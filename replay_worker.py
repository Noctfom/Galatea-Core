# ==================================================================================
#  Galatea AI Bot - 知识提取引擎 (终极防线版)
# ==================================================================================

import os
import time
import struct
import torch
import traceback
import numpy as np
import random

from yrp_parser import YrpParser
from galatea_env import GalateaEnv
from gamestate import DuelState, MessageParser
from deck_utils import Deck
from feature_encoder import GalateaEncoder
from ai_bot import AiBot
from data_types import GameAction

original_env_reset = GalateaEnv.reset

def patched_env_reset(self, deck0, deck1, seed=None, duel_flag=0, start_lp=8000, start_hand=5, draw_count=1):
    if self.pduel: 
        self.lib.end_duel(self.pduel)
        self.pduel = None
        
    duel_seed = seed if seed is not None else int(time.time()) & 0xFFFFFFFF
    is_replay = seed is not None
        
    self.pduel = self.lib.create_duel(duel_seed)
    self.lib.set_player_info(self.pduel, 0, start_lp, start_hand, draw_count)
    self.lib.set_player_info(self.pduel, 1, start_lp, start_hand, draw_count)
        
    def inject_deck(player_id, deck_obj):
        # 必须使用 [::-1]，底层 ocgcore push_back 是倒序的
        main_cards = deck_obj.main[::-1] if is_replay else deck_obj.main[:]
        if not is_replay: random.shuffle(main_cards) 
        for code in main_cards: 
            self.lib.new_card(self.pduel, code, player_id, player_id, 0x01, 0, 8)
        
        extra_cards = deck_obj.extra[::-1] if is_replay else deck_obj.extra[:]
        if not is_replay: random.shuffle(extra_cards)
        for code in extra_cards: 
            self.lib.new_card(self.pduel, code, player_id, player_id, 0x40, 0, 8)

    inject_deck(0, deck0)
    inject_deck(1, deck1)
    
    self.lib.start_duel(self.pduel, duel_flag)
    return self.step()

GalateaEnv.reset = patched_env_reset
encoder = GalateaEncoder()

def recover_human_macro(human_bytes, valid_actions, msg_type, active_player):
    """宏动作还原器"""
    if not human_bytes or len(human_bytes) < 2: return None
    act = GameAction(action_type=msg_type, index=len(valid_actions), desc_str="Human Macro")
    act.decision_bytes = human_bytes
    try:
        if msg_type in [15, 20]:
            count = human_bytes[0]
            if count > 1 and len(human_bytes) >= 1 + count:
                act.macro_targets = []
                for i in range(count):
                    val = human_bytes[1+i]
                    for va in valid_actions:
                        if va.index == val:
                            act.macro_targets.append(va.target_entity_idx)
                            break
                return act
        elif msg_type in [18, 24]:
            if len(human_bytes) >= 6 and len(human_bytes) % 3 == 0:
                act.macro_places = []
                for i in range(0, len(human_bytes), 3):
                    p, l, s = human_bytes[i], human_bytes[i+1], human_bytes[i+2]
                    p_abs = active_player if p == 0 else 1 - active_player
                    z = s | (16 if p_abs == 1 else 0) | (8 if l==8 else 0)
                    act.macro_places.append(z)
                return act
    except Exception as e:
        print(f"   ⚠️ 恢复人类宏动作时发生异常: {e}")
        traceback.print_exc()
    return None

def match_human_bytes_to_action(human_bytes, valid_actions, msg_type, active_player):
    """绝对精确且支持无序变长数组的智能解包器 (抗客户端畸形补零版)"""
    if not valid_actions or not human_bytes: return None

    padded = human_bytes.ljust(4, b'\x00')
    val_u32 = struct.unpack('<I', padded[:4])[0]
    val_i32 = struct.unpack('<i', padded[:4])[0]

    # 0. 优先拦截人类的右键取消 (Cancel)
    if val_i32 == -1 or human_bytes == b'\xff\xff\xff\xff':
        for idx, act in enumerate(valid_actions):
            if act.index == -1 or (hasattr(act, 'decision_bytes') and act.decision_bytes == struct.pack('<i', -1)): 
                return idx
        return None

    # 1. 宏动作智能比对
    if msg_type in [15, 18, 20, 23, 24, 25, 26]:
        for idx, act in enumerate(valid_actions):
            if not hasattr(act, 'decision_bytes') or not act.decision_bytes: continue
            db = act.decision_bytes
            
            # A. 字节完全相等
            if human_bytes.rstrip(b'\x00') == db.rstrip(b'\x00'):
                return idx
            
            # B. 变长选卡/同调素材：无序集合比对
            if msg_type in [15, 20, 23, 26]:
                h_count = human_bytes[0] if len(human_bytes) > 0 else 0
                d_count = db[0] if len(db) > 0 else 0
                if h_count == d_count and h_count > 0:
                    if set(human_bytes[1:1+h_count]) == set(db[1:1+d_count]):
                        return idx
                        
            # C. 位置选择 (18, 24)：坐标语义级无序比对！
            elif msg_type in [18, 24]:
                if len(human_bytes) >= len(db) and len(db) > 0:
                    valid_human = human_bytes[:len(db)]
                    
                    # 坐标解码器：透视 C++ 底层的坐标换算逻辑
                    def get_zones(b_array):
                        zones = set()
                        for i in range(0, len(b_array), 3):
                            if i + 2 < len(b_array):
                                p, l, s = b_array[i], b_array[i+1], b_array[i+2]
                                z_idx = s
                                if p != active_player: z_idx += 16
                                # 如果 l 不是 4 (MZONE)，底层一律算作 8 (SZONE)
                                if l != 4: z_idx += 8
                                zones.add(z_idx)
                        return zones
                        
                    h_zones = get_zones(valid_human)
                    d_zones = get_zones(db)
                    
                    if h_zones == d_zones:
                        return idx
        
        return None

    # 2. 基础单字节/双字节消息的对齐解包
    try:
        if msg_type in [10, 11]:
            h_idx = (val_u32 >> 16) & 0xFFFF
            h_cat = val_u32 & 0xFFFF
            for idx, act in enumerate(valid_actions):
                if act.index == h_idx and act.action_type == h_cat: return idx

        elif msg_type == 16:
            for idx, act in enumerate(valid_actions):
                if act.index == val_i32 or act.index == val_u32: return idx
            if len(human_bytes) >= 1:
                for idx, act in enumerate(valid_actions):
                    if act.index == human_bytes[0]: return idx

        elif msg_type == 19:
            if len(human_bytes) >= 1:
                for idx, act in enumerate(valid_actions):
                    if act.index == human_bytes[0]: return idx

        elif msg_type in [12, 13, 14, 140, 141, 142, 143]:
            for idx, act in enumerate(valid_actions):
                if act.index == val_u32 or act.desc_id == val_u32 or act.index == val_i32: 
                    return idx
            if msg_type in [12, 13] and len(human_bytes) >= 1:
                ans = 1 if human_bytes[0] > 0 else 0
                for idx, act in enumerate(valid_actions):
                    if act.index == ans: return idx
    except Exception as e:
        print(f"   ⚠️ 在智能解包阶段发生异常: {e}")

    # 3. 终极盲猜兜底
    for idx, act in enumerate(valid_actions):
        if act.index == val_i32 or act.index == val_u32: return idx
        if len(human_bytes) > 0 and act.index == human_bytes[0]: return idx
        
    return None

def play_replay(yrp_path, output_dir="./replay_data"):
    print(f"\n🎥 开始提取知识: {yrp_path}")
    parser = YrpParser(yrp_path)
    replay_data = parser.parse()
    if not replay_data: return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    p0_deck = Deck(name=replay_data['players'][0])
    p0_deck.main = replay_data['decks'][0]['main']
    p0_deck.extra = replay_data['decks'][0]['extra']
    p1_deck = Deck(name=replay_data['players'][1])
    p1_deck.main = replay_data['decks'][1]['main']
    p1_deck.extra = replay_data['decks'][1]['extra']

    env = GalateaEnv()
    bot = AiBot()
    
    duel_flag = replay_data.get('duel_flag', 0)
    raw_data = env.reset(p0_deck, p1_deck, 
                         seed=replay_data['seed'], 
                         duel_flag=replay_data['duel_flag'],
                         start_lp=replay_data['start_lp'],
                         start_hand=replay_data['start_hand'],
                         draw_count=replay_data['draw_count'])
    
    msg_queue = MessageParser.parse(raw_data) if raw_data else []
    state = DuelState(p0_deck.main, p0_deck.extra, p1_deck.main, p1_deck.extra)

    responses = replay_data['responses']
    resp_idx = 0
    dataset = []
    pending_experience = None  
    
    INTERACTION_MSGS = {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 140, 141, 142, 143}
    last_interaction_msg = None 
    last_payload = None
    last_valid_actions = []

    print(f"▶️ 知识提取引擎启动... (大师规则 Flag: 0x{duel_flag:X})")
    
    retry_count = 0

    try:
        while True:
            if not msg_queue:
                raw_data = env.step()
                if raw_data:
                    new_msgs = MessageParser.parse(raw_data)
                    if new_msgs:
                        first_msg_type = new_msgs[0][0]
                        if pending_experience:
                            if first_msg_type != 1: dataset.append(pending_experience)
                            pending_experience = None
                        msg_queue.extend(new_msgs)
                continue

            msg = msg_queue.pop(0)
            msg_type = msg[0]
            msg_payload = msg[1:]
            
            state.update(msg_type, msg_payload)
            
            if msg_type in INTERACTION_MSGS:
                last_interaction_msg = msg_type
                last_payload = msg_payload
                retry_count = 0

                # =================================================================
                # 1. 先请参谋部算好套餐，搞清楚当前这步“到底能不能取消”
                # =================================================================
                if msg_type in [15, 18, 20, 23, 24, 25, 26]:
                    import rule_bot 
                    macro_options = rule_bot.get_macro_options(msg_type, msg_payload)
                    if macro_options:
                        state.current_valid_actions = []
                        for i, opt in enumerate(macro_options):
                            act = GameAction(action_type=msg_type, index=i, desc_str=f"Macro {i}")
                            if 'locs' in opt: setattr(act, 'macro_targets', opt['locs'])
                            if 'places' in opt: setattr(act, 'macro_places', opt['places'])
                            setattr(act, 'decision_bytes', opt['bytes'])
                            state.current_valid_actions.append(act)

                # 拉取最新的全息快照和合法选项
                snap = state.get_snapshot(env)
                last_valid_actions = snap.valid_actions

                # =================================================================
                # 2. 动态判定：此时此刻允许取消吗？
                # =================================================================
                can_cancel = False
                for act in last_valid_actions:
                    if act.index == -1:
                        can_cancel = True
                        break
                    # 兼容 Rulebot 生成的结构体
                    if hasattr(act, 'decision_bytes') and act.decision_bytes == struct.pack('<i', -1):
                        can_cancel = True
                        break

                while resp_idx < len(responses) and responses[resp_idx] == b'\xff\xff\xff\xff':
                    if not can_cancel:
                        print("   ⏭️ 自动跳过无效操作 (当前不可取消)")
                        resp_idx += 1
                    else:
                        break

                if resp_idx >= len(responses):
                    print(f"\n✅ 录像播完了！成功提取了 {len(dataset)} 条高质量经验！")
                    break
                    
                human_bytes = responses[resp_idx]
                resp_idx += 1

                # =================================================================
                #  4. 将人类真实操作与套餐对齐
                # =================================================================
                if last_valid_actions:
                    target_action_idx = match_human_bytes_to_action(human_bytes, last_valid_actions, last_interaction_msg, state.active_player)
                    
                    if target_action_idx is not None:
                        # 1. 提取特征，生成 PPO 经验胶囊 (这部分代码保留不变)
                        obs_dict = encoder.encode(snap, state.active_player)
                        compressed_obs = {}
                        for k, v in obs_dict.items():
                            cpu_v = v.cpu().squeeze(0)
                            if cpu_v.dtype in [torch.long, torch.int64, torch.int32]: compressed_obs[k] = cpu_v.to(torch.int16)
                            elif cpu_v.dtype == torch.float32: compressed_obs[k] = cpu_v.to(torch.float16)
                            else: compressed_obs[k] = cpu_v
                                
                        pending_experience = {
                            'obs': compressed_obs,
                            'action': torch.tensor(target_action_idx, dtype=torch.int16),
                            'log_prob': torch.tensor(0.0, dtype=torch.float16),
                            'value': torch.tensor(1.0, dtype=torch.float16), 
                            'return': torch.tensor(1.0, dtype=torch.float16), 
                            'advantage': torch.tensor(1.0, dtype=torch.float16) 
                        }
                        
                        env.send_action(human_bytes)
                        msg_queue = []
                    else:
                        print(f"\n   ⚠️ [Mismatch 案发现场] 未能匹配人类操作！MsgType: {last_interaction_msg}")
                        print(f"      🙋 人类录像真实字节 (Hex): {human_bytes.hex(' ')}")
                        print(f"      🤖 Rulebot 提供的候选套餐 (共 {len(last_valid_actions)} 个):")
                        for display_idx, act in enumerate(last_valid_actions[:10]):
                            db_hex = act.decision_bytes.hex(' ') if hasattr(act, 'decision_bytes') and act.decision_bytes else "None"
                            target_info = ""
                            if hasattr(act, 'macro_targets') and act.macro_targets:
                                target_info = f" -> 实体: {act.macro_targets}"
                            elif hasattr(act, 'macro_places') and act.macro_places:
                                target_info = f" -> 坐标: {act.macro_places}"
                            print(f"         - [Idx:{act.index}] {act.desc_str} | Bytes: {db_hex}{target_info}")
                        if len(last_valid_actions) > 10:
                            print(f"         ... (省略其余 {len(last_valid_actions) - 10} 个套餐)")
                            
                        print("      ➡️ 采取行动: 发送原生人类字节强行推进...")
                        env.send_action(human_bytes)
                        msg_queue = []

            elif msg_type == 1:
                retry_count += 1
                # print(f"   ⚠️ 拦截到 MSG_RETRY (第 {retry_count} 次)") # 如果嫌吵可以注释掉这行
                
                # 🌟 把宽容度拉高到 50 次，静静地看着人类手残
                if retry_count > 50:
                    print(f"   💀 连续错误过多，无法从脱节中恢复！当前指针: {resp_idx}")
                    break
                
                if resp_idx < len(responses):
                    next_human_bytes = responses[resp_idx]
                    resp_idx += 1
                    # print(f"   🔄 引擎纠错：读取下一条人类指令 (Hex): {next_human_bytes.hex(' ')}")
                    env.send_action(next_human_bytes)
                else:
                    print("   💀 录像字节耗尽，无法恢复。")
                    break

            if resp_idx > 0 and resp_idx % 100 == 0 and not msg_queue:
                print(f"   ... 已同步 {resp_idx}/{len(responses)} 步, 截获 {len(dataset)} 帧黄金特征")

            if msg_type == 5:
                winner = msg[1] if len(msg) > 1 else -1
                print(f"\n🏆 提取结束！获胜者: 玩家 {winner}")
                break
                
    except Exception as e:
        print(f"💀 重演过程中发生崩溃: {e}")
        traceback.print_exc()
    finally:
        if env.pduel: 
            env.lib.end_duel(env.pduel)
        
        if dataset:
            ptr = len(dataset)
            batch_data = {
                'obs': {},
                'action': torch.stack([d['action'] for d in dataset]),
                'log_prob': torch.stack([d['log_prob'] for d in dataset]),
                'return': torch.stack([d['return'] for d in dataset]),
                'advantage': torch.stack([d['advantage'] for d in dataset])
            }
            for k in dataset[0]['obs'].keys():
                batch_data['obs'][k] = torch.stack([d['obs'][k] for d in dataset])
                
            batch_data['avg_rew'] = np.array([1.0], dtype=np.float32)
            batch_data['avg_len'] = np.array([ptr], dtype=np.float32)
            
            save_path = os.path.join(output_dir, f"tmp_rollout_replay_{int(time.time())}.pt")
            torch.save(batch_data, save_path)
            print(f"💾 PPO格式胶囊已保存至: {save_path} (有效动作: {ptr} 帧)")

if __name__ == "__main__":
    test_yrp = r"./replays/12-06「19：36：03」.yrp3d" 
    play_replay(test_yrp)