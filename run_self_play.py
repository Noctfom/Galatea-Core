'''
rulebot自决斗逻辑，包含详细战报和全息数据质检
'''

import struct
import io
import random
import os
import time
import sys
from collections import defaultdict
from rich.console import Console
from rich.table import Table

import deck_utils
import rule_bot
from galatea_env import GalateaEnv
from gamestate import DuelState, MessageParser
from card_reader import card_db
from game_constants import Zone, Phases, CardType, LocationInfo
from data_types import GameSnapshot, CardEntity
from feature_encoder import GalateaEncoder

# 配置
MAX_STEPS = 5000 
TOTAL_GAMES = 100

def get_summon_method(code):
    t = card_db.get_card_type(code)
    if t & 0x4000000: return "链接"
    if t & 0x800000: return "超量"
    if t & 0x2000: return "同调"
    if t & 0x40: return "融合"
    return "特殊"

def print_snapshot_inspection(snapshot: GameSnapshot, player_id: int):
    """打印指定玩家视角的快照信息"""
    g = snapshot.global_data
    print(f"\n🔍 [全息数据质检 - P{player_id} 视角]")
    print(f"   🌍 全局: Turn={g.turn_count}, Phase={Phases.get_str(g.phase_id)}, LP={g.my_lp}/{g.op_lp}")
    # 注意：这里改用了新的统计字段名
    print(f"   📚 资源: Deck={g.my_deck_len}, Hand={g.my_hand_len}, Grave={g.my_grave_len}, Extra={g.my_extra_len}")
    
    print(f"   🃏 场上实体采样 (按位置排序):")
    my_entities = [e for e in snapshot.entities if e.owner == player_id and e.code != 0]
    my_entities.sort(key=lambda x: (x.location, x.sequence))
    
    count = 0
    for entity in my_entities:
        loc_str = Zone.get_str(entity.location)
        card_name = card_db.get_card_name(entity.code)
        
        props = []
        if entity.type_mask & 0x1: props.append("怪")
        if entity.type_mask & 0x2: props.append("魔")
        if entity.type_mask & 0x4: props.append("陷")
        
        # [核心修正] 适配 V2.0 Schema
        # 在 V2.0 中，Rank/Link/Level 都统一存储在 level 字段
        if entity.type_mask & 0x800000: # TYPE_XYZ
            props.append(f"Rank{entity.level & 0xFFFF}")
        elif entity.type_mask & 0x4000000: # TYPE_LINK
            props.append(f"Link{entity.level & 0xFFFF}")
        elif (entity.type_mask & 0x1) and (entity.level > 0): # Monster with Level
            props.append(f"Lv{entity.level & 0xFFFF}")
        
        pos_str = "表攻"
        if entity.position == 0x4: pos_str = "表守"
        elif entity.position == 0x8: pos_str = "里守"
        elif entity.position == 0xA: pos_str = "表攻(里?)"
        
        print(f"      [{loc_str} {entity.sequence}] {card_name:<16} | {pos_str} | ATK {entity.current_atk} | {','.join(props)}")
        count += 1
        if count >= 8: break 
    print("-" * 50)


def run_single_game(env, deck1, deck2, name1="P0", name2="P1"):
    """
    执行单局游戏
    :return: (winner_id, turn_count, error_msg)
    """
    try:
        # 重置环境
        raw_data = env.reset(deck1, deck2)
        if not raw_data: return -1, 0, "Reset Failed (Empty Data)"
        
        # print(f"🕵️ [开局侦探] Raw Data Hex: {raw_data[:20].hex(' ')}...") # 可选调试
        msg_queue = MessageParser.parse(raw_data)
        
        brain_0 = DuelState(deck1.main, deck1.extra, deck2.main, deck2.extra)
        brain_1 = DuelState(deck1.main, deck1.extra, deck2.main, deck2.extra)

        encoder = GalateaEncoder()
        
        consecutive_retries = 0
        last_action_log = ""
        last_decision_value = None
        current_step_ignore_list = []
        
        step = 0
        winner = -1
        active_player = 0

        while step < MAX_STEPS:
            if not msg_queue:
                raw_data = env.step()
                if not raw_data: break
                msg_queue = MessageParser.parse(raw_data)
                continue
            
            msg = msg_queue.pop(0)
            msg_type = msg[0]
            msg_payload = msg[1:]
            
            brain_0.update(msg_type, msg_payload)
            brain_1.update(msg_type, msg_payload)
            
            # 追踪操作者
            if msg_type in [10, 11, 16, 15]:
                try: active_player = msg_payload[0] # 大部分交互消息第一个字节是 player
                except: pass
            
            brain = brain_0 if active_player == 0 else brain_1
            
            # ================= 🎙️ 详细战报 =================
            
            if msg_type == 40: # TURN
                print(f"\n======== ⏱️ P{msg_payload[0]} 的回合 ========")
            
            elif msg_type == 41: # PHASE
                try:
                    phase = struct.unpack('H', msg_payload[0:2])[0]
                    print(f"[阶段] {Phases.get_str(phase)}")
                except: pass

            elif msg_type == 50: # MOVE
                try:
                    code, old_raw, new_raw, reason = struct.unpack('<IIII', msg_payload)
                    oc, ol, os, _ = LocationInfo.decode(old_raw)
                    nc, nl, ns, _ = LocationInfo.decode(new_raw)
                    
                    # 只打印关键移动：送墓、回手、除外
                    if (nl & Zone.GRAVE) and not (ol & Zone.GRAVE):
                        name = card_db.get_card_name(code)
                        print(f"   💀 [墓地] P{nc} 的【{name}】送入墓地")
                except: pass

            elif msg_type == 60: # SUMMON
                code = struct.unpack('<I', msg_payload[0:4])[0]
                print(f"   🌟 [召唤] P{active_player} 通召: 【{card_db.get_card_name(code)}】")
            
            elif msg_type == 62: # SP_SUMMON
                code = struct.unpack('<I', msg_payload[0:4])[0]
                method = get_summon_method(code)
                print(f"   ✨ [特召] P{active_player} {method}召唤: 【{card_db.get_card_name(code)}】")

            elif msg_type == 70: # CHAIN
                code = struct.unpack('<I', msg_payload[0:4])[0]
                print(f"   ⛓️ [连锁] 发动: 【{card_db.get_card_name(code)}】")

            elif msg_type == 110: # ATTACK
                try:
                    atk_raw = struct.unpack('<I', msg_payload[0:4])[0]
                    ac, al, asq, _ = LocationInfo.decode(atk_raw)
                    atk_code = brain.get_card_code(ac, al, asq)
                    print(f"   ⚔️ [战斗] P{ac} 使用【{card_db.get_card_name(atk_code)}】攻击!")
                except: pass

            # ================= [NEW] 交互决策逻辑修正 =================
            # 包含所有需要 Bot 决策的消息 ID
            if msg_type in [10, 11, 16, 15, 12, 13, 14, 18, 19, 20, 22, 23, 26, 130, 131, 132, 133, 140, 141, 142, 143]:
                try: active_player = msg_payload[0]
                except: pass

                # [新增] --- 动作空间验证探针 ---
                # 只有当它是纯交互消息时才检查 (Type 11=Idle, 16=Chain, 15=Card)
                if msg_type in [11, 16, 15]:
                    snapshot = brain.get_snapshot()
                    actions = snapshot.valid_actions
                    
                    # 打印前3个动作来看看 (防止刷屏)
                    if len(actions) > 0:
                        print(f"   🤖 [动作感知] 捕获到 {len(actions)} 个合法操作:")
                        for i, act in enumerate(actions[:5]):
                            target_info = ""
                            if act.target_entity_idx >= 0:
                                # 反查一下是哪张卡，方便人类确认
                                if act.target_entity_idx < len(snapshot.entities):
                                    t_card = snapshot.entities[act.target_entity_idx]
                                    t_name = card_db.get_card_name(t_card.code)
                                    target_info = f" -> 指向 [{t_name}]"
                            
                            desc = act.desc_str if act.desc_str else f"Type={act.action_type}"
                            print(f"      [{i}] {desc} {target_info}")
                        if len(actions) > 5: print(f"      ... (还有 {len(actions)-5} 个)")
                    else:
                        print(f"   ⚠️ [动作感知] 警告: 收到交互请求 {msg_type} 但 valid_actions 为空!")
                # -----------------------------------

                # [NEW] 如果没有正在进行的重试，说明是新的题目，清空黑名单
                if consecutive_retries == 0:
                    current_step_ignore_list = []

                # [NEW] 将 ignore_actions 传入 Bot
                resp = rule_bot.get_rule_decision(
                    active_player, msg_type, msg, brain, 
                    ignore_actions=current_step_ignore_list
                )
                
                # [NEW] 记录这次决策，万一失败了要用
                last_decision_value = resp
                
                # 日志记录
                if msg_type == 11:
                    cat = resp & 0xFFFF
                    idx = (resp >> 16) & 0xFFFF
                    last_action_log = f"Idle操作: Cat={cat}, Idx={idx}"
                elif msg_type == 19:
                    last_action_log = f"位置选择: Val={resp}"
                else:
                    last_action_log = f"交互操作: Type={msg_type}"

                env.send_action(resp)
                msg_queue = [] # 发送动作后清空队列，等待新状态
            
            # ================= [最终修复] 核心拒绝 (RETRY) 处理 =================
            elif msg_type == 1: # MSG_RETRY
                consecutive_retries += 1
                if last_decision_value is not None:
                    current_step_ignore_list.append(last_decision_value)
                
                print(f"   ⚠️ [RETRY] 核心拒绝了操作 ({consecutive_retries}/10)")
                
                # 强力扰动 (Jitter)：针对顽固死锁
                if consecutive_retries > 6:
                    if "Idle操作: Cat=1" in last_action_log:
                        print("   ⚡ [Jitter] 暴力拉黑特召，范围扩大至 100...")
                        # [加固] 范围扩大到 100，确保覆盖所有可能的索引
                        for i in range(100):
                            current_step_ignore_list.append(f"1:{i}")
                    
                    elif "Type=18" in last_action_log:
                        print("   ⚡ [Jitter] 注入位置干扰 (Bytes)...")
                        # [绝对禁止] current_step_ignore_list.append(2)
                        # [正确做法]
                        current_step_ignore_list.append(b'\xFF\xFF\xFF')
                        current_step_ignore_list.append(b'\x00\x00\x00')
                        # 也可以注入一些可能合法的“盲猜”位置
                        current_step_ignore_list.append(b'\x00\x08\x05') # 尝试选场地
                    
                    elif "位置选择: Val=" in last_action_log: # Type 19
                        print("   ⚡ [Jitter] 检测到位置选择死锁，尝试随机扰动...")
                        # 注入几个常见的 Position 值 (1=表攻, 4=表守, 8=里守)
                        current_step_ignore_list.append(1)
                        current_step_ignore_list.append(4)
                        current_step_ignore_list.append(8)

                if consecutive_retries > 15:
                    print(f"   🛑 [熔断] 连续失败的操作: {last_action_log}")
                    print("   🔍 打印熔断时的全息快照:")
                    print_snapshot_inspection(brain.get_snapshot(), active_player)
                    print("   ☠️ 触发死循环保护，强制退出本局")
                    break 
                    
            else:
                # [关键修复] 智能记忆保留
                # 只有当游戏状态真正发生改变时，才清空“试错黑名单”。
                # 如果 Core 只是重发了 SELECT_IDLE 或 SELECT_CARD (交互请求)，说明还在同一个问题里纠缠，
                # 此时必须保留 current_step_ignore_list，否则 Bot 会忘记刚才试过的错选项。
                
                # 定义“状态变更”消息 ID 集合：
                # 40(Turn), 41(Phase), 50(Move), 60-65(Summon), 70(Chain), 90-94(HP/Draw)
                state_change_msgs = {40, 41, 50, 53, 54, 55, 56, 60, 61, 62, 70, 90, 91, 92, 94}
                
                # 定义“纯交互”消息 ID 集合：
                # 10, 11, 15, 16, 18, 19, 20, 22
                interaction_msgs = {10, 11, 15, 16, 18, 19, 20, 22, 26}
                
                # 逻辑：
                # 1. 如果是状态变更 -> 清空记忆
                # 2. 如果是交互请求 -> 保留记忆 (除非 consecutive_retries==0 表示是新的一轮)
                # 3. 杂项(Hint, Waiting) -> 不动
                
                if msg_type in state_change_msgs:
                    consecutive_retries = 0
                    current_step_ignore_list = []
                elif msg_type in interaction_msgs:
                    # 如果是新的一轮交互（retry归零），说明之前的操作成功了或者阶段变了
                    # 但如果是 Retry 后的重发，consecutive_retries 应该 > 0 (这由 RETRY 块控制)
                    # 等等，RETRY 消息本身不带交互数据。Core 发 RETRY 后紧接着发 IDLE。
                    # 所以 IDLE 来的时候，我们需要判断“这是 Retry 导致的重发”吗？
                    # 我们可以依赖 consecutive_retries。如果不为 0，说明正在重试中，不要清空！
                    if consecutive_retries == 0:
                        current_step_ignore_list = []
            
            # 胜利结算
            if msg_type == 5:
                # [修复] 防止 payload 为空导致崩溃
                if len(msg_payload) > 0:
                    winner = msg_payload[0]
                    if winner > 2: 
                        print(f"   ⚠️ [警告] 异常胜利ID {winner}，忽略...")
                        continue
                        
                    win_name = name1 if winner == 0 else name2
                    print(f"\n🎉 决斗结束！胜利者: P{winner} 【{win_name}】")
                    print_snapshot_inspection(brain_0.get_snapshot(), 0)

                    # ... 在 victory 打印之后，或者任意地方 ...
                    snapshot = brain_0.get_snapshot()
                    tensor_dict = encoder.encode(snapshot, player_id=0)
                    
                    # 打印看看形状对不对
                    print(f"Tensor Shape: {tensor_dict['card_idx'].shape}")
                    
                    # 【关键修正】这里必须是 return，不能是 break！
                    # break 会导致跳出循环后执行最后的 "return -1"
                    return winner, brain_0.turn, None
            
            step += 1

    except Exception as e:
        import traceback
        traceback.print_exc()
        return -1, 0, str(e)

    # 如果循环正常结束（即 step >= MAX_STEPS 且没分出胜负）
    # 这里的 return 只有在超时未分胜负时才应该被执行
    return -1, step, "Max Steps Reached"
