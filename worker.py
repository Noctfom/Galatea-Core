import torch
import numpy as np
import time
import random
import os
import struct
from galatea_env import GalateaEnv
from gamestate import MessageParser, DuelState
from ai_bot import AiBot
import deck_utils
import rule_bot
import warnings # [新增]
# [新增] 屏蔽 PyTorch 的 Nested Tensor 警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# 状态与消息定义
STATE_CHANGE_MSGS = {40, 41, 50, 53, 54, 55, 56, 60, 61, 62, 70, 90, 91, 92, 94}
INTERACTION_MSGS = {10, 11, 15, 16, 18, 19, 20, 22, 23, 24, 26, 130, 131, 132, 133}
AI_MANAGED_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 24]
DECISION_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 130, 131, 132, 133, 140, 141, 142, 143]

# GAE 参数 (和 Trainer 保持一致)
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_EPISODE_STEPS = 800

def worker_process(worker_id, net_config, weights, deck_dir, target_steps, device='cpu', req_q=None, resp_q=None):
    """
    工作进程：独立的 YGOPro 环境 + 独立的 AI (CPU模式)
    """
    # 🔴 [新增] 必须加这两行！限制 PyTorch 只能用单核
    # 防止 16 个进程每个都试图占用所有 CPU 核心导致死锁或卡顿
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        # 1. 初始化 AI (加载 Trainer 传来的权重)
        agent = AiBot(device=device, net_config=net_config)
        if weights is not None:
            agent.net.load_state_dict(weights, strict=False)
            agent.net.eval()
        
        env = GalateaEnv()
        memory = []
        collected_steps = 0
        
        # 统计数据
        episode_rewards = []
        episode_lens = []

        # print(f"👷 Worker {worker_id} 启动 | 目标: {target_steps} 步")

        while collected_steps < target_steps:
            # --- Reset 环境 ---
            try:
                res = deck_utils.get_random_deck_pair(ydk_dir=deck_dir)
                if not res or res[1] is None: 
                    time.sleep(1)
                    continue 
                d1_name, d1, d2_name, d2 = res
                raw_data = env.reset(d1, d2)
            except: continue
                
            if not raw_data: continue
            
            # [上帝视角] 直接读取 Deck 对象的属性
            p0_m, p0_e = d1.main, d1.extra
            p1_m, p1_e = d2.main, d2.extra
            
            brain = DuelState(p0_m, p0_e, p1_m, p1_e)
            try:
                msg_queue = MessageParser.parse(raw_data)
            except: continue

            # 局内状态
            game_buffer = {0: [], 1: []}
            winner = -1
            ep_steps = 0
            win_reason = 0
            
            # 短期记忆与 Retry 逻辑
            consecutive_retries = 0
            current_step_ignore_list = []
            last_decision_value = None
            last_interaction_msg = None
            last_act_time = time.time()

            while ep_steps < MAX_EPISODE_STEPS:
                # 超时保护
                if time.time() - last_act_time > 15.0:
                    break

                # 消息泵
                if not msg_queue:
                    try:
                        raw_data = env.step()
                    except OSError: break # Access Violation 保护
                    except Exception: break
                    
                    if not raw_data: break
                    
                    try:
                        msg_queue = MessageParser.parse(raw_data)
                    except: break
                    
                    last_act_time = time.time()
                    continue
                
                msg = msg_queue.pop(0)
                msg_type = msg[0]
                brain.update(msg_type, msg[1:])
                
                # 🟢 修改后 (增加读取 reason)
                if msg_type == 5: # Win
                    if len(msg[1:]) > 0: winner = msg[1:][0]
                    # 读取获胜原因: 0=投降, 1=LP归零, 2=卡组抽干(Deck Out), 3=超时
                    win_reason = msg[1:][1] if len(msg[1:]) > 1 else 0
                    break
                
                # ===========================
                # [关键逻辑] Retry 处理
                # ===========================
                if msg_type == 1: 
                    consecutive_retries += 1
                    
                    # 💥 [RL 惩罚] 不删除记忆，而是把刚才那步的 Value 强制拉低
                    snap = brain.get_snapshot()
                    player = snap.global_data.to_play
                    if game_buffer[player]:
                        # 给最后一步施加惩罚
                        game_buffer[player][-1]['value'] -= 0.01 
                    
                    if last_decision_value is not None:
                        current_step_ignore_list.append(last_decision_value)
                    
                    if consecutive_retries > 20: break
                    
                    if last_interaction_msg is not None:
                        msg = last_interaction_msg
                        msg_type = msg[0]
                    else: continue

                # 状态重置
                if msg_type in STATE_CHANGE_MSGS:
                    consecutive_retries = 0
                    current_step_ignore_list = []
                elif msg_type in INTERACTION_MSGS:
                    if consecutive_retries == 0: current_step_ignore_list = []

                # ===========================
                # [决策逻辑] AI vs RuleBot
                # ===========================
                if msg_type in DECISION_MSGS:
                    last_interaction_msg = msg
                    ai_handled = False
                    
                    # --- AI 尝试接管 ---
                    if msg_type in AI_MANAGED_MSGS and brain.current_valid_actions and consecutive_retries == 0:
                        try:
                            snap = brain.get_snapshot()
                            player = snap.global_data.to_play
                            tensor_dict = agent.encoder.encode(snap, player_id=player)
                            
                            # 🛡️ [终极防爆] 在源头直接将动作索引截断至 99，确保推理和训练全链路安全！
                            tensor_dict['act_card_idx'] = torch.clamp(tensor_dict['act_card_idx'], 0, 99)
                            
                            if req_q is not None and resp_q is not None:
                                # 🌟 [防爆降维：强类型 Numpy 字典]
                                # 彻底消灭 1D 拼接带来的 float32 内存膨胀！
                                numpy_dict = {}
                                for k, v in tensor_dict.items():
                                    if 'req' in k or 'mask' in k:
                                        numpy_dict[k] = v.numpy().astype(np.bool_) # 压缩 4 倍内存！
                                    elif 'feats' in k or 'num' in k or 'global' in k:
                                        numpy_dict[k] = v.numpy().astype(np.float16) # 半精度浮点
                                    else:
                                        # 改成 int32，防止未来词表扩容时 Hash ID 变成负数
                                        numpy_dict[k] = v.numpy().astype(np.int32)
                                        
                                req_q.put((worker_id, numpy_dict))
                                
                                # --- 2. [光速解包] ---
                                # 👇 [击毙幽灵 1：Numpy 解除] 收到 numpy 数组后还原为 Tensor
                                packed_res = torch.from_numpy(resp_q.get())
                                action_idx = packed_res[0].to(torch.long)
                                log_prob = packed_res[1]
                                value = packed_res[2]
                                
                                # 后续存入 game_buffer 还需要字典，所以这里保留转换
                                infer_dict = {k: v.cpu() for k, v in tensor_dict.items()}
                            else:
                                # 模式 B：同步本地模式 (兼容旧版回退)
                                with torch.no_grad():
                                    infer_dict = {k: v.to(device) for k, v in tensor_dict.items()}
                                    action_idx, log_prob, _, value = agent.get_action_and_value_from_tensor(infer_dict, snap.valid_actions)
                                action_idx = action_idx.cpu()
                                log_prob = log_prob.cpu()
                                value = value.cpu()
                                
                            sel_idx = action_idx.item()
                            if sel_idx < len(snap.valid_actions):
                                chosen = snap.valid_actions[sel_idx]
                            else:
                                chosen = random.choice(snap.valid_actions)
                            
                            # --- 动作翻译 ---
                            resp = b''
                            if msg_type in [10, 11, 15, 16]:
                                resp = agent._pack_response(chosen, msg_type=msg_type)
                            else:
                                val = chosen.index
                                # 位置处理
                                if msg_type in [18, 24]:
                                    zone_id = val
                                    p = 0; l = 0x04; s = 0
                                    if zone_id & 16: p = 1
                                    if zone_id & 8:  l = 0x08
                                    s = zone_id & 0x7
                                    req_p = 0
                                    if len(msg) > 1: req_p = msg[1]
                                    raw_p = req_p if p == 0 else (1 - req_p)
                                    final_p = 1 if raw_p == 1 else 0
                                    resp = bytes([final_p, l, s])
                                else:
                                    resp = bytes([val])
                            
                            env.send_action(resp)
                            last_decision_value = resp 
                            msg_queue = [] 
                            
                            # 🟢 2. 强制卸载：不管刚才在哪算的，存下来时必须 .cpu() 搬回系统内存
                            # 这样即使 16 个进程用 GPU，也不会导致 GPU 显存溢出 (OOM)
                            # ✅ 替换为:
                            game_buffer[player].append({
                                'obs': {k: v.cpu() for k, v in infer_dict.items()},
                                'action': action_idx.cpu(),
                                'log_prob': log_prob.cpu(),
                                'value': value.cpu()
                                # 删掉了 'valid_actions': snap.valid_actions
                            })
                            
                            ep_steps += 1
                            collected_steps += 1
                            ai_handled = True
                            last_act_time = time.time()
                        except Exception: 
                            ai_handled = False

                    # --- RuleBot 兜底 ---
                    if not ai_handled:
                        p = 0
                        if len(msg) > 1: p = msg[1]
                        
                        clean_ignore_list = []
                        for val in current_step_ignore_list:
                            clean_ignore_list.append(val)
                            try:
                                if isinstance(val, bytes):
                                    if len(val) >= 4:
                                        clean_ignore_list.append(struct.unpack('<I', val[:4])[0])
                                    elif len(val) >= 1:
                                        clean_ignore_list.append(val[0])
                            except: pass

                        resp = rule_bot.get_rule_decision(p, msg_type, msg, brain, ignore_actions=clean_ignore_list)
                        last_decision_value = resp 
                        msg_queue = [] 
                        env.send_action(resp)

            # --- 结算与 GAE ---
            if winner != -1 or ep_steps >= MAX_EPISODE_STEPS:
                episode_lens.append(ep_steps)
                for p in [0, 1]:
                    traj = game_buffer[p]
                    if not traj: continue
                    
                    final_reward = 0.0
                    if winner != -1 and winner <= 1:
                        if p == winner:
                            # 获取回合数
                            turns = brain.turn
                            
                            # === 针对性奖励修正 ===
                            if turns <= 20:
                                # 20回合内快速胜利：满分奖励
                                final_reward = 1.0
                            else:
                                # 长盘局 (turns > 20)
                                if win_reason == 2: 
                                    # [重罚] 靠抽干卡组赢的长盘 -> 只有 0.05 分
                                    # 告诉 AI: "虽然你赢了，但这很丢人，下次别这样"
                                    final_reward = 0.05
                                else:
                                    # [轻罚] 靠打死对面赢的长盘 -> 线性衰减，最低 0.5
                                    # 承认新手互啄打得慢是正常的，鼓励打死对面而不是抽干
                                    final_reward = max(0.5, 1.0 - (turns - 20) * 0.02)
                        else:
                            # 输了还是 -1
                            final_reward = -1.0
                    
                    if p == 0: episode_rewards.append(final_reward)

                    rewards = [0] * len(traj)
                    if rewards: rewards[-1] = final_reward
                    
                    advantages = []
                    last_gae_lam = 0
                    next_value = 0 

                    # [修改点 1] 初始化列式存储容器
                    if 'columns' not in locals():
                        columns = {
                            'obs': {}, 
                            'action': [],
                            'log_prob': [],
                            'return': [],
                            'advantage': []
                            # 删除了 valid_actions
                        }
                    
                    for t in reversed(range(len(traj))):
                        delta = rewards[t] + GAMMA * next_value - traj[t]['value'].item()
                        last_gae_lam = delta + GAMMA * GAE_LAMBDA * last_gae_lam
                        advantages.insert(0, last_gae_lam)
                        next_value = traj[t]['value'].item()
                    
                    # 打包发回 Trainer (列式存储)
                    for t in range(len(traj)):
                        # 1. 处理 Obs
                        for k, v in traj[t]['obs'].items():
                            if k not in columns['obs']: columns['obs'][k] = []
                            # [内存保护] 绝对保留布尔值，防止膨胀
                            if v.dtype == torch.bool:
                                columns['obs'][k].append(v)
                            elif v.is_floating_point():
                                columns['obs'][k].append(v.half())
                            else:
                                columns['obs'][k].append(v.to(dtype=torch.int32))
                        
                        # 2. 其他数据
                        columns['action'].append(traj[t]['action'])
                        columns['log_prob'].append(traj[t]['log_prob'].half())
                        
                        # Return/Advantage 转 Tensor 并压缩
                        ret_val = advantages[t] + traj[t]['value'].item()
                        columns['return'].append(torch.tensor(ret_val, dtype=torch.float16))
                        columns['advantage'].append(torch.tensor(advantages[t], dtype=torch.float16))
                        
                        # 删除了 columns['valid_actions'].append...
                        # 删除了 collected_steps += 1 (解决双重计数 Bug！)

        # 🛑 [修复] 使用 smart_concat 替代 stack
        if 'columns' not in locals() or len(columns['action']) == 0:
            return None, 0.0, 0.0

        def smart_concat(tensor_list):
            if not tensor_list: return torch.tensor([])
            if tensor_list[0].ndim == 0:
                return torch.stack(tensor_list)
            return torch.cat(tensor_list, dim=0)

        batch_data = {
            'obs': {k: smart_concat(v_list) for k, v_list in columns['obs'].items()},
            'action': smart_concat(columns['action']),
            'log_prob': smart_concat(columns['log_prob']),
            'return': smart_concat(columns['return']),
            'advantage': smart_concat(columns['advantage'])
            # 删除了 'valid_actions': columns['valid_actions']
        }
        
        avg_rew = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_len = np.mean(episode_lens) if episode_lens else 0.0
        
        return batch_data, avg_rew, avg_len

    except Exception as e:
        print(f"Worker {worker_id} Died: {e}")
        return None, 0.0, 0.0 # [修正] 必须返回 None，Trainer 才能正确识别并忽略