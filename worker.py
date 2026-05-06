# ==================================================================================
#  Worker Process for Galatea RL Training
#  每个 Worker 都是一个独立的 YGOPro 环境 + 独立的 AI (CPU模式)
#  负责与环境交互，收集经验，并通过队列与 Trainer 进行通信
# ==================================================================================


import torch
import numpy as np
import time
import random
import os
import gc
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
AI_MANAGED_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27]
DECISION_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 130, 131, 132, 133, 140, 141, 142, 143]

# GAE 参数 (和 Trainer 保持一致)
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_EPISODE_STEPS = 2000


def worker_process(worker_id, iteration, net_config, weight_file, deck_dir, target_steps, device='cpu', req_q=None, resp_q=None, opp_config=None):
    
    from system_logger import setup_global_logger
    setup_global_logger(prefix=f"Worker_{worker_id}")
    # =========================================================================
    #  [防卡死] 禁用 Windows 崩溃弹窗
    # 告诉操作系统：如果 OCGCore 发生致命写越界(Segfault)，直接静默杀死进程，不要弹窗也不要生成错误报告，以免训练被打断导致堆砌死锁
    if os.name == 'nt':
        import ctypes
        # SEM_FAILCRITICALERRORS (0x0001) | SEM_NOGPFAULTERRORBOX (0x0002) | SEM_NOOPENFILEERRORBOX (0x8000)
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    # =========================================================================
    
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    seed = (int(time.time() * 1000) % (2**31)) + (os.getpid() * 100) + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        agent = AiBot(device=device, net_config=net_config)
        #  从硬盘读取权重，斩断 Windows IPC 共享内存污染
        if weight_file and isinstance(weight_file, str) and os.path.exists(weight_file):
            weights = torch.load(weight_file, map_location=device, weights_only=False)
            agent.net.load_state_dict(weights, strict=False)
            agent.net.eval()
        
        # --- 新增：对手代理 P1 初始化 ---
        opp_agent = None
        if opp_config["mode"] == "ai":
            opp_agent = AiBot(device=device, net_config=net_config)
            opp_path = opp_config["path"]
            if opp_path and os.path.exists(opp_path):
                opp_weights = torch.load(opp_path, map_location=device, weights_only=False)
                # 处理保存的 checkpoint 字典或纯 state_dict
                sd = opp_weights['model_state_dict'] if isinstance(opp_weights, dict) and 'model_state_dict' in opp_weights else opp_weights
                opp_agent.net.load_state_dict(sd, strict=False)
                opp_agent.net.eval()
            else:
                # 如果是自对局且路径是 weight_file，直接共享内存或再次加载
                opp_agent.net.load_state_dict(agent.net.state_dict())
                opp_agent.net.eval()

        env = GalateaEnv()
        collected_steps = 0
        
        consecutive_ai_fails = 0 # 死亡熔断计数器
        
        # 统计数据
        episode_rewards = []
        episode_lens = []

        stats = {
            'wins_all_first': 0, 'games_all_first': 0,
            'wins_all_second': 0, 'games_all_second': 0,
            'wins_self_first': 0, 'games_self_first': 0,
            'wins_self_second': 0, 'games_self_second': 0,
            'wins_hist': 0, 'games_hist': 0,
            'wins_rule': 0, 'games_rule': 0,
            'deadlocks': 0, 'timeouts': 0, 'draws': 0
        }

        deck_records = []

        # print(f"👷 Worker {worker_id} 启动 | 目标: {target_steps} 步")

        while collected_steps < target_steps:
            # --- 每局开始前随机摇号决定座位 ---
            train_p_id = random.choice([0, 1]) 
            opp_p_id = 1 - train_p_id
            
            # 记录这局是打先手还是后手
            opp_type = opp_config.get("type", "self")
            if train_p_id == 0: 
                stats['games_all_first'] += 1
                if opp_type == "self": 
                    stats['games_self_first'] += 1
            else: 
                stats['games_all_second'] += 1
                if opp_type == "self": 
                    stats['games_self_second'] += 1
                
            if opp_type == "hist": 
                stats['games_hist'] += 1
            elif opp_type == "rule": 
                stats['games_rule'] += 1

            # --- Reset 环境 ---
            try:
                res = deck_utils.get_random_deck_pair(ydk_dir=deck_dir)
                if not res or res[1] is None: 
                    time.sleep(1)
                    continue 
                env_name, d1_name, d1, d2_name, d2 = res
                raw_data = env.reset(d1, d2)
            except Exception as e: 
                print(f"⚠️ [Worker {worker_id}] 环境Reset异常 (已记录并跳过): {e}")
                continue
                
            if not raw_data: continue
            
            # [上帝视角] 直接读取 Deck 对象的属性
            p0_m, p0_e = d1.main, d1.extra
            p1_m, p1_e = d2.main, d2.extra
            
            brain = DuelState(p0_m, p0_e, p1_m, p1_e)
            try:
                msg_queue = MessageParser.parse(raw_data)
            except Exception as e:
                print(f"⚠️ [Worker {worker_id}] 初始消息解析异常 (已记录并跳过): {e}")
                continue

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
                    print(f"⚠️ [Worker {worker_id}] 引擎 15 秒无响应，触发超时强制结算！")
                    ep_steps = MAX_EPISODE_STEPS  # 强制拉满，骗过底下的结算条件
                    break

                # 消息泵
                if not msg_queue:
                    try:
                        raw_data = env.step()
                    except OSError as e: 
                        print(f"⚠️ [Worker {worker_id}] OCGCore 引擎底层崩溃: {e}")
                        break 
                    except Exception as e: 
                        print(f"⚠️ [Worker {worker_id}] 环境 Step 发生未知错误: {e}")
                        import traceback; traceback.print_exc()
                        break
                    
                    if not raw_data: break
                    
                    try:
                        msg_queue = MessageParser.parse(raw_data)
                    except Exception as e: 
                        print(f"⚠️ [Worker {worker_id}] 消息解析失败 (静默退出): {e}")
                        break
                    
                    last_act_time = time.time()
                    continue
                
                msg = msg_queue.pop(0)
                msg_type = msg[0]
                brain.update(msg_type, msg[1:])
                
                # 增加读取 reason
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
                    
                    # 不删除记忆，而是把刚才那步的 Value 强制拉低
                    snap = brain.get_snapshot(env)
                    player = snap.global_data.to_play
                    if game_buffer[player]:
                        # 给最后一步施加惩罚
                        game_buffer[player][-1]['step_reward'] -= 0.05
                    
                    if last_decision_value is not None:
                        current_step_ignore_list.append(last_decision_value)
                    
                    current_limit = 100 if (last_interaction_msg and last_interaction_msg[0] == 142) else 40
                    if consecutive_retries > current_limit: 
                        # AI疯狂瞎按：直接判输，结束对局并惩罚
                        winner = 1 - player
                        win_reason = 0
                        break
                    
                    if last_interaction_msg is not None:
                        msg = last_interaction_msg
                        msg_type = msg[0]
                    else: 
                        continue

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

                    # 获取当前行动玩家
                    snap = brain.get_snapshot(env)
                    player = snap.global_data.to_play
                    
                    # --- 通过身份判断使用哪个脑子 ---
                    is_training_agent = (player == train_p_id)
                    current_agent = None
                    if is_training_agent:
                        current_agent = agent
                    elif opp_config["mode"] == "ai":
                        current_agent = opp_agent
                    
                    # --- AI 尝试接管 ---
                    if current_agent and msg_type in AI_MANAGED_MSGS and brain.current_valid_actions and consecutive_retries == 0:
                        
                        # 使用 current_agent 进行编码和推理
                        tensor_dict = current_agent.encoder.encode(snap, player_id=player)

                        # 向参谋部索要套餐，拦截并覆盖单选题
                        if msg_type in [15, 18, 20, 23, 24, 25]:
                            macro_options = rule_bot.get_macro_options(msg_type, msg[1:])
                            if macro_options:
                                from data_types import GameAction
                                brain.current_valid_actions = []
                                for i, opt in enumerate(macro_options):
                                    act = GameAction(action_type=msg_type, index=i, desc_str=f"Macro {i}")
                                    if 'locs' in opt: setattr(act, 'macro_targets', opt['locs'])
                                    if 'places' in opt: setattr(act, 'macro_places', opt['places']) # 🌟 绑定格子坐标
                                    setattr(act, 'decision_bytes', opt['bytes'])
                                    brain.current_valid_actions.append(act)
                        
                        try:
                            snap = brain.get_snapshot(env)
                            player = snap.global_data.to_play
                            tensor_dict = agent.encoder.encode(snap, player_id=player)
                            
                            # 防止嵌入层越界崩溃，设定索引上限（根据训练时的 vocab_size）
                            # 动态适配，不要写死 99
                            max_idx = tensor_dict['act_card_idx'].shape[-1] if len(tensor_dict['act_card_idx'].shape) > 2 else 119 # 保底 119
                            tensor_dict['act_card_idx'] = torch.clamp(tensor_dict['act_card_idx'], 0, 119)
                            tensor_dict['act_type'] = torch.clamp(tensor_dict['act_type'], 0, 255)
                            tensor_dict['act_desc'] = torch.clamp(tensor_dict['act_desc'], 0, 1023)
                            
                            # 是否异步模式分流
                            # [核心修复]：混合推理架构
                            # 只有 P0 才能使用异步推断服务器（因为服务器只有最新权重）。
                            # P1（历史模型）直接在 Worker 本地进行推理
                            if req_q is not None and resp_q is not None and is_training_agent:
                                # --- 模式 A: 真正的异步 Server 推理 (仅限 P0) ---
                                import queue
                                numpy_dict = {}
                                for k, v in tensor_dict.items():
                                    arr = v.cpu().numpy()
                                    # 极致压缩防止 Windows 管道爆炸
                                    if arr.dtype == np.float32: arr = arr.astype(np.float16)
                                    elif arr.dtype == np.bool_: pass 
                                    else: arr = arr.astype(np.int16) 
                                    numpy_dict[k] = arr
                                    
                                try:
                                    # 如果 Server 死掉导致队列满，Worker 等 2 秒就直接自毁
                                    req_q.put((worker_id, iteration, numpy_dict), timeout=2.0)
                                except queue.Full:
                                    raise RuntimeError("请求队列已满/Server可能已崩溃，触发 Worker 上行熔断")
                                
                                try:
                                    res_array = resp_q.get(timeout=15.0)
                                except queue.Empty:
                                    raise RuntimeError("推断服务器无响应，触发超时熔断机制")
                                    
                                packed_res = torch.from_numpy(res_array)
                                action_idx = packed_res[0].to(torch.long)
                                log_prob = packed_res[1]
                                value = packed_res[2]
                                rnd_reward = packed_res[3].item()
                                infer_dict = {k: v.cpu() for k, v in tensor_dict.items()}
                            else:
                                # --- 模式 B: 纯本地推理 (未开启 Server，或当前是 P1 行动) ---
                                with torch.no_grad():
                                    infer_dict = {k: v.to(device) for k, v in tensor_dict.items()}
                                    
                                    # 修复：使用 current_agent，而非定死 agent
                                    action_idx, log_prob, _, value, v_input = current_agent.get_action_and_value_from_tensor(infer_dict, snap.valid_actions)
                                    
                                    if is_training_agent:
                                        rnd_reward = current_agent.net.rnd(v_input.to(device)).item()
                                        current_agent.net.update_rnd_stats(v_input.to(device))
                                    else:
                                        rnd_reward = 0.0

                                # 新增 .detach()，确保放入字典的张量干干净净
                                action_idx = action_idx.detach().cpu()
                                log_prob = log_prob.detach().cpu()
                                value = value.detach().cpu()
                                infer_dict = {k: v.detach().cpu() for k, v in tensor_dict.items()}
                                
                            sel_idx = action_idx.item()
                            if sel_idx < len(snap.valid_actions):
                                chosen = snap.valid_actions[sel_idx]
                            else:
                                chosen = random.choice(snap.valid_actions)

                            # 软性时间惩罚 (Soft Enrage)
                            step_reward = 0.0
                            if chosen.action_type == 7: # 选择进入结束阶段 (To EP)
                                # 如果回合数大于 10，且这回合有操作空间但 AI 疯狂空过
                                if brain.turn > 10 and len(snap.valid_actions) > 2:
                                    step_reward -= 0.001 * (brain.turn - 10)

                            # --- 动作翻译 ---
                            resp = b''
                            
                            # 拦截代打：如果是包含预计算字节的套餐，直接越过打包器发送！
                            if hasattr(chosen, 'decision_bytes'):
                                resp = chosen.decision_bytes
                            elif msg_type in [10, 11, 15, 16]:
                                resp = agent._pack_response(chosen, msg_type=msg_type)
                            elif msg_type in [140, 141, 142]:
                                resp = struct.pack('<I', int(chosen.desc_id))
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
                            
                            # 内存脱水压缩：把 64位长整型全部压成 16位短整型
                            compressed_obs = {}
                            for k, v in infer_dict.items():
                                cpu_v = v.cpu()
                                if cpu_v.dtype in [torch.long, torch.int64, torch.int32]:
                                    compressed_obs[k] = cpu_v.to(torch.int16)
                                elif cpu_v.dtype == torch.float32:
                                    compressed_obs[k] = cpu_v.to(torch.float16)
                                else:
                                    compressed_obs[k] = cpu_v
                                    
                            # 注意：只有训练代理的数据才需要放入 game_buffer 供学习
                            if is_training_agent:
                                game_buffer[player].append({
                                    'obs': compressed_obs,
                                    'action': action_idx.cpu().to(torch.int16),
                                    'log_prob': log_prob.cpu().to(torch.float16),
                                    'value': value.cpu().to(torch.float16),
                                    'step_reward': step_reward
                                })
                                collected_steps += 1
                            
                            ep_steps += 1
                            ai_handled = True
                            last_act_time = time.time()
                        except OSError as e:
                            #  C++ DLL 底层内存越界
                            # 绝对不能让 RuleBot 接管，否则会引发无限死循环 DDOS 攻击主进程
                            print(f"💀 [Worker {worker_id}] C++引擎致命越界，强行终止本局以防死锁！({e})")
                            winner = 1 - player  # 惩罚导致崩溃的 AI
                            win_reason = 0
                            return
                            
                        except Exception as e: 
                            # 纯 Python 逻辑报错，允许 RuleBot 兜底
                            print(f"\n❌ [Worker {worker_id}] AI 逻辑计算崩溃: {e}")
                            import traceback
                            traceback.print_exc()
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
                            except Exception as e: 
                                print(f"⚠️ [Worker {worker_id}] ignore_list解析异常: {e}")

                        resp = rule_bot.get_rule_decision(p, msg_type, msg, brain, ignore_actions=clean_ignore_list)
                        
                        # 修复：不再强制计算 RuleBot 的 log_prob，也不把它放进 game_buffer
                        # 从 AI 的视角来看，这相当于对手或者系统强制替它走了一步，是环境状态的跃迁。
                        try:
                            env.send_action(resp)
                            # 我们只增加环境步数，不增加 collected_steps (因为它没有产出有效梯度数据)
                            ep_steps += 1
                        except OSError as e:
                            # 如果 RuleBot 发送后引擎崩溃，同样强行打断循环
                            print(f"💀 [Worker {worker_id}] RuleBot 踩雷导致引擎崩溃，强行终止本局！({e})")
                            winner = 1 - player
                            win_reason = 0
                            return
                        except Exception as e:
                            print(f"⚠️ [Worker {worker_id}] RuleBot 发送动作异常: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

            # --- 结算与 GAE ---
            if winner != -1 or ep_steps >= MAX_EPISODE_STEPS:
                if ep_steps == 0:
                    consecutive_ai_fails += 1
                    if consecutive_ai_fails >= 3:
                        print(f"💀 [Worker {worker_id}] AI 连续 3 局无法行动，触发物理自毁防止死锁！")
                        return
                else:
                    consecutive_ai_fails = 0
                    episode_lens.append(ep_steps)

                # 只处理主训练模型 (train_p_id) 的数据
                traj = game_buffer[train_p_id]
                if traj:
                    final_reward = 0.0

                    if ep_steps >= MAX_EPISODE_STEPS:
                        stats['timeouts'] += 1

                    if winner != -1 and winner <= 1:
                        if winner == train_p_id:
                            # 赢了！记录对应的胜率
                            if train_p_id == 0:
                                stats['wins_all_first'] += 1
                                if opp_type == "self": 
                                    stats['wins_self_first'] += 1
                            else:
                                stats['wins_all_second'] += 1
                                if opp_type == "self": 
                                    stats['wins_self_second'] += 1
                            my_deck = d1_name if train_p_id == 0 else d2_name
                            opp_deck = d2_name if train_p_id == 0 else d1_name
                            deck_records.append({
                                'env': env_name, 
                                'my_deck': my_deck, 
                                'opp_deck': opp_deck, # <--- 记录对手是谁
                                'is_first': (train_p_id == 0), 
                                'is_win': True
                            })
                                
                            if opp_type == "hist": 
                                stats['wins_hist'] += 1
                            elif opp_type == "rule": 
                                stats['wins_rule'] += 1
                            
                            turns = brain.turn
                            if turns <= 40: 
                                final_reward = 1.0
                            else: 
                                final_reward = 0.05
                        
                        elif winner == opp_p_id:
                            final_reward = -1.0 # 输了
                            my_deck = d1_name if train_p_id == 0 else d2_name
                            opp_deck = d2_name if train_p_id == 0 else d1_name
                            deck_records.append({
                                'env': env_name, 
                                'my_deck': my_deck, 
                                'opp_deck': opp_deck, 
                                'is_first': (train_p_id == 0), 
                                'is_win': False
                            })
                        else:
                            stats['draws'] += 1
                            final_reward = 0.0  # 平局
                    elif winner == -1:
                        if ep_steps < MAX_EPISODE_STEPS:
                            stats['deadlocks'] += 1
                        final_reward = -1.0 # 把死锁直接视为惨败
                    else:
                        stats['draws'] += 1
                        final_reward = 0.0
                    
                    episode_rewards.append(final_reward)

                    rewards = [0] * len(traj)
                    if rewards: rewards[-1] = final_reward
                    
                    advantages = []
                    last_gae_lam = 0
                    # 修复：区分“真死”和“超时截断”
                    next_value = 0.0
                    if ep_steps >= MAX_EPISODE_STEPS: # 如果是超时被强行截断的
                        # 使用最后一步的价值网络预测值作为未来收益的保底，而不是直接视为 0，这样可以让模型学会在超时前尽可能积累价值，而不是放弃抵抗
                        next_value = traj[-1]['value'].item()

                    # ==========================================
                    # 终极内存防爆：预分配连续内存，告别 list 与 torch.cat
                    # ==========================================
                    if 'columns' not in locals():
                        # 预先分配足够的空间 (目标步数 + 最大单局步数容错)
                        max_len = target_steps + MAX_EPISODE_STEPS + 100
                        columns = {
                            'obs': {}, 
                            'action': torch.zeros(max_len, dtype=torch.long),
                            'log_prob': torch.zeros(max_len, dtype=torch.float16),
                            'return': torch.zeros(max_len, dtype=torch.float16),
                            'advantage': torch.zeros(max_len, dtype=torch.float16)
                        }
                        ptr = 0 # 内存写入指针
                    
                    # 替换为这套融合了步进奖励的新逻辑：
                    for t in reversed(range(len(traj))):
                        # 获取这一步的面包屑奖励
                        s_rew = traj[t].get('step_reward', 0.0)
                        # 只有最后一步才加上胜负大奖 (rewards[t] 就是 final_reward)
                        actual_rew = s_rew + (rewards[t] if t == len(traj) - 1 else 0.0)
                        
                        delta = actual_rew + GAMMA * next_value - traj[t]['value'].item()
                        last_gae_lam = delta + GAMMA * GAE_LAMBDA * last_gae_lam
                        advantages.insert(0, last_gae_lam)
                        next_value = traj[t]['value'].item()
                    
                    # 动态初始化 obs 的预分配空间
                    if not columns['obs'] and len(traj) > 0:
                        for k, v in traj[0]['obs'].items():
                            columns['obs'][k] = torch.zeros((max_len,) + v.shape[1:], dtype=v.dtype)
                    
                    # 极限内存优化：流式填入数据，并立刻弹出销毁历史记录
                    for t in range(len(traj)):
                        obs_dict = traj[t].pop('obs') # 弹出并销毁，边填边释放历史数据，杜绝双重占用！
                        for k, v in obs_dict.items():
                            columns['obs'][k][ptr] = v[0]
                        del obs_dict
                        
                        columns['action'][ptr] = traj[t]['action']
                        columns['log_prob'][ptr] = traj[t]['log_prob'].half()
                        
                        ret_val = advantages[t] + traj[t]['value'].item()
                        columns['return'][ptr] = ret_val
                        columns['advantage'][ptr] = advantages[t]
                        
                        ptr += 1

        # 死锁保护
        tmp_file = f"tmp_rollout_iter_{iteration}_worker_{worker_id}.pt"
        if 'columns' not in locals() or ptr == 0:
            return # 改动：如果没数据直接返回，不生成空文件，主进程会自动忽略

        # 关键：使用纯净 Tensor 切片并 Clone，绝不触发 Pickle 内存翻倍！
        batch_data = {
            'obs': {k: v[:ptr].clone() for k, v in columns['obs'].items()},
            'action': columns['action'][:ptr].clone(),
            'log_prob': columns['log_prob'][:ptr].clone(),
            'return': columns['return'][:ptr].clone(),
            'advantage': columns['advantage'][:ptr].clone()
        }
        
        # 释放巨大的预分配内存池
        del columns
        gc.collect()
        
        avg_rew = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_len = np.mean(episode_lens) if episode_lens else 0.0
        batch_data['avg_rew'] = np.array([avg_rew], dtype=np.float32)
        batch_data['avg_len'] = np.array([avg_len], dtype=np.float32)

        # 把胜率字典打包成 numpy 发回去
        for k, v in stats.items():
            batch_data[f'stats_{k}'] = np.array([v], dtype=np.int32)

        batch_data['deck_records'] = deck_records
        
        # 原子写入：先写成临时文件，写完瞬间改名。绝不让 Trainer 读到损坏的残局！
        tmp_write_file = tmp_file + ".tmp"
        torch.save(batch_data, tmp_write_file)
        # 替换原来的 os.replace
        for _ in range(5):
            try:
                os.replace(tmp_write_file, tmp_file)
                break
            except PermissionError:
                time.sleep(0.1)
            except Exception as e:
                print(f"❌ [Worker {worker_id}] 文件写入失败: {e}")
                return
        
        return

    except Exception as e:
        # 打印具体的异常名称，MemoryError 不再隐形
        print(f"Worker {worker_id} Died: [{type(e).__name__}] {e}")
        import traceback
        traceback.print_exc()
        return