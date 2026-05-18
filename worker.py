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
AI_MANAGED_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 140, 141, 142, 143]
DECISION_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 130, 131, 132, 133, 140, 141, 142, 143]

# GAE 参数 (和 Trainer 保持一致)
GAMMA = 0.998
GAE_LAMBDA = 0.95
MAX_EPISODE_STEPS = 2000

def log_fatal_crash(worker_id, source, msg_type, resp, exc, valid_actions=None):
    """独立的跨进程安全黑匣子：只写入专用文件，绝不干涉主进程日志"""
    import datetime
    import os
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 防空数据解析
    if isinstance(resp, (bytes, bytearray)):
        safe_hex = bytes(resp).hex(' ') if len(resp) > 0 else "EMPTY_BYTES_b''"
        resp_repr = f"{safe_hex} (Hex, Len: {len(resp)})"
    elif isinstance(resp, int):
        resp_repr = f"{resp} (Int)"
    else:
        resp_repr = f"{resp} ({type(resp).__name__})"

    err_msg = f"\n[{timestamp}] 💀 [Worker {worker_id} 黑匣子] {source} 踩雷导致 C++ 引擎越界崩溃！\n"
    err_msg += f"   📍 崩溃前指令 Type: {msg_type}\n"
    err_msg += f"   📦 致命数据包: {resp_repr}\n"
    if valid_actions:
        err_msg += f"   🃏 案发现场选项池:\n"
        for idx, act in enumerate(valid_actions):
            err_msg += f"      [{idx}] {act.desc_str} (Code: {act.code if hasattr(act, 'code') else 'Unknown'})\n"
    err_msg += f"   🛑 异常堆栈: {exc}\n"

    print(err_msg) 
    
    # 放入标准的日志文件夹，纯粹的独立追加写入
    try:
        os.makedirs("./system_logs", exist_ok=True)
        crash_log_path = "./system_logs/engine_crashes.log"
        with open(crash_log_path, "a", encoding="utf-8") as f:
            f.write(err_msg)
    except Exception: 
        # 哪怕发生极端情况的并发踩踏，也静默吞下异常，保证 Worker 能够顺利走完后续的销毁流程
        pass

def worker_process(worker_id, iteration, net_config, weight_file, deck_dir, target_steps, device='cpu', req_q=None, resp_q=None, opp_config=None, worker_timeout=300, gamma=GAMMA, gae_lambda=GAE_LAMBDA, num_workers=4):
    
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

        worker_start_time = time.time()
        max_worker_uptime = float(worker_timeout) - 10.0
        
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
            if time.time() - worker_start_time > max_worker_uptime:
                break
            # --- 每局开始前随机摇号决定座位 ---
            train_p_id = random.choice([0, 1]) 
            opp_p_id = 1 - train_p_id
            ai_is_broken = {0: False, 1: False}

            current_turn = 0
            turn_steps = 0
            
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
                        
                # 只要底层吐出的新缓冲区里【完全没有】 RETRY(1)，才证明操作成功
                has_retry = any(m[0] == 1 for m in msg_queue)
                if msg_queue and not has_retry:
                    consecutive_retries = 0
                    current_step_ignore_list = []
                    current_step_ignore_idx_list = []
                    current_macro_pool = None
                            
            except Exception as e: 
                print(f"⚠️ [Worker {worker_id}] 消息解析引发严重错误: {e}")
                import traceback; traceback.print_exc()
                try: env._close_duel() # 🌟 强行摧毁坏掉的引擎
                except Exception: 
                    pass
                continue

            # 局内状态
            game_buffer = {0: [], 1: []}
            winner = -1
            ep_steps = 0
            win_reason = 0
            
            # 短期记忆与 Retry 逻辑
            consecutive_retries = 0
            current_step_ignore_list = []
            current_step_ignore_idx_list = []
            current_macro_pool = None
            last_decision_value = None
            last_decision_idx = None
            last_interaction_msg = None
            last_act_time = time.time()

            loop_tracker = {}
            last_valid_hash = ""

            while ep_steps < MAX_EPISODE_STEPS:
                resp = None  # ✅ 每次循环强制清空上一回合的残骸，防止变量逃逸
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
                        # [清除跨回合污染] 只要引擎发来的新状态里没有 RETRY，立刻清空错题本和缓存
                        has_retry = any(m[0] == 1 for m in msg_queue)
                        if msg_queue and not has_retry:
                            consecutive_retries = 0
                            current_step_ignore_list = []
                            current_step_ignore_idx_list = []
                            current_macro_pool = None
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
                        game_buffer[player][-1]['step_reward'] -= 0.001 # 惩罚力度可以调整
                    
                    if last_decision_value is not None:
                        current_step_ignore_list.append(last_decision_value)
                    if last_decision_idx is not None:
                        current_step_ignore_idx_list.append(last_decision_idx) # 记录失败的数字索引
                    
                    current_limit = 100 if (last_interaction_msg and last_interaction_msg[0] == 142) else 40
                    if consecutive_retries > current_limit: 
                        # AI疯狂瞎按：直接判输，结束对局并惩罚
                        winner = 1 - player
                        win_reason = 0
                        print(f"⚠️ [Worker {worker_id}] AI 连续 {consecutive_retries} 次触发 Retry，超过阈值 {current_limit}，强制判负, 消息类型： {last_interaction_msg[0] if last_interaction_msg else 'Unknown'}，最后操作值：{last_decision_value}")
                        break
                    
                    if last_interaction_msg is not None:
                        msg = last_interaction_msg
                        msg_type = msg[0]
                    else: 
                        continue

                # ===========================
                # [决策逻辑] AI vs RuleBot
                # ===========================
                if msg_type in DECISION_MSGS:
                    last_interaction_msg = msg
                    ai_handled = False
                    
                    # 每次到了能操作的阶段，检查下班时间
                    if time.time() - worker_start_time > max_worker_uptime:
                        print(f"⏳ [Worker {worker_id}] 达到最大服役时间上限，安全下班并上交数据...")
                        ep_steps = MAX_EPISODE_STEPS
                        break

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
                    is_macro_supported = msg_type in [15, 18, 20, 22, 23, 24, 25]
                    if current_agent and msg_type in AI_MANAGED_MSGS and (brain.current_valid_actions or is_macro_supported):
                        
                        # ========================================================
                        # [Two-Pass 架构] 拦截多选/排序等复杂指令，进行意图打分与降维
                        # ========================================================
                        # 严格限定只有 RuleBot 真实支持的 6 个组合指令
                        if msg_type in [15, 18, 20, 22, 23, 24, 25]:
                            # 只有在非重试阶段（或缓存为空时），才去生成新的 100 个套餐
                            if current_macro_pool is None:
                                # [前置意图感知] 在生成 5000 套餐前，先问问网络喜欢哪张卡
                                code_preferences = {}
                                try:
                                    with torch.no_grad():
                                        snap_pass1 = brain.get_snapshot(env)
                                        dict_pass1 = current_agent.encoder.encode(snap_pass1, player_id=player)
                                        v_input = {k: v.to(current_agent.device) for k, v in dict_pass1.items() if isinstance(v, torch.Tensor)}
                                        action_logits, _, _ = current_agent.net(v_input)
                                        probs = torch.softmax(action_logits, dim=-1).squeeze(0).cpu().numpy()
                                        
                                        # 将单卡概率映射到卡密上
                                        for i, act in enumerate(brain.current_valid_actions):
                                            if hasattr(act, 'code'):
                                                # 如果同一个卡密出现多次，取最高权重
                                                code_preferences[act.code] = max(code_preferences.get(act.code, 0), probs[i])
                                except Exception as e:
                                    pass # 如果前置感知失败，静默降级为无偏好生成
                                
                                # 将主将的偏好字典传给参谋部
                                large_options = rule_bot.get_macro_options(msg_type, msg[1:], brain, limit=5000, pref_weights=code_preferences)
                                
                                if large_options:
                                    idx_to_action_idx = {act.index: i for i, act in enumerate(brain.current_valid_actions)}
                                    
                                    # 为大池子打分
                                    scored_options = []
                                    for opt in large_options:
                                        score = 1e-4 # 基础探索分
                                        if len(opt['bytes']) == 4 and struct.unpack('<i', opt['bytes'])[0] == -1:
                                            score += 0.05 # 取消键额外权重
                                        elif msg_type in [18, 24]:
                                            for p in opt.get('places', []):
                                                if p in idx_to_action_idx: score += probs[idx_to_action_idx[p]]
                                        else:
                                            if len(opt['bytes']) > 1:
                                                for idx in opt['bytes'][1:]:
                                                    if idx in idx_to_action_idx: score += probs[idx_to_action_idx[idx]]
                                        opt['weight'] = score
                                        scored_options.append(opt)
                                        
                                    # 倾向性采样: 大池 -> 小池 100
                                    sample_size = min(100, len(scored_options))
                                    weights = np.array([o['weight'] for o in scored_options], dtype=np.float64)
                                    weights_sum = weights.sum()
                                    if weights_sum > 0: weights /= weights_sum
                                    else: weights = np.ones_like(weights) / len(weights)
                                    
                                    sampled_indices = np.random.choice(len(scored_options), size=sample_size, replace=False, p=weights)
                                    
                                    # 组装缓存的小池子
                                    from data_types import GameAction
                                    new_valid_actions = []
                                    for i, idx in enumerate(sampled_indices):
                                        opt = scored_options[idx]
                                        desc = f"Macro Action {i}"
                                        if len(opt['bytes']) == 4 and struct.unpack('<i', opt['bytes'])[0] == -1: desc = "Cancel"
                                        act = GameAction(action_type=msg_type, index=i, desc_str=desc)
                                        if 'locs' in opt: setattr(act, 'macro_targets', opt['locs'])
                                        if 'places' in opt: setattr(act, 'macro_places', opt['places'])
                                        setattr(act, 'decision_bytes', opt['bytes'])
                                        new_valid_actions.append(act)
                                        
                                    current_macro_pool = new_valid_actions
                                else:
                                    current_macro_pool = brain.current_valid_actions
                            
                            # 覆盖环境的合法选项 (保证 Retry 时池子永远不变)
                            brain.current_valid_actions = current_macro_pool

                        # [防死锁强力干预] 获取网络专属的数字错题本
                        clean_ignore_idx_list = [idx for idx in current_step_ignore_idx_list if 0 <= idx < len(brain.current_valid_actions)]
                        
                        try:
                            # Pass 2: 正式编码 (带着 100 个优质套餐，或者普通单选题)
                            snap = brain.get_snapshot(env)
                            player = snap.global_data.to_play
                            # 修复旧版错写成 agent.encoder 的 Bug，确保能兼容不同权重的对手
                            tensor_dict = current_agent.encoder.encode(snap, player_id=player) 
                            
                            # 核心防死锁：把在错题本里的选项强制 Mask 掉，迫使 AI 在下次重试时选择“备胎”
                            for bad_idx in clean_ignore_idx_list:
                                if tensor_dict['act_mask'].dim() == 2:
                                    if bad_idx < tensor_dict['act_mask'].shape[1]:
                                        tensor_dict['act_mask'][0, bad_idx] = False
                                else:
                                    if bad_idx < tensor_dict['act_mask'].shape[0]:
                                        tensor_dict['act_mask'][bad_idx] = False
                            
                            num_valid = len(brain.current_valid_actions)
                            is_exhausted = False
                            if tensor_dict['act_mask'].dim() == 2:
                                if not tensor_dict['act_mask'][0, :num_valid].any(): is_exhausted = True
                            else:
                                if not tensor_dict['act_mask'][:num_valid].any(): is_exhausted = True
                                
                            if is_exhausted:
                                # 获取具体被拒绝的样本名字
                                rejected_samples = [brain.current_valid_actions[i].desc_str for i in clean_ignore_idx_list[:8]]
                                
                                # 尝试挖掘上游触发指令的卡片
                                trigger_card = "未知机制/阶段动作"
                                from card_reader import card_db
                                if brain.chain_stack:
                                    # 如果有连锁正在处理，堆栈顶部就是发动效果的卡
                                    top_code = brain.chain_stack[-1]['code']
                                    trigger_card = f"连锁处理中 -> 【{card_db.get_card_name(top_code)}】"
                                elif brain.history_stack:
                                    # 如果没有连锁，看看上一张发动的卡是什么
                                    top_code = brain.history_stack[0]['code']
                                    trigger_card = f"最近发动 -> 【{card_db.get_card_name(top_code)}】"
                                elif last_interaction_msg:
                                    trigger_card = f"消息流头部 {last_interaction_msg[:4]}"
                                    
                                error_msg = (
                                    f"\n❌ [决策闭环崩溃] Worker {worker_id} 陷入死胡同！\n"
                                    f"   -> 崩溃阶段: Type {msg_type}\n"
                                    f"   -> 上下文线索: {trigger_card}\n"
                                    f"   -> 原始合法选项数: {num_valid}\n"
                                    f"   -> 已被错题本拉黑的选项: {rejected_samples}...\n"
                                    f"   -> 诊断: 选项已被全数屏蔽 (Mask 全黑)。请检查 gamestate 是否生成了乱序列表导致 Mask 错位，或 C++ 引擎是否根本不接受这些选项。"
                                )
                                raise RuntimeError(error_msg)

                            # 防止嵌入层越界崩溃
                            max_idx = tensor_dict['act_card_idx'].shape[-1] if len(tensor_dict['act_card_idx'].shape) > 2 else 119
                            bad_mask = tensor_dict['act_card_idx'] == -1
                            tensor_dict['act_card_idx'][bad_mask] = 120
                            tensor_dict['act_card_idx'] = torch.clamp(tensor_dict['act_card_idx'], 0, 120)
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
                                    
                                    '''
                                    if is_training_agent:
                                        rnd_reward = current_agent.net.rnd(v_input.to(device)).item()
                                        current_agent.net.update_rnd_stats(v_input.to(device))
                                    else:
                                        rnd_reward = 0.0
                                    '''

                                # 新增 .detach()，确保放入字典的张量干干净净
                                action_idx = action_idx.detach().cpu()
                                log_prob = log_prob.detach().cpu()
                                value = value.detach().cpu()
                                infer_dict = {k: v.detach().cpu() for k, v in tensor_dict.items()}
                                
                            if not snap.valid_actions:
                                raise RuntimeError("AI面临空选项池，主动放弃决策请求RuleBot兜底")
                            
                            step_reward = 0.0

                            sel_idx = action_idx.item()
                            last_decision_index = sel_idx
                            if sel_idx < len(snap.valid_actions):
                                chosen = snap.valid_actions[sel_idx]
                            else:
                                trigger_card = "未知时点/阶段"
                                from card_reader import card_db
                                if brain.chain_stack:
                                    trigger_card = f"连锁中 -> 【{card_db.get_card_name(brain.chain_stack[-1]['code'])}】"
                                elif brain.history_stack:
                                    trigger_card = f"上一动 -> 【{card_db.get_card_name(brain.history_stack[0]['code'])}】"
                                
                                print(f"⚠️ [网络幻觉] 源头: {trigger_card} | 引擎需要 {len(snap.valid_actions)} 个选项，网络却强行越界点选了槽位 [{sel_idx}]。已拦截并施加惩罚。")
                                chosen = snap.valid_actions[0] # 保底使用第一个选项
                                step_reward -= 0.005 # 告诉网络不要做梦
                                ai_is_broken[player] = True
                                # 强制修正送入 PPO 内存的标记，确保梯度的因果关系一致
                                action_idx = torch.tensor(0, dtype=torch.long) 
                                # (log_prob 会有略微偏差，但在 PPO 中会被容忍)

                            if snap.global_data.turn_count != current_turn:
                                current_turn = snap.global_data.turn_count
                                turn_steps = 0 # 切换回合，步数清零！
                            turn_steps += 1

                            # 软性时间惩罚 (Soft Enrage)
                            # 1. 超长盘疲劳：20回合以内绝对自由，20回合以后才施加极微小的推力
                            if current_turn > 20:
                                step_reward -= 0.0001

                            if turn_steps > 200:
                                step_reward -= 0.0005 # 如果单回合超过200步，说明可能卡在某个阶段了，施加额外惩罚
                                
                            # 2. 哈希查重：如果你当前的合法动作列表和上次一模一样，说明局势根本没推进
                            current_hash = "|".join([f"{a.action_type}_{a.index}" for a in snap.valid_actions])
                            if current_hash != last_valid_hash:
                                last_valid_hash = current_hash
                                loop_tracker.clear() # 局势变了，重置嫌疑
                                
                            # 记录该局势下，这个选项被选了多少次
                            loop_key = f"{current_hash}_{sel_idx}"
                            loop_tracker[loop_key] = loop_tracker.get(loop_key, 0) + 1
                            
                            # 3. 执法时刻：不论是 Cancel 还是什么神仙卡，同一个局势下连选 5 次，必是恶意拖延
                            if loop_tracker[loop_key] >= 5:
                                step_reward -= 0.005  # 重罚！

                            # --- 动作翻译 ---
                            resp = b''#如果数据包返回b''，说明 ai没有操作
                            
                            # 拦截代打：如果是包含预计算字节的套餐，直接越过打包器发送
                            if hasattr(chosen, 'decision_bytes') and chosen.decision_bytes:
                                resp = chosen.decision_bytes
                            else:
                                # 正常的单卡操作，交给 AI Bot 打包
                                resp = current_agent._pack_response(chosen, msg_type=msg_type, msg_args=msg[1:])
                            
                            env.send_action(resp)
                            last_decision_value = resp
                            last_decision_idx = sel_idx 
                            msg_queue = [] 
                            
                            # 内存脱水压缩：彻底抛弃 Tensor，转化为极其轻量的 Numpy 数组
                            compressed_obs = {}
                            for k, v in infer_dict.items():
                                cpu_v = v.cpu()
                                if cpu_v.dtype in [torch.long, torch.int64, torch.int32]:
                                    compressed_obs[k] = cpu_v.numpy().astype(np.int16)
                                elif cpu_v.dtype == torch.float32:
                                    compressed_obs[k] = cpu_v.numpy().astype(np.float16)
                                elif cpu_v.dtype == torch.bool:
                                    compressed_obs[k] = cpu_v.numpy().astype(np.bool_)
                                else:
                                    compressed_obs[k] = cpu_v.numpy()
                            
                            # 注意：只有训练代理的数据才需要放入 game_buffer 供学习
                            if is_training_agent and not ai_is_broken[player]:
                                game_buffer[player].append({
                                    'obs': compressed_obs,
                                    # 核心修复：连同外层变量一起切断 Tensor 关联
                                    'action': action_idx.cpu().numpy().astype(np.int16),
                                    'log_prob': log_prob.cpu().numpy().astype(np.float16),
                                    'value': value.cpu().numpy().astype(np.float16),
                                    'step_reward': step_reward
                                })
                                collected_steps += 1
                            
                            ep_steps += 1
                            ai_handled = True
                            last_act_time = time.time()
                        except OSError as e:
                            last_msg_type = msg_type
                            last_resp = resp
                            last_valid_actions = brain.current_valid_actions
                            if "access violation" in str(e).lower() or "exception" in str(e).lower():
                                    # 绝对不能让 RuleBot 接管，否则会引发无限死循环 DDOS 攻击主进程
                                    log_fatal_crash(worker_id, "【AI 神经网络】", last_msg_type, last_resp, e, last_valid_actions)
                                    winner = 1 - player  # 惩罚导致崩溃的 AI
                                    win_reason = 0
                                    return
                            else:
                                raise e
                            
                        except Exception as e: 
                            # 纯 Python 逻辑报错，允许 RuleBot 兜底
                            print(f"\n❌ [Worker {worker_id}] AI 逻辑计算崩溃: {e}")
                            if is_training_agent and not ai_is_broken[player]:
                                ai_is_broken[player] = True
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

                        rule_bot.sync_valid_actions(brain.current_valid_actions)
                        resp = rule_bot.get_rule_decision(p, msg_type, msg, brain, ignore_actions=clean_ignore_list)
                        last_decision_value = resp

                        # 修复：不再强制计算 RuleBot 的 log_prob，也不把它放进 game_buffer
                        # 从 AI 的视角来看，这相当于对手或者系统强制替它走了一步，是环境状态的跃迁。
                        # [黑匣子] 记录 RuleBot 动作
                        last_msg_type = msg_type
                        last_resp = resp
                        last_valid_actions = brain.current_valid_actions
                        
                        try:
                            env.send_action(resp)
                            msg_queue = []
                        except OSError as e:
                            if "access violation" in str(e).lower() or "exception" in str(e).lower():
                                # 如果 RuleBot 发送后引擎崩溃，同样强行打断循环
                                log_fatal_crash(worker_id, "【RuleBot 兜底逻辑】", last_msg_type, last_resp, e, last_valid_actions)
                                winner = 1 - player
                                win_reason = 0
                                return
                            else:
                                raise e
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
                        print(f"💀 [Worker {worker_id}] AI 连续 3 局无法行动，触发物理自毁防止死锁！可能卡组有问题: 训练方={d1_name}, 对手={d2_name}")
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
                            if ai_is_broken[train_p_id]:
                                final_reward = -1.0
                            elif turns <= 40: 
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
                            # 没到最大步数就停了，说明是引擎死锁了，这是致命的逻辑错误
                            stats['deadlocks'] += 1
                            final_reward = -1.0 
                        else:
                            # 达到了最大步数，说明是说书超时
                            stats['timeouts'] += 1
                            final_reward = 0.0
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
                        
                        delta = actual_rew + gamma * next_value - traj[t]['value'].item()
                        last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
                        advantages.insert(0, last_gae_lam)
                        next_value = traj[t]['value'].item()
                    
                    # 动态初始化 obs 的预分配空间
                    if not columns['obs'] and len(traj) > 0:
                        for k, v in traj[0]['obs'].items():
                            pt_dtype = torch.from_numpy(v).dtype
                            columns['obs'][k] = torch.zeros((max_len,) + v.shape[1:], dtype=pt_dtype)
                    
                    # 极限内存优化：流式填入数据，并立刻弹出销毁历史记录
                    for t in range(len(traj)):
                        obs_dict = traj[t].pop('obs') # 弹出并销毁，边填边释放历史数据，杜绝双重占用！
                        for k, v in obs_dict.items():
                            columns['obs'][k][ptr] = torch.from_numpy(v[0])
                        del obs_dict
                        
                        columns['action'][ptr] = torch.as_tensor(traj[t]['action'])
                        columns['log_prob'][ptr] = torch.as_tensor(traj[t]['log_prob'])
                        
                        ret_val = advantages[t] + traj[t]['value'].item()
                        columns['return'][ptr] = ret_val
                        columns['advantage'][ptr] = advantages[t]
                        
                        ptr += 1

        # 死锁保护
        tmp_file = f"tmp_rollout_iter_{iteration}_worker_{worker_id}.pt"
        if 'columns' not in locals() or ptr < (target_steps * 0.6):
            print(f"❌ [Worker {worker_id}] 数据严重残缺 ({ptr}/{target_steps})，拒绝生成诈尸 PT！")
            return

        batch_data = {'obs': {}}

        # 1. 逐个切片提取 obs 特征，提取完立刻焚毁原件，节省一半内存
        for k in list(columns['obs'].keys()):
            batch_data['obs'][k] = columns['obs'][k][:ptr].clone()
            del columns['obs'][k] # 切完立刻抹杀旧数据的这部分

        # 2. 提取外层特征并立即销毁
        for k in ['action', 'log_prob', 'return', 'advantage']:
            batch_data[k] = columns[k][:ptr].clone()
            del columns[k]

        # 释放只剩空壳的内存池字典
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