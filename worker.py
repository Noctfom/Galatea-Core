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
import zmq
import psutil
import sys

from galatea_env import GalateaEnv
from gamestate import MessageParser, DuelState
from ai_bot import AiBot
import deck_utils
import rule_bot
import warnings # [新增]
# [新增] 屏蔽 PyTorch 的 Nested Tensor 警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

if sys.platform == 'win32':
    # Windows 不完全支持 IPC，使用 TCP
    ZMQ_ADDR = "tcp://127.0.0.1:" 
else:
    # Linux 完美支持 IPC，走 /tmp 内存盘极速通信
    ZMQ_ADDR = "ipc:///tmp/galatea_zmq_"

# 状态与消息定义
STATE_CHANGE_MSGS = {40, 41, 50, 53, 54, 55, 56, 60, 61, 62, 70, 90, 91, 92, 94}
INTERACTION_MSGS = {10, 11, 15, 16, 18, 19, 20, 22, 23, 24, 26, 130, 131, 132, 133}
AI_MANAGED_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 140, 141, 142, 143]
DECISION_MSGS = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 130, 131, 132, 133, 140, 141, 142, 143]

# GAE 参数 (和 Trainer 保持一致)
GAMMA = 0.998
GAE_LAMBDA = 0.95
MAX_EPISODE_STEPS = 1500

def setup_optimal_cpu_threads():
    # 获取物理核心数
    physical_cores = psutil.cpu_count(logical=False) or 1
    # 获取逻辑线程数
    logical_threads = psutil.cpu_count(logical=True) or 1
    
    # 计算超线程倍率 (通常是 2)
    ht_ratio = logical_threads // physical_cores
    
    threads_per_worker = ht_ratio 
    
    torch.set_num_threads(threads_per_worker)
    torch.set_num_interop_threads(1)
    
    return threads_per_worker

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

def worker_process(worker_id, iteration, net_config, weight_file, deck_dir, target_steps, device='cpu',zmq_port=55555,opp_config=None, worker_timeout=300, gamma=GAMMA, gae_lambda=GAE_LAMBDA, num_workers=4,shared_buffers=None, shared_outputs=None, worker_events=None, use_onnx=False,shared_logits=None, standard_core=False):
    
    try:
        from system_logger import setup_global_logger
        # 让每个 Worker 生成自己专属的日志文件，比如 worker_0_2023xxxx.log
        setup_global_logger(prefix=f"worker_{worker_id}")
    except ImportError:
        pass

    import gamestate
    if standard_core:
        gamestate.CORE_HAS_GHOST_BYTE = False
        print(f"🔧 [Worker {worker_id}] 协议自适应：已关闭幽灵定界符 (Standard Core Mode)")
    else:
        gamestate.CORE_HAS_GHOST_BYTE = True

    # =========================================================================
    #  [防卡死] 禁用 Windows 崩溃弹窗
    # 告诉操作系统：如果 OCGCore 发生致命写越界(Segfault)，直接静默杀死进程，不要弹窗也不要生成错误报告，以免训练被打断导致堆砌死锁
    if os.name == 'nt':
        import ctypes
        # SEM_FAILCRITICALERRORS (0x0001) | SEM_NOGPFAULTERRORBOX (0x0002) | SEM_NOOPENFILEERRORBOX (0x8000)
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    # =========================================================================
    
    setup_optimal_cpu_threads()

    seed = (int(time.time() * 1000) % (2**31)) + (os.getpid() * 100) + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # [新增 ZMQ 连接初始化]
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    # 极速超时配置（防死锁），单位为毫秒
    socket.setsockopt(zmq.IDENTITY, f"worker_{worker_id}".encode('utf-8'))
    socket.setsockopt(zmq.RCVTIMEO, 60000)
    socket.setsockopt(zmq.SNDTIMEO, 15000)
    socket.connect(f"{ZMQ_ADDR}{zmq_port}")

    def build_ort_inputs(session, t_dict):
        inputs = {}
        for node in session.get_inputs():
            if node.name in t_dict:
                val = t_dict[node.name].numpy()
                # 使用 str().lower() 包含匹配，才能兼容所有 ONNX 版本
                ntype = str(node.type).lower()
                if 'int64' in ntype: val = val.astype(np.int64)
                elif 'int32' in ntype: val = val.astype(np.int32)
                elif 'int16' in ntype: val = val.astype(np.int16)
                elif 'int8' in ntype: val = val.astype(np.int8)
                elif 'float' in ntype: val = val.astype(np.float32)
                elif 'bool' in ntype: val = val.astype(np.bool_)
                inputs[node.name] = val
        return inputs

    use_onnx_p1 = False
    ort_session_p1 = None

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
            opp_agent = AiBot(device='cpu', net_config=net_config)
            opp_path = opp_config["path"]
            if opp_path and os.path.exists(opp_path):
                opp_agent.load_model(opp_path)
            else:
                cpu_state_dict = {k: v.cpu() for k, v in agent.net.state_dict().items()}
                opp_agent.net.load_state_dict(cpu_state_dict, strict=False)
                opp_agent.net.eval()

        time.sleep(worker_id * 0.4) # 让 0~N 号工人错开 0.4 秒启动
        env = GalateaEnv()
        collected_steps = 0
        
        consecutive_ai_fails = 0 # 死亡熔断计数器

        worker_start_time = time.time()
        # 给 Worker 留出充足的保存时间 (至少 30s)，防止 torch.save 大文件时被 p.terminate() 抢先杀掉
        max_worker_uptime = float(worker_timeout) - 30.0
        
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

        #提前拉起绝对连续的静态内存池
        max_len = target_steps + MAX_EPISODE_STEPS + 100
        columns = {
            'obs': {}, 
            'action': torch.zeros(max_len, dtype=torch.long),
            'log_prob': torch.zeros(max_len, dtype=torch.float32),
            'return': torch.zeros(max_len, dtype=torch.float32),
            'advantage': torch.zeros(max_len, dtype=torch.float32)
        }
        global_ptr = 0      # 全局写入指针
        ep_start_ptr = 0    # 当前对局的起始指针

        gc.collect()
        
        for k, v in shared_buffers[worker_id].items():
            shape = (max_len,) + v.shape 
            columns['obs'][k] = torch.empty(shape, dtype=v.dtype)

        # print(f"👷 Worker {worker_id} 启动 | 目标: {target_steps} 步")

        while collected_steps < target_steps:
            if time.time() - worker_start_time > max_worker_uptime:
                break

            # --- [挂载对手引擎] 解决历史模型穿越的 Bug ---
            if use_onnx and not use_onnx_p1:
                if opp_config.get("type") == "hist" and opp_config.get("path"):
                    # 如果是打历史模型，精准定位那个轮次的 .onnx
                    hist_onnx_path = opp_config["path"].replace(".pth", ".onnx")
                    if os.path.exists(hist_onnx_path):
                        try:
                            import onnxruntime as ort
                            sess_options = ort.SessionOptions()
                            sess_options.intra_op_num_threads = torch.get_num_threads()
                            ort_session_p1 = ort.InferenceSession(hist_onnx_path, sess_options, providers=['CPUExecutionProvider'])
                            use_onnx_p1 = True
                            print(f"🕰️ [Worker {worker_id}] 历史对手 ONNX 引擎 ({os.path.basename(hist_onnx_path)}) 挂载成功！")
                        except Exception:
                            pass

            perf_ledger = {
                'steps': 0,
                't_cpp_env': 0.0,    # C++ OCGCore 执行与 ctypes 格子解包 (含 get_snapshot 内部的 sync_active_field)
                't_encoder': 0.0,    # FeatureEncoder.encode 矩阵转换耗时
                't_zmq_p0': 0.0,    
                't_cpu_p1': 0.0,     
                't_pass1_cpu': 0.0, 
                't_rule_bot': 0.0    # RuleBot 穷举多选套餐与本地打分耗时
            }

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
                        t_start = time.time() # 计时器1
                        raw_data = env.step()
                        perf_ledger['t_cpp_env'] += (time.time() - t_start) # 计时器1
                        if raw_data and len(raw_data) > 500 * 1024: 
                            print(f"🚨 [防爆截断] Worker {worker_id} 引擎吐出畸形海量数据({len(raw_data)/1024:.1f}KB)，强行摧毁对局防止 RAM 溢出！")
                            break
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
                    t_start = time.time()  #计时器1

                    current_snap = brain.get_snapshot(env)
                    player = current_snap.global_data.to_play

                    perf_ledger['t_cpp_env'] += (time.time() - t_start) # 计时器1
                    
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
                        
                        # ==================================================================================
                        # [Two-Pass 架构] 拦截多选/排序等复杂指令，进行意图打分与降维 (NumPy 极限向量化版)
                        # ==================================================================================
                        if msg_type in [15, 18, 20, 22, 23, 24, 25]:
                            if current_macro_pool is None:
                                # [前置意图感知] 先问问网络喜欢哪张卡
                                code_preferences = {}
                                try:
                                    # 📍 桩点 B1：精准分离 Pass 1 的表征生成耗时
                                    t_enc_p1 = time.time()
                                    dict_pass1 = current_agent.encoder.encode(current_snap, player_id=player)
                                    perf_ledger['t_encoder'] += (time.time() - t_enc_p1)
                                    
                                    # 📍 桩点 C1：精准分离 Pass 1 的本地推理耗时
                                    t_infer_p1 = time.time()
                                    # 【大一统分流点 A】判断此轮推演是否具备全实时训练资格
                                    # 如果当前是主特工在操作（P0），或者是自对局模式下的对手（P1 Self），一律走 ZMQ 绑定 GPU 中枢
                                    if is_training_agent or opp_config["mode"] == "ai" and opp_config["type"] == "self":
                                        # 零拷贝填入超导通道
                                        for k, v in dict_pass1.items():
                                            if k in shared_buffers[worker_id]:
                                                shared_buffers[worker_id][k].copy_(v.squeeze(0))
                                        
                                        # 激活操作系统级信号量，呼叫服务端批处理
                                        try:
                                            socket.send(str(worker_id).encode('utf-8'))
                                            socket.recv()
                                        except zmq.error.Again:
                                            print(f"🛰️ [ZeroMQ 断联] Worker {worker_id} Pass1 超时，重建连接...")
                                            socket.close(linger=0)
                                            socket = context.socket(zmq.REQ)
                                            socket.setsockopt(zmq.IDENTITY, f"worker_{worker_id}".encode('utf-8'))
                                            socket.setsockopt(zmq.RCVTIMEO, 60000)
                                            socket.setsockopt(zmq.SNDTIMEO, 15000)
                                            socket.connect(f"{ZMQ_ADDR}{zmq_port}")
                                            raise RuntimeError("Pass1 ZMQ 首次通讯超时")
                                        
                                        # 极限极速：直接从 120 维超广阔共享大矩阵中捞取全量 Logits 并激活 Softmax
                                        raw_logits = shared_logits[worker_id].numpy()
                                        probs = torch.softmax(torch.tensor(raw_logits), dim=-1).numpy()
                                        perf_ledger['t_zmq_p0'] += (time.time() - t_infer_p1)
                                        
                                    # 如果是历史存档老模型（P1 Hist），执行宿主本地独立剥离推演（ONNX 优先，CPU 兜底）
                                    else:
                                        if use_onnx_p1 and ort_session_p1 is not None:
                                            ort_inputs = build_ort_inputs(ort_session_p1, dict_pass1)
                                            ort_outs = ort_session_p1.run(None, ort_inputs) 
                                            probs = torch.softmax(torch.tensor(ort_outs[0]), dim=-1).squeeze(0).numpy()
                                        else:
                                            with torch.no_grad():
                                                v_input = {k: v.to(current_agent.device) for k, v in dict_pass1.items() if isinstance(v, torch.Tensor)}
                                                action_logits, _, _ = current_agent.net(v_input)
                                                probs = torch.softmax(action_logits, dim=-1).squeeze(0).cpu().numpy()
                                        perf_ledger['t_pass1_cpu'] += (time.time() - t_infer_p1)
                                        
                                    for i, act in enumerate(brain.current_valid_actions):
                                        if hasattr(act, 'code'):
                                            code_preferences[act.code] = max(code_preferences.get(act.code, 0), probs[i])
                                except Exception as e:
                                    print(f"⚠️ [Worker {worker_id}] 意图感知阶段发生错误，跳过打分: {e}")
                                
                                t_rule_anchor = time.time()# 计时器4
                                large_options = rule_bot.get_macro_options(msg_type, msg[1:], brain, limit=5000, pref_weights=code_preferences)
                                
                                if large_options:
                                    # 构建 256 宽度的 NumPy 极速查表，彻底消灭字典 Lookup 损耗
                                    action_prob_map = np.zeros(256, dtype=np.float64)
                                    for i, act in enumerate(brain.current_valid_actions):
                                        if 0 <= act.index < 256:
                                            action_prob_map[act.index] = float(probs[i])
                                            
                                    scored_options = []
                                    is_place_msg = msg_type in [18, 24]
                                    
                                    # 斩断纯 Python 嵌套循环，全面转为 C 层次向量化运算
                                    for opt in large_options:
                                        score = 1e-4  # 基础探索分
                                        opt_bytes = opt['bytes']
                                        
                                        # 消除 struct.unpack：小端 -1 等价于 b'\xff\xff\xff\xff'
                                        if len(opt_bytes) == 4 and opt_bytes == b'\xff\xff\xff\xff':
                                            score += 0.05
                                        elif is_place_msg:
                                            # 物理格子类：直接利用 NumPy 批量求和
                                            places = opt.get('places', [])
                                            if places:
                                                score += action_prob_map[places].sum()
                                        else:
                                            # 卡片/素材选择类：用 np.frombuffer 零拷贝转为 uint8 数组批量求和
                                            if len(opt_bytes) > 1:
                                                card_idxs = np.frombuffer(opt_bytes, dtype=np.uint8, offset=1)
                                                score += action_prob_map[card_idxs].sum()
                                                
                                        opt['weight'] = score
                                        scored_options.append(opt)
                                        
                                    # 扩容到 120，完美饱和特征编码器最大深度，提供更广阔的 RL 探索视野
                                    sample_size = min(120, len(scored_options))
                                    weights = np.array([o['weight'] for o in scored_options], dtype=np.float64)
                                    weights_sum = weights.sum()
                                    if weights_sum > 0: 
                                        weights /= weights_sum
                                    else: 
                                        weights = np.ones_like(weights) / len(weights)
                                    
                                    sampled_indices = np.random.choice(len(scored_options), size=sample_size, replace=False, p=weights)
                                    
                                    # 组装缓存的小池子
                                    from data_types import GameAction
                                    new_valid_actions = []
                                    for i, idx in enumerate(sampled_indices):
                                        opt = scored_options[idx]
                                        desc = f"Macro Action {i}"
                                        if len(opt['bytes']) == 4 and opt['bytes'] == b'\xff\xff\xff\xff': 
                                            desc = "Cancel"
                                        act = GameAction(action_type=msg_type, index=i, desc_str=desc)
                                        if 'locs' in opt: setattr(act, 'macro_targets', opt['locs'])
                                        if 'places' in opt: setattr(act, 'macro_places', opt['places'])
                                        setattr(act, 'decision_bytes', opt['bytes'])
                                        new_valid_actions.append(act)
                                        
                                    current_macro_pool = new_valid_actions
                                else:
                                    current_macro_pool = brain.current_valid_actions

                                perf_ledger['t_rule_bot'] += (time.time() - t_rule_anchor) # 计时器4
                            
                            brain.current_valid_actions = current_macro_pool
                            current_snap.valid_actions = current_macro_pool


                        # [防死锁强力干预] 获取网络专属的数字错题本
                        clean_ignore_idx_list = [idx for idx in current_step_ignore_idx_list if 0 <= idx < len(brain.current_valid_actions)]
                        
                        try:
                            t_enc_anchor = time.time()# 计时器2
                            # 修复旧版错写成 agent.encoder 的 Bug，确保能兼容不同权重的对手
                            tensor_dict = current_agent.encoder.encode(current_snap, player_id=player)
                            perf_ledger['t_encoder'] += (time.time() - t_enc_anchor) # 计时器2
                            
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
                                print(error_msg)
                                winner = 1 - player
                                break

                            # 防止嵌入层越界崩溃
                            max_idx = tensor_dict['act_card_idx'].shape[-1] if len(tensor_dict['act_card_idx'].shape) > 2 else 119
                            bad_mask = tensor_dict['act_card_idx'] == -1
                            tensor_dict['act_card_idx'][bad_mask] = 120
                            tensor_dict['act_card_idx'] = torch.clamp(tensor_dict['act_card_idx'], 0, 120)
                            tensor_dict['act_type'] = torch.clamp(tensor_dict['act_type'], 0, 255)
                            tensor_dict['act_desc'] = torch.clamp(tensor_dict['act_desc'], 0, 1023)
                            
                            # 混合推理架构
                            t_ipc_anchor = time.time()
                            
                            # 【大一统分流点 B】主特工回合或自对局对手，全部绑上超导通道
                            if is_training_agent or opp_config["mode"] == "ai" and opp_config["type"] == "self":
                                t_zmq_anchor = time.time()
                                for k, v in tensor_dict.items():
                                    if k in shared_buffers[worker_id]:
                                        shared_buffers[worker_id][k].copy_(v.squeeze(0))
                                try:
                                    socket.send(str(worker_id).encode('utf-8'))
                                    reply = socket.recv()
                                except zmq.error.Again:
                                    print(f"🛰️ [ZeroMQ 断联] Worker {worker_id} 超时，正在执行神经连接重建...")
                                    # [核心修复] 必须彻底销毁并重建 Socket，否则 REQ 状态机会永久锁死
                                    socket.close(linger=0)
                                    socket = context.socket(zmq.REQ)
                                    socket.setsockopt(zmq.IDENTITY, f"worker_{worker_id}".encode('utf-8'))
                                    socket.setsockopt(zmq.RCVTIMEO, 60000)
                                    socket.setsockopt(zmq.SNDTIMEO, 15000)
                                    socket.connect(f"{ZMQ_ADDR}{zmq_port}")
                                    
                                    print("ZMQ 首次通讯超时，已强制重启连接")
                                
                                # 从 4 维基础槽中拉取采样出的动作面包屑
                                res_array = shared_outputs[worker_id].numpy()
                                action_idx = torch.tensor(int(res_array[0]), dtype=torch.long)
                                log_prob = torch.tensor(res_array[1], dtype=torch.float32)
                                value = torch.tensor(res_array[2], dtype=torch.float32)
                                infer_dict = {k: v.detach().cpu() for k, v in tensor_dict.items()}
                                
                                # 精细化分账统计
                                if is_training_agent:
                                    perf_ledger['t_zmq_p0'] += (time.time() - t_zmq_anchor)
                                else:
                                    perf_ledger['t_cpu_p1'] += (time.time() - t_zmq_anchor)

                            # 历史老模型独立推演流
                            else:
                                t_p1_anchor = time.time() 
                                with torch.no_grad():
                                    if use_onnx_p1 and ort_session_p1 is not None:
                                        ort_inputs = build_ort_inputs(ort_session_p1, tensor_dict)
                                        ort_outs = ort_session_p1.run(None, ort_inputs) 
                                        
                                        action_logits = torch.tensor(ort_outs[0]).squeeze(0)
                                        value = torch.tensor(ort_outs[1]).squeeze(0)
                                        
                                        # 严格在局部执行非法动作遮罩隔离
                                        act_mask_tensor = tensor_dict['act_mask'].squeeze(0)
                                        action_logits[~act_mask_tensor] = -65000.0
                                        
                                        dist = torch.distributions.Categorical(logits=action_logits)
                                        action_idx = dist.sample()
                                        log_prob = dist.log_prob(action_idx)
                                    else:
                                        infer_dict = {k: v.detach().to(current_agent.device) for k, v in tensor_dict.items()}
                                        action_idx, log_prob, _, value, _ = current_agent.get_action_and_value_from_tensor(infer_dict, current_snap.valid_actions)
                                        action_idx = action_idx.squeeze(0)
                                        log_prob = log_prob.squeeze(0)
                                        value = value.squeeze(0)
                                
                                action_idx = action_idx.detach().cpu()
                                log_prob = log_prob.detach().cpu()
                                value = value.detach().cpu()
                                infer_dict = {k: v.detach().cpu() for k, v in tensor_dict.items()}
                                
                                perf_ledger['t_cpu_p1'] += (time.time() - t_p1_anchor) # 只记录 P1 耗时

                            if not current_snap.valid_actions:
                                raise RuntimeError("AI面临空选项池，主动放弃决策请求RuleBot兜底")
                            
                            step_reward = 0.0

                            sel_idx = action_idx.item()
                            last_decision_index = sel_idx
                            if sel_idx < len(current_snap.valid_actions):
                                chosen = current_snap.valid_actions[sel_idx]
                            else:
                                trigger_card = "未知时点/阶段"
                                from card_reader import card_db
                                if brain.chain_stack:
                                    trigger_card = f"连锁中 -> 【{card_db.get_card_name(brain.chain_stack[-1]['code'])}】"
                                elif brain.history_stack:
                                    trigger_card = f"上一动 -> 【{card_db.get_card_name(brain.history_stack[0]['code'])}】"
                                
                                print(f"⚠️ [网络幻觉] 源头: {trigger_card} | 引擎需要 {len(current_snap.valid_actions)} 个选项，网络却强行越界点选了槽位 [{sel_idx}]。已拦截并施加惩罚。")
                                chosen = current_snap.valid_actions[0] # 保底使用第一个选项
                                step_reward -= 0.005 # 告诉网络不要做梦
                                ai_is_broken[player] = True
                                # 强制修正送入 PPO 内存的标记，确保梯度的因果关系一致
                                action_idx = torch.tensor(0, dtype=torch.long) 
                                # (log_prob 会有略微偏差，但在 PPO 中会被容忍)

                            if current_snap.global_data.turn_count != current_turn:
                                current_turn = current_snap.global_data.turn_count
                                turn_steps = 0 # 切换回合，步数清零！
                            turn_steps += 1

                            # 软性时间惩罚 (Soft Enrage)
                            # 1. 超长盘疲劳：20回合以内绝对自由，20回合以后才施加极微小的推力
                            if current_turn > 20:
                                step_reward -= 0.0001

                            if turn_steps > 200:
                                step_reward -= 0.0005 # 如果单回合超过200步，说明可能卡在某个阶段了，施加额外惩罚
                                
                            # 2. 哈希查重：如果你当前的合法动作列表和上次一模一样，说明局势根本没推进
                            current_hash = "|".join([f"{a.action_type}_{a.index}" for a in current_snap.valid_actions])
                            if current_hash != last_valid_hash:
                                last_valid_hash = current_hash
                                loop_tracker.clear() # 局势变了，重置嫌疑
                                
                            # 记录该局势下，这个选项被选了多少次
                            loop_key = f"{current_hash}_{sel_idx}"
                            loop_tracker[loop_key] = loop_tracker.get(loop_key, 0) + 1
                            
                            # 同一个局势下连选 10 次，是恶意拖延
                            if loop_tracker[loop_key] >= 10:
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
                            
                            # 仅当训练代理操作时才录入数据，绝对防止指针错位与重复
                            if is_training_agent and not ai_is_broken[player]:
                                # 内存直写：把特征直接强行塞进连续的静态池坑位里
                                for k, v in infer_dict.items():
                                    columns['obs'][k][global_ptr] = v.squeeze(0)

                                # game_buffer 极速瘦身：只存最轻量的 Python 原生数字！
                                game_buffer[player].append({
                                    'action': int(action_idx.item() if isinstance(action_idx, torch.Tensor) else action_idx),
                                    'log_prob': float(log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob),
                                    'value': float(value.item() if isinstance(value, torch.Tensor) else value),
                                    'step_reward': float(step_reward)
                                })
                                
                                global_ptr += 1  # 只有有效训练步才推针
                                collected_steps += 1
                            
                            ep_steps += 1
                            ai_handled = True
                            last_act_time = time.time()

                            perf_ledger['steps'] += 1 #计时器总步数加一

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
                    
                    total_t = perf_ledger['t_cpp_env'] + perf_ledger['t_encoder'] + perf_ledger['t_zmq_p0'] + perf_ledger['t_cpu_p1'] + perf_ledger['t_pass1_cpu'] + perf_ledger['t_rule_bot']
                    safe_t = max(0.0001, total_t)
                    print(f"\n📊 [Worker {worker_id} 审计报告] 单局步数: {perf_ledger['steps']} | 交互总耗时: {total_t:.2f}s")
                    print(f"对手类型: {opp_type} | 胜负结果: {'胜' if winner == train_p_id else '负' if winner == opp_p_id else '平'} | 最终奖励: {final_reward:.2f}")
                    print(f"   ├── 🧱 [环境/底盘] C++执行与解包: {perf_ledger['t_cpp_env']:.2f}s ({perf_ledger['t_cpp_env']/safe_t*100:.1f}%)")
                    print(f"   ├── 📐 [表征/特征] Encoder 矩阵生成: {perf_ledger['t_encoder']:.2f}s ({perf_ledger['t_encoder']/safe_t*100:.1f}%)")
                    print(f"   ├── 📡 [多进程管道] ZMQ+GPU推理(P0): {perf_ledger['t_zmq_p0']:.2f}s ({perf_ledger['t_zmq_p0']/safe_t*100:.1f}%)")
                    print(f"   ├── 💻 [本地推理] P1对手CPU推演 : {perf_ledger['t_cpu_p1']:.2f}s ({perf_ledger['t_cpu_p1']/safe_t*100:.1f}%)")
                    print(f"   ├── 👁️ [意图感知] Pass1 预查表耗时 : {perf_ledger['t_pass1_cpu']:.2f}s ({perf_ledger['t_pass1_cpu']/safe_t*100:.1f}%)")
                    print(f"   └── 🧠 [参谋部/规则] RuleBot 穷举打分: {perf_ledger['t_rule_bot']:.2f}s ({perf_ledger['t_rule_bot']/safe_t*100:.1f}%)")

                    rewards = [0] * len(traj)
                    if rewards: rewards[-1] = final_reward

                    ep_total_reward = sum(item.get('step_reward', 0.0) for item in traj) + final_reward
                    episode_rewards.append(ep_total_reward)
                    
                    advantages = []
                    last_gae_lam = 0
                    # 修复：区分“真死”和“超时截断”
                    next_value = 0.0
                    if ep_steps >= MAX_EPISODE_STEPS: # 如果是超时被强行截断的
                        # 使用最后一步的价值网络预测值作为未来收益的保底，而不是直接视为 0，这样可以让模型学会在超时前尽可能积累价值，而不是放弃抵抗
                        next_value = traj[-1]['value']

                    # 终极内存防爆：预分配连续内存，告别 list 与 torch.cat
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
                    
                    # 逆序计算 GAE，从后往前填充预分配的内存池
                    for t in reversed(range(len(traj))):
                        # 获取这一步的面包屑奖励
                        s_rew = traj[t].get('step_reward', 0.0)
                        # 只有最后一步才加上胜负大奖 (rewards[t] 就是 final_reward)
                        actual_rew = s_rew + (rewards[t] if t == len(traj) - 1 else 0.0)
                        
                        delta = actual_rew + gamma * next_value - traj[t]['value']
                        last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
                        traj[t]['advantage'] = last_gae_lam
                        traj[t]['return'] = last_gae_lam + traj[t]['value']
                        
                        next_value = traj[t]['value']
                    
                    for i, t_data in enumerate(traj):
                        idx = ep_start_ptr + i
                        columns['action'][idx] = t_data['action']
                        columns['log_prob'][idx] = t_data['log_prob']
                        columns['return'][idx] = t_data['return']
                        columns['advantage'][idx] = t_data['advantage']
                    
                    game_buffer[train_p_id].clear()
                    ep_start_ptr = global_ptr

        # 死锁保护 - 只要有数据就保存，宁可少也要有
        tmp_file = f"tmp_rollout_iter_{iteration}_worker_{worker_id}.pt"
        if 'columns' not in locals() or ep_start_ptr < 10:
            print(f"❌ [Worker {worker_id}] 数据几乎为空 ({ep_start_ptr}/{target_steps})，跳过保存")
            return
        if ep_start_ptr < (target_steps * 0.6):
            print(f"⚠️ [Worker {worker_id}] 数据仅收集了 {ep_start_ptr}/{target_steps} 步 (<60%)，但仍然保存以便训练继续")

        batch_data = {'obs': {}}

        # 提取外层特征
        for k in ['action', 'log_prob', 'return', 'advantage']:
            batch_data[k] = columns[k][:ep_start_ptr].clone()
            columns[k] = None
            
        # 提取 obs 内部特征
        for k in shared_buffers[worker_id].keys():
            if k in columns['obs']:
                batch_data['obs'][k] = columns['obs'][k][:ep_start_ptr].clone()
                columns['obs'][k] = None
            else:
                # 这一局压根没触发过这个特征，用 zeros 安全补齐
                ref_tensor = shared_buffers[worker_id][k]
                shape = list(ref_tensor.shape)
                shape[0] = ep_start_ptr
                batch_data['obs'][k] = torch.zeros(shape, dtype=ref_tensor.dtype)
            
        columns.clear()
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
    
    finally:
        # =========================================================================
        # [内存安全清理] 显式释放所有大型张量，防止 Windows 内存泄漏
        # =========================================================================
        try:
            # 1. 清理静态内存池 columns
            if 'columns' in dir():
                if columns is not None:
                    # 先清理 obs 字典内的所有张量
                    if 'obs' in columns and columns['obs']:
                        for k in list(columns['obs'].keys()):
                            columns['obs'][k] = None
                        columns['obs'].clear()
                    # 清理外层张量
                    for k in list(columns.keys()):
                        columns[k] = None
                    columns.clear()
            
            # 2. 清理 game_buffer
            if 'game_buffer' in dir() and game_buffer is not None:
                for k in list(game_buffer.keys()):
                    if game_buffer[k]:
                        game_buffer[k].clear()
                game_buffer.clear()
            
            # 3. 清理 batch_data (如果存在)
            if 'batch_data' in dir() and batch_data is not None:
                if 'obs' in batch_data:
                    for k in list(batch_data['obs'].keys()):
                        batch_data['obs'][k] = None
                for k in list(batch_data.keys()):
                    batch_data[k] = None
                batch_data = None
            
            # 4. 清理 AI 代理
            if 'agent' in dir() and agent is not None:
                agent = None
            if 'opp_agent' in dir() and opp_agent is not None:
                opp_agent = None
            
            # 5. 强制垃圾回收
            gc.collect()
            
        except Exception as cleanup_err:
            print(f"⚠️ [Worker {worker_id}] 内存清理时发生异常 (已忽略): {cleanup_err}")
        
        # 关闭 ZMQ 连接
        try:
            socket.setsockopt(zmq.LINGER, 0) 
            socket.close(linger=0)
            context.term()
        except Exception:
            pass
        
        print(f"👋 Worker {worker_id} 任务圆满完成，强制释放内存...")
        os._exit(0)
