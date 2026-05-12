# ==================================================================================
#  Galatea PPO Trainer (Deep Fix Version)
#  结合了 run_self_play 的鲁棒交互逻辑与 PPO 的训练管线
# ==================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import time
import datetime
import numpy as np
import random
import gc
import threading
import queue
import glob
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import struct

from galatea_env import GalateaEnv
from gamestate import MessageParser, DuelState
from worker import worker_process
from ai_bot import AiBot
import deck_utils
import rule_bot 
from feature_encoder import MAX_CARDS as MAX_SEQ_LEN
# [新增] 头部
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 屏蔽 PyTorch 2.0 啰嗦的警告

# === 超参数配置 ===
LR = 1e-4               # Learning Rate: 步长，决定学得有多快（太快容易震荡）
GAMMA = 0.998            # Discount Factor: 远视眼程度，0.998表示很看重未来收益
GAE_LAMBDA = 0.95       # GAE参数: 平衡方差和偏差的关键
UPDATE_TIMESTEPS = 2048 # Batch Size: 攒多少经验升一级
EPOCHS = 4              # PPO Update Epochs:同一批数据反复榨取几次
MINIBATCH_SIZE = 128    # Mini-batch: 梯度下降时的切片大小
CLIP_EPS = 0.2          # PPO Clip: 限制更新幅度，防止学“飘”了
ENTROPY_COEF = 0.03     # 熵正则化: 鼓励探索，防止过早收敛到局部最优
VALUE_LOSS_COEF = 0.5   # 价值网络权中
MAX_EPISODE_STEPS = 2000 # 单局最大步数，防止死循环
    
def worker_wrapper(worker_id, net_config, weights, deck_dir, target_steps, device, req_q, resp_q, result_q, worker_timeout=300, gamma=GAMMA, gae_lambda=GAE_LAMBDA, num_workers=4):
    """用于原生 Process 的安全包装器，防止 DLL 崩溃导致主进程死锁"""
    try:
        from worker import worker_process
        res = worker_process(worker_id, net_config, weights, deck_dir, target_steps, device, req_q, resp_q, worker_timeout=worker_timeout, gamma=gamma, gae_lambda=gae_lambda, num_workers=num_workers)
        result_q.put((worker_id, res))
    except Exception as e:
        print(f"Worker {worker_id} 发生异常退出: {e}")
        result_q.put((worker_id, None))

class PPOTrainer:
    def __init__(self, save_dir="./models", deck_dir="./decks", net_config=None, resume_path=None, 
                 update_timesteps=4096, mini_batch_size=512, num_workers=4, worker_device='cuda', async_infer=False, compile_model=True, worker_timeout=300, gamma=0.998, lr=1e-4, entropy=0.03, gae_lambda=0.95, clip_eps=0.2): # <--- 新增
        self.save_dir = save_dir
        self.deck_dir = deck_dir
        self.update_timesteps = update_timesteps
        self.mini_batch_size = mini_batch_size
        self.num_workers = num_workers
        self.worker_device = worker_device
        self.async_infer = async_infer
        self.scaler = torch.cuda.amp.GradScaler()
        os.makedirs(save_dir, exist_ok=True)

        # 默认配置
        if net_config is None:
            net_config = {'d_model': 256, 'n_heads': 4, 'n_layers': 2, 'vocab_size': 20000}
        
        self.net_config = net_config 
        self.worker_timeout = worker_timeout
        self.gamma = gamma
        self.lr = lr
        self.entropy = entropy
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps

        # 硬件检查与黑科技自动配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp_dtype = torch.float16 # 默认兼容老显卡
        self.enable_compile = compile_model

        if self.device.type == 'cuda':
            # 1. 开启 TF32 (30/40/50系福利)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("⚡ [Auto] Ampere+ 架构: 已开启 TF32 加速")
            
            # 2. 开启 BF16 (5070Ti 杀手锏)
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                print("⚡ [Auto] 支持 BF16: 已启用 BFloat16 混合精度")
            else:
                print("ℹ️ [Auto] 不支持 BF16: 回退至 Float16")

        # 探针：看看实例化模型前，显卡到底还有多少空余显存
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"🖥️ 探针报告 -> 当前显卡可用显存: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
        
        # 清理上一次意外中断留下的临时文件
        print("🧹 正在清理上一次训练遗留的临时通讯文件...")
        # 修复：加上 .pt* 就能同时匹配 .pt 和 .pt.tmp
        for f in glob.glob("tmp_rollout_*.pt*") + glob.glob("tmp_weights_*.pt*"):
            try: os.remove(f)
            except Exception as e: 
                print(f"[trainer]⚠️ 无法删除临时文件 {f}: {e}")

        # 初始化 AI
        self.agent = AiBot(device=self.device, net_config=self.net_config)
        # 内存布局优化
        #self.agent.net = self.agent.net.to(memory_format=torch.channels_last)
        self.agent.net.train()

        # 编译优化
        if self.enable_compile and self.device.type == 'cuda':
            try:
                print("🚀 [编译] 正在启用 torch.compile...")
                self.agent.net = torch.compile(self.agent.net, mode='reduce-overhead')
            except Exception as e:
                print(f"⚠️ 编译跳过: {e}")

        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=self.lr)
        # 初始化 Scaler (BF16时其实不需要缩放，但为了代码通用，我们保留它)
        # 使用新版 API，指定设备类型 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))
        # 初始化环境 (仅用于参数查询等，不参与对战)
        self.env = GalateaEnv()
        
        self.global_step = 0
        self.iteration = 0
        self.train_step = 0
        
        # [静态内存池] 预先设定最大容量，彻底消灭内存碎片
        self.buffer_allocated = False
        self.merged_memory = None
        # 容量 = 目标步数 + 容错余量
        self.max_buffer_steps = self.update_timesteps + (self.num_workers * 1000)

        # 初始化 TensorBoard 记录器
        # log_dir 可以按时间戳命名，方便区分不同次训练
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"./runs/galatea_{time_str}")
        self.run_id = time_str
        print(f"📊 TensorBoard 日志将保存至: ./runs/galatea_{time_str}")

        # Windows 必须设置
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass # 已经设置过了就算了

        if self.async_infer:
            # 只创建一次
            self.req_q = mp.Queue(maxsize=self.num_workers * 4)
            self.resp_qs = [mp.Queue(maxsize=4) for _ in range(self.num_workers)]
            self.server_stop_event = threading.Event()
            # 增加一个当前轮次的标记，用于过滤过期请求
            self.current_iter_id = mp.Value('i', 0) 
            
            self.infer_thread = threading.Thread(
                target=self.inference_server, 
                args=(self.req_q, self.resp_qs, self.server_stop_event, self.current_iter_id),
                daemon=True
            )
            self.infer_thread.start()
        else:
            self.req_q = None
            self.resp_qs = [None] * self.num_workers

        # 恢复训练逻辑简化为调用函数
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

    def load_checkpoint(self, path):
        """
        独立的加载函数，增强了对编译模型的兼容性
        """
        print(f"📥 正在从 {path} 恢复训练...")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # 1. 架构配置检查与重建
            if 'net_config' in checkpoint:
                saved_config = checkpoint['net_config']
                # 如果存档配置和当前不同，必须重建网络
                if saved_config != self.net_config:
                    print(f"⚠️ 架构变更! 重建网络: {saved_config}")
                    self.net_config = saved_config
                    
                    # 重新初始化 Agent
                    self.agent = AiBot(device=self.device, net_config=self.net_config)
                    # 重新应用内存布局优化
                    self.agent.net = self.agent.net.to(memory_format=torch.channels_last)
                    
                    # 如果启用了编译，重建后需要再次编译
                    if self.enable_compile and self.device.type == 'cuda':
                         try: 
                             self.agent.net = torch.compile(self.agent.net, mode='reduce-overhead')
                         except Exception as e:
                             print(f"[trainer]⚠️ 编译跳过: {e}")
                    
                    self.agent.net.train()
                    # 重建优化器 (因为网络参数对象变了)
                    self.optimizer = optim.Adam(self.agent.net.parameters(), lr=self.lr)

            # 2. 权重加载 (核心修复：处理 compile 产生的前缀)
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                # 如果存档里的 key 有 _orig_mod. 前缀 (说明是编译版存的)，去掉它
                # 这样无论是编译版还是普通版网络，都能匹配上
                name = k.replace("_orig_mod.", "")
                new_state_dict[name] = v
            
            # 使用 strict=False 容忍细微差异
            self.agent.net.load_state_dict(new_state_dict, strict=False)
            
            # 3. 恢复优化器和 Scaler
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            self.iteration = checkpoint.get('iteration', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.train_step = checkpoint.get('train_step', 0) # 不再用数学公式逆推
            
            print(f"✅ 恢复成功! Iter: {self.iteration} | Train_Step: {self.train_step}")

        except Exception as e:
            print(f"❌ 恢复失败: {e}")

    def collect_rollouts(self):
        print(f"📥 [Iter {self.iteration}] 唤醒 {self.num_workers} 个工人 | 目标: {self.update_timesteps} 步")
        t0 = time.time()

        raw_weights = self.agent.net.state_dict()
        cpu_weights = {k.replace("_orig_mod.", ""): v.cpu() for k, v in raw_weights.items()}
        
        # 将权重写进硬盘，禁止通过多进程参数传递 Tensor
        weight_file = f"tmp_weights_iter_{self.iteration}.pt"
        torch.save(cpu_weights, weight_file)

        del cpu_weights
        del raw_weights
        import gc; gc.collect()

        # --- 新增：联盟训练对手选择逻辑 ---
        # 扫描 models 文件夹下的所有历史存档
        historical_models = glob.glob(os.path.join(self.save_dir, "galatea_iter_*.pth"))
        
        worker_opp_configs = []
        for i in range(self.num_workers):
            roll = random.random()
            if roll < 0.15:
                # 15% 几率：对抗 RuleBot (基准锚点)
                worker_opp_configs.append({"mode": "rule", "type": "rule", "path": None})
            elif roll < 0.40 and historical_models:
                # 25% 几率：对抗历史随机模型 (防止策略退化)
                worker_opp_configs.append({"mode": "ai", "type": "hist", "path": random.choice(historical_models)})
            else:
                # 60% 几率：自对局 (追求当前最优对抗)
                worker_opp_configs.append({"mode": "ai", "type": "self", "path": weight_file})
        # --------------------------------------
        
        steps_per_worker = max(200, self.update_timesteps // self.num_workers)

        if self.async_infer:
            self.current_iter_id.value = self.iteration  # 更新当前轮次标记，供服务器过滤过期请求
            for q in self.resp_qs:
                while not q.empty():
                    try: 
                        q.get_nowait()
                    # 过滤掉过期的响应
                    except(queue.Empty, EOFError, OSError, ValueError):
                        pass

        # 🚀 启动工人
        processes = []
        for i in range(self.num_workers):
            # 根据配置，决定是否把通讯管道发给工人
            
            p = mp.Process(target=worker_process, args=(
                i, 
                self.iteration,
                self.net_config, 
                weight_file,         
                self.deck_dir, 
                steps_per_worker,
                self.worker_device,
                self.req_q, self.resp_qs[i],        # 恢复队列传参
                worker_opp_configs[i],
                self.worker_timeout,
                self.gamma,
                self.gae_lambda,
                self.num_workers
            ))
            p.daemon = True
            p.start()
            processes.append(p)
            
        print(f"   ... {'异步 GPU Server' if self.async_infer else '纯本地 CPU'} 运算中 ...")
        
        # 等待工人自然死亡
        start_wait = time.time()
        while time.time() - start_wait < self.worker_timeout:
            all_dead = True
            for p in processes:
                if p.is_alive(): all_dead = False
            if all_dead: break
            time.sleep(1.0)

        # 时间到，执行收尸
        for p in processes:
            if p.is_alive():
                print(f"⏳ [Trainer] 经过 {self.worker_timeout} 秒仍有 Worker 未完成采集。触发安全超时截断机制，终止冗余进程。")
                p.terminate() 
                p.join()
            try: 
                p.close() # 强制释放 Windows 进程句柄
            except Exception as e: 
                print(f"[trainer]⚠️ 无法关闭 Worker 进程 (可能已被系统回收): {e}")

        # 把权重文件删掉
        try: os.remove(weight_file)
        except Exception as e: 
            print(f"[trainer]⚠️ 无法删除权重文件 {weight_file}: {e}")

        # 3. 直接去硬盘收割
        file_list = []
        for i in range(self.num_workers):
            tmp_file = f"tmp_rollout_iter_{self.iteration}_worker_{i}.pt"
            if os.path.exists(tmp_file):
                file_list.append(tmp_file)

        if not file_list:
            print("❌ 所有 Worker 均未能产出数据！")
            return None
        
        print(f"⚡ 正在合并 {len(file_list)} 个数据块...")
        
        total_rewards = []
        total_lens = []

        ## 统计先后手胜率的全局计数器
        t_stats = {k: 0 for k in [
            'wins_all_first', 'games_all_first', 'wins_all_second', 'games_all_second',
            'wins_self_first', 'games_self_first', 'wins_self_second', 'games_self_second',
            'wins_hist', 'games_hist', 'wins_rule', 'games_rule','deadlocks', 'timeouts', 'draws'
        ]}

        t_deck_stats = {}
        all_match_records = []

        cursor = 0

        # 逐文件加载并直接注入预分配的静态内存池，避免一次性加载全部数据导致内存暴涨
        for i, f in enumerate(file_list):
            try:
                # 1. 仅将当前 1 个 Worker 的数据加载到物理内存
                data = torch.load(f, map_location='cpu', weights_only=False)
                s = data['action'].shape[0]

                # 2. 提取并累加统计信息
                r = data.get('avg_rew', np.array([0.0]))[0]
                l = data.get('avg_len', np.array([0.0]))[0]
                if r != 0: total_rewards.append(r)
                if l != 0: total_lens.append(l)
                
                # 极速自动累加所有字段
                for k in t_stats.keys():
                    t_stats[k] += data.get(f'stats_{k}', np.array([0]))[0]

                # 提取卡组雷达数据并聚合
                for record in data.get('deck_records', []):
                    record['iteration'] = self.iteration
                    all_match_records.append(record) # 需要在读取 file_list 前声明 all_match_records = []
                    dk = record['my_deck']
                    if dk not in t_deck_stats:
                        t_deck_stats[dk] = {'g_1st':0, 'w_1st':0, 'g_2nd':0, 'w_2nd':0}

                    if record['is_first']:
                        t_deck_stats[dk]['g_1st'] += 1
                        if record['is_win']: t_deck_stats[dk]['w_1st'] += 1
                    else:
                        t_deck_stats[dk]['g_2nd'] += 1
                        if record['is_win']: t_deck_stats[dk]['w_2nd'] += 1

                # 3. 如果是第一个文件，初始化静态内存池
                if not self.buffer_allocated:
                    print(f"📦 [内存管理] 首次初始化主进程静态内存池 (容量: {self.max_buffer_steps} 步)...")
                    self.merged_memory = {'obs': {}}
                    self.merged_memory['action'] = torch.empty(self.max_buffer_steps, dtype=data['action'].dtype)
                    self.merged_memory['log_prob'] = torch.empty(self.max_buffer_steps, dtype=data['log_prob'].dtype)
                    self.merged_memory['return'] = torch.empty(self.max_buffer_steps, dtype=data['return'].dtype)
                    self.merged_memory['advantage'] = torch.empty(self.max_buffer_steps, dtype=data['advantage'].dtype)
                    
                    for k, v in data['obs'].items():
                        shape = list(v.shape)
                        shape[0] = self.max_buffer_steps
                        self.merged_memory['obs'][k] = torch.empty(*shape, dtype=v.dtype)
                        
                    self.buffer_allocated = True

                # 4. 防御性截断，防止溢出缓冲池
                if cursor + s > self.max_buffer_steps:
                    print(f"⚠️ 警告: 采集步数({cursor+s})超过缓冲容量({self.max_buffer_steps})，自动截断！")
                    s = self.max_buffer_steps - cursor
                    if s <= 0: 
                        del data
                        break

                # 5. 零拷贝游标注入
                self.merged_memory['action'][cursor:cursor+s] = data['action'][:s]
                self.merged_memory['log_prob'][cursor:cursor+s] = data['log_prob'][:s]
                self.merged_memory['return'][cursor:cursor+s] = data['return'][:s]
                self.merged_memory['advantage'][cursor:cursor+s] = data['advantage'][:s]
                
                for k in self.merged_memory['obs'].keys():
                    self.merged_memory['obs'][k][cursor:cursor+s] = data['obs'][k][:s]
                    
                cursor += s
                
                # 6. 阅后即焚：彻底断开引用，强制回收这几百MB内存，并删掉硬盘文件
                del data
                try: os.remove(f)
                except Exception as e: print(f"[trainer]⚠️ 清理残余文件 {f} 失败: {e}")

                import gc; gc.collect()
                
            except Exception as e:
                print(f"❌ 读取/合并文件 {f} 失败: {e}") # 拒绝静默报错

        t_cost = time.time() - t0
        avg_rew = np.mean(total_rewards) if total_rewards else 0.0
        avg_len = np.mean(total_lens) if total_lens else 0.0
        print(f"✅ 采集完成! 耗时: {t_cost:.1f}s | 样本: {cursor} | Avg Reward: {avg_rew:.2f}")
        
        # 计算并写入胜率图表
        # 绘图面板分类排版
        def safe_div(w, g): return w / max(1, g)
        
        # 1. 总体大盘 (League_Overall)
        self.writer.add_scalar("League_Overall/WinRate_Total", safe_div(t_stats['wins_all_first'] + t_stats['wins_all_second'], t_stats['games_all_first'] + t_stats['games_all_second']), self.iteration)
        self.writer.add_scalar("League_Overall/WinRate_First", safe_div(t_stats['wins_all_first'], t_stats['games_all_first']), self.iteration)
        self.writer.add_scalar("League_Overall/WinRate_Second", safe_div(t_stats['wins_all_second'], t_stats['games_all_second']), self.iteration)

        # 2. 内战水平 (League_Self) - 注：由于是打自己，这个理应无限逼近 50%
        self.writer.add_scalar("League_Self/WinRate_Total", safe_div(t_stats['wins_self_first'] + t_stats['wins_self_second'], t_stats['games_self_first'] + t_stats['games_self_second']), self.iteration)
        self.writer.add_scalar("League_Self/WinRate_First", safe_div(t_stats['wins_self_first'], t_stats['games_self_first']), self.iteration)
        self.writer.add_scalar("League_Self/WinRate_Second", safe_div(t_stats['wins_self_second'], t_stats['games_self_second']), self.iteration)

        # 3. 外战水平 (League_Opponent) - 展现真实统治力的地方！
        self.writer.add_scalar("League_Opponent/Historical_AI", safe_div(t_stats['wins_hist'], t_stats['games_hist']), self.iteration)
        self.writer.add_scalar("League_Opponent/RuleBot", safe_div(t_stats['wins_rule'], t_stats['games_rule']), self.iteration)
        
        # 新增：废局率雷达 (占总对局的百分比)
        total_games = max(1, t_stats['games_all_first'] + t_stats['games_all_second'])
        self.writer.add_scalar("Rollout/Deadlock_Rate", t_stats['deadlocks'] / total_games, self.iteration)
        self.writer.add_scalar("Rollout/Timeout_Rate", t_stats['timeouts'] / total_games, self.iteration)
        self.writer.add_scalar("Rollout/Draw_Rate", t_stats['draws'] / total_games, self.iteration)

        # 卡组专属雷达面板 (Deck_WinRate)
        for key, ds in t_deck_stats.items():
            g_total = ds['g_1st'] + ds['g_2nd']
            w_total = ds['w_1st'] + ds['w_2nd']
            
            # TensorBoard 会自动根据 key 的斜杠 "/" 创建子文件夹
            if g_total > 0:
              self.writer.add_scalar(f"Deck_WinRates/{key}_Total", w_total / g_total, self.iteration)
            if ds['g_1st'] > 0:
                self.writer.add_scalar(f"Deck_WinRates/{key}_First", ds['w_1st'] / ds['g_1st'], self.iteration)
            if ds['g_2nd'] > 0:
                self.writer.add_scalar(f"Deck_WinRates/{key}_Second", ds['w_2nd'] / ds['g_2nd'], self.iteration)

        self.writer.add_scalar('Rollout/Average_Reward', avg_rew, self.iteration)
        self.writer.add_scalar('Rollout/Average_Length', avg_len, self.iteration)

        # WebUI 数据脱水：将本轮卡组胜率抛出给前端，零性能损耗
        if all_match_records:
            web_data_dir = "./web_data"
            os.makedirs(web_data_dir, exist_ok=True)
            csv_filename = f"match_history_{self.run_id}.csv"
            csv_path = os.path.join(web_data_dir, csv_filename)
            
            df_new = pd.DataFrame(all_match_records)
            try:
                if not os.path.exists(csv_path):
                    df_new.to_csv(csv_path, index=False, mode='w', encoding='utf-8')
                else:
                    df_new.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8')
            except Exception as e:
                print(f"⚠️ [WebUI] CSV 数据库写入冲突: {e}")

        self.global_step += cursor
        
        return cursor # 核心改动：不再返回内存大字典，而是返回有效步数！
    
    def inference_server(self, req_q, resp_qs, stop_event,iter_tracker):
        """
        [封包极速版] 接收压平的 Tensor，在 GPU 显存内进行光速切片
        """
        print("🚀 [Server] 异步推断服务器已启动，等待 Worker 请求...")
        self.agent.net.eval()
        while not stop_event.is_set():
            try:
                requests = []
                try:
                    # 1. 尝试获取第一个请求
                    item = req_q.get(timeout=0.05)
                    # 解包：现在是 3 个元素
                    wid, msg_iter, numpy_dict = item
                    
                    # 防污染安检：如果请求轮次落后于主线程轮次，直接丢弃
                    if msg_iter >= iter_tracker.value:
                        requests.append((wid, numpy_dict)) # 安检通过，剥离 iter_id 放入处理列表
                    
                    # 2. 尝试凑齐这批次的其它请求
                    while len(requests) < self.num_workers:
                        try:
                            item = req_q.get_nowait()
                            wid, msg_iter, numpy_dict = item
                            
                            # 同样进行防污染安检
                            if msg_iter >= iter_tracker.value:
                                requests.append((wid, numpy_dict))
                        except queue.Empty:
                            break
                except queue.Empty:
                    pass
                
                if not requests:
                    continue
                    
                # --- 1. 光速解包并恢复张量 ---
                worker_ids = [r[0] for r in requests]
                batch_obs = {}
                for k in requests[0][1].keys():
                    #把 np.stack 换成 np.concatenate，并且指定 axis=0
                    stacked = np.concatenate([r[1][k] for r in requests], axis=0)

                    # [ Fail-Fast 安检哨卡 ] 不洗白数据，一旦发现 NaN 或 Inf，当场抓出内鬼
                    if stacked.dtype in [np.float16, np.float32]:
                        if np.isnan(stacked).any() or np.isinf(stacked).any():
                            # 找出到底是哪个 Worker 传来的毒药
                            poisoned_workers = [req[0] for req in requests if np.isnan(req[1][k]).any() or np.isinf(req[1][k]).any()]
                            # 直接抛出异常，触发底下的 except 熔断
                            raise ValueError(f"🛑 发现致命毒素！Worker {poisoned_workers} 传输了包含 NaN/Inf 的极端超界特征 '{k}'！")
                    
                    # 按原样恢复成 PyTorch 认识的类型
                    if stacked.dtype == np.bool_:
                        tensor = torch.from_numpy(stacked).to(torch.bool)
                    elif stacked.dtype == np.float16:
                        tensor = torch.from_numpy(stacked).to(torch.float32) # 网络计算需要 fp32
                    else:
                        tensor = torch.from_numpy(stacked).to(torch.long)
                        
                    batch_obs[k] = tensor.to(self.device, non_blocking=True)
                
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    with torch.no_grad():
                        # 加上 v_inputs，接收第5个返回值
                        actions, log_probs, _, values, v_inputs = self.agent.get_action_and_value_from_tensor(batch_obs, None)
                        # 直接在 GPU 服务端计算出 RND 奖励(暂时舍弃)
                        #rnd_rewards = self.agent.net.rnd(v_inputs)
                        #self.agent.net.update_rnd_stats(v_inputs)
                
                # --- 4. 组装回传封包 ---
                # 新增 .detach()，彻底斩断与 GPU 计算图的最后一点阴阳联系
                packed_returns = torch.stack([
                    actions.to(torch.float32), 
                    log_probs.to(torch.float32), 
                    values.squeeze(-1).to(torch.float32),
                    torch.zeros_like(actions, dtype=torch.float32) #暂时舍弃rnd,用0占位
                ], dim=1).detach().cpu()
                
                for i, wid in enumerate(worker_ids):
                    # 转为 numpy 数组发回给 Worker
                    try:
                        resp_qs[wid].put(packed_returns[i].numpy(), timeout=1.0)
                    except queue.Full:
                        pass # 忽略死掉的 Worker

            except Exception as e:
                # 当批次中出现毒药数据(如 NaN)，拦截异常，仅抛弃这一批次
                print(f"\n🚨 [Server 局部熔断] 推断批次出现异常，已自动丢弃并恢复服务: {e}")
                # import traceback; traceback.print_exc() # 嫌吵可以注释掉堆栈打印
                time.sleep(0.1) # 稍微缓一下继续服务

    def update_policy(self, total_steps):
        """
        [安全防爆版] 将数据留在 CPU，每次只切片 mini_batch 送进 GPU
        """
        if total_steps == 0: return
        print("🔥 Training PPO (Action Head Mode)...")
        
        # 直接从静态缓冲池中切出有效数据
        cpu_obs = self.merged_memory['obs']
        cpu_actions = self.merged_memory['action'][:total_steps]
        cpu_log_probs = self.merged_memory['log_prob'][:total_steps]
        cpu_returns = self.merged_memory['return'][:total_steps]
        cpu_advantages = self.merged_memory['advantage'][:total_steps]

        # [优化 1] 全局优势归一化，稳定训练方向
        if len(cpu_advantages) > 1:
            adv_mean = cpu_advantages.mean()
            adv_std = cpu_advantages.std()
            
            # 严密防范方差除零 NaN
            if torch.isnan(adv_std) or adv_std < 1e-5:
                print(f"⚠️ 警告: 优势标准差过小 ({adv_std:.5f})，已自动跳过归一化以防止 NaN")
                cpu_advantages = cpu_advantages - adv_mean
            else:
                cpu_advantages = (cpu_advantages - adv_mean) / (adv_std + 1e-8)
        
        batch_size = cpu_actions.shape[0]

        for _ in range(EPOCHS):
            indices = torch.randperm(batch_size)
            
            # 每次只循环处理 mini_batch (比如 1024 个)
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                
                # 只有切出来的这 1024 个数据，才 .to(self.device) 上 GPU！
                mb_obs = {}
                for k, v in cpu_obs.items():
                    cpu_t = v[mb_idx]
                    if cpu_t.is_floating_point():
                        mb_obs[k] = cpu_t.to(device=self.device, dtype=torch.float32, non_blocking=True)
                    elif 'mask' in k:
                        mb_obs[k] = cpu_t.to(device=self.device, dtype=torch.bool, non_blocking=True)
                    else:
                        mb_obs[k] = cpu_t.to(device=self.device, dtype=torch.long, non_blocking=True)
                
                mb_actions = cpu_actions[mb_idx].to(self.device, dtype=torch.long, non_blocking=True)
                mb_old_log_probs = cpu_log_probs[mb_idx].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_returns = cpu_returns[mb_idx].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_advs = cpu_advantages[mb_idx].to(self.device, dtype=torch.float32, non_blocking=True)
                

                # --- 网络前向传播与反向传播 (完全保持原样) ---
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    logits, values, v_input = self.agent.net(mb_obs)
                    values = values.squeeze(1)

                    # 计算 RND 预测误差损失，让 Predictor 学习当前状态(暂时舍弃)
                    #rnd_loss = self.agent.net.rnd(v_input.detach()).mean()

                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                    
                    # 拆解 Loss，方便我们在图表里监控
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss_fn = nn.SmoothL1Loss()
                    value_loss = value_loss_fn(values, mb_returns)
                    entropy_loss = -entropy.mean()
                    
                    loss = policy_loss + VALUE_LOSS_COEF * value_loss + self.entropy * entropy_loss # + rnd_loss 暂时舍弃

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.train_step % 20 == 0:
                    self.writer.add_scalar('Train/Total_Loss', loss.item(), self.train_step)
                    self.writer.add_scalar('Train/Policy_Loss', policy_loss.item(), self.train_step) # 策略偏移
                    self.writer.add_scalar('Train/Value_Loss', value_loss.item(), self.train_step)   # 价值预测准确度
                    self.writer.add_scalar('Train/Entropy', entropy.mean().item(), self.train_step)  # 探索欲 (如果急剧降到0，说明AI变傻钻牛角尖了)
                self.train_step += 1

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.agent.net.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()

    def run_training_loop(self, max_iterations=1000):
        print(f"🚦 Starting PPO Training Loop...")
        # 如果是恢复训练，max_iterations 应该是“再练多少轮”或者“练到多少轮”
        # 这里假设是“总轮数”，所以如果恢复时已经是 1000，需要把目标设大一点
        target_iter = max_iterations
        if self.iteration >= target_iter:
            target_iter += 1000
            print(f"⚠️ 当前轮数已达目标，自动追加 1000 轮 (Target: {target_iter})")

        while self.iteration < target_iter:
            self.iteration += 1
            iter_start = time.time()
            
            # 1. 采集 (现在返回的是总步数)
            total_steps = self.collect_rollouts()
            
            # 2. 优化 (只有当样本足够时才更新)
            if total_steps is not None and total_steps >= self.mini_batch_size:
                self.update_policy(total_steps)
                # 单轮训练结束，清理主进程15GB 的静态内存池
                print("🧹 [内存调度] 训练完成，摧毁主进程内存池...")
                if self.merged_memory is not None:
                    self.merged_memory.clear()
                    del self.merged_memory
                
                self.merged_memory = None
                self.buffer_allocated = False  # 让下一轮 collect_rollouts 重新申请
                
                # 强制呼叫系统底层的垃圾车
                import gc
                gc.collect()
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()

            else:
                print(f"⚠️ 样本不足 ({total_steps if total_steps else 0} < {self.mini_batch_size})，跳过本轮训练")
            
            # 3. 保存 (打包保存)
            if self.iteration % 10 == 0:
                path = f"{self.save_dir}/galatea_iter_{self.iteration}.pth"
                
                # 在保存的源头剥离编译前缀，确保存档是纯净标准版
                raw_state = self.agent.net.state_dict()
                clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
                
                # 补全生命周期字段，确保 TensorBoard 曲线 100% 无缝对接
                checkpoint = {
                    'model_state_dict': clean_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'net_config': self.net_config, 
                    'iteration': self.iteration,
                    'train_step': self.train_step,   # [新增] TensorBoard 的 X 轴坐标
                    'global_step': self.global_step,  # [新增] 环境交互总步数
                    'gamma': self.gamma,
                    'lr': self.lr,
                    'entropy': self.entropy,
                    'gae_lambda': self.gae_lambda,
                    'clip_eps': self.clip_eps
                }
                torch.save(checkpoint, path)
                print(f"💾 Model saved: {path}")
            
            dt = time.time() - iter_start
            print(f"⏱️ Iteration {self.iteration} finished in {dt:.1f}s")

        print("🏁 训练结束！")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.run_training_loop()