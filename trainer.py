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
import zmq
import shutil
import psutil
import sys

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import struct

from galatea_env import GalateaEnv
from gamestate import MessageParser, DuelState
from worker import worker_process
from ai_bot import AiBot
from checkpoint_utils import (
    canonical_model_state_dict,
    load_training_checkpoint,
    restore_model_state_strict,
)
import deck_utils
import rule_bot 
from feature_encoder import MAX_CARDS as MAX_SEQ_LEN
# [新增] 头部
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 屏蔽 PyTorch 2.0 啰嗦的警告

if sys.platform == 'win32':
    # Windows 不完全支持 IPC，使用 TCP
    ZMQ_ADDR = "tcp://127.0.0.1:" 
else:
    # Linux 完美支持 IPC，走 /tmp 内存盘极速通信
    ZMQ_ADDR = "ipc:///tmp/galatea_zmq_"

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
MAX_EPISODE_STEPS = 1500 # 单局最大步数，防止死循环
    
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
                 update_timesteps=4096, mini_batch_size=512, num_workers=4, worker_device='cpu', async_infer=False, compile_model=True, worker_timeout=300, gamma=0.998, lr=1e-4, entropy=0.03, gae_lambda=0.95, clip_eps=0.2, use_onnx=False, standard_core=False):
        self.save_dir = save_dir
        self.deck_dir = deck_dir
        self.update_timesteps = update_timesteps
        self.mini_batch_size = mini_batch_size
        self.num_workers = num_workers
        self.worker_device = worker_device
        self.async_infer = async_infer
        self.use_onnx = use_onnx
        self.server_stop_event = threading.Event()
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
        self.standard_core = standard_core

        # 硬件检查与黑科技自动配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp_dtype = torch.float16 # 默认兼容老显卡
        self.enable_compile = compile_model

        if self.device.type == 'cuda':
            # 1. 开启 TF32 (30/40/50系福利)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
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

        resume_checkpoint = None
        if resume_path:
            print(f"📥 正在从 {resume_path} 恢复训练...")
            resume_checkpoint = load_training_checkpoint(resume_path, map_location="cpu")
            self.net_config = resume_checkpoint['net_config']

        # 始终先在未编译模型上严格恢复参数，再创建优化器并启用编译。
        self.agent = AiBot(device=self.device, net_config=self.net_config)
        self.agent.net.train()
        if resume_checkpoint is not None:
            restore_model_state_strict(self.agent.net, resume_checkpoint)

        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=self.lr)
        # 初始化 Scaler (BF16时其实不需要缩放，但为了代码通用，我们保留它)
        # 使用新版 API，指定设备类型 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_dtype == torch.float16))

        self.global_step = 0
        self.iteration = 0
        self.train_step = 0

        if resume_checkpoint is not None:
            self._restore_training_state(resume_checkpoint)
            del resume_checkpoint

        # 编译必须发生在严格恢复之后；OptimizedModule 与基础模型共享参数对象。
        if self.enable_compile and self.device.type == 'cuda':
            try:
                print("🚀 [编译] 正在启用 torch.compile...")
                self.agent.net = torch.compile(self.agent.net, mode='default')
            except Exception as e:
                print(f"⚠️ 编译跳过: {e}")

        # 初始化环境 (仅用于参数查询等，不参与对战)
        self.env = GalateaEnv()
        
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
        os.environ['GALATEA_RUN_ID'] = self.run_id
        print(f"📊 TensorBoard 日志将保存至: ./runs/galatea_{time_str}")

        # Windows 必须设置
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass # 已经设置过了就算了

        # 根据 feature_encoder.py 的最大常数，精准定义共享内存规格
        self.shared_buffers = []
        self.shared_outputs = []
        self.shared_logits = []
        self.worker_events = [mp.Event() for _ in range(self.num_workers)]
        
        # 严密对齐特工的观测维度 specs
        input_specs = {
            'global': ((15,), torch.float32),
            'card_idx': ((120,), torch.long),
            'card_overlay_idx': ((120,), torch.long),
            'card_race': ((120,), torch.long),
            'card_attr': ((120,), torch.long),
            'card_setcodes': ((120, 4), torch.long),
            'card_feats': ((120, 66), torch.float32),
            'padding_mask': ((120,), torch.bool),
            
            # ---前场/后场/手牌 语义大脑皮层槽位 ---
            'sem_category': ((120, 8, 8), torch.int16),
            'sem_req': ((120, 8, 16), torch.int8),
            'sem_setcode': ((120, 8, 4), torch.int16),
            'sem_number': ((120, 8, 4), torch.float16),
            'sem_ref': ((120, 8, 4), torch.int32),
            'sem_race': ((120, 8, 4), torch.int16),
            'sem_attr': ((120, 8, 4), torch.int16),
            'sem_code_idx': ((120, 8), torch.long),  # [新增]
            'sem_mask': ((120, 8), torch.bool),
            
            'deck_idx': ((75,), torch.long),
            'deck_race': ((75,), torch.long),
            'deck_attr': ((75,), torch.long),
            'deck_setcodes': ((75, 4), torch.long),
            'deck_mask': ((75,), torch.bool),
            
            # --- 上帝视角卡组残像 语义槽位 ---
            'd_sem_category': ((75, 8, 8), torch.int16),
            'd_sem_req': ((75, 8, 16), torch.int8),
            'd_sem_setcode': ((75, 8, 4), torch.int16),
            'd_sem_number': ((75, 8, 4), torch.float16),
            'd_sem_ref': ((75, 8, 4), torch.int32),
            'd_sem_race': ((75, 8, 4), torch.int16),
            'd_sem_attr': ((75, 8, 4), torch.int16),
            'd_sem_code_idx': ((75, 8), torch.long), # [新增]
            'd_sem_mask': ((75, 8), torch.bool),
            
            'c_mask': ((12,), torch.bool),
            
            # --- 瞬间时点连锁堆栈 语义槽位 ---
            'c_sem_category': ((12, 8, 8), torch.int16),
            'c_sem_req': ((12, 8, 16), torch.int8),
            'c_sem_setcode': ((12, 8, 4), torch.int16),
            'c_sem_number': ((12, 8, 4), torch.float16),
            'c_sem_ref': ((12, 8, 4), torch.int32),
            'c_sem_race': ((12, 8, 4), torch.int16),
            'c_sem_attr': ((12, 8, 4), torch.int16),
            'c_sem_code_idx': ((12, 8), torch.long), # [新增]
            'c_sem_mask': ((12, 8), torch.bool),
            'h_mask': ((8,), torch.bool),
            
            # --- 历史施法雷达 语义槽位 ---
            'h_sem_category': ((8, 8, 8), torch.int16),
            'h_sem_req': ((8, 8, 16), torch.int8),
            'h_sem_setcode': ((8, 8, 4), torch.int16),
            'h_sem_number': ((8, 8, 4), torch.float16),
            'h_sem_ref': ((8, 8, 4), torch.int32),
            'h_sem_race': ((8, 8, 4), torch.int16),
            'h_sem_attr': ((8, 8, 4), torch.int16),
            'h_sem_code_idx': ((8, 8), torch.long), # [新增]
            'h_sem_mask': ((8, 8), torch.bool),
            'act_card_idx': ((120, 5), torch.long),
            'act_type': ((120,), torch.long),
            'act_desc': ((120,), torch.long),
            'act_mask': ((120,), torch.bool),
            'act_race': ((120,), torch.long),
            'act_attr': ((120,), torch.long),
            'act_code': ((120,), torch.long),
            'act_place': ((120, 5), torch.long),
        }

        print("🧠 [Shared Memory] 正在开辟高带宽零拷贝多进程超导通道...")
        for _ in range(self.num_workers):
            # 输入槽
            buf = {}
            for k, (shape, dtype) in input_specs.items():
                t = torch.zeros(shape, dtype=dtype)
                t.share_memory_() # 赋予跨进程起死回生之力
                buf[k] = t
            self.shared_buffers.append(buf)
            
            # 输出槽 (包含: action, log_prob, value, rnd_reward 占位)
            out_t = torch.zeros((4,), dtype=torch.float32)
            out_t.share_memory_()
            self.shared_outputs.append(out_t)

            # 开辟 120 维全量动作分数共享内存，专为降维套餐设计
            logits_t = torch.zeros((120,), dtype=torch.float32)
            logits_t.share_memory_()
            self.shared_logits.append(logits_t)

        self.pinned_batch_buffers = {}
        print("📌 [Pinned Memory] 正在为主进程注入高效锁页多路聚合批处理器...")
        for k, (shape, dtype) in input_specs.items():
            # 一次性开辟物理连续、支持极速 DMA 异步搬运的 (num_workers, *dims) 锁页张量
            self.pinned_batch_buffers[k] = torch.zeros((self.num_workers, *shape), dtype=dtype).pin_memory()

        # [新增 ZMQ 配置]
        self.zmq_port = 55555  # 固定一个端口

        if self.async_infer:
            # 启动服务器，传入端口号
            self.infer_thread = threading.Thread(
                target=self.inference_server, 
                args=(self.shared_buffers, self.shared_outputs, self.server_stop_event, self.zmq_port),
                daemon=True
            )
            self.infer_thread.start()
        else:
            self.req_q = None

    def _restore_training_state(self, checkpoint):
        """Restore non-model training state after strict model loading."""
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.iteration = int(checkpoint['iteration'])
        self.global_step = int(checkpoint['global_step'])
        self.train_step = int(checkpoint['train_step'])
        print(f"✅ 恢复成功! Iter: {self.iteration} | Train_Step: {self.train_step}")

    def collect_rollouts(self):
        print(f"📥 [Iter {self.iteration}] 唤醒 {self.num_workers} 个工人 | 目标: {self.update_timesteps} 步")
        t0 = time.time()

        cpu_weights = canonical_model_state_dict(self.agent.net, to_cpu=True)
        
        # 将权重写进硬盘，禁止通过多进程参数传递 Tensor
        weight_file = f"tmp_weights_iter_{self.iteration}.pt"
        torch.save(cpu_weights, weight_file)

        del cpu_weights
        import gc; gc.collect()

        # --- 联盟训练对手选择逻辑 ---
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
            pass # 异步模式不需要额外的标记机制，服务器会根据 ZMQ 请求自动识别和处理最新的权重文件
            #self.current_iter_id.value = self.iteration  # 更新当前轮次标记，供服务器过滤过期请求

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
                self.zmq_port,
                worker_opp_configs[i],
                self.worker_timeout,
                self.gamma,
                self.gae_lambda,
                self.num_workers,
                self.shared_buffers,  # 新增直传
                self.shared_outputs,  
                self.worker_events,
                self.use_onnx,
                self.shared_logits,
                self.standard_core
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
            p.join(timeout=5.0)  # 增加等待时间，让系统有足够时间回收内存
            try: 
                p.close() # 强制释放 Windows 进程句柄
            except Exception as e: 
                print(f"[trainer]⚠️ 无法关闭 Worker 进程 (可能已被系统回收): {e}")
        
        orphan_tmps = glob.glob(f"tmp_rollout_iter_{self.iteration}_worker_*.pt.tmp")
        for f in orphan_tmps:
            try:
                os.remove(f)
                print(f"🧹 [清理] 已回收被强制截断的临时残骸: {f}")
            except Exception as e:
                print(f"⚠️ [清理] 无法删除临时残骸 {f}: {e}")

        # =========================================================================
        # [内存安全回收] 强制等待系统回收 Worker 进程的内存
        # =========================================================================
        gc.collect()
        
        # 检查系统可用内存，如果过低则等待更长时间
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            if available_gb < 1.0:  # 如果可用内存小于 1GB
                print(f"⚠️ [内存警告] 系统可用内存较低 ({available_gb:.1f}GB)，等待内存回收...")
                time.sleep(5.0)  # 额外等待 5 秒
                gc.collect()
                # 再次检查
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024**3)
                if available_gb < 1.0:
                    print(f"🚨 [内存危机] 可用内存仍然不足 ({available_gb:.1f}GB)，建议减少 Worker 数量或增加系统内存！")
        except Exception as mem_check_err:
            print(f"⚠️ 内存检查失败: {mem_check_err}")

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
                        for remain_f in file_list[i:]:
                            try: os.remove(remain_f)
                            except Exception as e: 
                                print(f"[trainer]⚠️ 清理残余文件 {remain_f} 失败: {e}")
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
        
        return cursor # 核心改动：不再返回内存大字典，而是返回有效步数

    def inference_server(self, shared_buffers, shared_outputs, stop_event, zmq_port, shared_logits=None):
        print(f" [Server] ZeroMQ ROUTER 硬件级微批处理中枢启动 (Port: {zmq_port})")
        
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
        socket.setsockopt(zmq.SNDHWM, 100)
        socket.setsockopt(zmq.RCVHWM, 100)
        socket.bind(f"{ZMQ_ADDR}{zmq_port}")
        
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        
        self.agent.net.eval()
        
        while not stop_event.is_set():
            try:
                socks = dict(poller.poll(100))
                if socket in socks:
                    requests = []
                    addresses = {}
                    
                    while True:
                        try:
                            addr, empty, payload = socket.recv_multipart(flags=zmq.NOBLOCK)
                            wid = int(payload.decode('utf-8'))
                            requests.append(wid)
                            addresses[wid] = addr
                        except zmq.Again:
                            break
                    
                    start_wait = time.perf_counter()
                    while len(requests) < self.num_workers and (time.perf_counter() - start_wait) < 0.003:
                        if poller.poll(0):
                            try:
                                addr, empty, payload = socket.recv_multipart(flags=zmq.NOBLOCK)
                                wid = int(payload.decode('utf-8'))
                                if wid not in requests:
                                    requests.append(wid)
                                    addresses[wid] = addr
                            except zmq.Again:
                                pass
                    
                    if not requests:
                        continue

                    # --- 4. GPU 极限推推演 (动态切片) ---
                    for k in self.pinned_batch_buffers.keys():
                        for wid in requests:
                            self.pinned_batch_buffers[k][wid].copy_(shared_buffers[wid][k])
                    
                    active_indices = torch.tensor(requests, dtype=torch.long)
                    batch_obs = {k: v[active_indices].to(self.device, non_blocking=True).detach() for k, v in self.pinned_batch_buffers.items()}
                    
                    # 核心大一统前向前向图计算
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        with torch.no_grad():
                            logits, values, v_input = self.agent.net(batch_obs)
                            
                            # 统一复刻决策采样（Action Head 模型规范）
                            dist = torch.distributions.Categorical(logits=logits)
                            actions = dist.sample()
                            log_probs = dist.log_prob(actions)
                    
                    # 5. 精准解包回写：单次前向输出同时灌入两个共享切片
                    for i, wid in enumerate(requests):
                        # A. 基础槽（供 Pass 2 或 自对局决策流抽样消费）
                        shared_outputs[wid][0] = actions[i].cpu().float()
                        shared_outputs[wid][1] = log_probs[i].cpu().float()
                        shared_outputs[wid][2] = values[i].squeeze(-1).cpu().float()
                        
                        # B. [精细新增] 分数槽（供 Pass 1 意图感知大矩阵查表消费）
                        if shared_logits is not None:
                            shared_logits[wid].copy_(logits[i].cpu().float())
                            
                        socket.send_multipart([addresses[wid], b'', b'OK'])
                    #del logits, values, v_input, batch_obs

            except Exception as e:
                print(f"\n🚨 [Server ZeroMQ 熔断] 核心推推演异常已拦截: {e}")
                time.sleep(0.01)
                
        socket.close()
        context.term()

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

        # 全局优势归一化，稳定训练方向
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

        gpu_mb_obs = {}
        for k, v in cpu_obs.items():
            dummy_slice = v[:self.mini_batch_size]
            if dummy_slice.is_floating_point():
                gpu_mb_obs[k] = torch.zeros_like(dummy_slice, device=self.device, dtype=torch.float32)
            elif dummy_slice.dtype == torch.bool:
                gpu_mb_obs[k] = torch.zeros_like(dummy_slice, device=self.device, dtype=torch.bool)
            else:
                gpu_mb_obs[k] = torch.zeros_like(dummy_slice, device=self.device, dtype=torch.long)
        
        # 其他零散张量也预分配
        gpu_actions = torch.zeros(self.mini_batch_size, dtype=torch.long, device=self.device)
        gpu_old_log_probs = torch.zeros(self.mini_batch_size, dtype=torch.float32, device=self.device)
        gpu_returns = torch.zeros(self.mini_batch_size, dtype=torch.float32, device=self.device)
        gpu_advs = torch.zeros(self.mini_batch_size, dtype=torch.float32, device=self.device)

        for _ in range(EPOCHS):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                if end > batch_size:
                    break
                mb_idx = indices[start:end]
                
                #循环内：使用 .copy_() 零拷贝写入，绝对不改变内存指针
                for k, v in cpu_obs.items():
                    gpu_mb_obs[k].copy_(v[mb_idx], non_blocking=True)
                
                gpu_actions.copy_(cpu_actions[mb_idx], non_blocking=True)
                gpu_old_log_probs.copy_(cpu_log_probs[mb_idx], non_blocking=True)
                gpu_returns.copy_(cpu_returns[mb_idx], non_blocking=True)
                gpu_advs.copy_(cpu_advantages[mb_idx], non_blocking=True)
                

                # --- 网络前向传播与反向传播 (完全保持原样) ---
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    logits, values, v_input = self.agent.net(gpu_mb_obs)
                    values = values.squeeze(1)

                    # 计算 RND 预测误差损失，让 Predictor 学习当前状态(暂时舍弃)
                    #rnd_loss = self.agent.net.rnd(v_input.detach()).mean()

                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(gpu_actions)
                    entropy = dist.entropy()
                    ratio = torch.exp(new_log_probs - gpu_old_log_probs)
                    surr1 = ratio * gpu_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gpu_advs
                    
                    # 拆解 Loss，方便在图表里监控
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss_fn = nn.SmoothL1Loss()
                    value_loss = value_loss_fn(values, gpu_returns)
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

                del logits, values, v_input, surr1, surr2, ratio, entropy, loss, policy_loss, value_loss, entropy_loss

    def run_training_loop(self, max_iterations=1000):
        print("🚦 Starting PPO Training Loop...")
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
                print("🧹 [内存调度] 训练完成，正在彻底销毁主进程 PPO 静态内存池，为下一轮 Worker 腾出系统资源")
                
                # 物理清空高达 10GB+ 的巨型经验池，把内存还给系统，防止下一轮加载 DLL 时 WinError 1455
                self.merged_memory = None
                
                # 重置标志位，告诉系统下一轮重新用 empty 申请，否则会报 NoneType 错误
                self.buffer_allocated = False
                
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
                clean_state = canonical_model_state_dict(self.agent.net)
                
                # 补全生命周期字段，确保 TensorBoard 曲线 100% 无缝对接
                checkpoint = {
                    'model_state_dict': clean_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'net_config': self.net_config, 
                    'iteration': self.iteration,
                    'train_step': self.train_step,   # TensorBoard 的 X 轴坐标
                    'global_step': self.global_step,  # 环境交互总步数
                    'gamma': self.gamma,
                    'lr': self.lr,
                    'entropy': self.entropy,
                    'gae_lambda': self.gae_lambda,
                    'clip_eps': self.clip_eps
                }
                torch.save(checkpoint, path)
                # 触发 ONNX 同步导出
                if self.use_onnx:
                    # 因为是同步的，主网络此时处于停滞状态，不需要克隆，直接传干净的字典即可
                    self._export_onnx_sync(clean_state, self.iteration)
                print(f"💾 Model saved: {path}")
            
            dt = time.time() - iter_start
            print(f"⏱️ Iteration {self.iteration} finished in {dt:.1f}s")

        print("🏁 训练结束！")
    
    def _export_onnx_sync(self, clean_state, current_iter):
        """[极简稳健版] 主线程同步导出 ONNX，牺牲 3 秒换取 100% 稳定"""
        try:
            import torch.onnx
            print(f"⏳ [ONNX] 正在主线程同步导出第 {current_iter} 轮引擎，这大约需要 3 秒钟...")
            
            # 建立临时的 CPU 感知皮层
            from galatea_net import GalateaNet
            export_net = GalateaNet(self.net_config).cpu().eval()
            export_net.load_state_dict(clean_state, strict=False)
            
            # 提取 Dummy Dict 
            dummy_dict = {k: v[0:1].clone().cpu() for k, v in self.pinned_batch_buffers.items()}
            keys = list(dummy_dict.keys())
            flat_inputs = tuple(dummy_dict[k] for k in keys)
            
            # 建立展平包装器
            class ONNXWrapper(torch.nn.Module):
                def __init__(self, net, keys):
                    super().__init__()
                    self.net = net
                    self.keys = keys
                    
                def forward(self, *args):
                    batch_dict = {k: v for k, v in zip(self.keys, args)}
                    logits, values, _ = self.net(batch_dict)
                    return logits, values
            
            wrapper_net = ONNXWrapper(export_net, keys).eval()
            onnx_archive = os.path.join(self.save_dir, f"galatea_iter_{current_iter}.onnx")
            
            # 执行同步导出
            torch.onnx.export(
                wrapper_net,
                flat_inputs,        
                onnx_archive,
                export_params=True,
                opset_version=18,   
                do_constant_folding=True, 
                input_names=keys,   
                output_names=['action_logits', 'values']
            )
            
            print(f"✅ [ONNX] 纯净无损静态计算图导出成功！历史引擎 {os.path.basename(onnx_archive)} 已就绪！")
            
        except Exception as e:
            print(f"⚠️ [ONNX] 同步导出失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.run_training_loop()
