# ==================================================================================
#  Galatea PPO Trainer (Deep Fix Version)
#  结合了 run_self_play 的鲁棒交互逻辑与 PPO 的训练管线
# ==================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
import numpy as np
import random
import gc
import threading
import queue
import glob

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
GAMMA = 0.99            # Discount Factor: 远视眼程度，0.99表示很看重未来收益
GAE_LAMBDA = 0.95       # GAE参数: 平衡方差和偏差的关键
UPDATE_TIMESTEPS = 2048 # Batch Size: 攒多少经验升一级
EPOCHS = 4              # PPO Update Epochs:同一批数据反复榨取几次
MINIBATCH_SIZE = 128    # Mini-batch: 梯度下降时的切片大小
CLIP_EPS = 0.2          # PPO Clip: 限制更新幅度，防止学“飘”了
ENTROPY_COEF = 0.02     # 熵正则化: 鼓励探索，防止过早收敛到局部最优
VALUE_LOSS_COEF = 0.5   # 价值网络权中
MAX_EPISODE_STEPS = 800 # 单局最大步数，防止死循环

class MemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
def worker_wrapper(worker_id, net_config, weights, deck_dir, target_steps, device, req_q, resp_q, result_q):
    """用于原生 Process 的安全包装器，防止 DLL 崩溃导致主进程死锁"""
    try:
        from worker import worker_process
        res = worker_process(worker_id, net_config, weights, deck_dir, target_steps, device, req_q, resp_q)
        result_q.put((worker_id, res))
    except Exception as e:
        print(f"Worker {worker_id} 发生异常退出: {e}")
        result_q.put((worker_id, None))

def collate_fn(batch):
    obs_list = [item['obs'] for item in batch]
    obs_batch = {}
    # 将字典中的 tensor 堆叠
    for k in obs_list[0].keys():
        obs_batch[k] = torch.cat([x[k] for x in obs_list], dim=0)
    
    actions = torch.stack([item['action'] for item in batch])
    old_log_probs = torch.stack([item['log_prob'] for item in batch])
    returns = torch.tensor([item['return'] for item in batch], dtype=torch.float32)
    advantages = torch.tensor([item['advantage'] for item in batch], dtype=torch.float32)
    valid_actions_list = [item['valid_actions'] for item in batch]
    
    return obs_batch, actions, old_log_probs, returns, advantages, valid_actions_list

class PPOTrainer:
    # [修改] 增加 compile_model 参数
    def __init__(self, save_dir="./models", deck_dir="./decks", net_config=None, resume_path=None, 
                 update_timesteps=4096, mini_batch_size=512, num_workers=4, worker_device='cuda', async_infer=False, compile_model=True): # <--- 新增
        self.save_dir = save_dir
        self.deck_dir = deck_dir
        self.update_timesteps = update_timesteps  # [新增] 存下来
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
        # 🌟 修复：加上 .pt* 就能同时匹配 .pt 和 .pt.tmp
        for f in glob.glob("tmp_rollout_*.pt*") + glob.glob("tmp_weights_*.pt*"):
            try: os.remove(f)
            except Exception as e: 
                print(f"[trainer]⚠️ 无法删除临时文件 {f}: {e}")

        # 初始化 AI
        self.agent = AiBot(device=self.device, net_config=self.net_config)
        # [新增] 内存布局优化
        #self.agent.net = self.agent.net.to(memory_format=torch.channels_last)
        self.agent.net.train()

        # [新增] 编译优化
        if self.enable_compile and self.device.type == 'cuda':
            try:
                print("🚀 [编译] 正在启用 torch.compile...")
                self.agent.net = torch.compile(self.agent.net, mode='reduce-overhead')
            except Exception as e:
                print(f"⚠️ 编译跳过: {e}")

        self.optimizer = optim.Adam(self.agent.net.parameters(), lr=LR)
        # 初始化 Scaler (BF16时其实不需要缩放，但为了代码通用，我们保留它)
        # [修改] 使用新版 API，指定设备类型 'cuda'
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

        # [新增] 初始化 TensorBoard 记录器
        # log_dir 可以按时间戳命名，方便区分不同次训练
        import datetime
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"./runs/galatea_{time_str}")
        print(f"📊 TensorBoard 日志将保存至: ./runs/galatea_{time_str}")

        # Windows 必须设置
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError: pass

        # [修改] 恢复训练逻辑简化为调用函数
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

    def load_checkpoint(self, path):
        """
        [新增] 独立的加载函数，增强了对编译模型的兼容性
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
                    self.optimizer = optim.Adam(self.agent.net.parameters(), lr=LR)

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
            print(f"✅ 恢复成功! Iter: {self.iteration}")
            # [优化 3] 恢复 train_step，衔接曲线
            # 假设每个 Iteration 大约更新 (32000 / 128) * 4 = 1000 次
            self.train_step = self.iteration * (self.update_timesteps // self.mini_batch_size) * EPOCHS

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
        
        steps_per_worker = max(200, self.update_timesteps // self.num_workers)
        
        # 每一轮动态创建专属队列和推断服务器
        if self.async_infer:
            req_q = mp.Queue(maxsize=self.num_workers * 2)
            resp_qs = [mp.Queue(maxsize=2) for _ in range(self.num_workers)]
            stop_event = threading.Event()
            infer_thread = threading.Thread(
                target=self.inference_server, 
                args=(req_q, resp_qs, stop_event),
                daemon=True
            )
            infer_thread.start()
        else:
            req_q = None
            resp_qs = [None] * self.num_workers

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
                req_q, resp_qs[i]        # 恢复队列传参
            ))
            p.daemon = True
            p.start()
            processes.append(p)
            
        print(f"   ... {'异步 GPU Server' if self.async_infer else '纯本地 CPU'} 运算中 ...")
        
        # 等待工人自然死亡
        for p in processes:
            p.join(timeout=300)
            if p.is_alive():
                print(f"⚠️ 侦测到 Worker 卡死在 C++ 引擎中，执行物理超度...")
                p.terminate() 
                p.join() # 🌟 必须收尸
            try: p.close() # 🌟 强制释放 Windows 进程句柄
            except: print(f"⚠️ 无法关闭 Worker 进程 (可能已被系统回收)，请检查系统资源管理器确认是否有残留的 Python 进程")
                
        #[核心修复]：彻底销毁队列，防止 Windows 句柄和后台线程泄露
        if self.async_infer:
            stop_event.set()
            # 1. 抽干残留数据，防止管道堵塞导致无法关闭
            while not req_q.empty():
                try: req_q.get_nowait()
                except: break
            
            # 2. 物理关闭底层管道并强杀后台喂食线程
            req_q.cancel_join_thread() # 取消等待，直接强杀
            req_q.close()
            
            for q in resp_qs:
                if q is not None:
                    q.cancel_join_thread() # 取消等待，直接强杀
                    q.close()
            
            infer_thread.join(timeout=2)
            del req_q
            del resp_qs
            import gc; gc.collect()

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
        
        total_steps = 0
        file_steps = []
        total_rewards = []
        total_lens = []

        # 步骤 A：测算总面积并提取评估数据
        for f in file_list:
            data = torch.load(f, map_location='cpu', weights_only=False)
            s = data['action'].shape[0]
            file_steps.append(s)
            total_steps += s
            
            # 提取存放在文件里的奖励数据
            r = data.get('avg_rew', np.array([0.0]))[0]
            l = data.get('avg_len', np.array([0.0]))[0]
            if r != 0: total_rewards.append(r)
            if l != 0: total_lens.append(l)
            
            del data
            import gc; gc.collect()
            
        # 步骤 B：预分配连续内存 (仅首次执行，彻底消灭内存碎片)
        if not self.buffer_allocated:
            print(f"📦 [内存管理] 首次初始化主进程静态内存池 (容量: {self.max_buffer_steps} 步)...")
            self.merged_memory = {'obs': {}}
            first_data = torch.load(file_list[0], map_location='cpu', weights_only=False)
            
            self.merged_memory['action'] = torch.empty(self.max_buffer_steps, dtype=first_data['action'].dtype)
            self.merged_memory['log_prob'] = torch.empty(self.max_buffer_steps, dtype=first_data['log_prob'].dtype)
            self.merged_memory['return'] = torch.empty(self.max_buffer_steps, dtype=first_data['return'].dtype)
            self.merged_memory['advantage'] = torch.empty(self.max_buffer_steps, dtype=first_data['advantage'].dtype)
            
            for k, v in first_data['obs'].items():
                shape = list(v.shape)
                shape[0] = self.max_buffer_steps
                self.merged_memory['obs'][k] = torch.empty(*shape, dtype=v.dtype)
                
            self.buffer_allocated = True
            del first_data
            import gc; gc.collect()

        # 步骤 C：流式注入并阅后即焚
        cursor = 0
        for i, f in enumerate(file_list):
            try:
                data = torch.load(f, map_location='cpu', weights_only=False)
                s = file_steps[i]
                
                # 防御性截断，防止溢出缓冲池
                if cursor + s > self.max_buffer_steps:
                    print(f"⚠️ 警告: 采集步数({cursor+s})超过缓冲容量({self.max_buffer_steps})，自动截断！")
                    s = self.max_buffer_steps - cursor
                    if s <= 0: break
                
                self.merged_memory['action'][cursor:cursor+s] = data['action'][:s]
                self.merged_memory['log_prob'][cursor:cursor+s] = data['log_prob'][:s]
                self.merged_memory['return'][cursor:cursor+s] = data['return'][:s]
                self.merged_memory['advantage'][cursor:cursor+s] = data['advantage'][:s]
                
                for k in self.merged_memory['obs'].keys():
                    self.merged_memory['obs'][k][cursor:cursor+s] = data['obs'][k][:s]
                    
                cursor += s
                del data
                import gc; gc.collect()
            except Exception as e:
                print(f"❌ 读取文件 {f} 失败: {e}") # 拒绝静默报错
            
            try: os.remove(f)
            except Exception as e: 
                print(f"[trainer]⚠️ 清理残余文件 {f} 失败: {e}")

        t_cost = time.time() - t0
        avg_rew = np.mean(total_rewards) if total_rewards else 0.0
        avg_len = np.mean(total_lens) if total_lens else 0.0
        print(f"✅ 采集完成! 耗时: {t_cost:.1f}s | 样本: {cursor} | Avg Reward: {avg_rew:.2f}")
        
        self.writer.add_scalar('Rollout/Average_Reward', avg_rew, self.iteration)
        self.writer.add_scalar('Rollout/Average_Length', avg_len, self.iteration)
        self.global_step += cursor
        
        return cursor # 核心改动：不再返回内存大字典，而是返回有效步数！
    
    def inference_server(self, req_q, resp_qs, stop_event):
        """
        [封包极速版] 接收压平的 Tensor，在 GPU 显存内进行光速切片
        """
        print("🚀 [Server] 异步推断服务器已启动，等待 Worker 请求...")
        self.agent.net.eval()
        
        while not stop_event.is_set():
            requests = []
            try:
                req = req_q.get(timeout=0.05)
                requests.append(req)
                while len(requests) < self.num_workers:
                    req = req_q.get_nowait()
                    requests.append(req)
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
                    actions, log_probs, _, values = self.agent.get_action_and_value_from_tensor(batch_obs, None)
            
            # --- 4. 组装回传封包 ---
            # 新增 .detach()，彻底斩断与 GPU 计算图的最后一点阴阳联系
            packed_returns = torch.stack([
                actions.to(torch.float32), 
                log_probs.to(torch.float32), 
                values.squeeze(-1).to(torch.float32)
            ], dim=1).detach().cpu()
            
            for i, wid in enumerate(worker_ids):
                # 👇 [击毙幽灵 1] 转为 numpy 数组发回给 Worker
                resp_qs[wid].put(packed_returns[i].numpy())

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
            cpu_advantages = (cpu_advantages - cpu_advantages.mean()) / (cpu_advantages.std() + 1e-8)
        
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
                    t = v[mb_idx].to(self.device, non_blocking=True)
                    if t.is_floating_point(): mb_obs[k] = t.to(dtype=torch.float32)
                    elif 'mask' in k: mb_obs[k] = t.to(dtype=torch.bool)
                    else: mb_obs[k] = t.to(dtype=torch.long)
                
                mb_actions = cpu_actions[mb_idx].to(self.device, non_blocking=True)
                mb_old_log_probs = cpu_log_probs[mb_idx].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_returns = cpu_returns[mb_idx].to(self.device, dtype=torch.float32, non_blocking=True)
                mb_advs = cpu_advantages[mb_idx].to(self.device, dtype=torch.float32, non_blocking=True)
                

                # --- 网络前向传播与反向传播 (完全保持原样) ---
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    logits, values = self.agent.net(mb_obs)
                    values = values.squeeze(1)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs
                    
                    # 拆解 Loss，方便我们在图表里监控
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                    entropy_loss = -entropy.mean()
                    
                    loss = policy_loss + VALUE_LOSS_COEF * value_loss + ENTROPY_COEF * entropy_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue

                if self.train_step % 20 == 0:
                    self.writer.add_scalar('Train/Total_Loss', loss.item(), self.train_step)
                    self.writer.add_scalar('Train/Policy_Loss', policy_loss.item(), self.train_step) # 策略偏移
                    self.writer.add_scalar('Train/Value_Loss', value_loss.item(), self.train_step)   # 价值预测准确度
                    self.writer.add_scalar('Train/Entropy', entropy.mean().item(), self.train_step)  # 🌟 探索欲 (如果急剧降到0，说明AI变傻钻牛角尖了)
                self.train_step += 1

                self.optimizer.zero_grad()
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
                
                # [关键] 保存所有信息
                checkpoint = {
                    'model_state_dict': self.agent.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'net_config': self.net_config, # 出生证明
                    'iteration': self.iteration,
                    'scaler_state_dict': self.scaler.state_dict(), # [新增] 保存 Scaler 状态
                }
                torch.save(checkpoint, path)
                print(f"💾 Model saved: {path}")
            
            dt = time.time() - iter_start
            print(f"⏱️ Iteration {self.iteration} finished in {dt:.1f}s")

        print("🏁 训练结束！")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.run_training_loop()