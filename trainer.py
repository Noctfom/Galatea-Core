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
ENTROPY_COEF = 0.01     # 熵正则化: 鼓励探索，防止过早收敛到局部最优
VALUE_LOSS_COEF = 0.5   # 价值网络权中
MAX_EPISODE_STEPS = 500 # 单局最大步数，防止死循环

class MemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

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

        # 初始化 AI
        self.agent = AiBot(device=self.device, net_config=self.net_config)
        # [新增] 内存布局优化
        self.agent.net = self.agent.net.to(memory_format=torch.channels_last)
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
        self.train_step = 0 # <--- [新增] 专门给 Train/Loss 画图用的步数器

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
            checkpoint = torch.load(path, map_location=self.device)
            
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
                         except: pass
                    
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

        except Exception as e:
            print(f"❌ 恢复失败: {e}")

    def collect_rollouts(self):
        """
        多进程采集核心逻辑
        """
        print(f"📥 [Iter {self.iteration}] 唤醒 {self.num_workers} 个工人 | 目标: {self.update_timesteps} 步")
        t0 = time.time()

        # --- [新增] 异步推断服务器初始化 ---
        if self.async_infer:
            m = mp.Manager() # Windows 下跨进程传队列必须用 Manager
            req_q = m.Queue()
            resp_qs = [m.Queue() for _ in range(self.num_workers)]
            stop_event = m.Event()
            
            # 启动后台发牌员线程
            infer_thread = threading.Thread(target=self.inference_server, args=(req_q, resp_qs, stop_event))
            infer_thread.start()
        else:
            req_q, resp_qs, stop_event = None, None, None
        
        # --- [黑科技 1] 权重共享机制 ---
        # 原理：在 Windows Spawn 模式下，普通传参会导致权重被复制 N 份。
        # 使用 share_memory_() 可以让所有子进程只读同一块物理内存，显着降低 RAM 占用。
        raw_weights = self.agent.net.state_dict()
        cpu_weights = {}
        for k, v in raw_weights.items():
            # 1. 剥离 compile 前缀
            clean_k = k.replace("_orig_mod.", "")
            # 2. 转到 CPU
            cpu_tensor = v.cpu()
            # 3. 关键：开启共享内存 (如果显存够大，这一步能省几 GB 内存)
            cpu_tensor.share_memory_() 
            cpu_weights[clean_k] = cpu_tensor
        
        # 2. 分配任务
        steps_per_worker = max(200, self.update_timesteps // self.num_workers)
        
        # 3. 启动进程池
        pool = mp.Pool(processes=self.num_workers)
        results = []
        
        for i in range(self.num_workers):
            res = pool.apply_async(worker_process, args=(
                i, 
                self.net_config, 
                cpu_weights if not self.async_infer else None, # 🔴 Async 模式下不再传权重！省内存！
                self.deck_dir, 
                steps_per_worker,
                self.worker_device if not self.async_infer else 'cpu', # Async 模式下 Worker 纯吃 CPU
                req_q,                          # 新增参数
                resp_qs[i] if self.async_infer else None  # 新增参数
            ))
            results.append(res)
            
        pool.close()
        
        # 等待所有工人完成
        # 注意: apply_async 是非阻塞的，这里会阻塞等待结果
        print(f"   ... cpu全核满载中 ...")
        pool.join()

        # --- [新增] 关闭异步服务器 ---
        if self.async_infer:
            stop_event.set()
            infer_thread.join()
        
        # 4. 汇总数据
        # [修改] 快速汇总
        # 我们现在收到的是 list of batch_dicts
        batch_dicts = []
        total_steps = 0
        total_rewards = []
        total_lens = []
        
        for res in results:
            try:
                data, r, l = res.get()
                
                # 🛑 [修复] 增加类型检查，防止旧版 worker 数据混入
                if data is not None:
                    if isinstance(data, dict): # 必须是字典
                        batch_dicts.append(data)
                        total_steps += len(data['action'])
                        if r != 0: total_rewards.append(r)
                        if l != 0: total_lens.append(l)
                    else:
                        print(f"⚠️ Worker Error: 返回了错误的数据类型 {type(data)}，已丢弃。请检查 worker.py!")
            except Exception as e:
                print(f"⚠️ Worker Error: {e}")
        
        # 如果没有收集到数据
        if not batch_dicts:
            return None
        
        # [核心优化] 极速拼接 (O(N_workers) vs O(N_steps))
        # 只需要 concat 16 次，而不是 stack 30000 次
        print(f"⚡ 正在合并 {len(batch_dicts)} 个数据块...")
        
        merged_memory = {
            'obs': {},
            'action': torch.cat([b['action'] for b in batch_dicts]),
            'log_prob': torch.cat([b['log_prob'] for b in batch_dicts]),
            'return': torch.cat([b['return'] for b in batch_dicts]),
            'advantage': torch.cat([b['advantage'] for b in batch_dicts]),
            'valid_actions': sum([b['valid_actions'] for b in batch_dicts], []) # list 相加就是通过 extend
        }
        
        # 合并 Obs
        first_obs = batch_dicts[0]['obs']
        for k in first_obs.keys():
            merged_memory['obs'][k] = torch.cat([b['obs'][k] for b in batch_dicts])

        t_cost = time.time() - t0
        avg_rew = np.mean(total_rewards) if total_rewards else 0.0
        avg_len = np.mean(total_lens) if total_lens else 0.0
        print(f"✅ 采集完成! 耗时: {t_cost:.1f}s | 样本: {total_steps} | Avg Reward: {avg_rew:.2f}")
        
        self.writer.add_scalar('Rollout/Average_Reward', avg_rew, self.iteration)
        self.writer.add_scalar('Rollout/Average_Steps', avg_len, self.iteration)
        self.writer.add_scalar('Rollout/Total_Samples', total_steps, self.iteration)
        # Log...
        self.global_step += total_steps
        return merged_memory # 直接返回大字典，不再返回 list
    
    def inference_server(self, req_q, resp_qs, stop_event):
        """
        [终极架构] 异步推断服务器：把零散请求打包送进 GPU
        """
        print("🚀 [Server] 异步推断服务器已启动，等待 Worker 请求...")
        self.agent.net.eval()
        
        while not stop_event.is_set():
            requests = []
            try:
                # 阻塞等待第一个请求，最多等 0.05 秒
                req = req_q.get(timeout=0.05)
                requests.append(req)
                # 贪婪模式：只要队列里还有，最多再拿 num_workers - 1 个，拼成一个大 Batch
                while len(requests) < self.num_workers:
                    req = req_q.get_nowait()
                    requests.append(req)
            except queue.Empty:
                pass
            
            if not requests:
                continue
                
            # --- 1. 组装 Batch ---
            batch_obs = {}
            first_obs = requests[0][1] # req[0]是worker_id, req[1]是obs字典
            for k in first_obs.keys():
                # 把大家的 tensor 拼在一起送上 GPU
                batch_obs[k] = torch.cat([r[1][k] for r in requests], dim=0).to(self.device, non_blocking=True)
                
            valid_actions_list = [r[2] for r in requests] # req[2]是合法动作列表
            worker_ids = [r[0] for r in requests]
            
            # --- 2. 批量推理 (GPU 秒杀时刻) ---
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                with torch.no_grad():
                    actions, log_probs, _, values = self.agent.get_action_and_value_from_tensor(batch_obs, valid_actions_list)
            
            # --- 3. 拆分结果寄回 ---
            for i, wid in enumerate(worker_ids):
                # 必须 .cpu()，让数据回到普通内存
                resp_qs[wid].put({
                    'action': actions[i].cpu(),
                    'log_prob': log_probs[i].cpu(),
                    'value': values[i].cpu()
                })

    def update_policy(self, memory):
        """
        [Action Head 适配版] 移除外部 Mask，完全信任网络输出
        """
        if memory is None: return
        print("🔥 Training PPO (Action Head Mode)...")
        
        # 1. 搬运到 GPU (保持不变)
        gpu_obs = {}
        for k, v in memory['obs'].items():
            t = v.to(self.device, non_blocking=True)
            if t.is_floating_point():
                gpu_obs[k] = t.to(dtype=torch.float32)
            elif 'mask' in k: 
                gpu_obs[k] = t.to(dtype=torch.bool)
            else:
                gpu_obs[k] = t.to(dtype=torch.long)

        gpu_actions = memory['action'].to(self.device, non_blocking=True)
        gpu_log_probs = memory['log_prob'].to(self.device, dtype=torch.float32, non_blocking=True)
        gpu_returns = memory['return'].to(self.device, dtype=torch.float32, non_blocking=True)
        gpu_advantages = memory['advantage'].to(self.device, dtype=torch.float32, non_blocking=True)
        
        batch_size = gpu_actions.shape[0]

        # ❌ [删除] 旧的 Mask 生成逻辑 (real_seq_len, cpu_full_mask...) 全部删掉
        # 因为 GalateaNet 现在内部已经处理了 act_mask

        # 2. 训练循环
        for _ in range(EPOCHS):
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                
                mb_obs = {k: v[mb_idx] for k, v in gpu_obs.items()}
                # ❌ [删除] mb_mask = gpu_full_mask[mb_idx]
                
                mb_actions = gpu_actions[mb_idx]
                mb_old_log_probs = gpu_log_probs[mb_idx]
                mb_returns = gpu_returns[mb_idx]
                mb_advs = gpu_advantages[mb_idx]
                
                if len(mb_advs) > 1:
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    logits, values = self.agent.net(mb_obs)
                    values = values.squeeze(1)
                    
                    # ✅ [直接使用] 网络输出的 logits 已经包含了 -1e9 的 mask
                    # masked_logits = logits + mb_mask  <-- 这一行删掉
                    
                    dist = torch.distributions.Categorical(logits=logits) # 直接用 logits
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy()
                    
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advs
                    
                    loss = -torch.min(surr1, surr2).mean() + \
                           VALUE_LOSS_COEF * 0.5 * ((values - mb_returns) ** 2).mean() + \
                           ENTROPY_COEF * -entropy.mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue

                # 每 20 个 mini_batch 记录一次
                if self.train_step % 20 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), self.train_step)
                self.train_step += 1 # <--- [新增] 每次梯度下降让画图步数 +1

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
            
            # 1. 采集
            memory = self.collect_rollouts()
            
            # 2. 优化
            # [修正] memory 是字典，len(memory) 是键的数量(6)，会导致永远跳过训练！
            # 我们应该检查数据的行数 (action 的长度)
            if memory is not None and memory['action'].shape[0] >= self.mini_batch_size:
                self.update_policy(memory)
                
                # 显式清理内存
                del memory
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            else:
                print(f"⚠️ 样本不足 ({memory['action'].shape[0] if memory else 0} < {self.mini_batch_size})，跳过本轮训练")
            
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