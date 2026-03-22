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
        [终极重构] 原生多进程架构，彻底击碎 Manager Socket 瓶颈
        """
        print(f"📥 [Iter {self.iteration}] 唤醒 {self.num_workers} 个工人 | 目标: {self.update_timesteps} 步")
        t0 = time.time()

        # --- 1. 使用原生 Queue 替代缓慢的 Manager Queue ---
        if self.async_infer:
            req_q = mp.Queue()
            resp_qs = [mp.Queue() for _ in range(self.num_workers)]
            stop_event = mp.Event()
            
            infer_thread = threading.Thread(target=self.inference_server, args=(req_q, resp_qs, stop_event))
            infer_thread.start()
        else:
            req_q, resp_qs, stop_event = None, None, None
        
        raw_weights = self.agent.net.state_dict()
        cpu_weights = {}
        for k, v in raw_weights.items():
            clean_k = k.replace("_orig_mod.", "")
            cpu_tensor = v.cpu()
            cpu_tensor.share_memory_() # 开启真正的物理共享内存
            cpu_weights[clean_k] = cpu_tensor
        
        steps_per_worker = max(200, self.update_timesteps // self.num_workers)
        
        # --- 2. 弃用 Pool，使用原生 Process 启动 ---
        result_q = mp.Queue()
        processes = []
        
        for i in range(self.num_workers):
            p = mp.Process(target=worker_wrapper, args=(
                i, 
                self.net_config, 
                cpu_weights if not self.async_infer else None, 
                self.deck_dir, 
                steps_per_worker,
                self.worker_device if not self.async_infer else 'cpu',
                req_q,
                resp_qs[i] if self.async_infer else None,
                result_q
            ))
            p.start()
            processes.append(p)
            
        print(f"   ... cpu运算中 ...")
        
        # --- 3. 收集数据 (带超时防死锁保护) ---
        batch_dicts = []
        total_steps = 0
        total_rewards = []
        total_lens = []
        
        # 等待所有人交作业
        for _ in range(self.num_workers):
            try:
                # 阻塞等待，加入超时机制，防止某个进程 C++ 引擎暴毙导致无限死等
                w_id, res = result_q.get(timeout=300) 
                if res is not None:
                    data, r, l = res
                    if data is not None and isinstance(data, dict):
                        batch_dicts.append(data)
                        total_steps += len(data['action'])
                        if r != 0: total_rewards.append(r)
                        if l != 0: total_lens.append(l)
            except queue.Empty:
                print("⚠️ 警告: 收集数据超时 (部分 Worker 的 C++ 引擎可能已崩溃)")

        for p in processes:
            # 优雅终止：如果进程还在，强制咔嚓掉，防止内存泄漏
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

        if self.async_infer:
            stop_event.set()
            infer_thread.join()
        
        if not batch_dicts:
            return None
        
        print(f"⚡ 正在合并 {len(batch_dicts)} 个数据块...")
        
        # --- 4. 修复 KeyError：彻底移除 valid_actions ---
        merged_memory = {
            'obs': {},
            'action': torch.cat([b['action'] for b in batch_dicts]),
            'log_prob': torch.cat([b['log_prob'] for b in batch_dicts]),
            'return': torch.cat([b['return'] for b in batch_dicts]),
            'advantage': torch.cat([b['advantage'] for b in batch_dicts])
            # (这里完美去除了导致报错的 valid_actions)
        }
        
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
        self.global_step += total_steps
        
        return merged_memory
    
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
                
            # --- 1. 组装压平的 Batch ---
            worker_ids = [r[0] for r in requests]
            # 👇 [击毙幽灵 1] 将 Worker 发来的 numpy 数组转回 Tensor
            packed_batch = torch.stack([torch.from_numpy(r[1]) for r in requests]).to(self.device, non_blocking=True) # [B, 6035]
            
            # --- 2. 在 GPU 上光速切片解包 (保持不变) ---
            batch_obs = {
                'global': packed_batch[:, :15],
                'card_idx': packed_batch[:, 15:115].to(torch.long),
                'card_race': packed_batch[:, 115:215].to(torch.long),
                'card_attr': packed_batch[:, 215:315].to(torch.long),
                'card_feats': packed_batch[:, 315:5615].view(-1, 100, 53),
                'padding_mask': packed_batch[:, 5615:5715].to(torch.bool),
                'act_card_idx': packed_batch[:, 5715:5795].to(torch.long),
                'act_type': packed_batch[:, 5795:5875].to(torch.long),
                'act_desc': packed_batch[:, 5875:5955].to(torch.long),
                'act_mask': packed_batch[:, 5955:6035].to(torch.bool)
            }
            
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                with torch.no_grad():
                    actions, log_probs, _, values = self.agent.get_action_and_value_from_tensor(batch_obs, None)
            
            # --- 4. 组装回传封包 ---
            packed_returns = torch.stack([actions.to(torch.float32), log_probs, values.squeeze(-1)], dim=1).cpu()
            
            for i, wid in enumerate(worker_ids):
                # 👇 [击毙幽灵 1] 转为 numpy 数组发回给 Worker
                resp_qs[wid].put(packed_returns[i].numpy())

    def update_policy(self, memory):
        """
        [安全防爆版] 将数据留在 CPU，每次只切片 mini_batch 送进 GPU
        """
        if memory is None: return
        print("🔥 Training PPO (Action Head Mode)...")
        
        # 数据先全留在 CPU 内存里
        cpu_obs = memory['obs']
        cpu_actions = memory['action']
        cpu_log_probs = memory['log_prob']
        cpu_returns = memory['return']
        cpu_advantages = memory['advantage']
        
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
                
                if len(mb_advs) > 1:
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

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
                    loss = -torch.min(surr1, surr2).mean() + \
                           VALUE_LOSS_COEF * 0.5 * ((values - mb_returns) ** 2).mean() + \
                           ENTROPY_COEF * -entropy.mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue

                if self.train_step % 20 == 0:
                    self.writer.add_scalar('Train/Loss', loss.item(), self.train_step)
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