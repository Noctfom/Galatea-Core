# ==================================================================================
#  Galatea AI Bot (Index Logic Fix)
#  修复了导致死锁的索引映射问题
# ==================================================================================

import torch
import torch.nn as nn
import os
import random
# 引入桥接后的 FeatureEncoder
try:
    from feature_encoder import GalateaEncoder as FeatureEncoder
except ImportError:
    # 兼容旧代码或测试环境
    from galatea_net import FeatureEncoder 

from galatea_net import GalateaNet

class AiBot:
    def __init__(self, device='cpu', net_config=None):
        if net_config is None:
            net_config = {'d_model': 256, 'n_heads': 4, 'n_layers': 2, 'vocab_size': 20000}
            
        self.net = GalateaNet(net_config).to(device)
        self.device = device
        self.encoder = FeatureEncoder()
        self.net.eval() # 默认推理模式

    def load_model(self, path):
        if not os.path.exists(path):
            print(f"⚠️ 模型文件不存在: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # [新逻辑] 检查是否包含配置字典
            if isinstance(checkpoint, dict) and 'net_config' in checkpoint:
                saved_config = checkpoint['net_config']
                print(f"📦 发现内嵌配置: {saved_config}")
                self.net = GalateaNet(saved_config).to(self.device)
                self.net.load_state_dict(checkpoint['model_state_dict'])
                self.net.eval()
                print(f"✅ 网络已自动重构并加载权重。")
                return True
            
            # [旧逻辑]
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                 self.net.load_state_dict(checkpoint['model_state_dict'])
                 return True
            else:
                self.net.load_state_dict(checkpoint)
                return True

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False

    def get_action_and_value_from_tensor(self, obs_dict, valid_actions_list=None):
        """
        [训练专用 - Action Head版] 获取动作概率和价值
        """
        # 1. 前向传播
        # logits: [B, MAX_ACTIONS] (已在网络内部Mask，无效动作是 -1e9)
        # value:  [B, 1]
        logits, value = self.net(obs_dict)
        
        # 2. 构建分布
        # Categorical 会自动对 logits 做 softmax
        # -1e9 的项概率会变成 0，不会被采样到
        dist = torch.distributions.Categorical(logits=logits)
        
        # 3. 采样
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # 返回: action(索引), log_prob, entropy(平均值), value
        return action, log_prob, entropy.mean(), value

    def get_decision(self, gamestate, msg_type, msg_args=None):
        self.net.eval()
        snap = gamestate.get_snapshot()
        if not snap.valid_actions: return None

        tensor_dict = self.encoder.encode(snap, player_id=snap.global_data.to_play)
        
        with torch.no_grad():
            gpu_dict = {k: v.unsqueeze(0).to(self.device) for k, v in tensor_dict.items()}
            
            # Logits 现在直接就是 [1, 80] 的动作分数
            logits, value = self.net(gpu_dict) 
            
            # 🌟 [终极清理] 既然网络已经内置了 act_mask 并把无效槽位变成了 -1e9
            # 我们根本不需要手动切片，直接无脑 Argmax，它绝对不可能选到 Padding！
            sel_idx = torch.argmax(logits[0]).item()

        if sel_idx < len(snap.valid_actions):
            chosen = snap.valid_actions[sel_idx]
        else:
            # 兜底：理论上不会走到这里，除非所有动作都被 mask 了
            chosen = random.choice(snap.valid_actions)

        resp = self._pack_response(chosen, msg_type, msg_args)
        return resp

    def _pack_response(self, action, msg_type=0, msg_args=None):
        """
        打包响应数据 (修复版 - 强制依赖 msg_type)
        """
        # 1. 必须返回 4 字节整数的消息类型
        # 10: Battle, 11: Idle (召唤/发动), 16: Chain
        if msg_type in [10, 11, 16]:
            if msg_type == 16:
                # 🌟 核心修复：必须加上 signed=True，否则 AI 选 -1 会直接抛出异常！
                return int(action.index).to_bytes(4, byteorder='little', signed=True)
            
            val = (action.index << 16) | action.action_type
            return int(val).to_bytes(4, byteorder='little')
        
        # 2. 选卡 (MSG_SELECT_CARD)
        elif msg_type == 15:
            if action.index < 0 or action.index > 255:
                return int(-1).to_bytes(4, byteorder='little', signed=True)
            return bytes([1, action.index])
        
        # 3. 其他单字节交互 (位置选择等)
        else:
            val = action.index
            # 特殊处理 Place / Disfield
            if msg_type in [18, 24]:
                zone_id = val
                p = 0; l = 0x04; s = 0
                if zone_id & 16: p = 1
                if zone_id & 8:  l = 0x08
                s = zone_id & 0x7
                
                req_p = 0
                if msg_args and len(msg_args) > 1: req_p = msg_args[1]
                raw_p = req_p if p == 0 else (1 - req_p)
                final_p = 1 if raw_p == 1 else 0
                return bytes([final_p, l, s])
            
            # 默认返回单字节
            return bytes([val])