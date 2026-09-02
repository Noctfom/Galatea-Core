# ==================================================================================
#  Galatea AI Bot (Index Logic Fix)
#  修复了导致死锁的索引映射问题
# ==================================================================================

import torch
import torch.nn as nn
import os
import random
import struct
from checkpoint_utils import load_training_checkpoint
# 引入桥接后的 FeatureEncoder
try:
    from feature_encoder import GalateaEncoder as FeatureEncoder
except ImportError:
    # 兼容旧代码或测试环境
    from galatea_net import FeatureEncoder 

from galatea_net import GalateaNet

class AiBot:
    def __init__(self, device='cpu', net_config=None, initialize_network=True):
        """初始化 AI 控制器；中心推理 Worker 可跳过本地网络以节省内存"""
        if net_config is None:
            net_config = {'d_model': 256, 'n_heads': 4, 'n_layers': 2, 'vocab_size': 20000}

        self.net = GalateaNet(net_config).to(device) if initialize_network else None
        self.device = device
        self.encoder = FeatureEncoder()
        if self.net is not None:
            self.net.eval() # 默认推理模式

    def load_model(self, path, expected_model_id=None):
        """严格加载当前检查点，并可核验联盟训练要求的模型 UUID"""
        if not os.path.exists(path):
            print(f"⚠️ 模型文件不存在: {path}")
            return False

        try:
            checkpoint = load_training_checkpoint(path, map_location=self.device)
            if (
                expected_model_id is not None
                and checkpoint['model_id'] != expected_model_id
            ):
                raise PermissionError(
                    "模型 UUID 鉴权失败: "
                    f"期望 {expected_model_id}, 实际 {checkpoint['model_id']}"
                )

            saved_config = checkpoint['net_config']
            print(f"📦 发现内嵌配置: {saved_config}")
            self.net = GalateaNet(saved_config).to(self.device)
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.net.eval()
            print(f"✅ 网络已自动重构并加载权重。")
            return True

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False

    def get_action_and_value_from_tensor(self, obs_dict, valid_actions_list=None):
        """
        [训练专用 - Action Head版] 获取动作概率和价值
        """
        if self.net is None:
            raise RuntimeError("当前 AI 控制器未加载本地推理网络")
        # 1. 前向传播
        # logits: [B, MAX_ACTIONS] (已在网络内部Mask，无效动作是 -1e9)
        # value:  [B, 1]
        logits, value, v_input = self.net(obs_dict)
        
        # 2. 构建分布
        # Categorical 会自动对 logits 做 softmax
        # -1e9 的项概率会变成 0，不会被采样到
        dist = torch.distributions.Categorical(logits=logits)
        
        # 3. 采样
        action = dist.sample()
        
        # 返回: action(索引), log_prob, entropy(平均值), value
        return action, dist.log_prob(action), dist.entropy().mean(), value, v_input

    def get_decision(self, gamestate, msg_type, msg_args=None):
        if self.net is None:
            raise RuntimeError("当前 AI 控制器未加载本地推理网络")
        self.net.eval()
        snap = gamestate.get_snapshot(self.env)
        if not snap.valid_actions: return None

        tensor_dict = self.encoder.encode(snap, player_id=snap.global_data.to_play)
        
        with torch.no_grad():
            gpu_dict = {k: v.to(self.device) for k, v in tensor_dict.items()}
            
            # Logits 现在直接就是 [1, 120] 的动作分数
            logits, value = self.net(gpu_dict) 
            
            # 网络已经内置了 act_mask 并把无效槽位变成了 -1e9
            # 不需要手动切片，直接 Argmax，不可能选到 Padding
            temperature = 0.5
            probs = torch.softmax(logits[0] / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            sel_idx = dist.sample().item()

        if sel_idx < len(snap.valid_actions):
            chosen = snap.valid_actions[sel_idx]
        else:
            # 兜底：理论上不会走到这里，除非所有动作都被 mask 了
            chosen = random.choice(snap.valid_actions)

        resp = self._pack_response(chosen, msg_type, msg_args)
        return resp

    def _pack_response(self, action, msg_type=0, msg_args=None):
        # ==========================================================
        # 优先检查动作是否携带有物理外挂（宏动作包裹）
        # 如果有 decision_bytes，说明这是经过 RuleBot 完美打包的套餐，直接透传
        # ==========================================================
        if hasattr(action, 'decision_bytes') and action.decision_bytes:
            return action.decision_bytes
        if getattr(action, 'decision_value', None) is not None:
            return int(action.decision_value)
        # ==========================================================
        # 1. 整型槽类 (调用 C++ set_responsei) - 绝对不能返回 bytes
        # 包含: 10(Battle), 11(Idle), 12(EffectYN), 13(YesNo), 
        #       14(Option), 16(Chain), 140~143(各类宣言)
        # ==========================================================
        if msg_type in [10, 11, 12, 13, 14, 16, 140, 141, 142, 143]:
            if msg_type in [10, 11]:
                return int((action.index << 16) | action.action_type)
            elif msg_type in [140, 141, 142]:
                return int(action.desc_id)
            else:
                return int(action.index)

        # ==========================================================
        # 2. 字节槽类 (调用 C++ set_responseb) - 必须带 count 字节
        # 包含: 15(SelectCard), 20(Tribute), 22(Counter), 26(Unselect)
        # ==========================================================
        elif msg_type in [15, 20, 22, 26]:
            if action.index < 0 or action.index > 255:
                # Cancel 指令 (-1)，转换为 4 字节的 0xFFFFFFFF
                return int(-1).to_bytes(4, byteorder='little', signed=True)
            # 兜底：数量(Count)=1, 后接选中的索引
            return bytes([1, action.index]) 
        
        # ==========================================================
        # 3. 物理格子类 (Place / Disfield) - 严格的 3 字节
        # ==========================================================
        elif msg_type in [18, 24]:
            zone_id = action.index
            p = 0; l = 0x04; s = 0
            if zone_id & 16: p = 1
            if zone_id & 8:  l = 0x08
            s = zone_id & 0x7
            
            req_p = 0
            if msg_args and len(msg_args) > 0: req_p = msg_args[0] 
            
            raw_p = req_p if p == 0 else (1 - req_p)
            final_p = 1 if raw_p == 1 else 0
            return bytes([final_p, l, s])
            
        # 兜底防护：所有未知指令全部返回整数，防止字节野指针
        return int(action.index)
