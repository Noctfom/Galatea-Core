# ==================================================================================
#  Galatea Network Architecture (Transformer-based)
#  Project Galatea V2.1 - The Brain
# ==================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_encoder import GalateaEncoder as FeatureEncoder

class GalateaNet(nn.Module):
    def __init__(self, config):
        """
        初始化神经网络
        :param config: 配置字典，包含 d_model, n_heads 等参数
        """
        super().__init__()
        self.d_model = config.get('d_model', 256)
        self.n_heads = config.get('n_heads', 4)
        self.n_layers = config.get('n_layers', 2)
        self.vocab_size = config.get('vocab_size', 20000) # 卡片种类数
        
        # --- 1. Embedding Layers (感知层) ---
        # A. 卡片 ID 编码 (0=Padding, 1=Unknown, 2...=Real)
        self.card_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        
        # B. 数值特征映射 (Owner, Loc, Atk, Def...) -> d_model
        # feature_encoder.py 中 card_feat_dim = 53
        self.feat_proj = nn.Linear(53, self.d_model)

        self.race_embed = nn.Embedding(30, self.d_model, padding_idx=0)
        self.attr_embed = nn.Embedding(10, self.d_model, padding_idx=0)
        
        # C. 全局特征映射 -> d_model
        # feature_encoder.py 中 global_dim = 15
        self.global_proj = nn.Linear(15, self.d_model)

        # --- 2. Transformer Encoder (推理层) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.n_heads, 
            dim_feedforward=self.d_model * 4, 
            batch_first=True, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # --- 3. 动作感知层 (Action Head 新增) ---
        # 动作类型的 Embedding (比如 攻击=1, 发动=5...)
        self.act_type_embed = nn.Embedding(256, self.d_model) # 假设类型ID不超过256
        # 描述文字的 Embedding
        self.desc_embed = nn.Embedding(1024, self.d_model) # Hash 后的 Desc ID

        # 最终打分网络: 输入是融合后的向量，输出 1 个分数
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Logit
        )

        # 价值网络 (保留)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, batch_dict):
        """
        前向传播
        :param batch_dict: feature_encoder 输出的字典
        :return: 
            logits: (Batch, SeqLen) 每张卡被选中的概率 Logit
            value: (Batch, 1) 当前局面评分
        """
        # [Batch, Seq]
        card_idx = batch_dict['card_idx'] 
        # [Batch, Seq, Feat]
        card_feats = batch_dict['card_feats']
        # [Batch, Seq] 卡片种族和属性
        card_race = batch_dict['card_race']
        card_attr = batch_dict['card_attr'] 
        # [Batch, Global]
        global_vec = batch_dict['global']
        # [Batch, Seq] (Bool, 1=Real, 0=Pad)
        padding_mask = batch_dict['padding_mask'] 
        
        # --- 1. Embedding ---
        # 将 ID 转换为向量
        x_code = self.card_embed(card_idx) # (B, S, D)
        # 将数值特征投影为向量
        x_feat = self.feat_proj(card_feats) # (B, S, D)
        # 激活种族和属性潜能
        x_race = self.race_embed(card_race)
        x_attr = self.attr_embed(card_attr)
        
        # 终极特征融合 (四大特征加在一起)
        x = x_code + x_feat + x_race + x_attr
        
        # --- 2. Transformer Reasoning ---
        # PyTorch Transformer 的 src_key_padding_mask 需要 True 为 Padding (忽略)
        # 我们的 mask 是 1=Valid, 0=Padding，所以要取反 (~mask)
        # 注意：padding_mask 是 BoolTensor，需确保 device 一致
        src_mask = ~padding_mask 
        
        # (B, S, D) -> 经过 Transformer 后的每张卡的深层语义向量
        memory = self.transformer(x, src_key_padding_mask=src_mask)
        
        # --- 3. Feature Aggregation (Global + Local) ---
        # 将全局特征投影
        g_embed = self.global_proj(global_vec).unsqueeze(1) # (B, 1, D)
        
        # 对场上所有卡做 Max Pooling，提取“最强的特征” (例如场上攻最高的怪)
        # 必须把 Padding 的位置 Mask 掉，防止取出 0
        masked_memory = memory.masked_fill(src_mask.unsqueeze(-1), -1e9)
        pooled = torch.max(masked_memory, dim=1)[0].unsqueeze(1) # (B, 1, D)
        
        # 融合 Global + Pooled Local
        # v_input: (B, 1, D)
        v_input = g_embed + pooled
        
        # --- 4. Value Prediction ---
        value = self.value_head(v_input.squeeze(1)) # (B, 1)

        # === [NEW] Action Head (动作打分) ===
        # 1. 获取动作数据
        act_card_idx = batch_dict['act_card_idx'] # [B, 80]
        act_type = batch_dict['act_type']         # [B, 80]
        act_desc = batch_dict['act_desc']         # [B, 80]
        act_mask = batch_dict['act_mask']         # [B, 80]

        # 2. "抓取" 目标卡片的特征
        # 我们知道动作指向了第几张卡 (act_card_idx)
        # 我们要从 memory (卡片特征库) 里把那张卡的向量拿出来
        # memory: [B, 100, D], act_card_idx: [B, 80]
        
        batch_size, max_acts = act_card_idx.shape
        # 扩展索引维度以匹配 gather 的要求: [B, 80, D]
        idx_expanded = act_card_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        
        # Gather: 这一步实现了 "绑定"
        target_card_vecs = torch.gather(memory, 1, idx_expanded) # [B, 80, D]

        # 3. 获取动作自身的特征
        type_vecs = self.act_type_embed(act_type) # [B, 80, D]
        desc_vecs = self.desc_embed(act_desc)     # [B, 80, D]

        # 4. 特征融合：卡片 + 类型 + 描述
        # 现在的 action_vec 包含了 "是对青眼白龙(Target) 做出的 攻击(Type)" 这一完整信息
        action_vecs = target_card_vecs + type_vecs + desc_vecs + v_input

        # 5. 打分
        logits = self.policy_head(action_vecs).squeeze(-1) # [B, 80]

        # 6. Mask 掉无效的动作槽位 (Padding)
        logits = logits.masked_fill(~act_mask, -1e9)

        return logits, value

if __name__ == "__main__":
    # 简单的冒烟测试 (Smoke Test)
    print("🚀 测试 GalateaNet...")
    config = {'vocab_size': 20000, 'd_model': 256}
    net = GalateaNet(config)
    
    # 伪造输入数据 (Batch=2, Seq=100)
    fake_batch = {
        'card_idx': torch.randint(0, 20000, (2, 100)),
        'card_feats': torch.randn(2, 100, 7),
        'global': torch.randn(2, 15),
        'padding_mask': torch.ones(2, 100).bool() # 全部有效
    }
    
    logits, val = net(fake_batch)
    print(f"✅ Logits Shape: {logits.shape} (Expected: [2, 100])")
    print(f"✅ Value Shape: {val.shape} (Expected: [2, 1])")
    print("🎉 网络结构验证通过！")