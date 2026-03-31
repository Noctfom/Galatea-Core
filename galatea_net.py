# ==================================================================================
#  Galatea Network Architecture (Transformer-based)
#  Project Galatea V3.0 - The Semantic Brain
# ==================================================================================

import torch
import torch.nn as nn

class GalateaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        self.vocab_size = config.get('vocab_size', 20000) 
        
        # --- 1. 基础物理感知层 (Physical Embeddings) ---
        self.card_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.feat_proj = nn.Linear(53, self.d_model)
        self.race_embed = nn.Embedding(30, self.d_model, padding_idx=0)
        self.attr_embed = nn.Embedding(10, self.d_model, padding_idx=0)
        self.setcode_embed = nn.Embedding(4096, self.d_model, padding_idx=0) 
        
        self.global_proj = nn.Linear(15, self.d_model)

        # ==========================================================
        # 🌟 2. 语义解析皮层 (Semantic Knowledge Modules)
        # ==========================================================
        self.d_sem = 128 # 语义特征在融合前所在的子空间维度
        
        # A. 主动作与 Hash (词表 4000，足以容纳目前 3415 个特殊效果)
        self.sem_cat_embed = nn.Embedding(4000, self.d_sem, padding_idx=0)
        # B. 发动条件与限制 (128维多热向量直接映射)
        self.sem_req_proj = nn.Linear(128, self.d_sem)
        # C. 关联字段 (与基础 setcode 隔离，专用于效果对象)
        self.sem_setcode_embed = nn.Embedding(4096, self.d_sem, padding_idx=0)
        # D. 魔法数字参数 (4个脱敏数字的提取)
        self.sem_num_proj = nn.Linear(4, self.d_sem)

        self.final_slot_norm = nn.LayerNorm(self.d_model)
        
        # E. 最终融合成 d_model 宽度的降维打击转换器
        self.sem_fusion_proj = nn.Sequential(
            nn.Linear(self.d_sem, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        # ==========================================================

        # --- 3. Transformer Encoder (逻辑推演引擎) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, 
            dim_feedforward=self.d_model * 4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # --- 4. Action Head (动作评估中枢) ---
        self.act_type_embed = nn.Embedding(256, self.d_model) 
        self.desc_embed = nn.Embedding(1024, self.d_model) 
        self.place_embed = nn.Embedding(32, self.d_model, padding_idx=0)
        
        self.intent_proj = nn.Linear(self.d_model, self.d_model)
        self.option_proj = nn.Linear(self.d_model, self.d_model)

        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model * 2, 256), # 双塔拼接，维度翻倍
            nn.ReLU(),
            nn.Linear(256, 1) 
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def process_semantics(self, sem_cat, sem_req, sem_sc, sem_num, sem_ref, sem_race, sem_attr):
        # 核心修复：将 int16 (Short) 强转为 long()，满足 PyTorch Embedding 的要求
        cat_v = self.sem_cat_embed(sem_cat.long()).sum(dim=-2)
        req_v = self.sem_req_proj(sem_req.to(torch.float32))
        sc_v = self.sem_setcode_embed(sem_sc.long()).sum(dim=-2)
        
        # 将 float16 转为 float32 满足 Linear 的要求
        num_v = self.sem_num_proj(sem_num.to(torch.float32))
        
        # 1. 基础语义聚合 (128维)
        sem_base = cat_v + req_v + sc_v + num_v # [B, S, 槽数, 128]
        
        # 2. 升维到 512 维，准备与物理特征接轨
        sem_base_512 = self.sem_fusion_proj(sem_base) # [B, S, 槽数, 512]
        
        # 3. 提取物理共鸣特征 (已经是 512 维了！)
        ref_v = self.card_embed(sem_ref.long()).sum(dim=-2)
        race_v = self.race_embed(sem_race.long()).sum(dim=-2)
        attr_v = self.attr_embed(sem_attr.long()).sum(dim=-2)
        
        # 4. 512维空间的终极融合
        slot_v = sem_base_512 + ref_v + race_v + attr_v
        
        # 将所有效果槽叠加
        card_sem_v = slot_v.sum(dim=-2) 
        return self.final_slot_norm(card_sem_v)

    def forward(self, batch_dict):
        # 物理基础感知
        x_code = self.card_embed(batch_dict['card_idx'])
        x_feat = self.feat_proj(batch_dict['card_feats'])
        x_race = self.race_embed(batch_dict['card_race'])
        x_attr = self.attr_embed(batch_dict['card_attr'])
        x_setcode = self.setcode_embed(batch_dict['card_setcodes']).sum(dim=-2)

        # 接入语义大脑！
        if 'sem_category' in batch_dict:
            x_sem = self.process_semantics(
                batch_dict['sem_category'], batch_dict['sem_req'], 
                batch_dict['sem_setcode'], batch_dict['sem_number'],
                batch_dict['sem_ref'], batch_dict['sem_race'], batch_dict['sem_attr'] # 🌟 补上这行！
            )
        else:
            x_sem = 0

        # 全息物理与语义的大一统！
        x = x_code + x_feat + x_race + x_attr + x_setcode + x_sem
        
        # --- Transformer 局势推演 ---
        src_mask = ~batch_dict['padding_mask'] 
        memory = self.transformer(x, src_key_padding_mask=src_mask)
        
        # --- 全局局面掌控 ---
        g_embed = self.global_proj(batch_dict['global']).unsqueeze(1) 
        masked_memory = memory.masked_fill(src_mask.unsqueeze(-1), -1e9)
        pooled = torch.max(masked_memory, dim=1)[0].unsqueeze(1) 
        
        # --- 上帝视角的语义化 ---
        if 'deck_idx' in batch_dict:
            e_d_code = self.card_embed(batch_dict['deck_idx'])
            e_d_race = self.race_embed(batch_dict['deck_race'])
            e_d_attr = self.attr_embed(batch_dict['deck_attr'])
            e_d_setcode = self.setcode_embed(batch_dict['deck_setcodes']).sum(dim=-2)
            
            if 'd_sem_category' in batch_dict:
                d_sem = self.process_semantics(
                    batch_dict['d_sem_category'], batch_dict['d_sem_req'],
                    batch_dict['d_sem_setcode'], batch_dict['d_sem_number'],
                    batch_dict['d_sem_ref'], batch_dict['d_sem_race'], batch_dict['d_sem_attr'] # 🌟 补上这行！
                )
            else:
                d_sem = 0
                
            x_deck = e_d_code + e_d_race + e_d_attr + e_d_setcode + d_sem # 连卡组都知道自己有什么效果了！
            
            d_mask_f = batch_dict['deck_mask'].float().unsqueeze(-1)
            x_deck_sum = (x_deck * d_mask_f).sum(dim=1)
            d_count = d_mask_f.sum(dim=1).clamp(min=1e-5) 
            deck_pooled = (x_deck_sum / d_count).unsqueeze(1) 
        else:
            deck_pooled = 0
            
        # 连锁雷达：嗅探正在发动的效果！
        if 'c_sem_category' in batch_dict:
            c_sem = self.process_semantics(
                batch_dict['c_sem_category'], batch_dict['c_sem_req'],
                batch_dict['c_sem_setcode'], batch_dict['c_sem_number'],
                batch_dict['c_sem_ref'], batch_dict['c_sem_race'], batch_dict['c_sem_attr']
            ) # [B, 5, 512]
            
            # 使用 mask 求平均
            c_mask_f = batch_dict['c_mask'].float().unsqueeze(-1)
            c_sem_sum = (c_sem * c_mask_f).sum(dim=1)
            c_count = c_mask_f.sum(dim=1).clamp(min=1e-5)
            chain_pooled = (c_sem_sum / c_count).unsqueeze(1) # [B, 1, 512]
        else:
            chain_pooled = 0

        # 大一统评分底蕴
        v_input = g_embed + pooled + deck_pooled + chain_pooled
        value = self.value_head(v_input.squeeze(1)) 

        # === Action Head (因果决策) ===
        act_card_idx = batch_dict['act_card_idx'] # 新形状: [B, 80, 5]
        act_mask = batch_dict['act_mask']         # 形状: [B, 80]
        
        # 把 5 张卡的 512维特征全部扫出来
        idx_expanded = act_card_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model) # [B, 80, 5, 512]
        memory_expanded = memory.unsqueeze(1).expand(-1, act_card_idx.shape[1], -1, -1) # [B, 80, 100, 512]
        gathered_vecs = torch.gather(memory_expanded, 2, idx_expanded) # [B, 80, 5, 512]
        

        is_sort = (batch_dict['act_type'] == 25).unsqueeze(-1).unsqueeze(-1).float() # [B, 80, 1, 1]
        # 创建衰减权重阵：1.0, 0.8, 0.6, 0.4, 0.2
        weights = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2], device=gathered_vecs.device).view(1, 1, 5, 1)
        # 巧妙融合：如果是 25，应用权重；如果不是，全部按 1.0 (等价于 Sum Pooling)
        w = is_sort * weights + (1.0 - is_sort) * 1.0
        
        target_card_vecs = (gathered_vecs * w).sum(dim=2) # [B, 80, 512]

        type_vecs = self.act_type_embed(batch_dict['act_type']) 
        desc_vecs = self.desc_embed(batch_dict['act_desc'])     
        
        act_race_vecs = self.race_embed(batch_dict['act_race'])
        act_attr_vecs = self.attr_embed(batch_dict['act_attr'])
        act_code_vecs = self.card_embed(batch_dict['act_code'])

        place_vecs_raw = self.place_embed(batch_dict['act_place']) # [B, 80, 5, 512]
        place_vecs = place_vecs_raw.sum(dim=2)                     # [B, 80, 512]

        # 终极双塔匹配机制 (Dual-Tower Matching)
        # 1. 意图塔 (Intent)：全局底蕴决定了ai想干什么
        intent_vec = self.intent_proj(v_input) 
        intent_vec = intent_vec.expand(-1, act_mask.shape[1], -1) 
        
        # 2. 选项塔 (Option)：把目标卡片、类型、隐藏语义全部融合
        raw_option = target_card_vecs + type_vecs + desc_vecs + act_race_vecs + act_attr_vecs + act_code_vecs + place_vecs
        option_vec = self.option_proj(raw_option)
        
        # 3. 交汇：意图与选项碰撞
        combined_vecs = torch.cat([intent_vec, option_vec], dim=-1)
        logits = self.policy_head(combined_vecs).squeeze(-1) 
        logits = logits.masked_fill(~act_mask, -1e9)

        return logits, value