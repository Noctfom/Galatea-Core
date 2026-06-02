# ==================================================================================
#  Galatea Network Architecture (Transformer-based)
#  Project Galatea V3.0 - The Semantic Brain
# ==================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class RunningMeanStd(nn.Module):
    # 动态记录输入的均值和方差，用于 RND 归一化
    def __init__(self, shape=()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

class SwiGLU(nn.Module):
    """
    现代化 SwiGLU 门控前馈网络 (取代传统 Linear->GELU->Linear)
    采用无偏置设计与 Tensor Core 硬件对齐优化
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, multiple_of=64):
        super().__init__()
        out_features = out_features or in_features
        # 如果未指定，采用业界标准的 8/3 缩放比例
        hidden_features = hidden_features or int(8 * in_features / 3)
        
        # 硬件级优化：自动向上补齐至 multiple_of (默认64) 的倍数，榨干显卡算力
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        # 核心逻辑：SiLU(Gate) * Up -> Down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class FiLMGenerator(nn.Module):
    """全局状态调制器：根据当前阶段/回合生成 Transformer 的缩放与偏移参数"""
    def __init__(self, condition_dim, d_model):
        super().__init__()
        # 输出 2 倍的 d_model，一半用于乘法缩放(gamma)，一半用于加法偏移(beta)
        self.proj = nn.Linear(condition_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, condition):
        out = self.proj(condition)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1) # [B, 1, d_model] 方便广播

class GalateaTransformerBlock(nn.Module):
    """单层游戏王思考核心：融合 FiLM 宏观调控、SwiGLU 门控逻辑 与 极速 SDPA"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 将 Q, K, V 合并为一个线性层，提速
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(in_features=d_model, multiple_of=64)

    def forward(self, x, padding_mask, gamma, beta):
        # --- 1. 意图调制 + 极速 SDPA (FlashAttention) ---
        residual = x
        x = self.norm1(x)
        x = x * (1.0 + gamma) + beta  # FiLM
        
        B, L, D = x.shape
        # 生成 QKV 并拆分
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # PyTorch SDPA 需要的 mask 是 True 代表保留，False 代表屏蔽
        # 你的 padding_mask 是 True 代表填充物(需屏蔽)，所以要取反 `~`
        attn_mask = (~padding_mask).unsqueeze(1).unsqueeze(2) if padding_mask is not None else None
        
        # 底层级加速调用
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = residual + x

        # --- 2. 意图调制 + 深度门控前馈 ---
        residual = x
        x = self.norm2(x)
        x = x * (1.0 + gamma) + beta  # FiLM
        x = self.ffn(x)
        x = residual + x
        return x

class GalateaTransformerStack(nn.Module):
    """全层 Transformer 堆叠容器：用于完美承接 PyTorch 的 checkpoint 机制"""
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            GalateaTransformerBlock(d_model, n_heads) 
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask, gamma, beta):
        for layer in self.layers:
            x = layer(x, padding_mask, gamma, beta)
        return self.final_norm(x)

class RNDModule(nn.Module): # 内在奖励模块：随机网络蒸馏 (RND),暂时不使用了，先留着代码
    def __init__(self, input_dim=512, hidden_dim=256, out_dim=128): 
        super().__init__()
        # Target 网络保持普通 MLP 且冻结，作为固定的随机指纹
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # Predictor 网络升级为 SwiGLU，加速追赶 target
        self.predictor = SwiGLU(input_features=input_dim, hidden_features=hidden_dim, out_features=out_dim)
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network (努力模仿 Target)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # 归一化输入特征 (防除 0)
        x_norm = (x - self.obs_norm.mean) / torch.sqrt(self.obs_norm.var + 1e-8)
        x_norm = torch.clamp(x_norm, -5.0, 5.0) # 截断极端值

        # 计算预测误差 (MSE) 作为内在奖励
        target_feat = self.target(x_norm)
        pred_feat = self.predictor(x_norm)
        return ((target_feat - pred_feat) ** 2).mean(dim=-1)

class GalateaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        self.vocab_size = config.get('vocab_size', 20000) 
        
        # --- 1. 基础物理感知层 (Physical Embeddings) ---
        self.card_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.feat_proj = nn.Linear(66, self.d_model)
        self.race_embed = nn.Embedding(30, self.d_model, padding_idx=0)
        self.attr_embed = nn.Embedding(10, self.d_model, padding_idx=0)
        self.setcode_embed = nn.Embedding(4096, self.d_model, padding_idx=0) 
        
        self.global_proj = nn.Linear(15, self.d_model)

        # ==========================================================
        # 2. 语义解析皮层 (Semantic Knowledge Modules)
        # ==========================================================
        self.d_sem = 128 # 语义特征在融合前所在的子空间维度
        
        # A. 主动作与 Hash (词表 4000，足以容纳目前的特殊效果)
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
        # 1. 挂载全局环境信号发生器
        self.film_gen = FiLMGenerator(condition_dim=15, d_model=self.d_model)
        
        # 2. 实例化定制的堆叠主干 (利用 config 字典解包)
        self.transformer = GalateaTransformerStack(
            d_model=self.d_model,
            n_heads=config['n_heads'],
            n_layers=config['n_layers']
        )

        # --- 4. Action Head (动作评估中枢) ---
        self.act_type_embed = nn.Embedding(256, self.d_model) 
        self.desc_embed = nn.Embedding(1024, self.d_model) 
        self.place_embed = nn.Embedding(33, self.d_model, padding_idx=0)
        
        # 使用 SwiGLU 将 15 维的全局状态精准升维
        self.intent_proj = SwiGLU(in_features=self.d_model, hidden_features=512, out_features=self.d_model)
        self.option_proj = nn.Linear(self.d_model, self.d_model)

        self.v_norm = nn.LayerNorm(self.d_model)
        self.fusion_norm = nn.LayerNorm(self.d_model * 2) # 双塔拼接后是 d_model * 2
        # 链式效果位置编码，最大长度 12 (需要与 feature_encoder.py 里的常量保持绝对一致)
        self.chain_pos_embed = nn.Parameter(torch.randn(1, 12, self.d_model) * 0.02)
        # 设定为 MAX_HISTORY = 8 (需要与 feature_encoder.py 里的常量保持绝对一致)
        self.history_pos_embed = nn.Parameter(torch.randn(1, 8, self.d_model) * 0.02)

        self.register_buffer("place_weights", torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]).view(1, 1, 5, 1))

        # 新代码：处理拼接后的两倍特征 (d_model * 2)，执行更深度的逻辑门控
        self.policy_head = nn.Sequential(
            SwiGLU(in_features=self.d_model * 2, hidden_features=512, out_features=256),
            nn.Linear(256, 1)
        )

        # 新代码：SwiGLU 过滤无效特征，再接一个 Linear 映射为单个估值标量
        self.value_head = nn.Sequential(
            SwiGLU(in_features=self.d_model, hidden_features=512, out_features=256),
            nn.Linear(256, 1)
        )

        #self.rnd = RNDModule(input_dim=self.d_model)
        self.overlay_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, 120, self.d_model) * 0.02) # 最大卡片数(SeqLen)是120

        for m in self.policy_head.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
                
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear) and m.out_features == 1:
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

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
        safe_sem_ref = torch.clamp(sem_ref.long(), 0, self.vocab_size - 1)
        ref_v = self.card_embed(safe_sem_ref).sum(dim=-2)
        race_v = self.race_embed(sem_race.long()).sum(dim=-2)
        attr_v = self.attr_embed(sem_attr.long()).sum(dim=-2)
        
        # 4. 512维空间的终极融合
        slot_v = sem_base_512 + ref_v + race_v + attr_v
        
        # 将所有效果槽叠加
        card_sem_v = slot_v.sum(dim=-2) 
        return self.final_slot_norm(card_sem_v)

    def forward(self, batch_dict):
        # --- 全局状态调制器 ---
        gamma, beta = self.film_gen(batch_dict['global'])

        # 物理基础感知
        x_code = self.card_embed(batch_dict['card_idx'])
        x_overlay = self.overlay_embed(batch_dict['card_overlay_idx'])
        x_feat = self.feat_proj(batch_dict['card_feats'])
        x_race = self.race_embed(batch_dict['card_race'])
        x_attr = self.attr_embed(batch_dict['card_attr'])
        x_setcode = self.setcode_embed(batch_dict['card_setcodes']).sum(dim=-2)

        # 接入语义大脑！
        if 'sem_category' in batch_dict:
            x_sem = self.process_semantics(
                batch_dict['sem_category'], batch_dict['sem_req'], 
                batch_dict['sem_setcode'], batch_dict['sem_number'],
                batch_dict['sem_ref'], batch_dict['sem_race'], batch_dict['sem_attr']
            )
        else:
            x_sem = 0.0

        # 全息物理与语义的大一统！
        x = x_code + x_overlay + x_feat + x_race + x_attr + x_setcode + x_sem
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        
        # --- Transformer 局势推演 ---
        src_mask = ~batch_dict['padding_mask'] 
        
        if self.training:
            # 强行向 PyTorch 声明这是一个需要计算梯度的连续隐空间，同时带入控制信号
            memory = checkpoint(self.transformer, x, src_mask, gamma, beta, use_reentrant=False)
        else:
            memory = self.transformer(x, src_mask, gamma, beta)
        
        # --- 全局局面掌控 ---
        g_embed = self.global_proj(batch_dict['global']).unsqueeze(1) 
        masked_memory = memory.masked_fill(src_mask.unsqueeze(-1), -1e4)
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
            
            seq_len = c_sem.shape[1]
            c_sem = c_sem + self.chain_pos_embed[:, :seq_len, :]

            # 使用 mask 求平均
            c_mask_f = batch_dict['c_mask'].float().unsqueeze(-1)
            c_sem_sum = (c_sem * c_mask_f).sum(dim=1)
            c_count = c_mask_f.sum(dim=1).clamp(min=1e-5)
            chain_pooled = (c_sem_sum / c_count).unsqueeze(1) # [B, 1, 512]
        else:
            chain_pooled = 0
        
        # 历史动作雷达：回想过去 8 步的施法记录
        if 'h_sem_category' in batch_dict:
            h_sem = self.process_semantics(
                batch_dict['h_sem_category'], batch_dict['h_sem_req'],
                batch_dict['h_sem_setcode'], batch_dict['h_sem_number'],
                batch_dict['h_sem_ref'], batch_dict['h_sem_race'], batch_dict['h_sem_attr']
            ) # [B, 8, 512]

            # 注入历史时序 (近期发生的在前面，远期发生的在后面)
            seq_len = h_sem.shape[1] 
            h_sem = h_sem + self.history_pos_embed[:, :seq_len, :]
            
            h_mask_f = batch_dict['h_mask'].float().unsqueeze(-1)
            h_sem_sum = (h_sem * h_mask_f).sum(dim=1)
            h_count = h_mask_f.sum(dim=1).clamp(min=1e-5)
            history_pooled = (h_sem_sum / h_count).unsqueeze(1) # [B, 1, 512]
        else:
            history_pooled = 0

        # 大一统评分底蕴：加入历史记忆
        v_input = g_embed + pooled + deck_pooled + chain_pooled + history_pooled
        v_input = self.v_norm(v_input)
        value = self.value_head(v_input.squeeze(1)) 

        # === Action Head (因果决策) ===
        act_card_idx = batch_dict['act_card_idx'] # 新形状: [B, 80, 5]
        act_mask = batch_dict['act_mask']         # 形状: [B, 80]
        
        B, A, M = act_card_idx.shape
        D = self.d_model
        
        # 1. 把索引展平 [B, 80, 5] -> [B, 400]
        flat_idx = act_card_idx.view(B, A * M)
        # 2. 扩充最后一个维度对接 d_model -> [B, 400, 512]
        flat_idx_expanded = flat_idx.unsqueeze(-1).expand(-1, -1, D)

        # 原本 memory 是 [B, 120, 512]，现在变成 [B, 121, 512]
        padding_vec = torch.zeros(B, 1, D, device=memory.device)
        memory_padded = torch.cat([memory, padding_vec], dim=1)

        # 3. 直接从原始 memory [B, 120, 512] 中捞取，彻底规避 4D 梯度爆炸
        gathered_flat = torch.gather(memory_padded, 1, flat_idx_expanded) # [B, 400, 512]
        # 4. 重新捏回需要的形状 -> [B, 80, 5, 512]
        gathered_vecs = gathered_flat.view(B, A, M, D)
        # =========================================================

        is_sort = (batch_dict['act_type'] == 25).unsqueeze(-1).unsqueeze(-1).float() # [B, 80, 1, 1]
        # 创建衰减权重阵：1.0, 0.8, 0.6, 0.4, 0.2
        weights = self.place_weights
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

        combined_vecs = self.fusion_norm(combined_vecs) # 双塔融合后的 LayerNorm

        logits = self.policy_head(combined_vecs).squeeze(-1) 
        logits = logits.masked_fill(~act_mask, -1e4)

        return logits, value, v_input.squeeze(1)
    
    def update_rnd_stats(self, v_input):
        with torch.no_grad():
            self.rnd.obs_norm.update(v_input)