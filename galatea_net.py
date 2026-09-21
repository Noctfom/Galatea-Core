# ==================================================================================
#  Galatea Network Architecture (Transformer-based)
#  Project Galatea V3.0 - The Semantic Brain
# ==================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from data_types import (
    ACTION_CONTEXT_DIM,
    ACTION_OPERATION_COUNT,
    ACTION_RESPONSE_BUCKETS,
    ACTION_SIGNATURE_BYTES,
    ACTION_TARGET_SLOTS,
    CHAIN_CONTEXT_DIM,
    CARD_NUMERIC_FEATURE_DIM,
    CARD_RELATION_TYPE_COUNT,
    CARD_STATUS_BITS,
    FIELD_ZONE_BITS,
    GLOBAL_FEATURE_DIM,
    PHASE_CATEGORY_COUNT,
    PLAYER_CONTEXT_SLOTS,
    PLAYER_ROLE_COUNT,
    POSITION_CATEGORY_COUNT,
    SUMMON_METHOD_COUNT,
    ZONE_CATEGORY_COUNT,
)
from protocol_schema import (
    MODEL_PROTOCOL_VERSION,
    apply_current_protocol_metadata,
)
from semantic_lookup import get_static_semantic_lookup

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


class OrderedContextPool(nn.Module):
    """用局部顺序混合和注意力池化保留短序列中的先后关系"""

    def __init__(self, d_model, max_length):
        super().__init__()
        score_hidden = max(8, d_model // 4)
        self.position_embed = nn.Parameter(
            torch.randn(1, max_length, d_model) * 0.02
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.local_mixer = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
            bias=False,
        )
        self.channel_mixer = nn.Linear(d_model, d_model, bias=False)
        self.output_norm = nn.LayerNorm(d_model)
        self.attention_score = nn.Sequential(
            nn.Linear(d_model, score_hidden),
            nn.Tanh(),
            nn.Linear(score_hidden, 1, bias=False),
        )

    def forward(self, tokens, valid_mask):
        """按固定槽位编码相邻关系，并只汇聚掩码标记的有效项"""
        sequence_length = tokens.shape[1]
        mask = valid_mask.bool()
        mask_values = mask.unsqueeze(-1).to(tokens.dtype)

        ordered = (
            tokens + self.position_embed[:, :sequence_length, :]
        ) * mask_values
        local_context = self.local_mixer(
            self.input_norm(ordered).transpose(1, 2)
        ).transpose(1, 2)
        local_context = self.channel_mixer(F.gelu(local_context))
        mixed = self.output_norm(ordered + local_context) * mask_values

        scores = self.attention_score(mixed).squeeze(-1).float()
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        return (
            mixed * weights.unsqueeze(-1).to(mixed.dtype)
        ).sum(dim=1)

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
        config = apply_current_protocol_metadata(config)
        self.model_protocol_version = MODEL_PROTOCOL_VERSION
        self.protocol_schema_revision = config['protocol_schema_revision']
        self.protocol_schema_hash = config['protocol_schema_hash']
        self.card_vocab_hash = config['card_vocab_hash']
        self.card_vocab_size = config['card_vocab_size']
        self.card_vocab_card_count = config['card_vocab_card_count']
        self.semantic_lookup_format_version = config[
            'semantic_lookup_format_version'
        ]
        self.semantic_lookup_hash = config['semantic_lookup_hash']
        self.semantic_lookup_card_count = config[
            'semantic_lookup_card_count'
        ]
        self.register_buffer(
            '_model_protocol_version',
            torch.tensor(MODEL_PROTOCOL_VERSION, dtype=torch.int32),
            persistent=True,
        )
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        self.vocab_size = config['vocab_size']

        semantic_lookup = get_static_semantic_lookup()
        self.register_buffer(
            'code_dict',
            torch.from_numpy(semantic_lookup.code_dictionary),
            persistent=False,
        )
        # 静态语义随模型驻留一次，逐步观测只需传递卡片与效果槽身份
        semantic_buffers = {
            'semantic_category_table': semantic_lookup.category,
            'semantic_requirement_table': semantic_lookup.requirement,
            'semantic_setcode_table': semantic_lookup.setcode,
            'semantic_number_table': semantic_lookup.number,
            'semantic_reference_table': semantic_lookup.reference,
            'semantic_race_table': semantic_lookup.race,
            'semantic_attribute_table': semantic_lookup.attribute,
            'semantic_code_index_table': semantic_lookup.code_index,
            'semantic_effect_mask_table': semantic_lookup.effect_mask,
        }
        for name, array in semantic_buffers.items():
            self.register_buffer(
                name,
                torch.from_numpy(array),
                persistent=False,
            )
        
        # --- 1. 基础物理感知层 (Physical Embeddings) ---
        self.card_embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.feat_proj = nn.Linear(CARD_NUMERIC_FEATURE_DIM, self.d_model)
        self.race_embed = nn.Embedding(30, self.d_model, padding_idx=0)
        self.attr_embed = nn.Embedding(10, self.d_model, padding_idx=0)
        self.setcode_embed = nn.Embedding(4096, self.d_model, padding_idx=0) 
        
        self.global_proj = nn.Linear(GLOBAL_FEATURE_DIM, self.d_model)
        self.player_role_embeds = nn.ModuleList(
            [
                nn.Embedding(PLAYER_ROLE_COUNT, self.d_model, padding_idx=0)
                for _ in range(PLAYER_CONTEXT_SLOTS)
            ]
        )
        self.phase_context_embed = nn.Embedding(
            PHASE_CATEGORY_COUNT,
            16,
            padding_idx=0,
        )
        self.phase_token_proj = nn.Linear(16, self.d_model, bias=False)
        self.zone_embed = nn.Embedding(
            ZONE_CATEGORY_COUNT,
            self.d_model,
            padding_idx=0,
        )
        self.position_embed = nn.Embedding(
            POSITION_CATEGORY_COUNT,
            self.d_model,
            padding_idx=0,
        )
        # V4 公共卡片状态使用紧凑投影，避免为每类 bit 建立庞大词表
        self.card_reason_proj = nn.Linear(32, self.d_model, bias=False)
        self.card_status_proj = nn.Linear(
            CARD_STATUS_BITS, self.d_model, bias=False
        )
        self.card_owner_role_embed = nn.Embedding(
            PLAYER_ROLE_COUNT, self.d_model, padding_idx=0
        )
        self.card_relation_type_embed = nn.Embedding(
            CARD_RELATION_TYPE_COUNT, self.d_model, padding_idx=0
        )
        self.card_counter_proj = nn.Linear(17, self.d_model, bias=False)
        self.field_zone_mask_proj = nn.Linear(
            FIELD_ZONE_BITS, self.d_model, bias=False
        )
        self.register_buffer(
            '_bit_masks',
            torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.long),
            persistent=False,
        )

        # ==========================================================
        # 2. 语义解析皮层 (Semantic Knowledge Modules)
        # ==========================================================
        self.d_sem = 128 # 语义特征在融合前所在的子空间维度
        
        # A. 主动作与 Hash (词表 4000，足以容纳目前的特殊效果)
        self.sem_cat_embed = nn.Embedding(4000, self.d_sem, padding_idx=0)
        self.code_vec_proj = nn.Sequential(
            nn.Linear(384, self.d_sem),
            nn.GELU(),
            nn.LayerNorm(self.d_sem)
        )
        # B. 发动条件与限制 (128维多热向量直接映射)
        self.sem_req_proj = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.d_sem)
        )
        # C. 关联字段 (与基础 setcode 隔离，专用于效果对象)
        self.sem_setcode_embed = nn.Embedding(4096, self.d_sem, padding_idx=0)
        # D. 魔法数字参数 (4个脱敏数字的提取)
        self.sem_num_proj = nn.Linear(4, self.d_sem)

        self.final_slot_norm = nn.LayerNorm(self.d_model)
        self.effect_slot_embed = nn.Embedding(8, self.d_model)

        self.slot_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,  # 你的特征融合后维度 (假设 d_sem 等拼接后是 512)
            num_heads=self.n_heads, 
            batch_first=True
        )
        self.query_proj = nn.Linear(self.d_model, self.d_model) # 用于生成 Slot Attention 的查询向量
        
        # E. 最终融合成 d_model 宽度的降维打击转换器
        self.sem_fusion_proj = nn.Sequential(
            nn.Linear(self.d_sem, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        # ==========================================================

        # --- 3. Transformer Encoder (逻辑推演引擎) ---
        # 1. 挂载全局环境信号发生器
        self.film_gen = FiLMGenerator(
            condition_dim=GLOBAL_FEATURE_DIM + 16,
            d_model=self.d_model,
        )
        
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
        self.action_operation_embed = nn.Embedding(
            ACTION_OPERATION_COUNT, self.d_model, padding_idx=0
        )
        self.action_summon_method_embed = nn.Embedding(
            SUMMON_METHOD_COUNT, self.d_model, padding_idx=0
        )
        self.action_response_embed = nn.Embedding(
            ACTION_RESPONSE_BUCKETS, self.d_model, padding_idx=0
        )
        self.action_signature_embeds = nn.ModuleList(
            [nn.Embedding(256, self.d_model) for _ in range(ACTION_SIGNATURE_BYTES)]
        )
        self.action_context_proj = nn.Linear(
            ACTION_CONTEXT_DIM, self.d_model, bias=False
        )
        self.action_target_value_proj = nn.Linear(2, self.d_model, bias=False)
        self.action_controller_embed = nn.Embedding(3, self.d_model, padding_idx=0)
        self.action_location_embed = nn.Embedding(
            ZONE_CATEGORY_COUNT,
            self.d_model,
            padding_idx=0,
        )
        self.action_sequence_embed = nn.Embedding(33, self.d_model, padding_idx=0)
        
        # 使用 SwiGLU 将 15 维的全局状态精准升维
        self.intent_proj = SwiGLU(in_features=self.d_model, hidden_features=512, out_features=self.d_model)
        self.option_proj = nn.Linear(self.d_model, self.d_model)

        self.v_norm = nn.LayerNorm(self.d_model)
        self.fusion_norm = nn.LayerNorm(self.d_model * 2) # 双塔拼接后是 d_model * 2
        # 连锁与历史使用顺序敏感聚合，避免位置向量在普通均值中被数学抵消
        self.chain_context_pool = OrderedContextPool(self.d_model, 12)
        self.history_context_pool = OrderedContextPool(self.d_model, 8)
        self.chain_metadata_proj = nn.Linear(
            CHAIN_CONTEXT_DIM,
            self.d_model,
            bias=False,
        )

        self.register_buffer(
            "place_weights",
            torch.linspace(1.0, 0.2, ACTION_TARGET_SLOTS).view(
                1, 1, ACTION_TARGET_SLOTS, 1
            ),
        )

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

    def process_semantics(
        self,
        sem_cat,
        sem_req,
        sem_sc,
        sem_num,
        sem_ref,
        sem_race,
        sem_attr,
        sem_code_idx,
        sem_mask,
        feat_vecs=None,
        return_slots=False,
    ):
        """聚合 Lua 代码语义；按需同时返回保留槽身份的效果向量"""
        cat_v = self.sem_cat_embed(sem_cat.long()).sum(dim=-2)
        
        # 残差融合代码语义
        sem_code_vec = F.embedding(sem_code_idx.long(), self.code_dict)
        code_v = self.code_vec_proj(sem_code_vec)
        cat_v = cat_v + code_v
        
        # 修复：兼容旧版本权重与新版超压缩索引的 GPU 极速还原
        if sem_req.dtype == torch.int8 or sem_req.shape[-1] == 16:
            B, N, S, K = sem_req.shape
            indices_flat = sem_req.view(-1, K).long()
            valid_mask = (indices_flat >= 0).float()
            indices_clamped = indices_flat.clamp(min=0)
            
            # 使用 scatter_ 零损耗展开回 128 维 One-Hot
            req_multi_hot = torch.zeros(indices_flat.size(0), 128, device=sem_req.device, dtype=torch.float32)
            req_multi_hot.scatter_(1, indices_clamped, valid_mask)
            req_multi_hot = req_multi_hot.view(B, N, S, 128)
        else:
            req_multi_hot = sem_req.to(torch.float32)
            
        req_v = self.sem_req_proj(req_multi_hot)
        sc_v = self.sem_setcode_embed(sem_sc.long()).sum(dim=-2)
        num_v = self.sem_num_proj(sem_num.to(torch.float32))
        
        sem_base = cat_v + req_v + sc_v + num_v 
        sem_base_512 = self.sem_fusion_proj(sem_base) 
        
        safe_sem_ref = torch.clamp(sem_ref.long(), 0, self.vocab_size - 1)
        ref_v = self.card_embed(safe_sem_ref).sum(dim=-2)
        race_v = self.race_embed(sem_race.long()).sum(dim=-2)
        attr_v = self.attr_embed(sem_attr.long()).sum(dim=-2)
        
        slot_positions = torch.arange(
            sem_base_512.shape[2],
            device=sem_base_512.device,
        )
        slot_position_v = self.effect_slot_embed(slot_positions).view(
            1, 1, -1, self.d_model
        )
        # 效果槽身份使 used_effect_mask 的第 N 位能够对应到第 N 个语义效果
        slot_v = sem_base_512 + ref_v + race_v + attr_v + slot_position_v
        
        B, N, S, D = slot_v.shape
        valid_semantic_slots = sem_mask.view(B * N, S)
        empty_semantic_rows = ~valid_semantic_slots.any(dim=1)
        # 全掩码注意力在部分 ONNX 后端会产生 NaN；临时开放零值哨兵，聚合后再归零
        fallback_slot = torch.arange(S, device=sem_mask.device).eq(0).view(1, S)
        safe_semantic_slots = valid_semantic_slots | (
            empty_semantic_rows.unsqueeze(1) & fallback_slot
        )
        slot_v_flat = slot_v.view(B * N, S, D)
        slot_v_flat = slot_v_flat * valid_semantic_slots.unsqueeze(-1)
        
        # 【核心】：如果有物理特征传入，用物理特征去查；否则用默认意图查
        if feat_vecs is not None:
            query_raw = feat_vecs.view(B * N, self.d_model)
            query = self.query_proj(query_raw).view(B * N, 1, D)
        else:
            query = sem_base_512[:, :, 0, :].view(B * N, 1, D)
        
        # 【核心】：构建 Padding Mask，屏蔽全 0 的无效效果槽
        key_padding_mask = ~safe_semantic_slots # PyTorch中True代表忽略

        attn_out, _ = self.slot_attention(
            query, slot_v_flat, slot_v_flat, 
            key_padding_mask=key_padding_mask
        )
        
        card_sem_v = self.final_slot_norm(attn_out.view(B, N, D))
        card_sem_v = card_sem_v.masked_fill(
            empty_semantic_rows.view(B, N, 1),
            0.0,
        )
        if return_slots:
            return card_sem_v, slot_v
        return card_sem_v

    def lookup_static_semantics(self, card_ids, effect_slots=None):
        """按卡片 token 与可选效果槽身份还原完整静态语义张量"""
        safe_card_ids = card_ids.long().clamp(
            0,
            self.semantic_category_table.shape[0] - 1,
        )
        semantic_inputs = (
            self.semantic_category_table[safe_card_ids],
            self.semantic_requirement_table[safe_card_ids],
            self.semantic_setcode_table[safe_card_ids],
            self.semantic_number_table[safe_card_ids],
            self.semantic_reference_table[safe_card_ids],
            self.semantic_race_table[safe_card_ids],
            self.semantic_attribute_table[safe_card_ids],
            self.semantic_code_index_table[safe_card_ids],
        )
        semantic_mask = self.semantic_effect_mask_table[safe_card_ids]

        # 复刻旧编码器的空语义哨兵，保证隐藏卡与空槽数值完全一致
        empty_rows = ~semantic_mask.any(dim=-1)
        fallback_slot = torch.arange(
            semantic_mask.shape[-1],
            device=semantic_mask.device,
        ).eq(0)
        semantic_mask = semantic_mask | (
            empty_rows.unsqueeze(-1) & fallback_slot
        )

        if effect_slots is not None:
            slot_ids = effect_slots.long()
            safe_slot_ids = (slot_ids - 1).clamp(
                0,
                semantic_mask.shape[-1] - 1,
            )
            selected_valid = torch.gather(
                semantic_mask,
                -1,
                safe_slot_ids.unsqueeze(-1),
            ).squeeze(-1)
            known_slots = (
                (slot_ids > 0)
                & (slot_ids <= semantic_mask.shape[-1])
                & selected_valid
            )
            focused_mask = torch.nn.functional.one_hot(
                safe_slot_ids,
                num_classes=semantic_mask.shape[-1],
            ).to(torch.bool)
            focus_selector = known_slots.unsqueeze(-1)
            semantic_mask = (
                (focused_mask & focus_selector)
                | (semantic_mask & ~focus_selector)
            )
        return (*semantic_inputs, semantic_mask)

    def _unpack_bit_bytes(self, packed):
        """把轨迹中的紧凑字节无损展开为供线性层使用的逐位特征"""
        expanded = torch.bitwise_and(
            packed.long().unsqueeze(-1),
            self._bit_masks,
        ).ne(0)
        return expanded.flatten(start_dim=-2).to(torch.float32)

    def forward(self, batch_dict):
        # --- 全局状态调制器 ---
        phase_context = self.phase_context_embed(
            batch_dict['phase'][:, 0].long()
        )
        film_context = torch.cat(
            [batch_dict['global'], phase_context],
            dim=-1,
        )
        gamma, beta = self.film_gen(film_context)

        # 物理基础感知
        x_code = self.card_embed(batch_dict['card_idx'])
        x_alias = self.card_embed(batch_dict['card_alias_idx'].long())
        x_overlay = self.overlay_embed(batch_dict['card_overlay_idx'])
        overlay_rest_idx = batch_dict['card_overlay_rest_idx'].long()
        overlay_rest_mask = overlay_rest_idx.ne(0).unsqueeze(-1)
        x_overlay_rest = self.overlay_embed(overlay_rest_idx)
        x_overlay_rest = (x_overlay_rest * overlay_rest_mask).sum(dim=-2) / (
            overlay_rest_mask.sum(dim=-2).clamp(min=1).to(x_overlay_rest.dtype)
        )
        x_feat = self.feat_proj(batch_dict['card_feats'])
        x_race = self.race_embed(batch_dict['card_race'])
        x_attr = self.attr_embed(batch_dict['card_attr'])
        x_setcode = self.setcode_embed(batch_dict['card_setcodes']).sum(dim=-2)
        x_reason = self.card_reason_proj(
            self._unpack_bit_bytes(batch_dict['card_reason_bytes'])
        )
        x_status = self.card_status_proj(
            batch_dict['card_status_bits'].to(torch.float32)
        )
        x_owner = self.card_owner_role_embed(
            batch_dict['card_owner_role'].long()
        )
        counter_input = torch.cat(
            [
                self._unpack_bit_bytes(batch_dict['card_counter_type_bytes']),
                batch_dict['card_counter_value'].to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )
        counter_mask = batch_dict['card_counter_type_bytes'].ne(0).any(dim=-1).unsqueeze(-1)
        x_counter_slots = self.card_counter_proj(counter_input)
        x_counter = (x_counter_slots * counter_mask).sum(dim=-2) / (
            counter_mask.sum(dim=-2).clamp(min=1).to(x_counter_slots.dtype)
        )

        # 接入语义大脑；生产路径只携带 card_idx，旧展开路径仅保留给数值等价测试
        if 'sem_category' in batch_dict:
            x_sem, x_sem_slots = self.process_semantics(
                batch_dict['sem_category'], batch_dict['sem_req'],
                batch_dict['sem_setcode'], batch_dict['sem_number'],
                batch_dict['sem_ref'], batch_dict['sem_race'], batch_dict['sem_attr'],
                batch_dict['sem_code_idx'], batch_dict['sem_mask'], x_feat,
                return_slots=True,
            )
        else:
            x_sem, x_sem_slots = self.process_semantics(
                *self.lookup_static_semantics(batch_dict['card_idx']),
                x_feat,
                return_slots=True,
            )
        # 全息物理与语义的大一统！
        x_zone = self.zone_embed(batch_dict['card_zone'].long())
        x_position = self.position_embed(batch_dict['card_position'].long())
        x_base = (
            x_code + x_alias + x_overlay + x_overlay_rest + x_feat
            + x_race + x_attr + x_setcode + x_reason + x_status
            + x_owner + x_counter + x_sem + x_zone + x_position
        )
        relation_idx = batch_dict['card_relation_idx'].long()
        relation_type = batch_dict['card_relation_type'].long()
        batch_size, card_count, relation_count = relation_idx.shape
        relation_padding = torch.zeros(
            batch_size, 1, self.d_model, device=x_base.device, dtype=x_base.dtype
        )
        relation_source = torch.cat([x_base, relation_padding], dim=1)
        flat_relation_idx = relation_idx.clamp(0, card_count).reshape(
            batch_size, card_count * relation_count
        )
        gathered_relations = torch.gather(
            relation_source,
            1,
            flat_relation_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
        ).reshape(batch_size, card_count, relation_count, self.d_model)
        relation_mask = (
            relation_type.ne(0) & relation_idx.ge(0) & relation_idx.lt(card_count)
        ).unsqueeze(-1)
        relation_vectors = gathered_relations + self.card_relation_type_embed(
            relation_type
        )
        x_relation = (relation_vectors * relation_mask).sum(dim=-2) / (
            relation_mask.sum(dim=-2).clamp(min=1).to(relation_vectors.dtype)
        )
        x = x_base + x_relation
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
        player_context = batch_dict['player_context'].long()
        player_role_embed = sum(
            embedding(player_context[:, slot])
            for slot, embedding in enumerate(self.player_role_embeds)
        )
        g_embed = (
            self.global_proj(batch_dict['global'])
            + player_role_embed
            + self.phase_token_proj(phase_context)
            + self.field_zone_mask_proj(
                self._unpack_bit_bytes(batch_dict['field_zone_mask'])
            )
        ).unsqueeze(1)
        masked_memory = memory.masked_fill(src_mask.unsqueeze(-1), -65000.0)
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
                    batch_dict['d_sem_ref'], batch_dict['d_sem_race'], batch_dict['d_sem_attr'],
                    batch_dict['d_sem_code_idx'], batch_dict['d_sem_mask'], None
                )
            else:
                d_sem = self.process_semantics(
                    *self.lookup_static_semantics(batch_dict['deck_idx']),
                    None,
                )
            x_deck = e_d_code + e_d_race + e_d_attr + e_d_setcode + d_sem # 连卡组都知道自己有什么效果了！
            
            d_mask_f = batch_dict['deck_mask'].float().unsqueeze(-1)
            x_deck_sum = (x_deck * d_mask_f).sum(dim=1)
            d_count = d_mask_f.sum(dim=1).clamp(min=1e-5) 
            deck_pooled = (x_deck_sum / d_count).unsqueeze(1) 
        else:
            deck_pooled = 0
            
        # 连锁雷达：嗅探正在发动的效果！
        if 'c_card_idx' in batch_dict:
            if 'c_sem_category' in batch_dict:
                c_semantic_inputs = (
                    batch_dict['c_sem_category'], batch_dict['c_sem_req'],
                    batch_dict['c_sem_setcode'], batch_dict['c_sem_number'],
                    batch_dict['c_sem_ref'], batch_dict['c_sem_race'], batch_dict['c_sem_attr'],
                    batch_dict['c_sem_code_idx'], batch_dict['c_sem_mask'],
                )
            else:
                c_semantic_inputs = self.lookup_static_semantics(
                    batch_dict['c_card_idx'],
                    batch_dict['c_effect_slot'],
                )
            c_sem = self.process_semantics(
                *c_semantic_inputs,
                None,
            ) # [B, 12, 512]
            c_sem = (
                c_sem
                + self.card_embed(batch_dict['c_card_idx'].long())
                + self.desc_embed(batch_dict['c_desc'].long())
                + self.zone_embed(batch_dict['c_zone'].long()).sum(dim=2)
                + self.position_embed(batch_dict['c_position'].long())
                + self.chain_metadata_proj(
                    batch_dict['c_context'].to(torch.float32)
                )
            )
            
            chain_pooled = self.chain_context_pool(
                c_sem,
                batch_dict['c_mask'],
            ).unsqueeze(1)
        else:
            chain_pooled = 0
        
        # 历史动作雷达：回想过去 8 步的施法记录
        if 'h_card_idx' in batch_dict:
            if 'h_sem_category' in batch_dict:
                h_semantic_inputs = (
                    batch_dict['h_sem_category'], batch_dict['h_sem_req'],
                    batch_dict['h_sem_setcode'], batch_dict['h_sem_number'],
                    batch_dict['h_sem_ref'], batch_dict['h_sem_race'], batch_dict['h_sem_attr'],
                    batch_dict['h_sem_code_idx'], batch_dict['h_sem_mask'],
                )
            else:
                h_semantic_inputs = self.lookup_static_semantics(
                    batch_dict['h_card_idx'],
                    batch_dict['h_effect_slot'],
                )
            h_sem = self.process_semantics(
                *h_semantic_inputs,
                None,
            ) # [B, 8, 512]

            history_pooled = self.history_context_pool(
                h_sem,
                batch_dict['h_mask'],
            ).unsqueeze(1)
        else:
            history_pooled = 0

        # 大一统评分底蕴：加入历史记忆
        v_input = g_embed + pooled + deck_pooled + chain_pooled + history_pooled
        v_input = self.v_norm(v_input)
        value = self.value_head(v_input.squeeze(1)) 

        # === Action Head (因果决策) ===
        act_card_idx = batch_dict['act_card_idx'] # 新形状: [B, 120, 5]
        act_mask = batch_dict['act_mask']         # 形状: [B, 120]
        
        B, A, M = act_card_idx.shape
        D = self.d_model
        
        # 1. 把索引展平 [B, 120, 5] -> [B, 600]
        flat_idx = act_card_idx.view(B, A * M)
        # 2. 扩充最后一个维度对接 d_model -> [B, 400, 512]
        flat_idx_expanded = flat_idx.unsqueeze(-1).expand(-1, -1, D)

        # 原本 memory 是 [B, 120, 512]，现在变成 [B, 121, 512]
        padding_vec = torch.zeros(B, 1, D, device=memory.device)
        memory_padded = torch.cat([memory, padding_vec], dim=1)

        # 3. 直接从原始 memory [B, 120, 512] 中捞取，彻底规避 4D 梯度爆炸
        gathered_flat = torch.gather(memory_padded, 1, flat_idx_expanded) # [B, 400, 512]
        # 4. 重新捏回需要的形状 -> [B, 120, 5, 512]
        gathered_vecs = gathered_flat.view(B, A, M, D)
        # =========================================================

        is_sort = (batch_dict['act_type'] == 25).unsqueeze(-1).unsqueeze(-1).float() # [B, 120, 1, 1]
        # 创建衰减权重阵：1.0, 0.8, 0.6, 0.4, 0.2
        weights = self.place_weights
        # 巧妙融合：如果是 25，应用权重；如果不是，全部按 1.0 (等价于 Sum Pooling)
        w = is_sort * weights + (1.0 - is_sort) * 1.0
        
        target_card_vecs = (gathered_vecs * w).sum(dim=2) # [B, 120, 512]

        type_vecs = self.act_type_embed(batch_dict['act_type']) 
        desc_vecs = self.desc_embed(batch_dict['act_desc'])     
        
        act_race_vecs = self.race_embed(batch_dict['act_race'])
        act_attr_vecs = self.attr_embed(batch_dict['act_attr'])
        act_code_vecs = self.card_embed(batch_dict['act_code'])

        # 通过实体指针和精确 Lua 槽位取出本次动作对应的代码语义；未知绑定保持零向量
        action_effect_vecs = torch.zeros_like(act_code_vecs)
        if x_sem_slots is not None:
            effect_slot_ids = batch_dict['act_effect_slot'].long()
            source_entity_ids = act_card_idx[..., 0].long()
            safe_source_entity_ids = source_entity_ids.clamp(
                0, x_sem_slots.shape[1] - 1
            )
            source_card_ids = torch.gather(
                batch_dict['card_idx'].long(),
                1,
                safe_source_entity_ids,
            )
            known_effects = (
                (effect_slot_ids > 0)
                & (effect_slot_ids <= x_sem_slots.shape[2])
                & (source_entity_ids >= 0)
                & (source_entity_ids < x_sem_slots.shape[1])
                & (source_card_ids == batch_dict['act_code'].long())
            )
            flat_slots = x_sem_slots.reshape(
                B,
                x_sem_slots.shape[1] * x_sem_slots.shape[2],
                D,
            )
            safe_slot_ids = (effect_slot_ids - 1).clamp(
                0, x_sem_slots.shape[2] - 1
            )
            flat_effect_ids = (
                safe_source_entity_ids * x_sem_slots.shape[2]
                + safe_slot_ids
            )
            action_effect_vecs = torch.gather(
                flat_slots,
                1,
                flat_effect_ids[..., None].expand(-1, -1, D),
            )
            action_effect_vecs = self.final_slot_norm(action_effect_vecs)
            action_effect_vecs = action_effect_vecs * known_effects.unsqueeze(-1)

        place_vecs_raw = self.place_embed(batch_dict['act_place']) # [B, 120, 5, 512]
        place_vecs = place_vecs_raw.sum(dim=2)                     # [B, 120, 512]

        # 动作协议 V2：显式融合响应语义、约束、隐藏目标代码与素材数值
        operation_vecs = self.action_operation_embed(
            batch_dict['act_operation'].long()
        )
        summon_method_vecs = self.action_summon_method_embed(
            batch_dict['act_summon_method'].long()
        )
        response_vecs = self.action_response_embed(
            batch_dict['act_response'].long()
        )
        signature_bytes = batch_dict['act_signature'].long()
        signature_vecs = sum(
            embedding(signature_bytes[..., byte_index])
            for byte_index, embedding in enumerate(self.action_signature_embeds)
        )
        context_vecs = self.action_context_proj(
            batch_dict['act_context'].to(torch.float32)
        )
        location_vecs = (
            self.action_controller_embed(batch_dict['act_controller'].long())
            + self.action_location_embed(batch_dict['act_location'].long())
            + self.action_sequence_embed(batch_dict['act_sequence'].long())
            + self.position_embed(batch_dict['act_position'].long())
        )

        target_code_vecs = self.card_embed(batch_dict['act_target_code'].long())
        target_value_vecs = self.action_target_value_proj(
            batch_dict['act_target_value'].to(torch.float32) / 255.0
        )
        # 代码与数量必须保持槽位关联，避免不同指示物分配在求和后再次折叠
        target_semantic_vecs = (
            (target_code_vecs + target_value_vecs) * self.place_weights
        ).sum(dim=2)

        # 终极双塔匹配机制 (Dual-Tower Matching)
        # 1. 意图塔 (Intent)：全局底蕴决定了ai想干什么
        intent_vec = self.intent_proj(v_input) 
        intent_vec = intent_vec.expand(-1, act_mask.shape[1], -1) 
        
        # 2. 选项塔 (Option)：把目标卡片、类型、隐藏语义全部融合
        raw_option = (
            target_card_vecs
            + type_vecs
            + desc_vecs
            + act_race_vecs
            + act_attr_vecs
            + act_code_vecs
            + action_effect_vecs
            + place_vecs
            + operation_vecs
            + summon_method_vecs
            + response_vecs
            + signature_vecs
            + context_vecs
            + location_vecs
            + target_semantic_vecs
        )
        option_vec = self.option_proj(raw_option)
        
        # 3. 交汇：意图与选项碰撞
        combined_vecs = torch.cat([intent_vec, option_vec], dim=-1)

        combined_vecs = self.fusion_norm(combined_vecs) # 双塔融合后的 LayerNorm

        logits = self.policy_head(combined_vecs).squeeze(-1) 
        logits = logits.masked_fill(~act_mask, -65000.0)

        return logits, value, v_input.squeeze(1)
    
    def update_rnd_stats(self, v_input):
        with torch.no_grad():
            self.rnd.obs_norm.update(v_input)
