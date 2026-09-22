# 本文件实现可复用、对卡片排列不敏感的卡组关系编码器
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DECK_LATENT_COUNT = 8
DECK_SECTION_COUNT = 4
DECK_COUNT_BINS = 256


class DeckFeedForward(nn.Module):
    """为卡组 latent 提供紧凑的门控前馈变换"""

    def __init__(self, d_model: int) -> None:
        """初始化卡组 latent 使用的轻量 SwiGLU 前馈层"""
        super().__init__()
        hidden_dim = max(64, 2 * int(d_model))
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.value = nn.Linear(d_model, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """使用 SwiGLU 组合卡组关系特征"""
        return self.output(F.silu(self.gate(value)) * self.value(value))


class DeckEncoder(nn.Module):
    """用固定 latent 查询汇聚无序卡组条目，并保留逐卡等变表示"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        latent_count: int = DECK_LATENT_COUNT,
    ) -> None:
        """初始化固定 latent、数量标签和双向交叉注意力"""
        super().__init__()
        if int(d_model) <= 0 or int(n_heads) <= 0:
            raise ValueError("deck encoder dimensions must be positive")
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("deck encoder d_model must be divisible by n_heads")
        if int(latent_count) <= 0:
            raise ValueError("deck encoder latent_count must be positive")

        self.d_model = int(d_model)
        self.latent_count = int(latent_count)
        self.hidden_dim = min(
            self.d_model,
            max(int(n_heads), (256 // int(n_heads)) * int(n_heads)),
        )
        self.input_proj = (
            nn.Linear(self.d_model, self.hidden_dim, bias=False)
            if self.hidden_dim != self.d_model
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(self.hidden_dim, self.d_model, bias=False)
            if self.hidden_dim != self.d_model
            else nn.Identity()
        )
        self.section_embed = nn.Embedding(
            DECK_SECTION_COUNT,
            self.hidden_dim,
            padding_idx=0,
        )
        self.initial_count_embed = nn.Embedding(
            DECK_COUNT_BINS,
            self.hidden_dim,
            padding_idx=0,
        )
        # 剩余数量为 0 是有效状态，不能把零号向量固定成 padding
        self.remaining_count_embed = nn.Embedding(
            DECK_COUNT_BINS,
            self.hidden_dim,
        )
        self.latent_queries = nn.Parameter(
            torch.randn(1, self.latent_count, self.hidden_dim) * 0.02
        )

        self.card_input_norm = nn.LayerNorm(self.hidden_dim)
        self.latent_cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.latent_cross_norm = nn.LayerNorm(self.hidden_dim)
        self.latent_self_attention = nn.MultiheadAttention(
            self.hidden_dim,
            n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.latent_self_norm = nn.LayerNorm(self.hidden_dim)
        self.latent_ffn = DeckFeedForward(self.hidden_dim)
        self.latent_output_norm = nn.LayerNorm(self.hidden_dim)

        self.card_cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.card_output_norm = nn.LayerNorm(self.hidden_dim)
        self.style_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        card_features: torch.Tensor,
        section: torch.Tensor,
        initial_count: torch.Tensor,
        remaining_count: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        return_per_card: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """返回卡组风格、固定 latent 和与输入顺序等变的逐卡表示"""
        if card_features.ndim != 3 or card_features.shape[-1] != self.d_model:
            raise ValueError("deck card_features must have shape [B, N, d_model]")
        expected_shape = card_features.shape[:2]
        for name, value in (
            ("section", section),
            ("initial_count", initial_count),
            ("remaining_count", remaining_count),
            ("valid_mask", valid_mask),
        ):
            if tuple(value.shape) != tuple(expected_shape):
                raise ValueError(
                    f"deck {name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected_shape)}"
                )

        mask = valid_mask.bool()
        empty_profiles = ~mask.any(dim=1)
        if mask.shape[1] <= 0:
            raise ValueError("deck encoder requires at least one padded entry")
        fallback_entry = torch.arange(
            mask.shape[1],
            device=mask.device,
        ).eq(0).unsqueeze(0)
        safe_mask = mask | (
            empty_profiles.unsqueeze(1) & fallback_entry
        )

        card_tokens = self.card_input_norm(
            self.input_proj(card_features)
            + self.section_embed(section.long().clamp(0, DECK_SECTION_COUNT - 1))
            + self.initial_count_embed(
                initial_count.long().clamp(0, DECK_COUNT_BINS - 1)
            )
            + self.remaining_count_embed(
                remaining_count.long().clamp(0, DECK_COUNT_BINS - 1)
            )
        )
        card_tokens = card_tokens * mask.unsqueeze(-1).to(card_tokens.dtype)

        latent_seed = self.latent_queries.expand(card_tokens.shape[0], -1, -1)
        cross_context, _ = self.latent_cross_attention(
            latent_seed,
            card_tokens,
            card_tokens,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        latents = self.latent_cross_norm(latent_seed + cross_context)
        self_context, _ = self.latent_self_attention(
            latents,
            latents,
            latents,
            need_weights=False,
        )
        latents = self.latent_self_norm(latents + self_context)
        latents = self.latent_output_norm(latents + self.latent_ffn(latents))

        per_card_features = None
        if return_per_card:
            # 当前策略只消费风格；组卡上层显式请求时才支付逐卡回写开销
            card_context, _ = self.card_cross_attention(
                card_tokens,
                latents,
                latents,
                need_weights=False,
            )
            per_card_features = self.output_proj(
                self.card_output_norm(card_tokens + card_context)
            )
            per_card_features = (
                per_card_features * mask.unsqueeze(-1).to(per_card_features.dtype)
            )
        latent_outputs = self.output_proj(latents)
        deck_style = self.style_norm(latent_outputs.mean(dim=1))

        empty_values = empty_profiles.unsqueeze(-1)
        deck_style = deck_style.masked_fill(empty_values, 0.0)
        latent_outputs = latent_outputs.masked_fill(
            empty_values.unsqueeze(-1),
            0.0,
        )
        return deck_style, latent_outputs, per_card_features
