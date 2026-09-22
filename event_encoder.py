"""将最近决策间状态转移编码为独立、顺序敏感的对局历史表示"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_types import (
    ACTION_OPERATION_COUNT,
    PHASE_CATEGORY_COUNT,
    PLAYER_ROLE_COUNT,
    POSITION_CATEGORY_COUNT,
    SUMMON_METHOD_COUNT,
    TRANSITION_BOUNDARY_COUNT,
    TRANSITION_EVENT_CHAIN_DIM,
    TRANSITION_EVENT_COUNT_DIM,
    TRANSITION_EVENT_HISTORY_SIZE,
    TRANSITION_EVENT_LOCATION_DIM,
    TRANSITION_EVENT_LP_DIM,
    TRANSITION_EVENT_MESSAGE_BYTES,
    TRANSITION_EVENT_RESULT_BYTES,
    TRANSITION_EVENT_TARGET_SLOTS,
    TRANSITION_ZONE_COUNT,
    ZONE_CATEGORY_COUNT,
)


EVENT_NUMERIC_DIM = (
    TRANSITION_EVENT_LP_DIM
    + 2 * TRANSITION_ZONE_COUNT
    + TRANSITION_EVENT_CHAIN_DIM
    + TRANSITION_EVENT_COUNT_DIM
)


class EventHistoryEncoder(nn.Module):
    """使用事件内目标聚合与事件间顺序混合生成固定宽度历史向量"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = int(d_model)
        self.prompt_embed = nn.Embedding(256, d_model, padding_idx=0)
        self.operation_embed = nn.Embedding(
            ACTION_OPERATION_COUNT, d_model, padding_idx=0
        )
        self.summon_method_embed = nn.Embedding(
            SUMMON_METHOD_COUNT, d_model, padding_idx=0
        )
        self.actor_role_embed = nn.Embedding(
            PLAYER_ROLE_COUNT, d_model, padding_idx=0
        )
        self.phase_embed = nn.Embedding(
            PHASE_CATEGORY_COUNT, d_model, padding_idx=0
        )
        self.boundary_embed = nn.Embedding(
            TRANSITION_BOUNDARY_COUNT, d_model, padding_idx=0
        )
        self.location_role_embed = nn.Embedding(
            PLAYER_ROLE_COUNT, d_model, padding_idx=0
        )
        self.zone_embed = nn.Embedding(
            ZONE_CATEGORY_COUNT, d_model, padding_idx=0
        )
        self.sequence_embed = nn.Embedding(33, d_model, padding_idx=0)
        self.position_embed = nn.Embedding(
            POSITION_CATEGORY_COUNT, d_model, padding_idx=0
        )
        self.result_proj = nn.Linear(
            TRANSITION_EVENT_RESULT_BYTES * 8,
            d_model,
            bias=False,
        )
        self.message_proj = nn.Linear(
            TRANSITION_EVENT_MESSAGE_BYTES * 8,
            d_model,
            bias=False,
        )
        self.numeric_proj = nn.Linear(EVENT_NUMERIC_DIM, d_model, bias=False)
        self.target_slot_embed = nn.Parameter(
            torch.randn(
                1,
                1,
                TRANSITION_EVENT_TARGET_SLOTS,
                d_model,
            ) * 0.02
        )
        self.event_position_embed = nn.Parameter(
            torch.randn(1, TRANSITION_EVENT_HISTORY_SIZE, d_model) * 0.02
        )
        self.event_input_norm = nn.LayerNorm(d_model)
        self.local_mixer = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
            bias=False,
        )
        self.channel_mixer = nn.Linear(d_model, d_model, bias=False)
        self.event_output_norm = nn.LayerNorm(d_model)
        score_hidden = max(8, d_model // 4)
        self.attention_score = nn.Sequential(
            nn.Linear(d_model, score_hidden),
            nn.Tanh(),
            nn.Linear(score_hidden, 1, bias=False),
        )
        self.register_buffer(
            "bit_masks",
            torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.long),
            persistent=False,
        )

    def _unpack_bit_bytes(self, packed):
        """把紧凑事件字节展开为确定的逐位浮点特征"""
        return torch.bitwise_and(
            packed.long().unsqueeze(-1),
            self.bit_masks,
        ).ne(0).flatten(start_dim=-2).to(torch.float32)

    def _encode_locations(self, context):
        """编码 `[角色, 区域, 序号, 表示]` 四元组"""
        if context.shape[-1] != TRANSITION_EVENT_LOCATION_DIM:
            raise ValueError("transition location context width is invalid")
        context = context.long()
        return (
            self.location_role_embed(context[..., 0])
            + self.zone_embed(context[..., 1])
            + self.sequence_embed(context[..., 2])
            + self.position_embed(context[..., 3])
        )

    def forward(self, batch_dict, source_semantic, target_semantic):
        """融合事件语义、公开变化与固定时间顺序"""
        mask = batch_dict["event_mask"].bool()
        mask_values = mask.unsqueeze(-1).to(source_semantic.dtype)

        source_context = self._encode_locations(
            batch_dict["event_source_context"]
        )
        target_context = self._encode_locations(
            batch_dict["event_target_context"]
        )
        target_mask = batch_dict["event_target_mask"].bool().unsqueeze(-1)
        target_tokens = (
            target_semantic + target_context + self.target_slot_embed
        ) * target_mask.to(target_semantic.dtype)
        target_summary = target_tokens.sum(dim=2) / target_mask.sum(
            dim=2
        ).clamp(min=1).to(target_tokens.dtype)

        numeric = torch.cat(
            [
                batch_dict["event_lp_delta"].to(torch.float32),
                batch_dict["event_zone_delta"].to(torch.float32) / 10.0,
                batch_dict["event_chain"].to(torch.float32) / 12.0,
                torch.stack(
                    (
                        batch_dict["event_counts"][..., 0].to(torch.float32)
                        / float(TRANSITION_EVENT_TARGET_SLOTS),
                        batch_dict["event_counts"][..., 1].to(torch.float32)
                        / 32.0,
                    ),
                    dim=-1,
                ),
            ],
            dim=-1,
        )
        if numeric.shape[-1] != EVENT_NUMERIC_DIM:
            raise ValueError("transition numeric feature width is invalid")

        tokens = (
            source_semantic
            + target_summary
            + self.prompt_embed(batch_dict["event_prompt"].long())
            + self.operation_embed(batch_dict["event_operation"].long())
            + self.summon_method_embed(
                batch_dict["event_summon_method"].long()
            )
            + self.actor_role_embed(batch_dict["event_actor_role"].long())
            + self.phase_embed(batch_dict["event_phase"].long())
            + self.boundary_embed(batch_dict["event_boundary"].long())
            + source_context
            + self.result_proj(
                self._unpack_bit_bytes(batch_dict["event_result_bytes"])
            )
            + self.message_proj(
                self._unpack_bit_bytes(batch_dict["event_message_bytes"])
            )
            + self.numeric_proj(numeric)
            + self.event_position_embed
        ) * mask_values

        local_context = self.local_mixer(
            self.event_input_norm(tokens).transpose(1, 2)
        ).transpose(1, 2)
        mixed = self.event_output_norm(
            tokens + self.channel_mixer(F.gelu(local_context))
        ) * mask_values
        scores = self.attention_score(mixed).squeeze(-1).float()
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-6)
        return (mixed * weights.unsqueeze(-1).to(mixed.dtype)).sum(dim=1)
