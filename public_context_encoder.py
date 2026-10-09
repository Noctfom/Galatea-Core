# 本文件将公开提示、已知卡组位置和有序通知轻量融合到卡片及全局表示，不扩展主干序列

import torch
import torch.nn as nn
import torch.nn.functional as F

from event_history import MAX_PUBLIC_NOTIFICATIONS
from public_observation import PUBLIC_CONTEXT_DIM, DECK_POSITION_CAPACITY


class PublicContextEncoder(nn.Module):
    """复用卡片/代码向量，以小维度分支编码动态公开认知"""

    def __init__(self, d_model):
        """建立有界残差分支，零门控使初始化不会强行改变原策略"""
        super().__init__()
        self.width = min(PUBLIC_CONTEXT_DIM, int(d_model))
        width = self.width
        self.kind = nn.Embedding(9, width, padding_idx=0)
        self.role = nn.Embedding(3, width, padding_idx=0)
        self.binding = nn.Embedding(3, width, padding_idx=0)
        self.zone = nn.Embedding(9, width, padding_idx=0)
        self.sequence = nn.Embedding(256, width, padding_idx=0)
        self.position = nn.Embedding(256, width, padding_idx=0)
        self.value = nn.ModuleList([nn.Embedding(256, width) for _ in range(4)])
        self.numeric = nn.Linear(3, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.score = nn.Linear(width, 1, bias=False)
        self.host_proj = nn.Linear(width, d_model, bias=False)
        self.context_proj = nn.Linear(4 * width + 6, d_model, bias=False)
        self.host_gate = nn.Parameter(torch.zeros(d_model))
        self.context_gate = nn.Parameter(torch.zeros(d_model))
        self.deck_position_gate = nn.Embedding(DECK_POSITION_CAPACITY, width)
        self.deck_faceup = nn.Embedding(2, width)
        self.notification_kind = nn.Embedding(256, width, padding_idx=0)
        self.notification_order = nn.Embedding(MAX_PUBLIC_NOTIFICATIONS, width)
        self.notification_numeric = nn.Linear(3, width, bias=False)
        self.notification_proj = nn.Linear(width, d_model, bias=False)
        self.notification_gate = nn.Parameter(torch.zeros(d_model))

    def _value_vector(self, values):
        """按字节位置编码完整32位身份，不使用取模合并提示编号"""
        return sum(embedding(values[..., index].long()) for index, embedding in enumerate(self.value))

    def _location(self, locations):
        """保持玩家、区域和表示类别独立，位置使用有界离散编码"""
        locations = locations.long()
        return (self.role(locations[..., 0]) + self.zone(locations[..., 1])
                + self.sequence(locations[..., 2]) + self.position(locations[..., 3]))

    def _pool(self, tokens, mask):
        """安全聚合集合；空提示输出严格为零，避免全掩码NaN"""
        scores = self.score(tokens).squeeze(-1).float().masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return (tokens * weights.unsqueeze(-1).to(tokens.dtype)).sum(dim=-2)

    def forward(self, batch, declared_cards, source_codes, deck_cards):
        """返回原120张卡的提示残差与供策略/价值共享的动态上下文"""
        mask = batch['hint_mask'].bool()
        values = batch['hint_value_bytes'].float()
        number = sum(values[..., byte] * float(256 ** byte) for byte in range(4))
        # 卡密/说明是身份而非大小；仅数量类提示使用量值分支
        number = number * (batch['hint_kind'].le(5) & batch['hint_kind'].ne(2))
        numeric = torch.stack((
            torch.log1p(number) / 32.0,
            torch.log1p(batch['hint_count'].float()) / 8.0,
            torch.log1p(batch['hint_age'].float()) / 8.0,
        ), dim=-1)
        tokens = F.gelu(self.norm(
            declared_cards + source_codes + self.kind(batch['hint_kind'].long())
            + self.role(batch['hint_role'].long()) + self.binding(batch['hint_binding'].long())
            + self._location(batch['hint_location']) + self._value_vector(batch['hint_value_bytes'])
            + self.numeric(numeric)
        )) * mask.unsqueeze(-1)
        host_count = 120
        hosts = tokens.new_zeros((tokens.shape[0], host_count + 1, self.width))
        host_indices = batch['hint_host'].long().unsqueeze(-1).expand(-1, -1, self.width)
        hosts = hosts.scatter_add(1, host_indices, tokens)
        host_context = self.host_proj(hosts[:, :host_count]) * torch.tanh(self.host_gate)
        hint_context = [self._pool(tokens, mask & batch['hint_role'].eq(role)) for role in (1, 2)]
        flags = batch['known_deck_flags']
        deck_mask = flags.ne(0)
        positions = torch.arange(DECK_POSITION_CAPACITY, device=deck_cards.device)
        # 位置与身份在聚合前相乘，防止普通均值再次消除排序关联
        deck_tokens = F.gelu(deck_cards * torch.sigmoid(self.deck_position_gate(positions))
                             + self.deck_faceup((flags.long() >> 1) & 1))
        deck_context = self._pool(deck_tokens, deck_mask)
        combined = torch.cat((*hint_context, deck_context[:, 0], deck_context[:, 1], batch['public_state_flags'].float()), dim=-1)
        context = self.context_proj(combined) * torch.tanh(self.context_gate)
        return host_context, context

    def encode_notifications(self, batch, cards):
        """编码消息内部顺序和处理链序，保持它们只是上下文而非因果标签"""
        mask = batch['event_public_mask'].bool()
        order = torch.arange(MAX_PUBLIC_NOTIFICATIONS, device=cards.device)
        metadata = torch.stack((
            batch['event_public_chain'].float() / 12.0,
            torch.log1p(batch['event_public_offset'].float()) / 8.0,
            batch['event_public_subtype'].float() / 7.0,
        ), dim=-1)
        tokens = F.gelu(
            cards + self.notification_kind(batch['event_public_kind'].long())
            + self._location(batch['event_public_location'])
            + self.notification_order(order)
            + self._value_vector(batch['event_public_value'])
            + self.notification_numeric(metadata)
        )
        summary = self._pool(tokens, mask)
        return self.notification_proj(summary) * torch.tanh(self.notification_gate)
