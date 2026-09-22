"""实现训练专用结构化辅助预测头、目标归一化与掩码损失"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from auxiliary_targets import AUXILIARY_ZONE_DIM, AuxiliaryHorizon


AUXILIARY_RESULT_BIT_COUNT = 10
AUXILIARY_MESSAGE_GROUP_COUNT = 10
AUXILIARY_BOUNDARY_CLASS_COUNT = 3
AUXILIARY_TERMINAL_OUTCOME_CLASS_COUNT = 3
AUXILIARY_HEAD_HIDDEN_DIM = 256
AUXILIARY_ZONE_COUNT = AUXILIARY_ZONE_DIM // 2
AUXILIARY_IMMEDIATE_DELTA_WIDTH = 2 + AUXILIARY_ZONE_DIM + 2
AUXILIARY_FUTURE_DELTA_WIDTH = 2 + AUXILIARY_ZONE_DIM
AUXILIARY_FUTURE_HORIZONS = (
    AuxiliaryHorizon.CHAIN_END,
    AuxiliaryHorizon.TURN_END,
    AuxiliaryHorizon.TERMINAL,
)
AUXILIARY_LP_SCALE = 8000.0
AUXILIARY_ZONE_SCALE = 5.0
AUXILIARY_CHAIN_SCALE = 16.0
AUXILIARY_MAX_REMAINING_EVENTS = 1500
AUXILIARY_LOSS_COEF = 0.1
AUXILIARY_BACKBONE_SCALE_MAX = 0.2
AUXILIARY_PROBE_STEPS = 100
AUXILIARY_RAMP_STEPS = 900
AUXILIARY_COMPONENT_WEIGHTS = {
    "Result_Loss": 1.0,
    "Message_Loss": 1.0,
    "Boundary_Loss": 0.5,
    "Immediate_Delta_Loss": 1.0,
    "Future_Delta_Loss": 1.0,
    "Terminal_Outcome_Loss": 0.5,
    "Terminal_Remaining_Loss": 0.25,
}


def scale_auxiliary_backbone_gradient(value, scale):
    """保持前向值不变，仅缩放辅助任务返回共享主干的梯度"""
    scale = float(scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"auxiliary backbone scale must be in [0, 1], got {scale}")
    if scale == 0.0:
        return value.detach()
    if scale == 1.0:
        return value
    return value.detach() + scale * (value - value.detach())


class StructuredAuxiliaryHeads(nn.Module):
    """从共享状态和本次已选动作表示预测可验证的真实后果"""

    def __init__(self, d_model, hidden_dim=AUXILIARY_HEAD_HIDDEN_DIM):
        super().__init__()
        context_dim = 2 * int(d_model)
        hidden_dim = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.result_head = nn.Linear(hidden_dim, AUXILIARY_RESULT_BIT_COUNT)
        self.message_head = nn.Linear(
            hidden_dim,
            AUXILIARY_MESSAGE_GROUP_COUNT,
        )
        self.boundary_head = nn.Linear(
            hidden_dim,
            AUXILIARY_BOUNDARY_CLASS_COUNT,
        )
        self.immediate_delta_head = nn.Linear(
            hidden_dim,
            AUXILIARY_IMMEDIATE_DELTA_WIDTH,
        )
        self.future_delta_head = nn.Linear(
            hidden_dim,
            len(AUXILIARY_FUTURE_HORIZONS) * AUXILIARY_FUTURE_DELTA_WIDTH,
        )
        self.terminal_outcome_head = nn.Linear(
            hidden_dim,
            AUXILIARY_TERMINAL_OUTCOME_CLASS_COUNT,
        )
        self.terminal_remaining_head = nn.Linear(hidden_dim, 1)

        for output in (
            self.result_head,
            self.message_head,
            self.boundary_head,
            self.immediate_delta_head,
            self.future_delta_head,
            self.terminal_outcome_head,
            self.terminal_remaining_head,
        ):
            nn.init.orthogonal_(output.weight, gain=0.01)
            nn.init.zeros_(output.bias)

    def forward(self, state_repr, selected_option_repr, backbone_scale):
        """生成训练预测；不会向策略或价值头回灌预测值"""
        if state_repr.shape != selected_option_repr.shape:
            raise ValueError("state and selected option representations must align")
        state_repr = scale_auxiliary_backbone_gradient(
            state_repr,
            backbone_scale,
        )
        selected_option_repr = scale_auxiliary_backbone_gradient(
            selected_option_repr,
            backbone_scale,
        )
        hidden = self.trunk(torch.cat([state_repr, selected_option_repr], dim=-1))
        batch_size = hidden.shape[0]
        return {
            "result_logits": self.result_head(hidden),
            "message_logits": self.message_head(hidden),
            "boundary_logits": self.boundary_head(hidden),
            "immediate_delta": self.immediate_delta_head(hidden),
            "future_delta": self.future_delta_head(hidden).view(
                batch_size,
                len(AUXILIARY_FUTURE_HORIZONS),
                AUXILIARY_FUTURE_DELTA_WIDTH,
            ),
            "terminal_outcome_logits": self.terminal_outcome_head(hidden),
            "terminal_remaining": self.terminal_remaining_head(hidden).squeeze(-1),
        }


def _relative_lp(raw_lp, actor):
    """把 P0/P1 LP 变化重排为当前执行者/对手顺序"""
    actor = actor.long()
    gather_index = torch.stack([actor, 1 - actor], dim=-1)
    while gather_index.ndim < raw_lp.ndim:
        gather_index = gather_index.unsqueeze(1)
    gather_index = gather_index.expand(*raw_lp.shape[:-1], 2)
    return torch.gather(raw_lp, -1, gather_index)


def _relative_zones(raw_zones, actor):
    """把 P0/P1 八区域变化重排为当前执行者/对手顺序"""
    reshaped = raw_zones.reshape(
        *raw_zones.shape[:-1],
        2,
        AUXILIARY_ZONE_COUNT,
    )
    actor = actor.long()
    prefix_shape = reshaped.shape[:-2]
    gather_index = actor.view(
        actor.shape[0],
        *([1] * (len(prefix_shape) - 1)),
        1,
        1,
    )
    self_index = gather_index.expand(*prefix_shape, 1, AUXILIARY_ZONE_COUNT)
    opponent_index = (1 - gather_index).expand(
        *prefix_shape,
        1,
        AUXILIARY_ZONE_COUNT,
    )
    own = torch.gather(reshaped, -2, self_index).squeeze(-2)
    opponent = torch.gather(reshaped, -2, opponent_index).squeeze(-2)
    return torch.cat([own, opponent], dim=-1)


def _masked_mean(values, mask):
    """仅对具备可靠标签的元素求均值，空掩码返回连图零值"""
    mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    weights = expanded.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_cross_entropy(logits, labels, mask):
    """计算带缺失掩码的分类损失"""
    per_item = F.cross_entropy(logits, labels, reduction="none")
    return _masked_mean(per_item, mask)


def _bit_targets(values, bit_count):
    """把紧凑整数位图展开成独立二分类标签"""
    masks = 1 << torch.arange(bit_count, device=values.device, dtype=torch.int32)
    return ((values.to(torch.int32).unsqueeze(-1) & masks) != 0).to(torch.float32)


def _normalized_delta_targets(targets):
    """生成玩家相对、尺度稳定的即时与多时间尺度资源目标"""
    actor = targets["actor"].long()
    lp = _relative_lp(targets["lp_delta"].to(torch.float32), actor)
    zones = _relative_zones(targets["zone_delta"].to(torch.float32), actor)
    lp = torch.clamp(lp / AUXILIARY_LP_SCALE, -4.0, 4.0)
    zones = torch.clamp(zones / AUXILIARY_ZONE_SCALE, -4.0, 4.0)

    immediate_index = int(AuxiliaryHorizon.NEXT_DECISION)
    chain_after_and_max = torch.clamp(
        targets["chain"][:, 1:].to(torch.float32) / AUXILIARY_CHAIN_SCALE,
        0.0,
        4.0,
    )
    immediate = torch.cat(
        [
            lp[:, immediate_index],
            zones[:, immediate_index],
            chain_after_and_max,
        ],
        dim=-1,
    )
    future_indices = torch.tensor(
        [int(item) for item in AUXILIARY_FUTURE_HORIZONS],
        dtype=torch.long,
        device=lp.device,
    )
    future = torch.cat(
        [
            lp.index_select(1, future_indices),
            zones.index_select(1, future_indices),
        ],
        dim=-1,
    )
    return immediate, future


def compute_structured_auxiliary_loss(predictions, targets):
    """计算七组归一化辅助任务，并返回可审计的分项指标"""
    valid = targets["valid"].bool()
    next_valid = valid[:, int(AuxiliaryHorizon.NEXT_DECISION)]
    result_targets = _bit_targets(
        targets["result_flags"],
        AUXILIARY_RESULT_BIT_COUNT,
    )
    message_targets = _bit_targets(
        targets["message_groups"],
        AUXILIARY_MESSAGE_GROUP_COUNT,
    )
    result_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            predictions["result_logits"],
            result_targets,
            reduction="none",
        ),
        next_valid,
    )
    message_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            predictions["message_logits"],
            message_targets,
            reduction="none",
        ),
        next_valid,
    )
    boundary_labels = (
        targets["boundary"].long() - int(1)
    ).clamp(0, AUXILIARY_BOUNDARY_CLASS_COUNT - 1)
    boundary_loss = _masked_cross_entropy(
        predictions["boundary_logits"],
        boundary_labels,
        next_valid,
    )

    immediate_targets, future_targets = _normalized_delta_targets(targets)
    immediate_loss = _masked_mean(
        F.smooth_l1_loss(
            predictions["immediate_delta"],
            immediate_targets,
            reduction="none",
        ),
        next_valid,
    )
    future_valid = valid.index_select(
        1,
        torch.tensor(
            [int(item) for item in AUXILIARY_FUTURE_HORIZONS],
            dtype=torch.long,
            device=valid.device,
        ),
    )
    future_loss = _masked_mean(
        F.smooth_l1_loss(
            predictions["future_delta"],
            future_targets,
            reduction="none",
        ),
        future_valid,
    )

    terminal_valid = valid[:, int(AuxiliaryHorizon.TERMINAL)]
    terminal_labels = (
        targets["terminal_outcome"].long() + 1
    ).clamp(0, AUXILIARY_TERMINAL_OUTCOME_CLASS_COUNT - 1)
    terminal_loss = _masked_cross_entropy(
        predictions["terminal_outcome_logits"],
        terminal_labels,
        terminal_valid,
    )
    remaining_target = torch.log1p(
        targets["terminal_remaining_events"].to(torch.float32).clamp(min=0)
    ) / math.log1p(AUXILIARY_MAX_REMAINING_EVENTS)
    remaining_loss = _masked_mean(
        F.smooth_l1_loss(
            predictions["terminal_remaining"],
            remaining_target,
            reduction="none",
        ),
        terminal_valid,
    )

    component_losses = {
        "Result_Loss": result_loss,
        "Message_Loss": message_loss,
        "Boundary_Loss": boundary_loss,
        "Immediate_Delta_Loss": immediate_loss,
        "Future_Delta_Loss": future_loss,
        "Terminal_Outcome_Loss": terminal_loss,
        "Terminal_Remaining_Loss": remaining_loss,
    }
    total_weight = sum(AUXILIARY_COMPONENT_WEIGHTS.values())
    total_loss = sum(
        component_losses[name] * weight
        for name, weight in AUXILIARY_COMPONENT_WEIGHTS.items()
    ) / total_weight

    with torch.no_grad():
        result_accuracy = _masked_mean(
            (
                (predictions["result_logits"] >= 0)
                == result_targets.bool()
            ).to(torch.float32),
            next_valid,
        )
        message_accuracy = _masked_mean(
            (
                (predictions["message_logits"] >= 0)
                == message_targets.bool()
            ).to(torch.float32),
            next_valid,
        )
        boundary_accuracy = _masked_mean(
            (
                predictions["boundary_logits"].argmax(dim=-1)
                == boundary_labels
            ).to(torch.float32),
            next_valid,
        )
        terminal_accuracy = _masked_mean(
            (
                predictions["terminal_outcome_logits"].argmax(dim=-1)
                == terminal_labels
            ).to(torch.float32),
            terminal_valid,
        )
        metrics = {
            "Result_Bit_Accuracy": result_accuracy,
            "Message_Bit_Accuracy": message_accuracy,
            "Boundary_Accuracy": boundary_accuracy,
            "Terminal_Accuracy": terminal_accuracy,
            "Future_Valid_Fraction": future_valid.to(torch.float32).mean(),
        }
    return total_loss, component_losses, metrics
