"""实现基于当前状态、稳定卡组画像和全局时局的目标规划隐变量"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from auxiliary_heads import (
    AUXILIARY_FUTURE_DELTA_WIDTH,
    AUXILIARY_FUTURE_HORIZONS,
    AUXILIARY_MAX_REMAINING_EVENTS,
    AUXILIARY_TERMINAL_OUTCOME_CLASS_COUNT,
    _masked_cross_entropy,
    _masked_mean,
    _normalized_delta_targets,
    scale_auxiliary_backbone_gradient,
)
from auxiliary_targets import (
    PLAN_PAIR_ROLE_FIRST,
    PLAN_PAIR_ROLE_SECOND,
    AuxiliaryHorizon,
)


PLAN_LATENT_DIM = 128
PLAN_HIDDEN_DIM = 256
PLAN_MAX_AMPLITUDE = 0.5
PLANNING_LOSS_COEF = 0.05
PLAN_CONSISTENCY_WEIGHT = 0.1
PLANNING_COMPONENT_WEIGHTS = {
    "Future_Loss": 1.0,
    "Terminal_Outcome_Loss": 0.5,
    "Terminal_Remaining_Loss": 0.25,
    "Consistency_Loss": PLAN_CONSISTENCY_WEIGHT,
}


class GoalTargetDecoder(nn.Module):
    """从目标隐变量重建可验证的多尺度未来摘要"""

    def __init__(self, plan_dim=PLAN_LATENT_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(plan_dim),
            nn.Linear(plan_dim, plan_dim),
            nn.GELU(),
        )
        self.future_delta_head = nn.Linear(
            plan_dim,
            len(AUXILIARY_FUTURE_HORIZONS) * AUXILIARY_FUTURE_DELTA_WIDTH,
        )
        self.terminal_outcome_head = nn.Linear(
            plan_dim,
            AUXILIARY_TERMINAL_OUTCOME_CLASS_COUNT,
        )
        self.terminal_remaining_head = nn.Linear(plan_dim, 1)
        for output in (
            self.future_delta_head,
            self.terminal_outcome_head,
            self.terminal_remaining_head,
        ):
            nn.init.orthogonal_(output.weight, gain=0.01)
            nn.init.zeros_(output.bias)

    def forward(self, plan_latent):
        """输出未来资源、终局胜负和剩余事件预测"""
        hidden = self.trunk(plan_latent)
        batch_size = hidden.shape[0]
        return {
            "future_delta": self.future_delta_head(hidden).view(
                batch_size,
                len(AUXILIARY_FUTURE_HORIZONS),
                AUXILIARY_FUTURE_DELTA_WIDTH,
            ),
            "terminal_outcome_logits": self.terminal_outcome_head(hidden),
            "terminal_remaining": self.terminal_remaining_head(hidden).squeeze(-1),
        }

    def evaluate_terminal(self, plan_latent):
        """仅解码当前状态的终局预测，供可选录像诊断使用"""
        hidden = self.trunk(plan_latent)
        return {
            "terminal_outcome_logits": self.terminal_outcome_head(hidden),
            "terminal_remaining": self.terminal_remaining_head(hidden).squeeze(-1),
        }


class GoalPlanner(nn.Module):
    """每个决策点重新生成目标隐变量，并保留独立监督解码器"""

    def __init__(
        self,
        d_model,
        global_context_dim,
        plan_dim=PLAN_LATENT_DIM,
        hidden_dim=PLAN_HIDDEN_DIM,
    ):
        super().__init__()
        d_model = int(d_model)
        plan_dim = int(plan_dim)
        hidden_dim = int(hidden_dim)
        self.state_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, plan_dim),
            nn.GELU(),
        )
        self.deck_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, plan_dim),
            nn.GELU(),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_context_dim),
            nn.Linear(global_context_dim, plan_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * plan_dim),
            nn.Linear(3 * plan_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, plan_dim),
            nn.LayerNorm(plan_dim),
        )
        self.target_decoder = GoalTargetDecoder(plan_dim)

    def forward(
        self,
        state_repr,
        deck_style,
        global_context,
        backbone_scale,
    ):
        """生成规划表示，并按训练进度限制其监督梯度返回旧主干"""
        state_repr = scale_auxiliary_backbone_gradient(
            state_repr,
            backbone_scale,
        )
        deck_style = scale_auxiliary_backbone_gradient(
            deck_style,
            backbone_scale,
        )
        global_context = scale_auxiliary_backbone_gradient(
            global_context,
            backbone_scale,
        )
        return self.fusion(torch.cat(
            [
                self.state_proj(state_repr),
                self.deck_proj(deck_style),
                self.global_proj(global_context),
            ],
            dim=-1,
        ))


def build_pair_preserving_indices(pair_roles):
    """按双样本单元打乱 PPO 样本，避免一致性配对跨越偶数小批次"""
    roles = pair_roles.to(device="cpu", dtype=torch.uint8).reshape(-1)
    count = int(roles.numel())
    paired = torch.zeros(count, dtype=torch.bool)
    if count > 1:
        starts = torch.nonzero(
            (roles[:-1] == PLAN_PAIR_ROLE_FIRST)
            & (roles[1:] == PLAN_PAIR_ROLE_SECOND),
            as_tuple=False,
        ).flatten()
        if starts.numel() > 0:
            # 目标构建器生成不重叠配对；此校验防止协议损坏造成重复采样
            if bool((starts[1:] <= starts[:-1] + 1).any()):
                raise ValueError("planning consistency pairs must not overlap")
            paired[starts] = True
            paired[starts + 1] = True
            pair_indices = torch.stack(
                [starts, starts + 1],
                dim=1,
            )
        else:
            pair_indices = torch.empty((0, 2), dtype=torch.long)
    else:
        pair_indices = torch.empty((0, 2), dtype=torch.long)
    singles = torch.nonzero(~paired, as_tuple=False).flatten()
    if singles.numel() > 0:
        singles = singles[torch.randperm(singles.numel())]
    paired_single_count = (singles.numel() // 2) * 2
    single_pairs = singles[:paired_single_count].reshape(-1, 2)
    units = torch.cat([pair_indices, single_pairs], dim=0)
    if units.numel() > 0:
        units = units[torch.randperm(units.shape[0])]
    result = units.reshape(-1)
    if paired_single_count < singles.numel():
        result = torch.cat([result, singles[-1:]])
    return result


def compute_goal_planning_loss(plan_latent, predictions, targets):
    """计算未来摘要监督和同回合同阶段相邻计划的一致性约束"""
    valid = targets["valid"].bool()
    future_indices = torch.tensor(
        [int(item) for item in AUXILIARY_FUTURE_HORIZONS],
        dtype=torch.long,
        device=valid.device,
    )
    future_valid = valid.index_select(1, future_indices)
    _, future_targets = _normalized_delta_targets(targets)
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

    pair_roles = targets["plan_pair_role"].reshape(-1)
    pair_mask = (
        (pair_roles[:-1] == PLAN_PAIR_ROLE_FIRST)
        & (pair_roles[1:] == PLAN_PAIR_ROLE_SECOND)
    )
    normalized_plan = F.normalize(plan_latent, dim=-1)
    pair_distance = 1.0 - (
        normalized_plan[:-1] * normalized_plan[1:]
    ).sum(dim=-1)
    consistency_loss = _masked_mean(pair_distance, pair_mask)

    total_loss = (
        PLANNING_COMPONENT_WEIGHTS["Future_Loss"] * future_loss
        + PLANNING_COMPONENT_WEIGHTS["Terminal_Outcome_Loss"] * terminal_loss
        + PLANNING_COMPONENT_WEIGHTS["Terminal_Remaining_Loss"] * remaining_loss
        + PLANNING_COMPONENT_WEIGHTS["Consistency_Loss"] * consistency_loss
    ) / sum(PLANNING_COMPONENT_WEIGHTS.values())

    with torch.no_grad():
        latent_std = plan_latent.float().std(dim=0, unbiased=False)
        latent_rms = plan_latent.float().square().mean().sqrt()
        active_fraction = (latent_std > 0.01).to(torch.float32).mean()
        pair_cosine = _masked_mean(1.0 - pair_distance, pair_mask)
        terminal_accuracy = _masked_mean(
            (
                predictions["terminal_outcome_logits"].argmax(dim=-1)
                == terminal_labels
            ).to(torch.float32),
            terminal_valid,
        )
        components = {
            "Future_Loss": future_loss,
            "Terminal_Outcome_Loss": terminal_loss,
            "Terminal_Remaining_Loss": remaining_loss,
            "Consistency_Loss": consistency_loss,
        }
        pair_valid_fraction = (
            pair_mask.to(torch.float32).mean()
            if pair_mask.numel() > 0
            else latent_rms.new_zeros(())
        )
        metrics = {
            "Latent_RMS": latent_rms,
            "Latent_Mean_Dimension_Std": latent_std.mean(),
            "Latent_Active_Dimension_Fraction": active_fraction,
            "Pair_Cosine": pair_cosine,
            "Pair_Valid_Fraction": pair_valid_fraction,
            "Terminal_Accuracy": terminal_accuracy,
            "Future_Valid_Fraction": future_valid.to(torch.float32).mean(),
        }
    return total_loss, components, metrics
