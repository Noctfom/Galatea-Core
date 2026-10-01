"""统一 PPO 更新默认参数、策略偏移检查和每轮更新审计"""

import math
import statistics

import torch


DEFAULT_GAE_LAMBDA = 0.98
DEFAULT_PPO_EPOCHS = 4
MAX_PPO_EPOCHS = 4
DEFAULT_TARGET_KL = 0.02
TARGET_KL_STOP_MULTIPLIER = 1.5


def policy_shift_statistics(new_log_probs, old_log_probs, clip_eps):
    """按已采样动作计算非负近似 KL 和超出裁剪区间的样本比例"""
    log_ratio = new_log_probs.detach().float() - old_log_probs.detach().float()
    ratio = torch.exp(log_ratio)
    approximate_kl = (torch.expm1(log_ratio) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > float(clip_eps)).float().mean()
    return approximate_kl, clip_fraction


def should_stop_for_kl(approximate_kl, target_kl):
    """策略偏移非有限或超过启用的目标阈值时停止本轮全部共享更新"""
    approximate_kl = float(approximate_kl)
    if not math.isfinite(approximate_kl):
        return True
    return float(target_kl) > 0 and approximate_kl > (
        TARGET_KL_STOP_MULTIPLIER * float(target_kl)
    )


def _quantile(values, probability):
    """对小规模 mini-batch 统计列表计算线性插值分位数"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class PPOUpdateAudit:
    """只保存每个小批次的标量摘要，避免审计占用 rollout 或计算图内存"""

    def __init__(self, sample_count, mini_batch_size, requested_epochs, target_kl):
        """初始化本轮更新计数和标量列表，不保留样本或计算图"""
        self.sample_count = int(sample_count)
        self.mini_batch_size = int(mini_batch_size)
        self.requested_epochs = int(requested_epochs)
        self.target_kl = float(target_kl)
        self.epochs_started = 0
        self.epochs_completed = 0
        self.optimizer_updates = 0
        self.samples_updated = 0
        self.kl_stopped = False
        self.nonfinite_batches = 0
        self.kl_values = []
        self.clip_values = []
        self.stop_kl = 0.0

    def observe(self, approximate_kl, clip_fraction):
        """登记本次前向的策略偏移并返回是否触发停止"""
        approximate_kl = float(approximate_kl)
        clip_fraction = float(clip_fraction)
        if math.isfinite(approximate_kl) and math.isfinite(clip_fraction):
            self.kl_values.append(approximate_kl)
            self.clip_values.append(clip_fraction)
        else:
            self.nonfinite_batches += 1
        if should_stop_for_kl(approximate_kl, self.target_kl):
            self.kl_stopped = True
            self.stop_kl = approximate_kl
            return True
        return False

    def record_update(self, sample_count):
        """只在优化器实际完成更新后累计样本复用次数"""
        self.optimizer_updates += 1
        self.samples_updated += int(sample_count)

    def scalars(self):
        """输出按采集轮次记录的更新次数、有效复用率和小批次分位数"""
        return {
            "Requested_Epochs": self.requested_epochs,
            "Epochs_Started": self.epochs_started,
            "Epochs_Completed": self.epochs_completed,
            "Optimizer_Updates": self.optimizer_updates,
            "Effective_Epochs": self.samples_updated / max(1, self.sample_count),
            "Samples_Updated": self.samples_updated,
            "Skipped_Tail_Samples_Per_Epoch": (
                self.sample_count % self.mini_batch_size
            ),
            "KL_Early_Stop": float(self.kl_stopped),
            "Target_KL": self.target_kl,
            "KL_Stop_Threshold": self.target_kl * TARGET_KL_STOP_MULTIPLIER,
            "Approx_KL_Mean": statistics.fmean(self.kl_values) if self.kl_values else 0.0,
            "Approx_KL_P50": _quantile(self.kl_values, 0.5),
            "Approx_KL_P95": _quantile(self.kl_values, 0.95),
            "Approx_KL_Max": max(self.kl_values, default=0.0),
            "Clip_Fraction_P50": _quantile(self.clip_values, 0.5),
            "Clip_Fraction_P95": _quantile(self.clip_values, 0.95),
            "Nonfinite_Batches": self.nonfinite_batches,
        }
