# 本文件在训练资源分配前统一校验模型结构和 PPO 参数

import math
import re
from numbers import Integral, Real

from protocol_schema import apply_current_protocol_metadata
from ppo_control import DEFAULT_PPO_EPOCHS, DEFAULT_TARGET_KL, MAX_PPO_EPOCHS

MODEL_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DEFAULT_LEAGUE_WEIGHTS = {'rule': 0.15, 'windbot': 0.0, 'hist': 0.25, 'self': 0.6}


def resolve_league_weights(weights=None, *, windbot_enabled=False, legacy_windbot_rate=None):
    """校验四类对手的绝对概率，默认由WindBot接替RuleBot基础份额"""
    if weights is not None and legacy_windbot_rate is not None:
        raise ValueError('联盟配比不能与旧版windbot-rate同时指定')
    if weights is None:
        weights = dict(DEFAULT_LEAGUE_WEIGHTS)
        if legacy_windbot_rate is not None:
            _require_finite_number('legacy_windbot_rate', legacy_windbot_rate)
            if not windbot_enabled or not 0 < legacy_windbot_rate <= 1:
                raise ValueError('旧版windbot-rate需要启用WindBot且位于(0,1]')
            weights = {key: value * (1 - legacy_windbot_rate) for key, value in weights.items()}
            weights['windbot'] = float(legacy_windbot_rate)
        elif windbot_enabled:
            weights['windbot'], weights['rule'] = weights['rule'], 0.0
    if not isinstance(weights, dict) or set(weights) != set(DEFAULT_LEAGUE_WEIGHTS):
        raise ValueError('联盟配比必须包含rule、windbot、hist、self四项')
    for name, value in weights.items():
        _require_finite_number(f'league.{name}', value)
        if not 0 <= value <= 1:
            raise ValueError(f'league.{name}必须位于[0,1]')
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError('四类联盟配比之和必须等于1，不进行隐式归一化')
    if weights['windbot'] and not windbot_enabled:
        raise ValueError('WindBot配比大于0时必须显式启用WindBot')
    return {key: float(weights[key]) for key in DEFAULT_LEAGUE_WEIGHTS}


def _require_positive_integer(name, value):
    """校验必须为正整数的训练参数"""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _require_finite_number(name, value):
    """校验必须为有限数值的训练参数"""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def validate_training_config(
    net_config,
    *,
    update_timesteps,
    mini_batch_size,
    num_workers,
    training_device,
    worker_timeout,
    gamma,
    learning_rate,
    entropy,
    gae_lambda,
    clip_eps,
    ppo_epochs=DEFAULT_PPO_EPOCHS,
    target_kl=DEFAULT_TARGET_KL,
):
    """统一拒绝会导致模型构建或 PPO 训练失效的非法配置"""
    if not isinstance(net_config, dict):
        raise ValueError("net_config must be a dictionary")

    for key in ("d_model", "n_heads", "n_layers", "vocab_size"):
        if key not in net_config:
            raise ValueError(f"net_config is missing required key: {key}")
        _require_positive_integer(f"net_config.{key}", net_config[key])

    apply_current_protocol_metadata(net_config)

    if net_config["d_model"] % net_config["n_heads"] != 0:
        raise ValueError("net_config.d_model must be divisible by net_config.n_heads")

    _require_positive_integer("update_timesteps", update_timesteps)
    _require_positive_integer("mini_batch_size", mini_batch_size)
    _require_positive_integer("num_workers", num_workers)
    _require_positive_integer("ppo_epochs", ppo_epochs)
    if ppo_epochs > MAX_PPO_EPOCHS:
        raise ValueError(f"ppo_epochs must not exceed {MAX_PPO_EPOCHS}")
    if mini_batch_size % 2:
        raise ValueError("mini_batch_size must be even for planning consistency pairs")
    if mini_batch_size > update_timesteps:
        raise ValueError("mini_batch_size must not exceed update_timesteps")

    if training_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training_device must be 'auto', 'cpu' or 'cuda'")

    _require_finite_number("worker_timeout", worker_timeout)
    if worker_timeout <= 30:
        raise ValueError("worker_timeout must be greater than 30 seconds")

    for name, value in (
        ("gamma", gamma),
        ("learning_rate", learning_rate),
        ("entropy", entropy),
        ("gae_lambda", gae_lambda),
        ("clip_eps", clip_eps),
        ("target_kl", target_kl),
    ):
        _require_finite_number(name, value)

    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in the interval (0, 1]")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if entropy < 0:
        raise ValueError("entropy must be greater than or equal to 0")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must be in the interval [0, 1]")
    if not 0 < clip_eps <= 1:
        raise ValueError("clip_eps must be in the interval (0, 1]")
    if target_kl < 0:
        raise ValueError("target_kl must be non-negative; 0 disables KL early stopping")


def validate_max_iterations(max_iterations):
    """校验训练循环目标轮数"""
    _require_positive_integer("max_iterations", max_iterations)


def validate_model_prefix(model_prefix):
    """校验可安全用于跨平台模型文件名的人工前缀。"""
    if not isinstance(model_prefix, str) or not MODEL_PREFIX_PATTERN.fullmatch(
        model_prefix
    ):
        raise ValueError(
            "model_prefix must start with an ASCII letter or digit and contain "
            "only letters, digits, '_' or '-', with a maximum length of 64"
        )
    return model_prefix


def resolve_training_target(
    current_iteration,
    *,
    target_iteration=None,
    additional_iterations=None,
):
    """将绝对目标或追加轮数解析为唯一且严格递增的停止轮次。"""
    if (
        isinstance(current_iteration, bool)
        or not isinstance(current_iteration, Integral)
        or current_iteration < 0
    ):
        raise ValueError("current_iteration must be a non-negative integer")
    if (target_iteration is None) == (additional_iterations is None):
        raise ValueError(
            "exactly one of target_iteration or additional_iterations is required"
        )

    if target_iteration is not None:
        _require_positive_integer("target_iteration", target_iteration)
        if target_iteration <= current_iteration:
            raise ValueError(
                "target_iteration must be greater than the checkpoint iteration"
            )
        return int(target_iteration)

    _require_positive_integer("additional_iterations", additional_iterations)
    return int(current_iteration + additional_iterations)
