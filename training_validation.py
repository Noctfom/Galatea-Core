# 本文件在训练资源分配前统一校验模型结构和 PPO 参数

import math
import re
from numbers import Integral, Real


MODEL_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


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
    worker_device,
    worker_timeout,
    gamma,
    learning_rate,
    entropy,
    gae_lambda,
    clip_eps,
):
    """统一拒绝会导致模型构建或 PPO 训练失效的非法配置"""
    if not isinstance(net_config, dict):
        raise ValueError("net_config must be a dictionary")

    for key in ("d_model", "n_heads", "n_layers", "vocab_size"):
        if key not in net_config:
            raise ValueError(f"net_config is missing required key: {key}")
        _require_positive_integer(f"net_config.{key}", net_config[key])

    if net_config["d_model"] % net_config["n_heads"] != 0:
        raise ValueError("net_config.d_model must be divisible by net_config.n_heads")

    _require_positive_integer("update_timesteps", update_timesteps)
    _require_positive_integer("mini_batch_size", mini_batch_size)
    _require_positive_integer("num_workers", num_workers)
    if mini_batch_size > update_timesteps:
        raise ValueError("mini_batch_size must not exceed update_timesteps")

    if worker_device not in {"cpu", "cuda"}:
        raise ValueError("worker_device must be either 'cpu' or 'cuda'")

    _require_finite_number("worker_timeout", worker_timeout)
    if worker_timeout <= 30:
        raise ValueError("worker_timeout must be greater than 30 seconds")

    for name, value in (
        ("gamma", gamma),
        ("learning_rate", learning_rate),
        ("entropy", entropy),
        ("gae_lambda", gae_lambda),
        ("clip_eps", clip_eps),
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
