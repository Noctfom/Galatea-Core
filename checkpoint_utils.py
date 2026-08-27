# 训练检查点协议版本、身份校验、严格恢复与规范化导出

import os
import stat
import uuid
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch

from training_validation import validate_model_prefix


# 仅表示训练检查点数据协议，必须独立于 app.py 中的框架版本维护
CHECKPOINT_FORMAT_VERSION = 1
DEFAULT_MODEL_PREFIX = "galatea"
MAX_CHECKPOINT_FILE_BYTES = 32 * 1024 * 1024 * 1024

REQUIRED_TRAINING_CHECKPOINT_KEYS = {
    "checkpoint_format_version",
    "model_id",
    "model_prefix",
    "run_id",
    "model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "net_config",
    "iteration",
    "train_step",
    "global_step",
}


def validate_model_id(model_id):
    """校验框架自动生成的模型 UUID，并返回规范字符串"""
    if not isinstance(model_id, str):
        raise ValueError("model_id must be a UUID string")
    try:
        normalized = str(uuid.UUID(model_id))
    except (ValueError, AttributeError) as error:
        raise ValueError("model_id must be a valid UUID") from error
    if model_id != normalized:
        raise ValueError("model_id must use canonical lowercase UUID format")
    return normalized


def generate_model_id():
    """为全新模型生成不可手动指定的随机 UUID"""
    return str(uuid.uuid4())


def get_checkpoint_format_warning(checkpoint):
    """返回检查点协议版本告警；版本一致时返回空值"""
    actual = checkpoint.get("checkpoint_format_version")
    if (
        isinstance(actual, int)
        and not isinstance(actual, bool)
        and actual == CHECKPOINT_FORMAT_VERSION
    ):
        return None
    return (
        "检查点协议版本不兼容: "
        f"文件={actual!r}, 当前={CHECKPOINT_FORMAT_VERSION}。"
        "该版本独立于 Galatea 框架版本，且当前没有对应迁移规则。"
    )


def safe_load_torch_checkpoint(
    path,
    map_location="cpu",
    *,
    materialize_tensors=True,
):
    """在固定文件句柄上预检危险全局对象，并用受限反序列化器加载检查点"""
    if not path:
        raise FileNotFoundError(f"training checkpoint does not exist: {path}")
    checkpoint_path = Path(path)
    if checkpoint_path.suffix.casefold() != ".pth":
        raise ValueError("training checkpoint filename must end with .pth")
    if checkpoint_path.is_symlink():
        raise ValueError("symbolic-link checkpoints are not allowed")

    try:
        with open(checkpoint_path, "rb") as stream:
            file_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("training checkpoint must be a regular file")
            if file_stat.st_size <= 0:
                raise ValueError("training checkpoint must not be empty")
            if file_stat.st_size > MAX_CHECKPOINT_FILE_BYTES:
                raise ValueError(
                    "training checkpoint exceeds the 32 GiB safety limit"
                )

            unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(
                stream
            )
            if unsafe_globals:
                preview = ", ".join(sorted(unsafe_globals)[:5])
                raise ValueError(
                    "training checkpoint contains non-allowlisted globals: "
                    f"{preview}"
                )
            stream.seek(0)
            if materialize_tensors:
                load_context = nullcontext()
            else:
                from torch._subclasses.fake_tensor import FakeTensorMode

                load_context = FakeTensorMode()
            with load_context:
                return torch.load(
                    stream,
                    map_location=map_location,
                    weights_only=True,
                    mmap=False,
                )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"training checkpoint does not exist: {checkpoint_path}"
        ) from None


def inspect_training_checkpoint(path, map_location="cpu"):
    """安全读取 WebUI 所需的检查点身份和架构元数据，不修改文件"""
    checkpoint = safe_load_torch_checkpoint(
        path,
        map_location=map_location,
        materialize_tensors=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("training checkpoint must be a dictionary")
    format_warning = get_checkpoint_format_warning(checkpoint)
    if format_warning is None:
        validate_model_id(checkpoint.get("model_id"))
        validate_model_prefix(checkpoint.get("model_prefix"))
        if (
            isinstance(checkpoint.get("iteration"), bool)
            or not isinstance(checkpoint.get("iteration"), int)
            or checkpoint["iteration"] < 0
        ):
            raise ValueError("checkpoint iteration must be a non-negative integer")
    return {
        "checkpoint_format_version": checkpoint.get("checkpoint_format_version"),
        "format_warning": format_warning,
        "model_id": checkpoint.get("model_id"),
        "model_prefix": checkpoint.get("model_prefix"),
        "iteration": checkpoint.get("iteration"),
        "run_id": checkpoint.get("run_id"),
        "net_config": checkpoint.get("net_config", {}),
    }


def validate_training_checkpoint(checkpoint, *, source_path=None):
    """校验已加载检查点的协议、UUID、结构和可选文件名一致性。"""
    if not isinstance(checkpoint, dict):
        raise TypeError("training checkpoint must be a dictionary")

    format_warning = get_checkpoint_format_warning(checkpoint)
    if format_warning:
        warnings.warn(format_warning, RuntimeWarning, stacklevel=2)
        raise ValueError(format_warning)

    missing = sorted(REQUIRED_TRAINING_CHECKPOINT_KEYS.difference(checkpoint))
    if missing:
        raise KeyError(f"training checkpoint is missing required keys: {missing}")

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("model_state_dict must be a non-empty dictionary")

    compiled_keys = [key for key in state_dict if key.startswith("_orig_mod.")]
    if compiled_keys:
        raise ValueError(
            "checkpoint model_state_dict must use canonical uncompiled keys; "
            f"found compiled key {compiled_keys[0]!r}"
        )

    validate_model_id(checkpoint["model_id"])
    validate_model_prefix(checkpoint["model_prefix"])
    if (
        isinstance(checkpoint["iteration"], bool)
        or not isinstance(checkpoint["iteration"], int)
        or checkpoint["iteration"] < 0
    ):
        raise ValueError("checkpoint iteration must be a non-negative integer")
    if not isinstance(checkpoint["run_id"], str) or not checkpoint["run_id"]:
        raise ValueError("checkpoint run_id must be a non-empty string")

    expected_name = f"{checkpoint['model_prefix']}_iter_{checkpoint['iteration']}.pth"
    if source_path is not None and Path(source_path).name != expected_name:
        warnings.warn(
            f"检查点文件名 {Path(source_path).name!r} 与内部身份不一致；"
            f"内部元数据 {expected_name!r} 将作为恢复依据。",
            RuntimeWarning,
            stacklevel=2,
        )

    return checkpoint


def load_training_checkpoint(path, map_location="cpu"):
    """加载并校验当前训练框架生成的完整检查点"""
    checkpoint = safe_load_torch_checkpoint(path, map_location=map_location)
    return validate_training_checkpoint(checkpoint, source_path=path)


def validate_training_checkpoint_file(path, map_location="cpu"):
    """不分配真实张量存储地校验外部检查点的完整协议与模型身份"""
    checkpoint = safe_load_torch_checkpoint(
        path,
        map_location=map_location,
        materialize_tensors=False,
    )
    return validate_training_checkpoint(checkpoint, source_path=path)


def restore_model_state_strict(model, checkpoint):
    """在 torch.compile 包装前严格恢复模型参数"""
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def canonical_model_state_dict(model, *, to_cpu=False):
    """导出不受 torch.compile 包装前缀影响的规范权重字典"""
    canonical = {}
    for key, value in model.state_dict().items():
        clean_key = key.removeprefix("_orig_mod.")
        if clean_key in canonical:
            raise ValueError(f"duplicate canonical model key: {clean_key!r}")
        canonical[clean_key] = value.cpu() if to_cpu else value
    return canonical
