# 训练检查点协议版本、身份校验、严格恢复与规范化导出

import os
import re
import stat
import uuid
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch

from protocol_schema import (
    MODEL_PROTOCOL_VERSION,
    apply_current_protocol_metadata,
    get_current_protocol_metadata,
    validate_protocol_metadata,
)
from training_validation import validate_model_prefix


# 仅表示训练检查点数据协议，必须独立于 app.py 中的框架版本维护
CHECKPOINT_FORMAT_VERSION = 3
DEFAULT_MODEL_PREFIX = "galatea"
MAX_CHECKPOINT_FILE_BYTES = 32 * 1024 * 1024 * 1024

REQUIRED_TRAINING_CHECKPOINT_KEYS = {
    "checkpoint_format_version",
    "model_protocol_version",
    "protocol_schema_revision",
    "protocol_schema_hash",
    "card_vocab_hash",
    "card_vocab_size",
    "card_vocab_card_count",
    "semantic_lookup_format_version",
    "semantic_lookup_hash",
    "semantic_lookup_card_count",
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


def validate_ppo_resume_stage(checkpoint):
    """Inference can load BC weights, but PPO must not restore a BC optimizer."""
    if checkpoint.get('training_stage', 'ppo') != 'ppo':
        raise ValueError('该检查点属于独立模仿学习，不能作为 PPO --resume；继续 PPO 请使用 train --init-from，只载入权重并创建新的 PPO 优化器。')
    return checkpoint


def get_training_lineage(checkpoint, _depth=0):
    """Keep cumulative PPO iterations and BC epochs separate from local counters."""
    if _depth > 16:
        raise ValueError('legacy training lineage is cyclic or too deep')
    lineage = checkpoint.get('training_lineage')
    if lineage is not None:
        if (not isinstance(lineage, dict) or type(lineage.get('format_version')) is not int or lineage.get('format_version') != 1
                or any(type(lineage.get(k)) is not int or lineage[k] < 0 for k in ('ppo_iteration', 'bc_epochs'))):
            raise ValueError('invalid training lineage counters')
        validate_model_prefix(lineage.get('base_prefix'))
        if checkpoint.get('training_stage') == 'imitation' and lineage['bc_epochs'] < checkpoint['iteration']:
            raise ValueError('cumulative BC count is less than local BC epoch')
        if (checkpoint.get('training_stage', 'ppo') == 'ppo'
                and lineage['ppo_iteration'] != checkpoint['iteration']):
            raise ValueError('PPO lineage iteration differs from checkpoint iteration')
        return dict(lineage)
    prefix, ppo, bc = checkpoint['model_prefix'], checkpoint['iteration'], 0
    if checkpoint.get('training_stage') == 'imitation':
        info = checkpoint.get('imitation') or {}
        parent = info.get('parent') or {}
        if parent.get('training_lineage'):
            prior = parent['training_lineage']
            prefix, ppo, bc = prior['base_prefix'], prior['ppo_iteration'], prior['bc_epochs']
        else:
            # Original 3.15.3 stored the parent filename and local BC iteration.
            # Use recorded provenance; never reinterpret the BC epoch as PPO age.
            parent_name = str(parent.get('checkpoint', '')).replace('\\', '/').rsplit('/', 1)[-1]
            match = re.fullmatch(r'(.+)_iter_(\d+)(?:_bc_(\d+))?\.pth', parent_name)
            if not match:
                raise ValueError('legacy BC checkpoint lacks recoverable PPO parent lineage')
            prefix, ppo, bc = match[1], int(match[2]), int(match[3] or 0)
            parent_file = Path(parent.get('checkpoint', ''))
            if prefix == checkpoint['model_prefix'] and parent_file.is_file() and not parent_file.is_symlink():
                previous = safe_load_torch_checkpoint(parent_file, materialize_tensors=False)
                if previous.get('model_id') == parent.get('model_id'):
                    prior = get_training_lineage(previous, _depth + 1)
                    prefix, ppo, bc = prior['base_prefix'], prior['ppo_iteration'], prior['bc_epochs']
        bc += checkpoint['iteration']
    validate_model_prefix(prefix)
    return {'format_version': 1, 'base_prefix': prefix, 'ppo_iteration': ppo, 'bc_epochs': bc}


def checkpoint_stem(checkpoint):
    """Display/save dual counters when lineage is explicitly present or inferred."""
    if checkpoint.get('training_lineage') is not None or checkpoint.get('training_stage') == 'imitation':
        info = get_training_lineage(checkpoint)
        suffix = f"_bc_{info['bc_epochs']}" if info['bc_epochs'] else ''
        return f"{info['base_prefix']}_iter_{info['ppo_iteration']}{suffix}"
    return f"{checkpoint['model_prefix']}_iter_{checkpoint['iteration']}"


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


def get_model_protocol_warning(checkpoint):
    """返回模型内部协议版本告警；一致时返回空值"""
    actual = checkpoint.get("model_protocol_version")
    if (
        isinstance(actual, int)
        and not isinstance(actual, bool)
        and actual == MODEL_PROTOCOL_VERSION
    ):
        return None
    return (
        "模型输入/动作协议版本不兼容: "
        f"文件={actual!r}, 当前={MODEL_PROTOCOL_VERSION}。"
        "该协议决定网络权重与动作张量结构，不能混用。"
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
    model_protocol_warning = get_model_protocol_warning(checkpoint)
    if format_warning is None and model_protocol_warning is None:
        validate_protocol_metadata(checkpoint, label="checkpoint")
        validate_model_id(checkpoint.get("model_id"))
        validate_model_prefix(checkpoint.get("model_prefix"))
        net_config = checkpoint.get("net_config")
        apply_current_protocol_metadata(net_config)
        if (
            isinstance(checkpoint.get("iteration"), bool)
            or not isinstance(checkpoint.get("iteration"), int)
            or checkpoint["iteration"] < 0
        ):
            raise ValueError("checkpoint iteration must be a non-negative integer")
    metadata = {
        "checkpoint_format_version": checkpoint.get("checkpoint_format_version"),
        "model_protocol_version": checkpoint.get("model_protocol_version"),
        "format_warning": format_warning,
        "model_protocol_warning": model_protocol_warning,
        "model_id": checkpoint.get("model_id"),
        "model_prefix": checkpoint.get("model_prefix"),
        "iteration": checkpoint.get("iteration"),
        "run_id": checkpoint.get("run_id"),
        "net_config": checkpoint.get("net_config", {}),
        "training_stage": checkpoint.get("training_stage", "ppo"),
        "windbot_training": checkpoint.get("windbot_training"),
        "league_weights": checkpoint.get("league_weights"),
    }
    for key in get_current_protocol_metadata():
        metadata[key] = checkpoint.get(key)
    if checkpoint.get('training_stage') == 'imitation':
        info = checkpoint.get('imitation', {})
        metadata['imitation'] = {key: info.get(key) for key in ('dataset', 'dataset_sha256', 'config', 'parent', 'counter_units')}
    metadata['training_lineage'] = get_training_lineage(checkpoint)
    metadata['display_name'] = checkpoint_stem(checkpoint)
    return metadata


def validate_training_checkpoint(
    checkpoint,
    *,
    source_path=None,
    card_vocabulary=None,
    semantic_root=".",
):
    """校验已加载检查点的协议、UUID、结构和可选文件名一致性"""
    if not isinstance(checkpoint, dict):
        raise TypeError("training checkpoint must be a dictionary")

    format_warning = get_checkpoint_format_warning(checkpoint)
    if format_warning:
        warnings.warn(format_warning, RuntimeWarning, stacklevel=2)
        raise ValueError(format_warning)

    model_protocol_warning = get_model_protocol_warning(checkpoint)
    if model_protocol_warning:
        warnings.warn(model_protocol_warning, RuntimeWarning, stacklevel=2)
        raise ValueError(model_protocol_warning)

    validate_protocol_metadata(
        checkpoint,
        label="checkpoint",
        card_vocabulary=card_vocabulary,
        semantic_root=semantic_root,
    )

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
    net_config = checkpoint["net_config"]
    validate_protocol_metadata(
        net_config,
        label="checkpoint net_config",
        card_vocabulary=card_vocabulary,
        semantic_root=semantic_root,
    )
    for key in get_current_protocol_metadata(
        card_vocabulary,
        semantic_root=semantic_root,
    ):
        if checkpoint.get(key) != net_config.get(key):
            raise ValueError(
                f"checkpoint top-level {key} does not match checkpoint net_config"
            )
    if (
        isinstance(checkpoint["iteration"], bool)
        or not isinstance(checkpoint["iteration"], int)
        or checkpoint["iteration"] < 0
    ):
        raise ValueError("checkpoint iteration must be a non-negative integer")
    if not isinstance(checkpoint["run_id"], str) or not checkpoint["run_id"]:
        raise ValueError("checkpoint run_id must be a non-empty string")

    if checkpoint.get('training_lineage') is not None:
        get_training_lineage(checkpoint)
    expected_name = (checkpoint_stem(checkpoint) if checkpoint.get('training_lineage') is not None else
                     f"{checkpoint['model_prefix']}_iter_{checkpoint['iteration']}") + '.pth'
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


def validate_training_checkpoint_file(
    path,
    map_location="cpu",
    *,
    card_vocabulary=None,
    semantic_root=".",
):
    """不分配真实张量存储地校验外部检查点的完整协议与模型身份"""
    checkpoint = safe_load_torch_checkpoint(
        path,
        map_location=map_location,
        materialize_tensors=False,
    )
    return validate_training_checkpoint(
        checkpoint,
        source_path=path,
        card_vocabulary=card_vocabulary,
        semantic_root=semantic_root,
    )


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


def initialize_ppo_from_weights(source_path, output_root='./models', *, learning_rate=1e-4,
                                model_prefix=None, checkpoint=None):
    """Create a separate PPO branch; never restore a BC optimizer or counters."""
    from galatea_net import GalateaNet
    from model_artifacts import write_checkpoint_artifact_manifest
    from core_trajectory import file_sha256
    checkpoint = (load_training_checkpoint(source_path) if checkpoint is None else
                  validate_training_checkpoint(checkpoint, source_path=source_path))
    lineage = get_training_lineage(checkpoint)
    prefix = validate_model_prefix(model_prefix or lineage['base_prefix'])
    lineage = dict(lineage, base_prefix=prefix)
    config = apply_current_protocol_metadata(checkpoint['net_config'])
    network = GalateaNet(config).cpu()
    restore_model_state_strict(network, checkpoint)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    model_id, run_id = generate_model_id(), uuid.uuid4().hex
    root = Path(output_root).resolve() / 'ppo_branches' / run_id
    root.mkdir(parents=True, exist_ok=False)
    payload = {'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION, **get_current_protocol_metadata(),
        'net_config': config, 'model_id': model_id, 'model_prefix': prefix, 'run_id': run_id,
        'training_stage': 'ppo', 'training_lineage': lineage,
        'iteration': lineage['ppo_iteration'], 'train_step': 0, 'global_step': 0,
        'model_state_dict': canonical_model_state_dict(network, to_cpu=True),
        'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': {},
        'initialization': {'checkpoint': str(Path(source_path).resolve()), 'sha256': file_sha256(source_path),
            'model_id': checkpoint['model_id'], 'training_stage': checkpoint.get('training_stage', 'ppo'),
            'optimizer_reset': True, 'ppo_update_counters_reset': True}}
    path = root / (checkpoint_stem(payload) + '.pth')
    temporary = path.with_suffix('.pth.tmp')
    torch.save(payload, temporary)
    temporary.rename(path)
    write_checkpoint_artifact_manifest(path, payload['iteration'], model_id=model_id, model_prefix=prefix,
                                      training_stage='ppo', training_lineage=lineage)
    return str(path), payload
