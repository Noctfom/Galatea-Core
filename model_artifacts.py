# 本文件统一管理检查点、ONNX 外置权重及部署包中的模型产物

import json
import os
import stat
from pathlib import Path

from checkpoint_utils import (
    CHECKPOINT_FORMAT_VERSION,
    inspect_training_checkpoint,
    validate_model_id,
)
from training_validation import validate_model_prefix


ARTIFACT_MANIFEST_FORMAT_VERSION = 2
ONNX_IDENTITY_KEYS = {
    "model_id": "galatea.model_id",
    "model_prefix": "galatea.model_prefix",
    "iteration": "galatea.iteration",
}


def model_artifact_stem(model_prefix, iteration):
    """根据已校验前缀和轮次生成一组模型产物的公共文件名"""
    validate_model_prefix(model_prefix)
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    return f"{model_prefix}_iter_{iteration}"


def is_primary_model_filename(filename):
    """判断文件是否为可独立选择的检查点或 ONNX 主图"""
    name = Path(filename).name
    return name.endswith(".pth") or name.endswith(".onnx")


def is_artifact_manifest_filename(filename):
    """判断文件是否为训练轮次产物清单"""
    return Path(filename).name.endswith(".artifacts.json")


def checkpoint_artifact_manifest_path(checkpoint_path):
    """返回与检查点同轮次的产物清单路径"""
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}.artifacts.json")


def _iter_onnx_tensors(graph):
    """递归遍历 ONNX 图及子图中的所有张量"""
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _iter_onnx_tensors(attribute.g)
            for child_graph in attribute.graphs:
                yield from _iter_onnx_tensors(child_graph)


def tag_onnx_model_identity(onnx_path, *, model_id, model_prefix, iteration):
    """把模型 UUID、前缀和轮次写入 ONNX 主图元数据并原子替换"""
    import onnx

    validate_model_id(model_id)
    validate_model_prefix(model_prefix)
    model_artifact_stem(model_prefix, iteration)
    graph_path = Path(onnx_path).resolve()
    model = onnx.load(str(graph_path), load_external_data=False)
    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(
        {
            ONNX_IDENTITY_KEYS["model_id"]: model_id,
            ONNX_IDENTITY_KEYS["model_prefix"]: model_prefix,
            ONNX_IDENTITY_KEYS["iteration"]: str(iteration),
        }
    )
    onnx.helper.set_model_props(model, properties)
    temporary_path = graph_path.with_suffix(graph_path.suffix + ".identity.tmp")
    try:
        onnx.save_model(model, str(temporary_path))
        os.replace(temporary_path, graph_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _safe_external_data_path(model_dir, location):
    """解析 ONNX 外置权重路径并拒绝越出模型目录的引用"""
    normalized = str(location).replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"ONNX external data path is unsafe: {location!r}")

    model_root = Path(model_dir).resolve()
    resolved = (model_root / relative).resolve()
    if os.path.commonpath((str(model_root), str(resolved))) != str(model_root):
        raise ValueError(f"ONNX external data escapes model directory: {location!r}")
    return resolved


def _read_artifact_manifest(marker_path):
    """读取并校验当前版本产物清单的基础结构"""
    with open(marker_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    manifest_version = payload.get("artifact_manifest_version")
    if (
        isinstance(manifest_version, bool)
        or manifest_version != ARTIFACT_MANIFEST_FORMAT_VERSION
    ):
        raise ValueError(f"artifact manifest version mismatch: {marker_path.name}")
    checkpoint_version = payload.get("checkpoint_format_version")
    if (
        isinstance(checkpoint_version, bool)
        or checkpoint_version != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(f"checkpoint format version mismatch: {marker_path.name}")
    validate_model_id(payload.get("model_id"))
    validate_model_prefix(payload.get("model_prefix"))
    iteration = payload.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"artifact iteration is invalid: {marker_path.name}")
    return payload


def _validate_onnx_artifact_manifest(graph_path, record, expected_model_id=None):
    """确认 ONNX 与产物清单属于同一模型且依赖文件完整"""
    marker_path = graph_path.with_name(f"{graph_path.stem}.artifacts.json")
    if not marker_path.is_file():
        if expected_model_id is not None:
            raise FileNotFoundError(
                f"ONNX identity manifest is missing: {marker_path.name}"
            )
        return

    payload = _read_artifact_manifest(marker_path)
    if expected_model_id is not None and payload["model_id"] != expected_model_id:
        raise PermissionError(
            f"ONNX model_id mismatch: expected {expected_model_id}, "
            f"found {payload['model_id']}"
        )

    onnx_record = payload.get("onnx") or {}
    if onnx_record.get("status") != "complete":
        raise RuntimeError(
            f"ONNX artifact is not marked complete in {marker_path.name}: "
            f"{onnx_record.get('status', 'unknown')}"
        )
    if onnx_record.get("files") != record["files"]:
        raise ValueError(f"ONNX artifact file list mismatch: {marker_path.name}")
    for key in ("model_id", "model_prefix", "iteration"):
        if onnx_record.get(key) != payload.get(key):
            raise ValueError(
                f"ONNX artifact {key} mismatch: {marker_path.name}"
            )
        if record.get(key) != payload.get(key):
            raise ValueError(
                f"ONNX embedded {key} mismatch: {marker_path.name}"
            )


def describe_onnx_artifact(
    onnx_path,
    *,
    require_complete=True,
    validate_manifest=True,
    expected_model_id=None,
    model_id=None,
    model_prefix=None,
    iteration=None,
):
    """读取 ONNX 主图引用并返回主图与全部外置权重的完整记录"""
    import onnx

    graph_path = Path(onnx_path).resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f"ONNX graph does not exist: {graph_path}")

    model = onnx.load(str(graph_path), load_external_data=False)
    properties = {item.key: item.value for item in model.metadata_props}
    embedded_model_id = properties.get(ONNX_IDENTITY_KEYS["model_id"])
    embedded_model_prefix = properties.get(ONNX_IDENTITY_KEYS["model_prefix"])
    embedded_iteration = properties.get(ONNX_IDENTITY_KEYS["iteration"])
    if embedded_iteration is not None:
        try:
            embedded_iteration = int(embedded_iteration)
        except ValueError as error:
            raise ValueError("ONNX embedded iteration must be an integer") from error

    embedded_identity = (
        embedded_model_id,
        embedded_model_prefix,
        embedded_iteration,
    )
    if any(value is not None for value in embedded_identity):
        if any(value is None for value in embedded_identity):
            raise ValueError("ONNX embedded model identity is incomplete")
        validate_model_id(embedded_model_id)
        validate_model_prefix(embedded_model_prefix)
    if expected_model_id is not None and embedded_model_id != expected_model_id:
        raise PermissionError(
            f"ONNX embedded model_id mismatch: expected {expected_model_id}, "
            f"found {embedded_model_id}"
        )
    for name, requested, embedded in (
        ("model_id", model_id, embedded_model_id),
        ("model_prefix", model_prefix, embedded_model_prefix),
        ("iteration", iteration, embedded_iteration),
    ):
        if requested is not None and requested != embedded:
            raise ValueError(
                f"ONNX embedded {name} does not match the requested identity"
            )
    locations = set()
    for tensor in _iter_onnx_tensors(model.graph):
        external = {item.key: item.value for item in tensor.external_data}
        if "location" in external:
            locations.add(external["location"])

    files = [graph_path.name]
    for location in sorted(locations):
        data_path = _safe_external_data_path(graph_path.parent, location)
        if require_complete and not data_path.is_file():
            raise FileNotFoundError(
                f"ONNX external data is missing for {graph_path.name}: {location}"
            )
        files.append(os.path.relpath(data_path, graph_path.parent).replace("\\", "/"))

    record = {
        "format": "onnx",
        "model_id": embedded_model_id,
        "model_prefix": embedded_model_prefix,
        "iteration": embedded_iteration,
        "primary": graph_path.name,
        "files": files,
        "external_data": files[1:],
        "status": "complete",
    }
    if validate_manifest:
        _validate_onnx_artifact_manifest(
            graph_path,
            record,
            expected_model_id=expected_model_id,
        )
    return record


def discover_checkpoint_artifacts(
    model_dir,
    *,
    model_prefix=None,
    model_id=None,
    include_orphan_checkpoints=False,
):
    """通过轻量产物清单发现检查点，并按前缀或模型 UUID 精确筛选"""
    model_root = Path(model_dir).resolve()
    records = []
    discovered_paths = set()
    if not model_root.is_dir():
        return records
    for marker_path in model_root.glob("*.artifacts.json"):
        try:
            payload = _read_artifact_manifest(marker_path)
            checkpoint = payload.get("checkpoint") or {}
            checkpoint_name = str(checkpoint.get("file", ""))
            if Path(checkpoint_name).name != checkpoint_name:
                continue
            checkpoint_path = model_root / checkpoint_name
            if checkpoint.get("status") != "complete" or not checkpoint_path.is_file():
                continue
            if (
                model_prefix is not None
                and payload["model_prefix"].casefold() != model_prefix.casefold()
            ):
                continue
            if model_id is not None and payload["model_id"] != model_id:
                continue
            discovered_paths.add(checkpoint_path.resolve())
            records.append(
                {
                    "model_id": payload["model_id"],
                    "model_prefix": payload["model_prefix"],
                    "iteration": int(payload["iteration"]),
                    "checkpoint_format_version": payload.get(
                        "checkpoint_format_version"
                    ),
                    "checkpoint_path": str(checkpoint_path),
                    "manifest_path": str(marker_path),
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

    if include_orphan_checkpoints:
        for checkpoint_path in model_root.glob("*.pth"):
            if checkpoint_path.resolve() in discovered_paths:
                continue
            if checkpoint_artifact_manifest_path(checkpoint_path).is_file():
                continue
            try:
                metadata = inspect_training_checkpoint(checkpoint_path)
                if metadata["format_warning"] is not None:
                    continue
                if (
                    model_prefix is not None
                    and metadata["model_prefix"].casefold() != model_prefix.casefold()
                ):
                    continue
                if model_id is not None and metadata["model_id"] != model_id:
                    continue
                records.append(
                    {
                        "model_id": metadata["model_id"],
                        "model_prefix": metadata["model_prefix"],
                        "iteration": int(metadata["iteration"]),
                        "checkpoint_format_version": metadata[
                            "checkpoint_format_version"
                        ],
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "manifest_path": None,
                    }
                )
            except (OSError, TypeError, ValueError, KeyError):
                continue
    return sorted(records, key=lambda item: item["iteration"])


def find_prefix_identity_conflicts(model_dir, model_prefix, expected_model_id):
    """查找同前缀但不属于预期 UUID 的检查点，供 WebUI 和 Trainer 告警"""
    validate_model_prefix(model_prefix)
    validate_model_id(expected_model_id)
    return [
        record
        for record in discover_checkpoint_artifacts(
            model_dir,
            model_prefix=model_prefix,
            include_orphan_checkpoints=True,
        )
        if record["model_id"] != expected_model_id
    ]


def find_model_prefix_namespace_files(model_dir, model_prefix):
    """查找会占用规范输出文件名空间的同前缀检查点。"""
    validate_model_prefix(model_prefix)
    model_root = Path(model_dir).resolve()
    canonical_prefix = f"{model_prefix}_iter_".casefold()
    if not model_root.is_dir():
        return []
    return sorted(
        str(path.resolve())
        for path in model_root.glob("*.pth")
        if path.name.casefold().startswith(canonical_prefix)
    )


def assert_checkpoint_target_identity(checkpoint_path, expected_model_id):
    """覆盖已有检查点前核验其清单 UUID，拒绝写坏同名前缀的其他模型"""
    validate_model_id(expected_model_id)
    checkpoint = Path(checkpoint_path).resolve()
    marker_path = checkpoint_artifact_manifest_path(checkpoint)
    if not checkpoint.exists() and not marker_path.exists():
        return
    if not marker_path.is_file():
        raise PermissionError(
            f"refusing to overwrite checkpoint without identity manifest: {checkpoint.name}"
        )
    payload = _read_artifact_manifest(marker_path)
    if payload["model_id"] != expected_model_id:
        raise PermissionError(
            f"refusing to overwrite model_id {payload['model_id']} with "
            f"{expected_model_id}: {checkpoint.name}"
        )


def collect_model_artifact_files(model_dir, selected_models):
    """展开用户选择，自动补齐 ONNX 外置权重和同轮次产物清单"""
    model_root = Path(model_dir).resolve()
    collected = []

    def append_once(relative_name):
        """按原顺序加入文件并避免重复打包"""
        normalized = str(relative_name).replace("\\", "/")
        if normalized not in collected:
            collected.append(normalized)

    for selected in selected_models:
        name = Path(selected).name
        if name != selected or not is_primary_model_filename(name):
            raise ValueError(f"invalid model artifact selection: {selected!r}")
        primary_path = model_root / name
        if not primary_path.is_file():
            raise FileNotFoundError(f"selected model does not exist: {primary_path}")

        if name.endswith(".onnx"):
            record = describe_onnx_artifact(primary_path, require_complete=True)
            for relative_name in record["files"]:
                append_once(relative_name)
        else:
            append_once(name)

        marker = f"{primary_path.stem}.artifacts.json"
        if (model_root / marker).is_file():
            append_once(marker)

    return collected


def build_package_model_records(model_dir, selected_models):
    """为部署包清单生成带轮次和依赖文件的模型记录"""
    model_root = Path(model_dir).resolve()
    records = []
    for selected in selected_models:
        name = Path(selected).name
        if name != selected or not is_primary_model_filename(name):
            raise ValueError(f"invalid model artifact selection: {selected!r}")
        primary_path = model_root / name
        if not primary_path.is_file():
            raise FileNotFoundError(f"selected model does not exist: {primary_path}")
        if name.endswith(".onnx"):
            records.append(describe_onnx_artifact(primary_path, require_complete=True))
        else:
            marker_path = checkpoint_artifact_manifest_path(primary_path)
            if not marker_path.is_file():
                raise FileNotFoundError(
                    f"checkpoint identity manifest is missing: {marker_path.name}"
                )
            payload = _read_artifact_manifest(marker_path)
            records.append(
                {
                    "format": "pytorch_checkpoint",
                    "model_id": payload["model_id"],
                    "model_prefix": payload["model_prefix"],
                    "iteration": payload["iteration"],
                    "primary": name,
                    "files": [name],
                    "status": "complete",
                }
            )
    return records


def write_checkpoint_artifact_manifest(
    checkpoint_path,
    iteration,
    *,
    model_id,
    model_prefix,
    checkpoint_format_version=CHECKPOINT_FORMAT_VERSION,
    onnx_record=None,
    onnx_error=None,
):
    """在保存检查点时写入同轮次产物清单，并标记 ONNX 是否完整"""
    validate_model_id(model_id)
    validate_model_prefix(model_prefix)
    checkpoint = Path(checkpoint_path).resolve()
    manifest_path = checkpoint_artifact_manifest_path(checkpoint)
    if onnx_error == "export_in_progress":
        onnx_status = "in_progress"
    elif onnx_error:
        onnx_status = "failed"
    else:
        onnx_status = "disabled"
    if onnx_record is not None:
        onnx_record = dict(onnx_record)
        onnx_record.update(
            {
                "model_id": model_id,
                "model_prefix": model_prefix,
                "iteration": int(iteration),
            }
        )
    payload = {
        "artifact_manifest_version": ARTIFACT_MANIFEST_FORMAT_VERSION,
        "checkpoint_format_version": int(checkpoint_format_version),
        "model_id": model_id,
        "model_prefix": model_prefix,
        "iteration": int(iteration),
        "checkpoint": {
            "format": "pytorch_checkpoint",
            "file": checkpoint.name,
            "status": "complete" if checkpoint.is_file() else "missing",
        },
        "onnx": onnx_record
        if onnx_record is not None
        else {
            "status": onnx_status,
            "error": str(onnx_error) if onnx_error else None,
        },
    }

    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary_path, manifest_path)
    return manifest_path


def safe_extract_zip(archive, target_dir):
    """校验 ZIP 成员路径和符号链接后安全解压部署包"""
    target_root = Path(target_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        relative = Path(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe archive member path: {info.filename!r}")
        destination = (target_root / relative).resolve()
        if os.path.commonpath((str(target_root), str(destination))) != str(target_root):
            raise ValueError(f"archive member escapes target directory: {info.filename!r}")
        unix_mode = info.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise ValueError(f"archive symbolic links are not allowed: {info.filename!r}")

    archive.extractall(target_root)
