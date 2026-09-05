# 本文件统一管理检查点、ONNX 外置权重及部署包中的模型产物

import json
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

from checkpoint_utils import (
    CHECKPOINT_FORMAT_VERSION,
    MODEL_PROTOCOL_VERSION,
    inspect_training_checkpoint,
    validate_training_checkpoint_file,
    validate_model_id,
)
from training_validation import validate_model_prefix
from semantic_assets import (
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
    HASH_MAPPING_FILENAME,
    KNOWLEDGE_BASE_FILENAME,
    SEMANTIC_ASSET_FILENAMES,
    validate_semantic_bundle,
)


ARTIFACT_MANIFEST_FORMAT_VERSION = 2
# 仅表示 .gkg 部署包协议，必须独立于 WebUI/框架版本维护
DEPLOY_PACKAGE_FORMAT_VERSION = 2
MAX_ONNX_GRAPH_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_MODEL_ARTIFACT_FILE_BYTES = 32 * 1024 * 1024 * 1024
MAX_MODEL_ARTIFACT_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_DEPLOY_PACKAGE_FILE_BYTES = 64 * 1024 * 1024 * 1024
MAX_DEPLOY_ARCHIVE_MEMBERS = 256
MAX_DEPLOY_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024 * 1024
MAX_DEPLOY_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_DEPLOY_COMPRESSION_RATIO = 1000
MAX_MANIFEST_FILE_BYTES = 2 * 1024 * 1024
SAFE_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SAFE_PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
MODEL_ARTIFACT_SUFFIXES = (".artifacts.json", ".onnx.data", ".onnx", ".pth")
DEPLOY_ROOT_FILES = {*SEMANTIC_ASSET_FILENAMES, "meta_staples.json"}
CODE_SEMANTIC_FILE_SET = {
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
}
ONNX_IDENTITY_KEYS = {
    "model_id": "galatea.model_id",
    "model_prefix": "galatea.model_prefix",
    "iteration": "galatea.iteration",
    "model_protocol_version": "galatea.model_protocol_version",
}


def validate_safe_filename(filename, *, allowed_suffixes=None):
    """校验不含路径、控制字符和跨平台保留名的普通文件名"""
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty string")
    if filename != Path(filename).name or "/" in filename or "\\" in filename:
        raise ValueError(f"filename must not contain a path: {filename!r}")
    if ".." in filename or not SAFE_FILENAME_PATTERN.fullmatch(filename):
        raise ValueError(f"filename contains unsupported characters: {filename!r}")
    if len(filename.encode("utf-8")) > 200:
        raise ValueError("filename exceeds the 200-byte safety limit")
    device_stem = filename.split(".", 1)[0].upper()
    if device_stem in WINDOWS_RESERVED_FILENAMES:
        raise ValueError(f"reserved filename is not allowed: {filename!r}")
    if allowed_suffixes and not any(
        filename.casefold().endswith(suffix.casefold()) for suffix in allowed_suffixes
    ):
        raise ValueError(f"filename has an unsupported suffix: {filename!r}")
    return filename


def validate_model_artifact_filename(filename):
    """校验检查点、ONNX、外置权重或轮次清单的模型产物文件名"""
    return validate_safe_filename(filename, allowed_suffixes=MODEL_ARTIFACT_SUFFIXES)


def validate_model_artifact_file(path):
    """校验模型制品是大小受限且不经过符号链接的普通文件"""
    artifact_path = Path(path)
    validate_model_artifact_filename(artifact_path.name)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError(f"model artifact must be a regular file: {artifact_path.name}")
    file_size = artifact_path.stat().st_size
    if file_size > MAX_MODEL_ARTIFACT_FILE_BYTES:
        raise ValueError(
            f"model artifact exceeds the 32 GiB safety limit: {artifact_path.name}"
        )
    return file_size


def validate_model_artifact_file_set(model_dir, filenames):
    """校验一组模型制品的单文件和总容量边界并返回总字节数"""
    model_root = Path(model_dir).resolve()
    total_size = 0
    for filename in filenames:
        safe_name = validate_model_artifact_filename(filename)
        total_size += validate_model_artifact_file(model_root / safe_name)
        if total_size > MAX_MODEL_ARTIFACT_TOTAL_BYTES:
            raise ValueError("model artifact set exceeds the 64 GiB safety limit")
    return total_size


def validate_package_name(package_name):
    """校验可安全用于跨平台部署包文件名的包名"""
    if not isinstance(package_name, str) or not SAFE_PACKAGE_NAME_PATTERN.fullmatch(
        package_name
    ):
        raise ValueError(
            "package_name must start with an ASCII letter or digit, contain only "
            "letters, digits, '.', '_' or '-', and be at most 64 characters"
        )
    if ".." in package_name or package_name.split(".", 1)[0].upper() in WINDOWS_RESERVED_FILENAMES:
        raise ValueError("package_name is not safe for a local filename")
    return package_name


def validate_deployment_package_filename(filename):
    """校验本地部署包必须是安全的单层 .gkg 文件名"""
    return validate_safe_filename(filename, allowed_suffixes=(".gkg",))


def model_artifact_stem(model_prefix, iteration):
    """根据已校验前缀和轮次生成一组模型产物的公共文件名"""
    validate_model_prefix(model_prefix)
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    return f"{model_prefix}_iter_{iteration}"


def is_primary_model_filename(filename):
    """判断文件是否为可独立选择的检查点或 ONNX 主图"""
    try:
        name = validate_safe_filename(filename, allowed_suffixes=(".pth", ".onnx"))
    except ValueError:
        return False
    return name.casefold().endswith(".pth") or name.casefold().endswith(".onnx")


def is_artifact_manifest_filename(filename):
    """判断文件是否为训练轮次产物清单"""
    try:
        validate_safe_filename(filename, allowed_suffixes=(".artifacts.json",))
    except ValueError:
        return False
    return filename.casefold().endswith(".artifacts.json")


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


def tag_onnx_model_identity(
    onnx_path,
    *,
    model_id,
    model_prefix,
    iteration,
    model_protocol_version=MODEL_PROTOCOL_VERSION,
):
    """把模型身份与模型协议版本写入 ONNX 主图并原子替换"""
    import onnx

    validate_model_id(model_id)
    validate_model_prefix(model_prefix)
    model_artifact_stem(model_prefix, iteration)
    if model_protocol_version != MODEL_PROTOCOL_VERSION:
        raise ValueError("ONNX model protocol version does not match the current protocol")
    source_graph_path = Path(onnx_path)
    if source_graph_path.is_symlink():
        raise ValueError("symbolic-link ONNX graphs are not allowed")
    graph_path = source_graph_path.resolve()
    validate_safe_filename(graph_path.name, allowed_suffixes=(".onnx",))
    model = onnx.load(str(graph_path), load_external_data=False)
    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(
        {
            ONNX_IDENTITY_KEYS["model_id"]: model_id,
            ONNX_IDENTITY_KEYS["model_prefix"]: model_prefix,
            ONNX_IDENTITY_KEYS["iteration"]: str(iteration),
            ONNX_IDENTITY_KEYS["model_protocol_version"]: str(
                model_protocol_version
            ),
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
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
        raise ValueError(f"ONNX external data path is unsafe: {location!r}")
    validate_safe_filename(relative.name, allowed_suffixes=(".onnx.data",))

    model_root = Path(model_dir).resolve()
    resolved = (model_root / relative.name).resolve()
    if os.path.commonpath((str(model_root), str(resolved))) != str(model_root):
        raise ValueError(f"ONNX external data escapes model directory: {location!r}")
    return resolved


def _read_artifact_manifest(marker_path):
    """读取并校验当前版本产物清单的基础结构"""
    marker_path = Path(marker_path)
    validate_safe_filename(marker_path.name, allowed_suffixes=(".artifacts.json",))
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"artifact manifest must be a regular file: {marker_path.name}")
    if marker_path.stat().st_size > MAX_MANIFEST_FILE_BYTES:
        raise ValueError(f"artifact manifest is too large: {marker_path.name}")
    with open(marker_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact manifest must be an object: {marker_path.name}")
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
    model_protocol_version = payload.get("model_protocol_version")
    if (
        isinstance(model_protocol_version, bool)
        or model_protocol_version != MODEL_PROTOCOL_VERSION
    ):
        raise ValueError(f"model protocol version mismatch: {marker_path.name}")
    validate_model_id(payload.get("model_id"))
    validate_model_prefix(payload.get("model_prefix"))
    iteration = payload.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"artifact iteration is invalid: {marker_path.name}")
    checkpoint = payload.get("checkpoint") or {}
    checkpoint_name = checkpoint.get("file")
    if checkpoint_name is not None:
        validate_safe_filename(checkpoint_name, allowed_suffixes=(".pth",))
    onnx_record = payload.get("onnx") or {}
    for filename in onnx_record.get("files") or []:
        validate_model_artifact_filename(filename)
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
    for key in ("model_id", "model_prefix", "iteration", "model_protocol_version"):
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
    model_protocol_version=None,
):
    """读取 ONNX 主图引用并返回主图与全部外置权重的完整记录"""
    import onnx

    source_graph_path = Path(onnx_path)
    if source_graph_path.is_symlink():
        raise ValueError("symbolic-link ONNX graphs are not allowed")
    graph_path = source_graph_path.resolve()
    validate_safe_filename(graph_path.name, allowed_suffixes=(".onnx",))
    if graph_path.is_symlink() or not graph_path.is_file():
        raise FileNotFoundError(f"ONNX graph does not exist: {graph_path}")
    if graph_path.stat().st_size > MAX_ONNX_GRAPH_FILE_BYTES:
        raise ValueError("ONNX graph exceeds the 2 GiB safety limit")

    model = onnx.load(str(graph_path), load_external_data=False)
    properties = {item.key: item.value for item in model.metadata_props}
    embedded_model_id = properties.get(ONNX_IDENTITY_KEYS["model_id"])
    embedded_model_prefix = properties.get(ONNX_IDENTITY_KEYS["model_prefix"])
    embedded_iteration = properties.get(ONNX_IDENTITY_KEYS["iteration"])
    embedded_model_protocol = properties.get(
        ONNX_IDENTITY_KEYS["model_protocol_version"]
    )
    if embedded_iteration is not None:
        try:
            embedded_iteration = int(embedded_iteration)
        except ValueError as error:
            raise ValueError("ONNX embedded iteration must be an integer") from error
    if embedded_model_protocol is not None:
        try:
            embedded_model_protocol = int(embedded_model_protocol)
        except ValueError as error:
            raise ValueError(
                "ONNX embedded model_protocol_version must be an integer"
            ) from error

    embedded_identity = (
        embedded_model_id,
        embedded_model_prefix,
        embedded_iteration,
        embedded_model_protocol,
    )
    if any(value is not None for value in embedded_identity):
        if any(value is None for value in embedded_identity):
            raise ValueError("ONNX embedded model identity is incomplete")
        validate_model_id(embedded_model_id)
        validate_model_prefix(embedded_model_prefix)
        if embedded_model_protocol != MODEL_PROTOCOL_VERSION:
            raise ValueError(
                "ONNX embedded model protocol does not match the current protocol"
            )
    if expected_model_id is not None and embedded_model_id != expected_model_id:
        raise PermissionError(
            f"ONNX embedded model_id mismatch: expected {expected_model_id}, "
            f"found {embedded_model_id}"
        )
    for name, requested, embedded in (
        ("model_id", model_id, embedded_model_id),
        ("model_prefix", model_prefix, embedded_model_prefix),
        ("iteration", iteration, embedded_iteration),
        (
            "model_protocol_version",
            model_protocol_version,
            embedded_model_protocol,
        ),
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
        if data_path.is_symlink():
            raise ValueError(
                f"ONNX external data must not be a symbolic link: {location}"
            )
        if require_complete and not data_path.is_file():
            raise FileNotFoundError(
                f"ONNX external data is missing for {graph_path.name}: {location}"
            )
        if data_path.is_file():
            validate_model_artifact_file(data_path)
        files.append(os.path.relpath(data_path, graph_path.parent).replace("\\", "/"))

    record = {
        "format": "onnx",
        "model_id": embedded_model_id,
        "model_prefix": embedded_model_prefix,
        "iteration": embedded_iteration,
        "model_protocol_version": embedded_model_protocol,
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
                    "model_protocol_version": payload.get(
                        "model_protocol_version"
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
                if (
                    metadata["format_warning"] is not None
                    or metadata["model_protocol_warning"] is not None
                ):
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
                        "model_protocol_version": metadata[
                            "model_protocol_version"
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


def assert_model_artifact_target_identity(model_dir, record):
    """导入覆盖前核验目标主文件的 UUID、前缀、轮次和格式完全一致"""
    validate_package_model_records([record], expected_model_id=record["model_id"])
    model_root = Path(model_dir).resolve()
    primary_name = validate_safe_filename(
        record["primary"], allowed_suffixes=(".pth", ".onnx")
    )
    target_primary = model_root / primary_name
    existing_files = [
        model_root / validate_model_artifact_filename(filename)
        for filename in record.get("files", [])
        if (model_root / filename).exists()
    ]
    target_marker = checkpoint_artifact_manifest_path(target_primary)
    if target_marker.exists() and target_marker not in existing_files:
        existing_files.append(target_marker)
    if not target_primary.exists():
        if existing_files:
            raise PermissionError(
                f"refusing to overwrite orphan target artifact: {existing_files[0].name}"
            )
        return
    existing = build_package_model_records(model_root, [primary_name])[0]
    for key in ("format", "model_id", "model_prefix", "iteration"):
        if existing.get(key) != record.get(key):
            raise PermissionError(
                f"refusing to overwrite {primary_name}: existing {key} differs"
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
        if not is_primary_model_filename(selected):
            raise ValueError(f"invalid model artifact selection: {selected!r}")
        name = validate_safe_filename(selected, allowed_suffixes=(".pth", ".onnx"))
        primary_path = model_root / name
        if primary_path.is_symlink() or not primary_path.is_file():
            raise FileNotFoundError(f"selected model does not exist: {primary_path}")

        if name.casefold().endswith(".onnx"):
            record = describe_onnx_artifact(primary_path, require_complete=True)
            for relative_name in record["files"]:
                append_once(relative_name)
        else:
            append_once(name)

        marker = f"{primary_path.stem}.artifacts.json"
        if (model_root / marker).is_file():
            append_once(marker)

    validate_model_artifact_file_set(model_root, collected)
    return collected


def build_package_model_records(model_dir, selected_models):
    """为部署包清单生成带轮次和依赖文件的模型记录"""
    model_root = Path(model_dir).resolve()
    records = []
    for selected in selected_models:
        if not is_primary_model_filename(selected):
            raise ValueError(f"invalid model artifact selection: {selected!r}")
        name = validate_safe_filename(selected, allowed_suffixes=(".pth", ".onnx"))
        primary_path = model_root / name
        if primary_path.is_symlink() or not primary_path.is_file():
            raise FileNotFoundError(f"selected model does not exist: {primary_path}")
        if name.casefold().endswith(".onnx"):
            records.append(describe_onnx_artifact(primary_path, require_complete=True))
        else:
            checkpoint = validate_training_checkpoint_file(
                primary_path,
                map_location="cpu",
            )
            marker_path = checkpoint_artifact_manifest_path(primary_path)
            if marker_path.is_file():
                payload = _read_artifact_manifest(marker_path)
                for key in (
                    "model_id",
                    "model_prefix",
                    "iteration",
                    "model_protocol_version",
                ):
                    if payload[key] != checkpoint[key]:
                        raise ValueError(
                            f"checkpoint internal {key} does not match "
                            f"{marker_path.name}"
                        )
                checkpoint_record = payload.get("checkpoint") or {}
                if (
                    checkpoint_record.get("status") != "complete"
                    or checkpoint_record.get("file") != name
                ):
                    raise ValueError(
                        f"checkpoint artifact record is invalid: {marker_path.name}"
                    )
            records.append(
                {
                    "format": "pytorch_checkpoint",
                    "model_id": checkpoint["model_id"],
                    "model_prefix": checkpoint["model_prefix"],
                    "iteration": checkpoint["iteration"],
                    "model_protocol_version": checkpoint[
                        "model_protocol_version"
                    ],
                    "primary": name,
                    "files": [name],
                    "status": "complete",
                }
            )
    return records


def get_model_iteration_mismatch(records):
    """比较同一选择中的 PTH 与 ONNX 轮次集合，并返回可展示的提示"""
    checkpoint_iterations = {
        record["iteration"]
        for record in records
        if record.get("format") == "pytorch_checkpoint"
    }
    onnx_iterations = {
        record["iteration"]
        for record in records
        if record.get("format") == "onnx"
    }
    if checkpoint_iterations and onnx_iterations and checkpoint_iterations != onnx_iterations:
        return (
            "PTH 与 ONNX 的内置轮次不一致: "
            f"PTH={sorted(checkpoint_iterations)}, ONNX={sorted(onnx_iterations)}"
        )
    return None


def validate_package_model_records(records, *, expected_model_id=None):
    """校验部署包只能包含同一 UUID 的模型，且双格式轮次必须成对一致"""
    if not isinstance(records, list):
        raise ValueError("model artifact records must be a list")
    if not records:
        return records
    model_ids = {record.get("model_id") for record in records}
    prefixes = {record.get("model_prefix") for record in records}
    model_protocol_versions = {
        record.get("model_protocol_version") for record in records
    }
    if None in model_ids or None in prefixes or None in model_protocol_versions:
        raise ValueError("model artifact identity is incomplete")
    for model_id in model_ids:
        validate_model_id(model_id)
    for prefix in prefixes:
        validate_model_prefix(prefix)
    if len(model_ids) > 1:
        raise ValueError("one deployment package cannot mix different model_id pools")
    if len(prefixes) > 1:
        raise ValueError("one model_id pool cannot contain different model prefixes")
    if model_protocol_versions != {MODEL_PROTOCOL_VERSION}:
        raise ValueError("deployment package model protocol version is incompatible")
    if expected_model_id is not None and model_ids and model_ids != {expected_model_id}:
        raise PermissionError("selected models do not belong to the requested model_id pool")
    artifact_keys = [
        (record.get("format"), record.get("iteration")) for record in records
    ]
    if len(artifact_keys) != len(set(artifact_keys)):
        raise ValueError("one deployment package cannot contain duplicate format/iteration artifacts")
    mismatch = get_model_iteration_mismatch(records)
    if mismatch:
        raise ValueError(mismatch)
    return records


def discover_model_repository(model_dir):
    """按模型 UUID 和内置轮次整理模型仓库，并列出无法归档的产物"""
    model_root = Path(model_dir).resolve()
    result = {"pools": {}, "invalid": []}
    if not model_root.is_dir():
        return result

    recognized_files = sorted(
        path
        for path in model_root.iterdir()
        if path.is_file()
        and any(path.name.casefold().endswith(suffix) for suffix in MODEL_ARTIFACT_SUFFIXES)
    )
    referenced_files = set()
    primary_paths = [path for path in recognized_files if is_primary_model_filename(path.name)]
    for primary_path in primary_paths:
        try:
            if primary_path.suffix.casefold() == ".pth":
                checkpoint = validate_training_checkpoint_file(
                    primary_path,
                    map_location="cpu",
                )
                marker_path = checkpoint_artifact_manifest_path(primary_path)
                if marker_path.is_file():
                    payload = _read_artifact_manifest(marker_path)
                    for key in (
                        "model_id",
                        "model_prefix",
                        "iteration",
                        "model_protocol_version",
                    ):
                        if payload[key] != checkpoint[key]:
                            raise ValueError(
                                f"checkpoint internal {key} does not match "
                                f"{marker_path.name}"
                            )
                    checkpoint_record = payload.get("checkpoint") or {}
                    if (
                        checkpoint_record.get("status") != "complete"
                        or checkpoint_record.get("file") != primary_path.name
                    ):
                        raise ValueError("checkpoint manifest does not name this PTH")
                    record = {
                        "format": "pytorch_checkpoint",
                        "model_id": checkpoint["model_id"],
                        "model_prefix": checkpoint["model_prefix"],
                        "iteration": checkpoint["iteration"],
                        "model_protocol_version": checkpoint[
                            "model_protocol_version"
                        ],
                        "primary": primary_path.name,
                        "files": [primary_path.name, marker_path.name],
                        "identity_source": "checkpoint",
                        "status": "complete",
                    }
                else:
                    record = {
                        "format": "pytorch_checkpoint",
                        "model_id": checkpoint["model_id"],
                        "model_prefix": checkpoint["model_prefix"],
                        "iteration": checkpoint["iteration"],
                        "model_protocol_version": checkpoint[
                            "model_protocol_version"
                        ],
                        "primary": primary_path.name,
                        "files": [primary_path.name],
                        "identity_source": "checkpoint",
                        "status": "complete",
                    }
            else:
                record = describe_onnx_artifact(
                    primary_path,
                    require_complete=True,
                    validate_manifest=checkpoint_artifact_manifest_path(primary_path).is_file(),
                )
                if record["model_id"] is None:
                    raise ValueError("ONNX embedded model identity is missing")
                record["identity_source"] = "onnx_metadata"
                marker_path = checkpoint_artifact_manifest_path(primary_path)
                if marker_path.is_file():
                    record["files"] = [*record["files"], marker_path.name]

            model_id = record["model_id"]
            iteration = record["iteration"]
            pool = result["pools"].setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "prefixes": set(),
                    "iterations": {},
                    "artifacts": [],
                },
            )
            pool["prefixes"].add(record["model_prefix"])
            pool["iterations"].setdefault(iteration, []).append(record)
            pool["artifacts"].append(record)
            referenced_files.update(record["files"])
        except Exception as error:
            result["invalid"].append(
                {"file": primary_path.name, "error": str(error)}
            )

    invalid_names = {item["file"] for item in result["invalid"]}
    for path in recognized_files:
        if path.name not in referenced_files and path.name not in invalid_names:
            result["invalid"].append(
                {"file": path.name, "error": "orphan or unverified model artifact"}
            )
    for pool in result["pools"].values():
        pool["prefixes"] = sorted(pool["prefixes"], key=str.casefold)
        pool["artifacts"].sort(key=lambda item: (item["iteration"], item["format"], item["primary"]))
    result["invalid"].sort(key=lambda item: item["file"].casefold())
    return result


def install_model_artifact_bundle(
    source_dir,
    target_dir,
    selected_models=None,
    *,
    expected_model_id=None,
    require_all_artifacts=False,
):
    """验证暂存模型制品的身份与依赖后，以临时文件原子安装到模型仓库"""
    source_root = Path(source_dir).resolve()
    target_root = Path(target_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"model artifact source does not exist: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    if selected_models is None:
        selected_models = sorted(
            path.name for path in source_root.iterdir() if is_primary_model_filename(path.name)
        )
    selected_models = list(dict.fromkeys(selected_models))
    records = build_package_model_records(source_root, selected_models)
    if expected_model_id is not None:
        validate_package_model_records(records, expected_model_id=expected_model_id)
    prefixes_by_model_id = {}
    for record in records:
        validate_package_model_records([record])
        prefixes_by_model_id.setdefault(record["model_id"], set()).add(
            record["model_prefix"]
        )
    if any(len(prefixes) != 1 for prefixes in prefixes_by_model_id.values()):
        raise ValueError("one model_id cannot be imported with different model prefixes")

    files = collect_model_artifact_files(source_root, selected_models)
    validate_model_artifact_file_set(source_root, files)
    for record in records:
        assert_model_artifact_target_identity(target_root, record)

    if require_all_artifacts:
        staged_artifacts = {
            path.name
            for path in source_root.iterdir()
            if path.is_file()
            and any(
                path.name.casefold().endswith(suffix)
                for suffix in MODEL_ARTIFACT_SUFFIXES
            )
        }
        unclaimed = staged_artifacts.difference(files)
        if unclaimed:
            raise ValueError(
                "unreferenced model artifacts are not allowed: "
                f"{sorted(unclaimed)}"
            )

    installed = []
    for filename in files:
        safe_name = validate_model_artifact_filename(filename)
        source_path = source_root / safe_name
        validate_model_artifact_file(source_path)
        target_path = target_root / safe_name
        temporary_path = None
        try:
            with open(source_path, "rb") as source_stream, tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target_path.name}.",
                suffix=".import.tmp",
                dir=target_root,
                delete=False,
            ) as temporary_stream:
                temporary_path = Path(temporary_stream.name)
                shutil.copyfileobj(source_stream, temporary_stream, length=1024 * 1024)
            os.replace(temporary_path, target_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        installed.append(safe_name)
    return {"records": records, "files": installed}


def write_checkpoint_artifact_manifest(
    checkpoint_path,
    iteration,
    *,
    model_id,
    model_prefix,
    checkpoint_format_version=CHECKPOINT_FORMAT_VERSION,
    model_protocol_version=MODEL_PROTOCOL_VERSION,
    onnx_record=None,
    onnx_error=None,
):
    """在保存检查点时写入同轮次产物清单，并标记 ONNX 是否完整"""
    validate_model_id(model_id)
    validate_model_prefix(model_prefix)
    if checkpoint_format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("artifact checkpoint format version does not match the current format")
    if model_protocol_version != MODEL_PROTOCOL_VERSION:
        raise ValueError("artifact model protocol version does not match the current protocol")
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
                "model_protocol_version": int(model_protocol_version),
            }
        )
    payload = {
        "artifact_manifest_version": ARTIFACT_MANIFEST_FORMAT_VERSION,
        "checkpoint_format_version": int(checkpoint_format_version),
        "model_protocol_version": int(model_protocol_version),
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


def create_deployment_package(
    target_path,
    model_dir,
    selected_models,
    *,
    package_name,
    extra_files=None,
):
    """校验模型池和附加文件后，原子生成带强制清单的部署包"""
    validate_package_name(package_name)
    target = Path(target_path)
    validate_deployment_package_filename(target.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink():
            raise ValueError("deployment package target must not be a symbolic link")
        raise FileExistsError(f"deployment package already exists: {target.name}")

    selected_models = list(dict.fromkeys(selected_models))
    records = build_package_model_records(model_dir, selected_models)
    validate_package_model_records(records)
    model_files = collect_model_artifact_files(model_dir, selected_models)
    model_total_size = validate_model_artifact_file_set(model_dir, model_files)
    extras = {}
    for archive_name, source_path in (extra_files or {}).items():
        if archive_name not in DEPLOY_ROOT_FILES:
            raise ValueError(f"unsupported deployment root file: {archive_name!r}")
        raw_source = Path(source_path)
        if raw_source.is_symlink():
            raise ValueError(f"deployment root file must not be a symlink: {archive_name}")
        source = raw_source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"deployment root file does not exist: {source}")
        extra_size = source.stat().st_size
        if extra_size > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
            raise ValueError(f"deployment root file exceeds 32 GiB: {archive_name}")
        model_total_size += extra_size
        if model_total_size > MAX_MODEL_ARTIFACT_TOTAL_BYTES:
            raise ValueError("deployment package inputs exceed the 64 GiB safety limit")
        extras[archive_name] = source

    included_code_semantics = CODE_SEMANTIC_FILE_SET.intersection(extras)
    if included_code_semantics and included_code_semantics != CODE_SEMANTIC_FILE_SET:
        raise ValueError("code semantic vectors and their index must be packaged together")
    includes_knowledge_base = KNOWLEDGE_BASE_FILENAME in extras
    includes_complete_code_semantics = included_code_semantics == CODE_SEMANTIC_FILE_SET
    if includes_knowledge_base is not includes_complete_code_semantics:
        raise ValueError(
            "knowledge_base.json, code_embeddings.npy and "
            "code_embeddings_idx.json must be packaged as one complete runtime semantic bundle"
        )
    if included_code_semantics:
        semantic_root = extras[CODE_EMBEDDINGS_FILENAME].parent
        validated_semantics = validate_semantic_bundle(semantic_root)
        if (
            validated_semantics["embedding_path"]
            != extras[CODE_EMBEDDINGS_FILENAME]
            or validated_semantics["index_path"]
            != extras[CODE_EMBEDDINGS_INDEX_FILENAME]
            or validated_semantics["knowledge_base_path"]
            != extras[KNOWLEDGE_BASE_FILENAME]
        ):
            raise ValueError("semantic assets must come from one coherent directory")
    if HASH_MAPPING_FILENAME in extras and KNOWLEDGE_BASE_FILENAME not in extras:
        raise ValueError("hash mapping requires knowledge_base.json")

    manifest = {
        "package_format_version": DEPLOY_PACKAGE_FORMAT_VERSION,
        "package_name": package_name,
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_included": selected_models,
        "model_artifacts": records,
        "model_files_included": model_files,
        "includes_kb": includes_knowledge_base,
        "includes_staples": "meta_staples.json" in extras,
        "includes_hash_mapping": HASH_MAPPING_FILENAME in extras,
        "includes_code_semantics": includes_complete_code_semantics,
    }

    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{target.stem}.",
        suffix=".gkg.tmp",
        dir=target.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for filename in model_files:
                safe_name = validate_model_artifact_filename(filename)
                archive.write(
                    Path(model_dir) / safe_name,
                    arcname=safe_name,
                    compress_type=zipfile.ZIP_STORED,
                )
            for archive_name, source in extras.items():
                archive.write(source, arcname=archive_name)
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
        if temporary_path.stat().st_size > MAX_DEPLOY_PACKAGE_FILE_BYTES:
            raise ValueError("deployment package exceeds the 64 GiB safety limit")
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest


def validate_deployment_package(stage_dir):
    """核验解包目录、部署协议清单、模型身份与实际文件集合完全一致"""
    raw_stage_root = Path(stage_dir)
    if raw_stage_root.is_symlink():
        raise ValueError("deployment stage must not be a symbolic link")
    stage_root = raw_stage_root.resolve()
    if not stage_root.is_dir():
        raise FileNotFoundError(f"deployment stage does not exist: {stage_root}")
    staged_paths = sorted(stage_root.iterdir())
    for path in staged_paths:
        validate_safe_filename(path.name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"deployment member must be a regular file: {path.name}")

    manifest_path = stage_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("deployment package manifest.json is required")
    if manifest_path.stat().st_size > MAX_MANIFEST_FILE_BYTES:
        raise ValueError("deployment package manifest is too large")
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("deployment package manifest must be an object")
    if manifest.get("package_format_version") != DEPLOY_PACKAGE_FORMAT_VERSION:
        raise ValueError(
            "deployment package format version mismatch: "
            f"file={manifest.get('package_format_version')!r}, "
            f"current={DEPLOY_PACKAGE_FORMAT_VERSION}"
        )
    validate_package_name(manifest.get("package_name"))

    actual_names = {path.name for path in staged_paths}
    allowed_names = {"manifest.json", *DEPLOY_ROOT_FILES}
    for name in actual_names.difference(allowed_names):
        validate_model_artifact_filename(name)
    primary_models = sorted(name for name in actual_names if is_primary_model_filename(name))
    declared_models = manifest.get("models_included")
    if not isinstance(declared_models, list) or len(declared_models) != len(set(declared_models)):
        raise ValueError("manifest models_included must be a unique list")
    for name in declared_models:
        if not is_primary_model_filename(name):
            raise ValueError(f"manifest contains an invalid primary model: {name!r}")
    if set(declared_models) != set(primary_models):
        raise ValueError("manifest primary model list does not match package contents")

    actual_records = build_package_model_records(stage_root, declared_models)
    validate_package_model_records(actual_records)
    declared_records = manifest.get("model_artifacts")
    if not isinstance(declared_records, list) or declared_records != actual_records:
        raise ValueError("manifest model identity records do not match embedded metadata")
    actual_model_files = collect_model_artifact_files(stage_root, declared_models)
    declared_model_files = manifest.get("model_files_included")
    if (
        not isinstance(declared_model_files, list)
        or len(declared_model_files) != len(set(declared_model_files))
        or set(declared_model_files) != set(actual_model_files)
    ):
        raise ValueError("manifest model file list does not match package contents")

    expected_names = {"manifest.json", *actual_model_files}
    for filename, flag in (
        (KNOWLEDGE_BASE_FILENAME, "includes_kb"),
        ("meta_staples.json", "includes_staples"),
        (HASH_MAPPING_FILENAME, "includes_hash_mapping"),
    ):
        included = filename in actual_names
        if manifest.get(flag) is not included:
            raise ValueError(f"manifest {flag} does not match package contents")
        if included:
            expected_names.add(filename)
    actual_code_semantics = CODE_SEMANTIC_FILE_SET.intersection(actual_names)
    includes_code_semantics = manifest.get("includes_code_semantics")
    if actual_code_semantics and actual_code_semantics != CODE_SEMANTIC_FILE_SET:
        raise ValueError("deployment package contains an incomplete code semantic pair")
    if includes_code_semantics is not (actual_code_semantics == CODE_SEMANTIC_FILE_SET):
        raise ValueError("manifest includes_code_semantics does not match package contents")
    has_knowledge_base = KNOWLEDGE_BASE_FILENAME in actual_names
    has_complete_code_semantics = actual_code_semantics == CODE_SEMANTIC_FILE_SET
    if has_knowledge_base is not has_complete_code_semantics:
        raise ValueError(
            "deployment package must contain knowledge_base.json, code_embeddings.npy and "
            "code_embeddings_idx.json as one complete runtime semantic bundle"
        )
    if actual_code_semantics:
        validate_semantic_bundle(stage_root)
        expected_names.update(CODE_SEMANTIC_FILE_SET)
    if HASH_MAPPING_FILENAME in actual_names and KNOWLEDGE_BASE_FILENAME not in actual_names:
        raise ValueError("hash mapping requires knowledge_base.json")
    if actual_names != expected_names:
        raise ValueError(
            f"deployment package contains undeclared files: {sorted(actual_names - expected_names)}"
        )
    return {"manifest": manifest, "records": actual_records, "files": actual_model_files}


def safe_extract_zip(archive, target_dir):
    """限制 ZIP 路径、类型、数量、展开体积和压缩比后流式解压"""
    raw_target = Path(target_dir)
    if raw_target.exists() and raw_target.is_symlink():
        raise ValueError("deployment extraction target must not be a symbolic link")
    target_root = raw_target.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    archive_filename = getattr(archive, "filename", None)
    if isinstance(archive_filename, (str, os.PathLike)):
        archive_path = Path(archive_filename)
        if archive_path.is_symlink() or not archive_path.is_file():
            raise ValueError("deployment package must be a regular file")
        if archive_path.stat().st_size > MAX_DEPLOY_PACKAGE_FILE_BYTES:
            raise ValueError("deployment package exceeds the 64 GiB safety limit")
    members = archive.infolist()
    if not members or len(members) > MAX_DEPLOY_ARCHIVE_MEMBERS:
        raise ValueError("deployment archive member count is outside the safety limit")

    total_size = 0
    seen_names = set()
    for info in members:
        if info.is_dir() or "/" in info.filename or "\\" in info.filename:
            raise ValueError(f"unsafe archive member path: {info.filename!r}")
        validate_safe_filename(info.filename)
        folded = info.filename.casefold()
        if folded in seen_names:
            raise ValueError(f"duplicate archive member is not allowed: {info.filename!r}")
        seen_names.add(folded)
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted archive member is not allowed: {info.filename!r}")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError(f"unsupported archive compression: {info.filename!r}")
        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if stat.S_ISLNK(unix_mode) or file_type not in {0, stat.S_IFREG}:
            raise ValueError(f"archive special files are not allowed: {info.filename!r}")
        if info.file_size < 0 or info.file_size > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
            raise ValueError(f"archive member is too large: {info.filename!r}")
        total_size += info.file_size
        if total_size > MAX_DEPLOY_ARCHIVE_TOTAL_BYTES:
            raise ValueError("deployment archive exceeds the 64 GiB expanded-size limit")
        if info.file_size > 1024 * 1024:
            if info.compress_size <= 0 or info.file_size / info.compress_size > MAX_DEPLOY_COMPRESSION_RATIO:
                raise ValueError(f"archive compression ratio is unsafe: {info.filename!r}")
        destination = target_root / info.filename
        if destination.exists():
            raise FileExistsError(f"archive target already exists: {info.filename!r}")

    created_files = []
    try:
        for info in members:
            destination = target_root / info.filename
            written = 0
            with archive.open(info, "r") as source, open(destination, "xb") as output:
                created_files.append(destination)
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size or written > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
                        raise ValueError(f"archive member expanded beyond its limit: {info.filename!r}")
                    output.write(chunk)
            if written != info.file_size:
                raise ValueError(f"archive member size mismatch: {info.filename!r}")
    except Exception:
        for path in reversed(created_files):
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return [info.filename for info in members]
