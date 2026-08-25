# 本文件统一管理检查点、ONNX 外置权重及部署包中的模型产物

import json
import os
import re
import stat
from pathlib import Path


_ITERATION_PATTERN = re.compile(r"^galatea_iter_(\d+)")


def parse_model_iteration(filename):
    """从标准模型文件名中提取训练轮数，无法识别时返回空值"""
    match = _ITERATION_PATTERN.match(Path(filename).name)
    return int(match.group(1)) if match else None


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


def _validate_onnx_artifact_manifest(graph_path, record):
    """若同轮次产物清单存在，确认其中的 ONNX 状态和文件记录完整"""
    marker_path = graph_path.with_name(f"{graph_path.stem}.artifacts.json")
    if not marker_path.is_file():
        return

    with open(marker_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("iteration") != record["iteration"]:
        raise ValueError(f"artifact manifest iteration mismatch: {marker_path.name}")

    onnx_record = payload.get("onnx") or {}
    if onnx_record.get("status") != "complete":
        raise RuntimeError(
            f"ONNX artifact is not marked complete in {marker_path.name}: "
            f"{onnx_record.get('status', 'unknown')}"
        )
    if onnx_record.get("files") != record["files"]:
        raise ValueError(f"ONNX artifact file list mismatch: {marker_path.name}")


def describe_onnx_artifact(
    onnx_path,
    *,
    require_complete=True,
    validate_manifest=True,
):
    """读取 ONNX 主图引用并返回主图与全部外置权重的完整记录"""
    import onnx

    graph_path = Path(onnx_path).resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(f"ONNX graph does not exist: {graph_path}")

    model = onnx.load(str(graph_path), load_external_data=False)
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
        "iteration": parse_model_iteration(graph_path.name),
        "primary": graph_path.name,
        "files": files,
        "external_data": files[1:],
        "status": "complete",
    }
    if validate_manifest:
        _validate_onnx_artifact_manifest(graph_path, record)
    return record


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

        iteration = parse_model_iteration(name)
        if iteration is not None:
            marker = f"galatea_iter_{iteration}.artifacts.json"
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
            records.append(
                {
                    "format": "pytorch_checkpoint",
                    "iteration": parse_model_iteration(name),
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
    onnx_record=None,
    onnx_error=None,
):
    """在保存检查点时写入同轮次产物清单，并标记 ONNX 是否完整"""
    checkpoint = Path(checkpoint_path).resolve()
    manifest_path = checkpoint_artifact_manifest_path(checkpoint)
    if onnx_error == "export_in_progress":
        onnx_status = "in_progress"
    elif onnx_error:
        onnx_status = "failed"
    else:
        onnx_status = "disabled"
    payload = {
        "schema_version": 1,
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
