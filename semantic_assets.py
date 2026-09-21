# 本文件统一校验、同步和安装语义知识库及代码语义向量资产

import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from effect_slot_binding import build_runtime_effect_binding_catalog


KNOWLEDGE_BASE_FILENAME = "knowledge_base.json"
HASH_MAPPING_FILENAME = "hash_mapping_report.json"
CODE_EMBEDDINGS_FILENAME = "code_embeddings.npy"
CODE_EMBEDDINGS_INDEX_FILENAME = "code_embeddings_idx.json"
STATIC_SEMANTIC_TABLE_FILENAME = "semantic_lookup_v1.npz"
STATIC_SEMANTIC_CATALOG_FILENAME = "semantic_lookup_v1.json"
STATIC_SEMANTIC_ASSET_FILENAMES = (
    STATIC_SEMANTIC_TABLE_FILENAME,
    STATIC_SEMANTIC_CATALOG_FILENAME,
)
CODE_SEMANTIC_FILENAMES = (
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
)
SEMANTIC_ASSET_FILENAMES = (
    KNOWLEDGE_BASE_FILENAME,
    HASH_MAPPING_FILENAME,
    *CODE_SEMANTIC_FILENAMES,
)
DEFAULT_SEMANTIC_REPOSITORY_URL = "https://github.com/Noctfom/Galatea-Core.git"
MAX_CODE_EMBEDDINGS_BYTES = 2 * 1024 * 1024 * 1024
MAX_CODE_EMBEDDING_INDEX_BYTES = 256 * 1024 * 1024
MAX_KNOWLEDGE_BASE_BYTES = 512 * 1024 * 1024


def invalidate_static_semantic_assets(target_directory):
    """删除可重建的静态语义编译资产，避免源文件更新后继续使用旧表"""
    root = Path(target_directory).resolve()
    removed = []
    for filename in STATIC_SEMANTIC_ASSET_FILENAMES:
        path = root / filename
        if path.is_symlink():
            raise ValueError(f"static semantic asset must not be a symlink: {path}")
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def semantic_sibling_url(base_url, filename):
    """把仓库或旧式 Raw 地址解析为指定语义资产的下载地址"""
    source = str(base_url or "").strip()
    parts = urllib.parse.urlsplit(source)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("semantic repository URL must use HTTP or HTTPS")
    if parts.username or parts.password:
        raise ValueError("semantic repository URL must not contain credentials")

    path_parts = [part for part in parts.path.split("/") if part]
    if parts.netloc.lower() == "github.com" and len(path_parts) >= 2:
        owner = path_parts[0]
        repository = path_parts[1]
        if repository.endswith(".git"):
            repository = repository[:-4]
        branch = "main"
        subdirectory = []
        if len(path_parts) >= 4 and path_parts[2] in {"tree", "blob"}:
            branch = path_parts[3]
            subdirectory = path_parts[4:]
            if path_parts[2] == "blob" and subdirectory:
                subdirectory = subdirectory[:-1]
        asset_path = "/".join([owner, repository, branch, *subdirectory, filename])
        return f"https://raw.githubusercontent.com/{asset_path}"

    known_filenames = set(SEMANTIC_ASSET_FILENAMES)
    if path_parts and path_parts[-1] in known_filenames:
        path_parts = path_parts[:-1]
    path = "/" + "/".join([*path_parts, filename])
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
    )


def clear_local_semantic_assets(
    target_directory,
    *,
    knowledge_base_filename=KNOWLEDGE_BASE_FILENAME,
):
    """物理删除本地语义资产，供明确选择的全量重建流程使用"""
    root = Path(target_directory).resolve()
    filenames = {
        knowledge_base_filename,
        HASH_MAPPING_FILENAME,
        *CODE_SEMANTIC_FILENAMES,
    }
    removed = []
    for filename in filenames:
        path = root / filename
        if path.is_symlink():
            raise ValueError(f"semantic asset must not be a symbolic link: {path}")
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    removed.extend(invalidate_static_semantic_assets(root))
    return removed


def _rebuild_hash_mapping(knowledge_base):
    """从结构化语义中的自定义标签重建可接续的 Hash 映射"""
    mapping = {}
    for card_id, card_data in knowledge_base.items():
        if not isinstance(card_data, dict):
            continue
        for fallback_slot, effect in enumerate(card_data.get("effects", []), start=1):
            if not isinstance(effect, dict):
                continue
            slot = int(effect.get("slot", fallback_slot) or fallback_slot)
            card_label = f"{card_id}_E{slot}"
            for category in effect.get("categories", []):
                category = str(category)
                if not category.startswith("CUSTOM_HASH_"):
                    continue
                record = mapping.setdefault(
                    category,
                    {"cards": [], "sample_code": ""},
                )
                if card_label not in record["cards"]:
                    record["cards"].append(card_label)
    return mapping


def _write_json_atomically(payload, destination):
    """把 JSON 资产写入同目录临时文件后原子替换"""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".sync.tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_json_object(path, *, max_bytes, label):
    """在大小限制内读取 JSON 对象，拒绝数组或其他顶层结构"""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{label} exceeds the safety size limit")
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def validate_code_semantic_assets(directory, *, required=False):
    """校验代码语义向量与索引必须成对存在且映射完整"""
    root = Path(directory).resolve()
    embedding_path = root / CODE_EMBEDDINGS_FILENAME
    index_path = root / CODE_EMBEDDINGS_INDEX_FILENAME
    existing = (embedding_path.is_file(), index_path.is_file())
    if not any(existing):
        if required:
            raise FileNotFoundError("code semantic embedding assets are missing")
        return None
    if not all(existing):
        raise ValueError("code_embeddings.npy and code_embeddings_idx.json must appear together")
    if embedding_path.is_symlink() or index_path.is_symlink():
        raise ValueError("code semantic assets must not be symbolic links")
    if embedding_path.stat().st_size > MAX_CODE_EMBEDDINGS_BYTES:
        raise ValueError("code_embeddings.npy exceeds the 2 GiB safety limit")

    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    try:
        shape = tuple(embeddings.shape)
        dtype = embeddings.dtype
        if embeddings.ndim != 2:
            raise ValueError("code_embeddings.npy must be a two-dimensional matrix")
        if dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("code_embeddings.npy must use float16 or float32")
        if shape[0] > 1_000_000 or shape[1] > 4096:
            raise ValueError("code_embeddings.npy shape exceeds semantic safety limits")
    finally:
        mmap_handle = getattr(embeddings, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    index = _load_json_object(
        index_path,
        max_bytes=MAX_CODE_EMBEDDING_INDEX_BYTES,
        label=CODE_EMBEDDINGS_INDEX_FILENAME,
    )
    values = []
    for key, value in index.items():
        if not isinstance(key, str) or not key:
            raise ValueError("code embedding index keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("code embedding index values must be integers")
        values.append(value)
    if len(index) != shape[0]:
        raise ValueError("code embedding row count does not match its index")
    if sorted(values) != list(range(shape[0])):
        raise ValueError("code embedding index must map exactly onto every matrix row")
    return {
        "shape": shape,
        "dtype": dtype,
        "index": index,
        "embedding_path": embedding_path,
        "index_path": index_path,
    }


def build_expected_code_semantic_keys(knowledge_base):
    """从结构化知识库生成模型实际使用的卡号与效果槽键集合"""
    if not isinstance(knowledge_base, dict):
        raise ValueError("knowledge_base.json must be a JSON object")
    expected_keys = set()
    for raw_card_id, card_data in knowledge_base.items():
        card_id = str(raw_card_id)
        if not card_id.isdigit() or not isinstance(card_data, dict):
            raise ValueError(f"invalid knowledge-base card record: {raw_card_id!r}")
        effects = card_data.get("effects", [])
        if not isinstance(effects, list):
            raise ValueError(f"knowledge-base effects must be a list: {card_id}")
        seen_slots = set()
        for fallback_slot, effect in enumerate(effects, start=1):
            if not isinstance(effect, dict):
                raise ValueError(f"knowledge-base effect must be an object: {card_id}")
            raw_slot = effect.get("slot", fallback_slot)
            if isinstance(raw_slot, bool):
                raise ValueError(f"invalid semantic slot for card {card_id}: {raw_slot!r}")
            try:
                slot = int(raw_slot)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid semantic slot for card {card_id}: {raw_slot!r}"
                ) from error
            if slot < 1:
                raise ValueError(f"semantic slot must be positive for card {card_id}")
            if slot > 8:
                continue
            if slot in seen_slots:
                raise ValueError(f"duplicate semantic slot {slot} for card {card_id}")
            seen_slots.add(slot)
            expected_keys.add(f"{card_id}_{slot - 1}")
    return expected_keys


def validate_semantic_bundle(
    directory,
    *,
    knowledge_base_filename=KNOWLEDGE_BASE_FILENAME,
    knowledge_base=None,
):
    """交叉校验知识库效果槽与代码向量索引必须完整且无过期项"""
    root = Path(directory).resolve()
    knowledge_base_path = root / knowledge_base_filename
    if knowledge_base is None:
        knowledge_base = _load_json_object(
            knowledge_base_path,
            max_bytes=MAX_KNOWLEDGE_BASE_BYTES,
            label=knowledge_base_filename,
        )
    elif not isinstance(knowledge_base, dict):
        raise ValueError("knowledge base must be a JSON object")

    code_assets = validate_code_semantic_assets(root, required=True)
    expected_keys = build_expected_code_semantic_keys(knowledge_base)
    # 运行时效果标识必须与 Lua 语义槽保持一对一，禁止歧义资产进入训练
    runtime_effect_bindings = build_runtime_effect_binding_catalog(knowledge_base)
    actual_keys = set(code_assets["index"])
    missing_keys = sorted(expected_keys.difference(actual_keys))
    stale_keys = sorted(actual_keys.difference(expected_keys))
    if missing_keys or stale_keys:
        details = []
        if missing_keys:
            details.append(
                f"missing={len(missing_keys)} sample={missing_keys[:5]}"
            )
        if stale_keys:
            details.append(
                f"stale={len(stale_keys)} sample={stale_keys[:5]}"
            )
        raise ValueError(
            "knowledge base and code semantic index do not match: "
            + "; ".join(details)
        )
    return {
        "knowledge_base": knowledge_base,
        "knowledge_base_path": knowledge_base_path,
        "effect_slot_count": len(expected_keys),
        "runtime_effect_binding_count": len(runtime_effect_bindings),
        **code_assets,
    }


def _download_to_path(url, target_path):
    """把远程资产流式下载到指定临时路径"""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with open(target_path, "wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)


def _replace_file_atomically(source, destination):
    """在目标目录生成临时副本后原子替换单个语义文件"""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".sync.tmp",
            dir=destination.parent,
            delete=False,
        ) as target_stream:
            temporary_path = Path(target_stream.name)
            with open(source, "rb") as source_stream:
                while True:
                    chunk = source_stream.read(1024 * 1024)
                    if not chunk:
                        break
                    target_stream.write(chunk)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def download_remote_semantic_bundle(remote_kb_url, target_directory):
    """下载同一远程目录的完整语义基座，并安装校验通过的向量对"""
    target_root = Path(target_directory).resolve()
    with tempfile.TemporaryDirectory(prefix="galatea_semantic_sync_") as temp_dir:
        temp_root = Path(temp_dir)
        downloaded = {}
        errors = {}
        for filename in SEMANTIC_ASSET_FILENAMES:
            target = temp_root / filename
            try:
                _download_to_path(
                    semantic_sibling_url(remote_kb_url, filename),
                    target,
                )
                downloaded[filename] = target
            except Exception as error:
                errors[filename] = str(error)

        if KNOWLEDGE_BASE_FILENAME not in downloaded:
            raise RuntimeError(
                "remote knowledge base download failed: "
                + errors.get(KNOWLEDGE_BASE_FILENAME, "unknown error")
            )
        knowledge_base = _load_json_object(
            downloaded[KNOWLEDGE_BASE_FILENAME],
            max_bytes=MAX_CODE_EMBEDDINGS_BYTES,
            label=KNOWLEDGE_BASE_FILENAME,
        )

        hash_mapping = None
        if HASH_MAPPING_FILENAME in downloaded:
            hash_mapping = _load_json_object(
                downloaded[HASH_MAPPING_FILENAME],
                max_bytes=MAX_CODE_EMBEDDING_INDEX_BYTES,
                label=HASH_MAPPING_FILENAME,
            )

        vector_names = set(CODE_SEMANTIC_FILENAMES)
        downloaded_vectors = vector_names.intersection(downloaded)
        if downloaded_vectors and downloaded_vectors != vector_names:
            for filename in downloaded_vectors:
                downloaded[filename].unlink(missing_ok=True)
            downloaded_vectors.clear()
            errors["code_semantic_pair"] = (
                "remote code semantic vectors were incomplete and were not installed"
            )
        if downloaded_vectors == vector_names:
            try:
                validate_semantic_bundle(
                    temp_root,
                    knowledge_base=knowledge_base,
                )
            except (OSError, ValueError) as error:
                downloaded_vectors.clear()
                errors["code_semantic_pair"] = str(error)
            else:
                for filename in CODE_SEMANTIC_FILENAMES:
                    _replace_file_atomically(
                        downloaded[filename],
                        target_root / filename,
                    )

    return {
        "knowledge_base": knowledge_base,
        "hash_mapping": hash_mapping,
        "installed_code_semantics": downloaded_vectors == vector_names,
        "errors": errors,
    }


def synchronize_remote_semantic_bundle(
    remote_source,
    target_directory,
    *,
    knowledge_base_filename=KNOWLEDGE_BASE_FILENAME,
):
    """仅同步远程语义资产，不扫描本地 Lua，也不生成新向量"""
    target_root = Path(target_directory).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    remote_kb_url = semantic_sibling_url(
        remote_source,
        KNOWLEDGE_BASE_FILENAME,
    )
    bundle = download_remote_semantic_bundle(remote_kb_url, target_root)
    hash_mapping = bundle["hash_mapping"]
    rebuilt_hash_mapping = hash_mapping is None
    if rebuilt_hash_mapping:
        hash_mapping = _rebuild_hash_mapping(bundle["knowledge_base"])

    _write_json_atomically(
        bundle["knowledge_base"],
        target_root / knowledge_base_filename,
    )
    _write_json_atomically(
        hash_mapping,
        target_root / HASH_MAPPING_FILENAME,
    )

    if not bundle["installed_code_semantics"]:
        # 远程结构语义变化后不能继续沿用旧向量，避免静默错位。
        for filename in CODE_SEMANTIC_FILENAMES:
            path = target_root / filename
            if path.is_file() and not path.is_symlink():
                path.unlink()

    invalidate_static_semantic_assets(target_root)

    return {
        **bundle,
        "remote_knowledge_base_url": remote_kb_url,
        "rebuilt_hash_mapping": rebuilt_hash_mapping,
    }
