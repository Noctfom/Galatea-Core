# 本文件统一校验、同步和安装语义知识库及代码语义向量资产

import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


KNOWLEDGE_BASE_FILENAME = "knowledge_base.json"
HASH_MAPPING_FILENAME = "hash_mapping_report.json"
CODE_EMBEDDINGS_FILENAME = "code_embeddings.npy"
CODE_EMBEDDINGS_INDEX_FILENAME = "code_embeddings_idx.json"
CODE_SEMANTIC_FILENAMES = (
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
)
SEMANTIC_ASSET_FILENAMES = (
    KNOWLEDGE_BASE_FILENAME,
    HASH_MAPPING_FILENAME,
    *CODE_SEMANTIC_FILENAMES,
)
MAX_CODE_EMBEDDINGS_BYTES = 2 * 1024 * 1024 * 1024
MAX_CODE_EMBEDDING_INDEX_BYTES = 256 * 1024 * 1024


def semantic_sibling_url(base_url, filename):
    """把知识库 URL 的文件名替换为同目录下的其他语义资产名"""
    parts = urllib.parse.urlsplit(base_url)
    parent = parts.path.rsplit("/", 1)[0]
    path = f"{parent}/{filename}" if parent else f"/{filename}"
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
    )


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
            validate_code_semantic_assets(temp_root, required=True)
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
