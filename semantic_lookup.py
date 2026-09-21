# -*- coding: utf-8 -*-
# 本文件把 Lua 静态语义编译为按精确卡片词表索引的模型侧查表，并维护可校验的语义身份

import hashlib
import json
import os
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np

from card_vocab import FIRST_CARD_TOKEN_ID, get_default_card_vocabulary
from effect_slot_binding import build_runtime_effect_binding_catalog
from semantic_assets import (
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
    KNOWLEDGE_BASE_FILENAME,
    STATIC_SEMANTIC_ASSET_FILENAMES,
    STATIC_SEMANTIC_CATALOG_FILENAME,
    STATIC_SEMANTIC_TABLE_FILENAME,
    validate_semantic_bundle,
)


STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION = 1
SEMANTIC_EFFECT_SLOTS = 8
SEMANTIC_CATEGORY_SLOTS = 8
SEMANTIC_REQUIREMENT_SLOTS = 16
SEMANTIC_RELATION_SLOTS = 4
SEMANTIC_REQUIREMENT_CAPACITY = 128
SEMANTIC_CATEGORY_CAPACITY = 4000
MAX_STATIC_SEMANTIC_TABLE_BYTES = 1024 * 1024 * 1024
MAX_STATIC_SEMANTIC_CATALOG_BYTES = 16 * 1024 * 1024

RACE_MAP = {
    "RACE_WARRIOR": 0x1,
    "RACE_SPELLCASTER": 0x2,
    "RACE_FAIRY": 0x4,
    "RACE_FIEND": 0x8,
    "RACE_ZOMBIE": 0x10,
    "RACE_MACHINE": 0x20,
    "RACE_AQUA": 0x40,
    "RACE_PYRO": 0x80,
    "RACE_ROCK": 0x100,
    "RACE_WINDBEAST": 0x200,
    "RACE_PLANT": 0x400,
    "RACE_INSECT": 0x800,
    "RACE_THUNDER": 0x1000,
    "RACE_DRAGON": 0x2000,
    "RACE_BEAST": 0x4000,
    "RACE_BEASTWARRIOR": 0x8000,
    "RACE_DINOSAUR": 0x10000,
    "RACE_FISH": 0x20000,
    "RACE_SEASERPENT": 0x40000,
    "RACE_REPTILE": 0x80000,
    "RACE_PSYCHO": 0x100000,
    "RACE_DEVINE": 0x200000,
    "RACE_CREATORGOD": 0x400000,
    "RACE_WYRM": 0x800000,
    "RACE_CYBERSE": 0x1000000,
    "RACE_ILLUSION": 0x2000000,
}
ATTR_MAP = {
    "ATTRIBUTE_EARTH": 0x01,
    "ATTRIBUTE_WATER": 0x02,
    "ATTRIBUTE_FIRE": 0x04,
    "ATTRIBUTE_WIND": 0x08,
    "ATTRIBUTE_LIGHT": 0x10,
    "ATTRIBUTE_DARK": 0x20,
    "ATTRIBUTE_DEVINE": 0x40,
}
REQUIREMENT_GROUPS = (
    "locations",
    "phases",
    "types",
    "summon_types",
    "reasons",
    "positions",
)


def _file_signature(path):
    """读取语义资产变化签名，使在线同步后自动失效旧查表缓存"""
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


def _canonical_array_bytes(array):
    """把数值数组转换为跨平台稳定的小端连续字节"""
    dtype = array.dtype
    if dtype.byteorder not in ("|", "<"):
        dtype = dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")


def _sha256_file(path):
    """流式计算编译资产摘要，避免校验大表时制造额外整文件副本"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compiled_asset_paths(directory):
    """返回静态语义数值表与轻量运行目录的固定路径"""
    root = Path(directory).resolve()
    return (
        root / STATIC_SEMANTIC_TABLE_FILENAME,
        root / STATIC_SEMANTIC_CATALOG_FILENAME,
    )


def _serialize_runtime_bindings(bindings):
    """把二元组键目录转成不依赖 pickle 的稳定 JSON 三元组。"""
    return [
        [int(card_code), int(runtime_desc), int(slot_index)]
        for (card_code, runtime_desc), slot_index in sorted(bindings.items())
    ]


def _deserialize_runtime_bindings(records):
    """严格还原运行时效果绑定，并拒绝重复键或越界槽位。"""
    if not isinstance(records, list):
        raise ValueError("static semantic runtime bindings must be a list")
    bindings = {}
    for record in records:
        if not isinstance(record, list) or len(record) != 3:
            raise ValueError("static semantic runtime binding must be a triple")
        card_code, runtime_desc, slot_index = record
        if any(isinstance(value, bool) or not isinstance(value, int) for value in record):
            raise ValueError("static semantic runtime binding values must be integers")
        if not 0 < card_code <= 0x7FFFFFFF or not 0 < runtime_desc <= 0xFFFFFFFF:
            raise ValueError("static semantic runtime binding identity is invalid")
        if not 0 <= slot_index < SEMANTIC_EFFECT_SLOTS:
            raise ValueError("static semantic runtime binding slot is invalid")
        key = (card_code, runtime_desc)
        if key in bindings:
            raise ValueError("static semantic runtime binding contains a duplicate key")
        bindings[key] = slot_index
    return bindings


def _deserialize_card_slots(records):
    """严格还原卡片到有效 Lua 效果槽的轻量审计目录"""
    if not isinstance(records, dict):
        raise ValueError("static semantic card slots must be an object")
    card_slots = {}
    for raw_card_code, raw_slots in records.items():
        if not str(raw_card_code).isdigit() or not isinstance(raw_slots, list):
            raise ValueError("static semantic card slot record is invalid")
        card_code = int(raw_card_code)
        slots = tuple(int(slot) for slot in raw_slots)
        if (
            card_code <= 0
            or any(isinstance(slot, bool) for slot in raw_slots)
            or tuple(sorted(set(slots))) != slots
            or any(not 0 <= slot < SEMANTIC_EFFECT_SLOTS for slot in slots)
        ):
            raise ValueError("static semantic card slot values are invalid")
        card_slots[card_code] = slots
    return card_slots


def _read_compiled_catalog(directory, card_vocabulary):
    """读取并校验轻量目录身份，不加载约 53 MiB 的模型侧数值表"""
    table_path, catalog_path = _compiled_asset_paths(directory)
    for path, size_limit in (
        (table_path, MAX_STATIC_SEMANTIC_TABLE_BYTES),
        (catalog_path, MAX_STATIC_SEMANTIC_CATALOG_BYTES),
    ):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"static semantic asset is missing: {path.name}")
        if path.stat().st_size <= 0 or path.stat().st_size > size_limit:
            raise ValueError(f"static semantic asset size is invalid: {path.name}")
    with open(catalog_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("static semantic catalog must be an object")
    if payload.get("format_version") != STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION:
        raise ValueError("static semantic catalog format version mismatch")
    if payload.get("table_file") != STATIC_SEMANTIC_TABLE_FILENAME:
        raise ValueError("static semantic catalog table filename mismatch")
    if payload.get("table_size") != table_path.stat().st_size:
        raise ValueError("static semantic table size does not match its catalog")
    if payload.get("card_vocab_hash") != card_vocabulary.vocabulary_hash:
        raise ValueError("static semantic catalog card vocabulary hash mismatch")
    if payload.get("card_vocab_size") != card_vocabulary.capacity:
        raise ValueError("static semantic catalog card vocabulary capacity mismatch")
    if payload.get("card_vocab_card_count") != card_vocabulary.card_count:
        raise ValueError("static semantic catalog card count mismatch")
    semantic_card_slots = _deserialize_card_slots(
        payload.get("semantic_card_slots")
    )
    runtime_effect_bindings = _deserialize_runtime_bindings(
        payload.get("runtime_effect_bindings")
    )
    return payload, semantic_card_slots, runtime_effect_bindings


def _validate_static_npz_container(table_path, card_vocabulary):
    """限制 NPZ 内层成员与展开体积，拒绝嵌套压缩炸弹和额外载荷"""
    expected_names = {f"{name}.npy" for name in _COMPILED_ARRAY_NAMES}
    with zipfile.ZipFile(table_path, "r") as archive:
        members = archive.infolist()
        actual_names = {item.filename for item in members}
        if len(members) != len(expected_names) or actual_names != expected_names:
            raise ValueError("static semantic NPZ members are invalid")
        if any(
            item.is_dir()
            or "/" in item.filename
            or "\\" in item.filename
            or item.compress_type != zipfile.ZIP_STORED
            or item.flag_bits & 0x1
            for item in members
        ):
            raise ValueError("static semantic NPZ member encoding is unsafe")
        max_payload_bytes = (
            64 * 1024
            + card_vocabulary.capacity
            * SEMANTIC_EFFECT_SLOTS
            * (
                SEMANTIC_CATEGORY_SLOTS * np.dtype(np.int16).itemsize
                + SEMANTIC_REQUIREMENT_SLOTS * np.dtype(np.int8).itemsize
                + SEMANTIC_RELATION_SLOTS
                * (
                    np.dtype(np.int16).itemsize * 3
                    + np.dtype(np.float16).itemsize
                    + np.dtype(np.int32).itemsize
                )
                + np.dtype(np.int32).itemsize
                + np.dtype(np.bool_).itemsize
            )
            + (card_vocabulary.capacity * SEMANTIC_EFFECT_SLOTS + 1)
            * 512
            * np.dtype(np.float32).itemsize
        )
        if sum(item.file_size for item in members) > max_payload_bytes:
            raise ValueError("static semantic NPZ expanded size exceeds protocol bounds")


_REGISTERED_CATALOGS = None


def _register_lookup_catalogs(semantic_card_slots, runtime_effect_bindings):
    """登记小型效果槽目录，确保 spawn Worker 与主进程看到同一语义"""
    global _REGISTERED_CATALOGS
    if (
        _REGISTERED_CATALOGS is not None
        and _REGISTERED_CATALOGS[0] is semantic_card_slots
        and _REGISTERED_CATALOGS[1] is runtime_effect_bindings
    ):
        return
    from effect_slot_binding import register_runtime_effect_binding_catalog
    from protocol_v3_audit import register_compiled_semantic_audit_catalog

    register_runtime_effect_binding_catalog(runtime_effect_bindings)
    register_compiled_semantic_audit_catalog(
        semantic_card_slots,
        runtime_effect_bindings,
    )
    _REGISTERED_CATALOGS = (semantic_card_slots, runtime_effect_bindings)


class StaticSemanticLookup:
    """保存按卡片 token 对齐的只读静态语义矩阵"""

    def __init__(
        self,
        *,
        card_vocabulary,
        category_to_index,
        requirement_to_index,
        category,
        requirement,
        setcode,
        number,
        reference,
        race,
        attribute,
        code_index,
        effect_mask,
        code_dictionary,
        semantic_card_slots,
        runtime_effect_bindings,
    ):
        self.card_vocabulary = card_vocabulary
        self.category_to_index = dict(category_to_index)
        self.requirement_to_index = dict(requirement_to_index)
        self.category = category
        self.requirement = requirement
        self.setcode = setcode
        self.number = number
        self.reference = reference
        self.race = race
        self.attribute = attribute
        self.code_index = code_index
        self.effect_mask = effect_mask
        self.code_dictionary = code_dictionary
        self.semantic_card_slots = semantic_card_slots
        self.runtime_effect_bindings = runtime_effect_bindings
        self._prefix_hashes = {}

    @property
    def table_arrays(self):
        """返回参与模型查表和语义身份计算的固定顺序数组"""
        return (
            self.category,
            self.requirement,
            self.setcode,
            self.number,
            self.reference,
            self.race,
            self.attribute,
            self.code_index,
            self.effect_mask,
        )

    def get_card_semantics(self, card_id):
        """按真实卡密返回旧编码器兼容的八组静态语义视图"""
        token_id = self.card_vocabulary.encode(card_id)
        return (
            self.category[token_id],
            self.requirement[token_id],
            self.setcode[token_id],
            self.number[token_id],
            self.reference[token_id],
            self.race[token_id],
            self.attribute[token_id],
            self.code_index[token_id],
        )

    def prefix_hash(self, card_count=None):
        """计算指定只追加词表前缀实际可见的逻辑语义指纹"""
        if card_count is None:
            card_count = self.card_vocabulary.card_count
        if isinstance(card_count, bool) or not isinstance(card_count, int):
            raise ValueError("semantic lookup card count must be an integer")
        if not 0 <= card_count <= self.card_vocabulary.card_count:
            raise ValueError("semantic lookup card count is outside the vocabulary")
        cached = self._prefix_hashes.get(card_count)
        if cached is not None:
            return cached

        prefix_vocabulary = self.card_vocabulary.prefix(card_count)
        token_limit = FIRST_CARD_TOKEN_ID + card_count
        used_categories = set(
            int(value)
            for value in np.unique(self.category[:token_limit])
            if int(value) > 0
        )
        used_requirements = set(
            int(value)
            for value in np.unique(self.requirement[:token_limit])
            if int(value) >= 0
        )
        category_labels = {
            label: index
            for label, index in self.category_to_index.items()
            if index in used_categories
        }
        requirement_labels = {
            label: index
            for label, index in self.requirement_to_index.items()
            if index in used_requirements
        }
        header = {
            "format_version": STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION,
            "card_count": card_count,
            "card_mapping": prefix_vocabulary.cards,
            "category_labels": category_labels,
            "requirement_labels": requirement_labels,
            "effect_slots": SEMANTIC_EFFECT_SLOTS,
            "code_dimension": int(self.code_dictionary.shape[1]),
        }
        digest = hashlib.sha256(
            json.dumps(
                header,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for array in self.table_arrays[:-2]:
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(_canonical_array_bytes(array[:token_limit]))
        digest.update(_canonical_array_bytes(self.effect_mask[:token_limit]))

        # code_index 的物理行号可在增量生成时变化；身份按实际槽位向量计算。
        for start in range(0, token_limit, 64):
            indices = self.code_index[start : min(start + 64, token_limit)]
            vectors = self.code_dictionary[indices]
            digest.update(_canonical_array_bytes(vectors))

        semantic_hash = digest.hexdigest()
        self._prefix_hashes[card_count] = semantic_hash
        return semantic_hash


def _build_index_vocabulary(knowledge_base):
    """按现有知识库稳定顺序生成分类与条件的离散编号"""
    category_to_index = {"<PAD>": 0, "<UNK>": 1}
    requirement_to_index = {}
    for card_data in knowledge_base.values():
        for effect in card_data.get("effects", []):
            for category in effect.get("categories", []):
                if category not in category_to_index:
                    category_to_index[category] = len(category_to_index)
            requirements = effect.get("requirements", {})
            for group in REQUIREMENT_GROUPS:
                for item in requirements.get(group, []):
                    if item not in requirement_to_index:
                        requirement_to_index[item] = len(requirement_to_index)
    if len(category_to_index) > SEMANTIC_CATEGORY_CAPACITY:
        raise ValueError("semantic category vocabulary exceeds network capacity")
    if len(requirement_to_index) > SEMANTIC_REQUIREMENT_CAPACITY:
        raise ValueError("semantic requirement vocabulary exceeds network capacity")
    return category_to_index, requirement_to_index


def _allocate_lookup_arrays(capacity):
    """一次性分配按卡片 token 对齐的紧凑静态语义数组"""
    return {
        "category": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_CATEGORY_SLOTS),
            dtype=np.int16,
        ),
        "requirement": np.full(
            (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_REQUIREMENT_SLOTS),
            -1,
            dtype=np.int8,
        ),
        "setcode": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
            dtype=np.int16,
        ),
        "number": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
            dtype=np.float16,
        ),
        "reference": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
            dtype=np.int32,
        ),
        "race": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
            dtype=np.int16,
        ),
        "attribute": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
            dtype=np.int16,
        ),
        "code_index": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS), dtype=np.int32
        ),
        "effect_mask": np.zeros(
            (capacity, SEMANTIC_EFFECT_SLOTS), dtype=np.bool_
        ),
    }


def _populate_card_semantics(
    arrays,
    token_id,
    card_id,
    card_data,
    *,
    card_vocabulary,
    category_to_index,
    requirement_to_index,
    code_embedding_index,
):
    """把单卡 Lua 效果写入其精确 token 行"""
    semantic_slots = []
    for fallback_slot, effect in enumerate(card_data.get("effects", []), start=1):
        try:
            slot = int(effect.get("slot", fallback_slot)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= slot < SEMANTIC_EFFECT_SLOTS:
            continue
        arrays["effect_mask"][token_id, slot] = True
        semantic_slots.append(slot)

        for index, category in enumerate(
            effect.get("categories", [])[:SEMANTIC_CATEGORY_SLOTS]
        ):
            arrays["category"][token_id, slot, index] = category_to_index.get(
                category, 1
            )

        requirements = effect.get("requirements", {})
        requirement_slot = 0
        for group in REQUIREMENT_GROUPS:
            for item in requirements.get(group, []):
                requirement_index = requirement_to_index.get(item)
                if (
                    requirement_index is not None
                    and requirement_index < SEMANTIC_REQUIREMENT_CAPACITY
                    and requirement_slot < SEMANTIC_REQUIREMENT_SLOTS
                ):
                    arrays["requirement"][
                        token_id, slot, requirement_slot
                    ] = requirement_index
                    requirement_slot += 1

        for index, raw_setcode in enumerate(
            requirements.get("setcodes", [])[:SEMANTIC_RELATION_SLOTS]
        ):
            try:
                text = str(raw_setcode)
                value = int(text, 16) if text.startswith("0x") else int(text)
                arrays["setcode"][token_id, slot, index] = value % 4096
            except (TypeError, ValueError):
                continue
        for index, race in enumerate(
            requirements.get("races", [])[:SEMANTIC_RELATION_SLOTS]
        ):
            if race in RACE_MAP:
                arrays["race"][token_id, slot, index] = RACE_MAP[race] % 30
        for index, attribute in enumerate(
            requirements.get("attributes", [])[:SEMANTIC_RELATION_SLOTS]
        ):
            if attribute in ATTR_MAP:
                arrays["attribute"][token_id, slot, index] = (
                    ATTR_MAP[attribute] % 10
                )

        number_slot = 0
        reference_slot = 0
        for raw_number in requirements.get("custom_numbers", []):
            try:
                value = float(raw_number)
            except (TypeError, ValueError):
                continue
            if value > 10000 and reference_slot < SEMANTIC_RELATION_SLOTS:
                arrays["reference"][token_id, slot, reference_slot] = (
                    card_vocabulary.encode(int(value))
                )
                reference_slot += 1
            elif number_slot < SEMANTIC_RELATION_SLOTS:
                arrays["number"][token_id, slot, number_slot] = value / 4000.0
                number_slot += 1

        code_key = f"{card_id}_{slot}"
        code_row = code_embedding_index.get(code_key)
        if code_row is None:
            raise ValueError(f"missing code semantic row for {code_key}")
        arrays["code_index"][token_id, slot] = int(code_row) + 1
    return tuple(sorted(set(semantic_slots)))


def build_static_semantic_lookup(
    directory=".",
    *,
    card_vocabulary=None,
    knowledge_base_filename=KNOWLEDGE_BASE_FILENAME,
):
    """校验完整语义资产并编译为网络可直接注册的稠密查表"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    root = Path(directory).resolve()
    validated = validate_semantic_bundle(
        root,
        knowledge_base_filename=knowledge_base_filename,
    )
    knowledge_base = validated["knowledge_base"]
    category_to_index, requirement_to_index = _build_index_vocabulary(
        knowledge_base
    )
    arrays = _allocate_lookup_arrays(vocabulary.capacity)
    semantic_card_slots = {}
    for card_id, token_id in vocabulary.cards.items():
        card_data = knowledge_base.get(str(card_id))
        if card_data is None:
            continue
        semantic_card_slots[card_id] = _populate_card_semantics(
            arrays,
            token_id,
            card_id,
            card_data,
            card_vocabulary=vocabulary,
            category_to_index=category_to_index,
            requirement_to_index=requirement_to_index,
            code_embedding_index=validated["index"],
        )

    code_embeddings = np.load(
        validated["embedding_path"], allow_pickle=False
    ).astype(np.float32, copy=False)
    code_dictionary = np.zeros(
        (code_embeddings.shape[0] + 1, code_embeddings.shape[1]),
        dtype=np.float32,
    )
    code_dictionary[1:] = code_embeddings
    return StaticSemanticLookup(
        card_vocabulary=vocabulary,
        category_to_index=category_to_index,
        requirement_to_index=requirement_to_index,
        category=arrays["category"],
        requirement=arrays["requirement"],
        setcode=arrays["setcode"],
        number=arrays["number"],
        reference=arrays["reference"],
        race=arrays["race"],
        attribute=arrays["attribute"],
        code_index=arrays["code_index"],
        effect_mask=arrays["effect_mask"],
        code_dictionary=code_dictionary,
        semantic_card_slots=semantic_card_slots,
        runtime_effect_bindings=build_runtime_effect_binding_catalog(
            knowledge_base
        ),
    )


_COMPILED_ARRAY_NAMES = (
    "category",
    "requirement",
    "setcode",
    "number",
    "reference",
    "race",
    "attribute",
    "code_index",
    "effect_mask",
    "code_dictionary",
)


def _lookup_array_mapping(lookup):
    """返回写入独立数值资产的固定数组集合"""
    return {
        "category": lookup.category,
        "requirement": lookup.requirement,
        "setcode": lookup.setcode,
        "number": lookup.number,
        "reference": lookup.reference,
        "race": lookup.race,
        "attribute": lookup.attribute,
        "code_index": lookup.code_index,
        "effect_mask": lookup.effect_mask,
        "code_dictionary": lookup.code_dictionary,
    }


def write_static_semantic_assets(
    source_directory=".",
    *,
    output_directory=None,
    card_vocabulary=None,
    knowledge_base_filename=KNOWLEDGE_BASE_FILENAME,
    lookup=None,
):
    """把已编译语义表原子写成安全 NPZ 与轻量运行目录"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    source_root = Path(source_directory).resolve()
    output_root = Path(output_directory or source_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    compiled = lookup or build_static_semantic_lookup(
        source_root,
        card_vocabulary=vocabulary,
        knowledge_base_filename=knowledge_base_filename,
    )
    if compiled.card_vocabulary.vocabulary_hash != vocabulary.vocabulary_hash:
        raise ValueError("compiled semantic lookup vocabulary mismatch")

    table_path, catalog_path = _compiled_asset_paths(output_root)
    temporary_table = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{table_path.name}.",
            suffix=".compile.tmp",
            dir=output_root,
            delete=False,
        ) as stream:
            temporary_table = Path(stream.name)
            np.savez(stream, **_lookup_array_mapping(compiled))
        os.replace(temporary_table, table_path)
        temporary_table = None
    finally:
        if temporary_table is not None and temporary_table.exists():
            temporary_table.unlink()

    source_records = {}
    for filename in (
        knowledge_base_filename,
        CODE_EMBEDDINGS_FILENAME,
        CODE_EMBEDDINGS_INDEX_FILENAME,
    ):
        source_path = source_root / filename
        stat = source_path.stat()
        source_records[filename] = {
            "size": stat.st_size,
            "sha256": _sha256_file(source_path),
        }
    arrays = _lookup_array_mapping(compiled)
    catalog = {
        "format_version": STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION,
        "table_file": STATIC_SEMANTIC_TABLE_FILENAME,
        "table_size": table_path.stat().st_size,
        "table_sha256": _sha256_file(table_path),
        "card_vocab_hash": vocabulary.vocabulary_hash,
        "card_vocab_size": vocabulary.capacity,
        "card_vocab_card_count": vocabulary.card_count,
        "semantic_lookup_hash": compiled.prefix_hash(),
        "semantic_lookup_card_count": vocabulary.card_count,
        "category_to_index": compiled.category_to_index,
        "requirement_to_index": compiled.requirement_to_index,
        "arrays": {
            name: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
            for name, array in arrays.items()
        },
        "semantic_card_slots": {
            str(card_code): list(slots)
            for card_code, slots in sorted(compiled.semantic_card_slots.items())
        },
        "runtime_effect_bindings": _serialize_runtime_bindings(
            compiled.runtime_effect_bindings
        ),
        "source_files": source_records,
    }
    temporary_catalog = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{catalog_path.name}.",
            suffix=".compile.tmp",
            dir=output_root,
            delete=False,
        ) as stream:
            temporary_catalog = Path(stream.name)
            json.dump(catalog, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary_catalog, catalog_path)
        temporary_catalog = None
    finally:
        if temporary_catalog is not None and temporary_catalog.exists():
            temporary_catalog.unlink()
    return compiled, {
        STATIC_SEMANTIC_TABLE_FILENAME: table_path,
        STATIC_SEMANTIC_CATALOG_FILENAME: catalog_path,
    }


def load_static_semantic_assets(
    directory=".",
    *,
    card_vocabulary=None,
    verify_source_files=False,
    verify_logical_hash=False,
):
    """从独立资产恢复模型查表，并验证数值表、词表和逻辑语义身份"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    root = Path(directory).resolve()
    table_path, _ = _compiled_asset_paths(root)
    catalog, semantic_card_slots, runtime_effect_bindings = _read_compiled_catalog(
        root,
        vocabulary,
    )
    if catalog.get("table_sha256") != _sha256_file(table_path):
        raise ValueError("static semantic table hash does not match its catalog")
    if verify_source_files:
        source_records = catalog.get("source_files")
        if not isinstance(source_records, dict):
            raise ValueError("static semantic source records are missing")
        source_names = set(source_records)
        if (
            len(source_names) != 3
            or CODE_EMBEDDINGS_FILENAME not in source_names
            or CODE_EMBEDDINGS_INDEX_FILENAME not in source_names
            or any(
                not isinstance(filename, str)
                or filename != Path(filename).name
                or filename in {"", ".", ".."}
                for filename in source_names
            )
        ):
            raise ValueError("static semantic source filename set is invalid")
        for filename, record in source_records.items():
            source_path = root / filename
            if source_path.is_symlink() or not source_path.is_file():
                raise FileNotFoundError(
                    f"static semantic source is missing: {filename}"
                )
            if (
                not isinstance(record, dict)
                or record.get("size") != source_path.stat().st_size
                or record.get("sha256") != _sha256_file(source_path)
            ):
                raise ValueError(
                    "semantic_lookup_hash source mismatch with compiled asset: "
                    f"{filename}"
                )

    expected_arrays = catalog.get("arrays")
    if not isinstance(expected_arrays, dict):
        raise ValueError("static semantic array manifest is missing")
    _validate_static_npz_container(table_path, vocabulary)
    with np.load(table_path, allow_pickle=False) as archive:
        if set(archive.files) != set(_COMPILED_ARRAY_NAMES):
            raise ValueError("static semantic table contains unexpected arrays")
        arrays = {}
        for name in _COMPILED_ARRAY_NAMES:
            array = np.asarray(archive[name])
            record = expected_arrays.get(name)
            if not isinstance(record, dict):
                raise ValueError(f"static semantic array record is missing: {name}")
            if record.get("dtype") != array.dtype.str or record.get("shape") != list(
                array.shape
            ):
                raise ValueError(f"static semantic array shape mismatch: {name}")
            if array.dtype.hasobject:
                raise ValueError(f"static semantic object array is not allowed: {name}")
            arrays[name] = np.array(array, copy=True)

    capacity = vocabulary.capacity
    expected_shapes = {
        "category": (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_CATEGORY_SLOTS),
        "requirement": (
            capacity,
            SEMANTIC_EFFECT_SLOTS,
            SEMANTIC_REQUIREMENT_SLOTS,
        ),
        "setcode": (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
        "number": (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
        "reference": (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
        "race": (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
        "attribute": (capacity, SEMANTIC_EFFECT_SLOTS, SEMANTIC_RELATION_SLOTS),
        "code_index": (capacity, SEMANTIC_EFFECT_SLOTS),
        "effect_mask": (capacity, SEMANTIC_EFFECT_SLOTS),
    }
    expected_dtypes = {
        "category": np.dtype(np.int16),
        "requirement": np.dtype(np.int8),
        "setcode": np.dtype(np.int16),
        "number": np.dtype(np.float16),
        "reference": np.dtype(np.int32),
        "race": np.dtype(np.int16),
        "attribute": np.dtype(np.int16),
        "code_index": np.dtype(np.int32),
        "effect_mask": np.dtype(np.bool_),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or arrays[name].dtype != expected_dtypes[name]:
            raise ValueError(f"static semantic array contract mismatch: {name}")
    code_dictionary = arrays["code_dictionary"]
    if (
        code_dictionary.ndim != 2
        or code_dictionary.dtype != np.float32
        or not 1 <= code_dictionary.shape[1] <= 512
        or code_dictionary.shape[0]
        > capacity * SEMANTIC_EFFECT_SLOTS + 1
    ):
        raise ValueError("static semantic code dictionary is invalid")
    if arrays["code_index"].size and (
        arrays["code_index"].min() < 0
        or arrays["code_index"].max() >= code_dictionary.shape[0]
    ):
        raise ValueError("static semantic code index exceeds its dictionary")

    category_to_index = catalog.get("category_to_index")
    requirement_to_index = catalog.get("requirement_to_index")
    if not isinstance(category_to_index, dict) or not isinstance(
        requirement_to_index, dict
    ):
        raise ValueError("static semantic label vocabularies are invalid")
    for mapping, lower_bound, upper_bound, label in (
        (
            category_to_index,
            0,
            SEMANTIC_CATEGORY_CAPACITY,
            "category",
        ),
        (
            requirement_to_index,
            0,
            SEMANTIC_REQUIREMENT_CAPACITY,
            "requirement",
        ),
    ):
        values = list(mapping.values())
        if (
            any(not isinstance(key, str) or not key for key in mapping)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not lower_bound <= value < upper_bound
                for value in values
            )
            or len(values) != len(set(values))
        ):
            raise ValueError(f"static semantic {label} vocabulary is invalid")
    if category_to_index.get("<PAD>") != 0 or category_to_index.get("<UNK>") != 1:
        raise ValueError("static semantic category reserved indices are invalid")
    bounded_arrays = (
        (arrays["category"], 0, SEMANTIC_CATEGORY_CAPACITY, "category"),
        (arrays["requirement"], -1, SEMANTIC_REQUIREMENT_CAPACITY, "requirement"),
        (arrays["setcode"], 0, 4096, "setcode"),
        (arrays["reference"], 0, capacity, "reference"),
        (arrays["race"], 0, 30, "race"),
        (arrays["attribute"], 0, 10, "attribute"),
    )
    for array, lower_bound, upper_bound, label in bounded_arrays:
        if array.size and (
            array.min() < lower_bound or array.max() >= upper_bound
        ):
            raise ValueError(f"static semantic {label} values are outside bounds")
    if not np.isfinite(arrays["number"]).all() or not np.isfinite(
        code_dictionary
    ).all():
        raise ValueError("static semantic floating-point values must be finite")
    lookup = StaticSemanticLookup(
        card_vocabulary=vocabulary,
        category_to_index=category_to_index,
        requirement_to_index=requirement_to_index,
        category=arrays["category"],
        requirement=arrays["requirement"],
        setcode=arrays["setcode"],
        number=arrays["number"],
        reference=arrays["reference"],
        race=arrays["race"],
        attribute=arrays["attribute"],
        code_index=arrays["code_index"],
        effect_mask=arrays["effect_mask"],
        code_dictionary=code_dictionary,
        semantic_card_slots=semantic_card_slots,
        runtime_effect_bindings=runtime_effect_bindings,
    )
    if catalog.get("semantic_lookup_card_count") != vocabulary.card_count:
        raise ValueError("static semantic lookup card count mismatch")
    logical_hash = catalog.get("semantic_lookup_hash")
    if (
        not isinstance(logical_hash, str)
        or len(logical_hash) != 64
        or any(char not in "0123456789abcdef" for char in logical_hash)
    ):
        raise ValueError("static semantic logical hash is invalid")
    if verify_logical_hash:
        if logical_hash != lookup.prefix_hash():
            raise ValueError("static semantic logical hash does not match its table")
    else:
        # 表文件摘要已验证，正常启动直接复用编译时逻辑身份，避免重复扫描向量
        lookup._prefix_hashes[vocabulary.card_count] = logical_hash
    _register_lookup_catalogs(semantic_card_slots, runtime_effect_bindings)
    return lookup


def ensure_static_semantic_assets(
    directory=".",
    *,
    card_vocabulary=None,
    knowledge_base_filename=KNOWLEDGE_BASE_FILENAME,
):
    """优先加载预编译资产；缺失时从完整源语义组一次性生成"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    root = Path(directory).resolve()
    source_paths = (
        root / knowledge_base_filename,
        root / CODE_EMBEDDINGS_FILENAME,
        root / CODE_EMBEDDINGS_INDEX_FILENAME,
    )
    has_complete_sources = all(
        path.is_file() and not path.is_symlink() for path in source_paths
    )
    try:
        return load_static_semantic_assets(
            root,
            card_vocabulary=vocabulary,
            verify_source_files=has_complete_sources,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        if not has_complete_sources:
            raise
    lookup = build_static_semantic_lookup(
        root,
        card_vocabulary=vocabulary,
        knowledge_base_filename=knowledge_base_filename,
    )
    write_static_semantic_assets(
        root,
        card_vocabulary=vocabulary,
        knowledge_base_filename=knowledge_base_filename,
        lookup=lookup,
    )
    _register_lookup_catalogs(
        lookup.semantic_card_slots,
        lookup.runtime_effect_bindings,
    )
    return lookup


_RUNTIME_CATALOG_CACHE = {}


def register_static_semantic_runtime_catalog(
    directory=".",
    *,
    card_vocabulary=None,
):
    """仅加载约 0.55 MiB 的效果槽目录，供无模型 Worker 保留 V4 观测"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    root = Path(directory).resolve()
    table_path, catalog_path = _compiled_asset_paths(root)
    if not table_path.is_file() or not catalog_path.is_file():
        ensure_static_semantic_assets(root, card_vocabulary=vocabulary)
    cache_key = (
        str(root),
        vocabulary.vocabulary_hash,
        _file_signature(table_path),
        _file_signature(catalog_path),
    )
    cached = _RUNTIME_CATALOG_CACHE.get(cache_key)
    if cached is None:
        try:
            _, card_slots, runtime_bindings = _read_compiled_catalog(
                root,
                vocabulary,
            )
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ):
            ensure_static_semantic_assets(root, card_vocabulary=vocabulary)
            _, card_slots, runtime_bindings = _read_compiled_catalog(
                root,
                vocabulary,
            )
        _RUNTIME_CATALOG_CACHE.clear()
        cached = (card_slots, runtime_bindings)
        _RUNTIME_CATALOG_CACHE[cache_key] = cached
    _register_lookup_catalogs(*cached)
    return {
        "semantic_card_count": len(cached[0]),
        "runtime_effect_binding_count": len(cached[1]),
    }


_LOOKUP_CACHE = {}
_LOOKUP_CACHE_LOCK = threading.Lock()


def get_static_semantic_lookup(
    directory=".",
    *,
    card_vocabulary=None,
    knowledge_base_filename=KNOWLEDGE_BASE_FILENAME,
):
    """按资产签名复用模型侧查表，并在文件更新后自动重建"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    root = Path(directory).resolve()
    asset_paths = tuple(root / filename for filename in STATIC_SEMANTIC_ASSET_FILENAMES)
    if not all(path.is_file() for path in asset_paths):
        ensure_static_semantic_assets(
            root,
            card_vocabulary=vocabulary,
            knowledge_base_filename=knowledge_base_filename,
        )
    cache_key = (
        str(root),
        knowledge_base_filename,
        vocabulary.vocabulary_hash,
        tuple(_file_signature(path) for path in asset_paths),
    )
    with _LOOKUP_CACHE_LOCK:
        lookup = _LOOKUP_CACHE.get(cache_key)
        if lookup is None:
            lookup = ensure_static_semantic_assets(
                root,
                card_vocabulary=vocabulary,
                knowledge_base_filename=knowledge_base_filename,
            )
            _LOOKUP_CACHE.clear()
            _LOOKUP_CACHE[cache_key] = lookup
        _register_lookup_catalogs(
            lookup.semantic_card_slots,
            lookup.runtime_effect_bindings,
        )
        return lookup


def clear_static_semantic_lookup_cache():
    """清除进程内查表，供资产更新流程和隔离测试显式调用"""
    global _REGISTERED_CATALOGS
    with _LOOKUP_CACHE_LOCK:
        _LOOKUP_CACHE.clear()
    _RUNTIME_CATALOG_CACHE.clear()
    _REGISTERED_CATALOGS = None
