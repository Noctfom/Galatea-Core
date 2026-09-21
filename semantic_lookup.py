# -*- coding: utf-8 -*-
# 本文件把 Lua 静态语义编译为按精确卡片词表索引的模型侧查表，并维护可校验的语义身份

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from card_vocab import FIRST_CARD_TOKEN_ID, get_default_card_vocabulary
from effect_slot_binding import build_runtime_effect_binding_catalog
from semantic_assets import (
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
    KNOWLEDGE_BASE_FILENAME,
    validate_semantic_bundle,
)


STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION = 1
SEMANTIC_EFFECT_SLOTS = 8
SEMANTIC_CATEGORY_SLOTS = 8
SEMANTIC_REQUIREMENT_SLOTS = 16
SEMANTIC_RELATION_SLOTS = 4
SEMANTIC_REQUIREMENT_CAPACITY = 128
SEMANTIC_CATEGORY_CAPACITY = 4000

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
    asset_paths = (
        root / knowledge_base_filename,
        root / CODE_EMBEDDINGS_FILENAME,
        root / CODE_EMBEDDINGS_INDEX_FILENAME,
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
            lookup = build_static_semantic_lookup(
                root,
                card_vocabulary=vocabulary,
                knowledge_base_filename=knowledge_base_filename,
            )
            _LOOKUP_CACHE.clear()
            _LOOKUP_CACHE[cache_key] = lookup
        return lookup


def clear_static_semantic_lookup_cache():
    """清除进程内查表，供资产更新流程和隔离测试显式调用"""
    with _LOOKUP_CACHE_LOCK:
        _LOOKUP_CACHE.clear()
