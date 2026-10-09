# 本文件从已有 Lua 代码语义资产编译公开提示目录，区分精确效果与父效果上下文

import re
from collections import defaultdict

from effect_slot_binding import build_runtime_effect_binding_catalog


PUBLIC_HINT_BINDING_FORMAT_VERSION = 1
PUBLIC_HINT_PARENT_DESCS_FIELD = 'public_hint_parent_desc_ids'
_CATALOG = {}
_DESCRIPTION = re.compile(
    r"\b\w+:SetDescription\(\s*(?:aux\.)?Stringid\(\s*(id|\d+)\s*,\s*(\d+)\s*\)\s*\)"
)


def _lua_code_only(source):
    """屏蔽 Lua 注释和字符串，避免其中的伪代码被当成静态绑定"""
    result, index = [], 0
    while index < len(source):
        comment = source.startswith("--", index)
        start = index + 2 if comment else index
        long_open = re.match(r"\[(=*)\[", source[start:]) if source[start:start + 1] == "[" else None
        if long_open:
            closing = "]" + long_open.group(1) + "]"
            end = source.find(closing, start + len(long_open.group(0)))
            index = len(source) if end < 0 else end + len(closing)
            result.append(" ")
        elif comment:
            end = source.find("\n", start)
            index = len(source) if end < 0 else end
            result.append(" ")
        elif source[index] in "\"'":
            quote = source[index]
            index += 1
            while index < len(source):
                char = source[index]
                index += 1
                if char == "\\":
                    index += 1
                elif char == quote:
                    break
            result.append(" ")
        else:
            result.append(source[index])
            index += 1
    return "".join(result)


def _parent_description_ids(effect, card):
    """读取持久绑定或当前向量原始代码证据，不用新脚本猜测旧向量来源"""
    stored = effect.get(PUBLIC_HINT_PARENT_DESCS_FIELD, [])
    if not isinstance(stored, list) or len(stored) > 4096:
        raise ValueError('invalid public hint parent description list')
    if any(isinstance(desc, bool) or not isinstance(desc, int) or not 0 < desc <= 0xFFFFFFFF for desc in stored):
        raise ValueError('invalid public hint parent description identity')
    desc_ids = set(stored)
    for owner, offset in _DESCRIPTION.findall(_lua_code_only(str(effect.get('raw_code', '')))):
        owner_code = card if owner == 'id' else int(owner)
        if 0 < owner_code <= 0x0FFFFFFF and 0 <= int(offset) <= 15:
            desc_ids.add((owner_code << 4) | int(offset))
    return sorted(desc_ids)


def apply_public_hint_metadata(card_data, card):
    """在原始代码仍在时保存轻量来源，供去掉raw_code的远程资产继续编译"""
    changed = False
    for effect in card_data.get('effects', []):
        if not isinstance(effect, dict) or 'raw_code' not in effect:
            continue
        # 有真实代码时重建而非合并陈旧记录；无代码的既有元数据原样保留
        fresh = dict(effect)
        fresh.pop(PUBLIC_HINT_PARENT_DESCS_FIELD, None)
        identities = _parent_description_ids(fresh, int(card))
        if effect.get(PUBLIC_HINT_PARENT_DESCS_FIELD) != identities:
            effect[PUBLIC_HINT_PARENT_DESCS_FIELD] = identities
            changed = True
    return changed


def build_public_hint_catalog(knowledge_base):
    """只绑定唯一且可证明的来源；动态提示引用父代码而不冒充初始化效果"""
    exact, contexts = defaultdict(set), defaultdict(set)
    for (card, desc), slot in build_runtime_effect_binding_catalog(knowledge_base).items():
        exact[desc].add((card, slot, 1))
    for raw_card, data in knowledge_base.items():
        if not str(raw_card).isdigit() or not isinstance(data, dict):
            continue
        card = int(raw_card)
        for fallback, effect in enumerate(data.get("effects", [])):
            if not isinstance(effect, dict):
                continue
            slot = int(effect.get("slot", fallback + 1)) - 1
            if not 0 <= slot < 8:
                continue
            for desc in _parent_description_ids(effect, card):
                contexts[desc].add((card, slot, 2))
    catalog = {}
    for desc in sorted(exact.keys() | contexts.keys()):
        candidates = exact.get(desc) or contexts[desc]
        sources = {(card, slot) for card, slot, _ in exact.get(desc, set()) | contexts.get(desc, set())}
        if len(sources) != 1:
            continue
        if len(candidates) == 1:
            catalog[desc] = next(iter(candidates))
    return catalog


def serialize_public_hint_catalog(catalog):
    """把提示目录序列化为安全且确定的整数四元组"""
    return [[desc, *record] for desc, record in sorted(catalog.items())]


def deserialize_public_hint_catalog(records):
    """验证提示来源、槽位与绑定级别，拒绝重复身份和越界记录"""
    if not isinstance(records, list) or len(records) > 200000:
        raise ValueError("invalid public hint catalog")
    catalog = {}
    for record in records:
        if not isinstance(record, list) or len(record) != 4:
            raise ValueError("invalid public hint catalog record")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in record):
            raise ValueError("public hint catalog values must be integers")
        desc, card, slot, binding = record
        if not 0 < desc <= 0xFFFFFFFF or not 0 < card <= 0x7FFFFFFF or not 0 <= slot < 8 or binding not in (1, 2) or desc in catalog:
            raise ValueError("invalid or duplicate public hint binding")
        catalog[desc] = (card, slot, binding)
    return catalog


def register_public_hint_catalog(catalog):
    """登记轻量目录，Worker 无需加载大语义表或句向量生成模型"""
    global _CATALOG
    _CATALOG = catalog


def resolve_public_hint(desc):
    """返回精确效果或明确标识的父上下文，未知提示保持未知"""
    return _CATALOG.get(int(desc), (0, -1, 0))
