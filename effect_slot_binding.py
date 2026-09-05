# 本文件负责把 Core 运行时效果标识精确绑定到 Lua 代码语义槽，不解析卡面描述文本

import re
from collections import defaultdict


MAX_MODELED_EFFECT_SLOTS = 8
RUNTIME_DESC_FIELD = "runtime_desc_ids"
_RUNTIME_EFFECT_SLOTS = {}


def _extract_initial_effect_body(lua_source):
    """截取 Lua 的 initial_effect 主体，避免同名局部变量跨函数串位"""
    start = lua_source.find(".initial_effect(c)")
    if start < 0:
        return ""
    end = lua_source.find("\nfunction ", start)
    return lua_source[start:] if end < 0 else lua_source[start:end]


def _resolve_stringid_owner(raw_owner, card_code):
    """只解析可证明的 Stringid 所属卡号，不猜测动态表达式"""
    owner = str(raw_owner).strip()
    if owner == "id":
        return int(card_code)
    try:
        value = int(owner, 0)
    except (TypeError, ValueError):
        return None
    return value if 0 < value <= 0x0FFFFFFF else None


def extract_runtime_effect_bindings(lua_source, card_code):
    """按 Effect.CreateEffect 对象绑定 SetDescription 的精确数值标识"""
    initial_body = _extract_initial_effect_body(str(lua_source or ""))
    creations = list(
        re.finditer(
            r"local\s+(e\d*)\s*=\s*Effect\.CreateEffect\(c\)",
            initial_body,
        )
    )
    desc_to_slots = defaultdict(set)
    unresolved_count = 0
    for slot_index, creation in enumerate(creations[:MAX_MODELED_EFFECT_SLOTS]):
        effect_name = re.escape(creation.group(1))
        description_calls = list(
            re.finditer(
                rf"\b{effect_name}:SetDescription\(\s*"
                r"(?:aux\.)?Stringid\(\s*([^,()]+)\s*,\s*(\d+)\s*\)\s*\)",
                initial_body,
            )
        )
        for match in description_calls:
            owner_code = _resolve_stringid_owner(match.group(1), card_code)
            string_index = int(match.group(2))
            if owner_code is None or not 0 <= string_index <= 15:
                unresolved_count += 1
                continue
            runtime_desc = ((owner_code << 4) | string_index) & 0xFFFFFFFF
            if runtime_desc != 0:
                desc_to_slots[runtime_desc].add(slot_index)

    bindings = {}
    ambiguous_desc_ids = []
    for runtime_desc, slots in desc_to_slots.items():
        if len(slots) == 1:
            bindings[runtime_desc] = next(iter(slots))
        else:
            ambiguous_desc_ids.append(runtime_desc)
    return {
        "bindings": bindings,
        "ambiguous_desc_ids": tuple(sorted(ambiguous_desc_ids)),
        "unresolved_count": unresolved_count,
    }


def apply_runtime_effect_bindings(card_data, lua_source, card_code):
    """把 Lua 对象绑定结果写回对应语义槽，返回内容是否发生变化"""
    if not isinstance(card_data, dict):
        return False
    effects = card_data.get("effects", [])
    if not isinstance(effects, list):
        return False
    extracted = extract_runtime_effect_bindings(lua_source, card_code)
    desc_ids_by_slot = defaultdict(list)
    for runtime_desc, slot_index in extracted["bindings"].items():
        desc_ids_by_slot[int(slot_index)].append(int(runtime_desc))

    changed = False
    for fallback_slot, effect in enumerate(effects, start=1):
        if not isinstance(effect, dict):
            continue
        try:
            slot_index = int(effect.get("slot", fallback_slot)) - 1
        except (TypeError, ValueError):
            continue
        desired = sorted(set(desc_ids_by_slot.get(slot_index, [])))
        previous = effect.get(RUNTIME_DESC_FIELD)
        if desired:
            if previous != desired:
                effect[RUNTIME_DESC_FIELD] = desired
                changed = True
        elif RUNTIME_DESC_FIELD in effect:
            del effect[RUNTIME_DESC_FIELD]
            changed = True
    return changed


def build_runtime_effect_binding_catalog(knowledge_base):
    """校验知识库中的运行时绑定，并构造卡号与完整 desc 到槽位的映射"""
    if not isinstance(knowledge_base, dict):
        raise ValueError("knowledge base must be a JSON object")
    catalog = {}
    for raw_card_code, card_data in knowledge_base.items():
        if not str(raw_card_code).isdigit() or not isinstance(card_data, dict):
            continue
        card_code = int(raw_card_code) & 0x7FFFFFFF
        effects = card_data.get("effects", [])
        if not isinstance(effects, list):
            continue
        for fallback_slot, effect in enumerate(effects, start=1):
            if not isinstance(effect, dict):
                continue
            try:
                slot_index = int(effect.get("slot", fallback_slot)) - 1
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid runtime effect slot for card {card_code}"
                ) from error
            raw_desc_ids = effect.get(RUNTIME_DESC_FIELD, [])
            if not isinstance(raw_desc_ids, list):
                raise ValueError(
                    f"{RUNTIME_DESC_FIELD} must be a list for card {card_code}"
                )
            if raw_desc_ids and not 0 <= slot_index < MAX_MODELED_EFFECT_SLOTS:
                raise ValueError(
                    f"runtime effect binding exceeds modeled slots for card {card_code}"
                )
            for raw_desc in raw_desc_ids:
                if isinstance(raw_desc, bool) or not isinstance(raw_desc, int):
                    raise ValueError(
                        f"runtime effect desc must be an integer for card {card_code}"
                    )
                runtime_desc = int(raw_desc)
                if not 0 < runtime_desc <= 0xFFFFFFFF:
                    raise ValueError(
                        f"runtime effect desc is outside uint32 for card {card_code}"
                    )
                key = (card_code, runtime_desc)
                previous_slot = catalog.get(key)
                if previous_slot is not None and previous_slot != slot_index:
                    raise ValueError(
                        f"runtime effect desc maps to multiple slots for card {card_code}: "
                        f"{runtime_desc}"
                    )
                catalog[key] = slot_index
    return catalog


def register_runtime_effect_bindings(knowledge_base):
    """登记当前知识库的运行时效果绑定，供 GameState 和编码器无 I/O 查询"""
    global _RUNTIME_EFFECT_SLOTS
    _RUNTIME_EFFECT_SLOTS = build_runtime_effect_binding_catalog(knowledge_base)
    return len(_RUNTIME_EFFECT_SLOTS)


def resolve_runtime_effect_slot(card_code, runtime_desc):
    """返回已证明的零基语义槽；缺失、动态或歧义绑定返回 None"""
    try:
        key = (
            int(card_code) & 0x7FFFFFFF,
            int(runtime_desc) & 0xFFFFFFFF,
        )
    except (TypeError, ValueError):
        return None
    if key[1] == 0:
        return None
    return _RUNTIME_EFFECT_SLOTS.get(key)
