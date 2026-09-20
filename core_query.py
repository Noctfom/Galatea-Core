# 本文件解析公开 ygopro-core 旧版 query_card 返回的卡片个体状态。

import io
import struct


QUERY_CODE = 0x1
QUERY_POSITION = 0x2
QUERY_ALIAS = 0x4
QUERY_TYPE = 0x8
QUERY_LEVEL = 0x10
QUERY_RANK = 0x20
QUERY_ATTRIBUTE = 0x40
QUERY_RACE = 0x80
QUERY_ATTACK = 0x100
QUERY_DEFENSE = 0x200
QUERY_BASE_ATTACK = 0x400
QUERY_BASE_DEFENSE = 0x800
QUERY_REASON = 0x1000
QUERY_REASON_CARD = 0x2000
QUERY_EQUIP_CARD = 0x4000
QUERY_TARGET_CARD = 0x8000
QUERY_OVERLAY_CARD = 0x10000
QUERY_COUNTERS = 0x20000
QUERY_OWNER = 0x40000
QUERY_STATUS = 0x80000
QUERY_LSCALE = 0x200000
QUERY_RSCALE = 0x400000
QUERY_LINK = 0x800000

STATUS_DISABLED = 0x1
STATUS_PROC_COMPLETE = 0x8
STATUS_FORBIDDEN = 0x4000000

PUBLIC_CARD_QUERY_FLAGS = (
    QUERY_CODE
    | QUERY_POSITION
    | QUERY_ALIAS
    | QUERY_TYPE
    | QUERY_LEVEL
    | QUERY_RANK
    | QUERY_ATTRIBUTE
    | QUERY_RACE
    | QUERY_ATTACK
    | QUERY_DEFENSE
    | QUERY_BASE_ATTACK
    | QUERY_BASE_DEFENSE
    | QUERY_REASON
    | QUERY_REASON_CARD
    | QUERY_EQUIP_CARD
    | QUERY_TARGET_CARD
    | QUERY_OVERLAY_CARD
    | QUERY_COUNTERS
    | QUERY_OWNER
    | QUERY_STATUS
    | QUERY_LSCALE
    | QUERY_RSCALE
    | QUERY_LINK
)


class CoreQueryError(ValueError):
    """表示公开 Core 查询返回了截断或不一致的数据。"""


def _read_exact(stream, size, label):
    """读取固定长度查询字段，并在截断时给出字段名称。"""
    value = stream.read(size)
    if len(value) != size:
        raise CoreQueryError(f"{label} is truncated")
    return value


def _read_u32(stream, label):
    """读取旧版查询协议的小端 uint32。"""
    return struct.unpack('<I', _read_exact(stream, 4, label))[0]


def _read_i32(stream, label):
    """读取旧版查询协议的小端 int32。"""
    return struct.unpack('<i', _read_exact(stream, 4, label))[0]


def parse_legacy_card_query(payload):
    """按公开旧版 query_card 位序解析完整卡片个体状态。"""
    raw = bytes(payload)
    if len(raw) < 8:
        raise CoreQueryError("query_card response is shorter than its header")
    stream = io.BytesIO(raw)
    data_length = _read_u32(stream, "data length")
    actual_flags = _read_u32(stream, "actual flags")
    if data_length < 8 or data_length > len(raw):
        raise CoreQueryError(
            f"query_card length {data_length} is outside buffer size {len(raw)}"
        )

    result = {
        "query_flags": actual_flags,
        "code": 0,
        "position_raw": 0,
        "current_code": 0,
        "current_type": 0,
        "current_level": 0,
        "current_rank": 0,
        "current_attr": 0,
        "current_race": 0,
        "current_atk": 0,
        "current_def": 0,
        "base_atk": 0,
        "base_def": 0,
        "reason": 0,
        "reason_card": 0,
        "equip_target": 0,
        "target_locations": [],
        "overlay_codes": [],
        "counter_items": [],
        "original_owner": -1,
        "status_mask": 0,
        "lscale": 0,
        "rscale": 0,
        "link_rating": 0,
        "link_marker": 0,
    }

    scalar_fields = (
        (QUERY_CODE, "code", _read_u32),
        (QUERY_POSITION, "position_raw", _read_u32),
        (QUERY_ALIAS, "current_code", _read_u32),
        (QUERY_TYPE, "current_type", _read_u32),
        (QUERY_LEVEL, "current_level", _read_u32),
        (QUERY_RANK, "current_rank", _read_u32),
        (QUERY_ATTRIBUTE, "current_attr", _read_u32),
        (QUERY_RACE, "current_race", _read_u32),
        (QUERY_ATTACK, "current_atk", _read_i32),
        (QUERY_DEFENSE, "current_def", _read_i32),
        (QUERY_BASE_ATTACK, "base_atk", _read_i32),
        (QUERY_BASE_DEFENSE, "base_def", _read_i32),
        (QUERY_REASON, "reason", _read_u32),
        (QUERY_REASON_CARD, "reason_card", _read_u32),
        (QUERY_EQUIP_CARD, "equip_target", _read_u32),
    )
    for flag, key, reader in scalar_fields:
        if actual_flags & flag:
            result[key] = reader(stream, key)

    if actual_flags & QUERY_TARGET_CARD:
        count = _read_i32(stream, "target card count")
        if count < 0 or count > 1024:
            raise CoreQueryError(f"invalid target card count: {count}")
        result["target_locations"] = [
            _read_u32(stream, f"target card {index}") for index in range(count)
        ]

    if actual_flags & QUERY_OVERLAY_CARD:
        count = _read_i32(stream, "overlay card count")
        if count < 0 or count > 1024:
            raise CoreQueryError(f"invalid overlay card count: {count}")
        result["overlay_codes"] = [
            _read_u32(stream, f"overlay card {index}") for index in range(count)
        ]

    if actual_flags & QUERY_COUNTERS:
        count = _read_i32(stream, "counter type count")
        if count < 0 or count > 1024:
            raise CoreQueryError(f"invalid counter type count: {count}")
        counter_items = []
        for index in range(count):
            packed = _read_u32(stream, f"counter item {index}")
            counter_items.append((packed & 0xFFFF, (packed >> 16) & 0xFFFF))
        result["counter_items"] = counter_items

    if actual_flags & QUERY_OWNER:
        result["original_owner"] = _read_i32(stream, "original owner")
    if actual_flags & QUERY_STATUS:
        result["status_mask"] = _read_u32(stream, "status")
    if actual_flags & QUERY_LSCALE:
        result["lscale"] = _read_u32(stream, "left scale")
    if actual_flags & QUERY_RSCALE:
        result["rscale"] = _read_u32(stream, "right scale")
    if actual_flags & QUERY_LINK:
        result["link_rating"] = _read_u32(stream, "link rating")
        result["link_marker"] = _read_u32(stream, "link marker")

    if stream.tell() != data_length:
        raise CoreQueryError(
            f"query_card consumed {stream.tell()} bytes but header reports {data_length}"
        )

    position_raw = result["position_raw"]
    result.update(
        controller=position_raw & 0xFF,
        location=(position_raw >> 8) & 0xFF,
        sequence=(position_raw >> 16) & 0xFF,
        position=(position_raw >> 24) & 0xFF,
        is_equip_source=bool(result["equip_target"]),
        counter_count=sum(count for _, count in result["counter_items"]),
        properly_summoned=bool(result["status_mask"] & STATUS_PROC_COMPLETE),
        is_disabled=bool(result["status_mask"] & STATUS_DISABLED),
        is_forbidden=bool(result["status_mask"] & STATUS_FORBIDDEN),
    )
    return result
