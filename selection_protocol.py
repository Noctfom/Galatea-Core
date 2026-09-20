# 本文件按 OCGCore 原生判定校验多选、解放、凑星与逐项选/撤选响应。

import io
import struct
from functools import lru_cache


CORE_RESPONSE_BUFFER_SIZE = 512
CANCEL_RESPONSE = b"\xff\xff\xff\xff"
SELECTION_RESPONSE_MSGS = frozenset({15, 20, 23, 26})


class SelectionProtocolError(ValueError):
    """表示选择响应不满足当前 Core 提示的原生约束。"""


def _read_exact(stream, size, label):
    """读取固定长度字段，截断时给出可定位的协议错误。"""
    data = stream.read(size)
    if len(data) != size:
        raise SelectionProtocolError(f"{label} is truncated")
    return data


def _read_u8(stream, label):
    """读取一个无符号字节。"""
    return _read_exact(stream, 1, label)[0]


def _read_u32(stream, label):
    """读取一个小端无符号四字节整数。"""
    return struct.unpack('<I', _read_exact(stream, 4, label))[0]


def parse_selection_prompt(msg_type, msg_payload):
    """把四类选择提示解析成统一约束，供生成器与回归测试共用。"""
    if msg_type not in SELECTION_RESPONSE_MSGS:
        raise SelectionProtocolError(f"message type {msg_type} is not a selection prompt")

    stream = io.BytesIO(bytes(msg_payload))
    if msg_type in (15, 20):
        prompt = {
            "player": _read_u8(stream, "player"),
            "cancelable": bool(_read_u8(stream, "cancelable")),
            "minimum": _read_u8(stream, "minimum"),
            "maximum": _read_u8(stream, "maximum"),
        }
        count = _read_u8(stream, "candidate count")
        candidates = []
        for index in range(count):
            code = _read_u32(stream, f"candidate {index} code")
            controller = _read_u8(stream, f"candidate {index} controller")
            location = _read_u8(stream, f"candidate {index} location")
            sequence = _read_u8(stream, f"candidate {index} sequence")
            extra = _read_u8(stream, f"candidate {index} extra value")
            candidates.append({
                "code": code,
                "controller": controller,
                "location": location,
                "sequence": sequence,
                "position": extra if msg_type == 15 else 0,
                "release_value": extra if msg_type == 20 else 0,
            })
        prompt["candidates"] = candidates
        return prompt

    if msg_type == 23:
        prompt = {
            "mode": _read_u8(stream, "sum mode"),
            "player": _read_u8(stream, "player"),
            "target": _read_u32(stream, "sum target"),
            "minimum": _read_u8(stream, "minimum"),
            "maximum": _read_u8(stream, "maximum"),
            "cancelable": False,
        }

        def read_sum_cards(count, label):
            """读取凑星协议中的必选或可选卡片列表。"""
            cards = []
            for index in range(count):
                cards.append({
                    "code": _read_u32(stream, f"{label} {index} code"),
                    "controller": _read_u8(stream, f"{label} {index} controller"),
                    "location": _read_u8(stream, f"{label} {index} location"),
                    "sequence": _read_u8(stream, f"{label} {index} sequence"),
                    "sum_value": _read_u32(stream, f"{label} {index} sum value"),
                })
            return cards

        mandatory_count = _read_u8(stream, "mandatory count")
        prompt["mandatory"] = read_sum_cards(mandatory_count, "mandatory card")
        candidate_count = _read_u8(stream, "candidate count")
        prompt["candidates"] = read_sum_cards(candidate_count, "candidate")
        return prompt

    prompt = {
        "player": _read_u8(stream, "player"),
        "finishable": bool(_read_u8(stream, "finishable")),
        "cancelable": bool(_read_u8(stream, "cancelable")),
        "minimum": _read_u8(stream, "minimum"),
        "maximum": _read_u8(stream, "maximum"),
    }

    def read_unselect_cards(count, label):
        """读取 Type 26 的可选或已选卡片列表。"""
        cards = []
        for index in range(count):
            cards.append({
                "code": _read_u32(stream, f"{label} {index} code"),
                "location_info": _read_u32(stream, f"{label} {index} location"),
            })
        return cards

    selectable_count = _read_u8(stream, "selectable count")
    prompt["selectable"] = read_unselect_cards(selectable_count, "selectable")
    selected_count = _read_u8(stream, "selected count")
    prompt["selected"] = read_unselect_cards(selected_count, "selected")
    return prompt


def _is_cancel_response(response):
    """识别 Core 以 int32(-1) 表示的取消或完成响应。"""
    return len(response) >= 4 and response[:4] == CANCEL_RESPONSE


def _decode_index_response(response):
    """读取 count+indices 响应并拒绝截断。"""
    if not response:
        raise SelectionProtocolError("selection response is empty")
    count = response[0]
    required_size = 1 + count
    if len(response) < required_size:
        raise SelectionProtocolError(
            f"selection response is truncated: need {required_size}, got {len(response)}"
        )
    return count, list(response[1:required_size])


def _validate_unique_indices(indices, candidate_count):
    """按 Core 的 check_response 规则校验索引范围与去重。"""
    if len(indices) != len(set(indices)):
        raise SelectionProtocolError("selection response contains duplicate indices")
    if any(index >= candidate_count for index in indices):
        raise SelectionProtocolError("selection response contains an out-of-range index")


def _sum_options(raw_value):
    """拆出 Core sum_param 中的低位值与可选高位值。"""
    first = raw_value & 0xFFFF
    second = (raw_value >> 16) & 0xFFFF
    values = [first]
    if second and second != first:
        values.append(second)
    return values


def _matches_exact_sum(raw_values, target):
    """逐分支复现 Core 的精确求和及最小素材边界。"""
    @lru_cache(maxsize=None)
    def check(index, remaining, previous_minimum):
        """按 playerop.cpp 的 select_sum_check1 顺序验证一个赋值分支。"""
        if remaining == 0 or index == len(raw_values):
            return False

        options = _sum_options(raw_values[index])
        if index == len(raw_values) - 1:
            return any(
                remaining == option
                and remaining + previous_minimum > option
                for option in options
            )

        return any(
            remaining > option
            and check(
                index + 1,
                remaining - option,
                min(option, previous_minimum),
            )
            for option in options
        )

    return check(0, target, 0xFFFF)


def _matches_at_least_sum(raw_values, target):
    """复现 Core 的 SumGreater 判定边界。"""
    if not raw_values:
        return target <= 0
    minimums = [min(_sum_options(value)) for value in raw_values]
    maximums = [max(_sum_options(value)) for value in raw_values]
    return sum(maximums) >= target and sum(minimums) - min(minimums) < target


def validate_selection_response(msg_type, msg_payload, response):
    """严格验证响应；成功时返回原始字节，失败时抛出协议错误。"""
    if isinstance(response, int):
        response = int(response).to_bytes(4, byteorder='little', signed=True)
    response = bytes(response)
    prompt = parse_selection_prompt(msg_type, msg_payload)

    if _is_cancel_response(response):
        if msg_type == 23:
            raise SelectionProtocolError("MSG_SELECT_SUM has no cancel response")
        if msg_type == 26:
            if not (prompt["cancelable"] or prompt["finishable"]):
                raise SelectionProtocolError("MSG_SELECT_UNSELECT_CARD cannot finish or cancel")
        elif not prompt["cancelable"]:
            raise SelectionProtocolError("selection prompt is not cancelable")
        return response

    count, indices = _decode_index_response(response)
    if msg_type == 15:
        if count < prompt["minimum"] or count > prompt["maximum"]:
            raise SelectionProtocolError("MSG_SELECT_CARD count is outside min/max")
        _validate_unique_indices(indices, len(prompt["candidates"]))
    elif msg_type == 20:
        if count > prompt["maximum"]:
            raise SelectionProtocolError("MSG_SELECT_TRIBUTE count exceeds max")
        _validate_unique_indices(indices, len(prompt["candidates"]))
        release_total = sum(
            prompt["candidates"][index]["release_value"] for index in indices
        )
        if release_total < prompt["minimum"]:
            raise SelectionProtocolError("MSG_SELECT_TRIBUTE release value is below min")
    elif msg_type == 26:
        if count != 1:
            raise SelectionProtocolError("MSG_SELECT_UNSELECT_CARD requires one index")
        _validate_unique_indices(
            indices,
            len(prompt["selectable"]) + len(prompt["selected"]),
        )
    else:
        mandatory = prompt["mandatory"]
        if count < len(mandatory):
            raise SelectionProtocolError("MSG_SELECT_SUM omits mandatory placeholders")
        optional_indices = indices[len(mandatory):]
        _validate_unique_indices(optional_indices, len(prompt["candidates"]))
        if prompt["mode"] == 0:
            if not (
                prompt["minimum"] + len(mandatory)
                <= count
                <= prompt["maximum"] + len(mandatory)
            ):
                raise SelectionProtocolError("MSG_SELECT_SUM count is outside min/max")
        raw_values = [card["sum_value"] for card in mandatory]
        raw_values.extend(
            prompt["candidates"][index]["sum_value"] for index in optional_indices
        )
        if prompt["mode"] == 0:
            valid_sum = _matches_exact_sum(raw_values, prompt["target"])
        else:
            valid_sum = _matches_at_least_sum(raw_values, prompt["target"])
        if not valid_sum:
            raise SelectionProtocolError("MSG_SELECT_SUM values do not satisfy target")
    return response


def is_selection_response_valid(msg_type, msg_payload, response):
    """返回选择响应是否会通过当前 Core 的对应边界检查。"""
    try:
        validate_selection_response(msg_type, msg_payload, response)
        return True
    except (SelectionProtocolError, OverflowError, TypeError, ValueError):
        return False


def build_core_response_buffer(response):
    """构造 Core 固定读取的 512 字节缓冲，避免截断与越界读取。"""
    raw = bytes(response)
    if len(raw) > CORE_RESPONSE_BUFFER_SIZE:
        raise SelectionProtocolError(
            f"Core response exceeds {CORE_RESPONSE_BUFFER_SIZE} bytes"
        )
    return raw.ljust(CORE_RESPONSE_BUFFER_SIZE, b'\x00')
