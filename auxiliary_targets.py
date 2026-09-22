"""构建辅助头训练所需的多尺度后验标签，并提供紧凑协议与审计统计"""

from enum import IntEnum

import torch

from data_types import TransitionBoundary, TransitionResultFlag
from event_history import (
    TRANSITION_MESSAGE_GROUP_COUNT,
    TRANSITION_ZONE_NAMES,
    encode_transition_message_groups,
    mask_transition_event_for_player,
)


AUXILIARY_TARGET_FORMAT_VERSION = 1


class AuxiliaryHorizon(IntEnum):
    """定义辅助标签采用的四个未来时间尺度"""

    NEXT_DECISION = 0
    CHAIN_END = 1
    TURN_END = 2
    TERMINAL = 3


AUXILIARY_HORIZON_COUNT = len(AuxiliaryHorizon)
AUXILIARY_LP_DIM = 2
AUXILIARY_ZONE_DIM = 2 * len(TRANSITION_ZONE_NAMES)
AUXILIARY_CHAIN_DIM = 3
_CHAIN_MESSAGE_TYPES = frozenset({70, 71, 72, 73, 74, 75, 76})

# 这些字段仅随 rollout 保存，不进入观测、中央推理或标准 ONNX 图
AUXILIARY_TARGET_SPECS = {
    "valid": ((AUXILIARY_HORIZON_COUNT,), torch.bool),
    "lp_delta": (
        (AUXILIARY_HORIZON_COUNT, AUXILIARY_LP_DIM),
        torch.int32,
    ),
    "zone_delta": (
        (AUXILIARY_HORIZON_COUNT, AUXILIARY_ZONE_DIM),
        torch.int16,
    ),
    "event_count": ((AUXILIARY_HORIZON_COUNT,), torch.int16),
    "result_flags": ((), torch.int16),
    "message_groups": ((), torch.int16),
    "message_count": ((), torch.int16),
    "boundary": ((), torch.uint8),
    "summon_method": ((), torch.uint8),
    "chain": ((AUXILIARY_CHAIN_DIM,), torch.uint8),
    "terminal_outcome": ((), torch.int8),
    "terminal_reason": ((), torch.uint8),
    "terminal_remaining_events": ((), torch.int16),
}
AUXILIARY_TARGET_BYTES_PER_STEP = sum(
    torch.empty(shape, dtype=dtype).numel()
    * torch.empty((), dtype=dtype).element_size()
    for shape, dtype in AUXILIARY_TARGET_SPECS.values()
)


def allocate_auxiliary_target_columns(length):
    """按固定协议预分配 Worker 使用的连续辅助标签列"""
    length = int(length)
    if length < 0:
        raise ValueError("auxiliary target length cannot be negative")
    return {
        name: torch.zeros((length, *shape), dtype=dtype)
        for name, (shape, dtype) in AUXILIARY_TARGET_SPECS.items()
    }


def _saturate_int16(value):
    """把事件数量安全压入有符号 16 位存储范围"""
    return max(-32768, min(32767, int(value)))


def _is_terminal_event(event):
    """判断事件是否由真实 Core 终局消息结束"""
    return (
        int(event.boundary) == int(TransitionBoundary.TERMINAL)
        and bool(int(event.result_flags) & int(TransitionResultFlag.TERMINAL))
    )


def _is_chain_event(event):
    """判断事件是否包含可验证的连锁上下文"""
    return (
        int(event.chain_depth_before) > 0
        or int(event.chain_depth_after) > 0
        or int(event.max_chain_depth) > 0
        or bool(_CHAIN_MESSAGE_TYPES.intersection(event.message_types))
    )


def _build_boundary_indices(events):
    """以一次反向扫描生成各步最近的连锁末、回合末和终局索引"""
    count = len(events)
    chain_ends = [None] * count
    turn_ends = [None] * count
    terminals = [None] * count
    next_chain_end = None
    next_terminal = None

    for index in range(count - 1, -1, -1):
        event = events[index]
        if _is_terminal_event(event):
            next_terminal = index
        terminals[index] = next_terminal

        chain_event = _is_chain_event(event)
        if chain_event and (
            _is_terminal_event(event)
            or 74 in event.message_types
            or (
                int(event.max_chain_depth) > 0
                and int(event.chain_depth_after) == 0
            )
        ):
            next_chain_end = index
        chain_ends[index] = next_chain_end if chain_event else None
        if not chain_event:
            next_chain_end = None

        if _is_terminal_event(event) or 40 in event.message_types:
            turn_ends[index] = index
        elif index + 1 < count:
            if int(events[index + 1].turn_count) != int(event.turn_count):
                turn_ends[index] = index
            else:
                turn_ends[index] = turn_ends[index + 1]

    return chain_ends, turn_ends, terminals


def _build_delta_prefixes(events):
    """构建 LP 与区域变化前缀和，使每个未来尺度均可常数时间汇总"""
    lp_prefix = [[0, 0]]
    zone_prefix = [[0] * AUXILIARY_ZONE_DIM]
    for event in events:
        lp_prefix.append([
            lp_prefix[-1][0] + int(event.lp_delta_p0),
            lp_prefix[-1][1] + int(event.lp_delta_p1),
        ])
        zone_prefix.append([
            zone_prefix[-1][index] + int(event.zone_deltas[index])
            for index in range(AUXILIARY_ZONE_DIM)
        ])
    return lp_prefix, zone_prefix


def _write_horizon(
    targets,
    row,
    horizon,
    start_index,
    end_index,
    lp_prefix,
    zone_prefix,
):
    """把一段真实事件累计为指定时间尺度的公开资源摘要"""
    if end_index is None:
        return
    targets["valid"][row, int(horizon)] = True
    targets["event_count"][row, int(horizon)] = _saturate_int16(
        end_index - start_index + 1
    )
    targets["lp_delta"][row, int(horizon), 0] = (
        lp_prefix[end_index + 1][0] - lp_prefix[start_index][0]
    )
    targets["lp_delta"][row, int(horizon), 1] = (
        lp_prefix[end_index + 1][1] - lp_prefix[start_index][1]
    )
    for zone_index in range(AUXILIARY_ZONE_DIM):
        value = (
            zone_prefix[end_index + 1][zone_index]
            - zone_prefix[start_index][zone_index]
        )
        targets["zone_delta"][row, int(horizon), zone_index] = (
            _saturate_int16(value)
        )


def _terminal_outcome_for_player(winner, player_id):
    """把公开终局胜者转换成当前训练玩家视角的胜负标签"""
    winner = int(winner)
    if winner == int(player_id):
        return 1
    if winner in (0, 1):
        return -1
    return 0


def build_auxiliary_targets(
    step_sequence_ids,
    completed_events,
    *,
    player_id,
    winner=-1,
    win_reason=0,
):
    """按训练步对齐完整事件流，生成下一决策、连锁末、回合末和终局标签"""
    if int(player_id) not in (0, 1):
        raise ValueError(f"player_id must be 0 or 1, got {player_id}")

    sequence_ids = [int(value) for value in step_sequence_ids]
    if sequence_ids != sorted(sequence_ids) or len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("training step sequence ids must be unique and ordered")

    events = list(completed_events)
    event_indices = {}
    previous_sequence = -1
    for index, event in enumerate(events):
        sequence_id = int(event.sequence_id)
        if sequence_id <= previous_sequence:
            raise ValueError("completed transition events must be strictly ordered")
        previous_sequence = sequence_id
        event_indices[sequence_id] = index

    targets = allocate_auxiliary_target_columns(len(sequence_ids))
    lp_prefix, zone_prefix = _build_delta_prefixes(events)
    chain_ends, turn_ends, terminals = _build_boundary_indices(events)
    max_completed_sequence = previous_sequence
    for row, sequence_id in enumerate(sequence_ids):
        event_index = event_indices.get(sequence_id)
        if event_index is None:
            # 正常截断只可能留下位于完整事件流末尾的一个挂起动作
            if sequence_id <= max_completed_sequence:
                raise ValueError(
                    f"transition event {sequence_id} is missing inside the completed stream"
                )
            continue

        event = events[event_index]
        if int(event.actor) != int(player_id):
            raise ValueError(
                f"training step {sequence_id} actor does not match player {player_id}"
            )
        visible_event = mask_transition_event_for_player(event, int(player_id))
        if visible_event is None:
            raise ValueError("a submitted action must remain visible to its actor")

        _write_horizon(
            targets,
            row,
            AuxiliaryHorizon.NEXT_DECISION,
            event_index,
            event_index,
            lp_prefix,
            zone_prefix,
        )
        chain_end = chain_ends[event_index]
        _write_horizon(
            targets,
            row,
            AuxiliaryHorizon.CHAIN_END,
            event_index,
            chain_end,
            lp_prefix,
            zone_prefix,
        )
        turn_end = turn_ends[event_index]
        _write_horizon(
            targets,
            row,
            AuxiliaryHorizon.TURN_END,
            event_index,
            turn_end,
            lp_prefix,
            zone_prefix,
        )
        terminal = terminals[event_index]
        _write_horizon(
            targets,
            row,
            AuxiliaryHorizon.TERMINAL,
            event_index,
            terminal,
            lp_prefix,
            zone_prefix,
        )

        targets["result_flags"][row] = int(visible_event.result_flags)
        targets["message_groups"][row] = encode_transition_message_groups(
            visible_event.message_types
        )
        targets["message_count"][row] = _saturate_int16(
            visible_event.message_count
        )
        targets["boundary"][row] = int(visible_event.boundary)
        targets["summon_method"][row] = int(visible_event.summon_method_id)
        targets["chain"][row, 0] = min(
            255,
            max(0, int(visible_event.chain_depth_before)),
        )
        targets["chain"][row, 1] = min(
            255,
            max(0, int(visible_event.chain_depth_after)),
        )
        targets["chain"][row, 2] = min(
            255,
            max(0, int(visible_event.max_chain_depth)),
        )

        if terminal is not None:
            targets["terminal_outcome"][row] = _terminal_outcome_for_player(
                winner,
                player_id,
            )
            targets["terminal_reason"][row] = min(
                255,
                max(0, int(win_reason)),
            )
            targets["terminal_remaining_events"][row] = _saturate_int16(
                terminal - event_index + 1
            )

    return targets


def summarize_auxiliary_targets(targets):
    """汇总标签覆盖率和即时结果分布，供 TensorBoard 审计使用"""
    valid = targets["valid"].bool()
    step_count = int(valid.shape[0])
    denominator = max(1, step_count)
    next_valid = valid[:, AuxiliaryHorizon.NEXT_DECISION]
    flags = targets["result_flags"].to(torch.int32)[next_valid]
    result_denominator = max(1, int(next_valid.sum().item()))

    def flag_rate(flag):
        return float(
            ((flags & int(flag)) != 0).sum().item() / result_denominator
        )

    return {
        "step_count": step_count,
        "coverage_next": float(valid[:, AuxiliaryHorizon.NEXT_DECISION].sum().item() / denominator),
        "coverage_chain": float(valid[:, AuxiliaryHorizon.CHAIN_END].sum().item() / denominator),
        "coverage_turn": float(valid[:, AuxiliaryHorizon.TURN_END].sum().item() / denominator),
        "coverage_terminal": float(valid[:, AuxiliaryHorizon.TERMINAL].sum().item() / denominator),
        "retry_rate": flag_rate(TransitionResultFlag.RETRY),
        "cancel_rate": flag_rate(TransitionResultFlag.CANCELLED),
        "state_changed_rate": flag_rate(TransitionResultFlag.STATE_CHANGED),
        "negated_rate": flag_rate(TransitionResultFlag.CHAIN_NEGATED),
        "disabled_rate": flag_rate(TransitionResultFlag.CHAIN_DISABLED),
    }


def get_auxiliary_target_protocol_descriptor():
    """返回 3.12.0 后验标签协议的机器可读说明"""
    return {
        "format_version": AUXILIARY_TARGET_FORMAT_VERSION,
        "horizons": tuple(item.name.lower() for item in AuxiliaryHorizon),
        "lp_order": ("p0", "p1"),
        "zone_order": tuple(
            f"p{player}_{zone}"
            for player in (0, 1)
            for zone in TRANSITION_ZONE_NAMES
        ),
        "message_group_count": TRANSITION_MESSAGE_GROUP_COUNT,
        "visibility": "actor-visible transition plus public state deltas",
        "network_input": False,
        "ppo_loss_consumed": False,
        "standard_onnx_output": False,
        "fields": {
            name: {
                "shape": tuple(shape),
                "dtype": str(dtype).replace("torch.", ""),
            }
            for name, (shape, dtype) in AUXILIARY_TARGET_SPECS.items()
        },
    }
