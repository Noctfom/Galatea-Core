"""维护决策之间的公开状态转移事件，并执行生命周期与可见性校验"""

from dataclasses import dataclass, replace
from typing import Iterable, Optional

from data_types import (
    ACTION_TARGET_SLOTS,
    ActionOperation,
    GameAction,
    StateTransitionEvent,
    SummonMethod,
    TRANSITION_EVENT_HISTORY_SIZE,
    TRANSITION_ZONE_COUNT,
    TransitionBoundary,
    TransitionResultFlag,
)
from game_constants import Zone


TRANSITION_EVENT_FORMAT_VERSION = 1
MAX_RECORDED_MESSAGE_TYPES = 32
ALL_PLAYERS_VISIBILITY_MASK = 0b11
TRANSITION_ZONE_ORDER = (
    Zone.DECK,
    Zone.HAND,
    Zone.MZONE,
    Zone.SZONE,
    Zone.GRAVE,
    Zone.REMOVED,
    Zone.EXTRA,
    Zone.OVERLAY,
)
TRANSITION_ZONE_NAMES = (
    "deck",
    "hand",
    "monster",
    "spell_trap",
    "grave",
    "removed",
    "extra",
    "overlay",
)

_PUBLIC_INTENT_OPERATIONS = {
    int(ActionOperation.NORMAL_SUMMON),
    int(ActionOperation.TRIBUTE_SUMMON),
    int(ActionOperation.SPECIAL_SUMMON),
    int(ActionOperation.CHANGE_POSITION),
    int(ActionOperation.MONSTER_SET),
    int(ActionOperation.SPELL_TRAP_SET),
    int(ActionOperation.ACTIVATE),
    int(ActionOperation.CHAIN),
    int(ActionOperation.PHASE),
    int(ActionOperation.DIRECT_ATTACK),
    int(ActionOperation.ATTACK),
}
_PUBLIC_SOURCE_MESSAGES = {60, 62, 64, 70}
_PUBLIC_EVENT_MESSAGES = {
    40, 41, 50, 53, 54, 55, 56,
    60, 61, 62, 63, 64, 65,
    70, 71, 72, 73, 74, 75, 76,
    83, 90, 91, 92, 93, 94, 95, 96, 97,
    100, 101, 102, 110, 111, 112, 113, 114,
}


@dataclass(frozen=True)
class TransitionStateDigest:
    """保存事件边界所需的轻量状态摘要"""

    lp_p0: int
    lp_p1: int
    zone_counts: tuple
    chain_depth: int


@dataclass
class _PendingTransition:
    sequence_id: int
    actor: int
    turn_count: int
    phase_id: int
    prompt_type: int
    operation_id: int
    summon_method_id: int
    source_code: int
    source_location_raw: int
    effect_slot: int
    target_locations: tuple
    target_count: int
    event_visibility_mask: int
    intent_visibility_mask: int
    source_visibility_mask: int
    target_visibility_mask: int
    start_digest: TransitionStateDigest
    result_flags: int
    message_types: list
    message_count: int
    max_chain_depth: int
    boundary: Optional[int] = None


class TransitionEventRecorder:
    """按动作提交和下一决策边界聚合最近 16 个状态转移事件"""

    def __init__(self):
        self._history = []
        self._pending = None
        self._next_sequence_id = 0

    @property
    def has_pending(self):
        return self._pending is not None

    @property
    def has_pending_boundary(self):
        return self._pending is not None and self._pending.boundary is not None

    def reset(self):
        """清空单局事件与未完成转移"""
        self._history.clear()
        self._pending = None
        self._next_sequence_id = 0

    def begin(
        self,
        *,
        actor,
        turn_count,
        phase_id,
        prompt_type,
        action,
        start_digest,
        source_visibility_mask,
        target_visibility_mask,
    ):
        """在动作已成功提交给 Core 后开启一个转移事件"""
        if self._pending is not None:
            raise RuntimeError("previous transition event has not reached a boundary")
        if actor not in (0, 1):
            raise ValueError(f"transition actor must be 0 or 1, got {actor}")

        action = action or GameAction(action_type=0, index=-1)
        operation_id = int(getattr(action, "operation_id", ActionOperation.DEFAULT))
        summon_method_id = int(
            getattr(action, "summon_method_id", SummonMethod.NONE)
        )
        actor_mask = 1 << actor
        intent_visibility = (
            ALL_PLAYERS_VISIBILITY_MASK
            if operation_id in _PUBLIC_INTENT_OPERATIONS
            else actor_mask
        )
        target_locations = tuple(
            int(value)
            for value in (getattr(action, "macro_target_locations", None) or ())
        )
        if not target_locations:
            location = int(getattr(action, "target_location_raw", -1))
            if location >= 0:
                target_locations = (location,)
        target_count = len(target_locations)
        target_locations = target_locations[:ACTION_TARGET_SLOTS]

        result_flags = TransitionResultFlag.NONE
        if operation_id == int(ActionOperation.CANCEL):
            result_flags |= TransitionResultFlag.CANCELLED
        elif operation_id == int(ActionOperation.NO):
            result_flags |= TransitionResultFlag.DECLINED
        elif operation_id == int(ActionOperation.FINISH):
            result_flags |= TransitionResultFlag.SELECTION_FINISHED

        self._pending = _PendingTransition(
            sequence_id=self._next_sequence_id,
            actor=int(actor),
            turn_count=int(turn_count),
            phase_id=int(phase_id),
            prompt_type=int(prompt_type),
            operation_id=operation_id,
            summon_method_id=summon_method_id,
            source_code=int(getattr(action, "code", 0)) & 0x7FFFFFFF,
            source_location_raw=int(getattr(action, "target_location_raw", -1)),
            effect_slot=int(getattr(action, "effect_slot", -1)),
            target_locations=target_locations,
            target_count=target_count,
            event_visibility_mask=int(intent_visibility),
            intent_visibility_mask=int(intent_visibility),
            source_visibility_mask=int(source_visibility_mask | actor_mask),
            target_visibility_mask=int(target_visibility_mask | actor_mask),
            start_digest=start_digest,
            result_flags=int(result_flags),
            message_types=[],
            message_count=0,
            max_chain_depth=int(start_digest.chain_depth),
        )
        self._next_sequence_id += 1

    def observe_message(self, msg_type, payload, current_chain_depth):
        """吸收动作之后、下一决策之前的 Core 消息标签"""
        pending = self._pending
        if pending is None or pending.boundary is not None:
            return
        msg_type = int(msg_type)
        if msg_type not in _PUBLIC_EVENT_MESSAGES:
            return
        pending.event_visibility_mask = ALL_PLAYERS_VISIBILITY_MASK
        pending.message_count += 1
        if msg_type not in pending.message_types:
            if len(pending.message_types) < MAX_RECORDED_MESSAGE_TYPES:
                pending.message_types.append(msg_type)
            else:
                pending.result_flags |= int(TransitionResultFlag.MESSAGE_OVERFLOW)
        pending.max_chain_depth = max(
            pending.max_chain_depth,
            int(current_chain_depth),
        )
        if msg_type == 75:
            pending.result_flags |= int(TransitionResultFlag.CHAIN_NEGATED)
        elif msg_type == 76:
            pending.result_flags |= int(TransitionResultFlag.CHAIN_DISABLED)

        # 召唤与连锁消息会公开来源卡；只在编号确实匹配时提升可见性
        if msg_type in _PUBLIC_SOURCE_MESSAGES and len(payload) >= 4:
            revealed_code = int.from_bytes(payload[:4], "little") & 0x7FFFFFFF
            if revealed_code and (
                pending.source_code == 0 or pending.source_code == revealed_code
            ):
                pending.source_code = revealed_code
                pending.source_visibility_mask = ALL_PLAYERS_VISIBILITY_MASK

    def mark_boundary(self, boundary):
        """登记下一决策、重试或终局边界，等待状态查询校准后结算"""
        if self._pending is None:
            return
        boundary = int(boundary)
        if boundary not in {int(item) for item in TransitionBoundary}:
            raise ValueError(f"unknown transition boundary: {boundary}")
        self._pending.boundary = boundary
        if boundary == int(TransitionBoundary.RETRY):
            self._pending.result_flags |= int(TransitionResultFlag.RETRY)
        elif boundary == int(TransitionBoundary.TERMINAL):
            self._pending.result_flags |= int(TransitionResultFlag.TERMINAL)

    def finalize(self, end_digest):
        """用同步后的状态摘要完成已抵达边界的事件"""
        pending = self._pending
        if pending is None or pending.boundary is None:
            return None
        zone_deltas = tuple(
            int(after) - int(before)
            for before, after in zip(
                pending.start_digest.zone_counts,
                end_digest.zone_counts,
            )
        )
        lp_delta_p0 = int(end_digest.lp_p0) - int(pending.start_digest.lp_p0)
        lp_delta_p1 = int(end_digest.lp_p1) - int(pending.start_digest.lp_p1)
        flags = int(pending.result_flags)
        if pending.boundary != int(TransitionBoundary.RETRY):
            flags |= int(TransitionResultFlag.RESOLVED)
        if (
            lp_delta_p0
            or lp_delta_p1
            or any(zone_deltas)
            or int(end_digest.chain_depth) != int(pending.start_digest.chain_depth)
            or pending.message_count
        ):
            flags |= int(TransitionResultFlag.STATE_CHANGED)
            pending.event_visibility_mask = ALL_PLAYERS_VISIBILITY_MASK

        event = StateTransitionEvent(
            sequence_id=pending.sequence_id,
            actor=pending.actor,
            turn_count=pending.turn_count,
            phase_id=pending.phase_id,
            prompt_type=pending.prompt_type,
            operation_id=pending.operation_id,
            summon_method_id=pending.summon_method_id,
            source_code=pending.source_code,
            source_location_raw=pending.source_location_raw,
            effect_slot=pending.effect_slot,
            target_locations=pending.target_locations,
            target_count=pending.target_count,
            event_visibility_mask=pending.event_visibility_mask,
            intent_visibility_mask=pending.intent_visibility_mask,
            source_visibility_mask=pending.source_visibility_mask,
            target_visibility_mask=pending.target_visibility_mask,
            boundary=int(pending.boundary),
            result_flags=flags,
            message_types=tuple(pending.message_types),
            message_count=pending.message_count,
            lp_delta_p0=lp_delta_p0,
            lp_delta_p1=lp_delta_p1,
            zone_deltas=zone_deltas,
            chain_depth_before=int(pending.start_digest.chain_depth),
            chain_depth_after=int(end_digest.chain_depth),
            max_chain_depth=max(
                pending.max_chain_depth,
                int(end_digest.chain_depth),
            ),
        )
        self._history.append(event)
        if len(self._history) > TRANSITION_EVENT_HISTORY_SIZE:
            self._history.pop(0)
        self._pending = None
        return event

    def snapshot(self):
        """返回按时间从旧到新排列的不可变事件副本"""
        return list(self._history)


def mask_transition_event_for_player(event, player_id):
    """按玩家视角抹除事件中的私有意图、来源和目标字段"""
    if player_id not in (0, 1):
        raise ValueError(f"player_id must be 0 or 1, got {player_id}")
    player_mask = 1 << player_id
    if not (event.event_visibility_mask & player_mask):
        return None
    updates = {}
    if not (event.intent_visibility_mask & player_mask):
        updates.update(
            prompt_type=0,
            operation_id=int(ActionOperation.DEFAULT),
            summon_method_id=int(SummonMethod.NONE),
            effect_slot=-1,
        )
    if not (event.source_visibility_mask & player_mask):
        updates.update(source_code=0, source_location_raw=-1)
    if not (event.target_visibility_mask & player_mask):
        updates.update(target_locations=(), target_count=0)
    return replace(event, **updates) if updates else event


def visible_transition_history(events, player_id):
    """过滤对指定玩家完全不可见的内部选择，并抹除字段级隐私"""
    visible = []
    for event in events:
        masked = mask_transition_event_for_player(event, player_id)
        if masked is not None:
            visible.append(masked)
    return visible[-TRANSITION_EVENT_HISTORY_SIZE:]


def validate_transition_history(events: Iterable[StateTransitionEvent]):
    """校验事件顺序、边界、长度及可见性掩码，发现污染时立即拒绝"""
    events = list(events)
    if len(events) > TRANSITION_EVENT_HISTORY_SIZE:
        raise ValueError("transition history exceeds its fixed capacity")
    previous_sequence = -1
    for event in events:
        if event.sequence_id <= previous_sequence:
            raise ValueError("transition sequence ids must be strictly increasing")
        previous_sequence = event.sequence_id
        if event.actor not in (0, 1):
            raise ValueError("transition actor is invalid")
        actor_mask = 1 << event.actor
        for mask in (
            event.event_visibility_mask,
            event.intent_visibility_mask,
            event.source_visibility_mask,
            event.target_visibility_mask,
        ):
            if mask & ~ALL_PLAYERS_VISIBILITY_MASK or not (mask & actor_mask):
                raise ValueError("transition visibility mask is invalid")
        if event.boundary not in {int(item) for item in TransitionBoundary}:
            raise ValueError("transition boundary is invalid")
        if len(event.zone_deltas) != 2 * TRANSITION_ZONE_COUNT:
            raise ValueError("transition zone delta width is invalid")
        if len(event.target_locations) > ACTION_TARGET_SLOTS:
            raise ValueError("transition target capacity is invalid")
        if len(event.message_types) > MAX_RECORDED_MESSAGE_TYPES:
            raise ValueError("transition message label capacity is invalid")
        if event.message_count < len(event.message_types):
            raise ValueError("transition message count is inconsistent")
    return True


def get_transition_event_protocol_descriptor():
    """返回独立于模型结构修订的 3.11.1 内部事件协议说明"""
    return {
        "format_version": TRANSITION_EVENT_FORMAT_VERSION,
        "history_size": TRANSITION_EVENT_HISTORY_SIZE,
        "zone_count_per_player": TRANSITION_ZONE_COUNT,
        "zone_order": TRANSITION_ZONE_NAMES,
        "target_slots": ACTION_TARGET_SLOTS,
        "message_type_slots": MAX_RECORDED_MESSAGE_TYPES,
        "order": "oldest_to_newest",
        "visibility": "field_level_player_bitmask",
        "network_consumed": False,
    }
