'''全息回放记录器：保存双方决策、Core事件、场面快照与动作协议语义。'''

import datetime
import json
import os
import struct

from card_reader import card_db
from data_types import ActionOperation
from game_constants import LocationInfo, Phases, Zone


REPLAY_FORMAT_VERSION = 2
REPLAY_EVENT_MSGS = frozenset({
    5, 40, 41, 50, 53, 54, 60, 61, 62, 63, 64, 65,
    70, 71, 72, 73, 74, 75, 76, 83, 90, 91, 92, 93, 94,
    96, 97, 100, 101, 102, 110, 111, 112, 113, 114,
})

_ZONE_NAMES = {
    Zone.DECK: "卡组",
    Zone.HAND: "手牌",
    Zone.MZONE: "怪兽区",
    Zone.SZONE: "魔陷区",
    Zone.GRAVE: "墓地",
    Zone.REMOVED: "除外区",
    Zone.EXTRA: "额外卡组",
    Zone.OVERLAY: "超量素材",
}

_OPERATION_NAMES = {
    int(ActionOperation.DEFAULT): "默认操作",
    int(ActionOperation.YES): "确认",
    int(ActionOperation.NO): "拒绝",
    int(ActionOperation.OPTION): "选择选项",
    int(ActionOperation.SELECT): "选择",
    int(ActionOperation.UNSELECT): "取消选择",
    int(ActionOperation.FINISH): "完成选择",
    int(ActionOperation.CANCEL): "取消",
    int(ActionOperation.POSITION_ATTACK): "表侧攻击",
    int(ActionOperation.POSITION_ATTACK_DOWN): "里侧攻击",
    int(ActionOperation.POSITION_DEFENSE): "表侧守备",
    int(ActionOperation.POSITION_SET): "里侧守备",
    int(ActionOperation.SHUFFLE): "洗牌",
    int(ActionOperation.DIRECT_ATTACK): "直接攻击",
    int(ActionOperation.ATTACK): "攻击",
    int(ActionOperation.ACTIVATE): "发动/执行",
    int(ActionOperation.CHAIN): "连锁响应",
    int(ActionOperation.PHASE): "阶段切换",
    int(ActionOperation.PLACE): "选择区域",
    int(ActionOperation.ANNOUNCE): "宣言",
    int(ActionOperation.MACRO_SELECT): "组合选择",
    int(ActionOperation.MACRO_SORT): "排序",
    int(ActionOperation.REMOVE_COUNTER): "移除指示物",
}

_EXTRA_SUMMON_TYPES = (
    (0x4000000, "连接召唤"),
    (0x800000, "超量召唤"),
    (0x2000, "同调召唤"),
    (0x40, "融合召唤"),
)


def _pure_code(code):
    """移除卡密中的公开状态标记并转换为普通整数。"""
    return int(code or 0) & 0x7FFFFFFF


def _card_name(code):
    """安全读取卡名，数据库缺失时回退到卡密。"""
    pure_code = _pure_code(code)
    if not pure_code:
        return "未知卡片"
    try:
        return card_db.get_card_name(pure_code) or str(pure_code)
    except Exception:
        return str(pure_code)


def _get_special_summon_details(action):
    """按候选来源与卡片类型生成特殊召唤方式、卡名和可读说明。"""
    code = _pure_code(getattr(action, "code", 0))
    card_name = _card_name(code)
    raw_location = int(getattr(action, "target_location_raw", -1))
    location = 0
    if raw_location >= 0:
        try:
            _, location, _, _ = LocationInfo.decode(raw_location)
        except (TypeError, ValueError, struct.error):
            location = 0

    summon_method = "特殊召唤"
    if location == Zone.EXTRA and code:
        try:
            card_type = int(card_db.get_full_stats(code)[0] or 0)
        except Exception:
            card_type = 0
        summon_method = next(
            (name for type_flag, name in _EXTRA_SUMMON_TYPES if card_type & type_flag),
            "从额外卡组特殊召唤",
        )
    elif location:
        source = _ZONE_NAMES.get(location, f"区域 {location}")
        summon_method = f"从{source}特殊召唤"

    return summon_method, card_name, f"尝试{summon_method} {card_name}"


def _get_extra_monster_category(code):
    """返回额外卡组怪兽类别，避免把效果拉起误写成正式召唤方式。"""
    try:
        card_type = int(card_db.get_full_stats(_pure_code(code))[0] or 0)
    except Exception:
        return ""
    for type_flag, summon_method in _EXTRA_SUMMON_TYPES:
        if card_type & type_flag:
            return summon_method.replace("召唤", "怪兽")
    return ""


def _serialize_card(entity):
    """把卡片实体转换为可写入 JSON 的稳定结构。"""
    return {
        "code": _pure_code(getattr(entity, "code", 0)),
        "owner": int(getattr(entity, "owner", -1)),
        "loc": int(getattr(entity, "location", 0)),
        "seq": int(getattr(entity, "sequence", 0)),
        "pos": int(getattr(entity, "position", 0)),
        "atk": int(getattr(entity, "current_atk", 0)),
        "def": int(getattr(entity, "current_def", 0)),
        "lvl": int(getattr(entity, "level", 0)),
        "counters": int(getattr(entity, "counter_count", 0)),
        "overlays": int(getattr(entity, "overlay_count", 0)),
    }


def _serialize_snapshot(snapshot):
    """把完整快照整理为回放棋盘能够直接消费的状态。"""
    zones = {
        (player, location): []
        for player in (0, 1)
        for location in (
            Zone.HAND, Zone.MZONE, Zone.SZONE, Zone.GRAVE,
            Zone.EXTRA, Zone.REMOVED,
        )
    }
    for entity in snapshot.entities:
        key = (int(getattr(entity, "owner", -1)), int(getattr(entity, "location", 0)))
        if key in zones:
            zones[key].append(_serialize_card(entity))
    for cards in zones.values():
        cards.sort(key=lambda card: card["seq"])

    global_data = snapshot.global_data
    return {
        "to_play": int(getattr(global_data, "to_play", 0)),
        "p0_lp": int(getattr(global_data, "my_lp", 8000)),
        "p1_lp": int(getattr(global_data, "op_lp", 8000)),
        "p0_deck_len": len(getattr(snapshot, "p0_deck_codes", [])),
        "p1_deck_len": len(getattr(snapshot, "p1_deck_codes", [])),
        "p0_extra_len": len(getattr(snapshot, "p0_extra_codes", [])),
        "p1_extra_len": len(getattr(snapshot, "p1_extra_codes", [])),
        "p0_hand": zones[(0, Zone.HAND)],
        "p0_mzone": zones[(0, Zone.MZONE)],
        "p0_szone": zones[(0, Zone.SZONE)],
        "p0_grave": zones[(0, Zone.GRAVE)],
        "p0_extra": zones[(0, Zone.EXTRA)],
        "p0_removed": zones[(0, Zone.REMOVED)],
        "p1_hand": zones[(1, Zone.HAND)],
        "p1_mzone": zones[(1, Zone.MZONE)],
        "p1_szone": zones[(1, Zone.SZONE)],
        "p1_grave": zones[(1, Zone.GRAVE)],
        "p1_extra": zones[(1, Zone.EXTRA)],
        "p1_removed": zones[(1, Zone.REMOVED)],
        "chain": list(getattr(snapshot, "chain_stack", [])),
        "history": list(getattr(snapshot, "history_stack", [])),
    }


def _find_entity_reference(snapshot, owner, location, sequence, code=0, position=0):
    """按 Core 坐标查找卡片，并保留离场后仍可绘制的坐标信息。"""
    owner = int(owner)
    location = int(location)
    sequence = int(sequence)
    for entity in snapshot.entities:
        if (
            int(getattr(entity, "owner", -1)) == owner
            and int(getattr(entity, "location", 0)) == location
            and int(getattr(entity, "sequence", -1)) == sequence
        ):
            return _serialize_card(entity)
    return {
        "code": _pure_code(code),
        "owner": owner,
        "loc": location,
        "seq": sequence,
        "pos": int(position),
    }


def _reference_from_raw(snapshot, raw_location, code=0):
    """把 Core 四字节位置值转换为回放卡片引用。"""
    owner, location, sequence, position = LocationInfo.decode(int(raw_location))
    return _find_entity_reference(
        snapshot, owner, location, sequence, code=code, position=position
    )


def _reference_from_chain(snapshot, chain_item):
    """把连锁栈条目转换为回放卡片引用。"""
    if not chain_item:
        return None
    return _find_entity_reference(
        snapshot,
        chain_item.get("c", -1),
        chain_item.get("l", 0),
        chain_item.get("s", 0),
        code=chain_item.get("code", 0),
    )


def _describe_action(action, msg_type):
    """结合消息类型生成不会混淆战斗与主要阶段的动作说明。"""
    if getattr(action, "desc_str", ""):
        return str(action.desc_str)
    action_type = int(getattr(action, "action_type", -1))
    if msg_type == 11:
        card_name = _card_name(getattr(action, "code", 0))
        if action_type == 1:
            return _get_special_summon_details(action)[2]
        return {
            0: f"通常召唤 {card_name}", 2: f"改变 {card_name} 的表示形式",
            3: f"盖放怪兽 {card_name}", 4: f"盖放魔陷 {card_name}",
            5: f"发动 {card_name} 的效果",
            6: "进入战斗阶段", 7: "结束回合", 8: "洗牌",
        }.get(action_type, f"Type={action_type}")
    if msg_type == 10:
        return {
            0: "发动效果", 1: "攻击宣言", 2: "进入主要阶段2", 3: "结束回合",
        }.get(action_type, f"Type={action_type}")
    return {
        12: "效果确认", 13: "确认选择", 14: "选择选项",
        15: "选择卡片", 16: "选择连锁", 18: "选择区域",
        19: "选择表示形式", 20: "选择祭品", 22: "分配指示物",
        23: "选择合计值", 24: "禁用区域", 25: "卡片排序",
        26: "动态选择", 140: "宣言种族", 141: "宣言属性",
        142: "宣言卡名", 143: "宣言数字",
    }.get(int(msg_type or -1), f"Type={action_type}")


def _serialize_action(snapshot, action, msg_type):
    """记录动作协议 V2 字段，并推导棋盘高亮和箭头端点。"""
    target_entity_idx = int(getattr(action, "target_entity_idx", -1))
    primary_card = None
    if 0 <= target_entity_idx < len(snapshot.entities):
        primary_card = _serialize_card(snapshot.entities[target_entity_idx])

    operation_id = int(getattr(action, "operation_id", 0))
    action_type = int(getattr(action, "action_type", -1))
    actor = None
    targets = []

    # 主要/战斗阶段卡片动作、效果确认与连锁候选的主指针都是动作发起者。
    if primary_card and (
        msg_type in (10, 11, 12, 16)
        or operation_id in (
            int(ActionOperation.ACTIVATE), int(ActionOperation.CHAIN),
            int(ActionOperation.ATTACK), int(ActionOperation.DIRECT_ATTACK),
        )
    ):
        actor = primary_card
    elif primary_card:
        targets.append(primary_card)

    # Type 11 的候选可能来自尚未进入实体表的额外卡组，使用原始坐标和卡密补回发起卡。
    target_location_raw = int(getattr(action, "target_location_raw", -1))
    if msg_type == 11 and actor is None and target_location_raw >= 0:
        actor = _reference_from_raw(
            snapshot,
            target_location_raw,
            code=getattr(action, "code", 0),
        )

    if msg_type == 16 and actor and getattr(snapshot, "chain_stack", None):
        previous_chain = _reference_from_chain(snapshot, snapshot.chain_stack[-1])
        if previous_chain:
            targets = [previous_chain]

    macro_locations = list(getattr(action, "macro_target_locations", None) or [])
    macro_codes = list(getattr(action, "macro_target_codes", None) or [])
    for index, raw_location in enumerate(macro_locations):
        code = macro_codes[index] if index < len(macro_codes) else 0
        reference = _reference_from_raw(snapshot, raw_location, code=code)
        if reference not in targets:
            targets.append(reference)

    semantic = {
        "operation_id": operation_id,
        "operation": _OPERATION_NAMES.get(operation_id, f"Operation {operation_id}"),
        "response_value": getattr(action, "response_value", None),
        "decision_value": getattr(action, "decision_value", None),
        "decision_bytes": bytes(getattr(action, "decision_bytes", b"")).hex(),
        "target_location_raw": int(getattr(action, "target_location_raw", -1)),
        "selection_min": int(getattr(action, "selection_min", 0)),
        "selection_max": int(getattr(action, "selection_max", 0)),
        "selection_count": int(getattr(action, "selection_count", 0)),
        "finishable": bool(getattr(action, "finishable", False)),
        "cancelable": bool(getattr(action, "cancelable", False)),
        "context_value": int(getattr(action, "context_value", 0)),
        "prompt_flags": int(getattr(action, "prompt_flags", 0)),
        "prompt_value": int(getattr(action, "prompt_value", 0)),
        "prompt_value2": int(getattr(action, "prompt_value2", 0)),
        "target_codes": [int(value) for value in macro_codes],
        "target_locations": [int(value) for value in macro_locations],
        "target_values": [
            int(value) for value in (getattr(action, "macro_target_values", None) or [])
        ],
        "places": [int(value) for value in (getattr(action, "macro_places", None) or [])],
    }
    if msg_type == 11 and action_type in (0, 1, 2, 3, 4, 5):
        semantic["card_name"] = _card_name(getattr(action, "code", 0))
    if msg_type == 11 and action_type == 1:
        semantic["summon_method"] = _get_special_summon_details(action)[0]
    return {
        "index": int(getattr(action, "index", -1)),
        "desc": _describe_action(action, msg_type),
        "action_type": action_type,
        "actor": actor,
        "target": targets[0] if targets else None,
        "targets": targets,
        "semantic": semantic,
    }


def _build_core_event(snapshot, msg_type, payload):
    """把关键 Core 消息翻译为带方向、数值变化和多目标信息的回放事件。"""
    payload = bytes(payload)
    event = {
        "msg_type": int(msg_type),
        "kind": "core",
        "label": f"Core 消息 {msg_type}",
        "actor": None,
        "target": None,
        "targets": [],
        "movements": [],
    }

    if msg_type == 5 and len(payload) >= 2:
        winner, reason = struct.unpack("<BB", payload[:2])
        event.update(kind="win", label=f"P{winner} 获胜", player=winner, reason=reason)
    elif msg_type == 40 and payload:
        player = payload[0]
        event.update(kind="turn", label=f"P{player} 的新回合", player=player)
    elif msg_type == 41 and len(payload) >= 2:
        phase = struct.unpack("<H", payload[:2])[0]
        event.update(kind="phase", label=f"进入{Phases.get_str(phase)}", phase=phase)
    elif msg_type == 50 and len(payload) >= 16:
        code, old_raw, new_raw, reason = struct.unpack("<IIII", payload[:16])
        source = _reference_from_raw(snapshot, old_raw, code=code)
        target = _reference_from_raw(snapshot, new_raw, code=code)
        old_zone = _ZONE_NAMES.get(source["loc"], f"区域 {source['loc']}")
        new_zone = _ZONE_NAMES.get(target["loc"], f"区域 {target['loc']}")
        movement = {"code": _pure_code(code), "from": source, "to": target, "reason": reason}
        event.update(
            kind="move",
            label=f"{_card_name(code)}：{old_zone} → {new_zone}",
            actor=source,
            target=target,
            targets=[target],
            movements=[movement],
            reason=reason,
        )
    elif msg_type == 53 and len(payload) >= 9:
        code, owner, location, sequence, previous, current = struct.unpack(
            "<IBBBBB", payload[:9]
        )
        actor = _find_entity_reference(snapshot, owner, location, sequence, code=code)
        event.update(
            kind="position",
            label=f"{_card_name(code)} 改变表示形式",
            actor=actor,
            previous_position=previous,
            current_position=current,
        )
    elif msg_type in (54, 60, 62, 64) and len(payload) >= 8:
        code, raw_location = struct.unpack("<II", payload[:8])
        actor = _reference_from_raw(snapshot, raw_location, code=code)
        kind, verb = {
            54: ("set", "盖放"), 60: ("summon", "通常召唤"),
            62: ("special_summon", "特殊召唤"), 64: ("flip_summon", "反转召唤"),
        }[msg_type]
        if msg_type == 62:
            category = _get_extra_monster_category(code)
            if category:
                verb = f"特殊召唤 {category}"
        event.update(kind=kind, label=f"{verb} {_card_name(code)}", actor=actor)
    elif msg_type in (61, 63, 65):
        kind, label = {
            61: ("summoned", "通常召唤成功"),
            63: ("special_summoned", "特殊召唤成功"),
            65: ("flip_summoned", "反转召唤成功"),
        }[msg_type]
        event.update(kind=kind, label=label)
    elif msg_type == 70 and len(payload) >= 16:
        code = struct.unpack("<I", payload[:4])[0]
        owner, location, sequence = payload[8], payload[9], payload[10]
        desc = struct.unpack("<I", payload[11:15])[0]
        chain_index = payload[15]
        actor = _find_entity_reference(snapshot, owner, location, sequence, code=code)
        target = None
        chain_stack = list(getattr(snapshot, "chain_stack", []))
        if len(chain_stack) >= 2:
            target = _reference_from_chain(snapshot, chain_stack[-2])
        event.update(
            kind="chain",
            label=f"连锁 {chain_index}：{_card_name(code)} 发动",
            actor=actor,
            target=target,
            targets=[target] if target else [],
            chain_index=chain_index,
            desc_id=desc,
        )
    elif msg_type in (71, 72, 73, 75, 76) and payload:
        chain_index = payload[0]
        kind, text = {
            71: ("chained", "连锁建立"), 72: ("chain_solving", "开始处理连锁"),
            73: ("chain_solved", "连锁处理完成"), 75: ("chain_negated", "连锁发动无效"),
            76: ("chain_disabled", "连锁效果无效"),
        }[msg_type]
        event.update(kind=kind, label=f"{text} {chain_index}", chain_index=chain_index)
    elif msg_type == 74:
        event.update(kind="chain_end", label="连锁处理结束")
    elif msg_type == 83 and payload:
        count = payload[0]
        targets = []
        offset = 1
        for _ in range(count):
            if offset + 4 > len(payload):
                break
            raw_location = struct.unpack("<I", payload[offset:offset + 4])[0]
            targets.append(_reference_from_raw(snapshot, raw_location))
            offset += 4
        event.update(
            kind="target", label=f"指定 {len(targets)} 个对象",
            target=targets[0] if targets else None, targets=targets,
        )
    elif msg_type == 90 and len(payload) >= 2:
        player, count = payload[0], payload[1]
        codes = []
        for offset in range(2, min(len(payload), 2 + count * 4), 4):
            if offset + 4 <= len(payload):
                codes.append(_pure_code(struct.unpack("<I", payload[offset:offset + 4])[0]))
        event.update(kind="draw", label=f"P{player} 抽取 {count} 张卡", player=player, codes=codes)
    elif msg_type in (91, 92, 94, 100) and len(payload) >= 5:
        player, value = struct.unpack("<BI", payload[:5])
        current_lp = int(getattr(snapshot.global_data, "my_lp" if player == 0 else "op_lp", 0))
        if msg_type in (91, 100):
            delta = -int(value)
            label = f"P{player} {'支付' if msg_type == 100 else '受到'} {value} LP"
        elif msg_type == 92:
            delta = int(value)
            label = f"P{player} 回复 {value} LP"
        else:
            delta = None
            label = f"P{player} LP 更新为 {value}"
            current_lp = int(value)
        event.update(
            kind="lp", label=label, player=player, lp_delta=delta,
            lp_after=current_lp,
            lp_before=(current_lp - delta) if delta is not None else None,
        )
    elif msg_type == 93 and len(payload) >= 8:
        equip_raw, target_raw = struct.unpack("<II", payload[:8])
        actor = _reference_from_raw(snapshot, equip_raw)
        target = _reference_from_raw(snapshot, target_raw)
        event.update(
            kind="equip", label="装备卡指定对象", actor=actor,
            target=target, targets=[target],
        )
    elif msg_type in (96, 97) and len(payload) >= 8:
        source_raw, target_raw = struct.unpack("<II", payload[:8])
        actor = _reference_from_raw(snapshot, source_raw)
        target = _reference_from_raw(snapshot, target_raw)
        event.update(
            kind="card_target" if msg_type == 96 else "card_target_cancel",
            label="建立卡片对象关系" if msg_type == 96 else "解除卡片对象关系",
            actor=actor, target=target, targets=[target],
        )
    elif msg_type in (101, 102) and len(payload) >= 7:
        counter_type, owner, location, sequence, count = struct.unpack("<HBBBH", payload[:7])
        actor = _find_entity_reference(snapshot, owner, location, sequence)
        verb = "增加" if msg_type == 101 else "移除"
        event.update(
            kind="counter", label=f"{verb} {count} 个指示物",
            actor=actor, counter_type=counter_type, counter_count=count,
        )
    elif msg_type == 110 and len(payload) >= 8:
        attacker_raw, defender_raw = struct.unpack("<II", payload[:8])
        actor = _reference_from_raw(snapshot, attacker_raw)
        direct_attack = defender_raw == 0
        target = (
            {
                "code": 0,
                "owner": 1 - int(actor.get("owner", 0)),
                "loc": 0x100,
                "seq": 0,
                "pos": 0,
            }
            if direct_attack
            else _reference_from_raw(snapshot, defender_raw)
        )
        event.update(
            kind="attack",
            label=(f"{_card_name(actor['code'])} 直接攻击" if direct_attack else f"{_card_name(actor['code'])} 发起攻击"),
            actor=actor,
            target=target,
            targets=[target],
            direct=direct_attack,
        )
    elif msg_type in (111, 112, 113, 114):
        kind, label = {
            111: ("battle", "战斗数值结算"), 112: ("damage_step_start", "进入伤害步骤"),
            113: ("damage_calculation", "进行伤害计算"), 114: ("damage_step_end", "伤害步骤结束"),
        }[msg_type]
        event.update(kind=kind, label=label)

    return event


class AIThoughtLogger:
    """记录竞技场双方决策和 Core 事件，生成全息回放 V2 文件。"""

    def __init__(self, player_name="Galatea_AI", opponent_name="RuleBot"):
        self.player_name = player_name
        self.opponent_name = opponent_name
        self.thoughts = []
        self.states = []
        self._state_ids = {}
        self.decklists = {}
        self.is_active = False

    def start_recording(self):
        """开始一局新录像并清空上一局缓存。"""
        self.thoughts = []
        self.states = []
        self._state_ids = {}
        self.decklists = {}
        self.is_active = True

    def set_decklists(
        self,
        p0_main,
        p0_extra,
        p1_main,
        p1_extra,
        p0_name="",
        p1_name="",
    ):
        """为当前录像保存双方开局携带卡组，避免在每个状态重复写入。"""
        if not self.is_active:
            return

        def normalize_codes(values):
            """把卡组输入整理为可序列化的公开卡密列表。"""
            result = []
            for value in values or []:
                try:
                    code = _pure_code(value)
                except (TypeError, ValueError):
                    continue
                if code > 0:
                    result.append(code)
            return result

        self.decklists = {
            "0": {
                "name": str(p0_name or ""),
                "main": normalize_codes(p0_main),
                "extra": normalize_codes(p0_extra),
            },
            "1": {
                "name": str(p1_name or ""),
                "main": normalize_codes(p1_main),
                "extra": normalize_codes(p1_extra),
            },
        }

    def _register_state(self, snapshot):
        """去重完整场面，只让时间轴帧保存轻量状态编号。"""
        state = _serialize_snapshot(snapshot)
        state_key = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        state_id = self._state_ids.get(state_key)
        if state_id is None:
            state_id = len(self.states)
            self._state_ids[state_key] = state_id
            self.states.append(state)
        return state_id

    def log_decision(
        self,
        turn,
        phase_id,
        snapshot,
        probs,
        chosen_index,
        player_id=0,
        msg_type=None,
        agent_name=None,
        source="model",
    ):
        """记录任意一方的完整候选、选择结果和动作协议 V2 语义。"""
        if not self.is_active:
            return
        try:
            options = []
            for index, action in enumerate(snapshot.valid_actions):
                option = _serialize_action(snapshot, action, msg_type)
                probability = 0.0
                if probs is not None and index < len(probs):
                    value = probs[index]
                    probability = float(value.item()) if hasattr(value, "item") else float(value)
                option.update(
                    confidence=probability,
                    is_chosen=(index == int(chosen_index)),
                )
                options.append(option)
            options.sort(key=lambda option: option["confidence"], reverse=True)
            self.thoughts.append({
                "frame_type": "decision",
                "turn": int(turn),
                "phase": Phases.get_str(phase_id),
                "phase_id": int(phase_id),
                "player": int(player_id),
                "agent": agent_name or (self.player_name if int(player_id) == 0 else self.opponent_name),
                "source": source,
                "msg_type": int(msg_type) if msg_type is not None else None,
                "state_id": self._register_state(snapshot),
                "options": options,
            })
        except Exception as error:
            print(f"\n[Logger Error] 决策记录失败: {error}")

    def log_external_decision(
        self,
        turn,
        phase_id,
        snapshot,
        msg_type,
        response,
        player_id,
        agent_name="RuleBot",
        chosen_index=None,
    ):
        """记录没有模型概率分布的规则对手决策。"""
        if not self.is_active:
            return
        try:
            response_text = response.hex() if isinstance(response, bytes) else str(response)
            options = []
            if chosen_index is not None and 0 <= int(chosen_index) < len(snapshot.valid_actions):
                for index, action in enumerate(snapshot.valid_actions):
                    option = _serialize_action(snapshot, action, msg_type)
                    option.update(
                        confidence=1.0 if index == int(chosen_index) else 0.0,
                        is_chosen=(index == int(chosen_index)),
                    )
                    options.append(option)
            else:
                options.append({
                    "index": -1,
                    "desc": f"RuleBot 响应 {response_text}",
                    "confidence": 1.0,
                    "is_chosen": True,
                    "action_type": int(msg_type),
                    "actor": None,
                    "target": None,
                    "targets": [],
                    "semantic": {"raw_response": response_text},
                })
            self.thoughts.append({
                "frame_type": "decision",
                "turn": int(turn),
                "phase": Phases.get_str(phase_id),
                "phase_id": int(phase_id),
                "player": int(player_id),
                "agent": agent_name,
                "source": "rule",
                "msg_type": int(msg_type),
                "state_id": self._register_state(snapshot),
                "options": options,
            })
        except Exception as error:
            print(f"\n[Logger Error] 规则决策记录失败: {error}")

    def log_core_event(self, turn, phase_id, snapshot, msg_type, payload):
        """记录会改变场面或解释动作结果的 Core 时间轴事件。"""
        if not self.is_active or msg_type not in REPLAY_EVENT_MSGS:
            return
        try:
            event = _build_core_event(snapshot, msg_type, payload)
            self.thoughts.append({
                "frame_type": "event",
                "turn": int(turn),
                "phase": Phases.get_str(phase_id),
                "phase_id": int(phase_id),
                "player": event.get("player"),
                "msg_type": int(msg_type),
                "state_id": self._register_state(snapshot),
                "event": event,
                "options": [],
            })
        except Exception as error:
            print(f"\n[Logger Error] Core 事件记录失败: {error}")

    def save(self, winner_id, game_idx, win_reason="正常结束"):
        """保存完整录像；没有有效帧时不创建空文件。"""
        if not self.is_active:
            return None
        self.is_active = False
        if not self.thoughts:
            return None
        os.makedirs("./ai_thoughts", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"./ai_thoughts/Game{game_idx}_{timestamp}_P{winner_id}Win.json"
        replay = {
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "model_name": self.player_name,
            "opponent_name": self.opponent_name,
            "players": {"0": self.player_name, "1": self.opponent_name},
            "winner": int(winner_id),
            "win_reason": win_reason,
            "decklists": self.decklists,
            "states": self.states,
            "frames": self.thoughts,
        }
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(replay, file, ensure_ascii=False, separators=(",", ":"))
        return filepath
