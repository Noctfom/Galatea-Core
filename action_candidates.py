# 构建训练与竞技场共用的复杂宏动作候选池

import struct

import numpy as np

import rule_bot
from data_types import ActionOperation, GameAction
from feature_encoder import MAX_ACTIONS
from game_constants import LocationInfo


MACRO_ACTION_MSGS = frozenset({15, 18, 20, 22, 23, 24, 25, 140, 141})
MODEL_ACTION_MSGS = frozenset(
    {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 140, 141, 142, 143}
)
_CANCEL_RESPONSE = b"\xff\xff\xff\xff"
MIN_MACRO_OPTION_WEIGHT = 1e-4


def _freeze_action_value(value):
    """把动作中的嵌套列表转换为可哈希元组，供循环状态键稳定比较"""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_action_value(item) for item in value)
    return value


def _read_macro_constraints(msg_type, msg_payload):
    """提取复杂消息的数量约束，供每个最终宏动作共享"""
    payload = bytes(msg_payload)
    constraints = {
        "selection_min": 0,
        "selection_max": 0,
        "cancelable": False,
        "context_value": 0,
    }
    if msg_type in (15, 20) and len(payload) >= 4:
        constraints.update(
            selection_min=payload[2],
            selection_max=payload[3],
            cancelable=bool(payload[1]),
        )
    elif msg_type in (18, 24) and len(payload) >= 2:
        constraints.update(selection_min=payload[1], selection_max=payload[1])
    elif msg_type == 22 and len(payload) >= 5:
        quantity = struct.unpack('<H', payload[3:5])[0]
        constraints.update(context_value=quantity)
    elif msg_type == 23 and len(payload) >= 8:
        constraints.update(selection_min=payload[6], selection_max=payload[7])
    elif msg_type in (25, 140, 141) and len(payload) >= 2:
        constraints.update(selection_min=payload[1], selection_max=payload[1])
        if msg_type in (140, 141):
            constraints["context_value"] = payload[1]
    return constraints


def build_action_state_key(snapshot, msg_type):
    """构造包含场面与完整候选身份的状态键，供训练和竞技场识别真实循环"""
    global_signature = tuple(vars(snapshot.global_data).values())
    entity_signature = tuple(
        (
            entity.code,
            entity.owner,
            entity.location,
            entity.sequence,
            entity.position,
            entity.current_atk,
            entity.current_def,
            entity.counter_count,
            entity.overlay_count,
            entity.is_equipped,
            entity.used_effect_mask,
        )
        for entity in snapshot.entities
    )
    action_signature = tuple(
        (
            action.action_type,
            action.index,
            action.target_entity_idx,
            action.desc_str,
            action.desc_id,
            getattr(action, "code", 0),
            bytes(getattr(action, "decision_bytes", b"")),
            getattr(action, "decision_value", None),
            getattr(action, "operation_id", 0),
            getattr(action, "response_value", None),
            getattr(action, "target_location_raw", -1),
            getattr(action, "selection_min", 0),
            getattr(action, "selection_max", 0),
            getattr(action, "selection_count", 0),
            bool(getattr(action, "finishable", False)),
            bool(getattr(action, "cancelable", False)),
            getattr(action, "context_value", 0),
            getattr(action, "prompt_flags", 0),
            getattr(action, "prompt_value", 0),
            getattr(action, "prompt_value2", 0),
            _freeze_action_value(getattr(action, "macro_targets", None) or ()),
            _freeze_action_value(getattr(action, "macro_places", None) or ()),
            _freeze_action_value(getattr(action, "macro_target_codes", None) or ()),
            _freeze_action_value(getattr(action, "macro_target_values", None) or ()),
            _freeze_action_value(getattr(action, "macro_target_locations", None) or ()),
        )
        for action in snapshot.valid_actions
    )
    chain_signature = tuple(
        tuple(sorted(item.items()))
        for item in snapshot.chain_stack
    )
    history_signature = tuple(
        tuple(sorted(item.items()))
        for item in snapshot.history_stack[-8:]
    )
    return (
        int(msg_type),
        global_signature,
        entity_signature,
        action_signature,
        chain_signature,
        history_signature,
    )


def _as_probabilities(action_probabilities):
    """将模型输出统一转换为一维双精度概率数组"""
    if hasattr(action_probabilities, "detach"):
        action_probabilities = action_probabilities.detach().cpu().numpy()
    return np.asarray(action_probabilities, dtype=np.float64).reshape(-1)


def build_macro_action_pool(
    msg_type,
    msg_payload,
    brain,
    base_actions,
    action_probabilities,
    *,
    option_limit=5000,
    max_actions=MAX_ACTIONS,
    rng=None,
):
    """按策略权重随机缩减复杂选择池，并生成可直接提交给引擎的宏动作

    返回动作保留原始位置标识，调用方需重新生成快照，将其映射为实体索引
    """
    if msg_type not in MACRO_ACTION_MSGS:
        raise ValueError(f"message type {msg_type} is not a macro action prompt")

    probabilities = _as_probabilities(action_probabilities)
    base_actions = list(base_actions)
    macro_constraints = _read_macro_constraints(msg_type, msg_payload)

    code_preferences = {}
    index_probabilities = np.zeros(256, dtype=np.float64)
    target_probabilities = {}

    for action_index, action in enumerate(base_actions):
        probability = float(probabilities[action_index]) if action_index < len(probabilities) else 0.0

        code = getattr(action, "code", 0)
        if code:
            code_preferences[code] = max(code_preferences.get(code, 0.0), probability)

        if 0 <= action.index < len(index_probabilities):
            index_probabilities[action.index] = max(
                index_probabilities[action.index], probability
            )

        target = getattr(action, "target_entity_idx", -1)
        if target >= 0:
            controller, location, sequence, _ = LocationInfo.decode(target)
            location_key = (controller, location, sequence)
            target_probabilities[location_key] = max(
                target_probabilities.get(location_key, 0.0), probability
            )

    options = rule_bot.get_macro_options(
        msg_type,
        msg_payload,
        brain,
        limit=option_limit,
        pref_weights=code_preferences,
    )
    if not options:
        return []

    scored_options = []
    for option in options:
        response = bytes(option.get("bytes", b""))
        response_value = option.get("value")
        # 给每个合法组合保留最低探索权重，避免低分选项被永久排除
        score = MIN_MACRO_OPTION_WEIGHT

        if response == _CANCEL_RESPONSE:
            score += 0.05
        elif option.get("places"):
            score += sum(
                index_probabilities[place]
                for place in option["places"]
                if 0 <= place < len(index_probabilities)
            )
        elif option.get("locs"):
            for raw_location in option["locs"]:
                controller, location, sequence, _ = LocationInfo.decode(raw_location)
                score += target_probabilities.get(
                    (controller, location, sequence), 0.0
                )
        elif option.get("indices"):
            score += sum(
                index_probabilities[index]
                for index in option["indices"]
                if 0 <= index < len(index_probabilities)
            )
        elif option.get("codes"):
            score += sum(
                code_preferences.get(code, 0.0)
                for code in option["codes"]
            )
        elif len(response) > 1:
            # Last-resort support for index-based response formats.
            response_indices = np.frombuffer(response, dtype=np.uint8, offset=1)
            score += index_probabilities[response_indices].sum()

        scored_options.append((option, score))

    if len(scored_options) > max_actions:
        weights = np.asarray([score for _, score in scored_options], dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            weights = np.full(len(scored_options), 1.0 / len(scored_options))
        else:
            weights /= weight_sum

        chooser = rng if rng is not None else np.random
        selected_indices = chooser.choice(
            len(scored_options),
            size=max_actions,
            replace=False,
            p=weights,
        )
        selected_options = [scored_options[int(index)][0] for index in selected_indices]
    else:
        selected_options = [option for option, _ in scored_options]

    macro_actions = []
    for pool_index, option in enumerate(selected_options):
        response = bytes(option.get("bytes", b""))
        response_value = option.get("value")
        description = "Cancel" if response == _CANCEL_RESPONSE else f"Macro Action {pool_index}"
        if response == _CANCEL_RESPONSE:
            operation = ActionOperation.CANCEL
        elif msg_type == 25:
            operation = ActionOperation.MACRO_SORT
        elif msg_type == 22:
            operation = ActionOperation.REMOVE_COUNTER
        elif msg_type in (18, 24):
            operation = ActionOperation.PLACE
        elif msg_type in (140, 141):
            operation = ActionOperation.ANNOUNCE
        else:
            operation = ActionOperation.MACRO_SELECT

        locations = list(option.get("locs", []))
        places = list(option.get("places", []))
        codes = list(option.get("codes", []))
        values = list(option.get("values", []))
        selected_count = len(locations) or len(places) or len(option.get("indices", []))
        macro_actions.append(
            GameAction(
                action_type=msg_type,
                index=pool_index,
                desc_str=description,
                operation_id=int(operation),
                response_value=response_value,
                target_location_raw=locations[0] if locations else -1,
                selection_min=macro_constraints["selection_min"],
                selection_max=macro_constraints["selection_max"],
                selection_count=selected_count,
                cancelable=macro_constraints["cancelable"],
                context_value=macro_constraints["context_value"],
                macro_targets=locations or None,
                macro_places=places or None,
                macro_target_codes=codes or None,
                macro_target_values=values or None,
                macro_target_locations=locations or None,
                decision_bytes=response,
                decision_value=response_value,
            )
        )

    return macro_actions
