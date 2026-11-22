# 构建训练与竞技场共用的复杂宏动作候选池

import struct

import numpy as np

import rule_bot
from data_types import ActionOperation, GameAction
from feature_encoder import MAX_ACTIONS
from game_constants import LocationInfo
from selection_protocol import CANCEL_RESPONSE, MATERIAL_PROPOSAL_REVISION, parse_selection_prompt


MACRO_ACTION_MSGS = frozenset({15, 18, 20, 21, 22, 23, 24, 25, 140, 141})
MODEL_ACTION_MSGS = frozenset(
    {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 140, 141, 142, 143}
)
MIN_MACRO_OPTION_WEIGHT = 1e-4
MODEL_CANDIDATE_PROPOSAL = f"model_pass1_v{MATERIAL_PROPOSAL_REVISION}"


def _freeze_action_value(value):
    """把动作与上下文中的嵌套容器转换为可哈希元组，供循环状态键稳定比较"""
    if isinstance(value, dict):
        return tuple((key, _freeze_action_value(item)) for key, item in sorted(value.items()))
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
        "prompt_flags": 0,
        "prompt_value": 0,
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
        prompt = parse_selection_prompt(msg_type, payload)
        constraints.update(
            selection_min=prompt["minimum"],
            selection_max=prompt["maximum"],
            context_value=prompt["target"],
            prompt_flags=prompt["mode"],
            prompt_value=len(prompt["mandatory"]),
        )
    elif msg_type in (21, 25, 140, 141) and len(payload) >= 2:
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
        _freeze_action_value(item)
        for item in snapshot.chain_stack
    )
    history_signature = tuple(
        _freeze_action_value(item)
        for item in snapshot.history_stack[-8:]
    )
    return (
        int(msg_type),
        global_signature,
        entity_signature,
        action_signature,
        chain_signature,
        history_signature,
        # 场面相同但询问用途或处理链变化时，不应被判为同一无进展循环
        int(getattr(snapshot, 'resolving_chain_index', 0)),
        _freeze_action_value(getattr(snapshot, 'recent_chain_targets', ())),
        _freeze_action_value(getattr(snapshot, 'prompt_context', ())),
    )


def _as_probabilities(action_probabilities):
    """将模型输出统一转换为一维双精度概率数组"""
    if hasattr(action_probabilities, "detach"):
        action_probabilities = action_probabilities.detach().cpu().numpy()
    return np.asarray(action_probabilities, dtype=np.float64).reshape(-1)


def _reduce_macro_options(scored, slots, rng, cover_sizes):
    """保留协议出口及不同有效张数，剩余位置仍按策略倾向无放回抽取"""
    if len(scored) <= slots:
        return [option for option, _ in scored]
    cancel = [i for i, (option, _) in enumerate(scored) if option.get('bytes') == CANCEL_RESPONSE]
    chosen = cancel[:1]
    remaining = [i for i, (option, _) in enumerate(scored) if option.get('bytes') != CANCEL_RESPONSE]
    slots -= len(chosen)

    def draw(indices, count, weights):
        weights = np.asarray(weights, dtype=np.float64)
        total = weights.sum()
        p = weights / total if np.isfinite(total) and total > 0 else np.full(len(indices), 1.0 / len(indices))
        return [indices[int(i)] for i in rng.choice(len(indices), size=count, replace=False, p=p)]

    by_size = {}
    if cover_sizes and slots:
        for i in remaining:
            # 同一询问必选张数固定，按响应总张数分组等价于按可选素材张数分组
            by_size.setdefault(scored[i][0]['bytes'][0], []).append(i)
    if len(by_size) > 1:
        sizes = list(by_size)
        if len(sizes) > slots:
            sizes = draw(sizes, slots, [max(scored[i][1] for i in by_size[size]) for size in sizes])
        for size in sizes:
            group = by_size[size]
            chosen.extend(group if len(group) == 1 else draw(group, 1, [scored[i][1] for i in group]))
        taken = set(chosen)
        remaining = [i for i in remaining if i not in taken]
        slots -= len(sizes)
    if slots:
        chosen.extend(draw(remaining, slots, [scored[i][1] for i in remaining]))
    return [scored[i][0] for i in chosen]


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
    if max_actions < 1 or option_limit < 1:
        raise ValueError("宏动作生成预算和动作槽数必须为正数")

    probabilities = _as_probabilities(action_probabilities)
    base_actions = list(base_actions)
    macro_constraints = _read_macro_constraints(msg_type, msg_payload)

    code_preferences = {}
    index_probabilities = np.zeros(256, dtype=np.float64)
    target_probabilities = {}

    for action_index, action in enumerate(base_actions):
        probability = float(probabilities[action_index]) if action_index < len(probabilities) else 0.0
        if not np.isfinite(probability) or probability < 0:
            probability = 0.0

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
        index_preferences=index_probabilities if msg_type in rule_bot.MATERIAL_SELECTION_MSGS else None,
    )
    if not options:
        return []

    scored_options = []
    for option in options:
        response = bytes(option.get("bytes", b""))
        # 给每个合法组合保留最低探索权重，避免低分选项被永久排除
        score = MIN_MACRO_OPTION_WEIGHT

        if response == CANCEL_RESPONSE:
            score += 0.05
        elif msg_type in rule_bot.MATERIAL_SELECTION_MSGS:
            # 原询问索引能区分同名素材和叠放层，必选占位不作为可选素材重复计分
            start = 1 + (macro_constraints['prompt_value'] if msg_type == 23 else 0)
            score += sum(index_probabilities[index] for index in response[start:])
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

    selected_options = _reduce_macro_options(scored_options, max_actions, rng if rng is not None else np.random,
                                            msg_type in rule_bot.MATERIAL_SELECTION_MSGS)

    macro_actions = []
    for pool_index, option in enumerate(selected_options):
        response = bytes(option.get("bytes", b""))
        response_value = option.get("value")
        description = "Cancel" if response == CANCEL_RESPONSE else f"Macro Action {pool_index}"
        if response == CANCEL_RESPONSE:
            operation = ActionOperation.CANCEL
        elif msg_type in (21, 25):
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
                prompt_flags=macro_constraints["prompt_flags"],
                prompt_value=macro_constraints["prompt_value"],
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
