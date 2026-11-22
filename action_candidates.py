# 构建训练与竞技场共用的复杂宏动作候选池

import struct
from typing import NamedTuple

import numpy as np

import rule_bot
from data_types import ActionOperation, GameAction
from feature_encoder import MAX_ACTIONS, entity_identity_visible, visible_chain_target
from game_constants import LocationInfo
from selection_protocol import CANCEL_RESPONSE, MATERIAL_PROPOSAL_REVISION, parse_selection_prompt


MACRO_ACTION_MSGS = frozenset({15, 18, 20, 21, 22, 23, 24, 25, 140, 141})
MODEL_ACTION_MSGS = frozenset(
    {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 140, 141, 142, 143}
)
MIN_MACRO_OPTION_WEIGHT = 1e-4
MODEL_CANDIDATE_PROPOSAL = f"model_pass1_v{MATERIAL_PROPOSAL_REVISION}"


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


class ActionStateKey(NamedTuple):
    """单局内同一座位的可见场面、询问用途和有序候选"""
    msg_type: int
    player: int
    world: tuple
    prompt: tuple
    actions: tuple

    @property
    def menu(self):
        return self.msg_type, self.prompt, self.actions


class DecisionVisits:
    """双方独立累计当前可见场面下各询问与候选的实际访问次数"""

    def __init__(self):
        self._worlds = {}
        self._visits = {}

    def enter(self, key):
        if self._worlds.get(key.player) == key.world:
            return False
        self._worlds[key.player] = key.world
        self._visits[key.player] = {}
        return True

    def counts_for(self, key):
        return self._visits[key.player].get(key.menu, {})

    def record(self, key, index):
        counts = self._visits[key.player].setdefault(key.menu, {})
        counts[index] = counts.get(index, 0) + 1
        return counts[index]


# 只取规则公开或持有者可见的标量，查询内部标志与未解析的位置引用不参与查重
_ENTITY_PROPERTIES = (
    'code', 'is_public', 'current_atk', 'current_def', 'base_atk', 'base_def',
    'type_mask', 'race', 'attribute', 'level', 'rank', 'link_rating', 'link_marker', 'lscale',
    'rscale', 'original_owner', 'reason', 'is_equipped', 'is_disabled', 'is_forbidden',
    'properly_summoned', 'used_effect_mask', 'counter_count', 'overlay_count',
)
_CHAIN_PROPERTIES = (
    'ct', 'code', 'desc', 'effect_slot', 'hc', 'hl', 'hs', 'hp', 'c', 'l', 's',
    'status_known', 'confirmed', 'solving', 'solved', 'activation_negated', 'effect_disabled', 'targets_known',
)
_GLOBAL_PROPERTIES = (
    'turn_count', 'phase_id', 'turn_player', 'starting_player', 'p0_turn_count', 'p1_turn_count',
    'my_lp', 'op_lp', 'my_hand_len', 'op_hand_len', 'my_deck_len', 'op_deck_len',
    'my_grave_len', 'op_grave_len', 'my_removed_len', 'op_removed_len',
    'my_extra_len', 'op_extra_len', 'disabled_field_mask',
)


def _visible_entity_key(entity, player, grave_blocked):
    slot = (entity.owner, entity.location, entity.sequence, entity.position)
    if not entity_identity_visible(entity, player, grave_blocked):
        return slot
    props = vars(entity)
    setcodes = props.get('setcodes', ())
    setcodes = tuple(setcodes) if isinstance(setcodes, (list, tuple)) else (setcodes,)
    return (slot, props.get('current_code') or entity.code, props.get('top_overlay_code') or 0,
            tuple(map(props.get, _ENTITY_PROPERTIES)),
            setcodes, tuple(props.get('overlay_codes') or ()),
            tuple(tuple(pair) for pair in props.get('counter_items', ())),
            tuple(zip(props.get('relation_types', ()), props.get('relation_indices', ()))))


def _visible_world_key(snapshot, player):
    """只使用决策时已有的场面事实，菜单切换与事件流水号不代表对局推进"""
    viewer = 1 << player
    flags = getattr(snapshot, 'public_state_flags', ())
    grave_blocked = bool(flags and flags[3 + player])
    entities = tuple(_visible_entity_key(entity, player, grave_blocked) for entity in snapshot.entities)
    chains = tuple((tuple(map(item.get, _CHAIN_PROPERTIES)),
                    tuple(visible_chain_target(target, player) for target in item.get('targets', ())))
                   for item in snapshot.chain_stack)
    hints = tuple((hint.player, hint.location, hint.kind, hint.value, hint.count, hint.turn)
                  for hint in getattr(snapshot, 'public_hints', ()) if hint.visibility & viewer)
    known_deck = tuple((card.player, card.sequence, card.code, card.faceup)
                       for card in getattr(snapshot, 'known_deck_cards', ()) if card.visibility & viewer)
    hand = getattr(snapshot, 'opponent_hand_memory', ((), ()))[1 - player]
    return (getattr(snapshot, 'loop_public_progress', 0),
            tuple(map(vars(snapshot.global_data).get, _GLOBAL_PROPERTIES)), entities, chains,
            int(getattr(snapshot, 'resolving_chain_index', 0)),
            tuple((item.get('code', 0), item.get('effect_slot', -1)) for item in snapshot.history_stack[:8]),
            hints, known_deck, tuple(hand), tuple(flags[:3]) + (grave_blocked,) + tuple(flags[5:]))


def _action_token(action):
    """按已投影的协议动作识别候选，显式应答可跨宏池临时序号重排"""
    explicit = bool(action.decision_bytes) or action.decision_value is not None
    return (action.action_type, None if explicit else action.index, action.target_entity_idx,
            action.desc_id, action.effect_slot, action.code, bytes(action.decision_bytes or b''),
            action.decision_value, action.operation_id, action.summon_method_id, action.response_value,
            action.target_location_raw, action.selection_min, action.selection_max, action.selection_count,
            action.finishable, action.cancelable, action.context_value, action.prompt_flags,
            action.prompt_value, action.prompt_value2, tuple(action.macro_targets or ()),
            tuple(action.macro_places or ()), tuple(action.macro_target_codes or ()),
            tuple(tuple(value) if isinstance(value, list) else value for value in (action.macro_target_values or ())),
            tuple(action.macro_target_locations or ()))


def build_action_state_key(snapshot, msg_type):
    """从实际行动座位构造查重键，不读取对手暗牌身份或复制模型张量"""
    player = getattr(snapshot.global_data, 'decision_player', -1)
    if player not in (0, 1):
        player = snapshot.global_data.to_play
    if player not in (0, 1):
        raise ValueError('动作查重需要确定 P0 或 P1 决策座位')
    viewer = 1 << player
    hints = tuple((hint.get('kind', 2), hint.get('subtype', 0), hint.get('wire_player', -1),
                   hint.get('value', 0), hint.get('code', 0), hint.get('chain', 0), hint.get('prompt_type', 0))
                  for hint in getattr(snapshot, 'prompt_context', ()) if hint.get('visibility_mask', 0) & viewer)
    targets = tuple(visible_chain_target(target, player) for target in getattr(snapshot, 'recent_chain_targets', ()))
    return ActionStateKey(int(msg_type), player, _visible_world_key(snapshot, player),
                          (hints, targets), tuple(_action_token(action) for action in snapshot.valid_actions))


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
