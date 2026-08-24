import numpy as np

import rule_bot
from data_types import GameAction
from feature_encoder import MAX_ACTIONS
from game_constants import LocationInfo


MACRO_ACTION_MSGS = frozenset({15, 18, 20, 22, 23, 24, 25})
MODEL_ACTION_MSGS = frozenset(
    {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 140, 141, 142, 143}
)
_CANCEL_RESPONSE = b"\xff\xff\xff\xff"


def _as_probabilities(action_probabilities):
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
    """Build policy-guided, engine-ready macro actions for complex prompts.

    ``base_actions`` and ``action_probabilities`` must describe the same
    pre-macro action list.  Returned actions retain raw location identifiers;
    callers must regenerate a DuelState snapshot so they are mapped to entity
    indices before feature encoding.
    """
    if msg_type not in MACRO_ACTION_MSGS:
        raise ValueError(f"message type {msg_type} is not a macro action prompt")

    probabilities = _as_probabilities(action_probabilities)
    base_actions = list(base_actions)

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
        response = bytes(option["bytes"])
        score = 1e-4

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
        response = bytes(option["bytes"])
        description = "Cancel" if response == _CANCEL_RESPONSE else f"Macro Action {pool_index}"
        macro_actions.append(
            GameAction(
                action_type=msg_type,
                index=pool_index,
                desc_str=description,
                macro_targets=list(option.get("locs", [])) or None,
                macro_places=list(option.get("places", [])) or None,
                decision_bytes=response,
            )
        )

    return macro_actions
