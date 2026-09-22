# 本文件统一维护 V4 模型协议字段清单与结构哈希

import hashlib
import json

from auxiliary_heads import (
    AUXILIARY_BACKBONE_SCALE_MAX,
    AUXILIARY_BOUNDARY_CLASS_COUNT,
    AUXILIARY_COMPONENT_WEIGHTS,
    AUXILIARY_FUTURE_DELTA_WIDTH,
    AUXILIARY_FUTURE_HORIZONS,
    AUXILIARY_HEAD_HIDDEN_DIM,
    AUXILIARY_IMMEDIATE_DELTA_WIDTH,
    AUXILIARY_LOSS_COEF,
    AUXILIARY_MESSAGE_GROUP_COUNT,
    AUXILIARY_PROBE_STEPS,
    AUXILIARY_RAMP_STEPS,
    AUXILIARY_RESULT_BIT_COUNT,
    AUXILIARY_TERMINAL_OUTCOME_CLASS_COUNT,
)
from auxiliary_targets import AUXILIARY_TARGET_FORMAT_VERSION
from card_vocab import (
    CARD_VOCAB_FORMAT_VERSION,
    CARD_VOCAB_RESERVED_TOKENS,
    FIRST_CARD_TOKEN_ID,
    get_default_card_vocabulary,
)
from deck_encoder import DECK_LATENT_COUNT, DECK_SECTION_COUNT
from deck_protocol import MAX_DECK_PROFILE_ENTRIES, DeckSection
from data_types import (
    ACTION_OPERATION_COUNT,
    CARD_ADDITIONAL_OVERLAY_SLOTS,
    CARD_COUNTER_SLOTS,
    CARD_COUNTER_TYPE_BYTES,
    CARD_RELATION_SLOTS,
    CARD_REASON_BYTES,
    CARD_STATUS_BITS,
    FIELD_ZONE_BITS,
    FIELD_ZONE_BYTES,
    GLOBAL_FEATURE_DIM,
    PHASE_CATEGORY_COUNT,
    POSITION_CATEGORY_COUNT,
    SUMMON_METHOD_COUNT,
    TRANSITION_EVENT_CHAIN_DIM,
    TRANSITION_EVENT_COUNT_DIM,
    TRANSITION_EVENT_HISTORY_SIZE,
    TRANSITION_EVENT_LOCATION_DIM,
    TRANSITION_EVENT_LP_DIM,
    TRANSITION_EVENT_MESSAGE_BYTES,
    TRANSITION_EVENT_RESULT_BYTES,
    TRANSITION_EVENT_TARGET_SLOTS,
    TRANSITION_ZONE_COUNT,
    ZONE_CATEGORY_COUNT,
    ActionOperation,
    SummonMethod,
)
from event_history import get_transition_event_protocol_descriptor
from semantic_lookup import (
    STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION,
    get_static_semantic_lookup,
)


MODEL_PROTOCOL_VERSION = 4
PROTOCOL_SCHEMA_REVISION = 12


def _schema_descriptor(card_vocabulary):
    """生成当前 V4 开发阶段实际启用的模型协议描述"""
    return {
        "model_protocol_version": MODEL_PROTOCOL_VERSION,
        "schema_revision": PROTOCOL_SCHEMA_REVISION,
        "card_identity": {
            "mapping": "append_only_exact_v1",
            "vocabulary_format_version": CARD_VOCAB_FORMAT_VERSION,
            "capacity": card_vocabulary.capacity,
            "first_card_token_id": FIRST_CARD_TOKEN_ID,
            "reserved_tokens": CARD_VOCAB_RESERVED_TOKENS,
            "indexed_inputs": [
                "card_idx",
                "card_alias_idx",
                "card_overlay_idx",
                "card_overlay_rest_idx",
                "deck_idx",
                "deck_profile_card_idx",
                "c_card_idx",
                "h_card_idx",
                "act_code",
                "act_target_code",
                "event_card_idx",
                "event_target_card_idx",
            ],
        },
        "global_state": {
            "continuous_input": "global",
            "continuous_dim": GLOBAL_FEATURE_DIM,
            "continuous_fields": [
                "turn_count",
                "is_current_players_turn",
                "current_player_went_first",
                "current_player_turn_count",
                "opponent_turn_count",
                "relative_resource_pairs",
            ],
            "player_context_input": "player_context",
            "player_context_slots": [
                "decision_player",
                "turn_player",
                "starting_player",
            ],
            "player_role_encoding": {
                "unknown": 0,
                "self": 1,
                "opponent": 2,
            },
            "categorical_phase_input": {
                "name": "phase",
                "shape": [1],
                "categories": PHASE_CATEGORY_COUNT,
            },
        },
        "categorical_locations": {
            "zone_categories": ZONE_CATEGORY_COUNT,
            "position_categories": POSITION_CATEGORY_COUNT,
            "card_inputs": ["card_zone", "card_position"],
            "chain_inputs": ["c_zone", "c_position"],
            "action_inputs": ["act_location", "act_position"],
        },
        "public_card_state": {
            "source": "public_legacy_ygopro_core_query_card",
            "query_mode": "full_snapshot_without_cache",
            "visibility": "masked_per_decision_player_after_query",
            "dynamic_identity_input": "card_alias_idx",
            "reason_bits_input": {
                "name": "card_reason_bytes",
                "shape": [120, CARD_REASON_BYTES],
                "storage": "little_endian_packed_bits",
                "expanded_bits": 32,
            },
            "status_bits_input": {
                "name": "card_status_bits",
                "shape": [120, CARD_STATUS_BITS],
                "values": ["disabled", "proc_complete", "forbidden"],
            },
            "original_owner_input": "card_owner_role",
            "relations": {
                "index_input": "card_relation_idx",
                "type_input": "card_relation_type",
                "slots": CARD_RELATION_SLOTS,
                "values": [
                    "none",
                    "equip_target",
                    "effect_target",
                    "reason_card",
                    "equip_source",
                ],
            },
            "overlay_inputs": {
                "first": "card_overlay_idx",
                "additional": "card_overlay_rest_idx",
                "additional_slots": CARD_ADDITIONAL_OVERLAY_SLOTS,
            },
            "typed_counters": {
                "type_bits_input": "card_counter_type_bytes",
                "type_bytes": CARD_COUNTER_TYPE_BYTES,
                "value_input": "card_counter_value",
                "slots": CARD_COUNTER_SLOTS,
            },
            "disabled_field_input": {
                "name": "field_zone_mask",
                "shape": [FIELD_ZONE_BYTES],
                "storage": "little_endian_packed_bits",
                "expanded_bits": FIELD_ZONE_BITS,
            },
            "dynamic_numeric_fields": [
                "level",
                "rank",
                "link_rating",
                "left_scale",
                "right_scale",
                "link_marker",
                "attack",
                "defense",
                "base_attack",
                "base_defense",
            ],
            "summon_state_boundary": {
                "available": "status_proc_complete_only",
                "unavailable": ["summon_type", "summon_location", "summon_player"],
                "policy": "do_not_infer_or_require_private_core_fork",
                "upstream_action": "track_public_api_or_submit_upstream_pr",
            },
        },
        "static_semantics": {
            "lookup_format_version": STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION,
            "index": "exact_card_token_and_lua_effect_slot",
            "effect_slots": 8,
            "lookup_owner": "model_side_registered_buffers",
            "trajectory_mode": "card_and_effect_slot_ids_only",
            "card_identity_inputs": [
                "card_idx",
                "deck_idx",
                "c_card_idx",
                "h_card_idx",
                "event_card_idx",
                "event_target_card_idx",
            ],
            "effect_slot_inputs": [
                "c_effect_slot",
                "h_effect_slot",
                "act_effect_slot",
                "event_effect_slot",
            ],
            "removed_expanded_prefixes": [
                "sem_",
                "d_sem_",
                "c_sem_",
                "h_sem_",
            ],
            "identity_policy": "append_only_card_prefix_with_logical_vectors",
        },
        "deck_profile": {
            "max_entries": MAX_DECK_PROFILE_ENTRIES,
            "identity_input": "deck_profile_card_idx",
            "section_input": "deck_profile_section",
            "section_categories": DECK_SECTION_COUNT,
            "section_values": {
                section.name.lower(): int(section)
                for section in DeckSection
            },
            "initial_count_input": "deck_profile_initial_count",
            "remaining_count_input": "deck_profile_remaining_count",
            "mask_input": "deck_profile_mask",
            "static_card_inputs": [
                "deck_profile_race",
                "deck_profile_attr",
                "deck_profile_setcodes",
            ],
            "latent_count": DECK_LATENT_COUNT,
            "encoder": "permutation_invariant_latent_cross_attention_v1",
            "semantic_summary": "masked_mean_of_all_lua_code_vectors",
            "current_policy_sections": [
                int(DeckSection.MAIN),
                int(DeckSection.EXTRA),
            ],
            "side_policy": "reserved_for_upper_layer_not_emitted_to_bo1_policy",
            "dynamic_overflow_fallback": "legacy_remaining_deck_sequence",
            "policy_fusion": "direct_policy_and_value_intent_summary",
        },
        "layered_film": {
            "scope": "per_transformer_layer_and_sublayer",
            "sublayers": ["attention", "ffn"],
            "parameters": ["gamma", "beta"],
            "global_condition": ["global", "phase_context"],
            "deck_condition": "deck_style",
            "deck_gate": "zero_initialized_per_layer_sublayer_parameter_channel",
            "deck_condition_dropout": {
                "probability": 0.1,
                "scope": "whole_training_duel",
                "mask_input": {
                    "name": "deck_film_mask",
                    "dtype": "bool",
                    "shape": [1],
                    "false_meaning": "disable_deck_film_only",
                },
                "ppo_consistency": "sampled_once_and_stored_per_step",
                "deployment_default": True,
            },
            "maximum_absolute_modulation": 0.5,
            "combination": "0.5*tanh(global_raw+tanh(deck_gate)*deck_raw)",
            "direct_deck_intent_path": True,
        },
        "action_semantics": {
            "operation_input": {
                "name": "act_operation",
                "shape": [120],
                "dtype": "uint8",
                "categories": ACTION_OPERATION_COUNT,
                "values": {
                    operation.name.lower(): int(operation)
                    for operation in ActionOperation
                },
                "visibility": "public_prompt",
                "source": "core_message_candidate_family",
                "lifecycle": "current_legal_action_snapshot",
                "missing_value": int(ActionOperation.DEFAULT),
                "consumers": [
                    "action_signature",
                    "policy_network",
                    "onnx",
                    "replay",
                ],
            },
            "summon_method_input": {
                "name": "act_summon_method",
                "shape": [120],
                "dtype": "uint8",
                "categories": SUMMON_METHOD_COUNT,
                "values": {
                    method.name.lower(): int(method)
                    for method in SummonMethod
                },
                "visibility": "public_prompt",
                "source": "core_message_or_verified_runtime_evidence",
                "lifecycle": "current_legal_action_snapshot",
                "missing_value": int(SummonMethod.NONE),
                "unknown_special_value": int(SummonMethod.SPECIAL_UNKNOWN),
                "consumers": [
                    "action_signature",
                    "policy_network",
                    "onnx",
                    "replay",
                ],
            },
            "selection_state_machines": {
                "iterative_messages": [26],
                "complete_combination_messages": [15, 20, 23],
                "type_15_cancel_source": "core_cancelable_flag",
                "type_20_legality": "max_card_count_and_min_release_value",
                "type_23_cancelable": False,
                "type_26_terminal_response": "int32_minus_one",
                "core_response_buffer_bytes": 512,
            },
        },
        "transition_history": {
            "event_protocol": get_transition_event_protocol_descriptor(),
            "history_size": TRANSITION_EVENT_HISTORY_SIZE,
            "order": "oldest_to_newest_after_visibility_filtering",
            "visibility": "field_level_player_mask_before_tensor_encoding",
            "inputs": {
                "mask": [TRANSITION_EVENT_HISTORY_SIZE],
                "source_card": [TRANSITION_EVENT_HISTORY_SIZE],
                "source_effect_slot": [TRANSITION_EVENT_HISTORY_SIZE],
                "source_context": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_LOCATION_DIM,
                ],
                "target_card": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_TARGET_SLOTS,
                ],
                "target_context": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_TARGET_SLOTS,
                    TRANSITION_EVENT_LOCATION_DIM,
                ],
                "target_mask": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_TARGET_SLOTS,
                ],
                "result_bytes": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_RESULT_BYTES,
                ],
                "message_bytes": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_MESSAGE_BYTES,
                ],
                "lp_delta": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_LP_DIM,
                ],
                "zone_delta": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    2 * TRANSITION_ZONE_COUNT,
                ],
                "chain": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_CHAIN_DIM,
                ],
                "counts": [
                    TRANSITION_EVENT_HISTORY_SIZE,
                    TRANSITION_EVENT_COUNT_DIM,
                ],
            },
            "source_semantics": "exact_card_and_lua_effect_slot",
            "target_semantics": "card_identity_and_all_lua_code_mean",
            "encoder": "target_pool_then_ordered_depthwise_context_pool_v1",
            "policy_value_fusion": "zero_initialized_bounded_residual_gate",
            "empty_history": "all_zero_masked_vector",
        },
        "training_auxiliary_heads": {
            "target_format_version": AUXILIARY_TARGET_FORMAT_VERSION,
            "inputs": ["shared_state", "selected_option"],
            "result_bits": AUXILIARY_RESULT_BIT_COUNT,
            "message_groups": AUXILIARY_MESSAGE_GROUP_COUNT,
            "boundary_classes": AUXILIARY_BOUNDARY_CLASS_COUNT,
            "terminal_outcome_classes": AUXILIARY_TERMINAL_OUTCOME_CLASS_COUNT,
            "head_hidden_dim": AUXILIARY_HEAD_HIDDEN_DIM,
            "immediate_delta_width": AUXILIARY_IMMEDIATE_DELTA_WIDTH,
            "future_horizons": [
                item.name.lower() for item in AUXILIARY_FUTURE_HORIZONS
            ],
            "future_resource_width": AUXILIARY_FUTURE_DELTA_WIDTH,
            "loss_coefficient": AUXILIARY_LOSS_COEF,
            "component_weights": AUXILIARY_COMPONENT_WEIGHTS,
            "policy_value_feedback": False,
            "standard_onnx_output": False,
            "backbone_gradient": {
                "method": "trainer_scaled_stop_gradient_bridge",
                "probe_updates": AUXILIARY_PROBE_STEPS,
                "ramp_updates": AUXILIARY_RAMP_STEPS,
                "maximum_scale": AUXILIARY_BACKBONE_SCALE_MAX,
            },
        },
        "inherited_protocol": "galatea_model_protocol_v3",
    }


def get_current_protocol_metadata(card_vocabulary=None, *, semantic_root="."):
    """返回检查点、ONNX 和部署包共用的 V4 协议身份"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    semantic_lookup = get_static_semantic_lookup(
        semantic_root,
        card_vocabulary=vocabulary,
    )
    descriptor = _schema_descriptor(vocabulary)
    encoded = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "model_protocol_version": MODEL_PROTOCOL_VERSION,
        "protocol_schema_revision": PROTOCOL_SCHEMA_REVISION,
        "protocol_schema_hash": hashlib.sha256(encoded).hexdigest(),
        "card_vocab_hash": vocabulary.vocabulary_hash,
        "card_vocab_size": vocabulary.capacity,
        "card_vocab_card_count": vocabulary.card_count,
        "semantic_lookup_format_version": STATIC_SEMANTIC_LOOKUP_FORMAT_VERSION,
        "semantic_lookup_hash": semantic_lookup.prefix_hash(),
        "semantic_lookup_card_count": vocabulary.card_count,
    }


def apply_current_protocol_metadata(
    config,
    *,
    card_vocabulary=None,
    semantic_root=".",
):
    """校验已有协议字段后，为新网络配置补全当前协议身份"""
    if not isinstance(config, dict):
        raise ValueError("network config must be a dictionary")
    metadata = get_current_protocol_metadata(
        card_vocabulary,
        semantic_root=semantic_root,
    )
    normalized = dict(config)
    present_keys = [key for key in metadata if key in normalized]
    if present_keys and len(present_keys) != len(metadata):
        raise ValueError("network config contains incomplete protocol metadata")
    if present_keys:
        validate_protocol_metadata(
            normalized,
            label="network config",
            card_vocabulary=card_vocabulary,
            semantic_root=semantic_root,
        )
    normalized.update(metadata)
    vocab_size = normalized.get("vocab_size", metadata["card_vocab_size"])
    if vocab_size != metadata["card_vocab_size"]:
        raise ValueError(
            "network vocab_size must equal the exact card vocabulary capacity"
        )
    normalized["vocab_size"] = metadata["card_vocab_size"]
    return normalized


def validate_protocol_metadata(
    container,
    *,
    label="artifact",
    card_vocabulary=None,
    semantic_root=".",
):
    """校验结构一致且词表是当前只追加前缀的产物"""
    if not isinstance(container, dict):
        raise ValueError(f"{label} protocol metadata must be a dictionary")
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    expected = get_current_protocol_metadata(
        vocabulary,
        semantic_root=semantic_root,
    )
    for key in (
        "model_protocol_version",
        "protocol_schema_revision",
        "protocol_schema_hash",
        "card_vocab_size",
        "semantic_lookup_format_version",
    ):
        expected_value = expected[key]
        if container.get(key) != expected_value:
            raise ValueError(
                f"{label} {key} does not match the current protocol: "
                f"file={container.get(key)!r}, current={expected_value!r}"
            )
    card_count = container.get("card_vocab_card_count")
    if isinstance(card_count, bool) or not isinstance(card_count, int):
        raise ValueError(f"{label} card_vocab_card_count must be an integer")
    if card_count < 0 or card_count > vocabulary.card_count:
        raise ValueError(
            f"{label} card vocabulary is not a prefix of the current vocabulary"
        )
    prefix_hash = vocabulary.prefix(card_count).vocabulary_hash
    if container.get("card_vocab_hash") != prefix_hash:
        raise ValueError(
            f"{label} card_vocab_hash does not match the current append-only prefix"
        )
    semantic_card_count = container.get("semantic_lookup_card_count")
    if semantic_card_count != card_count:
        raise ValueError(
            f"{label} semantic_lookup_card_count must equal card_vocab_card_count"
        )
    semantic_lookup = get_static_semantic_lookup(
        semantic_root,
        card_vocabulary=vocabulary,
    )
    expected_semantic_hash = semantic_lookup.prefix_hash(card_count)
    if container.get("semantic_lookup_hash") != expected_semantic_hash:
        raise ValueError(
            f"{label} semantic_lookup_hash does not match the current semantic prefix"
        )
    return expected
