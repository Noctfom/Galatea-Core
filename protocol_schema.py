# 本文件统一维护 V4 模型协议字段清单与结构哈希

import hashlib
import json

from card_vocab import (
    CARD_VOCAB_FORMAT_VERSION,
    CARD_VOCAB_RESERVED_TOKENS,
    FIRST_CARD_TOKEN_ID,
    get_default_card_vocabulary,
)
from data_types import (
    ACTION_OPERATION_COUNT,
    GLOBAL_FEATURE_DIM,
    PHASE_CATEGORY_COUNT,
    POSITION_CATEGORY_COUNT,
    SUMMON_METHOD_COUNT,
    ZONE_CATEGORY_COUNT,
    ActionOperation,
    SummonMethod,
)


MODEL_PROTOCOL_VERSION = 4
PROTOCOL_SCHEMA_REVISION = 5


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
                "card_overlay_idx",
                "deck_idx",
                "c_card_idx",
                "act_code",
                "act_target_code",
                "sem_ref",
                "d_sem_ref",
                "c_sem_ref",
                "h_sem_ref",
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
        "inherited_protocol": "galatea_model_protocol_v3",
    }


def get_current_protocol_metadata(card_vocabulary=None):
    """返回检查点、ONNX 和部署包共用的 V4 协议身份"""
    vocabulary = card_vocabulary or get_default_card_vocabulary()
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
    }


def apply_current_protocol_metadata(config, *, card_vocabulary=None):
    """校验已有协议字段后，为新网络配置补全当前协议身份"""
    if not isinstance(config, dict):
        raise ValueError("network config must be a dictionary")
    metadata = get_current_protocol_metadata(card_vocabulary)
    normalized = dict(config)
    present_keys = [key for key in metadata if key in normalized]
    if present_keys and len(present_keys) != len(metadata):
        raise ValueError("network config contains incomplete protocol metadata")
    if present_keys:
        validate_protocol_metadata(
            normalized,
            label="network config",
            card_vocabulary=card_vocabulary,
        )
    normalized.update(metadata)
    vocab_size = normalized.get("vocab_size", metadata["card_vocab_size"])
    if vocab_size != metadata["card_vocab_size"]:
        raise ValueError(
            "network vocab_size must equal the exact card vocabulary capacity"
        )
    normalized["vocab_size"] = metadata["card_vocab_size"]
    return normalized


def validate_protocol_metadata(container, *, label="artifact", card_vocabulary=None):
    """校验结构一致且词表是当前只追加前缀的产物"""
    if not isinstance(container, dict):
        raise ValueError(f"{label} protocol metadata must be a dictionary")
    vocabulary = card_vocabulary or get_default_card_vocabulary()
    expected = get_current_protocol_metadata(vocabulary)
    for key in (
        "model_protocol_version",
        "protocol_schema_revision",
        "protocol_schema_hash",
        "card_vocab_size",
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
    return expected
