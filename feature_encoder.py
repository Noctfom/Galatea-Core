# ==================================================================================
#  Galatea Feature Encoder（V4 精确卡片身份已启用）
# ==================================================================================

from collections import Counter

import torch
import numpy as np
from card_vocab import get_default_card_vocabulary
from deck_protocol import (
    MAX_DECK_PROFILE_ENTRIES,
    DeckProfile,
    DeckSection,
)
from data_types import (
    ACTION_CONTEXT_DIM,
    ACTION_OPERATION_COUNT,
    ACTION_RESPONSE_BUCKETS,
    ACTION_SIGNATURE_BYTES,
    ACTION_TARGET_SLOTS,
    CHAIN_CONTEXT_DIM,
    CARD_ADDITIONAL_OVERLAY_SLOTS,
    CARD_COUNTER_SLOTS,
    CARD_COUNTER_TYPE_BYTES,
    CARD_NUMERIC_FEATURE_DIM,
    CARD_REASON_BYTES,
    CARD_RELATION_SLOTS,
    CARD_STATUS_BITS,
    FIELD_ZONE_BYTES,
    GLOBAL_FEATURE_DIM,
    PHASE_CATEGORY_COUNT,
    PLAYER_CONTEXT_SLOTS,
    POSITION_CATEGORY_COUNT,
    SUMMON_METHOD_COUNT,
    ZONE_CATEGORY_COUNT,
    GameSnapshot,
)
from game_constants import LocationInfo, Position, Zone
from semantic_lookup import register_static_semantic_runtime_catalog

# --- 配置参数 ---
MAX_CARDS = 120
MAX_ACTIONS = 120
HIDDEN_OPPONENT_ZONES = {
    Zone.HAND,
    Zone.DECK,
    Zone.MZONE,
    Zone.SZONE,
    Zone.REMOVED,
    Zone.EXTRA,
}

class GalateaEncoder:
    def __init__(self, vocab_size=None, card_vocabulary=None):
        self.card_vocabulary = card_vocabulary or get_default_card_vocabulary()
        if vocab_size is not None and vocab_size != self.card_vocabulary.capacity:
            raise ValueError(
                "encoder vocab_size must equal the exact card vocabulary capacity"
            )
        self.vocab_size = self.card_vocabulary.capacity
        self.global_dim = GLOBAL_FEATURE_DIM
        self.card_feat_dim = 7
        self._deck_profile_cache = {}
        # spawn Worker 只加载轻量效果槽目录，不复制模型侧静态语义大表
        register_static_semantic_runtime_catalog(
            card_vocabulary=self.card_vocabulary,
        )

    def _encode_card_code(self, code):
        """把真实卡片代码映射为无碰撞的固定词表索引"""
        return self.card_vocabulary.encode(code)

    @staticmethod
    def _read_deck_card_metadata(code):
        """读取卡组卡片的稳定种族、属性和字段标签，异常时安全归零"""
        try:
            from card_reader import card_db

            stats = card_db.get_full_stats(code)
            race = int(stats[1]) % 30
            attribute = int(stats[2]) % 10
            raw_setcodes = (
                stats[10]
                if isinstance(stats[10], (list, tuple))
                else [stats[10]]
            )
            setcodes = [
                int(value) % 4096
                for value in (list(raw_setcodes) + [0] * 4)[:4]
            ]
            return race, attribute, setcodes
        except Exception:
            return 0, 0, [0, 0, 0, 0]

    def _encode_deck_profile(self, snapshot, player_id):
        """编码稳定初始画像及与画像条目对齐的逐步剩余数量"""
        if player_id == 0:
            initial_main = snapshot.p0_initial_deck_codes
            initial_extra = snapshot.p0_initial_extra_codes
            remaining_main = snapshot.p0_deck_codes
            remaining_extra = snapshot.p0_extra_codes
        else:
            initial_main = snapshot.p1_initial_deck_codes
            initial_extra = snapshot.p1_initial_extra_codes
            remaining_main = snapshot.p1_deck_codes
            remaining_extra = snapshot.p1_extra_codes

        cache_key = (tuple(initial_main), tuple(initial_extra))
        cached = self._deck_profile_cache.get(cache_key)
        if cached is None:
            profile = DeckProfile(
                main=tuple(initial_main),
                extra=tuple(initial_extra),
            )
            entries = profile.entries()
            if len(entries) > MAX_DECK_PROFILE_ENTRIES:
                raise ValueError(
                    f"deck profile has {len(entries)} unique entries; "
                    f"maximum is {MAX_DECK_PROFILE_ENTRIES}"
                )

            card_idx = np.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=np.int64)
            race = np.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=np.int64)
            attribute = np.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=np.int64)
            setcodes = np.zeros(
                (MAX_DECK_PROFILE_ENTRIES, 4),
                dtype=np.int64,
            )
            section = np.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=np.uint8)
            initial_count = np.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=np.uint8)
            mask = np.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=np.bool_)
            entry_keys = []
            for index, entry in enumerate(entries):
                card_idx[index] = self._encode_card_code(entry.code)
                race[index], attribute[index], setcodes[index] = (
                    self._read_deck_card_metadata(entry.code)
                )
                section[index] = int(entry.section)
                initial_count[index] = min(int(entry.copies), 255)
                mask[index] = True
                entry_keys.append((entry.section, int(entry.code)))
            cached = {
                "deck_profile_card_idx": card_idx,
                "deck_profile_race": race,
                "deck_profile_attr": attribute,
                "deck_profile_setcodes": setcodes,
                "deck_profile_section": section,
                "deck_profile_initial_count": initial_count,
                "deck_profile_mask": mask,
                "entry_keys": tuple(entry_keys),
            }
            self._deck_profile_cache[cache_key] = cached

        main_counts = Counter(int(code) for code in remaining_main)
        extra_counts = Counter(int(code) for code in remaining_extra)
        remaining_count = np.zeros(
            MAX_DECK_PROFILE_ENTRIES,
            dtype=np.uint8,
        )
        for index, (section, code) in enumerate(cached["entry_keys"]):
            counter = main_counts if section == DeckSection.MAIN else extra_counts
            remaining_count[index] = min(int(counter.get(code, 0)), 255)

        result = {
            key: torch.from_numpy(value).unsqueeze(0)
            for key, value in cached.items()
            if key != "entry_keys"
        }
        result["deck_profile_remaining_count"] = torch.from_numpy(
            remaining_count
        ).unsqueeze(0)
        return result

    @staticmethod
    def _encode_effect_slot(effect_slot):
        """把内部 0～7 效果槽编码为 1～8，并以 0 表示未知槽位"""
        try:
            slot_index = int(effect_slot)
        except (TypeError, ValueError):
            return 0
        return slot_index + 1 if 0 <= slot_index < 8 else 0

    @staticmethod
    def _hash_action_response(value):
        """把任意整数响应稳定映射到动作响应词表，并保留 0 作为空值"""
        if value is None:
            return 0
        return 1 + (int(value) & 0xFFFFFFFF) % (ACTION_RESPONSE_BUCKETS - 1)

    @staticmethod
    def _scale_action_context(value):
        """压缩动作约束数值，避免异常大值破坏网络数值范围"""
        return max(-4.0, min(4.0, float(value) / 16.0))

    @staticmethod
    def _split_target_value(value):
        """把素材的普通值或双值字段拆成两个紧凑的无符号特征"""
        if isinstance(value, (tuple, list)):
            low = int(value[0]) if value else 0
            high = int(value[1]) if len(value) > 1 else 0
        else:
            raw = int(value or 0) & 0xFFFFFFFF
            low = raw & 0xFFFF
            high = (raw >> 16) & 0xFFFF
        return [max(0, min(low, 255)), max(0, min(high, 255))]

    @staticmethod
    def _action_signature(action):
        """对完整动作语义生成稳定签名，继续区分超过显式目标槽的组合"""
        values = [
            action.action_type,
            getattr(action, 'operation_id', 0),
            getattr(action, 'summon_method_id', 0),
            getattr(action, 'response_value', None),
            getattr(action, 'desc_id', 0),
            getattr(action, 'code', 0),
            getattr(action, 'selection_min', 0),
            getattr(action, 'selection_max', 0),
            getattr(action, 'selection_count', 0),
            int(bool(getattr(action, 'finishable', False))),
            int(bool(getattr(action, 'cancelable', False))),
            getattr(action, 'context_value', 0),
            getattr(action, 'prompt_flags', 0),
            getattr(action, 'prompt_value', 0),
            getattr(action, 'prompt_value2', 0),
            getattr(action, 'decision_value', None),
        ]
        values.extend(bytes(getattr(action, 'decision_bytes', b'')))
        for attr_name in (
            'macro_targets',
            'macro_target_codes',
            'macro_target_values',
            'macro_target_locations',
            'macro_places',
        ):
            for item in getattr(action, attr_name, None) or ():
                if isinstance(item, (tuple, list)):
                    values.extend(item)
                else:
                    values.append(item)

        # FNV-1a 避免 Python hash 的进程随机盐导致 Worker 间编码不一致
        signature = 2166136261
        for value in values:
            normalized = -1 if value is None else int(value)
            signature ^= normalized & 0xFFFFFFFF
            signature = (signature * 16777619) & 0xFFFFFFFF
        return [
            (signature >> (byte_index * 8)) & 0xFF
            for byte_index in range(ACTION_SIGNATURE_BYTES)
        ]

    @staticmethod
    def _encode_target_location(action, player_id):
        """把引擎原始位置转成行动方视角的控制者、区域与序号"""
        raw_location = getattr(action, 'target_location_raw', -1)
        if raw_location is None or raw_location < 0:
            return 0, 0, 0, 0
        controller, location, sequence, position = LocationInfo.decode(raw_location)
        relative_controller = 1 if controller == player_id else 2
        return (
            relative_controller,
            GalateaEncoder._encode_zone_category(location),
            min(int(sequence), 31) + 1,
            GalateaEncoder._encode_position_category(position),
        )

    @staticmethod
    def _encode_chain_context(item, player_id):
        """把连锁位置与已确认的 Lua 效果槽压缩为行动方视角特征"""
        trigger_controller = int(item.get('c', -1))
        trigger_sequence = int(item.get('s', 0))
        handler_controller = int(item.get('hc', trigger_controller))
        handler_sequence = int(item.get('hs', trigger_sequence))
        effect_slot = int(item.get('effect_slot', -1))
        chain_index = int(item.get('ct', 0))

        def relative_controller(controller):
            """把绝对玩家编号转换为己方、对方或未知标量"""
            if controller not in (0, 1):
                return 0.0
            return 1.0 if controller == player_id else -1.0

        return [
            relative_controller(handler_controller),
            min(max(handler_sequence, 0), 31) / 10.0,
            relative_controller(trigger_controller),
            min(max(trigger_sequence, 0), 31) / 10.0,
            min(max(chain_index, 0), 12) / 12.0,
            (effect_slot + 1) / 8.0 if 0 <= effect_slot < 8 else 0.0,
        ]

    @staticmethod
    def _encode_phase_category(phase_id):
        """把 Core 阶段常量映射为紧凑且稳定的离散类别"""
        phase_ids = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200)
        try:
            return phase_ids.index(int(phase_id)) + 1
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _encode_zone_category(location):
        """把单一区域位映射为未知、卡组、手牌等九类离散编号"""
        zone_ids = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)
        try:
            return zone_ids.index(int(location)) + 1
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _encode_position_category(position):
        """只保留 Core 的单一实际表示及其公开标记，非法组合归入未知类别"""
        try:
            category = int(position)
        except (TypeError, ValueError):
            return 0
        if not 0 <= category < POSITION_CATEGORY_COUNT:
            return 0
        base_position = category & ~Position.REVEAL
        if base_position not in (
            0,
            Position.FACEUP_ATTACK,
            Position.FACEDOWN_ATTACK,
            Position.FACEUP_DEFENSE,
            Position.FACEDOWN_DEFENSE,
        ):
            return 0
        return category

    @staticmethod
    def _encode_global_vector(g, player_id):
        """把绝对全局状态转换为当前决策玩家的相对连续特征"""
        if player_id not in (0, 1):
            raise ValueError(f"player_id must be 0 or 1, got {player_id}")

        resource_pairs = [
            (g.my_lp, g.op_lp, 8000.0),
            (g.my_hand_len, g.op_hand_len, 10.0),
            (g.my_deck_len, g.op_deck_len, 40.0),
            (g.my_grave_len, g.op_grave_len, 20.0),
            (g.my_removed_len, g.op_removed_len, 10.0),
            (g.my_extra_len, g.op_extra_len, 15.0),
        ]

        if player_id == 1:
            resource_pairs = [(op, me, scale) for me, op, scale in resource_pairs]

        turn_player = getattr(g, 'turn_player', -1)
        if turn_player not in (0, 1):
            turn_player = g.to_play
        starting_player = getattr(g, 'starting_player', -1)
        turn_counts = [
            getattr(g, 'p0_turn_count', 0),
            getattr(g, 'p1_turn_count', 0),
        ]
        if player_id == 1:
            turn_counts.reverse()

        global_vec = [
            min(g.turn_count / 20.0, 5.0),
            1.0 if turn_player == player_id else 0.0,
            1.0 if starting_player == player_id else 0.0,
            min(turn_counts[0] / 10.0, 5.0),
            min(turn_counts[1] / 10.0, 5.0),
        ]
        for me, opponent, scale in resource_pairs:
            global_vec.extend([me / scale, opponent / scale])
        return global_vec

    @staticmethod
    def _encode_player_context(g, player_id):
        """将决策者、回合玩家和先手玩家编码为未知/己方/对方三类角色"""
        if player_id not in (0, 1):
            raise ValueError(f"player_id must be 0 or 1, got {player_id}")

        def relative_role(absolute_player):
            if absolute_player not in (0, 1):
                return 0
            return 1 if absolute_player == player_id else 2

        decision_player = getattr(g, 'decision_player', -1)
        if decision_player not in (0, 1):
            decision_player = g.to_play
        roles = [
            relative_role(decision_player),
            relative_role(getattr(g, 'turn_player', -1)),
            relative_role(getattr(g, 'starting_player', -1)),
        ]
        if len(roles) != PLAYER_CONTEXT_SLOTS:
            raise RuntimeError("player context slot count does not match the protocol")
        return roles

    @staticmethod
    def _encode_field_zone_mask(raw_mask, player_id):
        """把 Core 的双方禁用区域位转换为当前玩家视角的 32 位掩码"""
        mask = int(raw_mask or 0) & 0xFFFFFFFF
        if player_id == 1:
            mask = ((mask & 0xFFFF) << 16) | ((mask >> 16) & 0xFFFF)
        return np.asarray(
            [(mask >> (byte_index * 8)) & 0xFF for byte_index in range(FIELD_ZONE_BYTES)],
            dtype=np.uint8,
        )

    @staticmethod
    def _is_entity_visible_to_player(entity, player_id):
        if entity.owner == player_id:
            return True

        if entity.location in HIDDEN_OPPONENT_ZONES:
            return bool(entity.is_public)
        return True
    
    def _get_coords(self, player_id, owner, location, sequence):
        """将一维的 location 和 sequence 转换为二维平面坐标 (X, Y)"""
        # 非场上卡片，放入异次元坐标
        if location not in [Zone.MZONE, Zone.SZONE]:
            return -1.0, -1.0
            
        x, y = -1.0, -1.0
        is_mine = (owner == player_id)
        
        if location == Zone.MZONE:
            y = 0.3 if is_mine else 0.7
            if sequence <= 4:
                # 主怪兽区 0~4
                x = 0.1 + 0.2 * sequence if is_mine else 0.9 - 0.2 * sequence
            elif sequence == 5:
                # 左侧额外怪兽区 (对于控制者来说在 1 号位上方)
                x, y = (0.3, 0.5) if is_mine else (0.7, 0.5)
            elif sequence == 6:
                # 右侧额外怪兽区 (对于控制者来说在 3 号位上方)
                x, y = (0.7, 0.5) if is_mine else (0.3, 0.5)
                
        elif location == Zone.SZONE:
            y = 0.1 if is_mine else 0.9
            if sequence <= 4:
                # 魔陷区 0~4
                x = 0.1 + 0.2 * sequence if is_mine else 0.9 - 0.2 * sequence
            elif sequence == 5:
                # 场地区
                x, y = (0.0, 0.2) if is_mine else (1.0, 0.8)
                
        return x, y

    def encode_actions(self, valid_actions, snapshot, player_id):
        """把合法动作编码为动作协议 V2 的固定形状张量"""
        max_materials = ACTION_TARGET_SLOTS
        act_card_idxs, act_types, act_descs, act_effect_slots, masks = [], [], [], [], []
        act_races, act_attrs, act_codes, act_places = [], [], [], []
        act_operations, act_summon_methods, act_responses, act_signatures = [], [], [], []
        act_contexts, act_target_codes, act_target_values = [], [], []
        act_controllers, act_locations, act_sequences, act_positions = [], [], [], []

        for act in valid_actions[:MAX_ACTIONS]:
            if getattr(act, 'macro_targets', None):
                t_idxs = [
                    target if 0 <= target < MAX_CARDS else MAX_CARDS
                    for target in act.macro_targets[:max_materials]
                ]
                t_idxs.extend([MAX_CARDS] * (max_materials - len(t_idxs)))
            else:
                target = act.target_entity_idx
                target = target if 0 <= target < MAX_CARDS else MAX_CARDS
                t_idxs = [target] + [MAX_CARDS] * (max_materials - 1)
            act_card_idxs.append(t_idxs)

            if getattr(act, 'macro_places', None):
                places = list(act.macro_places[:max_materials])
                places.extend([0] * (max_materials - len(places)))
            else:
                place = (act.desc_id % 32) + 1 if act.action_type in [18, 24] else 0
                places = [place] + [0] * (max_materials - 1)
            act_places.append(places)

            raw_codes = list(getattr(act, 'macro_target_codes', None) or ())[:max_materials]
            target_codes = [self._encode_card_code(code) if code else 0 for code in raw_codes]
            target_codes.extend([0] * (max_materials - len(target_codes)))
            act_target_codes.append(target_codes)

            raw_values = list(getattr(act, 'macro_target_values', None) or ())[:max_materials]
            target_values = [self._split_target_value(value) for value in raw_values]
            target_values.extend([[0, 0]] * (max_materials - len(target_values)))
            act_target_values.append(target_values)

            act_types.append(act.action_type)
            act_descs.append(act.desc_id % 1024)
            effect_slot = int(getattr(act, 'effect_slot', -1))
            act_effect_slots.append(effect_slot + 1 if 0 <= effect_slot < 8 else 0)
            masks.append(True)
            operation_id = int(getattr(act, 'operation_id', 0))
            if not 0 <= operation_id < ACTION_OPERATION_COUNT:
                raise ValueError(f"action operation id is out of range: {operation_id}")
            summon_method_id = int(getattr(act, 'summon_method_id', 0))
            if not 0 <= summon_method_id < SUMMON_METHOD_COUNT:
                raise ValueError(
                    f"summon method id is out of range: {summon_method_id}"
                )
            act_operations.append(operation_id)
            act_summon_methods.append(summon_method_id)
            act_responses.append(
                self._hash_action_response(getattr(act, 'response_value', None))
            )
            act_signatures.append(self._action_signature(act))
            act_contexts.append([
                self._scale_action_context(getattr(act, 'selection_min', 0)),
                self._scale_action_context(getattr(act, 'selection_max', 0)),
                self._scale_action_context(getattr(act, 'selection_count', 0)),
                float(bool(getattr(act, 'finishable', False))),
                float(bool(getattr(act, 'cancelable', False))),
                self._scale_action_context(getattr(act, 'context_value', 0)),
            ])
            controller, location, sequence, position = self._encode_target_location(
                act,
                player_id,
            )
            act_controllers.append(controller)
            act_locations.append(location)
            act_sequences.append(sequence)
            act_positions.append(position)

            race_value, attr_value, code_value = 0, 0, 0
            if act.action_type == 140 and act.desc_id > 0:
                race_value = (act.desc_id.bit_length() - 1) % 30
            elif act.action_type == 141 and act.desc_id > 0:
                attr_value = (act.desc_id.bit_length() - 1) % 10
            elif act.action_type == 142:
                code_value = self._encode_card_code(act.desc_id)
            elif getattr(act, 'code', 0):
                code_value = self._encode_card_code(act.code)
            act_races.append(race_value)
            act_attrs.append(attr_value)
            act_codes.append(code_value)

        pad_len = MAX_ACTIONS - len(act_card_idxs)
        if pad_len > 0:
            act_card_idxs.extend([[MAX_CARDS] * max_materials] * pad_len)
            act_places.extend([[0] * max_materials] * pad_len)
            act_types.extend([0] * pad_len)
            act_descs.extend([0] * pad_len)
            act_effect_slots.extend([0] * pad_len)
            masks.extend([False] * pad_len)
            act_races.extend([0] * pad_len)
            act_attrs.extend([0] * pad_len)
            act_codes.extend([0] * pad_len)
            act_operations.extend([0] * pad_len)
            act_summon_methods.extend([0] * pad_len)
            act_responses.extend([0] * pad_len)
            act_signatures.extend([[0] * ACTION_SIGNATURE_BYTES] * pad_len)
            act_contexts.extend([[0.0] * ACTION_CONTEXT_DIM] * pad_len)
            act_target_codes.extend([[0] * max_materials] * pad_len)
            act_target_values.extend([[[0, 0]] * max_materials] * pad_len)
            act_controllers.extend([0] * pad_len)
            act_locations.extend([0] * pad_len)
            act_sequences.extend([0] * pad_len)
            act_positions.extend([0] * pad_len)

        return {
            'act_card_idx': torch.tensor(act_card_idxs, dtype=torch.long).unsqueeze(0),
            'act_type': torch.tensor(act_types, dtype=torch.long).unsqueeze(0),
            'act_desc': torch.tensor(act_descs, dtype=torch.long).unsqueeze(0),
            'act_effect_slot': torch.tensor(act_effect_slots, dtype=torch.uint8).unsqueeze(0),
            'act_mask': torch.tensor(masks, dtype=torch.bool).unsqueeze(0),
            'act_race': torch.tensor(act_races, dtype=torch.long).unsqueeze(0),
            'act_attr': torch.tensor(act_attrs, dtype=torch.long).unsqueeze(0),
            'act_code': torch.tensor(act_codes, dtype=torch.long).unsqueeze(0),
            'act_place': torch.tensor(act_places, dtype=torch.long).unsqueeze(0),
            'act_operation': torch.tensor(act_operations, dtype=torch.uint8).unsqueeze(0),
            'act_summon_method': torch.tensor(act_summon_methods, dtype=torch.uint8).unsqueeze(0),
            'act_response': torch.tensor(act_responses, dtype=torch.int16).unsqueeze(0),
            'act_signature': torch.tensor(act_signatures, dtype=torch.uint8).unsqueeze(0),
            'act_context': torch.tensor(act_contexts, dtype=torch.float16).unsqueeze(0),
            'act_target_code': torch.tensor(act_target_codes, dtype=torch.int32).unsqueeze(0),
            'act_target_value': torch.tensor(act_target_values, dtype=torch.uint8).unsqueeze(0),
            'act_controller': torch.tensor(act_controllers, dtype=torch.uint8).unsqueeze(0),
            'act_location': torch.tensor(act_locations, dtype=torch.uint8).unsqueeze(0),
            'act_sequence': torch.tensor(act_sequences, dtype=torch.uint8).unsqueeze(0),
            'act_position': torch.tensor(act_positions, dtype=torch.uint8).unsqueeze(0),
        }

    def encode(self, snapshot: GameSnapshot, player_id: int) -> dict:
        g = snapshot.global_data
        global_vec = self._encode_global_vector(g, player_id)
        player_context = self._encode_player_context(g, player_id)
        phase_category = self._encode_phase_category(g.phase_id)
        
        # 核心优化：直接预分配全量固定形状的 NumPy 数组，天然自带 Padding
        card_indices = np.full(
            MAX_CARDS,
            self.card_vocabulary.padding_id,
            dtype=np.int64,
        )
        card_overlay_indices = np.full(
            MAX_CARDS,
            self.card_vocabulary.padding_id,
            dtype=np.int64,
        )
        card_alias_indices = np.full(
            MAX_CARDS,
            self.card_vocabulary.padding_id,
            dtype=np.int16,
        )
        card_overlay_rest_indices = np.full(
            (MAX_CARDS, CARD_ADDITIONAL_OVERLAY_SLOTS),
            self.card_vocabulary.padding_id,
            dtype=np.int16,
        )
        card_reason_bytes = np.zeros(
            (MAX_CARDS, CARD_REASON_BYTES), dtype=np.uint8
        )
        card_status_bits = np.zeros(
            (MAX_CARDS, CARD_STATUS_BITS), dtype=np.uint8
        )
        card_owner_roles = np.zeros(MAX_CARDS, dtype=np.uint8)
        card_relation_indices = np.full(
            (MAX_CARDS, CARD_RELATION_SLOTS),
            MAX_CARDS,
            dtype=np.uint8,
        )
        card_relation_types = np.zeros(
            (MAX_CARDS, CARD_RELATION_SLOTS), dtype=np.uint8
        )
        card_counter_type_bytes = np.zeros(
            (MAX_CARDS, CARD_COUNTER_SLOTS, CARD_COUNTER_TYPE_BYTES),
            dtype=np.uint8,
        )
        card_counter_values = np.zeros(
            (MAX_CARDS, CARD_COUNTER_SLOTS), dtype=np.float16
        )
        card_races = np.zeros(MAX_CARDS, dtype=np.int64)
        card_attrs = np.zeros(MAX_CARDS, dtype=np.int64)
        card_setcodes = np.zeros((MAX_CARDS, 4), dtype=np.int64)
        card_feats = np.zeros(
            (MAX_CARDS, CARD_NUMERIC_FEATURE_DIM),
            dtype=np.float32,
        )
        card_zones = np.zeros(MAX_CARDS, dtype=np.int64)
        card_positions = np.zeros(MAX_CARDS, dtype=np.int64)
        masks = np.zeros(MAX_CARDS, dtype=np.bool_)

        # ==========================================
        # 1. 处理场上/手牌/墓地实体 
        # ==========================================
        op_known = []
        if hasattr(snapshot, 'known_hand_codes'):
            op_known = snapshot.known_hand_codes[1 - player_id].copy()
            hidden_capacity = 0
            for e in snapshot.entities:
                if e.owner != player_id:
                    if e.location == Zone.HAND: hidden_capacity += 1
                    elif e.location in [Zone.MZONE, Zone.SZONE] and not bool(e.is_public):
                        hidden_capacity += 1
                        
            while len(op_known) > hidden_capacity and len(op_known) > 0:
                op_known.pop(0)

        for i, e in enumerate(snapshot.entities[:MAX_CARDS]):
            card_zones[i] = self._encode_zone_category(e.location)
            card_positions[i] = self._encode_position_category(e.position)
            is_visible = self._is_entity_visible_to_player(e, player_id)
            has_public_state = is_visible
            is_tracked_by_memory = False
            visible_code = e.code

            if not is_visible and len(op_known) > 0:
                if e.location == Zone.HAND or (e.location in [Zone.MZONE, Zone.SZONE] and not is_visible):
                    visible_code = op_known.pop(0)
                    is_visible = True
                    is_tracked_by_memory = True

            if is_visible:
                card_indices[i] = self._encode_card_code(visible_code)
                pos_x, pos_y = self._get_coords(player_id, e.owner, e.location, e.sequence)

                if has_public_state:
                    visible_type = e.type_mask
                    visible_race = e.race
                    visible_attr = e.attribute
                    visible_level = e.level
                    visible_rank = e.rank
                    visible_link_rating = e.link_rating
                    visible_lscale = e.lscale
                    visible_rscale = e.rscale
                    visible_link_marker = e.link_marker
                    visible_current_atk = e.current_atk
                    visible_current_def = e.current_def
                    visible_base_atk = e.base_atk
                    visible_base_def = e.base_def
                    visible_overlay_count = e.overlay_count
                    visible_counter_count = e.counter_count
                    visible_is_equipped = e.is_equipped
                    visible_setcodes = e.setcodes
                    mask = getattr(e, 'used_effect_mask', 0)
                else:
                    # 记牌只恢复已知身份；隐藏后的 Core 动态状态不得随查询泄漏
                    from card_reader import card_db
                    printed = card_db.get_full_stats(visible_code)
                    visible_type = printed[0]
                    visible_race = printed[1]
                    visible_attr = printed[2]
                    visible_level = printed[3]
                    visible_lscale = printed[4]
                    visible_rscale = printed[5]
                    visible_link_marker = printed[6]
                    visible_rank = printed[7]
                    visible_link_rating = (
                        printed[3] if visible_type & 0x4000000 else 0
                    )
                    if visible_link_rating:
                        visible_level = 0
                    visible_current_atk = printed[8]
                    visible_current_def = printed[9]
                    visible_base_atk = printed[8]
                    visible_base_def = printed[9]
                    visible_overlay_count = 0
                    visible_counter_count = 0
                    visible_is_equipped = False
                    visible_setcodes = printed[10]
                    mask = 0
                
                used_eff_0 = 1.0 if (mask & (1 << 0)) else 0.0
                used_eff_1 = 1.0 if (mask & (1 << 1)) else 0.0
                used_eff_2 = 1.0 if (mask & (1 << 2)) else 0.0
                used_eff_3 = 1.0 if (mask & (1 << 3)) else 0.0
                used_eff_4 = 1.0 if (mask & (1 << 4)) else 0.0
                used_eff_5 = 1.0 if (mask & (1 << 5)) else 0.0
                used_eff_6 = 1.0 if (mask & (1 << 6)) else 0.0
                used_eff_7 = 1.0 if (mask & (1 << 7)) else 0.0

                feat_numeric = [
                    1.0 if e.owner == player_id else -1.0, e.sequence / 10.0,
                    visible_current_atk / 4000.0, visible_current_def / 4000.0,
                    visible_base_atk / 4000.0, visible_base_def / 4000.0,
                    pos_x, pos_y, visible_level / 12.0,
                    visible_lscale / 13.0, visible_rscale / 13.0,
                    1.0 if e.is_public else (0.5 if is_tracked_by_memory else 0.0),
                    min(visible_overlay_count / 5.0, 1.0),
                    min(visible_counter_count / 10.0, 1.0),
                    1.0 if visible_is_equipped else 0.0,
                    used_eff_0, used_eff_1, used_eff_2, used_eff_3, used_eff_4, used_eff_5, used_eff_6, used_eff_7,
                    visible_rank / 13.0, visible_link_rating / 8.0,
                ]
                feat = feat_numeric + [1.0 if (visible_type & (1<<idx)) else 0.0 for idx in range(32)] + [1.0 if (visible_link_marker & (1<<idx)) else 0.0 for idx in range(9)]
                card_feats[i] = feat

                card_races[i] = visible_race % 30
                card_attrs[i] = visible_attr % 10

                raw_sc = visible_setcodes if isinstance(visible_setcodes, (list, tuple)) else [visible_setcodes]
                card_setcodes[i] = [(s % 4096) for s in (list(raw_sc) + [0]*4)[:4]]
                masks[i] = True
                overlay_code = (
                    getattr(e, 'top_overlay_code', 0) if has_public_state else 0
                )
                card_overlay_indices[i] = (
                    self._encode_card_code(overlay_code)
                    if overlay_code
                    else self.card_vocabulary.padding_id
                )

                if has_public_state:
                    current_code = int(getattr(e, 'current_code', 0) or 0)
                    if current_code and current_code != int(e.code):
                        card_alias_indices[i] = self._encode_card_code(current_code)

                    for slot, overlay_code in enumerate(
                        list(getattr(e, 'overlay_codes', []))[1:1 + CARD_ADDITIONAL_OVERLAY_SLOTS]
                    ):
                        card_overlay_rest_indices[i, slot] = self._encode_card_code(
                            overlay_code
                        )

                    reason = int(getattr(e, 'reason', 0) or 0) & 0xFFFFFFFF
                    card_reason_bytes[i] = [
                        (reason >> (byte_index * 8)) & 0xFF
                        for byte_index in range(CARD_REASON_BYTES)
                    ]
                    card_status_bits[i] = [
                        int(bool(getattr(e, 'is_disabled', False))),
                        int(bool(getattr(e, 'properly_summoned', False))),
                        int(bool(getattr(e, 'is_forbidden', False))),
                    ]
                    original_owner = int(getattr(e, 'original_owner', -1))
                    if original_owner in (0, 1):
                        card_owner_roles[i] = 1 if original_owner == player_id else 2

                    relations = zip(
                        list(getattr(e, 'relation_indices', [])),
                        list(getattr(e, 'relation_types', [])),
                    )
                    for slot, (target_index, relation_type) in enumerate(relations):
                        if slot >= CARD_RELATION_SLOTS:
                            break
                        if 0 <= int(target_index) < MAX_CARDS:
                            card_relation_indices[i, slot] = int(target_index)
                            card_relation_types[i, slot] = int(relation_type)

                    for slot, (counter_type, counter_count) in enumerate(
                        list(getattr(e, 'counter_items', []))[:CARD_COUNTER_SLOTS]
                    ):
                        counter_type = int(counter_type) & 0xFFFF
                        card_counter_type_bytes[i, slot] = [
                            (counter_type >> (byte_index * 8)) & 0xFF
                            for byte_index in range(CARD_COUNTER_TYPE_BYTES)
                        ]
                        card_counter_values[i, slot] = min(
                            max(float(counter_count) / 10.0, 0.0),
                            25.0,
                        )

            else:
                if e.location == Zone.HAND:
                    card_indices[i] = self.card_vocabulary.hidden_hand_id
                elif e.location == Zone.SZONE:
                    card_indices[i] = self.card_vocabulary.hidden_szone_id
                else:
                    card_indices[i] = self.card_vocabulary.unknown_id
                card_overlay_indices[i] = self.card_vocabulary.padding_id
                masks[i] = True
                card_feats[i, :4] = [-1.0, e.sequence / 10.0, -1.0, -1.0]

        # ==========================================
        # 2. 处理上帝视角卡组残像 (MAX_DECK_CARDS = 75)
        # ==========================================
        MAX_DECK_CARDS = 75
        my_deck = (snapshot.p0_deck_codes + snapshot.p0_extra_codes) if player_id == 0 else (snapshot.p1_deck_codes + snapshot.p1_extra_codes)

        deck_idx = np.full(
            MAX_DECK_CARDS,
            self.card_vocabulary.padding_id,
            dtype=np.int64,
        )
        deck_race = np.zeros(MAX_DECK_CARDS, dtype=np.int64)
        deck_attr = np.zeros(MAX_DECK_CARDS, dtype=np.int64)
        deck_setcodes = np.zeros((MAX_DECK_CARDS, 4), dtype=np.int64)
        deck_masks = np.zeros(MAX_DECK_CARDS, dtype=np.bool_)

        for i, code in enumerate(my_deck[:MAX_DECK_CARDS]):
            deck_race[i], deck_attr[i], deck_setcodes[i] = (
                self._read_deck_card_metadata(code)
            )
                
            deck_idx[i] = self._encode_card_code(code)
            deck_masks[i] = True

        # ==========================================
        # 2.5 处理连锁堆栈 (MAX_CHAIN = 12)
        # ==========================================
        MAX_CHAIN = 12
        c_masks = np.zeros(MAX_CHAIN, dtype=np.bool_)
        c_card_idx = np.zeros(MAX_CHAIN, dtype=np.int64)
        c_effect_slots = np.zeros(MAX_CHAIN, dtype=np.uint8)
        c_desc = np.zeros(MAX_CHAIN, dtype=np.int64)
        c_context = np.zeros((MAX_CHAIN, CHAIN_CONTEXT_DIM), dtype=np.float16)
        c_zones = np.zeros((MAX_CHAIN, 2), dtype=np.int64)
        c_positions = np.zeros(MAX_CHAIN, dtype=np.int64)
        if hasattr(snapshot, 'chain_stack'):
            for i, item in enumerate(snapshot.chain_stack[:MAX_CHAIN]):
                c_card_idx[i] = self._encode_card_code(item['code'])
                c_effect_slots[i] = self._encode_effect_slot(
                    item.get('effect_slot', -1)
                )
                c_desc[i] = int(item.get('desc', 0)) % 1024
                c_context[i] = self._encode_chain_context(item, player_id)
                c_zones[i] = [
                    self._encode_zone_category(item.get('hl', 0)),
                    self._encode_zone_category(item.get('l', 0)),
                ]
                c_positions[i] = self._encode_position_category(item.get('hp', 0))
                c_masks[i] = True

        # ==========================================
        # 2.6 处理动作历史雷达 (MAX_HISTORY = 8)
        # ==========================================
        MAX_HISTORY = 8
        h_masks = np.zeros(MAX_HISTORY, dtype=np.bool_)
        h_card_idx = np.zeros(MAX_HISTORY, dtype=np.int64)
        h_effect_slots = np.zeros(MAX_HISTORY, dtype=np.uint8)

        if hasattr(snapshot, 'history_stack'):
            for i, item in enumerate(snapshot.history_stack[:MAX_HISTORY]):
                h_card_idx[i] = self._encode_card_code(item['code'])
                h_effect_slots[i] = self._encode_effect_slot(
                    item.get('effect_slot', -1)
                )
                h_masks[i] = True

        # ==========================================
        # 3. 最终打包 (直接包装为 Tensor)
        # ==========================================
        act_dict = self.encode_actions(snapshot.valid_actions, snapshot, player_id)

        # 哨兵雷达：在强转和 clip 之前，进行深度数值自检，绝不静默隐藏 Bug
        has_nan = np.isnan(card_feats).any()
        has_inf = np.isinf(card_feats).any()
        has_pos_extreme = (card_feats > 65500.0).any()
        has_neg_extreme = (card_feats < -65500.0).any() # 捕获异常负向数值（Underflow）

        if has_nan or has_inf or has_pos_extreme or has_neg_extreme:
            print("\n🛑 [FeatureEncoder 核心警报] 决斗管线中惊现破坏性投毒特征数据！已自动动态拦截排毒！")
            print("   -> 🔍 毒素成因: " + 
                  ("【NaN 空值】 " if has_nan else "") + 
                  ("【Inf 无穷大】 " if has_inf else "") + 
                  ("【正向超界 Max限】 " if has_pos_extreme else "") + 
                  ("【负向超界 Min限】" if has_neg_extreme else ""))
            
            # 开启全息追踪，遍历当前盘面实体，精准揪出下毒的卡片
            from card_reader import card_db
            found_culprit = False
            for idx, e in enumerate(snapshot.entities[:MAX_CARDS]):
                # 采用 abs() 绝对值判定，同时将正向膨胀与负向脏内存（如未初始化内存里的负数）一网打尽
                if abs(e.current_atk) > 260000000 or abs(e.current_def) > 260000000 or abs(e.base_atk) > 260000000 or abs(e.base_def) > 260000000 or e.code < 0:
                    found_culprit = True
                    c_name = "未知卡片"
                    try: c_name = card_db.get_card_name(e.code)
                    except Exception as ex:
                        print(f"   -> ⚠️ 追踪异常: 无法查询卡片代码 {e.code} 的名称，可能是非法代码或数据库未收录 | 错误详情: {ex}")
                    print(f"   ├─🎯 涉案实体索引: [{idx}] | 区域: {Zone.get_str(e.location)} | 槽位序号: {e.sequence}")
                    print(f"   ├─🃏 涉案卡片身份: 【{c_name}】 (真实卡密 Code: {e.code}) | 实际拥有者: 玩家 {e.owner}")
                    print(f"   └─📊 崩溃现场面板: 当前ATK={e.current_atk} | 当前DEF={e.current_def} | 原始ATK={e.base_atk} | 原始DEF={e.base_def}")
            
            if not found_culprit:
                print(f"   -> 💡 提示: 异常未源于可见怪兽面板，可能由于全局常数计算或隐藏特征通道越界（特征矩阵极值跨度: {np.min(card_feats)} ~ {np.max(card_feats)}）")
            print("   -> 🛡️ 安全状态: 雷达已强制重置该高维特征至 float16 物理极限安全带宽，对局继续，主进程 GPU 运算图保持绝对纯净。\n")

        # 有多大拉多大：利用 float16 临界区安全上限进行动态截断，吸收极值，阻止管道崩溃
        card_feats = np.clip(card_feats, -65500.0, 65500.0)
        
        base_dict = {
            'global': torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0),
            'phase': torch.tensor(
                [phase_category],
                dtype=torch.long,
            ).unsqueeze(0),
            'player_context': torch.tensor(
                player_context,
                dtype=torch.long,
            ).unsqueeze(0),
            # 部署/竞技默认启用；训练 Worker 会按整局覆写并把同一掩码存入 PPO
            'deck_film_mask': torch.ones((1, 1), dtype=torch.bool),
            
            'card_idx': torch.from_numpy(card_indices).unsqueeze(0),
            'card_alias_idx': torch.from_numpy(card_alias_indices).unsqueeze(0),
            'card_overlay_idx': torch.from_numpy(card_overlay_indices).unsqueeze(0),
            'card_overlay_rest_idx': torch.from_numpy(card_overlay_rest_indices).unsqueeze(0),
            'card_reason_bytes': torch.from_numpy(card_reason_bytes).unsqueeze(0),
            'card_status_bits': torch.from_numpy(card_status_bits).unsqueeze(0),
            'card_owner_role': torch.from_numpy(card_owner_roles).unsqueeze(0),
            'card_relation_idx': torch.from_numpy(card_relation_indices).unsqueeze(0),
            'card_relation_type': torch.from_numpy(card_relation_types).unsqueeze(0),
            'card_counter_type_bytes': torch.from_numpy(card_counter_type_bytes).unsqueeze(0),
            'card_counter_value': torch.from_numpy(card_counter_values).unsqueeze(0),
            'field_zone_mask': torch.from_numpy(
                self._encode_field_zone_mask(
                    getattr(g, 'disabled_field_mask', 0),
                    player_id,
                )
            ).unsqueeze(0),
            'card_race': torch.from_numpy(card_races).unsqueeze(0), 
            'card_attr': torch.from_numpy(card_attrs).unsqueeze(0), 
            'card_setcodes': torch.from_numpy(card_setcodes).unsqueeze(0), 
            'card_feats': torch.from_numpy(card_feats).unsqueeze(0),
            'card_zone': torch.from_numpy(card_zones).unsqueeze(0),
            'card_position': torch.from_numpy(card_positions).unsqueeze(0),
            'padding_mask': torch.from_numpy(masks).unsqueeze(0),
            
            'deck_idx': torch.from_numpy(deck_idx).unsqueeze(0),
            'deck_race': torch.from_numpy(deck_race).unsqueeze(0),
            'deck_attr': torch.from_numpy(deck_attr).unsqueeze(0),
            'deck_setcodes': torch.from_numpy(deck_setcodes).unsqueeze(0),
            'deck_mask': torch.from_numpy(deck_masks).unsqueeze(0),
            
            'c_mask': torch.from_numpy(c_masks).unsqueeze(0),
            'c_card_idx': torch.from_numpy(c_card_idx).unsqueeze(0),
            'c_effect_slot': torch.from_numpy(c_effect_slots).unsqueeze(0),
            'c_desc': torch.from_numpy(c_desc).unsqueeze(0),
            'c_context': torch.from_numpy(c_context).unsqueeze(0),
            'c_zone': torch.from_numpy(c_zones).unsqueeze(0),
            'c_position': torch.from_numpy(c_positions).unsqueeze(0),
            'h_mask': torch.from_numpy(h_masks).unsqueeze(0),
            'h_card_idx': torch.from_numpy(h_card_idx).unsqueeze(0),
            'h_effect_slot': torch.from_numpy(h_effect_slots).unsqueeze(0),
        }
        
        base_dict.update(self._encode_deck_profile(snapshot, player_id))
        base_dict.update(act_dict)
        return base_dict

if __name__ == "__main__":
    enc = GalateaEncoder()
    print("Encoder (with Semantic Active) Ready.")
