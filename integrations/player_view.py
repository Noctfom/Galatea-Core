"""Core-owned BO1 player-view ingress, separate from authoritative Core traces.

No native engine, transport connection, policy or training job is started here.
The legacy host profile must specify cached versus uncached query semantics.
Sparse/hidden attributes remain unknown in the sidecar cache. A diagnostic V4
projection is NOT an assertion of full native-observation equivalence.
"""
from copy import deepcopy
from dataclasses import dataclass
import struct

import core_query as q
from core_message_protocol import CORE_MESSAGE_PROTOCOL, decision_player_from_message, is_empty_chain_prompt
from integrations.duel_session import SessionGate, SessionProtocolError

PLAYER_VIEW_FORMAT_VERSION = 1
MAX_PACKET_BYTES = 65535
MAX_VIEW_MESSAGES = 100_000
ZONES = (1, 2, 4, 8, 16, 32, 64)
DECISIONS = frozenset((10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 132, 140, 141, 142, 143))
PUBLIC_MESSAGES = frozenset((5, 32, 34, 35, 36, 37, 40, 41, 42, 56, 61, 63, 65,
    70, 71, 72, 73, 74, 75, 76, 83, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102,
    110, 111, 112, 113, 114, 120, 130, 131, 133, 165, 170))

# This maps already-decoded fields, not a second binary query decoder.
QUERY_FIELDS = {
    q.QUERY_CODE: ('code',),
    q.QUERY_POSITION: ('position_raw', 'controller', 'location', 'sequence', 'position'),
    q.QUERY_ALIAS: ('current_code',), q.QUERY_TYPE: ('current_type',),
    q.QUERY_LEVEL: ('current_level',), q.QUERY_RANK: ('current_rank',),
    q.QUERY_ATTRIBUTE: ('current_attr',), q.QUERY_RACE: ('current_race',),
    q.QUERY_ATTACK: ('current_atk',), q.QUERY_DEFENSE: ('current_def',),
    q.QUERY_BASE_ATTACK: ('base_atk',), q.QUERY_BASE_DEFENSE: ('base_def',),
    q.QUERY_REASON: ('reason',), q.QUERY_REASON_CARD: ('reason_card',),
    q.QUERY_EQUIP_CARD: ('equip_target', 'is_equip_source'),
    q.QUERY_TARGET_CARD: ('target_locations',), q.QUERY_OVERLAY_CARD: ('overlay_codes',),
    q.QUERY_COUNTERS: ('counter_items', 'counter_count'), q.QUERY_OWNER: ('original_owner',),
    q.QUERY_STATUS: ('status_mask', 'properly_summoned', 'is_disabled', 'is_forbidden'),
    q.QUERY_LSCALE: ('lscale',), q.QUERY_RSCALE: ('rscale',),
    q.QUERY_LINK: ('link_rating', 'link_marker'),
}

# 在线接入合同独立于模型结构版本，本批只说明和审计既有输入
PLAYER_OBSERVATION_CONTRACT = "core_v4_player_observation_audit_v1"
V4_REQUIRED_QUERY_FLAGS = q.PUBLIC_CARD_QUERY_FLAGS & ~q.QUERY_REASON_CARD
INPUT_SOURCE_GROUPS = {
    "global": ("start_and_visible_events", "refresh_counts_without_hidden_identities"),
    "cards": ("visible_events_and_declared_query_cache", "missing_dynamic_values_are_not_cdb_defaults"),
    "deck": ("own_initial_deck_and_confirmed_movements", "no_opponent_construction_or_deck_order"),
    "chain": ("visible_chain_messages", "no_private_choice_reconstruction"),
    "effect_history": ("visible_activation_messages", "preserve_existing_eight_entry_semantics"),
    "transitions": ("confirmed_responses_and_visible_events", "opponent_response_boundaries_unproven"),
    "public_context": ("visible_hints_and_known_positions", "unsupported_delivery_is_not_an_empty_hint_list"),
    "actions": ("current_own_prompt_and_existing_packer", "preserve_candidate_order_masks_and_response_bytes"),
}


def input_source_group(name):
    """按实际模型输入名称定位来源组，不引入新张量或默认补值"""
    if name in ('global', 'phase', 'player_context', 'field_zone_mask'):
        return 'global'
    if name.startswith('card_') or name == 'padding_mask':
        return 'cards'
    if name.startswith('deck_'):
        return 'deck'
    if name.startswith('c_'):
        return 'chain'
    if name.startswith('h_'):
        return 'effect_history'
    if name.startswith('event_'):
        return 'transitions'
    if name.startswith(('hint_', 'known_deck_')) or name == 'public_state_flags':
        return 'public_context'
    if name.startswith('act_'):
        return 'actions'
    return 'unclassified'


def player_observation_contract():
    """返回不依赖引擎、模型或资产加载的在线来源说明"""
    return {'contract_id': PLAYER_OBSERVATION_CONTRACT, 'player_view_format_version': PLAYER_VIEW_FORMAT_VERSION,
        'core_message_protocol': CORE_MESSAGE_PROTOCOL,
        'groups': {name: {'source': source, 'rule': rule} for name, (source, rule) in INPUT_SOURCE_GROUPS.items()},
        'availability_states': ['known', 'known_zero', 'unknown', 'hidden', 'not_applicable', 'invalidated'],
        'query_contract': {'type6': 'replace_zone_slot_set', 'type7': 'update_one_slot',
            'required_model_flags': V4_REQUIRED_QUERY_FLAGS,
            'omitted_attributes': 'merge_only_with_declared_cache_and_same_instance',
            'omitted_equip': 'zero_only_with_requested_flag_evidence',
            'empty_row': 'length4_absent_slot', 'flags_zero_row': 'present_without_attributes'},
        'remaining_blockers': ['opponent_response_boundaries_unavailable', 'card_hint160_not_forwarded',
            'dynamic_attribute_coverage_unproven', 'reload162_not_supported_by_current_profile'],
        'permissions': {'inference': False, 'deterministic_replay': False, 'behavior_clone': False},
        'caller_can_override_readiness': False, 'changes_network_inputs': False}


def compare_encoded_observations(expected, actual, *, sample_limit=8):
    """只比较编码字段和差异位置，不把原始卡密、动作或输入值写入报告"""
    import torch
    if type(sample_limit) is not int or not 0 <= sample_limit <= 32:
        raise ValueError('comparison sample_limit must be 0..32')
    fields = []
    for name in sorted(set(expected) | set(actual)):
        item = {'field': name, 'group': input_source_group(name)}
        if name not in expected or name not in actual:
            item.update(equal=False, reason='missing_actual' if name not in actual else 'unexpected_actual')
        else:
            left, right = expected[name], actual[name]
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                raise TypeError('encoded observations must contain tensors only')
            item.update(shape=list(left.shape), dtype=str(left.dtype))
            if left.shape != right.shape or left.dtype != right.dtype:
                item.update(equal=False, reason='layout_mismatch', actual_shape=list(right.shape), actual_dtype=str(right.dtype))
            else:
                left, right = left.detach().cpu(), right.detach().cpu()
                valid = (not left.is_floating_point() or bool(torch.isfinite(left).all() and torch.isfinite(right).all()))
                equal = bool(valid and torch.equal(left, right))
                item.update(equal=equal, reason='equal' if equal else 'value_mismatch' if valid else 'nonfinite')
                if not equal:
                    different = torch.nonzero(left.ne(right), as_tuple=False)
                    item.update(different_elements=int(different.shape[0]), sample_indices=different[:sample_limit].tolist())
        fields.append(item)
    return {'fields': fields, 'field_count': len(fields), 'matching_fields': sum(row['equal'] for row in fields),
        'different_fields': [row['field'] for row in fields if not row['equal']],
        'all_fields_equal': bool(fields) and all(row['equal'] for row in fields),
        'unclassified_fields': [row['field'] for row in fields if row['group'] == 'unclassified']}


def public_query_updates(env, perspective):
    """在Core暂停边界生成派生查询刷新，不发送主卡组顺序或对手隐藏身份"""
    import ctypes
    if type(perspective) is not int or perspective not in (0, 1):
        raise SessionProtocolError('invalid query perspective')
    flags = q.PUBLIC_CARD_QUERY_FLAGS & ~q.QUERY_REASON_CARD
    buffer = (ctypes.c_byte * 8192)()
    for actor in (0, 1):
        for zone in (4, 8, 2, 16, 32, 64):
            count = 7 if zone == 4 else 8 if zone == 8 else env.query_zone_count(actor, zone)
            if count is None or not 0 <= count <= 256:
                raise SessionProtocolError('public Core query_field_count unavailable or invalid')
            rows = []
            for sequence in range(count):
                length = env.lib.query_card(env.pduel, actor, zone, sequence, flags, buffer, 0)
                if not 4 <= length <= ctypes.sizeof(buffer):
                    raise SessionProtocolError('external query_card length invalid')
                row = ctypes.string_at(ctypes.addressof(buffer), length)
                if struct.unpack_from('<I', row)[0] != len(row):
                    raise SessionProtocolError('external query_card length mismatch')
                if len(row) == 4:
                    rows.append(row)
                    continue
                actual = struct.unpack_from('<I', row, 4)[0]
                if actual & 3 != 3 or len(row) < 16:
                    raise SessionProtocolError('external query lacks code/position layout')
                public = zone == 16 or (zone != 2 and bool(row[15] & 5))
                if actor != perspective and not public:
                    row = struct.pack('<II', 8, 0)
                rows.append(row)
            yield encode_zone_refresh(actor, zone, rows)


class IncompletePlayerViewError(SessionProtocolError):
    """The source cannot prove all of the unchanged V4 model's inputs."""


@dataclass(frozen=True)
class ViewCard:
    known_flags: int
    masked: bool
    fields: dict


@dataclass(frozen=True)
class PlayerViewProjection:
    """Display/test-only snapshot plus an inseparable source/coverage report."""
    snapshot: object
    coverage: dict


class YGOProFrameDecoder:
    """Bounded incremental uint16-LE framing; payloads retain their STOC type.

    The caller handles lobby/control packets. Only STOC_GAME_MSG (type1) payloads
    belong in accept_game_message; network framing is never a raw Core chunk.
    """
    def __init__(self):
        self._buffer = bytearray()
        self.failed = False
        self.closed = False

    def feed(self, data):
        if self.failed or self.closed:
            raise SessionProtocolError('frame decoder failed/closed; create a new connection decoder')
        try:
            if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_PACKET_BYTES + 2:
                raise SessionProtocolError('invalid or oversized network input block')
            self._buffer.extend(data)
            packets, offset = [], 0
            while len(self._buffer) - offset >= 2:
                size = struct.unpack_from('<H', self._buffer, offset)[0]
                if size == 0:
                    raise SessionProtocolError('empty YGOPro packet')
                if len(self._buffer) - offset < size + 2:
                    break
                packets.append(bytes(self._buffer[offset + 2:offset + 2 + size]))
                offset += size + 2
                if len(packets) > 1024:
                    raise SessionProtocolError('network frame flood')
            del self._buffer[:offset]
            return tuple(packets)
        except ValueError:
            self._buffer.clear()
            self.failed = True
            raise

    def finish(self):
        self.closed = True
        if self.failed or self._buffer:
            self.failed = True
            self._buffer.clear()
            raise SessionProtocolError('connection ended with an incomplete/failed frame')


def encode_player_start(perspective, decks, *, lp=(8000, 8000), rule=5):
    """Build the pinned host Start layout without any hidden deck order/Side."""
    if type(perspective) is not int or perspective not in (0, 1) or type(rule) is not int or not 1 <= rule <= 5:
        raise SessionProtocolError('unsupported BO1 start perspective/rule')
    counts = [len(decks[seat][part]) for seat in ('0', '1') for part in ('main', 'extra')]
    if any(value > 256 for value in counts) or len(lp) != 2 or any(type(n) is not int or not 0 <= n < 2**32 for n in lp):
        raise SessionProtocolError('invalid start counts/LP')
    return bytes((4, perspective, rule)) + struct.pack('<IIHHHH', *lp, *counts)


def encode_zone_refresh(actor, zone, rows):
    """Build UpdateData; caller-owned query buffers are copied before sending."""
    if type(actor) is not int or actor not in (0, 1) or type(zone) is not int or zone not in ZONES:
        raise SessionProtocolError('invalid query zone/actor')
    if not isinstance(rows, (list, tuple)) or len(rows) > 256:
        raise SessionProtocolError('invalid query row collection')
    if any(not isinstance(row, (bytes, bytearray)) for row in rows):
        raise SessionProtocolError('query rows must be bytes')
    rows = tuple(bytes(row) for row in rows)
    if zone in (4, 8) and len(rows) != (7 if zone == 4 else 8):
        raise SessionProtocolError('invalid query row count')
    for row in rows:
        if not 4 <= len(row) <= 8192 or struct.unpack_from('<I', row)[0] != len(row):
            raise SessionProtocolError('invalid query row framing')
    message = bytes((6, actor, zone)) + b''.join(rows)
    if len(message) > MAX_PACKET_BYTES - 1:
        raise SessionProtocolError('query update exceeds STOC packet boundary')
    return message


def visible_core_message(message, perspective):
    """The existing WindBot masked BO1 profile, now owned by Core.

    Unknown kinds fail closed. This conservative profile omits card hints160;
    consuming it therefore does not prove complete native V4 public context.
    """
    if type(perspective) is not int or perspective not in (0, 1) or not message:
        raise SessionProtocolError('invalid player view')
    kind, raw = message[0], bytearray(message)
    if kind in DECISIONS:
        if decision_player_from_message(kind, raw[1:]) != perspective:
            return None
        groups = []
        if kind in (15, 20):
            groups.append((raw[5], 6, 8))
        elif kind == 26:
            count = raw[6]
            groups.append((count, 7, 8))
            offset = 7 + count * 8
            groups.append((raw[offset], offset + 1, 8))
        for count, offset, stride in groups:
            for index in range(count):
                entry = offset + index * stride
                if raw[entry + 4] != perspective:
                    raw[entry:entry + 4] = b'\0' * 4
        return bytes(raw)
    if kind == 2:
        hint, actor = raw[1:3]
        if actor not in (0, 1):
            raise SessionProtocolError('invalid hint recipient')
        if ((hint in (1, 2, 3, 4, 5) and actor == perspective)
                or (hint in (6, 7, 8, 9) and actor != perspective) or hint == 10):
            return bytes(raw)
        return None
    if kind in (90, 33, 39):
        actor, count = raw[1:3]
        if actor != perspective:
            for index in range(count):
                offset = 3 + index * 4
                code = struct.unpack_from('<I', raw, offset)[0]
                if kind != 90 or not code & 0x80000000:
                    raw[offset:offset + 4] = b'\0' * 4
        return bytes(raw)
    if kind == 50:
        actor, zone, position = raw[9], raw[10], raw[12]
        if actor != perspective and (zone & (1 | 2) or position & 0xA):
            raw[1:5] = b'\0' * 4
        return bytes(raw)
    if kind in (53, 54, 60, 62, 64):
        actor = raw[5]
        position = raw[9] if kind == 53 else raw[8]
        if kind == 54 or (actor != perspective and position & 0xA):
            raw[1:5] = b'\0' * 4
        return bytes(raw)
    if kind == 55:
        for offset in (1, 9):
            actor, zone, position = raw[offset + 4], raw[offset + 5], raw[offset + 7]
            if actor != perspective and (zone & (1 | 2) or position & 0xA):
                raw[offset:offset + 4] = b'\0' * 4
        return bytes(raw)
    if kind == 120:
        return bytes(raw) if raw[1] == perspective else None
    if kind == 31:
        from core_message_protocol import parse_confirm_cards
        reveal = parse_confirm_cards(raw[1:])
        cards = reveal['cards']
        hand_reveal = cards and all(card['location'] == 2 for card in cards)
        return bytes(raw) if reveal['player'] == perspective or hand_reveal else None
    if kind in (30, 38):
        return bytes(raw)
    if kind in (3, 4, 81, 160, 163, 164):
        return None
    if kind in PUBLIC_MESSAGES:
        return bytes(raw)
    raise SessionProtocolError(f'unsupported external-view Core message {kind}; no unsafe broadcast')


class PlayerViewAdapter:
    """One physical seat/one duel, with explicit query-cache semantics.

    Expects one unframed host GameMessage at a time, in arrival order. Start and
    query refreshes never call DuelState.update or create transition events.
    Omitted per-card flags are not invented. No opponent deck/Side is accepted.
    """
    def __init__(self, perspective, *, query_mode, requested_flags=None, own_main=None, own_extra=None):
        if type(perspective) is not int or perspective not in (0, 1) or query_mode not in ('replace', 'merge'):
            raise SessionProtocolError('invalid perspective/query cache mode')
        if requested_flags is not None and (type(requested_flags) is not int
                or requested_flags < 0 or requested_flags & ~q.PUBLIC_CARD_QUERY_FLAGS):
            raise SessionProtocolError('unsupported requested query flags')
        for cards in (own_main, own_extra):
            if cards is not None and (not isinstance(cards, (list, tuple)) or len(cards) > 256
                    or any(type(code) is not int or not 0 < code < 2**32 for code in cards)):
                raise SessionProtocolError('invalid own stable deck')
        self.perspective, self.query_mode, self.requested_flags = perspective, query_mode, requested_flags
        self._own_main = tuple(own_main) if own_main is not None else None
        self._own_extra = tuple(own_extra) if own_extra is not None else None
        self.brain = None
        self.gate = SessionGate()
        self.sequence = self.event_count = 0
        self.terminal = self.failed = False
        self._cards, self._refreshed = {}, set()
        self._pending_payload = None

    def _start(self, raw):
        if self.brain is not None or len(raw) != 19 or raw[1] != self.perspective or not 1 <= raw[2] <= 5:
            raise SessionProtocolError('invalid, spectator/tag or repeated Start')
        lp0, lp1, d0, e0, d1, e1 = struct.unpack_from('<IIHHHH', raw, 3)
        counts = ((d0, e0), (d1, e1))
        if max(d0, e0, d1, e1) > 256:
            raise SessionProtocolError('oversized Start zone counts')
        own_counts = counts[self.perspective]
        if any(cards is not None and len(cards) != n for cards, n in zip((self._own_main, self._own_extra), own_counts)):
            raise SessionProtocolError('own stable deck disagrees with Start')
        from gamestate import DuelState
        args = [None, None, None, None]
        args[self.perspective * 2:self.perspective * 2 + 2] = [self._own_main, self._own_extra]
        brain = DuelState(*args, audit_enabled=False)
        brain.my_lp, brain.op_lp = lp0, lp1
        brain._reload_counts = {actor: {zone: (counts[actor][0] if zone == 1 else
            counts[actor][1] if zone == 64 else 0) for zone in ZONES} for actor in (0, 1)}
        brain.public_observation.deck_counts = [d0, d1]
        brain.public_observation.remaining_composition_known = False
        # The conservative host profile cannot prove complete hint delivery.
        brain.public_observation.complete = False
        self.brain = brain

    @staticmethod
    def _rows(data):
        rows, offset = [], 0
        while offset < len(data):
            if len(data) - offset < 4:
                raise SessionProtocolError('truncated query row length')
            size = struct.unpack_from('<I', data, offset)[0]
            if size != 4 and not 8 <= size <= 8192:
                raise SessionProtocolError('invalid query row size')
            if offset + size > len(data) or len(rows) >= 256:
                raise SessionProtocolError('truncated/oversized query rows')
            row = data[offset:offset + size]
            if size == 4:
                rows.append(None)
            else:
                flags = struct.unpack_from('<I', row, 4)[0]
                if flags & ~q.PUBLIC_CARD_QUERY_FLAGS:
                    raise SessionProtocolError('unsupported query flags; fixed legacy profile required')
                parsed = q.parse_legacy_card_query(row)
                values = {name: parsed[name] for flag, names in QUERY_FIELDS.items() if flags & flag for name in names}
                rows.append((flags, values))
            offset += size
        return rows

    def _view_card(self, actor, zone, sequence, row):
        if row is None:
            return None
        flags, values = row
        key = actor, zone, sequence
        previous = self._cards.get(key)
        event = self.brain.field_map[actor].get(zone, {}).get(sequence, {})
        if flags & q.QUERY_POSITION and (values['controller'], values['location'], values['sequence']) != key:
            raise SessionProtocolError('query location disagrees with its envelope/slot')
        if 'code' in values:
            values['code'] &= 0x7FFFFFFF
        same = previous is not None and ('code' not in values or values['code'] == previous.fields.get('code'))
        fields = deepcopy(previous.fields) if self.query_mode == 'merge' and same else {}
        known = previous.known_flags if fields else 0
        fields.update(values)
        known |= flags
        position = fields.get('position', event.get('pos', 0))
        private = zone == 1 or (actor != self.perspective and
            (zone == 2 or (zone != 16 and not position & 5)))
        private |= zone == 16 and self.brain.public_observation.grave_blocked(self.perspective)
        if private:
            fields = {'controller': actor, 'location': zone, 'sequence': sequence}
            known = 0
            if position:
                fields['position'] = position
                known |= q.QUERY_POSITION
            return ViewCard(known, True, fields)
        # Public Core deliberately omits EQUIP_CARD when no equip target exists,
        # even without cache. Only an explicit requested-flags contract proves0.
        if self.requested_flags is not None and self.requested_flags & q.QUERY_EQUIP_CARD and not flags & q.QUERY_EQUIP_CARD:
            fields.update(equip_target=0, is_equip_source=False)
            known |= q.QUERY_EQUIP_CARD
        return ViewCard(known, False, fields)

    def _query(self, raw):
        minimum = 3 if raw[0] == 6 else 4
        if len(raw) < minimum:
            raise SessionProtocolError('truncated query envelope')
        actor, zone = raw[1:3]
        if actor not in (0, 1) or zone not in ZONES:
            raise SessionProtocolError('invalid query actor/zone')
        rows = self._rows(raw[minimum:])
        if raw[0] == 7:
            sequence = raw[3]
            limit = 7 if zone == 4 else 8 if zone == 8 else self.brain._reload_counts[actor][zone]
            if len(rows) != 1 or sequence >= limit:
                raise SessionProtocolError('invalid incremental query slot/row count')
            if rows[0] is None and zone not in (4, 8):
                raise SessionProtocolError('contiguous zone removal requires a full refresh')
            indices = [sequence]
        else:
            if zone in (4, 8) and len(rows) != (7 if zone == 4 else 8):
                raise SessionProtocolError('full field refresh has wrong slot count')
            if zone not in (4, 8) and any(row is None for row in rows):
                raise SessionProtocolError('contiguous zone refresh contains empty slots')
            indices = list(range(len(rows)))
        # Parse, validate and mask the complete envelope before changing a zone.
        changes = {index: self._view_card(actor, zone, index, row) for index, row in zip(indices, rows)}
        new_zone = {} if raw[0] == 6 else deepcopy(self.brain.field_map[actor].get(zone, {}))
        for index, card in changes.items():
            if card is None:
                new_zone.pop(index, None)
                continue
            old = self.brain.field_map[actor].get(zone, {}).get(index, {})
            fields = deepcopy(card.fields)
            entry = {'code': fields.get('code', 0), 'pos': fields.get('position', old.get('pos', 0)), 'owner': actor}
            entry.update(fields)
            entry['query_flags'] = card.known_flags
            for key in ('current_atk', 'current_def', 'base_atk', 'base_def'):
                if key in entry:
                    entry[key] = max(0, entry[key])
            if old.get('code', 0) and old.get('code') == entry['code']:
                entry['used_effect_mask'] = old.get('used_effect_mask', 0)
            new_zone[index] = entry
        if raw[0] == 6:
            for key in tuple(self._cards):
                if key[:2] == (actor, zone):
                    del self._cards[key]
        for index, card in changes.items():
            key = actor, zone, index
            if card is None:
                self._cards.pop(key, None)
            else:
                self._cards[key] = card
        # 主卡组不暴露牌序，额外库存只校准数量和稀疏缓存，不扩充原生V4实体序列
        if zone not in (1, 64):
            self.brain.field_map[actor][zone] = new_zone
        if raw[0] == 6:
            self.brain._reload_counts[actor][zone] = sum(row is not None for row in rows)
            if zone == 1:
                self.brain.public_observation.deck_counts[actor] = len(rows)
            self._refreshed.add((actor, zone))
        elif zone in (4, 8):
            self.brain._reload_counts[actor][zone] = len(new_zone)

    def _invalidate_instances(self, kind, payload):
        zones = set()
        if kind in (50, 55):
            offsets = (4, 8) if kind == 50 else (4, 12)
            for offset in offsets:
                actor, zone = payload[offset], payload[offset + 1] & 0x7F
                if actor in (0, 1) and zone in ZONES:
                    zones.add((actor, zone))
        elif kind in (33, 39):
            zones.add((payload[0], 2 if kind == 33 else 64))
        elif kind == 53:
            zones.add((payload[4], payload[5]))
        elif kind in (36, 162):
            zones = {(actor, zone) for actor in (0, 1) for zone in ZONES}
        for key in tuple(self._cards):
            if key[:2] in zones:
                del self._cards[key]
        for actor, zone in zones:
            for index, entry in self.brain.field_map[actor].get(zone, {}).items():
                # Keep only facts maintained by visible events, never reuse old
                # dynamic query attributes after a physical-instance boundary.
                self.brain.field_map[actor][zone][index] = {name: value for name, value in entry.items()
                    if name in ('code', 'pos', 'owner', 'used_effect_mask')}

    def accept_game_message(self, message, *, sequence=None):
        """Apply one ordered GameMessage. Failure poisons this adapter, not Core."""
        if self.failed:
            raise SessionProtocolError('player view failed; create a new duel adapter')
        try:
            if (not isinstance(message, (bytes, bytearray)) or not 1 <= len(message) <= MAX_PACKET_BYTES - 1
                    or self.sequence >= MAX_VIEW_MESSAGES
                    or sequence is not None and (type(sequence) is not int or sequence != self.sequence)):
                raise SessionProtocolError('invalid message size/order')
            raw = bytes(message)
            kind = raw[0]
            if kind == 4:
                self._start(raw)
                result = 'start'
            elif self.brain is None:
                raise SessionProtocolError('Start required before events/queries')
            elif kind in (6, 7):
                self._query(raw)
                result = 'query'
            else:
                from gamestate import MessageParser
                if MessageParser.parse(raw) != [raw]:
                    raise SessionProtocolError('expected exactly one fixed-profile Core event')
                if self.terminal or kind in (1, 35, 161):
                    raise SessionProtocolError('unsupported terminal/tag/bulk-exchange event')
                visible = visible_core_message(raw, self.perspective)
                result = 'ignored'
                if visible is not None:
                    self.brain.update(kind, visible[1:])
                    self._invalidate_instances(kind, visible[1:])
                    if kind in (33, 39):
                        actor, count = visible[1:3]
                        zone = 2 if kind == 33 else 64
                        self.brain.field_map[actor][zone] = {i: {'code': struct.unpack_from('<I', visible, 3 + 4*i)[0] & 0x7FFFFFFF,
                            'pos': 0, 'owner': actor} for i in range(count)}
                    if kind == 165 and self.brain.public_observation.grave_blocked(self.perspective):
                        # CARD_QUESTION restricts the viewer, not the graveyard
                        # owner. Do not expose either grave through old queries.
                        for actor in (0, 1):
                            for index, entry in self.brain.field_map[actor].get(16, {}).items():
                                position = entry.get('pos', 0)
                                self.brain.field_map[actor][16][index] = {'code': 0, 'pos': position, 'owner': actor}
                                key = actor, 16, index
                                if key in self._cards:
                                    fields = {'controller': actor, 'location': 16, 'sequence': index, 'position': position}
                                    self._cards[key] = ViewCard(q.QUERY_POSITION if position else 0, True, fields)
                    if (any(len(cards) > 256 for zones in self.brain.field_map.values() for cards in zones.values())
                            or any(len(codes) > 1024 for codes in self.brain.known_hand_codes.values())
                            or len(self.brain.recently_confirmed) > 1024):
                        raise SessionProtocolError('player-view state capacity exceeded')
                    self._refreshed.clear()
                    self.event_count += 1
                    if kind in DECISIONS:
                        self.gate.request(self.perspective, kind)
                        self._pending_payload = visible[1:]
                    if kind == 5:
                        self.terminal = True
                        self.gate.close()
                    result = 'event'
            self.sequence += 1
            return result
        except (ValueError, IndexError, KeyError, struct.error, RuntimeError) as error:
            self.failed = True
            self.gate.close()
            raise SessionProtocolError(f'player-view ingress failed: {error}') from error

    def record_submitted_response(self, submission):
        """Record an actually sent own response, never an unexecuted suggestion."""
        if self.failed or self.brain is None:
            raise SessionProtocolError('no active player view')
        request = self.gate.pending
        raw = self.gate.submit(submission)
        payload, self._pending_payload = self._pending_payload, None
        if is_empty_chain_prompt(request.prompt_type, payload):
            if raw[:4] != b'\xff' * 4:
                self.failed = True
                self.gate.close()
                raise SessionProtocolError('empty chain must submit -1')
        else:
            try:
                self.brain.begin_transition_event(self.perspective, request.prompt_type, raw)
            except (ValueError, RuntimeError):
                self.failed = True
                self.gate.close()
                raise
        return raw

    def card_at(self, actor, zone, sequence):
        """Return a detached field map; missing fields are unknown, not zero."""
        return deepcopy(self._cards.get((actor, zone, sequence)))

    def coverage(self):
        missing = []
        for (actor, zone, index), card in sorted(self._cards.items()):
            if not card.masked:
                absent = q.PUBLIC_CARD_QUERY_FLAGS & ~card.known_flags
                if absent:
                    missing.append({'actor': actor, 'zone': zone, 'sequence': index, 'missing_flags': absent})
        return {'format_version': PLAYER_VIEW_FORMAT_VERSION, 'core_message_protocol': CORE_MESSAGE_PROTOCOL,
            'visibility': 'player_view', 'perspective': self.perspective, 'sequence': self.sequence,
            'query_mode': self.query_mode, 'requested_flags': self.requested_flags,
            'started': self.brain is not None, 'failed': self.failed, 'terminal': self.terminal,
            'event_count': self.event_count, 'refreshed_zones': sorted(self._refreshed),
            'pending_refresh_zones': sorted({(p, z) for p in (0, 1) for z in ZONES if z != 1} - self._refreshed),
            'initial_own_deck_known': self._own_main is not None and self._own_extra is not None,
            'missing_card_attributes': missing, 'full_transition_history_available': False,
            'limitations': ['opponent_response_boundaries_unavailable', 'card_hint160_not_forwarded',
                            'single_player_visibility_not_full_core'],
            'v4_observation_complete': False, 'behavior_clone_eligible': False}

    def observation_readiness(self):
        """提供分组的接入阻塞原因，隐藏信息本身不是缺失公开属性"""
        coverage = self.coverage()
        reasons = []
        if not coverage['started']:
            reasons.append('start_missing')
        if coverage['failed']:
            reasons.append('session_failed')
        if coverage['terminal']:
            reasons.append('session_terminal')
        if not coverage['initial_own_deck_known']:
            reasons.append('own_stable_deck_missing')
        missing = [dict(item, missing_flags=item['missing_flags'] & V4_REQUIRED_QUERY_FLAGS)
                   for item in coverage['missing_card_attributes'] if item['missing_flags'] & V4_REQUIRED_QUERY_FLAGS]
        if missing:
            reasons.append('visible_card_attributes_missing')
        reasons.extend(player_observation_contract()['remaining_blockers'])
        return {'contract_id': PLAYER_OBSERVATION_CONTRACT, 'visibility': 'player_view',
            'perspective': self.perspective, 'core_message_protocol': CORE_MESSAGE_PROTOCOL,
            'query_mode': self.query_mode, 'requested_flags': self.requested_flags,
            'inference_ready': False, 'deterministic_replay_ready': False, 'behavior_clone_eligible': False,
            'blocking_reasons': reasons, 'missing_card_attributes': missing,
            'pending_refresh_zones': coverage['pending_refresh_zones'],
            'permissions_are_independent': True, 'observation_semantics_changed': False}

    def audit_native_snapshot(self, native_snapshot, encoder, *, sample_limit=8):
        """仅对照已有编码，不返回可供策略调用的诊断张量或提升在线资格"""
        from protocol_schema import get_current_protocol_metadata
        projection = self.diagnostic_projection()
        expected = encoder.encode(native_snapshot, player_id=self.perspective)
        actual = encoder.encode(projection.snapshot, player_id=self.perspective)
        report = compare_encoded_observations(expected, actual, sample_limit=sample_limit)
        report.update(contract_id=PLAYER_OBSERVATION_CONTRACT, perspective=self.perspective,
            model_identity=get_current_protocol_metadata(encoder.card_vocabulary),
            readiness=self.observation_readiness(), comparison_proves_online_support=False)
        return report

    def diagnostic_projection(self):
        if self.failed or self.brain is None:
            raise SessionProtocolError('no valid started player view')
        # get_snapshot may finalize pending transitions. A diagnostic read must
        # never create a boundary in the live adapter's event recorder.
        return PlayerViewProjection(deepcopy(self.brain).get_snapshot(), self.coverage())

    def encode_v4(self, encoder):
        """Fail closed until a later contract proves complete V4 input coverage."""
        raise IncompletePlayerViewError('player_view does not prove complete V4 history/hints/attributes; diagnostic projection only')


class PlayerViewConformanceAudit:
    """受控对局的有限玩家视角对照，不驱动模型或改变决策"""

    def __init__(self, *, max_checks=32):
        if type(max_checks) is not int or not 1 <= max_checks <= 256:
            raise ValueError('player-view audit checks must be 1..256')
        self.max_checks = max_checks
        self.views = {}
        self.checked = 0
        self.skipped = 0
        self.counts = {0: {'checks': 0, 'exact': 0}, 1: {'checks': 0, 'exact': 0}}
        self.differences = {}
        self.examples = []
        self.errors = []
        self.window_complete = False

    def start(self, deck0, deck1):
        """宿主只派生Start计数，各适配器仅持有本方稳定构筑"""
        decks = {str(seat): {'main': list(deck.main), 'extra': list(deck.extra)}
                 for seat, deck in enumerate((deck0, deck1))}
        for seat in (0, 1):
            own = decks[str(seat)]
            try:
                view = PlayerViewAdapter(seat, query_mode='replace', requested_flags=V4_REQUIRED_QUERY_FLAGS,
                    own_main=own['main'], own_extra=own['extra'])
                self.views[seat] = view
                view.accept_game_message(encode_player_start(seat, decks))
            except Exception as error:
                self._error(seat, error)

    def _error(self, seat, error):
        """错误仅标记该诊断适配器，不能中断正常RuleBot对局"""
        if seat in self.views:
            self.views[seat].failed = True
        if len(self.errors) < 16:
            self.errors.append({'perspective': seat, 'type': type(error).__name__, 'message': str(error)})

    def observe(self, message):
        """只向各视图传递自身可见的消息，禁止发送对手原始响应"""
        if self.window_complete:
            return
        for seat, view in self.views.items():
            if not view.failed:
                try:
                    view.accept_game_message(message)
                except Exception as error:
                    self._error(seat, error)

    def before_decision(self, env, native_brain, encoder, actor):
        """比较副本快照，额外查询不会更新原生决策所使用的状态"""
        if self.window_complete or self.checked >= self.max_checks:
            self.skipped += 1
            return
        view = self.views.get(actor)
        if view is None or view.failed:
            return
        self.checked += 1
        try:
            for message in public_query_updates(env, actor):
                view.accept_game_message(message)
            snapshot = deepcopy(native_brain).get_snapshot(env)
            report = view.audit_native_snapshot(snapshot, encoder)
            self.counts[actor]['checks'] += 1
            self.counts[actor]['exact'] += int(report['all_fields_equal'])
            for name in report['different_fields']:
                self.differences[name] = self.differences.get(name, 0) + 1
            if len(self.examples) < 8:
                self.examples.append(report)
        except Exception as error:
            self._error(actor, error)

    def submitted(self, actor, response):
        """只登记已提交的本方响应，保持空连锁自动应答语义"""
        if self.window_complete:
            return
        from integrations.duel_session import ResponseSubmission
        view = self.views.get(actor)
        if view is not None and not view.failed:
            try:
                request = view.gate.pending
                if request is None:
                    raise SessionProtocolError('audit response without current own prompt')
                view.record_submitted_response(ResponseSubmission(request.session_id, request.request_id, actor, response))
            except Exception as error:
                self._error(actor, error)
        if self.checked >= self.max_checks:
            self.window_complete = True

    def report(self):
        """相等样本只是当前受控证据，不能升级为真实服务器在线资格"""
        return {'format_version': 1, 'contract': player_observation_contract(),
            'source': 'controlled_masked_host_projection', 'max_checks': self.max_checks,
            'window_complete': self.window_complete,
            'checked_decisions': self.checked, 'skipped_after_limit': self.skipped,
            'perspectives': self.counts, 'different_field_counts': self.differences,
            'examples': self.examples, 'errors': self.errors,
            'readiness': {seat: view.observation_readiness() for seat, view in self.views.items()},
            'online_support_proven': False, 'raw_inputs_included': False}

    def save(self, directory='./system_logs/player_view_audit'):
        """每局保存一次有界报告，失败不改变对局结果"""
        import json
        import time
        from pathlib import Path
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'audit_player_view_{time.time_ns()}.json'
        path.write_text(json.dumps(self.report(), ensure_ascii=False, indent=2), encoding='utf-8')
        return str(path)
