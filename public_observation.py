# 本文件维护 Core 公开提示、已知卡组位置及快照恢复所需状态，不推测额外效果或奖励

from dataclasses import dataclass, replace

from game_constants import LocationInfo, Position, Zone
from data_types import TRANSITION_EVENT_HISTORY_SIZE
from event_history import MAX_PUBLIC_NOTIFICATIONS


MAX_PUBLIC_HINTS = 128
DECK_POSITION_CAPACITY = 256
PUBLIC_CONTEXT_DIM = 128
PUBLIC_OBSERVATION_FORMAT_VERSION = 1
CARD_QUESTION = 38723936
_CONTIGUOUS_ZONES = {Zone.DECK, Zone.HAND, Zone.GRAVE, Zone.REMOVED, Zone.EXTRA}

# 观测与共享内存共用规格；只传身份/参数，不在轨迹内保存展开的语义向量
PUBLIC_INPUT_SPECS = {
    "hint_mask": ((MAX_PUBLIC_HINTS,), "bool"),
    "hint_kind": ((MAX_PUBLIC_HINTS,), "uint8"),
    "hint_role": ((MAX_PUBLIC_HINTS,), "uint8"),
    "hint_host": ((MAX_PUBLIC_HINTS,), "uint8"),
    "hint_location": ((MAX_PUBLIC_HINTS, 4), "uint8"),
    "hint_value_bytes": ((MAX_PUBLIC_HINTS, 4), "uint8"),
    "hint_card_idx": ((MAX_PUBLIC_HINTS,), "int32"),
    "hint_semantic_card_idx": ((MAX_PUBLIC_HINTS,), "int32"),
    "hint_semantic_slot": ((MAX_PUBLIC_HINTS,), "uint8"),
    "hint_binding": ((MAX_PUBLIC_HINTS,), "uint8"),
    "hint_count": ((MAX_PUBLIC_HINTS,), "int32"),
    "hint_age": ((MAX_PUBLIC_HINTS,), "float16"),
    "known_deck_idx": ((2, DECK_POSITION_CAPACITY), "int32"),
    "known_deck_flags": ((2, DECK_POSITION_CAPACITY), "uint8"),
    "public_state_flags": ((6,), "uint8"),
    "event_public_mask": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS), "bool"),
    "event_public_kind": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS), "uint8"),
    "event_public_code": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS), "int32"),
    "event_public_location": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS, 4), "uint8"),
    "event_public_chain": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS), "uint8"),
    "event_public_offset": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS), "int32"),
    "event_public_subtype": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS), "uint8"),
    "event_public_value": ((TRANSITION_EVENT_HISTORY_SIZE, MAX_PUBLIC_NOTIFICATIONS, 4), "uint8"),
}


@dataclass(frozen=True)
class PublicHint:
    """保存提示事实；提示存在不等于对应规则当前生效"""

    player: int
    location: int
    kind: int
    value: int
    count: int = 1
    turn: int = 0
    visibility: int = 3


@dataclass(frozen=True)
class KnownDeckCard:
    """记录已合法公开的卡组位置，不保存隐藏洗牌顺序"""

    player: int
    sequence: int
    code: int
    faceup: bool = False
    visibility: int = 3


def location_key(raw):
    """把普通表示变化排除出实例位置键，叠放素材仍保留层序"""
    player, zone, sequence, position = LocationInfo.decode(int(raw))
    return player, zone, sequence, position if zone & Zone.OVERLAY else 0


def encode_location_key(key):
    """把实例坐标重新包装为 Core 的四字节位置值"""
    player, zone, sequence, position = key
    return player | (zone << 8) | (sequence << 16) | (position << 24)


class PublicObservationState:
    """按原始消息顺序维护公开状态，查询刷新不会覆盖这些提示"""

    def __init__(self, deck_counts=(None, None)):
        """初始化单局缓存；未知卡组数量不能被解释为零张"""
        self.deck_counts = list(deck_counts)
        self.deck_cards = [{}, {}]
        self.card_hints = {}
        self.player_hints = [{}, {}]
        self.reversed = False
        self.reversal_known = True
        self.complete = True
        self.remaining_composition_known = all(count is not None for count in deck_counts)

    def card_hint(self, raw, kind, value, turn):
        """更新最新数值提示或说明引用；撤销只删除一份引用"""
        key = location_key(raw)
        if key[0] not in (0, 1) or key[1] == 0 or not 1 <= kind <= 7:
            raise ValueError("invalid public card hint")
        scalar, descriptions = self.card_hints.get(key, (None, {}))
        descriptions = dict(descriptions)
        if kind <= 5:
            scalar = PublicHint(key[0], encode_location_key(key), kind, value, turn=turn)
        elif kind == 6:
            previous = descriptions.get(value)
            count = previous.count + 1 if previous else 1
            descriptions[value] = PublicHint(key[0], encode_location_key(key), 6, value, count, turn)
        elif value in descriptions:
            previous = descriptions[value]
            if previous.count > 1:
                descriptions[value] = replace(previous, count=previous.count - 1)
            else:
                del descriptions[value]
        else:
            # 中途初始化或客户端重建可缺少先前 ADD，不制造负引用
            self.complete = False
        if scalar is not None or descriptions:
            self.card_hints[key] = scalar, descriptions
        else:
            self.card_hints.pop(key, None)

    def player_hint(self, player, kind, desc, turn):
        """维护玩家说明的实际引用生命周期，不按回合自行清空"""
        if player not in (0, 1) or kind not in (6, 7):
            raise ValueError("invalid public player hint")
        hints = self.player_hints[player]
        previous = hints.get(desc)
        if kind == 6:
            hints[desc] = PublicHint(player, -1, 8, desc, previous.count + 1 if previous else 1, turn)
        elif previous is None:
            self.complete = False
        elif previous.count > 1:
            hints[desc] = replace(previous, count=previous.count - 1)
        else:
            del hints[desc]

    def grave_blocked(self, player):
        """遵守公开客户端的谜题查看限制，不抹去玩家先前合法记忆"""
        return CARD_QUESTION in self.player_hints[player]

    def clear_zone_hints(self, player, zone, *, descriptions_only=False):
        """在洗牌重建前清除对应映射，避免 Core 重发 ADD 后重复计数"""
        for key in tuple(self.card_hints):
            if key[:2] != (player, zone):
                continue
            scalar, _ = self.card_hints[key]
            if descriptions_only and scalar is not None:
                self.card_hints[key] = scalar, {}
            else:
                del self.card_hints[key]

    def shuffle_hints(self, player, zone, *, descriptions_reemitted=False):
        """洗牌丢弃无法追踪的实例提示，手牌说明可由 Core 后续 ADD 重建"""
        for key, (scalar, descriptions) in self.card_hints.items():
            if key[:2] == (player, zone) and (scalar or (descriptions and not descriptions_reemitted)):
                self.complete = False
        self.clear_zone_hints(player, zone)

    def remap_hints(self, mapping):
        """同时搬迁已知实例坐标，包含随宿主交换的超量素材提示"""
        moved = {}
        for old, new in mapping.items():
            for key in tuple(self.card_hints):
                is_host_material = key[:3] == (old[0], old[1] | Zone.OVERLAY, old[2])
                if key != old and not is_host_material:
                    continue
                scalar, descriptions = self.card_hints.pop(key)
                target = (new[0], new[1] | Zone.OVERLAY, new[2], key[3]) if is_host_material else new
                raw = encode_location_key(target)
                moved[target] = (
                    replace(scalar, player=new[0], location=raw) if scalar else None,
                    {desc: replace(hint, player=new[0], location=raw) for desc, hint in descriptions.items()},
                )
        self.card_hints.update(moved)

    def shuffle_set_hints(self, old_locations, new_locations):
        """只跟随公开的盖卡重排；隐藏目标位置时清理映射而不偷看查询身份"""
        mapping = {}
        for old_raw, new_raw in zip(old_locations, new_locations):
            old = location_key(old_raw)
            if new_raw:
                mapping[old] = location_key(new_raw)
            else:
                if self.card_hints.pop(old, None) is not None:
                    self.complete = False
        self.remap_hints(mapping)

    def _shift_hints(self, player, zone, sequence, delta):
        """连续区域插入/删除时平移实例提示，固定场上槽位不参与"""
        if zone not in _CONTIGUOUS_ZONES:
            return
        moved = {}
        for key in tuple(self.card_hints):
            if key[:2] == (player, zone) and key[2] >= sequence:
                scalar, descriptions = self.card_hints.pop(key)
                new_key = (*key[:2], key[2] + delta, key[3])
                raw = encode_location_key(new_key)
                moved[new_key] = (
                    replace(scalar, location=raw) if scalar else None,
                    {desc: replace(hint, location=raw) for desc, hint in descriptions.items()},
                )
        self.card_hints.update(moved)

    def _shift_overlay_hints(self, key, delta):
        """超量素材插拔时平移同一宿主下后续层序，不混入普通卡位"""
        moved = {}
        for other in tuple(self.card_hints):
            if other[:3] == key[:3] and other[3] >= key[3]:
                scalar, descriptions = self.card_hints.pop(other)
                target = (*other[:3], other[3] + delta)
                raw = encode_location_key(target)
                moved[target] = (replace(scalar, location=raw) if scalar else None,
                                 {desc: replace(hint, location=raw) for desc, hint in descriptions.items()})
        self.card_hints.update(moved)

    def move_hints(self, old_raw, new_raw):
        """随实例移动保留说明引用，普通移动清除最新数值提示"""
        old, new = location_key(old_raw), location_key(new_raw)
        if old == new:
            return
        _, descriptions = self.card_hints.pop(old, (None, {}))
        if not old[1] & Zone.OVERLAY:
            for key in tuple(self.card_hints):
                if key[:3] != (old[0], old[1] | Zone.OVERLAY, old[2]):
                    continue
                scalar, materials = self.card_hints.pop(key)
                if not new[1] & Zone.OVERLAY and new[1] in (Zone.MZONE, Zone.SZONE):
                    target = (new[0], new[1] | Zone.OVERLAY, new[2], key[3])
                    raw = encode_location_key(target)
                    self.card_hints[target] = (replace(scalar, player=new[0], location=raw) if scalar else None,
                                              {desc: replace(hint, player=new[0], location=raw) for desc, hint in materials.items()})
                else:
                    # 不能证明素材仍挂在离场宿主上时，不给新占位卡继承提示
                    self.complete = False
        if old[0] in (0, 1) and old[1]:
            if old[1] & Zone.OVERLAY:
                self._shift_overlay_hints((*old[:3], old[3] + 1), -1)
            else:
                self._shift_hints(old[0], old[1], old[2] + 1, -1)
        if new[0] in (0, 1) and new[1]:
            if new[1] & Zone.OVERLAY:
                self._shift_overlay_hints(new, 1)
            else:
                self._shift_hints(new[0], new[1], new[2], 1)
            # 新实例不能继承被占用固定槽位的旧提示
            self.card_hints.pop(new, None)
            if descriptions:
                raw = encode_location_key(new)
                self.card_hints[new] = None, {
                    desc: replace(hint, player=new[0], location=raw)
                    for desc, hint in descriptions.items()
                }

    def set_deck_card(self, player, sequence, raw_code):
        """登记合法公开位置；未知身份清除旧事实，高位只表示表侧"""
        if player not in (0, 1):
            raise ValueError("invalid deck-top player")
        count = self.deck_counts[player]
        if count is None:
            return
        if not 0 <= sequence < count:
            raise ValueError("public deck position is outside the logical deck")
        code = raw_code & 0x7FFFFFFF
        if code:
            self.deck_cards[player][sequence] = KnownDeckCard(player, sequence, code, bool(raw_code & 0x80000000))
        else:
            self.deck_cards[player].pop(sequence, None)

    def deck_top(self, player, offset, raw_code):
        """用消费该消息时的旧逻辑数量解释偏移，兼容先38后DRAW"""
        if player not in (0, 1):
            raise ValueError("invalid deck-top player")
        count = self.deck_counts[player]
        if count is not None:
            self.set_deck_card(player, count - 1 - offset, raw_code)

    def reverse_decks(self):
        """反转双方已知位置；不把未公开的卡组顺序变成已知"""
        self.reversed = not self.reversed
        for player in (0, 1):
            count = self.deck_counts[player]
            if count is not None:
                self.remap_hints({key: (player, Zone.DECK, count - 1 - key[2], key[3])
                                  for key in tuple(self.card_hints) if key[:2] == (player, Zone.DECK)})
            self.deck_cards[player] = {
                count - 1 - seq: replace(card, sequence=count - 1 - seq)
                for seq, card in self.deck_cards[player].items()
            } if count is not None else {}

    def draw(self, player, count, hand_start=0):
        """移除顶部位置并返回此前已知的抽入卡；不读取隐藏DRAW卡密"""
        size = self.deck_counts[player]
        if size is None:
            self.deck_cards[player].clear()
            return ()
        if count > size:
            raise ValueError("draw exceeds logical deck size")
        known = tuple(self.deck_cards[player].get(size - 1 - i) for i in range(count))
        for index in range(count):
            old = player | (Zone.DECK << 8) | ((size - 1 - index) << 16)
            new = player | (Zone.HAND << 8) | ((hand_start + index) << 16)
            self.move_hints(old, new)
        self.deck_counts[player] = size - count
        self.deck_cards[player] = {seq: card for seq, card in self.deck_cards[player].items() if seq < size - count}
        return known

    def move_deck_card(self, old_raw, new_raw, code, public):
        """跟随单卡移动更新位置，只有已公开身份能进入位置记牌"""
        old_p, old_z, old_s, _ = LocationInfo.decode(old_raw)
        new_p, new_z, new_s, _ = LocationInfo.decode(new_raw)
        known = None
        if old_p in (0, 1) and old_z == Zone.DECK:
            known = self.deck_cards[old_p].pop(old_s, None)
            if known is not None and code and known.code != code:
                known = None
                self.complete = False
            self.deck_cards[old_p] = {
                seq - int(seq > old_s): replace(card, sequence=seq - int(seq > old_s))
                for seq, card in self.deck_cards[old_p].items()
            }
            if self.deck_counts[old_p] is not None:
                self.deck_counts[old_p] = max(0, self.deck_counts[old_p] - 1)
        if new_p in (0, 1) and new_z == Zone.DECK:
            if self.deck_counts[new_p] is None:
                return known
            self.deck_cards[new_p] = {
                seq + int(seq >= new_s): replace(card, sequence=seq + int(seq >= new_s))
                for seq, card in self.deck_cards[new_p].items()
            }
            self.deck_counts[new_p] += 1
            if self.deck_counts[new_p] >= DECK_POSITION_CAPACITY:
                raise ValueError("deck exceeds fixed Core sequence capacity")
            if public or known is not None:
                visible_code = code if public else known.code
                self.set_deck_card(new_p, new_s, visible_code)
        return known

    def snapshot(self):
        """返回不可变记录组成的快照，旧训练样本不引用实时缓存"""
        hints = []
        for key in sorted(self.card_hints):
            scalar, descriptions = self.card_hints[key]
            if scalar is not None:
                hints.append(scalar)
            hints.extend(descriptions[desc] for desc in sorted(descriptions))
        for player in (0, 1):
            hints.extend(self.player_hints[player][desc] for desc in sorted(self.player_hints[player]))
        cards = tuple(card for mapping in self.deck_cards for _, card in sorted(mapping.items()))
        flags = (self.complete, self.reversal_known, self.reversed, self.grave_blocked(0), self.grave_blocked(1), self.remaining_composition_known)
        return tuple(hints), cards, tuple(bool(value) for value in flags)


def public_observation_descriptor():
    """公开固定输入规格和生命周期语义，参与 V4 结构身份校验"""
    return {
        "format_version": PUBLIC_OBSERVATION_FORMAT_VERSION,
        "inputs": PUBLIC_INPUT_SPECS,
        "hint_capacity": MAX_PUBLIC_HINTS,
        "deck_position_capacity": DECK_POSITION_CAPACITY,
        "context_width": PUBLIC_CONTEXT_DIM,
        "overflow": "reject_incomplete_observation",
        "hint_semantics": "exact_slot_or_explicit_parent_code_context",
        "lifetime": "core_notifications_and_client_instance_lifecycle",
        "reward_changes": False,
    }
