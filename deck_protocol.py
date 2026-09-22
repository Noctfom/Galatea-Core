# 本文件定义卡组构筑、单局画像与未来 BO3 上层之间的稳定数据协议
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Mapping, Optional, Sequence, Tuple


DECK_PROTOCOL_FORMAT_VERSION = 1
MAX_DECK_PROFILE_ENTRIES = 128


class DeckSection(IntEnum):
    """标记卡片在完整构筑中的分区，零值专供张量填充"""

    PADDING = 0
    MAIN = 1
    EXTRA = 2
    SIDE = 3


def _normalize_codes(values: Iterable[int], label: str) -> Tuple[int, ...]:
    """把外部卡号整理成不可变正整数序列，并尽早拒绝损坏数据"""
    normalized = []
    for value in values or ():
        if isinstance(value, bool):
            raise ValueError(f"{label} contains a boolean card code")
        try:
            code = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} contains a non-integer card code: {value!r}") from error
        if code <= 0:
            raise ValueError(f"{label} contains a non-positive card code: {code}")
        normalized.append(code)
    return tuple(normalized)


def _section_counts(values: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """生成与原始排列无关的卡号—数量序列"""
    return tuple(sorted(Counter(int(code) for code in values).items()))


def _content_hash(payload: Mapping[str, object]) -> str:
    """为规范化卡组内容生成稳定身份，不把显示名称混入身份"""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DeckEntry:
    """表示卡组画像中的一类卡片及其分区和投入数量"""

    code: int
    section: DeckSection
    copies: int

    def __post_init__(self) -> None:
        if int(self.code) <= 0:
            raise ValueError("deck entry code must be positive")
        try:
            section = DeckSection(self.section)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid deck section: {self.section!r}") from error
        if section == DeckSection.PADDING:
            raise ValueError("padding cannot be a real deck entry section")
        if int(self.copies) <= 0:
            raise ValueError("deck entry copies must be positive")
        object.__setattr__(self, "code", int(self.code))
        object.__setattr__(self, "section", section)
        object.__setattr__(self, "copies", int(self.copies))


@dataclass(frozen=True)
class DeckProfile:
    """单局策略可见的稳定己方卡组画像；结构上不包含 Side Deck"""

    main: Tuple[int, ...] = field(default_factory=tuple)
    extra: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "main", _normalize_codes(self.main, "main deck"))
        object.__setattr__(self, "extra", _normalize_codes(self.extra, "extra deck"))

    @property
    def profile_id(self) -> str:
        """返回只由主卡组和额外卡组构成的去重身份"""
        return _content_hash(
            {
                "format_version": DECK_PROTOCOL_FORMAT_VERSION,
                "main": _section_counts(self.main),
                "extra": _section_counts(self.extra),
            }
        )

    def entries(self) -> Tuple[DeckEntry, ...]:
        """按分区和卡号输出无序集合形式的模型画像"""
        entries = [
            DeckEntry(code, DeckSection.MAIN, copies)
            for code, copies in _section_counts(self.main)
        ]
        entries.extend(
            DeckEntry(code, DeckSection.EXTRA, copies)
            for code, copies in _section_counts(self.extra)
        )
        return tuple(entries)

    def to_payload(self) -> dict:
        """序列化单局画像；该载荷有意不存在 side 字段"""
        return {
            "format_version": DECK_PROTOCOL_FORMAT_VERSION,
            "profile_id": self.profile_id,
            "main": list(self.main),
            "extra": list(self.extra),
        }


@dataclass(frozen=True)
class DeckSpec:
    """保存完整构筑，供资源管理、组卡层和未来换备流程使用"""

    main: Tuple[int, ...] = field(default_factory=tuple)
    extra: Tuple[int, ...] = field(default_factory=tuple)
    side: Tuple[int, ...] = field(default_factory=tuple)
    name: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "main", _normalize_codes(self.main, "main deck"))
        object.__setattr__(self, "extra", _normalize_codes(self.extra, "extra deck"))
        object.__setattr__(self, "side", _normalize_codes(self.side, "side deck"))
        object.__setattr__(self, "name", str(self.name or ""))
        object.__setattr__(self, "source", str(self.source or ""))

    @property
    def deck_id(self) -> str:
        """返回包含 Side Deck、但不依赖文件名和卡片排列的构筑身份"""
        return _content_hash(
            {
                "format_version": DECK_PROTOCOL_FORMAT_VERSION,
                "main": _section_counts(self.main),
                "extra": _section_counts(self.extra),
                "side": _section_counts(self.side),
            }
        )

    def duel_profile(self) -> DeckProfile:
        """派生当前 BO1 策略唯一允许接收的主卡组/额外卡组画像"""
        return DeckProfile(main=self.main, extra=self.extra)

    def entries(self, *, include_side: bool = True) -> Tuple[DeckEntry, ...]:
        """输出完整构筑的无序分区条目，供未来组卡编码器复用"""
        entries = list(self.duel_profile().entries())
        if include_side:
            entries.extend(
                DeckEntry(code, DeckSection.SIDE, copies)
                for code, copies in _section_counts(self.side)
            )
        return tuple(entries)

    def to_payload(self) -> dict:
        """序列化完整构筑，供外部管理与未来比赛控制层保存"""
        return {
            "format_version": DECK_PROTOCOL_FORMAT_VERSION,
            "deck_id": self.deck_id,
            "name": self.name,
            "source": self.source,
            "main": list(self.main),
            "extra": list(self.extra),
            "side": list(self.side),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DeckSpec":
        """从受版本约束的外部载荷恢复完整构筑"""
        if not isinstance(payload, Mapping):
            raise ValueError("deck payload must be an object")
        version = payload.get("format_version", DECK_PROTOCOL_FORMAT_VERSION)
        if int(version) != DECK_PROTOCOL_FORMAT_VERSION:
            raise ValueError(f"unsupported deck protocol version: {version!r}")
        spec = cls(
            main=tuple(payload.get("main", ()) or ()),
            extra=tuple(payload.get("extra", ()) or ()),
            side=tuple(payload.get("side", ()) or ()),
            name=str(payload.get("name", "") or ""),
            source=str(payload.get("source", "") or ""),
        )
        declared_id = payload.get("deck_id")
        if declared_id not in (None, "") and str(declared_id) != spec.deck_id:
            raise ValueError("deck payload identity does not match its contents")
        return spec


@dataclass(frozen=True)
class DuelSummary:
    """为未来 BO3/组卡层保留的单局摘要，不作为当前策略网络输入"""

    duel_number: int
    winner_role: int
    self_went_first: bool
    turn_count: int
    own_profile_id: str = ""
    public_opponent_cards: Tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if int(self.duel_number) < 1:
            raise ValueError("duel_number must be at least 1")
        if int(self.winner_role) not in (-1, 0, 1):
            raise ValueError("winner_role must be -1, 0 or 1")
        if int(self.turn_count) < 0:
            raise ValueError("turn_count cannot be negative")
        object.__setattr__(
            self,
            "public_opponent_cards",
            _normalize_codes(self.public_opponent_cards, "public opponent cards"),
        )


@dataclass(frozen=True)
class MatchContext:
    """保存局间比赛状态；当前单局 DuelState 不读取此对象"""

    match_id: str = ""
    best_of: int = 1
    duel_number: int = 1
    self_wins: int = 0
    opponent_wins: int = 0
    previous_duels: Tuple[DuelSummary, ...] = field(default_factory=tuple)
    self_selected_first: Optional[bool] = None

    def __post_init__(self) -> None:
        best_of = int(self.best_of)
        duel_number = int(self.duel_number)
        self_wins = int(self.self_wins)
        opponent_wins = int(self.opponent_wins)
        if best_of < 1 or best_of % 2 == 0:
            raise ValueError("best_of must be a positive odd number")
        if duel_number < 1 or duel_number > best_of:
            raise ValueError("duel_number must be within the match length")
        if self_wins < 0 or opponent_wins < 0:
            raise ValueError("match wins cannot be negative")
        if self_wins + opponent_wins >= duel_number:
            raise ValueError("match wins are inconsistent with duel_number")
        object.__setattr__(self, "match_id", str(self.match_id or ""))
        object.__setattr__(self, "previous_duels", tuple(self.previous_duels or ()))

    def policy_inputs(self) -> dict:
        """明确当前协议不会把局间上下文或 Side Deck 注入单局网络"""
        return {}
