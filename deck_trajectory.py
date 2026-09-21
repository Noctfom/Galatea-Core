# 本文件实现卡组画像登记、轨迹静态字段去重与 PPO 批次无损重建
from __future__ import annotations

from typing import Dict, Mapping

import torch

from deck_protocol import DeckProfile


DECK_TRAJECTORY_FORMAT_VERSION = 1
MAX_DECK_PROFILE_ENTRIES = 128
DECK_STATIC_OBSERVATION_KEYS = frozenset({
    "deck_race",
    "deck_attr",
    "deck_setcodes",
})

_PROFILE_TENSOR_KEYS = (
    "profile_hash",
    "profile_card_idx",
    "profile_section",
    "profile_initial_count",
    "profile_mask",
)
_METADATA_TENSOR_KEYS = (
    "metadata_card_idx",
    "metadata_race",
    "metadata_attr",
    "metadata_setcodes",
)


def is_deck_static_observation(key: str) -> bool:
    """判断某个旧网络输入是否可由卡片 token 的静态元数据重建"""
    return key in DECK_STATIC_OBSERVATION_KEYS


def _profile_hash_tensor(profile_id: str) -> torch.Tensor:
    """把十六进制画像身份保存成 weights_only 可安全加载的定长字节"""
    try:
        raw = bytes.fromhex(str(profile_id))
    except ValueError as error:
        raise ValueError("deck profile id must be hexadecimal") from error
    if len(raw) != 32:
        raise ValueError("deck profile id must contain 32 SHA-256 bytes")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _profile_hash_key(value: torch.Tensor) -> bytes:
    """把画像哈希张量还原为可用于目录去重的不可变键"""
    row = value.detach().cpu().to(torch.uint8).reshape(-1)
    if row.numel() != 32:
        raise ValueError("deck profile hash tensor must have shape [32]")
    return bytes(row.tolist())


class DeckProfileRegistry:
    """Worker 侧目录：每种稳定画像只登记一次，每步仅记录其索引"""

    def __init__(self) -> None:
        self._profile_indices: Dict[str, int] = {}
        self._profiles = []
        self._metadata: Dict[int, tuple] = {}
        self._last_observation = None

    def register_profile(self, profile: DeckProfile, card_vocabulary) -> int:
        """登记不含 Side 的稳定卡组画像，并返回 Worker 局部索引"""
        if not isinstance(profile, DeckProfile):
            raise TypeError("profile must be a DeckProfile")
        profile_id = profile.profile_id
        existing = self._profile_indices.get(profile_id)
        if existing is not None:
            return existing

        entries = profile.entries()
        if len(entries) > MAX_DECK_PROFILE_ENTRIES:
            raise ValueError(
                f"deck profile has {len(entries)} unique entries; "
                f"maximum is {MAX_DECK_PROFILE_ENTRIES}"
            )
        card_idx = torch.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=torch.long)
        section = torch.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=torch.uint8)
        initial_count = torch.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=torch.uint8)
        mask = torch.zeros(MAX_DECK_PROFILE_ENTRIES, dtype=torch.bool)
        for index, entry in enumerate(entries):
            token = int(card_vocabulary.encode(entry.code))
            card_idx[index] = token
            section[index] = int(entry.section)
            initial_count[index] = min(int(entry.copies), 255)
            mask[index] = True

        local_index = len(self._profiles)
        self._profile_indices[profile_id] = local_index
        self._profiles.append(
            {
                "profile_hash": _profile_hash_tensor(profile_id),
                "profile_card_idx": card_idx,
                "profile_section": section,
                "profile_initial_count": initial_count,
                "profile_mask": mask,
            }
        )
        return local_index

    def register_observation(self, observation: Mapping[str, torch.Tensor]) -> None:
        """登记本轮实际出现的 token 静态元数据，覆盖罕见的动态入卡情况"""
        missing = {
            "deck_idx",
            "deck_race",
            "deck_attr",
            "deck_setcodes",
            "deck_mask",
        }.difference(observation)
        if missing:
            raise ValueError(f"deck observation is missing fields: {sorted(missing)}")

        tokens = observation["deck_idx"].detach().cpu().reshape(-1)
        races = observation["deck_race"].detach().cpu().reshape(-1)
        attrs = observation["deck_attr"].detach().cpu().reshape(-1)
        setcodes = observation["deck_setcodes"].detach().cpu().reshape(-1, 4)
        mask = observation["deck_mask"].detach().cpu().reshape(-1).bool()
        if not (
            tokens.numel() == races.numel() == attrs.numel() == mask.numel()
            and setcodes.shape[0] == tokens.numel()
        ):
            raise ValueError("deck observation fields have inconsistent shapes")

        current_observation = (tokens, races, attrs, setcodes, mask)
        if self._last_observation is not None and all(
            torch.equal(current, previous)
            for current, previous in zip(
                current_observation,
                self._last_observation,
            )
        ):
            return

        for position in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist():
            token = int(tokens[position])
            if token <= 0:
                continue
            metadata = (
                int(races[position]),
                int(attrs[position]),
                tuple(int(value) for value in setcodes[position].tolist()),
            )
            previous = self._metadata.get(token)
            if previous is not None and previous != metadata:
                raise ValueError(
                    f"deck token {token} has conflicting static metadata"
                )
            self._metadata[token] = metadata
        # 连续决策常共享同一卡组状态；保存小快照以跳过重复 Python 遍历
        self._last_observation = tuple(
            value.clone()
            for value in current_observation
        )

    def export(self) -> dict:
        """导出仅含张量的安全目录，供临时轨迹文件跨进程传递"""
        if not self._profiles:
            raise ValueError("cannot export an empty deck profile registry")
        payload = {
            "format_version": torch.tensor(
                [DECK_TRAJECTORY_FORMAT_VERSION],
                dtype=torch.int16,
            )
        }
        for key in _PROFILE_TENSOR_KEYS:
            payload[key] = torch.stack([row[key] for row in self._profiles], dim=0)

        metadata_tokens = sorted(self._metadata)
        payload["metadata_card_idx"] = torch.tensor(
            metadata_tokens,
            dtype=torch.long,
        )
        payload["metadata_race"] = torch.tensor(
            [self._metadata[token][0] for token in metadata_tokens],
            dtype=torch.long,
        )
        payload["metadata_attr"] = torch.tensor(
            [self._metadata[token][1] for token in metadata_tokens],
            dtype=torch.long,
        )
        payload["metadata_setcodes"] = torch.tensor(
            [self._metadata[token][2] for token in metadata_tokens],
            dtype=torch.long,
        ).reshape(-1, 4)
        return payload


def validate_deck_profile_bundle(bundle: Mapping[str, torch.Tensor]) -> None:
    """严格校验 Worker 卡组目录，拒绝损坏索引进入 PPO 合并池"""
    if not isinstance(bundle, Mapping):
        raise ValueError("deck profile bundle must be a mapping")
    required = {"format_version", *_PROFILE_TENSOR_KEYS, *_METADATA_TENSOR_KEYS}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"deck profile bundle is missing fields: {sorted(missing)}")
    version = bundle["format_version"].detach().cpu().reshape(-1)
    if version.numel() != 1 or int(version[0]) != DECK_TRAJECTORY_FORMAT_VERSION:
        raise ValueError("deck profile bundle format version does not match")

    profile_count = int(bundle["profile_hash"].shape[0])
    if profile_count <= 0 or tuple(bundle["profile_hash"].shape[1:]) != (32,):
        raise ValueError("deck profile hash table has an invalid shape")
    expected_profile_shapes = {
        "profile_card_idx": (profile_count, MAX_DECK_PROFILE_ENTRIES),
        "profile_section": (profile_count, MAX_DECK_PROFILE_ENTRIES),
        "profile_initial_count": (profile_count, MAX_DECK_PROFILE_ENTRIES),
        "profile_mask": (profile_count, MAX_DECK_PROFILE_ENTRIES),
    }
    for key, expected_shape in expected_profile_shapes.items():
        if tuple(bundle[key].shape) != expected_shape:
            raise ValueError(f"{key} has shape {tuple(bundle[key].shape)}, expected {expected_shape}")

    metadata_count = int(bundle["metadata_card_idx"].numel())
    if tuple(bundle["metadata_race"].shape) != (metadata_count,):
        raise ValueError("metadata_race shape does not match metadata_card_idx")
    if tuple(bundle["metadata_attr"].shape) != (metadata_count,):
        raise ValueError("metadata_attr shape does not match metadata_card_idx")
    if tuple(bundle["metadata_setcodes"].shape) != (metadata_count, 4):
        raise ValueError("metadata_setcodes shape does not match metadata_card_idx")


class DeckProfileCatalog:
    """Trainer 侧目录：合并各 Worker 的画像并构造按 token 查找的静态表"""

    def __init__(self) -> None:
        self._profile_indices: Dict[bytes, int] = {}
        self._profiles = []
        self._metadata: Dict[int, tuple] = {}

    @property
    def profile_count(self) -> int:
        """返回当前轮次已登记的全局画像数量"""
        return len(self._profiles)

    def merge_bundle(self, bundle: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """合并一个 Worker 目录并返回局部索引到全局索引的映射"""
        validate_deck_profile_bundle(bundle)
        local_count = int(bundle["profile_hash"].shape[0])
        remap = torch.empty(local_count, dtype=torch.long)
        for local_index in range(local_count):
            profile_key = _profile_hash_key(bundle["profile_hash"][local_index])
            existing = self._profile_indices.get(profile_key)
            row = {
                key: bundle[key][local_index].detach().cpu().clone()
                for key in _PROFILE_TENSOR_KEYS[1:]
            }
            if existing is None:
                existing = len(self._profiles)
                self._profile_indices[profile_key] = existing
                self._profiles.append(row)
            else:
                for key, value in row.items():
                    if not torch.equal(self._profiles[existing][key], value):
                        raise ValueError("identical deck profile ids contain different data")
            remap[local_index] = existing

        tokens = bundle["metadata_card_idx"].detach().cpu().long().reshape(-1)
        races = bundle["metadata_race"].detach().cpu().long().reshape(-1)
        attrs = bundle["metadata_attr"].detach().cpu().long().reshape(-1)
        setcodes = bundle["metadata_setcodes"].detach().cpu().long().reshape(-1, 4)
        for index, raw_token in enumerate(tokens.tolist()):
            token = int(raw_token)
            if token <= 0:
                raise ValueError("deck metadata token must be positive")
            metadata = (
                int(races[index]),
                int(attrs[index]),
                tuple(int(value) for value in setcodes[index].tolist()),
            )
            previous = self._metadata.get(token)
            if previous is not None and previous != metadata:
                raise ValueError(
                    f"deck token {token} differs between Worker metadata tables"
                )
            self._metadata[token] = metadata
        return remap

    def validate_references(self, references: torch.Tensor) -> None:
        """确认每个样本都引用当前轮次实际登记的画像"""
        values = references.detach().cpu().long().reshape(-1)
        if values.numel() == 0:
            return
        if self.profile_count <= 0:
            raise ValueError("deck profile references exist without a catalog")
        if int(values.min()) < 0 or int(values.max()) >= self.profile_count:
            raise ValueError("deck_profile_index is outside the merged catalog")

    def build_metadata_lookup(self, vocab_size: int) -> dict:
        """构造一次性小型查找表，供每个 PPO mini-batch 在设备侧重建旧输入"""
        size = int(vocab_size)
        if size <= 0:
            raise ValueError("vocab_size must be positive")
        lookup = {
            "deck_race": torch.zeros(size, dtype=torch.long),
            "deck_attr": torch.zeros(size, dtype=torch.long),
            "deck_setcodes": torch.zeros((size, 4), dtype=torch.long),
            "known": torch.zeros(size, dtype=torch.bool),
        }
        for token, metadata in self._metadata.items():
            if token >= size:
                raise ValueError(
                    f"deck metadata token {token} exceeds vocabulary size {size}"
                )
            lookup["deck_race"][token] = metadata[0]
            lookup["deck_attr"][token] = metadata[1]
            lookup["deck_setcodes"][token] = torch.tensor(
                metadata[2],
                dtype=torch.long,
            )
            lookup["known"][token] = True
        return lookup


def reconstruct_deck_static_observations(
    deck_idx: torch.Tensor,
    metadata_lookup: Mapping[str, torch.Tensor],
) -> dict:
    """按原 token 顺序逐元素重建旧网络需要的卡组静态字段"""
    indices = deck_idx.long()
    return {
        key: metadata_lookup[key][indices]
        for key in DECK_STATIC_OBSERVATION_KEYS
    }
