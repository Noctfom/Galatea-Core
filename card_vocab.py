# 本文件维护真实卡片代码到模型索引的只追加精确词表

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from semantic_assets import invalidate_static_semantic_assets


CARD_VOCAB_FILENAME = "card_vocab.json"
CARD_VOCAB_FORMAT_VERSION = 1
DEFAULT_CARD_VOCAB_CAPACITY = 20000
FIRST_CARD_TOKEN_ID = 10
CARD_VOCAB_RESERVED_TOKENS = {
    "PAD": 0,
    "UNK": 1,
    "HIDDEN_HAND": 2,
    "HIDDEN_SZONE": 3,
}
MAX_CARD_VOCAB_FILE_BYTES = 16 * 1024 * 1024


def _sha256_file(path):
    """流式计算词表来源文件的 SHA-256"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_mapping_payload(capacity, cards):
    """生成不受 JSON 排版影响的词表指纹载荷"""
    return {
        "format_version": CARD_VOCAB_FORMAT_VERSION,
        "capacity": int(capacity),
        "first_card_token_id": FIRST_CARD_TOKEN_ID,
        "reserved_tokens": CARD_VOCAB_RESERVED_TOKENS,
        "cards": {str(code): int(token_id) for code, token_id in cards.items()},
    }


def calculate_card_vocab_hash(capacity, cards):
    """计算模型与资产共同校验的稳定词表哈希"""
    payload = _canonical_mapping_payload(capacity, cards)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_card_codes_from_cdb(cdb_path="cards.cdb"):
    """以只读方式取出卡库中全部有效真实卡密"""
    source = Path(cdb_path).resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"cards database does not exist: {source}")
    connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        try:
            codes = sorted(
                {int(row[0]) for row in connection.execute("SELECT id FROM datas")}
            )
        except sqlite3.Error as error:
            raise ValueError("cards.cdb is not a valid card database") from error
    finally:
        connection.close()
    if not codes or codes[0] <= 0:
        raise ValueError("cards.cdb did not provide a valid positive card-code set")
    return codes


def find_card_vocabulary_cdb_gaps(vocabulary, cdb_path="cards.cdb"):
    """列出 cards.cdb 已收录但权威词表尚未分配编号的卡片。"""
    return [
        code
        for code in read_card_codes_from_cdb(cdb_path)
        if not vocabulary.contains(code)
    ]


def validate_card_vocabulary_cdb_coverage(vocabulary, cdb_path="cards.cdb"):
    """确保当前 cards.cdb 中的卡不会在模型端退化为 UNK"""
    missing = find_card_vocabulary_cdb_gaps(vocabulary, cdb_path)
    if missing:
        preview = ", ".join(str(code) for code in missing[:8])
        raise ValueError(
            f"card vocabulary is missing {len(missing)} cards from cards.cdb: {preview}"
        )
    return vocabulary


class CardVocabulary:
    """保存经过严格校验的精确卡片索引"""

    def __init__(self, cards, *, capacity=DEFAULT_CARD_VOCAB_CAPACITY):
        normalized = {}
        for raw_code, raw_token_id in cards.items():
            if isinstance(raw_code, bool) or isinstance(raw_token_id, bool):
                raise ValueError("card vocabulary entries must be integer pairs")
            try:
                code = int(raw_code)
                token_id = int(raw_token_id)
            except (TypeError, ValueError) as error:
                raise ValueError("card vocabulary entries must be integer pairs") from error
            if code <= 0 or str(code) != str(raw_code):
                raise ValueError(f"invalid canonical card code: {raw_code!r}")
            if code in normalized:
                raise ValueError(f"duplicate card code in vocabulary: {code}")
            normalized[code] = token_id

        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise ValueError("card vocabulary capacity must be an integer")
        if capacity <= FIRST_CARD_TOKEN_ID:
            raise ValueError("card vocabulary capacity is too small")
        ordered = sorted(normalized.items(), key=lambda item: item[1])
        expected_ids = list(range(FIRST_CARD_TOKEN_ID, FIRST_CARD_TOKEN_ID + len(ordered)))
        actual_ids = [token_id for _, token_id in ordered]
        if actual_ids != expected_ids:
            raise ValueError("card token ids must be unique and contiguous from index 10")
        if actual_ids and actual_ids[-1] >= capacity:
            raise ValueError("card vocabulary exceeds the configured embedding capacity")

        self.capacity = capacity
        self.cards = dict(ordered)
        self.card_count = len(self.cards)
        self.vocabulary_hash = calculate_card_vocab_hash(capacity, self.cards)

    @property
    def padding_id(self):
        return CARD_VOCAB_RESERVED_TOKENS["PAD"]

    @property
    def unknown_id(self):
        return CARD_VOCAB_RESERVED_TOKENS["UNK"]

    @property
    def hidden_hand_id(self):
        return CARD_VOCAB_RESERVED_TOKENS["HIDDEN_HAND"]

    @property
    def hidden_szone_id(self):
        return CARD_VOCAB_RESERVED_TOKENS["HIDDEN_SZONE"]

    def encode(self, code):
        """把真实卡片代码转换为唯一索引，未收录代码统一返回 UNK"""
        try:
            normalized = int(code)
        except (TypeError, ValueError):
            return self.unknown_id
        return self.cards.get(normalized, self.unknown_id)

    def contains(self, code):
        """判断真实卡片代码是否已经被精确词表收录"""
        try:
            return int(code) in self.cards
        except (TypeError, ValueError):
            return False

    def prefix(self, card_count):
        """截取指定长度的只追加前缀，用于校验旧 V4 产物"""
        if isinstance(card_count, bool) or not isinstance(card_count, int):
            raise ValueError("card vocabulary prefix length must be an integer")
        if card_count < 0 or card_count > self.card_count:
            raise ValueError("card vocabulary prefix length is outside the mapping")
        return CardVocabulary(
            dict(list(self.cards.items())[:card_count]),
            capacity=self.capacity,
        )

    def to_payload(self, *, source_cdb_sha256=None):
        """生成可写入磁盘且自带完整性信息的词表对象"""
        payload = _canonical_mapping_payload(self.capacity, self.cards)
        payload.update(
            {
                "card_count": self.card_count,
                "vocabulary_sha256": self.vocabulary_hash,
                "source_cdb_sha256": source_cdb_sha256,
            }
        )
        return payload


def load_card_vocabulary(path=CARD_VOCAB_FILENAME):
    """从普通 JSON 文件加载并严格校验精确卡片词表"""
    vocabulary_path = Path(path)
    if vocabulary_path.is_symlink() or not vocabulary_path.is_file():
        raise FileNotFoundError(f"card vocabulary does not exist: {vocabulary_path}")
    if vocabulary_path.stat().st_size > MAX_CARD_VOCAB_FILE_BYTES:
        raise ValueError("card vocabulary exceeds the 16 MiB safety limit")
    with open(vocabulary_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("card vocabulary must be a JSON object")
    if payload.get("format_version") != CARD_VOCAB_FORMAT_VERSION:
        raise ValueError("card vocabulary format version is incompatible")
    if payload.get("first_card_token_id") != FIRST_CARD_TOKEN_ID:
        raise ValueError("card vocabulary first token id is incompatible")
    if payload.get("reserved_tokens") != CARD_VOCAB_RESERVED_TOKENS:
        raise ValueError("card vocabulary reserved tokens are incompatible")
    cards = payload.get("cards")
    if not isinstance(cards, dict):
        raise ValueError("card vocabulary cards must be a JSON object")
    vocabulary = CardVocabulary(cards, capacity=payload.get("capacity"))
    if payload.get("card_count") != vocabulary.card_count:
        raise ValueError("card vocabulary card count does not match its mapping")
    if payload.get("vocabulary_sha256") != vocabulary.vocabulary_hash:
        raise ValueError("card vocabulary hash does not match its mapping")
    source_hash = payload.get("source_cdb_sha256")
    if source_hash is not None and (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
    ):
        raise ValueError("card vocabulary source CDB hash is invalid")
    return vocabulary


_DEFAULT_VOCAB_CACHE = None
_DEFAULT_VOCAB_CACHE_PATH = None
_DEFAULT_VOCAB_CACHE_SIGNATURE = None


def _card_vocab_file_signature(path):
    """读取词表文件变化签名，使 WebUI 子进程更新后可自动失效缓存"""
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


def get_default_card_vocabulary(path=CARD_VOCAB_FILENAME):
    """复用默认精确词表，避免每局重复解析一万余条 JSON"""
    global _DEFAULT_VOCAB_CACHE, _DEFAULT_VOCAB_CACHE_PATH
    global _DEFAULT_VOCAB_CACHE_SIGNATURE
    resolved = Path(path).resolve()
    signature = _card_vocab_file_signature(resolved)
    if (
        _DEFAULT_VOCAB_CACHE is None
        or _DEFAULT_VOCAB_CACHE_PATH != resolved
        or _DEFAULT_VOCAB_CACHE_SIGNATURE != signature
    ):
        _DEFAULT_VOCAB_CACHE = load_card_vocabulary(resolved)
        _DEFAULT_VOCAB_CACHE_PATH = resolved
        _DEFAULT_VOCAB_CACHE_SIGNATURE = signature
    return _DEFAULT_VOCAB_CACHE


def synchronize_authoritative_card_vocabulary(
    source_path,
    target_path=CARD_VOCAB_FILENAME,
):
    """仅在远程词表是本地只追加扩展时安全安装"""
    global _DEFAULT_VOCAB_CACHE, _DEFAULT_VOCAB_CACHE_PATH
    global _DEFAULT_VOCAB_CACHE_SIGNATURE
    source = Path(source_path).resolve()
    target = Path(target_path).resolve()
    authoritative = load_card_vocabulary(source)
    if target.exists():
        local = load_card_vocabulary(target)
        if local.capacity != authoritative.capacity:
            raise ValueError("authoritative card vocabulary capacity does not match local")
        shared_count = min(local.card_count, authoritative.card_count)
        if local.prefix(shared_count).cards != authoritative.prefix(shared_count).cards:
            raise ValueError(
                "authoritative card vocabulary diverges from the local append-only prefix"
            )
        if authoritative.card_count <= local.card_count:
            status = (
                "unchanged"
                if authoritative.card_count == local.card_count
                else "local_newer"
            )
            _DEFAULT_VOCAB_CACHE = local
            _DEFAULT_VOCAB_CACHE_PATH = target
            _DEFAULT_VOCAB_CACHE_SIGNATURE = _card_vocab_file_signature(target)
            return {"status": status, "vocabulary": local}

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with open(source, "rb") as source_stream, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".sync.tmp",
            dir=target.parent,
            delete=False,
        ) as target_stream:
            temporary_path = Path(target_stream.name)
            while chunk := source_stream.read(1024 * 1024):
                target_stream.write(chunk)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    _DEFAULT_VOCAB_CACHE = authoritative
    _DEFAULT_VOCAB_CACHE_PATH = target
    _DEFAULT_VOCAB_CACHE_SIGNATURE = _card_vocab_file_signature(target)
    invalidate_static_semantic_assets(target.parent)
    return {"status": "installed", "vocabulary": authoritative}


def update_card_vocabulary(
    cdb_path="cards.cdb",
    target_path=CARD_VOCAB_FILENAME,
    *,
    capacity=DEFAULT_CARD_VOCAB_CAPACITY,
):
    """从 cards.cdb 首次生成或只追加更新卡片词表"""
    global _DEFAULT_VOCAB_CACHE, _DEFAULT_VOCAB_CACHE_PATH
    global _DEFAULT_VOCAB_CACHE_SIGNATURE
    source = Path(cdb_path).resolve()
    target = Path(target_path).resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"cards database does not exist: {source}")
    if target.exists():
        existing = load_card_vocabulary(target)
        if existing.capacity != capacity:
            raise ValueError("existing card vocabulary capacity cannot be changed")
        cards = dict(existing.cards)
    else:
        cards = {}

    codes = read_card_codes_from_cdb(source)

    next_token_id = FIRST_CARD_TOKEN_ID + len(cards)
    for code in codes:
        if code not in cards:
            if next_token_id >= capacity:
                raise ValueError(
                    "card vocabulary capacity is exhausted; a model protocol change is required"
                )
            cards[code] = next_token_id
            next_token_id += 1

    vocabulary = CardVocabulary(cards, capacity=capacity)
    payload = vocabulary.to_payload(source_cdb_sha256=_sha256_file(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    _DEFAULT_VOCAB_CACHE = vocabulary
    _DEFAULT_VOCAB_CACHE_PATH = target
    _DEFAULT_VOCAB_CACHE_SIGNATURE = _card_vocab_file_signature(target)
    invalidate_static_semantic_assets(target.parent)
    return vocabulary


def main():
    """提供开发者使用的只追加词表更新命令"""
    parser = argparse.ArgumentParser(description="Build or append Galatea card vocabulary")
    parser.add_argument("--cdb", default="cards.cdb")
    parser.add_argument("--output", default=CARD_VOCAB_FILENAME)
    args = parser.parse_args()
    vocabulary = update_card_vocabulary(args.cdb, args.output)
    print(
        f"精确卡片词表已就绪: {vocabulary.card_count} 张 | "
        f"容量 {vocabulary.capacity} | SHA-256 {vocabulary.vocabulary_hash}"
    )


if __name__ == "__main__":
    main()
