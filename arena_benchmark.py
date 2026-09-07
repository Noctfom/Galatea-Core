# 本文件负责生成、校验和保存可复现的竞技场基准计划与结果

import datetime
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import deck_utils


BENCHMARK_PLAN_SCHEMA_VERSION = 1
BENCHMARK_RESULT_SCHEMA_VERSION = 1
MAX_BENCHMARK_JSON_BYTES = 16 * 1024 * 1024
MAX_BENCHMARK_GAMES = 10000
MAX_RESULT_SCAN_BYTES = 64 * 1024 * 1024
DEFAULT_BENCHMARK_ROOT = Path("arena_benchmarks")


def _safe_benchmark_label(value):
    """把基准名称限制为适合文件名的短标签"""
    text = str(value or "baseline").strip()
    safe = "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in text
    ).strip("._")
    return safe[:64] or "baseline"


def hash_file(path):
    """流式计算文件 SHA-256，避免大型模型或卡组文件进入内存"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path, payload):
    """使用同目录临时文件原子写入基准 JSON"""
    target = Path(path).resolve()
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
        if temporary_path.stat().st_size > MAX_BENCHMARK_JSON_BYTES:
            raise ValueError("竞技场基准 JSON 超过 16 MiB 安全上限")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def _deck_record(pick, digest_cache):
    """把一次已解析卡组转换为不含绝对路径的基准记录"""
    deck_hash = digest_cache.get(pick.deck_path)
    if deck_hash is None:
        deck_hash = hash_file(pick.deck_path)
        digest_cache[pick.deck_path] = deck_hash
    return {
        "range_kind": pick.range_kind,
        "range_name": pick.range_name,
        "pool_name": pick.pool_name,
        "deck_name": pick.deck_name,
        "sha256": deck_hash,
    }


def build_benchmark_plan(
    deck_dir,
    p0_source,
    p1_source,
    n_games,
    seed,
    name="baseline",
    catalog=None,
):
    """按固定随机种子预生成卡组、决斗种子与交替先后手赛程"""
    n_games = int(n_games)
    seed = int(seed)
    if not 1 <= n_games <= MAX_BENCHMARK_GAMES:
        raise ValueError(
            f"竞技场基准局数必须位于 1～{MAX_BENCHMARK_GAMES}"
        )
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("竞技场基准种子必须位于 uint32 范围")

    normalized_p0 = deck_utils.parse_arena_deck_source(p0_source)
    normalized_p1 = deck_utils.parse_arena_deck_source(
        p1_source,
        allow_follow=True,
    )
    rng = random.Random(seed)
    digest_cache = {}
    catalog = catalog or deck_utils.discover_arena_deck_catalog(deck_dir)
    pairs = []
    games = []
    for game_index in range(1, n_games + 1):
        pair = deck_utils.select_arena_deck_pair(
            ydk_dir=deck_dir,
            p0_source=normalized_p0,
            p1_source=normalized_p1,
            rng=rng,
            catalog=catalog,
        )
        pairs.append(pair)
        games.append({
            "game_index": game_index,
            "duel_seed": rng.randrange(1, 0x100000000),
            "swap_model_seats": game_index % 2 == 0,
            "p0": _deck_record(pair.p0, digest_cache),
            "p1": _deck_record(pair.p1, digest_cache),
        })

    plan = {
        "benchmark_plan_schema_version": BENCHMARK_PLAN_SCHEMA_VERSION,
        "name": _safe_benchmark_label(name),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "selection_seed": seed,
        "p0_source": deck_utils.format_arena_deck_source(normalized_p0),
        "p1_source": deck_utils.format_arena_deck_source(normalized_p1),
        "n_games": n_games,
        "seat_policy": "alternate_logical_players",
        "games": games,
    }
    return plan, pairs


def save_benchmark_plan(plan, root=DEFAULT_BENCHMARK_ROOT):
    """把新生成的基准计划保存到 plans 目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{_safe_benchmark_label(plan.get('name'))}_{timestamp}.json"
    return _atomic_write_json(Path(root) / "plans" / filename, plan)


def _read_benchmark_json(path):
    """安全读取有体积边界的普通基准 JSON 文件"""
    raw_source = Path(path)
    if raw_source.is_symlink():
        raise ValueError("竞技场基准文件不能是符号链接")
    source = raw_source.resolve()
    if source.suffix.casefold() != ".json":
        raise ValueError("竞技场基准文件必须使用 .json 后缀")
    if source.is_symlink() or not source.is_file():
        raise ValueError("竞技场基准文件必须是普通文件")
    if source.stat().st_size > MAX_BENCHMARK_JSON_BYTES:
        raise ValueError("竞技场基准文件超过 16 MiB 安全上限")
    with open(source, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("竞技场基准文件顶层必须是对象")
    return payload


def load_benchmark_plan(path, deck_dir, catalog=None):
    """加载既有赛程并校验当前卡组文件仍与计划哈希一致"""
    plan = _read_benchmark_json(path)
    if (
        plan.get("benchmark_plan_schema_version")
        != BENCHMARK_PLAN_SCHEMA_VERSION
    ):
        raise ValueError("竞技场基准计划版本不受支持")
    games = plan.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("竞技场基准计划没有有效对局")
    if int(plan.get("n_games", -1)) != len(games):
        raise ValueError("竞技场基准计划局数与赛程长度不一致")
    if len(games) > MAX_BENCHMARK_GAMES:
        raise ValueError("竞技场基准计划局数超过安全上限")
    if plan.get("seat_policy") != "alternate_logical_players":
        raise ValueError("竞技场基准计划座位策略不受支持")
    selection_seed = int(plan.get("selection_seed", -1))
    if not 0 <= selection_seed <= 0xFFFFFFFF:
        raise ValueError("竞技场基准计划选择种子越界")

    catalog = catalog or deck_utils.discover_arena_deck_catalog(deck_dir)
    digest_cache = {}
    pairs = []
    expected_game_index = 1
    for game in games:
        if not isinstance(game, dict):
            raise ValueError("竞技场基准赛程项必须是对象")
        if int(game.get("game_index", -1)) != expected_game_index:
            raise ValueError("竞技场基准赛程序号不连续")
        duel_seed = int(game.get("duel_seed", -1))
        if not 1 <= duel_seed <= 0xFFFFFFFF:
            raise ValueError("竞技场基准决斗种子越界")
        if (
            not isinstance(game.get("swap_model_seats"), bool)
            or game["swap_model_seats"] != (expected_game_index % 2 == 0)
        ):
            raise ValueError("竞技场基准先后手标记必须按奇偶局交替")

        picks = []
        for player_key in ("p0", "p1"):
            record = game.get(player_key)
            if not isinstance(record, dict):
                raise ValueError(f"竞技场基准缺少 {player_key} 卡组记录")
            pool_name = str(record.get("pool_name", ""))
            deck_name = str(record.get("deck_name", ""))
            exact_source = deck_utils.ArenaDeckSource(
                deck_utils.ARENA_SOURCE_DECK,
                f"{pool_name}/{deck_name}",
            )
            exact_pick = deck_utils.select_arena_deck(
                catalog,
                exact_source,
                rng=random.Random(0),
            )
            current_hash = digest_cache.get(exact_pick.deck_path)
            if current_hash is None:
                current_hash = hash_file(exact_pick.deck_path)
                digest_cache[exact_pick.deck_path] = current_hash
            if current_hash != record.get("sha256"):
                raise ValueError(
                    f"基准卡组已变化: {pool_name}/{deck_name}"
                )
            picks.append(deck_utils.ArenaDeckPick(
                range_kind=str(record.get("range_kind", exact_pick.range_kind)),
                range_name=str(record.get("range_name", exact_pick.range_name)),
                pool_name=exact_pick.pool_name,
                deck_name=exact_pick.deck_name,
                deck=exact_pick.deck,
                deck_path=exact_pick.deck_path,
            ))
        pairs.append(deck_utils.ArenaDeckPair(p0=picks[0], p1=picks[1]))
        expected_game_index += 1
    return plan, pairs


def describe_benchmark_model(path, fallback_name):
    """记录参与基准的模型文件名和内容哈希；RuleBot 使用固定标识"""
    if not path:
        return {"name": fallback_name, "kind": "rule"}
    resolved = Path(path).resolve()
    return {
        "name": resolved.name,
        "kind": "checkpoint",
        "sha256": hash_file(resolved),
    }


def _wilson_interval(wins, total):
    """计算胜率的 95% Wilson 区间"""
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize_benchmark_games(games):
    """汇总基准胜率、异常率、先后手表现和平均决策步数"""
    p0_wins = sum(game.get("winner") == 0 for game in games)
    p1_wins = sum(game.get("winner") == 1 for game in games)
    decisive = p0_wins + p1_wins
    abnormal = sum(bool(game.get("abnormal")) for game in games)
    p0_first = [
        game for game in games
        if not game.get("swap_model_seats") and game.get("winner") in (0, 1)
    ]
    p0_second = [
        game for game in games
        if game.get("swap_model_seats") and game.get("winner") in (0, 1)
    ]
    reason_counts = {}
    for game in games:
        reason = str(game.get("reason_label", "Unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    step_values = [
        int(game.get("steps", 0))
        for game in games
        if int(game.get("steps", 0)) >= 0
    ]
    win_rate = p0_wins / decisive if decisive else 0.0
    return {
        "games": len(games),
        "p0_wins": p0_wins,
        "p1_wins": p1_wins,
        "draws_or_aborts": len(games) - decisive,
        "p0_win_rate": win_rate,
        "p0_win_rate_95ci": _wilson_interval(p0_wins, decisive),
        "abnormal_games": abnormal,
        "abnormal_rate": abnormal / len(games) if games else 0.0,
        "model_fallbacks": sum(
            int(game.get("fallback_count", 0)) for game in games
        ),
        "average_decision_steps": (
            sum(step_values) / len(step_values) if step_values else 0.0
        ),
        "p0_as_first": {
            "games": len(p0_first),
            "wins": sum(game.get("winner") == 0 for game in p0_first),
        },
        "p0_as_second": {
            "games": len(p0_second),
            "wins": sum(game.get("winner") == 0 for game in p0_second),
        },
        "reason_counts": reason_counts,
    }


def save_benchmark_result(
    plan,
    plan_path,
    p0_model_path,
    p1_model_path,
    games,
    root=DEFAULT_BENCHMARK_ROOT,
):
    """保存带模型哈希、计划引用、逐局数据和统计摘要的结果"""
    p0_model = describe_benchmark_model(p0_model_path, "P0_AI")
    if (
        p0_model_path
        and p1_model_path
        and Path(p0_model_path).resolve() == Path(p1_model_path).resolve()
    ):
        p1_model = dict(p0_model)
    else:
        p1_model = describe_benchmark_model(p1_model_path, "RuleBot")
    payload = {
        "benchmark_result_schema_version": BENCHMARK_RESULT_SCHEMA_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "benchmark_name": plan.get("name", "baseline"),
        "plan_file": Path(plan_path).name,
        "selection_seed": plan.get("selection_seed"),
        "models": {
            "p0": p0_model,
            "p1": p1_model,
        },
        "summary": summarize_benchmark_games(games),
        "games": games,
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    p0_label = _safe_benchmark_label(payload["models"]["p0"]["name"])
    filename = (
        f"{_safe_benchmark_label(plan.get('name'))}_{p0_label}_{timestamp}.json"
    )
    return _atomic_write_json(Path(root) / "results" / filename, payload)


def load_benchmark_results(root=DEFAULT_BENCHMARK_ROOT, limit=100):
    """读取最近的安全基准结果，供 WebUI 横向比较"""
    result_dir = Path(root).resolve() / "results"
    if not result_dir.is_dir():
        return []
    candidates = []
    for path in result_dir.glob("*.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    paths = [
        path
        for _, path in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        )
    ][:max(1, min(int(limit), 1000))]
    results = []
    scanned_bytes = 0
    for path in paths:
        try:
            file_size = path.stat().st_size
            if scanned_bytes + file_size > MAX_RESULT_SCAN_BYTES:
                continue
            scanned_bytes += file_size
            payload = _read_benchmark_json(path)
            if (
                payload.get("benchmark_result_schema_version")
                != BENCHMARK_RESULT_SCHEMA_VERSION
            ):
                continue
            summary = payload.get("summary")
            models = payload.get("models")
            interval = (
                summary.get("p0_win_rate_95ci")
                if isinstance(summary, dict)
                else None
            )
            numeric_summary_fields = (
                "p0_wins",
                "p1_wins",
                "p0_win_rate",
                "abnormal_games",
                "average_decision_steps",
            )
            if (
                not isinstance(summary, dict)
                or not isinstance(models, dict)
                or not isinstance(interval, list)
                or len(interval) != 2
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in interval
                )
                or not all(
                    isinstance(summary.get(field), (int, float))
                    and math.isfinite(summary[field])
                    for field in numeric_summary_fields
                )
                or not all(
                    isinstance(models.get(player), dict)
                    and isinstance(models[player].get("name"), str)
                    for player in ("p0", "p1")
                )
            ):
                continue
            payload["_path"] = str(path)
            results.append(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return results
