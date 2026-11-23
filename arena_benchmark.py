# 本文件负责生成、校验和保存可复现的竞技场基准计划与结果

import datetime
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import deck_utils


BENCHMARK_PLAN_SCHEMA_VERSION = 2
BENCHMARK_RESULT_SCHEMA_VERSION = 1
MAX_BENCHMARK_JSON_BYTES = 16 * 1024 * 1024
MAX_BENCHMARK_GAMES = 10000
MAX_RESULT_SCAN_BYTES = 64 * 1024 * 1024
DEFAULT_BENCHMARK_ROOT = Path("arena_benchmarks")


def make_paired_deal(deck0, deck1, seed):
    """固定两名逻辑参赛者的实际注入顺序，各用独立随机流"""
    deal = {}
    for side, deck in enumerate((deck0, deck1)):
        rng = random.Random(f'galatea-deal-v1/{seed}/{side}')
        order = {part: list(deck[part]) for part in ('main', 'extra')}
        for cards in order.values():
            rng.shuffle(cards)
        deal[str(side)] = order
    return deal


def ordered_core_decks(deal, decks):
    """校验牌序与实际构筑相符，仅供内核开局使用"""
    if not isinstance(deal, dict) or set(deal) != {'0', '1'}:
        raise ValueError('发牌表需要双方完整牌序')
    ordered = []
    for seat in ('0', '1'):
        order = deal[seat]
        if not isinstance(order, dict) or set(order) != {'main', 'extra'}:
            raise ValueError('发牌表缺少主卡组或额外卡组')
        for part, cards in order.items():
            if (not isinstance(cards, list) or len(cards) != len(decks[seat][part])
                    or any(type(code) is not int or code <= 0 for code in cards)
                    or sorted(cards) != sorted(decks[seat][part])):
                raise ValueError('实际注入牌序与构筑不符')
        ordered.append(SimpleNamespace(**order))
    return ordered


def deal_digest(deal):
    return hashlib.sha256(json.dumps(deal, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def normalize_benchmark_inference(inference=None):
    """校验并规范化写入基准结果的竞技场推理策略"""
    if inference is None:
        return {"policy_mode": "greedy", "temperature": None}
    if not isinstance(inference, dict):
        raise ValueError("竞技场基准推理策略必须是对象")
    policy_mode = inference.get("policy_mode")
    temperature = inference.get("temperature")
    policy_metadata = {}
    if 'loop_policy' in inference or 'loop_repeat_threshold' in inference:
        if (inference.get('loop_policy') not in ('leaf_first_v1', 'leaf_first_v2')
                or type(inference.get('loop_repeat_threshold')) is not int
                or inference['loop_repeat_threshold'] != 5):
            raise ValueError('竞技场防循环策略或阈值无效')
        policy_metadata.update(loop_policy=inference['loop_policy'], loop_repeat_threshold=5)
    proposal = inference.get('candidate_proposal')
    if proposal is not None:
        if not isinstance(proposal, str) or not proposal.strip():
            raise ValueError('竞技场候选生成版本无效')
        policy_metadata['candidate_proposal'] = proposal
    if policy_mode == "greedy" and temperature is None:
        return {"policy_mode": policy_mode, "temperature": None, **policy_metadata}
    if (
        policy_mode == "training"
        and isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and math.isfinite(temperature)
        and float(temperature) == 1.0
    ):
        return {"policy_mode": policy_mode, "temperature": 1.0, **policy_metadata}
    if (
        policy_mode == "deployment"
        and isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and math.isfinite(temperature)
        and 0.05 <= float(temperature) <= 5.0
    ):
        return {"policy_mode": policy_mode, "temperature": float(temperature), **policy_metadata}
    raise ValueError("竞技场基准推理策略或温度无效")


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
            if stream.tell() > MAX_BENCHMARK_JSON_BYTES:
                # 长赛程的逐张牌序占用较多空间，必要时去掉排版空白
                stream.seek(0)
                stream.truncate()
                json.dump(payload, stream, ensure_ascii=False, separators=(',', ':'))
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
    deals = []
    for game_index in range(1, n_games + 1):
        if game_index % 2:
            pair = deck_utils.select_arena_deck_pair(
                ydk_dir=deck_dir, p0_source=normalized_p0, p1_source=normalized_p1,
                rng=rng, catalog=catalog)
            duel_seed = rng.randrange(1, 0x100000000)
            deals.append(make_paired_deal(vars(pair.p0.deck), vars(pair.p1.deck), duel_seed))
        pairs.append(pair)
        games.append({
            "game_index": game_index,
            "duel_seed": duel_seed,
            "deal_index": len(deals) - 1,
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
        "seat_policy": "paired_logical_players",
        "deals": deals,
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
    version = plan.get("benchmark_plan_schema_version")
    if type(version) is not int or version not in (1, BENCHMARK_PLAN_SCHEMA_VERSION):
        raise ValueError("竞技场基准计划版本不受支持")
    games = plan.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("竞技场基准计划没有有效对局")
    if int(plan.get("n_games", -1)) != len(games):
        raise ValueError("竞技场基准计划局数与赛程长度不一致")
    if len(games) > MAX_BENCHMARK_GAMES:
        raise ValueError("竞技场基准计划局数超过安全上限")
    if plan.get("seat_policy") != ("alternate_logical_players" if version == 1 else "paired_logical_players"):
        raise ValueError("竞技场基准计划座位策略不受支持")
    deals = plan.get('deals')
    if version == 2 and (not isinstance(deals, list) or len(deals) != (len(games) + 1) // 2):
        raise ValueError('基准发牌组数与赛程不符')
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
        if version == 2:
            index = (expected_game_index - 1) // 2
            if type(game.get('deal_index')) is not int or game['deal_index'] != index:
                raise ValueError('基准发牌组序号不匹配')
            ordered_core_decks(deals[index], {str(i): vars(pick.deck) for i, pick in enumerate(picks)})
            if expected_game_index % 2 == 0:
                first = games[expected_game_index - 2]
                if any(first[key] != game[key] for key in ('p0', 'p1', 'duel_seed')):
                    raise ValueError('同一发牌组的构筑或内核种子不一致')
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


def build_windbot_plan(executors, games_per_seat, seed=None):
    """每个执行器的每组开局固定两份独立牌序，模型带着自己的牌换座"""
    from integrations.duel_session import resolve_core_seed
    from integrations.windbot.runtime import executor_catalog, load_deck
    names = sorted(set(executors))
    if not names or type(games_per_seat) is not int or not 1 <= games_per_seat <= 1000:
        raise ValueError('请选择执行器，每个执行器可设置 1～1000 组发牌')
    count = len(names) * games_per_seat
    if 2 * count > MAX_BENCHMARK_GAMES:
        raise ValueError('每个模型的总局数超过基准上限')
    seed = resolve_core_seed(seed, count)
    catalog = executor_catalog()
    builds, deals = {}, []
    for name in names:
        info = catalog.get(name)
        if not info or not info['deck_available']:
            raise ValueError(f'缺少执行器或固定构筑 {name}')
        builds[name] = {'ydk_sha256': hash_file(info['deck_file']), 'level': info['level']}
        deck = load_deck(info['deck_file'])
        for _ in range(games_per_seat):
            core_seed = seed + len(deals)
            deals.append({'executor': name, 'seed': core_seed, 'order': make_paired_deal(deck, deck, core_seed)})
    return {'windbot_plan_version': 1, 'seed': seed, 'games_per_seat': games_per_seat,
            'bot_randomness_controlled': False, 'builds': builds, 'deals': deals}


def validate_windbot_plan(plan):
    """复用已保存的牌序并核对构筑和组号"""
    from collections import Counter
    from integrations.windbot.runtime import executor_catalog, load_deck
    builds, deals = plan.get('builds'), plan.get('deals')
    groups = plan.get('games_per_seat')
    if (plan.get('windbot_plan_version') != 1 or not isinstance(builds, dict) or not builds
            or type(groups) is not int or not 1 <= groups <= 1000 or not isinstance(deals, list)
            or len(deals) != len(builds) * groups or 2 * len(deals) > MAX_BENCHMARK_GAMES
            or type(plan.get('seed')) is not int or not 0 <= plan['seed'] <= 2**32 - len(deals)):
        raise ValueError('WindBot 配对赛程版本或组数无效')
    catalog, decks, counts = executor_catalog(), {}, Counter()
    for name, build in builds.items():
        info = catalog.get(name)
        if (not info or not info['deck_available'] or not isinstance(build, dict)
                or hash_file(info['deck_file']) != build.get('ydk_sha256')):
            raise ValueError(f'WindBot 固定构筑已变化或缺失 {name}')
        decks[name] = load_deck(info['deck_file'])
    for i, row in enumerate(deals):
        if (not isinstance(row, dict) or type(row.get('seed')) is not int or row['seed'] != plan['seed'] + i
                or not isinstance(row.get('executor'), str) or row['executor'] not in decks):
            raise ValueError('WindBot 发牌组的执行器或种子无效')
        deck = decks[row['executor']]
        ordered_core_decks(row.get('order'), {'0': deck, '1': deck})
        counts[row['executor']] += 1
    if any(counts[name] != groups for name in builds):
        raise ValueError('各执行器的发牌组数不符')


def play_windbot_benchmark_model(plan, checkpoint, output, *, runtime=None, mono=None, timeout=600,
                                 max_decisions=2000):
    """用同一检查点完成固定赛程，独立保存每个座位的真实终局与干预次数"""
    from integrations.windbot.runtime import executable_sha256
    from integrations.windbot.session import run_controlled_duel
    validate_windbot_plan(plan)
    checkpoint = Path(checkpoint)
    original_stat = checkpoint.stat()
    model = describe_benchmark_model(checkpoint, '')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    games = []
    for group, deal in enumerate(plan['deals']):
        for seat in (0, 1):
            current = checkpoint.stat()
            if (current.st_size, current.st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns):
                raise ValueError('评测期间检查点文件发生变化')
            result = run_controlled_duel(executor=deal['executor'], windbot_seat=1-seat,
                model=str(checkpoint), seed=deal['seed'], deal=deal['order'], runtime=runtime, mono=mono,
                max_decisions=max_decisions, timeout=timeout, policy_mode='greedy', session_purpose='arena_evaluation',
                capture_root=directory / 'captures', log_root=directory / 'logs', game_index=len(games)+1)
            winner = result['winner']
            if result['core_terminal'] and not result.get('native_query_errors') and winner in (0, 1, 2):
                outcome = 'draw' if winner == 2 else ('win' if winner == seat else 'loss')
            else:
                error = (result.get('error') or '').lower()
                outcome = 'truncated' if 'time limit' in error or 'decision limit' in error else 'aborted'
            game = {'group': group, 'executor': deal['executor'], 'seed': deal['seed'], 'model_seat': seat,
                    'outcome': outcome, 'audit': result['retry_audit'][str(seat)], 'seconds': result['seconds'],
                    'deal_sha256': result['deal_sha256'], 'inference': result['native_policy'],
                    'capture': Path(result['capture_path']).relative_to(directory.resolve()).as_posix() if result['capture_path'] else None,
                    'winner': winner, 'reason': result['reason'], 'error': result['error'],
                    'core_terminal': result['core_terminal'], 'query_errors': result.get('native_query_errors')}
            if game['deal_sha256'] != deal_digest(deal['order']):
                raise ValueError('实际发牌与赛程不符')
            _atomic_write_json(directory / f'game_{len(games)+1:05d}.json', game)
            games.append(game)
            print(f"[{model['name']}] {len(games)}/{2*len(plan['deals'])} {deal['executor']} P{seat} {outcome}", flush=True)
    if hash_file(checkpoint) != model['sha256']:
        raise ValueError('评测期间检查点内容发生变化')
    report = {'model': model, 'plan_sha256': deal_digest(plan), 'bot_sha256': executable_sha256(runtime, mono),
              'max_decisions': max_decisions, 'timeout': timeout, 'games': games}
    _atomic_write_json(directory / 'model_result.json', report)
    return report


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


def _windbot_counts(games):
    from collections import Counter
    outcomes = Counter(game['outcome'] for game in games)
    audit = Counter()
    for game in games:
        audit.update(game['audit'])
    completed = sum(outcomes[k] for k in ('win', 'draw', 'loss'))
    return {**{k: outcomes[k] for k in ('win', 'draw', 'loss', 'truncated', 'aborted')},
            'attempted': len(games), 'completed': completed,
            'win_rate': outcomes['win'] / completed if completed else None,
            'win_rate_95ci_descriptive': _wilson_interval(outcomes['win'], completed) if completed else None,
            'audit': dict(audit), 'interventions_per_1000': (1000 * audit['interventions'] / audit['model_decisions']
                if audit['model_decisions'] else None),
            'greedy_changes_per_1000': (1000 * audit['greedy_choices_changed'] / audit['greedy_decisions']
                if audit['greedy_decisions'] else None)}


def _paired_win_difference(deals, baseline, candidate):
    """将同一发牌的两个座位捆绑，在各执行器内重抽发牌组估计差值区间"""
    import numpy as np
    strata = {}
    for group, deal in enumerate(deals):
        pairs = [(baseline[group, seat], candidate[group, seat]) for seat in (0, 1)]
        if all(outcome in ('win', 'loss', 'draw') for pair in pairs for outcome in pair):
            difference = sum(int(b == 'win') - int(a == 'win') for a, b in pairs) / 2
            strata.setdefault(deal['executor'], []).append(difference)
    count = sum(map(len, strata.values()))
    result = {'matched_deals': count, 'excluded_deals': len(deals) - count, 'difference': None,
              'paired_bootstrap95': None, 'bounded95': None,
              'by_executor': {name: {'deals': len(values), 'difference': sum(values) / len(values)}
                              for name, values in strata.items()}}
    if count:
        mean = sum(map(sum, strata.values())) / count
        radius = math.sqrt(2 * math.log(40) / count)
        result.update(difference=mean, bounded95=[max(-1., mean-radius), min(1., mean+radius)])
        if min(map(len, strata.values())) >= 2 and any(len(set(v)) > 1 for v in strata.values()):
            rng = np.random.default_rng(3179)
            samples = np.zeros(4000)
            for values in strata.values():
                for start in range(0, len(samples), 100):
                    samples[start:start+100] += rng.choice(values, size=(100, len(values))).sum(axis=1)
            result['paired_bootstrap95'] = np.quantile(samples / count, [.025, .975]).tolist()
    return result


def summarize_windbot_benchmark(plan, outputs):
    """核对实际捕获的开局和运行身份，再按配对座位与模型决策数汇总"""
    from itertools import combinations
    from core_trajectory import _iter_records
    summaries, outcomes, identities = [], [], set()
    versions = set()
    seen_models = set()
    fingerprint = deal_digest(plan)
    limits, policies = set(), set()
    for directory in map(Path, outputs):
        report = _read_benchmark_json(directory / 'model_result.json')
        games, model = report['games'], report['model']
        if report['plan_sha256'] != fingerprint or len(games) != 2*len(plan['deals']):
            raise ValueError('模型结果与配对赛程不匹配')
        if model['sha256'] in seen_models:
            raise ValueError('配对比较要求不同的模型权重文件')
        seen_models.add(model['sha256'])
        limits.add((report['bot_sha256'], report['timeout'], report['max_decisions']))
        scores = {}
        for i, game in enumerate(games):
            group, seat = divmod(i, 2)
            deal = plan['deals'][group]
            if ((game['group'], game['model_seat'], game['executor'], game['seed']) !=
                    (group, seat, deal['executor'], deal['seed']) or game['deal_sha256'] != deal_digest(deal['order'])):
                raise ValueError('逐局的开局、座位或对手与赛程不匹配')
            if game['capture']:
                path = (directory / game['capture'].replace('\\', '/')).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError('评测捕获路径越界')
                stream = _iter_records(path)
                try:
                    header = next(stream)[0]
                finally:
                    stream.close()
                reset = header['reset']
                expected = {'0': deal['order'][str(seat)], '1': deal['order'][str(1-seat)]}
                if reset['seed'] != deal['seed'] or reset['players'] != expected:
                    raise ValueError('捕获记录的实际注入牌序与赛程不匹配')
                participants = header['external_session']['participants']
                if (participants[str(seat)]['kind'] != 'model'
                        or participants[str(1-seat)].get('executor') != deal['executor']
                        or participants[str(1-seat)].get('executable_sha256') != report['bot_sha256']):
                    raise ValueError('捕获记录的参赛者与赛程不符')
                identities.add(deal_digest(header['assets']))
                versions.add(header['framework_version'])
            elif game['outcome'] in ('win', 'draw', 'loss'):
                raise ValueError('有效终局缺少原始捕获')
            policies.add(deal_digest(game['inference']))
            scores[group, seat] = game['outcome']
        outcomes.append(scores)
        summaries.append({'model': model, **_windbot_counts(games),
            'by_seat': {str(seat): _windbot_counts([g for g in games if g['model_seat'] == seat]) for seat in (0, 1)},
            'by_executor': {name: _windbot_counts([g for g in games if g['executor'] == name]) for name in plan['builds']}})
    if len(limits) != 1 or len(policies) != 1 or len(identities) > 1:
        raise ValueError('参与比较的推理策略、资源、对手程序或对局上限不一致')
    comparisons = []
    for a, b in combinations(range(len(summaries)), 2):
        comparisons.append({'baseline': summaries[a]['model'], 'candidate': summaries[b]['model'],
            **_paired_win_difference(plan['deals'], outcomes[a], outcomes[b])})
    bot_sha256, timeout, max_decisions = next(iter(limits))
    return {'windbot_result_version': 1, 'plan_sha256': fingerprint, 'framework_versions': sorted(versions),
            'assets_sha256': next(iter(identities), None), 'bot_sha256': bot_sha256,
            'timeout': timeout, 'max_decisions': max_decisions,
            'bot_randomness_controlled': False, 'models': summaries, 'comparisons': comparisons}


def load_windbot_summary(path):
    summary = _read_benchmark_json(path)
    if (summary.get('windbot_result_version') != 1 or not isinstance(summary.get('models'), list)
            or not isinstance(summary.get('comparisons'), list)):
        raise ValueError('WindBot 配对评测结果格式无效')
    return summary


def run_windbot_benchmark(checkpoints, *, executors, games_per_seat, seed=None, plan_path=None,
                          root=DEFAULT_BENCHMARK_ROOT / 'windbot', runtime=None, mono=None, timeout=600,
                          max_decisions=2000):
    """在独立目录保存一个可复用的发牌表和整批模型的配对结果"""
    checkpoints = list(map(Path, checkpoints))
    if not checkpoints or any(not path.is_file() for path in checkpoints):
        raise ValueError('请选择存在的模型检查点')
    if len(set(map(Path.resolve, checkpoints))) != len(checkpoints):
        raise ValueError('请移除重复的评测模型')
    plan = (_read_benchmark_json(plan_path) if plan_path else
            build_windbot_plan(executors, games_per_seat, seed))
    validate_windbot_plan(plan)
    run = Path(root) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(run / 'plan.json', plan)
    outputs = []
    for i, checkpoint in enumerate(checkpoints):
        directory = run / f'model_{i+1}'
        play_windbot_benchmark_model(plan, checkpoint, directory, runtime=runtime, mono=mono,
                                    timeout=timeout, max_decisions=max_decisions)
        outputs.append(directory)
    return _atomic_write_json(run / 'summary.json', summarize_windbot_benchmark(plan, outputs))


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
    inference=None,
):
    """保存带模型、推理策略、赛程引用和统计摘要的基准结果"""
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
        "inference": normalize_benchmark_inference(inference),
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
            if "inference" not in payload:
                # 3.6.4 结果没有策略字段，按当时唯一存在的贪心模式解释
                payload["inference"] = normalize_benchmark_inference()
            else:
                payload["inference"] = normalize_benchmark_inference(
                    payload["inference"]
                )
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
