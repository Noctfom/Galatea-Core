# 本文件审查规范轨迹并按整局、精确构筑和稳定玩家的连通分组生成独立数据清单
import hashlib
import json
import math
from pathlib import Path

from core_trajectory import (
    _iter_records, _json_bytes, _replay_core_trajectory_source, canonical_response,
    decode_response, inspect_core_trajectory,
)
from trajectory_ingest import _freeze_source, validate_player_ids

DATASET_MANIFEST_VERSION = 1
MAX_DATASET_FILES = 10_000


def deck_identity(deck):
    """按主/额外/备牌分类及投入数量生成精确构筑身份，忽略文件名和卡片排列"""
    parts = {}
    for part in ('main', 'extra', 'side'):
        codes = deck.get(part, [])
        if (not isinstance(codes, list) or len(codes) > 256
                or any(type(code) is not int or not 0 < code < 2**32 for code in codes)):
            raise ValueError('invalid dataset deck section')
        parts[part] = sorted(codes)
    return hashlib.sha256(_json_bytes(parts)).hexdigest()


def _stable_players(header):
    """优先使用采集端稳定匿名 ID，模型对局可使用 UUID；昵称和座位不作玩家身份"""
    origin = header.get('ingress', {})
    if not isinstance(origin, dict):
        raise ValueError('invalid dataset ingress metadata')
    explicit = origin.get('player_ids')
    if explicit is None:
        explicit = header.get('player_ids')
    if explicit is not None:
        return validate_player_ids(explicit)
    models = header.get('models', {})
    if not isinstance(models, dict) or any(not isinstance(value, dict) for value in models.values()):
        raise ValueError('invalid dataset model identities')
    result = {'0': None, '1': None}
    for seat in result:
        model_id = models.get(seat, {}).get('model_id')
        if isinstance(model_id, str) and model_id:
            result[seat] = f'model:{model_id}'
        elif models.get(seat, {}).get('kind') == 'rule':
            result[seat] = 'rule:galatea_builtin'
    return validate_player_ids(result)


def inspect_dataset_entry(path):
    """固定输入后严格重放与检查来源，只有满足全部门禁的整局可进入候选数据集"""
    with _freeze_source(path) as frozen:
        report = inspect_core_trajectory(frozen)
        frozen.seek(0)
        file_digest = hashlib.sha256()
        for chunk in iter(lambda: frozen.read(1024 * 1024), b''):
            file_digest.update(chunk)
        header, footer = report['header'], report['footer']
        players = _stable_players(header)
        deck_ids = sorted({deck_identity(deck) for deck in header['decks'].values()})
        duel_digest = hashlib.sha256(_json_bytes({'reset': header['reset'], 'assets': header.get('assets')}))
        frozen.seek(0)
        for record, _ in _iter_records(frozen):
            if record['kind'] in ('chunk', 'response', 'automatic_response'):
                # 排除录制 UUID、展示文字和专家/模型来源标签，重复封装同一局仍归同组
                value = ({'kind': 'chunk', 'hex': record['hex']} if record['kind'] == 'chunk'
                         else {'kind': record['kind'], 'actor': record['actor'],
                               'response_buffer': canonical_response(decode_response(record['response'])).hex()})
                duel_digest.update(_json_bytes(value))
        reasons = []
        if header.get('visibility') != 'omniscient_local_core':
            reasons.append('incomplete_visibility')
        if header.get('source') not in ('arena', 'link', 'yrp'):
            reasons.append('unsupported_origin')
        origin = header.get('ingress', {})
        if header.get('source') in ('link', 'yrp') and (
                not isinstance(origin, dict) or origin.get('format_version') != 1
                or origin.get('kind') != header['source']
                or origin.get('original_core_output_available') is not True
                or origin.get('historical_assets_known') is not True):
            reasons.append('original_output_or_assets_unverified')
        if any(value is None for value in players.values()):
            reasons.append('stable_player_ids_missing')
        if footer['truncated']:
            reasons.append('truncated_recording')
        elif not footer['core_terminal']:
            reasons.append('no_real_terminal')
        elif footer['errors']:
            reasons.append('trajectory_quality_flags')
        if not reasons or all(reason == 'stable_player_ids_missing' for reason in reasons):
            frozen.seek(0)
            try:
                report = _replay_core_trajectory_source(frozen)
                if not report['behavior_clone_eligible']:
                    reasons.append('strict_replay_or_mapping_gate_failed')
            except (ValueError, RuntimeError, OSError, KeyError, EOFError, RecursionError) as error:
                reasons.append(f'replay_failed: {str(error)[:512]}')
        return {'trace_id': header['trace_id'], 'file_sha256': file_digest.hexdigest(),
                'duel_id': duel_digest.hexdigest(), 'deck_ids': deck_ids,
                'player_ids': sorted(set(value for value in players.values() if value is not None)),
                'source': header.get('source'),
                'assets_sha256': hashlib.sha256(_json_bytes(header.get('assets'))).hexdigest(),
                'winner': footer['winner'], 'eligible': not reasons,
                'rejection_reasons': reasons, 'replay_verified': report.get('replay_verified', False),
                'raw_output_exact': report.get('raw_output_exact', False),
                'response_count': report.get('response_count', 0),
                'observation_check_count': report.get('observation_check_count', 0)}


def assign_dataset_splits(entries, *, seed=20261009, validation_fraction=0.1, test_fraction=0.1):
    """把共享整局/构筑/玩家的传递连通分量放入同一集合，不按逐状态随机切分"""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('dataset seed must be uint32')
    for value in (validation_fraction, test_fraction):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError('invalid dataset split fraction')
    if validation_fraction + test_fraction >= 1:
        raise ValueError('dataset split must leave a positive training fraction')
    parents = list(range(len(entries)))

    def find(index):
        """压缩连通分组路径，避免共享玩家/卡组产生链式泄漏"""
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, entry in enumerate(entries):
        if not entry['eligible']:
            entry['split'] = 'quarantine'
            continue
        keys = [('duel', entry['duel_id'])]
        keys += [('deck', value) for value in entry['deck_ids']]
        keys += [('player', value) for value in entry['player_ids']]
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = {}
    for index, entry in enumerate(entries):
        if entry['eligible']:
            groups.setdefault(find(index), []).append(entry)
    train_boundary = 1 - validation_fraction - test_fraction
    for group in groups.values():
        group_id = hashlib.sha256(_json_bytes(sorted({entry['duel_id'] for entry in group}))).hexdigest()
        draw = int(hashlib.sha256(f'{seed}:{group_id}'.encode()).hexdigest()[:16], 16) / 2**64
        split = ('train' if draw < train_boundary else
                 'validation' if draw < train_boundary + validation_fraction else 'test')
        for entry in group:
            entry.update(split=split, group_id=group_id)
    return len(groups)


def build_trajectory_dataset(directory, output_path, *, seed=20261009,
                             validation_fraction=0.1, test_fraction=0.1):
    """扫描只读轨迹并新建质量/划分清单，不复制张量、不启动任何学习优化器"""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('dataset input must be a directory')
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError('dataset manifest already exists; choose a new output name')
    # 参数先验校验，不让错误配置在漫长重放后才失败
    assign_dataset_splits([], seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction)
    paths = []
    for path in root.rglob('*.core.jsonl.gz'):
        if path.is_symlink() or not path.is_file():
            continue
        path.resolve().relative_to(root)
        paths.append(path)
        if len(paths) > MAX_DATASET_FILES:
            raise ValueError('dataset file count limit exceeded')
    entries, duplicates = [], {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        try:
            entry = inspect_dataset_entry(path)
            if entry['duel_id'] in duplicates:
                entry['eligible'] = False
                entry['rejection_reasons'].append('duplicate_duel')
                entry['duplicate_of'] = duplicates[entry['duel_id']]
            elif entry['eligible']:
                duplicates[entry['duel_id']] = relative
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, EOFError, RecursionError) as error:
            entry = {'eligible': False, 'rejection_reasons': [f'format_or_quality_failed: {str(error)[:512]}']}
        entry['path'] = relative
        entries.append(entry)
        print(f"[数据门禁] {relative}: {'通过' if entry['eligible'] else '隔离'}", flush=True)
    group_count = assign_dataset_splits(entries, seed=seed,
                                       validation_fraction=validation_fraction, test_fraction=test_fraction)
    counts = {split: sum(entry['split'] == split for entry in entries)
              for split in ('train', 'validation', 'test', 'quarantine')}
    warnings = []
    if group_count < 3 or (validation_fraction and not counts['validation']) or (test_fraction and not counts['test']):
        warnings.append('insufficient_disjoint_groups_or_empty_holdout; do not claim independent validation')
    manifest = {'format_version': DATASET_MANIFEST_VERSION, 'input_root': str(root),
                'seed': seed, 'validation_fraction': validation_fraction, 'test_fraction': test_fraction,
                'isolation': ['complete_duel', 'exact_main_extra_side_deck', 'stable_player'],
                'group_count': group_count, 'counts': counts, 'warnings': warnings, 'entries': entries,
                'training_started': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest
