# 本文件审查规范轨迹并按整局、精确构筑和稳定玩家的连通分组生成独立数据清单
import hashlib
import json
import math
import io
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from core_trajectory import (
    _iter_records, _json_bytes, _replay_core_trajectory_source, canonical_response,
    decode_response, inspect_core_trajectory,
)
from trajectory_ingest import _freeze_source, validate_player_ids

DATASET_MANIFEST_VERSION = 1
MAX_DATASET_FILES = 10_000
EXPERT_READER_VERSION = 1
MAX_EXPERT_CACHE_BYTES = 512 * 1024 * 1024
MAX_EXPERT_SAMPLE_BYTES = 8 * 1024 * 1024


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


def _participants(header):
    """来源声明不是身份认证；不依据 ID 前缀猜测真人或程序"""
    players = _stable_players(header)
    origin = header.get('ingress', {})
    models = header.get('models', {})
    if not isinstance(models, dict):
        raise ValueError('invalid dataset model identities')
    session = origin.get('external_session')
    declared = session.get('participants') if isinstance(session, dict) else None
    if session is not None and (not isinstance(session, dict) or type(session.get('format_version')) is not int
            or session.get('format_version') != 1
            or not isinstance(declared, dict) or set(declared) != {'0', '1'}):
        raise ValueError('invalid external participant metadata')
    result = {}
    for seat, identity in players.items():
        participant = declared[seat] if declared is not None else models.get(seat, {})
        if not isinstance(participant, dict):
            raise ValueError('invalid participant metadata')
        kind = participant.get('kind', 'model' if participant.get('model_id') else 'unknown')
        if kind not in ('human', 'windbot', 'model', 'rule', 'unknown'):
            raise ValueError('unsupported participant kind')
        if declared is not None and participant.get('identity') != identity:
            raise ValueError('participant identity disagrees with stable player ID')
        result[seat] = {'identity': identity, 'kind': kind}
        for name in ('executor', 'upstream_commit', 'executable_sha256'):
            if name in participant:
                value = participant[name]
                if not isinstance(value, str) or len(value) > 512:
                    raise ValueError('invalid participant provenance')
                result[seat][name] = value
    return result


def inspect_dataset_entry(path):
    """固定输入后严格重放与检查来源，只有满足全部门禁的整局可进入候选数据集"""
    with _freeze_source(path) as frozen:
        return _inspect_dataset_source(frozen)


def _inspect_dataset_source(frozen, *, _on_decision=None):
    """已冻结输入的共用门禁；只读数据集审查和专家读取走同一条重放路径"""
    report = inspect_core_trajectory(frozen)
    frozen.seek(0)
    file_digest = hashlib.sha256()
    for chunk in iter(lambda: frozen.read(1024 * 1024), b''):
        file_digest.update(chunk)
    header, footer = report['header'], report['footer']
    players = _stable_players(header)
    participants = _participants(header)
    seat_counts = Counter()
    deck_ids = sorted({deck_identity(deck) for deck in header['decks'].values()})
    duel_digest = hashlib.sha256(_json_bytes({'reset': header['reset'], 'assets': header.get('assets')}))
    frozen.seek(0)
    for record, _ in _iter_records(frozen):
        if record['kind'] == 'response':
            seat_counts[str(record['actor'])] += 1
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
            report = _replay_core_trajectory_source(frozen, _on_decision=_on_decision)
            if not report['behavior_clone_eligible']:
                reasons.append('strict_replay_or_mapping_gate_failed')
        except (ValueError, RuntimeError, OSError, KeyError, EOFError, RecursionError) as error:
            reasons.append(f'replay_failed: {str(error)[:512]}')
    return {'trace_id': header['trace_id'], 'file_sha256': file_digest.hexdigest(),
            'duel_id': duel_digest.hexdigest(), 'deck_ids': deck_ids,
            'player_ids': sorted(set(value for value in players.values() if value is not None)),
            'participants': participants, 'response_counts_by_actor': dict(seat_counts),
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


def audit_program_identities(entries):
    """量化程序身份连边，不改动任何真实 split 或真人/未知身份隔离"""
    programs = {}
    diagnostic = []
    unknown = 0
    for entry in entries:
        copy = dict(entry)
        if entry['eligible']:
            participants = entry.get('participants', {})
            for participant in participants.values():
                identity, kind = participant['identity'], participant['kind']
                if kind in ('windbot', 'model', 'rule'):
                    programs.setdefault(identity, {'identity': identity, 'kinds': set(), 'duels': set()})
                    programs[identity]['kinds'].add(kind)
                    programs[identity]['duels'].add(entry['duel_id'])
                else:
                    unknown += kind == 'unknown'
        diagnostic.append(copy)
    # 同一身份只要有一次被标作真人或未知，就保守地保留其隔离连边
    non_program = {p['identity'] for e in entries if e['eligible']
                   for p in e.get('participants', {}).values() if p['kind'] not in ('windbot', 'model', 'rule')}
    for entry in entries:
        if entry['eligible']:
            non_program.update(set(entry['player_ids']) - {
                p['identity'] for p in entry.get('participants', {}).values()})
    for entry in diagnostic:
        if entry['eligible']:
            entry['player_ids'] = [identity for identity in entry['player_ids']
                                   if identity not in programs or identity in non_program]
    diagnostic_groups = assign_dataset_splits(diagnostic)
    shared = [{'identity': value['identity'], 'kinds': sorted(value['kinds']),
               'duel_count': len(value['duels'])} for value in programs.values() if len(value['duels']) > 1]
    return {'policy': 'all_stable_players_v1', 'split_policy_changed': False,
            'shared_program_identities': sorted(shared, key=lambda value: value['identity']),
            'unknown_kind_seat_count': unknown,
            'groups_without_program_edges_diagnostic_only': diagnostic_groups,
            'diagnostic_is_independent_validation': False}


class _VerifiedDemonstrations:
    """整局通过门禁后才暴露的临时 CPU 样本缓存；不保存永久张量资产"""

    def __init__(self, stream, seats, participants):
        self._stream, self._seats, self._participants = stream, seats, participants
        self._offsets = []
        self.report = None
        self._prompts = Counter()
        self._counts = Counter()

    def _append(self, record, observation):
        if record['actor'] not in self._seats:
            return
        import numpy as np
        import torch
        index = record['matches'][0]
        if (observation.get('act_mask') is None or observation['act_mask'].shape != (1, 120)
                or not bool(observation['act_mask'][0, index])):
            raise ValueError('expert label points to a masked or missing action')
        arrays = {}
        for key, tensor in observation.items():
            if (not isinstance(key, str) or key.startswith('__') or not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != 'cpu' or tensor.requires_grad or tensor.ndim == 0 or tensor.shape[0] != 1):
                raise ValueError('expert observation must contain only detached single-state CPU tensors')
            array = tensor.numpy()
            if array.dtype.kind not in 'biuf' or (array.dtype.kind == 'f' and not np.isfinite(array).all()):
                raise ValueError('invalid expert observation dtype or non-finite value')
            arrays[key] = array
        size = sum(value.nbytes for value in arrays.values())
        if size > MAX_EXPERT_SAMPLE_BYTES:
            raise ValueError('expert sample size limit exceeded')
        metadata = {'record_seq': record['seq'], 'actor': record['actor'],
                    'participant': self._participants[str(record['actor'])],
                    'prompt_type': record['prompt_type'], 'action_index': index,
                    'observation_sha256': record['observation_sha256'],
                    'response': record['response']}
        arrays['__metadata__'] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)
        with io.BytesIO() as buffer:
            np.savez(buffer, **arrays)  # 仅数值数组，读取时禁止 pickle
            raw = buffer.getvalue()
        if len(raw) > MAX_EXPERT_SAMPLE_BYTES or self._stream.tell() + len(raw) > MAX_EXPERT_CACHE_BYTES:
            raise ValueError('expert temporary cache size limit exceeded')
        self._offsets.append((self._stream.tell(), len(raw)))
        self._stream.write(raw)
        self._prompts[record['prompt_type']] += 1
        self._counts[record['actor']] += 1

    def __iter__(self):
        import numpy as np
        import torch
        if self.report is None or self._stream.closed:
            raise ValueError('expert reader is not verified or is closed')
        for offset, length in self._offsets:
            if self._stream.closed:
                raise ValueError('expert reader is closed')
            self._stream.seek(offset)
            with np.load(io.BytesIO(self._stream.read(length)), allow_pickle=False) as arrays:
                metadata = json.loads(arrays['__metadata__'].tobytes())
                observation = {key: torch.from_numpy(arrays[key]) for key in arrays.files if key != '__metadata__'}
            yield {'observation': observation, 'action_index': metadata['action_index'],
                   'metadata': dict(metadata, trace_id=self.report['trace_id'],
                                    file_sha256=self.report['file_sha256'], duel_id=self.report['duel_id'])}


@contextmanager
def open_verified_demonstrations(path, *, expert_seats=None, expected_file_sha256=None):
    """逐项读取完整已验证示范；默认仅选 WindBot，其他来源必须显式指定座位

    一个冻结副本、一次原生重放、至多512MiB匿名临时缓存。离开 with 后删除缓存
    不相信可修改的旧清单，不返回未验证前缀，不加载模型/优化器或构造 PPO 概率
    Core 回调为进程全局；勿与同进程其他决斗在线程中并发执行，批量并行使用独立进程
    """
    with _freeze_source(path) as frozen, tempfile.TemporaryFile(mode='w+b') as cache:
        header = inspect_core_trajectory(frozen)['header']
        participants = _participants(header)
        if expert_seats is None:
            seats = tuple(int(seat) for seat, p in participants.items() if p['kind'] == 'windbot')
        else:
            if not isinstance(expert_seats, (tuple, list)):
                raise ValueError('expert seats must be a list/tuple of physical seats')
            seats = tuple(expert_seats)
        if not seats or any(type(s) is not int or s not in (0, 1) for s in seats) or len(set(seats)) != len(seats):
            raise ValueError('select at least one unique physical expert seat; no default WindBot seat found')
        if expected_file_sha256 is not None:
            if (not isinstance(expected_file_sha256, str) or len(expected_file_sha256) != 64
                    or any(ch not in '0123456789abcdef' for ch in expected_file_sha256)):
                raise ValueError('invalid expected file SHA-256')
            frozen.seek(0)
            digest = hashlib.file_digest(frozen, 'sha256').hexdigest()
            if digest != expected_file_sha256:
                raise ValueError('trajectory file SHA-256 differs from expected manifest identity')
        reader = _VerifiedDemonstrations(cache, seats, participants)
        frozen.seek(0)
        entry = _inspect_dataset_source(frozen, _on_decision=reader._append)
        if not entry['eligible']:
            raise ValueError('expert trajectory rejected: ' + '; '.join(entry['rejection_reasons']))
        if not reader._offsets:
            raise ValueError('no expert decisions for selected seats')
        reader.report = dict(entry, reader_version=EXPERT_READER_VERSION,
                             expert_seats=list(seats), sample_count=len(reader._offsets),
                             sample_counts_by_actor=dict(reader._counts),
                             prompt_counts=dict(sorted(reader._prompts.items())),
                             temporary_cache_bytes=cache.tell(), training_started=False,
                             independent_validation_established=False)
        yield reader


def preview_verified_demonstrations(path, *, expert_seats=None, preview_rows=5, expected_file_sha256=None):
    """CLI/WebUI 用的有界预览，只输出标签/来源/张量规格，不导出隐藏观测"""
    if type(preview_rows) is not int or not 0 <= preview_rows <= 20:
        raise ValueError('preview rows must be between 0 and 20')
    with open_verified_demonstrations(path, expert_seats=expert_seats,
                                     expected_file_sha256=expected_file_sha256) as reader:
        result = dict(reader.report, preview=[], observation_spec={})
        if preview_rows:
            for sample in reader:
                result['preview'].append(sample['metadata'])
                if not result['observation_spec']:
                    result['observation_spec'] = {key: {'shape': list(t.shape), 'dtype': str(t.dtype)}
                                                   for key, t in sample['observation'].items()}
                if len(result['preview']) >= preview_rows:
                    break
        return result


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
    manifest['identity_audit'] = audit_program_identities(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest
