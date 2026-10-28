"""Lossless prepared CPU examples, source adapters and explicit holdout policies."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import zipfile

import numpy as np
import torch

from core_trajectory import (_iter_records, _json_bytes, canonical_response, decode_response,
                             encode_response, file_sha256, observation_sha256)
from deck_trajectory import DECK_PROFILE_STATIC_OBSERVATION_KEYS
from protocol_schema import get_current_protocol_metadata
from trajectory_dataset import _participants, deck_identity, open_verified_demonstrations

CACHE_FORMAT_VERSION = 1
MAX_BLOCK_BYTES = 32 * 1024 * 1024
MAX_CACHE_BYTES = 2 * 1024**3
ROLES = ('train', 'validation', 'test')


def read_json(path, limit=16 * 1024 * 1024):
    with open(path, 'rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('imitation JSON size limit exceeded')
    return json.loads(raw)


def write_json(path, value):
    """Only create new workflow artifacts; never replace a user's manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def data_identity(root='.'):
    """Once per workflow: content identity without constructing/preloading a Core."""
    root = Path(root)
    scripts = {}
    for directory in (root / 'script', root / 'script/official'):
        for path in directory.glob('*.lua'):
            scripts.setdefault(path.name, path)
    digest = hashlib.sha256()
    for name, path in sorted(scripts.items()):
        content = path.read_bytes()
        digest.update(_json_bytes({'name': name, 'size': len(content)}))
        digest.update(content)
    return {'protocol': get_current_protocol_metadata(),
            'core_sha256': file_sha256(root / ('ocgcore.dll' if platform.system() == 'Windows' else 'ocgcore.so')),
            'cdb_sha256': file_sha256(root / 'cards.cdb'), 'scripts_sha256': digest.hexdigest()}


def source_info(path):
    """Common physical-duel identity for raw and canonical wrappers of one game."""
    digest = None
    header = None
    for record, _ in _iter_records(path):
        if record['kind'] == 'header':
            header = record
            digest = hashlib.sha256(_json_bytes({'reset': header['reset'], 'assets': header.get('assets')}))
        elif record['kind'] in ('chunk', 'response', 'automatic_response'):
            value = ({'kind': 'chunk', 'hex': record['hex']} if record['kind'] == 'chunk' else
                     {'kind': 'response', 'actor': record['actor'],
                      'response_buffer': canonical_response(decode_response(record['response'])).hex()})
            digest.update(_json_bytes(value))
    if header is None:
        raise ValueError('missing imitation source header')
    if header.get('source') == 'link_capture':
        canonical_header = dict(header, ingress={'player_ids': header['player_ids'],
                                                'external_session': header.get('external_session')})
    else:
        canonical_header = header
    session = canonical_header.get('ingress', {}).get('external_session') or {}
    return {'source_path': str(Path(path).resolve()), 'file_sha256': file_sha256(path),
            'duel_id': digest.hexdigest(), 'trace_id': header['trace_id'],
            'assets': header.get('assets'),
            'decks': header['decks'],
            'opening_id': hashlib.sha256(_json_bytes({'reset': header['reset'], 'assets': header.get('assets')})).hexdigest(),
            'deck_ids': sorted({deck_identity(deck) for deck in header['decks'].values()}),
            'participants': _participants(canonical_header),
            'evaluation_only': session.get('purpose') == 'imitation_evaluation'}


def _load_arrays(path):
    # Prepared numeric arrays only: validate bounded decompression once per block.
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > MAX_BLOCK_BYTES:
            raise ValueError('imitation block size limit exceeded')
    with np.load(path, allow_pickle=False) as values:
        result = {key: values[key] for key in values.files}
    if any(array.dtype.kind not in 'biuf' for array in result.values()):
        raise ValueError('imitation cache must contain numeric arrays only')
    return result


class ExampleCacheWriter:
    """A bounded block staging area, published only after a complete source passes."""

    def __init__(self, root, identity, *, block_rows=32):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._temp = tempfile.TemporaryDirectory(prefix='.preparing_', dir=self.root)
        self.directory = Path(self._temp.name)
        self.block_rows = block_rows
        self.pending, self.blocks, self.profiles = [], [], {}
        self.schema = None
        self.counts, self.effective = Counter(), Counter()
        self.prompts, self.rejected = Counter(), Counter()
        self.bytes = 0
        self.closed = False

    def add(self, observation, *, actor, participant, prompt_type, action_index, response=None,
            record_seq=0):
        mask = observation['act_mask']
        if (type(action_index) is not int or not 0 <= action_index < 120
                or mask.shape != (1, 120) or not bool(mask[0, action_index])):
            raise ValueError('expert label must point to an available model candidate')
        schema = {key: {'shape': list(t.shape[1:]), 'dtype': str(t.numpy().dtype)}
                  for key, t in observation.items()}
        if self.schema is None:
            self.schema = schema
        if schema != self.schema:
            raise ValueError('imitation observation schema changed within one source')
        if any(t.device.type != 'cpu' or t.requires_grad or t.shape[0] != 1
               or (t.is_floating_point() and not torch.isfinite(t).all()) for t in observation.values()):
            raise ValueError('imitation requires finite detached CPU observations')
        static = {key: observation[key][0].numpy() for key in DECK_PROFILE_STATIC_OBSERVATION_KEYS}
        static_hash = observation_sha256({key: observation[key] for key in static})
        if static_hash not in self.profiles:
            name = f'profile_{len(self.profiles)}.npz'
            np.savez_compressed(self.directory / name, **static)
            self.profiles[static_hash] = name
        teacher = participant['identity']
        effective = int(mask.sum()) > 1
        metadata = {'actor': actor, 'participant': dict(participant), 'prompt_type': prompt_type,
                    'action_index': action_index, 'profile': self.profiles[static_hash],
                    'effective': effective, 'record_seq': record_seq,
                    'observation_sha256': observation_sha256(observation),
                    'response': encode_response(response) if response is not None else None}
        dynamic = {key: t[0].numpy().copy() for key, t in observation.items() if key not in static}
        self.pending.append((dynamic, metadata))
        self.counts[teacher] += 1
        self.effective[teacher] += effective
        self.prompts[str(prompt_type)] += 1
        if len(self.pending) >= self.block_rows:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        name = f'block_{len(self.blocks):05d}.npz'
        values = {key: np.stack([row[0][key] for row in self.pending]) for key in self.pending[0][0]}
        metadata = [row[1] for row in self.pending]
        values['__metadata__'] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)
        if sum(array.nbytes for array in values.values()) > MAX_BLOCK_BYTES:
            raise ValueError('imitation block exceeds its uncompressed bound')
        np.savez_compressed(self.directory / name, **values)
        self.bytes += (self.directory / name).stat().st_size
        if self.bytes > MAX_CACHE_BYTES:
            raise ValueError('imitation per-duel cache size limit exceeded')
        self.blocks.append({'file': name, 'sha256': file_sha256(self.directory / name), 'rows': len(self.pending)})
        self.pending.clear()

    def finish(self, source, *, verification):
        if source.get('evaluation_only'):
            raise ValueError('evaluation captures never become automatic training examples')
        self._flush()
        if not self.blocks:
            raise ValueError('no uniquely mapped expert examples in source')
        key = hashlib.sha256(_json_bytes({'source': source['file_sha256'], 'identity': self.identity,
                                         'teachers': sorted(self.counts), 'format': CACHE_FORMAT_VERSION})).hexdigest()
        profiles = {name: file_sha256(self.directory / name) for name in self.profiles.values()}
        manifest = {'format_version': CACHE_FORMAT_VERSION, 'cache_id': key, 'identity': self.identity,
                    'source': source, 'verification': verification, 'blocks': self.blocks, 'profiles': profiles,
                    'schema': self.schema, 'teacher_counts': dict(self.counts),
                    'effective_teacher_counts': dict(self.effective), 'prompt_counts': dict(self.prompts),
                    'rejected_labels': dict(self.rejected), 'training_started': False}
        write_json(self.directory / 'examples.json', manifest)
        target = self.root / key
        if target.exists():
            existing = read_json(target / 'examples.json')
            if existing != manifest:
                raise ValueError('existing imitation cache differs; choose a clean cache root')
        else:
            os.rename(self.directory, target)
        self.close()
        return str(target / 'examples.json')

    def close(self):
        if not self.closed:
            self.closed = True
            self._temp.cleanup()


def prepare_trace(path, cache_root, identity, *, expert_seats=None):
    """Legacy full traces replay once; immutable prepared cache is reused thereafter."""
    source = source_info(path)
    if source['evaluation_only']:
        raise ValueError('evaluation-only source cannot be prepared for learning')
    seats = expert_seats
    if seats is None:
        seats = [int(s) for s, p in source['participants'].items() if p['kind'] == 'windbot']
    teachers = sorted({source['participants'][str(s)]['identity'] for s in seats})
    key = hashlib.sha256(_json_bytes({'source': source['file_sha256'], 'identity': identity,
                                     'teachers': teachers, 'format': CACHE_FORMAT_VERSION})).hexdigest()
    cached = Path(cache_root) / key / 'examples.json'
    if cached.exists():
        check = read_json(cached)
        if check['identity'] != identity or check['source'] != source:
            raise ValueError('prepared source identity mismatch')
        return str(cached.resolve())
    writer = ExampleCacheWriter(cache_root, identity)
    try:
        with open_verified_demonstrations(path, expert_seats=seats,
                                         expected_file_sha256=source['file_sha256']) as examples:
            for sample in examples:
                meta = sample['metadata']
                writer.add(sample['observation'], actor=meta['actor'], participant=meta['participant'],
                           prompt_type=meta['prompt_type'], action_index=sample['action_index'],
                           response=decode_response(meta['response']), record_seq=meta['record_seq'])
        return writer.finish(source, verification='strict_original_replay')
    finally:
        writer.close()


def build_manifest(cache_paths, output, *, mode='episode', seed=73, validation_executors=(),
                   test_executors=(), fractions=(0.1, 0.1), declared_roles=None, plan=None):
    """Explicit evaluation goal, not one universal program/human identity split."""
    if mode not in ('none', 'episode', 'deck', 'executor') or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('invalid imitation split policy or seed')
    validation_executors, test_executors = set(validation_executors), set(test_executors)
    if validation_executors & test_executors:
        raise ValueError('an executor cannot be both validation and test')
    if len(fractions) != 2 or any(not 0 <= f < 1 for f in fractions) or sum(fractions) >= 1:
        raise ValueError('invalid imitation holdout fractions')
    entries, seen = [], set()
    for path in sorted(set(map(str, cache_paths))):
        meta = read_json(path)
        source = meta['source']
        if source['duel_id'] in seen:
            continue
        seen.add(source['duel_id'])
        participants = source['participants']
        executors = sorted({p['executor'] for p in participants.values() if p.get('kind') == 'windbot' and p.get('executor')})
        reserved = {'validation' for e in executors if e in validation_executors} | {'test' for e in executors if e in test_executors}
        if len(reserved) > 1:
            raise ValueError('one duel connects validation and test executors')
        role = next(iter(reserved), None)
        if declared_roles and source['duel_id'] in declared_roles:
            declared = declared_roles[source['duel_id']]
            if declared not in ROLES or (role is not None and role != declared):
                raise ValueError('declared source role conflicts with held-out executor')
            role = declared
        entries.append({'cache': str(Path(path).resolve()), 'cache_sha256': file_sha256(path),
                        'duel_id': source['duel_id'], 'deck_ids': source['deck_ids'],
                        'opening_id': source.get('opening_id', source['duel_id']),
                        'participants': participants, 'executors': executors, 'reserved_role': role,
                        'effective_count': sum(meta['effective_teacher_counts'].values()),
                        'sample_count': sum(meta['teacher_counts'].values())})
    # Keep duplicate games and all human/unknown identities together. Program IDs are provenance.
    parents = list(range(len(entries)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen_keys = {}
    nonprogram = {p['identity'] for e in entries for p in e['participants'].values()
                  if p['kind'] not in ('model', 'windbot', 'rule')}
    for i, entry in enumerate(entries):
        keys = [('duel', entry['duel_id']), ('opening', entry['opening_id'])]
        keys += [('human', p['identity']) for p in entry['participants'].values() if p['identity'] in nonprogram]
        if mode in ('deck', 'executor'):
            keys += [('deck', x) for x in entry['deck_ids']]
        if mode == 'executor':
            keys += [('executor', x) for x in entry['executors']]
        for key in keys:
            if key in seen_keys:
                parents[find(i)] = find(seen_keys[key])
            else:
                seen_keys[key] = i
    groups = {}
    for i, entry in enumerate(entries):
        groups.setdefault(find(i), []).append(entry)
    assignments = []
    for group in groups.values():
        roles = {e['reserved_role'] for e in group if e['reserved_role']}
        if len(roles) > 1:
            raise ValueError('held-out roles connected by shared deck/human/executor')
        group_id = hashlib.sha256(_json_bytes(sorted(e['duel_id'] for e in group))).hexdigest()
        role = next(iter(roles), None)
        assignments.append((group_id, group, role))
    # Split complete independent groups, keeping at least one training group.
    # A single game remains training-only instead of randomly becoming unusable.
    available = sorted((item for item in assignments if item[2] is None),
        key=lambda item: hashlib.sha256(f'{seed}:{item[0]}'.encode()).hexdigest())
    allocated = {}
    if mode in ('episode', 'deck'):
        remaining = max(0, len(available) - 1)
        group_count = len(available)
        for role, fraction in zip(('validation', 'test'), fractions):
            count = min(remaining, max(1, round(group_count * fraction))) if fraction and remaining else 0
            for group_id, _, _ in available[:count]:
                allocated[group_id] = role
            available = available[count:]
            remaining -= count
    for group_id, group, role in assignments:
        role = role or allocated.get(group_id, 'train')
        for entry in group:
            entry.update(split=role, group_id=group_id)
            entry.pop('reserved_role')
    identities = [read_json(e['cache'])['identity'] for e in entries]
    if identities and any(x != identities[0] for x in identities):
        raise ValueError('cannot combine imitation caches prepared with different assets')
    for entry in entries:
        entry['cache'] = Path(os.path.relpath(entry['cache'], Path(output).resolve().parent)).as_posix()
    result = {'format_version': 1, 'workflow': 'imitation', 'identity': identities[0] if identities else None,
              'split_mode': mode, 'seed': seed, 'entries': entries, 'collection_plan': plan,
              'counts': {role: sum(e['split'] == role for e in entries) for role in ROLES},
              'warnings': [], 'training_started': False}
    if not result['counts']['validation']:
        result['warnings'].append('no_validation_split; checkpoint selection cannot claim held-out validation')
    if not result['counts']['test']:
        result['warnings'].append('no_test_split; no final unseen-opponent claim')
    if mode == 'episode':
        result['warnings'].append('episode_holdout_only; programs and builds may be shared by design')
    write_json(output, result)
    return result


def recorded_teacher_names(source):
    """Read provenance headers for UI labels; this is not an eligibility check."""
    source = Path(source)
    paths = [source] if source.is_file() else list(source.rglob('*.link.jsonl.gz')) + list(source.rglob('*.core.jsonl.gz'))
    names = set()
    for path in paths:
        if not path.name.endswith(('.link.jsonl.gz', '.core.jsonl.gz')):
            continue
        records = _iter_records(path)
        try:
            header = next(records)[0]
            if header.get('source') == 'link_capture':
                header = dict(header, ingress={'player_ids': header['player_ids'],
                                               'external_session': header.get('external_session')})
            names.update(p['executor'] for p in _participants(header).values() if p.get('executor'))
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            continue
        finally:
            records.close()
    return sorted(names)


def imitation_artifacts(project_root='.', *, include_legacy=False):
    """Inventory owned batch files only; never follow dataset/source references."""
    root = Path(project_root).resolve()
    rows = []
    for parent, kind in (('replay_data/imitation', 'data'), ('models/imitation', 'model')):
        directory = root / parent
        if not directory.is_dir() or directory.is_symlink():
            continue
        for batch in sorted(directory.iterdir()):
            if not batch.is_dir() or batch.is_symlink() or batch.name.startswith('.'):
                continue
            source_label = batch.name
            try:
                if (batch / 'preparation.json').is_file():
                    source_label = str(read_json(batch / 'preparation.json').get('source', batch.name))
                elif (batch / 'plan.json').is_file():
                    source_label = ', '.join(str(item['executor']) for item in read_json(batch / 'plan.json')['executors'])
                elif (batch / 'evaluation_plan.json').is_file():
                    source_label = ', '.join(sorted({str(item['executor']) for item in read_json(batch / 'evaluation_plan.json')['rows']}))
            except (ValueError, OSError, KeyError, TypeError):
                pass
            for path in batch.rglob('*'):
                if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
                    continue
                relative = path.resolve().relative_to(root).as_posix()
                part = path.relative_to(batch).parts[0]
                category = ('model' if path.suffix == '.pth' else 'cache' if part == 'cache' else
                            'capture' if path.name.endswith('.link.jsonl.gz') else
                            'trajectory' if path.name.endswith('.core.jsonl.gz') else
                            'replay' if part == 'replays' else 'log' if path.suffix == '.log' else 'report')
                rows.append({'path': relative, 'batch': f'{kind}/{batch.name}', 'category': category,
                             'source': source_label[:200], 'bytes': path.stat().st_size})
    # BC TensorBoard files share their run ID with the model batch.
    runs = root / 'runs'
    for directory in sorted(runs.glob('imitation_*')):
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.rglob('*'):
            if path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents if p != root):
                rows.append({'path': path.resolve().relative_to(root).as_posix(),
                    'batch': 'model/' + directory.name.removeprefix('imitation_'),
                    'category': 'tensorboard', 'bytes': path.stat().st_size})
    if include_legacy:
        legacy = []
        for folder, pattern, category in (
                ('replays', '*.yrp', 'source'), ('replays', '*.yrp3d', 'source'),
                ('replays/link_captures', '*.link.jsonl.gz', 'capture'),
                ('replays/imported_trajectories', '*.core.jsonl.gz', 'trajectory'),
                ('replay_data/ingest_reports', '*.json', 'report'),
                ('replay_data/expert_reports', '*.json', 'report'),
                ('replay_data/datasets', '*.json', 'report')):
            directory = root / folder
            if directory.is_symlink():
                continue
            for path in (directory.rglob(pattern) if folder == 'replays' else directory.glob(pattern)):
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                    legacy.append({'path': path.relative_to(root).as_posix(), 'category': category,
                                   'bytes': path.stat().st_size})
        # Only explicit direct-file provenance joins legacy wrappers. A dataset's
        # list of many referenced games must never join the entire library.
        parents, keys = list(range(len(legacy))), {}
        def find(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        for i, row in enumerate(legacy):
            path = root / row['path']
            links = [('path', str(path.resolve())), ('digest', file_sha256(path))]
            value = {}
            try:
                if path.name.endswith(('.link.jsonl.gz', '.core.jsonl.gz')):
                    records = _iter_records(path)
                    try:
                        value = next(records)[0]
                    finally:
                        records.close()
                    origin = value.get('ingress') or {}
                    if origin.get('file_sha256'):
                        links.append(('digest', origin['file_sha256']))
                elif path.suffix == '.json':
                    value = read_json(path)
                if isinstance(value, dict):
                    if isinstance(value.get('file_sha256'), str):
                        links.append(('digest', value['file_sha256']))
                    for field in ('trajectory_path', 'source_path', 'path'):
                        if isinstance(value.get(field), str):
                            target = Path(value[field])
                            if not target.is_absolute():
                                target = root / target
                            if target.resolve().is_relative_to(root):
                                links.append(('path', str(target.resolve())))
            except (ValueError, OSError, KeyError, StopIteration, TypeError):
                pass  # Damaged files remain individually manageable.
            for key in links:
                if key in keys:
                    parents[find(i)] = find(keys[key])
                else:
                    keys[key] = i
        for i, row in enumerate(legacy):
            row['batch'] = 'source/' + legacy[find(i)]['path']
        rows.extend(legacy)
    return rows


def delete_imitation_artifacts(paths, project_root='.'):
    """Delete selected inventory members, without following manifests to sources."""
    root = Path(project_root).resolve()
    owned = {row['path'] for row in imitation_artifacts(root, include_legacy=True)}
    selected = set(paths)
    if not selected <= owned:
        raise ValueError('selection contains a missing file or a file outside managed imitation batches')
    targets = [(root / name).resolve() for name in sorted(selected)]
    if any(not path.is_relative_to(root) for path in targets):
        raise ValueError('imitation cleanup target escapes project root')
    for path in targets:
        path.unlink()
    # Remove empty directories bottom-up so deleted batches disappear from lists.
    parents = {parent for path in targets for parent in path.parents
               if parent != root and parent.is_relative_to(root)
               and (parent.is_relative_to(root / 'replay_data/imitation')
                    or parent.is_relative_to(root / 'models/imitation')
                    or parent.name.startswith('imitation_') and parent.parent == root / 'runs')}
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        if parent not in (root / 'replay_data/imitation', root / 'models/imitation'):
            try:
                parent.rmdir()
            except OSError:
                pass
    return len(targets)


class PreparedDataset:
    """Verify cache identity once, then stream lossless blocks without Core or replay."""

    def __init__(self, path, *, identity=None):
        self.path = Path(path).resolve()
        self.digest = file_sha256(self.path)
        self.manifest = read_json(self.path)
        if self.manifest.get('format_version') != 1 or self.manifest.get('workflow') != 'imitation':
            raise ValueError('not an imitation dataset manifest')
        if identity is not None and self.manifest['identity'] != identity:
            raise ValueError('prepared dataset assets changed; prepare a new cache')
        self.entries = []
        for entry in self.manifest['entries']:
            cache_path = Path(entry['cache'])
            if not cache_path.is_absolute():
                cache_path = self.path.parent / cache_path
            if file_sha256(cache_path) != entry['cache_sha256']:
                raise ValueError('imitation cache manifest changed')
            meta = read_json(cache_path)
            root = cache_path.parent
            if meta.get('format_version') != CACHE_FORMAT_VERSION or meta['identity'] != self.manifest['identity']:
                raise ValueError('imitation cache format/identity mismatch')
            if meta['source']['duel_id'] != entry['duel_id'] or entry['split'] not in ROLES:
                raise ValueError('prepared cache source/split mismatch')
            if self.entries and meta['schema'] != self.entries[0][1]['schema']:
                raise ValueError('prepared dataset observation schemas differ')
            for name, digest in [(b['file'], b['sha256']) for b in meta['blocks']] + list(meta['profiles'].items()):
                if Path(name).name != name or file_sha256(root / name) != digest:
                    raise ValueError('imitation cache block missing or changed')
            self.entries.append((entry, meta, root))

    def batches(self, split, batch_size, *, epoch=0, seed=73, balanced=True):
        if split not in ROLES or type(batch_size) is not int or not 1 <= batch_size <= 1024:
            raise ValueError('invalid imitation split/batch size')
        rng = random.Random((seed << 32) + epoch)
        blocks, teachers = [], Counter()
        for entry, meta, root in self.entries:
            if entry['split'] == split:
                teachers.update(meta['effective_teacher_counts'])
                blocks.extend((meta, root, block) for block in meta['blocks'])
        teachers = Counter({key: value for key, value in teachers.items() if value})
        total = sum(teachers.values())
        if split == 'train':
            rng.shuffle(blocks)
        pending = []
        for meta, root, block in blocks:
            arrays = _load_arrays(root / block['file'])
            rows = json.loads(arrays.pop('__metadata__').tobytes())
            order = list(range(len(rows)))
            if split == 'train':
                rng.shuffle(order)
            profiles = {name: _load_arrays(root / name) for name in {row['profile'] for row in rows}}
            for i in order:
                row = rows[i]
                if not row['effective']:
                    continue
                observation = {key: torch.from_numpy(value[i:i+1]) for key, value in arrays.items()}
                observation.update({key: torch.from_numpy(value).unsqueeze(0)
                                    for key, value in profiles[row['profile']].items()})
                weight = total / (len(teachers) * teachers[row['participant']['identity']]) if balanced else 1.0
                pending.append((observation, row, weight))
                if len(pending) == batch_size:
                    yield _collate(pending)
                    pending.clear()
        if pending:
            yield _collate(pending)


def _collate(rows):
    observations = {key: torch.cat([r[0][key] for r in rows], dim=0) for key in rows[0][0]}
    labels = torch.tensor([r[1]['action_index'] for r in rows], dtype=torch.long)
    if not bool(observations['act_mask'].gather(1, labels[:, None]).all()):
        raise ValueError('prepared expert label became masked')
    return observations, labels, torch.tensor([r[2] for r in rows]), [r[1] for r in rows]
