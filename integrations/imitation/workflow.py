"""Separate imitation collection, training and fixed-opponent evaluation workflows."""
from collections import Counter
import hashlib
import lzma
import math
from pathlib import Path
import uuid

import torch
import torch.nn.functional as F

from core_trajectory import file_sha256, matching_action_indices
from integrations.duel_session import WINDBOT_SELECTION_LABEL_V1, resolve_core_seed
from integrations.imitation import IMITATION_WORKFLOW_VERSION
from integrations.imitation.data import (
    ExampleCacheWriter, PreparedDataset, ROLES, build_manifest, data_identity,
    prepare_trace, read_json, source_info, write_json,
)


class NativeExpertSink:
    """Trusted in-process producer: one actual duel, no conversion/replay engines."""

    def __init__(self, cache_root, identity, *, expert_seats=None, source_override=None):
        from feature_encoder import GalateaEncoder
        self.encoder = GalateaEncoder()
        self.writer = ExampleCacheWriter(cache_root, identity)
        self.expert_seats, self.source_override = expert_seats, source_override

    def record(self, snapshot, message, response, actor, participant, sequence, *, observation=None):
        if (self.expert_seats is None and participant['kind'] != 'windbot'
                or self.expert_seats is not None and actor not in self.expert_seats):
            return
        matches = matching_action_indices(snapshot, response, message[0], message[1:],
            response_adapter=WINDBOT_SELECTION_LABEL_V1 if participant['kind'] == 'windbot' else None)
        if len(matches) != 1 or matches[0] >= 120:
            self.writer.rejected['unmapped_expert_action'] += 1
            return  # Do not invent a label or insert the response into the pool.
        if observation is None:
            observation = self.encoder.encode(snapshot, player_id=actor)
        self.writer.add(observation, actor=actor, participant=participant, prompt_type=message[0],
                        action_index=matches[0], response=response, record_seq=sequence)

    def finish(self, path, *, valid):
        try:
            if not valid or path is None:
                return None
            from trajectory_ingest import inspect_link_capture
            from core_trajectory import inspect_core_trajectory
            report = (inspect_link_capture(path) if str(path).endswith('.link.jsonl.gz') else
                      inspect_core_trajectory(path))  # Format/footer only, never starts Core.
            if (not report['footer']['core_terminal'] or report['footer']['truncated']
                    or report['footer']['errors']):
                return None
            source = source_info(path)
            if any(not p['identity'] for p in source['participants'].values()):
                raise ValueError('complete expert sources need stable participant identifiers')
            assets = source['assets']
            expected = self.writer.identity
            if (assets['core_sha256'] != expected['core_sha256'] or assets['cdb_sha256'] != expected['cdb_sha256']
                    or assets['loaded_scripts_sha256'] != expected['scripts_sha256']
                    or assets['protocol'] != expected['protocol']):
                raise ValueError('collection assets changed while preparing the imitation corpus')
            return self.writer.finish(self.source_override or source,
                verification='original_output_reconstruction' if self.source_override else 'native_complete_session')
        finally:
            self.close()

    def close(self):
        self.writer.close()


def collect_demonstrations(output, *, train_executors, validation_executors=(), test_executors=(),
                           games_per_executor=10, seed=None, runtime=None, mono=None,
                           timeout=600, max_decisions=4000, families=None, split_mode=None,
                           split_seed=73, validation_fraction=0.1, test_fraction=0.1):
    """Freeze role/configuration first. Held-out teachers never train the model."""
    from integrations.windbot.runtime import executor_catalog, load_deck, executable_sha256
    from integrations.windbot.session import run_controlled_duel
    from trajectory_dataset import deck_identity
    if type(games_per_executor) is not int or not 1 <= games_per_executor <= 10000:
        raise ValueError('games per executor must be 1..10000')
    roles = {'train': list(train_executors), 'validation': list(validation_executors), 'test': list(test_executors)}
    if families is not None and (not isinstance(families, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in families.items())):
        raise ValueError('family map must contain executor names and nonempty family strings')
    names = [name for values in roles.values() for name in values]
    if not roles['train'] or len(names) != len(set(names)):
        raise ValueError('choose training executors; roles must be disjoint and contain no duplicates')
    randomized_seed = seed is None
    seed = resolve_core_seed(seed, len(names) * games_per_executor)
    split_mode = split_mode or ('executor' if validation_executors or test_executors else 'episode')
    if (split_mode not in ('none', 'episode', 'deck', 'executor')
            or not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1
            or validation_fraction + test_fraction >= 1
            or type(split_seed) is not int or not 0 <= split_seed < 2**32):
        raise ValueError('invalid collection split settings')
    if split_mode != 'executor' and (validation_executors or test_executors):
        raise ValueError('separate executor pools require executor split mode')
    root = Path(output).resolve()
    if (root / 'plan.json').exists() or (root / 'dataset.json').exists():
        raise FileExistsError('collection workspace already exists; choose a new output directory')
    catalog, configs = executor_catalog(), []
    for role, values in roles.items():
        for name in values:
            if name not in catalog or not catalog[name]['deck_available']:
                raise ValueError(f'executor/build unavailable: {name}')
            config = catalog[name]
            configs.append({'executor': name, 'role': role, 'level': config['level'],
                            'deck_file': str(Path(config['deck_file']).resolve()),
                            'deck_sha256': file_sha256(config['deck_file']),
                            'deck_identity': deck_identity(load_deck(config['deck_file'])),
                            'family': (families or {}).get(name)})
    for i, config in enumerate(configs):
        for other in configs[:i]:
            if config['role'] != other['role'] and (config['deck_identity'] == other['deck_identity']
                    or (config['family'] and config['family'] == other['family'])):
                raise ValueError('same deck/declared family cannot cross executor-holdout roles')
    plan = {'format_version': 1, 'purpose': 'imitation_collection', 'seed': seed,
            'randomized_seed': randomized_seed, 'split_seed': split_seed, 'split_mode': split_mode,
            'games_per_executor': games_per_executor, 'executors': configs,
            'windbot_executable_sha256': executable_sha256(runtime, mono),
            'runtime': runtime, 'mono': mono, 'timeout': timeout, 'max_decisions': max_decisions,
            'bot_randomness_controlled': False}
    write_json(root / 'plan.json', plan)
    print(f'[Imitation/Collect] Core seed={seed}, randomized={randomized_seed}, split seed={split_seed}', flush=True)
    identity = data_identity()
    cache_paths, declared, reports = [], {}, []
    for config in configs:
        for game in range(games_per_executor):
            duel_seed = seed + len(reports)
            sink = NativeExpertSink(root / 'cache', identity)
            try:
                result = run_controlled_duel(executor=config['executor'], opponent_executor=config['executor'],
                    runtime=runtime, mono=mono, seed=duel_seed, timeout=timeout, max_decisions=max_decisions,
                    capture_root=str(root / 'captures'), log_root=str(root / 'bot_logs'),
                    expert_sink=sink, session_purpose='imitation_collection', verify=False)
            finally:
                sink.close()
            result['role'] = config['role']
            reports.append(result)
            write_json(root / 'games' / f'game_{len(reports):05d}.json', result)
            if result.get('expert_cache'):
                cache_paths.append(result['expert_cache'])
                if split_mode == 'executor':
                    declared[read_json(result['expert_cache'])['source']['duel_id']] = config['role']
            print(f"[Imitation/Collect] {config['executor']} {config['role']} {game + 1}/{games_per_executor}: "
                  f"{'ready' if result.get('expert_cache') else result.get('error') or result.get('expert_cache_error') or 'no examples'}", flush=True)
    if not cache_paths:
        raise ValueError('no complete expert games produced usable examples; inspect game reports')
    result = build_manifest(cache_paths, root / 'dataset.json', mode=split_mode, seed=split_seed,
                            declared_roles=declared, plan=plan, fractions=(validation_fraction, test_fraction))
    return dict(result, dataset_path=str(root / 'dataset.json'), attempted_games=len(reports))


def prepare_demonstrations(source, output, *, mode='episode', seed=73, expert_seats=None,
                           validation_executors=(), test_executors=(), validation_fraction=0.1,
                           test_fraction=0.1):
    """Explicit offline sources. Imported files are prepared once, never per epoch."""
    source, root = Path(source).resolve(), Path(output).resolve()
    if (root / 'dataset.json').exists():
        raise FileExistsError('imitation dataset already exists; use its prepared manifest or a new output')
    if source.is_dir():
        paths = (sorted(source.rglob('*.core.jsonl.gz')) + sorted(source.rglob('*.link.jsonl.gz'))
                 + sorted(source.rglob('*.yrp')) + sorted(source.rglob('*.yrp3d')))
    else:
        paths = [source]
    if not paths or len(paths) > 10000:
        raise ValueError('no supported full trajectories, or file-count limit exceeded')
    identity, caches, seen, refusals, processing = data_identity(), [], set(), [], []
    for path in paths:
        try:
            if path.suffix.lower() in ('.yrp', '.yrp3d'):
                from trajectory_ingest import import_yrp
                from yrp_ingest import inspect_yrp
                inspection = inspect_yrp(path)
                result = import_yrp(path, output_root=root / 'imported', _verify_replay=False)
                processing.append({'path': str(path), 'status': 'diagnostic_only', 'inspection': inspection,
                    'conversion': result, 'reason': 'original_output_or_assets_unverified'})
                refusals.append({'path': str(path), 'reason': 'original_output_or_assets_unverified'})
                continue
            info = source_info(path)
            if info['evaluation_only']:
                raise ValueError('evaluation-only capture is excluded from learning')
            if info['duel_id'] in seen:
                continue
            original = path
            if path.name.endswith('.link.jsonl.gz'):
                from trajectory_ingest import import_link_capture
                teachers = sorted({p['identity'] for s, p in info['participants'].items()
                    if (expert_seats is None and p['kind'] == 'windbot') or expert_seats is not None and int(s) in expert_seats})
                from core_trajectory import _json_bytes
                cache_id = hashlib.sha256(_json_bytes({'source': info['file_sha256'], 'identity': identity,
                    'teachers': teachers, 'format': 1})).hexdigest()
                cached = root / 'cache' / cache_id / 'examples.json'
                if cached.exists():
                    result = {'expert_cache': str(cached)}
                else:
                    sink = NativeExpertSink(root / 'cache', identity, expert_seats=expert_seats, source_override=info)
                    try:
                        result = import_link_capture(path, output_root=root / 'imported', _expert_sink=sink)
                    finally:
                        sink.close()
                if not result.get('expert_cache'):
                    raise ValueError('external full capture reconstruction failed: ' + str(result.get('verification_error') or result.get('conversion_error') or result.get('rejection_reasons')))
                caches.append(result['expert_cache'])
            else:
                caches.append(prepare_trace(path, root / 'cache', identity, expert_seats=expert_seats))
            seen.add(info['duel_id'])
            processing.append({'path': str(path), 'status': 'ready', 'cache': str(caches[-1])})
            print(f'[Imitation/Prepare] ready: {original}', flush=True)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, EOFError, lzma.LZMAError) as error:
            refusals.append({'path': str(path), 'reason': str(error)[:1024]})
            processing.append({'path': str(path), 'status': 'excluded', 'reason': str(error)[:1024]})
            print(f'[Imitation/Prepare] excluded: {path}: {error}', flush=True)
    preparation = {'source': str(source), 'refusals': refusals, 'processing': processing,
                   'ready_sources': len(caches), 'training_started': False}
    write_json(root / 'preparation.json', preparation)
    if not caches:
        return dict(preparation, dataset_path=None, report_path=str(root / 'preparation.json'),
                    counts={role: 0 for role in ROLES})
    result = build_manifest(caches, root / 'dataset.json', mode=mode, seed=seed,
        validation_executors=validation_executors, test_executors=test_executors,
        fractions=(validation_fraction, test_fraction))
    return dict(result, dataset_path=str(root / 'dataset.json'))


FROZEN_PREFIXES = ('value_head.', 'auxiliary_heads.', 'goal_planner.target_decoder.',
                   'plan_value_proj.', 'plan_value_gate')


def configure_policy_learning(network):
    """Do not directly supervise critic/diagnostic decoders; shared outputs may drift."""
    parameters = []
    for name, value in network.named_parameters():
        value.requires_grad_(not name.startswith(FROZEN_PREFIXES))
        if value.requires_grad:
            parameters.append(value)
    return parameters


def behavior_clone_loss(logits, labels, weights):
    if not torch.isfinite(logits).all():
        raise ValueError('non-finite imitation policy output')
    losses = F.cross_entropy(logits.float(), labels, reduction='none')
    return (losses * weights).mean(), losses


def _epoch(network, dataset, *, split, batch_size, epoch, seed, device, optimizer=None, scaler=None):
    training = optimizer is not None
    network.train(training)
    count = correct = 0
    loss_sum = 0.0
    balanced_loss_sum = 0.0
    breakdown = {}
    updates = 0
    for observations, labels, weights, metadata in dataset.batches(split, batch_size, epoch=epoch, seed=seed):
        observations = {key: value.to(device) for key, value in observations.items()}
        labels, weights = labels.to(device), weights.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == 'cuda'):
            logits, _, _ = network(observations)
            loss, losses = behavior_clone_loss(logits, labels, weights)
        if not torch.isfinite(loss):
            raise ValueError('non-finite imitation loss; no optimizer update applied')
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad], 0.5,
                                          error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
        hits = (logits.argmax(-1) == labels).detach().cpu().tolist()
        per_loss = losses.detach().cpu().tolist()
        count += len(metadata)
        correct += sum(hits)
        loss_sum += sum(per_loss)
        balanced_loss_sum += float((losses.detach() * weights).sum().cpu())
        for item, hit, row_loss in zip(metadata, hits, per_loss):
            for key in (f"teacher:{item['participant']['identity']}", f"prompt:{item['prompt_type']}"):
                metric = breakdown.setdefault(key, {'count': 0, 'correct': 0, 'loss_sum': 0.0})
                metric['count'] += 1
                metric['correct'] += hit
                metric['loss_sum'] += row_loss
    return {'effective_samples': count, 'cross_entropy': loss_sum / count if count else None,
            'balanced_cross_entropy': balanced_loss_sum / count if count else None,
            'top1_accuracy': correct / count if count else None, 'optimizer_updates': updates,
            'breakdown': {key: {'count': v['count'], 'top1_accuracy': v['correct'] / v['count'],
                                 'cross_entropy': v['loss_sum'] / v['count']} for key, v in breakdown.items()}}


def train_imitation(dataset_path, *, init_from=None, resume=None, output='./models/imitation',
                     epochs=5, batch_size=64, learning_rate=1e-5, device='auto', seed=73,
                     model_prefix=None, save_every=1, cpu_threads=1):
    """BC has its own optimizer, identity, budgets and checkpoints; never invokes PPO."""
    from checkpoint_utils import (CHECKPOINT_FORMAT_VERSION, canonical_model_state_dict,
                                  generate_model_id, load_training_checkpoint, get_training_lineage, checkpoint_stem)
    from galatea_net import GalateaNet
    from model_artifacts import write_checkpoint_artifact_manifest
    from protocol_schema import apply_current_protocol_metadata, get_current_protocol_metadata
    from training_lock import TrainerProcessLock
    from training_validation import validate_model_prefix
    from torch.utils.tensorboard import SummaryWriter
    if bool(init_from) == bool(resume):
        raise ValueError('choose exactly one: init_from weights or resume imitation checkpoint')
    if type(epochs) is not int or not 1 <= epochs <= 10000 or type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise ValueError('invalid imitation epoch/batch budget')
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.01 or not 1 <= save_every <= epochs:
        raise ValueError('invalid imitation learning rate/save frequency')
    if type(seed) is not int or not 0 <= seed < 2**32 or not 1 <= cpu_threads <= 64:
        raise ValueError('invalid imitation seed/CPU thread count')
    if model_prefix is not None:
        model_prefix = validate_model_prefix(model_prefix)
    if device not in ('auto', 'cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('requested imitation device is unavailable')
    device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
    with TrainerProcessLock() as lock:
        torch.set_num_threads(cpu_threads)
        torch.manual_seed(seed)
        dataset = PreparedDataset(dataset_path)
        # Prepared inputs are portable across hosts/OSs. Core/CDB/Lua fingerprints
        # describe collection, not a requirement to start a Core while doing BC.
        if dataset.manifest['identity'].get('protocol') != get_current_protocol_metadata():
            raise ValueError('prepared input vocabulary/semantic protocol differs from this model runtime')
        if not any(e['split'] == 'train' and e['effective_count'] for e, _, _ in dataset.entries):
            raise ValueError('dataset has no training examples with an actual choice')
        checkpoint = load_training_checkpoint(resume or init_from, map_location='cpu')
        source_lineage = get_training_lineage(checkpoint)
        if resume:
            if checkpoint.get('training_stage') != 'imitation':
                raise ValueError('imitation resume requires an imitation checkpoint, not a PPO optimizer')
            if checkpoint['imitation']['dataset_sha256'] != dataset.digest:
                raise ValueError('imitation resume dataset changed; use explicit init-from for a new experiment')
            previous = checkpoint['imitation']['config']
            current = {'batch_size': batch_size, 'learning_rate': learning_rate, 'seed': seed}
            if previous != current:
                raise ValueError('resume must keep batch size, learning rate and seed; use init-from to branch')
        config = apply_current_protocol_metadata(checkpoint['net_config'])
        network = GalateaNet(config)
        network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        parameters = configure_policy_learning(network)
        network.to(device)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())
        start_epoch, updates, examples = 0, 0, 0
        if resume:
            run_id, model_id, model_prefix = checkpoint['run_id'], checkpoint['model_id'], checkpoint['model_prefix']
            root = Path(resume).resolve().parent
            start_epoch, updates, examples = checkpoint['iteration'], checkpoint['train_step'], checkpoint['global_step']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'rng_state' in checkpoint['imitation']:
                torch.set_rng_state(checkpoint['imitation']['rng_state'])
            if device.type == 'cuda' and checkpoint['imitation'].get('cuda_rng_state'):
                torch.cuda.set_rng_state_all(checkpoint['imitation']['cuda_rng_state'])
            parent = checkpoint['imitation']['parent']
            epoch_offset = source_lineage['bc_epochs'] - start_epoch
            from checkpoint_utils import inspect_training_checkpoint
            for existing in root.glob('*.pth'):
                existing_meta = inspect_training_checkpoint(existing)
                if existing_meta['model_id'] == model_id and existing_meta['iteration'] > start_epoch:
                    raise ValueError('newer BC checkpoints already exist; resume the latest or init-from to branch')
        else:
            model_prefix = model_prefix or source_lineage['base_prefix']
            model_id, run_id = generate_model_id(), uuid.uuid4().hex
            root = Path(output).resolve() / run_id
            root.mkdir(parents=True, exist_ok=False)
            parent = {'checkpoint': str(Path(init_from).resolve()), 'sha256': file_sha256(init_from),
                      'model_id': checkpoint['model_id'], 'iteration': checkpoint['iteration'],
                      'training_lineage': source_lineage, 'training_stage': checkpoint.get('training_stage', 'ppo')}
            epoch_offset = source_lineage['bc_epochs']
        lineage = dict(source_lineage, base_prefix=(source_lineage['base_prefix'] if resume else model_prefix))
        previous_selection = checkpoint.get('imitation', {}).get('selection') if resume else None
        del checkpoint
        lock.set_run_id(run_id)
        writer = SummaryWriter(log_dir=str(Path('runs') / f'imitation_{run_id}'),
                               purge_step=start_epoch + 1 if resume else None)
        attempt_id = uuid.uuid4().hex[:8]
        history, saved = [], []
        try:
            # Test examples are never used for checkpoint selection or epoch monitoring.
            baseline = _epoch(network, dataset, split='validation', batch_size=batch_size, epoch=0,
                              seed=seed, device=device)
            if not (root / f'baseline_{start_epoch}.json').exists():
                write_json(root / f'baseline_{start_epoch}.json', baseline)
            best = baseline['balanced_cross_entropy'] if baseline['effective_samples'] else float('inf')
            best_checkpoint = str(Path(resume or init_from).resolve()) if baseline['effective_samples'] else None
            if (previous_selection and baseline['effective_samples']
                    and previous_selection.get('balanced_cross_entropy') is not None
                    and previous_selection['balanced_cross_entropy'] <= best):
                best = previous_selection['balanced_cross_entropy']
                best_checkpoint = previous_selection['checkpoint']
            for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
                train = _epoch(network, dataset, split='train', batch_size=batch_size, epoch=epoch,
                               seed=seed, device=device, optimizer=optimizer, scaler=scaler)
                validation = _epoch(network, dataset, split='validation', batch_size=batch_size,
                                    epoch=epoch, seed=seed, device=device)
                updates += train['optimizer_updates']
                examples += train['effective_samples']
                metric = validation['balanced_cross_entropy']
                record = {'epoch': epoch, 'train': train, 'validation': validation,
                          'new_best_validation': metric is not None and metric < best}
                if metric is not None:
                    best = min(best, metric)
                history.append(record)
                epoch_report = root / f'epoch_{epoch:05d}.json'
                if epoch_report.exists():
                    epoch_report = root / f'epoch_{epoch:05d}_{attempt_id}.json'
                write_json(epoch_report, record)
                for split, values in (('Train', train), ('Validation', validation)):
                    for name in ('cross_entropy', 'balanced_cross_entropy', 'top1_accuracy', 'effective_samples'):
                        if values[name] is not None:
                            writer.add_scalar(f'Imitation_{split}/{name}', values[name], epoch)
                writer.add_scalar('Imitation_Train/learning_rate', learning_rate, epoch)
                writer.add_scalar('Imitation_Train/optimizer_updates', updates, epoch)
                print(f"[Imitation/Train] Epoch {epoch}: CE={train['cross_entropy']:.5f}, "
                      f"Top1={train['top1_accuracy']:.3f}, validation={metric}", flush=True)
                if epoch % save_every == 0 or epoch == start_epoch + epochs or record['new_best_validation']:
                    lineage['bc_epochs'] = epoch_offset + epoch
                    path = root / (checkpoint_stem({'model_prefix': model_prefix, 'iteration': epoch,
                        'training_stage': 'imitation', 'training_lineage': lineage}) + '.pth')
                    if record['new_best_validation']:
                        best_checkpoint = str(path)
                    payload = {'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
                        **get_current_protocol_metadata(), 'net_config': config,
                        'model_id': model_id, 'model_prefix': model_prefix, 'run_id': run_id,
                        'iteration': epoch, 'train_step': updates, 'global_step': examples,
                        'training_stage': 'imitation', 'optimizer_state_dict': optimizer.state_dict(),
                        'training_lineage': dict(lineage),
                        'scaler_state_dict': scaler.state_dict(),
                        'model_state_dict': canonical_model_state_dict(network, to_cpu=True),
                        'imitation': {'workflow_version': IMITATION_WORKFLOW_VERSION, 'parent': parent,
                            'dataset': str(Path(dataset_path).resolve()), 'dataset_sha256': dataset.digest,
                            'config': {'batch_size': batch_size, 'learning_rate': learning_rate, 'seed': seed},
                            'selection': {'balanced_cross_entropy': best if baseline['effective_samples'] else None,
                                          'checkpoint': best_checkpoint},
                            'rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                            'counter_units': {'iteration': 'imitation_epoch', 'global_step': 'expert_examples'},
                            'frozen_prefixes': list(FROZEN_PREFIXES)}}
                    if path.exists():
                        raise FileExistsError('imitation epoch checkpoint already exists')
                    temporary = path.with_suffix('.pth.tmp')
                    torch.save(payload, temporary)
                    temporary.rename(path)
                    write_checkpoint_artifact_manifest(str(path), epoch, model_id=model_id, model_prefix=model_prefix,
                        training_stage='imitation', training_lineage=lineage)
                    saved.append(str(path))
                    del payload
            result = {'format_version': 1, 'workflow': 'imitation', 'run_id': run_id, 'model_id': model_id,
                      'dataset_sha256': dataset.digest, 'parent': parent, 'epochs_completed': start_epoch + epochs,
                      'checkpoints': saved, 'history': history, 'test_used_for_selection': False,
                      'best_validation_checkpoint': best_checkpoint,
                      'selection_metric': 'balanced_cross_entropy' if baseline['effective_samples'] else None,
                      'ppo_started': False, 'human_online_started': False}
            write_json(root / f'result_{start_epoch + epochs}.json', result)
            return result
        finally:
            writer.close()
            optimizer.zero_grad(set_to_none=True)


def evaluate_imitation(dataset_path, checkpoint, output, *, baseline=None, split='validation',
                        executors=(), games_per_seat=5, seed=None, runtime=None, mono=None,
                        policy_mode='training', temperature=0.8, timeout=600, record_replay=False, record_evaluations=None):
    """Explicit fixed-configuration BO1 comparison, never automatic online learning."""
    from checkpoint_utils import inspect_training_checkpoint
    from integrations.windbot.runtime import executor_catalog, load_deck
    from integrations.windbot.session import run_controlled_duel
    record_evaluations = bool(record_replay) if record_evaluations is None else bool(record_evaluations)
    if record_evaluations and not record_replay:
        raise ValueError('prediction curves require replay recording')
    if split not in ('validation', 'test') or not 1 <= games_per_seat <= 1000:
        raise ValueError('evaluation requires a held-out split and 1..1000 games per seat')
    manifest = read_json(dataset_path)
    if manifest.get('workflow') != 'imitation':
        raise ValueError('evaluation requires an imitation dataset')
    from trajectory_dataset import deck_identity
    cases = {}
    catalog = executor_catalog()
    if executors:
        for name in sorted(set(executors)):
            if name not in catalog or not catalog[name]['deck_available']:
                raise ValueError('evaluation executor/build is unavailable')
            deck = load_deck(catalog[name]['deck_file'])
            cases[(name, deck_identity(deck))] = deck
    else:
        for entry in manifest['entries']:
            if entry['split'] != split:
                continue
            cache = Path(entry['cache'])
            if not cache.is_absolute():
                cache = Path(dataset_path).resolve().parent / cache
            if file_sha256(cache) != entry['cache_sha256']:
                raise ValueError('evaluation source manifest changed')
            source = read_json(cache)['source']
            for seat, participant in source['participants'].items():
                name = participant.get('executor')
                if participant['kind'] == 'windbot' and name:
                    deck = source['decks'][seat]
                    cases[(name, deck_identity(deck))] = deck
    if not cases:
        raise ValueError('no held-out WindBot executors; explicitly select a regression opponent or prepare a holdout')
    for name, _ in cases:
        if name not in catalog or not catalog[name]['deck_available']:
            raise ValueError('evaluation executor/build is unavailable')
    randomized_seed = seed is None
    seed = resolve_core_seed(seed, len(cases) * games_per_seat)
    models = {'candidate': str(Path(checkpoint).resolve())}
    if baseline:
        models['baseline'] = str(Path(baseline).resolve())
    model_metadata = {role: inspect_training_checkpoint(path) for role, path in models.items()}
    root = Path(output).resolve()
    if (root / 'evaluation_plan.json').exists():
        raise FileExistsError('evaluation plan already exists; choose a new output directory')
    rows = []
    for i, ((name, deck_id), deck) in enumerate(sorted(cases.items())):
        deck_path = root / 'decks' / f'{deck_id}.ydk'
        if not deck_path.exists():
            deck_path.parent.mkdir(parents=True, exist_ok=True)
            with deck_path.open('x', encoding='utf-8') as stream:
                for part, marker in (('main', '#main'), ('extra', '#extra'), ('side', '!side')):
                    stream.write(marker + '\n' + ''.join(f'{code}\n' for code in deck.get(part, [])))
        if deck_identity(load_deck(deck_path)) != deck_id:
            raise ValueError('frozen evaluation build changed')
        rows.extend({'executor': name, 'seed': seed + i * games_per_seat + game, 'model_seat': seat,
                     'deck_file': str(deck_path), 'deck_sha256': file_sha256(deck_path)}
                    for game in range(games_per_seat) for seat in (0, 1))
    write_json(root / 'evaluation_plan.json', {'format_version': 1, 'split': split, 'rows': rows,
        'models': model_metadata, 'model_paths': models, 'policy_mode': policy_mode, 'temperature': temperature,
        'dataset_sha256': file_sha256(dataset_path), 'dataset_path': str(Path(dataset_path).resolve()),
        'seed': seed, 'randomized_seed': randomized_seed, 'bot_randomness_controlled': False,
        'holdout_scope': 'explicit_regression_override' if executors else manifest['split_mode'],
        'record_replay': bool(record_replay), 'record_evaluations': record_evaluations,
        'mirror_builds': True, 'training_started': False})
    results = {}
    for role, model in models.items():
        wins = losses = draws = aborted = 0
        by_executor = {}
        for index, row in enumerate(rows):
            result = run_controlled_duel(executor=row['executor'], model=model, windbot_seat=1-row['model_seat'],
                runtime=runtime, mono=mono, seed=row['seed'], timeout=timeout,
                policy_mode=policy_mode, temperature=temperature, verify=False,
                native_deck=row['deck_file'], capture_root=str(root / 'captures'), log_root=str(root / 'bot_logs'),
                windbot_deck=row['deck_file'],
                record_replay=record_replay, record_evaluations=record_evaluations,
                replay_root=str(root / 'replays' / role), game_index=index + 1,
                session_purpose='imitation_evaluation')
            counter = by_executor.setdefault(row['executor'], Counter())
            if not result['core_terminal']:
                aborted += 1
                counter['aborted'] += 1
            elif result['winner'] == row['model_seat']:
                wins += 1
                counter['wins'] += 1
            elif result['winner'] == 1-row['model_seat']:
                losses += 1
                counter['losses'] += 1
            else:
                draws += 1
                counter['draws'] += 1
            write_json(root / role / f'game_{index + 1:05d}.json', dict(result, model_seat=row['model_seat']))
        completed = wins + losses + draws
        if completed:
            p, z = wins / completed, 1.96
            center = (p + z*z/(2*completed)) / (1+z*z/completed)
            radius = z*math.sqrt(p*(1-p)/completed+z*z/(4*completed**2))/(1+z*z/completed)
            interval = [max(0.0, center-radius), min(1.0, center+radius)]
        else:
            interval = None
        results[role] = {'wins': wins, 'losses': losses, 'draws': draws, 'aborted': aborted,
                         'win_rate_completed': wins / completed if completed else None,
                         'wilson95_descriptive': interval, 'attempted': len(rows),
                         'by_executor': {key: dict(value) for key, value in by_executor.items()}}
    report = {'format_version': 1, 'results': results, 'training_started': False,
              'test_used_for_selection': False, 'note': 'paired seats share seeds; WindBot randomness is not controlled'}
    write_json(root / 'evaluation.json', report)
    return report
