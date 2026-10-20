"""Explicit controlled BO1 runner; never imported by ordinary PPO/Arena paths."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

from integrations.duel_session import (
    SessionGate, ResponseSubmission, finite_timeout, WINDBOT_SELECTION_LABEL_V1,
)
from integrations.windbot.bridge import DECISIONS, YGOProPeer, prompt_actor, visible_message
from integrations.windbot.runtime import ROOT, executor_catalog, executable_sha256, launch_bot, load_deck


def run_controlled_duel(*, executor, runtime=None, mono=None, opponent_executor=None,
        windbot_seat=1, model=None, native_deck=None, seed=73, timeout=600,
        response_timeout=30, max_decisions=2000, capture_root='./replays/link_captures',
        log_root='./system_logs/windbot', verify=False, policy_mode='training', temperature=0.8,
        native_deck_source=None, deck_dir='./decks', record_replay=False,
        record_evaluations=False, game_index=1, replay_root='./ai_thoughts'):
    """Drive one public Core and optional external clients, with exact full capture.

    External packets and masked query refreshes are derived transports, not raw
    Core chunks. Bot failures/Retry/timeouts seal a diagnostic prefix; no opponent
    replacement or fabricated terminal result is allowed. Expert actions are
    not PPO samples; independent replay does not imply all labels are usable.
    """
    from ai_bot import AiBot
    from core_trajectory import CoreTrajectoryInterpreter, runtime_asset_identity
    from galatea_env import GalateaEnv, get_callback_environment
    from gamestate import DuelState
    from core_message_protocol import is_empty_chain_prompt
    from trajectory_ingest import LinkCaptureWriter, prepare_canonical_decision_snapshot
    from model_versus import resolve_arena_policy, select_arena_action_index
    import numpy as np
    import rule_bot
    from action_candidates import MODEL_ACTION_MSGS

    timeout = finite_timeout(timeout)
    response_timeout = finite_timeout(response_timeout)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    if type(max_decisions) is not int or not 1 <= max_decisions <= 100000:
        raise ValueError('max_decisions must be 1..100000')
    if type(windbot_seat) is not int or windbot_seat not in (0, 1):
        raise ValueError('windbot_seat must be physical 0/1')
    if native_deck and native_deck_source:
        raise ValueError('native-deck and native-deck-source are mutually exclusive')
    if record_evaluations and (not record_replay or not model):
        raise ValueError('state evaluations require replay recording and a native model')
    if opponent_executor and (model or native_deck or native_deck_source):
        raise ValueError('two WindBots cannot be combined with a native model/deck')
    policy_mode, temperature = resolve_arena_policy(policy_mode, temperature)
    catalog = executor_catalog()
    if executor not in catalog or (opponent_executor and opponent_executor not in catalog):
        raise ValueError('unknown executor; use --inspect to list exact upstream names')
    executable_hash = executable_sha256(runtime, mono)  # Explicit dependency check, before Core creation.
    upstream = json.loads((ROOT / 'UPSTREAM.json').read_text(encoding='utf-8'))
    gate = SessionGate()
    peers = {}
    participants = {}
    deck_paths = {}
    bot_names = {windbot_seat: executor}
    if opponent_executor:
        bot_names[1 - windbot_seat] = opponent_executor
    if native_deck_source:
        import deck_utils
        pick = deck_utils.select_arena_deck(deck_utils.discover_arena_deck_catalog(deck_dir),
            native_deck_source, rng=random.Random(seed))
        native_deck = pick.deck_path
    for actor in (0, 1):
        deck_paths[actor] = (catalog[bot_names[actor]]['deck_file'] if actor in bot_names
                            else native_deck or catalog[executor]['deck_file'])
    decks = {str(actor): load_deck(deck_paths[actor]) for actor in (0, 1)}
    for actor in bot_names:
        deck_hash = hashlib.sha256(json.dumps(decks[str(actor)], sort_keys=True).encode()).hexdigest()
        identity = f'windbot:{executable_hash}:{bot_names[actor]}'
        participants[str(actor)] = {'kind': 'windbot', 'identity': identity,
            'executor': bot_names[actor], 'upstream_commit': upstream['commit'],
            'executable_sha256': executable_hash, 'deck_sha256': deck_hash}
    native_actor = None
    if not opponent_executor:
        native_actor = 1 - windbot_seat
        if model:
            bot = AiBot(device='cpu', initialize_network=False)
            if not bot.load_model(model):
                raise ValueError('native model checkpoint could not be loaded; no RuleBot substitution')
            participants[str(native_actor)] = {'kind': 'model', 'identity': f"model:{bot.model_metadata['model_id']}"}
        else:
            bot = AiBot.__new__(AiBot)
            participants[str(native_actor)] = {'kind': 'rule', 'identity': 'rule:galatea_builtin'}
    else:
        bot = None
    previous = get_callback_environment()
    env = capture = logger = evaluator = None
    begin = time.monotonic()
    random_state = random.getstate()
    random.seed(seed)
    rng = np.random.default_rng(seed)
    result = {'session_contract_version': 1, 'session_id': gate.session_id, 'participants': participants,
        'seed': seed, 'starting_player': 0, 'decisions': 0, 'winner': None, 'reason': None,
        'core_terminal': False, 'error': None, 'bot_logs': {}, 'capture_path': None,
        'deck_paths': {str(actor): str(path) for actor, path in deck_paths.items()},
        'native_deck_source': native_deck_source, 'replay_path': None,
        'native_policy': {'policy_mode': policy_mode, 'temperature': temperature} if model else None}
    try:
        print('🏟️ [WindBot/对局] 初始化本局 Core 与脚本；此阶段运行实际对局。')
        env = GalateaEnv()
        # Missing cards never silently become a different deck or placeholder.
        import galatea_env as core_env
        core_env._init_card_cache()
        missing = sorted({code for deck in decks.values() for part in ('main', 'extra')
                          for code in deck[part] if code not in core_env._GLOBAL_CARD_CACHE})
        if missing:
            raise ValueError(f'executor deck contains cards absent from CDB: {missing[:20]}')
        from types import SimpleNamespace
        native_decks = [SimpleNamespace(**decks[str(actor)]) for actor in (0, 1)]
        with ExitStack() as stack:
            for actor in sorted(bot_names):
                peer = YGOProPeer(actor, response_timeout)
                peers[actor] = peer
                stack.callback(peer.close)
                _, log = stack.enter_context(launch_bot(runtime=runtime, mono=mono,
                    executor=bot_names[actor], deck_file=deck_paths[actor], cdb_path=env.cdb_path,
                    port=peer.port, log_root=log_root, room_token=peer.room_token))
                result['bot_logs'][str(actor)] = log
                peer.accept(decks[str(actor)])
            raw = env.reset(*native_decks, seed=seed)
            reset = {**deepcopy(env.last_reset_metadata), 'initial_position': 0}
            capture = LinkCaptureWriter(visibility='full_core', reset=reset, decks=decks,
                assets=runtime_asset_identity(env), root=capture_root,
                player_ids={seat: value['identity'] for seat, value in participants.items()},
                session_metadata={'format_version': 1, 'session_id': gate.session_id, 'participants': participants})
            result['capture_path'] = capture.path
            interpreter = CoreTrajectoryInterpreter(DuelState(decks['0']['main'], decks['0']['extra'],
                decks['1']['main'], decks['1']['extra']))
            if record_replay:
                from thought_logger import AIThoughtLogger
                names = [f'WindBot ({bot_names[actor]})' if actor in bot_names else
                         Path(model).stem if model else 'RuleBot' for actor in (0, 1)]
                logger = AIThoughtLogger(*names)
                logger.start_recording()
                logger.set_decklists(decks['0']['main'], decks['0']['extra'],
                    decks['1']['main'], decks['1']['extra'],
                    p0_name=Path(deck_paths[0]).stem, p1_name=Path(deck_paths[1]).stem)
                policy = {'policy_mode': policy_mode, 'temperature': temperature}
                replay_models = {seat: {'kind': info['kind']} for seat, info in participants.items()}
                if model:
                    replay_models[str(native_actor)].update(bot.model_metadata)
                logger.set_recording_metadata({'source': 'windbot_controlled',
                    'session_id': gate.session_id, 'participants': participants,
                    'capture_path': capture.path, 'models': replay_models, **policy,
                    **({'evaluation_sampling': 'recorded_frames'} if record_evaluations else {})})
                if record_evaluations:
                    from arena_evaluation import ArenaStateEvaluator
                    model_bots = [None, None]
                    model_bots[native_actor] = bot
                    evaluator = ArenaStateEvaluator(model_bots, policy)
            for peer in peers.values():
                peer.start(decks)
            for _ in range(100000):
                if time.monotonic() - begin > timeout:
                    raise TimeoutError('controlled duel time limit; no terminal outcome fabricated')
                if not raw:
                    raise RuntimeError('Core stopped before a real terminal message')
                capture.record_chunk(raw)
                if capture._writer.stopped:
                    raise RuntimeError('full capture stopped; controlled session cannot claim faithful recording')
                messages = interpreter.parse_chunk(raw)
                if b''.join(messages) != bytes(raw):
                    raise ValueError('unparsed authoritative Core bytes')
                for index, message in enumerate(messages):
                    kind = message[0]
                    interpreter.apply_message(message)
                    if kind == 1:
                        raise ValueError('Core Retry rejected the submitted response; abort without substitution')
                    if kind in DECISIONS:
                        if index != len(messages) - 1:
                            raise ValueError('unconsumed messages after decision prompt')
                        if result['decisions'] >= max_decisions:
                            raise TimeoutError('controlled duel decision limit')
                        actor = prompt_actor(message)
                        decision_snapshot = None
                        selected = probabilities = None
                        request = gate.request(actor, kind)
                        for peer in peers.values():
                            peer.check_idle()
                        if is_empty_chain_prompt(kind, message[1:]):
                            response = -1
                        elif actor in peers:
                            response = peers[actor].respond(message, env)
                        elif model and kind in MODEL_ACTION_MSGS:
                            import torch
                            snapshot = prepare_canonical_decision_snapshot(interpreter, env, message, rng)
                            decision_snapshot = snapshot
                            observation = bot.encoder.encode(snapshot, player_id=actor)
                            with torch.no_grad():
                                logits, _, _ = bot.net(observation)
                            selected, _ = select_arena_action_index(logits[0], set(), set(),
                                policy_mode=policy_mode, temperature=temperature)
                            if logger:
                                probabilities = torch.softmax(logits[0] / (temperature or 1.0), dim=-1).cpu()
                            response = bot._pack_response(snapshot.valid_actions[selected], kind, message[1:])
                        else:
                            rule_bot.sync_valid_actions(interpreter.brain.current_valid_actions)
                            response = rule_bot.get_rule_decision(actor, kind, message, interpreter.brain)
                        raw_response = gate.submit(ResponseSubmission(request.session_id, request.request_id, actor, response))
                        env.send_action(raw_response)
                        capture.record_response(raw_response, actor)
                        if not is_empty_chain_prompt(kind, message[1:]):
                            if logger:
                                from core_trajectory import matching_action_indices
                                from arena_evaluation import record_frame_evaluations
                                # Recording never samples a macro pool or inserts the expert
                                # answer into policy candidates/the learning contract.
                                replay_snapshot = decision_snapshot or interpreter.snapshot(env)
                                previous_frames = len(logger.thoughts)
                                if selected is not None:
                                    logger.log_decision(interpreter.brain.turn, interpreter.brain.phase,
                                        replay_snapshot, probabilities, selected, player_id=actor,
                                        msg_type=kind, agent_name=names[actor])
                                else:
                                    matches = matching_action_indices(replay_snapshot, raw_response, kind, message[1:],
                                        response_adapter=WINDBOT_SELECTION_LABEL_V1 if actor in peers else None)
                                    logger.log_external_decision(interpreter.brain.turn, interpreter.brain.phase,
                                        replay_snapshot, kind, raw_response, actor, agent_name=names[actor],
                                        chosen_index=matches[0] if len(matches) == 1 else None,
                                        source='windbot' if actor in peers else 'rule', response_payload=message[1:])
                                record_frame_evaluations(logger, evaluator, replay_snapshot,
                                    interpreter.brain.transition_recorder.next_sequence_id,
                                    previous_frame_count=previous_frames, snapshot_source='decision_query')
                            interpreter.brain.begin_transition_event(actor, kind, raw_response)
                            result['decisions'] += 1
                    else:
                        if logger:
                            from thought_logger import REPLAY_EVENT_MSGS
                            if kind in REPLAY_EVENT_MSGS:
                                from arena_evaluation import record_frame_evaluations
                                event_snapshot = interpreter.snapshot()
                                previous_frames = len(logger.thoughts)
                                logger.log_core_event(interpreter.brain.turn, interpreter.brain.phase,
                                    event_snapshot, kind, message[1:])
                                record_frame_evaluations(logger, evaluator, event_snapshot,
                                    interpreter.brain.transition_recorder.completed_event_count,
                                    previous_frame_count=previous_frames)
                        for peer in peers.values():
                            masked = visible_message(message, peer.seat)
                            if masked is not None:
                                peer.game_message(masked)
                        if kind == 5:
                            if index != len(messages) - 1:
                                raise ValueError('unconsumed bytes after terminal')
                            result.update(winner=message[1], reason=message[2], core_terminal=True)
                if result['core_terminal']:
                    for peer in peers.values():
                        peer.send(0x16)
                    break
                raw = env.step()
            else:
                raise RuntimeError('controlled duel event limit')
    except Exception as error:
        result['error'] = str(error)
        result['core_terminal'] = False
        result['winner'] = result['reason'] = None
        if capture:
            capture._writer.flag(f'controlled_session_failed: {error}')
    finally:
        gate.close()
        if capture:
            capture.finish(winner=result['winner'] if result['core_terminal'] else -1,
                reason=result['reason'] if result['core_terminal'] else -1, core_terminal=result['core_terminal'])
        if logger:
            try:
                result['replay_path'] = logger.save(result['winner'] if result['core_terminal'] else -1,
                    game_index, str(result['reason']) if result['core_terminal'] else result['error'] or 'aborted',
                    core_terminal=result['core_terminal'], root=replay_root,
                    terminal_event_count=interpreter.brain.transition_recorder.completed_event_count)
            except Exception as error:
                result['replay_error'] = str(error)  # Capture/lifecycle cleanup must still complete.
        if env:
            result['native_query_errors'] = env.query_card_error_count
            env._close_duel()
            env.cdb.close()
            env.script_buffers.clear()
        if previous is not None:
            previous.install_callbacks()
        random.setstate(random_state)
        result['seconds'] = time.monotonic() - begin
    if verify and result['core_terminal']:
        from trajectory_ingest import import_link_capture
        print('🔎 [WindBot/校验] 开始离线转换与独立重放；将顺序初始化两次 Core 与脚本，不新增对局或训练。')
        verification_begin = time.monotonic()
        result['verification'] = import_link_capture(result['capture_path'])
        result['verification_seconds'] = time.monotonic() - verification_begin
    return result
