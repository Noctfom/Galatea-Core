# 本文件提供 Link 原始数据采集合同及 YRP/Link 到唯一 Core 轨迹解释器的离线入口
import hashlib
import random
import tempfile
import time
from collections import deque
from copy import deepcopy

from core_message_protocol import CORE_MESSAGE_PROTOCOL, is_empty_chain_prompt
from core_trajectory import (
    CoreTrajectoryInterpreter, CoreTrajectoryWriter, MAX_RECORDS, MAX_TRAJECTORY_BYTES,
    MAX_RECORD_BYTES, _iter_records, _decode_hex, canonical_response, decode_response,
    encode_response, encoder_decision_messages, framework_version,
    replay_core_trajectory, reset_core_from_metadata, runtime_asset_identity, validate_reset_metadata,
)
from yrp_ingest import read_yrp

LINK_CAPTURE_FORMAT_VERSION = 1
DEFAULT_INGEST_ROOT = './replays/imported_trajectories'
DEFAULT_LINK_ROOT = './replays/link_captures'


def validate_player_ids(player_ids):
    """接收来源命名空间内的稳定匿名玩家 ID，不把显示昵称当成可靠身份"""
    if player_ids is None:
        return {'0': None, '1': None}
    if not isinstance(player_ids, dict) or set(player_ids) != {'0', '1'}:
        raise ValueError('player_ids must contain physical seats 0/1')
    for value in player_ids.values():
        if value is not None and (not isinstance(value, str) or not 1 <= len(value) <= 256
                                  or any(ord(char) < 32 for char in value)):
            raise ValueError('invalid stable player identity')
    return dict(player_ids)


def _validate_link_header(header):
    """区分全量服务端捕获和单方视角，缺失信息不得伪装为已知"""
    if header.get('capture_format_version') != LINK_CAPTURE_FORMAT_VERSION:
        raise ValueError('unsupported Link capture format')
    if header.get('core_message_protocol') != CORE_MESSAGE_PROTOCOL:
        raise ValueError('Link Core message protocol mismatch')
    if header.get('visibility') not in ('full_core', 'player_view'):
        raise ValueError('invalid Link visibility')
    perspective = header.get('perspective')
    if perspective is not None and (type(perspective) is not int or perspective not in (0, 1)):
        raise ValueError('invalid Link physical perspective')
    if header['visibility'] == 'player_view' and perspective is None:
        raise ValueError('player-view Link capture requires a perspective')
    validate_player_ids(header.get('player_ids'))
    if header['visibility'] == 'full_core':
        validate_reset_metadata(header.get('reset'))
        decks = header.get('decks')
        if not isinstance(decks, dict) or set(decks) != {'0', '1'}:
            raise ValueError('full Link capture requires both complete decks')
        for seat in ('0', '1'):
            if not isinstance(decks[seat], dict):
                raise ValueError('invalid Link stable deck')
            for part in ('main', 'extra', 'side'):
                codes = decks[seat].get(part, [])
                if (not isinstance(codes, list) or len(codes) > 256
                        or any(type(code) is not int or not 0 < code < 2**32 for code in codes)):
                    raise ValueError('invalid Link stable deck cards')
                if part != 'side' and sorted(codes) != sorted(header['reset']['players'][seat][part]):
                    raise ValueError('Link stable deck and injection order disagree')
        if not isinstance(header.get('assets'), dict):
            raise ValueError('full Link capture requires original asset identities')


class LinkCaptureWriter:
    """供 Link 调用的有界采集 SDK；记录原始数据而不运行策略或生成 PPO 样本"""

    def __init__(self, *, visibility='player_view', perspective=None, player_ids=None,
                 reset=None, decks=None, assets=None, root=DEFAULT_LINK_ROOT):
        """开局建立捕获文件；普通在线视角可明确缺少种子、对方构筑及资产"""
        header = {'source': 'link_capture', 'capture_format_version': LINK_CAPTURE_FORMAT_VERSION,
                  'core_message_protocol': CORE_MESSAGE_PROTOCOL, 'framework_version': framework_version(),
                  'visibility': visibility, 'perspective': perspective,
                  'player_ids': validate_player_ids(player_ids), 'reset': deepcopy(reset),
                  'decks': deepcopy(decks), 'assets': deepcopy(assets)}
        _validate_link_header(header)
        self._writer = CoreTrajectoryWriter(header, root, file_kind='link')
        self.path = self._writer.path

    def record_chunk(self, raw):
        """只记录原始 Core 消息块，不接受含网络包头的字节或重新拼接的展示消息"""
        if not isinstance(raw, (bytes, bytearray, memoryview)) or not 0 < len(raw) <= 65536:
            raise ValueError('invalid Link Core chunk bytes')
        data = bytes(raw)
        if not 0 < len(data) <= 65536:
            raise ValueError('invalid Link Core chunk length')
        self._writer.write('chunk', hex=data.hex())

    def record_response(self, response, actor):
        """记录实际已提交响应及物理座位，不保存模型建议或尚未执行的动作"""
        if type(actor) is not int or actor not in (0, 1):
            raise ValueError('invalid Link response actor')
        self._writer.write('response', response=encode_response(response), actor=actor)

    def finish(self, *, winner=-1, reason=-1, core_terminal=False):
        """封存捕获摘要；异常/断线保留为诊断数据，不伪造真实终局"""
        self._writer.finish(winner=winner, reason=reason, core_terminal=core_terminal)


def inspect_link_capture(path):
    """在调用原生 Core 前校验 Link 捕获的边界、顺序、摘要与来源合同"""
    digest = hashlib.sha256()
    header = footer = None
    chunks = responses = count = 0
    for record, raw in _iter_records(path):
        if footer is not None or type(record.get('seq')) is not int or record['seq'] != count:
            raise ValueError('Link sequence/EOF mismatch')
        kind = record.get('kind')
        if count == 0:
            if kind != 'header':
                raise ValueError('missing Link header')
            _validate_link_header(record)
            header = record
        elif kind == 'chunk':
            if not _decode_hex(record['hex'], 65536):
                raise ValueError('empty Link Core chunk')
            chunks += 1
        elif kind == 'response':
            decode_response(record['response'])
            if type(record['actor']) is not int or record['actor'] not in (0, 1):
                raise ValueError('invalid Link response actor')
            responses += 1
        elif kind == 'footer':
            if (record.get('sha256') != digest.hexdigest() or record.get('record_count') != count
                    or any(type(record.get(key)) is not bool for key in ('core_terminal', 'truncated', 'mapping_complete'))
                    or type(record.get('winner')) is not int or record['winner'] not in (-1, 0, 1, 2)
                    or type(record.get('reason')) is not int or not -4 <= record['reason'] <= 255
                    or not isinstance(record.get('errors'), list) or len(record['errors']) > 32
                    or any(not isinstance(value, str) or len(value) > 256 for value in record['errors'])):
                raise ValueError('invalid Link checksum/footer')
            footer = record
        else:
            raise ValueError('unsupported Link capture record')
        if footer is None:
            digest.update(raw)
            count += 1
    if header is None or footer is None:
        raise ValueError('incomplete Link capture')
    return {'header': header, 'footer': footer, 'integrity_valid': True,
            'chunk_count': chunks, 'response_count': responses, 'behavior_clone_eligible': False}


def _freeze_source(path):
    """复制有界输入到匿名临时文件，避免校验后原路径被替换"""
    frozen = tempfile.TemporaryFile(mode='w+b')
    try:
        size = 0
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                size += len(chunk)
                if size > MAX_TRAJECTORY_BYTES + MAX_RECORD_BYTES:
                    raise ValueError('ingress compressed size limit exceeded')
                frozen.write(chunk)
        frozen.seek(0)
        return frozen
    except Exception:
        frozen.close()
        raise


def _ensure_known_cards(reset, decks):
    """原生执行前检查 CDB 与 V4 词表覆盖，未知新卡暂不转换"""
    from card_vocab import get_default_card_vocabulary
    import galatea_env
    from collections import Counter
    validate_reset_metadata(reset)
    vocabulary = get_default_card_vocabulary()
    galatea_env._init_card_cache()
    if not isinstance(decks, dict) or set(decks) != {'0', '1'}:
        raise ValueError('missing complete stable decks')
    unknown = []
    for seat in ('0', '1'):
        for part in ('main', 'extra'):
            if Counter(decks[seat][part]) != Counter(reset['players'][seat][part]):
                raise ValueError('stable decks and injection cards disagree')
            for code in reset['players'][seat][part]:
                if code not in galatea_env._GLOBAL_CARD_CACHE or not vocabulary.contains(code):
                    unknown.append(code)
    if unknown:
        raise ValueError(f'cards absent from local CDB/V4 vocabulary: {sorted(set(unknown))[:20]}')


def _offline_candidate_snapshot(interpreter, env, msg, rng):
    """复用现有宏候选构建器，以固定均匀先验还原离线合法池，不注入专家答案"""
    import numpy as np
    from action_candidates import MACRO_ACTION_MSGS, build_macro_action_pool
    brain = interpreter.brain
    snapshot = interpreter.snapshot(env)
    if msg[0] in MACRO_ACTION_MSGS:
        base = list(brain.current_valid_actions)
        brain.current_valid_actions = build_macro_action_pool(
            msg[0], msg[1:], brain, base, np.ones(len(base)), rng=rng)
        snapshot = interpreter.snapshot(env)
    return snapshot


def _convert(reset, decks, origin, output_root, *, link_source=None, responses=None,
             expected_assets=None, timeout_seconds=300):
    """用唯一解释器驱动公开 Core，逐条提交原响应并写出可独立重放的规范轨迹"""
    import numpy as np
    from feature_encoder import GalateaEncoder
    from galatea_env import GalateaEnv, get_callback_environment
    from gamestate import DuelState
    if not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 3600:
        raise ValueError('invalid ingress time budget')
    _ensure_known_cards(reset, decks)
    previous = get_callback_environment()
    env = writer = None
    random_state = random.getstate()
    begin = time.monotonic()
    input_count = 0
    try:
        env = GalateaEnv()
        assets = runtime_asset_identity(env)
        if expected_assets is not None and expected_assets != assets:
            raise ValueError('original Link Core/script/CDB/protocol assets do not match local assets')
        raw = reset_core_from_metadata(env, reset)
        header = {'source': origin['kind'], 'visibility': 'omniscient_local_core',
                  'core_message_protocol': CORE_MESSAGE_PROTOCOL, 'framework_version': framework_version(),
                  'assets': assets, 'reset': reset, 'decks': decks, 'ingress': origin}
        writer = CoreTrajectoryWriter(header, output_root)
        if origin['kind'] == 'yrp':
            writer.flag('historical_assets_unknown')
            writer.flag('original_core_output_unavailable')
        interpreter = CoreTrajectoryInterpreter(DuelState(
            decks['0']['main'], decks['0']['extra'], decks['1']['main'], decks['1']['extra']), writer)
        encoder = GalateaEncoder()
        rng = np.random.default_rng(reset['seed'])
        # 宏构建器的现有排序提示可能使用 Python random；局部固定并在 finally 恢复
        random.seed(reset['seed'])
        link_records = iter(_iter_records(link_source)) if link_source is not None else None
        if link_records is not None:
            next(link_records)  # 已完成格式校验，跳过来源头部
        response_iter = iter(responses or [])
        error = None
        try:
            for _ in range(MAX_RECORDS):
                if writer.stopped:
                    raise ValueError('canonical trajectory recording limit reached')
                if time.monotonic() - begin > timeout_seconds:
                    raise ValueError('ingress replay time budget exceeded')
                if not raw:
                    raise ValueError('Core stalled before a real terminal message')
                if link_records is not None:
                    record = next(link_records, ({}, None))[0]
                    if record.get('kind') != 'chunk' or bytes(raw) != _decode_hex(record.get('hex'), 65536):
                        raise ValueError('original Link raw Core output diverged')
                messages = deque(interpreter.parse_chunk(raw))
                if b''.join(messages) != bytes(raw):
                    raise ValueError('unparsed Core bytes during ingress')
                while messages:
                    msg = messages.popleft()
                    interpreter.apply_message(msg)
                    if msg[0] == 1:
                        raise ValueError('Core rejected original response (Retry); no response skipping allowed')
                    if msg[0] == 5:
                        if messages:
                            raise ValueError('unconsumed messages after terminal')
                        break
                    if msg[0] not in encoder_decision_messages():
                        continue
                    if messages:
                        raise ValueError('unconsumed messages after a decision prompt')
                    if link_records is not None:
                        record = next(link_records, ({}, None))[0]
                        if record.get('kind') != 'response' or record.get('actor') != msg[1]:
                            raise ValueError('Link response order/actor mismatch')
                        response = decode_response(record['response'])
                    else:
                        response = next(response_iter, None)
                        if response is None:
                            raise ValueError('YRP responses exhausted before terminal')
                    input_count += 1
                    if is_empty_chain_prompt(msg[0], msg[1:]):
                        if canonical_response(response) != canonical_response(-1):
                            raise ValueError('invalid original empty-chain response')
                        interpreter.record_automatic_response(msg[1:], msg[1])
                        env.send_action(-1)
                    else:
                        snapshot = _offline_candidate_snapshot(interpreter, env, msg, rng)
                        observation = encoder.encode(snapshot, player_id=msg[1])
                        interpreter.record_response(snapshot, response, msg[0], msg[1:], msg[1],
                                                    observation=observation, source=origin['kind'])
                        env.send_action(response)
                        interpreter.brain.begin_transition_event(msg[1], msg[0], response)
                if interpreter.terminal is not None:
                    if link_records is not None:
                        record = next(link_records, ({}, None))[0]
                        if (record.get('kind') != 'footer' or not record['core_terminal']
                                or (record['winner'], record['reason']) != interpreter.terminal):
                            raise ValueError('Link terminal metadata disagrees with real Core output')
                    elif next(response_iter, None) is not None:
                        raise ValueError('unused YRP responses after real terminal')
                    break
                raw = env.step()
            else:
                raise ValueError('ingress record count limit exceeded')
        except (ValueError, IndexError, KeyError, RuntimeError) as exception:
            error = str(exception)
            writer.flag(f'ingress_failed: {error}')
        if env.query_card_error_count:
            error = error or 'Core query failed during ingress'
            writer.flag('ingress_query_failed')
        terminal = interpreter.terminal if error is None else None
        writer.finish(winner=terminal[0] if terminal else -1,
                      reason=terminal[1] if terminal else -1, core_terminal=terminal is not None)
        result = {'converted': True, 'trajectory_path': writer.path, 'source': origin['kind'],
                  'source_response_count': input_count, 'conversion_error': error,
                  'behavior_clone_eligible': False}
        if error is None:
            try:
                result.update(replay_core_trajectory(writer.path))
            except (ValueError, RuntimeError) as exception:
                result['verification_error'] = str(exception)
        return result
    finally:
        random.setstate(random_state)
        if writer is not None:
            writer.finish()
        try:
            if env is not None:
                try:
                    env._close_duel()
                finally:
                    env.cdb.close()
        finally:
            if previous is not None:
                previous.install_callbacks()


def import_yrp(path, output_root=DEFAULT_INGEST_ROOT, *, player_ids=None, timeout_seconds=300):
    """把受支持 YRP 重建为诊断轨迹；缺原始报文/历史资产仍拒绝模仿资格"""
    data = read_yrp(path)
    if not data['replay_mode_supported']:
        raise ValueError('legacy non-uniform YRP can be inspected, but is not reconstructed with the fixed current Core profile')
    import numpy as np
    decks = {str(seat): {**deepcopy(deck), 'side': []} for seat, deck in enumerate(data['decks'])}
    # 公开 replay_mode::StartDuel 的 YRP1 路径先调用 std::mt19937(seed)()，不是直接送入头部 seed
    core_seed = (data['seed'] if data['seed_sequence'] is not None else
                 int(np.random.RandomState(data['seed']).randint(0, 2**32, dtype=np.uint32)))
    reset = {'seed': core_seed, 'lp': data['start_lp'], 'start_hand': data['start_hand'],
             'draw_count': data['draw_count'], 'duel_flags': data['duel_flag'], 'initial_position': 8,
             'players': {seat: {part: list(deck[part]) for part in ('main', 'extra')}
                         for seat, deck in decks.items()}}
    if data['seed_sequence'] is not None:
        reset['seed_sequence'] = data['seed_sequence']
    origin = {'format_version': 1, 'kind': 'yrp', 'file_sha256': data['file_sha256'],
              'replay_sha256': data['replay_sha256'], 'player_ids': validate_player_ids(player_ids),
              'original_core_output_available': False, 'historical_assets_known': False,
              'yrp_version': data['version'], 'container': data['container'], 'header_seed': data['seed'],
              'replay_profile': 'fluorohydride_uniform_mt19937_v1_seedseq_v2_direct_order'}
    return _convert(reset, decks, origin, output_root, responses=data['responses'], timeout_seconds=timeout_seconds)


def import_link_capture(path, output_root=DEFAULT_INGEST_ROOT, *, timeout_seconds=300):
    """全量 Link 严格比较原始报文，单方/截断捕获只返回诊断而不补造隐藏信息"""
    with _freeze_source(path) as frozen:
        report = inspect_link_capture(frozen)
        header, footer = report['header'], report['footer']
        if header['visibility'] != 'full_core' or footer['truncated'] or footer['errors']:
            return {**report, 'converted': False, 'limitations': ['partial_or_invalid_link_capture']}
        frozen.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: frozen.read(1024 * 1024), b''):
            digest.update(chunk)
        frozen.seek(0)
        origin = {'format_version': 1, 'kind': 'link', 'file_sha256': digest.hexdigest(),
                  'player_ids': validate_player_ids(header.get('player_ids')),
                  'original_core_output_available': True, 'historical_assets_known': True}
        return _convert(header['reset'], header['decks'], origin, output_root,
                        link_source=frozen, expected_assets=header['assets'], timeout_seconds=timeout_seconds)
