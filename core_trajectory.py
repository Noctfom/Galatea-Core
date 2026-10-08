"""流式记录公开 Core 原始轨迹，并复用现有解析器确定性校验消息、响应和观测"""

import ctypes
import gzip
import hashlib
import json
import numbers
import os
import struct
import tempfile
import uuid
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from data_types import GameAction
from selection_protocol import build_core_response_buffer

TRAJECTORY_SCHEMA_VERSION = 1
MAX_TRAJECTORY_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 1024 * 1024
MAX_RECORDS = 100_000
DEFAULT_TRAJECTORY_ROOT = './replays/core_trajectories'


def framework_version():
    """读取统一版本文件，避免后续发行忘记更新轨迹来源版本"""
    return Path(__file__).with_name('version.txt').read_text(encoding='utf-8').strip()


def _json_bytes(value):
    """使用唯一 JSON 编码计算校验摘要，拒绝非有限数字"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'


def file_sha256(path):
    """分块计算资产摘要，不把完整文件留在内存中"""
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_asset_identity(env):
    """绑定实际缓存的 Lua/卡片数据和当前公开 Core、观测协议身份"""
    import galatea_env
    from protocol_schema import get_current_protocol_metadata

    galatea_env._init_card_cache()
    scripts = hashlib.sha256()
    for name, (buffer, size) in sorted(env.script_buffers.items()):
        content = ctypes.string_at(ctypes.addressof(buffer), size)
        scripts.update(_json_bytes({'name': name, 'size': size}))
        scripts.update(content)
    card_cache = hashlib.sha256()
    for code, row in sorted(galatea_env._GLOBAL_CARD_CACHE.items()):
        card_cache.update(_json_bytes([code, list(row)]))
    return {
        'core_sha256': file_sha256(env.dll_path),
        'cdb_sha256': file_sha256(env.cdb_path),
        'loaded_card_cache_sha256': card_cache.hexdigest(),
        'loaded_scripts_sha256': scripts.hexdigest(),
        'protocol': get_current_protocol_metadata(),
    }


def encode_response(response):
    """显式区分整数与字节响应，禁止对象反序列化和超过 Core 边界的数据"""
    if isinstance(response, numbers.Integral) and not isinstance(response, bool):
        value = int(response)
        if not -(2**31) <= value < 2**32:
            raise ValueError('response integer outside uint32/int32')
        return {'kind': 'int', 'value': value}
    if isinstance(response, (bytes, bytearray)) and len(response) <= 512:
        return {'kind': 'bytes', 'hex': bytes(response).hex()}
    raise ValueError('unsupported Core response')


def decode_response(encoded):
    """从受限 JSON 响应恢复当前 Core 接口参数"""
    if not isinstance(encoded, dict):
        raise ValueError('invalid response record')
    if set(encoded) == {'kind', 'value'} and encoded['kind'] == 'int':
        value = encoded['value']
        if type(value) is int and -(2**31) <= value < 2**32:
            return value
    if set(encoded) == {'kind', 'hex'} and encoded['kind'] == 'bytes':
        value = _decode_hex(encoded['hex'], 512)
        return value
    raise ValueError('invalid response record')


def canonical_response(response):
    """按 Core 实际读取的 512 字节比较响应，不改动多选索引顺序"""
    encoded = encode_response(response)
    raw = (struct.pack('<I', encoded['value'] & 0xFFFFFFFF) if encoded['kind'] == 'int'
           else bytes.fromhex(encoded['hex']))
    return build_core_response_buffer(raw)


def _decode_hex(value, maximum):
    """限制原始消息或响应长度，拒绝不规范十六进制文本"""
    if not isinstance(value, str) or len(value) > maximum * 2:
        raise ValueError('oversized or invalid hex record')
    raw = bytes.fromhex(value)
    if raw.hex() != value:
        raise ValueError('noncanonical hex record')
    return raw


def serialize_actions(actions):
    """保存原始候选池的坐标与宏动作，避免重放重新抽样"""
    if len(actions) > 512:
        raise ValueError('candidate pool exceeds trajectory boundary')
    result = []
    for action in actions:
        item = {field.name: getattr(action, field.name) for field in fields(GameAction)}
        item['decision_bytes'] = bytes(item['decision_bytes']).hex()
        result.append(item)
    # 使用同一校验器确保写出的候选可安全还原。
    deserialize_actions(result)
    return result


def deserialize_actions(items):
    """仅还原白名单 GameAction 字段，不允许动态类或嵌套对象"""
    if not isinstance(items, list) or len(items) > 512:
        raise ValueError('invalid candidate pool')
    names = {field.name for field in fields(GameAction)}
    lists = {'macro_targets', 'macro_places', 'macro_target_codes',
             'macro_target_values', 'macro_target_locations'}
    optional = {'response_value', 'decision_value'}
    result = []
    for item in items:
        if not isinstance(item, dict) or set(item) != names:
            raise ValueError('unknown/missing action fields')
        values = dict(item)
        for key, value in item.items():
            if key == 'decision_bytes':
                values[key] = _decode_hex(value, 512)
            elif key == 'desc_str':
                if not isinstance(value, str) or len(value) > 4096:
                    raise ValueError('invalid action description')
            elif key in ('finishable', 'cancelable'):
                if type(value) is not bool:
                    raise ValueError('invalid action boolean')
            elif key in lists:
                if value is not None and (not isinstance(value, list) or len(value) > 512
                        or any(type(x) is not int or not -(2**64) <= x < 2**64 for x in value)):
                    raise ValueError('invalid action target list')
            elif not (key in optional and value is None):
                if type(value) is not int or not -(2**64) <= value < 2**64:
                    raise ValueError('invalid action integer')
        result.append(GameAction(**values))
    return result


def matching_action_indices(snapshot, response, msg_type, payload):
    """用既有包装器找全部响应匹配项，歧义不伪装成监督标签"""
    from ai_bot import AiBot
    packer = AiBot.__new__(AiBot)
    expected = canonical_response(response)
    matches = []
    for index, action in enumerate(snapshot.valid_actions):
        try:
            packed = packer._pack_response(action, msg_type=msg_type, msg_args=payload)
            if canonical_response(packed) == expected:
                matches.append(index)
        except (ValueError, TypeError, OverflowError, struct.error):
            continue
    return matches


def observation_sha256(batch):
    """摘要实际 CPU 编码观测，不保存完整张量或访问未来信息"""
    import torch
    digest = hashlib.sha256()
    for key, tensor in sorted(batch.items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(_json_bytes({'key': key, 'dtype': str(cpu.dtype),
                                   'shape': list(cpu.shape)}))
        digest.update(cpu.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()


class CoreTrajectoryWriter:
    """有容量上限的逐行压缩记录器；诊断失败不得终止对局"""

    def __init__(self, header, root=DEFAULT_TRAJECTORY_ROOT, max_bytes=MAX_TRAJECTORY_BYTES):
        """创建独立轨迹文件，摘要只校验损坏，不宣称来源认证"""
        os.makedirs(root, exist_ok=True)
        self.trace_id = str(uuid.uuid4())
        self.path = os.path.abspath(os.path.join(root, f'duel_{self.trace_id}.core.jsonl.gz'))
        self._stream = gzip.open(self.path, 'xb', compresslevel=3)
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._count = 0
        self._max_bytes = min(int(max_bytes), MAX_TRAJECTORY_BYTES)
        self.errors = []
        self.stopped = False
        self.mapping_complete = True
        self.write('header', schema_version=TRAJECTORY_SCHEMA_VERSION,
                   trace_id=self.trace_id, **header)

    def flag(self, error):
        """保留有界质量标记，避免重复错误累积"""
        if error not in self.errors and len(self.errors) < 32:
            self.errors.append(str(error)[:256])

    def write(self, kind, **data):
        """流式写入一条记录，容量或 I/O 故障仅停用记录功能"""
        if self.stopped:
            return
        try:
            raw = _json_bytes({'seq': self._count, 'kind': kind, **data})
            if (len(raw) > MAX_RECORD_BYTES or self._bytes + len(raw) > self._max_bytes
                    or self._count >= MAX_RECORDS):
                raise ValueError('trajectory recording limit reached')
            self._stream.write(raw)
            self._digest.update(raw)
            self._bytes += len(raw)
            self._count += 1
        except Exception as error:
            self.flag(str(error))
            self.stopped = True
            print(f'⚠️ 规范轨迹记录已停用，对局继续：{error}')

    def finish(self, *, winner=-1, reason=-1, core_terminal=False):
        """封存摘要和质量标记；截断或未映射数据不自动成为学习样本"""
        if self._stream is None:
            return
        try:
            footer = {'kind': 'footer', 'seq': self._count,
                      'sha256': self._digest.hexdigest(), 'record_count': self._count,
                      'winner': int(winner), 'reason': int(reason),
                      'core_terminal': bool(core_terminal), 'truncated': self.stopped,
                      'errors': self.errors, 'mapping_complete': self.mapping_complete}
            self._stream.write(_json_bytes(footer))
        except Exception as error:
            print(f'⚠️ 规范轨迹封存失败：{error}')
        finally:
            stream = self._stream
            self._stream = None
            try:
                stream.close()
            except Exception as error:
                print(f'⚠️ 规范轨迹关闭失败：{error}')


class CoreTrajectoryInterpreter:
    """记录和重放共用现有 MessageParser/DuelState，不另写 Core 解析逻辑"""

    def __init__(self, brain, writer=None):
        """绑定当前状态与可选记录器"""
        self.brain = brain
        self.writer = writer
        self.terminal = None

    def parse_chunk(self, raw):
        """记录完整 Core 数据块并按现有规则拆包"""
        from gamestate import MessageParser
        messages = MessageParser.parse(raw)
        if self.writer and not self.writer.stopped:
            parsed = b''.join(messages)
            self.writer.write('chunk', hex=bytes(raw).hex(),
                              parsed_bytes=len(parsed), message_count=len(messages),
                              parse_complete=parsed == bytes(raw))
            if parsed != bytes(raw):
                self.writer.flag('unparsed_core_bytes')
        return messages

    def apply_message(self, msg):
        """按原有时点更新状态，并记录真正消费的消息"""
        self.brain.update(msg[0], msg[1:])
        if msg[0] == 5 and len(msg) >= 3:
            self.terminal = (msg[1], msg[2])
        if self.writer and not self.writer.stopped:
            self.writer.write('message', hex=bytes(msg).hex())
            if msg[0] == 1:
                self.writer.flag('core_retry')
            if msg[0] == 1:
                self.writer.flag('core_retry')

    def snapshot(self, env=None):
        """保留查询时点与查询前候选，避免重放事件历史错位"""
        if self.writer and not self.writer.stopped:
            try:
                self.writer.write('snapshot', query=env is not None,
                                  actions=serialize_actions(self.brain.current_valid_actions))
            except Exception as error:
                self.writer.flag(f'candidate_capture_failed: {error}')
        return self.brain.get_snapshot(env)

    def record_response(self, snapshot, response, msg_type, payload, actor,
                        chosen_index=None, observation=None, evaluation=None, source='model'):
        """记录实际响应和严格映射，不能映射的规则宏动作只保留原始数据"""
        if not self.writer or self.writer.stopped:
            return
        try:
            matches = matching_action_indices(snapshot, response, msg_type, payload)
            mapped = len(matches) == 1 and matches[0] < 120
            if chosen_index is not None:
                mapped = mapped and matches[0] == chosen_index
            self.writer.mapping_complete &= mapped
            self.writer.write('response', response=encode_response(response),
                              prompt_type=int(msg_type), payload=bytes(payload).hex(),
                              actor=int(actor), source=source, matches=matches,
                              chosen_index=chosen_index,
                              observation_sha256=observation_sha256(observation) if observation else None,
                              evaluation=evaluation)
        except Exception as error:
            self.writer.mapping_complete = False
            self.writer.flag(f'response_capture_failed: {error}')


def _iter_records(path):
    """按行读取受限压缩 JSON，拒绝超大记录和解压膨胀"""
    total = 0
    with gzip.open(path, 'rb') as stream:
        for count in range(MAX_RECORDS + 2):
            raw = stream.readline(MAX_RECORD_BYTES + 1)
            if not raw:
                return
            total += len(raw)
            if len(raw) > MAX_RECORD_BYTES or total > MAX_TRAJECTORY_BYTES + MAX_RECORD_BYTES:
                raise ValueError('trajectory size limit exceeded')
            record = json.loads(raw)
            if not isinstance(record, dict) or _json_bytes(record) != raw:
                raise ValueError('noncanonical trajectory JSON')
            yield record, raw
        raise ValueError('too many trajectory records')


def inspect_core_trajectory(path):
    """先校验格式、摘要和边界，不调用 Core、不执行外部代码"""
    digest = hashlib.sha256()
    header = footer = None
    count = 0
    for record, raw in _iter_records(path):
        if footer is not None or type(record.get('seq')) is not int or record['seq'] != count:
            raise ValueError('trajectory sequence/EOF mismatch')
        kind = record.get('kind')
        if count == 0:
            if kind != 'header' or record.get('schema_version') != TRAJECTORY_SCHEMA_VERSION:
                raise ValueError('unsupported trajectory schema')
            header = record
            uuid.UUID(record['trace_id'])
            reset = record['reset']
            if (type(reset['seed']) is not int or not 0 <= reset['seed'] < 2**32
                    or (reset['lp'], reset['start_hand'], reset['draw_count'], reset['duel_flags'])
                    != (8000, 5, 1, 0)):
                raise ValueError('unsupported Core reset parameters')
            for decks in (record['decks'], reset['players']):
                if set(decks) != {'0', '1'}:
                    raise ValueError('missing trajectory deck/seat')
                for deck in decks.values():
                    for part in ('main', 'extra'):
                        codes = deck[part]
                        if (not isinstance(codes, list) or len(codes) > 256
                                or any(type(x) is not int or not 0 < x < 2**32 for x in codes)):
                            raise ValueError('invalid trajectory deck')
            if type(record['ghost_byte']) is not bool:
                raise ValueError('invalid Core dialect')
            for player in ('0', '1'):
                for part in ('main', 'extra'):
                    if sorted(record['decks'][player][part]) != sorted(reset['players'][player][part]):
                        raise ValueError('stable deck and injection cards disagree')
        elif kind == 'header':
            raise ValueError('duplicate trajectory header')
        elif kind in ('chunk', 'message'):
            if not _decode_hex(record['hex'], 65536):
                raise ValueError('empty Core message')
            if kind == 'chunk' and 'parse_complete' in record:
                if (type(record['parse_complete']) is not bool
                        or type(record.get('parsed_bytes')) is not int
                        or not 0 <= record['parsed_bytes'] <= 65536
                        or type(record.get('message_count')) is not int
                        or not 0 <= record['message_count'] <= 65536):
                    raise ValueError('invalid parse coverage metadata')
        elif kind == 'snapshot':
            if type(record['query']) is not bool:
                raise ValueError('invalid query marker')
            deserialize_actions(record['actions'])
        elif kind == 'response':
            decode_response(record['response'])
            _decode_hex(record['payload'], 65536)
            if (type(record['actor']) is not int or record['actor'] not in (0, 1)
                    or type(record['prompt_type']) is not int or not 0 < record['prompt_type'] < 256):
                raise ValueError('invalid prompt/actor')
            matches = record['matches']
            if (not isinstance(matches, list) or len(matches) > 512
                    or any(type(x) is not int or not 0 <= x < 512 for x in matches)
                    or sorted(set(matches)) != matches):
                raise ValueError('invalid candidate matches')
            chosen = record['chosen_index']
            if chosen is not None and (type(chosen) is not int or not 0 <= chosen < 512):
                raise ValueError('invalid selected candidate')
            observation = record['observation_sha256']
            if observation is not None and (len(_decode_hex(observation, 32)) != 32):
                raise ValueError('invalid observation digest')
        elif kind == 'runtime':
            if not isinstance(record.get('policy'), dict):
                raise ValueError('invalid policy runtime metadata')
        elif kind == 'footer':
            if (any(type(record.get(key)) is not bool for key in
                    ('core_terminal', 'truncated', 'mapping_complete'))
                    or type(record.get('winner')) is not int
                    or record['winner'] not in (-1, 0, 1, 2)
                    or type(record.get('reason')) is not int
                    or not -4 <= record['reason'] <= 255
                    or not isinstance(record.get('errors'), list) or len(record['errors']) > 32
                    or any(not isinstance(x, str) or len(x) > 256 for x in record['errors'])):
                raise ValueError('invalid terminal/quality flags')
            if record['sha256'] != digest.hexdigest() or record['record_count'] != count:
                raise ValueError('trajectory checksum mismatch')
            footer = record
        else:
            raise ValueError('unknown trajectory record')
        if footer is None:
            digest.update(raw)
            count += 1
    if header is None or footer is None:
        raise ValueError('incomplete trajectory')
    return {'header': header, 'footer': footer, 'integrity_valid': True,
            'replay_verified': False, 'behavior_clone_eligible': False}


def replay_core_trajectory(path):
    """固定有界输入副本，避免校验后路径被替换再送入原生 Core"""
    with open(path, 'rb') as source, tempfile.TemporaryFile(mode='w+b') as frozen:
        total = 0
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            total += len(chunk)
            if total > MAX_TRAJECTORY_BYTES + MAX_RECORD_BYTES:
                raise ValueError('compressed trajectory size limit exceeded')
            frozen.write(chunk)
        frozen.seek(0)
        return _replay_core_trajectory_source(frozen)


def _replay_core_trajectory_source(source):
    """在相同公开 Core/资产下重放，校验原始输出和实际编码观测"""
    import gamestate
    from feature_encoder import GalateaEncoder
    from galatea_env import GalateaEnv, get_callback_environment

    result = inspect_core_trajectory(source)
    source.seek(0)
    header, footer = result['header'], result['footer']
    if footer['truncated']:
        raise ValueError('truncated trajectory cannot be replay-verified')
    old_ghost = gamestate.CORE_HAS_GHOST_BYTE
    old_callback_environment = get_callback_environment()
    env = None
    try:
        gamestate.CORE_HAS_GHOST_BYTE = header['ghost_byte']
        env = GalateaEnv()
        if runtime_asset_identity(env) != header['assets']:
            raise ValueError('Core/script/CDB/vocabulary/semantic asset identity mismatch')
        injections = header['reset']['players']
        raw = env.reset(*(SimpleNamespace(**injections[str(p)]) for p in (0, 1)),
                        seed=header['reset']['seed'])
        decks = header['decks']
        brain = gamestate.DuelState(decks['0']['main'], decks['0']['extra'],
                                    decks['1']['main'], decks['1']['extra'])
        interpreter = CoreTrajectoryInterpreter(brain)
        encoder = GalateaEncoder()
        queue = []
        snapshot = None
        prompt = None
        mappings_complete = True
        parse_coverage_complete = True
        retry_seen = False
        raw_output_exact = True
        client_hint_reordered_chunks = 0
        observations = responses = 0
        for record, _ in _iter_records(source):
            kind = record['kind']
            if kind == 'chunk':
                if raw is None:
                    raw = env.step()
                expected_raw = _decode_hex(record['hex'], 65536)
                actual_raw = bytes(raw or b'')
                if actual_raw != expected_raw:
                    if not equivalent_client_hint_removals(expected_raw, actual_raw):
                        raise ValueError(
                            f"Core output diverged at record {record['seq']}; "
                            f"expected={record['hex'][:256]}, actual={actual_raw.hex()[:256]}"
                        )
                    # 只读诊断继续验证后续完整观测；原始数据不重写，模仿资格仍拒绝。
                    raw_output_exact = False
                    client_hint_reordered_chunks += 1
                if queue:
                    raise ValueError('unconsumed message queue before next chunk')
                queue = interpreter.parse_chunk(raw)
                coverage = b''.join(queue) == bytes(raw)
                parse_coverage_complete &= coverage
                if actual_raw != expected_raw:
                    queue = gamestate.MessageParser.parse(expected_raw)
                if 'parse_complete' in record and (
                        record['parse_complete'] != coverage
                        or record['parsed_bytes'] != sum(map(len, queue))
                        or record['message_count'] != len(queue)):
                    raise ValueError('parse coverage metadata diverged')
                raw = None
            elif kind == 'message':
                message = _decode_hex(record['hex'], 65536)
                if not queue or queue.pop(0) != message:
                    raise ValueError('processed Core message diverged')
                interpreter.apply_message(message)
                retry_seen |= message[0] == 1
                if message[0] in encoder_decision_messages():
                    prompt = message
            elif kind == 'snapshot':
                brain.current_valid_actions = deserialize_actions(record['actions'])
                snapshot = interpreter.snapshot(env if record['query'] else None)
            elif kind == 'response':
                if (snapshot is None or prompt is None or prompt[0] != record['prompt_type']
                        or bytes(prompt[1:]) != _decode_hex(record['payload'], 65536)
                        or prompt[1] != record['actor']):
                    raise ValueError('response does not belong to current prompt/actor')
                response = decode_response(record['response'])
                matches = matching_action_indices(snapshot, response, prompt[0], prompt[1:])
                if matches != record['matches']:
                    raise ValueError('response/candidate mapping diverged')
                unique = len(matches) == 1 and matches[0] < 120
                chosen = record['chosen_index']
                if chosen is not None and (type(chosen) is not int or chosen not in matches):
                    raise ValueError('selected candidate does not match response')
                mappings_complete &= unique
                expected_observation = record['observation_sha256']
                if expected_observation is not None:
                    actual = encoder.encode(snapshot, player_id=record['actor'])
                    if observation_sha256(actual) != expected_observation:
                        raise ValueError('encoded model observation diverged')
                    observations += 1
                env.send_action(response)
                brain.begin_transition_event(record['actor'], prompt[0], response)
                if queue:
                    mappings_complete = False
                queue = []
                responses += 1
                snapshot = None
        terminal_matches = interpreter.terminal == (footer['winner'], footer['reason'])
        if footer['core_terminal'] and not terminal_matches:
            raise ValueError('terminal outcome diverged')
        if env.query_card_error_count:
            raise ValueError('Core query failed during replay')
        result.update(replay_verified=True, response_count=responses,
                      observation_check_count=observations,
                      parse_coverage_complete=parse_coverage_complete,
                      retry_seen=retry_seen,
                      raw_output_exact=raw_output_exact,
                      client_hint_reordered_chunks=client_hint_reordered_chunks,
                      behavior_clone_eligible=bool(
                          terminal_matches and footer['core_terminal']
                          and footer['winner'] in (0, 1, 2)
                          and mappings_complete and footer['mapping_complete']
                          and not footer['errors']
                          and parse_coverage_complete and not retry_seen
                          and raw_output_exact
                          and observations == responses and responses > 0))
        return result
    finally:
        gamestate.CORE_HAS_GHOST_BYTE = old_ghost
        try:
            if env is not None:
                try:
                    env._close_duel()
                finally:
                    env.cdb.close()
        finally:
            if old_callback_environment is not None:
                old_callback_environment.install_callbacks()


def encoder_decision_messages():
    """复用 Core 已有交互消息集合，包含由规则处理的非模型提示"""
    from action_candidates import MODEL_ACTION_MSGS
    return frozenset(MODEL_ACTION_MSGS) | {132}


def equivalent_client_hint_removals(expected, actual):
    """仅识别同一卡片连续撤销提示的排列变化，不放宽严格训练资格"""
    from gamestate import MessageParser

    def normalized(raw):
        """同位置同类型的纯客户端提示撤销可交换，其余顺序原样保留"""
        messages = MessageParser.parse(raw)
        if b''.join(messages) != bytes(raw):
            return None
        result = []
        index = 0
        while index < len(messages):
            message = messages[index]
            # 公开旧 Core MSG_CARD_HINT：type + location(4) + hint_type + description(4)。
            if len(message) == 10 and message[0] == 160 and message[5] == 7:
                end = index + 1
                while (end < len(messages) and len(messages[end]) == 10
                       and messages[end][:6] == message[:6]):
                    end += 1
                result.extend(sorted(messages[index:end]))
                index = end
            else:
                result.append(message)
                index += 1
        return result

    expected_messages, actual_messages = normalized(expected), normalized(actual)
    return expected_messages is not None and expected_messages == actual_messages
