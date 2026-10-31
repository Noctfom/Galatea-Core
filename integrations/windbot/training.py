"""可选训练对手的构筑快照、留出分配与单局进程管理"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from integrations.windbot.bridge import YGOProPeer, DECISIONS, prompt_actor, visible_message
from integrations.windbot.runtime import executor_catalog, executable_sha256, launch_bot, load_deck


def training_catalog(*, diagnostics=None):
    """在调用线程独立读取卡库，避免WebUI重运行复用其他线程的SQLite连接"""
    import deck_utils
    from card_reader import CardReader
    from card_vocab import get_default_card_vocabulary

    entries = executor_catalog()
    report = {'cdb_path': deck_utils.DEFAULT_CDB_PATH,
              'vocabulary_path': deck_utils.DEFAULT_CARD_VOCAB_PATH,
              'total': len(entries), 'compatible': 0, 'skipped': []}
    database = CardReader(deck_utils.DEFAULT_CDB_PATH)
    result = {}
    try:
        if database.conn is None:
            raise RuntimeError(f"WindBot执行器扫描无法读取卡库: {deck_utils.DEFAULT_CDB_PATH}")
        vocabulary = get_default_card_vocabulary(deck_utils.DEFAULT_CARD_VOCAB_PATH)
        for name, entry in entries.items():
            reason, unsupported = None, []
            if not entry['deck_available']:
                reason = 'missing_deck'
            else:
                path = Path(entry['deck_file'])
                try:
                    raw_deck = load_deck(path)
                    checked = deck_utils.load_deck(str(path.parent), path.stem,
                        card_database=database, vocabulary=vocabulary)
                    if checked is None or not checked.main:
                        reason = 'empty_deck'
                    elif not deck_utils.is_deck_compatible(checked):
                        reason, unsupported = 'unsupported_cards', list(checked.invalid_codes)
                    else:
                        result[name] = dict(entry, deck=raw_deck)
                except (OSError, ValueError) as error:
                    reason = 'invalid_deck'
                    report['skipped'].append({'executor': name, 'reason': reason, 'detail': str(error)})
                    continue
            if reason:
                report['skipped'].append({'executor': name, 'reason': reason, 'unsupported_codes': unsupported})
        report['compatible'] = len(result)
        if diagnostics is not None:
            diagnostics.update(report)
        return result
    finally:
        if database.conn is not None:
            database.conn.close()


def suggest_training_split(catalog):
    """默认只用普通与简单执行器，每个可用层级留出一个完整执行器"""
    groups = {}
    for name, entry in sorted(catalog.items()):
        if entry['level'] in ('Normal', 'Easy'):
            groups.setdefault(entry['level'], []).append(name)
    holdouts = [names[0] for names in groups.values() if len(names) > 1]
    # 相同构筑的别名执行器跟随留出，避免名义留出但仍由同构筑机器人训练
    def identity(name):
        deck = catalog[name]['deck']
        return tuple((part, tuple(sorted(deck[part]))) for part in ('main', 'extra'))
    held_builds = {identity(name) for name in holdouts}
    eligible = [name for names in groups.values() for name in names]
    holdouts = sorted(name for name in eligible if identity(name) in held_builds)
    return sorted(set(eligible) - set(holdouts)), holdouts


def synchronize_training_split(catalog, selected, previous, other):
    """将本次增删同步到另一侧，同主额外构筑的别名始终一起转移"""
    valid = set(catalog)
    selected, previous, other = (set(values) & valid for values in (selected, previous, other))
    def identity(name):
        return tuple(tuple(sorted(catalog[name]['deck'][part])) for part in ('main', 'extra'))
    added = {identity(name) for name in selected - previous}
    removed = {identity(name) for name in previous - selected} - added
    moved_in = {name for name in valid if identity(name) in added}
    moved_out = {name for name in valid if identity(name) in removed}
    selected = (selected | moved_in) - moved_out
    other = (other | moved_out) - selected
    return sorted(selected), sorted(other)


def prepare_training_config(*, executors=None, holdouts=None, rate=0.15, deck_mode='mirror',
                            runtime=None, mono=None, response_timeout=10):
    """启动前冻结训练名单与构筑，恢复任务也需显式开启对手"""
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or not 0 < rate <= 1):
        raise ValueError('WindBot Worker比例必须在0到1之间且大于0')
    if deck_mode not in ('mirror', 'training-pool'):
        raise ValueError('WindBot模型侧卡组选法必须为mirror或training-pool')
    if (isinstance(response_timeout, bool) or not isinstance(response_timeout, (int, float))
            or not math.isfinite(response_timeout) or not 0 < response_timeout <= 10):
        raise ValueError('WindBot单次响应预算必须在0到10秒之间且大于0')
    catalog = training_catalog()
    default_train, default_holdouts = suggest_training_split(catalog)
    holdout_names = sorted(set(holdouts if holdouts is not None else default_holdouts if executors is None else []))
    train_names = sorted(set(executors if executors is not None else
                             set(default_train + default_holdouts) - set(holdout_names)))
    if not train_names or set(train_names) & set(holdout_names):
        raise ValueError('WindBot训练名单不能为空，且不能与留出名单重叠')
    unknown = set(train_names + holdout_names) - set(catalog)
    if unknown:
        raise ValueError(f'WindBot执行器缺失或构筑与卡库词表不兼容: {sorted(unknown)}')
    def record(name):
        entry = catalog[name]
        path = Path(entry['deck_file']).resolve()
        return {'executor': name, 'level': entry['level'], 'deck_file': str(path),
                'deck_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'deck': entry['deck']}
    training, reserved = [record(name) for name in train_names], [record(name) for name in holdout_names]
    def build_id(entry):
        return tuple(tuple(sorted(entry['deck'][part])) for part in ('main', 'extra'))
    if {build_id(e) for e in training} & {build_id(e) for e in reserved}:
        raise ValueError('WindBot训练与留出名单含相同构筑，请将同构筑执行器放在同一侧')
    return {'format_version': 1, 'rate': float(rate), 'deck_mode': deck_mode,
            'runtime': str(Path(runtime).resolve()) if runtime else None, 'mono': mono,
            'response_timeout': float(response_timeout), 'executable_sha256': executable_sha256(runtime, mono),
            'training': training, 'holdouts': reserved}


def choose_episode_decks(config, rng, train_seat, deck_dir):
    """WindBot固定使用冻结构筑，模型可镜像或沿用常规训练抽样"""
    entry = deepcopy(rng.choice(config['training']))
    bot_deck = SimpleNamespace(**entry['deck'])
    if config['deck_mode'] == 'mirror':
        pair = (f"windbot:{entry['executor']}", entry['executor'], bot_deck, entry['executor'], bot_deck)
    else:
        import deck_utils
        pair = deck_utils.get_random_deck_pair(ydk_dir=deck_dir)
        if pair is None:
            raise ValueError('常规训练池没有可用卡组')
        environment, first_name, first, second_name, second = pair
        pair = ((environment, first_name, first, entry['executor'], bot_deck) if train_seat == 0 else
                (environment, entry['executor'], bot_deck, second_name, second))
    return entry, pair


class TrainingOpponent:
    """复用当前Worker的Core，只负责机器人传输，不采集专家训练样本"""

    def __init__(self, config, entry, seat, env, decks):
        self.seat = seat
        self.entry = entry
        self.stack = ExitStack()
        self.responses = 0
        self.response_seconds = 0.0
        self.last_external_response = False
        self.registry = Path(config['process_registry']) if config.get('process_registry') else None
        try:
            self.peer = YGOProPeer(seat, config['response_timeout'])
            self.stack.callback(self.peer.close)
            process, self.log = self.stack.enter_context(launch_bot(
                runtime=config.get('runtime'), mono=config.get('mono'), executor=entry['executor'],
                deck_file=entry['deck_file'], cdb_path=env.cdb_path, port=self.peer.port,
                room_token=self.peer.room_token, log_root='./system_logs/windbot'))
            if self.registry is not None:
                import psutil
                self.registry.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.registry.with_suffix('.tmp')
                temporary.write_text(json.dumps({'pid': process.pid,
                    'create_time': psutil.Process(process.pid).create_time()}), encoding='utf-8')
                temporary.replace(self.registry)
            self.peer.accept(entry['deck'])
            self.peer.start(decks)
        except BaseException:
            self.close()
            raise

    def observe(self, message):
        """只发送当前玩家可见事件，模型侧提示不交给机器人"""
        if message[0] == 1 and self.last_external_response:
            raise ValueError('WindBot响应触发Core Retry，整局回滚且不替换对手')
        if message[0] in DECISIONS:
            self.peer.check_idle()
            if prompt_actor(message) != self.seat:
                self.last_external_response = False
        elif message[0] != 1:
            masked = visible_message(message, self.seat)
            if masked is not None:
                self.peer.game_message(masked)

    def respond(self, message, env):
        """机器人响应直接回交原Core，不进入PPO候选与概率计算"""
        started = time.monotonic()
        response = self.peer.respond(message, env)
        self.response_seconds += time.monotonic() - started
        self.responses += 1
        self.last_external_response = True
        return response

    def close(self):
        """正常结束与异常退出统一关闭连接并回收独立进程"""
        self.stack.close()
        if self.registry is not None:
            self.registry.unlink(missing_ok=True)


def reap_worker_bot(registry_path):
    """Worker被强制终止后，按进程创建时刻回收其登记的机器人"""
    import psutil
    path = Path(registry_path)
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        process = psutil.Process(record['pid'])
        if process.create_time() == record['create_time']:
            process.terminate()
            try:
                process.wait(timeout=2)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    except (FileNotFoundError, psutil.NoSuchProcess):
        pass
    finally:
        path.unlink(missing_ok=True)
