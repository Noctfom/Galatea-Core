# 本文件实现 Galatea 训练进程的跨平台单实例互斥锁

import json
import os
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


RUNTIME_TERMINAL_PHASES = {'completed', 'failed', 'interrupted', 'stopped'}


def _read_runtime_json(path):
    """读取小型运行状态文件，写入中断时保留页面可用"""
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def read_training_runtime(project_root):
    """根据进程创建时间恢复当前训练，避免刷新页面后丢失状态"""
    import psutil
    state = _read_runtime_json(Path(project_root) / '.galatea_train.lock.owner.json')
    alive = False
    try:
        process = psutil.Process(int(state['pid']))
        alive = (abs(process.create_time() - float(state['process_create_time'])) < 0.01
                 and process.status() not in {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD})
    except (KeyError, ValueError, TypeError, psutil.Error):
        pass
    state['process_alive'] = alive
    state['active'] = alive and state.get('phase') not in RUNTIME_TERMINAL_PHASES
    try:
        state['snapshot_age_seconds'] = max(0.0, time.time() - float(state.get('updated_at_epoch', 0)))
    except (ValueError, TypeError):
        state['snapshot_age_seconds'] = 0.0
    return state


def _control_directory(project_root, session_id):
    """请求按训练进程独立保存，不写入模型或部署包"""
    session_id = str(uuid.UUID(str(session_id)))
    return Path(project_root) / 'system_logs' / 'training_control' / session_id


def normalize_runtime_update(config, changes):
    """仅接受下一轮可生效的采集配置，不修改网络和优化参数"""
    from training_validation import resolve_league_weights, validate_collection_allocation
    if not isinstance(changes, dict) or not changes or set(changes) - {
            'league_weights', 'windbot_mirror_probability', 'allocation_mode'}:
        raise ValueError('动态配置仅支持采集调度、联盟比例和WindBot镜像比例')
    normalized = {}
    if 'allocation_mode' in changes:
        normalized['allocation_mode'] = validate_collection_allocation(changes['allocation_mode'])
    if 'league_weights' in changes:
        normalized['league_weights'] = resolve_league_weights(
            changes['league_weights'], windbot_enabled=config.get('windbot_enabled', False))
    if 'windbot_mirror_probability' in changes:
        probability = changes['windbot_mirror_probability']
        if not config.get('windbot_enabled'):
            raise ValueError('本次训练未初始化WindBot，需重新启动并启用后才能调整')
        if (isinstance(probability, bool) or not isinstance(probability, (float, int))
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('WindBot镜像比例必须位于0到1之间')
        normalized['windbot_mirror_probability'] = float(probability)
    return normalized


def pending_training_updates(project_root, session_id):
    """只查看当前训练的小型请求目录，不扫描模型或日志内容"""
    if not session_id:
        return []
    try:
        directory = _control_directory(project_root, session_id)
    except (ValueError, TypeError):
        return []
    return [_read_runtime_json(path) for path in sorted(directory.glob('*.json'))]


def request_training_update(project_root, session_id, revision, changes):
    """提交独立请求，由Trainer在下一轮采集前确认应用"""
    state = read_training_runtime(project_root)
    if not state.get('active') or state.get('session_id') != session_id:
        raise ValueError('训练已结束或运行身份已变化，请刷新当前训练状态')
    if state.get('runtime_version') != 1 or not state.get('config'):
        raise ValueError('当前训练尚未就绪或不支持动态配置')
    if state.get('config_revision', 0) != revision:
        raise ValueError('当前配置已更新，请刷新后重新调整')
    if state.get('target_iteration') is not None and state.get('iteration', 0) >= state['target_iteration']:
        raise ValueError('当前已是最后一轮，没有下一轮采集可应用修改')
    normalized = normalize_runtime_update(state['config'], changes)
    directory = _control_directory(project_root, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    payload = {'request_id': request_id, 'session_id': session_id, 'base_revision': revision,
               'changes': normalized, 'requested_at': datetime.now(timezone.utc).isoformat()}
    target = directory / f'{time.time_ns()}_{request_id}.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, target)
    return payload


class TrainerAlreadyRunningError(RuntimeError):
    """表示当前项目已经有一个 Trainer 持有训练锁"""


class TrainerProcessLock:
    """使用操作系统文件锁保证同一项目同时只有一个 Trainer"""

    def __init__(self, lock_path=None):
        project_root = Path(__file__).resolve().parent
        self.lock_path = Path(lock_path or project_root / ".galatea_train.lock")
        self.metadata_path = Path(f"{self.lock_path}.owner.json")
        self._handle = None
        self._locked = False
        self._last_runtime_write = 0.0
        self._runtime_warning = False
        import psutil
        self._metadata = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "run_id": None,
            'process_create_time': psutil.Process().create_time(),
            'session_id': str(uuid.uuid4()),
            'runtime_version': 1,
            'phase': 'initializing',
            'config_revision': 0,
        }

    def _read_owner(self):
        """读取锁持有者信息，仅用于生成可诊断的报错"""
        try:
            raw = self.metadata_path.read_text(encoding="utf-8").strip()
            if not raw:
                return {}
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _write_metadata(self, *, durable=True):
        """在持锁状态下写入 PID、启动时间和 run_id"""
        self._metadata['updated_at_epoch'] = time.time()
        payload = json.dumps(self._metadata, ensure_ascii=False, allow_nan=False) + "\n"
        temp_path = Path(f"{self.metadata_path}.{os.getpid()}.tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temp_path, self.metadata_path)

    def acquire(self):
        """非阻塞获取训练锁；锁被占用时立即给出明确错误"""
        if self._locked:
            return self

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\x00")
            handle.flush()
        handle.seek(0)

        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            handle.close()
            owner = self._read_owner()
            owner_pid = owner.get("pid", "未知")
            owner_run_id = owner.get("run_id") or "尚未生成"
            raise TrainerAlreadyRunningError(
                "检测到同一项目已有 Trainer 正在运行："
                f"PID={owner_pid}，run_id={owner_run_id}。"
                "当前版本不支持并行 Trainer，请先结束已有训练。"
            ) from error

        self._handle = handle
        self._locked = True
        try:
            self._write_metadata()
        except Exception:
            self.release()
            raise
        return self

    def set_run_id(self, run_id):
        """Trainer 创建 run_id 后更新锁文件中的诊断信息"""
        if not self._locked:
            raise RuntimeError("训练锁尚未获取，无法写入 run_id")
        self._metadata["run_id"] = str(run_id)
        self._write_metadata()

    def publish_runtime(self, *, force=False, **values):
        """阶段变化立即发布，普通进度最多每五秒写一次且不强制刷盘"""
        if not self._locked:
            return
        if values.get('phase') is not None and values['phase'] != self._metadata.get('phase'):
            values['phase_started_at_epoch'] = time.time()
        self._metadata.update(values)
        now = time.monotonic()
        if not force and now - self._last_runtime_write < 5.0:
            return
        self._last_runtime_write = now
        try:
            self._write_metadata(durable=False)
            self._runtime_warning = False
        except (OSError, ValueError, TypeError) as error:
            if not self._runtime_warning:
                print(f'⚠️ 训练状态面板暂不可更新，训练继续: {error}')
                self._runtime_warning = True

    def take_runtime_updates(self, config, iteration):
        """按提交顺序消费请求，过期页面的请求不会覆盖已应用的配置"""
        directory = _control_directory(self.lock_path.parent, self._metadata['session_id'])
        for path in sorted(directory.glob('*.json')):
            request = _read_runtime_json(path)
            result = {'request_id': request.get('request_id'), 'iteration': iteration,
                      'requested_at': request.get('requested_at'), 'status': 'rejected'}
            try:
                if request.get('session_id') != self._metadata['session_id']:
                    raise ValueError('请求不属于当前训练')
                if request.get('base_revision') != self._metadata['config_revision']:
                    raise ValueError('配置版本已变化，请根据最新配置重新提交')
                changes = normalize_runtime_update(config, request.get('changes'))
                result.update(status='applied', changes=changes)
                yield result
                config.update(changes)
                self._metadata['config_revision'] += 1
            except (ValueError, TypeError) as error:
                result['reason'] = str(error)
            self.publish_runtime(force=True, control_result=result, config=config)
            try:
                path.unlink()
            except OSError as error:
                print(f'⚠️ 已处理的配置请求暂未清理: {error}')

    def finish(self, phase='completed', error=None):
        """保留最近一次任务状态，页面可区分完成、中断与异常退出"""
        self.publish_runtime(force=True, phase=phase, error=error,
                             finished_at=datetime.now(timezone.utc).isoformat())

    def release(self):
        """释放训练锁；锁文件保留以避免删除与重新创建之间的竞态"""
        if not self._locked or self._handle is None:
            return

        if self._metadata.get('phase') not in RUNTIME_TERMINAL_PHASES:
            self.finish('stopped')

        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            self._locked = False

    def __enter__(self):
        """进入上下文时获取训练锁"""
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        """离开上下文时确保释放训练锁"""
        if exc_value is not None:
            self.finish('interrupted' if isinstance(exc_value, KeyboardInterrupt) else 'failed', str(exc_value))
        self.release()
        return False
