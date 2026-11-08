# 管理单个 Worker 中可回滚、可提交的采样数据写入游标

import time
import math
from pathlib import Path


COLLECTION_SOURCES = ('rule', 'windbot', 'hist', 'self')
COLLECTION_FIELDS = ('committed_steps', 'completed_games', 'truncated_games', 'aborted_games',
                     'worker_seconds', 't_cpp_env', 't_encoder', 't_central',
                     't_local_opponent', 't_pass1_cpu', 't_windbot', 't_rule_bot')
COLLECTION_BACKENDS = ('starting', 'central', 'ONNX', 'PyTorch', 'rule', 'windbot')
COLLECTION_SLOT_SIZE = 5 + len(COLLECTION_SOURCES) * len(COLLECTION_FIELDS)


class CollectionTelemetry:
    """复用单局审计账本，仅向固定共享槽发布计数，不传输观测张量"""

    def __init__(self, shared, source='self', started_at=None):
        self.shared = shared
        self.source = source if source in COLLECTION_SOURCES else 'self'
        self.sources = {name: dict.fromkeys(COLLECTION_FIELDS, 0.0) for name in COLLECTION_SOURCES}
        self.backend = 'starting'
        self.finished = False
        self.ledger = None
        self.segments = []
        self.last_tick = time.monotonic() if started_at is None else started_at
        self.last_publish = 0.0
        self.publish(force=True)

    def _account_time(self, now):
        self.sources[self.source]['worker_seconds'] += max(0.0, now - self.last_tick)
        self.last_tick = now

    def begin_episode(self, source, ledger):
        """补局按实际对手归类，重置或解析失败也计入该来源的占用时间"""
        if self.ledger is not None:
            self.end_episode(aborted=True)
        self._account_time(time.monotonic())
        self.source = source
        self.ledger = ledger
        self.publish()

    def end_episode(self, steps=0, *, truncated=False, aborted=False):
        """回滚不计训练样本，完整提交与可信截断分开计局"""
        if self.ledger is None:
            return
        row = self.sources[self.source]
        row['aborted_games' if aborted else 'truncated_games' if truncated else 'completed_games'] += 1
        if not aborted:
            row['committed_steps'] += steps
            if steps:
                self.segments.append((self.source, int(steps)))
        for key in COLLECTION_FIELDS[5:]:
            row[key] += float(self.ledger.get(key, 0.0))
        self.ledger = None
        self.publish()

    def publish(self, *, force=False, now=None):
        """单写者序号保护无锁快照，Worker被终止时读取方也不会等待锁"""
        now = time.monotonic() if now is None else now
        if not force and now - self.last_publish < 5.0:
            return
        self._account_time(now)
        self.last_publish = now
        if self.shared is None:
            return
        sequence = int(self.shared[0])
        self.shared[0] = sequence + 1
        values = [float(self.finished), COLLECTION_BACKENDS.index(self.backend),
                  COLLECTION_SOURCES.index(self.source), now]
        values.extend(self.sources[source][key] for source in COLLECTION_SOURCES for key in COLLECTION_FIELDS)
        self.shared[1:] = values
        self.shared[0] = sequence + 2

    def export(self):
        """保存提交片段，合并阶段可准确核对实际接收的各来源样本"""
        self.publish(force=True)
        return {'sources': {name: dict(row) for name, row in self.sources.items()},
                'segments': list(self.segments)}

    def finish(self):
        """收尾与序列化时间也计入Worker占用，异常局不遗留有效样本"""
        if self.ledger is not None:
            self.end_episode(aborted=True)
        self.finished = True
        self.publish(force=True)


def read_collection_slot(shared, previous=None):
    """只尝试读取三次，写入中断时沿用上一个完整快照"""
    for _ in range(3):
        sequence = shared[0]
        if int(sequence) % 2:
            continue
        values = list(shared[1:])
        if sequence == shared[0]:
            return values if sequence else previous
    return previous


def summarize_collection_slots(slots, previous, active_workers=(), *, now=None, ended_at=None):
    """展示Worker提交进度，合并成功后再以实际接收行数替换样本计数"""
    now = time.monotonic() if now is None else now
    sources = {name: dict.fromkeys(COLLECTION_FIELDS, 0.0) for name in COLLECTION_SOURCES}
    backends = {}
    for worker_id, slot in enumerate(slots):
        values = read_collection_slot(slot, previous.get(worker_id))
        if values is None:
            continue
        previous[worker_id] = values
        finished, backend, current_source, updated = values[:4]
        name = COLLECTION_BACKENDS[int(backend)]
        backends[name] = backends.get(name, 0) + 1
        for index, source in enumerate(COLLECTION_SOURCES):
            offset = 4 + index * len(COLLECTION_FIELDS)
            for j, key in enumerate(COLLECTION_FIELDS):
                sources[source][key] += values[offset + j]
        if not finished and (worker_id in active_workers or worker_id in (ended_at or {})):
            end = (ended_at or {}).get(worker_id, now)
            sources[COLLECTION_SOURCES[int(current_source)]]['worker_seconds'] += max(0.0, end - updated)
    return {'sources': sources, 'backends': backends, 'sample_stage': 'worker_committed'}


def accepted_source_steps(metadata, accepted_steps):
    """按成功合并的前缀计数，不把丢弃文件或回滚行计入吞吐"""
    counts = dict.fromkeys(COLLECTION_SOURCES, 0)
    remaining = int(accepted_steps)
    for source, length in metadata.get('segments', []):
        if source not in counts or type(length) is not int or length < 0:
            raise ValueError('采集来源片段无效')
        take = min(remaining, length)
        counts[source] += take
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError('采集来源片段与合并样本数不符')
    return counts


def rollout_chunk_path(iteration, worker_id, token, chunk_index, counts):
    """原子改名同时发布各来源行数，进程退出后无需加载张量核对配额"""
    rows = '_'.join(str(int(counts[name])) for name in COLLECTION_SOURCES)
    return f'tmp_rollout_iter_{iteration}_worker_{worker_id}_batch_{token}_{chunk_index}_rows_{rows}.pt'


def saved_collection_counts(iteration, worker_id, token):
    """只检查本次采集的完成文件名，补回落盘成功但未发送确认的行数"""
    counts = dict.fromkeys(COLLECTION_SOURCES, 0)
    for path in Path('.').glob(f'tmp_rollout_iter_{iteration}_worker_{worker_id}_batch_{token}_*_rows_*.pt'):
        values = [int(value) for value in path.stem.rsplit('_rows_', 1)[1].split('_')]
        if len(values) != len(COLLECTION_SOURCES) or any(value < 0 for value in values):
            raise ValueError('轨迹块文件名中的来源计数无效')
        for source, value in zip(COLLECTION_SOURCES, values):
            counts[source] += value
    return counts


class SampleQuotaScheduler:
    """父进程在整局边界分配来源，配额按有效模型动作而非对局胜负计算"""

    def __init__(self, target_steps, weights, *, has_history, max_episode_steps,
                 capacity, estimates=None):
        effective = dict(weights)
        if not has_history:
            effective['self'] += effective['hist']
            effective['hist'] = 0.0
        raw = {name: int(target_steps) * effective[name] for name in COLLECTION_SOURCES}
        self.targets = {name: math.floor(value) for name, value in raw.items()}
        remainder = int(target_steps) - sum(self.targets.values())
        for name in sorted(COLLECTION_SOURCES, key=lambda n: raw[n] - self.targets[n], reverse=True)[:remainder]:
            self.targets[name] += 1
        # 正比例至少尝试一局，零比例Rule补局独立于基础配额
        for name in COLLECTION_SOURCES:
            if effective[name] > 0:
                self.targets[name] = max(1, self.targets[name])
        self.rule_supplement = effective['rule'] == 0 and effective['windbot'] < 1
        if self.rule_supplement:
            self.targets['rule'] = 1
        self.capacity = int(capacity)
        self.max_episode_steps = int(max_episode_steps)
        self.counts = dict.fromkeys(COLLECTION_SOURCES, 0)
        self.reports = {}
        self.inflight = {}
        self.participants = {name: set() for name in COLLECTION_SOURCES}
        self.assignments = dict.fromkeys(COLLECTION_SOURCES, 0)
        self.previous = {}
        estimates = estimates or {}
        self.estimates = {name: {
            'rate': max(0.01, float(estimates.get(name, {}).get('rate', 1.0))),
            'episode_steps': max(1.0, float(estimates.get(name, {}).get('episode_steps', 128.0))),
        } for name in COLLECTION_SOURCES}

    def claim(self, worker_id, *, now=None):
        """用剩余预计工时均衡并发，给在途完整对局保留内存和来源配额"""
        if worker_id in self.inflight:
            raise RuntimeError('Worker尚有未结算的采集任务')
        if sum(self.counts.values()) + (len(self.inflight) + 1) * self.max_episode_steps > self.capacity:
            return None
        scores = {}
        for source in COLLECTION_SOURCES:
            tasks = [task for task in self.inflight.values() if task['source'] == source]
            remaining = self.targets[source] - self.counts[source]
            if remaining <= sum(task['reserved'] for task in tasks):
                continue
            scores[source] = remaining / self.estimates[source]['rate'] / (len(tasks) + 1)
        if not scores:
            return None
        source = max(scores, key=scores.get)
        previous = self.previous.get(worker_id)
        # 接近的优先级优先复用当前后端，避免无收益的频繁切换
        if previous in scores and scores[previous] >= scores[source] * 0.85:
            source = previous
        reserved = min(self.estimates[source]['episode_steps'],
                       self.targets[source] - self.counts[source] - sum(
                           task['reserved'] for task in self.inflight.values() if task['source'] == source))
        self.inflight[worker_id] = {'source': source, 'reserved': reserved,
                                    'started': time.monotonic() if now is None else now}
        self.previous[worker_id] = source
        self.participants[source].add(worker_id)
        self.assignments[source] += 1
        return source

    def report(self, worker_id, report, *, now=None):
        """累计提交与已保存计数分开，崩溃时只撤销尚未落盘的配额进度"""
        previous = self.reports.get(worker_id, {})
        counts = report.get('counts', {})
        for source in COLLECTION_SOURCES:
            self.counts[source] += int(counts.get(source, 0)) - int(previous.get('counts', {}).get(source, 0))
        task = self.inflight.pop(worker_id, None)
        if task:
            source = task['source']
            steps = int(counts.get(source, 0)) - int(previous.get('counts', {}).get(source, 0))
            elapsed = max(1e-6, (time.monotonic() if now is None else now) - task['started'])
            if steps > 0:
                estimate = self.estimates[source]
                estimate['rate'] = 0.75 * estimate['rate'] + 0.25 * (steps / elapsed)
                estimate['episode_steps'] = 0.75 * estimate['episode_steps'] + 0.25 * steps
        self.reports[worker_id] = report

    def worker_exited(self, worker_id, saved_counts=None):
        """回收未完成任务，不把丢失的内存中轨迹当作达标样本"""
        previous = self.reports.get(worker_id, {})
        saved = previous.get('saved_counts', {}) if saved_counts is None else saved_counts
        self.report(worker_id, {'counts': saved, 'saved_counts': saved})

    @property
    def stopped(self):
        """在途任务尚未结束时保留空闲Worker，异常释放后仍可重新分配"""
        if self.inflight:
            return False
        return (all(self.counts[name] >= self.targets[name] for name in COLLECTION_SOURCES)
                or sum(self.counts.values()) + self.max_episode_steps > self.capacity)

    def summary(self):
        return {'target_steps': dict(self.targets), 'committed_steps': dict(self.counts),
                'participating_workers': {name: len(ids) for name, ids in self.participants.items()},
                'inflight_workers': {name: sum(t['source'] == name for t in self.inflight.values())
                                     for name in COLLECTION_SOURCES},
                'assigned_games': dict(self.assignments), 'rule_supplement': self.rule_supplement,
                'capacity_limited': sum(self.counts.values()) + self.max_episode_steps > self.capacity}


class CollectionTaskClient:
    """每局边界通过小型管道领取任务，不搬运观测或策略输出"""

    def __init__(self, connection, initial, telemetry, deadline):
        self.connection = connection
        self.initial = initial
        self.telemetry = telemetry
        self.deadline = deadline
        self.saved_counts = dict.fromkeys(COLLECTION_SOURCES, 0)
        self.segment_start = 0

    def snapshot(self, kind):
        return {'kind': kind, 'counts': {name: int(row['committed_steps'])
                for name, row in self.telemetry.sources.items()}, 'saved_counts': dict(self.saved_counts)}

    def next_task(self):
        if self.initial is not None:
            initial, self.initial = self.initial, None
            return initial
        self.connection.send(self.snapshot('ready'))
        while time.monotonic() < self.deadline:
            if self.connection.poll(min(0.25, max(0.0, self.deadline - time.monotonic()))):
                return self.connection.recv()
        return None

    def mark_saved(self):
        self.saved_counts = self.snapshot('saved')['counts']
        self.segment_start = len(self.telemetry.segments)

    def close(self):
        try:
            self.connection.send(self.snapshot('finished'))
        except (OSError, EOFError):
            pass
        self.connection.close()


class RolloutCursor:
    """跟踪单个 Worker 的暂存行和已提交采样行"""

    def __init__(self):
        """初始化写入、提交和当前对局起点游标"""
        self.write_pos = 0
        self.committed_pos = 0
        self.collected_steps = 0
        self.episode_start_pos = 0
        self._episode_start_steps = 0
        self._episode_open = False

    @property
    def episode_open(self):
        """返回当前是否存在尚未提交或回滚的对局"""
        return self._episode_open

    @property
    def next_write_pos(self):
        """返回当前对局下一条观测应写入的位置"""
        if not self._episode_open:
            raise RuntimeError("cannot write a rollout row outside an episode")
        return self.write_pos

    def begin_episode(self):
        """从已提交位置开始一局新的暂存采样"""
        if self._episode_open:
            raise RuntimeError("previous rollout episode is still open")
        if self.write_pos != self.committed_pos:
            raise RuntimeError(
                f"uncommitted rollout rows remain: write={self.write_pos}, "
                f"committed={self.committed_pos}"
            )

        self.episode_start_pos = self.write_pos
        self._episode_start_steps = self.collected_steps
        self._episode_open = True

    def record_step(self):
        """记录一条已经完整写入的暂存观测"""
        index = self.next_write_pos
        self.write_pos += 1
        self.collected_steps += 1
        return index

    def validate_episode(self, trajectory_length):
        """校验观测写入行数与轨迹标签数量一致"""
        if not self._episode_open:
            raise RuntimeError("no rollout episode is open")

        written_rows = self.write_pos - self.episode_start_pos
        if trajectory_length != written_rows:
            raise RuntimeError(
                "rollout observation/trajectory mismatch: "
                f"observations={written_rows}, trajectory={trajectory_length}, "
                f"episode_start={self.episode_start_pos}, write={self.write_pos}"
            )

    def commit_episode(self, trajectory_length):
        """校验并提交当前对局的全部采样"""
        self.validate_episode(trajectory_length)
        self.committed_pos = self.write_pos
        self._episode_open = False

    def rollback_episode(self):
        """回滚当前异常对局占用的全部暂存行"""
        if not self._episode_open:
            raise RuntimeError("no rollout episode is open")

        self.write_pos = self.episode_start_pos
        self.collected_steps = self._episode_start_steps
        self._episode_open = False
