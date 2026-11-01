# 本文件定义中心推理的 ZMQ 请求协议，并校验共享内存结果是否属于当前请求。


class InferenceProtocolError(RuntimeError):
    """表示中心推理响应或共享结果与当前请求不匹配"""


def encode_inference_request(worker_id, request_id, *, phase=None, submitted_ns=None, input_fill_ns=0):
    """将 Worker 编号和请求号编码为服务端可解析的字节串"""
    if phase is not None:
        if phase not in ("Pass1", "Pass2") or type(submitted_ns) is not int or submitted_ns <= 0 or type(input_fill_ns) is not int or input_fill_ns < 0:
            raise InferenceProtocolError("invalid inference diagnostic metadata")
        return f"D1:{int(worker_id)}:{int(request_id)}:{phase}:{submitted_ns}:{input_fill_ns}".encode("ascii")
    return f"{int(worker_id)}:{int(request_id)}".encode("ascii")


def decode_inference_request(payload):
    """解析中心推理请求并拒绝格式错误或负数编号"""
    worker_id, request_id, _ = decode_inference_envelope(payload)
    return worker_id, request_id


def decode_inference_envelope(payload):
    """兼容既有请求，诊断扩展只携带同机单调时钟和阶段标签"""
    try:
        fields = payload.decode("ascii").split(":")
        metadata = None
        if fields[0] == "D1" and len(fields) == 6:
            _, worker_text, request_text, phase, submitted, filled = fields
            submitted_ns, input_fill_ns = int(submitted), int(filled)
            if phase not in ("Pass1", "Pass2") or submitted_ns <= 0 or input_fill_ns < 0:
                raise ValueError("invalid diagnostic metadata")
            metadata = {"phase": phase, "submitted_ns": submitted_ns, "input_fill_ns": input_fill_ns}
        elif len(fields) == 2:
            worker_text, request_text = fields
        else:
            raise ValueError("invalid request field count")
        worker_id = int(worker_text)
        request_id = int(request_text)
    except (AttributeError, UnicodeDecodeError, ValueError) as exc:
        raise InferenceProtocolError("invalid inference request payload") from exc

    if worker_id < 0 or request_id < 0:
        raise InferenceProtocolError("inference request ids must be non-negative")
    return worker_id, request_id, metadata


def encode_inference_success(request_id):
    """编码带请求号的成功响应"""
    return f"OK:{int(request_id)}".encode("ascii")


def encode_inference_error(request_id, reason):
    """编码服务端拒绝请求时的错误响应"""
    return f"ERR:{int(request_id)}:{reason}".encode("ascii", errors="replace")


def validate_inference_success(reply, request_id):
    """确认 ZMQ 响应确实对应当前请求"""
    expected = encode_inference_success(request_id)
    if reply != expected:
        raise InferenceProtocolError(
            f"unexpected inference reply: expected {expected!r}, got {reply!r}"
        )


def request_shared_inference_result(
    socket,
    worker_id,
    request_id,
    response_id_slot,
    result_slot,
    *,
    phase=None,
    input_fill_ns=0,
):
    """请求中心推理，并仅在响应号和共享槽完成号一致时复制结果"""
    if response_id_slot is None:
        raise InferenceProtocolError("missing shared inference response id slot")

    if phase is None:
        payload = encode_inference_request(worker_id, request_id)
    else:
        import time
        payload = encode_inference_request(worker_id, request_id, phase=phase,
            submitted_ns=time.perf_counter_ns(), input_fill_ns=input_fill_ns)
    socket.send(payload)
    reply = socket.recv()
    validate_inference_success(reply, request_id)

    completed_request_id = int(response_id_slot.item())
    if completed_request_id != request_id:
        raise InferenceProtocolError(
            "shared inference result id mismatch: "
            f"expected {request_id}, got {completed_request_id}"
        )

    # 返回独立快照，避免后续请求覆盖共享槽时影响当前决策
    return result_slot.detach().clone()


class PackedObservationLayout:
    """按原始dtype连续存储既有字段，只改变搬运布局，不量化或修改观测"""

    def __init__(self, specifications):
        import math
        self.fields = []
        self.widths = {}
        for name, (shape, dtype) in specifications.items():
            shape = tuple(shape)
            size = math.prod(shape)
            offset = self.widths.get(dtype, 0)
            self.fields.append((name, shape, dtype, offset, size))
            self.widths[dtype] = offset + size

    def allocate(self, rows, *, shared=False, pinned=False, device='cpu'):
        """每种dtype只分配一块CPU存储，Worker字段视图共用同一底层内存"""
        import torch
        blocks = {dtype: torch.zeros((rows, width), dtype=dtype, pin_memory=pinned, device=device)
                  for dtype, width in self.widths.items()}
        if shared:
            for block in blocks.values():
                block.share_memory_()
        return blocks

    def worker_views(self, blocks, rows):
        """保持Worker原来的字段字典接口与精度，避免改动编码器和轨迹"""
        return [{name: blocks[dtype][row, offset:offset + size].view(shape)
                 for name, shape, dtype, offset, size in self.fields} for row in range(rows)]

    def batch_views(self, blocks, rows, *, contiguous_floats=False):
        """设备侧拆回原字段，浮点输入保持原连续布局以复用相同线性算子"""
        result = {}
        for name, shape, dtype, offset, size in self.fields:
            value = blocks[dtype][:rows, offset:offset + size].view(rows, *shape)
            result[name] = value.contiguous() if contiguous_floats and value.is_floating_point() else value
        return result

    def gather(self, source, staging, workers):
        """只聚合本批有效Worker，顺序与原请求列表严格一致"""
        for dtype, destination in staging.items():
            for row, worker in enumerate(workers):
                destination[row].copy_(source[dtype][worker])

    def transfer(self, staging, rows, device, *, non_blocking, device_blocks=None):
        """每种dtype进行一次设备搬运，再还原原字段形状"""
        if device_blocks is None:
            blocks = {dtype: block[:rows].to(device, non_blocking=non_blocking).detach()
                      for dtype, block in staging.items()}
        else:
            for dtype, block in staging.items():
                device_blocks[dtype][:rows].copy_(block[:rows], non_blocking=non_blocking)
            blocks = device_blocks
        return self.batch_views(blocks, rows, contiguous_floats=True)


class CentralInferenceDiagnostics:
    """有限窗口的中央路径计时，不读取观测内容或消耗模型随机数"""

    def __init__(self, max_batches):
        import threading
        if type(max_batches) is not int or not 1 <= max_batches <= 10000:
            raise ValueError("inference profile batches must be 1..10000")
        self.max_batches = max_batches
        self._lock = threading.Lock()
        self._records = []
        self._iteration = None
        self._active = False

    def start(self, iteration):
        """仅在本轮Worker启动前重置窗口"""
        with self._lock:
            self._records = []
            self._iteration = int(iteration)
            self._active = True

    @property
    def active(self):
        return self._active and len(self._records) < self.max_batches

    def begin_batch(self, requests, received, metadata, device):
        """达到上限后不再创建计时器或CUDA事件"""
        with self._lock:
            if not self.active or not all(wid in received for wid in requests):
                return None
            record = _InferenceBatchMeasurement(requests, received, metadata, device)
            self._records.append(record)
            return record

    def finish(self):
        """Worker回收后一次性整理设备时间，禁止在每个算子间强制同步"""
        with self._lock:
            if not self._active:
                return None
            self._active = False
            records, self._records = self._records, []
            iteration = self._iteration
        from collections import Counter
        batches, requests, incomplete = [], [], 0
        for record in records:
            if not record.complete:
                incomplete += 1
                continue
            batch, rows = record.summarize()
            batches.append(batch)
            requests.extend(rows)
        def summary(values):
            values = sorted(values)
            if not values:
                return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
            def percentile(fraction):
                at = (len(values) - 1) * fraction
                low = int(at)
                high = min(low + 1, len(values) - 1)
                return values[low] + (values[high] - values[low]) * (at - low)
            return {"count": len(values), "mean": sum(values) / len(values),
                    "p50": percentile(0.5), "p95": percentile(0.95), "max": values[-1]}
        return {"format_version": 1, "iteration": iteration, "max_batches": self.max_batches,
            "measured_batches": len(batches), "incomplete_batches": incomplete,
            "measured_requests": len(requests),
            "cuda_timing_failures": sum(record.cuda_error is not None for record in records),
            "batch_size_histogram": dict(Counter(str(batch['batch_size']) for batch in batches)),
            "batch_ms": {name: summary([batch[name] for batch in batches if name in batch])
                         for name in sorted({key for batch in batches for key in batch if key != 'batch_size'})},
            "requests": {phase: {"count": len(group), "ms": {
                name: summary([row[name] for row in group if row.get(name) is not None])
                for name in ('input_fill', 'submit_to_receive', 'server_wait', 'submit_to_reply')}}
                for phase in ('Pass1', 'Pass2', 'unlabelled')
                for group in [[row for row in requests if row['phase'] == phase]]},
            "measurement": {"clock": "same_host_perf_counter_ns",
                "cpu": "host_stage_wall_time", "cuda": "stream_intervals_including_launch_gaps",
                "add_cpu_and_cuda": False, "includes_client_resume": False,
                "window": "first_valid_batches_per_iteration", "changes_model_or_rng": False}}


class _InferenceBatchMeasurement:
    """保存单批有限时间戳，不保留输入张量、动作或概率"""

    def __init__(self, requests, received, metadata, device):
        import time
        self.requests = [(wid, received[wid], dict(metadata.get(wid) or {})) for wid in requests]
        self.cpu = {"dispatch": time.perf_counter_ns()}
        self.cuda = {}
        self.cuda_error = None
        self.device = device
        self.complete = False

    def mark(self, name):
        """CPU记录提交耗时，CUDA事件仅在显式诊断窗口内创建"""
        import time
        self.cpu[name] = time.perf_counter_ns()
        if self.device.type == "cuda" and self.cuda_error is None and name in ('gathered', 'transferred', 'forwarded', 'sampled', 'returned'):
            import torch
            try:
                with torch.cuda.device(self.device):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    self.cuda[name] = event
            except Exception as error:
                # 可选计时器失效时仍按原流程推理，报告中单独计数
                self.cuda_error = type(error).__name__
                self.cuda.clear()
        if name == 'published':
            self.complete = True

    def summarize(self):
        """在采集屏障后查询已完成的CUDA事件"""
        stages = ('dispatch', 'gathered', 'transferred', 'forwarded', 'sampled', 'returned', 'published')
        names = ('gather', 'h2d_submit', 'forward_submit', 'sample_submit', 'result_copy', 'publish')
        batch = {'batch_size': len(self.requests)}
        for start, end, name in zip(stages, stages[1:], names):
            batch['cpu_' + name] = (self.cpu[end] - self.cpu[start]) / 1e6
        if self.cuda:
            self.cuda['returned'].synchronize()
            for start, end, name in zip(stages[1:5], stages[2:6], ('h2d', 'forward', 'sample', 'd2h')):
                batch['cuda_' + name] = self.cuda[start].elapsed_time(self.cuda[end])
        rows = []
        for _, received, metadata in self.requests:
            submitted = metadata.get('submitted_ns')
            valid_clock = submitted is not None and submitted <= received <= self.cpu['dispatch']
            rows.append({'phase': metadata.get('phase', 'unlabelled'),
                'input_fill': metadata.get('input_fill_ns', 0) / 1e6 if metadata else None,
                'submit_to_receive': (received - submitted) / 1e6 if valid_clock else None,
                'server_wait': (self.cpu['dispatch'] - received) / 1e6,
                'submit_to_reply': (self.cpu['published'] - submitted) / 1e6 if valid_clock else None})
        return batch, rows
