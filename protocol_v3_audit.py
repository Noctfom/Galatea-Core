# 本文件负责采集 Model Protocol V3 观测与效果槽映射审计，并输出 WebUI 可读报告

import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from checkpoint_utils import MODEL_PROTOCOL_VERSION
from effect_slot_binding import (
    build_runtime_effect_binding_catalog,
    extract_runtime_effect_bindings,
)


AUDIT_SCHEMA_VERSION = 2
MAX_EFFECT_OBSERVATIONS = 2048
MAX_AUDIT_REPORT_BYTES = 16 * 1024 * 1024
AUDIT_FLUSH_INTERVAL_SECONDS = 30.0
AUDIT_DIRECTORY = Path("system_logs") / "protocol_v3_audit"
_PROJECT_ROOT = Path(__file__).resolve().parent
_GLOBAL_RECORDER = None
_SEMANTIC_CARD_SLOTS = {}
_RUNTIME_DESC_SLOTS = {}


def _safe_label(value, fallback):
    """把运行来源压缩成只适合文件名使用的短标签"""
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return label[:64] or fallback


def register_semantic_audit_catalog(knowledge_base):
    """登记知识库中每张卡实际进入模型的效果槽，供运行时审计比对"""
    global _SEMANTIC_CARD_SLOTS, _RUNTIME_DESC_SLOTS
    catalog = {}
    if isinstance(knowledge_base, dict):
        for raw_card_id, card_data in knowledge_base.items():
            if not str(raw_card_id).isdigit() or not isinstance(card_data, dict):
                continue
            slots = set()
            effects = card_data.get("effects", [])
            if not isinstance(effects, list):
                continue
            for fallback_slot, effect in enumerate(effects, start=1):
                if not isinstance(effect, dict):
                    continue
                try:
                    slot = int(effect.get("slot", fallback_slot))
                except (TypeError, ValueError):
                    continue
                if 1 <= slot <= 8:
                    slots.add(slot - 1)
            catalog[int(raw_card_id)] = tuple(sorted(slots))
    _SEMANTIC_CARD_SLOTS = catalog
    _RUNTIME_DESC_SLOTS = build_runtime_effect_binding_catalog(knowledge_base)
    inspect_lua_description_slots.cache_clear()


@lru_cache(maxsize=4096)
def inspect_lua_description_slots(card_code):
    """解析单卡 initial_effect，返回完整运行时 desc 对应的 Lua 创建槽位。"""
    card_code = int(card_code) & 0x7FFFFFFF
    candidates = (
        _PROJECT_ROOT / "script" / f"c{card_code}.lua",
        _PROJECT_ROOT / "script" / "official" / f"c{card_code}.lua",
    )
    script_path = next((path for path in candidates if path.is_file()), None)
    if script_path is None:
        return {}
    try:
        content = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    extracted = extract_runtime_effect_bindings(content, card_code)
    return {
        int(runtime_desc): (int(slot_index),)
        for runtime_desc, slot_index in extracted["bindings"].items()
    }


def _classify_effect_slot(card_code, description_index, runtime_desc):
    """用内存中的 Lua 对象绑定分类运行时效果，不在对局期间读取文件。"""
    pure_code = int(card_code) & 0x7FFFFFFF
    semantic_slots = _SEMANTIC_CARD_SLOTS.get(pure_code)
    resolved_slot = _RUNTIME_DESC_SLOTS.get(
        (pure_code, int(runtime_desc) & 0xFFFFFFFF)
    )
    if semantic_slots is None:
        status = "missing_card_semantics"
    elif int(runtime_desc) == 0:
        status = "unverified"
    elif resolved_slot is None:
        status = "binding_missing"
    elif resolved_slot not in semantic_slots:
        status = "explicit_slot_missing_semantics"
    else:
        status = "exact"
    return status, semantic_slots or (), resolved_slot


def enrich_effect_slot_observation(observation):
    """在 WebUI 读取报告时用当前 Lua 对象绑定复核审计结论。"""
    result = dict(observation or {})
    try:
        card_code = int(result.get("card_code", 0)) & 0x7FFFFFFF
        description_index = int(result.get("description_index", -1))
        runtime_desc = int(result.get("runtime_desc", 0)) & 0xFFFFFFFF
    except (TypeError, ValueError):
        result["status"] = "invalid_observation"
        return result
    base_status = str(result.get("status", "unverified"))
    explicit_slots = (
        inspect_lua_description_slots(card_code).get(runtime_desc, ())
        if runtime_desc != 0
        else ()
    )
    result["explicit_lua_slots"] = list(explicit_slots)
    resolved_slot = result.get("resolved_effect_slot")
    if resolved_slot is None and len(explicit_slots) == 1:
        resolved_slot = explicit_slots[0]
    result["resolved_effect_slot"] = resolved_slot
    if base_status != "missing_card_semantics" and explicit_slots:
        semantic_slots = {
            int(slot) for slot in result.get("semantic_slots", [])
        }
        if not semantic_slots.intersection(explicit_slots):
            result["status"] = "explicit_slot_missing_semantics"
        else:
            result["status"] = "exact" if resolved_slot in explicit_slots else "mismatch"
    elif base_status != "missing_card_semantics" and runtime_desc != 0:
        result["status"] = "binding_missing"
    return result


class ProtocolV3AuditRecorder:
    """在单个进程内聚合协议观测，周期性原子写入一个 JSON 报告"""

    def __init__(self, source, run_label=None, output_directory=AUDIT_DIRECTORY):
        self.source = _safe_label(source, "process")
        self.run_label = _safe_label(run_label, "run")
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = (
            f"audit_{self.source}_{self.run_label}_{timestamp}_{os.getpid()}.json"
        )
        self.output_directory = Path(output_directory).resolve()
        self.report_path = self.output_directory / filename
        self.counters = Counter()
        self.message_counts = Counter()
        self.mapping_counts = Counter()
        self.effect_observations = {}
        self.last_flush_time = time.monotonic()
        self.dirty = False
        self.lock = threading.Lock()

    @staticmethod
    def _location_is_valid(location):
        """判断连锁位置是否为 Core 的单一区域位"""
        location = int(location)
        return location in (1, 2, 4, 8, 16, 32, 64, 128)

    def record_message(self, msg_type):
        """累计一条 Core 消息，并低频触发报告落盘"""
        with self.lock:
            self.counters["messages_total"] += 1
            self.message_counts[str(int(msg_type))] += 1
            self.dirty = True
            should_flush = (
                self.counters["messages_total"] % 256 == 0
                and time.monotonic() - self.last_flush_time
                >= AUDIT_FLUSH_INTERVAL_SECONDS
            )
        if should_flush:
            self.flush()

    def record_chain(
        self,
        *,
        code,
        desc,
        chain_index,
        chain_depth,
        handler_controller,
        handler_location,
        handler_sequence,
        trigger_controller,
        trigger_location,
        trigger_sequence,
    ):
        """累计连锁结构质量和运行时效果描述到语义槽的映射结果"""
        pure_code = int(code) & 0x7FFFFFFF
        runtime_desc = int(desc) & 0xFFFFFFFF
        description_index = runtime_desc & 0xF
        status, semantic_slots, resolved_slot = _classify_effect_slot(
            pure_code,
            description_index,
            runtime_desc,
        )
        with self.lock:
            self.counters["chain_events"] += 1
            self.counters["max_chain_depth"] = max(
                self.counters["max_chain_depth"], int(chain_depth)
            )
            if int(chain_depth) > 12:
                self.counters["chain_depth_overflow"] += 1
            if int(chain_index) <= 0:
                self.counters["invalid_chain_index"] += 1
            for label, controller in (
                ("handler", handler_controller),
                ("trigger", trigger_controller),
            ):
                if int(controller) not in (0, 1):
                    self.counters[f"invalid_{label}_controller"] += 1
            for label, location in (
                ("handler", handler_location),
                ("trigger", trigger_location),
            ):
                if int(location) == 0:
                    self.counters[f"unknown_{label}_location"] += 1
                elif not self._location_is_valid(location):
                    self.counters[f"invalid_{label}_location"] += 1
            for label, sequence in (
                ("handler", handler_sequence),
                ("trigger", trigger_sequence),
            ):
                if not 0 <= int(sequence) <= 31:
                    self.counters[f"invalid_{label}_sequence"] += 1

            self.mapping_counts[status] += 1
            observation_key = f"{pure_code}:{runtime_desc}:{resolved_slot}:{status}"
            observation = self.effect_observations.get(observation_key)
            if observation is None:
                if len(self.effect_observations) >= MAX_EFFECT_OBSERVATIONS:
                    self.counters["dropped_distinct_effect_observations"] += 1
                else:
                    observation = {
                        "card_code": pure_code,
                        "description_index": description_index,
                        "runtime_desc": runtime_desc,
                        "status": status,
                        "resolved_effect_slot": resolved_slot,
                        "semantic_slots": list(semantic_slots),
                        "explicit_lua_slots": (
                            [] if resolved_slot is None else [resolved_slot]
                        ),
                        "count": 0,
                    }
                    self.effect_observations[observation_key] = observation
            if observation is not None:
                observation["count"] += 1
            self.dirty = True

    def _payload(self):
        """构造稳定、可独立读取的审计报告对象"""
        return {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "model_protocol_version": MODEL_PROTOCOL_VERSION,
            "source": self.source,
            "run_label": self.run_label,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "counters": dict(sorted(self.counters.items())),
            "message_counts": dict(
                sorted(self.message_counts.items(), key=lambda item: int(item[0]))
            ),
            "mapping_counts": dict(sorted(self.mapping_counts.items())),
            "effect_observations": sorted(
                self.effect_observations.values(),
                key=lambda item: (
                    item["status"] == "exact",
                    -int(item["count"]),
                    int(item["card_code"]),
                ),
            ),
        }

    def flush(self, force=False):
        """将有变化的审计结果原子写入磁盘"""
        with self.lock:
            if not self.dirty and not force:
                return self.report_path if self.report_path.exists() else None
            payload = self._payload()
            self.output_directory.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix=f".{self.report_path.name}.",
                    suffix=".tmp",
                    dir=self.output_directory,
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    json.dump(payload, stream, ensure_ascii=False, indent=2)
                os.replace(temporary_path, self.report_path)
                temporary_path = None
                self.dirty = False
                self.last_flush_time = time.monotonic()
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
        return self.report_path


def configure_protocol_v3_audit(source, run_label=None, output_directory=AUDIT_DIRECTORY):
    """为当前训练 Worker、竞技场或自检进程启用一份独立审计报告"""
    global _GLOBAL_RECORDER
    _GLOBAL_RECORDER = ProtocolV3AuditRecorder(
        source,
        run_label=run_label,
        output_directory=output_directory,
    )
    return _GLOBAL_RECORDER.report_path


def disable_protocol_v3_audit(flush=False):
    """关闭当前进程审计；测试或嵌入式调用可选择先刷新已有结果"""
    global _GLOBAL_RECORDER
    recorder = _GLOBAL_RECORDER
    _GLOBAL_RECORDER = None
    if flush and recorder is not None:
        try:
            return recorder.flush(force=True)
        except Exception:
            return None
    return None


def record_protocol_message(msg_type):
    """在审计已启用时记录一条消息，审计异常不得影响正常对局"""
    if _GLOBAL_RECORDER is None:
        return
    try:
        _GLOBAL_RECORDER.record_message(msg_type)
    except Exception:
        return


def record_protocol_chain(**details):
    """在审计已启用时记录一条连锁，审计异常不得影响正常对局"""
    if _GLOBAL_RECORDER is None:
        return
    try:
        _GLOBAL_RECORDER.record_chain(**details)
    except Exception:
        return


def flush_protocol_v3_audit(force=False):
    """立即刷新当前进程的审计报告"""
    if _GLOBAL_RECORDER is None:
        return None
    try:
        return _GLOBAL_RECORDER.flush(force=force)
    except Exception:
        return None


def load_protocol_v3_audit_reports(directory=AUDIT_DIRECTORY, limit=200):
    """安全读取最近的 V3 审计报告，供 WebUI 聚合展示"""
    root = Path(directory).resolve()
    if not root.is_dir():
        return []
    reports = []
    paths = sorted(
        root.glob("audit_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: max(1, min(int(limit), 1000))]
    for path in paths:
        try:
            if path.is_symlink() or path.stat().st_size > MAX_AUDIT_REPORT_BYTES:
                continue
            with open(path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict):
                continue
            if payload.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
                continue
            payload["_path"] = str(path)
            reports.append(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return reports
