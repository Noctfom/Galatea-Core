# 本文件实现 Galatea 训练进程的跨平台单实例互斥锁

import json
import os
from datetime import datetime, timezone
from pathlib import Path


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
        self._metadata = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "run_id": None,
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

    def _write_metadata(self):
        """在持锁状态下写入 PID、启动时间和 run_id"""
        payload = json.dumps(self._metadata, ensure_ascii=False) + "\n"
        temp_path = Path(f"{self.metadata_path}.{os.getpid()}.tmp")
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
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

    def release(self):
        """释放训练锁；锁文件保留以避免删除与重新创建之间的竞态"""
        if not self._locked or self._handle is None:
            return

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
        self.release()
        return False
