import sys
import os
import datetime
import glob
import time

class DualLogger:
    def __init__(self, log_dir="./system_logs", prefix="main", max_logs=30):
        os.makedirs(log_dir, exist_ok=True)
        
        # 尝试从环境变量获取全局统一的训练批次 ID
        run_id = os.environ.get('GALATEA_RUN_ID', None)
        
        if run_id:
            # 如果有 run_id，强制使用统一命名 (如 worker_0_20231025_120000.log)
            self.log_file = os.path.join(log_dir, f"{prefix}_{run_id}.log")
        else:
            # 如果没有 (比如单独测试 worker)，则按当前时间戳生成
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
            
        # 自动清理旧日志 (排除当前正在使用的这个文件，防误删)
        try:
            existing_logs = sorted(glob.glob(os.path.join(log_dir, f"{prefix}_*.log")), key=os.path.getmtime)
            existing_logs = [f for f in existing_logs if os.path.abspath(f) != os.path.abspath(self.log_file)]
            
            while len(existing_logs) >= max_logs:
                oldest_log = existing_logs.pop(0)
                os.remove(oldest_log)
        except Exception:
            pass

        self.terminal = sys.stdout
        
        # "a" 追加模式是灵魂,同一次训练的后续 Iteration 会自动在文件末尾写
        self.file = open(self.log_file, "a", encoding="utf-8")
        
        # 写入分界线，方便你在几万行日志里看出这是哪一轮(Iteration)重启的
        start_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.file.write(f"\n\n{'='*60}\n🚀 [Process Spawned] {start_time} - Worker Active \n{'='*60}\n")
        
        # 性能核心：记录上一次 flush 的时间
        self.last_flush_time = time.time()

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        
        # Smart Flush 策略：
        # 1. 强制落盘条件：如果打印的内容包含报错关键字，立刻落盘
        # 2. 性能落盘条件：如果距离上次写入超过 1.0 秒，才落盘一次。防止高频 print 卡死 CPU
        now = time.time()
        if "Error" in message or "Exception" in message or "Traceback" in message or "⚠️" in message or "❌" in message or (now - self.last_flush_time > 1.0):
            self.file.flush()
            self.last_flush_time = now

    def flush(self):
        self.terminal.flush()
        self.file.flush()

class DualErrorLogger:
    def __init__(self, dual_logger):
        self.terminal = sys.stderr
        self.file = dual_logger.file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        # stderr 的内容通常是致命报错，必须 100% 实时落盘
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

def setup_global_logger(prefix="trainer"):
    logger = DualLogger(log_dir="./system_logs", prefix=prefix)
    sys.stdout = logger
    sys.stderr = DualErrorLogger(logger)
    # 这条信息一定会触发 1 秒规则被记录下来
    print(f"📡 [System] 高性能日志拦截器已启动 -> {logger.log_file}")