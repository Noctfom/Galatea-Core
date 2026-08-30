# 本文件实现 WebUI 所启动 Galatea 进程的身份标记、识别与安全清理。

import os

import psutil


MANAGED_PROCESS_MARKER = "GALATEA_MANAGED_PROCESS"
MANAGED_PROJECT_ROOT = "GALATEA_MANAGED_PROJECT_ROOT"


def normalize_project_root(project_root):
    """将项目目录规范化为可跨平台比较的绝对路径"""
    return os.path.normcase(os.path.realpath(os.fspath(project_root)))


def build_managed_process_env(project_root, base_env=None):
    """为 WebUI 启动的进程构造带项目归属标记的环境变量"""
    environment = dict(os.environ if base_env is None else base_env)
    environment[MANAGED_PROCESS_MARKER] = "1"
    environment[MANAGED_PROJECT_ROOT] = normalize_project_root(project_root)
    return environment


def process_matches_project(process, project_root):
    """确认进程是否带有当前 Galatea 项目的托管标记"""
    try:
        environment = process.environ()
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False
    marked_root = environment.get(MANAGED_PROJECT_ROOT)
    return (
        environment.get(MANAGED_PROCESS_MARKER) == "1"
        and bool(marked_root)
        and normalize_project_root(marked_root)
        == normalize_project_root(project_root)
    )


def process_identity_matches(process, expected_create_time):
    """用创建时间防止已结束任务的 PID 被系统复用后误杀新进程"""
    if expected_create_time is None:
        return False
    try:
        return abs(process.create_time() - float(expected_create_time)) < 0.01
    except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError, ValueError):
        return False


def purge_managed_processes(
    project_root,
    *,
    known_root_pid=None,
    known_root_create_time=None,
    current_pid=None,
    grace_seconds=2.0,
):
    """只清理当前项目标记进程及经身份校验的已登记进程树"""
    current_pid = os.getpid() if current_pid is None else int(current_pid)
    matched = {}

    for process in psutil.process_iter(["pid"]):
        if process.pid == current_pid:
            continue
        if process_matches_project(process, project_root):
            matched[process.pid] = process

    if known_root_pid:
        try:
            root_process = psutil.Process(int(known_root_pid))
            if (
                root_process.pid != current_pid
                and process_identity_matches(root_process, known_root_create_time)
            ):
                matched[root_process.pid] = root_process
                for child in root_process.children(recursive=True):
                    if child.pid != current_pid:
                        matched[child.pid] = child
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError, TypeError):
            pass

    matched_pids = sorted(matched)
    requested = []
    failed = []
    for pid in matched_pids:
        try:
            matched[pid].terminate()
            requested.append(matched[pid])
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError) as error:
            failed.append({"pid": pid, "error": str(error)})

    _, alive = psutil.wait_procs(requested, timeout=max(0.0, float(grace_seconds)))
    forced = []
    for process in alive:
        try:
            process.kill()
            forced.append(process.pid)
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError) as error:
            failed.append({"pid": process.pid, "error": str(error)})

    if alive:
        psutil.wait_procs(alive, timeout=1.0)

    return {
        "matched_pids": matched_pids,
        "forced_pids": sorted(forced),
        "failed": failed,
    }
