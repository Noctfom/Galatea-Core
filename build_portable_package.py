# 本文件校验一键包环境并生成带版本号的 Galatea-Core 便携 ZIP。

import argparse
import fnmatch
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT_NAME = "Galatea_Core"
EXCLUDED_TOP_LEVEL_DIRECTORIES = {
    ".git",
    ".vscode",
    "__pycache__",
    "ai_thoughts",
    "arena_benchmarks",
    "deploy_packages",
    "MDPro3-master-Tools-YGO Classes-ocgcore",
    "models",
    "ocgcore_linux_build",
    "ocgcore_windows_build",
    "replay_data",
    "replays",
    "runs",
    "system_logs",
    "tests",
    "web_data",
    "ygopro-core-master",
    "ygopro_linux_build",
    "ygopro_windows_build",
}
EXCLUDED_FILE_NAMES = {
    ".galatea_train.lock",
    ".galatea_train.lock.owner.json",
    "debug.log",
}
EXCLUDED_FILE_PATTERNS = (
    "Galatea_Core_V*.zip",
    "crash_report*",
    "tmp_rollout_*.pt",
    "tmp_rollout_*.pt.tmp",
    "tmp_weights_*.pt",
    "*.pyc",
)
STORED_SUFFIXES = {
    ".cdb",
    ".data",
    ".dll",
    ".npy",
    ".onnx",
    ".pth",
    ".pt",
    ".pyd",
    ".so",
    ".zip",
}


def read_release_version(version_path=None):
    """读取并校验用于压缩包文件名的三段式版本号"""
    version_path = Path(version_path or PROJECT_ROOT / "version.txt")
    version = version_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"version.txt 不是有效三段式版本号: {version!r}")
    return version


def is_excluded_file(path):
    """判断运行产物、日志、缓存或旧发布包是否应排除"""
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def iter_package_files(project_root=None):
    """遍历源码与一键环境，同时跳过开发目录和用户运行数据"""
    project_root = Path(project_root or PROJECT_ROOT).resolve()
    for current_root, directory_names, file_names in os.walk(project_root):
        current_path = Path(current_root)
        relative_current = current_path.relative_to(project_root)

        if relative_current == Path("."):
            directory_names[:] = [
                name
                for name in directory_names
                if name not in EXCLUDED_TOP_LEVEL_DIRECTORIES
            ]
        directory_names[:] = [
            name
            for name in directory_names
            if name != "__pycache__"
            and not (current_path / name).is_symlink()
        ]

        for file_name in file_names:
            path = current_path / file_name
            if path.is_symlink():
                raise ValueError(f"一键包不允许包含符号链接: {path}")
            if is_excluded_file(path):
                continue
            yield path


def validate_portable_environment(require_cuda=True):
    """使用便携解释器执行依赖、资源、内核与 CUDA 发布检查"""
    portable_python = PROJECT_ROOT / "python_env" / "python.exe"
    if not portable_python.is_file():
        raise FileNotFoundError("缺少一键包解释器: python_env/python.exe")

    command = [
        str(portable_python),
        "-X",
        "utf8",
        str(PROJECT_ROOT / "environment_setup.py"),
        "--verify-imports",
        "--verify-runtime-assets",
        "--require-portable-python",
    ]
    if require_cuda:
        command.append("--require-cuda")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError("一键包环境检查未通过，已停止打包")


def create_portable_archive(output_path, files):
    """通过临时文件原子生成支持 ZIP64 的便携压缩包"""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"目标发布包已存在，请先确认后删除: {output_path}")

    files = list(files)
    total_bytes = sum(path.stat().st_size for path in files)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for index, path in enumerate(files, start=1):
                relative_path = path.relative_to(PROJECT_ROOT)
                archive_name = (Path(ARCHIVE_ROOT_NAME) / relative_path).as_posix()
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix.casefold() in STORED_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(path, archive_name, compress_type=compression)
                if index % 1000 == 0 or index == len(files):
                    print(f"[一键包] 已写入 {index}/{len(files)} 个文件")

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(
        f"[一键包] 构建完成: {output_path}\n"
        f"[一键包] 文件数量: {len(files)} | 原始体积: {total_bytes / 1024**3:.2f} GiB | "
        f"压缩包体积: {output_path.stat().st_size / 1024**3:.2f} GiB"
    )
    return output_path


def main(argv=None):
    """解析打包参数，完成发布预检并生成版本化 ZIP"""
    parser = argparse.ArgumentParser(description="构建 Galatea-Core Windows 一键包")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="压缩包输出目录，默认项目根目录",
    )
    parser.add_argument(
        "--allow-cpu-only",
        action="store_true",
        help="允许便携环境不含 CUDA，仅用于明确发布 CPU 一键包",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只检查环境和待打包文件，不生成 ZIP",
    )
    args = parser.parse_args(argv)

    version = read_release_version()
    validate_portable_environment(require_cuda=not args.allow_cpu_only)
    files = list(iter_package_files())
    if not files:
        raise RuntimeError("没有找到可打包文件")

    total_bytes = sum(path.stat().st_size for path in files)
    print(
        f"[一键包] 版本: {version} | 文件: {len(files)} | "
        f"原始体积: {total_bytes / 1024**3:.2f} GiB"
    )
    if args.validate_only:
        print("[一键包] 发布预检通过，未生成压缩包。")
        return 0

    output_path = args.output_dir / f"Galatea_Core_V{version}.zip"
    create_portable_archive(output_path, files)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"[一键包] 构建失败: {exc}")
        raise SystemExit(1) from None
