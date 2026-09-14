# 本文件校验一键包环境并生成带版本号的 Galatea-Core 便携 ZIP。

import argparse
import fnmatch
import hashlib
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT_NAME = "Galatea_Core"
GITHUB_RELEASE_ASSET_LIMIT_BYTES = 2 * 1024**3
DEFAULT_RELEASE_PART_MIB = 1900
COPY_BUFFER_BYTES = 8 * 1024**2
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
    "Galatea_Core_V*.zip.part*",
    "Galatea_Core_V*.zip.sha256.txt",
    "Merge_Galatea_Core_V*.bat",
    "crash_report*",
    "tmp_rollout_*.pt",
    "tmp_rollout_*.pt.tmp",
    "tmp_weights_*.pt",
    "*.pyc",
)
STORED_SUFFIXES = {
    ".data",
    ".npy",
    ".onnx",
    ".pth",
    ".pt",
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


def split_release_archive(archive_path, part_size_bytes=None):
    """将超大 ZIP 无损切分为 GitHub Release 分卷并生成合并与校验文件"""
    input_path = Path(archive_path)
    if input_path.is_symlink():
        raise ValueError(f"待分卷 ZIP 不允许使用符号链接: {input_path}")
    archive_path = input_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"待分卷 ZIP 不存在或不是普通文件: {archive_path}")
    if archive_path.suffix.casefold() != ".zip":
        raise ValueError(f"只允许分卷 ZIP 文件: {archive_path.name}")
    if not zipfile.is_zipfile(archive_path):
        raise zipfile.BadZipFile(f"待分卷文件不是有效 ZIP: {archive_path.name}")

    part_size_bytes = part_size_bytes or DEFAULT_RELEASE_PART_MIB * 1024**2
    if part_size_bytes <= 0 or part_size_bytes >= GITHUB_RELEASE_ASSET_LIMIT_BYTES:
        raise ValueError("Release 分卷大小必须大于 0 且严格小于 2 GiB")

    archive_size = archive_path.stat().st_size
    part_count = max(1, (archive_size + part_size_bytes - 1) // part_size_bytes)
    number_width = max(3, len(str(part_count)))
    part_paths = [
        archive_path.with_name(
            f"{archive_path.name}.part{part_number:0{number_width}d}"
        )
        for part_number in range(1, part_count + 1)
    ]
    manifest_path = archive_path.with_name(f"{archive_path.name}.sha256.txt")
    merge_script_path = archive_path.with_name(f"Merge_{archive_path.stem}.bat")
    output_paths = [*part_paths, manifest_path, merge_script_path]
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(f"分卷输出已存在，请先确认后删除: {names}")

    temporary_outputs = []
    part_records = []
    archive_digest = hashlib.sha256()
    try:
        with archive_path.open("rb") as source:
            for part_path in part_paths:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{part_path.name}.",
                    suffix=".tmp",
                    dir=archive_path.parent,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    temporary_outputs.append((temporary_path, part_path))
                    part_digest = hashlib.sha256()
                    remaining = part_size_bytes
                    written_bytes = 0
                    while remaining > 0:
                        chunk = source.read(min(COPY_BUFFER_BYTES, remaining))
                        if not chunk:
                            break
                        temporary_file.write(chunk)
                        part_digest.update(chunk)
                        archive_digest.update(chunk)
                        written_bytes += len(chunk)
                        remaining -= len(chunk)
                part_records.append(
                    (part_path, written_bytes, part_digest.hexdigest())
                )

        expected_archive_hash = archive_digest.hexdigest()
        if sum(record[1] for record in part_records) != archive_size:
            raise OSError("分卷读取长度与原始 ZIP 不一致")

        manifest_lines = [
            "# Galatea-Core GitHub Release split archive checksums",
            f"# Original-Size: {archive_size}",
            f"{expected_archive_hash} *{archive_path.name}",
        ]
        manifest_lines.extend(
            f"{part_hash} *{part_path.name}"
            for part_path, _, part_hash in part_records
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=archive_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_manifest = Path(temporary_file.name)
            temporary_outputs.append((temporary_manifest, manifest_path))
            temporary_file.write("\n".join(manifest_lines) + "\n")

        copy_expression = "+".join(f'"{path.name}"' for path in part_paths)
        merge_script = (
            "@echo off\n"
            ":: This file reconstructs and verifies the Galatea-Core release ZIP.\n"
            "setlocal\n"
            "cd /d \"%~dp0\"\n"
            f"set \"ARCHIVE={archive_path.name}\"\n"
            f"set \"EXPECTED_SHA256={expected_archive_hash}\"\n"
            "if exist \"%ARCHIVE%\" (\n"
            "  echo [ERROR] The target ZIP already exists: %ARCHIVE%\n"
            "  echo Remove or move it only after confirming which copy to keep.\n"
            "  pause\n"
            "  exit /b 1\n"
            ")\n"
            f"copy /b {copy_expression} \"%ARCHIVE%\" >nul\n"
            "if errorlevel 1 (\n"
            "  echo [ERROR] Failed to merge parts. Download every part into this folder.\n"
            "  pause\n"
            "  exit /b 1\n"
            ")\n"
            "for /f %%H in ('powershell -NoProfile -Command \"(Get-FileHash -LiteralPath '%ARCHIVE%' -Algorithm SHA256).Hash.ToLowerInvariant()\"') do set \"ACTUAL_SHA256=%%H\"\n"
            "if /I not \"%ACTUAL_SHA256%\"==\"%EXPECTED_SHA256%\" (\n"
            "  echo [ERROR] SHA256 mismatch. A release part may be missing or damaged.\n"
            "  pause\n"
            "  exit /b 1\n"
            ")\n"
            "echo [OK] Release ZIP reconstructed and verified: %ARCHIVE%\n"
            "pause\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="",
            prefix=f".{merge_script_path.name}.",
            suffix=".tmp",
            dir=archive_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_script = Path(temporary_file.name)
            temporary_outputs.append((temporary_script, merge_script_path))
            temporary_file.write(merge_script)

        for temporary_path, final_path in temporary_outputs:
            os.replace(temporary_path, final_path)
        temporary_outputs.clear()
    finally:
        for temporary_path, _ in temporary_outputs:
            if temporary_path.exists():
                temporary_path.unlink()

    print(
        f"[Release 分卷] {archive_path.name} 已切分为 {len(part_paths)} 个文件，"
        f"单卷上限 {part_size_bytes / 1024**2:.0f} MiB"
    )
    print(
        f"[Release 分卷] GitHub 上传全部 .part 文件、{manifest_path.name} 与 "
        f"{merge_script_path.name}；不要上传超过 2 GiB 的原始 ZIP。"
    )
    return part_paths, manifest_path, merge_script_path


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
        help="仅跳过 CUDA 发布预检；不会删除当前环境中的 CUDA 文件或缩小包体",
    )
    parser.add_argument(
        "--release-part-mib",
        type=int,
        default=DEFAULT_RELEASE_PART_MIB,
        help="超出 GitHub 限制时的分卷大小（MiB），默认 1900；设为 0 禁用分卷",
    )
    parser.add_argument(
        "--split-existing",
        type=Path,
        help="只分卷一个已经生成的 ZIP，不重复执行环境预检和完整打包",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只检查环境和待打包文件，不生成 ZIP",
    )
    args = parser.parse_args(argv)

    if args.release_part_mib < 0 or args.release_part_mib >= 2048:
        parser.error("--release-part-mib 必须为 0 到 2047 之间的整数")
    part_size_bytes = args.release_part_mib * 1024**2
    if args.split_existing is not None:
        if args.validate_only or args.allow_cpu_only:
            parser.error("--split-existing 不能与 --validate-only/--allow-cpu-only 同时使用")
        if not part_size_bytes:
            parser.error("--split-existing 需要启用非零 Release 分卷大小")
        split_release_archive(args.split_existing, part_size_bytes)
        return 0

    version = read_release_version()
    if args.allow_cpu_only:
        print(
            "[一键包] 注意：--allow-cpu-only 只跳过 CUDA 探针，"
            "不会从现有 python_env 裁剪 CUDA 运行库。"
        )
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
    archive_path = create_portable_archive(output_path, files)
    if (
        part_size_bytes
        and archive_path.stat().st_size >= GITHUB_RELEASE_ASSET_LIMIT_BYTES
    ):
        split_release_archive(archive_path, part_size_bytes)
    elif archive_path.stat().st_size >= GITHUB_RELEASE_ASSET_LIMIT_BYTES:
        print(
            "[一键包] 警告：ZIP 已达到 GitHub Release 2 GiB 单文件上限，"
            "但分卷已禁用。"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"[一键包] 构建失败: {exc}")
        raise SystemExit(1) from None
