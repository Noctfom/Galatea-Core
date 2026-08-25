# 本文件检查并修复一键包 Python 环境中缺失的项目依赖

import argparse
import importlib
import importlib.util
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ImportError:
    # 便携环境至少包含 pip，可在 packaging 尚未独立安装时使用其内置解析器
    from pip._vendor.packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
IMPORT_NAME_OVERRIDES = {
    "pyzmq": "zmq",
    "sentence-transformers": "sentence_transformers",
    "sentence_transformers": "sentence_transformers",
}
MANUALLY_MANAGED_PACKAGES = {"torch"}


def normalize_package_name(name):
    """统一依赖包名称，便于匹配导入名覆盖规则"""
    return re.sub(r"[-_.]+", "-", name).lower()


def portable_project_path_is_configured(python_dir=None, project_root=None):
    """检查便携 Python 的 _pth 文件是否已经包含项目根目录"""
    python_dir = Path(python_dir or Path(sys.executable).resolve().parent).resolve()
    project_root = Path(project_root or PROJECT_ROOT).resolve()
    path_files = sorted(python_dir.glob("python*._pth"))
    if not path_files:
        return True

    for path_file in path_files:
        for raw_line in path_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("import "):
                continue
            if (python_dir / line).resolve() == project_root:
                return True
    return False


def configure_portable_project_path(python_dir=None, project_root=None):
    """把项目根目录原子写入便携 Python 的 _pth 搜索路径"""
    python_dir = Path(python_dir or Path(sys.executable).resolve().parent).resolve()
    project_root = Path(project_root or PROJECT_ROOT).resolve()
    path_files = sorted(python_dir.glob("python*._pth"))
    if not path_files or portable_project_path_is_configured(python_dir, project_root):
        return True

    relative_project_path = os.path.relpath(project_root, python_dir).replace("\\", "/")
    for path_file in path_files:
        lines = path_file.read_text(encoding="utf-8").splitlines()
        insert_at = next(
            (index for index, line in enumerate(lines) if line.strip().startswith("import ")),
            len(lines),
        )
        lines.insert(insert_at, relative_project_path)
        temporary_path = path_file.with_suffix(path_file.suffix + ".tmp")
        temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary_path, path_file)

    print(f"[环境修复] 已将项目目录加入便携 Python 搜索路径：{project_root}")
    return portable_project_path_is_configured(python_dir, project_root)


def parse_requirement_file(path):
    """读取普通 requirements 文件并返回安装规格与导入模块名"""
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement", "-c", "--constraint")):
            raise ValueError("nested requirement files are not supported")

        requirement = Requirement(line)
        package_name = requirement.name
        normalized_name = normalize_package_name(package_name)
        import_name = IMPORT_NAME_OVERRIDES.get(
            normalized_name,
            package_name.replace("-", "_"),
        )
        requirements.append((line, import_name))
    return requirements


def find_missing_requirements(requirements):
    """检查模块、发行包元数据和最低版本是否满足依赖清单"""
    missing = []
    for install_spec, import_name in requirements:
        requirement = Requirement(install_spec)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue

        try:
            module_spec = importlib.util.find_spec(import_name)
        except (ImportError, AttributeError, ValueError):
            module_spec = None
        try:
            installed_version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            installed_version = None

        version_matches = (
            installed_version is not None
            and (
                not requirement.specifier
                or requirement.specifier.contains(installed_version, prereleases=True)
            )
        )
        if module_spec is None or not version_matches:
            missing.append((install_spec, import_name))
    return missing


def verify_requirement_imports(requirements):
    """实际导入全部依赖，发现二进制损坏或版本不兼容问题"""
    failures = []
    for install_spec, import_name in requirements:
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            failures.append((install_spec, import_name, f"{type(exc).__name__}: {exc}"))
    return failures


def install_missing_requirements(missing):
    """调用当前解释器的 pip，仅安装缺失的依赖规格"""
    manual_specs = [
        install_spec
        for install_spec, _ in missing
        if normalize_package_name(Requirement(install_spec).name)
        in MANUALLY_MANAGED_PACKAGES
    ]
    if manual_specs:
        print(
            "[环境修复] 以下依赖需要根据 CPU/CUDA 类型手动安装，已停止自动修改："
            + ", ".join(manual_specs)
        )
        return 1

    install_specs = [install_spec for install_spec, _ in missing]
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        *install_specs,
    ]
    print("[环境修复] 正在安装缺失依赖：" + ", ".join(install_specs))
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def check_environment(requirements, verify_imports=False):
    """检查环境完整性，并输出适合启动器显示的诊断信息"""
    missing = find_missing_requirements(requirements)
    if missing:
        print("[环境检查] 缺失依赖：" + ", ".join(spec for spec, _ in missing))
        return False

    if verify_imports:
        failures = verify_requirement_imports(requirements)
        if failures:
            print("[环境检查] 以下依赖无法正常导入：")
            for install_spec, import_name, error in failures:
                print(f"  - {install_spec} ({import_name}): {error}")
            return False

    print("[环境检查] 一键包依赖完整。")
    return True


def main(argv=None):
    """执行一键环境检查，并按参数选择是否自动修复"""
    parser = argparse.ArgumentParser(description="检查并修复 Galatea 一键包环境")
    parser.add_argument("--repair", action="store_true", help="自动安装缺失依赖")
    parser.add_argument(
        "--verify-imports",
        action="store_true",
        help="实际导入依赖以检查二进制兼容性",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="依赖清单路径",
    )
    args = parser.parse_args(argv)

    requirements_path = args.requirements.resolve()
    if not requirements_path.is_file():
        print(f"[环境检查] 找不到依赖清单：{requirements_path}")
        return 1

    try:
        requirements = parse_requirement_file(requirements_path)
    except (OSError, ValueError) as exc:
        print(f"[环境检查] 无法解析依赖清单：{exc}")
        return 1

    if args.repair:
        if not configure_portable_project_path():
            print("[环境修复] 无法更新便携 Python 的项目搜索路径。")
            return 1
    elif not portable_project_path_is_configured():
        print("[环境检查] 便携 Python 尚未包含项目根目录，请使用 --repair。")
        return 1

    missing = find_missing_requirements(requirements)
    if missing and args.repair:
        if install_missing_requirements(missing) != 0:
            print("[环境修复] pip 安装失败，请检查网络或软件源设置。")
            return 1
        importlib.invalidate_caches()

    return 0 if check_environment(requirements, args.verify_imports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
