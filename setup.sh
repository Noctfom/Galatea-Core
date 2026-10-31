#!/usr/bin/env bash
# Galatea-Core v3.15.7 - Linux初始化与启动脚本
# 默认复用当前Python或Conda环境，不创建虚拟环境、不编译Core或WindBot

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SETUP_MODE="webui"
MODE_SELECTED=0
SKIP_WINDBOT=0
PYTHON_CMD="${GALATEA_PYTHON:-}"
TORCH_CONSTRAINT_FILE=""
LAUNCH_ARGS=()

usage() {
    cat <<'HELP'
用法: bash setup.sh [初始化选项] [操作及其参数]
  默认                 安装缺失依赖、检查运行前置并启动WebUI
  --install-only       初始化后退出，不启动界面或训练
  --check              只检查，不安装依赖或更新卡库
  --webui [参数...]    初始化后启动WebUI，后续参数交给Streamlit
  --train [参数...]    初始化后启动CLI训练，默认沿用--no_compile
  --duel [参数...]     初始化后启动竞技场
  --python PATH        使用指定解释器，否则优先使用当前PATH中的python
  --skip-windbot       跳过可选WindBot的系统依赖与运行检查
  -h, --help           显示帮助
环境变量:
  GALATEA_PYTHON           等同--python
  GALATEA_TORCH_INDEX_URL  仅在Torch未安装时指定其wheel源
先激活已有Conda/venv再运行即可，脚本不会另建环境或替换已安装Torch
HELP
}

fail() {
    printf '[错误] %s\n' "$*" >&2
    exit 1
}

select_mode() {
    [[ "$MODE_SELECTED" == 0 ]] || fail '一次只能指定一种启动或检查操作'
    SETUP_MODE="$1"
    MODE_SELECTED=1
}

while (($#)); do
    case "$1" in
        --python)
            [[ $# -ge 2 && -n "$2" ]] || fail '--python需要解释器路径'
            PYTHON_CMD="$2"
            shift 2
            ;;
        --skip-windbot) SKIP_WINDBOT=1; shift ;;
        --check) select_mode check; shift ;;
        --install-only) select_mode install; shift ;;
        --train|--duel|--webui)
            select_mode "${1#--}"
            shift
            LAUNCH_ARGS=("$@")
            break
            ;;
        -h|--help) usage; exit 0 ;;
        *) fail "未知初始化参数: $1，请使用--help查看用法" ;;
    esac
done

[[ "$(uname -s)" == Linux ]] || fail 'setup.sh用于Linux；Windows请使用现有BAT启动器'
cd -- "$PROJECT_DIR"
if [[ -z "$PYTHON_CMD" ]]; then
    # PATH顺序保留已激活环境，不能优先选中环境外的版本化系统Python
    for candidate in python python3 python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_CMD="$(command -v "$candidate")"
            break
        fi
    done
fi
[[ -n "$PYTHON_CMD" ]] || fail '未找到Python，请先安装或激活Python 3.10+环境，建议3.12'
command -v "$PYTHON_CMD" >/dev/null 2>&1 || fail "解释器不存在: $PYTHON_CMD"
"$PYTHON_CMD" -c 'import sys; print("[Python]", sys.executable, sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || fail '需要Python 3.10+；请激活正确环境或用--python显式指定'
"$PYTHON_CMD" -m pip --version >/dev/null || fail '当前解释器缺少pip，请在该环境安装pip后重试'

# 在下载依赖或调用sudo之前检查现有公共内核，不尝试更换系统glibc
[[ -f "$PROJECT_DIR/ocgcore.so" ]] || fail '缺少ocgcore.so，请使用携带公开Core产物的完整源码'
"$PYTHON_CMD" - "$PROJECT_DIR/ocgcore.so" <<'PY'
import ctypes
import sys
try:
    ctypes.CDLL(sys.argv[1])
except OSError as error:
    print(f'[错误] ocgcore.so无法加载: {error}', file=sys.stderr)
    print('请使用与当前Linux/架构兼容且ABI一致的公开Core产物；脚本不会升级glibc或编译替代内核', file=sys.stderr)
    sys.exit(1)
PY

if [[ "$SKIP_WINDBOT" == 0 ]]; then
    [[ -f "$PROJECT_DIR/integrations/windbot/runtime/WindBot.exe" ]] \
        || fail '缺少WindBot.exe，请拉取包含runtime的源码或解压配套运行包；不用WindBot可加--skip-windbot'
    # Ubuntu/Debian仅安装运行前置，不安装SDK、MSBuild或mono-devel
    if command -v apt-get >/dev/null 2>&1 && command -v dpkg-query >/dev/null 2>&1; then
        missing_system=()
        for package in mono-runtime libmono-system-data4.0-cil libmono-system-web4.0-cil libmono-system-runtime-serialization4.0-cil libsqlite3-0; do
            if ! package_status="$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null)" || [[ "$package_status" != 'install ok installed' ]]; then
                missing_system+=("$package")
            fi
        done
        if ((${#missing_system[@]})); then
            printf '[WindBot] 缺少系统运行依赖: %s\n' "${missing_system[*]}"
            [[ "$SETUP_MODE" != check ]] || fail '只读检查未通过；运行bash setup.sh --install-only可安装前置'
            installer=()
            if [[ "$EUID" != 0 ]]; then
                command -v sudo >/dev/null 2>&1 || fail '缺少系统依赖且没有sudo，请管理员安装上述运行包，或使用--skip-windbot'
                installer=(sudo)
            fi
            "${installer[@]}" apt-get update
            "${installer[@]}" apt-get install -y --no-install-recommends "${missing_system[@]}"
        fi
    fi
    # 其他发行版复用已有Mono与SQLite，缺依赖时由当前检查器明确报错
    "$PYTHON_CMD" - <<'PY'
import subprocess
from integrations.windbot.runtime import resolve_runtime
command = resolve_runtime()
subprocess.run([command[0], '--version'], check=True, stdout=subprocess.DEVNULL)
print('[WindBot] 公共程序、Mono和系统SQLite检查通过；此项不代表真实对局验收')
PY
fi

cleanup_constraint() {
    if [[ -n "$TORCH_CONSTRAINT_FILE" ]]; then
        rm -f -- "$TORCH_CONSTRAINT_FILE"
        TORCH_CONSTRAINT_FILE=""
    fi
}
trap cleanup_constraint EXIT

if [[ "$SETUP_MODE" != check ]]; then
    if ! "$PYTHON_CMD" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("torch") else 1)'; then
        if [[ -n "${GALATEA_TORCH_INDEX_URL:-}" ]]; then
            "$PYTHON_CMD" -m pip install --disable-pip-version-check 'torch>=2.0.0' --index-url "$GALATEA_TORCH_INDEX_URL"
        fi
    fi
    pip_constraints=()
    if "$PYTHON_CMD" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("torch") else 1)'; then
        # 本地CUDA版本后缀也保留，阻止其他依赖隐式替换已经配置好的Torch
        TORCH_VERSION="$("$PYTHON_CMD" -c 'import torch; print(torch.__version__)')" \
            || fail '已有Torch无法导入，请先修复当前环境；脚本不会自动覆盖它'
        TORCH_CONSTRAINT_FILE="$(mktemp "${TMPDIR:-/tmp}/galatea-torch-XXXXXX")"
        printf 'torch==%s\n' "$TORCH_VERSION" > "$TORCH_CONSTRAINT_FILE"
        pip_constraints=(-c "$TORCH_CONSTRAINT_FILE")
        printf '[Torch] 保留当前版本: %s\n' "$TORCH_VERSION"
    fi
    # 无Torch时沿requirements和现有pip源安装，不再硬编码cu121或猜测驱动对应的构建
    "$PYTHON_CMD" -m pip install --disable-pip-version-check "${pip_constraints[@]}" -r "$PROJECT_DIR/requirements.txt" \
        || fail '依赖安装失败；请查看pip报错，核对Torch约束并确认已激活可写的Conda/venv环境'
    cleanup_constraint
fi

"$PYTHON_CMD" "$PROJECT_DIR/environment_setup.py" --verify-imports
"$PYTHON_CMD" - <<'PY'
import shutil
import torch
print(f'[Torch] {torch.__version__} | CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    probe = torch.ones((2, 2), device='cuda')
    result = probe @ probe
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError('CUDA计算探针产生非有限结果')
    print(f'[CUDA] {torch.cuda.get_device_name(0)}计算检查通过')
elif shutil.which('nvidia-smi'):
    print('[提示] 检测到NVIDIA工具但当前Torch无法使用CUDA，请核对驱动和Torch构建；脚本不会自动替换环境')
PY

resources_ready() {
    "$PYTHON_CMD" - <<'PY'
from pathlib import Path
import sys
database, scripts = Path('cards.cdb'), Path('script')
sys.exit(0 if database.is_file() and database.stat().st_size and scripts.is_dir() and any(scripts.glob('*.lua')) else 1)
PY
}
if ! resources_ready; then
    [[ "$SETUP_MODE" != check ]] || fail '缺少卡库或Lua脚本；只读检查不会下载资源'
    "$PYTHON_CMD" "$PROJECT_DIR/main.py" update --data
    resources_ready || fail '资源同步后仍缺少卡库或Lua脚本，请查看资源同步日志'
fi

printf '[完成] 初始化与依赖检查通过，使用当前环境，不创建虚拟环境\n'
case "$SETUP_MODE" in
    check|install) ;;
    train) exec "$PYTHON_CMD" "$PROJECT_DIR/main.py" train --no_compile "${LAUNCH_ARGS[@]}" ;;
    duel) exec "$PYTHON_CMD" "$PROJECT_DIR/main.py" duel "${LAUNCH_ARGS[@]}" ;;
    webui) exec "$PYTHON_CMD" -m streamlit run "$PROJECT_DIR/app.py" "${LAUNCH_ARGS[@]}" ;;
esac
