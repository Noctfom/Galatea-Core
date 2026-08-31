@echo off
:: 强制设定编码为 UTF-8，确保中文显示正常
chcp 65001 >nul

:: 锁定工作目录
cd /d "%~dp0"

title Galatea AI 司令塔启动器
color 0A

echo ==========================================
echo       Galatea AI 司令塔 - 一键启动
echo ==========================================
echo.
echo 正在检查环境...

:: 1. 检查便携式 Python 环境
if not exist ".\python_env\python.exe" (
    echo [错误] 找不到内置的 Python 环境 python_env\python.exe
    echo 请确认您下载的是完整的整合包。
    pause
    exit /b
)

:: 2. 设置环境变量
set "PATH=%~dp0python_env;%~dp0python_env\Scripts;%PATH%"
set "PYTHONPATH=%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

:: 3. 检查并自动补齐依赖，同时验证二进制包能够实际导入
echo 正在核验一键包依赖完整性...
.\python_env\python.exe environment_setup.py --repair --verify-imports --verify-runtime-assets --require-portable-python
if errorlevel 1 (
    echo [错误] 一键包依赖检查或修复失败。
    echo 请检查网络连接后重试，或手动执行：
    echo .\python_env\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

:: 仅检查环境时不启动浏览器和 WebUI，供发布前回归验证使用
if /I "%~1"=="--check" (
    echo [成功] 一键包环境检查通过。
    exit /b 0
)

echo 环境就绪，准备启动...
echo.
echo ------------------------------------------
echo 💡 [操作提示]
echo 1. 启动成功后，程序会自动在浏览器打开网页
echo 2. 如果网页没弹出，请按住Ctrl键并点击下方出现的蓝色链接
echo ------------------------------------------
echo.

:: 4. 预先打开浏览器 (强制使用 127.0.0.1 彻底避开 localhost 解析大坑)
start "" "http://127.0.0.1:8501"

:: 5. 启动 Streamlit 服务
:: --server.address=127.0.0.1 强制将服务锁死在 IPv4 的本地环回地址上
.\python_env\python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false

:: 如果进程被手动终止
echo.
echo [提示] 司令塔服务已关闭。
pause
