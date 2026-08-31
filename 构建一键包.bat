@echo off
:: 本文件使用内置 Python 校验环境并生成带版本号的一键包。
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".\python_env\python.exe" (
    echo [错误] 找不到 python_env\python.exe，无法构建一键包。
    pause
    exit /b 1
)

echo 正在校验 CUDA 一键环境并构建发布包...
.\python_env\python.exe -X utf8 build_portable_package.py
if errorlevel 1 (
    echo [错误] 一键包构建失败，请查看上方诊断信息。
    pause
    exit /b 1
)

echo [成功] 一键包已生成。
pause
