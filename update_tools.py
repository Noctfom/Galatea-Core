# ==================================================================================
#  Galatea Update Tools (V2.0 - MyCard Data Source)
#  负责从萌卡官方拉取最新的中文卡片数据库
# ==================================================================================

import os
import subprocess
import urllib.request
import shutil
import tempfile
import zipfile
import io
import re

# 仓库地址（仅用于更新 Python 逻辑代码）
MY_REPO_URL = "https://github.com/Noctfom/Galatea-Core.git"

# 萌卡官方卡片数据库 (zh-CN 中文版)
# 路径：locales/zh-CN/cards.cdb
MOCKA_CDB_URL = "https://raw.githubusercontent.com/mycard/ygopro-database/master/locales/zh-CN/cards.cdb"

# 官方脚本库地址
OFFICIAL_SCRIPT_REPO = "https://github.com/Fluorohydride/ygopro-scripts.git"

def update_core_code():
    """更新本地核心代码。有 .git 则 git pull，否则从 GitHub ZIP Archive 覆盖更新"""
    print("🚀 正在检查并拉取核心代码更新...")

    # ----- 路径 A：标准 Git 仓库 -----
    if os.path.exists(".git"):
        try:
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                capture_output=True, text=True, check=True
            )
            print(f"✅ 代码更新成功:\n{result.stdout.strip()}")
            return True
        except Exception as e:
            print(f"❌ Git 更新失败: {e}")
            # 不直接 return False —— 下面还有 ZIP 兜底
            print("🔄 尝试使用 ZIP Archive 方式作为备用方案...")
    else:
        print("ℹ️  当前目录不是 Git 仓库，将使用 ZIP Archive 方式更新。")

    # ----- 路径 B：ZIP Archive 下载 + 覆盖（一键包 / 非 git 环境）-----
    return _update_core_via_zip()


def _update_core_via_zip():
    """通过 GitHub ZIP Archive 下载最新代码并覆盖 .py 文件"""
    zip_url = _git_url_to_zip_url(MY_REPO_URL)
    print(f"📥 正在下载最新代码包 (ZIP Archive)...")

    try:
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            root_prefix = _find_zip_root_prefix(zf)  # e.g. "Galatea-Core-master/"

            # 用户数据目录/文件，跳过不覆盖
            SKIP_PATTERNS = [
                'cards.cdb',
                'knowledge_base.json',
                'meta_staples.json',
                '一键包启动Webui.bat',
                '.gitignore',
                '.git/',
                'script/',
                'decks/',
                'models/',
                'runs/',
                'ai_thoughts/',
                'replays/',
                'replay_data/',
                'system_logs/',
                'web_data/',
                'deploy_packages/',
                'python_env/',
                'venv/',
                '__pycache__/',
                '.vscode/',
            ]

            updated_count = 0
            for member in zf.namelist():
                # 去掉仓库根目录前缀
                rel_path = member[len(root_prefix):] if root_prefix else member
                if not rel_path:
                    continue

                # 跳过目录条目
                if rel_path.endswith('/'):
                    continue

                # 跳过用户数据
                skip = False
                for pattern in SKIP_PATTERNS:
                    if pattern.endswith('/'):
                        if rel_path.startswith(pattern) or rel_path == pattern[:-1]:
                            skip = True
                            break
                    else:
                        if rel_path == pattern:
                            skip = True
                            break
                if skip:
                    continue

                # 只更新 .py / .md / .txt 等核心文件
                if not (rel_path.endswith('.py') or rel_path.endswith('.md') or rel_path.endswith('.txt')):
                    continue

                # 确保目标目录存在
                target_path = os.path.join('.', rel_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # 写入文件
                with zf.open(member) as src:
                    with open(target_path, 'wb') as dst:
                        dst.write(src.read())
                updated_count += 1

            if updated_count > 0:
                print(f"✅ 核心代码更新完成！共更新了 {updated_count} 个文件。")
                return True
            else:
                print("ℹ️  未发现需要更新的文件，当前已是最新版本。")
                return True

    except Exception as e:
        print(f"❌ ZIP 更新失败: {e}")
        return False

def update_data_and_scripts(repo_type='default', force=False):
    """更新来自萌卡的 cards.cdb 和官方脚本库"""
    print(f"🌐 正在启动数据同步模块...")
    
    # --- 1. 下载萌卡官方中文 cards.cdb ---
    cdb_path = "cards.cdb"
    print(f"📥 正在从萌卡官方拉取最新中文数据库...")
    try:
        # 模拟浏览器 User-Agent 防止被 GitHub 拦截
        req = urllib.request.Request(MOCKA_CDB_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(cdb_path, 'wb') as f:
                f.write(response.read())
        print("✅ 萌卡官方中文卡库下载并替换完成！")
    except Exception as e:
        print(f"❌ 卡库下载失败 (请检查网络): {e}")

    # --- 2. 更新 Script 文件夹 (GitHub ZIP Archive，不走 git clone) ---
    script_repo_url = OFFICIAL_SCRIPT_REPO if repo_type == 'default' else repo_type
    # 将 git URL 转换为 ZIP Archive 下载链接
    # 形如 https://github.com/owner/repo.git → https://github.com/owner/repo/archive/refs/heads/master.zip
    zip_url = _git_url_to_zip_url(script_repo_url)
    print(f"📥 正在拉取最新的官方 Lua 脚本库 (ZIP Archive)...")
    
    try:
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
        
        # 内存中直接解压
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            # 获取仓库内第一层目录前缀（如 ygopro-scripts-master/）
            root_prefix = _find_zip_root_prefix(zf)
            
            target_script_dir = "./script"
            if force and os.path.exists(target_script_dir):
                shutil.rmtree(target_script_dir)
            if not os.path.exists(target_script_dir):
                os.makedirs(target_script_dir)

            moved_count = 0
            for member in zf.namelist():
                if not member.endswith('.lua'):
                    continue
                # 去掉仓库根目录前缀
                rel_path = member[len(root_prefix):] if root_prefix else member
                # 只用文件名（官方脚本库的 lua 文件都在根目录）
                filename = os.path.basename(rel_path)
                if filename:
                    with zf.open(member) as src:
                        with open(os.path.join(target_script_dir, filename), 'wb') as dst:
                            dst.write(src.read())
                    moved_count += 1
                        
            print(f"✅ 脚本库同步完成！共合并了 {moved_count} 个 Lua 文件。")
            
    except Exception as e:
        print(f"❌ 脚本更新失败: {e}")


def _git_url_to_zip_url(git_url):
    """将 Git 仓库 URL 转为 GitHub Archive ZIP 下载链接"""
    # 匹配 https://github.com/owner/repo.git 或 https://github.com/owner/repo
    match = re.match(r'https://github\.com/([^/]+)/([^/]+?)(?:\.git)?$', git_url)
    if match:
        owner, repo = match.groups()
        return f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
    # 如果已经是 ZIP 链接或其他格式，直接返回
    return git_url


def _find_zip_root_prefix(zf):
    """找到 ZIP 中仓库的根目录前缀，如 'ygopro-scripts-master/' """
    # 取第一个文件路径，提取其顶层目录名
    for name in zf.namelist():
        if '/' in name:
            return name.split('/')[0] + '/'
    return ''
