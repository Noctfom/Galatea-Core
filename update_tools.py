# ==================================================================================
#  Galatea Update Tools (V2.0 - MyCard Data Source)
#  负责从萌卡官方拉取最新的中文卡片数据库
# ==================================================================================

import os
import subprocess
import urllib.request
import shutil
import tempfile

# 仓库地址（仅用于更新 Python 逻辑代码）
MY_REPO_URL = "https://github.com/Noctfom/astrbot-plugin-duel-galatea.git"

# 萌卡官方卡片数据库 (zh-CN 中文版)
# 路径：locales/zh-CN/cards.cdb
MOCKA_CDB_URL = "https://raw.githubusercontent.com/mycard/ygopro-database/master/locales/zh-CN/cards.cdb"

# 官方脚本库地址
OFFICIAL_SCRIPT_REPO = "https://github.com/Fluorohydride/ygopro-scripts.git"

def update_core_code():
    """更新本地核心代码 (相当于 git pull)"""
    print("🚀 正在检查并拉取核心代码更新...")
    try:
        if not os.path.exists(".git"):
            print("⚠️ 当前目录不是一个 Git 仓库，无法执行自动更新。")
            return False
        result = subprocess.run(["git", "pull", "origin", "main"], capture_output=True, text=True, check=True)
        print(f"✅ 代码更新成功:\n{result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ 代码更新失败: {e}")
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

    # --- 2. 更新 Script 文件夹 (Git 浅克隆) ---
    script_url = OFFICIAL_SCRIPT_REPO if repo_type == 'default' else repo_type
    print(f"📥 正在拉取最新的官方 Lua 脚本库...")
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # --depth 1 极速拉取
            subprocess.run(["git", "clone", "--depth", "1", script_url, tmp_dir], 
                           capture_output=True, text=True, check=True)
            
            target_script_dir = "./script"
            if force and os.path.exists(target_script_dir):
                shutil.rmtree(target_script_dir)
            
            if not os.path.exists(target_script_dir):
                os.makedirs(target_script_dir)

            moved_count = 0
            for root, _, files in os.walk(tmp_dir):
                if '.git' in root: continue 
                for file in files:
                    if file.endswith('.lua'):
                        shutil.copy2(os.path.join(root, file), os.path.join(target_script_dir, file))
                        moved_count += 1
                        
            print(f"✅ 脚本库同步完成！共合并了 {moved_count} 个 Lua 文件。")
            
    except Exception as e:
        print(f"❌ 脚本更新失败: {e}")