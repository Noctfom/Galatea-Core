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
import urllib.parse

from card_vocab import MAX_CARD_VOCAB_FILE_BYTES

# 仓库地址（仅用于更新 Python 逻辑代码）
MY_REPO_URL = "https://github.com/Noctfom/Galatea-Core.git"

# 萌卡官方卡片数据库 (zh-CN 中文版)
# 路径：locales/zh-CN/cards.cdb
MOCKA_CDB_URL = "https://raw.githubusercontent.com/mycard/ygopro-database/master/locales/zh-CN/cards.cdb"

# Galatea 仓库维护跨机器唯一的 V4 卡片编号表
CARD_VOCAB_URL = "https://raw.githubusercontent.com/Noctfom/Galatea-Core/main/card_vocab.json"

# 官方脚本库地址
OFFICIAL_SCRIPT_REPO = "https://github.com/Fluorohydride/ygopro-scripts.git"
MAX_CDB_FILE_BYTES = 1024 * 1024 * 1024


def _validate_http_url(url, label):
    """校验用户提供的远程资源地址，拒绝本地协议和内嵌凭据"""
    normalized = str(url or "").strip()
    parts = urllib.parse.urlsplit(normalized)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"{label} must use HTTP or HTTPS")
    if parts.username or parts.password:
        raise ValueError(f"{label} must not contain credentials")
    return normalized


def resolve_card_vocabulary_url(source=CARD_VOCAB_URL):
    """把词表仓库地址或直接文件地址统一解析为 card_vocab.json URL"""
    normalized = _validate_http_url(source, "card vocabulary source URL")
    parts = urllib.parse.urlsplit(normalized)
    path_parts = [part for part in parts.path.split("/") if part]
    if parts.netloc.lower() == "github.com" and len(path_parts) >= 2:
        owner = path_parts[0]
        repository = path_parts[1]
        if repository.endswith(".git"):
            repository = repository[:-4]
        branch = "main"
        subdirectory = []
        if len(path_parts) >= 4 and path_parts[2] in {"tree", "blob"}:
            branch = path_parts[3]
            subdirectory = path_parts[4:]
            if path_parts[2] == "blob" and subdirectory:
                subdirectory = subdirectory[:-1]
        asset_path = "/".join(
            [owner, repository, branch, *subdirectory, "card_vocab.json"]
        )
        return f"https://raw.githubusercontent.com/{asset_path}"
    if not parts.path.endswith(".json"):
        path = parts.path.rstrip("/") + "/card_vocab.json"
        return urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
        )
    return normalized


def _download_cdb_atomically(url, target_path="cards.cdb"):
    """流式下载并验证 CDB 后原子替换，避免自建源损坏现有卡库"""
    from card_vocab import read_card_codes_from_cdb

    target = os.path.abspath(target_path)
    target_directory = os.path.dirname(target)
    os.makedirs(target_directory, exist_ok=True)
    temporary_path = None
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as response, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".cards.",
            suffix=".download.tmp",
            dir=target_directory,
            delete=False,
        ) as stream:
            temporary_path = stream.name
            total_bytes = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_CDB_FILE_BYTES:
                    raise ValueError("remote cards.cdb exceeds the 1 GiB safety limit")
                stream.write(chunk)
        read_card_codes_from_cdb(temporary_path)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def _install_card_vocabulary_bytes(payload, target_path="card_vocab.json"):
    """验证仓库词表是本地前缀扩展后再原子安装"""
    if not payload or len(payload) > MAX_CARD_VOCAB_FILE_BYTES:
        raise ValueError("remote card vocabulary size is outside the safety limit")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".card_vocab.",
            suffix=".download.tmp",
            dir=".",
            delete=False,
        ) as stream:
            temporary_path = stream.name
            stream.write(payload)
        from card_vocab import synchronize_authoritative_card_vocabulary
        return synchronize_authoritative_card_vocabulary(
            temporary_path,
            target_path,
        )
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def sync_authoritative_card_vocabulary(url=CARD_VOCAB_URL):
    """从 Galatea 仓库拉取唯一词表，避免各机器独立编号"""
    resolved_url = resolve_card_vocabulary_url(url)
    request = urllib.request.Request(resolved_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request) as response:
        payload = response.read(MAX_CARD_VOCAB_FILE_BYTES + 1)
    return _install_card_vocabulary_bytes(payload)

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
    """通过 GitHub ZIP Archive 下载并更新代码、文档和启动脚本"""
    zip_url = _git_url_to_zip_url(MY_REPO_URL)
    print(f"📥 正在下载最新代码包 (ZIP Archive)...")

    try:
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            root_prefix = _find_zip_root_prefix(zf)  # e.g. "Galatea-Core-master/"
            vocabulary_payload = None

            # 用户数据目录/文件，跳过不覆盖
            SKIP_PATTERNS = [
                'cards.cdb',
                'knowledge_base.json',
                'meta_staples.json',
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

                if rel_path == 'card_vocab.json':
                    vocabulary_payload = zf.read(member)
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

                # 只更新代码、文档和跨平台启动脚本
                if not rel_path.endswith(('.py', '.md', '.txt', '.bat', '.sh')):
                    continue

                # 确保目标目录存在
                target_path = os.path.join('.', rel_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # 写入文件
                with zf.open(member) as src:
                    with open(target_path, 'wb') as dst:
                        dst.write(src.read())
                updated_count += 1

            if vocabulary_payload is not None:
                sync_result = _install_card_vocabulary_bytes(vocabulary_payload)
                if sync_result["status"] == "installed":
                    updated_count += 1
                print(
                    "✅ 权威卡片词表已校验: "
                    f"{sync_result['vocabulary'].card_count}/"
                    f"{sync_result['vocabulary'].capacity} "
                    f"({sync_result['status']})"
                )

            if updated_count > 0:
                print(f"✅ 核心代码更新完成！共更新了 {updated_count} 个文件。")
                return True
            else:
                print("ℹ️  未发现需要更新的文件，当前已是最新版本。")
                return True

    except Exception as e:
        print(f"❌ ZIP 更新失败: {e}")
        return False

def update_data_and_scripts(
    repo_type='default',
    force=False,
    *,
    cdb_url=MOCKA_CDB_URL,
    card_vocab_url=CARD_VOCAB_URL,
):
    """更新来自萌卡的 cards.cdb 和官方脚本库"""
    print(f"🌐 正在启动数据同步模块...")
    cdb_url = _validate_http_url(cdb_url, "CDB source URL")
    sync_errors = []
    
    # --- 1. 下载萌卡官方中文 cards.cdb ---
    cdb_path = "cards.cdb"
    print(f"📥 正在从萌卡官方拉取最新中文数据库...")
    try:
        _download_cdb_atomically(cdb_url, cdb_path)
        print("✅ 萌卡官方中文卡库下载并替换完成！")
    except Exception as e:
        print(f"❌ 卡库下载失败 (请检查网络): {e}")
        sync_errors.append(f"CDB: {e}")

    try:
        sync_result = sync_authoritative_card_vocabulary(card_vocab_url)
        vocabulary = sync_result["vocabulary"]
        print(
            f"✅ Galatea 权威卡片词表同步完成："
            f"{vocabulary.card_count}/{vocabulary.capacity} "
            f"({sync_result['status']})"
        )
    except Exception as e:
        print(f"⚠️ 权威卡片词表下载失败，已保留本地映射: {e}")
        sync_errors.append(f"card vocabulary: {e}")

    try:
        from card_vocab import (
            find_card_vocabulary_cdb_gaps,
            get_default_card_vocabulary,
        )
        vocabulary = get_default_card_vocabulary()
        missing_codes = find_card_vocabulary_cdb_gaps(vocabulary, cdb_path)
        if missing_codes:
            print(
                f"⚠️ 当前 CDB 有 {len(missing_codes)} 张新卡尚未进入权威词表；"
                "包含这些卡的卡组会暂时退出训练抽样。"
            )
        else:
            print("✅ 权威词表已完整覆盖当前 CDB。")
    except Exception as e:
        print(f"⚠️ 卡库/词表覆盖检查失败: {e}")

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
        sync_errors.append(f"Lua scripts: {e}")

    if sync_errors:
        raise RuntimeError(
            "data synchronization was incomplete: " + " | ".join(sync_errors)
        )


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
