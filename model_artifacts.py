# 本文件统一管理检查点、ONNX 外置权重及部署包中的模型产物

import json
import hashlib
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
import unicodedata
from pathlib import Path, PurePosixPath

from card_vocab import (
    CARD_VOCAB_FILENAME,
    get_default_card_vocabulary,
    load_card_vocabulary,
)
from checkpoint_utils import (
    CHECKPOINT_FORMAT_VERSION,
    MODEL_PROTOCOL_VERSION,
    inspect_training_checkpoint,
    validate_training_checkpoint_file,
    validate_model_id,
)
from protocol_schema import (get_current_protocol_metadata, validate_protocol_metadata,
                             SemanticLookupMismatchError, ObservationProtocolMismatchError)
from training_validation import validate_model_prefix
from semantic_assets import (
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
    HASH_MAPPING_FILENAME,
    KNOWLEDGE_BASE_FILENAME,
    SEMANTIC_ASSET_FILENAMES,
    STATIC_SEMANTIC_ASSET_FILENAMES,
    SEMANTIC_COMPATIBILITY_FILENAME,
    validate_semantic_bundle,
)
from semantic_lookup import (
    ensure_static_semantic_assets,
    load_static_semantic_assets,
    write_static_semantic_assets,
    get_static_semantic_lookup,
    semantic_compatibility_descriptor,
)
from code_semantic_provenance import CODE_EMBEDDINGS_META_FILENAME


ARTIFACT_MANIFEST_FORMAT_VERSION = 3
# 仅表示 .gkg 部署包协议，必须独立于 WebUI/框架版本维护
DEPLOY_PACKAGE_FORMAT_VERSION = 5
LEGACY_DEPLOY_PACKAGE_FORMAT_VERSION = 4
DEPLOY_CONTENT_FORMAT_VERSION = 1
MAX_ONNX_GRAPH_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_MODEL_ARTIFACT_FILE_BYTES = 32 * 1024 * 1024 * 1024
MAX_MODEL_ARTIFACT_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_DEPLOY_PACKAGE_FILE_BYTES = 64 * 1024 * 1024 * 1024
MAX_DEPLOY_ARCHIVE_MEMBERS = 256
MAX_DEPLOY_ARCHIVE_MEMBER_BYTES = 32 * 1024 * 1024 * 1024
MAX_DEPLOY_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_DEPLOY_COMPRESSION_RATIO = 1000
MAX_MANIFEST_FILE_BYTES = 2 * 1024 * 1024
MAX_DEPLOY_CONTENT_FILES = 16384
MAX_DEPLOY_CONTENT_MANIFEST_BYTES = 32 * 1024 * 1024
DEPLOY_CONTENT_SUFFIXES = {
    'decks': ('.ydk',),
    'replays': ('.json', '.jsonl', '.jsonl.gz', '.yrp', '.yrp2', '.yrp3', '.yrp3d', '.yrpx'),
    'documents': ('.md', '.txt', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'),
    'presets': ('.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.txt'),
}
SAFE_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SAFE_PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
MODEL_ARTIFACT_SUFFIXES = (".artifacts.json", ".onnx.data", ".onnx", ".pth")
DEPLOY_ROOT_FILES = {
    *SEMANTIC_ASSET_FILENAMES,
    *STATIC_SEMANTIC_ASSET_FILENAMES,
    CARD_VOCAB_FILENAME,
    "meta_staples.json",
}
CODE_SEMANTIC_FILE_SET = {
    CODE_EMBEDDINGS_FILENAME,
    CODE_EMBEDDINGS_INDEX_FILENAME,
}
STATIC_SEMANTIC_FILE_SET = set(STATIC_SEMANTIC_ASSET_FILENAMES)
ONNX_IDENTITY_KEYS = {
    "model_id": "galatea.model_id",
    "model_prefix": "galatea.model_prefix",
    "iteration": "galatea.iteration",
    "model_protocol_version": "galatea.model_protocol_version",
    "protocol_schema_revision": "galatea.protocol_schema_revision",
    "protocol_schema_hash": "galatea.protocol_schema_hash",
    "card_vocab_hash": "galatea.card_vocab_hash",
    "card_vocab_size": "galatea.card_vocab_size",
    "card_vocab_card_count": "galatea.card_vocab_card_count",
    "semantic_lookup_format_version": "galatea.semantic_lookup_format_version",
    "semantic_lookup_hash": "galatea.semantic_lookup_hash",
    "semantic_lookup_card_count": "galatea.semantic_lookup_card_count",
}


def validate_safe_filename(filename, *, allowed_suffixes=None):
    """校验不含路径、控制字符和跨平台保留名的普通文件名"""
    if not isinstance(filename, str) or not filename:
        raise ValueError("filename must be a non-empty string")
    if filename != Path(filename).name or "/" in filename or "\\" in filename:
        raise ValueError(f"filename must not contain a path: {filename!r}")
    if ".." in filename or not SAFE_FILENAME_PATTERN.fullmatch(filename):
        raise ValueError(f"filename contains unsupported characters: {filename!r}")
    if len(filename.encode("utf-8")) > 200:
        raise ValueError("filename exceeds the 200-byte safety limit")
    device_stem = filename.split(".", 1)[0].upper()
    if device_stem in WINDOWS_RESERVED_FILENAMES:
        raise ValueError(f"reserved filename is not allowed: {filename!r}")
    if allowed_suffixes and not any(
        filename.casefold().endswith(suffix.casefold()) for suffix in allowed_suffixes
    ):
        raise ValueError(f"filename has an unsupported suffix: {filename!r}")
    return filename


def validate_model_artifact_filename(filename):
    """校验检查点、ONNX、外置权重或轮次清单的模型产物文件名"""
    return validate_safe_filename(filename, allowed_suffixes=MODEL_ARTIFACT_SUFFIXES)


def validate_model_artifact_file(path):
    """校验模型制品是大小受限且不经过符号链接的普通文件"""
    artifact_path = Path(path)
    validate_model_artifact_filename(artifact_path.name)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise ValueError(f"model artifact must be a regular file: {artifact_path.name}")
    file_size = artifact_path.stat().st_size
    if file_size > MAX_MODEL_ARTIFACT_FILE_BYTES:
        raise ValueError(
            f"model artifact exceeds the 32 GiB safety limit: {artifact_path.name}"
        )
    return file_size


def validate_model_artifact_file_set(model_dir, filenames):
    """校验一组模型制品的单文件和总容量边界并返回总字节数"""
    model_root = Path(model_dir).resolve()
    total_size = 0
    for filename in filenames:
        safe_name = validate_model_artifact_filename(filename)
        total_size += validate_model_artifact_file(model_root / safe_name)
        if total_size > MAX_MODEL_ARTIFACT_TOTAL_BYTES:
            raise ValueError("model artifact set exceeds the 64 GiB safety limit")
    return total_size


def validate_package_name(package_name):
    """校验可安全用于跨平台部署包文件名的包名"""
    if not isinstance(package_name, str) or not SAFE_PACKAGE_NAME_PATTERN.fullmatch(
        package_name
    ):
        raise ValueError(
            "package_name must start with an ASCII letter or digit, contain only "
            "letters, digits, '.', '_' or '-', and be at most 64 characters"
        )
    if ".." in package_name or package_name.split(".", 1)[0].upper() in WINDOWS_RESERVED_FILENAMES:
        raise ValueError("package_name is not safe for a local filename")
    return package_name


def validate_deployment_package_filename(filename):
    """校验本地部署包必须是安全的单层 .gkg 文件名"""
    return validate_safe_filename(filename, allowed_suffixes=(".gkg",))


def _validate_content_relative_path(value, *, maximum_depth=16, maximum_bytes=1024):
    """保留中文与目录层级，同时拒绝跨平台歧义和越界路径"""
    if (not isinstance(value, str) or not value or '\\' in value
            or len(value.encode('utf-8')) > maximum_bytes):
        raise ValueError(f'unsafe content path: {value!r}')
    parts = value.split('/')
    if len(parts) > maximum_depth:
        raise ValueError(f'content path is too deep: {value!r}')
    for part in parts:
        if (part in ('', '.', '..') or part.startswith('.') or part.endswith(('.', ' '))
                or len(part.encode('utf-8')) > 200
                or any(ord(char) < 32 or ord(char) == 127 or char in '<>:"|?*' for char in part)
                or part.split('.', 1)[0].upper() in WINDOWS_RESERVED_FILENAMES):
            raise ValueError(f'unsafe content path: {value!r}')
    return value


def _validate_content_archive_path(value, *, category=None):
    """附件只能进入清单定义的内容分区，不能伪装为模型或根资产"""
    _validate_content_relative_path(value)
    parts = value.split('/')
    if len(parts) < 3 or parts[0] != 'content' or parts[1] not in DEPLOY_CONTENT_SUFFIXES:
        raise ValueError(f'unsafe content archive path: {value!r}')
    if category is not None and parts[1] != category:
        raise ValueError(f'content category does not match its path: {value!r}')
    if not value.casefold().endswith(DEPLOY_CONTENT_SUFFIXES[parts[1]]):
        raise ValueError(f'unsupported content file type: {value!r}')
    return parts[1], '/'.join(parts[2:])


def _content_path_key(value):
    """统一识别大小写与Unicode等价路径，避免不同平台覆盖同一文件"""
    return unicodedata.normalize('NFC', value).casefold()


def _check_deployment_path_collisions(paths):
    """预先拒绝文件与目录重名，以及大小写不同的目录别名"""
    nodes = {}
    for value in paths:
        parts = value.split('/')
        for index in range(1, len(parts) + 1):
            path = '/'.join(parts[:index])
            key = _content_path_key(path)
            kind = 'file' if index == len(parts) else 'directory'
            previous = nodes.get(key)
            if previous is not None and (previous != (path, kind) or kind == 'file'):
                raise ValueError(f'duplicate or conflicting archive path: {value!r}')
            nodes[key] = (path, kind)


def _reject_deployment_links(path):
    """文件和上层目录都不能经过符号链接或Windows目录联接"""
    raw = Path(path).absolute()
    for current in (raw, *raw.parents):
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError(f'deployment path must not traverse a link: {raw}')
    return raw


def discover_deployment_content(source_root, category):
    """只列出用户指定目录中的可携带附件，不读取内容或自动纳入训练"""
    if category not in DEPLOY_CONTENT_SUFFIXES:
        raise ValueError(f'unknown content category: {category!r}')
    root = _reject_deployment_links(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f'content directory does not exist: {root}')
    result = {}
    for directory, folders, names in os.walk(root, followlinks=False):
        folders[:] = sorted(name for name in folders if not name.startswith('.')
                            and name not in {'python_env', '__pycache__', 'node_modules'})
        for folder in list(folders):
            candidate = Path(directory) / folder
            if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
                folders.remove(folder)
        for name in sorted(names):
            source = Path(directory) / name
            if name.startswith('.') or not name.casefold().endswith(DEPLOY_CONTENT_SUFFIXES[category]):
                continue
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(root).as_posix()
            _validate_content_archive_path(f'content/{category}/{relative}')
            result[relative] = str(source)
            if len(result) > MAX_DEPLOY_CONTENT_FILES:
                raise ValueError(f'content directory exceeds {MAX_DEPLOY_CONTENT_FILES} files; select a smaller directory')
    _check_deployment_path_collisions(result)
    return result


def _prepare_deployment_content(optional_files):
    """展开显式选中的附件并检查数量、类型与容量，不自动扫描源目录"""
    if optional_files is None:
        return []
    if not isinstance(optional_files, dict) or set(optional_files) - set(DEPLOY_CONTENT_SUFFIXES):
        raise ValueError('optional_files must map content categories to relative-path/source mappings')
    records = []
    for category, files in sorted(optional_files.items()):
        if not isinstance(files, dict):
            raise ValueError(f'content category must contain a file mapping: {category}')
        for relative, source in sorted(files.items()):
            _validate_content_relative_path(relative)
            archive_path = f'content/{category}/{relative}'
            _validate_content_archive_path(archive_path, category=category)
            raw = _reject_deployment_links(source)
            if not raw.is_file() or not stat.S_ISREG(raw.stat().st_mode):
                raise ValueError(f'content source must be a regular file: {raw}')
            size = raw.stat().st_size
            if size > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
                raise ValueError(f'content source exceeds the member size limit: {raw}')
            records.append({'category': category, 'path': archive_path, 'relative_path': relative,
                            'size': size, 'source': raw})
            if len(records) > MAX_DEPLOY_CONTENT_FILES:
                raise ValueError(f'too many optional content files; limit={MAX_DEPLOY_CONTENT_FILES}')
    _check_deployment_path_collisions([row['path'] for row in records])
    return records


def _deployment_file_sha256(path):
    """分块计算附件摘要，避免读取大型录像到内存"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _deployment_content_records(manifest):
    """独立读取附件合同，旧包没有附件区，新包逐项声明路径与摘要"""
    version = manifest.get('package_format_version')
    if version == LEGACY_DEPLOY_PACKAGE_FORMAT_VERSION:
        if 'optional_content' in manifest:
            raise ValueError('GKG4 cannot declare optional content')
        return []
    if version != DEPLOY_PACKAGE_FORMAT_VERSION:
        raise ValueError(f'unsupported deployment package format: {version!r}')
    content = manifest.get('optional_content')
    if not isinstance(content, dict) or content.get('format_version') != DEPLOY_CONTENT_FORMAT_VERSION:
        raise ValueError('GKG5 requires the versioned optional_content section')
    records = content.get('files')
    if not isinstance(records, list) or not 0 < len(records) <= MAX_DEPLOY_CONTENT_FILES:
        raise ValueError('optional content file list is empty or exceeds its limit')
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('optional content records must be objects')
        category, relative = _validate_content_archive_path(row.get('path'), category=row.get('category'))
        if category != row.get('category') or relative != row.get('relative_path'):
            raise ValueError('optional content category or relative path does not match')
        size = row.get('size')
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
            raise ValueError('invalid optional content size')
        if not isinstance(row.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid optional content sha256')
    _check_deployment_path_collisions([row['path'] for row in records])
    return records


def _scan_deployment_stage(stage_root, *, has_content):
    """有界扫描暂存目录，模型保持单层，只有附件允许进入专属子目录"""
    files, directories, total = set(), set(), 0
    limit = MAX_DEPLOY_ARCHIVE_MEMBERS + (MAX_DEPLOY_CONTENT_FILES if has_content else 0)
    for directory, folders, names in os.walk(stage_root, followlinks=False):
        for name in folders:
            path = _reject_deployment_links(Path(directory) / name)
            relative = path.relative_to(stage_root).as_posix()
            _validate_content_relative_path(relative)
            parts = relative.split('/')
            if (not has_content or parts[0] != 'content'
                    or len(parts) > 1 and parts[1] not in DEPLOY_CONTENT_SUFFIXES):
                raise ValueError(f'undeclared deployment directory: {relative}')
            directories.add(relative)
        for name in names:
            path = _reject_deployment_links(Path(directory) / name)
            relative = path.relative_to(stage_root).as_posix()
            if '/' in relative:
                if not has_content:
                    raise ValueError(f'unsafe deployment member path: {relative}')
                _validate_content_archive_path(relative)
            else:
                validate_safe_filename(relative)
            if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(f'deployment member must be a regular file: {relative}')
            size = path.stat().st_size
            if size > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
                raise ValueError(f'deployment member is too large: {relative}')
            total += size
            files.add(relative)
            if len(files) > limit or total > MAX_DEPLOY_ARCHIVE_TOTAL_BYTES:
                raise ValueError('deployment stage exceeds its file count or total size limit')
    _check_deployment_path_collisions(sorted(files))
    return files, directories


def _content_import_manifest(stage_dir):
    """附件导入只读取自身合同，不加载模型或解释预设内容"""
    root = _reject_deployment_links(stage_dir).resolve()
    source = _reject_deployment_links(root / 'manifest.json')
    if not source.is_file() or source.stat().st_size > MAX_DEPLOY_CONTENT_MANIFEST_BYTES:
        raise ValueError('deployment manifest is absent or too large')
    manifest = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('deployment manifest must be an object')
    validate_package_name(manifest.get('package_name'))
    records = _deployment_content_records(manifest)
    return root, manifest, records


def _content_short_label(value, maximum=60):
    """缩短展示名称时保留完整中文字符，原始路径仍写入导入清单"""
    result = ''
    for character in value:
        if len((result + character).encode('utf-8')) > maximum:
            break
        result += character
    return result.rstrip('. ') or 'root'


def plan_deployment_content_import(stage_dir, project_root='.', *, selected_paths=None):
    """预览附件的独立落点，同名同内容复用，已有不同内容明确拒绝覆盖"""
    stage, manifest, records = _content_import_manifest(stage_dir)
    root = _reject_deployment_links(project_root).resolve()
    declared = {row['path']: row for row in records}
    selected = set(declared) if selected_paths is None else set(selected_paths)
    if selected - set(declared):
        raise ValueError('selected content is not declared by the package')
    model_references = [{key: record.get(key) for key in
        ('model_id', 'model_prefix', 'iteration', 'training_stage', 'training_lineage', 'format', 'primary')}
        for record in manifest.get('model_artifacts', [])]
    identity = hashlib.sha256(json.dumps({'package_name': manifest['package_name'],
        'models': model_references, 'files': records},
        sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
    group = f'{manifest["package_name"]}_{identity}'
    planned = []
    for name in sorted(selected):
        row = declared[name]
        source = _reject_deployment_links(stage / name)
        if (not source.is_file() or source.stat().st_size != row['size']
                or _deployment_file_sha256(source) != row['sha256']):
            raise ValueError(f'optional content checksum or size mismatch: {name}')
        relative = PurePosixPath(row['relative_path'])
        category = row['category']
        if category == 'decks':
            parent = relative.parent.as_posix()
            pool_hash = hashlib.sha256(parent.encode('utf-8')).hexdigest()[:16]
            label = _content_short_label(relative.parent.name or 'root')
            pool = f'GKG_{group}__{label}_{pool_hash}'
            # 原加载器按小写扩展名定位YDK，保留卡组名并只统一扩展名
            destination = PurePosixPath('decks', pool, relative.stem + '.ydk')
        elif category == 'replays' and relative.parts[0] == 'holographic':
            if relative.suffix.lower() != '.json':
                raise ValueError('holographic replay content must be a JSON file')
            path_hash = hashlib.sha256(row['relative_path'].encode('utf-8')).hexdigest()[:16]
            label = _content_short_label(relative.stem)
            destination = PurePosixPath('ai_thoughts', f'GKG_{group}_{path_hash}_{label}.json')
        elif category == 'replays':
            raw_relative = PurePosixPath(*relative.parts[1:]) if relative.parts[0] == 'raw' else relative
            if str(raw_relative) == '.':
                raise ValueError('raw replay path must identify a file')
            destination = PurePosixPath('replays', 'imported', group, raw_relative)
        else:
            destination = PurePosixPath('deploy_packages', 'imported', group, category, relative)
        target = _reject_deployment_links(root / destination)
        target.resolve().relative_to(root)
        _validate_content_relative_path(destination.as_posix(), maximum_depth=32, maximum_bytes=2048)
        status = 'new'
        if target.exists():
            status = ('present' if target.is_file() and target.stat().st_size == row['size']
                      and _deployment_file_sha256(target) == row['sha256'] else 'conflict')
        planned.append(dict(row, destination=destination.as_posix(), status=status))
    _check_deployment_path_collisions([row['destination'] for row in planned])
    return {'package_name': manifest['package_name'], 'group': group, 'files': planned,
            'models': model_references,
            'receipt': f'deploy_packages/imported/{group}/contents.json'}


def install_deployment_content(stage_dir, project_root='.', *, selected_paths=None):
    """只安装显式附件，失败回收本次新增文件，不修改模型资产或运行参数"""
    plan = plan_deployment_content_import(stage_dir, project_root, selected_paths=selected_paths)
    if any(row['status'] == 'conflict' for row in plan['files']):
        conflicts = [row['destination'] for row in plan['files'] if row['status'] == 'conflict']
        raise FileExistsError(f'content import would overwrite existing files: {conflicts}')
    if not plan['files']:
        return dict(plan, installed=0, reused=0)
    root = _reject_deployment_links(project_root).resolve()
    stage = _reject_deployment_links(stage_dir).resolve()
    created_files, created_directories, temporary_paths = [], [], []
    receipt_path = _reject_deployment_links(root / plan['receipt'])
    previous = {}
    if receipt_path.exists():
        if not receipt_path.is_file() or receipt_path.stat().st_size > 2 * MAX_DEPLOY_CONTENT_MANIFEST_BYTES:
            raise ValueError('existing content receipt is not a supported regular file')
        payload = json.loads(receipt_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or payload.get('group') != plan['group'] or not isinstance(payload.get('files'), list):
            raise ValueError('existing content receipt is invalid')
        previous = {row['path']: row for row in payload['files']}

    def prepare_parent(target):
        """只创建当前工程边界内的缺失目录，回滚时仅清理本次创建部分"""
        pending, parent = [], target.parent
        while parent != root and not parent.exists():
            pending.append(parent)
            parent = parent.parent
        for directory in reversed(pending):
            _reject_deployment_links(directory)
            directory.resolve().relative_to(root)
            directory.mkdir()
            created_directories.append(directory)

    try:
        pending_files = []
        for row in plan['files']:
            if row['status'] == 'present':
                continue
            target = root / row['destination']
            prepare_parent(target)
            with tempfile.NamedTemporaryFile(prefix='.gkg-', suffix='.tmp', dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
            temporary_paths.append(temporary)
            shutil.copyfile(stage / row['path'], temporary)
            if temporary.stat().st_size != row['size'] or _deployment_file_sha256(temporary) != row['sha256']:
                raise ValueError(f'content changed while importing: {row["path"]}')
            pending_files.append((temporary, target))
        for temporary, target in pending_files:
            _reject_deployment_links(target)
            target.resolve().relative_to(root)
            # 排他写入避免并发导入覆盖刚出现的文件
            with open(target, 'xb') as output:
                created_files.append(target)
                with open(temporary, 'rb') as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        prepare_parent(receipt_path)
        for row in plan['files']:
            previous[row['path']] = {key: value for key, value in row.items() if key != 'status'}
        receipt = {'format_version': DEPLOY_CONTENT_FORMAT_VERSION, 'package_name': plan['package_name'],
                   'group': plan['group'], 'models': plan['models'],
                   'files': [previous[key] for key in sorted(previous)]}
        receipt_bytes = json.dumps(receipt, ensure_ascii=False, indent=2).encode('utf-8')
        if len(receipt_bytes) > 2 * MAX_DEPLOY_CONTENT_MANIFEST_BYTES:
            raise ValueError('content import receipt is too large')
        with tempfile.NamedTemporaryFile(prefix='.gkg-', suffix='.tmp', dir=receipt_path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(receipt_bytes)
        temporary_paths.append(temporary)
        os.replace(temporary, receipt_path)
    except BaseException:
        for path in temporary_paths:
            path.resolve().relative_to(root)
            path.unlink(missing_ok=True)
        for path in reversed(created_files):
            path.resolve().relative_to(root)
            path.unlink(missing_ok=True)
        for path in reversed(created_directories):
            try:
                path.resolve().relative_to(root)
                path.rmdir()
            except OSError:
                pass
        raise
    finally:
        for path in temporary_paths:
            path.resolve().relative_to(root)
            path.unlink(missing_ok=True)
    return dict(plan, installed=len(created_files), reused=sum(row['status'] == 'present' for row in plan['files']))


def model_artifact_stem(model_prefix, iteration, bc_epochs=None):
    """根据已校验前缀和轮次生成一组模型产物的公共文件名"""
    validate_model_prefix(model_prefix)
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    if bc_epochs is not None and (type(bc_epochs) is not int or bc_epochs < 0):
        raise ValueError('BC epochs must be a non-negative integer')
    return f"{model_prefix}_iter_{iteration}" + (f"_bc_{bc_epochs}" if bc_epochs else '')


def is_primary_model_filename(filename):
    """判断文件是否为可独立选择的检查点或 ONNX 主图"""
    try:
        name = validate_safe_filename(filename, allowed_suffixes=(".pth", ".onnx"))
    except ValueError:
        return False
    return name.casefold().endswith(".pth") or name.casefold().endswith(".onnx")


def _onnx_update_examples(network):
    """生成固定输入合同的未知卡、语义上下文和词表尾部样本，不启动决斗"""
    import torch
    from feature_encoder import GalateaEncoder
    from gamestate import DuelState

    empty = GalateaEncoder().encode(DuelState((), (), (), ()).get_snapshot(), player_id=0)
    empty['card_idx'][:, 0] = 1
    empty['padding_mask'][:, 0] = True
    empty['act_mask'][:, 0] = True
    lookup = get_static_semantic_lookup()
    semantic_tokens = torch.nonzero(network.semantic_effect_mask_table.any(dim=-1)).flatten()[:12]
    tail_tokens = torch.tensor(sorted(lookup.card_vocabulary.cards.values())[-12:], dtype=torch.long)
    examples = [empty]
    for tokens in (semantic_tokens, tail_tokens):
        batch = {name: value.clone() for name, value in empty.items()}
        if not len(tokens):
            examples.append(batch)
            continue
        for key, mask in (('card_idx', 'padding_mask'), ('deck_idx', 'deck_mask'),
                          ('c_card_idx', 'c_mask'), ('h_card_idx', 'h_mask'),
                          ('event_card_idx', 'event_mask'), ('deck_profile_card_idx', 'deck_profile_mask')):
            count = min(len(tokens), batch[key].shape[1])
            batch[key][:, :count] = tokens[:count]
            batch[mask][:, :count] = True
        count = min(len(tokens), 12)
        batch['deck_profile_section'][:, :count] = 1
        batch['deck_profile_initial_count'][:, :count] = 1
        batch['deck_profile_remaining_count'][:, :count] = 1
        batch['c_context'][:, :count, 4] = torch.arange(1, count + 1, dtype=torch.float16) / 12.0
        batch['act_mask'][:, :2] = True
        batch['act_code'][:, :min(2, len(tokens))] = tokens[:2]
        batch['deck_film_mask'].fill_(True)
        # 验证当前提示、链节状态与公开指示对象，不能只导出全空新输入
        batch['current_context_mask'][:, :3] = True
        batch['current_context_kind'][:, :3] = torch.tensor([2, 70, 83], dtype=torch.uint8)
        batch['current_context_chain'][:, :3] = torch.tensor([0, 1, 1], dtype=torch.uint8)
        batch['current_context_subtype'][:, :3] = torch.tensor([3, 67, 3], dtype=torch.uint8)
        batch['current_context_value'][:, 0, :2] = torch.tensor([245, 1], dtype=torch.uint8)
        batch['current_context_location'][:, 2] = torch.tensor([2, 3, 1, 1], dtype=torch.uint8)
        batch['current_context_code'][:, 2] = tokens[0]
        examples.append(batch)
    return examples


def _export_updated_onnx(checkpoint, graph_path):
    """以当前兼容语义和原学习权重导出，并用三组输入对照CPU输出"""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from galatea_net import GalateaNet

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, network, keys):
            super().__init__()
            self.net = network
            self.keys = keys

        def forward(self, *values):
            logits, value, _ = self.net(dict(zip(self.keys, values)))
            return logits, value

    network = GalateaNet(checkpoint['net_config']).cpu().eval()
    network.load_state_dict(checkpoint['model_state_dict'], strict=True)
    examples = _onnx_update_examples(network)
    keys = list(examples[0])
    wrapper = ONNXWrapper(network, keys).eval()
    with torch.no_grad():
        torch.onnx.export(wrapper, tuple(examples[0][key] for key in keys), str(graph_path),
            export_params=True, opset_version=18, do_constant_folding=True, external_data=True,
            input_names=keys, output_names=['action_logits', 'values'], dynamo=False)
    tag_onnx_model_identity(graph_path, model_id=checkpoint['model_id'],
        model_prefix=checkpoint['model_prefix'], iteration=checkpoint['iteration'],
        **({'training_lineage': checkpoint['training_lineage']} if checkpoint.get('training_lineage') else {}))
    onnx.checker.check_model(str(graph_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph_path), sess_options=options, providers=['CPUExecutionProvider'])
    inputs = {item.name for item in session.get_inputs()}
    from public_observation import CURRENT_CONTEXT_INPUT_SPECS
    if (not inputs.issubset(keys) or not set(CURRENT_CONTEXT_INPUT_SPECS).issubset(inputs)
            or [item.name for item in session.get_outputs()] != ['action_logits', 'values']):
        raise ValueError('导出的ONNX输入或输出不符合当前模型推理合同')
    errors = []
    with torch.no_grad():
        for case_index, batch in enumerate(examples):
            expected = network(batch)[:2]
            actual = session.run(None, {key: batch[key].numpy() for key in inputs})
            case_errors = []
            for output_index, (reference, result) in enumerate(zip(expected, actual)):
                reference = reference.numpy()
                if reference.shape != result.shape or not np.isfinite(result).all():
                    raise ValueError('ONNX输出形状异常或包含非有限数值')
                np.testing.assert_allclose(result, reference, atol=1e-4, rtol=1e-4,
                    err_msg=f'ONNX验证样本{case_index}，输出{output_index}')
                case_errors.append(float(np.max(np.abs(reference - result))))
            errors.append(case_errors)
    return {'cases': len(examples), 'input_count': len(inputs), 'output_names': ['action_logits', 'values'],
            'max_absolute_errors': errors, 'atol': 1e-4, 'rtol': 1e-4}


def _check_onnx_update_assets():
    """只读核对现有运行资产，检查入口不触发编译或补写资产"""
    root = Path('.').resolve()
    vocabulary = load_card_vocabulary(root / CARD_VOCAB_FILENAME)
    source_names = (KNOWLEDGE_BASE_FILENAME, CODE_EMBEDDINGS_FILENAME, CODE_EMBEDDINGS_INDEX_FILENAME)
    load_static_semantic_assets(root, card_vocabulary=vocabulary,
        verify_source_files=all((root / name).is_file() for name in source_names))
    return get_current_protocol_metadata()


def _onnx_update_target(source, metadata):
    """读取旧图的身份和依赖供原位替换使用，不将旧图放行给推理"""
    import onnx

    graph = source.with_suffix('.onnx')
    marker = checkpoint_artifact_manifest_path(source)
    for path in (graph, marker):
        if path.is_symlink() or path.exists() and not path.is_file():
            raise ValueError(f'ONNX更新目标必须是普通文件: {path}')
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            payload = {}
        if isinstance(payload, dict) and payload.get('model_id') not in (None, metadata['model_id']):
            raise PermissionError(f'同名制品清单属于其他模型，不能覆盖: {marker.name}')
    if not graph.exists():
        return set()
    try:
        model = onnx.load(str(graph), load_external_data=False)
    except Exception:
        # 同名损坏图可从已验证PTH修复，不依赖损坏图反推权重
        return {f'{graph.name}.data'} if graph.with_name(f'{graph.name}.data').is_file() else set()
    properties = {item.key: item.value for item in model.metadata_props}
    expected = {'model_id': metadata['model_id'], 'model_prefix': metadata['model_prefix'],
                'iteration': str(metadata['training_lineage']['ppo_iteration'])}
    for key, value in expected.items():
        old = properties.get(ONNX_IDENTITY_KEYS[key])
        if old is not None and old != value:
            raise PermissionError(f'同名ONNX的{key}与PTH不一致，不能覆盖: {graph.name}')
    files = set()
    for tensor in _iter_onnx_tensors(model.graph):
        location = {item.key: item.value for item in tensor.external_data}.get('location')
        if location:
            _safe_external_data_path(graph.parent, location)
            if (graph.parent / location).is_symlink():
                raise ValueError(f'旧ONNX外置权重不能是符号链接: {location}')
            files.add(location)
    return files


def inspect_onnx_updates(model_directory='./models', *, checkpoint_paths=None):
    """检查模型目录内的同名PTH与ONNX，不导出、打包或修改文件"""
    protocol = _check_onnx_update_assets()
    pairs = set()
    if checkpoint_paths is None:
        root = Path(model_directory).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'模型目录不存在: {root}')
        for path in root.rglob('*'):
            if (path.suffix.lower() in ('.pth', '.onnx') and path.is_file() and not path.is_symlink()
                    and path.resolve().is_relative_to(root)
                    and not any(part.startswith(('.onnx_update_', '.onnx_recovery_')) for part in path.relative_to(root).parts)):
                pairs.add(path.with_suffix('.pth'))
    else:
        for path in checkpoint_paths:
            path = Path(path)
            if path.suffix.lower() not in ('.pth', '.onnx'):
                raise ValueError('检查对象必须是PTH或ONNX文件')
            pairs.add(path.absolute().with_suffix('.pth'))
    rows = []
    for source in sorted(pairs):
        graph = source.with_suffix('.onnx')
        row = {'checkpoint_path': str(source), 'onnx_path': str(graph), 'can_update': False,
               'needs_update': False, 'status': 'missing', 'reason': ''}
        metadata = None
        if source.is_file():
            try:
                metadata = inspect_training_checkpoint(source)
                if metadata.get('format_warning') or metadata.get('model_protocol_warning'):
                    raise ValueError(metadata.get('format_warning') or metadata['model_protocol_warning'])
                _onnx_update_target(source, metadata)
                row.update(can_update=True, model_id=metadata['model_id'],
                    ppo_iteration=metadata['training_lineage']['ppo_iteration'])
            except Exception as error:
                row.update(status='checkpoint_invalid', needs_update=True, reason=str(error))
                rows.append(row)
                continue
        try:
            record = describe_onnx_artifact(graph,
                expected_model_id=metadata['model_id'] if metadata else None)
            if record['model_id'] is None:
                raise ValueError('ONNX缺少Core模型身份')
            appended = record['card_vocab_card_count'] < protocol['card_vocab_card_count']
            row.update(status='append_only' if appended else 'current', needs_update=appended,
                reason='旧卡语义前缀一致，仍可加载；尚未包含新增卡片语义' if appended else '当前协议、语义及依赖文件一致')
            if not appended and metadata and any(metadata.get(key) != protocol[key] for key in protocol):
                row.update(status='checkpoint_assets_changed', needs_update=True,
                    reason='ONNX可用，PTH记录的词表或语义版本仍较旧，可一并同步资产标记')
        except SemanticLookupMismatchError as error:
            row.update(status='semantic_changed', needs_update=True, reason=str(error))
        except ObservationProtocolMismatchError as error:
            row.update(status='observation_changed', needs_update=True, reason=str(error))
        except FileNotFoundError as error:
            row.update(status='missing' if not graph.exists() else 'incomplete', needs_update=True, reason=str(error))
        except Exception as error:
            row.update(status='invalid', needs_update=True, reason=str(error))
        if not row['can_update']:
            row['reason'] += '；缺少同名PTH，不能在Core内重导出'
        rows.append(row)
    counts = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    return {'operation': 'onnx_check', **protocol, 'models': rows, 'counts': counts}


def _prepare_onnx_update_data(graph):
    """为新外置权重分配独立文件名，主图切换前不改动旧图引用的数据"""
    import uuid
    import onnx

    model = onnx.load(str(graph), load_external_data=False)
    locations = {}
    for tensor in _iter_onnx_tensors(model.graph):
        external = {item.key: item.value for item in tensor.external_data}
        old = external.get('location')
        if not old:
            continue
        if old not in locations:
            source = _safe_external_data_path(graph.parent, old)
            name = f'{graph.stem[:100]}.{uuid.uuid4().hex[:8]}.onnx.data'
            source.rename(graph.parent / name)
            locations[old] = name
        for item in tensor.external_data:
            if item.key == 'location':
                item.value = locations[old]
    if locations:
        onnx.save_model(model, str(graph))


def _refresh_checkpoint_assets(checkpoint, protocol):
    """更新PTH的资产和观测合同来源，学习状态与模型身份保持原样"""
    from protocol_schema import apply_current_protocol_metadata
    from checkpoint_utils import checkpoint_semantic_update, checkpoint_observation_history

    changed = any(checkpoint.get(key) != value for key, value in protocol.items())
    if not changed:
        return checkpoint, False
    refreshed = dict(checkpoint)
    history = list(checkpoint.get('semantic_asset_updates', []))
    asset_changed = any(checkpoint.get(key) != value for key, value in protocol.items()
                        if key not in ('protocol_schema_revision', 'protocol_schema_hash'))
    if asset_changed:
        update = checkpoint_semantic_update(checkpoint) or {
            'status': 'append_only_update',
            'source_semantic_lookup_hash': checkpoint['semantic_lookup_hash'],
            'runtime_semantic_lookup_hash': protocol['semantic_lookup_hash'],
            'message': '保留学习权重与模型身份，同步只追加词表和语义资产版本'}
        update = dict(update, source_card_vocab_hash=checkpoint['card_vocab_hash'],
            source_card_vocab_card_count=checkpoint['card_vocab_card_count'],
            runtime_card_vocab_hash=protocol['card_vocab_hash'],
            runtime_card_vocab_card_count=protocol['card_vocab_card_count'])
        if not history or history[-1] != update:
            history.append(update)
    refreshed.update(protocol)
    refreshed['net_config'] = apply_current_protocol_metadata(checkpoint['net_config'])
    refreshed['semantic_asset_updates'] = history
    refreshed['observation_protocol_updates'] = checkpoint_observation_history(checkpoint)
    compatibility = semantic_compatibility_descriptor(get_static_semantic_lookup())
    if compatibility is not None:
        refreshed['semantic_compatibility'] = compatibility
    else:
        refreshed.pop('semantic_compatibility', None)
        refreshed['net_config'].pop('semantic_compatibility', None)
    return refreshed, True


def _build_onnx_update_manifest(source, checkpoint, record, validation):
    """PTH与ONNX记录一致的当前资产版本，原资产来源留在更新历史中"""
    from checkpoint_utils import checkpoint_semantic_history, get_training_lineage

    payload = {'artifact_manifest_version': ARTIFACT_MANIFEST_FORMAT_VERSION,
        'checkpoint_format_version': checkpoint['checkpoint_format_version'],
        **{key: checkpoint[key] for key in get_current_protocol_metadata()},
        'model_id': checkpoint['model_id'], 'model_prefix': checkpoint['model_prefix'],
        'iteration': checkpoint['iteration'], 'training_stage': checkpoint.get('training_stage', 'ppo'),
        'checkpoint': {'format': 'pytorch_checkpoint', 'file': source.name, 'status': 'complete'},
        'onnx': record, 'onnx_update': {'validation': validation,
            'semantic_asset_updates': checkpoint_semantic_history(checkpoint)}}
    if checkpoint.get('training_lineage') is not None or checkpoint.get('training_stage') == 'imitation':
        payload['training_lineage'] = get_training_lineage(checkpoint)
    compatibility = checkpoint.get('semantic_compatibility') or checkpoint['net_config'].get('semantic_compatibility')
    if compatibility is not None:
        payload['semantic_compatibility'] = compatibility
    if checkpoint.get('semantic_asset_updates'):
        payload['semantic_asset_updates'] = checkpoint['semantic_asset_updates']
    if checkpoint.get('observation_protocol_updates'):
        payload['observation_protocol_updates'] = checkpoint['observation_protocol_updates']
    return payload


class _OnnxUpdateRecoveryError(RuntimeError):
    """替换回滚未完全成功时保留临时备份供恢复"""


def _install_onnx_update(source, stage, record, payload, old_data):
    """先记录更新中，再切换数据、主图和完成清单，失败时恢复原制品"""
    root = source.parent
    graph = source.with_suffix('.onnx')
    marker = checkpoint_artifact_manifest_path(source)
    backup = stage / 'previous'
    backup.mkdir()
    model_files = [*record['external_data'], graph.name]
    if (stage / source.name).is_file():
        model_files.append(source.name)
    managed = [*model_files, marker.name]
    existing = set()
    for name in managed:
        target = root / name
        if name in record['external_data'] and target.exists():
            raise FileExistsError(f'新ONNX外置权重文件名已被占用: {name}')
        if target.is_symlink() or target.exists() and not target.is_file():
            raise ValueError(f'ONNX更新目标不是普通文件: {target}')
        if target.is_file():
            shutil.copy2(target, backup / name)
            existing.add(name)
    pending = dict(payload, onnx={'status': 'in_progress', 'error': 'export_in_progress'})
    pending_file = stage / 'pending.artifacts.json'
    pending_file.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding='utf-8')
    completed = stage / marker.name
    completed.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    written = []
    try:
        os.replace(pending_file, marker)
        written.append(marker.name)
        for name in model_files:
            os.replace(stage / name, root / name)
            written.append(name)
        os.replace(completed, marker)
    except BaseException as error:
        failures = []
        for name in reversed(list(dict.fromkeys(written))):
            try:
                if name in existing:
                    os.replace(backup / name, root / name)
                else:
                    (root / name).unlink(missing_ok=True)
            except OSError as rollback_error:
                failures.append(str(rollback_error))
        if failures:
            raise _OnnxUpdateRecoveryError(f'ONNX替换失败且未完全恢复，原制品备份保留在 {backup}: {failures}') from error
        raise
    # 只清理已不被其他图引用的旧外置权重，不动PTH或其他模型的数据
    stale = old_data.difference(record['external_data'])
    if stale:
        import onnx
        for other in root.glob('*.onnx'):
            if other == graph:
                continue
            try:
                model = onnx.load(str(other), load_external_data=False)
                used = {item.value for tensor in _iter_onnx_tensors(model.graph)
                    for item in tensor.external_data if item.key == 'location'}
                stale.difference_update(used)
            except Exception:
                stale.clear()
                break
    for name in sorted(stale):
        target = _safe_external_data_path(root, name)
        try:
            if not target.is_symlink():
                target.unlink(missing_ok=True)
        except OSError as error:
            print(f'⚠️ [ONNX更新] 旧外置权重未清理 {name}: {error}')


def update_onnx_checkpoints(checkpoint_paths):
    """原位更新同名ONNX及PTH资产标记，学习状态和运行资产保持不变"""
    import torch
    from checkpoint_utils import load_training_checkpoint, get_training_lineage
    from training_lock import TrainerProcessLock

    sources = list(dict.fromkeys(Path(path).absolute() for path in checkpoint_paths))
    if not sources or any(path.suffix.lower() != '.pth' for path in sources):
        raise ValueError('ONNX更新需要选择对应PTH，只有ONNX时可检查但不能重导出')
    report = {'operation': 'onnx_update', 'models': [], 'learning_state_changed': False}
    with TrainerProcessLock():
        protocol = _check_onnx_update_assets()
        report.update(protocol)
        thread_count = torch.get_num_threads()
        mha_fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.set_num_threads(min(thread_count, 4))
            torch.backends.mha.set_fastpath_enabled(False)
            with torch.random.fork_rng(devices=[]):
                for source in sources:
                    source_stat = source.stat()
                    checkpoint = load_training_checkpoint(source, map_location='cpu')
                    lineage = get_training_lineage(checkpoint)
                    metadata = {'model_id': checkpoint['model_id'], 'model_prefix': checkpoint['model_prefix'],
                                'training_lineage': lineage}
                    old_data = _onnx_update_target(source, metadata)
                    graph = source.with_suffix('.onnx')
                    existed = graph.is_file()
                    stage = Path(tempfile.mkdtemp(prefix='.onnx_update_', dir=source.parent))
                    preserve = False
                    try:
                        print(f'⏳ [ONNX更新] {"替换" if existed else "补齐"} {graph.name}，使用CPU和当前兼容语义')
                        refreshed, checkpoint_updated = _refresh_checkpoint_assets(checkpoint, protocol)
                        export_checkpoint = dict(refreshed, iteration=lineage['ppo_iteration'])
                        if checkpoint.get('training_stage') == 'imitation':
                            export_checkpoint['training_lineage'] = lineage
                        validation = _export_updated_onnx(export_checkpoint, stage / graph.name)
                        _prepare_onnx_update_data(stage / graph.name)
                        record = describe_onnx_artifact(stage / graph.name, validate_manifest=False)
                        if checkpoint_updated:
                            torch.save(refreshed, stage / source.name)
                        payload = _build_onnx_update_manifest(source, refreshed, record, validation)
                        staged_marker = stage / checkpoint_artifact_manifest_path(source).name
                        staged_marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                        describe_onnx_artifact(stage / graph.name, expected_model_id=checkpoint['model_id'])
                        if get_current_protocol_metadata() != protocol:
                            raise ValueError('导出期间语义资产发生变化，本模型未替换，请重试')
                        current_stat = source.stat()
                        if (current_stat.st_mtime_ns, current_stat.st_size) != (source_stat.st_mtime_ns, source_stat.st_size):
                            raise ValueError('PTH在导出期间发生变化，本模型未替换，请结束其他写入后重试')
                        _install_onnx_update(source, stage, record, payload, old_data)
                        report['models'].append({'checkpoint_path': str(source), 'onnx_path': str(graph),
                            'action': 'replaced' if existed else 'created', 'model_id': checkpoint['model_id'],
                            'ppo_iteration': lineage['ppo_iteration'], 'bc_epochs': lineage['bc_epochs'],
                            'checkpoint_metadata_updated': checkpoint_updated, 'validation': validation})
                        print(f'✅ [ONNX更新] {graph.name} 已写回原目录，PTH资产标记{"已同步" if checkpoint_updated else "无需变更"}，学习状态保持不变')
                    except _OnnxUpdateRecoveryError:
                        preserve = True
                        raise
                    finally:
                        if not preserve:
                            shutil.rmtree(stage)
                    del checkpoint, refreshed, export_checkpoint
        finally:
            torch.backends.mha.set_fastpath_enabled(mha_fastpath)
            torch.set_num_threads(thread_count)
    return report


def is_artifact_manifest_filename(filename):
    """判断文件是否为训练轮次产物清单"""
    try:
        validate_safe_filename(filename, allowed_suffixes=(".artifacts.json",))
    except ValueError:
        return False
    return filename.casefold().endswith(".artifacts.json")


def checkpoint_artifact_manifest_path(checkpoint_path):
    """返回与检查点同轮次的产物清单路径"""
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}.artifacts.json")


def _iter_onnx_tensors(graph):
    """递归遍历 ONNX 图及子图中的所有张量"""
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _iter_onnx_tensors(attribute.g)
            for child_graph in attribute.graphs:
                yield from _iter_onnx_tensors(child_graph)


def tag_onnx_model_identity(
    onnx_path,
    *,
    model_id,
    model_prefix,
    iteration,
    model_protocol_version=MODEL_PROTOCOL_VERSION,
    training_lineage=None,
):
    """把模型身份与模型协议版本写入 ONNX 主图并原子替换"""
    import onnx

    protocol_metadata = get_current_protocol_metadata()
    validate_model_id(model_id)
    validate_model_prefix(model_prefix)
    model_artifact_stem(model_prefix, iteration)
    if model_protocol_version != MODEL_PROTOCOL_VERSION:
        raise ValueError("ONNX model protocol version does not match the current protocol")
    source_graph_path = Path(onnx_path)
    if source_graph_path.is_symlink():
        raise ValueError("symbolic-link ONNX graphs are not allowed")
    graph_path = source_graph_path.resolve()
    validate_safe_filename(graph_path.name, allowed_suffixes=(".onnx",))
    model = onnx.load(str(graph_path), load_external_data=False)
    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(
        {
            ONNX_IDENTITY_KEYS["model_id"]: model_id,
            ONNX_IDENTITY_KEYS["model_prefix"]: model_prefix,
            ONNX_IDENTITY_KEYS["iteration"]: str(iteration),
            ONNX_IDENTITY_KEYS["model_protocol_version"]: str(
                model_protocol_version
            ),
            ONNX_IDENTITY_KEYS["protocol_schema_revision"]: str(
                protocol_metadata["protocol_schema_revision"]
            ),
            ONNX_IDENTITY_KEYS["protocol_schema_hash"]: protocol_metadata[
                "protocol_schema_hash"
            ],
            ONNX_IDENTITY_KEYS["card_vocab_hash"]: protocol_metadata[
                "card_vocab_hash"
            ],
            ONNX_IDENTITY_KEYS["card_vocab_size"]: str(
                protocol_metadata["card_vocab_size"]
            ),
            ONNX_IDENTITY_KEYS["card_vocab_card_count"]: str(
                protocol_metadata["card_vocab_card_count"]
            ),
            ONNX_IDENTITY_KEYS["semantic_lookup_format_version"]: str(
                protocol_metadata["semantic_lookup_format_version"]
            ),
            ONNX_IDENTITY_KEYS["semantic_lookup_hash"]: protocol_metadata[
                "semantic_lookup_hash"
            ],
            ONNX_IDENTITY_KEYS["semantic_lookup_card_count"]: str(
                protocol_metadata["semantic_lookup_card_count"]
            ),
        }
    )
    if training_lineage is not None:
        from checkpoint_utils import get_training_lineage
        lineage = get_training_lineage({'training_lineage': training_lineage, 'iteration': iteration})
        properties['galatea.training_lineage'] = json.dumps(lineage, sort_keys=True)
    onnx.helper.set_model_props(model, properties)
    temporary_path = graph_path.with_suffix(graph_path.suffix + ".identity.tmp")
    try:
        onnx.save_model(model, str(temporary_path))
        os.replace(temporary_path, graph_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _safe_external_data_path(model_dir, location):
    """解析 ONNX 外置权重路径并拒绝越出模型目录的引用"""
    normalized = str(location).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
        raise ValueError(f"ONNX external data path is unsafe: {location!r}")
    validate_safe_filename(relative.name, allowed_suffixes=(".onnx.data",))

    model_root = Path(model_dir).resolve()
    resolved = (model_root / relative.name).resolve()
    if os.path.commonpath((str(model_root), str(resolved))) != str(model_root):
        raise ValueError(f"ONNX external data escapes model directory: {location!r}")
    return resolved


def _read_artifact_manifest(
    marker_path,
    *,
    card_vocabulary=None,
    semantic_root=".",
):
    """读取并校验当前版本产物清单的基础结构"""
    marker_path = Path(marker_path)
    validate_safe_filename(marker_path.name, allowed_suffixes=(".artifacts.json",))
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(f"artifact manifest must be a regular file: {marker_path.name}")
    if marker_path.stat().st_size > MAX_MANIFEST_FILE_BYTES:
        raise ValueError(f"artifact manifest is too large: {marker_path.name}")
    with open(marker_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact manifest must be an object: {marker_path.name}")
    manifest_version = payload.get("artifact_manifest_version")
    if (
        isinstance(manifest_version, bool)
        or manifest_version != ARTIFACT_MANIFEST_FORMAT_VERSION
    ):
        raise ValueError(f"artifact manifest version mismatch: {marker_path.name}")
    checkpoint_version = payload.get("checkpoint_format_version")
    if (
        isinstance(checkpoint_version, bool)
        or checkpoint_version != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(f"checkpoint format version mismatch: {marker_path.name}")
    model_protocol_version = payload.get("model_protocol_version")
    if (
        isinstance(model_protocol_version, bool)
        or model_protocol_version != MODEL_PROTOCOL_VERSION
    ):
        raise ValueError(f"model protocol version mismatch: {marker_path.name}")
    validate_protocol_metadata(
        payload,
        label=f"artifact manifest {marker_path.name}",
        card_vocabulary=card_vocabulary,
        semantic_root=semantic_root,
        allow_semantic_updates=True,
        allow_observation_updates=True,
    )
    validate_model_id(payload.get("model_id"))
    validate_model_prefix(payload.get("model_prefix"))
    iteration = payload.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"artifact iteration is invalid: {marker_path.name}")
    if payload.get('training_lineage') is not None:
        from checkpoint_utils import get_training_lineage
        get_training_lineage(payload)
    checkpoint = payload.get("checkpoint") or {}
    checkpoint_name = checkpoint.get("file")
    if checkpoint_name is not None:
        validate_safe_filename(checkpoint_name, allowed_suffixes=(".pth",))
    onnx_record = payload.get("onnx") or {}
    for filename in onnx_record.get("files") or []:
        validate_model_artifact_filename(filename)
    return payload


def _validate_onnx_artifact_manifest(
    graph_path,
    record,
    expected_model_id=None,
    *,
    card_vocabulary=None,
    semantic_root=".",
):
    """确认 ONNX 与产物清单属于同一模型且依赖文件完整"""
    marker_path = graph_path.with_name(f"{graph_path.stem}.artifacts.json")
    if not marker_path.is_file():
        if expected_model_id is not None:
            raise FileNotFoundError(
                f"ONNX identity manifest is missing: {marker_path.name}"
            )
        return

    payload = _read_artifact_manifest(
        marker_path,
        card_vocabulary=card_vocabulary,
        semantic_root=semantic_root,
    )
    if expected_model_id is not None and payload["model_id"] != expected_model_id:
        raise PermissionError(
            f"ONNX model_id mismatch: expected {expected_model_id}, "
            f"found {payload['model_id']}"
        )

    onnx_record = payload.get("onnx") or {}
    if onnx_record.get("status") != "complete":
        raise RuntimeError(
            f"ONNX artifact is not marked complete in {marker_path.name}: "
            f"{onnx_record.get('status', 'unknown')}"
        )
    if onnx_record.get("files") != record["files"]:
        raise ValueError(f"ONNX artifact file list mismatch: {marker_path.name}")
    if payload.get('training_lineage') != record.get('training_lineage'):
        raise ValueError(f'ONNX lineage differs from manifest: {marker_path.name}')
    from checkpoint_utils import get_training_lineage
    source_iteration = get_training_lineage(payload)['ppo_iteration']
    for key in (
        "model_id",
        "model_prefix",
        "iteration",
        "model_protocol_version",
        "protocol_schema_revision",
        "protocol_schema_hash",
        "card_vocab_hash",
        "card_vocab_size",
        "card_vocab_card_count",
        "semantic_lookup_format_version",
        "semantic_lookup_hash",
        "semantic_lookup_card_count",
    ):
        expected = source_iteration if key == 'iteration' else payload.get(key)
        if onnx_record.get(key) != expected:
            raise ValueError(
                f"ONNX artifact {key} mismatch: {marker_path.name}"
            )
        # 主清单、PTH及图内身份一致，更新来源记录不代替当前资产校验
        if record.get(key) != onnx_record.get(key):
            raise ValueError(
                f"ONNX embedded {key} mismatch: {marker_path.name}"
            )


def _validate_static_onnx_metadata(metadata, *, card_vocabulary=None, semantic_root='.'):
    """静态图沿用导出时的语义常量，内容更新需要重导出而不是只改元数据"""
    try:
        validate_protocol_metadata(metadata, label='ONNX metadata',
            card_vocabulary=card_vocabulary, semantic_root=semantic_root)
    except SemanticLookupMismatchError as error:
        raise SemanticLookupMismatchError(
            f'{error}; ONNX已固化导出时的静态语义，无法直接替换为当前语义，'
            '需用对应PTH重新导出；编号与编码器兼容的PTH仍可使用新语义'
        ) from error
    except ObservationProtocolMismatchError as error:
        raise ObservationProtocolMismatchError(
            f'{error}；旧ONNX没有当前询问与连锁输入，不能仅改元数据，'
            '请在资源同步中枢执行ONNX检查与更新'
        ) from error


def describe_onnx_artifact(
    onnx_path,
    *,
    require_complete=True,
    validate_manifest=True,
    expected_model_id=None,
    model_id=None,
    model_prefix=None,
    iteration=None,
    model_protocol_version=None,
    card_vocabulary=None,
    semantic_root=".",
):
    """读取 ONNX 主图引用并返回主图与全部外置权重的完整记录"""
    import onnx

    source_graph_path = Path(onnx_path)
    if source_graph_path.is_symlink():
        raise ValueError("symbolic-link ONNX graphs are not allowed")
    graph_path = source_graph_path.resolve()
    validate_safe_filename(graph_path.name, allowed_suffixes=(".onnx",))
    if graph_path.is_symlink() or not graph_path.is_file():
        raise FileNotFoundError(f"ONNX graph does not exist: {graph_path}")
    if graph_path.stat().st_size > MAX_ONNX_GRAPH_FILE_BYTES:
        raise ValueError("ONNX graph exceeds the 2 GiB safety limit")

    model = onnx.load(str(graph_path), load_external_data=False)
    properties = {item.key: item.value for item in model.metadata_props}
    embedded_model_id = properties.get(ONNX_IDENTITY_KEYS["model_id"])
    embedded_model_prefix = properties.get(ONNX_IDENTITY_KEYS["model_prefix"])
    embedded_iteration = properties.get(ONNX_IDENTITY_KEYS["iteration"])
    embedded_model_protocol = properties.get(
        ONNX_IDENTITY_KEYS["model_protocol_version"]
    )
    embedded_protocol_schema_revision = properties.get(
        ONNX_IDENTITY_KEYS["protocol_schema_revision"]
    )
    embedded_protocol_schema_hash = properties.get(
        ONNX_IDENTITY_KEYS["protocol_schema_hash"]
    )
    embedded_card_vocab_hash = properties.get(
        ONNX_IDENTITY_KEYS["card_vocab_hash"]
    )
    embedded_card_vocab_size = properties.get(
        ONNX_IDENTITY_KEYS["card_vocab_size"]
    )
    embedded_card_vocab_card_count = properties.get(
        ONNX_IDENTITY_KEYS["card_vocab_card_count"]
    )
    embedded_semantic_lookup_format_version = properties.get(
        ONNX_IDENTITY_KEYS["semantic_lookup_format_version"]
    )
    embedded_semantic_lookup_hash = properties.get(
        ONNX_IDENTITY_KEYS["semantic_lookup_hash"]
    )
    embedded_semantic_lookup_card_count = properties.get(
        ONNX_IDENTITY_KEYS["semantic_lookup_card_count"]
    )
    if embedded_iteration is not None:
        try:
            embedded_iteration = int(embedded_iteration)
        except ValueError as error:
            raise ValueError("ONNX embedded iteration must be an integer") from error
    if embedded_model_protocol is not None:
        try:
            embedded_model_protocol = int(embedded_model_protocol)
        except ValueError as error:
            raise ValueError(
                "ONNX embedded model_protocol_version must be an integer"
            ) from error
    for name, raw_value in (
        ("protocol_schema_revision", embedded_protocol_schema_revision),
        ("card_vocab_size", embedded_card_vocab_size),
        ("card_vocab_card_count", embedded_card_vocab_card_count),
        (
            "semantic_lookup_format_version",
            embedded_semantic_lookup_format_version,
        ),
        ("semantic_lookup_card_count", embedded_semantic_lookup_card_count),
    ):
        if raw_value is not None:
            try:
                parsed_value = int(raw_value)
            except ValueError as error:
                raise ValueError(f"ONNX embedded {name} must be an integer") from error
            if name == "protocol_schema_revision":
                embedded_protocol_schema_revision = parsed_value
            elif name == "card_vocab_size":
                embedded_card_vocab_size = parsed_value
            elif name == "card_vocab_card_count":
                embedded_card_vocab_card_count = parsed_value
            elif name == "semantic_lookup_format_version":
                embedded_semantic_lookup_format_version = parsed_value
            else:
                embedded_semantic_lookup_card_count = parsed_value

    embedded_identity = (
        embedded_model_id,
        embedded_model_prefix,
        embedded_iteration,
        embedded_model_protocol,
        embedded_protocol_schema_revision,
        embedded_protocol_schema_hash,
        embedded_card_vocab_hash,
        embedded_card_vocab_size,
        embedded_card_vocab_card_count,
        embedded_semantic_lookup_format_version,
        embedded_semantic_lookup_hash,
        embedded_semantic_lookup_card_count,
    )
    if any(value is not None for value in embedded_identity):
        if any(value is None for value in embedded_identity):
            raise ValueError("ONNX embedded model identity is incomplete")
        validate_model_id(embedded_model_id)
        validate_model_prefix(embedded_model_prefix)
        if embedded_model_protocol != MODEL_PROTOCOL_VERSION:
            raise ValueError(
                "ONNX embedded model protocol does not match the current protocol"
            )
        _validate_static_onnx_metadata(
            {
                "model_protocol_version": embedded_model_protocol,
                "protocol_schema_revision": embedded_protocol_schema_revision,
                "protocol_schema_hash": embedded_protocol_schema_hash,
                "card_vocab_hash": embedded_card_vocab_hash,
                "card_vocab_size": embedded_card_vocab_size,
                "card_vocab_card_count": embedded_card_vocab_card_count,
                "semantic_lookup_format_version": embedded_semantic_lookup_format_version,
                "semantic_lookup_hash": embedded_semantic_lookup_hash,
                "semantic_lookup_card_count": embedded_semantic_lookup_card_count,
            },
            card_vocabulary=card_vocabulary,
            semantic_root=semantic_root,
        )
    if expected_model_id is not None and embedded_model_id != expected_model_id:
        raise PermissionError(
            f"ONNX embedded model_id mismatch: expected {expected_model_id}, "
            f"found {embedded_model_id}"
        )
    for name, requested, embedded in (
        ("model_id", model_id, embedded_model_id),
        ("model_prefix", model_prefix, embedded_model_prefix),
        ("iteration", iteration, embedded_iteration),
        (
            "model_protocol_version",
            model_protocol_version,
            embedded_model_protocol,
        ),
    ):
        if requested is not None and requested != embedded:
            raise ValueError(
                f"ONNX embedded {name} does not match the requested identity"
            )
    locations = set()
    for tensor in _iter_onnx_tensors(model.graph):
        external = {item.key: item.value for item in tensor.external_data}
        if "location" in external:
            locations.add(external["location"])

    files = [graph_path.name]
    for location in sorted(locations):
        data_path = _safe_external_data_path(graph_path.parent, location)
        if data_path.is_symlink():
            raise ValueError(
                f"ONNX external data must not be a symbolic link: {location}"
            )
        if require_complete and not data_path.is_file():
            raise FileNotFoundError(
                f"ONNX external data is missing for {graph_path.name}: {location}"
            )
        if data_path.is_file():
            validate_model_artifact_file(data_path)
        files.append(os.path.relpath(data_path, graph_path.parent).replace("\\", "/"))

    training_lineage = json.loads(properties['galatea.training_lineage']) if 'galatea.training_lineage' in properties else None
    if training_lineage is not None:
        from checkpoint_utils import get_training_lineage
        get_training_lineage({'training_lineage': training_lineage, 'iteration': embedded_iteration})
    record = {
        "format": "onnx",
        "training_lineage": training_lineage,
        "model_id": embedded_model_id,
        "model_prefix": embedded_model_prefix,
        "iteration": embedded_iteration,
        "model_protocol_version": embedded_model_protocol,
        "protocol_schema_revision": embedded_protocol_schema_revision,
        "protocol_schema_hash": embedded_protocol_schema_hash,
        "card_vocab_hash": embedded_card_vocab_hash,
        "card_vocab_size": embedded_card_vocab_size,
        "card_vocab_card_count": embedded_card_vocab_card_count,
        "semantic_lookup_format_version": embedded_semantic_lookup_format_version,
        "semantic_lookup_hash": embedded_semantic_lookup_hash,
        "semantic_lookup_card_count": embedded_semantic_lookup_card_count,
        "primary": graph_path.name,
        "files": files,
        "external_data": files[1:],
        "status": "complete",
    }
    if validate_manifest:
        _validate_onnx_artifact_manifest(
            graph_path,
            record,
            expected_model_id=expected_model_id,
            card_vocabulary=card_vocabulary,
            semantic_root=semantic_root,
        )
    return record


def discover_checkpoint_artifacts(
    model_dir,
    *,
    model_prefix=None,
    model_id=None,
    include_orphan_checkpoints=False,
):
    """通过轻量产物清单发现检查点，并按前缀或模型 UUID 精确筛选"""
    model_root = Path(model_dir).resolve()
    records = []
    discovered_paths = set()
    if not model_root.is_dir():
        return records
    for marker_path in model_root.glob("*.artifacts.json"):
        try:
            payload = _read_artifact_manifest(marker_path)
            checkpoint = payload.get("checkpoint") or {}
            checkpoint_name = str(checkpoint.get("file", ""))
            if Path(checkpoint_name).name != checkpoint_name:
                continue
            checkpoint_path = model_root / checkpoint_name
            if checkpoint.get("status") != "complete" or not checkpoint_path.is_file():
                continue
            if (
                model_prefix is not None
                and payload["model_prefix"].casefold() != model_prefix.casefold()
            ):
                continue
            if model_id is not None and payload["model_id"] != model_id:
                continue
            discovered_paths.add(checkpoint_path.resolve())
            records.append(
                {
                    "model_id": payload["model_id"],
                    "model_prefix": payload["model_prefix"],
                    "iteration": int(payload["iteration"]),
                    "training_stage": payload.get('training_stage', 'ppo'),
                    "training_lineage": payload.get('training_lineage'),
                    "checkpoint_format_version": payload.get(
                        "checkpoint_format_version"
                    ),
                    "model_protocol_version": payload.get(
                        "model_protocol_version"
                    ),
                    "protocol_schema_revision": payload.get(
                        "protocol_schema_revision"
                    ),
                    "protocol_schema_hash": payload.get("protocol_schema_hash"),
                    "card_vocab_hash": payload.get("card_vocab_hash"),
                    "card_vocab_size": payload.get("card_vocab_size"),
                    "card_vocab_card_count": payload.get(
                        "card_vocab_card_count"
                    ),
                    "semantic_lookup_format_version": payload.get(
                        "semantic_lookup_format_version"
                    ),
                    "semantic_lookup_hash": payload.get(
                        "semantic_lookup_hash"
                    ),
                    "semantic_lookup_card_count": payload.get(
                        "semantic_lookup_card_count"
                    ),
                    "checkpoint_path": str(checkpoint_path),
                    "manifest_path": str(marker_path),
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

    if include_orphan_checkpoints:
        for checkpoint_path in model_root.glob("*.pth"):
            if checkpoint_path.resolve() in discovered_paths:
                continue
            if checkpoint_artifact_manifest_path(checkpoint_path).is_file():
                continue
            try:
                metadata = inspect_training_checkpoint(checkpoint_path)
                if (
                    metadata["format_warning"] is not None
                    or metadata["model_protocol_warning"] is not None
                ):
                    continue
                if (
                    model_prefix is not None
                    and metadata["model_prefix"].casefold() != model_prefix.casefold()
                ):
                    continue
                if model_id is not None and metadata["model_id"] != model_id:
                    continue
                records.append(
                    {
                        "model_id": metadata["model_id"],
                        "model_prefix": metadata["model_prefix"],
                        "iteration": int(metadata["iteration"]),
                        "checkpoint_format_version": metadata[
                            "checkpoint_format_version"
                        ],
                        "model_protocol_version": metadata[
                            "model_protocol_version"
                        ],
                        "protocol_schema_revision": metadata[
                            "protocol_schema_revision"
                        ],
                        "protocol_schema_hash": metadata["protocol_schema_hash"],
                        "card_vocab_hash": metadata["card_vocab_hash"],
                        "card_vocab_size": metadata["card_vocab_size"],
                        "card_vocab_card_count": metadata[
                            "card_vocab_card_count"
                        ],
                        "semantic_lookup_format_version": metadata[
                            "semantic_lookup_format_version"
                        ],
                        "semantic_lookup_hash": metadata[
                            "semantic_lookup_hash"
                        ],
                        "semantic_lookup_card_count": metadata[
                            "semantic_lookup_card_count"
                        ],
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "manifest_path": None,
                    }
                )
            except (OSError, TypeError, ValueError, KeyError):
                continue
    return sorted(records, key=lambda item: item["iteration"])


def find_prefix_identity_conflicts(model_dir, model_prefix, expected_model_id):
    """查找同前缀但不属于预期 UUID 的检查点，供 WebUI 和 Trainer 告警"""
    validate_model_prefix(model_prefix)
    validate_model_id(expected_model_id)
    return [
        record
        for record in discover_checkpoint_artifacts(
            model_dir,
            model_prefix=model_prefix,
            include_orphan_checkpoints=True,
        )
        if record["model_id"] != expected_model_id
    ]


def find_model_prefix_namespace_files(model_dir, model_prefix):
    """查找会占用规范输出文件名空间的同前缀检查点。"""
    validate_model_prefix(model_prefix)
    model_root = Path(model_dir).resolve()
    canonical_prefix = f"{model_prefix}_iter_".casefold()
    if not model_root.is_dir():
        return []
    return sorted(
        str(path.resolve())
        for path in model_root.glob("*.pth")
        if path.name.casefold().startswith(canonical_prefix)
    )


def assert_checkpoint_target_identity(checkpoint_path, expected_model_id):
    """覆盖已有检查点前核验其清单 UUID，拒绝写坏同名前缀的其他模型"""
    validate_model_id(expected_model_id)
    checkpoint = Path(checkpoint_path).resolve()
    marker_path = checkpoint_artifact_manifest_path(checkpoint)
    if not checkpoint.exists() and not marker_path.exists():
        return
    if not marker_path.is_file():
        raise PermissionError(
            f"refusing to overwrite checkpoint without identity manifest: {checkpoint.name}"
        )
    payload = _read_artifact_manifest(marker_path)
    if payload["model_id"] != expected_model_id:
        raise PermissionError(
            f"refusing to overwrite model_id {payload['model_id']} with "
            f"{expected_model_id}: {checkpoint.name}"
        )


def assert_model_artifact_target_identity(model_dir, record, *, card_vocabulary=None, semantic_root='.'):
    """导入覆盖前核验目标主文件的 UUID、前缀、轮次和格式完全一致"""
    validate_package_model_records([record], expected_model_id=record["model_id"],
        card_vocabulary=card_vocabulary, semantic_root=semantic_root)
    model_root = Path(model_dir).resolve()
    primary_name = validate_safe_filename(
        record["primary"], allowed_suffixes=(".pth", ".onnx")
    )
    target_primary = model_root / primary_name
    existing_files = [
        model_root / validate_model_artifact_filename(filename)
        for filename in record.get("files", [])
        if (model_root / filename).exists()
    ]
    target_marker = checkpoint_artifact_manifest_path(target_primary)
    if target_marker.exists() and target_marker not in existing_files:
        existing_files.append(target_marker)
    if not target_primary.exists():
        if existing_files:
            raise PermissionError(
                f"refusing to overwrite orphan target artifact: {existing_files[0].name}"
            )
        return
    existing = build_package_model_records(model_root, [primary_name])[0]
    for key in ("format", "model_id", "model_prefix", "iteration"):
        if existing.get(key) != record.get(key):
            raise PermissionError(
                f"refusing to overwrite {primary_name}: existing {key} differs"
            )


def collect_model_artifact_files(
    model_dir,
    selected_models,
    *,
    card_vocabulary=None,
    semantic_root=".",
):
    """展开用户选择，自动补齐 ONNX 外置权重和同轮次产物清单"""
    model_root = Path(model_dir).resolve()
    collected = []

    def append_once(relative_name):
        """按原顺序加入文件并避免重复打包"""
        normalized = str(relative_name).replace("\\", "/")
        if normalized not in collected:
            collected.append(normalized)

    for selected in selected_models:
        if not is_primary_model_filename(selected):
            raise ValueError(f"invalid model artifact selection: {selected!r}")
        name = validate_safe_filename(selected, allowed_suffixes=(".pth", ".onnx"))
        primary_path = model_root / name
        if primary_path.is_symlink() or not primary_path.is_file():
            raise FileNotFoundError(f"selected model does not exist: {primary_path}")

        if name.casefold().endswith(".onnx"):
            record = describe_onnx_artifact(
                primary_path,
                require_complete=True,
                card_vocabulary=card_vocabulary,
                semantic_root=semantic_root,
            )
            for relative_name in record["files"]:
                append_once(relative_name)
        else:
            append_once(name)

        marker = f"{primary_path.stem}.artifacts.json"
        if (model_root / marker).is_file():
            append_once(marker)

    validate_model_artifact_file_set(model_root, collected)
    return collected


def build_package_model_records(
    model_dir,
    selected_models,
    *,
    card_vocabulary=None,
    semantic_root=".",
):
    """为部署包清单生成带轮次和依赖文件的模型记录"""
    model_root = Path(model_dir).resolve()
    records = []
    for selected in selected_models:
        if not is_primary_model_filename(selected):
            raise ValueError(f"invalid model artifact selection: {selected!r}")
        name = validate_safe_filename(selected, allowed_suffixes=(".pth", ".onnx"))
        primary_path = model_root / name
        if primary_path.is_symlink() or not primary_path.is_file():
            raise FileNotFoundError(f"selected model does not exist: {primary_path}")
        if name.casefold().endswith(".onnx"):
            records.append(
                describe_onnx_artifact(
                    primary_path,
                    require_complete=True,
                    card_vocabulary=card_vocabulary,
                    semantic_root=semantic_root,
                )
            )
        else:
            checkpoint = validate_training_checkpoint_file(
                primary_path,
                map_location="cpu",
                card_vocabulary=card_vocabulary,
                semantic_root=semantic_root,
            )
            marker_path = checkpoint_artifact_manifest_path(primary_path)
            if marker_path.is_file():
                payload = _read_artifact_manifest(
                    marker_path,
                    card_vocabulary=card_vocabulary,
                    semantic_root=semantic_root,
                )
                for key in (
                    "model_id",
                    "model_prefix",
                    "iteration",
                    "model_protocol_version",
                    "protocol_schema_revision",
                    "protocol_schema_hash",
                    "card_vocab_hash",
                    "card_vocab_size",
                    "card_vocab_card_count",
                    "semantic_lookup_format_version",
                    "semantic_lookup_hash",
                    "semantic_lookup_card_count",
                ):
                    if payload[key] != checkpoint[key]:
                        raise ValueError(
                            f"checkpoint internal {key} does not match "
                            f"{marker_path.name}"
                        )
                if (checkpoint.get('training_lineage') is not None
                        and payload.get('training_lineage') != checkpoint['training_lineage']):
                    raise ValueError(f"checkpoint lineage does not match {marker_path.name}")
                checkpoint_record = payload.get("checkpoint") or {}
                if (
                    checkpoint_record.get("status") != "complete"
                    or checkpoint_record.get("file") != name
                ):
                    raise ValueError(
                        f"checkpoint artifact record is invalid: {marker_path.name}"
                    )
            records.append(
                {
                    "format": "pytorch_checkpoint",
                    "model_id": checkpoint["model_id"],
                    "model_prefix": checkpoint["model_prefix"],
                    "iteration": checkpoint["iteration"],
                    "training_stage": checkpoint.get('training_stage', 'ppo'),
                    "training_lineage": checkpoint.get('training_lineage'),
                    "model_protocol_version": checkpoint[
                        "model_protocol_version"
                    ],
                    "protocol_schema_revision": checkpoint[
                        "protocol_schema_revision"
                    ],
                    "protocol_schema_hash": checkpoint["protocol_schema_hash"],
                    "card_vocab_hash": checkpoint["card_vocab_hash"],
                    "card_vocab_size": checkpoint["card_vocab_size"],
                    "card_vocab_card_count": checkpoint[
                        "card_vocab_card_count"
                    ],
                    "semantic_lookup_format_version": checkpoint[
                        "semantic_lookup_format_version"
                    ],
                    "semantic_lookup_hash": checkpoint[
                        "semantic_lookup_hash"
                    ],
                    "semantic_lookup_card_count": checkpoint[
                        "semantic_lookup_card_count"
                    ],
                    "primary": name,
                    "files": [name],
                    "status": "complete",
                }
            )
            compatibility = checkpoint.get('semantic_compatibility') or checkpoint['net_config'].get('semantic_compatibility')
            if compatibility is not None:
                records[-1]['semantic_compatibility'] = compatibility
    return records


def get_model_iteration_mismatch(records):
    """比较同一选择中的 PTH 与 ONNX 轮次集合，并返回可展示的提示"""
    checkpoint_iterations = {
        (record.get('training_lineage') or {}).get('ppo_iteration', record["iteration"])
        for record in records
        if record.get("format") == "pytorch_checkpoint"
    }
    onnx_iterations = {
        record["iteration"]
        for record in records
        if record.get("format") == "onnx"
    }
    if checkpoint_iterations and onnx_iterations and checkpoint_iterations != onnx_iterations:
        return (
            "PTH 与 ONNX 的内置轮次不一致: "
            f"PTH={sorted(checkpoint_iterations)}, ONNX={sorted(onnx_iterations)}"
        )
    return None


def validate_package_model_records(
    records,
    *,
    expected_model_id=None,
    card_vocabulary=None,
    semantic_root=".",
):
    """校验部署包只能包含同一 UUID 的模型，且双格式轮次必须成对一致"""
    if not isinstance(records, list):
        raise ValueError("model artifact records must be a list")
    if not records:
        return records
    model_ids = {record.get("model_id") for record in records}
    prefixes = {record.get("model_prefix") for record in records}
    model_protocol_versions = {
        record.get("model_protocol_version") for record in records
    }
    if None in model_ids or None in prefixes or None in model_protocol_versions:
        raise ValueError("model artifact identity is incomplete")
    for model_id in model_ids:
        validate_model_id(model_id)
    for prefix in prefixes:
        validate_model_prefix(prefix)
    if len(model_ids) > 1:
        raise ValueError("one deployment package cannot mix different model_id pools")
    if len(prefixes) > 1:
        raise ValueError("one model_id pool cannot contain different model prefixes")
    if model_protocol_versions != {MODEL_PROTOCOL_VERSION}:
        raise ValueError("deployment package model protocol version is incompatible")
    for record in records:
        validate_protocol_metadata(
            record,
            label="deployment model record",
            card_vocabulary=card_vocabulary,
            semantic_root=semantic_root,
            allow_semantic_updates=record.get('format') == 'pytorch_checkpoint',
            allow_observation_updates=record.get('format') == 'pytorch_checkpoint',
        )
    if expected_model_id is not None and model_ids and model_ids != {expected_model_id}:
        raise PermissionError("selected models do not belong to the requested model_id pool")
    artifact_keys = [
        (record.get("format"), (record.get('training_lineage') or {}).get('ppo_iteration', record.get("iteration"))) for record in records
    ]
    if len(artifact_keys) != len(set(artifact_keys)):
        raise ValueError("one deployment package cannot contain duplicate format/iteration artifacts")
    mismatch = get_model_iteration_mismatch(records)
    if mismatch:
        raise ValueError(mismatch)
    return records


def discover_model_repository(model_dir):
    """按模型 UUID 和内置轮次整理模型仓库，并列出无法归档的产物"""
    model_root = Path(model_dir).resolve()
    result = {"pools": {}, "invalid": []}
    if not model_root.is_dir():
        return result

    recognized_files = sorted(
        path
        for path in model_root.iterdir()
        if path.is_file()
        and any(path.name.casefold().endswith(suffix) for suffix in MODEL_ARTIFACT_SUFFIXES)
    )
    referenced_files = set()
    primary_paths = [path for path in recognized_files if is_primary_model_filename(path.name)]
    for primary_path in primary_paths:
        try:
            if primary_path.suffix.casefold() == ".pth":
                checkpoint = validate_training_checkpoint_file(
                    primary_path,
                    map_location="cpu",
                )
                marker_path = checkpoint_artifact_manifest_path(primary_path)
                if marker_path.is_file():
                    payload = _read_artifact_manifest(marker_path)
                    for key in (
                        "model_id",
                        "model_prefix",
                        "iteration",
                        "model_protocol_version",
                        "protocol_schema_revision",
                        "protocol_schema_hash",
                        "card_vocab_hash",
                        "card_vocab_size",
                        "card_vocab_card_count",
                        "semantic_lookup_format_version",
                        "semantic_lookup_hash",
                        "semantic_lookup_card_count",
                    ):
                        if payload[key] != checkpoint[key]:
                            raise ValueError(
                                f"checkpoint internal {key} does not match "
                                f"{marker_path.name}"
                            )
                    checkpoint_record = payload.get("checkpoint") or {}
                    if (
                        checkpoint_record.get("status") != "complete"
                        or checkpoint_record.get("file") != primary_path.name
                    ):
                        raise ValueError("checkpoint manifest does not name this PTH")
                    record = {
                        "format": "pytorch_checkpoint",
                        "model_id": checkpoint["model_id"],
                        "model_prefix": checkpoint["model_prefix"],
                        "iteration": checkpoint["iteration"],
                        "model_protocol_version": checkpoint[
                            "model_protocol_version"
                        ],
                        "protocol_schema_revision": checkpoint[
                            "protocol_schema_revision"
                        ],
                        "protocol_schema_hash": checkpoint[
                            "protocol_schema_hash"
                        ],
                        "card_vocab_hash": checkpoint["card_vocab_hash"],
                        "card_vocab_size": checkpoint["card_vocab_size"],
                        "card_vocab_card_count": checkpoint[
                            "card_vocab_card_count"
                        ],
                        "semantic_lookup_format_version": checkpoint[
                            "semantic_lookup_format_version"
                        ],
                        "semantic_lookup_hash": checkpoint[
                            "semantic_lookup_hash"
                        ],
                        "semantic_lookup_card_count": checkpoint[
                            "semantic_lookup_card_count"
                        ],
                        "primary": primary_path.name,
                        "files": [primary_path.name, marker_path.name],
                        "identity_source": "checkpoint",
                        "status": "complete",
                    }
                else:
                    record = {
                        "format": "pytorch_checkpoint",
                        "model_id": checkpoint["model_id"],
                        "model_prefix": checkpoint["model_prefix"],
                        "iteration": checkpoint["iteration"],
                        "model_protocol_version": checkpoint[
                            "model_protocol_version"
                        ],
                        "protocol_schema_revision": checkpoint[
                            "protocol_schema_revision"
                        ],
                        "protocol_schema_hash": checkpoint[
                            "protocol_schema_hash"
                        ],
                        "card_vocab_hash": checkpoint["card_vocab_hash"],
                        "card_vocab_size": checkpoint["card_vocab_size"],
                        "card_vocab_card_count": checkpoint[
                            "card_vocab_card_count"
                        ],
                        "semantic_lookup_format_version": checkpoint[
                            "semantic_lookup_format_version"
                        ],
                        "semantic_lookup_hash": checkpoint[
                            "semantic_lookup_hash"
                        ],
                        "semantic_lookup_card_count": checkpoint[
                            "semantic_lookup_card_count"
                        ],
                        "primary": primary_path.name,
                        "files": [primary_path.name],
                        "identity_source": "checkpoint",
                        "status": "complete",
                    }
            else:
                record = describe_onnx_artifact(
                    primary_path,
                    require_complete=True,
                    validate_manifest=checkpoint_artifact_manifest_path(primary_path).is_file(),
                )
                if record["model_id"] is None:
                    raise ValueError("ONNX embedded model identity is missing")
                record["identity_source"] = "onnx_metadata"
                marker_path = checkpoint_artifact_manifest_path(primary_path)
                if marker_path.is_file():
                    record["files"] = [*record["files"], marker_path.name]

            if primary_path.suffix.casefold() == '.pth':
                from checkpoint_utils import checkpoint_stem, get_training_lineage
                record.update(training_stage=checkpoint.get('training_stage', 'ppo'),
                    training_lineage=get_training_lineage(checkpoint), display_name=checkpoint_stem(checkpoint))
                if (checkpoint.get('training_lineage') is not None and marker_path.is_file()
                        and payload.get('training_lineage') != checkpoint['training_lineage']):
                    raise ValueError('checkpoint lineage differs from manifest')
            else:
                record['display_name'] = primary_path.stem
                if marker_path.is_file():
                    source_manifest = _read_artifact_manifest(marker_path)
                    record['training_stage'] = source_manifest.get('training_stage', 'ppo')
                    record['source_iteration'] = source_manifest['iteration']
            model_id = record["model_id"]
            iteration = record.get('source_iteration', record["iteration"])
            pool = result["pools"].setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "prefixes": set(),
                    "iterations": {},
                    "artifacts": [],
                },
            )
            pool["prefixes"].add(record["model_prefix"])
            pool["iterations"].setdefault(iteration, []).append(record)
            pool["artifacts"].append(record)
            referenced_files.update(record["files"])
        except Exception as error:
            result["invalid"].append(
                {"file": primary_path.name, "error": str(error)}
            )

    invalid_names = {item["file"] for item in result["invalid"]}
    for path in recognized_files:
        if path.name not in referenced_files and path.name not in invalid_names:
            result["invalid"].append(
                {"file": path.name, "error": "orphan or unverified model artifact"}
            )
    for pool in result["pools"].values():
        pool["prefixes"] = sorted(pool["prefixes"], key=str.casefold)
        pool["artifacts"].sort(key=lambda item: (item["iteration"], item["format"], item["primary"]))
    result["invalid"].sort(key=lambda item: item["file"].casefold())
    return result


def install_model_artifact_bundle(
    source_dir,
    target_dir,
    selected_models=None,
    *,
    expected_model_id=None,
    require_all_artifacts=False,
    card_vocabulary=None,
    semantic_root='.',
):
    """验证暂存模型制品的身份与依赖后，以临时文件原子安装到模型仓库"""
    source_root = Path(source_dir).resolve()
    target_root = Path(target_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"model artifact source does not exist: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    if selected_models is None:
        selected_models = sorted(
            path.name for path in source_root.iterdir() if is_primary_model_filename(path.name)
        )
    selected_models = list(dict.fromkeys(selected_models))
    records = build_package_model_records(source_root, selected_models,
        card_vocabulary=card_vocabulary, semantic_root=semantic_root)
    if expected_model_id is not None:
        validate_package_model_records(records, expected_model_id=expected_model_id,
            card_vocabulary=card_vocabulary, semantic_root=semantic_root)
    prefixes_by_model_id = {}
    for record in records:
        validate_package_model_records([record], card_vocabulary=card_vocabulary, semantic_root=semantic_root)
        prefixes_by_model_id.setdefault(record["model_id"], set()).add(
            record["model_prefix"]
        )
    if any(len(prefixes) != 1 for prefixes in prefixes_by_model_id.values()):
        raise ValueError("one model_id cannot be imported with different model prefixes")

    files = collect_model_artifact_files(source_root, selected_models,
        card_vocabulary=card_vocabulary, semantic_root=semantic_root)
    validate_model_artifact_file_set(source_root, files)
    for record in records:
        assert_model_artifact_target_identity(target_root, record,
            card_vocabulary=card_vocabulary, semantic_root=semantic_root)

    if require_all_artifacts:
        staged_artifacts = {
            path.name
            for path in source_root.iterdir()
            if path.is_file()
            and any(
                path.name.casefold().endswith(suffix)
                for suffix in MODEL_ARTIFACT_SUFFIXES
            )
        }
        unclaimed = staged_artifacts.difference(files)
        if unclaimed:
            raise ValueError(
                "unreferenced model artifacts are not allowed: "
                f"{sorted(unclaimed)}"
            )

    installed = []
    for filename in files:
        safe_name = validate_model_artifact_filename(filename)
        source_path = source_root / safe_name
        validate_model_artifact_file(source_path)
        target_path = target_root / safe_name
        temporary_path = None
        try:
            with open(source_path, "rb") as source_stream, tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target_path.name}.",
                suffix=".import.tmp",
                dir=target_root,
                delete=False,
            ) as temporary_stream:
                temporary_path = Path(temporary_stream.name)
                shutil.copyfileobj(source_stream, temporary_stream, length=1024 * 1024)
            os.replace(temporary_path, target_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        installed.append(safe_name)
    return {"records": records, "files": installed}


def write_checkpoint_artifact_manifest(
    checkpoint_path,
    iteration,
    *,
    model_id,
    model_prefix,
    checkpoint_format_version=CHECKPOINT_FORMAT_VERSION,
    model_protocol_version=MODEL_PROTOCOL_VERSION,
    onnx_record=None,
    onnx_error=None,
    training_stage=None,
    training_lineage=None,
):
    """在保存检查点时写入同轮次产物清单，并标记 ONNX 是否完整"""
    protocol_metadata = get_current_protocol_metadata()
    validate_model_id(model_id)
    validate_model_prefix(model_prefix)
    if checkpoint_format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("artifact checkpoint format version does not match the current format")
    if model_protocol_version != MODEL_PROTOCOL_VERSION:
        raise ValueError("artifact model protocol version does not match the current protocol")
    checkpoint = Path(checkpoint_path).resolve()
    manifest_path = checkpoint_artifact_manifest_path(checkpoint)
    if onnx_error == "export_in_progress":
        onnx_status = "in_progress"
    elif onnx_error:
        onnx_status = "failed"
    else:
        onnx_status = "disabled"
    if onnx_record is not None:
        onnx_record = dict(onnx_record)
        onnx_record.update(
            {
                "model_id": model_id,
                "model_prefix": model_prefix,
                "iteration": int(iteration),
                **protocol_metadata,
            }
        )
    payload = {
        "artifact_manifest_version": ARTIFACT_MANIFEST_FORMAT_VERSION,
        "checkpoint_format_version": int(checkpoint_format_version),
        **protocol_metadata,
        "model_id": model_id,
        "model_prefix": model_prefix,
        "iteration": int(iteration),
        "checkpoint": {
            "format": "pytorch_checkpoint",
            "file": checkpoint.name,
            "status": "complete" if checkpoint.is_file() else "missing",
        },
        "onnx": onnx_record
        if onnx_record is not None
        else {
            "status": onnx_status,
            "error": str(onnx_error) if onnx_error else None,
        },
    }
    compatibility = semantic_compatibility_descriptor(get_static_semantic_lookup())
    if compatibility is not None:
        payload['semantic_compatibility'] = compatibility

    if training_stage is not None:
        payload['training_stage'] = training_stage
    if training_lineage is not None:
        from checkpoint_utils import get_training_lineage
        payload['training_lineage'] = get_training_lineage({**payload, 'training_lineage': training_lineage})
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary_path, manifest_path)
    return manifest_path


def create_deployment_package(
    target_path,
    model_dir,
    selected_models,
    *,
    package_name,
    extra_files=None,
    optional_files=None,
):
    """校验模型池和附加文件后，原子生成带强制清单的部署包"""
    validate_package_name(package_name)
    optional_records = _prepare_deployment_content(optional_files)
    package_format = (DEPLOY_PACKAGE_FORMAT_VERSION if optional_records else LEGACY_DEPLOY_PACKAGE_FORMAT_VERSION)
    target = Path(target_path)
    validate_deployment_package_filename(target.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink():
            raise ValueError("deployment package target must not be a symbolic link")
        raise FileExistsError(f"deployment package already exists: {target.name}")

    selected_models = list(dict.fromkeys(selected_models))
    requested_extras = dict(extra_files or {})
    requested_extras.setdefault(
        CARD_VOCAB_FILENAME,
        str(Path(CARD_VOCAB_FILENAME).resolve()),
    )
    raw_vocabulary_source = Path(requested_extras[CARD_VOCAB_FILENAME])
    if raw_vocabulary_source.is_symlink():
        raise ValueError("deployment card vocabulary must not be a symlink")
    packaged_vocabulary = load_card_vocabulary(raw_vocabulary_source.resolve())
    compiled_temp_directory = None
    semantic_root = Path(".").resolve()
    if all(
        filename in requested_extras
        for filename in (
            KNOWLEDGE_BASE_FILENAME,
            CODE_EMBEDDINGS_FILENAME,
            CODE_EMBEDDINGS_INDEX_FILENAME,
        )
    ):
        semantic_root = Path(
            requested_extras[KNOWLEDGE_BASE_FILENAME]
        ).resolve().parent
    source_semantic_root = semantic_root
    # 新语义源自动携带生成清单，避免用户选择向量后漏掉内容身份
    provenance_source = semantic_root / CODE_EMBEDDINGS_META_FILENAME
    if KNOWLEDGE_BASE_FILENAME in requested_extras and provenance_source.is_file():
        requested_extras.setdefault(CODE_EMBEDDINGS_META_FILENAME, str(provenance_source))
    # 部署包始终携带可直接运行的编译语义资产；源知识库仍是可选的维护组件
    if raw_vocabulary_source.resolve() == Path(CARD_VOCAB_FILENAME).resolve():
        ensure_static_semantic_assets(
            semantic_root,
            card_vocabulary=packaged_vocabulary,
        )
        compiled_semantic_paths = {
            filename: semantic_root / filename
            for filename in STATIC_SEMANTIC_ASSET_FILENAMES
        }
    else:
        compiled_temp_directory = tempfile.TemporaryDirectory(
            prefix="galatea_semantic_package_",
            dir=target.parent,
        )
        _, compiled_semantic_paths = write_static_semantic_assets(
            semantic_root,
            output_directory=compiled_temp_directory.name,
            card_vocabulary=packaged_vocabulary,
        )
    compiled_root = Path(
        compiled_semantic_paths[STATIC_SEMANTIC_ASSET_FILENAMES[0]]
    ).resolve().parent
    requested_extras.update(
        {
            filename: str(path)
            for filename, path in compiled_semantic_paths.items()
        }
    )
    compatibility_path = compiled_root / SEMANTIC_COMPATIBILITY_FILENAME
    if compatibility_path.is_file():
        requested_extras[SEMANTIC_COMPATIBILITY_FILENAME] = str(compatibility_path)
    records = build_package_model_records(
        model_dir,
        selected_models,
        card_vocabulary=packaged_vocabulary,
        semantic_root=compiled_root,
    )
    validate_package_model_records(
        records,
        card_vocabulary=packaged_vocabulary,
        semantic_root=compiled_root,
    )
    model_files = collect_model_artifact_files(
        model_dir,
        selected_models,
        card_vocabulary=packaged_vocabulary,
        semantic_root=compiled_root,
    )
    model_total_size = validate_model_artifact_file_set(model_dir, model_files)
    model_total_size += sum(row['size'] for row in optional_records)
    if model_total_size > MAX_DEPLOY_ARCHIVE_TOTAL_BYTES:
        raise ValueError('deployment package inputs exceed the 64 GiB safety limit')
    extras = {}
    for archive_name, source_path in requested_extras.items():
        if archive_name not in DEPLOY_ROOT_FILES:
            raise ValueError(f"unsupported deployment root file: {archive_name!r}")
        raw_source = Path(source_path)
        if raw_source.is_symlink():
            raise ValueError(f"deployment root file must not be a symlink: {archive_name}")
        source = raw_source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"deployment root file does not exist: {source}")
        extra_size = source.stat().st_size
        if extra_size > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
            raise ValueError(f"deployment root file exceeds 32 GiB: {archive_name}")
        model_total_size += extra_size
        if model_total_size > MAX_MODEL_ARTIFACT_TOTAL_BYTES:
            raise ValueError("deployment package inputs exceed the 64 GiB safety limit")
        extras[archive_name] = source

    included_code_semantics = CODE_SEMANTIC_FILE_SET.intersection(extras)
    included_static_semantics = STATIC_SEMANTIC_FILE_SET.intersection(extras)
    if included_static_semantics != STATIC_SEMANTIC_FILE_SET:
        raise ValueError("compiled static semantic table and catalog are both required")
    if included_code_semantics and included_code_semantics != CODE_SEMANTIC_FILE_SET:
        raise ValueError("code semantic vectors and their index must be packaged together")
    includes_knowledge_base = KNOWLEDGE_BASE_FILENAME in extras
    includes_complete_code_semantics = included_code_semantics == CODE_SEMANTIC_FILE_SET
    if CODE_EMBEDDINGS_META_FILENAME in extras and not includes_complete_code_semantics:
        raise ValueError('code semantic provenance requires the complete semantic source bundle')
    if includes_knowledge_base is not includes_complete_code_semantics:
        raise ValueError(
            "knowledge_base.json, code_embeddings.npy and "
            "code_embeddings_idx.json must be packaged as one complete semantic source bundle"
        )
    if included_code_semantics:
        source_semantic_root = extras[CODE_EMBEDDINGS_FILENAME].parent
        validated_semantics = validate_semantic_bundle(source_semantic_root)
        if CODE_EMBEDDINGS_META_FILENAME in extras and extras[CODE_EMBEDDINGS_META_FILENAME].parent != source_semantic_root:
            raise ValueError('code semantic provenance must come from the same directory')
        if (
            validated_semantics["embedding_path"]
            != extras[CODE_EMBEDDINGS_FILENAME]
            or validated_semantics["index_path"]
            != extras[CODE_EMBEDDINGS_INDEX_FILENAME]
            or validated_semantics["knowledge_base_path"]
            != extras[KNOWLEDGE_BASE_FILENAME]
        ):
            raise ValueError("semantic assets must come from one coherent directory")
    compiled_root = extras[STATIC_SEMANTIC_ASSET_FILENAMES[0]].parent
    if any(
        extras[filename].parent != compiled_root
        for filename in STATIC_SEMANTIC_ASSET_FILENAMES
    ):
        raise ValueError("compiled static semantic assets must come from one directory")
    load_static_semantic_assets(
        compiled_root,
        card_vocabulary=packaged_vocabulary,
        verify_source_files=(
            includes_complete_code_semantics
            and compiled_root == source_semantic_root
        ),
        verify_logical_hash=True,
    )
    semantic_root = compiled_root
    if HASH_MAPPING_FILENAME in extras and KNOWLEDGE_BASE_FILENAME not in extras:
        raise ValueError("hash mapping requires knowledge_base.json")

    protocol_metadata = get_current_protocol_metadata(
        packaged_vocabulary,
        semantic_root=semantic_root,
    )
    if records:
        for record in records:
            validate_protocol_metadata(
                record,
                label="deployment model record",
                card_vocabulary=packaged_vocabulary,
                semantic_root=semantic_root,
                allow_semantic_updates=record.get('format') == 'pytorch_checkpoint',
                allow_observation_updates=record.get('format') == 'pytorch_checkpoint',
            )

    manifest = {
        "package_format_version": package_format,
        **protocol_metadata,
        "package_name": package_name,
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_included": selected_models,
        "model_artifacts": records,
        "model_files_included": model_files,
        "includes_kb": includes_knowledge_base,
        "includes_staples": "meta_staples.json" in extras,
        "includes_hash_mapping": HASH_MAPPING_FILENAME in extras,
        "includes_code_semantics": includes_complete_code_semantics,
        "includes_static_semantics": True,
        "includes_card_vocab": True,
    }
    member_limit = MAX_DEPLOY_ARCHIVE_MEMBERS + (MAX_DEPLOY_CONTENT_FILES if optional_records else 0)
    if len(model_files) + len(extras) + len(optional_records) + 1 > member_limit:
        raise ValueError('deployment archive member count is outside the safety limit')

    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{target.stem}.",
        suffix=".gkg.tmp",
        dir=target.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for filename in model_files:
                safe_name = validate_model_artifact_filename(filename)
                archive.write(
                    Path(model_dir) / safe_name,
                    arcname=safe_name,
                    compress_type=zipfile.ZIP_STORED,
                )
            for archive_name, source in extras.items():
                archive.write(source, arcname=archive_name)
            if optional_records:
                declared_content = []
                for row in optional_records:
                    digest, written = hashlib.sha256(), 0
                    # 按实际入包字节计算摘要，压缩包和原始录像沿用存储模式
                    info = zipfile.ZipInfo(row['path'])
                    info.compress_type = (zipfile.ZIP_STORED if row['path'].lower().endswith(
                        ('.gz', '.yrp', '.yrp2', '.yrp3', '.yrp3d', '.yrpx', '.pdf', '.png', '.jpg', '.jpeg', '.webp'))
                        else zipfile.ZIP_DEFLATED)
                    info._compresslevel = 1
                    with open(row['source'], 'rb') as source, archive.open(info, 'w', force_zip64=True) as output:
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            written += len(chunk)
                            if written > row['size']:
                                raise ValueError(f'content changed while packaging: {row["path"]}')
                            digest.update(chunk)
                            output.write(chunk)
                    if written != row['size']:
                        raise ValueError(f'content changed while packaging: {row["path"]}')
                    declared_content.append({key: row[key] for key in ('category', 'path', 'relative_path', 'size')})
                    declared_content[-1]['sha256'] = digest.hexdigest()
                manifest['optional_content'] = {'format_version': DEPLOY_CONTENT_FORMAT_VERSION, 'files': declared_content}
            manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
            manifest_limit = MAX_DEPLOY_CONTENT_MANIFEST_BYTES if optional_records else MAX_MANIFEST_FILE_BYTES
            if len(manifest_bytes) > manifest_limit:
                raise ValueError('deployment package manifest is too large')
            archive.writestr(
                "manifest.json",
                manifest_bytes,
            )
        if temporary_path.stat().st_size > MAX_DEPLOY_PACKAGE_FILE_BYTES:
            raise ValueError("deployment package exceeds the 64 GiB safety limit")
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        if compiled_temp_directory is not None:
            compiled_temp_directory.cleanup()
    return manifest


def validate_deployment_package(stage_dir):
    """核验解包目录、部署协议清单、模型身份与实际文件集合完全一致"""
    raw_stage_root = _reject_deployment_links(stage_dir)
    stage_root = raw_stage_root.resolve()
    if not stage_root.is_dir():
        raise FileNotFoundError(f"deployment stage does not exist: {stage_root}")
    manifest_path = stage_root / "manifest.json"
    _reject_deployment_links(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError("deployment package manifest.json is required")
    if manifest_path.stat().st_size > MAX_DEPLOY_CONTENT_MANIFEST_BYTES:
        raise ValueError("deployment package manifest is too large")
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("deployment package manifest must be an object")
    if manifest.get("package_format_version") not in (LEGACY_DEPLOY_PACKAGE_FORMAT_VERSION, DEPLOY_PACKAGE_FORMAT_VERSION):
        raise ValueError(
            "deployment package format version mismatch: "
            f"file={manifest.get('package_format_version')!r}, "
            f"current={DEPLOY_PACKAGE_FORMAT_VERSION}"
        )
    validate_package_name(manifest.get("package_name"))
    content_records = _deployment_content_records(manifest)
    if not content_records and manifest_path.stat().st_size > MAX_MANIFEST_FILE_BYTES:
        raise ValueError('deployment package manifest is too large')
    actual_paths, actual_directories = _scan_deployment_stage(stage_root, has_content=bool(content_records))

    actual_names = {name for name in actual_paths if '/' not in name}
    allowed_names = {"manifest.json", *DEPLOY_ROOT_FILES}
    for name in actual_names.difference(allowed_names):
        validate_model_artifact_filename(name)
    primary_models = sorted(name for name in actual_names if is_primary_model_filename(name))
    declared_models = manifest.get("models_included")
    if not isinstance(declared_models, list) or len(declared_models) != len(set(declared_models)):
        raise ValueError("manifest models_included must be a unique list")
    for name in declared_models:
        if not is_primary_model_filename(name):
            raise ValueError(f"manifest contains an invalid primary model: {name!r}")
    if set(declared_models) != set(primary_models):
        raise ValueError("manifest primary model list does not match package contents")

    packaged_vocabulary = load_card_vocabulary(stage_root / CARD_VOCAB_FILENAME)
    actual_static_semantics = STATIC_SEMANTIC_FILE_SET.intersection(actual_names)
    if actual_static_semantics != STATIC_SEMANTIC_FILE_SET:
        raise ValueError(
            "deployment package requires the compiled static semantic table and catalog"
        )
    if manifest.get("includes_static_semantics") is not True:
        raise ValueError(
            "manifest includes_static_semantics does not match package contents"
        )
    staged_semantic_root = stage_root
    actual_records = build_package_model_records(
        stage_root,
        declared_models,
        card_vocabulary=packaged_vocabulary,
        semantic_root=staged_semantic_root,
    )
    validate_package_model_records(
        actual_records,
        card_vocabulary=packaged_vocabulary,
        semantic_root=staged_semantic_root,
    )
    declared_records = manifest.get("model_artifacts")
    if not isinstance(declared_records, list) or declared_records != actual_records:
        raise ValueError("manifest model identity records do not match embedded metadata")
    actual_model_files = collect_model_artifact_files(
        stage_root,
        declared_models,
        card_vocabulary=packaged_vocabulary,
        semantic_root=staged_semantic_root,
    )
    declared_model_files = manifest.get("model_files_included")
    if (
        not isinstance(declared_model_files, list)
        or len(declared_model_files) != len(set(declared_model_files))
        or set(declared_model_files) != set(actual_model_files)
    ):
        raise ValueError("manifest model file list does not match package contents")

    expected_names = {"manifest.json", *actual_model_files}
    for filename, flag in (
        (KNOWLEDGE_BASE_FILENAME, "includes_kb"),
        ("meta_staples.json", "includes_staples"),
        (HASH_MAPPING_FILENAME, "includes_hash_mapping"),
        (CARD_VOCAB_FILENAME, "includes_card_vocab"),
    ):
        included = filename in actual_names
        if manifest.get(flag) is not included:
            raise ValueError(f"manifest {flag} does not match package contents")
        if included:
            expected_names.add(filename)
    actual_code_semantics = CODE_SEMANTIC_FILE_SET.intersection(actual_names)
    includes_code_semantics = manifest.get("includes_code_semantics")
    if actual_code_semantics and actual_code_semantics != CODE_SEMANTIC_FILE_SET:
        raise ValueError("deployment package contains an incomplete code semantic pair")
    if includes_code_semantics is not (actual_code_semantics == CODE_SEMANTIC_FILE_SET):
        raise ValueError("manifest includes_code_semantics does not match package contents")
    has_knowledge_base = KNOWLEDGE_BASE_FILENAME in actual_names
    has_complete_code_semantics = actual_code_semantics == CODE_SEMANTIC_FILE_SET
    if has_knowledge_base is not has_complete_code_semantics:
        raise ValueError(
            "deployment package must contain knowledge_base.json, code_embeddings.npy and "
            "code_embeddings_idx.json as one complete semantic source bundle"
        )
    if actual_code_semantics:
        validate_semantic_bundle(stage_root)
        expected_names.update(CODE_SEMANTIC_FILE_SET)
    if CODE_EMBEDDINGS_META_FILENAME in actual_names:
        if not has_complete_code_semantics:
            raise ValueError('code semantic provenance requires the complete semantic source bundle')
        expected_names.add(CODE_EMBEDDINGS_META_FILENAME)
    load_static_semantic_assets(
        stage_root,
        card_vocabulary=packaged_vocabulary,
        verify_source_files=has_complete_code_semantics,
        verify_logical_hash=True,
    )
    expected_names.update(STATIC_SEMANTIC_FILE_SET)
    if HASH_MAPPING_FILENAME in actual_names and KNOWLEDGE_BASE_FILENAME not in actual_names:
        raise ValueError("hash mapping requires knowledge_base.json")
    validate_protocol_metadata(
        manifest,
        label="deployment package manifest",
        card_vocabulary=packaged_vocabulary,
        semantic_root=staged_semantic_root,
        allow_observation_updates=all(record.get('format') == 'pytorch_checkpoint' for record in actual_records),
    )
    for record in actual_records:
        validate_protocol_metadata(
            record,
            label="deployment model record",
            card_vocabulary=packaged_vocabulary,
            semantic_root=staged_semantic_root,
            allow_semantic_updates=record.get('format') == 'pytorch_checkpoint',
            allow_observation_updates=record.get('format') == 'pytorch_checkpoint',
        )
    if SEMANTIC_COMPATIBILITY_FILENAME in actual_names:
        from semantic_lookup import read_semantic_compatibility_registry
        read_semantic_compatibility_registry(stage_root)
        expected_names.add(SEMANTIC_COMPATIBILITY_FILENAME)
    for row in content_records:
        path = stage_root / row['path']
        if not path.is_file() or path.stat().st_size != row['size'] or _deployment_file_sha256(path) != row['sha256']:
            raise ValueError(f'optional content checksum or size mismatch: {row["path"]}')
        expected_names.add(row['path'])
    expected_directories = {'/'.join(row['path'].split('/')[:depth]) for row in content_records
                            for depth in range(1, len(row['path'].split('/')))}
    if actual_directories != expected_directories:
        raise ValueError('deployment package contains undeclared directories')
    if actual_paths != expected_names:
        raise ValueError(
            f"deployment package contains undeclared files: {sorted(actual_paths - expected_names)}"
        )
    return {"manifest": manifest, "records": actual_records, "files": actual_model_files,
            "optional_files": content_records}


def safe_extract_zip(archive, target_dir):
    """兼容单层旧包与附件新包，预检全部成员后有界解压并回收失败结果"""
    raw_target = _reject_deployment_links(target_dir)
    target_root = raw_target.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    archive_filename = getattr(archive, "filename", None)
    if isinstance(archive_filename, (str, os.PathLike)):
        archive_path = Path(archive_filename)
        if archive_path.is_symlink() or not archive_path.is_file():
            raise ValueError("deployment package must be a regular file")
        if archive_path.stat().st_size > MAX_DEPLOY_PACKAGE_FILE_BYTES:
            raise ValueError("deployment package exceeds the 64 GiB safety limit")
    members = archive.infolist()
    if not members or len(members) > MAX_DEPLOY_ARCHIVE_MEMBERS + MAX_DEPLOY_CONTENT_FILES:
        raise ValueError("deployment archive member count is outside the safety limit")

    total_size = 0
    seen_names = set()
    for info in members:
        if info.is_dir() or "\\" in info.filename:
            raise ValueError(f"unsafe archive member path: {info.filename!r}")
        if '/' in info.filename:
            _validate_content_archive_path(info.filename)
        else:
            validate_safe_filename(info.filename)
        folded = _content_path_key(info.filename)
        if folded in seen_names:
            raise ValueError(f"duplicate archive member is not allowed: {info.filename!r}")
        seen_names.add(folded)
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted archive member is not allowed: {info.filename!r}")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError(f"unsupported archive compression: {info.filename!r}")
        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if stat.S_ISLNK(unix_mode) or file_type not in {0, stat.S_IFREG}:
            raise ValueError(f"archive special files are not allowed: {info.filename!r}")
        if info.file_size < 0 or info.file_size > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
            raise ValueError(f"archive member is too large: {info.filename!r}")
        total_size += info.file_size
        if total_size > MAX_DEPLOY_ARCHIVE_TOTAL_BYTES:
            raise ValueError("deployment archive exceeds the 64 GiB expanded-size limit")
        if info.file_size > 1024 * 1024:
            if info.compress_size <= 0 or info.file_size / info.compress_size > MAX_DEPLOY_COMPRESSION_RATIO:
                raise ValueError(f"archive compression ratio is unsafe: {info.filename!r}")
        destination = target_root / info.filename
        _reject_deployment_links(destination)
        destination.resolve().relative_to(target_root)
        if destination.exists():
            raise FileExistsError(f"archive target already exists: {info.filename!r}")
    _check_deployment_path_collisions([info.filename for info in members])

    manifest_info = next((info for info in members if info.filename == 'manifest.json'), None)
    content_records = []
    if manifest_info is not None:
        if manifest_info.file_size > MAX_DEPLOY_CONTENT_MANIFEST_BYTES:
            raise ValueError('deployment package manifest is too large')
        manifest = json.loads(archive.read(manifest_info).decode('utf-8'))
        if not isinstance(manifest, dict):
            raise ValueError('deployment package manifest must be an object')
        content_records = _deployment_content_records(manifest)
    if not content_records:
        if len(members) > MAX_DEPLOY_ARCHIVE_MEMBERS:
            raise ValueError('deployment archive member count is outside the safety limit')
        if any('/' in info.filename for info in members):
            raise ValueError('nested content requires a GKG5 manifest')
        if manifest_info is not None and manifest_info.file_size > MAX_MANIFEST_FILE_BYTES:
            raise ValueError('deployment package manifest is too large')
    declared_content = {row['path'] for row in content_records}
    if {info.filename for info in members if '/' in info.filename} != declared_content:
        raise ValueError('archive content files do not match the manifest')

    created_files, created_directories = [], []
    try:
        for info in members:
            destination = target_root / info.filename
            pending = []
            parent = destination.parent
            while parent != target_root and not parent.exists():
                pending.append(parent)
                parent = parent.parent
            for directory in reversed(pending):
                _reject_deployment_links(directory)
                directory.resolve().relative_to(target_root)
                directory.mkdir()
                created_directories.append(directory)
            written = 0
            with archive.open(info, "r") as source, open(destination, "xb") as output:
                created_files.append(destination)
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size or written > MAX_DEPLOY_ARCHIVE_MEMBER_BYTES:
                        raise ValueError(f"archive member expanded beyond its limit: {info.filename!r}")
                    output.write(chunk)
            if written != info.file_size:
                raise ValueError(f"archive member size mismatch: {info.filename!r}")
    except Exception:
        for path in reversed(created_files):
            try:
                path.unlink()
            except OSError:
                pass
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except OSError:
                pass
        raise
    return [info.filename for info in members]
