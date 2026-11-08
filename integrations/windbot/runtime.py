"""Optional IceYGO lifecycle helpers; no dependency probing at import time."""
from contextlib import contextmanager, ExitStack
import ctypes
import ctypes.util
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def executor_catalog():
    """Read the pinned source declarations; never guess executor/deck correspondence."""
    result = {}
    pattern = re.compile(r'\[Deck\("([^"\r\n]+)",\s*"([^"\r\n]+)"(?:,\s*"([^"\r\n]+)")?\)\]')
    for source in sorted((ROOT / 'vendor/Game/AI/Decks').glob('*.cs')):
        for name, deck, level in pattern.findall(source.read_text(encoding='utf-8-sig')):
            path = ROOT / 'vendor/Decks' / (deck + '.ydk')
            if name in result or path.resolve().parent != (ROOT / 'vendor/Decks').resolve():
                raise ValueError(f'duplicate executor or missing/unsafe deck: {name}')
            result[name] = {'executor': name, 'deck_file': str(path), 'level': level or 'Normal',
                            'deck_available': path.is_file()}
    if not result:
        raise FileNotFoundError('WindBot optional source snapshot is absent')
    return result


def load_deck(path):
    """Read YDK without consulting a network source or modifying user files."""
    result = {'main': [], 'extra': [], 'side': []}
    section = None
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line in ('#main', '#extra', '!side'):
            section = {'#main': 'main', '#extra': 'extra', '!side': 'side'}[line]
        elif line and not line.startswith(('#', '!')):
            if section is None or not line.isdecimal() or not 0 < int(line) < 2**32 or len(result[section]) >= 256:
                raise ValueError(f'invalid YDK entry: {line}')
            result[section].append(int(line))
    if not result['main']:
        raise ValueError('empty Main Deck')
    return result


def runtime_directory():
    """仅为已有旧运行包保留平台子目录回退"""
    return ROOT / 'runtime' / ('windows' if sys.platform == 'win32' else 'linux')


@lru_cache(maxsize=1)
def linux_sqlite_library():
    """显式启用时验证系统SQLite，避免将Windows原生库交给Mono"""
    library = ctypes.util.find_library('sqlite3')
    if not library:
        raise FileNotFoundError('Linux缺少SQLite原生库，请安装系统libsqlite3运行库')
    try:
        native = ctypes.CDLL(library)
        native.sqlite3_libversion_number.restype = ctypes.c_int
        if native.sqlite3_libversion_number() < 3007000:
            raise RuntimeError('Linux SQLite版本过旧，需要SQLite 3.7或更新版本')
    except OSError as error:
        raise RuntimeError(f'Linux SQLite原生库无法加载: {library}: {error}') from error
    return library


def resolve_runtime(runtime=None, mono=None):
    common = ROOT / 'runtime/WindBot.exe'
    preferred = runtime_directory() / 'WindBot.exe'
    exe = Path(runtime or (common if common.is_file() else preferred)).resolve()
    if not exe.is_file():
        raise FileNotFoundError('WindBot.exe缺失，请将预编译WindBot运行包解压到项目根目录，或使用--runtime指定现有程序')
    if sys.platform == 'win32':
        return [str(exe)]
    runner = shutil.which(mono or 'mono')
    if runner is None:
        raise FileNotFoundError('WindBot requires Mono on Linux; install it separately or pass --mono')
    if sys.platform.startswith('linux'):
        linux_sqlite_library()
    return [runner, str(exe)]


def inspect_runtime(runtime=None, mono=None):
    provenance = json.loads((ROOT / 'UPSTREAM.json').read_text(encoding='utf-8'))
    report = {'upstream': provenance, 'executors': list(executor_catalog().values()), 'runtime_ready': False,
              'platform': sys.platform, 'platform_selection': 'automatic', 'verification': 'dependency_check_only',
              'backend': 'native_dotnet_framework' if sys.platform == 'win32' else 'mono'}
    if sys.platform.startswith('linux'):
        report['mono'] = shutil.which(mono or 'mono')
        try:
            report['sqlite_library'] = linux_sqlite_library()
        except (OSError, RuntimeError) as error:
            report['sqlite_error'] = str(error)
    try:
        report['command'] = resolve_runtime(runtime, mono)
        report['runtime_ready'] = True
    except (OSError, RuntimeError) as error:
        report['runtime_error'] = str(error)
    return report


def _linux_config(source, destination, library):
    """只为本次Linux进程生成映射，不改写原有Windows配置与上游文件"""
    tree = ET.parse(source) if source.is_file() else ET.ElementTree(ET.Element('configuration'))
    root = tree.getroot()
    if root.tag != 'configuration':
        raise ValueError('WindBot配置根节点必须是configuration')
    for element in list(root.findall('dllmap')):
        if element.get('dll') == 'sqlite3':
            root.remove(element)
    ET.SubElement(root, 'dllmap', {'dll': 'sqlite3', 'target': library, 'os': 'linux'})
    tree.write(destination, encoding='utf-8', xml_declaration=True)


class PreparedBotRuntime:
    """复用一轮内的只读启动资源，各局仍使用独立进程和工作目录"""

    def __init__(self, runtime=None, mono=None, cache_parent=None):
        self.request = (runtime, mono)
        self.executors = frozenset(executor_catalog())
        self.command = resolve_runtime(runtime, mono)
        exe = Path(self.command[-1])
        if cache_parent is not None:
            Path(cache_parent).mkdir(parents=True, exist_ok=True)
        self.storage = tempfile.TemporaryDirectory(prefix='galatea-windbot-', dir=cache_parent)
        self.closed = False
        try:
            self.directory = Path(self.storage.name).resolve()
            if sys.platform.startswith('linux'):
                shadow = self.directory / exe.name
                shutil.copy2(exe, shadow)
                for dependency in exe.parent.glob('*.dll'):
                    shutil.copy2(dependency, self.directory / dependency.name)
                _linux_config(Path(str(exe) + '.config'), Path(str(shadow) + '.config'), linux_sqlite_library())
                self.command[-1] = str(shadow)
            if (exe.parent / 'Dialogs').is_dir():
                shutil.copytree(exe.parent / 'Dialogs', self.directory, dirs_exist_ok=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        """所属进程均退出后释放启动资源"""
        if not self.closed:
            self.storage.cleanup()
            self.closed = True


@contextmanager
def launch_bot(*, runtime, executor, deck_file, cdb_path, port, log_root, mono=None, room_token='', prepared=None):
    """每局拥有独立进程与8 MiB日志，可复用调用方持有的只读启动资源"""
    with ExitStack() as stack:
        if prepared is None:
            prepared = PreparedBotRuntime(runtime, mono)
            stack.callback(prepared.close)
        if prepared.closed or prepared.request != (runtime, mono):
            raise ValueError('WindBot准备资源已关闭或运行文件配置不一致')
        if executor not in prepared.executors:
            raise ValueError(f'unknown WindBot executor: {executor}')
        cdb = Path(cdb_path).resolve()
        if not cdb.is_file():
            raise FileNotFoundError(cdb)
        os.makedirs(log_root, exist_ok=True)
        # 上游ReadFile支持父目录文件回退，单局目录不携带上一局的可写文件
        scratch = stack.enter_context(tempfile.TemporaryDirectory(prefix='duel-', dir=prepared.directory))
        command = list(prepared.command)
        log = tempfile.NamedTemporaryFile(prefix='windbot_', suffix='.log', dir=log_root, delete=False)
        try:
            # Argument lists preserve paths containing spaces, never invoke a shell.
            args = command + [f'Name=Galatea-WindBot', f'Deck={executor}',
                f'DeckFile={Path(deck_file).resolve()}', f'DbPath={cdb}',
                'Host=127.0.0.1', f'Port={port}', f'HostInfo={room_token}',
                'Version=4962', 'Chat=False', 'Debug=False']
            process = subprocess.Popen(args, cwd=scratch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            def drain_output():
                retained = 0
                for block in iter(lambda: process.stdout.read(65536), b''):
                    if retained < 8 * 1024**2:
                        chunk = block[:8 * 1024**2 - retained]
                        log.write(chunk)
                        log.flush()
                        retained += len(chunk)
            reader = threading.Thread(target=drain_output, daemon=True)
            reader.start()
            try:
                yield process, log.name
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                reader.join(timeout=5)
                process.stdout.close()
        finally:
            log.close()


def executable_sha256(runtime=None, mono=None):
    exe = Path(resolve_runtime(runtime, mono)[-1])
    digest = hashlib.sha256()
    with exe.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
