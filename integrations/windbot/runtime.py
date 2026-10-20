"""Optional IceYGO lifecycle helpers; no dependency probing at import time."""
from contextlib import contextmanager
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


def resolve_runtime(runtime=None, mono=None):
    exe = Path(runtime or ROOT / 'runtime/WindBot.exe').resolve()
    if not exe.is_file():
        raise FileNotFoundError('WindBot.exe missing: explicitly build/install the optional module and pass --runtime')
    if sys.platform == 'win32':
        return [str(exe)]
    runner = shutil.which(mono or 'mono')
    if runner is None:
        raise FileNotFoundError('WindBot requires Mono on Linux; install it separately or pass --mono')
    return [runner, str(exe)]


def inspect_runtime(runtime=None, mono=None):
    provenance = json.loads((ROOT / 'UPSTREAM.json').read_text(encoding='utf-8'))
    report = {'upstream': provenance, 'executors': list(executor_catalog().values()), 'runtime_ready': False}
    try:
        report['command'] = resolve_runtime(runtime, mono)
        report['runtime_ready'] = True
    except FileNotFoundError as error:
        report['runtime_error'] = str(error)
    return report


@contextmanager
def launch_bot(*, runtime, executor, deck_file, cdb_path, port, log_root, mono=None, room_token=''):
    """Own exactly one process; isolated cwd and an 8 MiB log cap, no HTTP server.

    Source executors/decisions are unmodified. An explicit runtime may be built by
    the user; the session records its executable hash rather than pretending it
    is authenticated by the pinned source commit.
    """
    if executor not in executor_catalog():
        raise ValueError(f'unknown WindBot executor: {executor}')
    command = resolve_runtime(runtime, mono)
    exe = Path(command[-1])
    cdb = Path(cdb_path).resolve()
    if not cdb.is_file():
        raise FileNotFoundError(cdb)
    os.makedirs(log_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='galatea-windbot-') as scratch:
        for directory in ('Dialogs',):
            if (exe.parent / directory).is_dir():
                shutil.copytree(exe.parent / directory, Path(scratch) / directory)
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
