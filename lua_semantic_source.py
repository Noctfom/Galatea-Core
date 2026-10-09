# 本文件静态提取 Lua 效果注册与回调代码来源，为向量接续提供内容摘要，不执行脚本或改变粗哈希

import hashlib
import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path


LUA_CODE_SOURCE_VERSION = 1
MAX_SOURCE_FUNCTIONS = 128
MAX_EFFECT_SOURCE_BYTES = 128 * 1024
_NATIVE_ROOTS = {'Duel', 'Card', 'Group', 'Effect', 'Debug', 'math', 'string', 'table', 'bit', 'bit32', 'coroutine'}
_LEX_TOKEN = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?)|\.\.\.|==|~=|<=|>=|\.\.|.')


def text_sha256(text):
    """计算原始代码内容身份，不使用粗哈希或效果描述序号"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class LuaToken:
    """保存词法位置，字符串和注释不会冒充函数及效果对象"""
    text: str
    start: int
    end: int
    literal: bool = False


def tokenize_lua(source):
    """跳过注释并完整读取字符串，保留代码片段的原始字符范围"""
    result, index = [], 0
    while index < len(source):
        if source[index].isspace():
            index += 1
            continue
        start = index
        comment = source.startswith('--', index)
        opening_at = index + 2 if comment else index
        opening = re.match(r'\[(=*)\[', source[opening_at:]) if source[opening_at:opening_at + 1] == '[' else None
        if opening:
            close = ']' + opening.group(1) + ']'
            end = source.find(close, opening_at + len(opening.group(0)))
            if end < 0:
                raise ValueError('unterminated Lua long string/comment')
            index = end + len(close)
            if not comment:
                result.append(LuaToken(source[start:index], start, index, True))
        elif comment:
            end = source.find('\n', opening_at)
            index = len(source) if end < 0 else end
        elif source[index] in "\"'":
            quote = source[index]
            index += 1
            while index < len(source):
                char = source[index]
                index += 1
                if char == '\\':
                    index += 1
                elif char == quote:
                    break
            else:
                raise ValueError('unterminated Lua quoted string')
            result.append(LuaToken(source[start:index], start, index, True))
        else:
            match = _LEX_TOKEN.match(source, index)
            index += len(match.group(0))
            result.append(LuaToken(match.group(0), start, index))
    return result


def _qualified_name(tokens, index):
    """读取静态函数/对象名字，不解析动态索引或调用表达式"""
    parts = []
    while index < len(tokens) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', tokens[index].text):
        parts.append(tokens[index].text)
        index += 1
        if index + 1 >= len(tokens) or tokens[index].text not in ('.', ':'):
            break
        parts.append(tokens[index].text)
        index += 1
    return ''.join(parts)


def _function_ranges(tokens):
    """用块栈读取命名和匿名函数，正确处理嵌套if/循环/repeat"""
    stack, functions = [], []
    for index, token in enumerate(tokens):
        if token.literal:
            continue
        value = token.text
        if value in ('function', 'if', 'for', 'while', 'repeat'):
            name = _qualified_name(tokens, index + 1) if value == 'function' else ''
            if value == 'function' and not name and index >= 2 and tokens[index - 1].text == '=':
                first = index - 2
                while first >= 2 and tokens[first - 1].text in ('.', ':'):
                    first -= 2
                name = _qualified_name(tokens, first)
            stack.append([value, index, name, value in ('for', 'while')])
        elif value == 'do':
            if stack and stack[-1][3]:
                stack[-1][3] = False
            else:
                stack.append(['do', index, '', False])
        elif value in ('end', 'until'):
            if not stack:
                raise ValueError('unbalanced Lua block ending')
            block = stack.pop()
            if (value == 'until') != (block[0] == 'repeat'):
                raise ValueError('mismatched Lua repeat/end')
            if block[0] == 'function':
                functions.append((block[2], block[1], index))
    if stack:
        raise ValueError('unterminated Lua function/block')
    return functions


def _closing_parenthesis(tokens, opening):
    """寻找调用的配对括号，匿名函数和嵌套参数不会被提前截断"""
    depth = 0
    for index in range(opening, len(tokens)):
        if tokens[index].literal:
            continue
        if tokens[index].text == '(':
            depth += 1
        elif tokens[index].text == ')':
            depth -= 1
            if depth == 0:
                return index
    raise ValueError('unterminated Lua call')


class LuaCodeSourceIndex:
    """缓存静态代码索引，辅助函数仅在被引用时加入效果来源"""

    def __init__(self, script_dir):
        """登记脚本目录，缓存仅用于一次提取，不参与模型推理"""
        self.root = Path(script_dir).resolve()
        self.units = OrderedDict()
        self.helpers = {}
        for filename in ('utility.lua', 'procedure.lua'):
            path = self.root / filename
            if path.is_file():
                unit = self._load(path)
                for name in unit['functions']:
                    self.helpers.setdefault(name, []).append(unit)

    @staticmethod
    def _canonical(name, card):
        """统一可证明的s/aux别名，不猜动态函数名"""
        if name.startswith('s.') and card is not None:
            return f'c{card}.' + name[2:]
        return 'Auxiliary.' + name[4:] if name.startswith('aux.') else name

    def _load(self, path, source=None):
        """读取并按词法构建函数索引，拒绝目录外的依赖路径"""
        path = Path(path)
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError('Lua dependency must stay inside the script directory')
        path = path.resolve()
        signature = (path.stat().st_mtime_ns, path.stat().st_size)
        if path.name in self.units and self.units[path.name]['signature'] == signature and (source is None or text_sha256(source) == self.units[path.name]['sha256']):
            self.units.move_to_end(path.name)
            return self.units[path.name]
        source = source if source is not None else path.read_text(encoding='utf-8', errors='ignore')
        tokens = tokenize_lua(source)
        ranges = _function_ranges(tokens)
        match = re.fullmatch(r'c(\d+)\.lua', path.name)
        card = int(match.group(1)) if match else None
        functions = {}
        for name, start, end in ranges:
            if name:
                canonical = self._canonical(name, card)
                functions.setdefault(canonical, []).append((start, end))
        top_ranges = sorted((tokens[start].start, tokens[end].end) for _, start, end in ranges)
        context, cursor = [], 0
        for start, end in top_ranges:
            if start >= cursor:
                context.append(source[cursor:start])
            cursor = max(cursor, end)
        context.append(source[cursor:])
        # 移除纯注释，但保留字面量和顶层常量/别名，不引入卡面描述文本
        context_tokens = tokenize_lua('\n'.join(context))
        context_code = ' '.join(token.text for token in context_tokens)
        unit = {'filename': path.name, 'source': source, 'tokens': tokens, 'card': card,
                'functions': functions, 'context': context_code, 'sha256': text_sha256(source), 'signature': signature}
        self.units[path.name] = unit
        while len(self.units) > 256:
            self.units.popitem(last=False)
        return unit

    def is_current(self, card_data, path):
        """同时核对提取版本、主脚本和实际依赖，缺来源的旧卡必须补提取"""
        identity = card_data.get('code_source', {})
        if identity.get('version') != LUA_CODE_SOURCE_VERSION:
            return False
        unit = self._load(path)
        dependencies = identity.get('files', {})
        if dependencies.get(unit['filename']) != unit['sha256']:
            return False
        for filename, digest in dependencies.items():
            target = self.root / filename
            if not target.is_file() or self._load(target)['sha256'] != digest:
                return False
        return all(effect.get('raw_code', '').strip() and effect.get('code_source', {}).get('sha256') == text_sha256(effect['raw_code'])
                   for effect in card_data.get('effects', [])[:8])

    def _resolve(self, name, unit):
        """只查找唯一静态函数，公开Core API不展开为私有实现"""
        name = self._canonical(name, unit['card'])
        if name in unit['functions']:
            return (unit, name) if len(unit['functions'][name]) == 1 else None
        owner = re.match(r'c(\d+)\.', name)
        if owner and (self.root / f'c{owner.group(1)}.lua').is_file():
            other = self._load(self.root / f'c{owner.group(1)}.lua')
            if len(other['functions'].get(name, [])) == 1:
                return other, name
        candidates = self.helpers.get(name, [])
        return (candidates[0], name) if len(candidates) == 1 and len(candidates[0]['functions'][name]) == 1 else None

    def _references(self, code, unit):
        """读取函数引用和回调参数，去掉CreateEffect等原生API"""
        tokens = tokenize_lua(code)
        names = []
        for index, token in enumerate(tokens):
            if token.literal or index and tokens[index - 1].text in ('.', ':'):
                continue
            name = _qualified_name(tokens, index)
            if '.' not in name and name not in unit['functions']:
                continue
            resolved = self._resolve(name, unit)
            if resolved is not None:
                names.append((name, resolved))
            elif name.split('.')[0] not in _NATIVE_ROOTS and re.match(r'(?:s|aux|Auxiliary|c\d+)\.', name):
                # Stringid/GetID和常量字段不是遗漏的效果回调
                following = index + len(re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[.:]', name))
                is_constant = re.search(r'\b' + re.escape(name).replace(r'\.', r'\s*\.\s*') + r'\s*=', unit['context'])
                if following < len(tokens) and tokens[following].text in ('(', ')', ',') and not name.endswith('.Stringid') and not is_constant:
                    names.append((name, None))
        return names

    def augment(self, card_data, path, source=None):
        """给原有效果槽补注册/Clone覆盖/回调闭包，不改变槽位和粗分类"""
        unit = self._load(path, source)
        initial = unit['functions'].get(f"c{card_data['id']}.initial_effect", [])
        if len(initial) != 1:
            raise ValueError(f"无法证明Lua initial_effect来源: {unit['filename']}")
        start, end = initial[0]
        tokens, active, registrations = unit['tokens'][start:end + 1], {}, []
        index = 0
        while index < len(tokens):
            values = [token.text for token in tokens[index:index + 9]]
            if len(values) >= 9 and values[:1] == ['local'] and re.fullmatch(r'e\d*', values[1]) and values[2:8] == ['=', 'Effect', '.', 'CreateEffect', '(', 'c'] and values[8] == ')':
                active[values[1]] = len(registrations)
                registrations.append([unit['source'][tokens[index].start:tokens[index + 8].end]])
                index += 9
                continue
            if len(values) >= 8 and values[0] == 'local' and values[2] == '=' and values[3] in active and values[4:8] == [':', 'Clone', '(', ')']:
                active[values[1]] = active[values[3]]
                registrations[active[values[1]]].append(unit['source'][tokens[index].start:tokens[index + 7].end])
                index += 8
                continue
            if len(values) >= 7 and values[1] == '=' and values[2] in active and values[3:7] == [':', 'Clone', '(', ')']:
                active[values[0]] = active[values[2]]
                registrations[active[values[0]]].append(unit['source'][tokens[index].start:tokens[index + 6].end])
                index += 7
                continue
            if len(values) >= 4 and values[1] == ':' and values[3] == '(':
                closing = _closing_parenthesis(tokens, index + 3)
                owner = active.get(values[0])
                if values[2] == 'RegisterEffect' and index + 4 < closing:
                    owner = active.get(tokens[index + 4].text)
                if owner is not None:
                    registrations[owner].append(unit['source'][tokens[index].start:tokens[closing].end])
                index = closing + 1
                continue
            index += 1
        if len(registrations) != len(card_data['effects']):
            raise ValueError(f"Lua代码来源槽数与结构解析不一致: {unit['filename']} ({len(registrations)}/{len(card_data['effects'])})")
        files = {unit['filename']: unit['sha256']}
        for effect, registration in zip(card_data['effects'], registrations):
            blocks = [('context', unit['filename'], unit['context']), ('registration', f"slot{effect['slot']}", '\n'.join(registration))]
            queue = deque((name, resolved) for name, resolved in self._references(blocks[-1][2], unit))
            seen, unresolved = set(), set()
            while queue:
                name, resolved = queue.popleft()
                if resolved is None:
                    unresolved.add(name)
                    continue
                target, canonical = resolved
                key = target['filename'], canonical
                if key in seen:
                    continue
                seen.add(key)
                if len(seen) > MAX_SOURCE_FUNCTIONS:
                    unresolved.add('function_capacity')
                    break
                first, last = target['functions'][canonical][0]
                code = target['source'][target['tokens'][first].start:target['tokens'][last].end]
                context = target['context'] if target is not unit and target['context'] and not any(block[0] == 'context' and block[1] == target['filename'] for block in blocks) else ''
                if sum(len(block[2].encode('utf-8')) for block in blocks) + len(code.encode('utf-8')) + len(context.encode('utf-8')) > MAX_EFFECT_SOURCE_BYTES:
                    unresolved.add('source_capacity')
                    continue
                if target['filename'] not in files:
                    files[target['filename']] = target['sha256']
                if context:
                    blocks.append(('context', target['filename'], context))
                blocks.append(('callback', canonical, code))
                queue.extend(self._references(code, target))
            text, spans = '', []
            for role, name, code in blocks:
                if not code.strip():
                    continue
                offset = len(text)
                text += code.strip() + '\n'
                spans.append({'role': role, 'name': name, 'start': offset, 'end': len(text) - 1})
            effect['raw_code'] = text.rstrip('\n')
            effect['code_source'] = {'version': LUA_CODE_SOURCE_VERSION, 'sha256': text_sha256(effect['raw_code']),
                                     'blocks': spans, 'complete': not unresolved, 'unresolved': sorted(unresolved)}
        card_data['code_source'] = {'version': LUA_CODE_SOURCE_VERSION, 'files': dict(sorted(files.items()))}
        return card_data
