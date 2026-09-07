'''
Deck 相关的工具函数 (增强版)
'''
import random
import os
import json
import time
import math
from dataclasses import dataclass
from card_reader import card_db

_last_io_check = {'global': 0, 'virtual': 0}

class Deck:
    def __init__(self, name="Unknown"):
        self.name = name # [新增] 记录卡组名
        self.main = []
        self.extra = [] 
        self.side = []


ARENA_SOURCE_WEIGHTED = "weighted"
ARENA_SOURCE_PHYSICAL = "physical"
ARENA_SOURCE_VIRTUAL = "virtual"
ARENA_SOURCE_DECK = "deck"
ARENA_SOURCE_SAME_RANGE = "same_range"
ARENA_SOURCE_SAME_DECK = "same_deck"


@dataclass(frozen=True)
class ArenaDeckSource:
    """描述竞技场一侧的卡组选择方式"""

    kind: str
    value: str = ""


@dataclass
class ArenaDeckPick:
    """保存一次已解析的卡组、所属区间与安全文件位置"""

    range_kind: str
    range_name: str
    pool_name: str
    deck_name: str
    deck: Deck
    deck_path: str


@dataclass
class ArenaDeckPair:
    """保存逻辑 P0/P1 的完整卡组抽取结果"""

    p0: ArenaDeckPick
    p1: ArenaDeckPick

def list_decks(deck_dir):
    """获取目录下所有卡组的名字列表 (不含.ydk后缀)"""
    if not os.path.exists(deck_dir):
        return []
    return [f[:-4] for f in os.listdir(deck_dir) if f.endswith('.ydk')]


def _list_arena_decks(deck_dir):
    """仅为竞技场列出普通卡组文件，并排除目录和符号链接"""
    if not os.path.isdir(deck_dir):
        return []
    decks = []
    with os.scandir(deck_dir) as entries:
        for entry in entries:
            if (
                entry.name.endswith('.ydk')
                and entry.is_file(follow_symlinks=False)
            ):
                decks.append(entry.name[:-4])
    return decks


def _read_json_object(filepath):
    """读取卡组调度 JSON，异常内容按空对象处理"""
    try:
        with open(filepath, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _safe_weight(value):
    """把外部权重限制为有限非负浮点数"""
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    return weight if math.isfinite(weight) and weight > 0.0 else 0.0


def discover_arena_deck_catalog(ydk_dir="./decks"):
    """扫描竞技场可选的物理池、虚拟池和单卡组标识"""
    root = os.path.realpath(ydk_dir)
    physical_pools = {}
    if not os.path.isdir(root):
        return {
            "root": root,
            "physical_pools": {},
            "virtual_pools": {},
            "global_weights": {},
            "weighted_ranges": [],
        }

    root_decks = sorted(_list_arena_decks(root), key=str.casefold)
    if root_decks:
        physical_pools["."] = root_decks

    subdir_names = []
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            names = sorted(_list_arena_decks(entry.path), key=str.casefold)
            if names:
                physical_pools[entry.name] = names
                subdir_names.append(entry.name)
    subdir_names.sort(key=str.casefold)

    virtual_payload = _read_json_object(os.path.join(root, "virtual_pools.json"))
    virtual_pools = {}
    for virtual_name, raw_config in virtual_payload.items():
        if not isinstance(virtual_name, str) or not isinstance(raw_config, dict):
            continue
        normalized = {
            pool_name: _safe_weight(raw_config.get(pool_name, 0.0))
            for pool_name in physical_pools
        }
        if any(weight > 0.0 for weight in normalized.values()):
            virtual_pools[virtual_name] = normalized

    global_weights = _read_json_object(os.path.join(root, "global_weights.json"))
    weighted_physical = subdir_names if subdir_names else (
        ["."] if root_decks else []
    )
    weighted_ranges = [
        {"kind": ARENA_SOURCE_PHYSICAL, "name": name}
        for name in weighted_physical
    ]
    weighted_ranges.extend(
        {"kind": ARENA_SOURCE_VIRTUAL, "name": name}
        for name in sorted(virtual_pools, key=str.casefold)
    )
    return {
        "root": root,
        "physical_pools": physical_pools,
        "virtual_pools": virtual_pools,
        "global_weights": global_weights,
        "weighted_ranges": weighted_ranges,
    }


def parse_arena_deck_source(value, *, allow_follow=False):
    """解析 CLI/WebUI 卡组来源，并拒绝未声明的模式"""
    if isinstance(value, ArenaDeckSource):
        source = value
    else:
        raw_value = str(value or ARENA_SOURCE_WEIGHTED).strip()
        kind, separator, detail = raw_value.partition(":")
        source = ArenaDeckSource(kind=kind, value=detail if separator else "")

    allowed = {
        ARENA_SOURCE_WEIGHTED,
        ARENA_SOURCE_PHYSICAL,
        ARENA_SOURCE_VIRTUAL,
        ARENA_SOURCE_DECK,
    }
    if allow_follow:
        allowed.update({ARENA_SOURCE_SAME_RANGE, ARENA_SOURCE_SAME_DECK})
    if source.kind not in allowed:
        raise ValueError(f"不支持的竞技场卡组来源: {source.kind!r}")
    if source.kind in {
        ARENA_SOURCE_PHYSICAL,
        ARENA_SOURCE_VIRTUAL,
        ARENA_SOURCE_DECK,
    } and not source.value:
        raise ValueError(f"竞技场卡组来源 {source.kind!r} 缺少选择值")
    if source.kind in {
        ARENA_SOURCE_WEIGHTED,
        ARENA_SOURCE_SAME_RANGE,
        ARENA_SOURCE_SAME_DECK,
    } and source.value:
        raise ValueError(f"竞技场卡组来源 {source.kind!r} 不接受附加值")
    return source


def format_arena_deck_source(source):
    """把卡组来源转换为稳定的 CLI 字符串"""
    parsed = parse_arena_deck_source(source, allow_follow=True)
    return parsed.kind if not parsed.value else f"{parsed.kind}:{parsed.value}"


def _load_arena_deck_pick(catalog, pool_name, deck_name, range_kind, range_name):
    """从已扫描目录中加载单一卡组，避免把用户字符串直接当路径"""
    available = catalog["physical_pools"].get(pool_name)
    if available is None or deck_name not in available:
        raise ValueError(f"竞技场卡组不存在: {pool_name}/{deck_name}")
    raw_pool_dir = (
        catalog["root"]
        if pool_name == "."
        else os.path.join(catalog["root"], pool_name)
    )
    if pool_name != "." and os.path.islink(raw_pool_dir):
        raise ValueError(f"竞技场物理池不能是符号链接: {pool_name}")
    pool_dir = os.path.realpath(raw_pool_dir)
    catalog_root = os.path.realpath(catalog["root"])
    if os.path.commonpath((catalog_root, pool_dir)) != catalog_root:
        raise ValueError("竞技场物理池路径越过卡组根目录")
    raw_deck_path = os.path.join(pool_dir, f"{deck_name}.ydk")
    if os.path.islink(raw_deck_path):
        raise ValueError(f"竞技场卡组不能是符号链接: {pool_name}/{deck_name}")
    deck_path = os.path.realpath(raw_deck_path)
    if os.path.commonpath((pool_dir, deck_path)) != pool_dir:
        raise ValueError("竞技场卡组路径越过所选物理池")
    if not os.path.isfile(deck_path):
        raise ValueError(f"竞技场卡组文件不可用: {pool_name}/{deck_name}")
    deck = load_deck(pool_dir, deck_name)
    if deck is None:
        raise ValueError(f"竞技场卡组加载失败: {pool_name}/{deck_name}")
    return ArenaDeckPick(
        range_kind=range_kind,
        range_name=range_name,
        pool_name=pool_name,
        deck_name=deck_name,
        deck=deck,
        deck_path=deck_path,
    )


def _choose_arena_item(items, weights, rng):
    """按权重选择一项；全部为零时恢复均匀随机"""
    if not items:
        raise ValueError("竞技场卡组候选为空")
    safe_weights = [_safe_weight(weight) for weight in weights]
    if sum(safe_weights) <= 0.0:
        safe_weights = [1.0] * len(items)
    return rng.choices(items, weights=safe_weights, k=1)[0]


def _pick_from_physical(catalog, pool_name, rng, range_kind, range_name):
    """从指定物理池随机抽取一副卡组"""
    names = catalog["physical_pools"].get(pool_name)
    if not names:
        raise ValueError(f"竞技场物理池不存在或为空: {pool_name}")
    deck_name = rng.choice(names)
    return _load_arena_deck_pick(
        catalog,
        pool_name,
        deck_name,
        range_kind,
        range_name,
    )


def _pick_from_virtual(catalog, virtual_name, rng):
    """按虚拟池配方先抽物理池，再抽具体卡组"""
    config = catalog["virtual_pools"].get(virtual_name)
    if not config:
        raise ValueError(f"竞技场虚拟池不存在或无有效权重: {virtual_name}")
    pool_names = list(config)
    pool_name = _choose_arena_item(
        pool_names,
        [config[name] for name in pool_names],
        rng,
    )
    return _pick_from_physical(
        catalog,
        pool_name,
        rng,
        ARENA_SOURCE_VIRTUAL,
        virtual_name,
    )


def select_arena_deck(
    catalog,
    source,
    rng=None,
    *,
    p0_pick=None,
    allow_follow=False,
):
    """按来源选择一副卡组，并为 P1 处理跟随 P0 的两种模式"""
    rng = rng or random
    source = parse_arena_deck_source(source, allow_follow=allow_follow)

    if source.kind == ARENA_SOURCE_SAME_DECK:
        if p0_pick is None:
            raise ValueError("P1 跟随同一卡组时缺少 P0 抽取结果")
        copied_deck = Deck(name=p0_pick.deck.name)
        copied_deck.main = list(p0_pick.deck.main)
        copied_deck.extra = list(p0_pick.deck.extra)
        copied_deck.side = list(p0_pick.deck.side)
        return ArenaDeckPick(
            range_kind=p0_pick.range_kind,
            range_name=p0_pick.range_name,
            pool_name=p0_pick.pool_name,
            deck_name=p0_pick.deck_name,
            deck=copied_deck,
            deck_path=p0_pick.deck_path,
        )

    if source.kind == ARENA_SOURCE_SAME_RANGE:
        if p0_pick is None:
            raise ValueError("P1 跟随同一区间时缺少 P0 抽取结果")
        source = ArenaDeckSource(p0_pick.range_kind, p0_pick.range_name)

    if source.kind == ARENA_SOURCE_WEIGHTED:
        ranges = catalog["weighted_ranges"]
        range_item = _choose_arena_item(
            ranges,
            [
                catalog["global_weights"].get(item["name"], 1.0)
                for item in ranges
            ],
            rng,
        )
        source = ArenaDeckSource(range_item["kind"], range_item["name"])

    if source.kind == ARENA_SOURCE_PHYSICAL:
        return _pick_from_physical(
            catalog,
            source.value,
            rng,
            ARENA_SOURCE_PHYSICAL,
            source.value,
        )
    if source.kind == ARENA_SOURCE_VIRTUAL:
        return _pick_from_virtual(catalog, source.value, rng)
    if source.kind == ARENA_SOURCE_DECK:
        pool_name, separator, deck_name = source.value.partition("/")
        if not separator or not pool_name or not deck_name:
            raise ValueError("单一卡组来源必须使用 deck:<物理池>/<卡组名>")
        return _load_arena_deck_pick(
            catalog,
            pool_name,
            deck_name,
            ARENA_SOURCE_PHYSICAL,
            pool_name,
        )
    raise ValueError(f"竞技场卡组来源无法解析: {source.kind}")


def select_arena_deck_pair(
    ydk_dir="./decks",
    p0_source=ARENA_SOURCE_WEIGHTED,
    p1_source=ARENA_SOURCE_SAME_RANGE,
    rng=None,
    catalog=None,
):
    """为竞技场逻辑 P0/P1 解析一组卡组"""
    rng = rng or random
    catalog = catalog or discover_arena_deck_catalog(ydk_dir)
    p0_pick = select_arena_deck(catalog, p0_source, rng=rng)
    p1_pick = select_arena_deck(
        catalog,
        p1_source,
        rng=rng,
        p0_pick=p0_pick,
        allow_follow=True,
    )
    return ArenaDeckPair(p0=p0_pick, p1=p1_pick)

def load_deck(base_dir, deck_name):
    """根据名字加载卡组"""
    filepath = os.path.join(base_dir, f"{deck_name}.ydk")
    d = Deck(name=deck_name)
    current_section = 'ignore' # 初始状态为忽略，直到碰到 #main
    
    if not os.path.exists(filepath):
        return None

    # 使用 errors='ignore' 防止因为奇怪字符导致崩溃
    with open(filepath, 'r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('!'): 
                current_section = 'ignore' # 暂时忽略 side
                continue
                
            # 核心修复：绝对白名单区域划分
            if line.startswith('#'):
                if line == '#main': current_section = 'main'
                elif line == '#extra': current_section = 'extra'
                else: current_section = 'ignore' # 屏蔽 #pickup, #case 等一切杂音
                continue
            
            # 如果处于被忽略的区域，直接跳过解析
            if current_section == 'ignore':
                continue
                
            try:
                raw_code = int(line)
                code = card_db.get_base_code(raw_code)
                if current_section == 'main': d.main.append(code)
                elif current_section == 'extra': d.extra.append(code)
            except Exception:
                print(f"[Deck] ⚠️ 解析 {deck_name}.ydk 时遇到非整数行: {line}")
            
    return d

# --- 双通道零 IO 缓存系统 ---
_cache_dict = {'global': {}, 'virtual': {}}
_mtime_dict = {'global': 0, 'virtual': 0}

def get_json_data(filepath, cache_key):
    if not os.path.exists(filepath): return {}
    try:
        now = time.time()
        # 冷却时间：10秒内绝不重新查询硬盘文件状态
        if now - _last_io_check[cache_key] > 10.0:
            mtime = os.path.getmtime(filepath)
            _last_io_check[cache_key] = now
            if mtime != _mtime_dict[cache_key]:
                with open(filepath, 'r', encoding='utf-8') as f:
                    _cache_dict[cache_key] = json.load(f)
                _mtime_dict[cache_key] = mtime
        return _cache_dict[cache_key]
    except Exception: 
        return _cache_dict[cache_key]

def get_random_deck_pair(ydk_dir='./decks'):
    """随机返回环境名和两副卡组；无法组成对局时统一返回空值"""
    if not os.path.exists(ydk_dir):
        return None
    subdirs = [os.path.join(ydk_dir, d) for d in os.listdir(ydk_dir) if os.path.isdir(os.path.join(ydk_dir, d))]

    if not subdirs:
        names = list_decks(ydk_dir)
        if len(names) < 2:
            return None
        n1, n2 = random.choice(names), random.choice(names)
        return "Root_Mix", n1, load_deck(ydk_dir, n1), n2, load_deck(ydk_dir, n2)

    # 1. 分别加载全局权重与虚拟池配方
    global_file = os.path.join(ydk_dir, 'global_weights.json')
    virtual_file = os.path.join(ydk_dir, 'virtual_pools.json')

    global_weights = get_json_data(global_file, 'global')
    virtual_pools = get_json_data(virtual_file, 'virtual')

    # 2. 候选名单 = 所有物理文件夹 + 所有虚拟池
    subdir_names = [os.path.basename(os.path.normpath(d)) for d in subdirs]
    env_choices = subdir_names + list(virtual_pools.keys())

    # 3. 提取全局权重 (如果没配，默认给 1.0)
    weights = [float(global_weights.get(name, 1.0)) for name in env_choices]
    if sum(weights) <= 0: weights = [1.0] * len(env_choices)

    chosen_env = random.choices(env_choices, weights=weights, k=1)[0]

    # --- 路径 A：抽中了虚拟拼装池 ---
    if chosen_env in virtual_pools:
        pool_cfg = virtual_pools[chosen_env]
        # 在虚拟池内，根据配方权重重新抽取物理池
        v_weights = [float(pool_cfg.get(name, 0.0)) for name in subdir_names]
        if sum(v_weights) <= 0:
            return None

        c_env1 = random.choices(subdirs, weights=v_weights, k=1)[0]
        c_env2 = random.choices(subdirs, weights=v_weights, k=1)[0]

        names1, names2 = list_decks(c_env1), list_decks(c_env2)
        if not names1 or not names2:
            return None

        n1, n2 = random.choice(names1), random.choice(names2)
        return chosen_env, n1, load_deck(c_env1, n1), n2, load_deck(c_env2, n2)

    # --- 路径 B：抽中了物理池 (内战) ---
    else:
        chosen_dir = os.path.join(ydk_dir, chosen_env)
        names = list_decks(chosen_dir)
        if len(names) < 1:
            return None
        n1, n2 = random.choice(names), random.choice(names)
        return chosen_env, n1, load_deck(chosen_dir, n1), n2, load_deck(chosen_dir, n2)
