'''
Deck 相关的工具函数 (增强版)
'''
import random
import os
import json
from card_reader import card_db

class Deck:
    def __init__(self, name="Unknown"):
        self.name = name # [新增] 记录卡组名
        self.main = []
        self.extra = [] 
        self.side = []

def list_decks(deck_dir):
    """获取目录下所有卡组的名字列表 (不含.ydk后缀)"""
    if not os.path.exists(deck_dir):
        return []
    return [f[:-4] for f in os.listdir(deck_dir) if f.endswith('.ydk')]

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
        mtime = os.path.getmtime(filepath)
        if mtime != _mtime_dict[cache_key]:
            with open(filepath, 'r', encoding='utf-8') as f:
                _cache_dict[cache_key] = json.load(f)
            _mtime_dict[cache_key] = mtime
        return _cache_dict[cache_key]
    except Exception: return _cache_dict[cache_key]

def get_random_deck_pair(ydk_dir='./decks'):
    if not os.path.exists(ydk_dir): return None, None, None, None
    subdirs = [os.path.join(ydk_dir, d) for d in os.listdir(ydk_dir) if os.path.isdir(os.path.join(ydk_dir, d))]
    
    if not subdirs:
        names = list_decks(ydk_dir)
        if len(names) < 2: return None, None, None, None
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
        if sum(v_weights) <= 0: return None, None, None, None 
        
        c_env1 = random.choices(subdirs, weights=v_weights, k=1)[0]
        c_env2 = random.choices(subdirs, weights=v_weights, k=1)[0]
        
        names1, names2 = list_decks(c_env1), list_decks(c_env2)
        if not names1 or not names2: return None, None, None, None
        
        n1, n2 = random.choice(names1), random.choice(names2)
        return chosen_env, n1, load_deck(c_env1, n1), n2, load_deck(c_env2, n2)

    # --- 路径 B：抽中了物理池 (内战) ---
    else:
        chosen_dir = os.path.join(ydk_dir, chosen_env)
        names = list_decks(chosen_dir)
        if len(names) < 1: return None, None, None, None
        n1, n2 = random.choice(names), random.choice(names)
        return chosen_env, n1, load_deck(chosen_dir, n1), n2, load_deck(chosen_dir, n2)