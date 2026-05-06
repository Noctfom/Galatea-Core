'''
Deck 相关的工具函数 (增强版)
'''
import random
import os

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
                code = int(line)
                if current_section == 'main': d.main.append(code)
                elif current_section == 'extra': d.extra.append(code)
            except Exception:
                print(f"[Deck] ⚠️ 解析 {deck_name}.ydk 时遇到非整数行: {line}")
            
    return d

def get_random_deck_pair(ydk_dir='./decks'):
    """
    随机选两个卡组 (支持子文件夹环境隔离)
    返回: (name1, deck1_obj, name2, deck2_obj)
    """
    if not os.path.exists(ydk_dir):
        return None, None, None, None

    # 1. 寻找所有子文件夹 (代表不同的环境/卡池)
    subdirs = [os.path.join(ydk_dir, d) for d in os.listdir(ydk_dir) if os.path.isdir(os.path.join(ydk_dir, d))]
    
    # 如果没有子文件夹，就把根目录当成默认环境
    if not subdirs:
        subdirs = [ydk_dir]
        
    # 2. 随机选中一个环境 (比如 2024_meta)
    chosen_env = random.choice(subdirs)
    
    # 3. 在该环境下抽取两个卡组
    names = list_decks(chosen_env)
    if len(names) < 1:
        print(f"[Deck] ⚠️ 文件夹 {chosen_env} 下没有找到 .ydk 文件!")
        return None, None, None, None
        
    # 随机抽两个名字（允许同名，即镜像对局）
    n1 = random.choice(names)
    n2 = random.choice(names)
    
    d1 = load_deck(chosen_env, n1)
    d2 = load_deck(chosen_env, n2)
    env_name = os.path.basename(os.path.normpath(chosen_env))
    
    return env_name, n1, d1, n2, d2