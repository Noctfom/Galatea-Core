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
    current_section = 'main'
    
    if not os.path.exists(filepath):
        return None

    # 使用 errors='ignore' 防止因为奇怪字符导致崩溃
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('!'): continue
            if line.startswith('#'):
                if 'extra' in line: current_section = 'extra'
                elif 'main' in line: current_section = 'main'
                continue
            
            try:
                code = int(line)
                if current_section == 'main': d.main.append(code)
                elif current_section == 'extra': d.extra.append(code)
            except: pass
            
    return d

def get_random_deck_pair(ydk_dir='./decks'):
    """
    随机选两个卡组
    返回: (name1, deck1_obj, name2, deck2_obj)
    """
    names = list_decks(ydk_dir)
    if len(names) < 1:
        print(f"[Deck] No .ydk files found in {ydk_dir}!")
        return None, None, None, None
    
    # 随机抽两个名字（允许同名，即镜像对局）
    n1 = random.choice(names)
    n2 = random.choice(names)
    
    d1 = load_deck(ydk_dir, n1)
    d2 = load_deck(ydk_dir, n2)
    
    return n1, d1, n2, d2