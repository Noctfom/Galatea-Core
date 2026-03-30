# -*- coding: utf-8 -*-
# 语义知识库模块
# 负责解析和存储卡片效果的语义信息，供模型训练时使用

import json
import numpy as np

# 🌟 从 common.h 映射的统一规则字典
RACE_MAP = {'RACE_WARRIOR': 0x1, 'RACE_SPELLCASTER': 0x2, 'RACE_FAIRY': 0x4, 'RACE_FIEND': 0x8, 'RACE_ZOMBIE': 0x10, 'RACE_MACHINE': 0x20, 'RACE_AQUA': 0x40, 'RACE_PYRO': 0x80, 'RACE_ROCK': 0x100, 'RACE_WINDBEAST': 0x200, 'RACE_PLANT': 0x400, 'RACE_INSECT': 0x800, 'RACE_THUNDER': 0x1000, 'RACE_DRAGON': 0x2000, 'RACE_BEAST': 0x4000, 'RACE_BEASTWARRIOR': 0x8000, 'RACE_DINOSAUR': 0x10000, 'RACE_FISH': 0x20000, 'RACE_SEASERPENT': 0x40000, 'RACE_REPTILE': 0x80000, 'RACE_PSYCHO': 0x100000, 'RACE_DEVINE': 0x200000, 'RACE_CREATORGOD': 0x400000, 'RACE_WYRM': 0x800000, 'RACE_CYBERSE': 0x1000000, 'RACE_ILLUSION': 0x2000000}
ATTR_MAP = {'ATTRIBUTE_EARTH': 0x01, 'ATTRIBUTE_WATER': 0x02, 'ATTRIBUTE_FIRE': 0x04, 'ATTRIBUTE_WIND': 0x08, 'ATTRIBUTE_LIGHT': 0x10, 'ATTRIBUTE_DARK': 0x20, 'ATTRIBUTE_DEVINE': 0x40}

class SemanticKnowledgeBase:
    def __init__(self, kb_path='knowledge_base.json', vocab_size=20000):
        self.vocab_size = vocab_size
        self.reserved_ids = 10 
        #print(f"🧠 正在连接卡片效果语义知识库...")
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.kb = json.load(f)
        except Exception as e:
            print(f"⚠️ 无法加载知识库 {kb_path}: {e}，将使用空知识库。")
            self.kb = {}
            
        self.cat2idx = {'<PAD>': 0, '<UNK>': 1}
        self.req2idx = {}
        
        for cid_str, card_data in self.kb.items():
            for eff in card_data.get('effects', []):
                for cat in eff.get('categories', []):
                    if cat not in self.cat2idx: self.cat2idx[cat] = len(self.cat2idx)
                reqs = eff.get('requirements', {})
                for key in ['locations', 'phases', 'types', 'summon_types', 'reasons', 'positions']:
                    for item in reqs.get(key, []):
                        if item not in self.req2idx: self.req2idx[item] = len(self.req2idx)
                        
        self.num_cats = len(self.cat2idx)
        self.req_dim = 128 
        #print(f"✅ 知识库加载完毕！包含 {self.num_cats} 种动作，已实现表征大一统！")

    def get_card_semantics(self, card_id):
        # 压缩：动作词表不到4000，int16(2字节)足够
        cat_out = np.zeros((8, 8), dtype=np.int16)
        # 压缩：掩码只有 0/1，必须用 bool_(1字节)
        req_out = np.zeros((8, 128), dtype=np.bool_)
        # 压缩：字段同样用 int16
        set_out = np.zeros((8, 4), dtype=np.int16)
        # 压缩：数值除以 4000 后很小，float16(2字节)足够
        num_out = np.zeros((8, 4), dtype=np.float16)
        
        # 压缩：卡密可能超过 32767，用 int32(4字节) 安全
        ref_out = np.zeros((8, 4), dtype=np.int32)  
        race_out = np.zeros((8, 4), dtype=np.int16) 
        attr_out = np.zeros((8, 4), dtype=np.int16)
        
        card_id_str = str(card_id)
        if card_id_str not in self.kb:
            return cat_out, req_out, set_out, num_out, ref_out, race_out, attr_out
            
        effects = self.kb[card_id_str].get('effects', [])
        
        for i, eff in enumerate(effects):
            if i >= 8: break 
            
            for j, cat in enumerate(eff.get('categories', [])[:8]):
                cat_out[i, j] = self.cat2idx.get(cat, 1) 
                
            reqs = eff.get('requirements', {})
            for key in ['locations', 'phases', 'types', 'summon_types', 'reasons', 'positions']:
                for item in reqs.get(key, []):
                    if item in self.req2idx and self.req2idx[item] < 128:
                        req_out[i, self.req2idx[item]] = 1.0
                        
            for j, scode in enumerate(reqs.get('setcodes', [])[:4]):
                try: set_out[i, j] = (int(scode, 16) if scode.startswith('0x') else int(scode)) % 4096 
                except Exception as e: 
                    print(f"[semantic_kb]⚠️ setcode解析异常: {e} (scode={scode})")

            # 🌟 种族/属性 大一统解析
            for j, r in enumerate(reqs.get('races', [])[:4]):
                if r in RACE_MAP: race_out[i, j] = RACE_MAP[r] % 30
            for j, a in enumerate(reqs.get('attributes', [])[:4]):
                if a in ATTR_MAP: attr_out[i, j] = ATTR_MAP[a] % 10

            n_idx, r_idx = 0, 0
            for cnum in reqs.get('custom_numbers', []):
                try: 
                    val = float(cnum)
                    if val > 10000 and r_idx < 4: 
                        # 卡密：转换为 Hash Embedding 索引
                        ref_out[i, r_idx] = (int(val) % (self.vocab_size - self.reserved_ids)) + self.reserved_ids
                        r_idx += 1
                    elif n_idx < 4:
                        # 常规数值：直接除以 4000.0，与 feature_encoder.py 完全对齐！
                        num_out[i, n_idx] = val / 4000.0
                        n_idx += 1
                except Exception as e: 
                    print(f"[semantic_kb]⚠️ custom_number解析异常: {e} (cnum={cnum})")
                    
        return cat_out, req_out, set_out, num_out, ref_out, race_out, attr_out