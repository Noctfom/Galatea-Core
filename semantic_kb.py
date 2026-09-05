# -*- coding: utf-8 -*-
# 语义知识库模块
# 负责解析和存储卡片效果的语义信息，供模型训练时使用

import numpy as np
import time
import random
from pathlib import Path

from protocol_v3_audit import register_semantic_audit_catalog
from semantic_assets import validate_semantic_bundle
from effect_slot_binding import register_runtime_effect_bindings

#从 common.h 映射的统一规则字典
RACE_MAP = {'RACE_WARRIOR': 0x1, 'RACE_SPELLCASTER': 0x2, 'RACE_FAIRY': 0x4, 'RACE_FIEND': 0x8, 'RACE_ZOMBIE': 0x10, 'RACE_MACHINE': 0x20, 'RACE_AQUA': 0x40, 'RACE_PYRO': 0x80, 'RACE_ROCK': 0x100, 'RACE_WINDBEAST': 0x200, 'RACE_PLANT': 0x400, 'RACE_INSECT': 0x800, 'RACE_THUNDER': 0x1000, 'RACE_DRAGON': 0x2000, 'RACE_BEAST': 0x4000, 'RACE_BEASTWARRIOR': 0x8000, 'RACE_DINOSAUR': 0x10000, 'RACE_FISH': 0x20000, 'RACE_SEASERPENT': 0x40000, 'RACE_REPTILE': 0x80000, 'RACE_PSYCHO': 0x100000, 'RACE_DEVINE': 0x200000, 'RACE_CREATORGOD': 0x400000, 'RACE_WYRM': 0x800000, 'RACE_CYBERSE': 0x1000000, 'RACE_ILLUSION': 0x2000000}
ATTR_MAP = {'ATTRIBUTE_EARTH': 0x01, 'ATTRIBUTE_WATER': 0x02, 'ATTRIBUTE_FIRE': 0x04, 'ATTRIBUTE_WIND': 0x08, 'ATTRIBUTE_LIGHT': 0x10, 'ATTRIBUTE_DARK': 0x20, 'ATTRIBUTE_DEVINE': 0x40}

class SemanticKnowledgeBase:
    def __init__(self, kb_path='knowledge_base.json', vocab_size=20000):
        """严格加载结构语义与代码向量，拒绝不完整或互相错位的资产"""
        self._cache = {}
        self.vocab_size = vocab_size
        self.reserved_ids = 10 
        time.sleep(random.uniform(0.1, 1.5))
        kb_path_obj = Path(kb_path).resolve()
        try:
            validated_bundle = validate_semantic_bundle(
                kb_path_obj.parent,
                knowledge_base_filename=kb_path_obj.name,
            )
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"语义资产完整性校验失败，已拒绝启动模型: {error}"
            ) from error
        kb_data = validated_bundle["knowledge_base"]
        register_runtime_effect_bindings(kb_data)
        register_semantic_audit_catalog(kb_data)
            
        self.cat2idx = {'<PAD>': 0, '<UNK>': 1}
        self.req2idx = {}
        
        for cid_str, card_data in kb_data.items():
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
        self.code_dim = 384
        self.code_embeddings = np.load(
            validated_bundle["embedding_path"],
            allow_pickle=False,
        )
        self.hash2idx = validated_bundle["index"]

        for cid_str in kb_data.keys():
            card_id = int(cid_str)
            self._cache[card_id] = self._build_card_semantics(card_id, kb_data[cid_str])
            
        del kb_data
        import gc; gc.collect()

    def _build_card_semantics(self, card_id, card_data):
        cat_out = np.zeros((8, 8), dtype=np.int16)
        req_out = np.full((8, 16), -1, dtype=np.int8)  # 🛡️ 核心修复 3：128维 bool 坍缩为 16 维紧凑 Index
        set_out = np.zeros((8, 4), dtype=np.int16)
        num_out = np.zeros((8, 4), dtype=np.float16)
        ref_out = np.zeros((8, 4), dtype=np.int32)  
        race_out = np.zeros((8, 4), dtype=np.int16) 
        attr_out = np.zeros((8, 4), dtype=np.int16)
        code_idx_out = np.zeros((8,), dtype=np.int32)

        effects = card_data.get('effects', [])
        
        for fallback_slot, eff in enumerate(effects, start=1):
            try:
                i = int(eff.get('slot', fallback_slot)) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= i < 8:
                continue
            
            for j, cat in enumerate(eff.get('categories', [])[:8]):
                cat_out[i, j] = self.cat2idx.get(cat, 1) 
                
            reqs = eff.get('requirements', {})
            req_idx = 0
            for key in ['locations', 'phases', 'types', 'summon_types', 'reasons', 'positions']:
                for item in reqs.get(key, []):
                    if item in self.req2idx and self.req2idx[item] < 128:
                        if req_idx < 16:
                            req_out[i, req_idx] = self.req2idx[item]
                            req_idx += 1
                        
            for j, scode in enumerate(reqs.get('setcodes', [])[:4]):
                try: set_out[i, j] = (int(scode, 16) if scode.startswith('0x') else int(scode)) % 4096 
                except Exception: pass

            for j, r in enumerate(reqs.get('races', [])[:4]):
                if r in RACE_MAP: race_out[i, j] = RACE_MAP[r] % 30
            for j, a in enumerate(reqs.get('attributes', [])[:4]):
                if a in ATTR_MAP: attr_out[i, j] = ATTR_MAP[a] % 10

            n_idx, r_idx = 0, 0
            for cnum in reqs.get('custom_numbers', []):
                try: 
                    val = float(cnum)
                    if val > 10000 and r_idx < 4: 
                        ref_out[i, r_idx] = (int(val) % (self.vocab_size - self.reserved_ids)) + self.reserved_ids
                        r_idx += 1
                    elif n_idx < 4:
                        num_out[i, n_idx] = val / 4000.0
                        n_idx += 1
                except Exception: pass

            key = f"{card_id}_{i}"
            if self.code_embeddings is not None and key in self.hash2idx:
                idx = self.hash2idx[key]
                code_idx_out[i] = idx + 1
                    
        return (cat_out, req_out, set_out, num_out, ref_out, race_out, attr_out, code_idx_out)

    def get_card_semantics(self, card_id):
        if card_id in self._cache:
            return self._cache[card_id]
        return (
            np.zeros((8, 8), dtype=np.int16),
            np.full((8, 16), -1, dtype=np.int8),
            np.zeros((8, 4), dtype=np.int16),
            np.zeros((8, 4), dtype=np.float16),
            np.zeros((8, 4), dtype=np.int32),
            np.zeros((8, 4), dtype=np.int16),
            np.zeros((8, 4), dtype=np.int16),
            np.zeros((8,), dtype=np.int32)
        )
