# ==================================================================================
#  Galatea Feature Encoder (特征编码器 V3.0 - Semantic Active)
# ==================================================================================

import torch
import numpy as np
from data_types import GameSnapshot
from game_constants import Zone
from semantic_kb import SemanticKnowledgeBase  # 导入语义库

# --- 配置参数 ---
MAX_CARDS = 120          
VOCAB_SIZE = 20000       
UNK_CODE_IDX = 1         
PAD_CODE_IDX = 0         
MAX_ACTIONS = 120  

_GLOBAL_SEM_KB = None

class GalateaEncoder:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.reserved_ids = 10 
        self.global_dim = 15
        self.card_feat_dim = 7
        
        # 单例模式：防止每开一局卡顿，所有环境共享一个缓存！
        global _GLOBAL_SEM_KB
        if _GLOBAL_SEM_KB is None:
            _GLOBAL_SEM_KB = SemanticKnowledgeBase('knowledge_base.json')
        self.sem_kb = _GLOBAL_SEM_KB

    def _hash_code(self, code):
        if code == 0: return UNK_CODE_IDX 
        return (code % (self.vocab_size - self.reserved_ids)) + self.reserved_ids
    
    def _get_coords(self, player_id, owner, location, sequence):
        """将一维的 location 和 sequence 转换为二维平面坐标 (X, Y)"""
        # 非场上卡片，放入异次元坐标
        if location not in [Zone.MZONE, Zone.SZONE]:
            return -1.0, -1.0
            
        x, y = -1.0, -1.0
        is_mine = (owner == player_id)
        
        if location == Zone.MZONE:
            y = 0.3 if is_mine else 0.7
            if sequence <= 4:
                # 主怪兽区 0~4
                x = 0.1 + 0.2 * sequence if is_mine else 0.9 - 0.2 * sequence
            elif sequence == 5:
                # 左侧额外怪兽区 (对于控制者来说在 1 号位上方)
                x, y = (0.3, 0.5) if is_mine else (0.7, 0.5)
            elif sequence == 6:
                # 右侧额外怪兽区 (对于控制者来说在 3 号位上方)
                x, y = (0.7, 0.5) if is_mine else (0.3, 0.5)
                
        elif location == Zone.SZONE:
            y = 0.1 if is_mine else 0.9
            if sequence <= 4:
                # 魔陷区 0~4
                x = 0.1 + 0.2 * sequence if is_mine else 0.9 - 0.2 * sequence
            elif sequence == 5:
                # 场地区
                x, y = (0.0, 0.2) if is_mine else (1.0, 0.8)
                
        return x, y

    def encode_actions(self, valid_actions, snapshot):
        MAX_MATERIALS = 5 # 最多融合同调 5 张素材
        act_card_idxs, act_types, act_descs, masks = [], [], [], []
        act_races, act_attrs, act_codes = [], [], [] 
        act_places = [] # 新增：空间坐标数组

        for act in valid_actions[:MAX_ACTIONS]:
            # 1. 提取多目标实体/素材 (支持 5 张卡)
            if hasattr(act, 'macro_targets') and act.macro_targets:
                t_idxs = [t for t in act.macro_targets if t >= 0][:MAX_MATERIALS]
                t_idxs.extend([0] * (MAX_MATERIALS - len(t_idxs))) 
            else:
                t_idx = act.target_entity_idx if act.target_entity_idx >= 0 else 0
                t_idxs = [t_idx] + [0] * (MAX_MATERIALS - 1)
            act_card_idxs.append(t_idxs)
            
            # 2. 提取多重格子坐标 (支持同时锁 5 个格子)
            if hasattr(act, 'macro_places') and act.macro_places:
                p_vals = act.macro_places[:MAX_MATERIALS]
                p_vals.extend([0] * (MAX_MATERIALS - len(p_vals)))
            else:
                p_val = (act.desc_id % 32) + 1 if act.action_type in [18, 24] else 0
                p_vals = [p_val] + [0] * (MAX_MATERIALS - 1)
            act_places.append(p_vals)
            
            act_types.append(act.action_type)
            act_descs.append(act.desc_id % 1024)
            masks.append(True)
            
            # 3. 宣言类附加语义
            r_val, a_val, c_val = 0, 0, 0
            if act.action_type == 140: r_val = (act.desc_id.bit_length() - 1) % 30
            elif act.action_type == 141: a_val = (act.desc_id.bit_length() - 1) % 10
            elif act.action_type == 142: c_val = self._hash_code(act.desc_id)
                
            act_races.append(r_val)
            act_attrs.append(a_val)
            act_codes.append(c_val)
            
        # 4. 长度对齐 Padding
        pad_len = MAX_ACTIONS - len(act_card_idxs)
        if pad_len > 0:
            act_card_idxs.extend([[0]*MAX_MATERIALS] * pad_len) # 二维 Padding
            act_places.extend([[0]*MAX_MATERIALS] * pad_len)    # 二维 Padding
            act_types.extend([0] * pad_len)
            act_descs.extend([0] * pad_len)
            masks.extend([False] * pad_len)
            act_races.extend([0] * pad_len)
            act_attrs.extend([0] * pad_len)
            act_codes.extend([0] * pad_len)
            
        return {
            'act_card_idx': torch.tensor(act_card_idxs, dtype=torch.long).unsqueeze(0), # [1, 80, 5]
            'act_type': torch.tensor(act_types, dtype=torch.long).unsqueeze(0),
            'act_desc': torch.tensor(act_descs, dtype=torch.long).unsqueeze(0),
            'act_mask': torch.tensor(masks, dtype=torch.bool).unsqueeze(0),
            'act_race': torch.tensor(act_races, dtype=torch.long).unsqueeze(0),
            'act_attr': torch.tensor(act_attrs, dtype=torch.long).unsqueeze(0),
            'act_code': torch.tensor(act_codes, dtype=torch.long).unsqueeze(0),
            'act_place': torch.tensor(act_places, dtype=torch.long).unsqueeze(0)        # 🌟 挂载 2D 坐标 [1, 80, 5]
        }

    def encode(self, snapshot: GameSnapshot, player_id: int) -> dict:
        g = snapshot.global_data
        global_vec = [
            min(g.turn_count / 20.0, 5.0), g.phase_id / 10.0,
            1.0 if g.to_play == player_id else 0.0,
            g.my_lp / 8000.0, g.op_lp / 8000.0,
            g.my_hand_len / 10.0, g.op_hand_len / 10.0,
            g.my_deck_len / 40.0, g.op_deck_len / 40.0,
            g.my_grave_len / 20.0, g.op_grave_len / 20.0,
            g.my_removed_len / 10.0, g.op_removed_len / 10.0,
            g.my_extra_len / 15.0, g.op_extra_len / 15.0
        ]
        
        card_indices, card_feats, card_races, card_attrs, card_setcodes, masks = [], [], [], [], [], []
        card_overlay_indices = []

        # 场上实体的语义特征容器
        sem_cats, sem_reqs, sem_scs, sem_nums = [], [], [], []
        sem_refs, sem_races, sem_attrs = [], [], []
        
        # ==========================================
        # 1. 处理场上/手牌/墓地实体 (MAX_CARDS = 120)
        # ==========================================

        # [新增] 提取对方的小本本情报
        op_known = []
        if hasattr(snapshot, 'known_hand_codes'):
            op_known = snapshot.known_hand_codes[1 - player_id].copy()
            
            # 【终极防幻觉：容量坍缩校验】
            # 对方能藏牌的地方 = 手牌总数 + 场上的里侧表示卡
            hidden_capacity = 0
            for e in snapshot.entities:
                if e.owner != player_id:
                    if e.location == Zone.HAND:
                        hidden_capacity += 1
                    elif e.location in [Zone.MZONE, Zone.SZONE] and not (e.position & 0x1 or e.position & 0x4):
                        hidden_capacity += 1
                        
            # 如果记忆的牌数 > 对方实际能藏的牌数，说明有牌被消耗但没抓到（比如暗牌除外），强制遗忘
            while len(op_known) > hidden_capacity and len(op_known) > 0:
                op_known.pop(0)

        for e in snapshot.entities[:MAX_CARDS]:
            is_visible = True
            is_tracked_by_memory = False
            if e.owner != player_id:
                if e.location in [Zone.HAND, Zone.DECK]:
                    if not e.is_public: is_visible = False
                if e.location in [Zone.MZONE, Zone.SZONE] and (e.position & 0xA):
                     if not e.is_public: is_visible = False
            
            # 【状态附着】
            # 管它在哪，只要遇到一个“瞎子槽位”(手牌或场上暗卡)，并且我们还有记忆，就把记忆塞给它
            if not is_visible and len(op_known) > 0:
                # 只有手牌或场上暗牌才能承载记忆
                if e.location == Zone.HAND or (e.location in [Zone.MZONE, Zone.SZONE] and not is_visible):
                    e.code = op_known.pop(0) 
                    is_visible = True
                    is_tracked_by_memory = True

            if is_visible:
                c_idx = self._hash_code(e.code)

                pos_x, pos_y = self._get_coords(player_id, e.owner, e.location, e.sequence)

                feat_numeric = [
                    1.0 if e.owner == player_id else -1.0, e.location / 100.0, e.sequence / 10.0,
                    e.current_atk / 4000.0, e.current_def / 4000.0, e.base_atk / 4000.0, e.base_def / 4000.0,
                    pos_x,pos_y,
                    e.level / 12.0, e.lscale / 13.0, e.rscale / 13.0,
                    e.position / 10.0,
                    1.0 if e.is_public else (0.5 if is_tracked_by_memory else 0.0),
                    min(e.overlay_count / 5.0, 1.0),   # 缩放：最多算5个素材
                    min(e.counter_count / 10.0, 1.0),  # 缩放：最多算10个指示物
                    1.0 if e.is_equipped else 0.0      # 布尔值
                ]
                feat = feat_numeric + [1.0 if (e.type_mask & (1<<i)) else 0.0 for i in range(32)] + [1.0 if (e.link_marker & (1<<i)) else 0.0 for i in range(9)]
                r_idx, a_idx = e.race % 30, e.attribute % 10

                raw_sc = e.setcodes if isinstance(e.setcodes, (list, tuple)) else [e.setcodes]
                sc_hashed = [(s % 4096) for s in (list(raw_sc) + [0]*4)[:4]]
                
                # 查询语义库
                cat_out, req_out, set_out, num_out, ref_out, race_out, attr_out = self.sem_kb.get_card_semantics(e.code)
                overlay_idx = self._hash_code(getattr(e, 'top_overlay_code', 0))
            else:
                c_idx, r_idx, a_idx, sc_hashed = UNK_CODE_IDX, 0, 0, [0, 0, 0, 0]
                overlay_idx = PAD_CODE_IDX
                feat = [-1.0, e.location / 100.0, e.sequence / 10.0, -1.0, -1.0] + [0.0] * 53
                # 未知卡片返回全零语义
                cat_out, req_out, set_out, num_out, ref_out, race_out, attr_out = self.sem_kb.get_card_semantics(0)
            
            card_indices.append(c_idx); card_feats.append(feat); card_races.append(r_idx)
            card_attrs.append(a_idx); card_setcodes.append(sc_hashed); masks.append(1.0)
            card_overlay_indices.append(overlay_idx)
            sem_cats.append(cat_out); sem_reqs.append(req_out); sem_scs.append(set_out); sem_nums.append(num_out);sem_refs.append(ref_out); sem_races.append(race_out); sem_attrs.append(attr_out)

        # Padding
        pad_len = MAX_CARDS - len(card_indices)
        if pad_len > 0:
            card_indices.extend([PAD_CODE_IDX] * pad_len)
            card_overlay_indices.extend([PAD_CODE_IDX] * pad_len)
            card_races.extend([0] * pad_len)
            card_attrs.extend([0] * pad_len)
            masks.extend([0.0] * pad_len)
            for _ in range(pad_len):
                card_setcodes.append([0, 0, 0, 0])
                card_feats.append([0.0] * 58)
                sem_cats.append(np.zeros((8, 8), dtype=np.int16))
                sem_reqs.append(np.zeros((8, 128), dtype=np.bool_))
                sem_scs.append(np.zeros((8, 4), dtype=np.int16))
                sem_nums.append(np.zeros((8, 4), dtype=np.float16))
                sem_refs.append(np.zeros((8, 4), dtype=np.int32))
                sem_races.append(np.zeros((8, 4), dtype=np.int16))
                sem_attrs.append(np.zeros((8, 4), dtype=np.int16))

        # ==========================================
        # 2. 处理上帝视角卡组残像 (MAX_DECK_CARDS = 75)
        # ==========================================
        MAX_DECK_CARDS = 75
        my_deck = (snapshot.p0_deck_codes + snapshot.p0_extra_codes) if player_id == 0 else (snapshot.p1_deck_codes + snapshot.p1_extra_codes)

        deck_idx, deck_race, deck_attr, deck_setcodes, deck_masks = [], [], [], [], []
        # 卡组实体的语义特征容器
        d_sem_cats, d_sem_reqs, d_sem_scs, d_sem_nums = [], [], [], []
        d_sem_refs, d_sem_races, d_sem_attrs = [], [], []

        from card_reader import card_db
        for code in my_deck[:MAX_DECK_CARDS]:
            try:
                stats = card_db.get_full_stats(code)
                deck_race.append(stats[1] % 30)
                deck_attr.append(stats[2] % 10)
                # 容错保护
                raw_dsc = stats[10] if isinstance(stats[10], (list, tuple)) else [stats[10]]
                deck_setcodes.append([(s % 4096) for s in (list(raw_dsc) + [0]*4)[:4]])
            except Exception as e:
                print(f"⚠️ [FeatureEncoder] 卡片数据获取异常 (已记录并跳过): {e}")
                deck_race.append(0); deck_attr.append(0); deck_setcodes.append([0, 0, 0, 0])
                
            deck_idx.append(self._hash_code(code))
            deck_masks.append(True)
            
            # 给卡组里的卡也装上语义 (带大一统特征)
            dc_out, dr_out, ds_out, dn_out, dref_out, drace_out, dattr_out = self.sem_kb.get_card_semantics(code)
            d_sem_cats.append(dc_out); d_sem_reqs.append(dr_out)
            d_sem_scs.append(ds_out); d_sem_nums.append(dn_out)
            d_sem_refs.append(dref_out); d_sem_races.append(drace_out); d_sem_attrs.append(dattr_out)

        # Padding 卡组
        deck_pad_len = MAX_DECK_CARDS - len(deck_idx)
        if deck_pad_len > 0:
            deck_idx.extend([PAD_CODE_IDX] * deck_pad_len)
            deck_race.extend([0] * deck_pad_len)
            deck_attr.extend([0] * deck_pad_len)
            deck_masks.extend([False] * deck_pad_len)
            for _ in range(deck_pad_len):
                deck_setcodes.append([0, 0, 0, 0])
                d_sem_cats.append(np.zeros((8, 8), dtype=np.int16))
                d_sem_reqs.append(np.zeros((8, 128), dtype=np.bool_))
                d_sem_scs.append(np.zeros((8, 4), dtype=np.int16))
                d_sem_nums.append(np.zeros((8, 4), dtype=np.float16))
                d_sem_refs.append(np.zeros((8, 4), dtype=np.int32))
                d_sem_races.append(np.zeros((8, 4), dtype=np.int16))
                d_sem_attrs.append(np.zeros((8, 4), dtype=np.int16))
        
        # ==========================================
        # 2.5 处理连锁堆栈 (MAX_CHAIN = 5)
        # ==========================================
        MAX_CHAIN = 5
        c_sem_cats, c_sem_reqs, c_sem_scs, c_sem_nums, c_sem_refs, c_sem_races, c_sem_attrs = [], [], [], [], [], [], []
        c_masks = []
        
        # 提取堆栈里正在发动的卡片语义
        if hasattr(snapshot, 'chain_stack'):
            for item in snapshot.chain_stack[:MAX_CHAIN]:
                cc_out, cr_out, cs_out, cn_out, cref_out, crace_out, cattr_out = self.sem_kb.get_card_semantics(item['code'])
                c_sem_cats.append(cc_out); c_sem_reqs.append(cr_out); c_sem_scs.append(cs_out); c_sem_nums.append(cn_out)
                c_sem_refs.append(cref_out); c_sem_races.append(crace_out); c_sem_attrs.append(cattr_out)
                c_masks.append(True)
                
        # Padding
        c_pad_len = MAX_CHAIN - len(c_masks)
        if c_pad_len > 0:
            c_masks.extend([False] * c_pad_len)
            for _ in range(c_pad_len):
                c_sem_cats.append(np.zeros((8, 8), dtype=np.int16))
                c_sem_reqs.append(np.zeros((8, 128), dtype=np.bool_))
                c_sem_scs.append(np.zeros((8, 4), dtype=np.int16))
                c_sem_nums.append(np.zeros((8, 4), dtype=np.float16))
                c_sem_refs.append(np.zeros((8, 4), dtype=np.int32))
                c_sem_races.append(np.zeros((8, 4), dtype=np.int16))
                c_sem_attrs.append(np.zeros((8, 4), dtype=np.int16))

        # ==========================================
        # 2.6 处理动作历史雷达 (MAX_HISTORY = 8)
        # ==========================================
        MAX_HISTORY = 8
        h_sem_cats, h_sem_reqs, h_sem_scs, h_sem_nums, h_sem_refs, h_sem_races, h_sem_attrs = [], [], [], [], [], [], []
        h_masks = []
        
        if hasattr(snapshot, 'history_stack'):
            for item in snapshot.history_stack[:MAX_HISTORY]:
                hc_out, hr_out, hs_out, hn_out, href_out, hrace_out, hattr_out = self.sem_kb.get_card_semantics(item['code'])
                h_sem_cats.append(hc_out); h_sem_reqs.append(hr_out); h_sem_scs.append(hs_out); h_sem_nums.append(hn_out)
                h_sem_refs.append(href_out); h_sem_races.append(hrace_out); h_sem_attrs.append(hattr_out)
                h_masks.append(True)
                
        h_pad_len = MAX_HISTORY - len(h_masks)
        if h_pad_len > 0:
            h_masks.extend([False] * h_pad_len)
            for _ in range(h_pad_len):
                h_sem_cats.append(np.zeros((8, 8), dtype=np.int16))
                h_sem_reqs.append(np.zeros((8, 128), dtype=np.bool_))
                h_sem_scs.append(np.zeros((8, 4), dtype=np.int16))
                h_sem_nums.append(np.zeros((8, 4), dtype=np.float16))
                h_sem_refs.append(np.zeros((8, 4), dtype=np.int32))
                h_sem_races.append(np.zeros((8, 4), dtype=np.int16))
                h_sem_attrs.append(np.zeros((8, 4), dtype=np.int16))

        # ==========================================
        # 3. 最终打包
        # ==========================================
        act_dict = self.encode_actions(snapshot.valid_actions, snapshot)
        
        base_dict = {
            'global': torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0),
            
            'card_idx': torch.tensor(card_indices, dtype=torch.long).unsqueeze(0),
            'card_overlay_idx': torch.tensor(card_overlay_indices, dtype=torch.long).unsqueeze(0),
            'card_race': torch.tensor(card_races, dtype=torch.long).unsqueeze(0), 
            'card_attr': torch.tensor(card_attrs, dtype=torch.long).unsqueeze(0), 
            'card_setcodes': torch.tensor(card_setcodes, dtype=torch.long).unsqueeze(0), 
            'card_feats': torch.tensor(card_feats, dtype=torch.float32).unsqueeze(0),
            'padding_mask': torch.tensor(masks, dtype=torch.bool).unsqueeze(0),
            
            # 挂载场上实体的语义张量 [Batch, 100, 4, ...]
            'sem_category': torch.from_numpy(np.array(sem_cats)).unsqueeze(0),
            'sem_req': torch.from_numpy(np.array(sem_reqs)).unsqueeze(0),
            'sem_setcode': torch.from_numpy(np.array(sem_scs)).unsqueeze(0),
            'sem_number': torch.from_numpy(np.array(sem_nums)).unsqueeze(0),
            'sem_ref': torch.from_numpy(np.array(sem_refs)).unsqueeze(0),
            'sem_race': torch.from_numpy(np.array(sem_races)).unsqueeze(0),
            'sem_attr': torch.from_numpy(np.array(sem_attrs)).unsqueeze(0),
            
            'deck_idx': torch.tensor(deck_idx, dtype=torch.long).unsqueeze(0),
            'deck_race': torch.tensor(deck_race, dtype=torch.long).unsqueeze(0),
            'deck_attr': torch.tensor(deck_attr, dtype=torch.long).unsqueeze(0),
            'deck_setcodes': torch.tensor(deck_setcodes, dtype=torch.long).unsqueeze(0),
            'deck_mask': torch.tensor(deck_masks, dtype=torch.bool).unsqueeze(0),
            
            # 挂载上帝视角的语义张量 [Batch, 75, 4, ...]
            'd_sem_category': torch.from_numpy(np.array(d_sem_cats)).unsqueeze(0),
            'd_sem_req': torch.from_numpy(np.array(d_sem_reqs)).unsqueeze(0),
            'd_sem_setcode': torch.from_numpy(np.array(d_sem_scs)).unsqueeze(0),
            'd_sem_number': torch.from_numpy(np.array(d_sem_nums)).unsqueeze(0),
            'd_sem_ref': torch.from_numpy(np.array(d_sem_refs)).unsqueeze(0),
            'd_sem_race': torch.from_numpy(np.array(d_sem_races)).unsqueeze(0),
            'd_sem_attr': torch.from_numpy(np.array(d_sem_attrs)).unsqueeze(0),

            'c_mask': torch.tensor(c_masks, dtype=torch.bool).unsqueeze(0),
            'c_sem_category': torch.from_numpy(np.array(c_sem_cats)).unsqueeze(0),
            'c_sem_req': torch.from_numpy(np.array(c_sem_reqs)).unsqueeze(0),
            'c_sem_setcode': torch.from_numpy(np.array(c_sem_scs)).unsqueeze(0),
            'c_sem_number': torch.from_numpy(np.array(c_sem_nums)).unsqueeze(0),
            'c_sem_ref': torch.from_numpy(np.array(c_sem_refs)).unsqueeze(0),
            'c_sem_race': torch.from_numpy(np.array(c_sem_races)).unsqueeze(0),
            'c_sem_attr': torch.from_numpy(np.array(c_sem_attrs)).unsqueeze(0),
        
            'h_mask': torch.tensor(h_masks, dtype=torch.bool).unsqueeze(0),
            'h_sem_category': torch.from_numpy(np.array(h_sem_cats)).unsqueeze(0),
            'h_sem_req': torch.from_numpy(np.array(h_sem_reqs)).unsqueeze(0),
            'h_sem_setcode': torch.from_numpy(np.array(h_sem_scs)).unsqueeze(0),
            'h_sem_number': torch.from_numpy(np.array(h_sem_nums)).unsqueeze(0),
            'h_sem_ref': torch.from_numpy(np.array(h_sem_refs)).unsqueeze(0),
            'h_sem_race': torch.from_numpy(np.array(h_sem_races)).unsqueeze(0),
            'h_sem_attr': torch.from_numpy(np.array(h_sem_attrs)).unsqueeze(0),
        }
        
        base_dict.update(act_dict)
        return base_dict

if __name__ == "__main__":
    enc = GalateaEncoder()
    print("Encoder (with Semantic Active) Ready.")