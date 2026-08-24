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
HIDDEN_OPPONENT_ZONES = {
    Zone.HAND,
    Zone.DECK,
    Zone.MZONE,
    Zone.SZONE,
    Zone.REMOVED,
    Zone.EXTRA,
}

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
        if code == 0:
            return UNK_CODE_IDX
        return (code % (self.vocab_size - self.reserved_ids)) + self.reserved_ids

    @staticmethod
    def _encode_global_vector(g, player_id):
        """Encode fixed P0/P1 state fields from the acting player's view."""
        if player_id not in (0, 1):
            raise ValueError(f"player_id must be 0 or 1, got {player_id}")

        resource_pairs = [
            (g.my_lp, g.op_lp, 8000.0),
            (g.my_hand_len, g.op_hand_len, 10.0),
            (g.my_deck_len, g.op_deck_len, 40.0),
            (g.my_grave_len, g.op_grave_len, 20.0),
            (g.my_removed_len, g.op_removed_len, 10.0),
            (g.my_extra_len, g.op_extra_len, 15.0),
        ]

        if player_id == 1:
            resource_pairs = [(op, me, scale) for me, op, scale in resource_pairs]

        global_vec = [
            min(g.turn_count / 20.0, 5.0),
            g.phase_id / 10.0,
            1.0 if g.to_play == player_id else 0.0,
        ]
        for me, opponent, scale in resource_pairs:
            global_vec.extend([me / scale, opponent / scale])
        return global_vec

    @staticmethod
    def _is_entity_visible_to_player(entity, player_id):
        if entity.owner == player_id:
            return True

        if entity.location in HIDDEN_OPPONENT_ZONES:
            return bool(entity.is_public)
        return True
    
    def _get_sem_mask(self, cat_out, code_idx_out):
        """动态侦测哪些槽位是有效的，生成 Attention 掩码"""
        m = np.zeros(8, dtype=np.bool_)
        for j in range(8):
            # 【修改】如果分类非 0，或者索引非 0，说明该槽位有效
            if cat_out[j, 0] != 0 or code_idx_out[j] != 0:
                m[j] = True
        if not np.any(m): m[0] = True
        return m
    
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
                t_idxs = [t if (0 <= t < MAX_CARDS) else MAX_CARDS for t in act.macro_targets][:MAX_MATERIALS]
                t_idxs.extend([MAX_CARDS] * (MAX_MATERIALS - len(t_idxs)))
            else:
                t_idx = act.target_entity_idx if (0 <= act.target_entity_idx < MAX_CARDS) else MAX_CARDS
                t_idxs = [t_idx] + [MAX_CARDS] * (MAX_MATERIALS - 1)
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
            act_card_idxs.extend([[120]*MAX_MATERIALS] * pad_len) # 二维 Padding
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
        global_vec = self._encode_global_vector(g, player_id)
        
        # 核心优化：直接预分配全量固定形状的 NumPy 数组，天然自带 Padding
        card_indices = np.full(MAX_CARDS, PAD_CODE_IDX, dtype=np.int64)
        card_overlay_indices = np.full(MAX_CARDS, PAD_CODE_IDX, dtype=np.int64)
        card_races = np.zeros(MAX_CARDS, dtype=np.int64)
        card_attrs = np.zeros(MAX_CARDS, dtype=np.int64)
        card_setcodes = np.zeros((MAX_CARDS, 4), dtype=np.int64)
        card_feats = np.zeros((MAX_CARDS, 66), dtype=np.float32)
        masks = np.zeros(MAX_CARDS, dtype=np.bool_)

        # 语义大矩阵全量预分配，消灭碎片
        sem_cats = np.zeros((MAX_CARDS, 8, 8), dtype=np.int16)
        sem_reqs = np.full((MAX_CARDS, 8, 16), -1, dtype=np.int8)
        sem_scs = np.zeros((MAX_CARDS, 8, 4), dtype=np.int16)
        sem_nums = np.zeros((MAX_CARDS, 8, 4), dtype=np.float16)
        sem_refs = np.zeros((MAX_CARDS, 8, 4), dtype=np.int32)
        sem_races = np.zeros((MAX_CARDS, 8, 4), dtype=np.int16)
        sem_attrs = np.zeros((MAX_CARDS, 8, 4), dtype=np.int16)
        sem_code_idx = np.zeros((MAX_CARDS, 8), dtype=np.int32)
        sem_mask = np.zeros((MAX_CARDS, 8), dtype=np.bool_)
        sem_mask[:, 0] = True  # 兜底：默认第一个语义槽位永远有效，防止全空 NaN 崩溃

        # ==========================================
        # 1. 处理场上/手牌/墓地实体 
        # ==========================================
        op_known = []
        if hasattr(snapshot, 'known_hand_codes'):
            op_known = snapshot.known_hand_codes[1 - player_id].copy()
            hidden_capacity = 0
            for e in snapshot.entities:
                if e.owner != player_id:
                    if e.location == Zone.HAND: hidden_capacity += 1
                    elif e.location in [Zone.MZONE, Zone.SZONE] and not (e.position & 0x1 or e.position & 0x4):
                        hidden_capacity += 1
                        
            while len(op_known) > hidden_capacity and len(op_known) > 0:
                op_known.pop(0)

        for i, e in enumerate(snapshot.entities[:MAX_CARDS]):
            is_visible = self._is_entity_visible_to_player(e, player_id)
            is_tracked_by_memory = False
            visible_code = e.code

            if not is_visible and len(op_known) > 0:
                if e.location == Zone.HAND or (e.location in [Zone.MZONE, Zone.SZONE] and not is_visible):
                    visible_code = op_known.pop(0)
                    is_visible = True
                    is_tracked_by_memory = True

            if is_visible:
                card_indices[i] = self._hash_code(visible_code)
                pos_x, pos_y = self._get_coords(player_id, e.owner, e.location, e.sequence)

                mask = getattr(e, 'used_effect_mask', 0)  # 使用 getattr 安全获取实体属性
                
                used_eff_0 = 1.0 if (mask & (1 << 0)) else 0.0
                used_eff_1 = 1.0 if (mask & (1 << 1)) else 0.0
                used_eff_2 = 1.0 if (mask & (1 << 2)) else 0.0
                used_eff_3 = 1.0 if (mask & (1 << 3)) else 0.0
                used_eff_4 = 1.0 if (mask & (1 << 4)) else 0.0
                used_eff_5 = 1.0 if (mask & (1 << 5)) else 0.0
                used_eff_6 = 1.0 if (mask & (1 << 6)) else 0.0
                used_eff_7 = 1.0 if (mask & (1 << 7)) else 0.0

                feat_numeric = [
                    1.0 if e.owner == player_id else -1.0, e.location / 100.0, e.sequence / 10.0,
                    e.current_atk / 4000.0, e.current_def / 4000.0, e.base_atk / 4000.0, e.base_def / 4000.0,
                    pos_x, pos_y, e.level / 12.0, e.lscale / 13.0, e.rscale / 13.0, e.position / 10.0,
                    1.0 if e.is_public else (0.5 if is_tracked_by_memory else 0.0),
                    min(e.overlay_count / 5.0, 1.0), min(e.counter_count / 10.0, 1.0), 1.0 if e.is_equipped else 0.0,
                    used_eff_0, used_eff_1, used_eff_2, used_eff_3, used_eff_4, used_eff_5, used_eff_6, used_eff_7
                ]
                feat = feat_numeric + [1.0 if (e.type_mask & (1<<idx)) else 0.0 for idx in range(32)] + [1.0 if (e.link_marker & (1<<idx)) else 0.0 for idx in range(9)]
                card_feats[i] = feat
                
                card_races[i] = e.race % 30
                card_attrs[i] = e.attribute % 10

                raw_sc = e.setcodes if isinstance(e.setcodes, (list, tuple)) else [e.setcodes]
                card_setcodes[i] = [(s % 4096) for s in (list(raw_sc) + [0]*4)[:4]]
                masks[i] = True
                card_overlay_indices[i] = self._hash_code(getattr(e, 'top_overlay_code', 0))

                # 写入预分配矩阵对应切片
                cat_out, req_out, set_out, num_out, ref_out, race_out, attr_out, code_out = self.sem_kb.get_card_semantics(visible_code)
            else:
                if e.location == Zone.HAND:
                    card_indices[i] = 2 
                elif e.location == Zone.SZONE:
                    card_indices[i] = 3
                else:
                    card_indices[i] = UNK_CODE_IDX
                card_overlay_indices[i] = PAD_CODE_IDX
                masks[i] = True
                card_feats[i, :5] = [-1.0, e.location / 100.0, e.sequence / 10.0, -1.0, -1.0]
                cat_out, req_out, set_out, num_out, ref_out, race_out, attr_out, code_out = self.sem_kb.get_card_semantics(0)

            sem_cats[i] = cat_out; sem_reqs[i] = req_out; sem_scs[i] = set_out
            sem_nums[i] = num_out; sem_refs[i] = ref_out; sem_races[i] = race_out; sem_attrs[i] = attr_out
            sem_code_idx[i] = code_out
            sem_mask[i] = self._get_sem_mask(cat_out, code_out)

        # ==========================================
        # 2. 处理上帝视角卡组残像 (MAX_DECK_CARDS = 75)
        # ==========================================
        MAX_DECK_CARDS = 75
        my_deck = (snapshot.p0_deck_codes + snapshot.p0_extra_codes) if player_id == 0 else (snapshot.p1_deck_codes + snapshot.p1_extra_codes)

        deck_idx = np.full(MAX_DECK_CARDS, PAD_CODE_IDX, dtype=np.int64)
        deck_race = np.zeros(MAX_DECK_CARDS, dtype=np.int64)
        deck_attr = np.zeros(MAX_DECK_CARDS, dtype=np.int64)
        deck_setcodes = np.zeros((MAX_DECK_CARDS, 4), dtype=np.int64)
        deck_masks = np.zeros(MAX_DECK_CARDS, dtype=np.bool_)

        d_sem_cats = np.zeros((MAX_DECK_CARDS, 8, 8), dtype=np.int16)
        d_sem_reqs = np.full((MAX_DECK_CARDS, 8, 16), -1, dtype=np.int8)
        d_sem_scs = np.zeros((MAX_DECK_CARDS, 8, 4), dtype=np.int16)
        d_sem_nums = np.zeros((MAX_DECK_CARDS, 8, 4), dtype=np.float16)
        d_sem_refs = np.zeros((MAX_DECK_CARDS, 8, 4), dtype=np.int32)
        d_sem_races = np.zeros((MAX_DECK_CARDS, 8, 4), dtype=np.int16)
        d_sem_attrs = np.zeros((MAX_DECK_CARDS, 8, 4), dtype=np.int16)
        d_sem_code_idx = np.zeros((MAX_DECK_CARDS, 8), dtype=np.int32)
        d_sem_mask = np.zeros((MAX_DECK_CARDS, 8), dtype=np.bool_)
        d_sem_mask[:, 0] = True  # 兜底：默认第一个语义槽位永远有效，防止全空 NaN 崩溃

        from card_reader import card_db
        for i, code in enumerate(my_deck[:MAX_DECK_CARDS]):
            try:
                stats = card_db.get_full_stats(code)
                deck_race[i] = stats[1] % 30
                deck_attr[i] = stats[2] % 10
                raw_dsc = stats[10] if isinstance(stats[10], (list, tuple)) else [stats[10]]
                deck_setcodes[i] = [(s % 4096) for s in (list(raw_dsc) + [0]*4)[:4]]
            except Exception:
                pass
                
            deck_idx[i] = self._hash_code(code)
            deck_masks[i] = True
            
            dc_out, dr_out, ds_out, dn_out, dref_out, drace_out, dattr_out, dcode_out = self.sem_kb.get_card_semantics(code)
            d_sem_cats[i] = dc_out; d_sem_reqs[i] = dr_out; d_sem_scs[i] = ds_out
            d_sem_nums[i] = dn_out; d_sem_refs[i] = dref_out; d_sem_races[i] = drace_out; d_sem_attrs[i] = dattr_out
            d_sem_code_idx[i] = dcode_out
            d_sem_mask[i] = self._get_sem_mask(dc_out, dcode_out)

        # ==========================================
        # 2.5 处理连锁堆栈 (MAX_CHAIN = 12)
        # ==========================================
        MAX_CHAIN = 12
        c_masks = np.zeros(MAX_CHAIN, dtype=np.bool_)
        c_sem_cats = np.zeros((MAX_CHAIN, 8, 8), dtype=np.int16)
        c_sem_reqs = np.full((MAX_CHAIN, 8, 16), -1, dtype=np.int8)
        c_sem_scs = np.zeros((MAX_CHAIN, 8, 4), dtype=np.int16)
        c_sem_nums = np.zeros((MAX_CHAIN, 8, 4), dtype=np.float16)
        c_sem_refs = np.zeros((MAX_CHAIN, 8, 4), dtype=np.int32)
        c_sem_races = np.zeros((MAX_CHAIN, 8, 4), dtype=np.int16)
        c_sem_attrs = np.zeros((MAX_CHAIN, 8, 4), dtype=np.int16)
        c_sem_code_idx = np.zeros((MAX_CHAIN, 8), dtype=np.int32)
        c_sem_mask = np.zeros((MAX_CHAIN, 8), dtype=np.bool_)
        c_sem_mask[:, 0] = True  # 兜底：默认第一个语义槽位永远有效，防止全空 NaN 崩溃
        if hasattr(snapshot, 'chain_stack'):
            for i, item in enumerate(snapshot.chain_stack[:MAX_CHAIN]):
                cc_out, cr_out, cs_out, cn_out, cref_out, crace_out, cattr_out, ccode_out = self.sem_kb.get_card_semantics(item['code'])
                c_sem_cats[i] = cc_out; c_sem_reqs[i] = cr_out; c_sem_scs[i] = cs_out
                c_sem_nums[i] = cn_out; c_sem_refs[i] = cref_out; c_sem_races[i] = crace_out; c_sem_attrs[i] = cattr_out
                c_sem_code_idx[i] = ccode_out
                c_sem_mask[i] = self._get_sem_mask(cc_out, ccode_out)
                c_masks[i] = True

        # ==========================================
        # 2.6 处理动作历史雷达 (MAX_HISTORY = 8)
        # ==========================================
        MAX_HISTORY = 8
        h_masks = np.zeros(MAX_HISTORY, dtype=np.bool_)
        h_sem_cats = np.zeros((MAX_HISTORY, 8, 8), dtype=np.int16)
        h_sem_reqs = np.full((MAX_HISTORY, 8, 16), -1, dtype=np.int8)
        h_sem_scs = np.zeros((MAX_HISTORY, 8, 4), dtype=np.int16)
        h_sem_nums = np.zeros((MAX_HISTORY, 8, 4), dtype=np.float16)
        h_sem_refs = np.zeros((MAX_HISTORY, 8, 4), dtype=np.int32)
        h_sem_races = np.zeros((MAX_HISTORY, 8, 4), dtype=np.int16)
        h_sem_attrs = np.zeros((MAX_HISTORY, 8, 4), dtype=np.int16)
        h_sem_code_idx = np.zeros((MAX_HISTORY, 8), dtype=np.int32)
        h_sem_mask = np.zeros((MAX_HISTORY, 8), dtype=np.bool_)
        h_sem_mask[:, 0] = True  # 兜底：默认第一个语义槽位永远有效，防止全空 NaN 崩溃

        if hasattr(snapshot, 'history_stack'):
            for i, item in enumerate(snapshot.history_stack[:MAX_HISTORY]):
                hc_out, hr_out, hs_out, hn_out, href_out, hrace_out, hattr_out, hcode_out = self.sem_kb.get_card_semantics(item['code'])
                h_sem_cats[i] = hc_out; h_sem_reqs[i] = hr_out; h_sem_scs[i] = hs_out
                h_sem_nums[i] = hn_out; h_sem_refs[i] = href_out; h_sem_races[i] = hrace_out; h_sem_attrs[i] = hattr_out
                h_sem_code_idx[i] = hcode_out
                h_sem_mask[i] = self._get_sem_mask(hc_out, hcode_out)
                h_masks[i] = True

        # ==========================================
        # 3. 最终打包 (直接包装为 Tensor)
        # ==========================================
        act_dict = self.encode_actions(snapshot.valid_actions, snapshot)

        # 哨兵雷达：在强转和 clip 之前，进行深度数值自检，绝不静默隐藏 Bug
        has_nan = np.isnan(card_feats).any()
        has_inf = np.isinf(card_feats).any()
        has_pos_extreme = (card_feats > 65500.0).any()
        has_neg_extreme = (card_feats < -65500.0).any() # 捕获异常负向数值（Underflow）

        if has_nan or has_inf or has_pos_extreme or has_neg_extreme:
            print("\n🛑 [FeatureEncoder 核心警报] 决斗管线中惊现破坏性投毒特征数据！已自动动态拦截排毒！")
            print("   -> 🔍 毒素成因: " + 
                  ("【NaN 空值】 " if has_nan else "") + 
                  ("【Inf 无穷大】 " if has_inf else "") + 
                  ("【正向超界 Max限】 " if has_pos_extreme else "") + 
                  ("【负向超界 Min限】" if has_neg_extreme else ""))
            
            # 开启全息追踪，遍历当前盘面实体，精准揪出下毒的卡片
            from card_reader import card_db
            found_culprit = False
            for idx, e in enumerate(snapshot.entities[:MAX_CARDS]):
                # 采用 abs() 绝对值判定，同时将正向膨胀与负向脏内存（如未初始化内存里的负数）一网打尽
                if abs(e.current_atk) > 260000000 or abs(e.current_def) > 260000000 or abs(e.base_atk) > 260000000 or abs(e.base_def) > 260000000 or e.code < 0:
                    found_culprit = True
                    c_name = "未知卡片"
                    try: c_name = card_db.get_card_name(e.code)
                    except Exception as ex:
                        print(f"   -> ⚠️ 追踪异常: 无法查询卡片代码 {e.code} 的名称，可能是非法代码或数据库未收录 | 错误详情: {ex}")
                    print(f"   ├─🎯 涉案实体索引: [{idx}] | 区域: {Zone.get_str(e.location)} | 槽位序号: {e.sequence}")
                    print(f"   ├─🃏 涉案卡片身份: 【{c_name}】 (真实卡密 Code: {e.code}) | 实际拥有者: 玩家 {e.owner}")
                    print(f"   └─📊 崩溃现场面板: 当前ATK={e.current_atk} | 当前DEF={e.current_def} | 原始ATK={e.base_atk} | 原始DEF={e.base_def}")
            
            if not found_culprit:
                print(f"   -> 💡 提示: 异常未源于可见怪兽面板，可能由于全局常数计算或隐藏特征通道越界（特征矩阵极值跨度: {np.min(card_feats)} ~ {np.max(card_feats)}）")
            print("   -> 🛡️ 安全状态: 雷达已强制重置该高维特征至 float16 物理极限安全带宽，对局继续，主进程 GPU 运算图保持绝对纯净。\n")

        # 有多大拉多大：利用 float16 临界区安全上限进行动态截断，吸收极值，阻止管道崩溃
        card_feats = np.clip(card_feats, -65500.0, 65500.0)
        
        base_dict = {
            'global': torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0),
            
            'card_idx': torch.from_numpy(card_indices).unsqueeze(0),
            'card_overlay_idx': torch.from_numpy(card_overlay_indices).unsqueeze(0),
            'card_race': torch.from_numpy(card_races).unsqueeze(0), 
            'card_attr': torch.from_numpy(card_attrs).unsqueeze(0), 
            'card_setcodes': torch.from_numpy(card_setcodes).unsqueeze(0), 
            'card_feats': torch.from_numpy(card_feats).unsqueeze(0),
            'padding_mask': torch.from_numpy(masks).unsqueeze(0),
            
            'sem_category': torch.from_numpy(sem_cats).unsqueeze(0),
            'sem_req': torch.from_numpy(sem_reqs).unsqueeze(0),
            'sem_setcode': torch.from_numpy(sem_scs).unsqueeze(0),
            'sem_number': torch.from_numpy(sem_nums).unsqueeze(0),
            'sem_ref': torch.from_numpy(sem_refs).unsqueeze(0),
            'sem_race': torch.from_numpy(sem_races).unsqueeze(0),
            'sem_attr': torch.from_numpy(sem_attrs).unsqueeze(0),
            'sem_code_idx': torch.from_numpy(sem_code_idx).unsqueeze(0),
            'sem_mask': torch.from_numpy(sem_mask).unsqueeze(0),
            
            'deck_idx': torch.from_numpy(deck_idx).unsqueeze(0),
            'deck_race': torch.from_numpy(deck_race).unsqueeze(0),
            'deck_attr': torch.from_numpy(deck_attr).unsqueeze(0),
            'deck_setcodes': torch.from_numpy(deck_setcodes).unsqueeze(0),
            'deck_mask': torch.from_numpy(deck_masks).unsqueeze(0),
            
            'd_sem_category': torch.from_numpy(d_sem_cats).unsqueeze(0),
            'd_sem_req': torch.from_numpy(d_sem_reqs).unsqueeze(0),
            'd_sem_setcode': torch.from_numpy(d_sem_scs).unsqueeze(0),
            'd_sem_number': torch.from_numpy(d_sem_nums).unsqueeze(0),
            'd_sem_ref': torch.from_numpy(d_sem_refs).unsqueeze(0),
            'd_sem_race': torch.from_numpy(d_sem_races).unsqueeze(0),
            'd_sem_attr': torch.from_numpy(d_sem_attrs).unsqueeze(0),
            'd_sem_code_idx': torch.from_numpy(d_sem_code_idx).unsqueeze(0),
            'd_sem_mask': torch.from_numpy(d_sem_mask).unsqueeze(0),

            'c_mask': torch.from_numpy(c_masks).unsqueeze(0),
            'c_sem_category': torch.from_numpy(c_sem_cats).unsqueeze(0),
            'c_sem_req': torch.from_numpy(c_sem_reqs).unsqueeze(0),
            'c_sem_setcode': torch.from_numpy(c_sem_scs).unsqueeze(0),
            'c_sem_number': torch.from_numpy(c_sem_nums).unsqueeze(0),
            'c_sem_ref': torch.from_numpy(c_sem_refs).unsqueeze(0),
            'c_sem_race': torch.from_numpy(c_sem_races).unsqueeze(0),
            'c_sem_attr': torch.from_numpy(c_sem_attrs).unsqueeze(0),
            'c_sem_code_idx': torch.from_numpy(c_sem_code_idx).unsqueeze(0),
            'c_sem_mask': torch.from_numpy(c_sem_mask).unsqueeze(0),

            'h_mask': torch.from_numpy(h_masks).unsqueeze(0),
            'h_sem_category': torch.from_numpy(h_sem_cats).unsqueeze(0),
            'h_sem_req': torch.from_numpy(h_sem_reqs).unsqueeze(0),
            'h_sem_setcode': torch.from_numpy(h_sem_scs).unsqueeze(0),
            'h_sem_number': torch.from_numpy(h_sem_nums).unsqueeze(0),
            'h_sem_ref': torch.from_numpy(h_sem_refs).unsqueeze(0),
            'h_sem_race': torch.from_numpy(h_sem_races).unsqueeze(0),
            'h_sem_attr': torch.from_numpy(h_sem_attrs).unsqueeze(0),
            'h_sem_code_idx': torch.from_numpy(h_sem_code_idx).unsqueeze(0),
            'h_sem_mask': torch.from_numpy(h_sem_mask).unsqueeze(0),
        }
        
        base_dict.update(act_dict)
        return base_dict

if __name__ == "__main__":
    enc = GalateaEncoder()
    print("Encoder (with Semantic Active) Ready.")
