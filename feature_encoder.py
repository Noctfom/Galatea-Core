# ==================================================================================
#  Galatea Feature Encoder (特征编码器 V2.0 - Hash Stable)
# ==================================================================================
#  功能：将 GameSnapshot 转换为 PyTorch Tensor
#  改进：
#  1. 移除动态字典，改用 Hash 映射，确保多进程/重启后 ID 一致性。
#  2. 增强了特征归一化的健壮性。
# ==================================================================================

import torch
import numpy as np
from data_types import GameSnapshot
from game_constants import Zone

# --- 配置参数 ---
MAX_CARDS = 100          # 序列最大长度
VOCAB_SIZE = 20000       # 词表大小 (必须与 main.py 中的 vocab_size 一致)
UNK_CODE_IDX = 1         # 未知/特殊 Token
PAD_CODE_IDX = 0         # Padding Token

MAX_ACTIONS = 80  # 单个时点下最大动作数

class GalateaEncoder:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        # 预留前 10 个 ID 给特殊用途 (0=Pad, 1=Unk, 2=Covered...)
        self.reserved_ids = 10 
        
        # Global: Turn, Phase, ToPlay, MyLP, OpLP, + 10个区域统计 = 15
        self.global_dim = 15
        # Card Feat: Owner, Loc, Seq, Atk, Def, Level, Disabled = 7
        self.card_feat_dim = 7

    def _hash_code(self, code):
        """
        [关键修复] 静态哈希映射
        保证无论哪个进程、无论何时运行，同一个 code 永远映射到同一个 idx
        """
        if code == 0: return UNK_CODE_IDX # 0 通常是未知或掩盖的卡
        # 简单取模哈希，避开预留位
        return (code % (self.vocab_size - self.reserved_ids)) + self.reserved_ids
    
    def encode_actions(self, valid_actions, snapshot):
        """
        将动作列表转换为 Tensor 矩阵
        """
        act_card_idxs = [] # 动作指向哪张卡 (在 entities 里的下标)
        act_types = []     # 动作类型 (召唤/攻击/发动...)
        act_descs = []     # 效果 ID (Hash后)
        masks = []         # 有效位标记

        # 遍历前 MAX_ACTIONS 个动作
        for act in valid_actions[:MAX_ACTIONS]:
            # 1. 目标卡片索引
            # 如果动作不指向特定卡(如进战阶)，我们指向 Index 0 (Padding位/全局位)
            # act.target_entity_idx 是在 get_snapshot 里计算好的 entities 下标
            t_idx = act.target_entity_idx if act.target_entity_idx >= 0 else 0
            act_card_idxs.append(t_idx)
            
            # 2. 动作类型
            act_types.append(act.action_type)
            
            # 3. 描述 ID (简单 Hash 到 1024 以内)
            # 这里的 desc_id 就是你在 data_types.py 里新加的字段
            desc_hash = act.desc_id % 1024
            act_descs.append(desc_hash)
            
            masks.append(True)
            
        # Padding 补齐 (补 0)
        pad_len = MAX_ACTIONS - len(act_card_idxs)
        if pad_len > 0:
            act_card_idxs.extend([0] * pad_len)
            act_types.extend([0] * pad_len)
            act_descs.extend([0] * pad_len)
            masks.extend([False] * pad_len)
            
        return {
            'act_card_idx': torch.tensor(act_card_idxs, dtype=torch.long).unsqueeze(0),
            'act_type': torch.tensor(act_types, dtype=torch.long).unsqueeze(0),
            'act_desc': torch.tensor(act_descs, dtype=torch.long).unsqueeze(0),
            'act_mask': torch.tensor(masks, dtype=torch.bool).unsqueeze(0)
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
        
        card_indices = []
        card_feats = []
        card_races = []  # 种族
        card_attrs = []  # 属性
        card_setcodes = [] # 字段列表
        masks = []
        
        for e in snapshot.entities[:MAX_CARDS]:
            is_visible = True
            if e.owner != player_id:
                if e.location in [Zone.HAND, Zone.DECK]:
                    if not e.is_public: is_visible = False
                if e.location in [Zone.MZONE, Zone.SZONE] and (e.position & 0xA):
                     if not e.is_public: is_visible = False
            
            if is_visible:
                c_idx = self._hash_code(e.code)
                # 1. 基础数值特征 (12维)
                feat_numeric = [
                    1.0 if e.owner == player_id else -1.0, 
                    e.location / 100.0, e.sequence / 10.0,
                    e.current_atk / 4000.0, e.current_def / 4000.0,
                    e.base_atk / 4000.0, e.base_def / 4000.0,
                    e.level / 12.0, e.lscale / 13.0, e.rscale / 13.0,
                    e.position / 10.0, 1.0 if e.is_public else 0.0
                ]
                
                # 2. 类型展开 (32维) - 完美解析魔法/陷阱/各种怪兽
                type_bits = [1.0 if (e.type_mask & (1 << i)) else 0.0 for i in range(32)]
                
                # 3. 连接箭头展开 (9维) - 左下、下、右下、左、右...
                link_bits = [1.0 if (e.link_marker & (1 << i)) else 0.0 for i in range(9)]
                
                feat = feat_numeric + type_bits + link_bits # 总长 12+32+9 = 53维
                
                # Hash 种族(不超过30种) 和 属性(不超过10种)
                r_idx = e.race % 30
                a_idx = e.attribute % 10

                # 提取并补齐字段到固定的 4 个槽位，哈希映射到 4096 以内
                sc = list(e.setcodes)
                sc = (sc + [0]*4)[:4]
                sc_hashed = [s % 4096 for s in sc]

            else:
                c_idx = UNK_CODE_IDX 
                feat = [-1.0, e.location / 100.0, e.sequence / 10.0] + [0.0] * 50
                r_idx = 0
                a_idx = 0
                sc_hashed = [0, 0, 0, 0] # 未知卡的字段全为 0
            
            card_indices.append(c_idx)
            card_feats.append(feat)
            card_races.append(r_idx)
            card_attrs.append(a_idx)
            card_setcodes.append(sc_hashed)
            masks.append(1.0)

        # Padding 补齐
        pad_len = MAX_CARDS - len(card_indices)
        if pad_len > 0:
            card_indices.extend([PAD_CODE_IDX] * pad_len)
            card_races.extend([0] * pad_len)
            card_attrs.extend([0] * pad_len)
            masks.extend([0.0] * pad_len)
            for _ in range(pad_len):
                card_setcodes.append([0, 0, 0, 0])
                card_feats.append([0.0] * 53)

        act_dict = self.encode_actions(snapshot.valid_actions, snapshot)
        
        base_dict = {
            'global': torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0),
            'card_idx': torch.tensor(card_indices, dtype=torch.long).unsqueeze(0),
            'card_race': torch.tensor(card_races, dtype=torch.long).unsqueeze(0), # [新增]
            'card_attr': torch.tensor(card_attrs, dtype=torch.long).unsqueeze(0), # [新增]
            'card_setcodes': torch.tensor(card_setcodes, dtype=torch.long).unsqueeze(0), # 🌟 [新增] shape: [1, Seq, 4]
            'card_feats': torch.tensor(card_feats, dtype=torch.float32).unsqueeze(0),
            'padding_mask': torch.tensor(masks, dtype=torch.bool).unsqueeze(0)
        }
        base_dict.update(act_dict)
        return base_dict

if __name__ == "__main__":
    enc = GalateaEncoder()
    print(f"Encoder Ready. Hash Check: Code 3001 -> {enc._hash_code(3001)}")