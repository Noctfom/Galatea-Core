#决斗录像模块

import json
import datetime
import os
from card_reader import card_db
from game_constants import Phases

class AIThoughtLogger:
    def __init__(self, player_name="Galatea_AI"):
        self.player_name = player_name
        self.thoughts = []
        self.is_active = False # 控制当前局是否需要记录
        
    def start_recording(self):
        """开启一局新的录制"""
        self.thoughts = []
        self.is_active = True
        
    def log_decision(self, turn, phase_id, snapshot, probs, chosen_index):
        """记录单步决策"""
        if not self.is_active: return
        
        step_log = {"turn": turn, "phase": Phases.get_str(phase_id), "options": []}
        
        for i, act in enumerate(snapshot.valid_actions):
            # 翻译动作描述
            desc = act.desc_str if act.desc_str else f"Type={act.action_type}"
            target_info = ""
            if act.target_entity_idx >= 0 and act.target_entity_idx < len(snapshot.entities):
                t_card = snapshot.entities[act.target_entity_idx]
                target_info = f" -> [{card_db.get_card_name(t_card.code)}]"
            
            # 提取概率值 (安全转换 tensor 为 float)
            prob_val = float(probs[i].item()) if hasattr(probs[i], 'item') else float(probs[i])
            
            step_log["options"].append({
                "index": i,
                "desc": f"{desc}{target_info}",
                "confidence": prob_val,
                "is_chosen": (i == chosen_index)
            })
            
        # 按信心值从高到低排序，一目了然
        step_log["options"].sort(key=lambda x: x["confidence"], reverse=True)
        self.thoughts.append(step_log)
        
    def save(self, winner_id, game_idx):
        """保存为 JSON"""
        if not self.is_active or not self.thoughts: return None
        
        os.makedirs("./ai_thoughts", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"./ai_thoughts/Game{game_idx}_{timestamp}_P{winner_id}Win.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.player_name,
                "winner": winner_id,
                "decisions": self.thoughts
            }, f, ensure_ascii=False, indent=4)
            
        self.is_active = False # 保存完自动关闭，等待下一次唤醒
        return filepath