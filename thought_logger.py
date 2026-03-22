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
        """记录单步决策 (包含场面状态)"""
        if not self.is_active: return
        
        try:
            # 🌟 核心修复 3：将 controller 改为真实的属性名 owner！
            p0_hand = sum(1 for c in snapshot.entities if getattr(c, 'owner', 0) == 0 and getattr(c, 'location', 0) == 0x02)
            p0_mzone = sum(1 for c in snapshot.entities if getattr(c, 'owner', 0) == 0 and getattr(c, 'location', 0) == 0x04)
            p1_hand = sum(1 for c in snapshot.entities if getattr(c, 'owner', 1) == 1 and getattr(c, 'location', 0) == 0x02)
            p1_mzone = sum(1 for c in snapshot.entities if getattr(c, 'owner', 1) == 1 and getattr(c, 'location', 0) == 0x04)

            step_log = {
                "turn": turn, 
                "phase": Phases.get_str(phase_id),
                "state": {
                    "p0_lp": getattr(snapshot.global_data, 'my_lp', 8000),
                    "p1_lp": getattr(snapshot.global_data, 'op_lp', 8000),
                    "p0_hand": p0_hand, "p0_mzone": p0_mzone,
                    "p1_hand": p1_hand, "p1_mzone": p1_mzone
                },
                "options": []
            }
            
            action_dict = {
                0: "通常召唤", 1: "特殊召唤", 2: "改变表示形式",
                3: "盖放怪兽", 4: "盖放魔陷", 5: "发动效果",
                6: "进入战斗阶段", 7: "结束回合", 8: "洗牌",
                15: "选择卡片/目标", 16: "选择位置/区域"
            }
            
            for i, act in enumerate(snapshot.valid_actions):
                desc = act.desc_str if act.desc_str else action_dict.get(act.action_type, f"Type={act.action_type}")

                target_info = ""
                if act.target_entity_idx >= 0 and act.target_entity_idx < len(snapshot.entities):
                    t_card = snapshot.entities[act.target_entity_idx]
                    code = getattr(t_card, 'code', 0)
                    name = card_db.get_card_name(code) if code != 0 else "盖卡/未知"
                    target_info = f" -> [{name}]"
                
                prob_val = float(probs[i].item()) if hasattr(probs[i], 'item') else float(probs[i])
                
                step_log["options"].append({
                    "index": i,
                    "desc": f"{desc}{target_info}",
                    "confidence": prob_val,
                    "is_chosen": (i == chosen_index)
                })
                
            step_log["options"].sort(key=lambda x: x["confidence"], reverse=True)
            self.thoughts.append(step_log)
        except Exception as e:
            print(f"\n[Logger Error] 心声记录器抛出异常: {e}")

    def save(self, winner_id, game_idx):
        """保存为 JSON"""
        if not self.is_active: return None
        self.is_active = False 
        
        if not self.thoughts: return None
        
        os.makedirs("./ai_thoughts", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"./ai_thoughts/Game{game_idx}_{timestamp}_P{winner_id}Win.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.player_name,
                "winner": winner_id,
                "decisions": self.thoughts
            }, f, ensure_ascii=False, indent=4)
            
        return filepath