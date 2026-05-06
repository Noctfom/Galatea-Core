import json
import datetime
import os
from card_reader import card_db
from game_constants import Phases

class AIThoughtLogger:
    def __init__(self, player_name="Galatea_AI"):
        self.player_name = player_name
        self.thoughts = []
        self.is_active = False
        
    def start_recording(self):
        self.thoughts = []
        self.is_active = True
        
    def log_decision(self, turn, phase_id, snapshot, probs, chosen_index):
        if not self.is_active: return
        
        try:
            def get_zone_cards(owner_id, loc_id):
                cards = []
                for c in snapshot.entities:
                    if getattr(c, 'owner', -1) == owner_id and getattr(c, 'location', -1) == loc_id:
                        code = getattr(c, 'code', 0) & 0x7FFFFFFF
                        cards.append({
                            "code": code, 
                            "seq": getattr(c, 'sequence', 0), 
                            "pos": getattr(c, 'position', 0),
                            "atk": getattr(c, 'current_atk', 0),
                            "def": getattr(c, 'current_def', 0),
                            "lvl": getattr(c, 'level', 0),
                            "counters": getattr(c, 'counter_count', 0),
                            "overlays": getattr(c, 'overlay_count', 0)
                        })
                cards.sort(key=lambda x: x["seq"])
                return cards

            # 🌟 修复 1：补齐主卡组与额外卡组的真实长度读取
            p0_deck_len = len(getattr(snapshot, 'p0_deck_codes', []))
            p1_deck_len = len(getattr(snapshot, 'p1_deck_codes', []))
            p0_extra_len = len(getattr(snapshot, 'p0_extra_codes', []))
            p1_extra_len = len(getattr(snapshot, 'p1_extra_codes', []))
            
            to_play = getattr(snapshot.global_data, 'to_play', 0)

            step_log = {
                "turn": turn, 
                "phase": Phases.get_str(phase_id),
                "state": {
                    "to_play": to_play,
                    "p0_lp": getattr(snapshot.global_data, 'my_lp', 8000),
                    "p1_lp": getattr(snapshot.global_data, 'op_lp', 8000),
                    
                    # 🌟 修复 1：将 Extra 的真实长度也写入 json
                    "p0_deck_len": p0_deck_len, "p1_deck_len": p1_deck_len,
                    "p0_extra_len": p0_extra_len, "p1_extra_len": p1_extra_len,
                    
                    "p0_hand": get_zone_cards(0, 0x02), "p0_mzone": get_zone_cards(0, 0x04),
                    "p0_szone": get_zone_cards(0, 0x08), "p0_grave": get_zone_cards(0, 0x10),
                    "p0_extra": get_zone_cards(0, 0x40), "p0_removed": get_zone_cards(0, 0x20),
                    "p1_hand": get_zone_cards(1, 0x02), "p1_mzone": get_zone_cards(1, 0x04),
                    "p1_szone": get_zone_cards(1, 0x08), "p1_grave": get_zone_cards(1, 0x10),
                    "p1_extra": get_zone_cards(1, 0x40), "p1_removed": get_zone_cards(1, 0x20),
                    "chain": getattr(snapshot, 'chain_stack', []), 
                    "history": getattr(snapshot, 'history_stack', []) 
                },
                "options": []
            }
            
            action_dict = {
                0: "通常召唤", 1: "特殊召唤/攻击", 2: "改变表示形式",
                3: "盖放怪兽", 4: "盖放魔陷", 5: "发动效果",
                6: "进入战斗阶段", 7: "结束回合", 8: "洗牌",
                15: "选择目标", 16: "选择位置", 26: "复杂选卡"
            }
            
            current_actor = None
            if hasattr(snapshot, 'chain_stack') and len(snapshot.chain_stack) > 0:
                top_chain = snapshot.chain_stack[-1]
                current_actor = {
                    "code": top_chain.get("code", 0) & 0x7FFFFFFF,
                    "owner": top_chain.get("c", -1), "loc": top_chain.get("l", -1), "seq": top_chain.get("s", -1)
                }

            for i, act in enumerate(snapshot.valid_actions):
                desc = act.desc_str if act.desc_str else action_dict.get(act.action_type, f"Type={act.action_type}")
                
                opt_actor = current_actor
                opt_target = None
                target_info = ""

                if act.target_entity_idx >= 0 and act.target_entity_idx < len(snapshot.entities):
                    t_card = snapshot.entities[act.target_entity_idx]
                    code = getattr(t_card, 'code', 0) & 0x7FFFFFFF
                    owner = getattr(t_card, 'owner', 0) 
                    loc = getattr(t_card, 'location', 0)
                    seq = getattr(t_card, 'sequence', 0)
                    
                    card_dict = {"code": code, "owner": owner, "loc": loc, "seq": seq}
                    name = card_db.get_card_name(code) if code != 0 else "未知卡片"
                    loc_str = {0x01:"卡组", 0x02:"手牌", 0x04:"怪兽区", 0x08:"魔陷区", 0x10:"墓地", 0x20:"除外区", 0x40:"额外"}.get(loc, "区域")
                    owner_str = "我方" if owner == 0 else "敌方"

                    if act.action_type in [0, 1, 3, 4, 5]: 
                        opt_actor = card_dict
                        target_info = f" -> [{owner_str}{loc_str}的 {name}]"
                    else:
                        opt_target = card_dict
                        target_info = f" -> [目标: {owner_str}{loc_str}的 {name}]"
                
                prob_val = float(probs[i].item()) if hasattr(probs[i], 'item') else float(probs[i])
                step_log["options"].append({
                    "index": i, "desc": f"{desc}{target_info}",
                    "confidence": prob_val, "is_chosen": (i == chosen_index),
                    "action_type": act.action_type,
                    "actor": opt_actor, "target": opt_target
                })
                
            step_log["options"].sort(key=lambda x: x["confidence"], reverse=True)
            self.thoughts.append(step_log)
        except Exception as e:
            print(f"\n[Logger Error] 心声记录器抛出异常: {e}")

    # 修复 4：接收竞技场传来的真实胜利原因
    def save(self, winner_id, game_idx, win_reason="正常结束"):
        if not self.is_active or not self.thoughts: return None
        self.is_active = False 
        os.makedirs("./ai_thoughts", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filepath = f"./ai_thoughts/Game{game_idx}_{timestamp}_P{winner_id}Win.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.player_name, 
                "winner": winner_id, 
                "win_reason": win_reason,
                "decisions": self.thoughts
            }, f, ensure_ascii=False, indent=4)
        return filepath