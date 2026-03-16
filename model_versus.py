# ==================================================================================
#  Galatea Model Versus (Arena Mode) - Enhanced Logging
# ==================================================================================

import os
import time
import random
import torch
import struct
import torch.nn.functional as F

import rule_bot
from galatea_env import GalateaEnv
from gamestate import MessageParser, DuelState
from ai_bot import AiBot
import deck_utils
from thought_logger import AIThoughtLogger

class ModelArena:
    # 增加 config 参数
    def __init__(self, model_p0_path, model_p1_path=None, device='cpu', deck_dir="./decks", config=None):
        self.deck_dir = deck_dir

        self.thought_freq = self.net_config.get('thought_freq', 0)
        # 初始化记录器
        p0_name = os.path.basename(model_p0_path) if model_p0_path else "P0_AI"
        self.logger = AIThoughtLogger(player_name=p0_name)
        
        # 处理设备参数
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 默认配置 (保底用)
        default_config = {
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 2,
            'vocab_size': 20000
        }
        # 如果传入了 config 就用传入的，否则用默认
        self.net_config = config if config else default_config
        
        print(f"⚙️ [Arena] 模型配置: {self.net_config}")
        
        # P0
        # 传递 config 给 GalateaNet
        # 注意：这里假设 AiBot 内部初始化 GalateaNet 时接收 config
        # 如果 AiBot.__init__ 还没改，记得去 ai_bot.py 里把 self.net = GalateaNet(config) 改好
        self.p0_bot = AiBot(device=self.device, net_config=self.net_config) 
        self.p0_bot.load_model(model_p0_path)
        self.p0_bot.net.eval()
        print(f"🤖 [AiBot] 成功加载模型权重: {model_p0_path}")

        # P1: 对手
        self.p1_bot = None
        if model_p1_path:
            self.p1_bot = AiBot(device=self.device)
            self.p1_bot.load_model(model_p1_path)
            self.p1_bot.net.eval()
            print(f"🤖 [Opponent] 成功加载模型权重: {model_p1_path}")
        else:
            print(f"🤖 [Opponent] 使用 RuleBot (内置规则脚本)")

        self.env = GalateaEnv()

    def run_duel(self, game_idx=1):
        """
        返回: (winner_index, reason_code)
        reason_code: 
           0-4: 游戏规则胜利 (投降, 0LP, 0卡组等)
           -1: AI死锁 (Retry过多)
           -2: 超时 (Steps过多)
           -3: 初始化失败
        """
        res = deck_utils.get_random_deck_pair(ydk_dir=self.deck_dir)
        if not res: return 0, -3
        d1_name, d1, d2_name, d2 = res
        
        raw_data = self.env.reset(d1, d2)
        if not raw_data: return 0, -3

        brain = DuelState()
        msg_queue = MessageParser.parse(raw_data)
        
        consecutive_retries = 0
        current_step_ignore_list = []
        last_decision_value = None

        ai_fallback_count = 0
        
        state_change_msgs = {40, 41, 50, 53, 54, 55, 56, 60, 61, 62, 70, 90, 91, 92, 94}
        interaction_msgs = {10, 11, 15, 16, 18, 19, 20, 22, 26, 130, 131, 132, 133}
        ai_managed_msgs = [10, 11, 12, 13, 14, 15, 16, 18, 19, 24] 

        steps = 0
        # 增加步数上限到 2000，防止慢速卡组被误判
        while steps < 2000: 
            if not msg_queue:
                raw_data = self.env.step()
                if not raw_data: break
                msg_queue = MessageParser.parse(raw_data)
                # 只要新来的数据包不是以 RETRY (1) 开头，说明上一回合的动作必定被引擎接受了！
                if msg_queue and msg_queue[0][0] != 1:
                    consecutive_retries = 0
                    current_step_ignore_list.clear()
                continue
            
            msg = msg_queue.pop(0)
            msg_type = msg[0]
            brain.update(msg_type, msg[1:])
            
            if msg_type == 5: # 胜利 MSG_WIN
                # msg格式通常是 [5, winner, reason]
                winner = msg[1:][0]
                reason = msg[1:][1] if len(msg[1:]) > 1 else 0
                # [新增] 比赛结束，保存这局的日记
                if self.logger.is_active:
                    saved_path = self.logger.save(winner, game_idx)
                    print(f"\n🧠 [AI 读心] 第 {game_idx} 局的心声已保存至 {saved_path}")
                return winner, reason, ai_fallback_count # 补齐 3 个返回值

            # --- Retry 处理 ---
            if msg_type == 1:
                consecutive_retries += 1
                if last_decision_value is not None:
                    current_step_ignore_list.append(last_decision_value)

                # [新增] 如果是 AI 刚刚操作完导致的 Retry，记一笔
                # 我们怎么知道是 AI？看 consecutive_retries == 1 且上一步是 AI 决策
                if consecutive_retries == 1:
                     # 简单粗暴统计：只要发生 Retry 就算一次不稳定
                     # 因为 RuleBot 理论上不应该 Retry
                     ai_fallback_count += 1
                
                # 连续 20 次非法操作 -> 判定为 AI 死锁
                if consecutive_retries > 20:
                    return -1, -1, ai_fallback_count # <--- 返回 count
                continue

            # --- 决策 ---
            if msg_type in [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 26, 130, 131, 132, 133, 140, 141, 142, 143]:
                player_to_act = 0
                if len(msg) > 1: player_to_act = msg[1]
                
                resp = None
                is_p0_turn = (player_to_act == 0)
                active_bot = self.p0_bot if is_p0_turn else self.p1_bot

                # 尝试 AI 决策
                if active_bot and msg_type in ai_managed_msgs and brain.current_valid_actions and consecutive_retries == 0:
                    try:
                        # 1. 提取当前状态快照
                        snap = brain.get_snapshot()
                        player = snap.global_data.to_play
                        
                        # 2. 调用 V3 编码器生成 53 维字典，并挂载到设备 (GPU/CPU)
                        tensor_dict = active_bot.encoder.encode(snap, player_id=player)
                        infer_dict = {k: v.to(self.device) for k, v in tensor_dict.items()}
                        
                        # 3. 神经网络前向传播
                        with torch.no_grad():
                            logits, _ = active_bot.net(infer_dict)
                            # 👑 [竞技场核心] 绝对贪婪策略：不掷骰子，永远选打分最高的操作！
                            action_idx = torch.argmax(logits, dim=-1)
                            
                        # 🌟 [新增] 截获 AI 的胜率打分并记录
                        if is_p0_turn and self.logger.is_active:
                            # 用 Softmax 把原始的 logits 分数转换为 0~1 的概率
                            probs = F.softmax(logits.squeeze(0), dim=-1)
                            self.logger.log_decision(
                                turn=brain.turn,
                                phase_id=brain.phase,
                                snapshot=snap,
                                probs=probs,
                                chosen_index=action_idx.item()
                            )

                        # 4. 取出动作
                        sel_idx = action_idx.item()
                        if sel_idx < len(snap.valid_actions):
                            chosen = snap.valid_actions[sel_idx]
                        else:
                            chosen = random.choice(snap.valid_actions)
                            
                        # 5. 翻译动作为 YGOPro 底层 Bytes
                        resp = b''
                        
                        if msg_type == 15:
                            val = chosen.index
                            if val == -1: # AI 决定 Cancel
                                resp = struct.pack('<i', -1)
                            else:
                                # msg[3] 就是真正的 Min (第0个是Type, 第1个是P, 第2个是Cancelable)
                                min_c = msg[3] if len(msg) >= 4 else 1
                                count = max(1, min_c)
                                
                                available_indices = [a.index for a in snap.valid_actions if a.index != -1 and a.index != val]
                                selected = [val] + available_indices[:count-1]
                                
                                resp_buf = bytearray([len(selected)]) # 第一位必须是 Count！
                                for i in selected: resp_buf.append(i) # 后面跟着索引
                                resp = bytes(resp_buf)
                                
                        elif msg_type in [10, 11, 16]:
                            resp = active_bot._pack_response(chosen, msg_type=msg_type)
                        else:
                            val = chosen.index
                            if msg_type in [18, 24]:
                                zone_id = val
                                p = 0; l = 0x04; s = 0
                                if zone_id & 16: p = 1
                                if zone_id & 8:  l = 0x08
                                s = zone_id & 0x7
                                # 直接用 msg[1] 读取 PlayerID
                                req_p = msg[1] if len(msg) > 1 else 0
                                raw_p = req_p if p == 0 else (1 - req_p)
                                final_p = 1 if raw_p == 1 else 0
                                resp = bytes([final_p, l, s])
                            else:
                                resp = bytes([val])
                                
                        if resp:
                            last_decision_value = resp
                    except Exception as e:
                        # 如果出现未适配的异常，无缝切给 RuleBot
                        # print(f"[Arena Warning] AI 推理回退: {e}")
                        resp = None

                # RuleBot 兜底
                if resp is None:
                    # 🌟 [关键修复] 如果刚刚是 AI 翻车了(导致了第1次Retry)，
                    # 我们必须清空 Ignore List！防止 AI 的错误格式误伤 RuleBot，
                    # 导致 RuleBot 被禁止选择正确的逻辑选项！
                    if consecutive_retries == 1:
                        current_step_ignore_list.clear()

                    clean_ignore = []
                    for val in current_step_ignore_list:
                        clean_ignore.append(val)
                        if isinstance(val, bytes):
                            if len(val)>=4: clean_ignore.append(struct.unpack('<I', val[:4])[0])
                            elif len(val)>=1: clean_ignore.append(val[0])
                    
                   # 直接传 msg
                    resp = rule_bot.get_rule_decision(player_to_act, msg_type, msg, brain, clean_ignore)
                    last_decision_value = resp

                self.env.send_action(resp)
                msg_queue = []
                steps += 1
        
        return -1, -2, ai_fallback_count # 超时

    def run_tournament(self, n_games=10):
        print(f"🚀 开始 {n_games} 场对决...")
        p0_wins = 0
        p1_wins = 0
        draws = 0
        
        # [新增] 统计 AI 掉线/回退次数
        total_ai_fallbacks = 0

        # 统计详细原因
        reasons = {
            'Surrender': 0, 
            'LP_0': 0, 
            'Deck_0': 0, 
            'TimeLimit': 0,
            'Deadlock': 0,
            'StepsOut': 0
        }
        
        for i in range(n_games,n_games=10):
            # [新增] 如果开启了记录，并且到达了指定的间隔局数，唤醒 Logger
            if self.thought_freq > 0 and (i + 1) % self.thought_freq == 0:
                self.logger.start_recording()

            # [修改] 把局数 i+1 传进去
            w, r, fallback_cnt = self.run_duel(game_idx=i+1)

            total_ai_fallbacks += fallback_cnt
            
            reason_str = "Unknown"
            is_abnormal = False
            
            # 解析原因代码 (根据 YGOPro 核心定义)
            if r == 0: reason_str = "Surrender"; reasons['Surrender'] += 1
            elif r == 1: reason_str = "LP -> 0"; reasons['LP_0'] += 1
            elif r == 2: reason_str = "Deck -> 0"; reasons['Deck_0'] += 1
            elif r == 3: reason_str = "Time Limit"; reasons['TimeLimit'] += 1
            elif r == -1: reason_str = "❌ AI Deadlock"; is_abnormal = True; reasons['Deadlock'] += 1
            elif r == -2: reason_str = "⌛ Steps Limit"; is_abnormal = True; reasons['StepsOut'] += 1
            elif r == -3: reason_str = "⚠️ Init Fail"; is_abnormal = True
            
            if w == 0: p0_wins += 1
            elif w == 1: p1_wins += 1
            else: draws += 1
            
            # 格式化输出
            score_str = f"Score: {p0_wins}-{p1_wins}"
            if is_abnormal:
                print(f"⚠️ Game {i+1}: Aborted ({reason_str}) | {score_str}")
            else:
                # 正常局只在最后或者每10局打印一次，避免刷屏，或者像你之前一样用 \r
                print(f"   Game {i+1}: Winner P{w} ({reason_str}) | {score_str}", end="\r")

            # 打印时带上 Fallback 信息
            fallback_info = f" | ⚠️ AI Fallbacks: {fallback_cnt}" if fallback_cnt > 0 else ""
            if is_abnormal:
                print(f"⚠️ Game {i+1}: Aborted ({reason_str}) | {score_str}{fallback_info}")
            else:
                print(f"   Game {i+1}: Winner P{w} ({reason_str}) | {score_str}{fallback_info}", end="\r")
        
        print(f"\n\n🏆 最终比分: AI(P0) {p0_wins} : {p1_wins} RuleBot(P1)")
        print("📊 胜负原因统计:")
        for k, v in reasons.items():
            if v > 0: print(f"   - {k}: {v}")