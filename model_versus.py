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

        # 1. 先处理设备
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 2. 再配置网络参数
        default_config = {
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 2,
            'vocab_size': 20000
        }
        self.net_config = config if config else default_config
        
        # 3. 这时候再去取 thought_freq 就绝对安全了！
        self.thought_freq = self.net_config.get('thought_freq', 0)
        
        # 4. 初始化记录器
        p0_name = os.path.basename(model_p0_path) if model_p0_path else "P0_AI"
        self.logger = AIThoughtLogger(player_name=p0_name)
        
        print(f"⚙️ [Arena] 模型配置: {self.net_config}")
        
        # P0
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

        last_valid_hash = ""
        action_history = []

        # 🌟 [新增] 合法死锁断路器
        loop_tracker = {}
        banned_actions_for_state = {}
        
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

            # 🌟 [新增] 阶段切换时，场面刷新，清空拉黑记录
            if msg_type in [40, 41]:
                loop_tracker.clear()
                banned_actions_for_state.clear()
            
            if msg_type == 5: # 胜利 MSG_WIN
                # msg格式通常是 [5, winner, reason]
                winner = msg[1:][0]
                reason = msg[1:][1] if len(msg[1:]) > 1 else 0
                # 🚨 [黑匣子] 抓取异常胜利代码
                if reason not in [0, 1, 2, 3, 4]:
                    print(f"\n🚨 [黑匣子触发] 捕获异常 WIN_REASON: {reason} | 赢家: P{winner}")
                    print(f"   -> 崩溃前的最近 5 次底层动作记录: {action_history}")
                    print(f"   -> 原始封包 MSG 字节内容: {msg}")
                    # 如果有记录器，保留它以供事后尸检
                    if self.logger.is_active:
                        path = self.logger.save(winner, game_idx)
                        print(f"   -> 尸检报告已保存至: {path}")

                # [新增] 比赛结束，保存这局的日记
                elif self.logger.is_active:
                    saved_path = self.logger.save(winner, game_idx)
                    print(f"\n🧠 [AI 读心] 第 {game_idx} 局的心声已保存至 {saved_path}")
                return winner, reason, ai_fallback_count # 补齐 3 个返回值

            # --- Retry 处理 ---
            if msg_type == 1:
                consecutive_retries += 1
                if last_decision_value is not None:
                    current_step_ignore_list.append(last_decision_value)

                if consecutive_retries == 1:
                     ai_fallback_count += 1
                
                # 💥 [关键同步] 引入 run_self_play.py 的 Jitter 扰动机制！
                # 当 AI 和 RuleBot 都陷入死锁时，强制注入噪音打破僵局！
                if consecutive_retries > 6:
                    current_step_ignore_list.append(b'\xFF\xFF\xFF')
                    current_step_ignore_list.append(b'\x00\x00\x00')
                    current_step_ignore_list.append(1)
                    current_step_ignore_list.append(4)
                
                if consecutive_retries > 20:
                    return -1, -1, ai_fallback_count
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

                        current_hash = "|".join([f"{a.action_type}_{a.index}" for a in snap.valid_actions])
                        if current_hash != last_valid_hash:
                            last_valid_hash = current_hash
                            current_step_ignore_list.clear() # 场面推进了，清空黑名单
                            action_history.clear()
                        
                        # 2. 调用 V3 编码器生成 53 维字典，并挂载到设备 (GPU/CPU)
                        tensor_dict = active_bot.encoder.encode(snap, player_id=player)
                        tensor_dict['act_card_idx'] = torch.clamp(tensor_dict['act_card_idx'], 0, 99)
                        infer_dict = {k: v.to(self.device) for k, v in tensor_dict.items()}
                        
                        # 3. 神经网络前向传播
                        with torch.no_grad():
                            logits, _ = active_bot.net(infer_dict)
                            
                            # 🛡️ [断路器生效] 强制将已拉黑的动作得分降为极小值 (-1e9)
                            for bad_idx in banned_actions_for_state[current_hash]:
                                if bad_idx < logits.shape[-1]:
                                    logits[0, bad_idx] = -1e9
                                    
                            # 👑 [竞技场核心] 绝对贪婪策略
                            action_idx = torch.argmax(logits, dim=-1)
                            
                        # 🌟 [新增] 记录动作频率，侦测死循环
                        sel_idx = action_idx.item()
                        loop_key = f"{current_hash}_{sel_idx}"
                        loop_tracker[loop_key] = loop_tracker.get(loop_key, 0) + 1
                        
                        # 事不过三，超过 3 次同样的动作，立刻拉黑！
                        if loop_tracker[loop_key] >= 3:
                            banned_actions_for_state[current_hash].add(sel_idx)
                            print(f"\n   ⚡ [断路器] 侦测到合法死循环！屏蔽该状态下动作索引: {sel_idx}")
                            
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
                            
                        # 5. 翻译动作为 YGOPro 底层指令
                        resp = None
                        
                        if msg_type == 15:
                            val = chosen.index
                            if val == -1: 
                                resp = struct.pack('<i', -1)
                            else:
                                min_c = msg[3] if len(msg) >= 4 else 1
                                count = max(1, min_c)
                                available_indices = [a.index for a in snap.valid_actions if a.index != -1 and a.index != val]
                                selected = [val] + available_indices[:count-1]
                                
                                if len(selected) < min_c:
                                    resp = None # AI 凑不够卡，直接让 RuleBot 兜底
                                else:
                                    resp_buf = bytearray([len(selected)])
                                    for i in selected: resp_buf.append(i)
                                    resp = bytes(resp_buf)
                                
                        # 🌟 核心修复：10, 11, 16 彻底抛弃 bytes 转换，直接传原生整数！
                        elif msg_type in [10, 11]:
                            resp = int((chosen.index << 16) | chosen.action_type)
                        elif msg_type == 16:
                            resp = int(chosen.index)
                        elif msg_type in [12, 13, 14]:
                            resp = int(chosen.index)
                            
                        # 19 和 区域选择 依然保留 bytes，galatea_env 会帮我们安全补全 64 字节
                        elif msg_type == 19:
                            val = chosen.index
                            resp = bytes([val])
                        elif msg_type in [18, 24]:
                            zone_id = chosen.index
                            p = 0; l = 0x04; s = 0
                            if zone_id & 16: p = 1
                            if zone_id & 8:  l = 0x08
                            s = zone_id & 0x7
                            req_p = msg[1] if len(msg) > 1 else 0
                            raw_p = req_p if p == 0 else (1 - req_p)
                            final_p = 1 if raw_p == 1 else 0
                            resp = bytes([final_p, l, s])
                            
                        if resp is not None:
                            last_decision_value = resp
                                
                    except Exception as e:
                        resp = None

                # RuleBot 兜底
                if resp is None:
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
        
        # [防漏电] 如果因为超时或死锁非正常退出，强制关闭录像机
        if self.logger.is_active:
            self.logger.is_active = False
        
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
        
        # 修复了致命的语法错误：只需要 range(n_games) 即可
        for i in range(n_games):
            # [新增] 如果开启了记录，并且到达了指定的间隔局数，唤醒 Logger
            if self.thought_freq > 0 and (i + 1) % self.thought_freq == 0:
                self.logger.start_recording()

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
            else: # 👇 [新增] 把未知的数字打印出来！
                reason_str = f"Special({r})"
                reasons[reason_str] = reasons.get(reason_str, 0) + 1
            
            if w == 0: p0_wins += 1
            elif w == 1: p1_wins += 1
            else: draws += 1
            
            # 删除了冗余的双重打印逻辑，只保留带 Fallback 信息的最完美输出格式
            score_str = f"Score: {p0_wins}-{p1_wins}"
            fallback_info = f" | ⚠️ AI Fallbacks: {fallback_cnt}" if fallback_cnt > 0 else ""
            
            if is_abnormal:
                print(f"⚠️ Game {i+1}: Aborted ({reason_str}) | {score_str}{fallback_info}")
            else:
                # 正常局使用 \r 覆盖打印，保持控制台整洁
                print(f"   Game {i+1}: Winner P{w} ({reason_str}) | {score_str}{fallback_info}", end="\r")
        
        print(f"\n\n🏆 最终比分: AI(P0) {p0_wins} : {p1_wins} RuleBot(P1)")
        print("📊 胜负原因统计:")
        for k, v in reasons.items():
            if v > 0: print(f"   - {k}: {v}")