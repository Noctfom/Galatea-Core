# ==================================================================================
#  Galatea Model Versus (Arena Mode) - Enhanced Logging
# ==================================================================================

import os
import numpy as np
import torch
import struct
import traceback
import torch.nn.functional as F

import rule_bot
from action_candidates import MACRO_ACTION_MSGS, MODEL_ACTION_MSGS, build_macro_action_pool
from galatea_env import GalateaEnv
from gamestate import MessageParser, DuelState
from ai_bot import AiBot
from feature_encoder import MAX_CARDS
import deck_utils
from thought_logger import AIThoughtLogger


class ModelArena:
    # 增加 config 参数
    def __init__(self, model_p0_path, model_p1_path=None, device='cpu', deck_dir="./decks", config=None, standard_core=False):
        self.deck_dir = deck_dir
        self.standard_core = standard_core

        import gamestate
        if standard_core:
            gamestate.CORE_HAS_GHOST_BYTE = False
            print("🔧 [Arena] 协议自适应：已关闭幽灵定界符 (Standard Core Mode)")
        else:
            gamestate.CORE_HAS_GHOST_BYTE = True

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
        
        # 3. 这时候再去取 thought_freq 就绝对安全了
        self.thought_freq = self.net_config.get('thought_freq', 0)
        
        # 4. 初始化记录器
        p0_name = os.path.basename(model_p0_path) if model_p0_path else "P0_AI"
        self.logger = AIThoughtLogger(player_name=p0_name)
        
        print(f"⚙️ [Arena] 模型配置: {self.net_config}")
        
        # P0
        self.p0_bot = AiBot(device=self.device, net_config=self.net_config)
        self._require_model_loaded(self.p0_bot, model_p0_path, "P0")
        self.p0_bot.net.eval()
        print(f"🤖 [AiBot] 成功加载模型权重: {model_p0_path}")

        # P1: 对手
        self.p1_bot = None
        if model_p1_path:
            self.p1_bot = AiBot(device=self.device)
            self._require_model_loaded(self.p1_bot, model_p1_path, "P1")
            self.p1_bot.net.eval()
            print(f"🤖 [Opponent] 成功加载模型权重: {model_p1_path}")
        else:
            print(f"🤖 [Opponent] 使用 RuleBot (内置规则脚本)")

        self.env = GalateaEnv()

    @staticmethod
    def _require_model_loaded(bot, path, player_label):
        if not path:
            raise ValueError(f"{player_label} model path is required")
        if not bot.load_model(path):
            raise RuntimeError(f"{player_label} model failed to load: {path}")

    @staticmethod
    def _validate_action_indices(tensor_dict):
        indices = tensor_dict['act_card_idx']
        if torch.any(indices < 0) or torch.any(indices > MAX_CARDS):
            minimum = int(indices.min().item())
            maximum = int(indices.max().item())
            raise ValueError(
                f"act_card_idx outside [0, {MAX_CARDS}]: min={minimum}, max={maximum}"
            )

    def run_duel(self, game_idx=1):
        """
        返回: (winner_index, reason_code, model_fallback_count)
        reason_code:
           0-4: 游戏规则胜利 (投降, 0LP, 0卡组等)
           -1: AI死锁 (Retry过多)
           -2: 超时 (Steps过多)
           -3: 初始化失败
           -4: 模型决策链失败
        """
        try:
            res = deck_utils.get_random_deck_pair(ydk_dir=self.deck_dir)
            if not res:
                return -1, -3, 0
            env_name, d1_name, d1, d2_name, d2 = res

            raw_data = self.env.reset(d1, d2)
            if not raw_data:
                return -1, -3, 0

            brain = DuelState(d1.main, d1.extra, d2.main, d2.extra)
            msg_queue = MessageParser.parse(raw_data)
        except Exception as error:
            print(f"\n❌ [Arena] 对局初始化失败: {error}")
            traceback.print_exc()
            return -1, -3, 0
        
        consecutive_retries = 0
        current_step_ignore_list = []
        last_decision_value = None
        last_decision_index = None
        last_interaction_msg = None
        current_macro_pool = None

        ai_fallback_count = 0

        last_valid_hash = ""
        action_history = []

        # 合法死锁断路器
        loop_tracker = {}
        banned_actions_for_state = {}
        
        # 替换为最全的常量集合
        STATE_CHANGE_MSGS = {40, 41, 50, 53, 54, 55, 56, 60, 61, 62, 70, 90, 91, 92, 94}
        INTERACTION_MSGS = {10, 11, 15, 16, 18, 19, 20, 22, 23, 24, 26, 130, 131, 132, 133}
        DECISION_MSGS = {10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 130, 131, 132, 133, 140, 141, 142, 143}
        AI_MANAGED_MSGS = MODEL_ACTION_MSGS
        macro_rng = np.random.default_rng(game_idx)

        steps = 0
        # 增加步数上限到 5000，防止慢速卡组被误判
        while steps < 5000: 
            if not msg_queue:
                raw_data = self.env.step()
                if not raw_data: break
                msg_queue = MessageParser.parse(raw_data)
                # 只要新来的数据包不是以 RETRY (1) 开头，说明上一回合的动作必定被引擎接受了！
                if msg_queue and msg_queue[0][0] != 1:
                    consecutive_retries = 0
                    current_step_ignore_list.clear()
                    current_macro_pool = None
                continue
            
            msg = msg_queue.pop(0)
            msg_type = msg[0]
            brain.update(msg_type, msg[1:])

            if msg_type in DECISION_MSGS:
                current_macro_pool = None
                last_interaction_msg = msg

            # 状态重置：如果发生局势变动，清空 Retry 计数和黑名单
            if msg_type in STATE_CHANGE_MSGS:
                consecutive_retries = 0
                current_step_ignore_list.clear()
                if msg_type in [40, 41]:
                    loop_tracker.clear()
                    banned_actions_for_state.clear()
            elif msg_type in INTERACTION_MSGS:
                if consecutive_retries == 0:
                    current_step_ignore_list.clear()
            
            if msg_type == 5: # 胜利 MSG_WIN
                # msg格式通常是 [5, winner, reason]
                winner = msg[1:][0]
                reason = msg[1:][1] if len(msg[1:]) > 1 else 0

                # 🌟 修复 4：翻译底层的胜利原因代码
                r_map = {0: "投降 (Surrender)", 1: "LP 归零", 2: "卡组抽干 (Deck Out)", 3: "时间耗尽", 4: "特殊胜利"}
                reason_str = r_map.get(reason, f"未知代码({reason})")
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
                    # 将 reason_str 透传给记录器
                    saved_path = self.logger.save(winner, game_idx, reason_str)
                    print(f"\n🧠 [AI 读心] 第 {game_idx} 局的心声已保存至 {saved_path}")
                return winner, reason, ai_fallback_count

            # --- Retry 处理 ---
            if msg_type == 1:
                consecutive_retries += 1
                if last_decision_value is not None:
                    current_step_ignore_list.append(last_decision_value)

                if last_decision_index is not None and last_valid_hash:
                    banned_actions_for_state.setdefault(last_valid_hash, set()).add(last_decision_index)

                # [犯罪现场记录仪] 如果连 RuleBot 都卡死了，立刻在终端打印尸检报告
                if consecutive_retries > 6:
                    print(f"\n   [🚨 深度追凶] 连续被引擎拒绝 {consecutive_retries} 次！")
                    print(f"      - 引擎正在追问的 MsgType: {last_interaction_msg[0] if last_interaction_msg else 'Unknown'}")
                    print(f"      - 刚被引擎拒绝的动作代码 (Raw Bytes/Int): {last_decision_value}")
                    print(f"      - 当前 RuleBot 忽略名单 (黑名单): {current_step_ignore_list}")
                    if brain.current_valid_actions:
                        print(f"      - 当前回合可用的合法选项: {[f'类型:{a.action_type}_索引:{a.index}' for a in brain.current_valid_actions]}")
                
                current_limit = 100 if (last_interaction_msg and last_interaction_msg[0] == 142) else 20
                if consecutive_retries > current_limit:
                    return -1, -1, ai_fallback_count
                
                # 时空回溯
                if last_interaction_msg is not None:
                    msg = last_interaction_msg
                    msg_type = msg[0]
                else:
                    continue

            # --- 决策 ---
            if msg_type in DECISION_MSGS:
                player_to_act = msg[1] if len(msg) > 1 else 0
                is_p0_turn = player_to_act == 0
                active_bot = self.p0_bot if is_p0_turn else self.p1_bot
                model_should_act = active_bot is not None and msg_type in AI_MANAGED_MSGS

                if model_should_act:
                    try:
                        snap = brain.get_snapshot(self.env)
                        player = snap.global_data.to_play

                        if msg_type in MACRO_ACTION_MSGS:
                            if current_macro_pool is None:
                                base_actions = list(brain.current_valid_actions)
                                base_tensor_dict = active_bot.encoder.encode(snap, player_id=player)
                                self._validate_action_indices(base_tensor_dict)
                                base_infer_dict = {
                                    key: value.to(self.device)
                                    for key, value in base_tensor_dict.items()
                                }
                                with torch.no_grad():
                                    base_logits, _, _ = active_bot.net(base_infer_dict)
                                    base_probabilities = F.softmax(
                                        base_logits.squeeze(0), dim=-1
                                    )

                                current_macro_pool = build_macro_action_pool(
                                    msg_type,
                                    msg[1:],
                                    brain,
                                    base_actions,
                                    base_probabilities,
                                    rng=macro_rng,
                                )
                                if not current_macro_pool:
                                    raise RuntimeError(
                                        f"no legal macro actions generated for message {msg_type}"
                                    )

                            brain.current_valid_actions = current_macro_pool
                            # Map raw macro locations to entity indices exactly as training does.
                            snap = brain.get_snapshot(self.env)

                        if not snap.valid_actions:
                            raise RuntimeError(f"model received no valid actions for message {msg_type}")

                        current_hash = "|".join(
                            f"{action.action_type}_{action.index}_{action.desc_id}_"
                            f"{bytes(getattr(action, 'decision_bytes', b'')).hex()}"
                            for action in snap.valid_actions
                        )
                        if current_hash != last_valid_hash:
                            last_valid_hash = current_hash
                            current_step_ignore_list.clear()
                            action_history.clear()

                        tensor_dict = active_bot.encoder.encode(snap, player_id=player)
                        self._validate_action_indices(tensor_dict)
                        infer_dict = {
                            key: value.to(self.device)
                            for key, value in tensor_dict.items()
                        }

                        with torch.no_grad():
                            logits, _, _ = active_bot.net(infer_dict)
                            valid_count = min(
                                len(snap.valid_actions), logits.shape[-1]
                            )
                            if valid_count == 0:
                                raise RuntimeError("model produced no encodable actions")
                            valid_logits = logits[0, :valid_count].clone()
                            for bad_idx in banned_actions_for_state.get(current_hash, set()):
                                if 0 <= bad_idx < valid_count:
                                    valid_logits[bad_idx] = -65000.0

                            if not torch.isfinite(valid_logits).any() or torch.all(
                                valid_logits <= -64000.0
                            ):
                                raise RuntimeError("all model actions are banned or non-finite")
                            sel_idx = int(torch.argmax(valid_logits).item())

                        loop_key = f"{current_hash}_{sel_idx}"
                        loop_tracker[loop_key] = loop_tracker.get(loop_key, 0) + 1
                        if loop_tracker[loop_key] >= 3:
                            banned_actions_for_state.setdefault(current_hash, set()).add(sel_idx)

                        if is_p0_turn and self.logger.is_active:
                            probs = F.softmax(logits.squeeze(0), dim=-1)
                            self.logger.log_decision(
                                turn=brain.turn,
                                phase_id=brain.phase,
                                snapshot=snap,
                                probs=probs,
                                chosen_index=sel_idx,
                            )

                        chosen = snap.valid_actions[sel_idx]
                        last_decision_index = sel_idx
                        resp = active_bot._pack_response(
                            chosen,
                            msg_type=msg_type,
                            msg_args=msg[1:],
                        )
                        last_decision_value = resp
                        action_history.append((msg_type, sel_idx, resp))
                        action_history[:] = action_history[-5:]

                    except Exception as e:
                        ai_fallback_count += 1
                        print(f"\n❌ [Arena] 模型决策链失败，拒绝 RuleBot 代打: {e}")
                        traceback.print_exc()
                        if self.logger.is_active:
                            self.logger.save(-1, game_idx, f"模型决策链失败: {e}")
                        return -1, -4, ai_fallback_count
                else:
                    clean_ignore = []
                    for value in current_step_ignore_list:
                        clean_ignore.append(value)
                        if isinstance(value, bytes):
                            if len(value) >= 4:
                                clean_ignore.append(struct.unpack('<I', value[:4])[0])
                            elif len(value) >= 1:
                                clean_ignore.append(value[0])

                    rule_bot.sync_valid_actions(brain.current_valid_actions)
                    resp = rule_bot.get_rule_decision(
                        player_to_act,
                        msg_type,
                        msg,
                        brain,
                        clean_ignore,
                    )
                    last_decision_value = resp

                self.env.send_action(resp)
                msg_queue = []
                steps += 1
        
        # [防漏电] 如果因为超时或死锁非正常退出，强制关闭录像机
        if self.logger.is_active:
            self.logger.save(-1, game_idx, "超时强制截断/死锁熔断")
        
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
            'StepsOut': 0,
            'InitFail': 0,
            'ModelError': 0,
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
            elif r == -3: reason_str = "⚠️ Init Fail"; is_abnormal = True; reasons['InitFail'] += 1
            elif r == -4: reason_str = "❌ Model Decision Error"; is_abnormal = True; reasons['ModelError'] += 1
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
