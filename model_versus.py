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
from action_candidates import (
    MACRO_ACTION_MSGS,
    MODEL_ACTION_MSGS,
    build_action_state_key,
    build_macro_action_pool,
)
from galatea_env import GalateaEnv
from gamestate import MessageParser, DuelState
from ai_bot import AiBot
from feature_encoder import MAX_CARDS
from checkpoint_utils import MODEL_PROTOCOL_VERSION
import deck_utils
from thought_logger import AIThoughtLogger, REPLAY_EVENT_MSGS
from protocol_v3_audit import (
    configure_protocol_v3_audit,
    flush_protocol_v3_audit,
)


ARENA_LOOP_SOFT_BAN_THRESHOLD = 5


def select_arena_action_index(
    valid_logits,
    retry_bans,
    loop_bans,
    loop_counts=None,
):
    """应用硬/软禁用；软禁用耗尽时改选访问次数最低的合法候选"""
    if valid_logits.dim() != 1 or valid_logits.numel() == 0:
        raise RuntimeError("model produced no encodable actions")

    finite_mask = torch.isfinite(valid_logits)
    if not finite_mask.any():
        raise RuntimeError("model produced no finite action logits")

    # 网络内部使用极小值屏蔽 act_mask=False 的槽位；有效前缀不应全部落入该范围
    available_mask = finite_mask & (valid_logits > -64000.0)
    if not available_mask.any():
        raise RuntimeError("model or encoder masked every valid action")

    for action_index in retry_bans:
        if 0 <= action_index < available_mask.numel():
            available_mask[action_index] = False
    if not available_mask.any():
        raise RuntimeError("all model actions were rejected by engine retry")

    loop_filtered_mask = available_mask.clone()
    for action_index in loop_bans:
        if 0 <= action_index < loop_filtered_mask.numel():
            loop_filtered_mask[action_index] = False

    ignored_exhaustive_loop_bans = not loop_filtered_mask.any()
    if ignored_exhaustive_loop_bans:
        loop_filtered_mask = available_mask
        if loop_counts is not None:
            available_indices = torch.nonzero(
                available_mask,
                as_tuple=False,
            ).flatten().tolist()
            minimum_count = min(
                int(loop_counts.get(index, 0))
                for index in available_indices
            )
            for action_index in available_indices:
                if int(loop_counts.get(action_index, 0)) != minimum_count:
                    loop_filtered_mask[action_index] = False

    masked_logits = valid_logits.clone()
    masked_logits[~loop_filtered_mask] = -torch.inf
    return int(torch.argmax(masked_logits).item()), ignored_exhaustive_loop_bans


def describe_arena_loop_state(snapshot, msg_type, state_key, loop_tracker):
    """生成防循环软禁用耗尽时的候选与重复次数摘要"""
    details = []
    for index, action in enumerate(snapshot.valid_actions):
        description = action.desc_str or f"Type={action.action_type}"
        count = loop_tracker.get((state_key, index), 0)
        details.append(f"[{index}] {description}×{count}")
    return f"MsgType={msg_type} | " + "; ".join(details[:12])


def find_matching_action_index(snapshot, response, msg_type, msg_args, packer):
    """按真实 Core 响应反查 RuleBot 选择的普通候选索引"""
    expected = bytes(response) if isinstance(response, (bytes, bytearray)) else response
    for index, action in enumerate(snapshot.valid_actions):
        try:
            packed = packer._pack_response(action, msg_type=msg_type, msg_args=msg_args)
            actual = bytes(packed) if isinstance(packed, (bytes, bytearray)) else packed
            if actual == expected:
                return index
        except Exception:
            continue
    return None


class ModelArena:
    # 增加 config 参数
    def __init__(self, model_p0_path, model_p1_path=None, device='cpu', deck_dir="./decks", config=None, standard_core=False, protocol_audit=False):
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
            
        # 2. 竞技场控制参数与模型架构分离；网络结构始终以检查点内置配置为准。
        self.arena_config = dict(config or {})
        self.thought_freq = self.arena_config.get('thought_freq', 0)
        
        # 4. 初始化记录器
        p0_name = os.path.basename(model_p0_path) if model_p0_path else "P0_AI"
        p1_name = os.path.basename(model_p1_path) if model_p1_path else "RuleBot"
        self.p0_name = p0_name
        self.p1_name = p1_name
        if protocol_audit:
            configure_protocol_v3_audit(
                source="arena",
                run_label=f"{os.path.splitext(p0_name)[0]}_vs_{os.path.splitext(p1_name)[0]}",
            )
        self.logger = AIThoughtLogger(player_name=p0_name, opponent_name=p1_name)
        
        # P0
        self.p0_bot = AiBot(device=self.device, initialize_network=False)
        self._require_model_loaded(self.p0_bot, model_p0_path, "P0")
        self.p0_bot.net.eval()
        self.net_config = {
            'd_model': self.p0_bot.net.d_model,
            'n_heads': self.p0_bot.net.n_heads,
            'n_layers': self.p0_bot.net.n_layers,
            'vocab_size': self.p0_bot.net.vocab_size,
            'model_protocol_version': MODEL_PROTOCOL_VERSION,
        }
        print(f"⚙️ [Arena] P0 模型架构（检查点）: {self.net_config}")
        print(f"🤖 [AiBot] 成功加载模型权重: {model_p0_path}")

        # P1: 对手
        self.p1_bot = None
        if model_p1_path:
            self.p1_bot = AiBot(device=self.device, initialize_network=False)
            self._require_model_loaded(self.p1_bot, model_p1_path, "P1")
            self.p1_bot.net.eval()
            print(f"🤖 [Opponent] 成功加载模型权重: {model_p1_path}")
        else:
            print(f"🤖 [Opponent] 使用 RuleBot (内置规则脚本)")

        self.env = GalateaEnv()

    @staticmethod
    def _require_model_loaded(bot, path, player_label):
        """确保指定玩家模型成功加载，失败时立即终止初始化"""
        if not path:
            raise ValueError(f"{player_label} model path is required")
        if not bot.load_model(path):
            raise RuntimeError(f"{player_label} model failed to load: {path}")

    @staticmethod
    def _validate_action_indices(tensor_dict):
        """校验动作卡片索引，同时保留 MAX_CARDS 作为无目标哨兵"""
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
            if res is None:
                return -1, -3, 0
            env_name, d1_name, d1, d2_name, d2 = res

            raw_data = self.env.reset(d1, d2)
            if not raw_data:
                return -1, -3, 0

            brain = DuelState(d1.main, d1.extra, d2.main, d2.extra)
            self.logger.set_decklists(
                d1.main,
                d1.extra,
                d2.main,
                d2.extra,
                p0_name=d1_name,
                p1_name=d2_name,
            )
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

        last_model_state_key = None
        action_history = []

        # 合法死锁断路器
        loop_tracker = {}
        retry_bans_for_state = {}
        loop_bans_for_state = {}
        
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

            # 回放 V2 同时记录 Core 状态事件，避免只看到模型决策而看不到移动与结算。
            if self.logger.is_active and msg_type in REPLAY_EVENT_MSGS:
                event_snapshot = brain.get_snapshot()
                self.logger.log_core_event(
                    turn=brain.turn,
                    phase_id=brain.phase,
                    snapshot=event_snapshot,
                    msg_type=msg_type,
                    payload=msg[1:],
                )

            if msg_type in DECISION_MSGS:
                current_macro_pool = None
                last_interaction_msg = msg

            # 状态重置：如果发生局势变动，清空 Retry 计数和黑名单
            if msg_type in STATE_CHANGE_MSGS:
                consecutive_retries = 0
                current_step_ignore_list.clear()
                if msg_type in [40, 41]:
                    loop_tracker.clear()
                    retry_bans_for_state.clear()
                    loop_bans_for_state.clear()
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

                if last_decision_index is not None and last_model_state_key is not None:
                    retry_bans_for_state.setdefault(last_model_state_key, set()).add(
                        last_decision_index
                    )

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

                        current_state_key = build_action_state_key(snap, msg_type)
                        if current_state_key != last_model_state_key:
                            last_model_state_key = current_state_key
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
                            sel_idx, ignored_loop_bans = select_arena_action_index(
                                valid_logits,
                                retry_bans_for_state.get(current_state_key, set()),
                                loop_bans_for_state.get(current_state_key, set()),
                                loop_counts={
                                    index: loop_tracker.get(
                                        (current_state_key, index),
                                        0,
                                    )
                                    for index in range(valid_count)
                                },
                            )
                            if ignored_loop_bans:
                                loop_bans_for_state.pop(current_state_key, None)
                                fallback_action = snap.valid_actions[sel_idx]
                                fallback_description = (
                                    fallback_action.desc_str
                                    or f"Type={fallback_action.action_type}"
                                )
                                print(
                                    "⚠️ [Arena] 防循环软禁用已覆盖全部候选，"
                                    "本次改选重复次数最低的合法候选\n"
                                    f"   ↳ 本次选择: [{sel_idx}] {fallback_description}\n"
                                    f"   ↳ {describe_arena_loop_state(snap, msg_type, current_state_key, loop_tracker)}"
                                )

                        loop_key = (current_state_key, sel_idx)
                        loop_tracker[loop_key] = loop_tracker.get(loop_key, 0) + 1
                        if loop_tracker[loop_key] >= ARENA_LOOP_SOFT_BAN_THRESHOLD:
                            loop_bans_for_state.setdefault(current_state_key, set()).add(
                                sel_idx
                            )

                        if self.logger.is_active:
                            probs = F.softmax(logits.squeeze(0), dim=-1)
                            self.logger.log_decision(
                                turn=brain.turn,
                                phase_id=brain.phase,
                                snapshot=snap,
                                probs=probs,
                                chosen_index=sel_idx,
                                player_id=player_to_act,
                                msg_type=msg_type,
                                agent_name=self.p0_name if is_p0_turn else self.p1_name,
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
                    if self.logger.is_active:
                        rule_snapshot = brain.get_snapshot(self.env)
                        rule_chosen_index = find_matching_action_index(
                            rule_snapshot,
                            resp,
                            msg_type,
                            msg[1:],
                            self.p0_bot,
                        )
                        self.logger.log_external_decision(
                            turn=brain.turn,
                            phase_id=brain.phase,
                            snapshot=rule_snapshot,
                            msg_type=msg_type,
                            response=resp,
                            player_id=player_to_act,
                            agent_name="RuleBot",
                            chosen_index=rule_chosen_index,
                        )

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

            # 初始化阶段若未产生任何帧，也必须关闭本局录像状态，避免串入下一局。
            if self.logger.is_active:
                self.logger.save(-1, i + 1, "对局未产生可保存的回放帧")

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
        
        print(f"\n\n🏆 最终比分: {self.p0_name}(P0) {p0_wins} : {p1_wins} {self.p1_name}(P1)")
        print("📊 胜负原因统计:")
        for k, v in reasons.items():
            if v > 0: print(f"   - {k}: {v}")
        audit_path = flush_protocol_v3_audit(force=True)
        if audit_path is not None:
            print(f"🧪 V3 观测审计已保存: {audit_path}")
