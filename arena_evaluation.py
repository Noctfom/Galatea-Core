"""竞技场专用双视角状态评估：读取已记录快照，不查询 Core、不采样、不产生训练行"""

from dataclasses import replace

import torch

from decision_evaluation import serialize_state_evaluation


class ArenaStateEvaluator:
    """为模型侧的事件/决策快照补充终局预测，保持物理玩家与快照时点一致"""

    def __init__(self, bots, policy_metadata):
        """绑定双方现有模型，不分配新网络或读取额外环境状态"""
        self.bots = tuple(bots)
        self.policy_metadata = dict(policy_metadata)
        self.failures = {}

    def evaluate(self, snapshot, sequence_id, *, existing=None, snapshot_source='message_mirror'):
        """各方按自身可见信息推理；行动方已有预测直接复用，失败只记录一次告警"""
        result = {}
        existing = existing or {}
        # 被动观测不借用当前行动者候选，避免隐藏候选泄露与伪造本方决策
        state_snapshot = replace(snapshot, valid_actions=[])
        for player, bot in enumerate(self.bots):
            if bot is None:
                continue
            try:
                evaluation = existing.get(player)
                if evaluation is None:
                    observation = bot.encoder.encode(state_snapshot, player_id=player)
                    state_input = {key: value.to(bot.device) for key, value in observation.items()}
                    with torch.no_grad():
                        value, _, predictions = bot.net.evaluate_state(state_input)
                    evaluation = serialize_state_evaluation(
                        predictions, value, player=player, sequence_id=sequence_id,
                        model_metadata=getattr(bot, 'model_metadata', {}),
                        policy_metadata=self.policy_metadata,
                    )
                result[str(player)] = dict(evaluation, snapshot_source=snapshot_source)
            except Exception as error:
                previous = self.failures.get(str(player), 0)
                self.failures[str(player)] = previous + 1
                if previous == 0:
                    print(f'⚠️ [Arena] P{player} 状态预测不可用，保留原对局流程：{error}')
        return result


def record_frame_evaluations(logger, evaluator, snapshot, sequence_id, *,
                             previous_frame_count, existing=None, snapshot_source='message_mirror'):
    """只向刚刚成功记录的帧附加预测，不增删帧、不调用引擎或修改决策"""
    if evaluator is None or not logger.is_active or len(logger.thoughts) != previous_frame_count + 1:
        return
    logger.thoughts[-1]['evaluations'] = evaluator.evaluate(
        snapshot, sequence_id, existing=existing, snapshot_source=snapshot_source)
    logger.recording_metadata['evaluation_failures'] = dict(evaluator.failures)
