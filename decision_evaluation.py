"""序列化状态级终局预测，并为全息回放提供独立双视角曲线"""

import math

EVALUATION_FORMAT_VERSION = 1
MAX_DISPLAY_EVENTS = 1_000_000


def serialize_state_evaluation(predictions, value, *, player, sequence_id,
                               model_metadata, policy_metadata):
    """将单状态预测转为标量，不把未来答案或预测反哺动作选择"""
    from auxiliary_heads import AUXILIARY_MAX_REMAINING_EVENTS

    logits = predictions['terminal_outcome_logits'].detach().float()
    remaining = predictions['terminal_remaining'].detach().float()
    if tuple(logits.shape) != (1, 3) or remaining.numel() != 1:
        raise ValueError('evaluation requires one state and three terminal classes')
    probabilities = logits.softmax(dim=-1)[0].cpu().tolist()
    length = float(remaining.item())
    critic = float(value.detach().float().reshape(-1)[0].item())
    if not all(math.isfinite(x) for x in probabilities + [length, critic]):
        raise ValueError('nonfinite state evaluation')
    log_events = max(0.0, length) * math.log1p(AUXILIARY_MAX_REMAINING_EVENTS)
    return {
        'format_version': EVALUATION_FORMAT_VERSION,
        'kind': 'state_terminal', 'perspective': int(player),
        'sequence_id': int(sequence_id),
        'model': {key: (model_metadata or {}).get(key) for key in
                  ('model_id', 'model_prefix', 'iteration', 'train_step', 'checkpoint_sha256')},
        'policy': dict(policy_metadata or {}),
        'calibration': {'status': 'uncalibrated', 'version': None},
        'probabilities': dict(zip(('loss', 'draw', 'win'), probabilities)),
        'remaining_events': math.expm1(min(log_events, math.log1p(MAX_DISPLAY_EVENTS))),
        'remaining_events_clipped': log_events > math.log1p(MAX_DISPLAY_EVENTS),
        'length_unit': 'transition_events', 'discounted_value': critic,
    }


def replay_evaluation_points(replay):
    """提取有效逐决策曲线点，旧录像和损坏预测自动跳过"""
    points = []
    for frame_index, frame in enumerate(replay.get('frames', []), 1):
        if not isinstance(frame, dict):
            continue
        evaluation = frame.get('evaluation') or {}
        if (not isinstance(evaluation, dict)
                or evaluation.get('format_version') != EVALUATION_FORMAT_VERSION
                or evaluation.get('kind') != 'state_terminal'
                or evaluation.get('length_unit') != 'transition_events'
                or evaluation.get('perspective') not in (0, 1)
                or type(evaluation.get('perspective')) is not int
                or evaluation.get('perspective') != frame.get('player')):
            continue
        try:
            probabilities = evaluation['probabilities']
            loss, draw, win = (float(probabilities[k]) for k in ('loss', 'draw', 'win'))
            remaining = float(evaluation['remaining_events'])
            if (not all(math.isfinite(x) for x in (loss, draw, win, remaining))
                    or min(loss, draw, win, remaining) < 0
                    or max(loss, draw, win) > 1
                    or abs(loss + draw + win - 1) > 1e-4):
                continue
        except (KeyError, ValueError, TypeError):
            continue
        points.append({
            'frame': frame_index, 'perspective': f"P{evaluation['perspective']}",
            'win_probability': win, 'draw_probability': draw,
            'expected_score': win + 0.5 * draw, 'remaining_events': remaining,
        })
    return points
