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


def _frame_evaluations(frame):
    """读取同帧多视角扩展并兼容旧单视角字段，严格保留预测所属玩家"""
    multiple = frame.get('evaluations')
    if isinstance(multiple, dict):
        for key, evaluation in multiple.items():
            yield (int(key) if key in ('0', '1') else None), evaluation
    elif multiple is not None:
        yield frame.get('player'), multiple
    player = frame.get('player')
    if (frame.get('evaluation') is not None
            and (not isinstance(multiple, dict) or str(player) not in multiple)):
        yield player, frame['evaluation']


def replay_evaluation_actual(frame, player):
    """读取对应玩家的真实终局对照，不把行动方答案挂到另一方曲线"""
    multiple = frame.get('evaluation_actuals')
    if isinstance(multiple, dict) and str(player) in multiple:
        return multiple[str(player)]
    return frame.get('evaluation_actual') if frame.get('player') == player else None


def _validated_prediction(evaluation, player):
    """校验既有单预测合同，返回数值或明确原因，不放宽视角/概率校验"""
    if (not isinstance(evaluation, dict)
            or evaluation.get('format_version') != EVALUATION_FORMAT_VERSION
            or evaluation.get('kind') != 'state_terminal'
            or evaluation.get('length_unit') != 'transition_events'):
        return None, 'format'
    if (type(player) is not int or player not in (0, 1)
            or type(evaluation.get('perspective')) is not int
            or evaluation.get('perspective') != player):
        return None, 'perspective'
    try:
        probabilities = evaluation['probabilities']
        loss, draw, win = (float(probabilities[k]) for k in ('loss', 'draw', 'win'))
        remaining = float(evaluation['remaining_events'])
        if (not all(math.isfinite(x) for x in (loss, draw, win, remaining))
                or min(loss, draw, win, remaining) < 0
                or max(loss, draw, win) > 1
                or abs(loss + draw + win - 1) > 1e-4):
            return None, 'numeric'
    except (KeyError, ValueError, TypeError, OverflowError):
        return None, 'numeric'
    return (loss, draw, win, remaining), None


def replay_evaluation_summary(replay):
    """统一校验预测并统计双方覆盖情况，不补造事件帧或未来预测"""
    from replay_utils import get_replay_frames

    points = []
    frames = get_replay_frames(replay)
    coverage = {side: {'valid': 0, 'missing': 0, 'invalid': 0,
                       'first_frame': None, 'last_frame': None} for side in ('P0', 'P1')}
    reasons = {}
    for frame_index, frame in enumerate(frames, 1):
        if not isinstance(frame, dict):
            continue
        entries = list(_frame_evaluations(frame))
        for player, evaluation in entries:
            side = f'P{player}' if type(player) is int and player in (0, 1) else None
            values, reason = _validated_prediction(evaluation, player)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                if side:
                    coverage[side]['invalid'] += 1
                continue
            loss, draw, win, remaining = values
            points.append({
                'frame': frame_index, 'perspective': side,
                'win_probability': win, 'draw_probability': draw, 'loss_probability': loss,
                'expected_score': win + 0.5 * draw, 'remaining_events': remaining,
                'snapshot_source': evaluation.get('snapshot_source', 'legacy_decision'),
            })
            stats = coverage[side]
            stats['valid'] += 1
            if stats['first_frame'] is None:
                stats['first_frame'] = frame_index
            stats['last_frame'] = frame_index
        metadata = replay.get('recording_metadata') or {}
        if isinstance(metadata, dict) and metadata.get('evaluation_sampling') == 'recorded_frames':
            models = metadata.get('models') or {}
            for player in (0, 1):
                model = models.get(str(player)) if isinstance(models, dict) else None
                if (isinstance(model, dict) and model.get('kind') not in ('rule', 'windbot')
                        and player not in [owner for owner, _ in entries]):
                    coverage[f'P{player}']['missing'] += 1
        elif frame.get('source') == 'model' and not entries:
            player = frame.get('player')
            if type(player) is int and player in (0, 1):
                coverage[f'P{player}']['missing'] += 1
    return {'points': points, 'coverage': coverage, 'invalid_reasons': reasons,
            'frame_count': len(frames)}


def replay_evaluation_points(replay):
    """提取有效逐决策曲线点，旧录像和损坏预测自动跳过"""
    return replay_evaluation_summary(replay)['points']


def latest_replay_evaluations(points, frame_number):
    """分别读取截至指定一基帧号的最近预测，绝不借用未来或另一方的值"""
    latest = {'P0': None, 'P1': None}
    for point in points:
        if point['frame'] <= frame_number:
            side = point['perspective']
            if latest[side] is None or point['frame'] > latest[side]['frame']:
                latest[side] = point
    return latest
