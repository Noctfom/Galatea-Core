"""全息预测展示：本地化双视角曲线、持久显示开关、播放游标及预测覆盖诊断"""

from decision_evaluation import latest_replay_evaluations, replay_evaluation_summary
from replay_utils import get_replay_frames


def build_prediction_chart(points, step, frame_count, field, translate):
    """绘制真实预测点与零基播放游标，不向空白区延长曲线或伪造预测"""
    import altair as alt

    _ = translate
    probability = field == 'win_probability'
    values = [dict(point, step=point['frame'] - 1) for point in points]
    current = [dict(point, step=point['frame'] - 1)
               for point in latest_replay_evaluations(points, step + 1).values() if point]
    x = alt.X('step:Q', title=_('回放步数（从 0 开始，包含事件帧）',
                                'Replay step (zero-based, including events)'),
              scale=alt.Scale(domain=[0, max(1, frame_count - 1)], nice=False))
    y = alt.Y(f'{field}:Q', title=(_('AI 预测胜利概率', 'AI-predicted win probability')
                                  if probability else _('预计剩余转移事件数', 'Estimated remaining transitions')),
              scale=alt.Scale(domain=[0, 1] if probability else
                              [0, max(1, max(point[field] for point in points) * 1.05)], nice=False),
              axis=alt.Axis(format='.0%' if probability else ',.0f'))
    color = alt.Color('perspective:N', title=_('预测视角', 'Prediction perspective'),
                      scale=alt.Scale(domain=['P0', 'P1'], range=['#0072B2', '#E69F00']),
                      legend=alt.Legend(orient='bottom'))
    tooltip = [
        alt.Tooltip('step:Q', title=_('来源步数', 'Source step'), format=',.0f'),
        alt.Tooltip('perspective:N', title=_('预测视角', 'Prediction perspective')),
        alt.Tooltip('win_probability:Q', title=_('预测胜利', 'Predicted win'), format='.1%'),
        alt.Tooltip('draw_probability:Q', title=_('预测平局', 'Predicted draw'), format='.1%'),
        alt.Tooltip('loss_probability:Q', title=_('预测失败', 'Predicted loss'), format='.1%'),
        alt.Tooltip('remaining_events:Q', title=_('预计剩余事件', 'Estimated remaining events'), format=',.1f'),
    ]
    # 显式内联数据兼容超过 5,000 个预测点的长局，不修改 Altair 全局数据转换器
    line = alt.Chart(alt.Data(values=values)).mark_line(point=alt.OverlayMarkDef(size=10)).encode(
        x=x, y=y, color=color, order=alt.Order('step:Q'), tooltip=tooltip)
    highlights = alt.Chart(alt.Data(values=current)).mark_point(size=95, filled=True).encode(
        x=x, y=y, color=color, tooltip=tooltip)
    cursor = alt.Chart(alt.Data(values=[{'step': step}])).mark_rule(
        color='#D55E00', strokeDash=[5, 4], strokeWidth=2).encode(
        x=x, tooltip=[alt.Tooltip('step:Q', title=_('当前播放步数', 'Current replay step'), format=',.0f')])
    return (line + highlights + cursor).properties(height=210 if probability else 170)


def render_replay_evaluations(replay, step, translate):
    """持续展示双方最近预测及来源，暂停、步进和事件帧不隐藏面板"""
    import inspect
    import streamlit as st

    _ = translate
    # 使用带 key 的开关而不是每次重绘 expanded=False 的折叠框，兼容旧版 Streamlit
    if not st.toggle(_('📈 AI 预测曲线', '📈 AI prediction curves'), value=True,
                     key='replay_show_predictions', help=_(
                         '暂停或拖动时间轴不会关闭。圆点标出各方截至当前步的最近预测，虚线标出当前播放位置。',
                         'Stays open on pause or seek. Dots mark each side’s latest recorded prediction; the dashed line marks the current replay position.')):
        return
    summary = replay_evaluation_summary(replay)
    points = summary['points']
    st.caption(_(
        'P0/P1 各按自己的可见信息预测，不是互补胜率。曲线只连接实际模型决策的预测点；事件帧不重新预测。数字沿用各方截至当前步的最近预测，并非当前场面的重新计算。剩余长度是动作转移事件数，不是回合数。',
        'P0/P1 predict from their own visible information, not complementary win rates. Lines connect recorded model-decision predictions; events do not trigger predictions. Figures retain each side’s latest prediction, not a recomputation of the current state. Length counts action transitions, not turns.'))
    if points:
        chart_args = {'width': 'stretch'} if 'width' in inspect.signature(st.altair_chart).parameters else {'use_container_width': True}
        # 旧版 Streamlit 不支持图表 key；不为显示修补提高环境版本下限
        keyed_charts = 'key' in inspect.signature(st.altair_chart).parameters
        for field in ('win_probability', 'remaining_events'):
            key_args = {'key': f'replay_prediction_{field}'} if keyed_charts else {}
            st.altair_chart(build_prediction_chart(points, step, summary['frame_count'], field, _),
                            **chart_args, **key_args)
    else:
        st.info(_('该录像没有有效状态预测。需在竞技场启用预测记录；旧录像不会自动补算。',
                  'No valid state predictions. Enable prediction recording in Arena; old replays are not retroactively evaluated.'))

    latest = latest_replay_evaluations(points, step + 1)
    frames = get_replay_frames(replay)
    metadata = replay.get('recording_metadata') or {}
    models = metadata.get('models', {}) if isinstance(metadata, dict) else {}
    for side, point in latest.items():
        st.markdown(f'**{side} ' + _('视角 · AI 预测', 'perspective · AI predictions') + '**')
        if point is None:
            model = models.get(side[1:], {}) if isinstance(models, dict) else {}
            is_rule = isinstance(model, dict) and model.get('kind') == 'rule'
            st.caption(_('RuleBot 不提供模型预测。', 'RuleBot does not provide model predictions.') if is_rule else _(
                '截至当前步还没有该方的有效预测，不使用未来值或另一方数据填充。',
                'No valid prediction for this side yet; future values and the other side’s data are not substituted.'))
            continue
        source_step = point['frame'] - 1
        age = step - source_step
        st.caption(_(f'来源：第 {source_step} 步；' + ('当前步的预测。' if age == 0 else f'沿用此前预测，距当前 {age} 帧。'),
                     f'Source: step {source_step}; ' + ('prediction at this step.' if age == 0 else f'previous prediction, {age} frames ago.')))
        columns = st.columns(3)
        columns[0].metric(f'{side} ' + _('AI 预测胜利概率', 'AI-predicted win probability'), f"{point['win_probability']:.1%}")
        columns[1].metric(f'{side} ' + _('AI 预测平局概率', 'AI-predicted draw probability'), f"{point['draw_probability']:.1%}")
        columns[2].metric(f'{side} ' + _('预计剩余转移事件', 'Estimated remaining transitions'), f"{point['remaining_events']:.1f}")
        actual = frames[source_step].get('evaluation_actual')
        if isinstance(actual, dict):
            outcome = actual.get('outcome')
            result = {1: _('胜', 'win'), 0: _('平', 'draw'), -1: _('负', 'loss')}.get(outcome, _('未知', 'unknown'))
            st.caption(_(
                f"{side} 真实终局对照（不参与预测）：{result}；从来源步到终局实际剩余 {actual.get('remaining_events', '?')} 个转移事件。",
                f"{side} actual terminal reference (not a prediction): {result}; {actual.get('remaining_events', '?')} transitions remained from the source step."))

    with st.expander(_('ℹ️ 预测覆盖与曲线空白说明', 'ℹ️ Prediction coverage and gaps')):
        st.caption(_('首尾留白通常是该方尚未进行模型决策或之后没有新决策；不会把最后预测延长到终局。缺记录数只统计明确标记为模型来源但未保存预测的决策，并不包含普通事件或规则决策。',
                     'Blank starts/ends usually mean this side had no model decision there. The last prediction is not extended to the terminal frame. Missing counts include only explicitly model-sourced decisions without saved predictions, not ordinary events or rule decisions.'))
        for side, stats in summary['coverage'].items():
            span = (_('无', 'none') if stats['first_frame'] is None else
                    f"{stats['first_frame'] - 1}–{stats['last_frame'] - 1}")
            st.caption(_(
                f"{side}：有效预测 {stats['valid']} 个；来源步数 {span}；模型决策未记录 {stats['missing']} 个；无效预测跳过 {stats['invalid']} 个。",
                f"{side}: {stats['valid']} valid predictions; source steps {span}; {stats['missing']} model decisions without predictions; {stats['invalid']} invalid predictions skipped."))
        reason_labels = {'format': _('格式/单位不兼容', 'incompatible format/units'),
                         'perspective': _('玩家视角不匹配', 'player perspective mismatch'),
                         'numeric': _('概率/长度数值无效', 'invalid probabilities/length')}
        if summary['invalid_reasons']:
            st.warning(_('跳过原因：', 'Skipped because: ') + '；'.join(
                f'{reason_labels[reason]} × {count}' for reason, count in summary['invalid_reasons'].items()))
