"""全息预测展示：本地化双视角曲线、持久显示开关、播放游标及预测覆盖诊断"""

from decision_evaluation import latest_replay_evaluations, replay_evaluation_summary, replay_evaluation_actual
from replay_utils import get_replay_frames


def build_prediction_chart(points, step, frame_count, field, translate):
    """绘制真实预测点与零基播放游标，不向空白区延长曲线或伪造预测"""
    import altair as alt

    _ = translate
    probability = field == 'win_probability'
    sources = {'message_mirror': _('已消费消息的场面镜像', 'Consumed-message state mirror'),
               'decision_query': _('决策点查询校准', 'Query-corrected decision state'),
               'legacy_decision': _('旧决策记录', 'Legacy decision recording')}
    values = [dict(point, step=point['frame'] - 1,
                   observation_scope=sources.get(point.get('snapshot_source'), _('记录快照', 'Recorded snapshot')))
              for point in points]
    current_sources = {(point['frame'], point['perspective'])
                       for point in latest_replay_evaluations(points, step + 1).values() if point}
    current = [point for point in values if (point['frame'], point['perspective']) in current_sources]
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
        alt.Tooltip('observation_scope:N', title=_('快照来源', 'Snapshot source')),
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
    metadata = replay.get('recording_metadata') or {}
    continuous = isinstance(metadata, dict) and metadata.get('evaluation_sampling') == 'recorded_frames'
    st.caption(_(
        'P0/P1 各按自己的可见信息预测，不是互补胜率。新录像对每个已记录事件/决策快照分别评估模型侧；事件点只用已消费消息维护的场面，不提前查询后续Core状态，动态属性以最近校准为准。旧录像仍只显示原有预测。剩余长度是动作转移事件数，不是回合数。',
        'P0/P1 use their own visible information, not complementary win rates. New recordings evaluate model sides on each recorded event/decision snapshot. Events use consumed-message mirrors without querying ahead; dynamic attributes retain the latest correction. Old replays retain their original predictions. Length counts action transitions, not turns.'))
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
    models = metadata.get('models', {}) if isinstance(metadata, dict) else {}
    for side, point in latest.items():
        st.markdown(f'**{side} ' + _('视角 · AI 预测', 'perspective · AI predictions') + '**')
        if point is None:
            model = models.get(side[1:], {}) if isinstance(models, dict) else {}
            if isinstance(model, dict) and model.get('kind') == 'windbot':
                st.caption(_('WindBot 是外部执行器，不提供模型预测。', 'WindBot is an external executor and provides no model predictions.'))
                continue
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
        actual = replay_evaluation_actual(frames[source_step], int(side[1:]))
        if isinstance(actual, dict):
            outcome = actual.get('outcome')
            result = {1: _('胜', 'win'), 0: _('平', 'draw'), -1: _('负', 'loss')}.get(outcome, _('未知', 'unknown'))
            st.caption(_(
                f"{side} 真实终局对照（不参与预测）：{result}；从来源步到终局实际剩余 {actual.get('remaining_events', '?')} 个转移事件。",
                f"{side} actual terminal reference (not a prediction): {result}; {actual.get('remaining_events', '?')} transitions remained from the source step."))

    with st.expander(_('ℹ️ 预测覆盖与曲线空白说明', 'ℹ️ Prediction coverage and gaps')):
        if continuous:
            st.caption(_('本录像逐帧记录模型侧状态预测，未记录数统计该模型侧应有但缺失的帧；RuleBot/WindBot 侧不计算。不会用未来值或终局答案补曲线。',
                         'This replay records model-side state predictions per frame. Missing counts refer to expected but absent model-side frame predictions, excluding RuleBot/WindBot. Future values and actual outcomes never fill gaps.'))
        else:
            st.caption(_('旧录像只记录模型决策点，首尾留白及事件间隙不是丢失；未记录数只统计明确模型来源但缺预测的决策。不会倒填新预测。',
                         'Legacy replays record model decisions only; blank spans and events are not missing data. Missing counts cover explicitly model-sourced decisions without predictions. New predictions are not backfilled.'))
        for side, stats in summary['coverage'].items():
            span = (_('无', 'none') if stats['first_frame'] is None else
                    f"{stats['first_frame'] - 1}–{stats['last_frame'] - 1}")
            st.caption(_(
                f"{side}：有效预测 {stats['valid']} 个；来源步数 {span}；{'预测帧' if continuous else '模型决策'}未记录 {stats['missing']} 个；无效预测跳过 {stats['invalid']} 个。",
                f"{side}: {stats['valid']} valid predictions; source steps {span}; {stats['missing']} {'frames' if continuous else 'model decisions'} without predictions; {stats['invalid']} invalid predictions skipped."))
        reason_labels = {'format': _('格式/单位不兼容', 'incompatible format/units'),
                         'perspective': _('玩家视角不匹配', 'player perspective mismatch'),
                         'numeric': _('概率/长度数值无效', 'invalid probabilities/length')}
        if summary['invalid_reasons']:
            st.warning(_('跳过原因：', 'Skipped because: ') + '；'.join(
                f'{reason_labels[reason]} × {count}' for reason, count in summary['invalid_reasons'].items()))
