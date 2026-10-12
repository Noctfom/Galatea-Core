"""卡组大盘矩阵显示：保留完整统计，按渲染预算分页着色并导出完整 CSV"""

import math


MATCHUP_STYLED_CELL_BUDGET = 10_000
MATCHUP_PAGE_SIDE = 50


def matchup_page_shape(rows, columns, styler_limit):
    """按单元格预算选择页大小，不修改 Pandas 全局配置或原始统计"""
    limit = max(0, int(styler_limit))
    budget = min(MATCHUP_STYLED_CELL_BUDGET, limit) if limit else MATCHUP_STYLED_CELL_BUDGET
    if rows * columns <= budget:
        return rows, columns
    page_columns = min(columns, MATCHUP_PAGE_SIDE, budget)
    page_rows = min(rows, MATCHUP_PAGE_SIDE, max(1, budget // max(1, page_columns)))
    return page_rows, page_columns


def render_matchup_matrix(matrix, translate, *, key):
    """显示有界矩阵页；未知对局留空，完整导出仍包含所有卡组组合"""
    import inspect
    import pandas as pd
    import streamlit as st

    _ = translate
    if matrix.empty:
        st.info(_('暂无可用的卡组克制统计。', 'No matchup statistics available.'))
        return
    rows, columns = matrix.shape
    styler_limit = int(pd.get_option('styler.render.max_elements'))
    page_rows, page_columns = matchup_page_shape(rows, columns, styler_limit)
    row_pages = math.ceil(rows / page_rows)
    column_pages = math.ceil(columns / page_columns)
    row_page = column_page = 1
    if row_pages > 1 or column_pages > 1:
        st.info(_(
            f'完整矩阵为 {rows:,} 行 × {columns:,} 列（{matrix.size:,} 个格子），当前分页着色；统计和完整导出不截断。',
            f'The full matrix has {rows:,} rows × {columns:,} columns ({matrix.size:,} cells). Styling is paginated; statistics and full exports are not truncated.'))
        controls = st.columns(2)
        for position, (pages, label, page_key) in enumerate((
            (row_pages, _('我方卡组页', 'My-deck page'), f'{key}_row_page'),
            (column_pages, _('对手卡组页', 'Opponent-deck page'), f'{key}_column_page'),
        )):
            # 训练轮次或数据源筛选后，旧页号可能越界；仅在本轮创建控件前修正
            previous = st.session_state.get(page_key, 1)
            if type(previous) is not int or not 1 <= previous <= pages:
                st.session_state[page_key] = 1
            selected = controls[position].number_input(
                label, min_value=1, max_value=pages, value=1, step=1, key=page_key,
                disabled=pages == 1, help=_(
                    f'共 {pages} 页。只切换显示区域，不改变胜率、样本数或 CSV 导出范围。',
                    f'{pages} pages. Changes only the displayed region, not win rates, sample counts or the CSV export.'))
            if position == 0:
                row_page = int(selected)
            else:
                column_page = int(selected)
        st.download_button(
            _('📥 导出完整克制矩阵 CSV', '📥 Export full matchup matrix CSV'),
            matrix.to_csv(na_rep='', encoding='utf-8-sig').encode('utf-8-sig'),
            file_name='matchup_matrix.csv', mime='text/csv', key=f'{key}_export', help=_(
                '导出当前筛选范围的全部卡组组合，数值是 0–1 胜率；空白表示没有对局记录，不是 0% 胜率。',
                'Exports every matchup in the current filter as win rates from 0–1. Blank means no recorded games, not a 0% win rate.'))
    row_start = (row_page - 1) * page_rows
    column_start = (column_page - 1) * page_columns
    page = matrix.iloc[row_start:row_start + page_rows, column_start:column_start + page_columns]
    st.caption(_(
        f'显示行 {row_start + 1}–{row_start + len(page)} / {rows}，列 {column_start + 1}–{column_start + len(page.columns)} / {columns}。比例是此筛选范围内的训练对局胜率；— 表示无对局。',
        f'Rows {row_start + 1}–{row_start + len(page)} of {rows}; columns {column_start + 1}–{column_start + len(page.columns)} of {columns}. Rates describe training games within this filter; — means no games.'))
    display = page
    if styler_limit > 0:
        # 分页后每次着色都低于预算；固定 0–1 色标使不同页的颜色具有相同含义
        display = page.style.format('{:.1%}', na_rep='—').background_gradient(
            cmap='RdYlGn', axis=None, vmin=0, vmax=1).highlight_null(color='lightgrey')
    else:
        st.caption(_('当前环境禁用了 Styler 渲染，显示原始 0–1 比例。',
                     'Styler rendering is disabled in this environment; showing raw rates from 0–1.'))
    width_args = {'width': 'stretch'} if 'width' in inspect.signature(st.dataframe).parameters else {'use_container_width': True}
    st.dataframe(display, **width_args)
