# webui
#输入命令：streamlit run app.py

import streamlit as st
import pandas as pd
import os
import glob
import subprocess
import torch
import psutil
import json
import shutil
import streamlit.components.v1 as components
import socket
import time
import sys
import tempfile
import inspect
import html
from collections import deque
from card_reader import CardReader
from checkpoint_utils import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_MODEL_PREFIX,
    MODEL_PROTOCOL_VERSION,
    inspect_training_checkpoint,
)
from model_artifacts import (
    discover_model_repository,
    discover_checkpoint_artifacts,
    find_model_prefix_namespace_files,
    install_model_artifact_bundle,
    is_primary_model_filename,
    validate_model_artifact_filename,
)
from managed_processes import (
    build_managed_process_env,
    process_identity_matches,
    process_matches_project,
    purge_managed_processes,
)
from training_validation import resolve_training_target, validate_model_prefix
from replay_utils import (
    format_action_semantics,
    get_replay_frame_state,
    get_replay_decklists,
    get_frame_visuals,
    get_replay_frames,
    get_selected_replay_option_index,
    group_replay_card_codes,
    queue_replay_cursor,
    set_replay_cursor,
    sync_replay_session,
)

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.realpath(os.path.dirname(__file__))

# 强制使用绝对路径挂载 CDB，防止 WebUI 运行目录错位导致查不到卡名
card_db_ui = CardReader(db_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'cards.cdb')))


def get_stretch_width_args(widget):
    """按当前 Streamlit 版本选择无弃用警告的拉伸宽度参数。"""
    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


REPLAY_BUTTON_WIDTH = get_stretch_width_args(st.button)
REPLAY_DATAFRAME_WIDTH = get_stretch_width_args(st.dataframe)

st.set_page_config(page_title="Galatea 司令塔", page_icon="🤖", layout="wide")

# ==========================================
# 🚀 全局版本控制与智能探测器
# ==========================================
LOCAL_VERSION = "3.6.0"  # 当前本地版本号 (每次更新时手动改一下这里)
REMOTE_VERSION_URL = "https://raw.githubusercontent.com/Noctfom/Galatea-Core/main/version.txt"

@st.cache_data(ttl=10800, show_spinner=False) # 缓存 3 小时，绝不拖慢用户启动速度
def check_remote_version():
    import requests
    try:
        # 设置极短的超时时间，防止断网时卡死
        resp = requests.get(REMOTE_VERSION_URL, timeout=3)
        if resp.status_code == 200:
            return resp.text.strip()
    except:
        pass
    return LOCAL_VERSION

remote_version = check_remote_version()

# 解析语义化版本号 X.Y.Z
def parse_version(v_str):
    try: return [int(x) for x in v_str.split(".")]
    except: return [0, 0, 0]

v_local = parse_version(LOCAL_VERSION)
v_remote = parse_version(remote_version)

has_critical_update = (v_remote[0] > v_local[0]) or (v_remote[0] == v_local[0] and v_remote[1] > v_local[1])
has_patch_update = (not has_critical_update) and (v_remote[2] > v_local[2])

# 初始化全局导航劫持状态
if "jump_to_update" not in st.session_state:
    st.session_state.jump_to_update = False

CACHE_KEYS = {
    't_steps': 5000, 't_batch': 4096, 't_mini': 256, 't_workers': 6, 't_timeout': 300,
    't_gamma': 0.998, 't_lr': 0.0001, 't_entropy': 0.03, 't_gae': 0.95, 't_clip': 0.2,
    't_device': 'auto', 't_d_model': 256, 't_n_heads': 4, 't_n_layers': 2,
    't_model_prefix': DEFAULT_MODEL_PREFIX,
    'sp_games': 100, 'sp_freq': 50
}

if 'ui_cache' not in st.session_state:
    st.session_state.ui_cache = CACHE_KEYS.copy()

def cache_val(key):
    st.session_state.ui_cache[key] = st.session_state[f"widget_{key}"]


def validate_local_asset_name(filename, *, required_suffix=None):
    """校验允许中文但不含路径、控制字符或 Windows 保留设备名的本地资产名"""
    if not isinstance(filename, str) or not filename or filename in {".", ".."}:
        raise ValueError("文件名不能为空或使用相对路径标记")
    if filename != os.path.basename(filename) or "/" in filename or "\\" in filename:
        raise ValueError("文件名不得包含目录路径")
    if filename[-1] in {" ", "."} or any(ord(char) < 32 for char in filename):
        raise ValueError("文件名包含跨平台不支持的字符")
    if len(filename.encode("utf-8")) > 200:
        raise ValueError("文件名超过 200 字节限制")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{index}" for index in range(1, 10))
    reserved.update(f"LPT{index}" for index in range(1, 10))
    if filename.split(".", 1)[0].upper() in reserved:
        raise ValueError("文件名使用了 Windows 保留设备名")
    if required_suffix and not filename.casefold().endswith(required_suffix.casefold()):
        raise ValueError(f"文件名必须以 {required_suffix} 结尾")
    return filename

# ==========================================
# 🛠️ 进程管理辅助函数与状态初始化
# ==========================================
def terminate_process(pid, expected_create_time=None):
    """跨平台安全终止进程及其所有子进程"""
    try:
        parent = psutil.Process(pid)
        if expected_create_time is None:
            if not process_matches_project(parent, PROJECT_ROOT):
                return False
        elif not process_identity_matches(parent, expected_create_time):
            return False
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    except Exception as e:
        print(f"终止进程错误: {e}")
        return False

def is_process_alive(pid, expected_create_time=None):
    """精准判断进程是否存活，剔除僵尸进程"""
    if not pid or not psutil.pid_exists(pid):
        return False
    try:
        p = psutil.Process(pid)
        if expected_create_time is None:
            if not process_matches_project(p, PROJECT_ROOT):
                return False
        elif not process_identity_matches(p, expected_create_time):
            return False
        if p.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
            return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def launch_managed_task(command):
    """启动带项目归属标记的后台任务并登记可靠的进程身份。"""
    process = subprocess.Popen(
        command,
        env=build_managed_process_env(PROJECT_ROOT),
    )
    st.session_state.running_pid = process.pid
    try:
        st.session_state.running_process_create_time = psutil.Process(
            process.pid
        ).create_time()
    except psutil.Error:
        st.session_state.running_process_create_time = None
    return process

if 'running_pid' not in st.session_state:
    st.session_state.running_pid = None
if 'running_process_create_time' not in st.session_state:
    st.session_state.running_process_create_time = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'tensorboard_pid' not in st.session_state:
    st.session_state.tensorboard_pid = None
if 'tensorboard_process_create_time' not in st.session_state:
    st.session_state.tensorboard_process_create_time = None

if st.session_state.running_pid:
    if st.session_state.running_process_create_time is None:
        try:
            registered_process = psutil.Process(st.session_state.running_pid)
            if process_matches_project(registered_process, PROJECT_ROOT):
                st.session_state.running_process_create_time = registered_process.create_time()
            else:
                st.session_state.running_pid = None
        except psutil.Error:
            st.session_state.running_pid = None
    if not is_process_alive(
        st.session_state.running_pid,
        st.session_state.running_process_create_time,
    ):
        st.session_state.running_pid = None
        st.session_state.running_process_create_time = None
        st.session_state.auto_refresh = False # 进程结束，自动关闭刷新开关

# ==========================================
# 🧠 全局辅助：读取存档身份与模型架构
# 🌟 必须写在最外层全局，否则缓存会失效
# ==========================================
@st.cache_data(show_spinner=False)
def load_model_metadata(path):
    """读取恢复训练所需的协议版本、UUID、前缀、轮次和架构信息"""
    try:
        metadata = inspect_training_checkpoint(path, map_location='cpu')
        metadata['load_error'] = None
        return metadata
    except Exception as e:
        print(f"读取配置失败: {e}")
        return {'load_error': str(e), 'net_config': {}}


def get_model_identity_signature(model_dir):
    """生成模型身份相关文件的轻量缓存签名，文件变化时自动失效"""
    paths = []
    for pattern in ("*.pth", "*.onnx", "*.onnx.data", "*.artifacts.json"):
        paths.extend(glob.glob(os.path.join(model_dir, pattern)))
    records = []
    for path in paths:
        try:
            records.append((path, os.path.getmtime(path), os.path.getsize(path)))
        except OSError:
            continue
    return tuple(sorted(records))


@st.cache_data(show_spinner=False)
def load_folder_model_identities(model_dir, file_signature):
    """缓存目录身份扫描，避免 WebUI 重绘时反复读取孤立大检查点"""
    del file_signature
    return discover_checkpoint_artifacts(
        model_dir,
        include_orphan_checkpoints=True,
    )


@st.cache_data(show_spinner=False)
def load_model_repository(model_dir, file_signature):
    """缓存按 UUID 和轮次整理的模型仓库视图，避免重复解析大文件"""
    del file_signature
    return discover_model_repository(model_dir)

# ==========================================
# 🎨 前端 CSS 魔法
# ==========================================
st.markdown("""
<style>
/* 🔪 精准隐藏 Deploy 按钮 */
.stAppDeployButton {display: none !important;}
/* 🔪 隐藏底部 Streamlit 水印 */
footer {visibility: hidden !important;}
/* 🛡️ 强制保护侧边栏展开按钮 */
[data-testid="collapsedControl"] {visibility: visible !important; display: flex !important; z-index: 9999 !important;}
.stDataFrame {width: 100%;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🌐 国际化与侧边栏
# ==========================================
lang = st.sidebar.radio("🌐 Language / 语言", ["🇨🇳 中文", "🇺🇸 English"])
def _(zh, en): return zh if lang == "🇨🇳 中文" else en

menu_options = [
    _("📈 卡组生态大盘", "📈 Meta Dashboard"), 
    _("📉 训练流形图", "📉 TensorBoard"), 
    _("⚔️ 启动与监控中枢", "⚔️ Control & Logs"),
    _("🗃️ 资产与卡组管理", "🗃️ Assets & Decks"),  
    _("🔄 资源同步中枢", "🔄 Update Manager"),  
    _("🧠 语义知识库引擎", "🧠 Semantic KB Engine"), 
    _("📁 存储与日志仓库", "📁 Storage & Logs"), 
    _("👁️ 全息读心回放", "👁️ Holographic Replay"),
    _("📦 模型部署与打包", "📦 Model Deployment")
]

# 如果检测到跳转指令，强行覆盖当前菜单的 index
if st.session_state.jump_to_update:
    st.session_state.main_menu_radio = _("🔄 资源同步中枢", "🔄 Update Manager")
    st.session_state.jump_to_update = False

st.sidebar.title(_("🎛️ 司令塔菜单", "🎛️ Command Menu"))
# 🌟 核心：绑定 key="main_menu_radio"
menu = st.sidebar.radio("Navigation", menu_options, key="main_menu_radio", label_visibility="collapsed")

csv_path = "./web_data/match_history.csv"

if has_critical_update:
    st.warning(f"🚀 **{_('发现重要核心更新！', 'Critical Update Available!')}** | {_('当前版本:', 'Current:')} `v{LOCAL_VERSION}` ➡️ {_('最新版本:', 'Latest:')} `v{remote_version}`", icon="✨")
    if st.button("👉 " + _("立即前往同步中枢查看详情", "Go to Update Manager"), type="primary"):
        st.session_state.jump_to_update = True
        st.rerun()
elif has_patch_update:
    # 小修小补只给个极其轻量的提示
    st.toast(f"ℹ️ {_('检测到小版本更新', 'Minor patch available')}: v{remote_version}", icon="🔧")

# ==========================================
# 📈 模块一：卡组生态大盘 (代码保持不变)
# ==========================================
if menu == _("📈 卡组生态大盘", "📈 Meta Dashboard"):
    st.title(_("📈 实时 Meta 卡组天梯", "📈 Live Meta Tier List"))
    if st.sidebar.button(_("🔄 刷新数据", "🔄 Refresh Data"), use_container_width=True):
        st.rerun()

    data_dir = "./web_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')], reverse=True)

    @st.cache_data(show_spinner=False, max_entries=10)
    def load_csv_cached(filepath, mtime):
        # 参数中传入 mtime(文件最后修改时间)。只要文件没被训练进程写入新数据，就直接走内存秒读！
        temp_df = pd.read_csv(filepath)
        run_id = os.path.basename(filepath).replace("match_history_", "").replace(".csv", "")
        temp_df['run_id'] = run_id
        return temp_df

    if not csv_files:
        st.info(_("数据库为空，等待训练进程写入数据...", "Database is empty. Waiting for training process..."))
    else:
        # 🌟 新增：多选数据源
        selected_csvs = st.multiselect(_("📂 选择要分析的对局数据源 (可多选对比)", "📂 Select Match Data Sources"), csv_files, default=[csv_files[0]] if csv_files else [])
        
        if not selected_csvs:
            st.warning(_("请至少选择一个数据源进行分析。", "Please select at least one data source."))
        else:
            # 🌟 新增：合并多个 CSV，并打上来源标签
            all_dfs = []
            for f in selected_csvs:
                filepath = os.path.join(data_dir, f)
                mtime = os.path.getmtime(filepath) # 获取文件当前的物理修改时间
                # 如果文件没变，这行代码只需不到 1 毫秒！
                all_dfs.append(load_csv_cached(filepath, mtime)) 
            
            df = pd.concat(all_dfs, ignore_index=True)

            min_iter, max_iter = int(df['iteration'].min()), int(df['iteration'].max())
            if min_iter == max_iter:
                st.info(_("当前数据仅包含一个迭代轮次，无法选择范围。", "Only one iteration available, range selection disabled."))
                selected_range = (min_iter, max_iter)   # 单点范围
            else:
                # 🌟 核心修复：用 form 表单上锁，阻断拖动时的疯狂重绘
                st.markdown(_("⏳ **选择分析的训练轮次范围**", "⏳ **Select Iteration Range**"))
                with st.form("range_filter_form"):
                    c_slider, c_btn = st.columns([8, 2])
                    with c_slider:
                        selected_range = st.slider(
                            "Range", # Label 隐藏掉
                            min_value=min_iter, max_value=max_iter, value=(max(min_iter, max_iter-200), max_iter),
                            label_visibility="collapsed"
                        )
                    with c_btn:
                        st.form_submit_button("✅ " + _("应用筛选", "Apply"), use_container_width=True)
                        
            filtered_df = df[(df['iteration'] >= selected_range[0]) & (df['iteration'] <= selected_range[1])]
            
            if filtered_df.empty:
                st.warning(_("选定范围内无数据", "No data in selected range."))
            else:
                main_tab_overview, main_tab_compare= st.tabs([
                    _("📊 综合大盘战报", "📊 Overall Dashboard"), 
                    _("⚔️ 双卡组胜率对比", "⚔️ Deck Comparison")
                ])
                
                with main_tab_overview:
                    env_pools = filtered_df['env'].unique().tolist()
                    sub_tabs = st.tabs(env_pools)
                    for tab, env in zip(sub_tabs, env_pools):
                        with tab:
                            env_df = filtered_df[filtered_df['env'] == env]
                            st.markdown(_(f"### 📊 【{env}】卡池综合战报", f"### 📊 [{env}] Pool Overview"))
                            
                            stats = env_df.groupby('my_deck').agg(
                                总场数=('is_win', 'count'),
                                总胜场=('is_win', 'sum'),
                                先手场数=('is_first', 'sum')
                            ).reset_index()
                            
                            first_wins = env_df[env_df['is_first'] == True].groupby('my_deck')['is_win'].sum().reset_index(name='先手胜场')
                            second_wins = env_df[env_df['is_first'] == False].groupby('my_deck')['is_win'].sum().reset_index(name='后手胜场')
                            second_games = env_df[env_df['is_first'] == False].groupby('my_deck')['is_first'].count().reset_index(name='后手场数')
                            
                            stats = stats.merge(first_wins, on='my_deck', how='left').merge(second_wins, on='my_deck', how='left').merge(second_games, on='my_deck', how='left').fillna(0)
                            stats['总胜率'] = stats['总胜场'] / stats['总场数']
                            stats['先手胜率'] = stats['先手胜场'] / stats['先手场数'].replace(0, 1)
                            stats['后手胜率'] = stats['后手胜场'] / stats['后手场数'].replace(0, 1)
                            stats = stats[stats['总场数'] >= 5]
                            
                            if not stats.empty:
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.caption(_("🏆 综合胜率榜", "🏆 Overall Win Rate"))
                                    df_overall = stats[['my_deck', '总场数', '总胜率']].sort_values(by='总胜率', ascending=False)
                                    st.dataframe(df_overall.style.format({'总胜率': "{:.1%}"}).background_gradient(cmap='RdYlGn', subset=['总胜率']), hide_index=True)
                                with col_b:
                                    st.caption(_("🥇 先手胜率榜", "🥇 Going 1st Win Rate"))
                                    df_first = stats[['my_deck', '先手场数', '先手胜率']].sort_values(by='先手胜率', ascending=False)
                                    st.dataframe(df_first.style.format({'先手胜率': "{:.1%}"}).background_gradient(cmap='Greens', subset=['先手胜率']), hide_index=True)
                                with col_c:
                                    st.caption(_("🥈 后手突破榜", "🥈 Going 2nd Win Rate"))
                                    df_second = stats[['my_deck', '后手场数', '后手胜率']].sort_values(by='后手胜率', ascending=False)
                                    st.dataframe(df_second.style.format({'后手胜率': "{:.1%}"}).background_gradient(cmap='Oranges', subset=['后手胜率']), hide_index=True)
                                
                                st.divider()
                                st.markdown(_("#### ⚔️ 卡组克制矩阵 (行: 我方, 列: 敌方)", "#### ⚔️ Matchup Matrix (Row: My Deck, Col: Opponent)"))
                                pivot_df = env_df.pivot_table(index='my_deck', columns='opp_deck', values='is_win', aggfunc='mean')
                                st.dataframe(pivot_df.style.format("{:.1%}").background_gradient(cmap='RdYlGn', axis=None).highlight_null(color='lightgrey'))
                            else:
                                st.info(_("该池子有效对局数不足。", "Not enough valid games in this pool."))
                                
                with main_tab_compare:
                    st.markdown(_("### ⚔️ 双卡组核心指标 PK", "### ⚔️ Dual Deck PK"))
                    
                    env_pools_compare = filtered_df['env'].unique().tolist()
                    if not env_pools_compare:
                        st.info(_("无可用的环境池数据。", "No environment pools available."))
                    else:
                        selected_env = st.selectbox(_("1️⃣ 首先，选择要对比的环境池", "1️⃣ First, Select Environment Pool"), env_pools_compare, key="compare_env_select")
                        compare_df = filtered_df[filtered_df['env'] == selected_env]
                        all_decks_in_env = sorted(compare_df['my_deck'].unique().tolist())
                        
                        if len(all_decks_in_env) >= 2:
                            st.markdown(_("### 2️⃣ 接下来，选择要对决的卡组", "### 2️⃣ Next, Select Decks"))
                            col_pk1, col_pk2 = st.columns(2)
                            with col_pk1:
                                deck_a = st.selectbox(_("选择卡组 A (蓝方)", "Select Deck A (Blue)"), all_decks_in_env, index=0)
                            with col_pk2:
                                deck_b = st.selectbox(_("选择卡组 B (红方)", "Select Deck B (Red)"), all_decks_in_env, index=1 if len(all_decks_in_env) > 1 else 0)
                                
                            if deck_a and deck_b:
                                df_a = compare_df[compare_df['my_deck'] == deck_a]
                                df_b = compare_df[compare_df['my_deck'] == deck_b]
                                
                                def calc_metrics(df_sub):
                                    total = len(df_sub)
                                    wins = df_sub['is_win'].sum()
                                    first_games = len(df_sub[df_sub['is_first'] == True])
                                    first_wins = df_sub[df_sub['is_first'] == True]['is_win'].sum()
                                    second_games = len(df_sub[df_sub['is_first'] == False])
                                    second_wins = df_sub[df_sub['is_first'] == False]['is_win'].sum()
                                    return {
                                        "total": total,
                                        "win_rate": (wins / total) if total > 0 else 0,
                                        "first_wr": (first_wins / first_games) if first_games > 0 else 0,
                                        "second_wr": (second_wins / second_games) if second_games > 0 else 0
                                    }
                                    
                                metrics_a = calc_metrics(df_a)
                                metrics_b = calc_metrics(df_b)
                                
                                st.markdown("---")
                                c1, c2, c3 = st.columns(3)
                                
                                c1.metric(label=_("总胜率对比", "Overall Win Rate"), 
                                          value=f"{deck_a}: {metrics_a['win_rate']:.1%}", 
                                          delta=f"{metrics_a['win_rate'] - metrics_b['win_rate']:.1%} (vs {deck_b})",
                                          delta_color="normal" if metrics_a['win_rate'] >= metrics_b['win_rate'] else "inverse")
                                c1.caption(f"{deck_b}: {metrics_b['win_rate']:.1%}")
                                
                                c2.metric(label=_("先手压制力对比", "Going 1st WR"), 
                                          value=f"{deck_a}: {metrics_a['first_wr']:.1%}", 
                                          delta=f"{metrics_a['first_wr'] - metrics_b['first_wr']:.1%} (vs {deck_b})",
                                          delta_color="normal" if metrics_a['first_wr'] >= metrics_b['first_wr'] else "inverse")
                                c2.caption(f"{deck_b}: {metrics_b['first_wr']:.1%}")
                                
                                c3.metric(label=_("后手突破力对比", "Going 2nd WR"), 
                                          value=f"{deck_a}: {metrics_a['second_wr']:.1%}", 
                                          delta=f"{metrics_a['second_wr'] - metrics_b['second_wr']:.1%} (vs {deck_b})",
                                          delta_color="normal" if metrics_a['second_wr'] >= metrics_b['second_wr'] else "inverse")
                                c3.caption(f"{deck_b}: {metrics_b['second_wr']:.1%}")
                                
                                st.markdown("---")
                                st.markdown(_("#### 🥊 直接交锋 (Head-to-Head)", "#### 🥊 Head-to-Head"))
                                h2h_df = df_a[df_a['opp_deck'] == deck_b]
                                h2h_games = len(h2h_df)
                                if h2h_games > 0:
                                    h2h_wins = h2h_df['is_win'].sum()
                                    st.info(f"在限定轮次内，**{deck_a}** 与 **{deck_b}** 共交手 **{h2h_games}** 次。")
                                    st.success(f"**{deck_a}** 赢了 **{h2h_wins}** 次 (胜率: {h2h_wins/h2h_games:.1%})")
                                else:
                                    st.info(_("在限定轮次内，这两个卡组没有直接交手记录。", "No direct matches between these two decks in the selected range."))
                        else:
                            st.info(_("所选环境池中卡组种类不足 2 个，无法进行对比。", "Not enough deck types in this pool for comparison."))

# ==========================================
# ⚔️ 模块二：启动与监控中枢
# ==========================================
elif menu == _("⚔️ 启动与监控中枢", "⚔️ Control & Logs"):
    st.title(_("🚀 进程控制与全息监控", "🚀 Process Control & Live Logs"))
    
    # 新增：全局进程控制台
    is_running = False
    if st.session_state.running_pid:
        is_running = is_process_alive(
            st.session_state.running_pid,
            st.session_state.running_process_create_time,
        )
        if not is_running:
            st.session_state.running_pid = None
            st.session_state.running_process_create_time = None
            
    col_status, col_kill, col_purge = st.columns([6, 2, 2])
    with col_status:
        if is_running:
            st.success(f"🟢 {_('检测到后台正在运行任务', 'Task running')} (PID: {st.session_state.running_pid})")
        else:
            st.info(f"⚪ {_('当前没有任务在后台运行', 'No tasks running')}")
    
    with col_kill:
        if is_running:
            if st.button("🛑 " + _("紧急制动", "Abort"), type="primary", use_container_width=True):
                terminate_process(
                    st.session_state.running_pid,
                    st.session_state.running_process_create_time,
                )
                st.session_state.running_pid = None
                st.session_state.running_process_create_time = None
                st.rerun()

    with col_purge:
        if st.button(
            "🧨 " + _("净化僵尸进程", "Purge Zombies"),
            use_container_width=True,
            help=_(
                "仅终止由本项目 WebUI 标记的后台任务及其子进程，不会终止其他 Python 程序或当前 WebUI。",
                "Only stop background processes owned by this project; unrelated Python tasks and this UI are preserved.",
            ),
        ):
            result = purge_managed_processes(
                PROJECT_ROOT,
                known_root_pid=st.session_state.running_pid,
                known_root_create_time=st.session_state.running_process_create_time,
            )
            if result["matched_pids"]:
                st.session_state.running_pid = None
                st.session_state.running_process_create_time = None
                st.session_state.auto_refresh = False
                st.success(
                    _(
                        f"已清理本项目托管进程: {result['matched_pids']}",
                        f"Purged project-owned processes: {result['matched_pids']}",
                    )
                )
            else:
                st.info(_("未发现本项目托管的残留进程。", "No project-owned zombie process was found."))
            if result["failed"]:
                st.warning(_(f"部分进程清理失败: {result['failed']}", f"Some processes could not be stopped: {result['failed']}"))
    st.divider()
    
    tab_train, tab_duel, tab_selfcheck = st.tabs([
        _("🔥 发起训练 (Train)", "🔥 Start Training"), 
        _("🏟️ 发起竞技 (Duel)", "🏟️ Start Arena"),
        _("🛠️ 规则自检压测 (Self-Check)", "🛠️ Rules Self-Check") # <--- 新增
    ])
    models = ["None"] + sorted(
        (
            path
            for path in glob.glob("./models/*.pth")
            if is_primary_model_filename(os.path.basename(path))
            and not os.path.islink(path)
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    
    # --- 🔥 训练控制台 ---
    with tab_train:
        # 🌟 核心修复：将选择框移出表单！这样一选中就会立刻刷新网页！
        st.markdown("### 📂 " + _("存档加载器", "Checkpoint Loader"))
        t_resume = st.selectbox(
            _("选择恢复存档 (Resume)", "Resume Checkpoint"), models, 
            help=_("选择一个之前训练好的 .pth 文件继续训练。如果选 None 则是从零开始。", "Select a checkpoint to resume training.")
        )
        
        is_resume = (t_resume != "None")
        folder_identity_records = load_folder_model_identities(
            "./models",
            get_model_identity_signature("./models"),
        )
        saved_meta = load_model_metadata(t_resume) if is_resume else {}
        saved_cfg = saved_meta.get('net_config', {})
        resume_metadata_error = saved_meta.get('load_error') if is_resume else None
        resume_format_warning = saved_meta.get('format_warning') if is_resume else None
        resume_model_protocol_warning = (
            saved_meta.get('model_protocol_warning') if is_resume else None
        )

        if resume_metadata_error:
            st.error(_(
                f"无法读取检查点元数据: {resume_metadata_error}",
                f"Unable to read checkpoint metadata: {resume_metadata_error}",
            ))
        elif resume_format_warning:
            st.warning(_(
                f"⚠️ {resume_format_warning}",
                f"⚠️ {resume_format_warning}",
            ))
        elif resume_model_protocol_warning:
            st.warning(_(
                f"⚠️ {resume_model_protocol_warning}",
                f"⚠️ {resume_model_protocol_warning}",
            ))
        
        # 将读取到的配置安全转化为整数，防止格式报错
        default_d_model = int(saved_cfg.get('d_model', 256))
        default_n_heads = int(saved_cfg.get('n_heads', 4))
        default_n_layers = int(saved_cfg.get('n_layers', 2))

        # 只有按下这个按钮才提交训练
        # --- 模型架构参数区 ---
        with st.expander("🧠 " + _("模型架构参数 (Model Architecture)", "Model Architecture"), expanded=not is_resume):
            if is_resume:
                st.success(_("🔒 已选择恢复存档，架构参数已自动读取并锁定！", "Architecture locked to the selected checkpoint."))

            default_model_prefix = (
                saved_meta.get('model_prefix')
                if is_resume and saved_meta.get('model_prefix')
                else st.session_state.ui_cache['t_model_prefix']
            )
            t_model_prefix = st.text_input(
                _("模型前缀 (Model Prefix)", "Model Prefix"),
                value=default_model_prefix,
                disabled=is_resume,
                key=f"widget_t_model_prefix_{t_resume}",
                help=_(
                    "仅允许字母、数字、下划线和短横线，最长 64 字符。恢复训练时由检查点自动继承。",
                    "Letters, digits, underscores and hyphens only, up to 64 characters. Inherited on resume.",
                ),
            )
            if not is_resume:
                st.session_state.ui_cache['t_model_prefix'] = t_model_prefix

            identity_conflicts = []
            prefix_namespace_files = []
            if is_resume and saved_meta.get('model_id') and saved_meta.get('model_prefix'):
                identity_conflicts = [
                    record
                    for record in folder_identity_records
                    if record['model_prefix'].casefold()
                    == saved_meta['model_prefix'].casefold()
                    and record['model_id'] != saved_meta['model_id']
                ]
            elif not is_resume:
                try:
                    validate_model_prefix(t_model_prefix)
                    identity_conflicts = [
                        record
                        for record in folder_identity_records
                        if record['model_prefix'].casefold()
                        == t_model_prefix.casefold()
                    ]
                    prefix_namespace_files = find_model_prefix_namespace_files(
                        "./models",
                        t_model_prefix,
                    )
                except ValueError:
                    identity_conflicts = []

            if is_resume and saved_meta.get('model_id'):
                st.caption(_(
                    f"模型 UUID（只读）: {saved_meta['model_id']} | 当前轮次: {saved_meta.get('iteration')} | 检查点协议: {saved_meta.get('checkpoint_format_version')}/{CHECKPOINT_FORMAT_VERSION} | 模型协议: {saved_meta.get('model_protocol_version')}/{MODEL_PROTOCOL_VERSION}",
                    f"Model UUID (read-only): {saved_meta['model_id']} | Current iteration: {saved_meta.get('iteration')} | Checkpoint protocol: {saved_meta.get('checkpoint_format_version')}/{CHECKPOINT_FORMAT_VERSION} | Model protocol: {saved_meta.get('model_protocol_version')}/{MODEL_PROTOCOL_VERSION}",
                ))
            
            m1, m2, m3 = st.columns(3)
            with m1:
                t_d_model = st.number_input("d_model", value=default_d_model, step=64, disabled=is_resume, 
                                            help=_("卡片向量的维度。越高越聪明，但计算越慢。必须能被 n_heads 整除。", "Feature dimension. Must be divisible by n_heads."))
            with m2:
                t_n_heads = st.number_input("n_heads", value=default_n_heads, step=1, disabled=is_resume, 
                                            help=_("注意力头数。AI同时观察局面的视角数量(例如有的头看手牌，有的头看墓地)。", "Num of attention heads."))
            with m3:
                t_n_layers = st.number_input("n_layers", value=default_n_layers, step=1, disabled=is_resume,
                                                help=_("神经网络的思考深度。2层适合简单尝试，6层适合复杂的长线战术推演。", "Num of Transformer layers."))

        if identity_conflicts:
            conflict_ids = sorted({item['model_id'] for item in identity_conflicts})
            st.warning(_(
                f"目录中存在同前缀但不同 UUID 的模型: {conflict_ids}。训练加载会按 UUID 严格隔离。",
                f"The folder contains the same prefix under different UUIDs: {conflict_ids}. Loading is UUID-isolated.",
            ))
        elif prefix_namespace_files:
            st.warning(_(
                f"该前缀的规范文件名空间已被现有检查点占用: {prefix_namespace_files}",
                f"The canonical filename namespace is already occupied: {prefix_namespace_files}",
            ))

        st.markdown("---")
        
        # --- 训练超参数区 ---
        st.markdown("### ⚙️ " + _("训练环境配置", "Training Hyperparameters"))
        # 1. 常规配置区 (恢复三列布局，把 device 加回来)
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            iteration_mode = st.radio(
                _("轮次计算方式", "Iteration Mode"),
                [_("训练至指定轮次", "Train to iteration"), _("追加训练轮数", "Additional iterations")],
                horizontal=True,
                help=_("两种方式互斥，不再自动追加 1000 轮。", "The modes are exclusive; no implicit 1000-iteration extension."),
            )
            t_steps = st.number_input(_("轮次数值", "Iteration Value"),
                                    value=st.session_state.ui_cache['t_steps'], step=100,
                                    key="widget_t_steps", on_change=cache_val, args=('t_steps',),
                                    help=_("根据上方模式解释为绝对停止轮次或追加轮数。", "Interpreted as an absolute target or additional count."))
            t_workers = st.number_input(_("进程数 (CPU Workers)", "CPU Workers"), 
                                        value=st.session_state.ui_cache['t_workers'], min_value=1, max_value=32, 
                                        key="widget_t_workers", on_change=cache_val, args=('t_workers',),
                                        help=_("同时开启几个后台 YGOPro 环境。不要超过 CPU 物理核心数！", "Number of parallel environment processes."))
        with col_r2:
            t_batch = st.number_input(_("总经验池 (Batch Size)", "Batch Size"), 
                                    value=st.session_state.ui_cache['t_batch'], step=512, 
                                    key="widget_t_batch", on_change=cache_val, args=('t_batch',),
                                    help=_("每次网络更新前收集的总步数。越大越稳定，但内存需求大。", "Total steps before updating policy."))
            t_timeout = st.number_input(_("超时强杀 (Timeout/s)", "Collection Timeout"), 
                                        value=st.session_state.ui_cache['t_timeout'], step=10, 
                                        key="widget_t_timeout", on_change=cache_val, args=('t_timeout',),
                                        help=_("单个 Worker 采集数据的最长等待时间，防止进程僵死。", "Max time to wait for a worker to collect data."))
        with col_r3:
            t_mini = st.number_input(_("切片大小 (Mini Batch)", "Mini Batch"),
                                    value=st.session_state.ui_cache['t_mini'], step=64,
                                    key="widget_t_mini", on_change=cache_val, args=('t_mini',),
                                    help=_("PPO 梯度下降时每次送入所选训练设备的数据量。", "Data slice size for updates on the selected training device."))
            training_devices = ["auto", "cpu", "cuda"]
            cached_device = st.session_state.ui_cache['t_device']
            device_index = (
                training_devices.index(cached_device)
                if cached_device in training_devices
                else 0
            )
            t_device = st.selectbox(_("训练设备", "Training Device"), training_devices,
                                    index=device_index, key="widget_t_device", on_change=cache_val, args=('t_device',),
                                    help=_("auto 自动使用可用 CUDA，否则仅用 CPU；所有 Worker 始终使用 CPU。", "Auto uses CUDA when available; all workers always stay on CPU."))

        # 高级超参数区 (完整找回 t_gae 和 t_clip)
        with st.expander(_("🛠️ 深度学习核心超参数", "Advanced Hyperparameters")):
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                t_gamma = st.number_input(_("折扣因子 (Gamma)", "Gamma"), 
                                            value=float(st.session_state.ui_cache['t_gamma']), format="%.3f", step=0.001, 
                                            key="widget_t_gamma", on_change=cache_val, args=('t_gamma',),
                                            help=_("目光远视程度。推荐 0.998 以应对长盘对局。", "Discount factor for future rewards."))
                t_clip = st.number_input(_("截断阈值 (Clip)", "PPO Clip"), 
                                            value=float(st.session_state.ui_cache['t_clip']), format="%.2f", step=0.05, 
                                            key="widget_t_clip", on_change=cache_val, args=('t_clip',),
                                            help=_("限制单次更新的策略变动幅度，防止学‘飘’了。", "PPO policy clipping epsilon."))
            with hc2:
                t_lr = st.number_input(_("学习率 (LR)", "Learning Rate"), 
                                        value=float(st.session_state.ui_cache['t_lr']), format="%.5f", step=0.00001, 
                                        key="widget_t_lr", on_change=cache_val, args=('t_lr',),
                                        help=_("大脑神经元重塑速度。太大会导致训练震荡。", "Adam optimizer learning rate."))
                t_entropy = st.number_input(_("探索系数 (Entropy)", "Entropy Coef"), 
                                            value=float(st.session_state.ui_cache['t_entropy']), format="%.3f", step=0.005, 
                                            key="widget_t_entropy", on_change=cache_val, args=('t_entropy',),
                                            help=_("鼓励 AI 尝试新操作；训练期间保持所设定的系数。", "Encourages exploration and remains at the configured value during training."))
            with hc3:
                t_gae = st.number_input(_("GAE Lambda", "GAE Lambda"), 
                                        value=float(st.session_state.ui_cache['t_gae']), format="%.2f", step=0.01, 
                                        key="widget_t_gae", on_change=cache_val, args=('t_gae',),
                                        help=_("广义优势估计参数。用于平衡预测的偏差和方差。", "Generalized Advantage Estimation parameter."))
        
        # --- 高级开关 ---
        st.write(_("⚡ 高级开关", "⚡ Advanced Toggles"))
        st.caption(_(
            "中央批量推理服务固定启用；采集 Worker 固定使用 CPU。",
            "Central batched inference is always enabled; collection workers always use CPU.",
        ))
        c_nocomp = st.checkbox(_("禁用模型编译 (--no_compile)", "Disable Torch Compile"), value=True, 
                                help=_("Windows 系统下 PyTorch 2.0+ 的 compile 极易报错，勾选此项牺牲 5% 速度换取绝对稳定。", "Disable torch.compile for Windows compatibility."))
        c_onnx = st.checkbox(_("同时导出 ONNX 静态模型 (--use_onnx)", "Export ONNX Model (--use_onnx)"), value=True, 
                             help=_("开启后会在保存 Checkpoint 时同步导出优化后的 .onnx 文件，大幅加速老模型对打和自决斗时的 CPU 推理速度。", 
                                   "Export .onnx graphs synchronously to accelerate CPU inference during arena duels."))
        c_std_core = st.checkbox(_("关闭幽灵字节解析 (--standard_core)", "Disable Ghost Byte"), value=False, 
                                 help=_("如果使用自编译的无幽灵字节内核(Standard Core)，请勾选此项以防止解析错位。", "Check this if using a custom core without ghost bytes at 16/31 messages."))
        
        if st.button("🔥 " + _("在后台启动训练", "Start Training Process"), use_container_width=True):
            if is_running:
                st.error(_("⚠️ 请先终止当前任务！", "⚠️ Stop current task first!"))
            else:
                launch_error = None
                try:
                    if t_device == "cuda" and not torch.cuda.is_available():
                        raise RuntimeError("当前 PyTorch 环境无法使用 CUDA，请选择 auto 或 cpu")
                    validate_model_prefix(t_model_prefix)
                    if is_resume:
                        if resume_metadata_error:
                            raise ValueError(resume_metadata_error)
                        if resume_format_warning:
                            raise ValueError(resume_format_warning)
                        if resume_model_protocol_warning:
                            raise ValueError(resume_model_protocol_warning)
                        if not saved_meta.get('model_id'):
                            raise ValueError("检查点缺少自动生成的 model_id")
                    elif identity_conflicts or prefix_namespace_files:
                        raise PermissionError(
                            "新模型前缀已被现有 UUID 使用，请更换模型前缀"
                        )
                    current_iteration = int(saved_meta.get('iteration', 0)) if is_resume else 0
                    if iteration_mode == _("训练至指定轮次", "Train to iteration"):
                        resolve_training_target(
                            current_iteration,
                            target_iteration=int(t_steps),
                        )
                    else:
                        resolve_training_target(
                            current_iteration,
                            additional_iterations=int(t_steps),
                        )
                except (ValueError, PermissionError, RuntimeError) as error:
                    launch_error = str(error)

                if launch_error:
                    st.error(_(
                        f"训练参数校验失败: {launch_error}",
                        f"Training validation failed: {launch_error}",
                    ))
                    st.stop()

                iteration_flag = (
                    "--target-iteration"
                    if iteration_mode == _("训练至指定轮次", "Train to iteration")
                    else "--additional-iterations"
                )
                cmd = [
                    sys.executable, "main.py", "train",
                    iteration_flag, str(t_steps),
                    "--model-prefix", str(t_model_prefix),
                    "--batch_size", str(t_batch),
                    "--mini_batch", str(t_mini),
                    "--workers", str(t_workers),
                    "--device", str(t_device),
                    "--d_model", str(int(t_d_model)),                # 🌟 补回
                    "--n_heads", str(int(t_n_heads)),                # 🌟 补回
                    "--n_layers", str(int(t_n_layers)),              # 🌟 补回
                    "--timeout", str(t_timeout),
                    "--gamma", str(t_gamma),
                    "--lr", str(t_lr),
                    "--entropy", str(t_entropy),
                    "--gae_lambda", str(t_gae),
                    "--clip_eps", str(t_clip)
                ]
                if t_resume != "None": cmd.extend(["--resume", t_resume])
                if c_nocomp: cmd.append("--no_compile")
                if c_onnx: cmd.append("--use_onnx")
                if c_std_core: cmd.append("--standard_core")
                p = launch_managed_task(cmd)
                st.success(_(f"指令已发送 (PID: {p.pid})！", f"Dispatched (PID: {p.pid})!"))
                time.sleep(0.5)
                st.rerun()

    # --- 🏟️ 竞技场表单 ---
    with tab_duel:
        st.markdown("### ⚔️ " + _("发起模型对决", "Initiate Model Duel"))
        
        # 同样把竞技场的出战模型拉到表单外面
        c_p0, c_p1 = st.columns(2)
        with c_p0:
            d_p0 = st.selectbox(_("出战模型 (P0)", "Select P0 Model"), models, index=1 if len(models)>1 else 0)
        with c_p1:
            d_p1 = st.selectbox(_("守擂模型/规则 (P1)", "Select P1 Model/RuleBot"), models, index=0, help=_("如果选 None，对手就是内置的规则脚本 (RuleBot)。", "Opponent. None means RuleBot."))
            
        with st.form("duel_form"):
            c1, c2 = st.columns(2)
            with c1:
                d_num = st.number_input(_("对战局数", "Number of Games"), value=100, step=10, 
                                        help=_("让双方在竞技场打多少局。打完会在终端打印胜负原因统计。", "Total games to play."))
            with c2:
                d_freq = st.number_input("🧠 " + _("读心频率 (导出 JSON)", "Thought Log Freq"), value=5, 
                                         help=_("每隔 N 局保存一次极其详尽的 AI 脑电波日志（存放于 ./ai_thoughts/），用于后续读心回放复盘。设为 0 关闭。", "Save AI probability dist every N games for replay."))
            d_std_core = st.checkbox(_("关闭幽灵字节解析 (--standard_core)", "Disable Ghost Byte"), value=False)
            
            if st.form_submit_button("⚔️ " + _("在后台启动竞技场", "Start Arena Process"), use_container_width=True):
                if is_running:
                    st.error(_("⚠️ 请先终止当前任务！", "⚠️ Stop current task first!"))
                elif d_p0 != "None":
                    if d_p0 != "None":
                        cmd = [sys.executable, "main.py", "duel", "--p0", d_p0, "--num", str(d_num), "--thought_freq", str(d_freq)]
                        if d_p1 != "None": cmd.extend(["--p1", d_p1])
                        if d_std_core: cmd.append("--standard_core")
                        p = launch_managed_task(cmd)
                        st.success(_(f"竞技场启动 (PID: {p.pid})！", f"Arena started (PID: {p.pid})!"))
                        
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(_("请至少为 P0 选择一个有效的出战模型！", "Please select a valid P0 model!"))

    # --- 🛠️ 规则自检测试舱 ---
    with tab_selfcheck:
        st.markdown("### 🛠️ 核心规则系统与环境压测")
        st.info(_("通过让内置的纯规则 Bot (RuleBot) 进行超高频盲打，可以瞬间检验卡片脚本、底层 C++ 引擎以及消息解析队列是否存在死锁或崩溃 Bug。", 
                  "Run high-speed RuleBot self-play to debug engine, Lua scripts, and message parsers without neural network overhead."))
        
        c_check, c_run = st.columns([1, 1])
        with c_check:
            st.markdown("**🔍 运行环境基准探测**")
            import platform
            dll_name = "ocgcore.dll" if platform.system() == "Windows" else "ocgcore.so"
            st.write(f"⚙️ 核心引擎 (`{dll_name}`): ", "✅ 存在" if os.path.exists(f"./{dll_name}") else "❌ 缺失")
            st.write("🗃️ 卡片数据库 (`cards.cdb`): ", "✅ 存在" if os.path.exists("./cards.cdb") else "❌ 缺失")
            st.write("📜 脚本目录 (`./script/`): ", "✅ 存在" if os.path.exists("./script") else "❌ 缺失")
        
        with c_run:
            with st.form("selfcheck_form"):
                sc_num = st.number_input(_("极端压测局数", "Number of Games"), min_value=1, value=50, step=10)
                sc_std_core = st.checkbox(_("关闭幽灵字节解析 (--standard_core)", "Disable Ghost Byte"), value=False)

                if st.form_submit_button("🚀 " + _("启动 RuleBot 压测", "Start Self-Check"), type="primary", use_container_width=True):
                    if is_running:
                        st.error(_("⚠️ 请先终止当前任务！", "⚠️ Stop current task first!"))
                    else:
                        cmd = [sys.executable, "main.py", "play", "-n", str(sc_num)]
                        if sc_std_core:
                            cmd.append("--standard_core")
                        p = launch_managed_task(cmd)
                        st.success(_(f"自检压测启动 (PID: {p.pid})！", f"Self-Check started (PID: {p.pid})!"))
                        time.sleep(0.5)
                        st.rerun()

    st.divider()
    
    # --- 终端日志全息映射 (滚动 + 自动刷新版) ---
    st.divider()
    
    # --- 终端日志全息映射 ---
    col_log, col_btn_manual, col_btn_auto = st.columns([6, 2, 2])
    with col_log:
        st.subheader("📝 " + _("实时终端映射 (Live Logs)", "Live Logs"))
    with col_btn_manual:
        if st.button("🔄 " + _("手动刷新", "Refresh"), use_container_width=True):
            pass # 点击触发 Rerun
    with col_btn_auto:
        auto_refresh = st.toggle("⏱️ " + _("自动刷新", "Auto"), value=False)

    # 🌟 修改：抓取所有的 log 文件，不局限于 Trainer
    log_files = glob.glob("./system_logs/*.log")
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        
        # 🌟 补回丢失的控制器栏位，定义 display_limit
        c_info, c_limit = st.columns([8, 2])
        with c_info:
            st.caption(_(f"正在监视文件: `{latest_log}`", f"Monitoring: `{latest_log}`"))
        with c_limit:
            display_limit = st.selectbox(
                _("左侧显示行数", "Display Lines"), 
                [500, 1000, 3000, 5000, 10000], 
                index=0, 
                help=_("调大可看更早的完整上下文，但前端浏览器可能会变卡", "Increase to see older context, but may lag browser")
            )
        
        try:
            with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
                # 🚀 性能革命：绝对禁止用 readlines() 吞入动辄几十MB的整个文件！
                # 使用 deque 在底层极速截取最后 30000 行
                # 彻底解决 WebUI 刷新导致的 CPU 100% 卡死和 C++ 引擎内存越界问题！
                lines = list(deque(f, maxlen=30000))
                
                # 根据用户在右上角的选择动态切片
                display_lines = lines[-display_limit:] if len(lines) > display_limit else lines
                log_text = "".join(display_lines)
                
                # --- 🚨 错误自动提纯引擎 (高性能优化版) ---
                error_lines = []
                # 转为元组，匹配速度更快
                keywords = ("🚨", "❌", "Error", "Exception", "Traceback", "死循环", "熔断")
                
                # 🛡️ 防御性编程：只扫描最后 20000 行。
                # 即使是老日志，也不会导致 while 循环把 CPU 撑爆
                scan_lines = lines[-20000:] if len(lines) > 20000 else lines
                scan_offset = len(lines) - len(scan_lines)
                
                i = 0
                while i < len(scan_lines):
                    # 高速匹配
                    if any(kw in scan_lines[i] for kw in keywords):
                        start = max(0, i - 5)
                        end = min(len(scan_lines), i + 11)
                        block = "".join(scan_lines[start:end])
                        actual_line_num = scan_offset + i + 1
                        error_lines.append(f"--- Line {actual_line_num} Context ---\n{block}")
                        i = end
                    else:
                        i += 1
                        
                if len(error_lines) > 30:
                    error_lines = ["... " + _("(错误过多，仅展示最近 30 个)", "(Too many errors, showing last 30)") + " ...\n"] + error_lines[-30:]
                    
                error_text = "\n\n".join(error_lines)
                # --------------------------------
            
            # 🌟 核心升级：日志双列视图 (左侧完整，右侧提纯)
            log_col1, log_col2 = st.columns([6, 4])
            
            with log_col1:
                # [修复 4] 添加中英双语切换
                st.markdown("**📜 " + _("完整终端日志 (Full Logs)", "Full Terminal Logs") + "**")
                with st.container(height=550):
                    st.code(log_text, language="bash")
                    
            with log_col2:
                # [修复 4] 添加中英双语切换
                st.markdown("**🚨 " + _("异常警报分离器 (Alerts Extraction)", "Alerts Extraction") + "**")
                with st.container(height=550):
                    if error_lines:
                        # 用红色醒目标注
                        st.error(_("检测到严重的报错或异常！(已过滤常规 Retry)", "Critical errors detected! (Normal retries filtered)"))
                        st.code(error_text, language="bash")
                    else:
                        st.success(_("✅ 全局扫描未检测到严重报错。一切平稳！", "✅ No critical errors detected in the entire log. All good!"))
                
        except Exception as e:
            st.error(_(f"读取日志失败: {e}", f"Failed to read logs: {e}"))
    else:
        st.info(_("暂无日志文件生成。请点击上方的启动按钮。", "No logs generated yet. Click start above."))

    # 🌟 自动刷新引擎 (放在文件最末尾)
    if auto_refresh:
        import time
        time.sleep(2) # 每 2 秒刷新一次
        st.rerun()

# ==========================================
# 🔄 模块三：资源同步中枢 (带版本检测与日志阅读器)
# ==========================================
elif menu == _("🔄 资源同步中枢", "🔄 Update Manager"):
    st.title(_("🌐 核心数据与代码同步", "🌐 Data & Core Sync"))
    
    # 顶部状态栏
    c_ver1, c_ver2 = st.columns(2)
    c_ver1.info(f"📌 **{_('本地当前版本', 'Local Version')}**: `v{LOCAL_VERSION}`")
    if has_critical_update: c_ver2.error(f"✨ **{_('云端最新版本', 'Remote Version')}**: `v{remote_version}` ({_('强烈建议更新', 'Update Recommended')})")
    elif has_patch_update: c_ver2.warning(f"🔧 **{_('云端最新版本', 'Remote Version')}**: `v{remote_version}` ({_('可选热修复', 'Optional Patch')})")
    else: c_ver2.success(f"✅ **{_('云端最新版本', 'Remote Version')}**: `v{remote_version}` ({_('已是最新', 'Up to date')})")

    st.write("")
    
    tab_sync, tab_changelog = st.tabs([
        _("🔄 同步控制台", "🔄 Sync Dashboard"), 
        _("📜 更新日志", "📜 Changelog")
    ])
    
    with tab_sync:
        st.markdown(_("从官方与萌卡仓库拉取最新的环境依赖，确保您的 AI 掌握最新的卡片效果与规则。", 
                      "Fetch latest databases and Lua scripts to keep your AI up to date."))
        
        with st.form("update_form"):
            st.subheader(_("选择同步目标", "Select Sync Targets"))
            c_core = st.checkbox(_("🔄 更新 Galatea 核心代码 (Git Pull)", "Sync Core Code (Git Pull)"), value=has_critical_update, 
                                 help=_("从 GitHub 拉取最新的框架 Python 代码。", "Pull the latest framework Python code from GitHub."))
            c_data = st.checkbox(_("🃏 更新 CDB卡库与官方 Lua 脚本", "Sync CDB & Scripts"), value=True, 
                                 help=_("从萌卡拉取最新的 cards.cdb，并从官方仓库同步 script 文件夹。", "Fetch the latest cards.cdb from MyCard and sync the script folder from the official repo."))
            
            st.subheader(_("高级选项", "Advanced Options"))
            t_repo = st.text_input(_("脚本仓库源 (留空为官方)", "Script Repo Source"), value="default")
            c_force = st.checkbox(_("⚠️ 覆盖模式 (强制覆盖本地修改)", "Force Overwrite"), value=False)
            
            if st.form_submit_button("🚀 " + _("立即开始同步", "Start Synchronization"), type="primary", use_container_width=True):
                if not c_core and not c_data:
                    st.warning(_("请至少勾选一个同步目标！", "Please select at least one target!"))
                else:
                    cmd = [sys.executable, "main.py", "update"]
                    if c_core: cmd.append("--core")
                    if c_data: cmd.append("--data")
                    if t_repo != "default": cmd.extend(["--repo", t_repo])
                    if c_force: cmd.append("--force")
                    
                    with st.spinner(_("⏳ 正在全力同步中，这可能需要几分钟时间，请勿刷新页面...", "⏳ Syncing... Please wait.")):
                        try:
                            custom_env = os.environ.copy()
                            custom_env["PYTHONIOENCODING"] = "utf-8"
                            result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', env=custom_env)
                            
                            if result.returncode == 0:
                                st.success(_("✅ 同步完成！建议重启整个系统以加载最新代码。", "✅ Synchronization Complete! Restart recommended."))
                            else:
                                st.error(_("⚠️ 同步遇到问题，请查看下方日志：", "⚠️ Sync encountered issues:"))
                            
                            import re
                            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                            clean_log = ansi_escape.sub('', result.stdout + "\n" + result.stderr)
                            st.code(clean_log, language="bash")
                            
                        except Exception as e:
                            st.error(f"执行失败: {e}")

    with tab_changelog:
        st.markdown("### " + _("📝 版本更新日志", "📝 Release Notes"))
        
        # 智能判定读取中文还是英文文档
        doc_filename = "changelog.md" if lang == "🇨🇳 中文" else "changelog_en.md"
        doc_path = os.path.join(".", "docs", doc_filename)
        
        if os.path.exists(doc_path):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # 用一个带边框的容器装起来，显得更正式
                with st.container(border=True):
                    st.markdown(markdown_content, unsafe_allow_html=True)
            except Exception as e:
                st.error(_(f"读取日志文件失败: {e}", f"Failed to read changelog: {e}"))
        else:
            st.info(_(f"未找到日志文件: `{doc_path}`。您可以去 GitHub 仓库查看最新变更。", 
                      f"Changelog file not found: `{doc_path}`. Please check the GitHub repository."))

# ==========================================
# 🧠 模块四：语义知识库引擎
# ==========================================
elif menu == _("🧠 语义知识库引擎", "🧠 Semantic KB Engine"):
    st.title(_("🧠 语义知识库引擎", "🧠 Semantic KB Engine"))
    st.markdown(_("将 Lua 脚本降维提纯，生成供神经网络食用的高维特征字典。", 
                  "Compress Lua scripts into high-dimensional semantic features for neural networks."))
    
    tab_exec, tab_hash, tab_card = st.tabs([
        _("⚙️ 执行中枢", "⚙️ Execution Hub"), 
        _("🧬 特殊效果图鉴", "🧬 Custom Hash Explorer"),
        _("🔍 单卡语义解剖", "🔍 Card Semantic Viewer")
    ])
    
    # --- 1. 执行中枢 ---
    with tab_exec:
        st.markdown("### " + _("扫描与构建", "Scan & Build"))
        with st.form("parse_form"):
            st.info(_("一键扫描 `./script` 目录下的所有 Lua 脚本，利用正则与降维算法提取出所有卡片的动作条件与种类。", 
                      "Scan all Lua scripts to extract semantic requirements and categories."))
            p_clear = st.checkbox(_("🧨 物理清空本地旧数据 (--clear)", "Clear Local KB"), value=False,
                                  help=_("彻底删除本地知识库、映射表和代码语义向量，重新全量解析。", "Delete the local KB, mapping, and code-semantic vectors before a full rebuild."))
            p_sync = st.checkbox(_("🌐 从 Github 拉取基础卡库同步 (--sync)", "Sync Base KB from Github"), value=False,
                                 help=_("同步主仓库的知识库、Hash 映射、代码语义向量和索引，解析新增卡片后自动接续代码向量。", "Sync the remote KB, Hash map, code-semantic matrix, and index, then automatically append vectors for newly parsed cards."))
            p_url = st.text_input(_("远程基座 URL (可选)", "Remote Base URL (Optional)"), value="https://raw.githubusercontent.com/Noctfom/Galatea-Core/main/knowledge_base.json")
            
            # 👇 [新增] 代码语义化特征提取开关
            p_embed = st.checkbox(_("🧬 提取代码语义特征 (--embed)", "Extract Code Semantic Features"), value=False,
                                  help=_("用于不启用同步的本地更新；同步模式已自动接续。仅提取新增 Lua 效果槽，资产不一致时才全量重建。",
                                         "For local updates without sync; sync mode already continues automatically. Only new Lua effect slots are encoded unless assets are incompatible."))
            
            if st.form_submit_button("🧠 " + _("开始提取卡片语义", "Start Semantic Parsing"), use_container_width=True):
                cmd = [sys.executable, "main.py", "parse"]
                if p_clear: cmd.append("--clear")
                if p_sync: 
                    cmd.append("--sync")
                    if p_url: cmd.extend(["--remote_url", p_url])
                
                # 👇 [新增] 捕捉勾选状态并传递给 main.py
                if p_embed: cmd.append("--embed")
                
                with st.spinner(_("⏳ 正在暴力解析全卡池 Lua 脚本中，请耐心等待...", "⏳ Parsing all Lua scripts... Please wait.")):
                    try:
                        custom_env = os.environ.copy()
                        custom_env["PYTHONIOENCODING"] = "utf-8"
                        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', env=custom_env)
                        
                        import re
                        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                        clean_log = ansi_escape.sub('', result.stdout + "\n" + result.stderr)
                        
                        if result.returncode == 0:
                            st.success(_("✅ 解析完成！", "✅ Parsing Complete!"))
                        else:
                            st.error(_("⚠️ 解析遭遇异常：", "⚠️ Parsing encountered issues:"))
                        st.code(clean_log, language="bash")
                    except Exception as e:
                        st.error(f"执行失败: {e}")

    # --- 2. 特殊效果图鉴 ---
    with tab_hash:
        st.markdown("### " + _("共用底层逻辑探测仪", "Shared Logic Detector"))
        hash_file = "hash_mapping_report.json"
        if os.path.exists(hash_file):
            with open(hash_file, 'r', encoding='utf-8') as f:
                hash_data = json.load(f)
            
            if hash_data:
                # 按共用这张卡的数量从大到小排序
                hash_keys = sorted(list(hash_data.keys()), key=lambda k: len(hash_data[k]["cards"]), reverse=True)
                
                # 提取格式供展示
                display_options = [f"{k} ({_('共用卡数:', 'Shared Cards:')} {len(hash_data[k]['cards'])})" for k in hash_keys]
                selected_idx = st.selectbox(_("选择降维 Hash 标签查看", "Select Custom Hash Tag"), range(len(display_options)), format_func=lambda x: display_options[x])
                
                if selected_idx is not None:
                    selected_hash = hash_keys[selected_idx]
                    h_info = hash_data[selected_hash]
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"**🛠️ {_('极致压缩提取的底层逻辑 (机器码):', 'Compressed Lua Logic:')}**")
                        st.code(h_info.get("sample_code", "N/A"), language="lua")
                    
                    with c2:
                        st.markdown(f"**🔗 {_('共用此逻辑的卡片族群', 'Cards sharing this logic')} (Total: {len(h_info['cards'])}):**")
                        # 格式化输出带名称的卡片列表
                        card_displays = []
                        for cl in h_info["cards"]:
                            c_id = cl.split('_')[0]
                            # 使用强绑定的 UI 专属 DB 读取
                            c_name = card_db_ui.get_card_name(int(c_id)) 
                            # 如果实在查不到(比如是先行卡/未同步)，给个干净的占位符
                            if c_name.startswith("Code "): c_name = "未收录卡片"
                            
                            ygocdb_link = f"https://ygocdb.com/card/{c_id}"
                            card_displays.append(f"- **{c_name}** (`{c_id}`) → [🌐 百鸽 YGOCDB]({ygocdb_link}) *(Slot: {cl.split('_')[1]})*")
                        
                        with st.container(height=400):
                            st.markdown("\n".join(card_displays))
            else:
                st.info(_("映射表为空。", "Hash mapping report is empty."))
        else:
            st.info(_("请先在 [执行中枢] 提取语义数据。", "Please run the parser in Execution Hub first."))

    # --- 3. 单卡语义解剖 ---
    with tab_card:
        st.markdown("### " + _("AI 视觉下的单卡解剖", "AI Vision Semantic Viewer"))
        kb_file = "knowledge_base.json"
        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
                
            search_id = st.text_input(_("输入 8 位卡片密码 (Code) 进行搜索", "Enter 8-digit Card Code"), placeholder="例如: 14558127")
            if search_id:
                if search_id in kb_data:
                    c_name = card_db_ui.get_card_name(int(search_id))
                    st.success(f"**{c_name}** (`{search_id}`)")
                    st.markdown(f"🔗 **[点击此处在 百鸽 (YGOCDB) 中查看卡图与详细裁定](https://ygocdb.com/card/{search_id})** *(鸣谢: YGOCDB)*")
                    st.json(kb_data[search_id])
                else:
                    st.warning(_("知识库中未找到该卡片的独立效果，这可能是一张白板怪兽，或脚本极其特殊。", "Card not found in Knowledge Base. It might be a Normal Monster or has no extractable effects."))
        else:
            st.info(_("请先在 [执行中枢] 提取语义数据。", "Please run the parser in Execution Hub first."))

# ==========================================
# 🗃️ 模块五：资产与卡组管理
# ==========================================
elif menu == _("🗃️ 资产与卡组管理", "🗃️ Assets & Decks"):
    st.title(_("🗃️ 资产与卡组管理", "🗃️ Assets & Decks Manager"))
    
    tab_staples, tab_decks, tab_global, tab_virtual,tab_online= st.tabs([
        _("🃏 泛用卡池配置 (142 兜底)", "🃏 Meta Staples (142 Cache)"),
        _("📂 卡组与环境管理", "📂 Deck & Pool Manager"),
        _("⚖️ 动态环境权重调度", "⚖️ Dynamic Pool Weights"),
        _("🧠 虚拟环境构建器", "🧠 Virtual Mix Pool Builder"),
        _("🌐 在线动态环境构建", "🌐 Online Fetcher")
    ])
    
    # --- 1. 泛用卡池配置 ---
    with tab_staples:
        st.markdown(_("当引擎发来极限刁钻的 142 宣言条件时，AI 将优先从这个纯净池里挑选答案。", 
                      "AI will prioritize this clean pool when facing extreme 142 card announce conditions."))
        
        staples_file = 'meta_staples.json'
        # 确保文件存在
        if not os.path.exists(staples_file):
            with open(staples_file, 'w', encoding='utf-8') as f: 
                json.dump([14558127, 23434538, 10045474, 24094653, 73642296, 32807846], f)
        
        with open(staples_file, 'r', encoding='utf-8') as f:
            staples_list = json.load(f)
            
        c_add, c_del = st.columns(2)
        with c_add:
            st.subheader(_("➕ 添加新卡", "Add New Card"))
            new_code = st.text_input(_("输入 8 位卡片密码", "Enter 8-digit Card Code"), placeholder="例如: 14558127")
            if st.button(_("添加到泛用池", "Add to Pool"), use_container_width=True):
                if new_code.isdigit():
                    code_int = int(new_code)
                    if code_int in staples_list:
                        st.warning(_("该卡片已在池中！", "Card already in pool!"))
                    else:
                        c_name = card_db_ui.get_card_name(code_int)
                        if c_name.startswith("Code "):
                            st.error(_("数据库中查无此卡，请检查卡密！", "Card not found in database, check code!"))
                        else:
                            staples_list.append(code_int)
                            with open(staples_file, 'w', encoding='utf-8') as f:
                                json.dump(staples_list, f, indent=4)
                            st.success(_(f"✅ 成功添加: {c_name}", f"✅ Added: {c_name}"))
                            st.rerun()
                else:
                    st.error(_("请输入纯数字卡密！", "Please enter numeric code only!"))
                    
        with c_del:
            st.subheader(_("🗑️ 移除卡片", "Remove Card"))
            # 构建字典供选择
            options = {code: f"{card_db_ui.get_card_name(code)} ({code})" for code in staples_list}
            to_delete = st.multiselect(_("选择要移除的卡片", "Select cards to remove"), options=list(options.keys()), format_func=lambda x: options[x])
            if st.button(_("从泛用池中移除", "Remove Selected"), type="primary", use_container_width=True):
                if to_delete:
                    staples_list = [c for c in staples_list if c not in to_delete]
                    with open(staples_file, 'w', encoding='utf-8') as f:
                        json.dump(staples_list, f, indent=4)
                    st.success(_("✅ 移除成功！", "✅ Removed successfully!"))
                    st.rerun()
                else:
                    st.warning(_("请先选择卡片！", "Please select cards first!"))
        
        st.divider()
        st.write(_("📜 当前卡池列表预览：", "📜 Current Staples Preview:"))
        # 用易读的格式展示
        display_list = [f"- **{card_db_ui.get_card_name(code)}** (`{code}`)" for code in staples_list]
        st.markdown("\n".join(display_list))

    # --- 2. 资产与卡组管理：全息构筑与环境池中枢 ---
    with tab_decks:
        import deck_utils 
        deck_root = "./decks"
        os.makedirs(deck_root, exist_ok=True)

        # 1. 环境池基础导航
        deck_root_real = os.path.realpath(deck_root)
        pools = []
        for dirname in os.listdir(deck_root):
            try:
                validate_local_asset_name(dirname)
            except ValueError:
                continue
            candidate = os.path.join(deck_root, dirname)
            resolved = os.path.realpath(candidate)
            if (
                os.path.isdir(candidate)
                and not os.path.islink(candidate)
                and os.path.commonpath([deck_root_real, resolved]) == deck_root_real
                and resolved != deck_root_real
            ):
                pools.append(dirname)
        pools.insert(0, ".")
        
        c_nav, c_del_pool = st.columns([3, 1])
        with c_nav:
            sel_pool = st.selectbox(_("📂 当前操作环境池 (文件夹)", "📂 Current Meta Pool"), pools, key="sel_p_nav")
            pool_path = deck_root if sel_pool == "." else os.path.join(deck_root, sel_pool)
        
        with c_del_pool:
            st.write(""); st.write("")
            if sel_pool != ".":
                if st.button("🗑️ " + _("删除环境池", "Delete Pool"), use_container_width=True, help=_("注意：将删除该文件夹下所有卡组！", "Warning: This will delete all decks in this folder!")):
                    resolved_pool = os.path.realpath(pool_path)
                    if (
                        os.path.commonpath([deck_root_real, resolved_pool]) != deck_root_real
                        or resolved_pool == deck_root_real
                    ):
                        raise ValueError("环境池目录边界校验失败")
                    shutil.rmtree(resolved_pool)
                    st.toast(_("环境池已移除", "Pool removed"))
                    st.rerun()

        # 2. 快捷创建与上传区 (收纳进折叠面板)
        with st.expander(_("🛠️ 快速新建与上传", "🛠️ Quick Create & Upload")):
            m1, m2 = st.columns(2)
            with m1:
                st.caption(_("新建子环境池 (文件夹)", "Create Sub-Pool Folder"))
                new_p_name = st.text_input("Folder Name", placeholder="e.g. tier_1", label_visibility="collapsed", key="in_new_p")
                if st.button(_("确认创建文件夹", "Confirm Create Folder"), use_container_width=True):
                    if new_p_name:
                        try:
                            safe_pool_name = validate_local_asset_name(new_p_name)
                            os.makedirs(os.path.join(deck_root, safe_pool_name), exist_ok=False)
                            st.success(_("文件夹创建成功", "Folder Created"))
                            st.rerun()
                        except Exception as error:
                            st.error(_(f"环境池创建失败: {error}", f"Pool creation failed: {error}"))
            with m2:
                st.caption(_("在此池中新建空白卡组 (.ydk)", "Create Empty Deck in this Pool"))
                new_d_name = st.text_input("Deck Name", placeholder="e.g. MyNewDeck", label_visibility="collapsed", key="in_new_d")
                if st.button(_("确认创建空白卡组", "Confirm Create Deck"), use_container_width=True):
                    if new_d_name:
                        try:
                            deck_filename = validate_local_asset_name(
                                f"{new_d_name}.ydk",
                                required_suffix=".ydk",
                            )
                            new_f_path = os.path.join(pool_path, deck_filename)
                            with open(new_f_path, 'x', encoding='utf-8') as f:
                                f.write("#main\n#extra\n!side\n")
                            st.success(_("空白卡组已就绪", "Empty Deck Created"))
                            st.rerun()
                        except FileExistsError:
                            st.error(_("卡组已存在", "Deck already exists"))
                        except Exception as error:
                            st.error(_(f"卡组创建失败: {error}", f"Deck creation failed: {error}"))
            
            st.divider()
            uploaded_files = st.file_uploader(_("📤 上传本地 .ydk 文件", "📤 Upload local .ydk"), accept_multiple_files=True, type=['ydk'])
            if uploaded_files and st.button(_("💾 执行保存", "Execute Save"), use_container_width=True):
                try:
                    for f in uploaded_files:
                        safe_upload_name = validate_local_asset_name(
                            f.name,
                            required_suffix=".ydk",
                        )
                        destination = os.path.join(pool_path, safe_upload_name)
                        f.seek(0)
                        with tempfile.NamedTemporaryFile(
                            prefix=f".{safe_upload_name}.",
                            suffix=".upload.tmp",
                            dir=pool_path,
                            delete=False,
                        ) as temporary_stream:
                            temporary_path = temporary_stream.name
                            shutil.copyfileobj(f, temporary_stream, length=1024 * 1024)
                        try:
                            os.replace(temporary_path, destination)
                        finally:
                            if os.path.exists(temporary_path):
                                os.remove(temporary_path)
                    st.success(_("上传成功", "Upload successful")); st.rerun()
                except Exception as error:
                    st.error(_(f"上传被拒绝: {error}", f"Upload rejected: {error}"))

        st.divider()

        # 3. 核心：全息构筑编辑器
        files = deck_utils.list_decks(pool_path)
        if files:
            st.subheader(_("👁️ 全息预览与构筑", "👁️ Holographic Editor"))
            view_deck_name = st.selectbox(_("选择要编辑/预览的卡组", "Select deck to edit/view"), ["None"] + files, key="sb_edit_deck")
            
            if view_deck_name != "None":
                deck_filename = validate_local_asset_name(
                    f"{view_deck_name}.ydk",
                    required_suffix=".ydk",
                )
                deck_full_path = os.path.join(pool_path, deck_filename)
                
                # 初始化/同步缓存
                if 'editor_content' not in st.session_state or st.session_state.get('active_deck_path') != deck_full_path:
                    try:
                        with open(deck_full_path, 'r', encoding='utf-8') as f:
                            st.session_state.editor_content = f.read()
                        st.session_state.active_deck_path = deck_full_path
                    except: st.error("File Error")

                # YDK 解析
                def parse_ydk_fast(txt):
                    m, e, s = [], [], []
                    curr = None
                    for line in txt.split('\n'):
                        l = line.strip()
                        if l == '#main': curr = m
                        elif l == '#extra': curr = e
                        elif l == '!side': curr = s
                        elif l.isdigit() and curr is not None: curr.append(l)
                    return m, e, s

                main_list, extra_list, side_list = parse_ydk_fast(st.session_state.editor_content)

                # 快速操作区
                op_c1, op_c2 = st.columns([1, 1])
                with op_c1:
                    st.write(_("➕ **添加卡片 (输入卡密)**", "➕ **Quick Add**"))
                    new_code = st.text_input("Passcode", placeholder="8位卡片密码", label_visibility="collapsed", key="ed_add")
                    if new_code and new_code.isdigit():
                        c_name = card_db_ui.get_card_name(int(new_code))
                        st.caption(f"✨ 识别: **{c_name}**" if c_name != "Unknown" else "❌ 未知卡片")
                        btn_c = st.columns(3)
                        if btn_c[0].button(_("主卡", "Main"), use_container_width=True): main_list.append(new_code)
                        if btn_c[1].button(_("额外", "Extra"), use_container_width=True): extra_list.append(new_code)
                        if btn_c[2].button(_("副卡", "Side"), use_container_width=True): side_list.append(new_code)
                        st.session_state.editor_content = f"#main\n" + "\n".join(main_list) + f"\n#extra\n" + "\n".join(extra_list) + f"\n!side\n" + "\n".join(side_list)

                with op_c2:
                    st.write(_("💾 **保存修改**", "💾 **Save Config**"))
                    st.write("")
                    if st.button("🚀 " + _("将构筑写入 .ydk 文件", "Save to YDK File"), type="primary", use_container_width=True):
                        with open(deck_full_path, 'w', encoding='utf-8') as f:
                            f.write(st.session_state.editor_content)
                        st.success(_("✅ 构筑已持久化保存！", "✅ Saved successfully!"))

                # 渲染卡组网格
                api_lang = "sc" if lang == "🇨🇳 中文" else "en"
                def render_deck_grid(card_list, title):
                    st.markdown(f"**{title} ({len(card_list)})**")
                    if not card_list: return
                    cols = st.columns(10)
                    for i, code in enumerate(card_list):
                        with cols[i % 10]:
                            img_url = f"https://cdn.233.momobako.com/ygoimg/{api_lang}/{code}.webp!half"
                            # 图片带查卡链接
                            st.markdown(f'<a href="https://ygocdb.com/card/{code}" target="_blank"><img src="{img_url}" style="width:100%; border-radius:3px; border: 1px solid #444;"></a>', unsafe_allow_html=True)
                            # 删除小按钮
                            if st.button("🗑️", key=f"d_{title}_{i}_{code}", help=_("移除", "Remove")):
                                card_list.pop(i)
                                st.session_state.editor_content = f"#main\n" + "\n".join(main_list) + f"\n#extra\n" + "\n".join(extra_list) + f"\n!side\n" + "\n".join(side_list)
                                st.rerun()

                render_deck_grid(main_list, _("📜 主卡组", "📜 Main"))
                render_deck_grid(extra_list, _("🎴 额外卡组", "🎴 Extra"))
                render_deck_grid(side_list, _("🃏 副卡组", "🃏 Side"))

            st.divider()

            # 4. 批量管理 (原功能保持)
            st.write(_(f"📑 批量管理 `{sel_pool}` 中的文件：", f"📑 Batch Manage in `{sel_pool}`:"))
            sel_files = st.multiselect(_("选择文件", "Select Files"), [f"{n}.ydk" for n in files])
            
            bm1, bm2 = st.columns(2)
            with bm1:
                target_p = st.selectbox(_("移动到...", "Move to..."), pools, key="mv_dest")
                if st.button(_("➡️ 执行移动", "Move Selected"), use_container_width=True) and sel_files:
                    dest = deck_root if target_p == "." else os.path.join(deck_root, target_p)
                    for f in sel_files: shutil.move(os.path.join(pool_path, f), os.path.join(dest, f))
                    st.rerun()
            with bm2:
                st.write(""); st.write("")
                if st.button(_("🗑️ 批量删除选中卡组", "Delete Selected"), type="primary", use_container_width=True) and sel_files:
                    for f in sel_files: os.remove(os.path.join(pool_path, f))
                    st.rerun()
        else:
            st.info(_("此文件夹暂无卡组，请使用上方面板新建或上传。", "No decks. Use the panel above to create or upload."))

    # --- 3. 全局环境调度 (Global Weights - 分类+大滑块版) ---
    with tab_global:
        st.markdown("### ⚖️ " + _("全局训练环境调度台", "Global Training Environment Weights"))
        st.info(_("此面板控制 AI 最终遇到哪一个环境池。为了方便管理，已将环境按来源分类。上方的大滑块可一键覆盖同类别的所有权重。", "Control environment probabilities. Pools are grouped by source. Use master sliders to bulk apply weights."))
        
        deck_root = "./decks"
        os.makedirs(deck_root, exist_ok=True)
        subdirs = [os.path.basename(os.path.normpath(d)) for d in os.listdir(deck_root) if os.path.isdir(os.path.join(deck_root, d))]
        
        virtual_file = os.path.join(deck_root, "virtual_pools.json")
        v_pools = []
        if os.path.exists(virtual_file):
            try:
                with open(virtual_file, 'r') as f: v_pools = list(json.load(f).keys())
            except: pass
            
        all_candidates = subdirs + v_pools
        if not all_candidates:
            st.warning(_("暂无任何环境池。", "No pools available."))
        else:
            global_file = os.path.join(deck_root, "global_weights.json")
            current_g_weights = {}
            if os.path.exists(global_file):
                try:
                    with open(global_file, 'r') as f: current_g_weights = json.load(f)
                except: pass
            
            # 🌟 为池子分类
            online_pools = [c for c in all_candidates if c.startswith("ygopd_")]
            virtual_pools = [c for c in all_candidates if c in v_pools]
            local_pools = [c for c in all_candidates if c not in online_pools and c not in virtual_pools]
            
            new_g_weights = {}
            
            with st.form("global_weights_form"):
                # --- 分类 1: 在线抓取池 ---
                if online_pools:
                    with st.expander("🌐 " + _("在线动态抓取池 (Online Pools)", "Online Fetch Pools"), expanded=True):
                        c_bulk, c_btn = st.columns([3, 1])
                        bulk_val_online = c_bulk.number_input(_("批量设值 (Bulk Set)", "Bulk Set"), 0.0, 10.0, 1.0, 0.1, key="blk_on")
                        apply_online = c_btn.form_submit_button(_("⬇️ 向下应用", "Apply Below"), key="btn_app_on")
                        
                        # 🌟 核心修复：直接覆写 session_state 内存状态
                        if apply_online:
                            for c_name in online_pools:
                                st.session_state[f"g_w_{c_name}"] = bulk_val_online
                                
                        st.markdown("---")
                        for c_name in online_pools:
                            new_g_weights[c_name] = st.slider(f"☁️ {c_name}", 0.0, 10.0, float(current_g_weights.get(c_name, 1.0)), 0.1, key=f"g_w_{c_name}")
                
                # --- 分类 2: 虚拟拼装池 ---
                if virtual_pools:
                    with st.expander("🧬 " + _("虚拟拼装乱斗池 (Virtual Mix Pools)", "Virtual Mix Pools"), expanded=True):
                        c_bulk, c_btn = st.columns([3, 1])
                        bulk_val_virt = c_bulk.number_input(_("批量设值 (Bulk Set)", "Bulk Set"), 0.0, 10.0, 1.0, 0.1, key="blk_vi")
                        apply_virt = c_btn.form_submit_button(_("⬇️ 向下应用", "Apply Below"), key="btn_app_vi")
                        
                        # 🌟 核心修复
                        if apply_virt:
                            for c_name in virtual_pools:
                                st.session_state[f"g_w_{c_name}"] = bulk_val_virt
                                
                        st.markdown("---")
                        for c_name in virtual_pools:
                            new_g_weights[c_name] = st.slider(f"🧪 {c_name}", 0.0, 10.0, float(current_g_weights.get(c_name, 1.0)), 0.1, key=f"g_w_{c_name}")

                # --- 分类 3: 本地管理池 ---
                if local_pools:
                    with st.expander("📂 " + _("本地环境管理池 (Local Pools)", "Local Pools"), expanded=True):
                        c_bulk, c_btn = st.columns([3, 1])
                        bulk_val_loc = c_bulk.number_input(_("批量设值 (Bulk Set)", "Bulk Set"), 0.0, 10.0, 1.0, 0.1, key="blk_lo")
                        apply_loc = c_btn.form_submit_button(_("⬇️ 向下应用", "Apply Below"), key="btn_app_lo")
                        
                        # 🌟 核心修复
                        if apply_loc:
                            for c_name in local_pools:
                                st.session_state[f"g_w_{c_name}"] = bulk_val_loc
                                
                        st.markdown("---")
                        for c_name in local_pools:
                            new_g_weights[c_name] = st.slider(f"📁 {c_name}", 0.0, 10.0, float(current_g_weights.get(c_name, 1.0)), 0.1, key=f"g_w_{c_name}")

                st.write("")
                # 最终保存按钮
                if st.form_submit_button("💾 " + _("保存并应用全局权重", "Save Global Weights"), type="primary", use_container_width=True):
                    with open(global_file, 'w', encoding='utf-8') as f:
                        json.dump(new_g_weights, f, indent=4)
                    st.toast(_("✅ 全局权重已更新！Worker 下局生效。", "✅ Global weights updated!"))

    # --- 4. 虚拟拼装池构建 (Virtual Mix Pools) ---
    with tab_virtual:
        st.markdown("### 🧠 " + _("虚拟拼装池制药厂", "Virtual Mix Pool Builder"))
        st.info(_("在这里配置特定的物理池混合配方。建好后，它会作为一个新环境出现在【全局环境权重】中供你调度！", 
                  "Create recipes mixing different physical pools. They will appear in Global Weights."))
        
        deck_root = "./decks"
        subdirs = [os.path.basename(os.path.normpath(d)) for d in os.listdir(deck_root) if os.path.isdir(os.path.join(deck_root, d))]
        
        if not subdirs:
            st.warning(_("无物理文件夹，无法拼装。", "No subfolders found."))
        else:
            v_file = os.path.join(deck_root, "virtual_pools.json")
            try:
                with open(v_file, 'r', encoding='utf-8') as f: all_v_pools = json.load(f)
            except: all_v_pools = {}

            with st.expander(_("➕ 创建新的拼装池", "Create New Mix Pool")):
                c_name, c_btn = st.columns([3, 1])
                new_pool_name = c_name.text_input(_("拼装池名称", "Pool Name"), placeholder="e.g. Meta_VS_Fun")
                if c_btn.button(_("创建配方", "Create Recipe"), use_container_width=True):
                    if not new_pool_name.strip():
                        idx = 1
                        while f"Virtual_Mix_{idx}" in all_v_pools: idx += 1
                        new_pool_name = f"Virtual_Mix_{idx}"
                    
                    if new_pool_name in all_v_pools: st.error("已存在！")
                    else:
                        all_v_pools[new_pool_name] = {name: 0.0 for name in subdirs}
                        with open(v_file, 'w', encoding='utf-8') as f: json.dump(all_v_pools, f, indent=4)
                        st.success(f"✅ 创建成功！快去下方调配它吧！")
                        st.rerun()

            st.divider()
            
            if not all_v_pools:
                st.info(_("暂无虚拟拼装池。", "No virtual pools."))
            else:
                # 对所有物理池进行分类
                online_subdirs = [c for c in subdirs if c.startswith("ygopd_")]
                local_subdirs = [c for c in subdirs if not c.startswith("ygopd_")]
                
                for p_name in list(all_v_pools.keys()):
                    with st.container(border=True):
                        col_h, col_del = st.columns([8, 1])
                        col_h.subheader(f"🧪 配方: {p_name}")
                        if col_del.button(_("🗑️ 删除配方", "🗑️ Delete Recipe"), key=f"del_{p_name}"):
                            del all_v_pools[p_name]
                            with open(v_file, 'w', encoding='utf-8') as f: json.dump(all_v_pools, f, indent=4)
                            st.rerun()
                        
                        p_cfg = all_v_pools[p_name]
                        new_cfg = {}
                        
                        # --- 虚拟池配方分类: 在线池 ---
                        if online_subdirs:
                            with st.expander("🌐 " + _("混入在线抓取池", "Mix Online Pools"), expanded=True):
                                c_bulk, c_btn = st.columns([3, 1])
                                bulk_val_on = c_bulk.number_input(_("批量设值", "Bulk Set"), 0.0, 10.0, 0.0, 0.1, key=f"vb_on_{p_name}")
                                apply_on = c_btn.button(_("⬇️ 向下应用", "Apply"), key=f"va_on_{p_name}")
                                
                                # 🌟 核心修复：修改底层 session_state
                                if apply_on:
                                    for s_name in online_subdirs:
                                        st.session_state[f"vsld_{p_name}_{s_name}"] = bulk_val_on
                                        
                                st.markdown("---")
                                cols = st.columns(2)
                                for i, s_name in enumerate(online_subdirs):
                                    with cols[i % 2]:
                                        new_cfg[s_name] = st.slider(f"☁️ {s_name}", 0.0, 10.0, float(p_cfg.get(s_name, 0.0)), 0.1, key=f"vsld_{p_name}_{s_name}")
                        
                        # --- 虚拟池配方分类: 本地池 ---
                        if local_subdirs:
                            with st.expander("📂 " + _("混入本地环境池", "Mix Local Pools"), expanded=True):
                                c_bulk, c_btn = st.columns([3, 1])
                                bulk_val_loc = c_bulk.number_input(_("批量设值", "Bulk Set"), 0.0, 10.0, 0.0, 0.1, key=f"vb_lo_{p_name}")
                                apply_loc = c_btn.button(_("⬇️ 向下应用", "Apply"), key=f"va_lo_{p_name}")
                                
                                # 🌟 核心修复
                                if apply_loc:
                                    for s_name in local_subdirs:
                                        st.session_state[f"vsld_{p_name}_{s_name}"] = bulk_val_loc
                                        
                                st.markdown("---")
                                cols = st.columns(2)
                                for i, s_name in enumerate(local_subdirs):
                                    with cols[i % 2]:
                                        new_cfg[s_name] = st.slider(f"📁 {s_name}", 0.0, 10.0, float(p_cfg.get(s_name, 0.0)), 0.1, key=f"vsld_{p_name}_{s_name}")
                        
                        st.write("")
                        if st.button("💾 " + _("保存此混合配方", "Save Mix Recipe"), key=f"vsave_{p_name}", type="primary", use_container_width=True):
                            all_v_pools[p_name] = new_cfg
                            with open(v_file, 'w', encoding='utf-8') as f: json.dump(all_v_pools, f, indent=4)
                            st.toast(f"✅ 【{p_name}】 配方已保存！")
    
    # --- 5. 在线动态环境与抓取引擎 ---
    with tab_online:
        st.markdown("### 🌐 " + _("在线动态环境构建与同步", "Online Meta Builder"))
        
        import threading
        tasks_file = "./decks/fetch_tasks.json"
        daemon_status_file = "./decks/daemon_status.txt"
        
        # 🌟 UI 顶部：数据源声明与连通性测试
        c_source, c_test = st.columns([3, 1])
        with c_source:
            source = st.selectbox(_("🔌 数据源声明 (鸣谢)", "Data Source"), ["YGOProDeck (海外最大卡组库)", "萌卡 MyCard (国内) [待开发]"], disabled=True)
        with c_test:
            st.write(""); st.write("")
            if st.button("📡 " + _("探测数据源 API 状态", "Test Connection"), use_container_width=True):
                import online_fetcher
                fetcher = online_fetcher.YGOProDeckFetcher()
                with st.spinner("正在发送探测封包..."):
                    succ, msg = fetcher.test_connection()
                    if succ: st.success(msg)
                    else: st.error(msg)
        
        st.divider()
        daemon_running = False
        if os.path.exists(daemon_status_file):
            with open(daemon_status_file, "r", encoding="utf-8") as f: d_status = f.read()
            if d_status.startswith("RUNNING"):
                daemon_running = True
                st.success("🔄 " + d_status)
        
        # 🌟 核心：映射官方真实底层 API 标签 (全量豪华版)
        api_tags = {
            # --- 🏆 比赛上位 (Tournament) ---
            "🏆 TCG 比赛上位 (Tournament TCG)": "Tournament Meta Decks",
            "🏆 OCG 比赛上位 (Tournament OCG)": "Tournament Meta Decks OCG",
            "🏆 OCG-CN 比赛上位 (Tournament OCG-CN)": "Tournament Meta Decks OCG-CN",
            "🏆 世界赛上位 (Tournament Worlds)": "Tournament Meta Decks World Championship",
            
            # --- ⚔️ 竞技环境 (Competitive) ---
            "⚔️ 竞技天梯 (Meta Decks)": "Meta Decks",
            "⚔️ 历届世界赛构筑 (World Championship)": "World Championship Decks",
            
            # --- 🎉 休闲与娱乐 (Casual) ---
            "🎉 非主流/绝活 (Non-Meta)": "Non-Meta Decks",
            "📺 动漫主题卡组 (Anime Decks)": "Anime Decks",
            "🎈 纯娱乐卡组 (Fun/Casual)": "Fun/Casual Decks",
            "🧠 构筑研讨 (Theorycrafting)": "Theorycrafting Decks",
            
            # --- 🎮 其他游戏与特殊赛制 ---
            "🎮 大师决斗 (Master Duel)": "Master Duel Decks",
            "🕰️ Edison 环境 (Edison Format)": "Edison Format",
            "🕰️ Goat 环境 (Goat Format)": "Goat Format",
            "🕰️ 疾速决斗 (Speed Duel)": "Speed Duel Decks",
            "🌍 全部分类无限制 (All)": "All"
        }

        # 1. 抓取与添加新卡池
        with st.expander(_("➕ 初始化/拉取新卡池", "Fetch New Pool"), expanded=True):
            f1, f2, f3 = st.columns([4, 3, 3])
            with f1: fetch_label = st.selectbox(_("选择目标卡组池标签 (API 映射)", "Select Target Pool Label"), list(api_tags.keys()), key="ftag")
            with f2: fetch_mode = st.radio(_("抓取深度模式", "Fetch Depth Mode"), [_("🆕 最新顺序", "🆕 Latest"), _("🌌 历史随机", "🌌 Random")], horizontal=True, key="fmode")
            with f3: fetch_limit = st.number_input(_("抓取数量", "Fetch Quantity"), min_value=5, max_value=200, value=30, step=10, key="flimit")
            
            real_api_tag = api_tags[fetch_label]
            is_rand = "🌌" in fetch_mode
            mode_tag = "Rand" if is_rand else "Latest"
            auto_folder_name = f"ygopd_{real_api_tag.replace(' ', '')}_{mode_tag}"
            
            if st.button("📥 " + _(f"抓取并添加到订阅清单: {auto_folder_name}", "Fetch & Add to List"), type="primary", use_container_width=True):
                with st.spinner("正在突破天梯拉取数据..."):
                    import online_fetcher, random
                    fetcher = online_fetcher.YGOProDeckFetcher()
                    # 🌟 核心突破：强制将随机偏移量乘以 20，完美骗过 WordPress 的分页系统
                    offset = random.randint(1, 100) * 20 if is_rand else 0
                    
                    succ, msg = fetcher.fetch_decks(limit=fetch_limit, target_dir=os.path.join("./decks", auto_folder_name), api_category=real_api_tag, offset=offset)
                    
                    if succ:
                        try:
                            with open(tasks_file, 'r', encoding='utf-8') as f: tasks = json.load(f)
                        except: tasks = {}
                        
                        if auto_folder_name not in tasks:
                            tasks[auto_folder_name] = {
                                "api_category": real_api_tag, "is_rand": is_rand,
                                "base_limit": fetch_limit, "update_limit": fetch_limit, "auto_update": False,
                                "last_update": time.strftime("%m-%d %H:%M")
                            }
                        else: tasks[auto_folder_name]["last_update"] = time.strftime("%m-%d %H:%M")
                        
                        with open(tasks_file, 'w', encoding='utf-8') as f: json.dump(tasks, f, indent=4)
                        st.success(f"{msg} (偏移深度: {offset})")
                        time.sleep(1.5)
                        st.rerun()
                    else: st.error(msg)
        
        # 2. 全局守护进程控制器
        st.markdown("#### 🤖 " + _("后台自动更新守护进程", "Auto-Update Daemon"))
        col_int, col_btn = st.columns([3, 1])
        with col_int:
            global_interval = st.number_input(_("全局循环间隔 (小时) | 建议 0.5 ~ 24", "Global Interval (Hours)"), min_value=0.5, max_value=72.0, value=12.0, step=0.25)
            st.caption(_("守护进程会将下方【允许自动】的任务均匀分散到此时间段内执行，完美规避 API 封禁。", ""))
        
        with col_btn:
            st.write(""); st.write("")
            if daemon_running:
                if st.button("🛑 " + _("停止守护", "Stop Daemon"), type="primary", use_container_width=True):
                    with open(daemon_status_file, "w", encoding="utf-8") as f: f.write("STOPPED")
                    st.rerun()
            else:
                if st.button("⚙️ " + _("启动后台守护", "Start Daemon"), type="secondary", use_container_width=True):
                    with open(daemon_status_file, "w", encoding="utf-8") as f: f.write(f"RUNNING: 守护中 (全局间隔 {global_interval}H)")
                    
                    def bg_daemon_task(interval_hrs):
                        import online_fetcher, random, time
                        while True:
                            try:
                                with open(tasks_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    tasks = json.loads(content) if content else {}
                            except: tasks = {}
                            
                            active_tasks = [k for k, v in tasks.items() if v.get('auto_update', False)]
                            if not active_tasks:
                                for _ in range(60):
                                    if os.path.exists(daemon_status_file) and open(daemon_status_file, "r", encoding="utf-8").read().strip() == "STOPPED":
                                        os.remove(daemon_status_file); return
                                    time.sleep(1)
                                continue
                                
                            interval_sec = int(interval_hrs * 3600)
                            gap_sec = max(10, interval_sec // len(active_tasks)) 
                            
                            for task_name in active_tasks:
                                if os.path.exists(daemon_status_file) and open(daemon_status_file, "r", encoding="utf-8").read().strip() == "STOPPED":
                                    os.remove(daemon_status_file); return
                                
                                cfg = tasks[task_name]
                                try:
                                    fetcher = online_fetcher.YGOProDeckFetcher()
                                    offset = random.randint(1, 100) * 20 if cfg.get('is_rand') else 0
                                    fetcher.fetch_decks(limit=cfg.get('update_limit', 10), target_dir=os.path.join("./decks", task_name), api_category=cfg.get('api_category'), offset=offset)
                                    
                                    tasks[task_name]['last_update'] = time.strftime("%m-%d %H:%M")
                                    with open(tasks_file, 'w', encoding='utf-8') as f: json.dump(tasks, f, indent=4)
                                except: pass
                                
                                for _ in range(gap_sec):
                                    if os.path.exists(daemon_status_file) and open(daemon_status_file, "r", encoding="utf-8").read().strip() == "STOPPED":
                                        os.remove(daemon_status_file); return
                                    time.sleep(1)
                                    
                    threading.Thread(target=bg_daemon_task, args=(global_interval,), daemon=True).start()
                    st.rerun()

        # 3. 缓存清单管理列表
        st.markdown("#### 📋 " + _("卡池更新管理列表", "Subscription List"))
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f: current_tasks = json.load(f)
        except: current_tasks = {}
        
        if not current_tasks:
            st.info("暂无抓取记录。请在上方执行首次抓取！")
        else:
            for t_name, t_cfg in current_tasks.items():
                with st.container(border=True):
                    col_name, col_lim, col_tog, col_man, col_del = st.columns([3, 2, 2, 2, 1])
                    col_name.write(f"🏷️ **{t_name}**\n\n<small>{_('上次:', 'Last Updated')}: {t_cfg.get('last_update', 'N/A')}</small>", unsafe_allow_html=True)
                    
                    new_lim = col_lim.number_input(_("更新量", "Update Quantity"), min_value=5, max_value=t_cfg.get('base_limit', 50), value=t_cfg.get('update_limit', 10), step=5, key=f"ulim_{t_name}")
                    st.write("")
                    new_tog = col_tog.toggle(_("🔄 允许自动", "🔄 Allow Auto-Update"), value=t_cfg.get('auto_update', False), key=f"utog_{t_name}")
                    
                    if new_lim != t_cfg.get('update_limit') or new_tog != t_cfg.get('auto_update'):
                        current_tasks[t_name]['update_limit'] = new_lim
                        current_tasks[t_name]['auto_update'] = new_tog
                        with open(tasks_file, 'w', encoding='utf-8') as f: json.dump(current_tasks, f, indent=4)
                    
                    if col_man.button("⚡ " + _("立即覆盖更新", "Force Update"), key=f"uman_{t_name}"):
                        import online_fetcher, random
                        with st.spinner(f"正在更新 {t_name}..."):
                            fetcher = online_fetcher.YGOProDeckFetcher()
                            offset = random.randint(1, 100) * 20 if t_cfg.get('is_rand') else 0
                            succ, msg = fetcher.fetch_decks(limit=new_lim, target_dir=os.path.join("./decks", t_name), api_category=t_cfg.get('api_category'), offset=offset)
                            if succ:
                                current_tasks[t_name]['last_update'] = time.strftime("%m-%d %H:%M")
                                with open(tasks_file, 'w', encoding='utf-8') as f: json.dump(current_tasks, f, indent=4)
                                st.toast(f"✅ {t_name} 手动更新完毕！")
                            else: st.error(msg)
                            
                    if col_del.button("🗑️ " + _("删除任务", "Delete Task"), key=f"udel_{t_name}"):
                        del current_tasks[t_name]
                        with open(tasks_file, 'w', encoding='utf-8') as f: json.dump(current_tasks, f, indent=4)
                        st.rerun()

# ==========================================
# 📁 模块六：存储与日志仓库
# ==========================================
elif menu == _("📁 存储与日志仓库", "📁 Storage & Logs"):
    st.title(_("📁 存储与日志仓库", "📁 Storage & Logs Manager"))
    st.markdown(_("管理系统运行期间产生的所有日志、模型文件与 AI 心声记录。", 
                  "Manage all logs, models, and AI thoughts generated during system operation."))
    
    tab_logs, tab_thoughts, tab_models, tab_data, tab_tb = st.tabs([ # <--- 增加了 tab_tb
        _("📜 系统日志", "📜 System Logs"),
        _("🧠 读心记录", "🧠 Thought Replays"),
        _("🤖 模型仓库", "🤖 Model Storage"),
        _("📊 对局大盘数据", "📊 Match Data"),
        _("📉 TensorBoard 数据", "📉 TensorBoard Runs") # <--- 增加这行
    ])
    
    # 🌟 定义一个通用的文件管理器模板函数，直接消灭重复代码
    def build_file_manager(folder_path, ext, title_str, allow_view=False, allow_upload=False):
        os.makedirs(folder_path, exist_ok=True)
        files = sorted(glob.glob(os.path.join(folder_path, f"*{ext}")), key=os.path.getmtime, reverse=True)
        file_names = [os.path.basename(f) for f in files]
        
        # --- 导入/上传功能 ---
        if allow_upload:
            uploaded = st.file_uploader(_(f"📥 上传 {ext} 文件", f"📥 Upload {ext} file"), type=[ext.replace('.', '')], accept_multiple_files=True, key=f"up_{folder_path}")
            if uploaded and st.button(_("💾 保存上传", "Save Uploads"), key=f"save_{folder_path}"):
                try:
                    for f in uploaded:
                        safe_name = validate_local_asset_name(f.name, required_suffix=ext)
                        f.seek(0)
                        with open(os.path.join(folder_path, safe_name), "wb") as out:
                            shutil.copyfileobj(f, out, length=1024 * 1024)
                    st.success(_("✅ 上传成功！", "✅ Uploaded!"))
                    st.rerun()
                except Exception as error:
                    st.error(_(f"上传被拒绝: {error}", f"Upload rejected: {error}"))
            st.divider()

        if not files:
            st.info(_(f"{folder_path} 目录下暂无 {ext} 文件。", f"No {ext} files found in {folder_path}."))
            return

        # --- 导出/查看功能 ---
        sel_view = st.selectbox(_("🔍 选择文件操作 (查看/导出)", "🔍 Select file for View/Export"), file_names, key=f"view_{folder_path}")
        if sel_view:
            file_to_read = os.path.join(folder_path, sel_view)
            with open(file_to_read, "rb") as f: file_bytes = f.read()
            
            st.download_button(_("⬇️ 导出 (下载) 此文件", "Export (Download) this file"), data=file_bytes, file_name=sel_view, use_container_width=True, key=f"dl_{folder_path}")
            
            if allow_view:
                with st.expander(_("👀 预览文件尾部内容", "Preview File Content (Tail)")):
                    try:
                        content = file_bytes.decode('utf-8', errors='replace')
                        # 日志太长会导致浏览器卡死，只展示最后 10000 个字符
                        display_content = content[-10000:] if len(content) > 10000 else content
                        if len(content) > 10000: st.caption("*(文件过长，仅展示尾部内容)*")
                        st.code(display_content, language="bash" if ext == ".log" else "json")
                    except Exception as e:
                        st.warning(_("无法预览此文件。", "Cannot preview this file."))
        st.divider()
        
        # --- 批量删除功能 ---
        c_del, c_clr = st.columns([2, 1])
        with c_del:
            st.markdown(f"**🗑️ {_('批量删除', 'Batch Delete')}**")
            sel_del = st.multiselect(_("勾选要删除的文件", "Select files to delete"), file_names, key=f"del_{folder_path}")
            if st.button(_("删除选中文件", "Delete Selected"), key=f"btn_del_{folder_path}"):
                if sel_del:
                    for f in sel_del: os.remove(os.path.join(folder_path, f))
                    st.success(_("✅ 删除成功！", "✅ Deleted!"))
                    st.rerun()
                else:
                    st.warning(_("请先勾选文件！", "Please select files first!"))
                    
        # --- 安全清空功能 (双重确认锁) ---
        with c_clr:
            st.markdown(f"**🧨 {_('危险操作区', 'Danger Zone')}**")
            with st.expander(_("一键清空...", "Clear All...")):
                st.error(_(f"将彻底删除 `{folder_path}` 下所有文件！", f"Will completely delete all files in `{folder_path}`!"))
                confirm = st.checkbox(_("我确认清空", "I confirm to clear"), key=f"chk_clr_{folder_path}")
                # 只有勾选了确认框，删除按钮才会解锁 (disabled=not confirm)
                if st.button(_("彻底清空所有文件", "Clear ALL Files"), type="primary", disabled=not confirm, key=f"btn_clr_{folder_path}"):
                    for f in files: os.remove(f)
                    st.success(_("✅ 已全部清空！", "✅ Cleared all!"))
                    st.rerun()

    # 渲染标签页的内容
    with tab_logs:
        build_file_manager("./system_logs", ".log", _("系统运行日志", "System Logs"), allow_view=True, allow_upload=False)
    with tab_thoughts:
        build_file_manager("./ai_thoughts", ".json", _("AI 读心记录", "AI Thought Records"), allow_view=True, allow_upload=False)
    with tab_models:
        st.markdown("### 🧬 " + _("按模型 UUID 管理制品", "Artifacts grouped by model UUID"))
        st.caption(_(
            "PTH、ONNX 主图、ONNX 外置权重和轮次清单会按同一模型 UUID 与内置轮次归档。",
            "PTH, ONNX graphs, external ONNX data and iteration manifests are grouped by embedded UUID and iteration.",
        ))
        model_dir = os.path.abspath("./models")
        os.makedirs(model_dir, exist_ok=True)

        uploaded_models = st.file_uploader(
            _("📥 上传模型制品（可一次选择完整 ONNX 制品组）", "📥 Upload model artifacts as a complete bundle"),
            type=["pth", "onnx", "data", "json"],
            accept_multiple_files=True,
            key="model_artifact_bundle_upload",
            help=_(
                "ONNX 使用外置权重时必须同时选择 .onnx 与对应 .onnx.data；轮次清单可一并上传。",
                "External-data ONNX uploads must include both .onnx and the referenced .onnx.data file; manifests may be uploaded too.",
            ),
        )
        if uploaded_models and st.button(_("💾 校验并保存模型制品", "💾 Validate and save artifacts"), key="save_model_artifact_bundle"):
            try:
                seen_upload_names = set()
                with tempfile.TemporaryDirectory(prefix=".web_model_upload_", dir=model_dir) as stage_dir:
                    for uploaded in uploaded_models:
                        safe_name = validate_model_artifact_filename(uploaded.name)
                        folded_name = safe_name.casefold()
                        if folded_name in seen_upload_names:
                            raise ValueError(f"上传文件名重复: {safe_name}")
                        seen_upload_names.add(folded_name)
                        uploaded.seek(0)
                        with open(os.path.join(stage_dir, safe_name), "xb") as output:
                            shutil.copyfileobj(uploaded, output, length=1024 * 1024)
                    installed = install_model_artifact_bundle(
                        stage_dir,
                        model_dir,
                        require_all_artifacts=True,
                    )
                load_model_repository.clear()
                st.success(_(
                    f"✅ 已安全导入 {len(installed['files'])} 个模型制品文件。",
                    f"✅ Safely imported {len(installed['files'])} model artifact files.",
                ))
                st.rerun()
            except Exception as error:
                st.error(_(f"模型制品导入被拒绝: {error}", f"Model artifact import rejected: {error}"))

        st.divider()
        repository = load_model_repository(
            model_dir,
            get_model_identity_signature(model_dir),
        )
        pools = repository["pools"]
        invalid_artifacts = repository["invalid"]

        if pools:
            pool_ids = sorted(
                pools,
                key=lambda model_id: (
                    ",".join(pools[model_id]["prefixes"]).casefold(),
                    model_id,
                ),
            )

            def format_model_pool(model_id):
                """把模型 UUID 池格式化为便于选择的前缀与轮次摘要"""
                pool = pools[model_id]
                iterations = sorted(pool["iterations"])
                return (
                    f"{', '.join(pool['prefixes'])} | {model_id} | "
                    f"iter {iterations[0]}–{iterations[-1]} ({len(iterations)})"
                )

            selected_pool_id = st.selectbox(
                _("先选择模型池（UUID）", "Select model pool (UUID) first"),
                pool_ids,
                format_func=format_model_pool,
                key="storage_model_pool",
            )
            selected_pool = pools[selected_pool_id]
            if len(selected_pool["prefixes"]) > 1:
                st.warning(_(
                    "同一模型 UUID 出现多个前缀，已标记为身份异常；打包前会再次拒绝校验。",
                    "This UUID uses multiple prefixes. It is an identity anomaly and packaging will reject it.",
                ))
            st.code(selected_pool_id, language=None)

            artifact_rows = []
            pool_files = set()
            for artifact in selected_pool["artifacts"]:
                pool_files.update(artifact["files"])
                artifact_rows.append(
                    {
                        _("轮次", "Iteration"): artifact["iteration"],
                        _("格式", "Format"): artifact["format"],
                        _("主文件", "Primary"): artifact["primary"],
                        _("完整制品文件", "Artifact files"): ", ".join(artifact["files"]),
                    }
                )
            st.dataframe(artifact_rows, use_container_width=True, hide_index=True)

            downloadable_files = sorted(pool_files, key=str.casefold)
            selected_download = st.selectbox(
                _("选择单个制品文件下载（包含 .onnx.data）", "Select an individual artifact to download (including .onnx.data)"),
                downloadable_files,
                key="storage_model_download",
            )
            download_path = os.path.join(model_dir, selected_download)
            download_size = os.path.getsize(download_path)
            if download_size <= 256 * 1024 * 1024:
                with open(download_path, "rb") as download_stream:
                    st.download_button(
                        _("⬇️ 下载所选制品", "⬇️ Download selected artifact"),
                        data=download_stream.read(),
                        file_name=selected_download,
                        key="download_model_artifact",
                        use_container_width=True,
                    )
            else:
                st.info(_(
                    f"该文件为 {download_size / (1024 ** 3):.2f} GiB。为避免 WebUI 内存翻倍，请从本地路径复制：{download_path}",
                    f"This file is {download_size / (1024 ** 3):.2f} GiB. Copy it from {download_path} to avoid duplicating it in WebUI memory.",
                ))

            iteration_options = sorted(selected_pool["iterations"])
            delete_iterations = st.multiselect(
                _("选择要删除的完整轮次制品", "Select complete artifact iterations to delete"),
                iteration_options,
                key="storage_delete_model_iterations",
            )
            confirm_delete_iterations = st.checkbox(
                _("确认删除所选轮次的 PTH、ONNX、外置权重及清单", "Confirm deleting PTH, ONNX, external data and manifests for selected iterations"),
                key="confirm_delete_model_iterations",
            )
            if st.button(
                _("🗑️ 删除所选完整轮次", "🗑️ Delete selected complete iterations"),
                disabled=not (delete_iterations and confirm_delete_iterations),
                key="delete_model_iterations",
            ):
                files_to_delete = set()
                for iteration in delete_iterations:
                    for artifact in selected_pool["iterations"][iteration]:
                        files_to_delete.update(artifact["files"])
                for filename in files_to_delete:
                    safe_name = validate_model_artifact_filename(filename)
                    candidate = os.path.join(model_dir, safe_name)
                    if os.path.isfile(candidate) and not os.path.islink(candidate):
                        os.remove(candidate)
                load_model_repository.clear()
                st.success(_("✅ 已删除所选完整轮次。", "✅ Selected complete iterations deleted."))
                st.rerun()
        else:
            st.info(_("模型仓库中暂无可验证的当前协议模型。", "No verifiable current-protocol model is available."))

        if invalid_artifacts:
            with st.expander(_(
                f"⚠️ 无法归档或孤立的制品（{len(invalid_artifacts)}）",
                f"⚠️ Invalid or orphan artifacts ({len(invalid_artifacts)})",
            )):
                st.dataframe(invalid_artifacts, use_container_width=True, hide_index=True)
                invalid_names = [item["file"] for item in invalid_artifacts]
                delete_invalid = st.multiselect(
                    _("选择要删除的异常文件", "Select invalid files to delete"),
                    invalid_names,
                    key="delete_invalid_model_artifacts",
                )
                if st.button(
                    _("删除所选异常文件", "Delete selected invalid files"),
                    disabled=not delete_invalid,
                    key="delete_invalid_model_artifacts_button",
                ):
                    for filename in delete_invalid:
                        safe_name = validate_model_artifact_filename(filename)
                        candidate = os.path.join(model_dir, safe_name)
                        if os.path.isfile(candidate) and not os.path.islink(candidate):
                            os.remove(candidate)
                    load_model_repository.clear()
                    st.rerun()
    with tab_data: # <--- 新增
        build_file_manager("./web_data", ".csv", _("天梯对局数据", "Match Data"), allow_view=True, allow_upload=False)
    # 🌟 修复：TensorBoard 专属多选删除与一键清空逻辑
    with tab_tb:
        st.markdown("### 📉 " + _("TensorBoard 运行记录管理", "TensorBoard Runs Management"))
        st.markdown(_("TensorBoard 日志为庞大的二进制流，此处提供分文件夹批量删除与一键清理功能。", 
                      "Manage TensorBoard event files. Support for batch deletion and one-click purging."))
        
        tb_dir = "./runs"
        os.makedirs(tb_dir, exist_ok=True)
        
        # 获取所有运行记录文件夹
        run_folders = sorted([f for f in os.listdir(tb_dir) if os.path.isdir(os.path.join(tb_dir, f))], reverse=True)
        
        # 实时容量统计
        tb_size = sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(tb_dir) for f in filenames)
        st.info(f"**{_('当前 ./runs 目录占用空间', 'Current Disk Usage')}:** {tb_size / (1024 * 1024):.2f} MB")
        
        if run_folders:
            # 🌟 新增：批量多选删除区
            st.markdown("#### 📂 " + _("记录列表", "Run Records"))
            selected_runs = st.multiselect(_("选择要删除的特定记录", "Select specific runs to delete"), run_folders)
            if st.button(_("🗑️ 删除选中记录", "Delete Selected Runs"), type="secondary", disabled=not selected_runs):
                for folder in selected_runs:
                    shutil.rmtree(os.path.join(tb_dir, folder))
                st.success(_("✅ 指定记录已清理！", "✅ Selected runs cleared!"))
                st.rerun()

            # 原有的危险操作区
            with st.expander(_("🧨 危险操作区 (一键清空...)", "Danger Zone (Clear All...)")):
                st.error(_(f"将彻底删除 `{tb_dir}` 下所有文件！注意：这会重置所有历史训练曲线！", 
                           f"Will completely delete all files in `{tb_dir}`!"))
                confirm_tb = st.checkbox(_("我确认清空", "I confirm to clear"), key="chk_clr_runs")
                
                if st.button(_("彻底清空所有数据", "Clear ALL Data"), type="primary", disabled=not confirm_tb):
                    shutil.rmtree(tb_dir)
                    os.makedirs(tb_dir, exist_ok=True)
                    st.success(_("✅ 已全部清空记录！", "✅ Cleared all!"))
                    st.rerun()
        else:
            st.info(_("暂无运行记录", "No run records found."))

# ==========================================
# 👁️ 模块七：全息读心回放 (战术沙盘 V7.0)
# ==========================================
elif menu == _("👁️ 全息读心回放", "👁️ Holographic Replay"):
    import time
    import re
    
    st.title(_("👁️ 全息战术沙盘", "👁️ Holographic Tactical Board"))
    st.markdown(_("真实坐标系连线，沉浸式悬停状态窗，动态墓地/额外/除外实体渲染。", 
                  "True coordinate SVG tracking, immersive hover tooltips, and dynamic off-field rendering."))
    
    log_dir = "./ai_thoughts"
    os.makedirs(log_dir, exist_ok=True)
    json_files = sorted(glob.glob(os.path.join(log_dir, "*.json")), key=os.path.getmtime, reverse=True)
    
    if not json_files:
        st.info(_("暂无录像文件。请先在竞技场生成对局。", "No replay files found."))
    else:
        c_file, c_space = st.columns([2, 1])
        with c_file:
            sel_file = st.selectbox(_("📂 选择录像文件", "Select Replay File"), [os.path.basename(f) for f in json_files])
        
        with open(os.path.join(log_dir, sel_file), "r", encoding="utf-8") as f:
            try: replay_data = json.load(f)
            except Exception: replay_data = {}
            
        replay_frames = get_replay_frames(replay_data)
        if not replay_frames:
            st.warning(_("录像格式过旧或损坏。", "Replay is empty or old format."))
        else:
            max_steps = len(replay_frames) - 1
            sync_replay_session(st.session_state, sel_file, max_steps)
            
            # 提前注册透视开关状态，防止回收
            if "tgl_p1_hand" not in st.session_state: st.session_state.tgl_p1_hand = False
            if "tgl_p0_hand" not in st.session_state: st.session_state.tgl_p0_hand = True
            if "tgl_p1_set" not in st.session_state: st.session_state.tgl_p1_set = False
            if "tgl_p0_set" not in st.session_state: st.session_state.tgl_p0_set = True
            if "tgl_rotate_p1" not in st.session_state: st.session_state.tgl_rotate_p1 = True
            if "tgl_p1_confidence" not in st.session_state: st.session_state.tgl_p1_confidence = True

            st.markdown(f"**🤖 {_('模型', 'Model')}:** `{replay_data.get('model_name', 'Unknown')}` &nbsp;|&nbsp; **🏆 {_('胜者', 'Winner')}:** `P{replay_data.get('winner', '?')} ({replay_data.get('win_reason', '结束')})`")
            
            # --- 🎛️ 控制台 (绝对去重版) ---
            st.markdown("---")
            ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1,1,1,2,5])
            
            def step_prev():
                set_replay_cursor(st.session_state, st.session_state.replay_step - 1, max_steps)
                st.session_state.is_playing = False
            def step_next():
                set_replay_cursor(st.session_state, st.session_state.replay_step + 1, max_steps)
                st.session_state.is_playing = False
            def toggle_play():
                st.session_state.is_playing = not st.session_state.is_playing
            def on_slider():
                st.session_state.replay_step = st.session_state.step_slider_widget
                st.session_state.is_playing = False
            
            ctrl1.button("⏮️ 上一步", key="btn_prev_step", on_click=step_prev, **REPLAY_BUTTON_WIDTH)
            play_label = "⏸️ 暂停播放" if st.session_state.is_playing else "▶️ 自动播放"
            ctrl2.button(play_label, type="primary" if st.session_state.is_playing else "secondary", key="btn_toggle_play", on_click=toggle_play, **REPLAY_BUTTON_WIDTH)
            ctrl3.button("⏭️ 下一步", key="btn_next_step", on_click=step_next, **REPLAY_BUTTON_WIDTH)

            play_speed = ctrl4.slider("⏱️ 播放间隔", 0.5, 5.0, 1.5, 0.5, label_visibility="collapsed", key="slider_play_speed")
            if max_steps > 0:
                st.slider("时间轴", 0, max_steps, format="Step %d", label_visibility="collapsed", key="step_slider_widget", on_change=on_slider)
            else:
                st.caption(_("该录像仅包含一个有效帧。", "This replay contains one frame."))
                
            t_col1, t_col2, t_col3, t_col4, t_col5, t_col6 = st.columns(6)
            t_col1.toggle("👁️ P1 手牌", key="tgl_p1_hand")
            t_col2.toggle("👁️ P0 手牌", key="tgl_p0_hand")
            t_col3.toggle("👁️ P1 盖卡", key="tgl_p1_set")
            t_col4.toggle("👁️ P0 盖卡", key="tgl_p0_set")
            t_col5.toggle("🔄 P1 翻转180°", key="tgl_rotate_p1")
            t_col6.toggle(_("📊 P1 置信度", "📊 P1 Confidence"), key="tgl_p1_confidence")

            step_data = replay_frames[st.session_state.replay_step]
            state = get_replay_frame_state(replay_data, step_data)
            replay_decklists = get_replay_decklists(replay_data)
            api_lang = "sc" if lang == "🇨🇳 中文" else "en"

            frame_player = step_data.get("player")
            show_option_table = (
                frame_player != 1 or st.session_state.tgl_p1_confidence
            )
            preview_frame_id = f"{sel_file}:{st.session_state.replay_step}"
            if st.session_state.get("replay_preview_frame_id") != preview_frame_id:
                st.session_state.replay_preview_frame_id = preview_frame_id
                st.session_state.replay_preview_option_index = None
            preview_option_index = (
                st.session_state.get("replay_preview_option_index")
                if show_option_table
                else None
            )
            frame_visuals = get_frame_visuals(step_data, preview_option_index)
            chosen_opt = frame_visuals["chosen"]
            preview_opt = frame_visuals["preview"]
            action_desc = chosen_opt.get("desc", "")
            actor_data = frame_visuals["actor"]
            target_data = frame_visuals["targets"][0] if frame_visuals["targets"] else None
            target_keys = {
                (target.get("owner"), target.get("loc"), target.get("seq"))
                for target in frame_visuals["targets"]
                if target
            }
            event_data = frame_visuals["event"]

            css_style = """
            <style>
            .ygo-hand { display: flex; justify-content: center; gap: 5px; min-height: 90px; margin: 10px 0; background: rgba(0,0,0,0.2); padding: 5px; border-radius: 8px;}
            .ygo-board-wrapper { position: relative; width: max-content; margin: 0 auto; }
            .ygo-board {
                display: grid;
                grid-template-columns: 75px repeat(5, 75px) 75px;
                grid-template-rows: 106px 106px 106px 106px 106px;
                gap: 6px; justify-content: center;
                background: #1e1e1e; padding: 20px; border-radius: 12px; box-shadow: inset 0 0 20px #000;
            }
            .ygo-cell {
                width: 75px; height: 106px;
                border: 2px dashed #444; border-radius: 5px; display: flex; justify-content: center; align-items: center;
                background: rgba(0,0,0,0.4); position: relative;
            }
            .ygo-cell.empty-space { border: none; background: transparent; }
            
            .ygo-card-wrapper { transition: transform 0.2s; z-index: 10; display: flex; align-items: center; justify-content: center; height: 100%; width: 100%; position: relative;}
            .ygo-card-wrapper:hover { transform: scale(1.4); z-index: 999; }
            
            .ygo-card { width: 100%; height: auto; aspect-ratio: 59/86; object-fit: contain; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.6); pointer-events: auto;}
            .facedown { background: repeating-linear-gradient(45deg, #4a2e15, #4a2e15 10px, #36210f 10px, #36210f 20px); border: 2px solid #222; }
            
            /* 🌟 核心新增：透视幽灵特效 */
            @keyframes ghostPulse {
                0% { opacity: 0.5; box-shadow: 0 0 5px rgba(0, 255, 255, 0.4); }
                50% { opacity: 0.9; box-shadow: 0 0 15px rgba(0, 255, 255, 0.8); }
                100% { opacity: 0.5; box-shadow: 0 0 5px rgba(0, 255, 255, 0.4); }
            }
            .xray-ghost {
                animation: ghostPulse 2.5s infinite ease-in-out;
                border: 2px dashed rgba(0, 255, 255, 0.7) !important;
            }

            @keyframes pulseActor { 0% {box-shadow: 0 0 10px #0ff; border: 2px solid #0ff;} 50% {box-shadow: 0 0 25px #08f; border: 2px solid #08f;} 100% {box-shadow: 0 0 10px #0ff; border: 2px solid #0ff;} }
            .hl-actor { animation: pulseActor 1.5s infinite; border-radius: 4px;}
            @keyframes pulseTarget { 0% {box-shadow: 0 0 10px #f00; border: 2px solid #f00;} 50% {box-shadow: 0 0 25px #fa0; border: 2px solid #fa0;} 100% {box-shadow: 0 0 10px #f00; border: 2px solid #f00;} }
            .hl-target { animation: pulseTarget 1.5s infinite; border-radius: 4px;}
            
            .svg-overlay { position: absolute; top: 20px; left: 20px; width: calc(100% - 40px); height: calc(100% - 40px); pointer-events: none; z-index: 50; overflow: visible; }
            .dash-line { stroke-dasharray: 10; animation: dashAnim 1s linear infinite; }
            @keyframes dashAnim { to { stroke-dashoffset: -20; } }
            .deck-card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(52px, 1fr)); gap: 7px; padding: 8px 2px; }
            .deck-card-item { position: relative; min-width: 0; }
            .deck-card-item a { display: block; }
            .deck-card-count { position: absolute; right: -3px; bottom: -3px; min-width: 22px; padding: 1px 4px; border-radius: 10px; background: #00a9c7; color: white; font-size: 11px; font-weight: 700; text-align: center; box-shadow: 0 1px 4px #000; }
            .deck-list-empty { color: #888; padding: 12px; text-align: center; }
            [data-testid="column"] { padding: 0.2rem !important; }
            </style>
            """
            
            ATTR_MAP = {1:"地", 2:"水", 4:"炎", 8:"风", 16:"光", 32:"暗", 64:"神"}
            RACE_MAP = {1:"战士", 2:"魔法师", 4:"天使", 8:"恶魔", 16:"不死", 32:"机械", 64:"水", 128:"炎", 256:"岩石", 512:"鸟兽", 1024:"植物", 2048:"昆虫", 4096:"雷", 8192:"龙", 16384:"兽", 32768:"兽战士", 65536:"恐龙", 131072:"鱼", 262144:"海龙", 524288:"爬虫", 1048576:"念动力", 2097152:"幻神兽", 4194304:"创造神", 8388608:"幻龙", 16777216:"电子界", 33554432:"幻想魔"}

            def get_grid_pos(owner, loc, seq):
                if loc == 0x100:
                    return (3, 6) if owner == 0 else (3, 2)
                if owner == 0:
                    if loc == 0x04:
                        if seq < 5: return (4, seq + 2)
                        elif seq == 5: return (3, 3)
                        elif seq == 6: return (3, 5)
                    elif loc == 0x08:
                        if seq < 5: return (5, seq + 2)
                        elif seq == 5: return (4, 1)
                    elif loc == 0x10: return (4, 7)
                    elif loc == 0x20: return (3, 7)
                    elif loc == 0x40: return (5, 1)
                    elif loc == 0x01: return (5, 7)
                    elif loc == 0x02: return (6, 4)
                elif owner == 1:
                    if loc == 0x04:
                        if seq < 5: return (2, 6 - seq)
                        elif seq == 5: return (3, 5)
                        elif seq == 6: return (3, 3)
                    elif loc == 0x08:
                        if seq < 5: return (1, 6 - seq)
                        elif seq == 5: return (2, 7)
                    elif loc == 0x10: return (2, 1)
                    elif loc == 0x20: return (3, 1)
                    elif loc == 0x40: return (1, 7)
                    elif loc == 0x01: return (1, 1)
                    elif loc == 0x02: return (0, 4)
                return None

            def render_card(c, loc, owner, is_hand=False, force_faceup=False):
                if not c: return "" 
                code = c.get("code", 0)
                pos = c.get("pos", 0)
                seq = c.get("seq", 0)
                
                is_mine = (owner == 0)
                is_set = (pos in [0x2, 0x8, 0xa]) or (code == 0)
                
                should_hide = False
                if not force_faceup:
                    if is_hand: should_hide = not st.session_state.tgl_p0_hand if is_mine else not st.session_state.tgl_p1_hand
                    elif is_set: should_hide = not st.session_state.tgl_p0_set if is_mine else not st.session_state.tgl_p1_set
                
                is_def = pos in [0x4, 0x8, 0xa]
                is_monster_zone = (loc == 0x04) 
                
                rotation = 0
                if is_def and is_monster_zone:
                    rotation = 90 if is_mine else (-90 if st.session_state.tgl_rotate_p1 else 90)
                elif not is_mine and st.session_state.tgl_rotate_p1 and loc in [0x04, 0x08, 0x40, 0x10, 0x20]:
                    rotation = 180

                hl_class = ""
                if actor_data and actor_data.get('owner') == owner and actor_data.get('loc') == loc and actor_data.get('seq') == seq:
                    hl_class = " hl-actor"
                elif (owner, loc, seq) in target_keys:
                    hl_class = " hl-target"

                # 🌟 核心优化：透视特效仅作用于场上的里侧卡片 (排除手牌)
                # loc == 0x04 是怪兽区，loc == 0x08 是魔陷区
                is_on_field = (loc == 0x04 or loc == 0x08)
                xray_class = " xray-ghost" if (is_set and not should_hide and code != 0 and is_on_field) else ""

                c_name = card_db_ui.get_card_name(code) if code else "未知卡片"
                
                title = ""
                if is_set: title += "[面朝下] "
                elif is_hand: title += "[手牌] "
                title += f"【{c_name}】 ({code})"
                
                if code != 0:
                    stats = card_db_ui.get_full_stats(code)
                    t_race = RACE_MAP.get(stats[1], "未知")
                    t_attr = ATTR_MAP.get(stats[2], "未知")
                    if stats[8] > 0 or stats[9] > 0 or loc == 0x04: 
                        title += f"&#10;⚔️ ATK: {c.get('atk', stats[8])} | 🛡️ DEF: {c.get('def', stats[9])}"
                    if stats[3] > 0: title += f"&#10;⭐ 星级/阶级: {c.get('lvl', stats[3])}"
                    title += f"&#10;🧬 种族: {t_race} | 🔮 属性: {t_attr}"
                    if c.get("overlays", 0) > 0: title += f"&#10;💿 Xyz 素材数: {c.get('overlays', 0)}"
                    if c.get("counters", 0) > 0: title += f"&#10;🎲 指示物: {c.get('counters', 0)}"

                transform_style = f"transform: rotate({rotation}deg);" if rotation != 0 else ""
                
                if should_hide or code == 0:
                    return f'<div class="ygo-card-wrapper"><div class="ygo-card facedown{hl_class}" style="{transform_style}" title="{title}"></div></div>'
                else:
                    img_url = f"https://cdn.233.momobako.com/ygoimg/{api_lang}/{code}.webp!half"
                    # 🌟 注入幽灵特效类 xray_class
                    return f'<div class="ygo-card-wrapper"><a href="https://ygocdb.com/card/{code}" target="_blank" style="display:block; width:100%; height:100%;"><img src="{img_url}" class="ygo-card{hl_class}{xray_class}" style="{transform_style}" title="{title}"></a></div>'

            def render_carried_deck(codes):
                """把完整卡组按同名卡合并为带数量角标的卡图网格。"""
                cards_html = []
                for code, count in group_replay_card_codes(codes):
                    card_name = html.escape(
                        card_db_ui.get_card_name(code) or str(code),
                        quote=True,
                    )
                    img_url = f"https://cdn.233.momobako.com/ygoimg/{api_lang}/{code}.webp!half"
                    count_badge = (
                        f'<span class="deck-card-count">×{count}</span>'
                        if count > 1 else ""
                    )
                    cards_html.append(
                        '<div class="deck-card-item">'
                        f'<a href="https://ygocdb.com/card/{code}" target="_blank">'
                        f'<img src="{img_url}" class="ygo-card" loading="lazy" title="【{card_name}】 ({code}) ×{count}">'
                        f'</a>{count_badge}</div>'
                    )
                if not cards_html:
                    return '<div class="deck-list-empty">无卡片 / Empty</div>'
                return '<div class="deck-card-grid">' + "".join(cards_html) + '</div>'

            grid_html = { (r, c): '<div class="ygo-cell"></div>' for r in range(1, 6) for c in range(1, 8) }
            for c in [2, 4, 6]: grid_html[(3, c)] = '<div class="ygo-cell empty-space"></div>'

            to_play = state.get("to_play", 0)
            turn_str = "我方回合" if to_play == 0 else "敌方回合"
            grid_html[(3, 4)] = f'<div style="text-align:center;color:#fff;"><b>Turn {step_data.get("turn", "?")}</b><br><span style="color:#0ff">{step_data.get("phase", "?")}</span><br><span style="font-size:12px;color:#aaa;">({turn_str})</span></div>'

            # 🌟 修复 2：在空位注入双方 LP 显示 (行3列2为 P1_LP，行3列6为 P0_LP)
            grid_html[(3, 2)] = f'<div class="ygo-cell empty-space" style="display:flex; flex-direction:column; color:#ff4444; font-weight:bold; font-size:18px; text-shadow: 1px 1px 2px #000;"><span>P1 LP</span><span>{state.get("p1_lp", 8000)}</span></div>'
            grid_html[(3, 6)] = f'<div class="ygo-cell empty-space" style="display:flex; flex-direction:column; color:#44ff44; font-weight:bold; font-size:18px; text-shadow: 1px 1px 2px #000;"><span>P0 LP</span><span>{state.get("p0_lp", 8000)}</span></div>'

            occupied_cells = set()

            def fill_board(zone_list, owner, loc):
                """填充场地区卡片并记录已占用坐标。"""
                for c in zone_list:
                    rc = get_grid_pos(owner, loc, c['seq'])
                    if rc:
                        occupied_cells.add(rc)
                        grid_html[rc] = f'<div class="ygo-cell">{render_card(c, loc, owner)}</div>'
                    
            fill_board(state.get('p0_mzone', []), 0, 0x04); fill_board(state.get('p0_szone', []), 0, 0x08)
            fill_board(state.get('p1_mzone', []), 1, 0x04); fill_board(state.get('p1_szone', []), 1, 0x08)

            # 🌟 修复魔法区越界 Bug：强制检查 expected_seq
            def fixed_cell(r, c, label, count, loc_id, owner_id, expected_seq=-1):
                """绘制固定区域计数，并在事件命中时显示对应卡片。"""
                active_c = None
                for act_data in [actor_data] + frame_visuals["targets"]:
                    if act_data and act_data.get('loc') == loc_id and act_data.get('owner') == owner_id:
                        if expected_seq == -1 or act_data.get('seq') == expected_seq:
                            active_c = act_data
                            break
                if active_c:
                    content = render_card(active_c, loc_id, owner_id, force_faceup=True)
                else:
                    content = f'<div style="color:gray;font-weight:bold;text-align:center;width:100%;">{label}<br>{count}</div>'
                grid_html[(r, c)] = f'<div class="ygo-cell" style="border: 2px solid #555;">{content}</div>'

            # 🌟 修复 1：读取刚刚在 Logger 中增加的 _len 字段 (并保留旧日志兼容)
            fixed_cell(5, 7, "Deck", state.get("p0_deck_len", "?"), 0x01, 0)
            fixed_cell(1, 1, "Deck", state.get("p1_deck_len", "?"), 0x01, 1)
            fixed_cell(4, 7, "Grave", len(state.get('p0_grave', [])), 0x10, 0)
            fixed_cell(2, 1, "Grave", len(state.get('p1_grave', [])), 0x10, 1)
            fixed_cell(3, 7, "Removed", len(state.get('p0_removed', [])), 0x20, 0)
            fixed_cell(3, 1, "Removed", len(state.get('p1_removed', [])), 0x20, 1)
            
            # 使用 get 获取新加的 extra_len 字段，如果没有（读取老日志）则 fallback 退回读列表长度
            fixed_cell(5, 1, "Extra", state.get("p0_extra_len", len(state.get('p0_extra', []))), 0x40, 0)
            fixed_cell(1, 7, "Extra", state.get("p1_extra_len", len(state.get('p1_extra', []))), 0x40, 1)
            
            # 🌟 强制检查 seq==5 才算场地区
            fixed_cell(4, 1, "Field", "", 0x08, 0, expected_seq=5)
            fixed_cell(2, 7, "Field", "", 0x08, 1, expected_seq=5)

            # 离场移动后源卡已经不在快照中，使用事件幽灵卡保留移动起点。
            for event_card in [actor_data] + frame_visuals["targets"]:
                if not event_card or event_card.get("loc") not in (0x04, 0x08):
                    continue
                rc = get_grid_pos(
                    event_card.get("owner", -1),
                    event_card.get("loc", 0),
                    event_card.get("seq", 0),
                )
                if rc and rc not in occupied_cells:
                    grid_html[rc] = (
                        '<div class="ygo-cell" style="opacity:0.72;">'
                        + render_card(
                            event_card,
                            event_card.get("loc", 0),
                            event_card.get("owner", -1),
                            force_faceup=True,
                        )
                        + '</div>'
                    )

            # 🏹 按动作、连锁、攻击和移动类型绘制多目标有向箭头。
            svg_html = ""
            arrow_colors = {
                "move": "#00d4ff", "attack": "#ff3b30", "chain": "#c65cff",
                "equip": "#ffd60a", "card_target": "#ff9500", "action": "#ff3b30",
            }
            svg_lines = []
            svg_markers = []
            for arrow_index, arrow in enumerate(frame_visuals["arrows"]):
                source = arrow.get("from") or {}
                target = arrow.get("to") or {}
                arc = get_grid_pos(source.get('owner'), source.get('loc'), source.get('seq', 0))
                trc = get_grid_pos(target.get('owner'), target.get('loc'), target.get('seq', 0))
                if not arc or not trc or arc == trc:
                    continue
                color = arrow_colors.get(arrow.get("kind"), "#ff3b30")
                marker_id = f"arrowhead-{arrow_index}"
                ax = (arc[1] - 1) * 81 + 37.5; ay = (arc[0] - 1) * 112 + 53
                tx = (trc[1] - 1) * 81 + 37.5; ty = (trc[0] - 1) * 112 + 53
                svg_markers.append(
                    f'<marker id="{marker_id}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
                    f'<polygon points="0 0, 10 3.5, 0 7" fill="{color}" /></marker>'
                )
                svg_lines.append(
                    f'<line x1="{ax}" y1="{ay}" x2="{tx}" y2="{ty}" stroke="{color}" '
                    f'stroke-width="4" class="dash-line" marker-end="url(#{marker_id})" />'
                )
            if svg_lines:
                svg_html = (
                    '<svg class="svg-overlay"><defs>'
                    + ''.join(svg_markers)
                    + '</defs>'
                    + ''.join(svg_lines)
                    + '</svg>'
                )

            board_html = '<div class="ygo-board-wrapper"><div class="ygo-board">'
            for r in range(1, 6):
                for c in range(1, 8): board_html += grid_html[(r, c)]
            board_html += '</div>' + svg_html + '</div>'

            p1_hand_html = '<div class="ygo-hand">' + "".join([f'<div style="width:60px;">{render_card(c, 0x02, 1, True)}</div>' for c in state.get('p1_hand', [])]) + '</div>'
            p0_hand_html = '<div class="ygo-hand">' + "".join([f'<div style="width:60px;">{render_card(c, 0x02, 0, True)}</div>' for c in state.get('p0_hand', [])]) + '</div>'

            st.markdown(css_style, unsafe_allow_html=True)
            c_left, c_right = st.columns([6, 4])
            
            with c_left:
                st.markdown(p1_hand_html, unsafe_allow_html=True)
                st.markdown(board_html, unsafe_allow_html=True)
                st.markdown(p0_hand_html, unsafe_allow_html=True)

                with st.expander("📚 " + _("查看双方完整卡组", "View Both Full Decks")):
                    st.caption(_(
                        "这里显示开局携带清单，仅用于赛后审计，不会进入模型观测。",
                        "Initial carried lists for post-game review only; they are not model observations.",
                    ))
                    if replay_decklists:
                        deck_tabs = []
                        deck_sections = []
                        for player in ("0", "1"):
                            deck_entry = replay_decklists.get(player, {})
                            deck_name = deck_entry.get("name") or _("未命名卡组", "Unnamed Deck")
                            for section, section_label in (
                                ("main", _("主卡组", "Main")),
                                ("extra", _("额外卡组", "Extra")),
                            ):
                                codes = deck_entry.get(section, [])
                                deck_tabs.append(
                                    f"P{player} {section_label} · {deck_name} ({len(codes)})"
                                )
                                deck_sections.append(codes)
                        for tab, codes in zip(st.tabs(deck_tabs), deck_sections):
                            with tab:
                                st.markdown(
                                    render_carried_deck(codes),
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.info(_(
                            "这份旧录像没有保存开局卡组；重新录制的对局会自动包含该信息。",
                            "This older replay has no initial deck list; newly recorded games include it automatically.",
                        ))

                with st.expander("📂 " + _("展开查看 墓地 / 除外 / 额外 详情", "Expand Grave / Banished / Extra View")):
                    col_p1, col_p0 = st.columns(2)
                    with col_p1:
                        st.write("P1 墓地 Grave"); st.markdown('<div class="ygo-hand" style="flex-wrap:wrap;">' + "".join([f'<div style="width:50px;">{render_card(c, 0x10, 1, force_faceup=True)}</div>' for c in state.get('p1_grave', [])]) + '</div>', unsafe_allow_html=True)
                        st.write("P1 除外 Banished"); st.markdown('<div class="ygo-hand" style="flex-wrap:wrap;">' + "".join([f'<div style="width:50px;">{render_card(c, 0x20, 1, force_faceup=True)}</div>' for c in state.get('p1_removed', [])]) + '</div>', unsafe_allow_html=True)
                        st.write("P1 额外 Extra"); st.markdown('<div class="ygo-hand" style="flex-wrap:wrap;">' + "".join([f'<div style="width:50px;">{render_card(c, 0x40, 1, force_faceup=True)}</div>' for c in state.get('p1_extra', [])]) + '</div>', unsafe_allow_html=True)
                    with col_p0:
                        st.write("P0 墓地 Grave"); st.markdown('<div class="ygo-hand" style="flex-wrap:wrap;">' + "".join([f'<div style="width:50px;">{render_card(c, 0x10, 0, force_faceup=True)}</div>' for c in state.get('p0_grave', [])]) + '</div>', unsafe_allow_html=True)
                        st.write("P0 除外 Banished"); st.markdown('<div class="ygo-hand" style="flex-wrap:wrap;">' + "".join([f'<div style="width:50px;">{render_card(c, 0x20, 0, force_faceup=True)}</div>' for c in state.get('p0_removed', [])]) + '</div>', unsafe_allow_html=True)
                        st.write("P0 额外 Extra"); st.markdown('<div class="ygo-hand" style="flex-wrap:wrap;">' + "".join([f'<div style="width:50px;">{render_card(c, 0x40, 0, force_faceup=True)}</div>' for c in state.get('p0_extra', [])]) + '</div>', unsafe_allow_html=True)

            with c_right:
                frame_agent = step_data.get("agent") or (
                    replay_data.get("players", {}).get(str(frame_player), "Core")
                    if frame_player is not None else "Core"
                )
                st.caption(
                    f"Frame {st.session_state.replay_step + 1}/{len(replay_frames)} · "
                    f"{step_data.get('frame_type', 'decision')} · "
                    f"{('P' + str(frame_player) + ' / ') if frame_player is not None else ''}{frame_agent} · "
                    f"MsgType {step_data.get('msg_type', '?')}"
                )

                if event_data:
                    st.subheader("🎬 " + _("Core 事件", "Core Event"))
                    st.info(event_data.get("label", f"Core {event_data.get('msg_type', '?')}"))
                    if event_data.get("kind") == "lp":
                        lp_before = event_data.get("lp_before")
                        lp_after = event_data.get("lp_after")
                        lp_delta = event_data.get("lp_delta")
                        delta_text = f"{lp_delta:+d}" if isinstance(lp_delta, int) else None
                        st.metric(
                            f"P{event_data.get('player', '?')} LP",
                            lp_after,
                            delta=delta_text,
                        )
                        if lp_before is not None:
                            st.caption(f"{lp_before} → {lp_after}")

                chain_list = state.get("chain", [])
                hist_list = state.get("history", [])
                
                if chain_list or hist_list:
                    st.subheader("📜 " + _("场面态势与连锁", "Board History & Chain"))
                    if chain_list:
                        for ch in chain_list:
                            cname = card_db_ui.get_card_name(ch.get('code', 0))
                            st.warning(f"⚡ 连锁 {ch.get('ct', '?')}: {cname}")
                    if hist_list:
                        h_names = [card_db_ui.get_card_name(h.get('code', 0)) for h in hist_list[:5]]
                        st.caption(f"最近动作: {' ➡️ '.join(h_names)}")
                        
                st.subheader("🧠 " + _("双方决策与动作推演", "Both Players' Operations"))
                if action_desc:
                    arr_hint = " 🎯 [箭头方向：发起者 → 目标]" if svg_html else ""
                    player_label = f"P{frame_player}" if frame_player is not None else "AI"
                    st.info(f"👉 **{player_label} 最终决定:** {action_desc}{arr_hint}")
                    semantic_details = format_action_semantics(chosen_opt.get("semantic"))
                    if semantic_details:
                        st.caption(" ｜ ".join(semantic_details))

                if preview_option_index is not None and preview_opt:
                    preview_desc = preview_opt.get("desc", "") or _(
                        "未命名候选",
                        "Unnamed option",
                    )
                    st.info(_(
                        f"🔎 当前候选预览：[{preview_option_index}] {preview_desc}",
                        f"🔎 Candidate preview: [{preview_option_index}] {preview_desc}",
                    ))
                    preview_semantics = format_action_semantics(
                        preview_opt.get("semantic")
                    )
                    if preview_semantics:
                        st.caption(" ｜ ".join(preview_semantics))

                    # 候选涉及的卡片在表格下直接展示，同时在左侧棋盘标亮对应位置。
                    preview_cards = []
                    seen_preview_cards = set()
                    for role, card in [
                        (_("发起者", "Actor"), frame_visuals["actor"]),
                        *[
                            (_("目标/素材", "Target/Material"), target)
                            for target in frame_visuals["targets"]
                        ],
                    ]:
                        if not card:
                            continue
                        identity = (
                            card.get("owner"),
                            card.get("loc"),
                            card.get("seq"),
                            card.get("code"),
                        )
                        if identity in seen_preview_cards:
                            continue
                        seen_preview_cards.add(identity)
                        preview_cards.append((role, card))
                    if preview_cards:
                        preview_card_html = '<div class="ygo-hand" style="flex-wrap:wrap;">'
                        for role, card in preview_cards[:8]:
                            preview_card_html += (
                                '<div style="width:76px;text-align:center;color:#bbb;">'
                                f'<div style="font-size:11px;margin-bottom:3px;">{role}</div>'
                                + render_card(
                                    card,
                                    card.get("loc", 0),
                                    card.get("owner", -1),
                                    force_faceup=True,
                                )
                                + '</div>'
                            )
                        preview_card_html += '</div>'
                        st.markdown(preview_card_html, unsafe_allow_html=True)
                        if len(preview_cards) > 8:
                            st.caption(_(
                                f"另有 {len(preview_cards) - 8} 个目标/素材未展开。",
                                f"{len(preview_cards) - 8} more targets/materials are hidden.",
                            ))
                
                df_data = []
                for opt in step_data.get("options", []):
                    df_data.append({
                        _("采纳", "Action"): "✅" if opt.get("is_chosen") else "",
                        _("信心", "Confidence"): f"{opt.get('confidence', 0)*100:.1f}%",
                        _("动作推演", "Operation Detail"): opt.get("desc", ""),
                        _("协议语义", "Protocol Semantics"): " ｜ ".join(
                            format_action_semantics(opt.get("semantic"))
                        ),
                    })
                
                df = pd.DataFrame(df_data)
                def highlight_row(row):
                    if row[_("采纳", "Action")] == "✅": return ['background-color: rgba(0, 255, 255, 0.2); font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                if not df.empty and show_option_table:
                    option_table_key = (
                        f"replay_option_table::{sel_file}::"
                        f"{st.session_state.replay_step}"
                    )
                    table_event = st.dataframe(
                        df.style.apply(highlight_row, axis=1),
                        height=550,
                        hide_index=True,
                        key=option_table_key,
                        on_select="rerun",
                        selection_mode="single-row",
                        **REPLAY_DATAFRAME_WIDTH,
                    )
                    selected_option_index = get_selected_replay_option_index(
                        table_event,
                        len(df_data),
                    )
                    if selected_option_index != preview_option_index:
                        st.session_state.replay_preview_option_index = selected_option_index
                        st.session_state.is_playing = False
                        st.rerun()
                elif not df.empty and frame_player == 1:
                    st.caption(_(
                        "P1 候选置信度表已隐藏；可使用顶部开关重新显示。",
                        "P1 confidence table is hidden; use the top toggle to show it.",
                    ))

            if st.session_state.is_playing:
                time.sleep(play_speed)
                if st.session_state.replay_step < max_steps:
                    # 滑块已经在本轮实例化，只推进逻辑游标并交给下一轮同步控件键。
                    queue_replay_cursor(st.session_state, st.session_state.replay_step + 1, max_steps)
                    st.rerun()
                else:
                    st.session_state.is_playing = False
                    st.rerun()

# ==========================================
# 📉 模块 X：TensorBoard 训练流形图
# ==========================================
elif menu == _("📉 训练流形图", "📉 TensorBoard"):
    st.title(_("📉 TensorBoard 训练大盘", "📉 TensorBoard Dashboard"))
    st.markdown(_("在这里直接观测 AI 训练的 Loss 曲线、熵值(探索欲)、以及各卡组的胜率走势！", 
                  "View training curves, entropy, and win rates here directly!"))

    # 🌟 修复 1: 强制使用 127.0.0.1 替代 localhost，防止 IPv6 解析横跳导致的闪烁
    def is_tb_running(port=6006):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0

    def clear_tensorboard_registration():
        """清除当前 WebUI 会话登记的 TensorBoard 进程身份。"""
        st.session_state.tensorboard_pid = None
        st.session_state.tensorboard_process_create_time = None

    def launch_tensorboard_service(port=6006):
        """使用当前便携 Python 启动 TensorBoard，避免依赖系统 PATH。"""
        command = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            os.path.join(PROJECT_ROOT, "runs"),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--samples_per_plugin",
            "scalars=500,images=0,audio=0",
            "--max_reload_threads",
            "1",
            "--reload_interval",
            "60",
        ]
        process_options = {
            "cwd": PROJECT_ROOT,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": build_managed_process_env(PROJECT_ROOT),
        }
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(command, **process_options)
        st.session_state.tensorboard_pid = process.pid
        try:
            st.session_state.tensorboard_process_create_time = psutil.Process(
                process.pid
            ).create_time()
        except psutil.Error:
            st.session_state.tensorboard_process_create_time = None
        return process

    def adopt_managed_tensorboard_service(port=6006):
        """重新登记同项目先前启动的 TensorBoard，兼容 WebUI 页面重连。"""
        port_text = str(port)
        for process in psutil.process_iter(["pid", "cmdline"]):
            if process.pid == os.getpid() or not process_matches_project(
                process,
                PROJECT_ROOT,
            ):
                continue
            try:
                command = list(process.cmdline())
                module_index = command.index("-m") if "-m" in command else -1
                port_index = command.index("--port") if "--port" in command else -1
                is_tensorboard = (
                    module_index >= 0
                    and module_index + 1 < len(command)
                    and command[module_index + 1] == "tensorboard.main"
                    and port_index >= 0
                    and port_index + 1 < len(command)
                    and command[port_index + 1] == port_text
                )
                if not is_tensorboard:
                    continue
                st.session_state.tensorboard_pid = process.pid
                st.session_state.tensorboard_process_create_time = process.create_time()
                return True
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
        return False

    if not st.session_state.tensorboard_pid:
        adopt_managed_tensorboard_service(6006)

    registered_tb_alive = is_process_alive(
        st.session_state.tensorboard_pid,
        st.session_state.tensorboard_process_create_time,
    )
    if st.session_state.tensorboard_pid and not registered_tb_alive:
        clear_tensorboard_registration()

    tb_running = is_tb_running(6006)

    # 顶部的控制按钮 (去除了 use_container_width 以防后续版本报错)
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        if not tb_running:
            if st.button(_("🚀 启动 TensorBoard", "🚀 Start TensorBoard"), type="primary"):
                try:
                    process = launch_tensorboard_service(6006)
                    # 最多等待 5 秒，兼容首次扫描日志稍慢的机器。
                    for _ in range(20):
                        if process.poll() is not None or is_tb_running(6006):
                            break
                        time.sleep(0.25)
                    if process.poll() is not None or not is_tb_running(6006):
                        terminate_process(
                            st.session_state.tensorboard_pid,
                            st.session_state.tensorboard_process_create_time,
                        )
                        clear_tensorboard_registration()
                        st.error(_(
                            "TensorBoard 启动失败，请确认一键环境中的 tensorboard 依赖完整。",
                            "TensorBoard failed to start. Check the bundled tensorboard dependency.",
                        ))
                    else:
                        st.rerun()
                except (OSError, subprocess.SubprocessError) as error:
                    clear_tensorboard_registration()
                    st.error(_(
                        f"TensorBoard 启动失败：{error}",
                        f"TensorBoard failed to start: {error}",
                    ))
        else:
            if st.button(_("🛑 关闭 TensorBoard", "🛑 Stop TensorBoard")):
                if registered_tb_alive:
                    terminate_process(
                        st.session_state.tensorboard_pid,
                        st.session_state.tensorboard_process_create_time,
                    )
                    clear_tensorboard_registration()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning(_(
                        "6006 端口上的服务不是由当前 WebUI 会话启动，已拒绝跨进程强制结束。",
                        "The service on port 6006 was not started by this WebUI session and will not be terminated.",
                    ))
    with col2:
        if tb_running:
            if st.button(_("🔄 刷新图表", "🔄 Refresh Charts")):
                st.rerun()

    st.divider()

    # 展示区
    if tb_running:
        st.success(_("🟢 TensorBoard 正在后台运行！", "🟢 TensorBoard is running!"))
        
        # 🌟 修复 3: 提供极强提示的外部链接作为兜底，这是最完美的浏览体验
        st.markdown(_("### 👉 **[点此在新标签页中全屏打开 TensorBoard](http://127.0.0.1:6006)**", "### 👉 **[Click here to open TensorBoard in a new tab](http://127.0.0.1:6006)**"))
        st.caption(_("如果下方内嵌窗口空白或闪烁，是因为部分浏览器严格拦截了本地端口的 iframe 嵌套，请直接点击上方链接使用浏览器原生标签页浏览。", "If the iframe below is blank, use the link above."))
        
        # 尝试嵌入，如果被拦截，用户还可以通过上面的链接走
        components.iframe("http://127.0.0.1:6006", height=850, scrolling=True)
    else:
        st.info(_("⚪ TensorBoard 尚未启动，请点击左上方按钮启动服务器。", "⚪ TensorBoard is not running. Click Start."))
    
# ==========================================
# 📦 模块八：模型部署与打包 (Model Deployment V3.0.0 本地直连版)
# ==========================================
elif menu == _("📦 模型部署与打包", "📦 Model Deployment"):
    import zipfile
    import shutil
    import time
    import platform
    import subprocess
    from model_artifacts import (
        create_deployment_package,
        get_model_iteration_mismatch,
        install_model_artifact_bundle,
        safe_extract_zip,
        validate_deployment_package,
        validate_deployment_package_filename,
        validate_package_name,
        validate_safe_filename,
    )
    
    st.title(_("📦 模型打包与发布工厂", "📦 Model Deployment Factory"))
    st.markdown(_("将训练好的模型与知识库打包为 `.gkg` 部署包。已启用纯本地原生 I/O 加速，彻底免除网页端大文件传输导致的卡顿与内存溢出。", 
                  "Package models and knowledge bases into `.gkg` files. Native local I/O enabled to prevent browser freezing and OOM."))
    
    deploy_dir = os.path.abspath("./deploy_packages")
    unpack_dir = os.path.join(deploy_dir, "unpacked")
    os.makedirs(deploy_dir, exist_ok=True)
    os.makedirs(unpack_dir, exist_ok=True)
    
    # 跨平台打开文件夹辅助函数
    def open_local_folder(path):
        """按当前系统打开已经确定的本地目录"""
        try:
            if platform.system() == "Windows": os.startfile(path)
            elif platform.system() == "Darwin": subprocess.Popen(["open", path])
            else: subprocess.Popen(["xdg-open", path])
        except Exception as e:
            st.error(f"无法打开文件夹: {e}")

    def remove_staged_package_folder(path):
        """确认暂存目录没有越出部署暂存根目录后再删除"""
        stage_root = os.path.realpath(unpack_dir)
        target = os.path.realpath(path)
        if os.path.commonpath([stage_root, target]) != stage_root or target == stage_root:
            raise ValueError("暂存目录边界校验失败")
        shutil.rmtree(target)

    def list_safe_deployment_packages():
        """只列出通过单层安全文件名校验的本地部署包"""
        valid = []
        invalid = []
        for filename in os.listdir(deploy_dir):
            if not filename.casefold().endswith(".gkg"):
                continue
            try:
                validate_deployment_package_filename(filename)
                path = os.path.join(deploy_dir, filename)
                if os.path.isfile(path) and not os.path.islink(path):
                    valid.append(filename)
                else:
                    invalid.append(filename)
            except ValueError:
                invalid.append(filename)
        return sorted(valid, reverse=True), sorted(invalid, reverse=True)

    def list_safe_staged_folders():
        """只列出真实位于部署暂存根目录内的普通目录"""
        root = os.path.realpath(unpack_dir)
        valid = []
        for dirname in os.listdir(unpack_dir):
            try:
                validate_safe_filename(dirname)
            except ValueError:
                continue
            path = os.path.join(unpack_dir, dirname)
            resolved = os.path.realpath(path)
            if (
                os.path.isdir(path)
                and not os.path.islink(path)
                and resolved != root
                and os.path.commonpath([root, resolved]) == root
            ):
                valid.append(dirname)
        return sorted(valid, reverse=True)

    def deployment_stage_signature(stage_path):
        """生成暂存包文件签名，让安全验证仅在文件变化后重跑"""
        records = []
        for filename in os.listdir(stage_path):
            path = os.path.join(stage_path, filename)
            if os.path.isfile(path):
                records.append((filename, os.path.getmtime(path), os.path.getsize(path)))
        return tuple(sorted(records))

    @st.cache_data(show_spinner=False)
    def load_staged_package_validation(stage_path, file_signature):
        """缓存完整部署包验证结果，避免 WebUI 重绘时重复加载 PTH"""
        del file_signature
        return validate_deployment_package(stage_path)

    tab_pack, tab_unpack = st.tabs([
        _("📥 封装部署文件 (Export)", "📥 Export Package"), 
        _("📤 解包与选择性导入 (Unpack & Import)", "📤 Unpack & Import")
    ])
    
    # --- 1. 打包导出 ---
    with tab_pack:
        col_form, col_list = st.columns([6, 4])
        
        with col_form:
            st.markdown("### 🔧 " + _("构建新的 .gkg 部署包", "Build new .gkg Package"))
            package_repository = load_model_repository(
                os.path.abspath("./models"),
                get_model_identity_signature(os.path.abspath("./models")),
            )
            package_pools = package_repository["pools"]
            package_pool_ids = sorted(
                package_pools,
                key=lambda model_id: (
                    ",".join(package_pools[model_id]["prefixes"]).casefold(),
                    model_id,
                ),
            )
            selected_package_pool = None
            if package_pool_ids:
                selected_package_pool_id = st.selectbox(
                    _("① 选择模型池（UUID）", "① Select model pool (UUID)"),
                    package_pool_ids,
                    format_func=lambda model_id: (
                        f"{', '.join(package_pools[model_id]['prefixes'])} | "
                        f"{model_id} | {len(package_pools[model_id]['iterations'])} iterations"
                    ),
                    key="deployment_model_pool",
                )
                selected_package_pool = package_pools[selected_package_pool_id]
                artifact_by_primary = {
                    artifact["primary"]: artifact
                    for artifact in selected_package_pool["artifacts"]
                }
                sel_models = st.multiselect(
                    _("② 选择该模型池中的文件", "② Select files from this model pool"),
                    list(artifact_by_primary),
                    format_func=lambda name: (
                        f"iter {artifact_by_primary[name]['iteration']} | "
                        f"{artifact_by_primary[name]['format']} | {name}"
                    ),
                    key="deployment_model_files",
                )
                preview_records = [artifact_by_primary[name] for name in sel_models]
            else:
                sel_models = []
                preview_records = []
                st.warning(_(
                    "暂无可验证的模型池；仍可只打包知识库。",
                    "No verified model pool is available; a knowledge-only package is still allowed.",
                ))

            selection_warning = get_model_iteration_mismatch(preview_records)
            prefix_warning = (
                selected_package_pool is not None
                and len(selected_package_pool["prefixes"]) != 1
            )
            if selection_warning:
                st.warning(_(
                    f"⚠️ {selection_warning}。请补齐相同轮次的另一格式，或只选择一种格式。",
                    f"⚠️ {selection_warning}. Select matching iterations in both formats, or only one format.",
                ))
            if prefix_warning:
                st.error(_(
                    "同一 UUID 池出现多个模型前缀，身份异常，已禁止打包。",
                    "This UUID pool contains multiple prefixes; packaging is disabled.",
                ))
            if package_repository["invalid"]:
                st.caption(_(
                    f"另有 {len(package_repository['invalid'])} 个异常或孤立制品未进入可选池。",
                    f"{len(package_repository['invalid'])} invalid or orphan artifacts were excluded.",
                ))

            with st.form("pack_form"):
                st.markdown("##### 🗂️ 附加数据组件")
                c1, c2, c3, c4 = st.columns(4)
                with c1: inc_kb = st.checkbox("包含 结构语义", value=True, help="knowledge_base.json + hash_mapping_report.json")
                with c2: inc_code_semantics = st.checkbox("包含 代码语义", value=True, help="code_embeddings.npy + code_embeddings_idx.json")
                with c3: inc_staples = st.checkbox("包含 兜底池", value=True, help="meta_staples.json")
                with c4: st.checkbox("强制安全清单", value=True, disabled=True, help="manifest.json")
                
                st.markdown("##### 🏷️ 包体信息")
                pkg_name = st.text_input(_("自定义包名 (留空默认自动生成)", "Package Name"), placeholder="e.g. Galatea_V3_Full")
                
                if st.form_submit_button(
                    "🔨 " + _("原生极速打包 (.gkg)", "Generate .gkg Fast"),
                    type="primary",
                    use_container_width=True,
                    disabled=bool(selection_warning or prefix_warning),
                ):
                    missing = []
                    if inc_kb and not os.path.exists("knowledge_base.json"): missing.append("knowledge_base.json")
                    if inc_code_semantics:
                        for semantic_filename in ("code_embeddings.npy", "code_embeddings_idx.json"):
                            if not os.path.exists(semantic_filename):
                                missing.append(semantic_filename)
                    if inc_staples and not os.path.exists("meta_staples.json"): missing.append("meta_staples.json")

                    if missing:
                        st.error(_(f"缺少勾选的组件: {', '.join(missing)}。请先生成或取消勾选！", f"Missing files: {', '.join(missing)}."))
                    elif inc_code_semantics and not inc_kb:
                        st.error(_("代码语义必须和对应知识库一起打包。", "Code semantics must be packaged with their knowledge base."))
                    elif not sel_models and not (inc_kb or inc_code_semantics or inc_staples):
                        st.error("包体不能为空，请至少选择一个模型或组件！")
                    else:
                        try:
                            with st.spinner("正在校验身份并进行本地封装..."):
                                final_name = pkg_name.strip() if pkg_name.strip() else f"Galatea_Pkg_{int(time.time())}"
                                validate_package_name(final_name)
                                target_zip = os.path.join(deploy_dir, f"{final_name}.gkg")
                                extra_files = {}
                                if inc_kb:
                                    extra_files["knowledge_base.json"] = "knowledge_base.json"
                                    if os.path.exists("hash_mapping_report.json"):
                                        extra_files["hash_mapping_report.json"] = "hash_mapping_report.json"
                                if inc_code_semantics:
                                    extra_files["code_embeddings.npy"] = "code_embeddings.npy"
                                    extra_files["code_embeddings_idx.json"] = "code_embeddings_idx.json"
                                if inc_staples: extra_files["meta_staples.json"] = "meta_staples.json"
                                create_deployment_package(
                                    target_zip,
                                    "./models",
                                    sel_models,
                                    package_name=final_name,
                                    extra_files=extra_files,
                                )
                            st.success(_(f"✅ 打包成功！文件位于 `{target_zip}`", f"✅ Successfully packaged to `{target_zip}`"))
                            time.sleep(1)
                            st.rerun()
                        except Exception as error:
                            st.error(_(f"部署包生成被拒绝: {error}", f"Package generation rejected: {error}"))

        with col_list:
            st.markdown("### 🗄️ " + _("本地部署包仓库", "Local Packages"))
            gkgs, invalid_gkgs = list_safe_deployment_packages()
            if invalid_gkgs:
                st.warning(_(
                    f"已忽略 {len(invalid_gkgs)} 个文件名不安全的 .gkg 文件。",
                    f"Ignored {len(invalid_gkgs)} .gkg files with unsafe names.",
                ))
            
            if st.button("📂 " + _("打开本地部署包文件夹", "Open Local Folder"), use_container_width=True):
                open_local_folder(deploy_dir)
                
            st.divider()
            if gkgs:
                for gm in gkgs:
                    with st.container(border=True):
                        st.markdown(f"📦 **{gm}** `({os.path.getsize(os.path.join(deploy_dir, gm)) // (1024*1024)} MB)`")
                        if st.button("🗑️ " + _("删除该包", "Delete"), key=f"del_{gm}", use_container_width=True):
                            os.remove(os.path.join(deploy_dir, gm))
                            st.rerun()
            else:
                st.info("暂无生成的部署包。")

    # --- 2. 解包与暂存库 ---
    with tab_unpack:
        col_up, col_mgr = st.columns([1, 1])

        # 【左侧：从本地读取解包】
        with col_up:
            st.markdown("### 📥 " + _("从本地读取并解压", "Read & Unpack Locally"))
            st.info("将外部获取的 `.gkg` 包拖入本地部署文件夹，然后在下方选择解压到暂存区。")
            
            if st.button("📂 " + _("打开本地文件夹放入 .gkg 包", "Drop .gkg here"), type="secondary", use_container_width=True):
                open_local_folder(deploy_dir)
            
            st.write("")
            local_gkgs, invalid_local_gkgs = list_safe_deployment_packages()
            if invalid_local_gkgs:
                st.warning(_(
                    f"已忽略 {len(invalid_local_gkgs)} 个文件名不安全的 .gkg 文件。",
                    f"Ignored {len(invalid_local_gkgs)} .gkg files with unsafe names.",
                ))
            if not local_gkgs:
                st.warning("⚠️ 本地仓库未检测到 `.gkg` 包。请先点击上方按钮拖入文件。")
            else:
                sel_gkg = st.selectbox("选择要解压的本地部署包：", local_gkgs)
                if st.button("⚡ " + _("执行本地极速解包", "Unpack Fast"), type="primary", use_container_width=True):
                    with st.spinner("正在进行本地原生解构，绕过所有上传限制..."):
                        target_extract_dir = None
                        try:
                            validate_deployment_package_filename(sel_gkg)
                            pkg_name_no_ext = os.path.splitext(sel_gkg)[0]
                            target_extract_dir = os.path.join(unpack_dir, f"{pkg_name_no_ext}_{int(time.time())}")
                            os.makedirs(target_extract_dir, exist_ok=False)
                            with zipfile.ZipFile(os.path.join(deploy_dir, sel_gkg), 'r') as gkg_zip:
                                safe_extract_zip(gkg_zip, target_extract_dir)
                            validated_package = validate_deployment_package(target_extract_dir)
                            st.success(f"✅ 解包成功！已存入暂存区：`{os.path.basename(target_extract_dir)}`")
                            st.caption(_(
                                f"模型池 UUID: {', '.join(sorted({record['model_id'] for record in validated_package['records']})) or '无模型'}",
                                f"Model pool UUID: {', '.join(sorted({record['model_id'] for record in validated_package['records']})) or 'No model'}",
                            ))
                            time.sleep(1.0)
                            st.rerun()
                        except Exception as e:
                            if target_extract_dir and os.path.isdir(target_extract_dir):
                                remove_staged_package_folder(target_extract_dir)
                            st.error(f"解压失败，包体可能损坏: {str(e)}")

        # 【右侧：暂存库与导入管理】
        with col_mgr:
            st.markdown("### 🚀 " + _("暂存库管理与导入", "Staging & Selective Import"))
            st.info("浏览解包后的暂存区，精准勾选需要导入的文件。")
            
            staged_folders = list_safe_staged_folders()
            
            if not staged_folders:
                st.warning("暂存区为空。请先在左侧解压一个 `.gkg` 包。")
            else:
                sel_stage = st.selectbox("📂 选择解压暂存目录", staged_folders)
                stage_path = os.path.join(unpack_dir, sel_stage)
                staged_files = os.listdir(stage_path)

                if not staged_files:
                    st.info("该暂存区为空。")
                else:
                    try:
                        stage_validation = load_staged_package_validation(
                            stage_path,
                            deployment_stage_signature(stage_path),
                        )
                    except Exception as error:
                        stage_validation = None
                        st.error(_(
                            f"该暂存包未通过安全与身份校验，禁止导入: {error}",
                            f"This staged package failed safety and identity validation: {error}",
                        ))

                    if stage_validation is not None:
                        manifest = stage_validation["manifest"]
                        models_in_stage = manifest["models_included"]
                        package_model_ids = sorted(
                            {record["model_id"] for record in stage_validation["records"]}
                        )
                        if package_model_ids:
                            st.caption(_(
                                f"已鉴权模型池 UUID: {package_model_ids[0]}",
                                f"Authenticated model pool UUID: {package_model_ids[0]}",
                            ))

                        with st.form("import_form"):
                            st.write("**勾选需要部署进当前系统的文件：**")
                            selected_stage_models = []
                            selected_root_files = []

                            if models_in_stage:
                                st.markdown("🤖 **模型制品组（依赖文件将自动补齐）**")
                                record_by_primary = {
                                    record["primary"]: record
                                    for record in stage_validation["records"]
                                }
                                for model_name in models_in_stage:
                                    record = record_by_primary[model_name]
                                    label = (
                                        f"iter {record['iteration']} | {record['format']} | "
                                        f"{model_name}"
                                    )
                                    if st.checkbox(
                                        f"📄 {label}",
                                        value=True,
                                        key=f"chk_{sel_stage}_{model_name}",
                                    ):
                                        selected_stage_models.append(model_name)

                            root_file_groups = [
                                (
                                    "结构化语义（知识库 + Hash 接续索引）",
                                    [
                                        filename
                                        for filename in ("knowledge_base.json", "hash_mapping_report.json")
                                        if filename in staged_files
                                    ],
                                ),
                                (
                                    "代码语义向量（向量 + 索引）",
                                    [
                                        filename
                                        for filename in ("code_embeddings.npy", "code_embeddings_idx.json")
                                        if filename in staged_files
                                    ],
                                ),
                                (
                                    "142 泛用宣言池",
                                    ["meta_staples.json"] if "meta_staples.json" in staged_files else [],
                                ),
                            ]
                            root_file_groups = [group for group in root_file_groups if group[1]]
                            if root_file_groups:
                                st.markdown("📝 **字典文件（将覆盖系统根目录同名文件）**")
                                for group_label, group_files in root_file_groups:
                                    if st.checkbox(
                                        f"⚙️ {group_label}: {', '.join(group_files)}",
                                        value=True,
                                        key=f"chk_{sel_stage}_{group_files[0]}",
                                    ):
                                        selected_root_files.extend(group_files)

                            st.write("")
                            if st.form_submit_button("📥 " + _("执行精准导入", "Execute Import"), type="primary", use_container_width=True):
                                if not selected_stage_models and not selected_root_files:
                                    st.warning("未勾选任何文件！")
                                elif (
                                    "code_embeddings.npy" in selected_root_files
                                    and "knowledge_base.json" not in selected_root_files
                                ):
                                    st.warning(_(
                                        "导入代码语义时必须同时导入同包知识库，避免向量索引错配。",
                                        "Import the matching knowledge base together with code semantics to avoid index mismatches.",
                                    ))
                                else:
                                    try:
                                        if selected_stage_models:
                                            install_model_artifact_bundle(
                                                stage_path,
                                                "./models",
                                                selected_stage_models,
                                                expected_model_id=(
                                                    package_model_ids[0]
                                                    if package_model_ids
                                                    else None
                                                ),
                                            )
                                        for filename in selected_root_files:
                                            source = os.path.join(stage_path, filename)
                                            destination = os.path.abspath(filename)
                                            with tempfile.NamedTemporaryFile(
                                                prefix=f".{filename}.",
                                                suffix=".import.tmp",
                                                dir=os.path.dirname(destination),
                                                delete=False,
                                            ) as temporary_stream:
                                                temporary = temporary_stream.name
                                            try:
                                                shutil.copy2(source, temporary)
                                                os.replace(temporary, destination)
                                            finally:
                                                if os.path.exists(temporary):
                                                    os.remove(temporary)
                                        load_model_repository.clear()
                                        st.success("✅ 导入成功！系统环境已更新。")
                                    except Exception as e:
                                        st.error(f"导入中途失败: {str(e)}")

                if st.button("🗑️ " + _("清空此暂存目录", "Delete this Staged Folder"), use_container_width=True):
                    remove_staged_package_folder(stage_path)
                    st.rerun()
