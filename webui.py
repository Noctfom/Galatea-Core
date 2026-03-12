import streamlit as st
import subprocess
import os
import time
import socket

# --- 页面配置 ---
st.set_page_config(page_title="Galatea AI 控制中心", page_icon="🤖", layout="wide")
st.title("🤖 Galatea AI 核心控制面板")

# --- 后台进程管理功能 ---
@st.cache_resource
def get_process_manager():
    return {"train_proc": None, "tb_proc": None}

pm = get_process_manager()

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_tensorboard():
    if not is_port_in_use(6006):
        # 启动 TensorBoard 后台进程
        pm["tb_proc"] = subprocess.Popen(
            ["tensorboard", "--logdir=runs", "--port=6006"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(2) # 等待启动

# 自动尝试启动 TensorBoard
start_tensorboard()

# --- 主界面布局：左右分栏 ---
# 左侧控制台，右侧 TensorBoard 监控
col_control, col_monitor = st.columns([1, 2])

with col_control:
    st.header("⚙️ 参数配置")
    
    # 模式选择
    mode = st.radio("选择运行模式", ["训练模式 (Train)", "测试模式 (Play)", "竞技场模式 (Duel)"])
    
    # 基础路径配置 (所有模式通用)
    deck_dir = st.text_input("卡组文件夹路径 (--deck_dir)", value="./decks")
    
    st.divider()
    
    # --- 训练模式参数 ---
    if mode == "训练模式 (Train)":
        st.subheader("🚀 训练参数")
        
        # 动态生成命令行参数
        save_dir = st.text_input("模型保存路径 (--dir)", value="./models")
        resume_path = st.text_input("恢复训练权重路径 (--resume) [留空为从头开始]", value="")
        
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input("训练总轮数 (--steps)", value=1000, step=100)
            batch_size = st.number_input("采集批量 (--batch_size)", value=4096, step=1024)
            workers = st.number_input("CPU 进程数 (--workers)", value=4, min_value=1)
        with col2:
            mini_batch = st.number_input("训练批量 (--mini_batch)", value=512, step=128)
            worker_device = st.selectbox("采集设备 (--worker_device)", ["cuda", "cpu"])
            
        st.markdown("**网络架构参数**")
        c1, c2, c3 = st.columns(3)
        d_model = c1.number_input("d_model", value=256, step=64)
        n_heads = c2.number_input("n_heads", value=4, step=1)
        n_layers = c3.number_input("n_layers", value=2, step=1)
        
        async_infer = st.checkbox("启用异步推断 (--async_infer)", value=True)
        no_compile = st.checkbox("禁用模型编译 (--no_compile)", value=False)
        
        if st.button("🔥 启动训练", use_container_width=True, type="primary"):
            cmd = [
                "python", "main.py", "train",
                "--dir", save_dir,
                "--deck_dir", deck_dir,
                "--steps", str(steps),
                "--batch_size", str(batch_size),
                "--mini_batch", str(mini_batch),
                "--workers", str(workers),
                "--worker_device", worker_device,
                "--d_model", str(d_model),
                "--n_heads", str(n_heads),
                "--n_layers", str(n_layers)
            ]
            if resume_path: cmd.extend(["--resume", resume_path])
            if async_infer: cmd.append("--async_infer")
            if no_compile: cmd.append("--no_compile")
            
            st.info(f"执行命令: {' '.join(cmd)}")
            # 以后台进程启动
            pm["train_proc"] = subprocess.Popen(cmd)
            st.success("训练已在后台启动！请在右侧观察 TensorBoard，或查看控制台日志。")

    # --- 测试模式参数 ---
    elif mode == "测试模式 (Play)":
        st.subheader("⚔️ 测试参数")
        num_games = st.number_input("对局数量 (-n)", value=10, step=10)
        
        if st.button("▶️ 开始测试 (控制台查看日志)", use_container_width=True):
            cmd = ["python", "main.py", "play", "-n", str(num_games), "--deck_dir", deck_dir]
            st.info(f"执行命令: {' '.join(cmd)}")
            subprocess.Popen(cmd) # 在后台运行，日志输出在原终端
            st.success("测试已启动，请查看运行该 WebUI 的终端黑框以获取战报！")

    # --- 竞技场模式参数 ---
    elif mode == "竞技场模式 (Duel)":
        st.subheader("🏟️ 竞技场参数")
        p0_path = st.text_input("P0 模型路径 (--p0)")
        p1_path = st.text_input("P1 模型路径 (--p1)")
        duel_games = st.number_input("对战局数 (-n)", value=100, step=10)
        device = st.selectbox("推理设备 (--device)", ["cuda", "cpu"])
        
        st.markdown("**网络架构参数 (需与模型匹配)**")
        c1, c2, c3 = st.columns(3)
        d_model = c1.number_input("d_model ", value=256, step=64)
        n_heads = c2.number_input("n_heads ", value=4, step=1)
        n_layers = c3.number_input("n_layers ", value=2, step=1)
        
        if st.button("⚔️ 开启决斗 (控制台查看日志)", use_container_width=True):
            cmd = [
                "python", "main.py", "duel",
                "--deck_dir", deck_dir,
                "-n", str(duel_games),
                "--device", device,
                "--d_model", str(d_model),
                "--n_heads", str(n_heads),
                "--n_layers", str(n_layers)
            ]
            if p0_path: cmd.extend(["--p0", p0_path])
            if p1_path: cmd.extend(["--p1", p1_path])
            
            st.info(f"执行命令: {' '.join(cmd)}")
            subprocess.Popen(cmd)
            st.success("竞技场已启动，请查看终端输出的胜率战报！")

    st.divider()
    if st.button("🛑 强制终止后台训练进程", type="secondary"):
        if pm["train_proc"] is not None:
            pm["train_proc"].terminate()
            pm["train_proc"] = None
            st.warning("已终止训练进程！")
        else:
            st.info("当前没有通过 WebUI 启动的训练进程。")

with col_monitor:
    st.header("📈 TensorBoard 实时监控")
    if is_port_in_use(6006):
        # 完美内嵌 TensorBoard 网页
        st.components.v1.iframe("http://localhost:6006", height=800, scrolling=True)
    else:
        st.error("TensorBoard 未启动。请确保 runs 文件夹存在，或手动运行 tensorboard --logdir=runs")