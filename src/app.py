# src/app.py
import streamlit as st
import pandas as pd
import os
import sys
import threading
import shutil
from pathlib import Path
from datetime import datetime

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import load_and_preprocess_dataset
from src.augment.traditional import apply_traditional_augmentation
from src.augment.cgan_256 import train_cgan_256

# 页面配置
st.set_page_config(page_title="工业缺陷数据增强系统", layout="wide", page_icon="🏭")

st.title("🏭 工业表面缺陷数据增强系统")

# === Session State 初始化 ===
# 用于存储当前的输出目录路径，以便跨操作保持状态
if 'output_path' not in st.session_state:
    st.session_state.output_path = "results/gui_gan_default"

# --- 侧边栏 ---
st.sidebar.header("⚙️ 参数配置")

# 1. 路径配置
default_raw = "data/raw/NEU-DET/train/images"
raw_dir = st.sidebar.text_input("原始数据路径", default_raw)
processed_dir = "data/processed/gui_temp"

# 2. 方法选择
method = st.sidebar.radio("选择增强模式", ("深度学习增强 (GAN)", "传统增强 (Traditional)"))

# 3. 具体参数
if method == "传统增强 (Traditional)":
    num_samples = st.sidebar.slider("生成样本总数", 50, 2000, 200)
    # 传统增强通常是一次性的，不需要复杂路径管理，也可以加时间戳，这里暂保持简单
    output_dir = "results/gui_traditional"

else:
    st.sidebar.subheader("GAN 训练参数")

    # === 训练模式选择 ===
    train_mode = st.sidebar.radio(
        "训练模式",
        ("断点续训 (Resume)", "重新开始 (Restart) - 生成新文件夹"),
        index=0
    )
    is_resume = True if "Resume" in train_mode else False

    # === 动态路径管理 ===
    if not is_resume:
        st.sidebar.info("💡 提示：点击“开始”后，系统将自动创建一个带时间戳的新文件夹。")
        # 重新开始模式下，输出路径是动态生成的，不需要用户填
        current_display_path = "(将在启动时自动生成...)"
    else:
        # 续训模式下，允许用户修改路径，以便续训以前的某个特定文件夹
        st.sidebar.markdown("⬇️ **请确认续训目录**：")
        st.session_state.output_path = st.sidebar.text_input(
            "输出/续训路径",
            value=st.session_state.output_path
        )
        current_display_path = st.session_state.output_path

    # 其他参数
    epochs = st.sidebar.number_input("训练轮数 (Epochs)", 10, 5000, 400, step=10)
    batch_size = st.sidebar.selectbox("Batch Size", [2, 4, 8, 16], index=2)
    lr = st.sidebar.number_input("学习率", value=0.0002, format="%.5f")
    save_int = st.sidebar.number_input("保存间隔 (Epochs)", 1, 100, 10)

# 4. 启动按钮
start_btn = st.sidebar.button("🚀 开始任务", type="primary")


# --- 后台任务函数 ---
def run_gan_thread(target_output_dir, resume_flag):
    try:
        # 预处理
        p_dir = load_and_preprocess_dataset(raw_dir, processed_dir, size=256)

        # 训练
        train_cgan_256(
            data_dir=p_dir,
            output_dir=target_output_dir,  # 使用传入的动态路径
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_interval=save_int,
            resume=resume_flag
        )
    except Exception as e:
        print(f"Error: {e}")


# --- 逻辑处理 ---
if start_btn:
    if not os.path.exists(raw_dir):
        st.error(f"❌ 路径不存在: {raw_dir}")
    else:
        if method == "传统增强 (Traditional)":
            with st.spinner("正在执行传统增强..."):
                p_dir = load_and_preprocess_dataset(raw_dir, processed_dir, size=256)
                apply_traditional_augmentation(p_dir, output_dir, num_samples=num_samples)
            st.success(f"✅ 传统增强完成！保存至 {output_dir}")
        else:
            # === 关键修改：处理路径生成 ===
            if not is_resume:
                # 生成带时间戳的新路径
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_output_dir = f"results/gan_run_{timestamp}"

                # 更新 Session State，这样页面刷新后也能记住这个新路径
                st.session_state.output_path = final_output_dir

                st.success(f"🆕 已创建新任务目录: {final_output_dir}")
            else:
                # 续训模式，直接使用输入框里的路径
                final_output_dir = st.session_state.output_path
                st.info(f"🔄 正在续训目录: {final_output_dir}")

            st.info("🔥 训练已在后台启动！请观察右侧图表更新。")

            # 启动线程
            t = threading.Thread(
                target=run_gan_thread,
                args=(final_output_dir, is_resume)
            )
            t.start()

st.divider()

# --- 结果展示区 ---
st.header("📊 监控与结果")

# 获取当前实际要展示的目录
active_dir = output_dir if method == "传统增强 (Traditional)" else st.session_state.output_path

st.caption(f"当前监控目录: `{active_dir}`")

if method == "深度学习增强 (GAN)":
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📉 训练损失实时曲线")
        log_file = os.path.join(active_dir, "training_log.csv")

        if st.button("🔄 刷新图表"):
            if os.path.exists(log_file):
                try:
                    df = pd.read_csv(log_file)
                    st.line_chart(df.set_index("epoch")[["D_loss", "G_loss"]])
                except Exception:
                    st.warning("日志文件读取中...")
            else:
                st.warning("等待数据写入...")
        else:
            # 自动尝试加载
            if os.path.exists(log_file):
                try:
                    df = pd.read_csv(log_file)
                    st.line_chart(df.set_index("epoch")[["D_loss", "G_loss"]])
                except:
                    pass

    with col2:
        st.subheader("🖼️ 最新生成样本")
        if st.button("🔄 刷新图片"):
            pass

        if os.path.exists(active_dir):
            files = sorted([f for f in os.listdir(active_dir) if f.startswith("epoch_") and f.endswith(".png")])
            if files:
                latest_file = files[-1]
                st.image(os.path.join(active_dir, latest_file), caption=latest_file, use_container_width=True)
            else:
                st.info("等待生成第一批图像...")
        else:
            st.info("目录尚未创建")

elif method == "传统增强 (Traditional)":
    if os.path.exists(active_dir):
        images = [f for f in os.listdir(active_dir) if f.endswith(".png")]
        if images:
            st.subheader(f"预览 (共 {len(images)} 张)")
            cols = st.columns(4)
            for i, img_name in enumerate(images[:8]):
                with cols[i % 4]:
                    st.image(os.path.join(active_dir, img_name), caption=img_name, use_container_width=True)
        else:
            st.write("暂无结果")