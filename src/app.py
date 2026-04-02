import os
import re
import sys
import threading
from datetime import datetime
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augment.cgan_256 import train_cgan_256
from src.augment.traditional import apply_traditional_augmentation
from src.dataset_loader import load_and_preprocess_dataset


st.set_page_config(page_title="工业缺陷数据增强系统", layout="wide", page_icon="🏭")
st.title("🏭 工业表面缺陷数据增强系统")

EPOCH_IMAGE_PATTERN = re.compile(r"^epoch_(\d+)_class_.+\.png$")
EPOCH_DETAIL_PATTERN = re.compile(r"^epoch_(\d+)_class_(.+?)(?:_s(\d+))?\.png$")


def _init_state():
    defaults = {
        "raw_dir": "data/raw/NEU-DET/train/images",
        "output_path": f"results/gan_run_{datetime.now().strftime('%Y%m%d')}",
        "gan_thread": None,
        "pause_event": None,
        "stop_event": None,
        "gan_control": {"running": False, "state": "idle", "epoch": 0, "epochs": 0, "error": ""},
        "best_epoch_result": None,
        "dialog_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _pick_folder_via_dialog(initial_dir):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=initial_dir if os.path.isdir(initial_dir) else os.getcwd(),
            title="选择文件夹",
        )
        root.destroy()
        return selected
    except Exception as exc:
        st.session_state.dialog_error = f"无法打开文件管理器: {exc}"
        return None


def _collect_epoch_images(run_dir):
    epoch_to_paths = {}
    if not os.path.isdir(run_dir):
        return epoch_to_paths
    for file_name in os.listdir(run_dir):
        match = EPOCH_IMAGE_PATTERN.match(file_name)
        if not match:
            continue
        epoch = int(match.group(1))
        epoch_to_paths.setdefault(epoch, []).append(os.path.join(run_dir, file_name))
    return epoch_to_paths


def _collect_epoch_details(run_dir):
    epoch_to_items = {}
    if not os.path.isdir(run_dir):
        return epoch_to_items

    for file_name in os.listdir(run_dir):
        match = EPOCH_DETAIL_PATTERN.match(file_name)
        if not match:
            continue
        epoch = int(match.group(1))
        class_name = match.group(2)
        sample_idx = int(match.group(3)) if match.group(3) is not None else -1
        epoch_to_items.setdefault(epoch, []).append(
            {
                "file_name": file_name,
                "path": os.path.join(run_dir, file_name),
                "class_name": class_name,
                "sample_idx": sample_idx,
            }
        )

    for epoch in epoch_to_items:
        epoch_to_items[epoch].sort(key=lambda x: (x["class_name"], x["sample_idx"], x["file_name"]))
    return epoch_to_items


def _compute_epoch_quality(image_paths):
    images = []
    sharpness_values = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img.astype(np.float32))
        sharpness_values.append(cv2.Laplacian(img, cv2.CV_64F).var())
    if len(images) < 2:
        return None

    pairwise_diffs = []
    for a, b in combinations(images, 2):
        pairwise_diffs.append(float(np.mean(np.abs(a - b)) / 255.0))

    return {
        "sharpness": float(np.mean(sharpness_values)),
        "diversity": float(np.mean(pairwise_diffs)),
        "num_images": len(images),
    }


def find_best_epoch(run_dir):
    epoch_to_paths = _collect_epoch_images(run_dir)
    rows = []
    for epoch, paths in sorted(epoch_to_paths.items()):
        metrics = _compute_epoch_quality(paths)
        if metrics is None:
            continue
        rows.append(
            {
                "epoch": epoch,
                "sharpness": metrics["sharpness"],
                "diversity": metrics["diversity"],
                "num_images": metrics["num_images"],
            }
        )

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    sharpness_range = df["sharpness"].max() - df["sharpness"].min()
    diversity_range = df["diversity"].max() - df["diversity"].min()
    df["sharpness_norm"] = (
        (df["sharpness"] - df["sharpness"].min()) / sharpness_range if sharpness_range > 0 else 0.0
    )
    df["diversity_norm"] = (
        (df["diversity"] - df["diversity"].min()) / diversity_range if diversity_range > 0 else 0.0
    )
    df["score"] = 0.6 * df["sharpness_norm"] + 0.4 * df["diversity_norm"]
    best_row = df.sort_values("score", ascending=False).iloc[0]
    return best_row.to_dict(), df.sort_values("epoch").reset_index(drop=True)


def _run_gan_thread(
    raw_dir,
    processed_dir,
    output_dir,
    epochs,
    batch_size,
    lr_g,
    lr_d,
    image_size,
    save_interval,
    num_preview,
    resume_flag,
    pause_event,
    stop_event,
    control,
):
    try:
        control.update({"running": True, "state": "preprocessing", "error": ""})
        processed = load_and_preprocess_dataset(raw_dir, processed_dir, size=image_size)

        control.update({"state": "running", "epoch": 0, "epochs": epochs})
        result = train_cgan_256(
            data_dir=processed,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr_g=lr_g,
            lr_d=lr_d,
            image_size=image_size,
            save_interval=save_interval,
            num_test_samples=num_preview,
            resume=resume_flag,
            pause_event=pause_event,
            stop_event=stop_event,
            status_callback=lambda payload: control.update(payload),
        )
        control.update({"state": result.get("status", "finished")})
    except Exception as exc:
        control.update({"state": "error", "error": str(exc)})
    finally:
        control["running"] = False


_init_state()

# If thread finished on previous run, sync status
thread_obj = st.session_state.gan_thread
if thread_obj is not None and not thread_obj.is_alive():
    st.session_state.gan_control["running"] = False


st.sidebar.header("⚙️ 参数配置")

# Raw path + folder picker
raw_text = st.sidebar.text_input("原始数据路径", value=st.session_state.raw_dir)
if raw_text != st.session_state.raw_dir:
    st.session_state.raw_dir = raw_text
if st.sidebar.button("📂 选择原始数据文件夹"):
    picked = _pick_folder_via_dialog(st.session_state.raw_dir)
    if picked:
        st.session_state.raw_dir = picked
        st.rerun()

if st.session_state.dialog_error:
    st.sidebar.warning(st.session_state.dialog_error)
    st.session_state.dialog_error = ""

processed_dir = "data/processed/gui_temp"
method = st.sidebar.radio("选择增强模式", ("深度学习增强 (GAN)", "传统增强 (Traditional)"))

if method == "传统增强 (Traditional)":
    num_samples = st.sidebar.slider("生成样本总数", 50, 2000, 200)
    output_dir = "results/gui_traditional"
else:
    st.sidebar.subheader("GAN 训练参数")
    image_size = st.sidebar.selectbox("训练分辨率", [128, 256], index=0)
    train_mode = st.sidebar.radio(
        "训练模式",
        ("断点续训 (Resume)", "重新开始 (Restart) - 生成新文件夹"),
        index=0,
    )
    is_resume = "Resume" in train_mode

    if is_resume:
        output_text = st.sidebar.text_input("输出/续训路径", value=st.session_state.output_path)
        if output_text != st.session_state.output_path:
            st.session_state.output_path = output_text
        if st.sidebar.button("📂 选择输出/续训文件夹"):
            picked = _pick_folder_via_dialog(st.session_state.output_path)
            if picked:
                st.session_state.output_path = picked
                st.rerun()
    else:
        st.sidebar.info("点击开始后将自动创建 `results/gan_run_时间戳` 目录。")

    epochs = st.sidebar.number_input("训练轮数 (Epochs)", 10, 5000, 400, step=10)
    batch_size = st.sidebar.selectbox("Batch Size", [2, 4, 8, 16], index=1)
    lr_g = st.sidebar.number_input("生成器学习率 (lr_g)", value=0.00010, format="%.5f")
    lr_d = st.sidebar.number_input("判别器学习率 (lr_d)", value=0.00010, format="%.5f")
    save_int = st.sidebar.number_input("保存间隔 (Epochs)", 1, 100, 10)
    num_preview = st.sidebar.number_input("每类保存样本数", 1, 16, 8)

is_running = bool(
    st.session_state.gan_thread is not None
    and st.session_state.gan_thread.is_alive()
    and st.session_state.gan_control.get("running", False)
)

start_btn = st.sidebar.button("🚀 开始任务", type="primary", disabled=is_running)

if method == "深度学习增强 (GAN)" and is_running:
    st.sidebar.markdown("### 训练控制")
    control_col1, control_col2, control_col3 = st.sidebar.columns(3)
    if control_col1.button("⏸️ 暂停"):
        st.session_state.pause_event.set()
    if control_col2.button("▶️ 继续"):
        st.session_state.pause_event.clear()
    if control_col3.button("⏹️ 停止"):
        st.session_state.stop_event.set()
        st.session_state.pause_event.clear()

    state_text = st.session_state.gan_control.get("state", "running")
    epoch = st.session_state.gan_control.get("epoch", 0)
    total_epochs = st.session_state.gan_control.get("epochs", 0)
    st.sidebar.caption(f"状态: {state_text} | Epoch: {epoch}/{total_epochs}")

if start_btn:
    if not os.path.exists(st.session_state.raw_dir):
        st.error(f"❌ 路径不存在: {st.session_state.raw_dir}")
    elif method == "传统增强 (Traditional)":
        with st.spinner("正在执行传统增强..."):
            p_dir = load_and_preprocess_dataset(st.session_state.raw_dir, processed_dir, size=256)
            apply_traditional_augmentation(p_dir, output_dir, num_samples=num_samples)
        st.success(f"✅ 传统增强完成！保存至 {output_dir}")
    else:
        if is_resume:
            final_output_dir = st.session_state.output_path
            os.makedirs(final_output_dir, exist_ok=True)
            st.info(f"🔄 继续训练目录: {final_output_dir}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_dir = f"results/gan_run_{timestamp}"
            st.session_state.output_path = final_output_dir
            st.success(f"🆕 新任务目录: {final_output_dir}")

        st.session_state.best_epoch_result = None
        st.session_state.pause_event = threading.Event()
        st.session_state.stop_event = threading.Event()
        st.session_state.gan_control = {"running": True, "state": "starting", "epoch": 0, "epochs": int(epochs), "error": ""}

        thread = threading.Thread(
            target=_run_gan_thread,
            args=(
                st.session_state.raw_dir,
                processed_dir,
                final_output_dir,
                int(epochs),
                int(batch_size),
                float(lr_g),
                float(lr_d),
                int(image_size),
                int(save_int),
                int(num_preview),
                bool(is_resume),
                st.session_state.pause_event,
                st.session_state.stop_event,
                st.session_state.gan_control,
            ),
            daemon=True,
        )
        st.session_state.gan_thread = thread
        thread.start()
        st.info("🔥 训练已在后台启动。可用侧边栏按钮暂停、继续或停止。")


st.divider()
st.header("📊 监控与结果")

active_dir = output_dir if method == "传统增强 (Traditional)" else st.session_state.output_path
st.caption(f"当前监控目录: `{active_dir}`")

if method == "深度学习增强 (GAN)":
    control = st.session_state.gan_control
    if control.get("state") == "error":
        st.error(f"训练错误: {control.get('error', '未知错误')}")
    elif control.get("state") == "stopped":
        st.warning(f"训练已停止，最后完成 epoch: {control.get('epoch', 0)}")
    elif control.get("state") == "paused":
        st.info(f"训练已暂停在 epoch {control.get('epoch', 0)}。")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("📉 训练损失曲线")
        log_file = os.path.join(active_dir, "training_log.csv")
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                if not df.empty:
                    st.line_chart(df.set_index("epoch")[["D_loss", "G_loss"]])
                else:
                    st.info("日志文件为空。")
            except Exception as exc:
                st.warning(f"日志读取失败: {exc}")
        else:
            st.info("等待日志写入...")

    with col2:
        st.subheader("🖼️ 最新生成样本")
        if os.path.exists(active_dir):
            files = sorted([f for f in os.listdir(active_dir) if f.startswith("epoch_") and f.endswith(".png")])
            if files:
                latest_file = files[-1]
                st.image(os.path.join(active_dir, latest_file), caption=latest_file, use_container_width=True)
            else:
                st.info("等待生成样本...")
        else:
            st.info("目录尚未创建")

    st.divider()
    st.subheader("🏅 最佳 Epoch 推荐")
    if st.button("🔍 分析当前目录最佳 Epoch"):
        best_result, score_df = find_best_epoch(active_dir)
        if best_result is None:
            st.warning("当前目录样本不足，无法分析最佳 epoch。")
        else:
            st.session_state.best_epoch_result = {
                "active_dir": active_dir,
                "best_result": best_result,
                "score_df": score_df,
            }

    cached = st.session_state.get("best_epoch_result")
    if cached and cached.get("active_dir") == active_dir:
        best = cached["best_result"]
        score_df = cached["score_df"]
        best_epoch = int(best["epoch"])
        st.success(
            f"推荐 Epoch: {best_epoch} | score={best['score']:.4f} "
            f"(sharpness={best['sharpness']:.2f}, diversity={best['diversity']:.4f})"
        )

        preview_files = sorted(
            [
                f
                for f in os.listdir(active_dir)
                if f.startswith(f"epoch_{best_epoch:04d}_class_") and f.endswith(".png")
            ]
        )
        if preview_files:
            st.image(
                os.path.join(active_dir, preview_files[0]),
                caption=f"最佳 Epoch 示例: {preview_files[0]}",
                use_container_width=True,
            )

        show_cols = ["epoch", "sharpness", "diversity", "score", "num_images"]
        st.dataframe(score_df[show_cols].sort_values("score", ascending=False).head(10), use_container_width=True)

    st.divider()
    st.subheader("🧭 指定 Epoch 浏览")
    epoch_items = _collect_epoch_details(active_dir)
    available_epochs = sorted(epoch_items.keys())
    if available_epochs:
        default_idx = len(available_epochs) - 1
        selected_epoch = st.selectbox("选择要查看的 Epoch", available_epochs, index=default_idx)
        all_classes = sorted({item["class_name"] for item in epoch_items[selected_epoch]})
        selected_classes = st.multiselect("筛选类别（不选则显示全部）", all_classes, default=all_classes)

        selected_items = [
            item
            for item in epoch_items[selected_epoch]
            if not selected_classes or item["class_name"] in selected_classes
        ]
        st.caption(f"Epoch {selected_epoch} 共 {len(selected_items)} 张样本")

        if selected_items:
            cols = st.columns(4)
            for i, item in enumerate(selected_items):
                caption = (
                    f"{item['class_name']} | s{item['sample_idx']:02d}"
                    if item["sample_idx"] >= 0
                    else item["class_name"]
                )
                with cols[i % 4]:
                    st.image(item["path"], caption=caption, use_container_width=True)
        else:
            st.info("该 Epoch 下没有匹配筛选条件的样本。")
    else:
        st.info("当前目录还没有可供浏览的 epoch 样本。")

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
            st.info("暂无结果")
