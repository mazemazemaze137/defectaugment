import os
import json
import re
import shutil
import sys
import threading
from datetime import datetime
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augment.cgan_256 import train_cgan_256
from src.augment.cgan_256 import export_generated_samples
from src.augment.refine_generated_samples import refine_generated_samples
from src.augment.traditional import apply_traditional_augmentation
from src.dataset_loader import (
    load_and_preprocess_dataset,
    load_and_preprocess_dataset_from_annotations,
)
from src.evaluate.classifier_validation import parse_class_weights, run_classification_validation
from src.evaluate.build_defense_summary import build_summary as build_defense_summary
from src.evaluate.build_reproducibility_manifest import build_manifest as build_reproducibility_manifest
from src.evaluate.defense_demo_profile import DEMO_PROFILE, build_demo_profile
from src.evaluate.detection_validation import run_detection_validation
from src.evaluate.industrial_readiness import analyze_industrial_readiness, parse_severity_overrides
from src.evaluate.make_evidence_figures import create_evidence_figures
from src.evaluate.metrics import evaluate_generated_dataset
from src.evaluate.smoke_check import run_smoke_check


GAN_METHOD = "深度学习增强 (GAN)"
TRADITIONAL_METHOD = "传统增强 (Traditional)"
EPOCH_IMAGE_PATTERN = re.compile(r"^epoch_(\d+)_class_.+\.png$")
EPOCH_DETAIL_PATTERN = re.compile(r"^epoch_(\d+)_class_(.+?)(?:_s(\d+))?\.png$")

st.set_page_config(page_title="工业表面缺陷数据增强系统", layout="wide", page_icon="DA")
st.title("工业表面缺陷数据增强系统")


def _init_state():
    defaults = {
        "raw_dir": "data/raw/NEU-DET/train/images",
        "annotation_dir": "data/raw/NEU-DET/train/annotations",
        "output_path": f"results/gan_run_{datetime.now().strftime('%Y%m%d')}",
        "gan_thread": None,
        "pause_event": None,
        "stop_event": None,
        "gan_control": {
            "running": False,
            "state": "idle",
            "epoch": 0,
            "epochs": 0,
            "error": "",
        },
        "best_epoch_result": None,
        "dialog_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _device_label():
    if torch.cuda.is_available():
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    return "CPU"


def _read_text_file(path, max_chars=12000):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...（内容较长，已截断预览）"
    return text


def _path_status(path):
    path = Path(path)
    if path.exists():
        if path.is_dir():
            count = sum(1 for _ in path.rglob("*") if _.is_file())
            return "存在", f"{count} 个文件"
        return "存在", f"{path.stat().st_size / 1024:.1f} KB"
    return "缺失", "-"


def _health_rows():
    rows = []
    checks = [
        ("训练集", "data/raw/NEU-DET/train/images"),
        ("验证集", "data/raw/NEU-DET/validation/images"),
        ("40轮cGAN-v2导出样本", "results/cgan_v2_roi_40ep/export_100_per_class"),
        ("600张GAN子集", "results/ratio_ablation/cgan_v2_40ep_seed42/subsets/gan_100_per_class"),
        ("比例消融结果", "results/ratio_ablation/cgan_v2_40ep_seed42/ratio_ablation_summary.csv"),
        ("600张多种子结果", "results/multiseed/cgan_v2_40ep_600/multiseed_summary.csv"),
        ("答辩汇总", "docs/defense_summary.md"),
        ("复现环境清单", "docs/reproducibility_manifest.md"),
        ("答辩演示清单", "docs/defense_demo_checklist.md"),
        ("现场演示配置", "docs/defense_live_demo_profile.md"),
        ("Word论文初稿", "毕业论文_初稿_格式化.docx"),
        ("目标检测验证脚本", "src/evaluate/detection_validation.py"),
        ("分类器模型对比脚本", "src/evaluate/run_classifier_model_comparison.py"),
    ]
    for name, path in checks:
        status, detail = _path_status(path)
        rows.append({"项目": name, "路径": path, "状态": status, "详情": detail})
    return rows


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
        st.session_state.dialog_error = f"无法打开文件管理器：{exc}"
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
    rows = []
    for epoch, paths in sorted(_collect_epoch_images(run_dir).items()):
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
    annotation_dir,
    processed_dir,
    output_dir,
    epochs,
    batch_size,
    lr_g,
    lr_d,
    image_size,
    model_variant,
    diff_augment_enabled,
    ema_decay,
    save_interval,
    num_preview,
    resume_flag,
    use_roi,
    roi_margin,
    enhance_contrast,
    denoise,
    min_box_size,
    strict_label_match,
    pause_event,
    stop_event,
    control,
):
    try:
        control.update({"running": True, "state": "preprocessing", "error": ""})
        if os.path.isdir(processed_dir):
            shutil.rmtree(processed_dir, ignore_errors=True)

        if use_roi:
            processed = load_and_preprocess_dataset_from_annotations(
                images_root=raw_dir,
                annotations_dir=annotation_dir,
                processed_dir=processed_dir,
                size=image_size,
                grayscale=True,
                roi_margin=roi_margin,
                enhance_contrast=enhance_contrast,
                denoise=denoise,
                min_box_size=min_box_size,
                strict_label_match=strict_label_match,
            )
        else:
            processed = load_and_preprocess_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                size=image_size,
                grayscale=True,
                enhance_contrast=enhance_contrast,
                denoise=denoise,
            )

        control.update({"state": "running", "epoch": 0, "epochs": epochs})
        result = train_cgan_256(
            data_dir=processed,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr_g=lr_g,
            lr_d=lr_d,
            image_size=image_size,
            model_variant=model_variant,
            diff_augment_enabled=diff_augment_enabled,
            ema_decay=ema_decay,
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

thread_obj = st.session_state.gan_thread
if thread_obj is not None and not thread_obj.is_alive():
    st.session_state.gan_control["running"] = False

st.sidebar.header("参数配置")
st.sidebar.caption(f"训练设备：{_device_label()}")

raw_text = st.sidebar.text_input("原始数据路径", value=st.session_state.raw_dir)
if raw_text != st.session_state.raw_dir:
    st.session_state.raw_dir = raw_text
if st.sidebar.button("选择原始数据文件夹"):
    picked = _pick_folder_via_dialog(st.session_state.raw_dir)
    if picked:
        st.session_state.raw_dir = picked
        st.rerun()

anno_text = st.sidebar.text_input("标注文件路径（XML 目录）", value=st.session_state.annotation_dir)
if anno_text != st.session_state.annotation_dir:
    st.session_state.annotation_dir = anno_text
if st.sidebar.button("选择标注文件夹"):
    picked = _pick_folder_via_dialog(st.session_state.annotation_dir)
    if picked:
        st.session_state.annotation_dir = picked
        st.rerun()

if st.session_state.dialog_error:
    st.sidebar.warning(st.session_state.dialog_error)
    st.session_state.dialog_error = ""

module = st.sidebar.radio(
    "功能模块",
    ("数据增强训练", "数据质量评估", "下游分类验证", "目标检测验证", "工业应用评估", "答辩材料与健康检查"),
)

if module == "答辩材料与健康检查":
    st.header("答辩材料与健康检查")
    st.caption("集中查看关键实验结论、复现环境、答辩演示清单和系统路径状态，适合答辩前快速自检。")

    health_df = pd.DataFrame(_health_rows())
    ok_count = int((health_df["状态"] == "存在").sum())
    total_count = len(health_df)
    cuda_text = _device_label()
    git_commit = ""
    manifest_path = Path("docs/reproducibility_manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            git_commit = manifest.get("git_commit", "")
        except Exception:
            git_commit = ""

    metric_cols = st.columns(4)
    metric_cols[0].metric("健康检查", f"{ok_count}/{total_count}")
    metric_cols[1].metric("训练设备", cuda_text)
    metric_cols[2].metric("论文字符数", len(_read_text_file("毕业论文_初稿.md", max_chars=200000)))
    metric_cols[3].metric("Git提交", git_commit or "未生成")

    st.subheader("关键路径状态")
    st.dataframe(health_df, use_container_width=True, hide_index=True)

    action_cols = st.columns(4)
    if action_cols[0].button("重新生成证据图"):
        try:
            result = create_evidence_figures("assets/figures")
        except Exception as exc:
            st.error(f"证据图生成失败：{exc}")
        else:
            st.success(f"已生成 {len(result['figures'])} 张证据图。")
    if action_cols[1].button("重新生成答辩汇总"):
        try:
            output = build_defense_summary("docs/defense_summary.md")
        except Exception as exc:
            st.error(f"答辩汇总生成失败：{exc}")
        else:
            st.success(f"已更新：{output}")
    if action_cols[2].button("重新生成复现清单"):
        try:
            output = build_reproducibility_manifest("docs/reproducibility_manifest.md")
        except Exception as exc:
            st.error(f"复现清单生成失败：{exc}")
        else:
            st.success(f"已更新：{output}")
    if action_cols[3].button("生成现场演示配置"):
        try:
            output = build_demo_profile()
        except Exception as exc:
            st.error(f"演示配置生成失败：{exc}")
        else:
            st.success(f"已更新：{output['markdown']}")

    if st.button("运行答辩烟测"):
        result = run_smoke_check()
        if result["passed"]:
            st.success("烟测通过：关键文档、图表、实验结果和 CUDA 状态均满足当前检查条件。")
        else:
            st.error("烟测未通过：请查看下方失败项。")
        st.dataframe(pd.DataFrame(result["checks"]), use_container_width=True, hide_index=True)

    st.subheader("答辩文档")
    doc_options = {
        "答辩实验汇总": "docs/defense_summary.md",
        "答辩演示流程清单": "docs/defense_demo_checklist.md",
        "答辩现场演示配置": "docs/defense_live_demo_profile.md",
        "复现环境清单": "docs/reproducibility_manifest.md",
        "分类器模型对比": "docs/classifier_model_comparison_cgan_v2_600_seed42.md",
        "论文初稿": "毕业论文_初稿.md",
    }
    selected_doc = st.selectbox("选择文档", list(doc_options.keys()))
    selected_path = doc_options[selected_doc]
    text = _read_text_file(selected_path)
    if text:
        st.markdown(text)
    else:
        st.warning(f"文档不存在：{selected_path}")

    st.subheader("核心证据图")
    figure_paths = [
        ("困难类别样本对比", "assets/figures/pitted_surface_refinement_grid.png"),
        ("生成比例消融", "assets/figures/ratio_ablation_cgan_v2_40ep.png"),
        ("600张多种子复验", "assets/figures/multiseed_cgan_v2_600.png"),
        ("工业门槛对比", "assets/figures/industrial_gate_comparison.png"),
        ("系统闭环流程", "assets/figures/system_workflow_industrial.png"),
    ]
    fig_tabs = st.tabs([name for name, _ in figure_paths])
    for tab, (name, path) in zip(fig_tabs, figure_paths):
        with tab:
            if os.path.exists(path):
                st.image(path, caption=name, use_container_width=True)
            else:
                st.warning(f"图片不存在：{path}")
    st.stop()

if module == "数据质量评估":
    st.header("数据质量评估")
    st.caption("对比真实样本与生成样本，计算 FID、SSIM 和 PSNR。FID 越低越好，SSIM/PSNR 越高越好。")
    eval_demo_mode = st.checkbox(
        "答辩演示模式（快速质量评估）",
        value=False,
        help="现场演示默认少量配对并跳过 FID；正式论文结果使用已保存的完整质量评估。",
    )
    if eval_demo_mode:
        st.info("演示模式会降低配对数量并默认不计算 FID，适合现场快速展示评估入口。")

    eval_real_dir = st.text_input("真实数据集目录", value="data/processed/gui_temp")
    eval_generated_dir = st.text_input("生成数据集目录", value=st.session_state.output_path)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        eval_image_size = st.number_input("SSIM/PSNR 图像尺寸", min_value=64, max_value=512, value=256, step=32)
    with col_b:
        eval_max_pairs = st.number_input(
            "SSIM/PSNR 最大配对数",
            min_value=10,
            max_value=2000,
            value=int(DEMO_PROFILE["live_demo"]["quality_eval_max_pairs"]) if eval_demo_mode else 200,
            step=10,
        )
    with col_c:
        eval_fid_batch = st.number_input("FID Batch Size", min_value=1, max_value=128, value=32, step=1)

    eval_fid = st.checkbox("计算 FID（较慢，但论文价值最高）", value=not eval_demo_mode)

    if st.button("开始评估", type="primary"):
        with st.spinner("正在计算质量指标..."):
            try:
                metrics = evaluate_generated_dataset(
                    real_dir=eval_real_dir,
                    generated_dir=eval_generated_dir,
                    image_size=int(eval_image_size),
                    max_pairs=int(eval_max_pairs),
                    fid_batch_size=int(eval_fid_batch),
                    calculate_fid_metric=bool(eval_fid),
                )
            except Exception as exc:
                st.error(f"评估失败：{exc}")
            else:
                metric_cols = st.columns(3 if eval_fid else 2)
                metric_cols[0].metric("SSIM", f"{metrics['ssim']:.4f}")
                metric_cols[1].metric("PSNR", f"{metrics['psnr']:.2f} dB")
                if eval_fid:
                    metric_cols[2].metric("FID", f"{metrics['fid']:.2f}")

                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "real_count": metrics["real_count"],
                                "generated_count": metrics["generated_count"],
                                "paired_images": metrics["pairs"],
                                **({"fid": metrics["fid"]} if "fid" in metrics else {}),
                                "ssim": metrics["ssim"],
                                "psnr": metrics["psnr"],
                            }
                        ]
                    ),
                    use_container_width=True,
                )
    st.stop()

if module == "下游分类验证":
    st.header("下游分类验证")
    st.caption("训练一个轻量 CNN 分类器，用验证集准确率检验增强数据是否提升下游任务效果。")
    cls_demo_mode = st.checkbox(
        "答辩演示模式（短轮次，只证明验证链路）",
        value=False,
        help="现场演示建议跑短轮次；正式结论使用已保存的 20 轮对比实验和多随机种子结果。",
    )
    if cls_demo_mode:
        st.info("演示模式会把默认训练轮数降到 3 轮，并保留低置信度样本导出，用于现场快速证明分类验证流程可运行。")

    cls_train_dir = st.text_input("训练集目录", value="data/raw/NEU-DET/train/images")
    cls_val_dir = st.text_input("验证集目录", value="data/raw/NEU-DET/validation/images")
    cls_generated_dir = st.text_input("生成样本目录（可留空做原始基线）", value="")
    cls_output_dir = st.text_input(
        "结果输出目录",
        value=f"results/classifier_validation_{datetime.now().strftime('%Y%m%d')}",
    )

    cls_col1, cls_col2, cls_col3, cls_col4, cls_col5 = st.columns(5)
    with cls_col1:
        cls_epochs = st.number_input(
            "训练轮数",
            min_value=1,
            max_value=100,
            value=int(DEMO_PROFILE["live_demo"]["classifier_epochs"]) if cls_demo_mode else 5,
            step=1,
        )
    with cls_col2:
        cls_batch_size = st.number_input("Batch Size", min_value=4, max_value=128, value=32, step=4)
    with cls_col3:
        cls_image_size = st.number_input("图像尺寸", min_value=64, max_value=256, value=128, step=32)
    with cls_col4:
        cls_lr = st.number_input("学习率", min_value=0.00001, max_value=0.01, value=0.001, format="%.5f")
    with cls_col5:
        cls_patience = st.number_input("早停耐心", min_value=0, max_value=20, value=4, step=1)
    cls_model_name = st.selectbox(
        "分类器模型",
        ["small_cnn", "resnet18", "mobilenet_v3_small"],
        index=0,
    )
    cls_low_conf_threshold = st.slider("低置信度导出阈值", 0.0, 1.0, 0.70, 0.05)
    cls_class_weights = st.text_input(
        "类别损失权重（可选，用于困难类别）",
        value="",
        placeholder="例如：pitted-surface=1.5,crazing=1.2",
    )
    auto_industrial_eval = st.checkbox("分类验证后自动生成工业应用评估", value=True)

    if st.button("开始分类验证", type="primary"):
        generated_dir_value = cls_generated_dir.strip() or None
        with st.spinner("正在训练分类器并验证效果..."):
            try:
                summary = run_classification_validation(
                    train_dir=cls_train_dir,
                    val_dir=cls_val_dir,
                    generated_dir=generated_dir_value,
                    output_dir=cls_output_dir,
                    image_size=int(cls_image_size),
                    epochs=int(cls_epochs),
                    batch_size=int(cls_batch_size),
                    lr=float(cls_lr),
                    num_workers=0,
                    early_stopping_patience=int(cls_patience),
                    class_weights=parse_class_weights(cls_class_weights),
                    model_name=cls_model_name,
                    low_confidence_threshold=float(cls_low_conf_threshold),
                )
            except Exception as exc:
                st.error(f"分类验证失败：{exc}")
            else:
                metric_cols = st.columns(4)
                metric_cols[0].metric("最佳验证准确率", f"{summary['best_val_accuracy']:.2%}")
                metric_cols[1].metric("最终验证准确率", f"{summary['final_val_accuracy']:.2%}")
                metric_cols[2].metric("训练耗时", f"{summary['elapsed_seconds']:.1f}s")
                metric_cols[3].metric("训练设备", summary["device"])
                st.caption(
                    f"实际完成轮数：{summary.get('completed_epochs', summary['epochs'])} / "
                    f"{summary['epochs']}；最佳轮数：{summary.get('best_epoch', '-')}"
                )

                st.dataframe(pd.DataFrame([summary["counts"]]), use_container_width=True)
                history_path = os.path.join(summary["output_dir"], "history.png")
                if os.path.exists(history_path):
                    st.image(history_path, caption="训练曲线", use_container_width=True)
                confusion_path = os.path.join(summary["output_dir"], "confusion_matrix.png")
                if os.path.exists(confusion_path):
                    st.image(confusion_path, caption="混淆矩阵", use_container_width=True)
                low_conf = summary.get("low_confidence", {})
                if low_conf.get("num_records", 0):
                    st.warning(
                        f"已导出 {low_conf['num_records']} 个低置信度/误分类样本："
                        f"{low_conf.get('csv_path', '')}"
                    )
                if auto_industrial_eval:
                    try:
                        industrial = analyze_industrial_readiness(
                            result_dir=summary["output_dir"],
                            output_dir=os.path.join(summary["output_dir"], "industrial_readiness"),
                        )
                    except Exception as exc:
                        st.warning(f"工业应用评估生成失败：{exc}")
                    else:
                        status = "通过" if industrial["passed"] else "未通过"
                        st.info(
                            f"工业应用评估：{status}；最低类别召回率 "
                            f"{industrial['min_class_recall']:.2%}，代价加权错误率 "
                            f"{industrial['weighted_error_rate']:.2%}。"
                        )
                st.success(f"分类验证结果已保存至：{summary['output_dir']}")
    st.stop()

if module == "目标检测验证":
    st.header("目标检测验证")
    st.caption("基于 NEU-DET XML 标注训练 Faster R-CNN MobileNet-FPN，输出 AP50、Precision 和 Recall。当前 GAN ROI 样本没有边界框标注，因此不直接加入检测训练。")
    det_demo_mode = st.checkbox(
        "答辩演示模式（小样本 smoke）",
        value=True,
        help="现场只验证 XML 读取、检测训练和指标输出链路；正式检测 mAP 可作为后续扩展。",
    )
    if det_demo_mode:
        st.info("演示模式会限制训练/验证样本数并只跑 1 轮，避免现场检测训练耗时过长。")

    det_train_images = st.text_input("检测训练图像目录", value="data/raw/NEU-DET/train/images")
    det_train_ann = st.text_input("检测训练标注目录", value="data/raw/NEU-DET/train/annotations")
    det_val_images = st.text_input("检测验证图像目录", value="data/raw/NEU-DET/validation/images")
    det_val_ann = st.text_input("检测验证标注目录", value="data/raw/NEU-DET/validation/annotations")
    det_output_dir = st.text_input("检测结果输出目录", value="results/detection_validation/gui")

    det_col1, det_col2, det_col3, det_col4 = st.columns(4)
    with det_col1:
        det_epochs = st.number_input(
            "训练轮数",
            min_value=1,
            max_value=50,
            value=int(DEMO_PROFILE["live_demo"]["detector_epochs"]) if det_demo_mode else 3,
            step=1,
        )
    with det_col2:
        det_batch = st.selectbox("Batch Size", [1, 2, 4], index=1)
    with det_col3:
        det_image_size = st.selectbox("输入尺寸", [256, 320, 416], index=1)
    with det_col4:
        det_lr = st.number_input("学习率", min_value=0.00001, max_value=0.01, value=0.0005, format="%.5f")

    det_limit_col1, det_limit_col2, det_limit_col3 = st.columns(3)
    with det_limit_col1:
        det_max_train = st.number_input(
            "最大训练样本（0为不限）",
            min_value=0,
            max_value=2000,
            value=int(DEMO_PROFILE["live_demo"]["detector_max_train"]) if det_demo_mode else 80,
            step=20,
        )
    with det_limit_col2:
        det_max_val = st.number_input(
            "最大验证样本（0为不限）",
            min_value=0,
            max_value=1000,
            value=int(DEMO_PROFILE["live_demo"]["detector_max_val"]) if det_demo_mode else 40,
            step=10,
        )
    with det_limit_col3:
        det_score = st.slider("预测分数阈值", 0.05, 0.95, 0.30, 0.05)

    if st.button("开始目标检测验证", type="primary"):
        with st.spinner("正在训练并评估目标检测模型..."):
            try:
                det_summary = run_detection_validation(
                    train_images=det_train_images,
                    train_annotations=det_train_ann,
                    val_images=det_val_images,
                    val_annotations=det_val_ann,
                    output_dir=det_output_dir,
                    epochs=int(det_epochs),
                    batch_size=int(det_batch),
                    image_size=int(det_image_size),
                    lr=float(det_lr),
                    max_train=int(det_max_train) or None,
                    max_val=int(det_max_val) or None,
                    score_threshold=float(det_score),
                    quiet=True,
                )
            except Exception as exc:
                st.error(f"目标检测验证失败：{exc}")
            else:
                st.metric("mAP@0.5", f"{det_summary['map50']:.2%}")
                st.write(
                    f"训练样本：{det_summary['train_samples']}，验证样本：{det_summary['val_samples']}，"
                    f"耗时：{det_summary['elapsed_seconds']:.2f} 秒。"
                )
                st.dataframe(pd.DataFrame(det_summary["class_metrics"]), use_container_width=True)
                st.info(det_summary["note"])
                st.success(f"检测验证结果已保存至：{det_summary['output_dir']}")
    st.stop()

if module == "工业应用评估":
    st.header("工业应用评估")
    st.caption("从混淆矩阵计算类别召回率、精确率、代价加权错误率和上线门槛，辅助判断实验结果是否具备试运行价值。")

    ind_result_dir = st.text_input(
        "分类验证结果目录",
        value="results/ablation_earlystop/cgan_v2_40ep_filtered_300",
    )
    ind_output_dir = st.text_input("评估报告输出目录（留空则写入结果目录下）", value="")

    ind_col1, ind_col2, ind_col3 = st.columns(3)
    with ind_col1:
        min_best_accuracy = st.number_input("最低最佳准确率", min_value=0.0, max_value=1.0, value=0.98, step=0.01)
    with ind_col2:
        min_class_recall = st.number_input("最低类别召回率", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    with ind_col3:
        max_weighted_error = st.number_input("最高加权错误率", min_value=0.0, max_value=1.0, value=0.06, step=0.01)

    severity_text = st.text_input(
        "类别严重度权重（可选）",
        value="crazing=5,inclusion=4,patches=3,pitted-surface=4,rolled-in-scale=3,scratches=4",
    )

    if st.button("生成工业应用评估", type="primary"):
        with st.spinner("正在生成工业应用评估报告..."):
            try:
                result = analyze_industrial_readiness(
                    result_dir=ind_result_dir,
                    output_dir=ind_output_dir.strip() or None,
                    min_best_accuracy=float(min_best_accuracy),
                    min_class_recall=float(min_class_recall),
                    max_weighted_error=float(max_weighted_error),
                    severity_overrides=parse_severity_overrides(severity_text),
                )
            except Exception as exc:
                st.error(f"工业应用评估失败：{exc}")
            else:
                status = "通过" if result["passed"] else "未通过"
                metric_cols = st.columns(4)
                metric_cols[0].metric("评估状态", status)
                metric_cols[1].metric("最低类别召回率", f"{result['min_class_recall']:.2%}")
                metric_cols[2].metric("加权错误率", f"{result['weighted_error_rate']:.2%}")
                metric_cols[3].metric("总错误数", result["total_errors"])

                st.dataframe(pd.DataFrame(result["class_metrics"]), use_container_width=True)
                recall_path = os.path.join(result["output_dir"], "industrial_recall_by_class.png")
                if os.path.exists(recall_path):
                    st.image(recall_path, caption="类别召回率与上线门槛", use_container_width=True)
                for item in result["recommendations"]:
                    st.write(f"- {item}")
                st.success(f"工业应用评估已保存至：{result['output_dir']}")
    st.stop()

processed_dir = "data/processed/gui_temp"
method = st.sidebar.radio("选择增强模式", (GAN_METHOD, TRADITIONAL_METHOD))

if method == TRADITIONAL_METHOD:
    num_samples = st.sidebar.slider("生成样本总数", 50, 2000, 200)
    output_dir = "results/gui_traditional"
else:
    st.sidebar.subheader("GAN 训练参数")
    gan_demo_mode = st.sidebar.checkbox(
        "答辩演示模式（短训练）",
        value=False,
        help="现场只跑短训练证明流程；正式效果展示 40 轮 cGAN-v2 和保存的实验结果。",
    )
    if gan_demo_mode:
        st.sidebar.info("演示模式建议只跑 3 轮，保存间隔为 1。现场不要用短训练结果代表最终生成质量。")
    image_size = st.sidebar.selectbox("训练分辨率", [128, 256], index=0)
    gan_variant_label = st.sidebar.selectbox(
        "GAN 模型版本",
        ["cGAN-v2 Projection + Hinge", "cGAN-v1 LSGAN"],
        index=0,
        help="cGAN-v2 更适合正式实验；cGAN-v1 保留为对照模型，用于说明模型结构升级带来的稳定性差异。",
    )
    model_variant = "projection_hinge" if "v2" in gan_variant_label else "legacy"
    if model_variant == "projection_hinge":
        st.sidebar.caption("推荐：cGAN-v2 使用 Projection Discriminator、Hinge Loss、DiffAugment 和 EMA，生成纹理更稳定。")
    else:
        st.sidebar.caption("对照：cGAN-v1 使用 LSGAN，可用于答辩中说明早期模型效果和改进过程。")
    diff_augment_enabled = st.sidebar.checkbox("启用 DiffAugment", value=model_variant == "projection_hinge")
    ema_decay = st.sidebar.number_input("EMA 衰减", min_value=0.0, max_value=0.9999, value=0.999, format="%.4f")
    use_roi = st.sidebar.checkbox("使用标注框 ROI 裁剪", value=True)
    roi_margin = st.sidebar.slider("ROI 边界扩展比例", 0.0, 0.30, 0.08, 0.01)
    enhance_contrast = st.sidebar.checkbox("预处理对比度增强（CLAHE）", value=True)
    denoise = st.sidebar.checkbox("预处理去噪（中值滤波）", value=True)
    min_box_size = st.sidebar.number_input("最小缺陷框像素", min_value=2, max_value=32, value=6, step=1)
    strict_label_match = st.sidebar.checkbox("仅保留同类标注框", value=True)
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
        if st.sidebar.button("选择输出/续训文件夹"):
            picked = _pick_folder_via_dialog(st.session_state.output_path)
            if picked:
                st.session_state.output_path = picked
                st.rerun()
    else:
        st.sidebar.info("点击开始后会自动创建 `results/gan_run_时间戳` 目录。")

    epochs = st.sidebar.number_input(
        "训练轮数 (Epochs)",
        1,
        5000,
        int(DEMO_PROFILE["live_demo"]["gan_epochs"]) if gan_demo_mode else 400,
        step=1 if gan_demo_mode else 10,
    )
    batch_size = st.sidebar.selectbox("Batch Size", [2, 4, 8, 16], index=1)
    lr_g = st.sidebar.number_input("生成器学习率 (lr_g)", value=0.00010, format="%.5f")
    lr_d = st.sidebar.number_input("判别器学习率 (lr_d)", value=0.00010, format="%.5f")
    save_int = st.sidebar.number_input(
        "保存间隔 (Epochs)",
        1,
        100,
        int(DEMO_PROFILE["live_demo"]["gan_save_interval"]) if gan_demo_mode else 10,
    )
    num_preview = st.sidebar.number_input("每类保存样本数", 1, 16, 4 if gan_demo_mode else 8)

is_running = bool(
    st.session_state.gan_thread is not None
    and st.session_state.gan_thread.is_alive()
    and st.session_state.gan_control.get("running", False)
)
show_controls = is_running or st.session_state.gan_control.get("state") in {
    "paused",
    "running",
    "preprocessing",
    "starting",
}

start_btn = st.sidebar.button("开始任务", type="primary", disabled=is_running)

if method == GAN_METHOD and show_controls:
    st.sidebar.markdown("### 训练控制")
    control_col1, control_col2, control_col3 = st.sidebar.columns(3)
    if control_col1.button("暂停"):
        st.session_state.pause_event.set()
    if control_col2.button("继续"):
        st.session_state.pause_event.clear()
    if control_col3.button("停止"):
        st.session_state.stop_event.set()
        st.session_state.pause_event.clear()

    state_text = st.session_state.gan_control.get("state", "running")
    epoch = st.session_state.gan_control.get("epoch", 0)
    total_epochs = st.session_state.gan_control.get("epochs", 0)
    st.sidebar.caption(f"状态：{state_text} | Epoch: {epoch}/{total_epochs}")

if start_btn:
    if not os.path.exists(st.session_state.raw_dir):
        st.error(f"路径不存在：{st.session_state.raw_dir}")
    elif method == TRADITIONAL_METHOD:
        with st.spinner("正在执行传统增强..."):
            p_dir = load_and_preprocess_dataset(st.session_state.raw_dir, processed_dir, size=256)
            apply_traditional_augmentation(p_dir, output_dir, num_samples=num_samples)
        st.success(f"传统增强完成，保存至 {output_dir}")
    else:
        if use_roi and not os.path.exists(st.session_state.annotation_dir):
            st.error(f"标注路径不存在：{st.session_state.annotation_dir}")
            st.stop()

        if is_resume:
            final_output_dir = st.session_state.output_path
            os.makedirs(final_output_dir, exist_ok=True)
            st.info(f"续训目录：{final_output_dir}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_dir = f"results/gan_run_{timestamp}"
            st.session_state.output_path = final_output_dir
            st.success(f"新任务目录：{final_output_dir}")

        st.session_state.best_epoch_result = None
        st.session_state.pause_event = threading.Event()
        st.session_state.stop_event = threading.Event()
        st.session_state.gan_control = {
            "running": True,
            "state": "starting",
            "epoch": 0,
            "epochs": int(epochs),
            "error": "",
        }

        thread = threading.Thread(
            target=_run_gan_thread,
            args=(
                st.session_state.raw_dir,
                st.session_state.annotation_dir,
                processed_dir,
                final_output_dir,
                int(epochs),
                int(batch_size),
                float(lr_g),
                float(lr_d),
                int(image_size),
                model_variant,
                bool(diff_augment_enabled),
                float(ema_decay),
                int(save_int),
                int(num_preview),
                bool(is_resume),
                bool(use_roi),
                float(roi_margin),
                bool(enhance_contrast),
                bool(denoise),
                int(min_box_size),
                bool(strict_label_match),
                st.session_state.pause_event,
                st.session_state.stop_event,
                st.session_state.gan_control,
            ),
            daemon=True,
        )
        st.session_state.gan_thread = thread
        thread.start()
        st.info("训练已在后台启动，可用侧边栏按钮暂停、继续或停止。")
        st.rerun()


st.divider()
st.header("监控与结果")

active_dir = output_dir if method == TRADITIONAL_METHOD else st.session_state.output_path
st.caption(f"当前监控目录：`{active_dir}`")

if method == GAN_METHOD:
    control = st.session_state.gan_control
    if control.get("state") == "error":
        st.error(f"训练错误：{control.get('error', '未知错误')}")
    elif control.get("state") == "stopped":
        st.warning(f"训练已停止，最后完成 epoch：{control.get('epoch', 0)}")
    elif control.get("state") == "paused":
        st.info(f"训练已暂停在 epoch {control.get('epoch', 0)}。")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader("训练损失曲线")
        log_file = os.path.join(active_dir, "training_log.csv")
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                if not df.empty:
                    st.line_chart(df.set_index("epoch")[["D_loss", "G_loss"]])
                else:
                    st.info("日志文件为空。")
            except Exception as exc:
                st.warning(f"日志读取失败：{exc}")
        else:
            st.info("等待日志写入...")

    with col2:
        st.subheader("最新生成样本")
        if os.path.exists(active_dir):
            files = sorted([f for f in os.listdir(active_dir) if f.startswith("epoch_") and f.endswith(".png")])
            if files:
                latest_file = files[-1]
                st.image(os.path.join(active_dir, latest_file), caption=latest_file, use_container_width=True)
            else:
                st.info("等待生成样本...")
        else:
            st.info("目录尚未创建。")

    st.divider()
    st.subheader("最佳 Epoch 推荐")
    if st.button("分析当前目录最佳 Epoch"):
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
                caption=f"最佳 Epoch 示例：{preview_files[0]}",
                use_container_width=True,
            )

        show_cols = ["epoch", "sharpness", "diversity", "score", "num_images"]
        st.dataframe(score_df[show_cols].sort_values("score", ascending=False).head(10), use_container_width=True)

    st.divider()
    st.subheader("高质量样本导出")
    st.caption("用于改善生成样本的展示观感：先多生成候选样本，再按清晰度、灰度对比和亮度范围优选，可选真实统计匹配。正式论文指标仍建议重新做下游验证。")
    export_profile = st.selectbox(
        "导出策略",
        ["平衡模式（推荐）", "训练优先（少后处理）", "展示优先（纹理更明显）"],
        index=0,
        help="平衡模式适合答辩展示和后续训练预览；展示优先会增强纹理，但不建议直接替代正式实验结果。",
    )
    profile_settings = {
        "训练优先（少后处理）": {"truncation": 0.95, "oversample": 2, "refine": False, "std_scale": 0.80, "clahe": 0.0, "sharpen": 0.0},
        "平衡模式（推荐）": {"truncation": 0.85, "oversample": 3, "refine": True, "std_scale": 0.85, "clahe": 0.8, "sharpen": 0.08},
        "展示优先（纹理更明显）": {"truncation": 0.80, "oversample": 4, "refine": True, "std_scale": 0.90, "clahe": 1.2, "sharpen": 0.15},
    }
    selected_profile = profile_settings[export_profile]
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        checkpoint_path = st.text_input("Checkpoint路径", value=os.path.join(active_dir, "checkpoint_latest.pth"))
    with exp_col2:
        export_output_dir = st.text_input("导出目录", value=os.path.join(active_dir, "export_quality_selected"))
    with exp_col3:
        export_samples_per_class = st.number_input("每类导出数量", min_value=1, max_value=500, value=50, step=10)

    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        export_truncation = st.slider("潜变量截断", 0.50, 1.20, float(selected_profile["truncation"]), 0.05)
    with adv_col2:
        export_oversample = st.number_input("候选倍数", min_value=1, max_value=8, value=int(selected_profile["oversample"]), step=1)
    with adv_col3:
        export_refine = st.checkbox("真实统计匹配", value=bool(selected_profile["refine"]))

    refine_real_dir = st.text_input("真实样本统计目录", value=processed_dir)
    if st.button("导出高质量生成样本"):
        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint 不存在：{checkpoint_path}")
        else:
            try:
                raw_export_dir = export_output_dir
                summary = export_generated_samples(
                    checkpoint_path=checkpoint_path,
                    output_dir=raw_export_dir,
                    samples_per_class=int(export_samples_per_class),
                    image_size=int(image_size),
                    batch_size=int(batch_size),
                    truncation=float(export_truncation),
                    oversample_factor=int(export_oversample),
                    quality_select=True,
                )
                final_dir = raw_export_dir
                refine_summary = None
                if export_refine:
                    refined_dir = f"{raw_export_dir}_refined"
                    refine_summary = refine_generated_samples(
                        generated_dir=raw_export_dir,
                        output_dir=refined_dir,
                        real_dir=refine_real_dir if os.path.exists(refine_real_dir) else None,
                        target_std_scale=float(selected_profile["std_scale"]),
                        clahe_clip=float(selected_profile["clahe"]),
                        sharpen_amount=float(selected_profile["sharpen"]),
                    )
                    final_dir = refined_dir
                st.success(f"导出完成：{final_dir}")
                st.json(
                    {
                        "export": {
                            "written": summary["written"],
                            "truncation": summary["truncation"],
                            "oversample_factor": summary["oversample_factor"],
                            "quality_select": summary["quality_select"],
                        },
                        "refine": {
                            "enabled": export_refine,
                            "output_dir": final_dir,
                            "classes": list((refine_summary or {}).get("classes", {}).keys())[:6],
                        },
                    }
                )
                preview_files = []
                for root, _, files in os.walk(final_dir):
                    for file in files:
                        if file.lower().endswith((".png", ".jpg", ".jpeg")):
                            preview_files.append(os.path.join(root, file))
                    if len(preview_files) >= 8:
                        break
                if preview_files:
                    cols = st.columns(4)
                    for idx, path in enumerate(preview_files[:8]):
                        with cols[idx % 4]:
                            st.image(path, caption=Path(path).parent.name, use_container_width=True)
            except Exception as exc:
                st.error(f"导出失败：{exc}")

    st.divider()
    st.subheader("指定 Epoch 浏览")
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

elif method == TRADITIONAL_METHOD:
    if os.path.exists(active_dir):
        images = [f for f in os.listdir(active_dir) if f.endswith(".png")]
        if images:
            st.subheader(f"预览（共 {len(images)} 张）")
            cols = st.columns(4)
            for i, img_name in enumerate(images[:8]):
                with cols[i % 4]:
                    st.image(os.path.join(active_dir, img_name), caption=img_name, use_container_width=True)
        else:
            st.info("暂无结果。")
