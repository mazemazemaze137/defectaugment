# DefectAugment — 工业表面缺陷数据增强系统

## 功能
- 工业缺陷图像预处理
- 传统数据增强（旋转、翻转等）
- GAN 生成式数据增强（DCGAN）
- 增强质量评估（SSIM/PSNR/FID + 可视化）

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 将原始数据集放入 `data/raw/`
3. 运行：`python src/main.py`

## 目录说明
- `data/`: 原始与预处理数据
- `results/`: 增强结果与评估输出
- `src/`: 核心代码