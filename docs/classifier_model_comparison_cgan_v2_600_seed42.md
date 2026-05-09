# cGAN-v2 600张增强分类器模型对比

## 实验设置

- 分支：`defense-eval-extensions`
- 训练集：`data/raw/NEU-DET/train/images`
- 验证集：`data/raw/NEU-DET/validation/images`
- 生成样本：`results/ratio_ablation/cgan_v2_40ep_seed42/subsets/gan_100_per_class`
- 生成样本数量：每类 100 张，共 600 张
- 输入尺寸：128x128
- batch size：16
- 训练轮数：20
- 早停耐心：4
- 随机种子：42
- 低置信度阈值：0.70

## 结果

| 模型 | 最佳验证准确率 | 最终验证准确率 | 最佳验证损失 | 最佳轮数 | 实际轮数 | 低置信度/误分类样本数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small_cnn | 96.11% | 92.50% | 0.1225 | 8 | 12 | 42 |
| resnet18 | 93.33% | 90.83% | 0.2682 | 5 | 9 | 37 |
| mobilenet_v3_small | 95.56% | 87.22% | 0.1403 | 7 | 11 | 28 |

## 低置信度样本分布

| 模型 | 主要低置信度类别 |
| --- | --- |
| small_cnn | inclusion 21、pitted-surface 11、patches 6、scratches 4 |
| resnet18 | inclusion 20、pitted-surface 11、scratches 3、crazing 2、rolled-in-scale 1 |
| mobilenet_v3_small | crazing 11、inclusion 9、pitted-surface 5、scratches 3 |

## 分析

本次实验说明，更复杂的分类 backbone 并不会在默认超参数下自动优于项目中使用的 small CNN。ResNet18 和 MobileNetV3-Small 参数量更大，但在当前 128x128 灰度缺陷图、20 轮训练和统一学习率设置下，最佳验证准确率低于 small CNN，说明它们需要单独调学习率、训练轮数、正则化策略或预训练权重适配。

从低置信度样本看，`inclusion` 与 `pitted-surface` 仍是主要不确定类别。这与此前生成质量和工业召回率分析一致：系统后续优化应继续围绕夹杂、麻点等纹理相近或低对比度类别展开。

答辩中建议谨慎表述：本系统已经支持 ResNet18/MobileNet 对比和低置信度样本导出，但当前正式对比结果显示，默认参数下 small CNN 更适合本项目的轻量验证任务；这不代表大模型无效，而是说明工业缺陷任务需要结合数据规模、输入分辨率和训练策略进行模型选择。
