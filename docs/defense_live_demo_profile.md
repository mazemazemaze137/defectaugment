# 答辩现场演示配置

## 现场演示原则

现场运行短流程，正式结论展示已保存的 40 轮 cGAN-v2 结果、比例消融、多随机种子和工业门槛评估。这样既能证明系统可运行，又能避免把答辩时间消耗在随机训练上。

## 推荐现场参数

- GAN 演示训练轮数：3 轮
- GAN 演示分辨率：128x128
- 分类验证演示轮数：3 轮
- 目标检测 smoke 轮数：1 轮
- 质量评估演示配对数：80，现场默认不计算 FID

## 正式结果引用

- cgan_v2_40ep_fid: 178.41
- ratio_ablation_best: 600 generated samples, best validation accuracy 99.72%, final validation accuracy 98.61%
- multiseed_best: cGAN-v2 600 samples average best validation accuracy 99.44%, baseline 98.15%
- industrial_gate: 600-sample setting passed: min recall 98.33%, cost-weighted error 0.29%
- classifier_comparison: small_cnn 96.11%, mobilenet_v3_small 95.56%, resnet18 93.33% in the formal backbone comparison run

## 证据路径检查

- OK demo_checklist: `docs/defense_demo_checklist.md`
- OK defense_summary: `docs/defense_summary.md`
- OK formatted_thesis: `毕业论文_初稿_格式化.docx`
- OK ratio_ablation_figure: `assets/figures/ratio_ablation_cgan_v2_40ep.png`
- OK multiseed_figure: `assets/figures/multiseed_cgan_v2_600.png`
- OK industrial_gate_figure: `assets/figures/industrial_gate_comparison.png`
- OK classifier_comparison: `docs/classifier_model_comparison_cgan_v2_600_seed42.md`

## 答辩表述建议

- 现场短轮次训练用于展示数据读取、ROI 预处理、GPU 调用、Loss 曲线、断点续训和文件输出。
- 正式生成效果不依赖现场随机训练，而是展示 40 轮 cGAN-v2 模型、质量指标、比例消融和多随机种子验证。
- 若老师要求看完整训练，可说明完整实验已保存 checkpoint、summary、曲线和证据图，现场重新训练会浪费答辩时间。
- 目标检测验证当前证明 XML 标注读取和检测训练链路可运行；GAN ROI 样本没有 bbox，因此不直接混入检测训练。
