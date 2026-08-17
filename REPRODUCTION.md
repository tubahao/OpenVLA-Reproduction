# OpenVLA-Reproduction

本仓库是 **OpenVLA: An Open-Source Vision-Language-Action Model**（arXiv:2406.09246v3）在单张 RTX 4090（24GB）上的复现工作：

- 官方 checkpoint 推理基准与 LIBERO-Spatial 仿真评估（100 trials，成功率 84.0%，论文报告 84.7±0.9%）
- 基于官方 [openvla/openvla](https://github.com/openvla/openvla)（commit c8f03f4）的 LoRA 微调复现实验与完整根因分析

## 内容结构

- eproduction/OpenVLA_模型复现报告.md — 完整复现报告
- eproduction/figures/ — 报告图表（训练曲线、分任务成功率对比）
- eproduction/remote_scripts/ — 复现使用的脚本（LoRA 微调启动、环境检查、结果解析等）
- 其余目录为官方 openvla/openvla 代码，仅修改 la-scripts/finetune.py 与 xperiments/robot/openvla_utils.py

详细实验设置、结果与结论见复现报告。
