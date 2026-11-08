# R2：CQS-P 与 CQS-R 架构融合实验

本目录提供新融合模型的实际实验代码。模型将可观测量依赖掩码约束的 Fourier 乘积直接输入共享 GRU，同时维护累计乘积状态和循环状态。FSPT 可以跨前缀查询复用状态，TFIM 保持原始标量监督，VQE 使用原始测量矩监督。

包含两组已经完成的实验：主实验中的 66 个融合模型拟合和 33 组 VQE 搜索，以及五个数据池、四种架构、每次训练 120 秒的 20 个可行性拟合。必要数据、模型检查点、预测及搜索轨迹通过本仓库的 Release 附件提供，代码、配置与结果说明放在本目录。

## 在 CPU 上复核已报告结果

使用 Python 3.12，在 `R2` 目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_artifacts.py
python reproduce.py --suite all
```

如果下载的是没有 Git 信息的源码 ZIP，为下载命令增加 `--repository OWNER/REPOSITORY`。已下载附件时可以用 `--archive PATH`。不需要私人服务器、账户凭证或真实量子设备。

复核会从完整测试预测数组重算 66 个主模型的指标，对每个主模型检查点重新预测 16 个测试输入，独立计算 33 组搜索全部 660 个终点的真实电路能量，并复核原始基线预测和终点。还会重算 20 个可行性拟合的指标，对各检查点重新预测 16 个输入，并验证汇总均值和样本标准差。检查点跨后端预测采用绝对和相对容差各 0.000020，实际最大误差完整记录在 `reproduction_report.json`。

这是对已完成实验的数值复核，完整重新训练命令见英文说明。原始 GPU 验证集重放记录保存在各实验的 `verification.json` 中。

## 文件与论文实验对应

| 实验 | 位置 | 规模 |
| --- | --- | --- |
| 主 TFIM，英文论文表 65 | `paper_iclr2027/rebuttal_20261107/hybrid_main_20261108/` | 4 个规模 × 3 次原始划分，共 12 个拟合 |
| 主 FSPT，表 66 | 同上 | 4 个规模 × 3 个时域 × 3 次原始划分，共 36 个拟合 |
| 主 VQE，表 67–68 | 同上 | 6 个电路设置 × 3 套原始数据，共 18 个拟合和 33 组搜索 |
| 限时可行性，表 63–64 | `paper_iclr2027/rebuttal_20261107/hybrid_feasibility_20261108/` | 5 个原有独立数据池 × 4 个架构，共 20 个拟合 |

`main_models.py` 是三个主任务的融合架构实现。可行性目录下的 `models.py` 实现融合模型及循环、乘积、加法对照。`run_main.py` 和 `run_hybrid.py` 训练模型，`analyze_main.py` 和 `analyze_hybrid.py` 执行原始核验与评分流程。依赖的电路、掩码和训练工具保存在 `world_model_surrogate/` 中。

各目录中的 `protocol.json` 为冻结配置，`rows.json` 为逐次结果，`summary.json` 为汇总结果，`实验结果.md` 保留三位小数，以均值 ± 样本标准差展示结果。原始数据保持完整精度。主模型拟合在 `fits/` 中，可行性拟合在 `pools/` 中，主 VQE 搜索轨迹在 `search/` 中。

主实验融合模型在 3/4 个 TFIM 设置中取得最低平均 MAE，在全部 12 个 FSPT 设置中优于 CQS-P，在 6 个设置中优于原 CQS-R。直接 VQE 搜索未超过六个主要设置中的 CQS-P+C。TFIM/FSPT 三次运行是原始数据池的配对划分，VQE 沿用原始独立数据集。更强的 24 周期分阶段 CQS-R 另有 12 周期预训练，结果中单独标明。

## 运行现有测试

以下命令分别启动进程，避免实验本地模块同名：

```bash
python -m unittest discover -s paper_iclr2027/rebuttal_20261107/hybrid_main_20261108 -p 'test*.py' -v
python -m unittest discover -s paper_iclr2027/rebuttal_20261107/hybrid_feasibility_20261108 -p 'test*.py' -v
```

共 13 个测试，覆盖融合梯度、掩码、前缀输出和梯度等价、因果性、任务适配、参数匹配、数据划分与搜索完整性。

## 重新训练

先下载数值附件，再执行 `python prepare_fresh.py ../R2-fresh`，生成只包含代码、原始输入和冻结协议的新目录，保留已有实验结果。完整训练与评分命令见 [README.md](README.md)。原始训练采用 Python 3.10、PyTorch 2.6.0、CUDA 12.4、NumPy 2.0.2、两条 CPU 线程及 RTX 4090，对应依赖见 `requirements-training-cuda.txt`。应等待 GPU 空闲后再运行，限时训练结果会受到设备性能影响。

本次上传执行数值复核，不重复全部训练。主实验沿用原始数据、预算、监督方式和验证集选择。数值附件的文件清单见 `artifact_inventory.json`。JSON 中的工作站路径已转为相对路径，数值数组保持原样。私人服务器连接与远程作业启动脚本未包含在内。
