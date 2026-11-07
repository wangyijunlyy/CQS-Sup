# R1 实验代码与复现说明

本目录覆盖完整回复中的四项问题，包括掩码与 Fourier 对照、共同系数惩罚、并行轨迹模型、等训练时间 Query-GRU、十套新数据的一维链固定数据搜索、ansatz 能量参考、网格延拓对照、测量增强链优化、五套新数据局部优化确认，以及强耦合网格实验。

源码与冻结配置放在 Git 中。实际数据、检查点、验证记录、完整搜索与测量轨迹放在同仓库的 Release 附件中，由下载脚本获取。不是只有汇总表或一份结果声明。

## 复现现有结果

使用 Python 3.12，在本 R1 目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_artifacts.py
python reproduce.py --suite all
```

下载器从当前 Git 仓库的 origin 自动识别附件地址。若使用网页下载的源码 ZIP，需要向下载器传入 `--repository OWNER/REPOSITORY`。不需要作者电脑、私人服务器、硬件账号或 QPU。

默认流程会重新计算数值证据，包括检查点预测、验证选择、Fourier 拟合、真实终点能量、测量接受决策和 shots 预算。它不会假装重新训练所有模型。结果写入 `reproduction_report.json`，其他核验结果放在 `outputs/`。

并行模型部分会从保存的预测重新计算全部45项测试指标，并在 CPU 上重新求值45个检查点。其中25个 GRU/MLP 检查点作严格预测匹配，20个 Transformer 的 CPU 输出作为跨后端诊断记录。原 PyTorch 2.6 CUDA 融合 Transformer 路径与 CPU/非融合路径存在数值差异，已通过相同检查点、相同输入和 CUDA 快速路径开关定位。模型与容差均未为此改变。要严格重放全部45个检查点，请在 NVIDIA GPU 和 Python 3.10 环境安装 `requirements-parallel-cuda.txt`，运行 `python reproduce.py --suite parallel --device cuda`。

可以单独运行：

```bash
python reproduce.py --suite masks
python reproduce.py --suite common
python reproduce.py --suite parallel
python reproduce.py --suite query
python reproduce.py --suite fixed-chain
python reproduce.py --suite reference
python reproduce.py --suite grid-continuation
python reproduce.py --suite local-chain
python reproduce.py --suite local-confirmation
python reproduce.py --suite strong-grid
python reproduce.py --suite timing
```

英文 README 的对应表逐项列出了 Weakness/Question、论文表格和复现入口。十套新数据的固定数据链实验直接对应回复中的新增表，不虚构其在当前论文中的表号。

## 重新训练与搜索

`rerun.py` 将原始程序的输出重定向到独立目录，保留完整模型、验证网格、随机种子、测量预算与搜索设置。例如：

```bash
python rerun.py fixed-chain --out outputs/new_fixed_chain
python rerun.py parallel --out outputs/new_parallel --device cuda
python rerun.py query --out outputs/new_query --device cuda
python rerun.py common --out outputs/new_common
python rerun.py local-chain --out outputs/new_local_chain
python rerun.py local-confirmation --out outputs/new_local_confirmation
python rerun.py strong-grid --out outputs/new_strong_grid
```

新目录用于全新运行，已完整生成的同目录结果会继续复用，避免无提示覆盖。完整命令和 Fourier 局部校正、直接优化器的独立入口见英文 README。十套新链实验默认完成310个乘积拟合、360次搜索；也可用 `--seeds 3101` 运行一个完整数据集，但不能把单数据集当成十数据集汇总。

`requirements.txt` 固定已经测试的 Python 3.12核验环境。`requirements-chain-training.txt` 记录新链训练与搜索实际采用的 Python 3.10环境。各次拟合的元数据也保留其库版本。相同墙钟训练预算在不同硬件上的更新次数会不同，训练和计时结果不应被理解为跨硬件逐位一致。

所有比较保留各自的数据、全部耦合和全部终点。固定数据+C、参考附近初值、局部测量增强、网格热启动无平滑搜索分别记录，不能混作同一实验。预训练链的局部测量扩展总预算为1686000次 shots，固定数据新链每池采集24000次 shots、搜索不新增测量。

内容已移除个人姓名、本机路径、私人服务器和账号信息、凭据。数值、权重、种子、数据用途和实际测量结果保留。GitHub 托管账号及仓库活动信息不属于内容匿名化范围。
