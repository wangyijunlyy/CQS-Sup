# R3 新增实验复现包

本目录包含新增实验及依赖代码。早先实验的实际数据、划分、模型参数、预测、验证历史、优化轨迹和测量记录，放在同一仓库的 `r3-repro-20261108` 发布附件中。不是仅提供本地路径；运行下载脚本即可取得文件。

在本目录中使用 Python 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_artifacts.py
python reproduce.py --suite all
```

共有 8,492 份数据与记录文件，包含 456 个检查点。`artifact_inventory.json` 给出清单。默认复现从真实数据、保存的模型与搜索记录重新计算结果，不需要访问作者电脑、服务器或量子硬件账户。

可以用 `--suite hardware`、`w1`、`w2`、`query`、`chain` 分别复算。完整训练和搜索入口、GPU 计时条件及重跑方法见英文 README。

代码与记录已去除作者身份、主机路径和账号信息，实验数值、随机种子和数据划分保留。文档中的相对目录均属于实际发布包内部。

另附 TFIM 原训练标签、检查点和基线记录，可使用 `python retrain_tfim.py --check-inputs` 核验全部12组原始输入，或执行 `python retrain_tfim.py --n 8 --seed 42` 重新拟合调参 NC。
# 跨哈密顿量复用实验

新增实验的代码、实际测量数据、20个原始完整检查点和全部搜索记录，已直接放入独立的 `hamiltonian_reuse_20261117` 目录。该实验无需下载个人账号的Release附件，无需作者本机或量子硬件账号。

在R3目录下运行：

```bash
python -m venv .venv-reuse
source .venv-reuse/bin/activate
python -m pip install -r hamiltonian_reuse_20261117/requirements.txt
python reproduce.py --suite hamiltonian-reuse
```

原始运行使用Python 3.10，完整复算也在Python 3.12下通过。实验保留两类电路、两种耦合顺序、全部独立预算组，共360个序列和72,000个终点。SPSA允许复用历史测量与热启动，每个新任务均保留扰动起点。代理初始采集成本全部计入，跨哈密顿量期间不重新拟合模型。结果报告同时给出平滑顺序及链式固定代理的适用边界。

复现会重建历史测量、参数更新、保留决策和累计成本，重算400次代理目标优化，并复核全部终点能量。还可使用独立目录重新生成全部工作负载：

```bash
python hamiltonian_reuse_20261117/reproduce.py --fresh --out fresh_reuse --workers 2
```

以上重新运行使用发布的原始测量池、未改动的完整检查点和冻结的SPSA设置，不重新训练原始模型。完整设置见该目录的README和结果报告。下文为早先R3实验的说明。
