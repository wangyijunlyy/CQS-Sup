# R3 新增实验复现包

本目录包含新增实验及依赖代码。实际数据、划分、模型参数、预测、验证历史、优化轨迹和测量记录，放在同一仓库的 `r3-repro-20261108` 发布附件中。不是仅提供本地路径；运行下载脚本即可取得文件。

在本目录中使用 Python 3.12：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_artifacts.py
python reproduce.py --suite all
```

共有 8,244 份数据与记录文件，包含 432 个检查点。`artifact_inventory.json` 给出清单。默认复现从真实数据、保存的模型与搜索记录重新计算结果，不需要访问作者电脑、服务器或量子硬件账户。

可以用 `--suite hardware`、`w1`、`w2`、`query`、`chain` 分别复算。完整训练和搜索入口、GPU 计时条件及重跑方法见英文 README。

代码与记录已去除作者身份、主机路径和账号信息，实验数值、随机种子和数据划分保留。文档中的相对目录均属于实际发布包内部。
