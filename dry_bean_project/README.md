# 基于形态特征的干豆品种分类及模型比较

## 作业目标与进度

完成七分类监督学习实验，比较逻辑回归、随机森林、RBF 支持向量机，并分析类别不平衡、标准化与误分类。多数类分类器作为最低性能基线。

当前阶段：已完成数据准备、基线比较与预设网格调参。已检查无缺失值，删除 68 条完全重复记录，清洗后 13,543 条，训练集 10,834 条、测试集 2,709 条。按五折 CV 宏平均 F1 选择 SVM（C=10、gamma=scale），分数为 0.943764，选择已冻结。逻辑回归调参后为 0.936557，随机森林为 0.936317。下一步运行 evaluate，独立测试集尚未评价，完整结果与论文仍待完成。详见 实验记录.md。

课件要求：报告用学术论文格式，必须包括“模型描述”和“实验结果”。AI 辅助需附 prompt/MD 文件与参与内容列表。课件写明 2026 年 10 月 27 日交纸质报告，10 月 31 日或 11 月 1 日展示数据及代码运行，具体安排以课程通知为准。

## 在 PowerShell 中运行

不需要激活虚拟环境，直接使用项目里的 Python。先进入目录：

```powershell
cd 'D:\1111\PR&ML\dry_bean_project'
```

1. 查看与检查数据：

```powershell
.\.venv\Scripts\python.exe train.py --step prepare
```

2. 基线比较：

```powershell
.\.venv\Scripts\python.exe train.py --step baseline
```

每个模型在训练集上做五折交叉验证。先看 `cv_macro_f1`，再看准确率和训练/验证差距。`cv_f1_std` 为五折分数的标准差（ddof=0），表示折间波动，不是置信区间。这里输出的 `test_*` 字段来自 sklearn 的交叉验证接口，含义是每折验证分数；脚本表格已改名为 `cv_*`，没有评价独立测试集。

3. 在训练集内部调参：

```powershell
.\.venv\Scripts\python.exe train.py --step tune
```

逻辑回归搜索 C = 0.1、1、10。随机森林搜索最大深度 None/20、叶节点最小样本数 1/3，树数固定 200。SVM 搜索 C = 1、10、100，gamma = scale/0.01。每种模型用五折宏平均 F1 选参数，再按同一指标选最终模型。两个并行任务限制资源消耗，实际耗时以本机运行记录为准。

4. 模型选择冻结后，评价独立测试集：

```powershell
.\.venv\Scripts\python.exe train.py --step evaluate
```

输出基线和调参模型的测试成绩供报告比较，但最终模型仍由训练集交叉验证决定。看到测试成绩后不要继续根据测试集调参。脚本会阻止这种调参顺序。

每一步运行后，把末尾表格发到聊天里，我们一起解释，再进行下一步。

## 数据与实验设计

- 原始数据：13,611 行，16 个特征，7 类。类别列 Class 是标签。使用现成形态特征，不涉及从图片提取特征。
- 检查缺失值、数值有效性、重复行及相同特征对应冲突标签。相同特征但标签不同会中止并要求检查。完全重复行先删除，避免相同记录落入不同集合。
- 分层划分 80% 训练集、20% 测试集，随机种子 42。实际样本数以 `outputs/data_summary.json` 为准。
- 训练集内部五折分层交叉验证。StandardScaler 与模型放入 Pipeline，在每折内部拟合，避免标准化泄漏。
- 主指标：宏平均 F1，每类同等权重。辅助指标：Accuracy、Balanced Accuracy、每类 Precision/Recall/F1 与混淆矩阵。
- 搜索得到的最佳 CV 分数用于模型选择，可能有选择偏差。独立测试集用于最终评价。单次划分不能证明跨场景泛化。

## 输出文件

`outputs/data_summary.json`：数据与环境摘要。`class_counts.csv`：各类在全部数据、训练集和测试集中的数量。

`baseline_cv.csv`：基线交叉验证结果。`tuned_cv.csv`：各模型最佳参数的交叉验证成绩。`*_grid.csv`：完整参数搜索记录。两个表的折间标准差均使用 ddof=0，不能直接当成显著性检验。

`selection.json`：测试前冻结的模型选择。`test_metrics.csv`：测试指标。`selected_classification_report.csv`：最终模型各类指标。`test_predictions.csv` 与 `misclassified_samples.csv`：逐样本预测与误分类。`error_pairs.csv`：常见混淆方向。

`class_distribution.png` 与 `confusion_matrix.png`：报告图。`models/`：训练得到的完整模型及标准化流水线。

## 复现与来源

已安装依赖记录在 `requirements-lock.txt`。当前虚拟环境使用 Python 3.12。环境移动到其他电脑后需要重新创建，可用 Python 3.12 创建 `.venv` 再安装锁定依赖。Git 不提交环境、原始工作簿及压缩包。克隆仓库后先进入 dry_bean_project，再执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe download_data.py
```

然后按上文分步运行。数据校验值匹配时可以沿用已提交的划分；需要重新准备时，在尚未训练的阶段运行 prepare。

数据来源：https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

数据 DOI：10.24432/C50S4B。许可 CC BY 4.0。原始压缩包、工作簿及来源/校验值保存在 `data/raw/`。

方法参考：https://scikit-learn.org/stable/common_pitfalls.html

班级内数据集重复情况未知。本项目通过去重、规范验证和误分类分析形成自己的实验，不保证与其他同学无重复。
