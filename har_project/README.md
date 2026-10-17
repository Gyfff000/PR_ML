# UCI HAR：人工特征与时序特征融合研究

截至 2026-10-17，已完成监督分类、分组交叉验证、调参、PCA 对照、神经网络单分支及融合实验、融合模型随机种子复验，以及冻结模型后的官方测试。论文正在整理，仓库中的结果来自实际运行记录。

## 数据与评估约定

- 数据：UCI Human Activity Recognition Using Smartphones，六类活动。
- 官方训练集：7352 个窗口、21 名受试者；官方测试集：2947 个窗口、9 名受试者。两者受试者不重叠。
- 人工特征：每个窗口 561 维；时间序列：9 个通道 × 128 个时间点。
- 模型选择使用官方训练集的五折 GroupKFold，以受试者为分组单位。
- 神经网络在每个外层训练折内再用 GroupShuffleSplit 留出受试者选择轮数，然后重新初始化、在该外层全部训练数据上重训，最后评估外层验证集。
- 标准化仅拟合相应训练部分。独立测试在模型配置和权重固定后执行，不用于调参。

官方数据与引用：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones ，DOI：10.24432/C54S4K。

## 已完成的实验

以下为训练集模型选择阶段的五折平均宏平均 F1。神经网络主对照使用基础种子 42，每折训练种子为基础种子加折号减一。传统模型的调参均值是模型选择结果，不作为独立测试成绩。

| 模型/配置 | CV 宏平均 F1 | 折间标准差 |
|---|---:|---:|
| 逻辑回归，C=0.1，无 PCA | 0.933897 | 0.0427 |
| 双分支融合 | 0.931812 | 0.043830 |
| RBF SVM，C=10，gamma=0.001 | 0.929178 | 0.0389 |
| MLP | 0.928606 | 0.042736 |
| 1D CNN | 0.924130 | 0.043785 |
| 随机森林，max_depth=20，min_samples_leaf=3 | 0.916476 | 0.0453 |

PCA 保留率 90%、95%、99% 的对照结果见 `outputs/pca_compare_cv.csv`，该实验固定逻辑回归 C=0.1。

融合模型在基础种子 42、2024、2025 下的五折均值为 0.931812、0.933000、0.929527；三次均值平均为 0.931447，种子间样本标准差（ddof=1）为 0.001765。该统计与折间标准差分别报告。

## 最终独立测试

交叉验证均值最高的逻辑回归作为预先指定的主模型，融合模型作为研究对照。融合模型固定基础种子 42、训练 9 轮；9 是三次实验共 15 个内层所选轮数的中位数，规则在测试前确定。

| 模型 | 测试准确率 | 测试宏平均 F1 | 错误窗口数 |
|---|---:|---:|---:|
| 逻辑回归 | 0.949779 | 0.950026 | 148 |
| 融合模型 | 0.955548 | 0.956384 | 131 |

融合模型在本次测试中净减少 17 个错误，动态活动的 F1 有所改善，坐/站立的 F1 有所下降。没有进行显著性检验，不声称普遍优于其他模型。论文研究重点为双分支特征融合及其局限，不将常见拼接结构宣称为全新架构。

## 在 Visual Studio 中复现

使用 Visual Studio（紫色图标）打开本目录，选择项目 `.venv` 解释器，将需要运行的脚本设为启动项或选择“当前文档”，按 Ctrl+F5。

当前记录环境为 Python 3.14.3、NumPy 2.5.4、scikit-learn 1.9.1、PyTorch 2.14.1+cpu。依赖锁定文件包含 CPU 轮子的索引地址，可在 VS 的 PowerShell 终端执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

从官方地址下载并解压数据，使其包含 `data/UCI HAR Dataset/train`、`test` 及各自的 `Inertial Signals` 子目录。数据集、预处理数组、虚拟环境和 VS 缓存不提交 Git。

脚本按编号分阶段运行：

1. `01_inspect.py`：数据检查。`02_baseline .py`（文件名含空格）到 `08_pca_compare.py`：传统模型、调参及 PCA 对照。
2. `10_check_torch.py`：环境检查；`11_prepare_neural_data.py`：对齐两种输入并保存固定外层折下标。
3. `12_mlp_pilot.py`、`14_cnn_pilot.py`、`16_fusion_pilot.py`：内层试训练；`13_mlp_group_cv.py`、`15_cnn_group_cv.py`、`17_fusion_group_cv.py`：正式五折实验。
4. `18_fusion_stability.py`：固定内层划分、改变训练基础种子为 2024/2025 的复验。
5. `19_train_final_models.py`：在完整官方训练集上训练并保存最终模型及标准化器。
6. `20_evaluate_test.py`：加载保存的模型进行独立测试，不重新拟合标准化器、不更新网络权重。

保存的模型位于 `models/final`，已有实验记录位于 `outputs`。重新运行训练会生成新的模型文件；仓库保存的原始测试结果同时记录了当时模型文件的 SHA-256。融合网络源文件的原始字节通过 `.gitattributes` 保留，以满足冻结配置中的源文件哈希校验。

## 结果文件

- `outputs/*group_cv.csv`：五折正式指标；`*inner_history.csv`：内层训练日志。
- `outputs/*oof_predictions.npz`：外层验证预测；`fusion_seed_2024`、`fusion_seed_2025`：复验记录。
- `outputs/final_test`：最终指标、分类报告、混淆矩阵和逐窗口预测。
- `models/final`：逻辑回归流水线、融合网络权重、融合预处理参数、固定配置和最终训练日志。

AI 参与及关键提示见 [AI使用记录.md](AI使用记录.md)。训练由学生在 Visual Studio 中执行，代码示例、实验设计和结果解读由 AI 辅助。论文已形成完整 LaTeX 初稿，仍待学生核实、填写署名及修改定稿。


## 论文图表

`21_make_paper_figures.py` 从已保存的实验记录生成四张图：融合网络结构、六模型五折分组交叉验证、两个冻结模型的测试混淆矩阵、测试各类别 F1。输出位于 `outputs/paper_figures`，包括 300 dpi PNG、四页矢量 PDF、图注、绘图数值表及来源哈希清单。

在 HAR 项目的 VS PowerShell 终端中安装绘图依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-figures.txt
```

随后选择 `21_make_paper_figures.py`，按 Ctrl+F5。脚本不加载模型、不训练或重新测试，依赖现有 `outputs` 和冻结网络源文件。训练环境锁定文件保持原样，绘图依赖单列。

初版图表已由 AI 使用本机现有 Matplotlib 3.11.2 环境生成并检查，实际绘图环境版本和来源文件 SHA-256 见 `plot_manifest.json`；学生可在 HAR 的 VS 环境安装上述依赖后自行复现。交叉验证图采用原始基础种子42，误差线为五折标准差（ddof=0），不是置信区间；独立测试与交叉验证分图呈现。


## 中文学术论文

完整 LaTeX 初稿位于 [paper](paper/README.md)，入口为 `paper/main.tex`，编译结果为 `paper/main.pdf`。采用开源 ElegantPaper v0.12，保留固定版本、完整上游模板及许可证；参考文献按 GB/T 7714 数字顺序编码排版。包含中英文摘要与关键词、数据描述、模型公式、实验协议、CV/PCA/稳定性/独立测试分析、局限与结论、参考文献。AI 参与清单和关键提示单独保存在 [AI使用记录.md](AI使用记录.md)，提交时与论文分别提供。

可在 Visual Studio 编辑 `.tex`，在该目录的 PowerShell 终端执行 `.\build.ps1`；详细说明见论文 README。正常编译不运行训练代码，也不需要 Python。先在 `author_info.tex` 填写本人真实姓名、班级和学号，再核对正文及其对应实验。论文中的表格与配对统计均由保存结果生成，但正文结论仍需本人理解和审核。
