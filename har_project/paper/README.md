# 中文学术论文 LaTeX 项目

这是完整论文初稿及其可编译源文件，不是只有标题的模板。模型训练和最终测试成绩来自现有实验留档；排版不触发训练或测试。提交前仍须学生核对内容、理解代码，并填写真实署名。

## 开源模板与规范

采用 [ElegantPaper](https://github.com/ElegantLaTeX/ElegantPaper)，固定提交 `40275032d6e3073e658038de30a98e1b4785f052`，类文件版本 v0.12。原始模板完整保存在 `vendor/ElegantPaper`，根目录类文件是原样副本。LPPL-1.3c 或更高版本的许可证、作者署名与来源信息均保留，详见 `TEMPLATE_SOURCE.md`。论文使用 A4 单栏、中英文摘要与关键词、编号公式、图题在下表题在上、三线表、文中交叉引用、GB/T 7714 数字顺序编码参考文献、AI 参与附录。不伪造期刊卷期、DOI 或作者身份。

## 在 Visual Studio 中修改和编译

1. 用 VS（紫色图标）的“打开文件夹”打开 `har_project/paper`，或在当前工程中打开这些 `.tex` 文件。
2. 先编辑 `author_info.tex`，在 `\author{}` 中填真实姓名，在 `\institute{}` 中填班级和学号。学校若要求单位名称，也在此填写。当前留空。
3. 正文在 `sections`，主入口是 `main.tex`，文献在 `references.bib`。正常修改论文不需要运行 Python。
4. 打开 VS 的 PowerShell 终端，进入论文目录后执行：

```powershell
Set-Location -LiteralPath 'D:\1111\PR_ML\har_project\paper'
.\build.ps1
```

本机已安装 TeX Live 2026，脚本依次运行 XeLaTeX、Biber、两次 XeLaTeX，结果为 `main.pdf`。LaTeX 编译器与 Python 解释器互相独立，不需要给 Python 安装同名 LaTeX 包。脚本若被本机执行策略阻止，可直接运行以下四条命令，不必更改执行策略：

```powershell
& 'C:\texlive\2026\bin\windows\xelatex.exe' -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
& 'C:\texlive\2026\bin\windows\biber.exe' main
& 'C:\texlive\2026\bin\windows\xelatex.exe' -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
& 'C:\texlive\2026\bin\windows\xelatex.exe' -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
```

如果打开的 PDF 被其他程序锁定，先关闭该 PDF 再编译。也可在 Overleaf 上传完整源文件夹，主文件选 `main.tex`，编译器选 XeLaTeX；模板在非 Windows 平台由 CTeX 选择可用中文字体。

## 内容与实验对应

- `01_introduction`：研究问题、相关工作与贡献边界。
- `02_data`：官方数据、输入表示、标签与受试者隔离。
- `03_methods`：标准化、传统方法、MLP、CNN、拼接融合和损失函数。
- `04_protocol`：五折分组、内层选轮数、最终冻结、评价指标与环境。
- `05_results`：真实 CV/PCA/复验/独立测试、类别与受试者误差、局限。
- `06_conclusion`：结论与研究边界。
- `appendix_ai`：AI 辅助及复现说明；另见上级目录 `AI使用记录.md`。

## 数据来源与更新

`prepare_assets.py` 读取已提交的 CSV/JSON 与外层预测，不加载训练模型、不读取原始数据集、不重新预测，生成 `tables` 与 `figures`。`asset_manifest.json` 记录来源哈希；`audit_statistics.json` 保存配对错误、受试者统计和三种子均值。初稿素材使用已有 Matplotlib 环境生成，版本以清单为准。源项目已包含所有编译所需图表，正常编译不依赖此 Python 脚本或上级实验目录。

只有实验记录更新时才需要重新运行该脚本（需 NumPy、Matplotlib 和中文字体），并重新核对正文手写的结果描述；表格自动更新不会自动改写结论。正文、图表、配置和独立测试日志必须对应同一冻结实验。不得依据当前测试结果继续调参后，仍将其称为未触碰的最终测试。

## 学生核对顺序

先读“数据描述”和“评价协议”，确认能解释为什么不能随机拆分重叠窗口；再对照 `fusion_model.py` 逐层讲清图 1；最后读结果，解释为什么 CV 最好的仍是逻辑回归、为什么融合测试更高不代表每一类别都提高。务必理解模型复杂度、选择偏差与测试集独立性的区别。文献已按一手来源核验题名、作者、年份与可核对链接；提交前应实际阅读与自己论述有关的部分。

## 随源码保留的排版依赖

本机 TeX Live 安装缺少 multirow、placeins、abstract、appendix、footmisc。为保持原模板类文件不变，将这五个官方包的运行文件及完整 CTAN 源分发保存在 `vendor/tex-packages`，来源与下载哈希见该目录 `SOURCE.json`，原作者和许可声明随上游源码保留。主文件已配置搜索路径，编译无需升级 TeX Live 或改动 Python 环境。
