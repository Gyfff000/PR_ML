# 智能手机人体活动识别：学习项目

当前完成数据读取与划分检查，尚未训练分类模型。

## 在 Visual Studio 中运行

通过“文件 → 打开 → 文件夹”打开 `D:\1111\PR_ML\har_project`。选择工作区环境 `.venv (3.14, 64-bit)`，解释器应为 `D:\1111\PR_ML\har_project\.venv\Scripts\python.exe`。

将 `01_inspect.py` 设为启动项，保存后按 Ctrl+F5 运行。数据目录根据脚本所在位置解析，移动项目后无需修改绝对路径。

## 环境与数据

Python 3.14.3，NumPy 2.5.4，依赖见 requirements-lock.txt。虚拟环境、数据及 VS 缓存不提交 Git。

官方数据：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

数据解压后应包含 `data/UCI HAR Dataset/train/X_train.txt`、`train/y_train.txt`、`train/subject_train.txt` 和 `test/subject_test.txt`。官方压缩包可能需要解压多层，运行前确认 train/test 文件夹位于该数据目录中。

2026-10-14 数据检查：训练特征为 (7352, 561)，标签为 (7352,)，训练受试者 21 人，测试受试者 9 人，受试者交集为空，特征均为有效数值。

## AI 辅助记录

选题、数据读取代码和配置指导由 AI 辅助提供；学生在 Visual Studio 中编辑与运行。具体参与内容见 AI使用记录.md。后续模型训练、调参和论文需分阶段完成。
