# 当前大作业：UCI HAR 人体活动识别

当前训练、融合实验、最终模型和独立测试见 [HAR 项目说明](har_project/README.md)，AI 参与见 [AI 使用记录](har_project/AI使用记录.md)。下方保留早期干豆分类练习的说明，两项目结果分别管理。

---

## 监督学习基线练习（历史实验）

用 UCI Dry Bean 数据集完成七分类监督学习，比较逻辑回归、随机森林和 RBF 支持向量机。

实验代码、数据、报告提纲和 AI 披露记录在 [supervised_learning_baseline（监督学习基线练习）](supervised_learning_baseline/README.md)。

当前已完成数据检查、固定划分、基线比较与预设网格调参。按训练集五折交叉验证选择最终 SVM（C=10、gamma=scale），CV 宏平均 F1 为 0.943764，选择已冻结。独立测试集尚未评价。下一步按项目说明运行 evaluate，再根据真实结果完成论文。
