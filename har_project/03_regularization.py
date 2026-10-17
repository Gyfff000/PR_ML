from pathlib import Path
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. 只读取官方训练集
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "UCI HAR Dataset"

X = np.loadtxt(data_dir / "train/X_train.txt")
y = np.loadtxt(data_dir / "train/y_train.txt", dtype=int)
subjects = np.loadtxt(
    data_dir / "train/subject_train.txt", dtype=int
)

# 2. 每一折先标准化，再训练逻辑回归
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.01, max_iter=3000)
)

# 3. 按受试者分成五折
cv = GroupKFold(n_splits=5)

scores = cross_validate(
    model,
    X,
    y,
    groups=subjects,
    cv=cv,
    scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
    return_train_score=True,
    n_jobs=1,
    error_score="raise"
)

# 4. 显示每折结果
for i in range(5):
    print(
        f"第 {i + 1} 折："
        f"验证准确率={scores['test_accuracy'][i]:.4f}，"
        f"验证宏平均F1={scores['test_macro_f1'][i]:.4f}，"
        f"训练宏平均F1={scores['train_macro_f1'][i]:.4f}"
    )

print("\n五折平均准确率：", scores["test_accuracy"].mean())
print("五折平均宏平均F1：", scores["test_macro_f1"].mean())
print("宏平均F1折间标准差：", scores["test_macro_f1"].std())

# 5. 保存真实结果，后续用于论文
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

results = np.column_stack([
    np.arange(1, 6),
    scores["test_accuracy"],
    scores["test_macro_f1"],
    scores["train_macro_f1"]
])

np.savetxt(
    output_dir / "logistic_C0p01_group_cv.csv",
    results,
    delimiter=",",
    header="fold,val_accuracy,val_macro_f1,train_macro_f1",
    comments=""
)