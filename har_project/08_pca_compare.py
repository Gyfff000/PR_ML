from pathlib import Path
import csv
import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "UCI HAR Dataset"

X = np.loadtxt(data_dir / "train/X_train.txt")
y = np.loadtxt(data_dir / "train/y_train.txt", dtype=int)
subjects = np.loadtxt(
    data_dir / "train/subject_train.txt", dtype=int
)

rows = []

# None 是不降维的对照组
for ratio in [None, 0.90, 0.95, 0.99]:
    label = "No PCA" if ratio is None else f"PCA {ratio:.0%}"
    print(f"\n开始：{label}", flush=True)

    steps = [StandardScaler()]

    if ratio is not None:
        steps.append(
            PCA(n_components=ratio, svd_solver="full")
        )

    steps.append(LogisticRegression(C=0.1, max_iter=3000))
    model = make_pipeline(*steps)

    start = time.perf_counter()

    scores = cross_validate(
        model,
        X,
        y,
        groups=subjects,
        cv=GroupKFold(n_splits=5),
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        return_train_score=True,
        return_estimator=True,
        n_jobs=1,
        verbose=2,
        error_score="raise"
    )

    elapsed = time.perf_counter() - start

    # 每折的 PCA 仅根据该折训练数据拟合，维数可能不同
    dimensions = [
        estimator.named_steps["pca"].n_components_
        if ratio is not None else X.shape[1]
        for estimator in scores["estimator"]
    ]

    row = {
        "method": label,
        "mean_components": float(np.mean(dimensions)),
        "min_components": int(min(dimensions)),
        "max_components": int(max(dimensions)),
        "train_macro_f1": float(scores["train_macro_f1"].mean()),
        "cv_macro_f1": float(scores["test_macro_f1"].mean()),
        "cv_f1_std": float(scores["test_macro_f1"].std()),
        "cv_accuracy": float(scores["test_accuracy"].mean()),
        "seconds": elapsed
    }
    rows.append(row)

    print(
        f"{label}：各折维数={dimensions}，"
        f"训练F1={row['train_macro_f1']:.4f}，"
        f"验证F1={row['cv_macro_f1']:.4f}，"
        f"标准差={row['cv_f1_std']:.4f}，"
        f"耗时={elapsed:.1f}秒"
    )

output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

with open(
    output_dir / "pca_compare_cv.csv",
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

best = max(rows, key=lambda row: row["cv_macro_f1"])
print("\n本轮验证F1最高的方法：", best["method"])
print("交叉验证宏平均F1：", best["cv_macro_f1"])