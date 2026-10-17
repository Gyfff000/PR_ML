from pathlib import Path
import csv
import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "UCI HAR Dataset"

X = np.loadtxt(data_dir / "train/X_train.txt")
y = np.loadtxt(data_dir / "train/y_train.txt", dtype=int)
subjects = np.loadtxt(
    data_dir / "train/subject_train.txt", dtype=int
)

# 随机森林不需要标准化
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=2
)

param_grid = {
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 3]
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=GroupKFold(n_splits=5),
    return_train_score=True,
    refit=False,
    n_jobs=1,
    verbose=2,
    error_score="raise"
)

start = time.perf_counter()
search.fit(X, y, groups=subjects)
elapsed = time.perf_counter() - start

results = search.cv_results_
rows = []

for i, params in enumerate(results["params"]):
    depth = params["max_depth"]
    leaf = params["min_samples_leaf"]
    train_f1 = results["mean_train_score"][i]
    val_f1 = results["mean_test_score"][i]
    std = results["std_test_score"][i]

    print(
        f"max_depth={depth}, min_samples_leaf={leaf}："
        f"训练F1={train_f1:.4f}，"
        f"验证F1={val_f1:.4f}，标准差={std:.4f}"
    )
    rows.append([str(depth), leaf, train_f1, val_f1, std])

print("\n最佳参数：", search.best_params_)
print("最佳交叉验证宏平均F1：", search.best_score_)
print(f"搜索总耗时：{elapsed:.1f} 秒")

output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

with open(
    output_dir / "forest_grid_cv.csv",
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "max_depth", "min_samples_leaf",
        "train_macro_f1", "cv_macro_f1", "cv_f1_std"
    ])
    writer.writerows(rows)