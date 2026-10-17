from pathlib import Path
import csv
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "UCI HAR Dataset"

X = np.loadtxt(data_dir / "train/X_train.txt")
y = np.loadtxt(data_dir / "train/y_train.txt", dtype=int)
subjects = np.loadtxt(
    data_dir / "train/subject_train.txt", dtype=int
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=3000)
)

# make_pipeline 自动给逻辑回归命名为 logisticregression
param_grid = {
    "logisticregression__C": [0.001, 0.01, 0.1, 1, 10]
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=GroupKFold(n_splits=5),
    return_train_score=True,
    refit=False,
    n_jobs=1,
    verbose=1,
    error_score="raise"
)

search.fit(X, y, groups=subjects)

results = search.cv_results_
rows = []

for i, params in enumerate(results["params"]):
    C = params["logisticregression__C"]
    train_f1 = results["mean_train_score"][i]
    val_f1 = results["mean_test_score"][i]
    std = results["std_test_score"][i]

    print(
        f"C={C:g}：训练F1={train_f1:.4f}，"
        f"验证F1={val_f1:.4f}，标准差={std:.4f}"
    )
    rows.append([C, train_f1, val_f1, std])

print("\n最佳参数：", search.best_params_)
print("最佳交叉验证宏平均F1：", search.best_score_)

output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

with open(
    output_dir / "logistic_grid_cv.csv",
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "C", "train_macro_f1", "cv_macro_f1", "cv_f1_std"
    ])
    writer.writerows(rows)