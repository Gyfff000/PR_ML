from pathlib import Path
import numpy as np

data_dir = Path(__file__).resolve().parent / "data" / "UCI HAR Dataset"

X = np.loadtxt(data_dir / "train/X_train.txt")
y = np.loadtxt(data_dir / "train/y_train.txt", dtype=int)
subjects = np.loadtxt(data_dir / "train/subject_train.txt", dtype=int)
test_subjects = np.loadtxt(
    data_dir / "test/subject_test.txt", dtype=int
)

print("训练特征形状：", X.shape)
print("训练标签形状：", y.shape)
print("各类标签及样本数：", np.unique(y, return_counts=True))
print("训练受试者人数：", len(np.unique(subjects)))
print("测试受试者人数：", len(np.unique(test_subjects)))
print("全部特征是否为有效数值：", np.isfinite(X).all())

overlap = np.intersect1d(subjects, test_subjects)
print("训练与测试共有的受试者：", overlap)

assert len(X) == len(y) == len(subjects)
assert X.shape[1] == 561
assert len(overlap) == 0