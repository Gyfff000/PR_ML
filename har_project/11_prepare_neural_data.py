from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


root = Path(__file__).resolve().parent
data_dir = root / "data" / "UCI HAR Dataset"
output_dir = root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 读取官方训练集的人工提取特征、活动标签和受试者编号
features = np.loadtxt(
    data_dir / "train" / "X_train.txt",
    dtype=np.float32,
)
original_labels = np.loadtxt(
    data_dir / "train" / "y_train.txt",
    dtype=np.int64,
)
subjects = np.loadtxt(
    data_dir / "train" / "subject_train.txt",
    dtype=np.int64,
)

# PyTorch 分类损失要求类别编号从 0 开始
labels = original_labels - 1

# 2. 读取同一批样本对应的九通道时间序列
channel_names = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

channels = []
for name in channel_names:
    channel = np.loadtxt(
        data_dir / "train" / "Inertial Signals" / f"{name}_train.txt",
        dtype=np.float32,
    )
    channels.append(channel)

# 每个通道原本是：样本数 × 时间点数
# 合并后是：样本数 × 通道数 × 时间点数
signals = np.stack(channels, axis=1)

# 3. 检查数据是否符合预期
assert features.shape == (7352, 561)
assert signals.shape == (7352, 9, 128)
assert labels.shape == subjects.shape == (7352,)
assert np.array_equal(np.unique(labels), np.arange(6))
assert len(np.unique(subjects)) == 21
assert np.isfinite(features).all()
assert np.isfinite(signals).all()

print("人工特征形状：", features.shape)
print("时间序列形状：", signals.shape)
print("类别编号：", np.unique(labels))
print("受试者人数：", len(np.unique(subjects)))

# 4. 固定五折划分，与前面的 GroupKFold 实验保持一致
cv = GroupKFold(n_splits=5)
fold_indices = {}

for fold, (train_idx, val_idx) in enumerate(
    cv.split(features, labels, groups=subjects),
    start=1,
):
    train_subjects = np.unique(subjects[train_idx])
    val_subjects = np.unique(subjects[val_idx])

    # 同一受试者不能同时出现在本折的训练部分和验证部分
    assert np.intersect1d(train_subjects, val_subjects).size == 0

    fold_indices[f"train_idx_{fold}"] = train_idx
    fold_indices[f"val_idx_{fold}"] = val_idx

    print(
        f"第 {fold} 折："
        f"训练样本={len(train_idx)}，"
        f"验证样本={len(val_idx)}"
    )
    print("  验证受试者：", val_subjects.tolist())

# 5. 保存数据和划分，供后续三个模型复用
output_path = output_dir / "har_train.npz"

np.savez_compressed(
    output_path,
    features=features,
    signals=signals,
    labels=labels,
    subjects=subjects,
    channel_names=np.array(channel_names),
    **fold_indices,
)

print("\n准备完成，保存到：", output_path)
print("本步骤没有训练模型，也没有读取官方测试集。")