from pathlib import Path
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from fusion_model import FusionModel


SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def as_tensor(array):
    return torch.from_numpy(array.astype(np.float32))


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 同时读取人工特征与时间序列
    with np.load(root / "data" / "processed" / "har_train.npz") as data:
        features = data["features"]
        signals = data["signals"]
        labels = data["labels"]
        subjects = data["subjects"]
        outer_train_idx = data["train_idx_1"]
        outer_val_idx = data["val_idx_1"]

    # 2. 使用与前面两个模型相同的内层划分
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED,
    )
    inner_train, inner_val = next(
        splitter.split(
            features[outer_train_idx],
            labels[outer_train_idx],
            groups=subjects[outer_train_idx],
        )
    )

    train_idx = outer_train_idx[inner_train]
    val_idx = outer_train_idx[inner_val]

    assert np.intersect1d(
        subjects[train_idx], subjects[val_idx]
    ).size == 0
    assert np.intersect1d(
        subjects[outer_train_idx], subjects[outer_val_idx]
    ).size == 0

    # 3. 人工特征：只用内层训练数据拟合标准化器
    feature_scaler = StandardScaler()
    train_features = as_tensor(
        feature_scaler.fit_transform(features[train_idx])
    )
    val_features = as_tensor(
        feature_scaler.transform(features[val_idx])
    )

    # 4. 时间序列：只用内层训练数据计算每通道统计量
    raw_train_signals = signals[train_idx]
    signal_mean = raw_train_signals.mean(
        axis=(0, 2), keepdims=True
    )
    signal_std = raw_train_signals.std(
        axis=(0, 2), keepdims=True
    )
    signal_std = np.maximum(signal_std, 1e-6)

    train_signals = as_tensor(
        (raw_train_signals - signal_mean) / signal_std
    )
    val_signals = as_tensor(
        (signals[val_idx] - signal_mean) / signal_std
    )

    # 两种输入使用同一组下标，保证样本对应
    train_loader = DataLoader(
        TensorDataset(
            train_features,
            train_signals,
            torch.from_numpy(labels[train_idx]).long(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_signals),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = FusionModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    print("运行设备：CPU")
    print("训练人工特征形状：", tuple(train_features.shape))
    print("训练时间序列形状：", tuple(train_signals.shape))
    print("内层验证受试者：", np.unique(subjects[val_idx]).tolist())
    print("可训练参数量：", sum(p.numel() for p in model.parameters()))
    print("\n开始融合模型十轮试训练……")

    records = []
    total_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        # 5. 联合训练两条分支
        model.train()
        total_loss = 0.0

        for batch_features, batch_signals, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)

            scores = model(batch_features, batch_signals)
            loss = loss_fn(scores, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_idx)

        # 6. 验证
        model.eval()
        prediction_batches = []

        with torch.no_grad():
            for batch_features, batch_signals in val_loader:
                scores = model(batch_features, batch_signals)
                predictions = scores.argmax(dim=1).numpy()
                prediction_batches.append(predictions)

        predictions = np.concatenate(prediction_batches)
        val_accuracy = accuracy_score(labels[val_idx], predictions)
        val_f1 = f1_score(
            labels[val_idx],
            predictions,
            labels=np.arange(6),
            average="macro",
            zero_division=0,
        )
        seconds = time.perf_counter() - epoch_start

        records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "inner_val_accuracy": val_accuracy,
            "inner_val_macro_f1": val_f1,
            "seconds": seconds,
        })

        print(
            f"第 {epoch:02d}/{EPOCHS} 轮："
            f"训练损失={train_loss:.4f}，"
            f"内层验证准确率={val_accuracy:.4f}，"
            f"内层验证F1={val_f1:.4f}，"
            f"耗时={seconds:.2f}秒",
            flush=True,
        )

    log_path = output_dir / "fusion_pilot_history.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    elapsed = time.perf_counter() - total_start
    print(f"\n训练及验证总耗时：{elapsed:.1f}秒")
    print("日志已保存：", log_path)
    print("本次仅为内层试训练，未使用外层验证集和官方测试集。")


if __name__ == "__main__":
    main()