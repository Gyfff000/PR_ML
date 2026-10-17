from pathlib import Path
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit


SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)

    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 这次读取九通道信号，不使用 561 个人工特征
    with np.load(root / "data" / "processed" / "har_train.npz") as data:
        signals = data["signals"]
        labels = data["labels"]
        subjects = data["subjects"]
        outer_train_idx = data["train_idx_1"]
        outer_val_idx = data["val_idx_1"]

    # 2. 使用与 MLP 试训练相同的内层受试者划分
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED,
    )
    inner_train, inner_val = next(
        splitter.split(
            signals[outer_train_idx],
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

    # 3. 每个通道单独标准化
    # 均值和标准差只从内层训练样本中计算
    train_signals = signals[train_idx]
    mean = train_signals.mean(axis=(0, 2), keepdims=True)
    std = train_signals.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)

    x_train = torch.from_numpy(
        ((train_signals - mean) / std).astype(np.float32)
    )
    x_val = torch.from_numpy(
        ((signals[val_idx] - mean) / std).astype(np.float32)
    )
    y_train = torch.from_numpy(labels[train_idx]).long()
    y_val = labels[val_idx]

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        TensorDataset(x_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 4. 三层一维卷积网络
    model = nn.Sequential(
        nn.Conv1d(9, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2),

        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(128, 6),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    print("运行设备：CPU")
    print("内层训练输入形状：", tuple(x_train.shape))
    print("内层验证输入形状：", tuple(x_val.shape))
    print("内层验证受试者：", np.unique(subjects[val_idx]).tolist())
    print("可训练参数量：", sum(p.numel() for p in model.parameters()))
    print("\n开始 CNN 十轮试训练……")

    records = []
    total_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        # 5. 训练
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_idx)

        # 6. 分批验证，保持原始验证样本顺序
        model.eval()
        prediction_batches = []

        with torch.no_grad():
            for (batch_x,) in val_loader:
                predictions = model(batch_x).argmax(dim=1)
                prediction_batches.append(predictions.numpy())

        predictions = np.concatenate(prediction_batches)
        val_accuracy = accuracy_score(y_val, predictions)
        val_f1 = f1_score(
            y_val,
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

    log_path = output_dir / "cnn_pilot_history.csv"
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