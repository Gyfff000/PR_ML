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

    # 1. 读取特征和第一折训练样本的下标
    with np.load(root / "data" / "processed" / "har_train.npz") as data:
        features = data["features"]
        labels = data["labels"]
        subjects = data["subjects"]
        outer_train_idx = data["train_idx_1"]
        outer_val_idx = data["val_idx_1"]

    # 2. 从第一折训练部分中，再划出内层验证受试者
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

    print("运行设备：CPU")
    print("内层训练样本数：", len(train_idx))
    print("内层验证样本数：", len(val_idx))
    print("内层训练受试者：", np.unique(subjects[train_idx]).tolist())
    print("内层验证受试者：", np.unique(subjects[val_idx]).tolist())

    # 3. 标准化参数只从内层训练数据中计算
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train_idx])
    x_val = scaler.transform(features[val_idx])

    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_val = torch.from_numpy(x_val.astype(np.float32))
    y_train = torch.from_numpy(labels[train_idx]).long()
    y_val = labels[val_idx]

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )

    # 4. 网络结构：561 → 128 → 64 → 6
    model = nn.Sequential(
        nn.Linear(561, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 6),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    print("\n网络结构：")
    print(model)
    print("可训练参数量：", sum(p.numel() for p in model.parameters()))
    print("\n开始 10 轮试训练……")

    records = []
    total_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        # 5. 训练：计算损失并更新网络权重
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)

            scores = model(batch_x)
            loss = loss_fn(scores, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_idx)

        # 6. 验证：只预测，不更新网络权重
        model.eval()
        with torch.no_grad():
            predictions = model(x_val).argmax(dim=1).numpy()

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

    # 7. 保存每轮日志，后续可以用于绘制训练曲线
    log_path = output_dir / "mlp_pilot_history.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    elapsed = time.perf_counter() - total_start
    print(f"\n训练及验证总耗时：{elapsed:.1f} 秒")
    print("日志已保存：", log_path)
    print("本次仅为内层试训练，不是正式五折评估。")


if __name__ == "__main__":
    main()