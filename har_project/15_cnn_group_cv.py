from pathlib import Path
import csv
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
#from sklearn.preprocessing import StandardScaler


SEED = 42
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def build_model(seed):
    torch.manual_seed(seed)

    return nn.Sequential(
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


def make_loader(x, y, seed):
    return DataLoader(
        TensorDataset(x, torch.from_numpy(y).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)

    return total_loss / len(loader.dataset)


def evaluate(model, x, y):
    model.eval()

    loader = DataLoader(
        TensorDataset(x),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    prediction_batches = []

    with torch.no_grad():
        for (batch_x,) in loader:
            predictions = model(batch_x).argmax(dim=1)
            prediction_batches.append(predictions.numpy())

    predictions = np.concatenate(prediction_batches)

    accuracy = accuracy_score(y, predictions)
    macro_f1 = f1_score(
        y,
        predictions,
        labels=np.arange(6),
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1, predictions


def as_tensor(array):
    return torch.from_numpy(array.astype(np.float32))

def standardize_signals(train_signals, val_signals):
    # 每个通道分别计算均值和标准差
    # axis=(0, 2)：沿样本维度和时间维度汇总
    mean = train_signals.mean(axis=(0, 2), keepdims=True)
    std = train_signals.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)

    # 验证数据使用训练数据的统计量
    x_train = as_tensor((train_signals - mean) / std)
    x_val = as_tensor((val_signals - mean) / std)

    return x_train, x_val


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    np.random.seed(SEED)
    torch.set_num_threads(4)

    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(root / "data" / "processed" / "har_train.npz") as data:
        signals = data["signals"]
        labels = data["labels"]
        subjects = data["subjects"]

        folds = [
            (data[f"train_idx_{fold}"], data[f"val_idx_{fold}"])
            for fold in range(1, 6)
        ]

    results = []
    history = []
    oof_predictions = np.full(len(labels), -1, dtype=np.int64)
    oof_folds = np.zeros(len(labels), dtype=np.int64)
    loss_fn = nn.CrossEntropyLoss()

    for fold, (outer_train_idx, outer_val_idx) in enumerate(folds, 1):
        fold_start = time.perf_counter()
        seed = SEED + fold - 1

        print(f"\n{'=' * 45}")
        print(f"第 {fold}/5 折")

        assert np.intersect1d(
            subjects[outer_train_idx],
            subjects[outer_val_idx],
        ).size == 0

        # A. 只在外层训练部分中划分内层训练、验证数据
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

                # 内层标准化只能使用内层训练数据
        x_train, x_val = standardize_signals(
            signals[train_idx],
            signals[val_idx],
        )
        loader = make_loader(x_train, labels[train_idx], seed)

        model = build_model(seed)
        optimizer = make_optimizer(model)

        best_f1 = -1.0
        best_epoch = 1
        no_improvement = 0

        # B. 用内层验证 F1 选择训练轮数
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_one_epoch(model, loader, optimizer, loss_fn)
            accuracy, macro_f1, _ = evaluate(
                model, x_val, labels[val_idx]
            )

            history.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": loss,
                "inner_val_accuracy": accuracy,
                "inner_val_macro_f1": macro_f1,
            })

            improved = macro_f1 > best_f1
            if improved:
                best_f1 = macro_f1
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1

            if improved or epoch % 5 == 0:
                print(
                    f"内层第 {epoch:03d} 轮："
                    f"损失={loss:.4f}，"
                    f"验证F1={macro_f1:.4f}",
                    flush=True,
                )

            if no_improvement >= PATIENCE:
                print(f"连续 {PATIENCE} 轮未提升，停止内层训练。")
                break

        print(f"选择训练轮数：{best_epoch}")

               # C. 重新计算标准化参数，使用该折全部训练样本
        x_full_train, x_outer_val = standardize_signals(
            signals[outer_train_idx],
            signals[outer_val_idx],
        )

        # 重新初始化网络，不沿用内层训练得到的权重
        model = build_model(seed)
        optimizer = make_optimizer(model)
        full_loader = make_loader(
            x_full_train, labels[outer_train_idx], seed
        )

        for _ in range(best_epoch):
            train_one_epoch(model, full_loader, optimizer, loss_fn)

        # D. 重训结束后，评估外层验证集
        _, train_f1, _ = evaluate(
            model, x_full_train, labels[outer_train_idx]
        )
        val_accuracy, val_f1, predictions = evaluate(
            model, x_outer_val, labels[outer_val_idx]
        )

        oof_predictions[outer_val_idx] = predictions
        oof_folds[outer_val_idx] = fold
        seconds = time.perf_counter() - fold_start

        results.append({
            "fold": fold,
            "seed": seed,
            "selected_epochs": best_epoch,
            "best_inner_macro_f1": best_f1,
            "train_macro_f1": train_f1,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1,
            "seconds": seconds,
        })

        print(
            f"第 {fold} 折正式结果："
            f"训练F1={train_f1:.4f}，"
            f"外层验证准确率={val_accuracy:.4f}，"
            f"外层验证F1={val_f1:.4f}，"
            f"耗时={seconds:.1f}秒"
        )

    assert np.all(oof_predictions >= 0)
    assert np.all(oof_folds > 0)

    save_csv(output_dir / "cnn_group_cv.csv", results)
    save_csv(output_dir / "cnn_inner_history.csv", history)

    # 保存每个样本在其外层验证折中的预测，便于后续分析
    np.savez_compressed(
        output_dir / "cnn_oof_predictions.npz",
        predictions=oof_predictions,
        labels=labels,
        subjects=subjects,
        folds=oof_folds,
    )

    accuracies = np.array([row["val_accuracy"] for row in results])
    f1_scores = np.array([row["val_macro_f1"] for row in results])

    print("\n五折平均准确率：", accuracies.mean())
    print("五折平均宏平均F1：", f1_scores.mean())
    print("宏平均F1折间标准差：", f1_scores.std(ddof=0))
    print("各折选择的训练轮数：",
          [row["selected_epochs"] for row in results])
    print("五折总耗时：",
          round(sum(row["seconds"] for row in results), 1), "秒")
    print("结果已保存到：", output_dir)
    print("本实验没有读取官方测试集。")


if __name__ == "__main__":
    main()