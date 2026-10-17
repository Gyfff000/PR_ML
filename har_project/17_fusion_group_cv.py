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
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def as_tensor(array):
    return torch.from_numpy(array.astype(np.float32))


def prepare_datasets(features, signals, labels, train_idx, val_idx):
    # 人工特征：只用当前训练部分拟合标准化器
    scaler = StandardScaler()
    train_features = as_tensor(
        scaler.fit_transform(features[train_idx])
    )
    val_features = as_tensor(
        scaler.transform(features[val_idx])
    )

    # 时间序列：只用当前训练部分计算每通道统计量
    train_raw = signals[train_idx]
    mean = train_raw.mean(axis=(0, 2), keepdims=True)
    std = train_raw.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)

    train_signals = as_tensor((train_raw - mean) / std)
    val_signals = as_tensor((signals[val_idx] - mean) / std)

    train_data = TensorDataset(
        train_features,
        train_signals,
        torch.from_numpy(labels[train_idx]).long(),
    )
    val_data = TensorDataset(
        val_features,
        val_signals,
        torch.from_numpy(labels[val_idx]).long(),
    )

    return train_data, val_data


def build_model(seed):
    torch.manual_seed(seed)
    return FusionModel()


def make_train_loader(dataset, seed):
    return DataLoader(
        dataset,
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

    for batch_features, batch_signals, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)

        scores = model(batch_features, batch_signals)
        loss = loss_fn(scores, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)

    return total_loss / len(loader.dataset)


def evaluate(model, dataset):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    prediction_batches = []
    label_batches = []

    with torch.no_grad():
        for batch_features, batch_signals, batch_y in loader:
            scores = model(batch_features, batch_signals)
            prediction_batches.append(scores.argmax(dim=1).numpy())
            label_batches.append(batch_y.numpy())

    predictions = np.concatenate(prediction_batches)
    true_labels = np.concatenate(label_batches)

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=np.arange(6),
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1, predictions


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
        features = data["features"]
        signals = data["signals"]
        labels = data["labels"]
        subjects = data["subjects"]

        folds = [
            (data[f"train_idx_{fold}"], data[f"val_idx_{fold}"])
            for fold in range(1, 6)
        ]

    assert len(features) == len(signals) == len(labels) == len(subjects)

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

        # A. 在外层训练部分内部划分受试者
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

        inner_train_data, inner_val_data = prepare_datasets(
            features, signals, labels, train_idx, val_idx
        )
        loader = make_train_loader(inner_train_data, seed)

        model = build_model(seed)
        optimizer = make_optimizer(model)

        best_f1 = -1.0
        best_epoch = 1
        no_improvement = 0

        # B. 用内层验证选择轮数
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_one_epoch(model, loader, optimizer, loss_fn)
            accuracy, macro_f1, _ = evaluate(model, inner_val_data)

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

        # C. 用该折全部训练样本重新计算标准化参数
        full_train_data, outer_val_data = prepare_datasets(
            features,
            signals,
            labels,
            outer_train_idx,
            outer_val_idx,
        )

        # 重新初始化网络，用选出的轮数重训
        model = build_model(seed)
        optimizer = make_optimizer(model)
        full_loader = make_train_loader(full_train_data, seed)

        for _ in range(best_epoch):
            train_one_epoch(model, full_loader, optimizer, loss_fn)

        # D. 重训结束后评估外层验证集
        _, train_f1, _ = evaluate(model, full_train_data)
        val_accuracy, val_f1, predictions = evaluate(
            model, outer_val_data
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

    save_csv(output_dir / "fusion_group_cv.csv", results)
    save_csv(output_dir / "fusion_inner_history.csv", history)

    np.savez_compressed(
        output_dir / "fusion_oof_predictions.npz",
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