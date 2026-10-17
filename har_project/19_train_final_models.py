from pathlib import Path
import csv
import hashlib
import json
import time

import joblib
import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fusion_model import FusionModel


SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def main():
    np.random.seed(SEED)
    torch.set_num_threads(4)

    root = Path(__file__).resolve().parent
    model_dir = root / "models" / "final"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. 根据已有训练集实验记录，确定融合模型训练轮数
    cv_paths = [
        root / "outputs" / "fusion_group_cv.csv",
        root / "outputs" / "fusion_seed_2024" / "fusion_group_cv.csv",
        root / "outputs" / "fusion_seed_2025" / "fusion_group_cv.csv",
    ]

    selected_epochs = []

    for path in cv_paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))

        assert len(rows) == 5
        assert {int(row["fold"]) for row in rows} == set(range(1, 6))

        selected_epochs.extend(
            int(row["selected_epochs"]) for row in rows
        )

    assert all(1 <= epoch <= 100 for epoch in selected_epochs)
    fusion_epochs = int(np.median(selected_epochs))

    # 2. 只读取官方训练数据
    with np.load(root / "data" / "processed" / "har_train.npz") as data:
        features = data["features"]
        signals = data["signals"]
        labels = data["labels"]
        subjects = data["subjects"]
        channel_names = data["channel_names"].tolist()

    assert features.shape == (7352, 561)
    assert signals.shape == (7352, 9, 128)
    assert labels.shape == (7352,)
    assert len(np.unique(subjects)) == 21
    assert np.array_equal(np.unique(labels), np.arange(6))
    assert np.isfinite(features).all()
    assert np.isfinite(signals).all()

    # 逻辑回归使用原始文本中的双精度特征
    logistic_features = np.loadtxt(
        root / "data" / "UCI HAR Dataset" / "train" / "X_train.txt",
        dtype=np.float64,
    )
    assert np.allclose(logistic_features, features)

    # 3. 保存测试前固定的实验配置
    config = {
        "primary_model": "logistic",
        "comparison_model": "fusion",
        "seed": SEED,
        "training_samples": len(labels),
        "training_subjects": len(np.unique(subjects)),
        "label_encoding": "official label minus 1",
        "class_names": ["步行", "上楼", "下楼", "坐", "站立", "躺卧"],
        "logistic": {
            "C": 0.1,
            "max_iter": 3000,
            "pca": False,
        },
        "fusion": {
            "epochs": fusion_epochs,
            "epoch_rule": "median of 15 inner-selected epoch counts",
            "source_epoch_counts": selected_epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "channel_names": channel_names,
        },
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": str(torch.__version__),
        },
        "fusion_source_sha256": hashlib.sha256(
            (root / "fusion_model.py").read_bytes()
        ).hexdigest(),
    }

    config_path = model_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("最终主模型：逻辑回归，C=0.1，不使用 PCA")
    print("融合模型最终训练轮数：", fusion_epochs)
    print("融合模型随机种子：", SEED)
    print("训练样本数：", len(labels))

    # 4. 在全部官方训练数据上训练逻辑回归
    print("\n开始训练最终逻辑回归……", flush=True)
    logistic_start = time.perf_counter()

    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3000),
    )
    logistic_model.fit(logistic_features, labels)

    joblib.dump(
        logistic_model,
        model_dir / "logistic_model.joblib",
    )

    print(
        f"逻辑回归训练完成，"
        f"耗时 {time.perf_counter() - logistic_start:.1f} 秒"
    )

    # 5. 为融合模型拟合完整训练集的标准化参数
    feature_scaler = StandardScaler()
    normalized_features = feature_scaler.fit_transform(features)

    signal_mean = signals.mean(axis=(0, 2), keepdims=True)
    signal_std = signals.std(axis=(0, 2), keepdims=True)
    signal_std = np.maximum(signal_std, 1e-6)
    normalized_signals = (signals - signal_mean) / signal_std

    dataset = TensorDataset(
        torch.from_numpy(normalized_features.astype(np.float32)),
        torch.from_numpy(normalized_signals.astype(np.float32)),
        torch.from_numpy(labels).long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )

    # 6. 从头训练最终融合模型，固定轮数
    torch.manual_seed(SEED)
    model = FusionModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("\n开始训练最终融合模型……", flush=True)
    history = []

    for epoch in range(1, fusion_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0

        for batch_features, batch_signals, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch_features, batch_signals)
            loss = loss_fn(scores, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(dataset)
        seconds = time.perf_counter() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": seconds,
        })

        print(
            f"第 {epoch:02d}/{fusion_epochs} 轮："
            f"训练损失={train_loss:.4f}，"
            f"耗时={seconds:.2f}秒",
            flush=True,
        )

    # 7. 保存网络权重和预测时必须使用的预处理参数
    model.eval()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": SEED,
            "epochs": fusion_epochs,
        },
        model_dir / "fusion_model.pt",
    )

    joblib.dump(
        {
            "feature_scaler": feature_scaler,
            "signal_mean": signal_mean,
            "signal_std": signal_std,
            "channel_names": channel_names,
        },
        model_dir / "fusion_preprocessing.joblib",
    )

    with (model_dir / "fusion_train_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print("\n最终模型训练并保存完成：", model_dir)
    print("本步骤没有读取官方测试集。")


if __name__ == "__main__":
    main()