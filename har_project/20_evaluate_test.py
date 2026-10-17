from pathlib import Path
import csv
import hashlib
import json

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from fusion_model import FusionModel


def save_report(name, labels, predictions, class_names, output_dir):
    report_text = classification_report(
        labels,
        predictions,
        labels=np.arange(6),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        labels,
        predictions,
        labels=np.arange(6),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "errors": int(np.sum(labels != predictions)),
    }

    (output_dir / f"{name}_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    (output_dir / f"{name}_classification_report.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    matrix = confusion_matrix(
        labels, predictions, labels=np.arange(6)
    )
    with (output_dir / f"{name}_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["真实类别/预测类别", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    print(f"\n{'=' * 45}")
    print("模型：", name)
    print(f"测试准确率：{metrics['accuracy']:.6f}")
    print(f"测试宏平均F1：{metrics['macro_f1']:.6f}")
    print("错误样本数：", metrics["errors"])
    print("\n分类报告：")
    print(report_text)

    return metrics


def main():
    torch.set_num_threads(4)

    root = Path(__file__).resolve().parent
    model_dir = root / "models" / "final"
    data_dir = root / "data" / "UCI HAR Dataset"
    output_dir = root / "outputs" / "final_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载测试前已固定的配置
    config = json.loads(
        (model_dir / "experiment_config.json").read_text(
            encoding="utf-8"
        )
    )
    class_names = config["class_names"]
    channel_names = config["fusion"]["channel_names"]

    source_hash = hashlib.sha256(
        (root / "fusion_model.py").read_bytes()
    ).hexdigest()
    if source_hash != config["fusion_source_sha256"]:
        raise RuntimeError(
            "fusion_model.py 与最终训练时不一致，请先核对网络文件。"
        )

    # 2. 读取官方测试集
    features = np.loadtxt(
        data_dir / "test" / "X_test.txt",
        dtype=np.float64,
    )
    labels = np.loadtxt(
        data_dir / "test" / "y_test.txt",
        dtype=np.int64,
    ) - 1
    subjects = np.loadtxt(
        data_dir / "test" / "subject_test.txt",
        dtype=np.int64,
    )

    signals = np.stack(
        [
            np.loadtxt(
                data_dir / "test" / "Inertial Signals"
                / f"{name}_test.txt",
                dtype=np.float32,
            )
            for name in channel_names
        ],
        axis=1,
    )

    # 只读取训练受试者编号，用于检查划分
    train_subjects = np.loadtxt(
        data_dir / "train" / "subject_train.txt",
        dtype=np.int64,
    )
    overlap = np.intersect1d(train_subjects, subjects)

    assert features.shape == (2947, 561)
    assert signals.shape == (2947, 9, 128)
    assert labels.shape == subjects.shape == (2947,)
    assert len(np.unique(subjects)) == 9
    assert overlap.size == 0
    assert np.array_equal(np.unique(labels), np.arange(6))
    assert np.isfinite(features).all()
    assert np.isfinite(signals).all()

    print("测试样本数：", len(labels))
    print("测试受试者人数：", len(np.unique(subjects)))
    print("训练与测试共有的受试者：", overlap.tolist())

    # 3. 加载逻辑回归，直接预测
    logistic_model = joblib.load(
        model_dir / "logistic_model.joblib"
    )
    assert np.array_equal(logistic_model.classes_, np.arange(6))

    logistic_predictions = logistic_model.predict(features)

    # 4. 使用保存的标准化参数处理融合模型输入
    preprocessing = joblib.load(
        model_dir / "fusion_preprocessing.joblib"
    )
    assert preprocessing["channel_names"] == channel_names

    normalized_features = preprocessing["feature_scaler"].transform(
        features.astype(np.float32)
    )
    normalized_signals = (
        signals - preprocessing["signal_mean"]
    ) / preprocessing["signal_std"]

    dataset = TensorDataset(
        torch.from_numpy(normalized_features.astype(np.float32)),
        torch.from_numpy(normalized_signals.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["fusion"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # 5. 加载融合模型权重，进入预测模式
    checkpoint = torch.load(
        model_dir / "fusion_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["epochs"] == config["fusion"]["epochs"]
    assert checkpoint["seed"] == config["seed"]

    fusion_model = FusionModel()
    fusion_model.load_state_dict(checkpoint["state_dict"])
    fusion_model.eval()

    prediction_batches = []
    with torch.no_grad():
        for batch_features, batch_signals in loader:
            scores = fusion_model(batch_features, batch_signals)
            prediction_batches.append(
                scores.argmax(dim=1).numpy()
            )

    fusion_predictions = np.concatenate(prediction_batches)
    assert fusion_predictions.shape == labels.shape

    # 6. 输出并保存两个模型的评估结果
    results = {
        "primary_model": config["primary_model"],
        "comparison_model": config["comparison_model"],
        "selection_basis": "training cross-validation",
        "test_samples": len(labels),
        "test_subjects": len(np.unique(subjects)),
        "class_names": class_names,
        "models": {},
    }

    results["models"]["logistic"] = save_report(
        "logistic", labels, logistic_predictions,
        class_names, output_dir,
    )
    results["models"]["fusion"] = save_report(
        "fusion", labels, fusion_predictions,
        class_names, output_dir,
    )

    # 记录本次评估对应的模型文件，便于追溯
    results["artifact_sha256"] = {
        filename: hashlib.sha256(
            (model_dir / filename).read_bytes()
        ).hexdigest()
        for filename in [
            "experiment_config.json",
            "logistic_model.joblib",
            "fusion_model.pt",
            "fusion_preprocessing.joblib",
        ]
    }

    (output_dir / "final_test_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 7. 保存逐样本预测，类别编号恢复为官方的 1～6
    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "sample_index",
            "subject_id",
            "true_label",
            "true_activity",
            "logistic_prediction",
            "fusion_prediction",
            "logistic_correct",
            "fusion_correct",
        ])

        for i in range(len(labels)):
            writer.writerow([
                i + 1,
                int(subjects[i]),
                int(labels[i]) + 1,
                class_names[int(labels[i])],
                int(logistic_predictions[i]) + 1,
                int(fusion_predictions[i]) + 1,
                bool(logistic_predictions[i] == labels[i]),
                bool(fusion_predictions[i] == labels[i]),
            ])

    print("\n结果保存到：", output_dir)
    print("本次仅加载模型进行预测，没有重新训练。")


if __name__ == "__main__":
    main()