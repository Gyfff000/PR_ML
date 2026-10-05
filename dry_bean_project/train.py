"""Dry Bean classification: run prepare, baseline, tune, evaluate one step at a time."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                    cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
SEED = 42
def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require(path, step):
    if not path.exists():
        raise SystemExit(f"Missing {path.name}. Run --step {step} first.")


def read_data():
    raw = pd.read_excel(ROOT / "data/raw/Dry_Bean_Dataset.xlsx")
    if raw.shape != (13611, 17) or "Class" not in raw:
        raise ValueError(f"Unexpected dataset shape/columns: {raw.shape}")
    if raw.isna().any().any():
        raise ValueError("Missing values found. Inspect before continuing.")
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in raw.columns if c != "Class"):
        raise ValueError("Non-numeric feature found.")
    if not np.isfinite(raw.drop(columns="Class").to_numpy()).all():
        raise ValueError("Non-finite feature found.")
    # Same feature vector with conflicting labels requires manual investigation.
    feature_columns = raw.columns.drop("Class").tolist()
    if raw.groupby(feature_columns)["Class"].nunique().max() > 1:
        raise ValueError("Identical features have conflicting labels.")
    clean = raw.drop_duplicates().reset_index(drop=True)
    return raw, clean


def load_split():
    require(OUT / "split.json", "prepare")
    _, data = read_data()
    split = json.loads((OUT / "split.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    if digest != split["clean_sha256"]:
        raise ValueError("Data changed since prepare. Start a new experiment.")
    train, test = data.iloc[split["train_indices"]], data.iloc[split["test_indices"]]
    return train.drop(columns="Class"), test.drop(columns="Class"), train.Class, test.Class


def cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def candidates():
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=4000)),
        "forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
        "svm": make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", kernel="rbf")),
    }


def prepare():
    # Prevent changes to split metadata after training starts.
    if (OUT / "baseline_cv.csv").exists():
        raise SystemExit("Preparation already complete and training exists. Keep the fixed split.")
    raw, data = read_data()
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=.2,
                                          stratify=data.Class, random_state=SEED)
    save_json(OUT / "split.json", {
        "seed": SEED, "test_size": .2,
        "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(),
        "clean_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
    })
    counts = pd.DataFrame({"all": data.Class.value_counts(),
                           "train": data.iloc[train_idx].Class.value_counts(),
                           "test": data.iloc[test_idx].Class.value_counts()}).sort_index()
    counts.to_csv(OUT / "class_counts.csv", index_label="class")
    stats = {"raw_rows": len(raw), "duplicate_rows_removed": len(raw)-len(data),
             "clean_rows": len(data), "features": data.columns.drop("Class").tolist(),
             "classes": sorted(data.Class.unique().tolist()), "missing_values": 0,
             "train_rows": len(train_idx), "test_rows": len(test_idx),
             "python": platform.python_version(), "sklearn": sklearn.__version__}
    save_json(OUT / "data_summary.json", stats)
    data.describe().to_csv(OUT / "feature_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    counts["all"].plot.bar(ax=ax, color="#3676a8")
    ax.set(xlabel="Bean variety", ylabel="Samples", title="Class distribution after deduplication")
    fig.tight_layout(); fig.savefig(OUT / "class_distribution.png", dpi=180); plt.close(fig)
    print(json.dumps(stats, indent=2))
    print(counts.to_string())
    print("\nReady. No model has been trained. Next: --step baseline")


def baseline():
    if (OUT / "selection.json").exists():
        raise SystemExit("Model selection already complete; baseline is frozen.")
    X, _, y, _ = load_split()
    rows = []
    for name, model in candidates().items():
        print(f"\n{name}: five-fold training-set CV ...", flush=True)
        start = time.perf_counter()
        scores = cross_validate(model, X, y, cv=cv(),
                                scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
                                n_jobs=2, return_train_score=True, error_score="raise")
        fitted = clone(model).fit(X, y)
        joblib.dump(fitted, MODELS / f"{name}_baseline.joblib")
        row = {"model": name, "cv_accuracy": float(scores["test_accuracy"].mean()),
               "cv_macro_f1": float(scores["test_macro_f1"].mean()),
               "cv_f1_std": float(scores["test_macro_f1"].std(ddof=0)),
               "train_macro_f1": float(scores["train_macro_f1"].mean()),
               "seconds": time.perf_counter()-start}
        rows.append(row)
        print(pd.Series(row).to_string(), flush=True)
        pd.DataFrame(rows).to_csv(OUT / "baseline_cv_partial.csv", index=False)
    table = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    table.to_csv(OUT / "baseline_cv.csv", index=False)
    print("\n", table.to_string(index=False))
    print("\nThese are cross-validation scores; the held-out test set is still untouched.")


def tune():
    require(OUT / "baseline_cv.csv", "baseline")
    if (OUT / "test_metrics.csv").exists():
        raise SystemExit("Test results already revealed. Do not retune using the same test set.")
    X, _, y, _ = load_split()
    grids = {
        "logistic": {"logisticregression__C": [.1, 1, 10]},
        "forest": {"max_depth": [None, 20], "min_samples_leaf": [1, 3]},
        "svm": {"svc__C": [1, 10, 100], "svc__gamma": ["scale", .01]},
    }
    rows = []
    for name, params in grids.items():
        print(f"\nTuning {name} using training CV only ...", flush=True)
        search = GridSearchCV(candidates()[name], params, scoring="f1_macro", cv=cv(),
                              n_jobs=2, return_train_score=True, error_score="raise", verbose=1)
        start = time.perf_counter(); search.fit(X, y)
        joblib.dump(search.best_estimator_, MODELS / f"{name}_tuned.joblib")
        pd.DataFrame(search.cv_results_).to_csv(OUT / f"{name}_grid.csv", index=False)
        row = {"model": name, "cv_macro_f1": float(search.best_score_),
               "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
               "parameters": json.dumps(search.best_params_),
               "seconds": time.perf_counter()-start}
        rows.append(row); print(row, flush=True)
    table = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    table.to_csv(OUT / "tuned_cv.csv", index=False)
    best = table.iloc[0]
    save_json(OUT / "selection.json", {"model": best["model"], "selected_by": "training_cv_macro_f1",
                                       "cv_macro_f1": float(best["cv_macro_f1"]),
                                       "parameters": json.loads(best["parameters"])})
    print("\nSelected before test evaluation:", best["model"])


def evaluate():
    require(OUT / "selection.json", "tune")
    if (OUT / "test_metrics.csv").exists():
        print(pd.read_csv(OUT / "test_metrics.csv").to_string(index=False))
        print("Existing test results displayed. Models were not refitted.")
        return
    _, X, _, y = load_split()
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    rows = []
    for name in candidates():
        for variant in (["baseline"] if name == "majority" else ["baseline", "tuned"]):
            model = joblib.load(MODELS / f"{name}_{variant}.joblib")
            start = time.perf_counter(); pred = model.predict(X)
            rows.append({"model": name, "variant": variant,
                         "test_accuracy": accuracy_score(y, pred),
                         "test_macro_f1": f1_score(y, pred, average="macro"),
                         "test_balanced_accuracy": balanced_accuracy_score(y, pred),
                         "predict_seconds": time.perf_counter()-start})
            if name == selection["model"] and variant == "tuned":
                selected_model, selected_pred = model, pred
    classes = selected_model.classes_
    detail = pd.DataFrame(classification_report(y, selected_pred, output_dict=True, zero_division=0)).T
    detail.to_csv(OUT / "selected_classification_report.csv", index_label="class")
    predictions = X.copy(); predictions["true"] = y; predictions["predicted"] = selected_pred
    predictions.to_csv(OUT / "test_predictions.csv", index_label="clean_row_id")
    errors = predictions.loc[predictions["true"] != predictions["predicted"]]
    errors.to_csv(OUT / "misclassified_samples.csv", index_label="clean_row_id")
    errors.groupby(["true", "predicted"]).size().sort_values(ascending=False).to_csv(OUT / "error_pairs.csv", header=["count"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(y, selected_pred, labels=classes, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Held-out test confusion matrix: {selection['model']}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=180); plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "test_metrics.csv", index=False)
    save_json(OUT / "evaluation_record.json", {"selected_model": selection["model"],
                                               "selection_frozen_before_test": True,
                                               "test_rows": len(y), "errors": len(errors)})
    print(table.to_string(index=False))
    print("\nSelection stays fixed by training CV, even if another model has a higher test score.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=["prepare", "baseline", "tune", "evaluate"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    globals()[args.step]()


if __name__ == "__main__":
    main()
