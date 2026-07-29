import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "3.5_peak_corrected"
MODEL_PATH = PROJECT_ROOT / "outputs" / "final" / "svm_pipeline.joblib"
CONFIG_PATH = PROJECT_ROOT / "outputs" / "final" / "config.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"

STRESS_LABEL = 0
NONSTRESS_LABEL = 1
OVERWRITE = True


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def load_config_feature_names(config_path):
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("feature_names")


def load_subject_feature_file(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int8) if "y" in data else None
    feature_names = [str(name) for name in data["feature_names"]]
    subject = str(np.asarray(data["subject"]).item()) if "subject" in data else npz_path.stem

    finite_rows = np.all(np.isfinite(X), axis=1)
    if "nan_rows" in data:
        finite_rows = finite_rows & ~np.asarray(data["nan_rows"], dtype=bool)

    result = {
        "subject": subject,
        "X": X[finite_rows],
        "feature_names": feature_names,
        "kept_indices": np.flatnonzero(finite_rows),
        "removed_rows": int(np.sum(~finite_rows)),
        "source": str(npz_path),
    }
    if y is not None:
        result["y"] = y[finite_rows]
    return result


def stress_decision_scores(model, X):
    scores = model.decision_function(X)
    svm = model.named_steps["svm"] if hasattr(model, "named_steps") else model
    classes = list(svm.classes_)
    if classes == [STRESS_LABEL, NONSTRESS_LABEL]:
        return -scores
    if classes == [NONSTRESS_LABEL, STRESS_LABEL]:
        return scores
    raise ValueError(f"unexpected SVM classes: {classes}")


def stress_intensity_scores(stress_scores):
    """Convert SVM stress decision scores to user-facing 0-100 stress scores.

    This function is intentionally separated so the scoring rule can be changed
    later without touching classification logic. The current rule is a simple
    sigmoid mapping: larger stress-side decision scores produce higher values.
    """
    stress_scores = np.asarray(stress_scores, dtype=np.float64)
    return 100.0 / (1.0 + np.exp(-stress_scores))


def stress_auc_score(y_true, stress_scores):
    stress_binary = (y_true == STRESS_LABEL).astype(np.int8)
    if len(np.unique(stress_binary)) < 2:
        return np.nan
    return float(roc_auc_score(stress_binary, stress_scores))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT_DIR.exists() and not OVERWRITE:
        existing = list(OUTPUT_DIR.glob("*"))
        if existing:
            raise FileExistsError(f"output directory already contains files: {OUTPUT_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model file not found: {MODEL_PATH}")

    npz_paths = sorted(FEATURE_DIR.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no feature NPZ files found in {FEATURE_DIR}")

    model = joblib.load(MODEL_PATH)
    expected_feature_names = load_config_feature_names(CONFIG_PATH)

    subject_rows = []
    prediction_rows = []
    all_y_true = []
    all_y_pred = []
    all_scores = []

    print(f"feature_dir: {FEATURE_DIR}")
    print(f"model_path: {MODEL_PATH}")
    print(f"output_dir: {OUTPUT_DIR}")
    print(f"subjects: {len(npz_paths)}")
    print()

    for npz_path in npz_paths:
        subject_data = load_subject_feature_file(npz_path)
        if expected_feature_names is not None and subject_data["feature_names"] != expected_feature_names:
            raise ValueError(
                f"feature name mismatch in {npz_path}: "
                f"{subject_data['feature_names']} != {expected_feature_names}"
            )

        X = subject_data["X"]
        y_pred = model.predict(X)
        stress_scores = stress_decision_scores(model, X)
        stress_scores_0_100 = stress_intensity_scores(stress_scores)
        has_label = "y" in subject_data

        row = {
            "subject": subject_data["subject"],
            "source": subject_data["source"],
            "samples": int(len(X)),
            "removed_rows": subject_data["removed_rows"],
            "pred_stress": int(np.sum(y_pred == STRESS_LABEL)),
            "pred_nonstress": int(np.sum(y_pred == NONSTRESS_LABEL)),
        }

        if has_label:
            y_true = subject_data["y"]
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_scores.append(stress_scores)

            accuracy = accuracy_score(y_true, y_pred)
            f1_stress = f1_score(
                y_true,
                y_pred,
                pos_label=STRESS_LABEL,
                labels=[STRESS_LABEL, NONSTRESS_LABEL],
                zero_division=0,
            )
            auc_stress = stress_auc_score(y_true, stress_scores)
            cm = confusion_matrix(y_true, y_pred, labels=[STRESS_LABEL, NONSTRESS_LABEL])
            row.update(
                {
                    "true_stress": int(np.sum(y_true == STRESS_LABEL)),
                    "true_nonstress": int(np.sum(y_true == NONSTRESS_LABEL)),
                    "accuracy": float(accuracy),
                    "f1_stress": float(f1_stress),
                    "auc_stress": float(auc_stress),
                    "tp_stress": int(cm[0, 0]),
                    "fn_stress": int(cm[0, 1]),
                    "fp_stress": int(cm[1, 0]),
                    "tn_stress": int(cm[1, 1]),
                }
            )

        subject_rows.append(row)

        for local_index, original_index in enumerate(subject_data["kept_indices"]):
            pred = int(y_pred[local_index])
            pred_name = "stress" if pred == STRESS_LABEL else "non-stress"
            output_row = {
                "subject": subject_data["subject"],
                "sample_index": int(original_index),
                "y_pred": pred,
                "prediction": pred_name,
                "stress_score": float(stress_scores[local_index]),
                "stress_score_0_100": float(stress_scores_0_100[local_index]),
            }
            if has_label:
                truth = int(subject_data["y"][local_index])
                output_row["y_true"] = truth
                output_row["correct"] = int(truth == pred)
            prediction_rows.append(output_row)

        print(f"[ok] {subject_data['subject']}")
        print(f"  samples: {len(X)}")
        print(f"  predicted stress/non-stress: {row['pred_stress']}/{row['pred_nonstress']}")
        if has_label:
            print(f"  accuracy: {row['accuracy']:.4f}")
            print(f"  f1_stress: {row['f1_stress']:.4f}")
        print()

    summary_rows = []
    if all_y_true:
        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)
        scores_all = np.concatenate(all_scores)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[STRESS_LABEL, NONSTRESS_LABEL])
        summary_rows.append(
            {
                "samples": int(len(y_true_all)),
                "true_stress": int(np.sum(y_true_all == STRESS_LABEL)),
                "true_nonstress": int(np.sum(y_true_all == NONSTRESS_LABEL)),
                "pred_stress": int(np.sum(y_pred_all == STRESS_LABEL)),
                "pred_nonstress": int(np.sum(y_pred_all == NONSTRESS_LABEL)),
                "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
                "f1_stress": float(
                    f1_score(
                        y_true_all,
                        y_pred_all,
                        pos_label=STRESS_LABEL,
                        labels=[STRESS_LABEL, NONSTRESS_LABEL],
                        zero_division=0,
                    )
                ),
                "auc_stress": stress_auc_score(y_true_all, scores_all),
                "tp_stress": int(cm[0, 0]),
                "fn_stress": int(cm[0, 1]),
                "fp_stress": int(cm[1, 0]),
                "tn_stress": int(cm[1, 1]),
            }
        )

    subject_fieldnames = [
        "subject",
        "source",
        "samples",
        "removed_rows",
        "true_stress",
        "true_nonstress",
        "pred_stress",
        "pred_nonstress",
        "accuracy",
        "f1_stress",
        "auc_stress",
        "tp_stress",
        "fn_stress",
        "fp_stress",
        "tn_stress",
    ]
    prediction_fieldnames = [
        "subject",
        "sample_index",
        "y_true",
        "y_pred",
        "prediction",
        "stress_score",
        "stress_score_0_100",
        "correct",
    ]

    write_csv(OUTPUT_DIR / "subject_predictions_summary.csv", subject_rows, subject_fieldnames)
    write_csv(OUTPUT_DIR / "predictions.csv", prediction_rows, prediction_fieldnames)
    if summary_rows:
        write_csv(
            OUTPUT_DIR / "summary.csv",
            summary_rows,
            [
                "samples",
                "true_stress",
                "true_nonstress",
                "pred_stress",
                "pred_nonstress",
                "accuracy",
                "f1_stress",
                "auc_stress",
                "tp_stress",
                "fn_stress",
                "fp_stress",
                "tn_stress",
            ],
        )

    print(f"results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
