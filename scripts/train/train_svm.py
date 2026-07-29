import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "3.5_peak_corrected"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "3.5_peak_corrected"

STRESS_LABEL = 0
NONSTRESS_LABEL = 1
RANDOM_SEED = 42


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def load_subject_feature_file(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int8)
    feature_names = [str(name) for name in data["feature_names"]]
    subject = str(np.asarray(data["subject"]).item()) if "subject" in data else npz_path.stem

    finite_rows = np.all(np.isfinite(X), axis=1)
    if "nan_rows" in data:
        finite_rows = finite_rows & ~np.asarray(data["nan_rows"], dtype=bool)

    X = X[finite_rows]
    y = y[finite_rows]

    return {
        "subject": subject,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "removed_rows": int(np.sum(~finite_rows)),
        "source": str(npz_path),
    }


def build_model(class_weight, seed):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def count_label(y, label):
    return int(np.sum(y == label))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stress_decision_scores(model, X):
    scores = model.decision_function(X)
    classes = list(model.named_steps["svm"].classes_)
    if classes == [STRESS_LABEL, NONSTRESS_LABEL]:
        return -scores
    if classes == [NONSTRESS_LABEL, STRESS_LABEL]:
        return scores
    raise ValueError(f"unexpected SVM classes: {classes}")


def stress_auc_score(y_true, stress_scores):
    stress_binary = (y_true == STRESS_LABEL).astype(np.int8)
    if len(np.unique(stress_binary)) < 2:
        return np.nan
    return float(roc_auc_score(stress_binary, stress_scores))


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate SVM with Leave-One-Subject-Out validation."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_DIR,
        help="Directory containing subject feature NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write LOSO SVM results.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="none",
        help="SVM class weight. Default: none. Use balanced to compensate class imbalance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducible runs. Default: {RANDOM_SEED}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing result files.",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.output_dir.exists() and not args.overwrite:
        existing = list(args.output_dir.glob("*"))
        if existing:
            raise FileExistsError(
                f"output directory already contains files: {args.output_dir} "
                "(use --overwrite to replace them)"
            )

    npz_paths = sorted(args.feature_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no feature NPZ files found in {args.feature_dir}")

    subjects = [load_subject_feature_file(path) for path in npz_paths]
    feature_names = subjects[0]["feature_names"]
    for subject_data in subjects:
        if subject_data["feature_names"] != feature_names:
            raise ValueError(f"feature name mismatch in {subject_data['source']}")
        if subject_data["X"].shape[1] != len(feature_names):
            raise ValueError(f"feature shape mismatch in {subject_data['source']}")

    class_weight = "balanced" if args.class_weight == "balanced" else None

    fold_rows = []
    prediction_rows = []

    print(f"feature_dir: {args.feature_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"subjects: {len(subjects)}")
    print(f"features: {', '.join(feature_names)}")
    print(
        "model: StandardScaler + "
        f"SVC(kernel=rbf, C=1.0, gamma=scale, class_weight={class_weight}, "
        f"random_state={args.seed})"
    )
    print("metrics: accuracy, f1_score(pos_label=stress=0), auc_stress")
    print(f"seed: {args.seed}")
    print()

    for test_index, test_subject_data in enumerate(subjects):
        train_subjects = [
            subject_data for index, subject_data in enumerate(subjects) if index != test_index
        ]

        X_train = np.vstack([subject_data["X"] for subject_data in train_subjects])
        y_train = np.concatenate([subject_data["y"] for subject_data in train_subjects])
        X_test = test_subject_data["X"]
        y_test = test_subject_data["y"]

        model = build_model(class_weight, args.seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        stress_scores = stress_decision_scores(model, X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_stress = f1_score(
            y_test,
            y_pred,
            pos_label=STRESS_LABEL,
            labels=[STRESS_LABEL, NONSTRESS_LABEL],
            zero_division=0,
        )
        auc_stress = stress_auc_score(y_test, stress_scores)
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[STRESS_LABEL, NONSTRESS_LABEL],
        )
        tn_stress = int(cm[1, 1])
        fp_stress = int(cm[1, 0])
        fn_stress = int(cm[0, 1])
        tp_stress = int(cm[0, 0])

        row = {
            "test_subject": test_subject_data["subject"],
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_stress": count_label(y_train, STRESS_LABEL),
            "train_nonstress": count_label(y_train, NONSTRESS_LABEL),
            "test_stress": count_label(y_test, STRESS_LABEL),
            "test_nonstress": count_label(y_test, NONSTRESS_LABEL),
            "removed_test_rows": test_subject_data["removed_rows"],
            "accuracy": float(accuracy),
            "f1_stress": float(f1_stress),
            "auc_stress": float(auc_stress),
            "tp_stress": tp_stress,
            "fn_stress": fn_stress,
            "fp_stress": fp_stress,
            "tn_stress": tn_stress,
        }
        fold_rows.append(row)

        for sample_index, (truth, pred, score) in enumerate(zip(y_test, y_pred, stress_scores)):
            prediction_rows.append(
                {
                    "test_subject": test_subject_data["subject"],
                    "sample_index": sample_index,
                    "y_true": int(truth),
                    "y_pred": int(pred),
                    "stress_score": float(score),
                    "correct": int(truth == pred),
                }
            )

        print(f"[fold] test={row['test_subject']}")
        print(f"  train samples: {row['train_samples']}")
        print(f"  test samples: {row['test_samples']}")
        print(f"  accuracy: {row['accuracy']:.4f}")
        print(f"  f1_stress: {row['f1_stress']:.4f}")
        print(f"  auc_stress: {row['auc_stress']:.4f}")
        print(f"  confusion stress-positive: TP={tp_stress}, FN={fn_stress}, FP={fp_stress}, TN={tn_stress}")
        print()

    accuracies = np.asarray([row["accuracy"] for row in fold_rows], dtype=np.float64)
    f1_scores = np.asarray([row["f1_stress"] for row in fold_rows], dtype=np.float64)
    auc_scores = np.asarray([row["auc_stress"] for row in fold_rows], dtype=np.float64)

    summary_rows = [
        {
            "metric": "accuracy",
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies)),
        },
        {
            "metric": "f1_stress",
            "mean": float(np.mean(f1_scores)),
            "std": float(np.std(f1_scores, ddof=1)) if len(f1_scores) > 1 else 0.0,
            "min": float(np.min(f1_scores)),
            "max": float(np.max(f1_scores)),
        },
        {
            "metric": "auc_stress",
            "mean": float(np.nanmean(auc_scores)),
            "std": float(np.nanstd(auc_scores, ddof=1)) if np.sum(~np.isnan(auc_scores)) > 1 else 0.0,
            "min": float(np.nanmin(auc_scores)),
            "max": float(np.nanmax(auc_scores)),
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "loso_results.csv",
        fold_rows,
        [
            "test_subject",
            "train_samples",
            "test_samples",
            "train_stress",
            "train_nonstress",
            "test_stress",
            "test_nonstress",
            "removed_test_rows",
            "accuracy",
            "f1_stress",
            "auc_stress",
            "tp_stress",
            "fn_stress",
            "fp_stress",
            "tn_stress",
        ],
    )
    write_csv(
        args.output_dir / "predictions.csv",
        prediction_rows,
        ["test_subject", "sample_index", "y_true", "y_pred", "stress_score", "correct"],
    )
    write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
        ["metric", "mean", "std", "min", "max"],
    )

    np.savez_compressed(
        args.output_dir / "metadata.npz",
        feature_names=np.asarray(feature_names),
        class_weight=np.asarray(args.class_weight),
        random_seed=np.asarray(args.seed),
        stress_label=np.asarray(STRESS_LABEL),
        nonstress_label=np.asarray(NONSTRESS_LABEL),
        subjects=np.asarray([subject_data["subject"] for subject_data in subjects]),
    )

    print("[summary]")
    print(f"  accuracy mean/std: {np.mean(accuracies):.4f} / {np.std(accuracies, ddof=1):.4f}")
    print(f"  f1_stress mean/std: {np.mean(f1_scores):.4f} / {np.std(f1_scores, ddof=1):.4f}")
    print(f"  auc_stress mean/std: {np.nanmean(auc_scores):.4f} / {np.nanstd(auc_scores, ddof=1):.4f}")
    print(f"results written to: {args.output_dir}")


if __name__ == "__main__":
    main()
