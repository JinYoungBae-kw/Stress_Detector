import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "train_features"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

STRESS_LABEL = 0
NONSTRESS_LABEL = 1
RANDOM_SEED = 42
BASELINE_FEATURES = [
    "hrm_bpm",
    "hrsd_bpm",
    "mnn_ms",
    "pnn50_percent",
    "total_power_ms2",
    "approx_entropy",
    "correlation_dimension_d2",
]
METRIC_COLUMNS = [
    "accuracy",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "mcc",
]


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def load_subject_feature_file(npz_path, selected_features=None):
    data = np.load(npz_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int8)
    all_feature_names = [str(name) for name in data["feature_names"]]
    feature_names = selected_features or all_feature_names
    missing = sorted(set(feature_names) - set(all_feature_names))
    if missing:
        raise ValueError(f"missing features in {npz_path}: {missing}")
    indices = [all_feature_names.index(name) for name in feature_names]
    X = X[:, indices]
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


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_data_hash(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def prepare_output_dirs(paths, overwrite):
    for path in paths:
        if path.exists() and any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"output directory already contains files: {path} "
                    "(use --overwrite to replace them)"
                )
            resolved = path.resolve()
            if resolved in {Path(resolved.anchor), PROJECT_ROOT.resolve(), OUTPUT_ROOT.resolve()}:
                raise ValueError(f"refusing to clear protected directory: {resolved}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def stress_decision_scores(model, X):
    scores = model.decision_function(X)
    classes = list(model.named_steps["svm"].classes_)
    if classes == [STRESS_LABEL, NONSTRESS_LABEL]:
        return -scores
    if classes == [NONSTRESS_LABEL, STRESS_LABEL]:
        return scores
    raise ValueError(f"unexpected SVM classes: {classes}")


def calculate_metrics(y_true, y_pred, stress_scores):
    truth = (np.asarray(y_true) == STRESS_LABEL).astype(np.int8)
    predicted = (np.asarray(y_pred) == STRESS_LABEL).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    has_both_classes = len(np.unique(truth)) == 2
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "roc_auc": (
            float(roc_auc_score(truth, stress_scores)) if has_both_classes else np.nan
        ),
        "average_precision": (
            float(average_precision_score(truth, stress_scores))
            if has_both_classes
            else np.nan
        ),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, np.nan, np.nan
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    margin = critical * float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - margin, mean + margin


def validate_comparison_configs(output_root):
    configs = {}
    for feature_set in ("baseline", "extended"):
        path = output_root / "loso" / feature_set / "train_result" / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"missing training configuration: {path}")
        configs[feature_set] = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "feature_data_sha256",
        "training_script_sha256",
        "seed",
        "class_weight",
    ):
        if configs["baseline"].get(field) != configs["extended"].get(field):
            raise ValueError(f"baseline and extended settings differ: {field}")
    return int(configs["extended"]["seed"])


def save_metrics_difference(output_root, iterations=2000):
    seed = validate_comparison_configs(output_root)
    result_root = output_root / "loso"
    frames = {}
    required = ["test_subject", "sample_index", "y_true", "y_pred", "stress_score"]
    for feature_set in ("baseline", "extended"):
        path = result_root / feature_set / "train_result" / "predictions.csv"
        frame = pd.read_csv(path)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"missing columns in {path}: {missing}")
        frames[feature_set] = frame[required]

    paired = frames["baseline"].merge(
        frames["extended"],
        on=["test_subject", "sample_index"],
        suffixes=("_baseline", "_extended"),
        validate="one_to_one",
    )
    if len(paired) != len(frames["baseline"]) or len(paired) != len(frames["extended"]):
        raise ValueError("baseline and extended predictions cover different windows")
    if not np.array_equal(paired["y_true_baseline"], paired["y_true_extended"]):
        raise ValueError("baseline and extended labels differ")

    subject_metrics = {feature_set: [] for feature_set in ("baseline", "extended")}
    for subject, group in paired.groupby("test_subject", sort=False):
        for feature_set in subject_metrics:
            subject_metrics[feature_set].append(
                {
                    "subject": subject,
                    **calculate_metrics(
                        group["y_true_baseline"],
                        group[f"y_pred_{feature_set}"],
                        group[f"stress_score_{feature_set}"],
                    ),
                }
            )
    subject_metrics = {
        key: pd.DataFrame(value).set_index("subject") for key, value in subject_metrics.items()
    }
    if not subject_metrics["baseline"].index.equals(subject_metrics["extended"].index):
        raise ValueError("baseline and extended subject order differs")

    observed_baseline = subject_metrics["baseline"][METRIC_COLUMNS].mean()
    observed_extended = subject_metrics["extended"][METRIC_COLUMNS].mean()
    subject_differences = (
        subject_metrics["extended"][METRIC_COLUMNS]
        - subject_metrics["baseline"][METRIC_COLUMNS]
    ).to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((iterations, len(METRIC_COLUMNS)), dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(subject_differences), len(subject_differences))
        bootstrap[index] = np.nanmean(subject_differences[sample], axis=0)

    rows = []
    for index, metric in enumerate(METRIC_COLUMNS):
        low, high = np.percentile(bootstrap[:, index], [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "baseline": observed_baseline[metric],
                "extended": observed_extended[metric],
                "difference": observed_extended[metric] - observed_baseline[metric],
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
                "higher_is_better": True,
                "bootstrap_iterations": iterations,
            }
        )
    output_path = result_root / "metrics_difference.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def compute_and_save_shap(model, X, feature_names, output_dir, seed):
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required to generate explanation results. "
            "Install it with: pip install shap"
        ) from exc

    rng = np.random.default_rng(seed)
    background_size = min(50, len(X))
    explain_size = min(200, len(X))
    background_indices = rng.choice(len(X), size=background_size, replace=False)
    explain_indices = rng.choice(len(X), size=explain_size, replace=False)
    background = X[background_indices]
    X_explain = X[explain_indices]

    explainer = shap.KernelExplainer(
        lambda values: stress_decision_scores(model, values),
        background,
    )
    shap_values = np.asarray(
        explainer.shap_values(X_explain, nsamples="auto"),
        dtype=np.float64,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "shap_values.npz",
        shap_values=shap_values,
        X_explain=X_explain,
        feature_names=np.asarray(feature_names),
        background_indices=background_indices,
        explain_indices=explain_indices,
    )

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    write_csv(
        output_dir / "shap_feature_importance.csv",
        [
            {
                "rank": rank + 1,
                "feature": feature_names[index],
                "mean_abs_shap": float(mean_abs[index]),
            }
            for rank, index in enumerate(order)
        ],
        ["rank", "feature", "mean_abs_shap"],
    )

    plt.figure(figsize=(9, 5))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "background_size": int(background_size),
        "explain_size": int(explain_size),
    }


def train_and_save_full_model(
    subjects,
    feature_names,
    class_weight,
    seed,
    model_dir,
    shap_dir,
):
    X = np.vstack([subject_data["X"] for subject_data in subjects])
    y = np.concatenate([subject_data["y"] for subject_data in subjects])

    model = build_model(class_weight, seed)
    model.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "svm_pipeline.joblib"
    joblib.dump(model, model_path)

    scaler = model.named_steps["scaler"]
    svm = model.named_steps["svm"]
    parameter_path = model_dir / "svm_parameters.npz"
    np.savez_compressed(
        parameter_path,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        support_vectors=svm.support_vectors_,
        support_indices=svm.support_,
        dual_coef=svm.dual_coef_,
        intercept=svm.intercept_,
        n_support=svm.n_support_,
        classes=svm.classes_,
        feature_names=np.asarray(feature_names),
    )

    shap_info = compute_and_save_shap(
        model,
        X,
        feature_names,
        shap_dir,
        seed,
    )

    metadata = {
        "model_path": str(model_path),
        "parameter_path": str(parameter_path),
        "training_scope": "all available subjects after LOSO evaluation",
        "subjects": [subject_data["subject"] for subject_data in subjects],
        "training_samples": int(len(y)),
        "stress_samples": int(np.sum(y == STRESS_LABEL)),
        "nonstress_samples": int(np.sum(y == NONSTRESS_LABEL)),
        "feature_names": feature_names,
        "random_seed": seed,
        "model": {
            "pipeline": "StandardScaler + SVC",
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": class_weight,
        },
        "shap": shap_info,
    }
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return model_path, parameter_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline or extended SVM models with LOSO validation."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=FEATURE_DIR,
        help="Directory containing subject feature NPZ files.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["baseline", "extended"],
        default="extended",
        help="Use the original seven or all ten stress features. Default: extended.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the automatic train_result directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override the automatic model directory.",
    )
    parser.add_argument(
        "--shap-dir",
        type=Path,
        default=None,
        help="Override the automatic SHAP directory.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="none",
        help="SVM class weight. Default: none.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed. Default: {RANDOM_SEED}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files for the selected feature set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    experiment_dir = args.output_root / "loso" / args.feature_set
    train_result_dir = args.output_dir or experiment_dir / "train_result"
    model_dir = args.model_dir or experiment_dir / "model"
    shap_dir = args.shap_dir or experiment_dir / "shap"
    standard_output_layout = all(
        path is None for path in (args.output_dir, args.model_dir, args.shap_dir)
    )
    comparison_path = args.output_root / "loso" / "metrics_difference.csv"
    if standard_output_layout and args.feature_set == "baseline":
        comparison_path.unlink(missing_ok=True)
    prepare_output_dirs(
        [train_result_dir, model_dir, shap_dir],
        overwrite=args.overwrite,
    )

    npz_paths = sorted(args.feature_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no feature NPZ files found in {args.feature_dir}")

    selected_features = BASELINE_FEATURES if args.feature_set == "baseline" else None
    subjects = [
        load_subject_feature_file(path, selected_features=selected_features)
        for path in npz_paths
    ]
    feature_names = subjects[0]["feature_names"]
    for subject_data in subjects:
        if subject_data["feature_names"] != feature_names:
            raise ValueError(f"feature name mismatch in {subject_data['source']}")
        if subject_data["X"].shape[1] != len(feature_names):
            raise ValueError(f"feature shape mismatch in {subject_data['source']}")

    class_weight = "balanced" if args.class_weight == "balanced" else None
    config = {
        "feature_dir": str(args.feature_dir.resolve()),
        "feature_data_sha256": feature_data_hash(npz_paths),
        "training_script_sha256": file_sha256(Path(__file__).resolve()),
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "subjects": [subject_data["subject"] for subject_data in subjects],
        "seed": args.seed,
        "class_weight": args.class_weight,
        "evaluation": "leave-one-subject-out",
        "positive_class": "stress (source label 0)",
        "model": "StandardScaler + SVC(kernel=rbf, C=1.0, gamma=scale)",
    }
    (train_result_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    subject_rows = []
    prediction_rows = []
    print(f"feature_set: {args.feature_set}")
    print(f"feature_dir: {args.feature_dir}")
    print(f"subjects: {len(subjects)}")
    print(f"features ({len(feature_names)}): {', '.join(feature_names)}")
    print(
        "model: StandardScaler + "
        f"SVC(kernel=rbf, C=1.0, gamma=scale, class_weight={class_weight})"
    )
    print()

    for test_index, test_subject_data in enumerate(subjects):
        train_subjects = [
            subject_data
            for index, subject_data in enumerate(subjects)
            if index != test_index
        ]
        X_train = np.vstack([subject_data["X"] for subject_data in train_subjects])
        y_train = np.concatenate([subject_data["y"] for subject_data in train_subjects])
        X_test = test_subject_data["X"]
        y_test = test_subject_data["y"]

        model = build_model(class_weight, args.seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        stress_scores = stress_decision_scores(model, X_test)
        metrics = calculate_metrics(y_test, y_pred, stress_scores)
        row = {
            "test_subject": test_subject_data["subject"],
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_stress": count_label(y_train, STRESS_LABEL),
            "train_nonstress": count_label(y_train, NONSTRESS_LABEL),
            "test_stress": count_label(y_test, STRESS_LABEL),
            "test_nonstress": count_label(y_test, NONSTRESS_LABEL),
            "removed_test_rows": test_subject_data["removed_rows"],
            **{metric: metrics[metric] for metric in METRIC_COLUMNS},
            "tp_stress": metrics["tp"],
            "fn_stress": metrics["fn"],
            "fp_stress": metrics["fp"],
            "tn_stress": metrics["tn"],
        }
        subject_rows.append(row)

        for sample_index, (truth, prediction, score) in enumerate(
            zip(y_test, y_pred, stress_scores)
        ):
            prediction_rows.append(
                {
                    "test_subject": test_subject_data["subject"],
                    "sample_index": sample_index,
                    "y_true": int(truth),
                    "y_pred": int(prediction),
                    "stress_score": float(score),
                    "correct": int(truth == prediction),
                }
            )

        print(
            f"[{test_index + 1:02d}/{len(subjects)}] "
            f"{row['test_subject']}: accuracy={row['accuracy']:.3f}, "
            f"F1={row['f1']:.3f}, AUC={row['roc_auc']:.3f}"
        )

    metrics_by_subject = pd.DataFrame(subject_rows)
    metrics_by_subject.to_csv(
        train_result_dir / "metrics_by_subject.csv",
        index=False,
    )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(train_result_dir / "predictions.csv", index=False)

    summary_rows = []
    for metric in METRIC_COLUMNS:
        values = metrics_by_subject[metric]
        estimate, low, high = confidence_interval(values)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "min": float(finite.min()),
                "max": float(finite.max()),
                "n_subjects": int(len(finite)),
                "aggregation": "mean_across_LOSO_subjects_t_interval",
            }
        )
    metrics_summary = pd.DataFrame(summary_rows)
    metrics_summary.to_csv(train_result_dir / "metrics_summary.csv", index=False)

    model_path, parameter_path = train_and_save_full_model(
        subjects=subjects,
        feature_names=feature_names,
        class_weight=class_weight,
        seed=args.seed,
        model_dir=model_dir,
        shap_dir=shap_dir,
    )

    comparison_output = None
    if standard_output_layout and args.feature_set == "extended":
        comparison_output = save_metrics_difference(args.output_root)

    print("\n=== SVM LOSO Training Complete ===")
    display = metrics_summary[["metric", "estimate", "ci95_low", "ci95_high"]].copy()
    for column in ("estimate", "ci95_low", "ci95_high"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    print(display.to_string(index=False))
    print(f"Training results: {train_result_dir}")
    print(f"Full-data model: {model_path}")
    print(f"SVM parameters: {parameter_path}")
    print(f"SHAP: {shap_dir}")
    if comparison_output is not None:
        print(f"Model comparison: {comparison_output}")


if __name__ == "__main__":
    main()
