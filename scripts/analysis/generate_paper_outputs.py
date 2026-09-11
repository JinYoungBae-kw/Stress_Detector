"""Generate manuscript tables and figures from completed stress experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "loso"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "train_features"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "paper"
MODELS = ("baseline", "extended")
KEY_METRICS = (
    "accuracy",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "mcc",
)
DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "MCC",
}
COLORS = {"baseline": "#527D9B", "extended": "#D96745"}


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}")
        resolved = path.resolve()
        if resolved in {Path(resolved.anchor), PROJECT_ROOT.resolve()}:
            raise ValueError(f"refusing to clear protected directory: {resolved}")
        shutil.rmtree(path)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)


def load_predictions(root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    required = {"test_subject", "sample_index", "y_true", "y_pred", "stress_score"}
    for model in MODELS:
        path = root / model / "train_result" / "predictions.csv"
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")
        if frame.duplicated(["test_subject", "sample_index"]).any():
            raise ValueError(f"duplicate LOSO windows found in {path}")
        frames[model] = frame

    keys = ["test_subject", "sample_index"]
    paired = frames["baseline"][keys + ["y_true"]].merge(
        frames["extended"][keys + ["y_true"]],
        on=keys,
        suffixes=("_baseline", "_extended"),
        validate="one_to_one",
    )
    if len(paired) != len(frames["baseline"]) or len(paired) != len(frames["extended"]):
        raise ValueError("baseline and extended predictions cover different windows")
    if not np.array_equal(paired["y_true_baseline"], paired["y_true_extended"]):
        raise ValueError("baseline and extended labels differ")
    return frames


def stress_binary(values: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=int) == 0).astype(np.int8)


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = stress_binary(frame["y_true"])
    predicted = stress_binary(frame["y_pred"])
    scores = frame["stress_score"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": accuracy_score(y, predicted),
        "roc_auc": roc_auc_score(y, scores) if has_both_classes else np.nan,
        "average_precision": (
            average_precision_score(y, scores) if has_both_classes else np.nan
        ),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "mcc": matthews_corrcoef(y, predicted),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def combine_training_tables(root: Path, filename: str, model_column: str) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        frame = pd.read_csv(root / model / "train_result" / filename)
        frame.insert(0, model_column, model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_dataset_tables(
    predictions: dict[str, pd.DataFrame],
    feature_dir: Path,
    table_dir: Path,
) -> list[tuple[str, str, str]]:
    reference = predictions["baseline"]
    dataset_summary = pd.DataFrame(
        [
            {
                "subjects": reference["test_subject"].nunique(),
                "windows": len(reference),
                "stress_windows": int((reference["y_true"] == 0).sum()),
                "nonstress_windows": int((reference["y_true"] == 1).sum()),
                "stress_rate": float((reference["y_true"] == 0).mean()),
                "evaluation": "leave-one-subject-out",
            }
        ]
    )
    dataset_summary.to_csv(table_dir / "dataset_summary.csv", index=False)

    subject_distribution = (
        reference.assign(stress=(reference["y_true"] == 0).astype(int))
        .groupby("test_subject", as_index=False)
        .agg(windows=("y_true", "size"), stress_windows=("stress", "sum"))
    )
    subject_distribution["nonstress_windows"] = (
        subject_distribution["windows"] - subject_distribution["stress_windows"]
    )
    subject_distribution["stress_rate"] = (
        subject_distribution["stress_windows"] / subject_distribution["windows"]
    )
    subject_distribution.to_csv(table_dir / "subject_distribution.csv", index=False)

    feature_rows = []
    for path in sorted(feature_dir.glob("S*.npz")):
        data = np.load(path, allow_pickle=True)
        X = np.asarray(data["X"], dtype=float)
        names = [str(value) for value in data["feature_names"]]
        invalid = ~np.isfinite(X)
        for column, name in enumerate(names):
            feature_rows.append(
                {
                    "subject": path.stem,
                    "feature": name,
                    "rows": len(X),
                    "invalid_count": int(invalid[:, column].sum()),
                    "invalid_rate": float(invalid[:, column].mean()),
                }
            )
    feature_quality = pd.DataFrame(feature_rows)
    feature_quality.to_csv(table_dir / "feature_quality_by_subject.csv", index=False)
    feature_quality.groupby("feature", as_index=False).agg(
        rows=("rows", "sum"),
        invalid_count=("invalid_count", "sum"),
    ).assign(
        invalid_rate=lambda frame: frame["invalid_count"] / frame["rows"]
    ).to_csv(table_dir / "feature_quality.csv", index=False)
    return [
        ("tables/dataset_summary.csv", "table", "Cohort and class distribution"),
        ("tables/subject_distribution.csv", "table", "Window and class counts by subject"),
        ("tables/feature_quality.csv", "table", "Non-finite values by feature"),
        ("tables/feature_quality_by_subject.csv", "table", "Feature quality by subject"),
    ]


def save_performance_tables(
    predictions: dict[str, pd.DataFrame],
    experiment_root: Path,
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[str, str, str]]]:
    subject_mean = combine_training_tables(
        experiment_root, "metrics_summary.csv", "model"
    )
    subject_mean.to_csv(table_dir / "performance_subject_mean.csv", index=False)
    by_subject = combine_training_tables(
        experiment_root, "metrics_by_subject.csv", "model"
    )
    by_subject.to_csv(table_dir / "performance_by_subject.csv", index=False)

    pooled_rows = []
    confusion_rows = []
    for model, frame in predictions.items():
        metrics = calculate_metrics(frame)
        pooled_rows.append({"model": model, **metrics})
        confusion_rows.extend(
            [
                {"model": model, "actual": "nonstress", "predicted": "nonstress", "count": metrics["tn"]},
                {"model": model, "actual": "nonstress", "predicted": "stress", "count": metrics["fp"]},
                {"model": model, "actual": "stress", "predicted": "nonstress", "count": metrics["fn"]},
                {"model": model, "actual": "stress", "predicted": "stress", "count": metrics["tp"]},
            ]
        )
    pd.DataFrame(pooled_rows).to_csv(
        table_dir / "performance_pooled_oof.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(table_dir / "confusion_matrix.csv", index=False)

    difference = pd.read_csv(experiment_root / "metrics_difference.csv")
    difference.to_csv(table_dir / "performance_difference_bootstrap.csv", index=False)

    baseline = by_subject[by_subject["model"] == "baseline"].set_index("test_subject")
    extended = by_subject[by_subject["model"] == "extended"].set_index("test_subject")
    if not baseline.index.equals(extended.index):
        extended = extended.reindex(baseline.index)
    subject_difference = extended[list(KEY_METRICS)] - baseline[list(KEY_METRICS)]
    subject_difference.index.name = "test_subject"
    subject_difference.reset_index().to_csv(
        table_dir / "performance_difference_by_subject.csv", index=False
    )
    return subject_mean, difference, by_subject, [
        ("tables/performance_subject_mean.csv", "table", "Primary LOSO subject-mean metrics and confidence intervals"),
        ("tables/performance_pooled_oof.csv", "table", "Secondary pooled out-of-fold metrics"),
        ("tables/performance_difference_bootstrap.csv", "table", "Paired subject-bootstrap model differences"),
        ("tables/performance_by_subject.csv", "table", "LOSO performance for every held-out subject"),
        ("tables/performance_difference_by_subject.csv", "table", "Extended minus baseline for every subject"),
        ("tables/confusion_matrix.csv", "table", "Pooled confusion-matrix counts"),
    ]


def save_importance_tables(
    experiment_root: Path,
    table_dir: Path,
) -> list[tuple[str, str, str]]:
    rows = []
    for model in MODELS:
        path = experiment_root / model / "shap" / "shap_feature_importance.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "model", model)
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(table_dir / "shap_importance.csv", index=False)

    baseline_features = set(
        json.loads(
            (experiment_root / "baseline" / "train_result" / "config.json").read_text(
                encoding="utf-8"
            )
        )["feature_names"]
    )
    transferred = combined.loc[
        (combined["model"] == "extended")
        & (~combined["feature"].isin(baseline_features))
    ].copy()
    transferred = transferred.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )
    transferred["rank_among_extended"] = np.arange(1, len(transferred) + 1)
    transferred.to_csv(table_dir / "extended_feature_shap.csv", index=False)
    return [
        ("tables/shap_importance.csv", "table", "Full-data SHAP ranking for both feature sets"),
        ("tables/extended_feature_shap.csv", "table", "SHAP ranking restricted to added stress features"),
    ]


def save_feature_sets(experiment_root: Path, table_dir: Path) -> tuple[str, str, str]:
    rows = []
    baseline_features = set()
    for model in MODELS:
        config = json.loads(
            (experiment_root / model / "train_result" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        if model == "baseline":
            baseline_features = set(config["feature_names"])
        for order, feature in enumerate(config["feature_names"], start=1):
            rows.append(
                {
                    "model": model,
                    "order": order,
                    "feature": feature,
                    "feature_group": (
                        "baseline" if feature in baseline_features else "extended_only"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(table_dir / "feature_sets.csv", index=False)
    return (
        "tables/feature_sets.csv",
        "table",
        "Features included in the baseline and extended stress models",
    )


def plot_performance(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        "accuracy",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
    ]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    indexed = summary.set_index(["model", "metric"])
    for offset, model in zip((-width / 2, width / 2), MODELS):
        rows = indexed.loc[model].loc[metrics]
        values = rows["estimate"].to_numpy(dtype=float)
        low = rows["ci95_low"].to_numpy(dtype=float)
        high = rows["ci95_high"].to_numpy(dtype=float)
        ax.bar(x + offset, values, width, color=COLORS[model], label=model.title())
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="none",
            ecolor="#273746",
            capsize=3,
            linewidth=1,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(x, [DISPLAY_NAMES[metric] for metric in metrics], rotation=28, ha="right")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_difference(difference: pd.DataFrame, output: Path) -> None:
    frame = difference.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    values = frame["difference"].to_numpy(dtype=float)
    low = frame["ci95_low"].to_numpy(dtype=float)
    high = frame["ci95_high"].to_numpy(dtype=float)
    colors = np.where(frame["ci_excludes_zero"], "#D96745", "#98A5AE")
    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.axvline(0, color="#273746", linewidth=1)
    for index, color in enumerate(colors):
        ax.errorbar(
            values[index],
            y[index],
            xerr=[[values[index] - low[index]], [high[index] - values[index]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )
    ax.set_yticks(y, [DISPLAY_NAMES.get(metric, metric) for metric in frame["metric"]])
    ax.set_xlabel("Extended - Baseline")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mean_loso_roc(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    grid = np.linspace(0, 1, 201)
    curves = []
    aucs = []
    for _, subject in frame.groupby("test_subject", sort=False):
        y = stress_binary(subject["y_true"])
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, subject["stress_score"])
        curves.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(y, subject["stress_score"]))
    curves = np.asarray(curves)
    return (
        grid,
        curves.mean(axis=0),
        np.percentile(curves, 2.5, axis=0),
        np.percentile(curves, 97.5, axis=0),
        float(np.mean(aucs)),
    )


def _mean_loso_pr(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    grid = np.linspace(0, 1, 201)
    curves = []
    scores = []
    for _, subject in frame.groupby("test_subject", sort=False):
        y = stress_binary(subject["y_true"])
        if len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, subject["stress_score"])
        order = np.argsort(recall)
        curves.append(np.interp(grid, recall[order], precision[order]))
        scores.append(average_precision_score(y, subject["stress_score"]))
    curves = np.asarray(curves)
    return (
        grid,
        curves.mean(axis=0),
        np.percentile(curves, 2.5, axis=0),
        np.percentile(curves, 97.5, axis=0),
        float(np.mean(scores)),
    )


def plot_curves(predictions: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for model, frame in predictions.items():
        grid, mean, low, high, auc = _mean_loso_roc(frame)
        ax.plot(grid, mean, color=COLORS[model], linewidth=2, label=f"{model.title()} (mean AUC={auc:.3f})")
        ax.fill_between(grid, low, high, color=COLORS[model], alpha=0.12)
    ax.plot([0, 1], [0, 1], "--", color="#98A5AE", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_comparison_mean_loso.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for model, frame in predictions.items():
        grid, mean, low, high, ap = _mean_loso_pr(frame)
        ax.plot(grid, mean, color=COLORS[model], linewidth=2, label=f"{model.title()} (mean AP={ap:.3f})")
        ax.fill_between(grid, low, high, color=COLORS[model], alpha=0.12)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(figure_dir / "pr_comparison_mean_loso.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(predictions: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, model in zip(axes, MODELS):
        y = stress_binary(predictions[model]["y_true"])
        predicted = stress_binary(predictions[model]["y_pred"])
        matrix = confusion_matrix(y, predicted, labels=[0, 1])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                ax.text(column, row, f"{matrix[row, column]}\n({normalized[row, column]:.1%})", ha="center", va="center")
        ax.set_title(model.title())
        ax.set_xticks([0, 1], ["Nonstress", "Stress"])
        ax.set_yticks([0, 1], ["Nonstress", "Stress"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.04)
    fig.subplots_adjust(wspace=0.35, right=0.9)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subject_differences(table_dir: Path, output: Path) -> None:
    frame = pd.read_csv(table_dir / "performance_difference_by_subject.csv").set_index(
        "test_subject"
    )
    metrics = ["accuracy", "roc_auc", "average_precision", "f1"]
    values = frame[metrics].to_numpy(dtype=float)
    limit = max(0.01, float(np.nanmax(np.abs(values))))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), [DISPLAY_NAMES[metric] for metric in metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(frame)), frame.index)
    ax.set_xlabel("Extended - Baseline")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_extended_shap(table_dir: Path, output: Path) -> None:
    frame = pd.read_csv(table_dir / "extended_feature_shap.csv").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(frame["feature"], frame["mean_abs_shap"], color=COLORS["extended"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest(output: Path, rows: list[tuple[str, str, str]]) -> None:
    pd.DataFrame(rows, columns=["artifact", "type", "meaning"]).to_csv(
        output / "paper_output_manifest.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stress-study manuscript outputs.")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output(args.output, args.overwrite)
    table_dir = args.output / "tables"
    figure_dir = args.output / "figures"
    predictions = load_predictions(args.experiment_root)

    manifest = save_dataset_tables(predictions, args.feature_dir, table_dir)
    summary, difference, _, performance_manifest = save_performance_tables(
        predictions, args.experiment_root, table_dir
    )
    manifest.extend(performance_manifest)
    manifest.extend(save_importance_tables(args.experiment_root, table_dir))
    manifest.append(save_feature_sets(args.experiment_root, table_dir))

    plot_performance(summary, figure_dir / "performance_comparison.png")
    plot_difference(difference, figure_dir / "performance_difference.png")
    plot_curves(predictions, figure_dir)
    plot_confusion(predictions, figure_dir / "confusion_matrix_comparison.png")
    plot_subject_differences(table_dir, figure_dir / "subject_metric_differences.png")
    plot_extended_shap(table_dir, figure_dir / "extended_feature_shap.png")
    manifest.extend(
        [
            ("figures/performance_comparison.png", "figure", "Primary LOSO subject-mean performance with confidence intervals"),
            ("figures/performance_difference.png", "figure", "Paired subject-bootstrap differences and confidence intervals"),
            ("figures/roc_comparison_mean_loso.png", "figure", "Mean subject-level LOSO ROC curves"),
            ("figures/pr_comparison_mean_loso.png", "figure", "Mean subject-level LOSO precision-recall curves"),
            ("figures/confusion_matrix_comparison.png", "figure", "Pooled out-of-fold confusion matrices"),
            ("figures/subject_metric_differences.png", "figure", "Performance changes for each held-out subject"),
            ("figures/extended_feature_shap.png", "figure", "SHAP importance of the added stress features"),
        ]
    )
    write_manifest(args.output, manifest)
    print(f"Paper outputs: {args.output}")
    print(f"Tables: {len(list(table_dir.glob('*.csv')))}")
    print(f"Figures: {len(list(figure_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
