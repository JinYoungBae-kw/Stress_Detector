import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "3.5_peak_corrected"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"

STRESS_LABEL = 0
NONSTRESS_LABEL = 1
RANDOM_SEED = 42
CLASS_WEIGHT = None
OVERWRITE = True


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

    return {
        "subject": subject,
        "X": X[finite_rows],
        "y": y[finite_rows],
        "feature_names": feature_names,
        "removed_rows": int(np.sum(~finite_rows)),
        "source": str(npz_path),
        "samples_before": int(len(y)),
        "samples_after": int(np.sum(finite_rows)),
    }


def build_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight=CLASS_WEIGHT,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def stress_decision_scores(model, X):
    scores = model.decision_function(X)
    classes = list(model.named_steps["svm"].classes_)
    if classes == [STRESS_LABEL, NONSTRESS_LABEL]:
        return -scores
    if classes == [NONSTRESS_LABEL, STRESS_LABEL]:
        return scores
    raise ValueError(f"unexpected SVM classes: {classes}")


def compute_and_save_shap(model, X, feature_names, output_dir):
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required to generate explanation images. "
            "Install it with: pip install shap"
        ) from exc

    rng = np.random.default_rng(RANDOM_SEED)
    background_size = min(50, len(X))
    explain_size = min(200, len(X))

    background_indices = rng.choice(len(X), size=background_size, replace=False)
    explain_indices = rng.choice(len(X), size=explain_size, replace=False)
    background = X[background_indices]
    X_explain = X[explain_indices]

    explainer = shap.KernelExplainer(lambda values: stress_decision_scores(model, values), background)
    shap_values = explainer.shap_values(X_explain, nsamples="auto")
    shap_values = np.asarray(shap_values, dtype=np.float64)

    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        shap_dir / "shap_values.npz",
        shap_values=shap_values,
        X_explain=X_explain,
        feature_names=np.asarray(feature_names),
        background_indices=background_indices,
        explain_indices=explain_indices,
    )

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    rows = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[index],
            "mean_abs_shap": float(mean_abs[index]),
        }
        for rank, index in enumerate(order)
    ]
    write_csv(
        shap_dir / "shap_feature_importance.csv",
        rows,
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
    plt.savefig(shap_dir / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "background_size": int(background_size),
        "explain_size": int(explain_size),
        "shap_dir": str(shap_dir),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    np.random.seed(RANDOM_SEED)

    if OUTPUT_DIR.exists() and not OVERWRITE:
        existing = list(OUTPUT_DIR.glob("*"))
        if existing:
            raise FileExistsError(f"output directory already contains files: {OUTPUT_DIR}")

    npz_paths = sorted(FEATURE_DIR.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no feature NPZ files found in {FEATURE_DIR}")

    subjects = [load_subject_feature_file(path) for path in npz_paths]
    feature_names = subjects[0]["feature_names"]
    for subject_data in subjects:
        if subject_data["feature_names"] != feature_names:
            raise ValueError(f"feature name mismatch in {subject_data['source']}")
        if subject_data["X"].shape[1] != len(feature_names):
            raise ValueError(f"feature shape mismatch in {subject_data['source']}")

    X = np.vstack([subject_data["X"] for subject_data in subjects])
    y = np.concatenate([subject_data["y"] for subject_data in subjects])

    model = build_model()
    model.fit(X, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "svm_pipeline.joblib"
    joblib.dump(model, model_path)

    shap_info = compute_and_save_shap(model, X, feature_names, OUTPUT_DIR)

    subject_rows = []
    for subject_data in subjects:
        subject_y = subject_data["y"]
        subject_rows.append(
            {
                "subject": subject_data["subject"],
                "source": subject_data["source"],
                "samples_before": subject_data["samples_before"],
                "samples_after": subject_data["samples_after"],
                "removed_rows": subject_data["removed_rows"],
                "stress_samples": int(np.sum(subject_y == STRESS_LABEL)),
                "nonstress_samples": int(np.sum(subject_y == NONSTRESS_LABEL)),
            }
        )

    write_csv(
        OUTPUT_DIR / "training_subjects.csv",
        subject_rows,
        [
            "subject",
            "source",
            "samples_before",
            "samples_after",
            "removed_rows",
            "stress_samples",
            "nonstress_samples",
        ],
    )

    config = {
        "feature_dir": str(FEATURE_DIR),
        "model_path": str(model_path),
        "feature_names": feature_names,
        "stress_label": STRESS_LABEL,
        "nonstress_label": NONSTRESS_LABEL,
        "random_seed": RANDOM_SEED,
        "model": {
            "pipeline": "StandardScaler + SVC",
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": CLASS_WEIGHT,
        },
        "training_samples": int(len(y)),
        "stress_samples": int(np.sum(y == STRESS_LABEL)),
        "nonstress_samples": int(np.sum(y == NONSTRESS_LABEL)),
        "subjects": [subject_data["subject"] for subject_data in subjects],
        "expected_pipeline": {
            "bvp_source": "wrist BVP",
            "bandpass_hz": [0.5, 3.5],
            "window_seconds": 360,
            "stride_seconds": 30,
            "window_function": "hanning",
            "peak_correction": "short NN interval correction",
            "nn_interval_valid_range_sec": [0.3, 2.0],
        },
        "shap": shap_info,
    }

    with (OUTPUT_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    np.savez_compressed(
        OUTPUT_DIR / "metadata.npz",
        feature_names=np.asarray(feature_names),
        class_weight=np.asarray("none" if CLASS_WEIGHT is None else CLASS_WEIGHT),
        random_seed=np.asarray(RANDOM_SEED),
        stress_label=np.asarray(STRESS_LABEL),
        nonstress_label=np.asarray(NONSTRESS_LABEL),
        subjects=np.asarray([subject_data["subject"] for subject_data in subjects]),
        training_samples=np.asarray(len(y)),
        stress_samples=np.asarray(int(np.sum(y == STRESS_LABEL))),
        nonstress_samples=np.asarray(int(np.sum(y == NONSTRESS_LABEL))),
    )

    print(f"feature_dir: {FEATURE_DIR}")
    print(f"output_dir: {OUTPUT_DIR}")
    print(f"model saved: {model_path}")
    print(f"shap results: {shap_info['shap_dir']}")
    print(f"subjects: {len(subjects)}")
    print(f"training samples: {len(y)}")
    print(f"stress samples: {int(np.sum(y == STRESS_LABEL))}")
    print(f"non-stress samples: {int(np.sum(y == NONSTRESS_LABEL))}")
    print(f"features: {', '.join(feature_names)}")


if __name__ == "__main__":
    main()
