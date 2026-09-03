import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "windowed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "peaks"


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def local_maxima_indices(values):
    values = np.asarray(values).reshape(-1)
    if len(values) < 3:
        return np.empty((0,), dtype=np.int64)

    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1


def detect_window_peaks(window, threshold):
    window = np.asarray(window).reshape(-1)
    candidate_indices = local_maxima_indices(window)

    if len(candidate_indices) == 0:
        return candidate_indices, np.empty((0,), dtype=window.dtype)

    keep_mask = window[candidate_indices] >= threshold
    peak_indices = candidate_indices[keep_mask]
    peak_values = window[peak_indices]
    return peak_indices.astype(np.int64), peak_values.astype(np.float64)


def process_subject(npz_path, output_path, overwrite=False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} "
            "(use --overwrite to replace it)"
        )

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    threshold = float(np.mean(X))

    all_peak_indices = []
    all_peak_values = []
    thresholds = []
    peak_counts = []

    for window in X:
        peak_indices, peak_values = detect_window_peaks(window, threshold)
        all_peak_indices.append(peak_indices)
        all_peak_values.append(peak_values)
        thresholds.append(threshold)
        peak_counts.append(len(peak_indices))

    peak_indices_array = np.asarray(all_peak_indices, dtype=object)
    peak_values_array = np.asarray(all_peak_values, dtype=object)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    peak_counts = np.asarray(peak_counts, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        peak_indices=peak_indices_array,
        peak_values=peak_values_array,
        peak_counts=peak_counts,
        thresholds=thresholds,
        y=y,
        starts=data["starts"],
        ends=data["ends"],
        start_seconds=data["start_seconds"],
        end_seconds=data["end_seconds"],
        subject=data["subject"],
        bvp_hz=data["bvp_hz"],
        window_seconds=data["window_seconds"],
        stride_seconds=data["stride_seconds"],
        window_function=(
            data["window_function"]
            if "window_function" in data
            else np.asarray("unknown")
        ),
        threshold_scope=np.asarray("subject_windowed_signal_mean"),
        threshold_value=np.asarray(threshold),
        source_npz=np.asarray(str(npz_path)),
    )

    return {
        "subject": str(np.asarray(data["subject"]).item()),
        "input_npz": str(npz_path),
        "output_npz": str(output_path),
        "windows": int(len(X)),
        "threshold": threshold,
        "min_peaks": int(np.min(peak_counts)) if len(peak_counts) else 0,
        "max_peaks": int(np.max(peak_counts)) if len(peak_counts) else 0,
        "mean_peaks": float(np.mean(peak_counts)) if len(peak_counts) else 0.0,
        "median_peaks": float(np.median(peak_counts)) if len(peak_counts) else 0.0,
        "stress_windows": int(np.sum(y == 0)),
        "nonstress_windows": int(np.sum(y == 1)),
    }


def write_summary(summary_path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "input_npz",
        "output_npz",
        "windows",
        "threshold",
        "min_peaks",
        "max_peaks",
        "mean_peaks",
        "median_peaks",
        "stress_windows",
        "nonstress_windows",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect BVP peaks using LMM and a subject-level mean threshold "
            "computed over all extracted windows."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing Hanning-windowed BVP NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write detected peak NPZ files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing peak NPZ files and summary.csv.",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.input_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no windowed NPZ files found in {args.input_dir}")

    if args.output_dir.exists() and not args.overwrite:
        existing_outputs = list(args.output_dir.glob("*.npz")) + list(
            args.output_dir.glob("summary.csv")
        )
        if existing_outputs:
            raise FileExistsError(
                f"output directory already contains files: {args.output_dir} "
                "(use --overwrite to replace them)"
            )

    rows = []
    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print("method: Local Maxima Method, keep peaks >= subject-level mean amplitude")
    print(f"subjects: {len(npz_paths)}")
    print()

    for npz_path in npz_paths:
        output_path = args.output_dir / npz_path.name
        row = process_subject(npz_path, output_path, overwrite=args.overwrite)
        rows.append(row)

        print(f"[ok] {row['subject']}")
        print(f"  output: {row['output_npz']}")
        print(f"  windows: {row['windows']}")
        print(f"  threshold: {row['threshold']:.6f}")
        print(
            "  peaks/window: "
            f"min={row['min_peaks']}, "
            f"median={row['median_peaks']:.1f}, "
            f"mean={row['mean_peaks']:.1f}, "
            f"max={row['max_peaks']}"
        )
        print()

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
