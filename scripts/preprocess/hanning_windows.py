import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from common import subject_sort_key

BVP_HZ = 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_FUNCTION = "hanning"


def iter_windows(signal_length, window_samples, stride_samples):
    start = 0
    while start + window_samples <= signal_length:
        end = start + window_samples
        yield start, end
        start += stride_samples


def apply_hanning_window(window):
    return window * np.hanning(len(window))


def process_subject(pkl_path, output_path, window_seconds, stride_seconds):
    with pkl_path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    subject = data.get("subject", pkl_path.stem)
    bvp = np.asarray(data["signal"]["wrist"]["BVP"]).reshape(-1)
    if "label_bvp" not in data:
        raise KeyError(f"{pkl_path} does not contain data['label_bvp']")

    label_bvp = np.asarray(data["label_bvp"]).reshape(-1)
    if len(bvp) != len(label_bvp):
        raise ValueError(
            f"{pkl_path}: BVP length ({len(bvp)}) and label_bvp length "
            f"({len(label_bvp)}) do not match"
        )

    window_samples = int(window_seconds * BVP_HZ)
    stride_samples = int(stride_seconds * BVP_HZ)

    windows = []
    labels = []
    starts = []
    ends = []
    dropped_mixed = 0
    dropped_ignore = 0
    total_windows = 0

    for start, end in iter_windows(len(bvp), window_samples, stride_samples):
        total_windows += 1
        window_labels = label_bvp[start:end]
        unique_labels = np.unique(window_labels)

        if len(unique_labels) != 1:
            dropped_mixed += 1
            continue

        label = int(unique_labels[0])
        if label == 2:
            dropped_ignore += 1
            continue

        window = apply_hanning_window(bvp[start:end])
        windows.append(window)
        labels.append(label)
        starts.append(start)
        ends.append(end)

    if windows:
        X = np.stack(windows).astype(np.float32)
        y = np.asarray(labels, dtype=np.int8)
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
    else:
        X = np.empty((0, window_samples), dtype=np.float32)
        y = np.empty((0,), dtype=np.int8)
        starts = np.empty((0,), dtype=np.int64)
        ends = np.empty((0,), dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        starts=starts,
        ends=ends,
        start_seconds=starts / BVP_HZ,
        end_seconds=ends / BVP_HZ,
        subject=np.asarray(subject),
        bvp_hz=np.asarray(BVP_HZ),
        window_seconds=np.asarray(window_seconds),
        stride_seconds=np.asarray(stride_seconds),
        window_function=np.asarray(WINDOW_FUNCTION),
    )

    stress_windows = int(np.sum(y == 0))
    nonstress_windows = int(np.sum(y == 1))

    return {
        "subject": subject,
        "input_pkl": str(pkl_path),
        "output_npz": str(output_path),
        "signal_samples": len(bvp),
        "duration_seconds": len(bvp) / BVP_HZ,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "window_function": WINDOW_FUNCTION,
        "total_windows": total_windows,
        "kept_windows": len(y),
        "stress_windows": stress_windows,
        "nonstress_windows": nonstress_windows,
        "dropped_mixed": dropped_mixed,
        "dropped_ignore": dropped_ignore,
    }


def write_summary(summary_path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "input_pkl",
        "output_npz",
        "signal_samples",
        "duration_seconds",
        "window_seconds",
        "stride_seconds",
        "window_samples",
        "stride_samples",
        "window_function",
        "total_windows",
        "kept_windows",
        "stress_windows",
        "nonstress_windows",
        "dropped_mixed",
        "dropped_ignore",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Create fixed-length BVP windows from preprocessed WESAD PKL files."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed" / "moving_average",
        help=(
            "Directory containing final preprocessed subject PKL files. "
            "Default: <project>/data/preprocessed/moving_average"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "windowed",
        help="Directory to write subject NPZ window files.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=360,
        help="Window length in seconds. Default: 360",
    )
    parser.add_argument(
        "--stride-seconds",
        type=int,
        default=30,
        help="Sliding stride in seconds. Default: 30",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing NPZ files and summary.csv.",
    )
    args = parser.parse_args()

    pkl_paths = sorted(
        args.preprocessed_dir.glob("S*/S*.pkl"),
        key=subject_sort_key,
    )
    if not pkl_paths:
        raise FileNotFoundError(
            f"no preprocessed PKL files found in {args.preprocessed_dir}"
        )

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
    print(f"preprocessed_dir: {args.preprocessed_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"window: {args.window_seconds}s")
    print(f"stride: {args.stride_seconds}s")
    print(f"window_function: {WINDOW_FUNCTION}")
    print(f"subjects: {len(pkl_paths)}")
    print()

    for pkl_path in pkl_paths:
        subject = pkl_path.stem
        output_path = args.output_dir / f"{subject}.npz"
        row = process_subject(
            pkl_path=pkl_path,
            output_path=output_path,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
        )
        rows.append(row)

        print(f"[ok] {row['subject']}")
        print(f"  output: {row['output_npz']}")
        print(f"  total windows: {row['total_windows']}")
        print(f"  kept: {row['kept_windows']}")
        print(f"  stress: {row['stress_windows']}")
        print(f"  non-stress: {row['nonstress_windows']}")
        print(f"  dropped mixed: {row['dropped_mixed']}")
        print(f"  dropped ignore: {row['dropped_ignore']}")
        print()

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
