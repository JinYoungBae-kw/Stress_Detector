import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "features" / "peaks"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "intervals"

LOWER_NN_SEC = 0.30
UPPER_NN_SEC = 2.00
MAX_DELETIONS_PER_WINDOW = 200


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def scalar_to_string(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def nn_intervals_sec(peak_indices, bvp_hz):
    peak_indices = np.asarray(peak_indices, dtype=np.int64)
    if len(peak_indices) < 2:
        return np.empty((0,), dtype=np.float64)
    return np.diff(peak_indices) / float(bvp_hz)


def interval_counts(intervals, lower_sec, upper_sec):
    intervals = np.asarray(intervals, dtype=np.float64)
    short_count = int(np.sum(intervals < lower_sec))
    long_count = int(np.sum(intervals > upper_sec))
    abnormal_count = short_count + long_count
    ratio = abnormal_count / len(intervals) if len(intervals) else 0.0
    return short_count, long_count, abnormal_count, ratio


def robust_reference_interval(intervals, lower_sec, upper_sec):
    intervals = np.asarray(intervals, dtype=np.float64)
    valid = intervals[(intervals >= lower_sec) & (intervals <= upper_sec)]
    if len(valid):
        return float(np.median(valid))
    if len(intervals):
        return float(np.median(intervals))
    return 1.0


def sequence_score(peak_indices, bvp_hz, reference_nn_sec, lower_sec, upper_sec):
    intervals = nn_intervals_sec(peak_indices, bvp_hz)
    if len(intervals) == 0:
        return 1_000_000.0

    short_count, long_count, abnormal_count, _ = interval_counts(
        intervals,
        lower_sec,
        upper_sec,
    )
    valid = intervals[(intervals >= lower_sec) & (intervals <= upper_sec)]
    if len(valid) == 0:
        rhythm_penalty = 1_000.0
    else:
        rhythm_penalty = float(np.mean(np.abs(valid - reference_nn_sec)))
        rhythm_penalty += float(np.std(valid))

    return abnormal_count * 1_000.0 + short_count * 100.0 + long_count * 200.0 + rhythm_penalty


def amplitude_tie_penalty(delete_index, peak_values):
    values = np.asarray(peak_values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    value_range = float(np.max(values) - np.min(values))
    if value_range <= 0:
        return 0.0
    normalized = (float(values[delete_index]) - float(np.min(values))) / value_range
    return normalized * 0.01


def delete_peak(peak_indices, peak_values, delete_index):
    return (
        np.delete(peak_indices, delete_index).astype(np.int64),
        np.delete(peak_values, delete_index).astype(np.float64),
    )


def correct_window_peaks(peak_indices, peak_values, bvp_hz, lower_sec, upper_sec):
    peak_indices = np.asarray(peak_indices, dtype=np.int64).copy()
    peak_values = np.asarray(peak_values, dtype=np.float64).copy()
    deleted_count = 0
    deleted_peak_indices = []
    deleted_peak_values = []

    for _ in range(MAX_DELETIONS_PER_WINDOW):
        intervals = nn_intervals_sec(peak_indices, bvp_hz)
        short_positions = np.flatnonzero(intervals < lower_sec)
        if len(short_positions) == 0:
            break

        short_pos = int(short_positions[0])
        if len(peak_indices) < 3:
            break

        reference_nn = robust_reference_interval(intervals, lower_sec, upper_sec)
        current_score = sequence_score(
            peak_indices,
            bvp_hz,
            reference_nn,
            lower_sec,
            upper_sec,
        )

        candidates = []
        for delete_index in (short_pos, short_pos + 1):
            candidate_indices, candidate_values = delete_peak(
                peak_indices,
                peak_values,
                delete_index,
            )
            score = sequence_score(
                candidate_indices,
                bvp_hz,
                reference_nn,
                lower_sec,
                upper_sec,
            )
            score += amplitude_tie_penalty(delete_index, peak_values)
            candidates.append((score, delete_index, candidate_indices, candidate_values))

        best_score, best_delete_index, best_indices, best_values = min(
            candidates,
            key=lambda item: item[0],
        )

        if best_score >= current_score:
            break

        deleted_peak_indices.append(int(peak_indices[best_delete_index]))
        deleted_peak_values.append(float(peak_values[best_delete_index]))
        peak_indices = best_indices
        peak_values = best_values
        deleted_count += 1

    return {
        "peak_indices": peak_indices,
        "peak_values": peak_values,
        "deleted_count": deleted_count,
        "deleted_peak_indices": np.asarray(deleted_peak_indices, dtype=np.int64),
        "deleted_peak_values": np.asarray(deleted_peak_values, dtype=np.float64),
    }


def process_subject(npz_path, output_path, overwrite=False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")

    data = np.load(npz_path, allow_pickle=True)
    bvp_hz = int(np.asarray(data["bvp_hz"]).item())
    peak_indices = data["peak_indices"]
    peak_values = data["peak_values"]

    corrected_peak_indices = []
    corrected_peak_values = []
    corrected_nn_intervals = []
    corrected_nn_counts = []
    deleted_peak_indices = []
    deleted_peak_values = []
    deleted_peak_counts = []

    before_short_total = 0
    before_long_total = 0
    before_abnormal_total = 0
    before_interval_total = 0
    after_short_total = 0
    after_long_total = 0
    after_abnormal_total = 0
    after_interval_total = 0

    for window_peak_indices, window_peak_values in zip(peak_indices, peak_values):
        before_intervals = nn_intervals_sec(window_peak_indices, bvp_hz)
        before_short, before_long, before_abnormal, _ = interval_counts(
            before_intervals,
            LOWER_NN_SEC,
            UPPER_NN_SEC,
        )
        before_short_total += before_short
        before_long_total += before_long
        before_abnormal_total += before_abnormal
        before_interval_total += len(before_intervals)

        corrected = correct_window_peaks(
            window_peak_indices,
            window_peak_values,
            bvp_hz,
            LOWER_NN_SEC,
            UPPER_NN_SEC,
        )

        after_intervals = nn_intervals_sec(corrected["peak_indices"], bvp_hz)
        after_short, after_long, after_abnormal, _ = interval_counts(
            after_intervals,
            LOWER_NN_SEC,
            UPPER_NN_SEC,
        )
        after_short_total += after_short
        after_long_total += after_long
        after_abnormal_total += after_abnormal
        after_interval_total += len(after_intervals)

        corrected_peak_indices.append(corrected["peak_indices"])
        corrected_peak_values.append(corrected["peak_values"])
        corrected_nn_intervals.append(after_intervals)
        corrected_nn_counts.append(len(after_intervals))
        deleted_peak_indices.append(corrected["deleted_peak_indices"])
        deleted_peak_values.append(corrected["deleted_peak_values"])
        deleted_peak_counts.append(corrected["deleted_count"])

    corrected_peak_indices = np.asarray(corrected_peak_indices, dtype=object)
    corrected_peak_values = np.asarray(corrected_peak_values, dtype=object)
    corrected_nn_intervals = np.asarray(corrected_nn_intervals, dtype=object)
    corrected_nn_counts = np.asarray(corrected_nn_counts, dtype=np.int64)
    deleted_peak_indices = np.asarray(deleted_peak_indices, dtype=object)
    deleted_peak_values = np.asarray(deleted_peak_values, dtype=object)
    deleted_peak_counts = np.asarray(deleted_peak_counts, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        nn_intervals_sec=corrected_nn_intervals,
        nn_intervals_ms=corrected_nn_intervals * 1000.0,
        nn_counts=corrected_nn_counts,
        corrected_peak_indices=corrected_peak_indices,
        corrected_peak_values=corrected_peak_values,
        deleted_peak_indices=deleted_peak_indices,
        deleted_peak_values=deleted_peak_values,
        deleted_peak_counts=deleted_peak_counts,
        y=data["y"],
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
        correction_method=np.asarray("short_interval_delete_peak_by_local_naturalness"),
        lower_nn_sec=np.asarray(LOWER_NN_SEC),
        upper_nn_sec=np.asarray(UPPER_NN_SEC),
        source_npz=np.asarray(str(npz_path)),
    )

    before_ratio = before_abnormal_total / before_interval_total if before_interval_total else 0.0
    after_ratio = after_abnormal_total / after_interval_total if after_interval_total else 0.0
    deleted_total = int(np.sum(deleted_peak_counts))

    return {
        "subject": scalar_to_string(data["subject"]),
        "input_npz": str(npz_path),
        "output_npz": str(output_path),
        "windows": int(len(peak_indices)),
        "before_intervals": int(before_interval_total),
        "after_intervals": int(after_interval_total),
        "before_short_count": int(before_short_total),
        "after_short_count": int(after_short_total),
        "before_long_count": int(before_long_total),
        "after_long_count": int(after_long_total),
        "before_abnormal_ratio": float(before_ratio),
        "after_abnormal_ratio": float(after_ratio),
        "deleted_peak_count": deleted_total,
        "stress_windows": int(np.sum(data["y"] == 0)),
        "nonstress_windows": int(np.sum(data["y"] == 1)),
    }


def write_summary(summary_path, rows):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "input_npz",
        "output_npz",
        "windows",
        "before_intervals",
        "after_intervals",
        "before_short_count",
        "after_short_count",
        "before_long_count",
        "after_long_count",
        "before_abnormal_ratio",
        "after_abnormal_ratio",
        "deleted_peak_count",
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
            "Correct short peak intervals and calculate NN intervals "
            "for each BVP window."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing detected peak NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write corrected peak and NN interval NPZ files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing interval NPZ files and summary.csv.",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.input_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no peak NPZ files found in {args.input_dir}")

    if args.output_dir.exists() and not args.overwrite:
        existing_outputs = list(args.output_dir.glob("*.npz")) + list(
            args.output_dir.glob("summary.csv")
        )
        if existing_outputs:
            raise FileExistsError(
                f"output directory already contains files: {args.output_dir} "
                "(use --overwrite to replace them)"
            )

    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"correction range: short < {LOWER_NN_SEC}s, long > {UPPER_NN_SEC}s")
    print("correction: only short intervals are corrected by deleting one nearby peak")
    print(f"subjects: {len(npz_paths)}")
    print()

    rows = []
    for npz_path in npz_paths:
        row = process_subject(
            npz_path,
            args.output_dir / npz_path.name,
            overwrite=args.overwrite,
        )
        rows.append(row)
        print(f"[ok] {row['subject']}")
        print(
            "  before short/long: "
            f"{row['before_short_count']}/{row['before_long_count']}"
        )
        print(
            "  after short/long:  "
            f"{row['after_short_count']}/{row['after_long_count']}"
        )
        print(
            "  abnormal ratio: "
            f"{row['before_abnormal_ratio'] * 100:.4f}% -> "
            f"{row['after_abnormal_ratio'] * 100:.4f}%"
        )
        print(f"  deleted peaks: {row['deleted_peak_count']}")
        print()

    write_summary(args.output_dir / "summary.csv", rows)

    total_before_intervals = sum(row["before_intervals"] for row in rows)
    total_after_intervals = sum(row["after_intervals"] for row in rows)
    total_before_short = sum(row["before_short_count"] for row in rows)
    total_after_short = sum(row["after_short_count"] for row in rows)
    total_before_long = sum(row["before_long_count"] for row in rows)
    total_after_long = sum(row["after_long_count"] for row in rows)
    total_deleted = sum(row["deleted_peak_count"] for row in rows)
    total_before_abnormal = total_before_short + total_before_long
    total_after_abnormal = total_after_short + total_after_long
    before_ratio = (
        total_before_abnormal / total_before_intervals
        if total_before_intervals
        else 0.0
    )
    after_ratio = (
        total_after_abnormal / total_after_intervals
        if total_after_intervals
        else 0.0
    )

    print("[total]")
    print(f"before short count: {total_before_short}")
    print(f"after short count:  {total_after_short}")
    print(f"before long count:  {total_before_long}")
    print(f"after long count:   {total_after_long}")
    print(f"before abnormal ratio: {before_ratio * 100:.4f}%")
    print(f"after abnormal ratio:  {after_ratio * 100:.4f}%")
    print(f"deleted peak count: {total_deleted}")
    print(f"summary: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
