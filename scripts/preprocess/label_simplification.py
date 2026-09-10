import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from common import subject_sort_key

LABEL_HZ = 700
BVP_HZ = 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def simplify_labels(labels):
    labels = np.asarray(labels)
    simplified = np.full(labels.shape, 2, dtype=np.int8)
    simplified[labels == 2] = 0
    simplified[labels == 1] = 1
    return simplified


def contiguous_segments(labels):
    change_points = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, len(labels)]
    return zip(starts, ends, labels[starts])


def build_bvp_labels(labels, bvp_length):
    labels = np.asarray(labels)
    bvp_labels = np.empty(bvp_length, dtype=labels.dtype)

    for start, end, label_value in contiguous_segments(labels):
        bvp_start = round((start / LABEL_HZ) * BVP_HZ)
        bvp_end = round((end / LABEL_HZ) * BVP_HZ)

        bvp_start = max(0, min(bvp_start, bvp_length))
        bvp_end = max(0, min(bvp_end, bvp_length))

        if end == len(labels):
            bvp_end = bvp_length

        if bvp_end > bvp_start:
            bvp_labels[bvp_start:bvp_end] = label_value

    return bvp_labels


def counts_by_seconds(labels, sample_rate):
    counts = Counter(np.asarray(labels).tolist())
    return {
        int(label): {
            "samples": int(count),
            "seconds": count / sample_rate,
        }
        for label, count in sorted(counts.items())
    }


def format_counts(counts):
    return ", ".join(
        f"{label}: {values['samples']} ({values['seconds']:.1f}s)"
        for label, values in counts.items()
    )


def process_subject(raw_pkl_path, labeled_pkl_path, overwrite=False):
    if labeled_pkl_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {labeled_pkl_path} "
            "(use --overwrite to replace it)"
        )

    with raw_pkl_path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    original_label = np.asarray(data["label"])
    simplified_label = simplify_labels(original_label)

    bvp = np.asarray(data["signal"]["wrist"]["BVP"])
    bvp_length = len(bvp)

    label_duration = len(simplified_label) / LABEL_HZ
    bvp_duration = bvp_length / BVP_HZ
    if not np.isclose(label_duration, bvp_duration, atol=1 / BVP_HZ):
        raise ValueError(
            f"duration mismatch in {raw_pkl_path}: "
            f"label={label_duration:.6f}s, BVP={bvp_duration:.6f}s"
        )

    bvp_label = build_bvp_labels(simplified_label, bvp_length)

    data["label"] = simplified_label
    data["label_bvp"] = bvp_label

    labeled_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = labeled_pkl_path.with_suffix(labeled_pkl_path.suffix + ".tmp")
    with temp_path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(labeled_pkl_path)

    return {
        "subject": data.get("subject", raw_pkl_path.stem),
        "raw_pkl": raw_pkl_path,
        "labeled_pkl": labeled_pkl_path,
        "label_shape": simplified_label.shape,
        "bvp_shape": bvp.shape,
        "label_bvp_shape": bvp_label.shape,
        "label_counts": counts_by_seconds(simplified_label, LABEL_HZ),
        "label_bvp_counts": counts_by_seconds(bvp_label, BVP_HZ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Label all WESAD PKL files using baseline-only non-stress: "
            "2->stress, 1->non-stress, all other labels->ignore."
        )
    )
    parser.add_argument(
        "--raw-dir",
        default=PROJECT_ROOT / "data" / "raw",
        type=Path,
        help="Directory containing raw WESAD subject folders. Default: <project>/data/raw",
    )
    parser.add_argument(
        "--labeled-dir",
        default=PROJECT_ROOT / "data" / "labeled",
        type=Path,
        help="Directory to write labeled subject folders. Default: <project>/data/labeled",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing labeled PKL files.",
    )
    args = parser.parse_args()

    subject_dirs = sorted(
        [path for path in args.raw_dir.iterdir() if path.is_dir() and path.name.startswith("S")],
        key=subject_sort_key,
    )

    if not subject_dirs:
        raise FileNotFoundError(f"no subject folders found in {args.raw_dir}")

    print(f"raw_dir: {args.raw_dir}")
    print(f"labeled_dir: {args.labeled_dir}")
    print("mapping: original 2 -> 0 stress, original 1 -> 1 non-stress, others -> 2 ignore")
    print(f"subjects: {len(subject_dirs)}")
    print()

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        raw_pkl_path = subject_dir / f"{subject}.pkl"
        labeled_pkl_path = args.labeled_dir / subject / f"{subject}.pkl"

        if not raw_pkl_path.exists():
            print(f"[skip] {subject}: missing {raw_pkl_path}")
            continue

        result = process_subject(
            raw_pkl_path=raw_pkl_path,
            labeled_pkl_path=labeled_pkl_path,
            overwrite=args.overwrite,
        )

        print(f"[ok] {result['subject']}")
        print(f"  output: {result['labeled_pkl']}")
        print(f"  BVP shape: {result['bvp_shape']}")
        print(f"  label shape: {result['label_shape']}")
        print(f"  label_bvp shape: {result['label_bvp_shape']}")
        print(f"  label counts: {format_counts(result['label_counts'])}")
        print(f"  label_bvp counts: {format_counts(result['label_bvp_counts'])}")
        print()


if __name__ == "__main__":
    main()
