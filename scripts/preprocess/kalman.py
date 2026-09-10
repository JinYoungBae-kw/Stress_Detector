import argparse
import pickle
from pathlib import Path

import numpy as np

from common import format_signal_summary, save_pickle, subject_sort_key, summarize_signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "bandpass"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "kalman"

PROCESS_VARIANCE = 1e-5
MEASUREMENT_VARIANCE = 1e-2
INITIAL_ESTIMATE_ERROR = 1.0


def kalman_filter_1d(
    values,
    process_variance=PROCESS_VARIANCE,
    measurement_variance=MEASUREMENT_VARIANCE,
    initial_estimate_error=INITIAL_ESTIMATE_ERROR,
):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return values

    estimates = np.empty_like(values, dtype=np.float64)
    estimate = values[0]
    estimate_error = initial_estimate_error

    for index, measurement in enumerate(values):
        prediction = estimate
        prediction_error = estimate_error + process_variance

        kalman_gain = prediction_error / (prediction_error + measurement_variance)
        estimate = prediction + kalman_gain * (measurement - prediction)
        estimate_error = (1.0 - kalman_gain) * prediction_error

        estimates[index] = estimate

    return estimates


def process_subject(input_pkl_path, output_pkl_path, overwrite=False):
    if output_pkl_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_pkl_path} "
            "(use --overwrite to replace it)"
        )

    with input_pkl_path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    bvp = np.asarray(data["signal"]["wrist"]["BVP"])
    original_shape = bvp.shape
    before = summarize_signal(bvp)

    filtered_bvp = kalman_filter_1d(bvp).reshape(original_shape).astype(np.float64)
    after = summarize_signal(filtered_bvp)

    data["signal"]["wrist"]["BVP"] = filtered_bvp

    save_pickle(data, output_pkl_path)

    return {
        "subject": data.get("subject", input_pkl_path.stem),
        "input_pkl": input_pkl_path,
        "output_pkl": output_pkl_path,
        "shape": original_shape,
        "before": before,
        "after": after,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply conservative 1D Kalman filtering to wrist BVP."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=(
            "Directory containing bandpass subject PKL files. "
            "Default: <project>/data/preprocessed/bandpass"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Directory to write Kalman-filtered subject PKL files. "
            "Default: <project>/data/preprocessed/kalman"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Kalman PKL files.",
    )
    args = parser.parse_args()

    pkl_paths = sorted(args.input_dir.glob("S*/S*.pkl"), key=subject_sort_key)
    if not pkl_paths:
        raise FileNotFoundError(f"no bandpass PKL files found in {args.input_dir}")

    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(
        "filter: 1D Kalman "
        f"process_variance={PROCESS_VARIANCE}, "
        f"measurement_variance={MEASUREMENT_VARIANCE}, "
        f"initial_estimate_error={INITIAL_ESTIMATE_ERROR}"
    )
    print(f"subjects: {len(pkl_paths)}")
    print()

    for input_pkl_path in pkl_paths:
        subject = input_pkl_path.stem
        output_pkl_path = args.output_dir / subject / f"{subject}.pkl"

        result = process_subject(
            input_pkl_path=input_pkl_path,
            output_pkl_path=output_pkl_path,
            overwrite=args.overwrite,
        )

        print(f"[ok] {result['subject']}")
        print(f"  input: {result['input_pkl']}")
        print(f"  output: {result['output_pkl']}")
        print(f"  BVP shape: {result['shape']}")
        print(f"  before: {format_signal_summary(result['before'])}")
        print(f"  after:  {format_signal_summary(result['after'])}")
        print()


if __name__ == "__main__":
    main()
