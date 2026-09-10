import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np

from common import format_signal_summary, save_pickle, subject_sort_key, summarize_signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "kalman"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "moving_average"
FINAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "all_preprocessed"

WINDOW_SIZE = 3


def moving_average(values, window_size=WINDOW_SIZE):
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered moving average")

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0 or window_size == 1:
        return values

    kernel = np.ones(window_size, dtype=np.float64) / window_size
    pad_width = window_size // 2
    padded = np.pad(values, pad_width, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def process_subject(
    input_pkl_path,
    output_pkl_path,
    final_output_pkl_path,
    overwrite=False,
):
    for path in [output_pkl_path, final_output_pkl_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output already exists: {path} "
                "(use --overwrite to replace it)"
            )

    with input_pkl_path.open("rb") as file:
        data = pickle.load(file, encoding="latin1")

    bvp = np.asarray(data["signal"]["wrist"]["BVP"])
    original_shape = bvp.shape
    before = summarize_signal(bvp)

    filtered_bvp = moving_average(bvp).reshape(original_shape).astype(np.float64)
    after = summarize_signal(filtered_bvp)

    data["signal"]["wrist"]["BVP"] = filtered_bvp

    save_pickle(data, output_pkl_path)
    final_output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_pkl_path, final_output_pkl_path)

    return {
        "subject": data.get("subject", input_pkl_path.stem),
        "input_pkl": input_pkl_path,
        "output_pkl": output_pkl_path,
        "final_output_pkl": final_output_pkl_path,
        "shape": original_shape,
        "before": before,
        "after": after,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply 3-point moving average smoothing to wrist BVP."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=(
            "Directory containing Kalman-filtered subject PKL files. "
            "Default: <project>/data/preprocessed/kalman"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Directory to write moving-average subject PKL files. "
            "Default: <project>/data/preprocessed/moving_average"
        ),
    )
    parser.add_argument(
        "--final-output-dir",
        type=Path,
        default=FINAL_OUTPUT_DIR,
        help=(
            "Directory to write final all-preprocessed subject PKL files. "
            "Default: <project>/data/preprocessed/all_preprocessed"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing moving-average and final PKL files.",
    )
    args = parser.parse_args()

    pkl_paths = sorted(args.input_dir.glob("S*/S*.pkl"), key=subject_sort_key)
    if not pkl_paths:
        raise FileNotFoundError(f"no Kalman PKL files found in {args.input_dir}")

    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"final_output_dir: {args.final_output_dir}")
    print(f"filter: {WINDOW_SIZE}-point centered moving average")
    print(f"subjects: {len(pkl_paths)}")
    print()

    for input_pkl_path in pkl_paths:
        subject = input_pkl_path.stem
        output_pkl_path = args.output_dir / subject / f"{subject}.pkl"
        final_output_pkl_path = args.final_output_dir / subject / f"{subject}.pkl"

        result = process_subject(
            input_pkl_path=input_pkl_path,
            output_pkl_path=output_pkl_path,
            final_output_pkl_path=final_output_pkl_path,
            overwrite=args.overwrite,
        )

        print(f"[ok] {result['subject']}")
        print(f"  input: {result['input_pkl']}")
        print(f"  output: {result['output_pkl']}")
        print(f"  final: {result['final_output_pkl']}")
        print(f"  BVP shape: {result['shape']}")
        print(f"  before: {format_signal_summary(result['before'])}")
        print(f"  after:  {format_signal_summary(result['after'])}")
        print()


if __name__ == "__main__":
    main()
