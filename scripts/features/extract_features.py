import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.signal import welch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "features" / "intervals"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features" / "train_features"

TIME_DOMAIN_FEATURE_NAMES = [
    "hrm_bpm",
    "hrsd_bpm",
    "mnn_ms",
    "pnn50_percent",
]

FREQUENCY_DOMAIN_FEATURE_NAMES = [
    "total_power_ms2",
]

NONLINEAR_FEATURE_NAMES = [
    "approx_entropy",
    "correlation_dimension_d2",
]

SHORT_TERM_HR_FEATURE_NAMES = [
    "reactivity_peak_rise_bpm",
    "short_hr_max_bpm",
    "short_hr_range_bpm",
]

FEATURE_NAMES = (
    TIME_DOMAIN_FEATURE_NAMES
    + FREQUENCY_DOMAIN_FEATURE_NAMES
    + NONLINEAR_FEATURE_NAMES
    + SHORT_TERM_HR_FEATURE_NAMES
)

HRV_INTERP_FS = 4.0
WELCH_NPERSEG = "full"
HRV_FREQ_BANDS = {
    "ulf": (0.0, 0.0033),
    "vlf": (0.0033, 0.04),
    "lf": (0.04, 0.15),
    "hf": (0.15, 0.4),
}

APEN_M = 2
APEN_R_FACTOR = 0.2

D2_EMBED_DIM = 10
D2_LAG = 1
D2_NUM_RADII = 20
D2_MIN_POINTS = 20

HR_BIN_SECONDS = 10
MIN_HR_PROFILE_COVERAGE = 0.8
EARLY_BASELINE_SECONDS = 60


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def as_float_array(values):
    return np.asarray(values, dtype=np.float64).reshape(-1)


def feature_hrm(nn_sec):
    nn_sec = as_float_array(nn_sec)
    if len(nn_sec) == 0:
        return np.nan
    hr_bpm = 60.0 / nn_sec
    return float(np.mean(hr_bpm))


def feature_hrsd(nn_sec):
    nn_sec = as_float_array(nn_sec)
    if len(nn_sec) < 2:
        return np.nan
    hr_bpm = 60.0 / nn_sec
    return float(np.std(hr_bpm, ddof=1))


def feature_mnn(nn_sec):
    nn_sec = as_float_array(nn_sec)
    if len(nn_sec) == 0:
        return np.nan
    nn_ms = nn_sec * 1000.0
    return float(np.mean(nn_ms))


def feature_pnn50(nn_sec):
    nn_sec = as_float_array(nn_sec)
    if len(nn_sec) < 2:
        return np.nan
    nn_ms = nn_sec * 1000.0
    successive_diff_ms = np.abs(np.diff(nn_ms))
    return float(100.0 * np.sum(successive_diff_ms > 50.0) / len(successive_diff_ms))


def extract_time_domain_features(nn_sec):
    return np.asarray(
        [
            feature_hrm(nn_sec),
            feature_hrsd(nn_sec),
            feature_mnn(nn_sec),
            feature_pnn50(nn_sec),
        ],
        dtype=np.float64,
    )


def band_power(frequencies, psd, low_hz, high_hz):
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def feature_total_power(nn_sec, interp_fs=HRV_INTERP_FS):
    nn_sec = as_float_array(nn_sec)
    if len(nn_sec) < 4:
        return np.nan

    nn_ms = nn_sec * 1000.0
    beat_times = np.cumsum(nn_sec)
    beat_times = beat_times - beat_times[0]

    if beat_times[-1] <= 0:
        return np.nan

    uniform_times = np.arange(0, beat_times[-1], 1.0 / interp_fs)
    if len(uniform_times) < 4:
        return np.nan

    tachogram = np.interp(uniform_times, beat_times, nn_ms)
    tachogram = tachogram - np.mean(tachogram)

    nperseg = len(tachogram) if WELCH_NPERSEG == "full" else min(WELCH_NPERSEG, len(tachogram))
    frequencies, psd = welch(tachogram, fs=interp_fs, nperseg=nperseg)

    return float(
        sum(
            band_power(frequencies, psd, low_hz, high_hz)
            for low_hz, high_hz in HRV_FREQ_BANDS.values()
        )
    )


def extract_frequency_domain_features(nn_sec):
    return np.asarray(
        [
            feature_total_power(nn_sec),
        ],
        dtype=np.float64,
    )


def _phi_approx_entropy(values, m, r):
    n = len(values)
    templates = np.array([values[i : i + m] for i in range(n - m + 1)])
    if len(templates) == 0:
        return np.nan

    counts = []
    for template in templates:
        distances = np.max(np.abs(templates - template), axis=1)
        counts.append(np.mean(distances <= r))

    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    return float(np.mean(np.log(counts)))


def feature_approximate_entropy(nn_sec, m=APEN_M, r_factor=APEN_R_FACTOR):
    nn_sec = as_float_array(nn_sec)
    if len(nn_sec) <= m + 1:
        return np.nan

    nn_ms = nn_sec * 1000.0
    std = np.std(nn_ms, ddof=0)
    if std <= 0:
        return np.nan

    r = r_factor * std
    phi_m = _phi_approx_entropy(nn_ms, m, r)
    phi_m1 = _phi_approx_entropy(nn_ms, m + 1, r)
    if np.isnan(phi_m) or np.isnan(phi_m1):
        return np.nan
    return float(phi_m - phi_m1)


def delay_embedding(values, embed_dim, lag):
    values = as_float_array(values)
    n_vectors = len(values) - (embed_dim - 1) * lag
    if n_vectors <= 0:
        return np.empty((0, embed_dim), dtype=np.float64)
    return np.column_stack(
        [values[i * lag : i * lag + n_vectors] for i in range(embed_dim)]
    )


def feature_correlation_dimension(
    nn_sec,
    embed_dim=D2_EMBED_DIM,
    lag=D2_LAG,
    num_radii=D2_NUM_RADII,
    min_points=D2_MIN_POINTS,
):
    nn_sec = as_float_array(nn_sec)
    if len(nn_sec) < min_points:
        return np.nan

    nn_ms = nn_sec * 1000.0
    embedded = delay_embedding(nn_ms, embed_dim, lag)
    if len(embedded) < min_points:
        return np.nan

    distances = pdist(embedded, metric="euclidean")
    distances = distances[distances > 0]
    if len(distances) < min_points:
        return np.nan

    low = np.percentile(distances, 10)
    high = np.percentile(distances, 90)
    if low <= 0 or high <= low:
        return np.nan

    radii = np.logspace(np.log10(low), np.log10(high), num_radii)
    corr_sums = np.asarray([np.mean(distances < radius) for radius in radii])

    valid = (corr_sums > 0) & (corr_sums < 1)
    if np.sum(valid) < 2:
        return np.nan

    slope, _ = np.polyfit(np.log(radii[valid]), np.log(corr_sums[valid]), 1)
    return float(slope)


def extract_nonlinear_features(nn_sec):
    return np.asarray(
        [
            feature_approximate_entropy(nn_sec),
            feature_correlation_dimension(nn_sec),
        ],
        dtype=np.float64,
    )


def heart_rate_profile(
    peak_indices,
    bvp_hz,
    window_seconds,
    bin_seconds=HR_BIN_SECONDS,
):
    profile_size = int(round(window_seconds / bin_seconds))
    profile = np.full(profile_size, np.nan, dtype=np.float64)

    peaks = np.unique(np.asarray(peak_indices, dtype=np.float64).reshape(-1))
    peaks = peaks[np.isfinite(peaks)]
    if len(peaks) < 2:
        return profile

    nn_seconds = np.diff(peaks) / bvp_hz
    beat_seconds = peaks[1:] / bvp_hz
    valid = np.isfinite(nn_seconds) & (nn_seconds > 0)
    heart_rate = 60.0 / nn_seconds[valid]
    beat_seconds = beat_seconds[valid]

    for index in range(profile_size):
        lower = index * bin_seconds
        upper = (index + 1) * bin_seconds
        in_bin = (beat_seconds >= lower) & (beat_seconds < upper)
        if np.any(in_bin):
            profile[index] = np.median(heart_rate[in_bin])

    minimum_bins = int(np.ceil(profile_size * MIN_HR_PROFILE_COVERAGE))
    finite = np.isfinite(profile)
    if np.sum(finite) >= minimum_bins:
        positions = np.arange(profile_size)
        profile = np.interp(positions, positions[finite], profile[finite])

    return profile


def extract_short_term_hr_features(peak_indices, bvp_hz, window_seconds):
    heart_rate = heart_rate_profile(
        peak_indices,
        bvp_hz=bvp_hz,
        window_seconds=window_seconds,
    )
    if not np.all(np.isfinite(heart_rate)):
        return np.full(
            len(SHORT_TERM_HR_FEATURE_NAMES),
            np.nan,
            dtype=np.float64,
        )

    early_bins = max(1, int(round(EARLY_BASELINE_SECONDS / HR_BIN_SECONDS)))
    early_heart_rate = np.median(heart_rate[:early_bins])
    maximum_heart_rate = np.max(heart_rate)

    return np.asarray(
        [
            maximum_heart_rate - early_heart_rate,
            maximum_heart_rate,
            np.ptp(heart_rate),
        ],
        dtype=np.float64,
    )


def extract_features(nn_sec, peak_indices, bvp_hz, window_seconds):
    return np.concatenate(
        [
            extract_time_domain_features(nn_sec),
            extract_frequency_domain_features(nn_sec),
            extract_nonlinear_features(nn_sec),
            extract_short_term_hr_features(
                peak_indices,
                bvp_hz,
                window_seconds,
            ),
        ]
    )


def process_subject(npz_path, output_path, overwrite=False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} "
            "(use --overwrite to replace it)"
        )

    data = np.load(npz_path, allow_pickle=True)
    nn_intervals_sec = data["nn_intervals_sec"]
    corrected_peak_indices = data["corrected_peak_indices"]
    y = data["y"].astype(np.int8)
    bvp_hz = float(np.asarray(data["bvp_hz"]).item())
    window_seconds = float(np.asarray(data["window_seconds"]).item())

    if len(nn_intervals_sec) != len(corrected_peak_indices):
        raise ValueError(f"window alignment mismatch in {npz_path}")

    feature_rows = [
        extract_features(
            window_nn_sec,
            window_peak_indices,
            bvp_hz,
            window_seconds,
        )
        for window_nn_sec, window_peak_indices in zip(
            nn_intervals_sec,
            corrected_peak_indices,
        )
    ]
    X_features = np.vstack(feature_rows).astype(np.float64)
    feature_names = np.asarray(FEATURE_NAMES)
    nan_rows = np.any(np.isnan(X_features), axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_features,
        y=y,
        feature_names=feature_names,
        nan_rows=nan_rows,
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
        hrv_interp_fs=np.asarray(HRV_INTERP_FS),
        welch_nperseg=np.asarray(str(WELCH_NPERSEG)),
        frequency_bands=np.asarray(
            [f"{name}:{low}-{high}" for name, (low, high) in HRV_FREQ_BANDS.items()]
        ),
        apen_m=np.asarray(APEN_M),
        apen_r_factor=np.asarray(APEN_R_FACTOR),
        d2_embed_dim=np.asarray(D2_EMBED_DIM),
        d2_lag=np.asarray(D2_LAG),
        d2_num_radii=np.asarray(D2_NUM_RADII),
        d2_min_points=np.asarray(D2_MIN_POINTS),
        short_term_hr_feature_names=np.asarray(SHORT_TERM_HR_FEATURE_NAMES),
        hr_bin_seconds=np.asarray(HR_BIN_SECONDS),
        minimum_hr_profile_coverage=np.asarray(MIN_HR_PROFILE_COVERAGE),
        early_baseline_seconds=np.asarray(EARLY_BASELINE_SECONDS),
        source_npz=np.asarray(str(npz_path)),
    )

    return {
        "subject": str(np.asarray(data["subject"]).item()),
        "input_npz": str(npz_path),
        "output_npz": str(output_path),
        "windows": int(len(y)),
        "features": int(X_features.shape[1]),
        "nan_windows": int(np.sum(nan_rows)),
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
        "features",
        "nan_windows",
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
            "Extract baseline and short-term HR feature matrices from "
            "corrected NN interval files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing NN interval NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write feature NPZ files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing feature NPZ files and summary.csv.",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.input_dir.glob("S*.npz"), key=subject_sort_key)
    if not npz_paths:
        raise FileNotFoundError(f"no interval NPZ files found in {args.input_dir}")

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
    print(f"features: {', '.join(FEATURE_NAMES)}")
    print(f"subjects: {len(npz_paths)}")
    print()

    for npz_path in npz_paths:
        output_path = args.output_dir / npz_path.name
        row = process_subject(npz_path, output_path, overwrite=args.overwrite)
        rows.append(row)

        print(f"[ok] {row['subject']}")
        print(f"  output: {row['output_npz']}")
        print(f"  windows: {row['windows']}")
        print(f"  features: {row['features']}")
        print(f"  NaN windows: {row['nan_windows']}")
        print()

    summary_path = args.output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
