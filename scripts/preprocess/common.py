import pickle

import numpy as np


def subject_sort_key(path):
    name = path.stem
    if name.startswith("S") and name[1:].isdigit():
        return int(name[1:])
    return name


def summarize_signal(values):
    values = np.asarray(values).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def format_signal_summary(summary):
    return (
        f"mean={summary['mean']:.6f}, std={summary['std']:.6f}, "
        f"min={summary['min']:.6f}, max={summary['max']:.6f}"
    )


def save_pickle(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)
