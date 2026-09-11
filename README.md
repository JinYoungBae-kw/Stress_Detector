# Stress Detector

This project classifies stress from wrist BVP in the WESAD dataset. The
pipeline prepares the signal, corrects implausibly short NN intervals, extracts
seven baseline features and three short-term heart-rate features, and compares
the two feature sets with the same Leave-One-Subject-Out (LOSO) evaluation.

## Configuration

- Data: WESAD wrist BVP from 15 participants
- Sampling rate: 64 Hz
- Classes: stress versus baseline
- Windows: 360 seconds with a 30-second stride
- Model: `StandardScaler` and RBF SVM
- Evaluation: LOSO cross-validation
- Feature sets: Baseline (7) and Extended (10)

## Setup

The dataset is not included. Download WESAD from the
[official dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
and place the subject files under `data/raw`:

```text
data/raw/S2/S2.pkl
data/raw/S3/S3.pkl
...
data/raw/S17/S17.pkl
```

Create the environment from the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Run these commands from the repository root in order:

```powershell
python scripts/preprocess/label_simplification.py --overwrite
python scripts/preprocess/bandpass.py --overwrite
python scripts/preprocess/kalman.py --overwrite
python scripts/preprocess/moving_average.py --overwrite
python scripts/preprocess/hanning_windows.py --overwrite
python scripts/features/detect_peaks.py --overwrite
python scripts/features/nn_intervals.py --overwrite
python scripts/features/extract_features.py --overwrite
python scripts/train/train_svm.py --feature-set baseline --overwrite
python scripts/train/train_svm.py --feature-set extended --overwrite
python scripts/analysis/generate_paper_outputs.py --overwrite
```

The Extended run creates the paired Baseline-versus-Extended comparison after
both training runs are available.

## Pipeline

| Stage | Output |
| --- | --- |
| Label preparation | `data/labeled/S*/S*.pkl` |
| Band-pass, Kalman, and moving-average filtering | `data/preprocessed/` |
| Windowing | `data/windowed/S*.npz` |
| Peak and NN interval processing | `data/features/peaks/` and `data/features/intervals/` |
| Final feature extraction | `data/features/train_features/S*.npz` |
| Model evaluation | `outputs/loso/` |
| Summary tables and figures | `outputs/paper/` |

See [data/README.md](data/README.md) and
[outputs/README.md](outputs/README.md) for the generated layouts. All scripts
derive paths from the project root and support `--help` where command-line
options are available.

## Notes

- WESAD labels 2 and 1 are used as stress and non-stress, respectively. Other
  protocol labels are excluded.
- Baseline and Extended models use the same subjects, windows, model settings,
  and random seed. Only the selected features differ.
- The source dataset and all generated artifacts are ignored by Git.
- This is research software and is not intended for clinical use.

## License

Source code is released under the [MIT License](LICENSE). WESAD remains subject
to its original terms and is not covered by this repository's license.
