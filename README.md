# Physiologically Corrected Wrist-PPG Stress Classification

This repository contains the final wrist-PPG stress-classification pipeline
used in our study. It processes WESAD wrist blood volume pulse (BVP), corrects
detected peaks that produce physiologically implausible short NN intervals,
extracts seven HRV-related features, and evaluates an RBF SVM with
Leave-One-Subject-Out (LOSO) cross-validation.

The implementation is based on the signal-processing and feature framework
described by Jahanjoo et al. and retains only the final experimental
configuration used in this project.

## Study scope

- Dataset: WESAD, 15 subjects
- Sensor: Empatica E4 wrist BVP, sampled at 64 Hz
- Task: binary stress classification
- Labels: WESAD stress (2) vs. baseline (1)
- Validation: LOSO cross-validation
- Classifier: StandardScaler followed by an RBF SVM
- Final preprocessing: 0.5-3.5 Hz band-pass, Kalman filter, 3-point moving average
- Segmentation: 360-second Hann windows with a 30-second stride
- Peak correction: remove one adjacent peak when a detected NN interval is shorter than 0.30 seconds

Amusement and all other protocol labels are excluded from model training.

## Repository structure

```text
.
|-- data/
|   `-- README.md
|-- outputs/
|   `-- README.md
|-- scripts/
|   |-- preprocess/
|   |   |-- label_simplification.py
|   |   |-- bandpass.py
|   |   |-- kalman.py
|   |   |-- moving_average.py
|   |   `-- hanning_windows.py
|   |-- features/
|   |   |-- detect_peaks.py
|   |   |-- nn_intervals.py
|   |   `-- extract_features.py
|   `-- train/
|       `-- train_svm.py
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Dataset setup

The dataset is not redistributed in this repository. Download WESAD from the
[official WESAD page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
and place the original subject files as follows:

```text
data/raw/S2/S2.pkl
data/raw/S3/S3.pkl
...
data/raw/S17/S17.pkl
```

The expected subject IDs are S2-S11 and S13-S17. See
[data/README.md](data/README.md) for the data policy and directory details.

## Environment

The recorded environment uses Python 3.11.9. Create a virtual environment and
install the pinned dependencies.

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce the pipeline

Run the scripts from the repository root in this order:

```powershell
python scripts/preprocess/label_simplification.py --overwrite
python scripts/preprocess/bandpass.py --overwrite
python scripts/preprocess/kalman.py --overwrite
python scripts/preprocess/moving_average.py --overwrite
python scripts/preprocess/hanning_windows.py --overwrite
python scripts/features/detect_peaks.py --overwrite
python scripts/features/nn_intervals.py --overwrite
python scripts/features/extract_features.py --overwrite
python scripts/train/train_svm.py --overwrite
```

Each script also supports `--help`. Input and output directories can be
overridden through CLI arguments, except for the fixed physiological and model
parameters listed below.

## Processing stages

| Stage | Script | Main operation |
|---|---|---|
| Label preparation | `label_simplification.py` | Maps stress to 0, baseline to 1, and all other labels to ignore (2); aligns labels to 64 Hz BVP |
| Band-pass filtering | `bandpass.py` | Fourth-order Butterworth filter at 0.5-3.5 Hz |
| Kalman filtering | `kalman.py` | One-dimensional smoothing with process variance `1e-5` and measurement variance `1e-2` |
| Moving average | `moving_average.py` | Three-point centered moving average |
| Segmentation | `hanning_windows.py` | 360-second Hann windows, 30-second stride; mixed and ignored-label windows are removed |
| Peak detection | `detect_peaks.py` | Local maxima above the subject-level mean windowed-signal amplitude |
| Peak correction | `nn_intervals.py` | Corrects short NN intervals by deleting the locally less plausible adjacent peak |
| Feature extraction | `extract_features.py` | Produces seven time-, frequency-, and nonlinear-domain features |
| Classification | `train_svm.py` | LOSO evaluation and final full-data SVM/SHAP generation |

Long NN intervals above 2.00 seconds are reported for quality assessment but
are not corrected.

## Extracted features

| Feature | Description |
|---|---|
| `hrm_bpm` | Mean heart rate |
| `hrsd_bpm` | Standard deviation of heart rate |
| `mnn_ms` | Mean NN interval |
| `pnn50_percent` | Percentage of successive NN differences greater than 50 ms |
| `total_power_ms2` | Sum of ULF, VLF, LF, and HF spectral power |
| `approx_entropy` | Approximate entropy of the NN sequence |
| `correlation_dimension_d2` | Correlation-dimension estimate of NN dynamics |

Rows containing non-finite feature values are excluded inside each LOSO fold
before model fitting and evaluation.

## Model and evaluation

The classifier is:

```text
StandardScaler
-> SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=None, random_state=42)
```

Stress is label 0 and is explicitly treated as the positive class for F1 and
ROC AUC. Reported values are the mean of the 15 subject-level LOSO folds.

| Metric | Mean |
|---|---:|
| Accuracy | 94.08% |
| Stress F1 | 88.96% |
| Stress ROC AUC | 99.91% |

The final full-data model is generated only after LOSO evaluation. Its SHAP
results explain that fitted model and are not used to calculate the LOSO
performance. See [outputs/README.md](outputs/README.md) for generated files.

## Reproducibility notes

- The random seed is 42.
- The default SVM uses no class weighting.
- All scientific Python dependencies are pinned in `requirements.txt`.
- Intermediate data and trained artifacts are ignored by Git.
- Reproducing the table requires the unchanged original WESAD PKL files and
  the default parameters in the scripts.
- The repository contains research code and is not a medical device.

## References

1. A. Jahanjoo, N. TaheriNejad, and A. Aminifar, "High-Accuracy Stress
   Detection Using Wrist-Worn PPG Sensors," *2024 IEEE International Symposium
   on Circuits and Systems (ISCAS)*, pp. 1-5, 2024.
   [doi:10.1109/ISCAS58744.2024.10558012](https://doi.org/10.1109/ISCAS58744.2024.10558012)
2. P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, and K. Van Laerhoven,
   "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect
   Detection," *Proceedings of the 20th ACM International Conference on
   Multimodal Interaction*, pp. 400-408, 2018.
   [doi:10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)

## License

The source code is released under the [MIT License](LICENSE). WESAD data is not
included and remains subject to the original dataset terms.
