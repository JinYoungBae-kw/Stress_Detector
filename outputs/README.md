# Outputs

This directory contains generated artifacts and is not version controlled.

Running `scripts/train/train_svm.py --overwrite` with the final 10-feature
files creates:

```text
outputs/
|-- train_result/
|   |-- loso_results.csv
|   |-- predictions.csv
|   |-- summary.csv
|   `-- metadata.npz
|-- model/
|   |-- svm_pipeline.joblib
|   |-- svm_parameters.npz
|   `-- model_metadata.json
`-- shap/
    |-- shap_feature_importance.csv
    |-- shap_summary_bar.png
    |-- shap_summary_beeswarm.png
    `-- shap_values.npz
```

The files in `train_result` report held-out Leave-One-Subject-Out evaluation.
The model and SHAP files are generated after fitting one final pipeline on all
available subjects; they must not be interpreted as additional held-out
performance estimates.

`summary.csv` reports the unweighted mean, standard deviation, minimum, and
maximum across the 15 subject-level folds. Stress is label 0 and is treated as
the positive class for F1 and ROC AUC.

The expected final mean scores are:

| Metric | Mean |
|---|---:|
| Accuracy | 97.01% |
| Stress F1 | 94.88% |
| Stress ROC AUC | 99.98% |

All generated files under `outputs` are ignored by Git. Only this README is
tracked.
