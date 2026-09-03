# Outputs

This directory contains generated artifacts and is not version controlled.

Running `scripts/train/train_svm.py` creates:

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
