# Outputs

Generated results are excluded from Git. Training the Baseline model first and
the Extended model second creates this layout:

```text
outputs/
|-- loso/
|   |-- metrics_difference.csv
|   |-- baseline/
|   |   |-- train_result/
|   |   |-- model/
|   |   `-- shap/
|   `-- extended/
|       |-- train_result/
|       |-- model/
|       `-- shap/
`-- paper/
    |-- tables/
    |-- figures/
    `-- paper_output_manifest.csv
```

Key files:

- `metrics_summary.csv`: mean LOSO performance and 95% confidence intervals.
- `metrics_by_subject.csv`: metrics for each held-out participant.
- `predictions.csv`: held-out predictions and stress decision scores.
- `metrics_difference.csv`: paired participant-bootstrap differences between
  the two feature sets.
- `shap_feature_importance.csv`: mean absolute SHAP ranking.
- `paper_output_manifest.csv`: list and meaning of generated summary artifacts.

The saved full-data model and its SHAP values are separate from held-out LOSO
performance. Use `--overwrite` only when intentionally replacing a completed
run.
