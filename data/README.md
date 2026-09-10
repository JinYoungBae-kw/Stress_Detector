# Data

The WESAD dataset and all derived data are intentionally excluded from this
repository.

Download WESAD from the [official dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
and extract the subject folders into `data/raw`:

```text
data/
|-- raw/
|   |-- S2/S2.pkl
|   |-- S3/S3.pkl
|   |-- ...
|   `-- S17/S17.pkl
|-- labeled/S*/S*.pkl
|-- preprocessed/
|   |-- bandpass/S*/S*.pkl
|   |-- kalman/S*/S*.pkl
|   |-- moving_average/S*/S*.pkl
|   `-- all_preprocessed/S*/S*.pkl
|-- windowed/S*.npz
`-- features/
    |-- peaks/S*.npz
    |-- intervals/S*.npz
    `-- train_features/S*.npz
```

The pipeline creates every directory except `data/raw`. The expected WESAD
cohort contains 15 subject files: S2-S11 and S13-S17. The final
`train_features` files contain seven baseline features and three added
short-term HR features in one matrix.

All raw and derived files under `data` are ignored by Git. Only this README is
tracked. Use `--overwrite` when intentionally regenerating an existing stage.

WESAD is available for scientific, non-commercial use under the terms stated
by its owners. It is not covered by this repository's MIT license. Only load
trusted PKL files because Python pickle files can execute code while loading.
