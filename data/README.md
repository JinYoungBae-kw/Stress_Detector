# Data

WESAD and all generated data are excluded from Git. Place the original 15
subject files in `data/raw` using their original folder names.

```text
data/
|-- raw/S*/S*.pkl
|-- labeled/S*/S*.pkl
|-- preprocessed/
|   |-- bandpass/S*/S*.pkl
|   |-- kalman/S*/S*.pkl
|   `-- moving_average/S*/S*.pkl
|-- windowed/S*.npz
`-- features/
    |-- peaks/S*.npz
    |-- intervals/S*.npz
    `-- train_features/S*.npz
```

The pipeline creates every directory except `data/raw`. Expected subjects are
S2-S11 and S13-S17. The final NPZ files contain the same 10 ordered features
for each participant.

WESAD is governed by its original terms and is not covered by this project's
MIT License. Only load trusted PKL files because pickle loading can execute
code.
