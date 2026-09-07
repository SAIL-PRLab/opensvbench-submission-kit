# VoxPopuli

The kit downloads only the selected English data required by the core trial.

```bash
./local/download_voxpopuli.sh --data "${DATA_ROOT}"
./local/prepare_voxpopuli.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/voxpopuli/raw_data`; prepared manifests and
trials go under `${DATA_ROOT}/voxpopuli/prepared`.

Core scenario: `voxpopuli_cross_time`.

Follow the upstream VoxPopuli terms.
