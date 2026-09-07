# FFSVC 2022

The kit does not include FFSVC audio.

```bash
./local/download_ffsvc2022.sh --data "${DATA_ROOT}"
./local/prepare_ffsvc2022.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/ffsvc2022/raw_data`; prepared manifests and
trials go under `${DATA_ROOT}/ffsvc2022/prepared`.

Core scenarios: `ffsvc2022_cross_channel` and `ffsvc2022_cross_domain`.

Follow the upstream FFSVC 2022 terms.
