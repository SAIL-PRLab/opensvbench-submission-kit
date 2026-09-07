# Google Speech Commands

The kit does not include Speech Commands audio.

```bash
./local/download_gsc.sh --data "${DATA_ROOT}"
./local/prepare_gsc.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/gsc/raw_data`; prepared manifests and trials
go under `${DATA_ROOT}/gsc/prepared`.

Core scenario: `gsc_short`, built from the official testing split.

Follow the upstream Speech Commands terms.
