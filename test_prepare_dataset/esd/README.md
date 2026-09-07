# Emotional Speech Dataset (ESD)

The kit does not include ESD audio. The downloader accepts the upstream
Hugging Face files or a local zip.

```bash
./local/download_esd.sh --data "${DATA_ROOT}"
./local/prepare_esd.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/esd/raw_data`; prepared manifests and trials
go under `${DATA_ROOT}/esd/prepared`.

Core scenario: `esd`.

Follow the upstream ESD terms.
