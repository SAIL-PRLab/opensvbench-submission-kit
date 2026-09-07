# CN-Celeb

The kit does not include CN-Celeb audio or evaluation archives.

```bash
./local/download_cnceleb.sh --data "${DATA_ROOT}"
./local/prepare_cnceleb.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/cnceleb/raw_data`; prepared manifests and
trials go under `${DATA_ROOT}/cnceleb/prepared`.

Core scenarios: `cnceleb`, `cnceleb_genre`, and `cnceleb_short`.

Follow the upstream CN-Celeb terms.
