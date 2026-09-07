# CHiME-6

The kit does not include CHiME-6 audio or transcripts.

```bash
./local/download_chime6.sh --data "${DATA_ROOT}"
./local/prepare_chime6.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/chime6/raw_data`; prepared manifests,
segments, and trials go under `${DATA_ROOT}/chime6/prepared`.

Core scenarios: `chime6_domestic_farfield` and `chime6_overlap`.

Follow the upstream CHiME-6 terms.
