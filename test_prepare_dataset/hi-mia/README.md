# HI-MIA

The kit does not include HI-MIA audio.

```bash
./local/download_hi-mia.sh --data "${DATA_ROOT}"
./local/prepare_hi-mia.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/hi-mia/raw_data`; prepared manifests,
short-utterance files, and trials go under `${DATA_ROOT}/hi-mia/prepared`.

Core scenario: `hi_mia`, built from the public test split.

Follow the upstream HI-MIA/OpenSLR terms.
