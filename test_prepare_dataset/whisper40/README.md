# Whisper40

The kit does not include the Whisper40 archive or audio. Set
`WHISPER40_ARCHIVE` to the upstream `whisper40.rar`, or place it under
`${DATA_ROOT}/whisper40/download_data/`.

```bash
./local/download_whisper40.sh --data "${DATA_ROOT}"
./local/prepare_whisper40.sh --data "${DATA_ROOT}"
```

Prepared manifests and trials go under `${DATA_ROOT}/whisper40/prepared`.
The extracted archive should provide `nature/` and `whsp/` trees.

Core scenario: `whisper40_whisper`.
