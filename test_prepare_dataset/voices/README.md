# VOiCES

The kit does not include VOiCES audio.

```bash
./local/download_voices.sh --data "${DATA_ROOT}"
./local/prepare_voices.sh --data "${DATA_ROOT}"
```

The downloader extracts the files required by the core protocol. Raw files go
under `${DATA_ROOT}/voices/raw_data`; prepared manifests and trials go under
`${DATA_ROOT}/voices/prepared`.

Core scenario: `voices_noise_reverb`.

Follow the upstream VOiCES terms.
