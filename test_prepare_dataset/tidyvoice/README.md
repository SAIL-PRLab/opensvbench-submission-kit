# TidyVoiceX2-ASV

The kit does not include the TidyVoiceX2-ASV evaluation package. Place it at:

```text
${DATA_ROOT}/tidyvoice/raw_data/TidyVoiceX2_ASV/
```

Then run:

```bash
./local/download_tidyvoice.sh --data "${DATA_ROOT}"
./local/prepare_tidyvoice.sh --data "${DATA_ROOT}"
```

Prepared manifests and trials go under `${DATA_ROOT}/tidyvoice/prepared`.
Core scenario: `tidyvoice`.

Follow the upstream TidyVoiceX2-ASV terms.
