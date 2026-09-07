# Manual Dataset Placement

Use the same `DATA_ROOT` as in `configs/full_core.env`. After placing a dataset,
run its downloader and then its preparation script.

## TidyVoiceX2-ASV

Place the evaluation package at:

```text
${DATA_ROOT}/tidyvoice/raw_data/TidyVoiceX2_ASV/
```

It must include the evaluation audio and:

```text
TidyVoiceX2_ASV/trials/tv26_eval-A.txt
TidyVoiceX2_ASV/trials/tv26_eval-U.txt
```

```bash
./local/download_tidyvoice.sh --data "${DATA_ROOT}"
```

## VoxCeleb1

Place extracted audio under:

```text
${DATA_ROOT}/voxceleb/raw_data/wav/<speaker>/<video>/<utterance>.wav
```

Alternatively set `VOX1_ZIP_DIR` to a directory containing the official
`vox1_test_wav.zip`, `vox1_dev_wav.zip`, or split `vox1_dev_wav_parta*` files.

```bash
./local/download_voxceleb.sh --data "${DATA_ROOT}"
```

## Whisper40

Set:

```bash
WHISPER40_ARCHIVE="/path/to/whisper40.rar"
```

or place the archive at:

```text
${DATA_ROOT}/whisper40/download_data/whisper40.rar
```

Then run:

```bash
./local/download_whisper40.sh --data "${DATA_ROOT}"
```

The extracted data should contain `nature/` and `whsp/` trees.

## Hugging Face datasets

GLOBE, ESD, VoxKnesset, and VoxPopuli use the local Hugging Face configuration.
If direct access is unavailable, use an approved proxy or cache outside this
repository. Keep the relative GLOBE shard paths listed in
`resources/canonical/globe/shards.tsv`.
