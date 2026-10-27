# OpenSVBench Submission Kit

Local tools for preparing benchmark data, generating trials, extracting speaker
embeddings, scoring systems, and creating an archive for the OpenSVBench
leaderboard.

Leaderboard: <https://opensvbench.xyz/>

## Contents

- `local/`: dataset download, placement, and preparation scripts
- `tools/`: trial generation, embedding extraction, scoring, and checks
- `scripts/`: entrypoints for the full core workflow
- `resources/`: frozen trial and canonical short-audio protocol resources
- `test_prepare_dataset/`: short notes for each dataset

The kit does not include audio, model weights, embedding caches, databases,
credentials, or previous submissions. Each dataset remains subject to its own
upstream license and terms of use.

## Quick start

Install the common dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create the configuration file:

```bash
cp configs/full_core.example.env configs/full_core.env
```

Set at least these fields:

```bash
DATA_ROOT="/path/to/opensvbench_data"
MODEL_NAME="/path/to/model"
BACKEND="wespeaker"
GPU=0
SUBMISSION_NAME="my-model"
```

Run the workflow in order:

```bash
./scripts/00_check_env.sh
./scripts/01_download_full_core.sh
./scripts/02_prepare_full_core.sh
./scripts/03_precheck_full_core.sh
./scripts/04_generate_submission_zip.sh
```

GSC `gsc_short` 默认使用仓库中提交的 canonical manifest 和 trial 文件。准备
流程会校验官方压缩包，不会重新采样 utterance 或重新生成 trial。固定的哈希值
以及仅供实验使用的 legacy 随机模式见 `resources/canonical/gsc/README.md`。

Keep `DATA_ROOT` outside the kit if possible. Data preparation can require
substantial disk space, and some datasets require an application or manual
placement. See `DATASET_ACCESS_MATRIX.md` and `MANUAL_PLACEMENT.md`.

## Full-core and partial modes

The default is a full-core submission covering all 26 current core trial sets.
Use partial mode when testing only a subset:

```bash
python3 tools/generate_submission.py \
  --data "/path/to/opensvbench_data" \
  --model-name "/path/to/model" \
  --backend wespeaker \
  --submission-mode partial \
  --scenario-ids tidyvoice globe
```

Partial submissions are useful for local checks and are not complete core
results for the public ranking.

## Output and upload

`scripts/04_generate_submission_zip.sh` writes a submission directory and zip
under `OUTPUT_DIR`. The upload archive contains only:

```text
submission.json
scores/*.csv
```

Local metrics, precheck reports, generation reports, and embedding caches stay
outside the upload archive. See `scripts/05_upload_instructions.md`.

## Frozen trials

The current core collection has 26 trial sets. Their fingerprints are recorded
in `resources/core_trial_fingerprints.tsv`. Run the precheck before generating a
submission. If a fingerprint does not match, prepare that dataset again rather
than editing the score CSV by hand.

See `TRIAL_SET_GUIDE.md` for the data split, protocol source, construction
method, and prepared trial path for every core trial set.

对于 `gsc_short`，canonical utterance 子集提交在
`resources/canonical/gsc/manifest.tsv`，canonical 220,000 行 trial 文件提交在
`resources/canonical/gsc/trials_short.txt`。对应的 trial SHA-256 为
`24c93fc5799f2a24afa603eaf410dfc24e9bd70ec896e41e2877e551bacf0229`。

Some protocols repeat the same utterance pair. The kit writes one score for
each unique pair; the leaderboard expands it against the frozen trial file.

VoxCeleb Short and CN-Celeb Short also use frozen crop manifests under
`resources/canonical/short_crops/`. The manifests fix the source utterance,
start sample, length, sample rate, and decoded PCM SHA-256 for all 40,000 short
utterances. Data preparation reuses matching files, creates missing files, and
stops on any mismatch. Submission precheck validates these PCM fingerprints
before cached embeddings can be reused.

## Datasets

The kit provides download and preparation code but does not redistribute source
audio. Read `DATASET_ACCESS_MATRIX.md` before running the scripts and follow
each dataset's license, access restrictions, and citation requirements. GLOBE
uses the fixed manifest, shard list, and trial file under
`resources/canonical/globe/` so that users run the same protocol.
GSC 同样使用 `resources/canonical/gsc/` 下的固定 manifest 和 trial 文件；源音频
仍需从上游 Speech Commands 压缩包下载。

## License

The code and documentation in this kit are released under the Apache License
2.0; see `LICENSE`. This does not grant additional rights to datasets, models,
or other upstream files.
