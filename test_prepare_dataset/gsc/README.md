# Google Speech Commands

The kit does not include Speech Commands audio.

```bash
./local/download_gsc.sh --data "${DATA_ROOT}"
./local/prepare_gsc.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/gsc/raw_data`; prepared manifests and trials
go under `${DATA_ROOT}/gsc/prepared`.

Core scenario: `gsc_short`，基于官方 testing split 构建。默认 preparation 会
应用 `resources/canonical/gsc/` 下已提交的 canonical resources：

- `manifest.tsv`: 10,000 utterances from 250 speakers;
- `trials_short.txt`: 220,000 rows;
- trial SHA-256:
  `24c93fc5799f2a24afa603eaf410dfc24e9bd70ec896e41e2877e551bacf0229`.

脚本会校验压缩包 checksum、`testing_list.txt`、每个 canonical 音频路径和 trial
指纹，不会随机 subsample 或重新生成 trial。自定义随机子集仅供实验使用：

```bash
./local/prepare_gsc.sh \
  --data "${DATA_ROOT}" \
  --legacy-random true \
  --max-utts 10000
```

legacy 输出不适用于 core leaderboard protocol。所有固定哈希见
`resources/canonical/gsc/README.md`。

Follow the upstream Speech Commands terms.
