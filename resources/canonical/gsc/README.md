# GSC canonical protocol

`gsc_short` 是基于 Google Speech Commands v0.02 官方 testing split 构建的
OpenSVBench 协议。本目录正式提交固定的 utterance 子集和 trial 行，确保所有
提交使用完全相同的评测输入。Submission kit 不重新分发源音频。

## 固定输入与资源

| 项目 | 数量 | SHA-256 |
| --- | ---: | --- |
| `speech_commands_v0.02.tar.gz` | 2,428,923,189 bytes | `af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58` |
| extracted `testing_list.txt` | 11,005 lines | `2d17c6b3faf63be43eda93cfeb0c747cfd79b7b236282039dbac65a2cb5f1df5` |
| `manifest.tsv` | 10,000 utterances / 250 speakers | `8c19839809cbd4e08a5266076c830e49fcb649524d928edb7af114908664dcb3` |
| `trials_short.txt` | 220,000 rows | `24c93fc5799f2a24afa603eaf410dfc24e9bd70ec896e41e2877e551bacf0229` |

官方压缩包地址：

```text
http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
```

`manifest.tsv` 的列为：

```text
utt_id    rel_path    speaker_id    word
```

`rel_path` 是相对于整理后的 `gsc/raw_data` 的路径。Submission kit 会把压缩包
整理为 `speaker/word/audio.wav`。该 manifest 是历史上与冻结的
`24c93fc5...` trial 文件对应的 10,000 条 utterance 子集，行顺序也已固定。

## 准备方式

正常命令会应用仓库中提交的资源，并依次校验源压缩包、testing list、manifest、
音频路径和 trial 指纹：

```bash
./local/download_gsc.sh --data "${DATA_ROOT}"
./local/prepare_gsc.sh --data "${DATA_ROOT}"
```

canonical 模式是 core protocol 唯一使用的模式，不会重新采样 utterance，也不会
重新生成 trial。为避免协议漂移，canonical 模式会拒绝非默认的
`--max-utts` 值。

如果只是做本地实验，仍可显式选择旧的随机生成器。该输出不属于 core
`gsc_short` protocol，不能用于 leaderboard 提交：

```bash
./local/prepare_gsc.sh \
  --data "${DATA_ROOT}" \
  --legacy-random true \
  --max-utts 10000
```

不要编辑 `trials_short.txt` 或替换仓库中的 manifest。如果 core 指纹检查失败，
应当从官方压缩包重新执行 canonical preparation，而不是修改 score 文件。
