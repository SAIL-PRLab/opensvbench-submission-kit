# Changelog

## 0.2.2 - 2026-10-27

- 将历史 10,000 条 GSC manifest 及对应的 `24c93fc5...` trial 文件提交到仓库，
  作为唯一 canonical `gsc_short` protocol。
- canonical GSC preparation 在写入 metadata 前校验官方压缩包、testing list、
  manifest、源文件路径和冻结 trial 指纹。
- 旧的随机 GSC 子集/trial 生成器仅通过显式的 `--legacy-random true` 开启。

## 0.2.1 - 2026-10-18

- Freeze the VoxCeleb Short and CN-Celeb Short crop source IDs, sample offsets,
  lengths, sample rates, and decoded PCM SHA-256 fingerprints.
- Verify all canonical short audio before precheck, scoring, or embedding-cache
  reuse.
- Invalidate only stale short-utterance entries in legacy embedding caches.
