# Changelog

## 0.2.1 - 2026-10-18

- Freeze the VoxCeleb Short and CN-Celeb Short crop source IDs, sample offsets,
  lengths, sample rates, and decoded PCM SHA-256 fingerprints.
- Verify all canonical short audio before precheck, scoring, or embedding-cache
  reuse.
- Invalidate only stale short-utterance entries in legacy embedding caches.
