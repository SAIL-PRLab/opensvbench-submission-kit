# 3D-Speaker

The kit does not include the 3D-Speaker audio or metadata.

```bash
./local/download_3dspeaker.sh --data "${DATA_ROOT}"
./local/prepare_3dspeaker.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/3dspeaker/raw_data`; prepared manifests and
trials go under `${DATA_ROOT}/3dspeaker/prepared`.

Core scenarios: `speaker3d_device`, `speaker3d_distance`, and
`speaker3d_dialect`.

Follow the upstream 3D-Speaker terms.
