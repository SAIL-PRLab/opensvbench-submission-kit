# VoxCeleb1

The kit does not include VoxCeleb audio. Place or unpack the official audio at:

```text
${DATA_ROOT}/voxceleb/raw_data/wav/<speaker>/<video>/<utterance>.wav
```

Then run:

```bash
./local/download_voxceleb.sh --data "${DATA_ROOT}"
./local/prepare_voxceleb.sh --data "${DATA_ROOT}"
```

Prepared manifests and trials go under `${DATA_ROOT}/voxceleb/prepared`.
Core scenarios: `voxceleb_o`, `voxceleb_e`, `voxceleb_h`, and
`voxceleb_short`.

Follow the official VoxCeleb terms.
