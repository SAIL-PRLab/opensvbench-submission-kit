# Dataset Access Matrix

工具包不包含任何上游音频。下表只说明脚本会做什么；下载、使用和再分发仍
以数据集所有者公布的条款为准。

| Dataset | Core use | Automatic step | Manual input |
| --- | --- | --- | --- |
| TidyVoiceX2-ASV | Cross-lingual SV | Check local placement | Evaluation package and audio |
| HI-MIA | Short-duration SV | Download public test split | None when OpenSLR is reachable |
| Google Speech Commands | Short-duration SV | Download and unpack | None |
| GLOBE | Accent robustness | Download pinned Hugging Face shards | Hugging Face access if required |
| CN-Celeb | Official, genre, and short-duration trials | Download evaluation package | Full audio if the package is unavailable |
| VoxCeleb1 | Vox1-O/E/H and short-duration trials | Download trial lists | Official audio |
| 3D-Speaker | Device, distance, and dialect trials | Download test package and metadata | None when the mirror is reachable |
| FFSVC 2022 | Cross-channel and cross-domain trials | Download public dev package | None when Zenodo is reachable |
| Whisper40 | Neutral and whispered speech | Check and unpack local archive | `whisper40.rar` |
| Lombard Grid | Plain and Lombard speech | Download and unpack public release | None when Zenodo is reachable |
| VOiCES | Noise and reverberation | Download public devkit | None when AWS is reachable |
| CHiME-6 | Far-field and overlap trials | Download public dev package | None when OpenSLR is reachable |
| ESD | Cross-emotion speech | Download from Hugging Face or use local zip | Local zip if needed |
| AliMeeting | Near/far and overlap trials | Download test package | None when the mirror is reachable |
| VoxKnesset | Longitudinal speaker verification | Check Hugging Face access | Proxy, cache, or local access may be needed |
| VoxPopuli | English cross-time trials | Download selected Hugging Face shards | Proxy, cache, or local access may be needed |

## Notes

- Read and accept the upstream terms before downloading or sharing any data.
- Keep raw and prepared data under `DATA_ROOT`, outside the repository when
  possible.
- The GLOBE protocol uses the fixed files under
  `resources/canonical/globe/`; do not replace them with a different subset.
- `LICENSE` applies to this kit's code and documentation, not to the datasets.
