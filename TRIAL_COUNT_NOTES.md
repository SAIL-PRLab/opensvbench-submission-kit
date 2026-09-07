# Frozen Trial Counts

These are the core trial files used by the current leaderboard. `Frozen rows` is
the row count in the fixed trial file; `Unique score rows` is the number of
unique pairs required in the submission CSV.

For the source split, protocol type, construction method, and prepared trial
path for each entry, see `TRIAL_SET_GUIDE.md`.

| Trial set | Frozen rows | Unique score rows |
| --- | ---: | ---: |
| tidyvoice | 200000 | 200000 |
| hi_mia | 660000 | 654706 |
| gsc_short | 220000 | 220000 |
| globe | 56848 | 56848 |
| cnceleb | 3484292 | 3484292 |
| cnceleb_genre | 440000 | 440000 |
| cnceleb_short | 546964 | 542958 |
| voxceleb_o | 37611 | 37274 |
| voxceleb_e | 579818 | 577310 |
| voxceleb_h | 550894 | 548491 |
| voxceleb_short | 394724 | 392927 |
| speaker3d_device | 180000 | 180000 |
| speaker3d_distance | 175163 | 175163 |
| speaker3d_dialect | 180000 | 180000 |
| ffsvc2022_cross_channel | 72000 | 71959 |
| ffsvc2022_cross_domain | 66546 | 66508 |
| whisper40_whisper | 17600 | 17600 |
| lombardgrid_lombard | 29524 | 29524 |
| voices_noise_reverb | 55000 | 55000 |
| chime6_domestic_farfield | 36487 | 36487 |
| chime6_overlap | 39600 | 39600 |
| esd | 437408 | 437408 |
| alimeeting_near_far | 220000 | 220000 |
| alimeeting_overlap | 165000 | 165000 |
| voxknesset | 158312 | 158312 |
| voxpopuli_cross_time | 146575 | 146575 |

Some trial files contain repeated canonical pairs. The kit writes one score per
unique pair, and the leaderboard expands it against the frozen file. Do not pad
the CSV manually.

The canonical GLOBE manifest, shard list, and trial file are under
`resources/canonical/globe/`. Run `./scripts/03_precheck_full_core.sh` to check
the local files and their fingerprints.
