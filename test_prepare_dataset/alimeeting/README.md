# AliMeeting

The kit does not include AliMeeting audio or annotations.

```bash
./local/download_alimeeting.sh --data "${DATA_ROOT}"
./local/prepare_alimeeting.sh --data "${DATA_ROOT}"
```

Raw files go under `${DATA_ROOT}/alimeeting/raw_data`; prepared segments,
manifests, and trials go under `${DATA_ROOT}/alimeeting/prepared`.

Core scenarios: `alimeeting_near_far` and `alimeeting_overlap`.

Follow the upstream AliMeeting terms.
