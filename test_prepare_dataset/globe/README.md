# GLOBE

The kit includes the canonical manifest, shard list, and trial file, but not
the source audio.

```bash
./local/download_globe.sh --data "${DATA_ROOT}"
./local/prepare_globe.sh --data "${DATA_ROOT}"
```

The downloader uses the shard paths in
`resources/canonical/globe/shards.tsv`. Raw audio goes under
`${DATA_ROOT}/globe/raw_data`; prepared files and the frozen trial go under
`${DATA_ROOT}/globe/prepared`.

Core scenario: `globe`.

Follow the upstream GLOBE terms. The OpenSVBench trial file is maintained in
this kit and is not an upstream benchmark release.
