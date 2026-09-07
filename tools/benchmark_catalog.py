#!/usr/bin/env python3
"""Shared benchmark dataset and scenario catalog for OpenSVBench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    collection: str
    benchmark_condition: str
    source_limit: str
    trial_source: str
    license_note: str
    download_method: str
    raw_rel_dir: str
    prepared_rel_dir: str
    metadata_files: tuple[str, ...]
    readme_rel_path: str
    doc_status: str
    notes: str

    def raw_dir(self, data_root: Path | None = None) -> Path:
        return (data_root or DEFAULT_DATA_ROOT) / self.raw_rel_dir

    def prepared_dir(self, data_root: Path | None = None) -> Path:
        return (data_root or DEFAULT_DATA_ROOT) / self.prepared_rel_dir

    def readme_path(self) -> Path:
        return REPO_ROOT / self.readme_rel_path


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    dataset_id: str
    display_name: str
    variant: str
    collection: str
    trial_rel_path: str
    prep_rel_dir: str
    submission_filename: str
    protocol_source: str = "Self-constructed protocol"
    source_split: str = "unspecified"
    protocol_note: str = ""

    def trial_path(self, data_root: Path | None = None) -> Path:
        return (data_root or DEFAULT_DATA_ROOT) / self.trial_rel_path

    def prep_dir(self, data_root: Path | None = None) -> Path:
        return (data_root or DEFAULT_DATA_ROOT) / self.prep_rel_dir

    def trial_metadata_path(self, data_root: Path | None = None) -> Path:
        return self.prep_dir(data_root) / f"trial_metadata_{self.scenario_id}.csv.gz"


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    display_name: str
    short_label: str
    collection: str
    summary: str
    lens: str
    scenario_ids: tuple[str, ...]


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="tidyvoice",
        display_name="TidyVoiceX2-ASV",
        collection="core",
        benchmark_condition="cross-lingual",
        source_limit="<=10000 clean source utterances",
        trial_source="Official TidyVoiceX2-ASV eval A/U keys with reproducible sub-sampling and matched negatives",
        license_note="Follow the official TidyVoice / TidyVoiceX terms before redistribution or public use.",
        download_method="Manual placement supported; local preparation expects DATA_ROOT/tidyvoice/raw_data/TidyVoiceX2_ASV.",
        raw_rel_dir="tidyvoice/raw_data",
        prepared_rel_dir="tidyvoice/prepared",
        metadata_files=("utt2spk", "utt2lang", "spk2utt", "stats_tidyvoice.json"),
        readme_rel_path="test_prepare_dataset/tidyvoice/README.md",
        doc_status="ready",
        notes="Current core cross-lingual evaluation set.",
    ),
    DatasetSpec(
        dataset_id="hi-mia",
        display_name="HI-MIA",
        collection="core",
        benchmark_condition="short-duration",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed from the public test split",
        license_note="See OpenSLR 85 licensing terms.",
        download_method="Automatic OpenSLR test_v2 download via local/download_hi-mia.sh.",
        raw_rel_dir="hi-mia/raw_data",
        prepared_rel_dir="hi-mia/prepared",
        metadata_files=("utt2spk", "spk2utt"),
        readme_rel_path="test_prepare_dataset/hi-mia/README.md",
        doc_status="ready",
        notes="Short-duration evaluation from the public HI-MIA test set.",
    ),
    DatasetSpec(
        dataset_id="gsc",
        display_name="Google Speech Commands",
        collection="core",
        benchmark_condition="short-duration",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed from the official testing split",
        license_note="See the TensorFlow Speech Commands license and distribution terms.",
        download_method="Automatic download via local/download_gsc.sh.",
        raw_rel_dir="gsc/raw_data",
        prepared_rel_dir="gsc/prepared",
        metadata_files=("utt2spk", "utt2text", "spk2utt"),
        readme_rel_path="test_prepare_dataset/gsc/README.md",
        doc_status="ready",
        notes="Current core keeps the short-duration text-independent trial family; archived text-dependent variants live on the archive/text-dependent-scenario branch.",
    ),
    DatasetSpec(
        dataset_id="globe",
        display_name="GLOBE",
        collection="core",
        benchmark_condition="accent robustness",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed from the downloaded evaluation pool",
        license_note="Follow the upstream GLOBE / Hugging Face dataset terms.",
        download_method="Automatic Hugging Face download of the frozen OpenSVBench canonical shards via local/download_globe.sh.",
        raw_rel_dir="globe/raw_data",
        prepared_rel_dir="globe/prepared",
        metadata_files=("utt2spk", "utt2accent", "spk2utt"),
        readme_rel_path="test_prepare_dataset/globe/README.md",
        doc_status="ready",
        notes="English-accent robustness benchmark derived from GLOBE; the kit pins and materializes only the canonical 10k-utterance subset.",
    ),
    DatasetSpec(
        dataset_id="cnceleb",
        display_name="CN-Celeb",
        collection="core",
        benchmark_condition="official in-domain + genre shift + short-duration",
        source_limit="No 10000 limit; official trials preserved",
        trial_source="Official eval protocol for the main set plus self-constructed genre-shift and short-duration supplements",
        license_note="See OpenSLR 82 licensing terms.",
        download_method="Automatic evaluation-package download, with manual full-set placement supported.",
        raw_rel_dir="cnceleb/raw_data",
        prepared_rel_dir="cnceleb/prepared",
        metadata_files=("utt2spk", "utt2genre", "utt2session", "spk2utt"),
        readme_rel_path="test_prepare_dataset/cnceleb/README.md",
        doc_status="ready",
        notes="Official-trial dataset exempt from the 10000 clean-source limit, with the main protocol, genre-shift supplement, and short-duration supplement kept in the active core.",
    ),
    DatasetSpec(
        dataset_id="voxceleb",
        display_name="VoxCeleb1",
        collection="core",
        benchmark_condition="official Vox1-O / Vox1-E / Vox1-H + short-duration",
        source_limit="No 10000 limit; official trials preserved",
        trial_source="Official VoxCeleb1 verification protocols plus a short-duration supplement",
        license_note="Requires acceptance of the official VoxCeleb terms.",
        download_method="Automatic trial download; audio must be placed manually or unpacked from local zips.",
        raw_rel_dir="voxceleb/raw_data",
        prepared_rel_dir="voxceleb/prepared",
        metadata_files=("utt2spk", "spk2utt"),
        readme_rel_path="test_prepare_dataset/voxceleb/README.md",
        doc_status="ready",
        notes="Official-trial dataset exempt from the 10000 clean-source limit.",
    ),
    DatasetSpec(
        dataset_id="3dspeaker",
        display_name="3D-Speaker",
        collection="core",
        benchmark_condition="cross-device + cross-distance + cross-dialect",
        source_limit="No 10000 limit; official trials preserved",
        trial_source="Official 3D-Speaker trial files",
        license_note="Follow the official 3D-Speaker dataset terms.",
        download_method="Automatic download from the official 3D-Speaker mirrors.",
        raw_rel_dir="3dspeaker/raw_data",
        prepared_rel_dir="3dspeaker/prepared",
        metadata_files=("utt2spk", "utt2device", "utt2distance", "utt2dialect", "spk2utt"),
        readme_rel_path="test_prepare_dataset/3dspeaker/README.md",
        doc_status="ready",
        notes="Official-trial dataset exempt from the 10000 clean-source limit, with device, distance, and dialect trials split into separate ranked entries.",
    ),
    DatasetSpec(
        dataset_id="ffsvc2022",
        display_name="FFSVC 2022",
        collection="core",
        benchmark_condition="cross-channel + near/far distance",
        source_limit="No 10000 limit; official dev trials preserved",
        trial_source="Official FFSVC 2022 dev trial pairs filtered into channel and near/far robustness views",
        license_note="Follow the official FFSVC 2022 / Zenodo release terms.",
        download_method="Automatic public dev-set download via local/download_ffsvc2022.sh; eval wav remains opt-in because it is large.",
        raw_rel_dir="ffsvc2022/raw_data",
        prepared_rel_dir="ffsvc2022/prepared",
        metadata_files=(
            "utt2spk",
            "utt2visit",
            "utt2session",
            "utt2device",
            "utt2distance",
            "utt2text",
            "utt2style",
            "utt2domain",
            "spk2utt",
            "stats_ffsvc2022.json",
        ),
        readme_rel_path="test_prepare_dataset/ffsvc2022/README.md",
        doc_status="ready",
        notes="Public dev-set integration. Aging evaluation is intentionally omitted because the visit span is short for longitudinal speaker-robustness evaluation.",
    ),
    DatasetSpec(
        dataset_id="whisper40",
        display_name="Whisper40",
        collection="core",
        benchmark_condition="neutral-to-whisper speaking-style shift",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed same-text neutral-to-whisper trials from the manually provided Whisper40 archive",
        license_note="Follow the upstream Whisper40 paper / repository terms before redistribution or public use.",
        download_method="Manual archive placement expected; local/download_whisper40.sh extracts a user-provided whisper40.rar archive.",
        raw_rel_dir="whisper40/raw_data",
        prepared_rel_dir="whisper40/prepared",
        metadata_files=("utt2spk", "utt2lang", "utt2style", "utt2text", "utt2split", "spk2utt", "stats_whisper40.json"),
        readme_rel_path="test_prepare_dataset/whisper40/README.md",
        doc_status="ready",
        notes="Chinese same-text whisper robustness set built from neutral and whispered speech pairs with text-matched impostors.",
    ),
    DatasetSpec(
        dataset_id="lombardgrid",
        display_name="Lombard Grid",
        collection="core",
        benchmark_condition="plain-to-Lombard speaking-style shift",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed plain-to-Lombard trials from the public Lombard Grid audio release",
        license_note="CC BY 4.0 per the Zenodo Lombard Grid release; cite Alghamdi et al. (2018).",
        download_method="Automatic Zenodo download via local/download_lombardgrid.sh; preparation extracts lombardgrid_audio.zip under DATA_ROOT/lombardgrid/raw_data.",
        raw_rel_dir="lombardgrid/raw_data",
        prepared_rel_dir="lombardgrid/prepared",
        metadata_files=(
            "utt2spk",
            "utt2lang",
            "utt2style",
            "utt2text",
            "utt2duration",
            "spk2utt",
            "stats_lombardgrid_source.json",
            "stats_lombardgrid_lombard.json",
        ),
        readme_rel_path="test_prepare_dataset/lombardgrid/README.md",
        doc_status="ready",
        notes="English Lombard-effect robustness set built from plain reference speech and Lombard speech elicited by noise exposure.",
    ),
    DatasetSpec(
        dataset_id="voices",
        display_name="VOiCES",
        collection="core",
        benchmark_condition="noise/reverb",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed easy-to-hard in-corpus trials from the public VOiCES devkit test split",
        license_note="CC BY 4.0 per the official VOiCES release; follow the upstream terms before redistribution.",
        download_method="Automatic public AWS devkit download via local/download_voices.sh; preparation selectively extracts the VOiCES devkit test speech/source files.",
        raw_rel_dir="voices/raw_data",
        prepared_rel_dir="voices/prepared",
        metadata_files=(
            "utt2spk",
            "utt2lang",
            "utt2gender",
            "utt2room",
            "utt2noise",
            "utt2channel",
            "utt2mic",
            "utt2mic_type",
            "utt2mic_location",
            "utt2source_segment",
            "spk2utt",
            "stats_voices_source.json",
            "stats_voices_noise_reverb.json",
        ),
        readme_rel_path="test_prepare_dataset/voices/README.md",
        doc_status="ready",
        notes="VOiCES devkit test-set integration using a canonical easy enroll condition (`rm1-none-mc01`) against harder noise/reverb/mic test conditions.",
    ),
    DatasetSpec(
        dataset_id="chime6",
        display_name="CHiME-6",
        collection="core",
        benchmark_condition="domestic far-field",
        source_limit="<=10000 prepared utterances per active trial protocol from the dev split",
        trial_source="Self-constructed domestic far-field and overlap-robustness trials from the public CHiME-6 dev split",
        license_note="CC BY-SA 4.0 per the OpenSLR CHiME-6 release; cite the CHiME-5 and CHiME-6 challenge papers.",
        download_method="Automatic OpenSLR download via local/download_chime6.sh; preparation extracts CHiME6_dev and CHiME6_transcriptions under DATA_ROOT/chime6/raw_data.",
        raw_rel_dir="chime6/raw_data",
        prepared_rel_dir="chime6/prepared",
        metadata_files=(
            "utt2spk",
            "utt2lang",
            "utt2session",
            "utt2location",
            "utt2channel",
            "utt2channel_type",
            "utt2array_id",
            "utt2array_channel",
            "utt2overlap_ratio",
            "utt2text",
            "utt2duration",
            "spk2utt",
            "stats_chime6_source.json",
            "stats_chime6_domestic_farfield.json",
        ),
        readme_rel_path="test_prepare_dataset/chime6/README.md",
        doc_status="ready",
        notes="CHiME-6 dev-set integration with a domestic far-field wearable-to-array trial and a same-channel overlap-robustness trial (wearable-wearable plus array-array).",
    ),
    DatasetSpec(
        dataset_id="esd",
        display_name="ESD",
        collection="core",
        benchmark_condition="cross-emotion",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed from public ESD speakers and emotions",
        license_note="Follow the upstream ESD terms before redistribution.",
        download_method="Automatic download with manual zip fallback via local/download_esd.sh.",
        raw_rel_dir="esd/raw_data",
        prepared_rel_dir="esd/prepared",
        metadata_files=("utt2spk", "utt2emotion", "utt2lang", "spk2utt"),
        readme_rel_path="test_prepare_dataset/esd/README.md",
        doc_status="ready",
        notes="Cross-emotion robustness benchmark.",
    ),
    DatasetSpec(
        dataset_id="alimeeting",
        display_name="AliMeeting",
        collection="core",
        benchmark_condition="near/far distance mismatch + overlap robustness",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed from public test-set segments",
        license_note="CC BY-SA 4.0 per the AliMeeting release.",
        download_method="Automatic download from the official AliMeeting mirror.",
        raw_rel_dir="alimeeting/raw_data",
        prepared_rel_dir="alimeeting/prepared",
        metadata_files=("utt2spk", "utt2channel", "utt2session", "spk2utt"),
        readme_rel_path="test_prepare_dataset/alimeeting/README.md",
        doc_status="ready",
        notes="Meeting speaker verification with a ranked near/far distance trial and overlap-stratified far-field trials.",
    ),
    DatasetSpec(
        dataset_id="voxknesset",
        display_name="VoxKnesset",
        collection="core",
        benchmark_condition="aging / longitudinal",
        source_limit="<=10000 clean source utterances",
        trial_source="Self-constructed reproducible protocol on the official VoxKnesset test split",
        license_note="Released under the ivrit.ai license; follow the upstream dataset terms.",
        download_method="Hugging Face access check via local/download_voxknesset.sh; configure proxy, mirror, or local cache if direct access fails.",
        raw_rel_dir="voxknesset/raw_data",
        prepared_rel_dir="voxknesset/prepared",
        metadata_files=("utt2spk", "utt2age", "utt2gender", "spk2utt", "stats_voxknesset.json"),
        readme_rel_path="test_prepare_dataset/voxknesset/README.md",
        doc_status="partially reproducible",
        notes="Current core aging set using the official VoxKnesset test split and age-derived time gaps.",
    ),
    DatasetSpec(
        dataset_id="voxpopuli",
        display_name="VoxPopuli",
        collection="core",
        benchmark_condition="aging / longitudinal",
        source_limit="<=10000 selected English utterances from the train split",
        trial_source="Self-constructed protocol on the public VoxPopuli English train split",
        license_note="CC0-1.0 / other as listed on the Hugging Face VoxPopuli dataset card; follow upstream terms before redistribution.",
        download_method="Automatic Hugging Face streaming access check via local/download_voxpopuli.sh; preparation materializes only the selected English wav subset.",
        raw_rel_dir="voxpopuli/raw_data",
        prepared_rel_dir="voxpopuli/prepared",
        metadata_files=(
            "utt2spk",
            "utt2gender",
            "utt2lang",
            "utt2date",
            "utt2year",
            "utt2duration",
            "spk2utt",
            "stats_voxpopuli_source.json",
            "stats_voxpopuli_cross_time.json",
        ),
        readme_rel_path="test_prepare_dataset/voxpopuli/README.md",
        doc_status="ready",
        notes="English-only VoxPopuli integration using European Parliament recording dates parsed from audio_id to build multi-year aging trials.",
    ),
)


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("tidyvoice", "tidyvoice", "TidyVoiceX2-ASV", "clean", "core",
                 "tidyvoice/prepared/trials_cross_lingual.txt", "tidyvoice/prepared", "tidyvoice_scores.csv",
                 "Official eval-key subset + constructed completion", "TidyVoiceX2-ASV eval A/U",
                 "Uses released labeled TidyVoiceX2-ASV eval A/U keys with reproducible subsampling and deterministic matched non-target completion."),
    ScenarioSpec("hi_mia", "hi-mia", "HI-MIA", "clean", "core",
                 "hi-mia/prepared/trials_short.txt", "hi-mia/prepared", "hi_mia_scores.csv",
                 "Self-constructed protocol", "Public test split",
                 "No upstream SV trial keys are used; trials are generated from the test split."),
    ScenarioSpec("gsc_short", "gsc", "GSC Short", "clean", "core",
                 "gsc/prepared/trials_short.txt", "gsc/prepared", "gsc_short_scores.csv",
                 "Self-constructed protocol", "Official testing split",
                 "Uses the Speech Commands testing split to build a short-duration SV protocol."),
    ScenarioSpec("globe", "globe", "GLOBE", "clean", "core",
                 "globe/prepared/trials_accent.txt", "globe/prepared", "globe_scores.csv",
                 "Self-constructed protocol", "Pinned canonical HF train+test shards",
                 "No upstream SV trial keys are used; trials are generated from a frozen 10k-utterance subset spanning selected Hugging Face train and test shards."),
    ScenarioSpec("cnceleb", "cnceleb", "CN-Celeb", "clean", "core",
                 "cnceleb/prepared/trials_cnceleb.txt", "cnceleb/prepared", "cnceleb_scores.csv",
                 "Official protocol", "Official evaluation protocol",
                 "Keeps the upstream CN-Celeb evaluation trial protocol."),
    ScenarioSpec("cnceleb_genre", "cnceleb", "CN-Celeb Genre", "clean", "core",
                 "cnceleb/prepared/trials_cnceleb_genre.txt", "cnceleb/prepared", "cnceleb_genre_scores.csv",
                 "Self-constructed supplement", "CN-Celeb prepared evaluation pool",
                 "Uses CN-Celeb metadata to build a genre-shift supplement outside the official trial file."),
    ScenarioSpec("cnceleb_short", "cnceleb", "CN-Celeb Short", "clean", "core",
                 "cnceleb/prepared/trials_short_dur.txt", "cnceleb/prepared", "cnceleb_short_scores.csv",
                 "Self-constructed supplement", "CN-Celeb prepared evaluation pool",
                 "Uses generated short utterances as a duration-robustness supplement."),
    ScenarioSpec("voxceleb_o", "voxceleb", "VoxCeleb1-O", "clean", "core",
                 "voxceleb/prepared/trials_vox1_O.txt", "voxceleb/prepared", "voxceleb_o_scores.csv",
                 "Official protocol", "VoxCeleb1-O",
                 "Keeps the upstream VoxCeleb1-O verification protocol."),
    ScenarioSpec("voxceleb_e", "voxceleb", "VoxCeleb1-E", "clean", "core",
                 "voxceleb/prepared/trials_vox1_E.txt", "voxceleb/prepared", "voxceleb_e_scores.csv",
                 "Official protocol", "VoxCeleb1-E",
                 "Keeps the upstream VoxCeleb1-E verification protocol."),
    ScenarioSpec("voxceleb_h", "voxceleb", "VoxCeleb1-H", "clean", "core",
                 "voxceleb/prepared/trials_vox1_H.txt", "voxceleb/prepared", "voxceleb_h_scores.csv",
                 "Official protocol", "VoxCeleb1-H",
                 "Keeps the upstream VoxCeleb1-H verification protocol."),
    ScenarioSpec("voxceleb_short", "voxceleb", "VoxCeleb Short", "clean", "core",
                 "voxceleb/prepared/trials_short_dur.txt", "voxceleb/prepared", "voxceleb_short_scores.csv",
                 "Self-constructed supplement", "VoxCeleb1 evaluation audio",
                 "Uses generated short utterances as a duration-robustness supplement."),
    ScenarioSpec("speaker3d_device", "3dspeaker", "3D-Speaker Device", "clean", "core",
                 "3dspeaker/prepared/trials_speaker3d_device.txt", "3dspeaker/prepared", "speaker3d_device_scores.csv",
                 "Official protocol", "3D-Speaker device trial file",
                 "Keeps the upstream cross-device trial file."),
    ScenarioSpec("speaker3d_distance", "3dspeaker", "3D-Speaker Distance", "clean", "core",
                 "3dspeaker/prepared/trials_speaker3d_distance.txt", "3dspeaker/prepared", "speaker3d_distance_scores.csv",
                 "Official protocol", "3D-Speaker distance trial file",
                 "Keeps the upstream cross-distance trial file."),
    ScenarioSpec("speaker3d_dialect", "3dspeaker", "3D-Speaker Dialect", "clean", "core",
                 "3dspeaker/prepared/trials_speaker3d_dialect.txt", "3dspeaker/prepared", "speaker3d_dialect_scores.csv",
                 "Official protocol", "3D-Speaker dialect trial file",
                 "Keeps the upstream cross-dialect trial file."),
    ScenarioSpec("ffsvc2022_cross_channel", "ffsvc2022", "FFSVC 2022 Cross-Channel", "clean", "core",
                 "ffsvc2022/prepared/trials_ffsvc2022_cross_channel.txt", "ffsvc2022/prepared", "ffsvc2022_cross_channel_scores.csv",
                 "Official protocol subset", "FFSVC 2022 public dev trials",
                 "Filters the official dev trial pairs to the channel/device robustness view."),
    ScenarioSpec("ffsvc2022_cross_domain", "ffsvc2022", "FFSVC 2022 Cross-Domain", "clean", "core",
                 "ffsvc2022/prepared/trials_ffsvc2022_cross_domain.txt", "ffsvc2022/prepared", "ffsvc2022_cross_domain_scores.csv",
                 "Official protocol subset", "FFSVC 2022 public dev trials",
                 "Filters the official dev trial pairs to the near/far domain robustness view."),
    ScenarioSpec("whisper40_whisper", "whisper40", "Whisper40 Whisper", "clean", "core",
                 "whisper40/prepared/trials_whisper40_whisper.txt", "whisper40/prepared", "whisper40_whisper_scores.csv",
                 "Self-constructed protocol", "Provided Whisper40 archive",
                 "Builds same-text neutral-to-whisper SV trials because no upstream SV trial keys are used."),
    ScenarioSpec("lombardgrid_lombard", "lombardgrid", "Lombard Grid Lombard", "clean", "core",
                 "lombardgrid/prepared/trials_lombardgrid_lombard.txt", "lombardgrid/prepared", "lombardgrid_lombard_scores.csv",
                 "Self-constructed protocol", "Public Lombard Grid audio release",
                 "Builds plain-to-Lombard SV trials from the public Lombard Grid release because no upstream SV trial keys are used."),
    ScenarioSpec("voices_noise_reverb", "voices", "VOiCES Noise/Reverb", "clean", "core",
                 "voices/prepared/trials_voices_noise_reverb.txt", "voices/prepared", "voices_noise_reverb_scores.csv",
                 "Self-constructed protocol subset", "VOiCES devkit test split",
                 "Uses the 6,400-recording devkit test split, not the VOiCES 2019 challenge trial keys."),
    ScenarioSpec("chime6_domestic_farfield", "chime6", "CHiME-6 Domestic Far-Field", "clean", "core",
                 "chime6/prepared/trials_chime6_domestic_farfield.txt", "chime6/prepared", "chime6_domestic_farfield_scores.csv",
                 "Self-constructed protocol", "CHiME-6 dev split",
                 "Uses dev sessions with location/ref metadata to build wearable-to-array domestic far-field trials."),
    ScenarioSpec("chime6_overlap", "chime6", "CHiME-6 Overlap", "clean", "core",
                 "chime6/prepared_overlap/trials_chime6_overlap.txt", "chime6/prepared_overlap", "chime6_overlap_scores.csv",
                 "Self-constructed protocol", "CHiME-6 dev split",
                 "Uses dev sessions to build clean-enroll to overlap-test same-channel trials."),
    ScenarioSpec("esd", "esd", "ESD", "clean", "core",
                 "esd/prepared/trials_emotion.txt", "esd/prepared", "esd_scores.csv",
                 "Self-constructed protocol", "Public ESD recordings",
                 "Builds cross-emotion SV trials from speaker and emotion labels."),
    ScenarioSpec("alimeeting_near_far", "alimeeting", "AliMeeting Near/Far", "clean", "core",
                 "alimeeting/prepared/trials_alimeeting_near_far.txt", "alimeeting/prepared", "alimeeting_near_far_scores.csv",
                 "Self-constructed protocol", "Public AliMeeting test-set segments",
                 "Builds near/far distance trials from public test-set segment metadata."),
    ScenarioSpec("alimeeting_overlap", "alimeeting", "AliMeeting Overlap", "clean", "core",
                 "alimeeting/prepared/trials_alimeeting_overlap.txt", "alimeeting/prepared", "alimeeting_overlap_scores.csv",
                 "Self-constructed protocol", "Public AliMeeting test-set segments",
                 "Builds overlap-stratified trials from speaker-time annotations."),
    ScenarioSpec("voxknesset", "voxknesset", "VoxKnesset", "clean", "core",
                 "voxknesset/prepared/trials_cross_time.txt", "voxknesset/prepared", "voxknesset_scores.csv",
                 "Self-constructed protocol", "Official VoxKnesset test split",
                 "Builds aging trials from the official test split and speaker age/time metadata."),
    ScenarioSpec("voxpopuli_cross_time", "voxpopuli", "VoxPopuli Aging", "clean", "core",
                 "voxpopuli/prepared/trials_voxpopuli_cross_time.txt", "voxpopuli/prepared", "voxpopuli_cross_time_scores.csv",
                 "Self-constructed protocol", "VoxPopuli English train split",
                 "Builds English-only aging trials from European Parliament recording dates parsed from audio_id."),
)


CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        capability_id="cross_lingual",
        display_name="Cross-Lingual Robustness",
        short_label="Cross-lingual",
        collection="core",
        summary="How well speaker identity survives enrollment-test language mismatch.",
        lens="Language mismatch",
        scenario_ids=("tidyvoice",),
    ),
    CapabilitySpec(
        capability_id="short_duration",
        display_name="Short-Duration Robustness",
        short_label="Short-duration",
        collection="core",
        summary="How much performance holds up when speech evidence is limited by duration.",
        lens="Short-duration speech",
        scenario_ids=("hi_mia", "gsc_short", "cnceleb_short", "voxceleb_short"),
    ),
    CapabilitySpec(
        capability_id="genre_shift",
        display_name="Genre-Shift Robustness",
        short_label="Genre shift",
        collection="core",
        summary="Whether performance holds when CN-Celeb enrollment and test speech come from different source genres.",
        lens="Source-genre variation",
        scenario_ids=("cnceleb_genre",),
    ),
    CapabilitySpec(
        capability_id="accent_dialect_robustness",
        display_name="Accent / Dialect Robustness",
        short_label="Accent/dialect",
        collection="core",
        summary="How reliably the model tracks identity across accent and dialect mismatch.",
        lens="Accent and dialect variation",
        scenario_ids=("globe", "speaker3d_dialect"),
    ),
    CapabilitySpec(
        capability_id="in_the_wild_robustness",
        display_name="In-The-Wild Robustness",
        short_label="In-the-wild",
        collection="core",
        summary="How strong the model is on unconstrained celebrity and media speech across official CN-Celeb and VoxCeleb protocols.",
        lens="Open-domain media speech",
        scenario_ids=("cnceleb", "voxceleb_o", "voxceleb_e", "voxceleb_h"),
    ),
    CapabilitySpec(
        capability_id="channel_device_robustness",
        display_name="Channel / Device Robustness",
        short_label="Channel/device",
        collection="core",
        summary="How resilient the model is to device and channel mismatch.",
        lens="Device and channel variation",
        scenario_ids=("speaker3d_device", "ffsvc2022_cross_channel"),
    ),
    CapabilitySpec(
        capability_id="speaking_state_robustness",
        display_name="Speaking-Style Robustness",
        short_label="Speaking style",
        collection="core",
        summary="Whether the model can preserve identity across emotion-driven change, whispered speech, and noise-induced Lombard speaking style.",
        lens="Speaking style shift",
        scenario_ids=("esd", "whisper40_whisper", "lombardgrid_lombard"),
    ),
    CapabilitySpec(
        capability_id="distance_robustness",
        display_name="Distance Robustness",
        short_label="Distance",
        collection="core",
        summary="How much performance changes across meeting and domestic distance mismatch.",
        lens="Distance mismatch",
        scenario_ids=("speaker3d_distance", "alimeeting_near_far", "chime6_domestic_farfield", "ffsvc2022_cross_domain"),
    ),
    CapabilitySpec(
        capability_id="noise_reverb_robustness",
        display_name="Noise / Reverb Robustness",
        short_label="Noise/reverb",
        collection="core",
        summary="How reliably the model preserves identity when room acoustics, distractor noise, and microphone placement deviate from an easier in-corpus reference condition.",
        lens="Noise and reverberation",
        scenario_ids=("voices_noise_reverb",),
    ),
    CapabilitySpec(
        capability_id="overlap_robustness",
        display_name="Overlap Robustness",
        short_label="Overlap",
        collection="core",
        summary="How well speaker identity survives light, mid, and heavy overlap in both meeting and domestic recordings.",
        lens="Overlapping speakers",
        scenario_ids=("alimeeting_overlap", "chime6_overlap"),
    ),
    CapabilitySpec(
        capability_id="cross_time",
        display_name="Aging Robustness",
        short_label="Aging",
        collection="core",
        summary="How stable identity representations remain across age-derived and longitudinal recording gaps.",
        lens="Speaker aging and time gaps",
        scenario_ids=("voxknesset", "voxpopuli_cross_time"),
    ),
)


def iter_datasets(collection: str | None = None) -> Iterable[DatasetSpec]:
    for spec in DATASETS:
        if collection is None or spec.collection == collection:
            yield spec


def iter_scenarios(collection: str = "core") -> Iterable[ScenarioSpec]:
    for spec in SCENARIOS:
        if spec.collection == collection:
            yield spec


def iter_capabilities(collection: str = "core") -> Iterable[CapabilitySpec]:
    for spec in CAPABILITIES:
        if spec.collection == collection:
            yield spec


def dataset_by_id(dataset_id: str) -> DatasetSpec:
    for spec in DATASETS:
        if spec.dataset_id == dataset_id:
            return spec
    raise KeyError(dataset_id)


def scenario_by_id(scenario_id: str) -> ScenarioSpec:
    for spec in SCENARIOS:
        if spec.scenario_id == scenario_id:
            return spec
    raise KeyError(scenario_id)


def capability_by_id(capability_id: str) -> CapabilitySpec:
    for spec in CAPABILITIES:
        if spec.capability_id == capability_id:
            return spec
    raise KeyError(capability_id)
