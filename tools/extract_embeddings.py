#!/usr/bin/env python3
# OpenSVBench embedding extraction entrypoint.

import argparse
import os
import sys
import pickle
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Keep routine Transformers logs out of the extraction output.
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

logging.getLogger().setLevel(logging.ERROR)

# SpeechBrain 1.x expects list_audio_backends, which was removed in some
# torchaudio 2.x releases.
import torchaudio as _torchaudio
if not hasattr(_torchaudio, "list_audio_backends"):
    _torchaudio.list_audio_backends = lambda: []


def get_args():
    parser = argparse.ArgumentParser(description="Extract embeddings")
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True, help="Path to local model folder OR HF repo name")
    parser.add_argument("--backend", default="wespeaker",
                        choices=["wespeaker", "huggingface", "modelscope", "speechbrain", "redimnet2"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-chunk-seconds", type=float, default=20.0,
                        help="Split long utterances into chunks of at most this many seconds "
                             "before embedding extraction. Use 0 to disable chunking. "
                             "For WeSpeaker this is used as an OOM-safe fallback path. "
                             "(default: 20.0)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="For non-WeSpeaker backends, process this many single-chunk "
                             "utterances together per forward pass. "
                             "Use 0 to pick a backend-specific default. (default: 0)")
    parser.add_argument("--io-workers", type=int, default=0,
                        help="For non-WeSpeaker backends, use this many workers to decode "
                             "audio ahead of GPU extraction. Use 0 to pick a backend-specific "
                             "default. (default: 0)")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Flush partial embedding cache to disk every N newly extracted "
                             "utterances so interrupted jobs can resume. (default: 2000)")
    parser.add_argument(
        "--waveform-cmvn",
        choices=["none", "utterance"],
        default="none",
        help="Waveform-level CMVN mode before backend inference. "
             "'none' follows most model-card inference usage; "
             "'utterance' applies per-utterance mean/std normalization. (default: none)",
    )
    parser.add_argument(
        "--speechbrain-normalize",
        choices=["auto", "true", "false"],
        default="auto",
        help="Control SpeechBrain encode_batch(normalize=...). "
             "'auto' uses OpenSVBench default, 'true' forces normalize=True, "
             "'false' forces normalize=False. (default: auto)",
    )
    return parser.parse_args()


def default_batch_size(backend: str) -> int:
    return {
        "huggingface": 16,
        # Keep ModelScope on single-utterance forward by default to stay
        # close to official model-card style inference.
        "modelscope": 1,
        "speechbrain": 32,
        # ReDimNet2 quality is sensitive to variable-length batch padding;
        # keep single-utterance forward by default.
        "redimnet2": 1,
    }.get(backend, 1)


def default_io_workers(backend: str) -> int:
    return {
        "huggingface": 8,
        "modelscope": 8,
        "speechbrain": 8,
        "redimnet2": 8,
    }.get(backend, 1)


def resolve_speechbrain_normalize_mode(mode: str) -> bool | None:
    if mode == "true":
        return True
    if mode == "false":
        return False
    # "auto": keep close to SpeechBrain model-card examples that call encode_batch
    # without explicitly passing normalize=...
    return False

def load_wespeaker_model(model_name, device_str):
    import wespeaker

    print(f"[Extract] Loading WeSpeaker model: {model_name} ...")
    model = wespeaker.load_model(model_name)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        if hasattr(model, "set_device"):
            model.set_device(device_str)
        elif hasattr(model, "set_gpu"):
            gpu_id = int(device_str.split(":")[1])
            model.set_gpu(gpu_id)
        else:
            try:
                model.model.to(device_str)
                model.device = torch.device(device_str)
            except Exception:
                pass
    else:
        if hasattr(model, "set_device"):
            model.set_device("cpu")
    return model

def load_hf_model(model_name, device):
    """
    Load a Hugging Face model from a local path or repository identifier.
    """
    print(f"[Extract] Loading HuggingFace model from: {model_name} ...")
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    # 1. Load the feature extractor.
    # Some Orange checkpoints omit preprocessor_config.json, so use the
    # WavLM configuration when the model directory has no usable extractor.
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    except (OSError, ValueError):
        print(f"  [Info] Local config not found or invalid. Falling back to 'microsoft/wavlm-large' for feature extractor...")
        # This fallback may require network access unless the WavLM
        # configuration is already available locally.
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large", trust_remote_code=True)

    # 2. Load the model weights from model_name.
    try:
        model = AutoModelForAudioXVector.from_pretrained(model_name, trust_remote_code=True).to(device)
    except OSError as e:
        print(f"\n[Error] Failed to load local model from {model_name}.")
        print("Please check if 'config.json' and 'model.safetensors' exist in that folder.")
        raise e

    model.eval()
    return feature_extractor, model

def load_modelscope_model(model_name, device):
    print(f"[Extract] Loading ModelScope model: {model_name} ...")
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    if device.type == "cuda" and device.index is not None:
        ms_device = f"gpu:{device.index}"
    else:
        ms_device = "cpu"
    sv_pipeline = pipeline(task=Tasks.speaker_verification, model=model_name, device=ms_device)
    model = sv_pipeline.model
    model.to(device)
    model.eval()
    return model


def load_speechbrain_model(model_name, device):
    """
    Load a SpeechBrain EncoderClassifier.

    model_name may be a Hugging Face repository or a local path. Downloaded
    files are cached under pretrained_models/<basename>.
    """
    print(f"[Extract] Loading SpeechBrain model: {model_name} ...")
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # older speechbrain API

    basename = os.path.basename(model_name.rstrip("/"))
    # Prefer the local pretrained_models/<basename> cache.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    savedir = os.path.join(repo_root, "pretrained_models", basename)
    os.makedirs(savedir, exist_ok=True)

    run_opts = {"device": str(device)}

    # Use the local directory when hyperparams.yaml is already available.
    runtime_savedir = savedir
    if os.path.exists(os.path.join(savedir, "hyperparams.yaml")):
        source = prepare_local_speechbrain_source(savedir, basename)
        runtime_savedir = source
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    else:
        source = model_name  # Let SpeechBrain download the model.

    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir=runtime_savedir,
        run_opts=run_opts,
    )
    classifier.eval()
    return classifier


def parse_redimnet2_model_name(model_name: str) -> tuple[str, str]:
    raw_name = os.path.basename(model_name.rstrip("/"))
    for pattern in (
        r"^redimnet2[_:-](b[0-6])[_:-](ptn|lm)$",
        r"^(b[0-6])[_:-](ptn|lm)$",
    ):
        match = re.fullmatch(pattern, raw_name)
        if match:
            return match.group(1), match.group(2)
    raise ValueError(
        "Unsupported redimnet2 model name. Use formats like "
        "'redimnet2_b6_lm' or 'b6_lm'."
    )


def resolve_redimnet2_repo() -> Path:
    env_path = os.environ.get("OPENSVBENCH_REDIMNET2_REPO")
    if env_path:
        repo_path = Path(env_path).expanduser().resolve()
        if repo_path.is_dir():
            return repo_path

    repo_root = Path(__file__).resolve().parents[1]
    default_path = repo_root / "pretrained_models" / "redimnet2_repo"
    if default_path.is_dir():
        return default_path

    raise FileNotFoundError(
        "redimnet2 repo not found. Expected it at "
        f"{default_path} or via OPENSVBENCH_REDIMNET2_REPO."
    )


def load_redimnet2_model(model_name, device):
    model_variant, train_type = parse_redimnet2_model_name(model_name)
    repo_path = resolve_redimnet2_repo()
    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)

    print(
        f"[Extract] Loading ReDimNet2 model: variant={model_variant}, "
        f"train_type={train_type}, repo={repo_path_str} ..."
    )
    from redimnet2 import ReDimNet2

    model = ReDimNet2.from_pretrained(model_variant, train_type=train_type)
    model = model.to(device)
    model.eval()
    return model


def prepare_local_speechbrain_source(savedir: str, basename: str) -> str:
    """
    Patch a local SpeechBrain source when hyperparams still reference a
    Hugging Face repository. The patched source points the pretrainer at local
    checkpoint and label files.
    """
    source_dir = Path(savedir).resolve()
    hyperparams_path = source_dir / "hyperparams.yaml"
    if not hyperparams_path.is_file():
        return str(source_dir)

    patched_dir = source_dir / ".local_speechbrain_source"
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_hparams = patched_dir / "hyperparams.yaml"

    text = hyperparams_path.read_text(encoding="utf-8")
    local_ref = source_dir.as_posix()

    text = re.sub(
        r"^pretrained_path:\s*.*$",
        f"pretrained_path: {local_ref}",
        text,
        flags=re.MULTILINE,
    )
    text = text.replace(f"speechbrain/{basename}", local_ref)
    patched_hparams.write_text(text, encoding="utf-8")
    return str(patched_dir)

def read_wav_scp(wav_scp_path):
    wav_paths = {}
    with open(wav_scp_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                wav_paths[parts[0]] = parts[1]
    return wav_paths


def load_existing_embeddings(output_path: str) -> dict:
    output = Path(output_path)
    if not output.is_file() or output.stat().st_size <= 16:
        return {}
    try:
        with output.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        print(f"[Extract] Warning: failed to load existing cache {output}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}

def load_waveform_robust(path, target_sr=16000):
    """
    Load audio for inference with deterministic normalization steps: prefer
    torchaudio, fall back to soundfile, convert to mono, resample, and add a
    tiny perturbation to completely silent audio.
    """
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        try:
            import soundfile as sf
            wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)
        except Exception:
            return None

    # 1. Convert to mono.
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # 2. Resample.
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)

    wav = wav.squeeze()

    # 3. Reject empty audio.
    if wav.numel() == 0:
        return None

    # Add a tiny deterministic perturbation to completely silent audio so the
    # backend does not discard the sample or produce NaNs.
    if wav.abs().max() < 1e-6:
        wav = wav.clone()
        wav[0] = 1e-4

    return wav


def split_waveform(wav, max_chunk_samples, min_chunk_samples=16000):
    """Split a long waveform into non-overlapping chunks."""
    if max_chunk_samples is None or max_chunk_samples <= 0 or wav.numel() <= max_chunk_samples:
        return [wav]

    chunks = []
    total = wav.numel()
    for start in range(0, total, max_chunk_samples):
        chunk = wav[start:start + max_chunk_samples]
        if chunk.numel() < min_chunk_samples and chunks:
            # Avoid sending tiny trailing fragments (e.g. VoxCeleb 20.04 s
            # files leave a 40 ms tail) through pooling layers that need at
            # least two valid time frames for a finite std statistic.
            chunks[-1] = torch.cat([chunks[-1], chunk], dim=0)
        elif chunk.numel() >= min_chunk_samples:
            chunks.append(chunk)

    return chunks if chunks else [wav]


def build_chunk_attempts(max_chunk_samples):
    chunk_attempts = [max_chunk_samples]
    if max_chunk_samples and max_chunk_samples > 16000:
        chunk_size = max_chunk_samples // 2
        while chunk_size >= 16000:
            chunk_attempts.append(chunk_size)
            chunk_size //= 2
    if chunk_attempts == [None]:
        chunk_attempts = [None]
    return chunk_attempts


def prepare_waveform_for_embedding(wav, backend=None, cmvn_mode="none"):
    if wav is None:
        return None
    if wav.shape[0] < 400:
        wav = F.pad(wav, (0, 400 - wav.shape[0]), mode="replicate")
    if cmvn_mode == "utterance":
        wav_std = wav.std()
        if torch.isfinite(wav_std) and wav_std > 1e-6:
            wav = (wav - wav.mean()) / (wav_std + 1e-6)
    return wav


def iter_prepared_waveforms(
    wav_items,
    target_sr=16000,
    io_workers=1,
    prefetch_factor=4,
    backend=None,
    cmvn_mode="none",
):
    if io_workers <= 1:
        for utt_id, wav_path in wav_items:
            wav = load_waveform_robust(wav_path, target_sr=target_sr)
            yield utt_id, wav_path, prepare_waveform_for_embedding(
                wav, backend=backend, cmvn_mode=cmvn_mode
            )
        return

    wav_iter = iter(wav_items)
    max_pending = max(io_workers * prefetch_factor, io_workers)

    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        pending = []

        def submit_next():
            try:
                utt_id, wav_path = next(wav_iter)
            except StopIteration:
                return False
            future = executor.submit(load_waveform_robust, wav_path, target_sr)
            pending.append((utt_id, wav_path, future))
            return True

        for _ in range(max_pending):
            if not submit_next():
                break

        while pending:
            utt_id, wav_path, future = pending.pop(0)
            wav = future.result()
            yield utt_id, wav_path, prepare_waveform_for_embedding(
                wav, backend=backend, cmvn_mode=cmvn_mode
            )
            submit_next()


def stack_wave_batch(wavs):
    max_len = max(wav.numel() for wav in wavs)
    padded = []
    lengths = []
    for wav in wavs:
        wav_len = wav.numel()
        lengths.append(wav_len / max_len)
        if wav_len < max_len:
            pad_val = wav[-1].expand(max_len - wav_len)
            wav = torch.cat([wav, pad_val], dim=0)
        padded.append(wav)
    return torch.stack(padded, dim=0), torch.tensor(lengths, dtype=torch.float32)


def merge_chunk_embeddings(chunk_embeddings):
    """Average chunk embeddings and renormalize the merged representation."""
    finite_chunks = []
    for emb in chunk_embeddings:
        if emb is None:
            continue
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        emb = emb.detach().cpu()
        if torch.isfinite(emb).all():
            finite_chunks.append(emb)

    if not finite_chunks:
        return None

    emb = torch.stack(finite_chunks, dim=0).mean(dim=0, keepdim=True)
    emb = F.normalize(emb, p=2, dim=1).squeeze(0)
    if not torch.isfinite(emb).all():
        return None
    return emb.cpu().numpy()


def normalize_embedding_array(emb):
    if emb is None:
        return None
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    emb = emb.detach().cpu().float().flatten()
    if emb.numel() == 0 or not torch.isfinite(emb).all():
        return None
    emb = F.normalize(emb.unsqueeze(0), p=2, dim=1).squeeze(0)
    if not torch.isfinite(emb).all():
        return None
    return emb.cpu().numpy()


def normalize_embedding_batch_tensor(emb_t):
    if not isinstance(emb_t, torch.Tensor):
        emb_t = torch.as_tensor(emb_t)
    emb_t = emb_t.detach().float()
    if emb_t.ndim == 1:
        emb_t = emb_t.unsqueeze(0)
    elif emb_t.ndim == 3 and emb_t.shape[1] == 1:
        emb_t = emb_t.squeeze(1)
    elif emb_t.ndim > 2:
        emb_t = emb_t.reshape(emb_t.shape[0], -1)

    emb_t = F.normalize(emb_t, p=2, dim=1)
    outputs = []
    for row in emb_t:
        row = row.cpu()
        outputs.append(row if torch.isfinite(row).all() else None)
    return outputs


def extract_wespeaker_chunk_embedding(model, wav_chunk):
    emb = model.extract_embedding_from_pcm(wav_chunk.unsqueeze(0), 16000)
    return normalize_embedding_array(emb)


def extract_wespeaker_embedding_from_waveform(model, wav_path, chunk_attempts, device):
    """
    Robust WeSpeaker fallback:
    1) Decode audio with robust loader.
    2) Try progressively smaller chunk sizes when OOM happens.
    3) Merge chunk embeddings by mean + L2 normalization.
    """
    # Some CN-Celeb FLAC files fail in torchaudio but can still be decoded by soundfile.
    wav = load_waveform_robust(wav_path, target_sr=16000)
    wav = prepare_waveform_for_embedding(wav, backend="wespeaker", cmvn_mode="none")
    if wav is None:
        return None

    for chunk_size in chunk_attempts:
        chunks = split_waveform(wav, max_chunk_samples=chunk_size)
        chunk_embeddings = []
        attempt_failed = False

        for chunk in chunks:
            try:
                emb = extract_wespeaker_chunk_embedding(model, chunk)
                if emb is None:
                    attempt_failed = True
                    break
                chunk_embeddings.append(emb)
            except Exception as exc:
                if device.type == "cuda" and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                attempt_failed = True
                break

        if not attempt_failed:
            merged = merge_chunk_embeddings(chunk_embeddings)
            if merged is not None:
                return merged

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return None


def save_embeddings_atomic(output_path: str, embeddings: dict) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(embeddings, f)
    os.replace(tmp_path, output)


def extract_hf_chunk_embedding(wav_chunk, feature_extractor, model, device):
    inputs = feature_extractor(
        wav_chunk.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    emb_t = None
    if hasattr(outputs, "embeddings"):
        emb_t = outputs.embeddings
    elif hasattr(outputs, "projected_states"):
        emb_t = outputs.projected_states
    elif isinstance(outputs, tuple) and outputs[0].ndim == 2:
        emb_t = outputs[0]

    if emb_t is None:
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]

        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        emb_t = sum_embeddings / sum_mask

    emb_t = F.normalize(emb_t, p=2, dim=1)
    return emb_t.squeeze(0).detach().cpu()


def extract_hf_batch_embeddings(wav_chunks, feature_extractor, model, device):
    inputs = feature_extractor(
        [wav_chunk.numpy() for wav_chunk in wav_chunks],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    emb_t = None
    if hasattr(outputs, "embeddings"):
        emb_t = outputs.embeddings
    elif hasattr(outputs, "projected_states"):
        emb_t = outputs.projected_states
    elif isinstance(outputs, tuple) and outputs[0].ndim == 2:
        emb_t = outputs[0]

    if emb_t is None:
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0]

        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        emb_t = sum_embeddings / sum_mask

    return normalize_embedding_batch_tensor(emb_t)


def extract_modelscope_chunk_embedding(wav_chunk, model, device):
    wav_t = wav_chunk.to(device).unsqueeze(0)
    with torch.no_grad():
        out = model(wav_t)

    if isinstance(out, torch.Tensor):
        emb_t = out.squeeze()
    elif isinstance(out, dict) and "embedding" in out:
        emb_t = out["embedding"].squeeze()
        if not isinstance(emb_t, torch.Tensor):
            emb_t = torch.as_tensor(emb_t)
    else:
        raise RuntimeError(f"Unexpected ModelScope output type: {type(out)}")

    emb_t = F.normalize(emb_t.unsqueeze(0), p=2, dim=1).squeeze(0)
    return emb_t.detach().cpu()


def extract_modelscope_batch_embeddings(wav_chunks, model, device):
    wav_batch, _ = stack_wave_batch(wav_chunks)
    wav_batch = wav_batch.to(device)
    with torch.no_grad():
        out = model(wav_batch)

    if isinstance(out, torch.Tensor):
        emb_t = out
    elif isinstance(out, dict) and "embedding" in out:
        emb_t = out["embedding"]
        if not isinstance(emb_t, torch.Tensor):
            emb_t = torch.as_tensor(emb_t)
    else:
        raise RuntimeError(f"Unexpected ModelScope output type: {type(out)}")

    return normalize_embedding_batch_tensor(emb_t)


def extract_speechbrain_chunk_embedding(wav_chunk, model, device, sb_normalize):
    wav_t = wav_chunk.to(device).unsqueeze(0)
    with torch.no_grad():
        if sb_normalize is None:
            emb_t = model.encode_batch(wav_t)
        else:
            try:
                emb_t = model.encode_batch(wav_t, normalize=sb_normalize)
            except TypeError:
                emb_t = model.encode_batch(wav_t)
    emb_t = emb_t.squeeze()
    emb_t = F.normalize(emb_t.unsqueeze(0), p=2, dim=1).squeeze(0)
    return emb_t.detach().cpu()


def extract_speechbrain_batch_embeddings(wav_chunks, model, device, sb_normalize):
    wav_batch, wav_lens = stack_wave_batch(wav_chunks)
    wav_batch = wav_batch.to(device)
    wav_lens = wav_lens.to(device)
    with torch.no_grad():
        if sb_normalize is None:
            try:
                emb_t = model.encode_batch(wav_batch, wav_lens)
            except TypeError:
                emb_t = model.encode_batch(wav_batch)
        else:
            try:
                emb_t = model.encode_batch(wav_batch, wav_lens, normalize=sb_normalize)
            except TypeError:
                try:
                    emb_t = model.encode_batch(wav_batch, wav_lens)
                except TypeError:
                    emb_t = model.encode_batch(wav_batch)
    return normalize_embedding_batch_tensor(emb_t)


def extract_redimnet2_chunk_embedding(wav_chunk, model, device):
    wav_t = wav_chunk.to(device).unsqueeze(0)
    with torch.no_grad():
        emb_t = model(wav_t)
    emb_t = emb_t.squeeze()
    emb_t = F.normalize(emb_t.unsqueeze(0), p=2, dim=1).squeeze(0)
    return emb_t.detach().cpu()


def extract_redimnet2_batch_embeddings(wav_chunks, model, device):
    wav_batch, _ = stack_wave_batch(wav_chunks)
    wav_batch = wav_batch.to(device)
    with torch.no_grad():
        emb_t = model(wav_batch)
    return normalize_embedding_batch_tensor(emb_t)


def extract_non_wespeaker_chunk_embedding(
    backend,
    wav_chunk,
    hf_fe,
    hf_model,
    ms_model,
    sb_model,
    redimnet2_model,
    device,
    sb_normalize,
):
    if backend == "huggingface":
        return extract_hf_chunk_embedding(wav_chunk, hf_fe, hf_model, device)
    if backend == "modelscope":
        return extract_modelscope_chunk_embedding(wav_chunk, ms_model, device)
    if backend == "speechbrain":
        return extract_speechbrain_chunk_embedding(wav_chunk, sb_model, device, sb_normalize)
    if backend == "redimnet2":
        return extract_redimnet2_chunk_embedding(wav_chunk, redimnet2_model, device)
    raise ValueError(f"Unsupported backend: {backend}")


def extract_non_wespeaker_batch_embeddings(
    backend,
    wav_chunks,
    hf_fe,
    hf_model,
    ms_model,
    sb_model,
    redimnet2_model,
    device,
    sb_normalize,
):
    if backend == "huggingface":
        return extract_hf_batch_embeddings(wav_chunks, hf_fe, hf_model, device)
    if backend == "modelscope":
        return extract_modelscope_batch_embeddings(wav_chunks, ms_model, device)
    if backend == "speechbrain":
        return extract_speechbrain_batch_embeddings(wav_chunks, sb_model, device, sb_normalize)
    if backend == "redimnet2":
        return extract_redimnet2_batch_embeddings(wav_chunks, redimnet2_model, device)
    raise ValueError(f"Unsupported backend: {backend}")


def extract_non_wespeaker_embeddings_adaptive(
    backend,
    wav_chunks,
    hf_fe,
    hf_model,
    ms_model,
    sb_model,
    redimnet2_model,
    device,
    sb_normalize,
):
    if not wav_chunks:
        return []
    if backend == "huggingface" and len(wav_chunks) > 16:
        # Very long recordings can produce hundreds of chunks. Sending all of
        # them through one padded HF batch is fragile and can poison retries
        # with OOM/non-finite outputs, so keep the fallback bounded.
        out = []
        for start in range(0, len(wav_chunks), 16):
            out.extend(
                extract_non_wespeaker_embeddings_adaptive(
                    backend,
                    wav_chunks[start : start + 16],
                    hf_fe,
                    hf_model,
                    ms_model,
                    sb_model,
                    redimnet2_model,
                    device,
                    sb_normalize,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return out
    if backend == "modelscope":
        # Keep ModelScope extraction as single-utterance forwards for
        # reproducibility and to avoid cross-device/batch-shape edge cases.
        return [
            extract_modelscope_chunk_embedding(wav_chunk, ms_model, device)
            for wav_chunk in wav_chunks
        ]
    if backend == "redimnet2":
        # For long-utterance fallback, running all chunks as one large batch
        # can explode memory. Keep chunk-level forwards to guarantee progress.
        out = []
        for idx, wav_chunk in enumerate(wav_chunks):
            out.append(extract_redimnet2_chunk_embedding(wav_chunk, redimnet2_model, device))
            if device.type == "cuda" and (idx + 1) % 64 == 0:
                torch.cuda.empty_cache()
        return out
    try:
        return extract_non_wespeaker_batch_embeddings(
            backend,
            wav_chunks,
            hf_fe,
            hf_model,
            ms_model,
            sb_model,
            redimnet2_model,
            device,
            sb_normalize,
        )
    except Exception as batch_exc:
        if device.type == "cuda" and "out of memory" in str(batch_exc).lower():
            torch.cuda.empty_cache()
        if len(wav_chunks) == 1:
            return [
                extract_non_wespeaker_chunk_embedding(
                    backend,
                    wav_chunks[0],
                    hf_fe,
                    hf_model,
                    ms_model,
                    sb_model,
                    redimnet2_model,
                    device,
                    sb_normalize,
                )
            ]
        mid = len(wav_chunks) // 2
        left = extract_non_wespeaker_embeddings_adaptive(
            backend,
            wav_chunks[:mid],
            hf_fe,
            hf_model,
            ms_model,
            sb_model,
            redimnet2_model,
            device,
            sb_normalize,
        )
        right = extract_non_wespeaker_embeddings_adaptive(
            backend,
            wav_chunks[mid:],
            hf_fe,
            hf_model,
            ms_model,
            sb_model,
            redimnet2_model,
            device,
            sb_normalize,
        )
        return left + right

def main():
    args = get_args()
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    backend = args.backend.lower()
    print(f"[Extract] Backend: {backend}, Path: {args.model_name}")
    print(f"[Extract] Waveform CMVN: {args.waveform_cmvn}")

    wespeaker_model = None
    hf_fe, hf_model = None, None
    ms_model = None
    sb_model = None
    redimnet2_model = None

    if backend == "wespeaker":
        wespeaker_model = load_wespeaker_model(args.model_name, str(device))
    elif backend == "huggingface":
        hf_fe, hf_model = load_hf_model(args.model_name, device)
    elif backend == "modelscope":
        ms_model = load_modelscope_model(args.model_name, device)
    elif backend == "speechbrain":
        sb_model = load_speechbrain_model(args.model_name, device)
    elif backend == "redimnet2":
        redimnet2_model = load_redimnet2_model(args.model_name, device)

    wav_paths = read_wav_scp(args.wav_scp)
    print(f"[Extract] Found {len(wav_paths)} utterances.")
    sb_normalize = None
    if backend == "speechbrain":
        sb_normalize = resolve_speechbrain_normalize_mode(args.speechbrain_normalize)
        print(f"[Extract] SpeechBrain normalize: {sb_normalize}")
    max_chunk_samples = None
    max_chunk_seconds = args.max_chunk_seconds
    if args.max_chunk_seconds > 0:
        max_chunk_samples = int(max_chunk_seconds * 16000)
    chunk_attempts = build_chunk_attempts(max_chunk_samples)
    if backend in {"redimnet2", "modelscope"} and (not chunk_attempts or chunk_attempts[0] is not None):
        # Match official-style full-utterance inference first, then
        # fall back to chunked retries only when needed.
        chunk_attempts = [None] + chunk_attempts
    batch_size = 1 if backend == "wespeaker" else max(1, args.batch_size or default_batch_size(backend))
    io_workers = 0 if backend == "wespeaker" else max(1, args.io_workers or default_io_workers(backend))
    save_every = max(1, args.save_every)

    embeddings = load_existing_embeddings(args.output)
    if embeddings:
        print(f"[Extract] Resuming from existing cache with {len(embeddings)} embeddings: {args.output}")
    num_err = 0
    new_embeddings_since_save = 0

    pending_batch = []

    def maybe_save_partial(force=False):
        nonlocal new_embeddings_since_save
        if new_embeddings_since_save == 0:
            return
        if not force and new_embeddings_since_save < save_every:
            return
        save_embeddings_atomic(args.output, embeddings)
        print(f"[Extract] Checkpointed {len(embeddings)} embeddings to {args.output}")
        new_embeddings_since_save = 0

    def store_embedding(utt_id, emb):
        nonlocal num_err, new_embeddings_since_save
        emb_np = normalize_embedding_array(emb)
        if emb_np is None:
            num_err += 1
            if num_err <= 5:
                print(f"Warning: Failed to extract {utt_id}: extraction returned no finite embedding")
            return
        embeddings[utt_id] = emb_np
        new_embeddings_since_save += 1
        maybe_save_partial()

    def flush_pending_batch():
        nonlocal num_err
        if not pending_batch:
            return

        utt_ids = [utt_id for utt_id, _ in pending_batch]
        wavs = [wav for _, wav in pending_batch]
        batch_embeddings = extract_non_wespeaker_embeddings_adaptive(
            backend,
            wavs,
            hf_fe,
            hf_model,
            ms_model,
            sb_model,
            redimnet2_model,
            device,
            sb_normalize,
        )

        for utt_id, emb in zip(utt_ids, batch_embeddings):
            store_embedding(utt_id, emb)

        pending_batch.clear()

    if backend == "wespeaker":
        for utt_id, wav_path in tqdm(wav_paths.items(), desc="Extracting"):
            if utt_id in embeddings:
                continue
            try:
                # Use the same robust waveform path for every WeSpeaker utterance.
                # Relying on model.extract_embedding first made long-audio behavior
                # depend on whether a machine happened to OOM and enter fallback.
                emb = extract_wespeaker_embedding_from_waveform(
                    wespeaker_model,
                    wav_path,
                    chunk_attempts=chunk_attempts,
                    device=device,
                )
                if emb is None:
                    raise RuntimeError("extraction returned no finite embedding")
                embeddings[utt_id] = emb
                new_embeddings_since_save += 1
                maybe_save_partial()
            except Exception as e:
                num_err += 1
                if device.type == "cuda" and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                if num_err <= 5:
                    print(f"Warning: Failed to extract {utt_id}: {e}")
    else:
        remaining_items = [(utt_id, wav_path) for utt_id, wav_path in wav_paths.items() if utt_id not in embeddings]
        with tqdm(total=len(wav_paths), initial=len(embeddings), desc="Extracting") as pbar:
            for utt_id, _wav_path, wav in iter_prepared_waveforms(
                remaining_items,
                target_sr=16000,
                io_workers=io_workers,
                backend=backend,
                cmvn_mode=args.waveform_cmvn,
            ):
                pbar.update(1)
                try:
                    if wav is None:
                        continue

                    initial_chunks = split_waveform(wav, max_chunk_samples=max_chunk_samples)
                    if len(initial_chunks) == 1 and batch_size > 1:
                        pending_batch.append((utt_id, initial_chunks[0]))
                        if len(pending_batch) >= batch_size:
                            flush_pending_batch()
                        continue

                    flush_pending_batch()

                    merged = None
                    max_used_chunks = 1
                    for chunk_size in chunk_attempts:
                        try:
                            chunks = split_waveform(wav, max_chunk_samples=chunk_size)
                            max_used_chunks = max(max_used_chunks, len(chunks))
                            chunk_embeddings = extract_non_wespeaker_embeddings_adaptive(
                                backend,
                                chunks,
                                hf_fe,
                                hf_model,
                                ms_model,
                                sb_model,
                                redimnet2_model,
                                device,
                                sb_normalize,
                            )
                            merged = merge_chunk_embeddings(chunk_embeddings)
                        except Exception as chunk_exc:
                            if device.type == "cuda" and "out of memory" in str(chunk_exc).lower():
                                torch.cuda.empty_cache()
                            # Continue to finer chunk retries.
                            continue
                        if merged is not None:
                            break

                    if merged is not None:
                        store_embedding(utt_id, merged)
                    else:
                        raise RuntimeError(
                            f"failed to extract finite embedding after {len(chunk_attempts)} chunk attempts"
                        )

                    if device.type == "cuda" and max_used_chunks > 1:
                        torch.cuda.empty_cache()

                except Exception as e:
                    num_err += 1
                    if device.type == "cuda" and "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                    if num_err <= 5:
                        print(f"Warning: Failed to extract {utt_id}: {e}")

        flush_pending_batch()

    if wav_paths and not embeddings:
        raise RuntimeError(
            f"No embeddings were extracted from {args.wav_scp}. "
            "Check backend/device configuration and input audio decoding."
        )

    maybe_save_partial(force=True)
    save_embeddings_atomic(args.output, embeddings)

    print(f"[Extract] Saved {len(embeddings)} embeddings to {args.output}")

if __name__ == "__main__":
    main()
