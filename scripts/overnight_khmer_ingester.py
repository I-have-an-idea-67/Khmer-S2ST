"""
STAGE 08 - Crash-Proof, Resumable Overnight Khmer Dataset Sharder.

Streams Khmer audio-text pairs lazily from Hugging Face Hub candidates in
priority order:
  1. `phonsobon/openslr42-khmer-male`        (Studio male Khmer, ~2,900 clips)
  2. `seanghay/km-speech-corpus`             (Clean read Khmer, ~14,900 clips)
  3. `rinabuoy/khm-asr-open`                 (Spontaneous & read Khmer, ~26,300 clips)
  4. `Digital-Divide-Data/khm-asr-cultural`  (Multi-speaker cultural Khmer, ~56,700 clips)
  5. `google/fleurs` (`km_kh` config)        (FLEURS Khmer, multi-split fallback)

Storage & Performance Optimizations:
  - RAM stays flat (< 1.5 GB) by writing atomic shards of `--shard-size` clips
    (default 500 clips).
  - Storage footprint cut by 50%: Audio resampled to 16,000 Hz and 24,000 Hz,
    peak-normalized to 0.90, and quantized to PCM16 (`torch.int16`).
  - Pre-tokenizes Khmer transcript into 7-slot tensors (`onset_ids`, `coeng_ids`,
    `medial_ids`, `register_ids`, `vowel_ids`, `coda_ids`, `sesqui_ids`) via the
    Rust-accelerated `BilingualCodeSwitchDecomposer`.
  - Atomically writes `shard_XXXXX.pt` and tracks state in `progress.json` for
    zero-overhead, crash-proof resumption.
  - Monitors free disk space and cleanly halts if `--min-free-gb` threshold is hit.

Usage:
  python scripts/overnight_khmer_ingester.py --shard-size 50 --max-clips 100
  python scripts/overnight_khmer_ingester.py --shard-size 500 --max-clips 15000 --min-free-gb 15.0
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch

# Ensure UTF-8 output on Windows console
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.contracts.schemas import SyllableTensorBatch  # noqa: E402
from backend.linguistics.syllable_decomposer import (  # noqa: E402
    BilingualCodeSwitchDecomposer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR_16K = 16_000
SR_24K = 24_000
MIN_DURATION_SEC = 1.2
MAX_DURATION_SEC = 12.0
PEAK_TARGET = 0.90

SLOT_NAMES = (
    "onset_ids",
    "coeng_ids",
    "medial_ids",
    "register_ids",
    "vowel_ids",
    "coda_ids",
    "sesqui_ids",
)

# Multi-corpus priority list
CANDIDATES: List[Tuple[str, Optional[str], str]] = [
    ("phonsobon/openslr42-khmer-male", None, "train"),
    ("seanghay/km-speech-corpus", None, "train"),
    ("rinabuoy/khm-asr-open", None, "train"),
    ("Digital-Divide-Data/khm-asr-cultural", None, "train"),
    ("google/fleurs", "km_kh", "train"),
    ("google/fleurs", "km_kh", "validation"),
    ("google/fleurs", "km_kh", "test"),
]

_OFFLINE_PATCHES: List[Any] = []


# ============================================================================
# Hugging Face Cache & Offline Helpers
# ============================================================================
def _is_cached(repo: str, config: Optional[str]) -> bool:
    """True when HuggingFace datasets cache holds this dataset."""
    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "datasets")
    )
    flat = repo.replace("/", "___")
    base = cache_root / flat
    if not base.is_dir():
        return False
    if config is None:
        return any(base.iterdir())
    cfg_dir = base / config
    return cfg_dir.is_dir() and any(cfg_dir.iterdir())


def _set_offline(enabled: bool) -> Optional[bool]:
    """Enable or disable HuggingFace offline mode."""
    if not enabled:
        return None
    os.environ["HF_HUB_OFFLINE"] = "1"
    previous: List[Optional[bool]] = []
    for module_name in ("huggingface_hub.constants", "datasets.config"):
        try:
            module = __import__(module_name, fromlist=["*"])
            if hasattr(module, "HF_HUB_OFFLINE"):
                previous.append((module, module.HF_HUB_OFFLINE))
                module.HF_HUB_OFFLINE = True
        except Exception:
            pass
    _OFFLINE_PATCHES.extend(previous)
    return True


def _restore_offline() -> None:
    """Undo every patch applied by _set_offline(True)."""
    while _OFFLINE_PATCHES:
        module, value = _OFFLINE_PATCHES.pop()
        try:
            module.HF_HUB_OFFLINE = value
        except Exception:
            pass
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        os.environ.pop("HF_HUB_OFFLINE", None)


def _open_stream(repo: str, config: Optional[str], split: str, offline: bool = False):
    """Open streaming IterableDataset with raw audio bytes (avoiding torchcodec)."""
    from datasets import Audio, load_dataset

    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if offline:
        _set_offline(True)
    else:
        _restore_offline()

    if config is not None:
        ds = load_dataset(repo, config, **kwargs)
    else:
        ds = load_dataset(repo, **kwargs)
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds


# ============================================================================
# Audio & Khmer Processing Helpers
# ============================================================================
def _decode_audio(audio_field: Any) -> Tuple[Optional[Any], Optional[int]]:
    """Decode raw audio bytes via soundfile to mono float32 numpy array."""
    import numpy as np
    import soundfile as sf

    if isinstance(audio_field, dict):
        raw = audio_field.get("bytes")
        if raw is None:
            return None, None
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    else:
        if hasattr(audio_field, "get_array"):
            data = audio_field.get_array().numpy()
            sr = int(audio_field.sampling_rate)
        else:
            return None, None
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def _resample_to(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample mono float32 waveform using torchaudio.functional.resample."""
    import torchaudio.functional as AF

    if orig_sr == target_sr:
        return wav
    return AF.resample(wav, orig_sr, target_sr)


def _peak_normalize(wav: torch.Tensor, peak: float = PEAK_TARGET) -> torch.Tensor:
    """Peak-normalize waveform to `peak` target."""
    mx = float(wav.abs().max().item())
    if mx <= 0.0 or not torch.isfinite(torch.tensor(mx)).item():
        return wav
    return wav * (peak / mx)


def _quantize_pcm16(wav: torch.Tensor) -> torch.Tensor:
    """Quantize normalized float32 waveform (-1.0 to 1.0) into torch.int16."""
    return (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)


def _has_khmer(text: str) -> bool:
    """True if string contains at least one Khmer unicode character."""
    return any("\u1780" <= ch <= "\u17ff" for ch in text)


def _extract_transcript(example: Dict[str, Any]) -> str:
    """Extract candidate transcript string from row."""
    for key in (
        "transcript",
        "sentence",
        "transcription",
        "raw_transcription",
        "text",
        "normalized_text",
    ):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_clip_id(example: Dict[str, Any], repo: str, idx: int) -> str:
    """Derive deterministic, collision-resistant clip identifier."""
    for key in ("id", "sentence_id", "audio_id", "path"):
        val = example.get(key)
        if val is not None and str(val).strip():
            return f"{repo}:{str(val).strip()}"
    audio = example.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return f"{repo}:{Path(audio['path']).stem}"
    return f"{repo}:clip_{idx:07d}"


# ============================================================================
# Resumable State Tracker
# ============================================================================
class ProgressTracker:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.progress_file = out_dir / "progress.json"
        self.completed_shards: List[str] = []
        self.ingested_ids: Set[str] = set()
        self.total_clips_saved: int = 0
        self.next_shard_index: int = 0
        self.load()

    def load(self) -> None:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.completed_shards = data.get("completed_shards", [])
                self.ingested_ids = set(data.get("ingested_ids", []))
                self.total_clips_saved = data.get("total_clips_saved", 0)
                self.next_shard_index = data.get(
                    "next_shard_index", len(self.completed_shards)
                )
                return
            except Exception as e:
                print(f"  [WARN] Failed to read {self.progress_file}: {e}")

        # Fallback: inspect disk for existing shards
        existing_shards = sorted(self.out_dir.glob("shard_*.pt"))
        if existing_shards:
            self.completed_shards = [s.name for s in existing_shards]
            self.next_shard_index = len(self.completed_shards)
            # Count clips across existing shards
            total = 0
            for shard_path in existing_shards:
                try:
                    shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
                    total += len(shard_data)
                    for item in shard_data:
                        if "clip_id" in item:
                            self.ingested_ids.add(item["clip_id"])
                except Exception:
                    pass
            self.total_clips_saved = total

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = self.progress_file.with_suffix(".json.tmp")
        data = {
            "total_clips_saved": self.total_clips_saved,
            "next_shard_index": self.next_shard_index,
            "completed_shards": self.completed_shards,
            "ingested_ids": list(self.ingested_ids),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self.progress_file)


# ============================================================================
# Candidate Probe & Streaming Pipeline
# ============================================================================
def _iter_viable_candidates(
    probe_timeout_sec: float = 25.0,
) -> Iterator[Tuple[str, Optional[str], str, Any]]:
    """Probe candidate datasets in priority order and yield active streams."""
    for repo, config, split in CANDIDATES:
        label = f"{repo}" + (f"/{config}" if config else "") + f" [{split}]"
        print(f"\n[candidate-probe] Probing {label}...", flush=True)

        cached = _is_cached(repo, config)
        order = (True, False) if cached else (False,)

        for offline in order:
            mode_desc = "cached offline" if offline else "live stream"
            print(f"  -> attempting {mode_desc}...", flush=True)
            try:
                ds = _open_stream(repo, config, split, offline=offline)
            except Exception as exc:
                print(
                    f"  -> SKIPPED (cannot open): {type(exc).__name__}: "
                    f"{str(exc).splitlines()[0][:120]}",
                    flush=True,
                )
                continue

            q: "queue.Queue[Any]" = queue.Queue(maxsize=1)

            def _pull(iterator, out_q):
                try:
                    out_q.put(("ok", next(iterator)))
                except BaseException as b_exc:
                    out_q.put(("err", b_exc))

            thread = threading.Thread(
                target=_pull, args=(iter(ds), q), daemon=True
            )
            thread.start()

            try:
                status, payload = q.get(timeout=probe_timeout_sec)
            except queue.Empty:
                print(
                    f"  -> SKIPPED (no first row within {probe_timeout_sec:.0f}s timeout)",
                    flush=True,
                )
                continue

            if status == "err":
                print(
                    f"  -> SKIPPED (row fetch error): {type(payload).__name__}: "
                    f"{str(payload).splitlines()[0][:120]}",
                    flush=True,
                )
                continue

            print(
                f"  -> OK (candidate verified active, first row received: {label})",
                flush=True,
            )
            try:
                yield repo, config, split, ds
            finally:
                if offline:
                    _restore_offline()
            # If this candidate opened and yielded, advance once exhausted


# ============================================================================
# Main Sharder Engine
# ============================================================================
def run_ingestion(
    out_dir_path: Path,
    shard_size: int = 500,
    max_clips: int = 15000,
    min_free_gb: float = 15.0,
    probe_timeout: float = 25.0,
) -> int:
    t_start = time.time()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    tracker = ProgressTracker(out_dir_path)

    print("=" * 78)
    print("STAGE 08 - RESUMABLE OVERNIGHT KHMER DATASET SHARDER")
    print("=" * 78)
    print(f"Output directory : {out_dir_path}")
    print(f"Shard size       : {shard_size} clips / shard")
    print(f"Max clips target : {max_clips}")
    print(f"Min free disk    : {min_free_gb:.1f} GB")
    print(f"Duration window  : [{MIN_DURATION_SEC}, {MAX_DURATION_SEC}] s")
    print(f"Audio storage    : PCM16 (torch.int16) @ 16kHz & 24kHz")
    print(f"Already ingested : {tracker.total_clips_saved} clips across {len(tracker.completed_shards)} shards")

    # Fast resume exit if target already met
    if tracker.total_clips_saved >= max_clips:
        print(
            f"\n[RESUME] Target of {max_clips} clips already reached "
            f"({tracker.total_clips_saved} clips on disk). "
            f"0 duplicate shards written in {time.time()-t_start:.2f}s."
        )
        return 0

    # Initial disk check
    free_gb = shutil.disk_usage(out_dir_path).free / (1024**3)
    if free_gb < min_free_gb:
        print(
            f"[HALT] Insufficient free disk space ({free_gb:.2f} GB < {min_free_gb:.2f} GB). "
            f"Exiting cleanly without shard creation."
        )
        return 0

    decomposer = BilingualCodeSwitchDecomposer()
    current_shard_buffer: List[Dict[str, Any]] = []
    current_shard_ids: List[str] = []
    scanned_total = 0
    rejected: Dict[str, int] = {
        "duration": 0,
        "transcript": 0,
        "no_khmer": 0,
        "decode": 0,
        "already_ingested": 0,
    }

    def _flush_shard():
        nonlocal current_shard_buffer, current_shard_ids
        if not current_shard_buffer:
            return

        # Check disk space before saving
        f_gb = shutil.disk_usage(out_dir_path).free / (1024**3)
        if f_gb < min_free_gb:
            print(
                f"\n[DISK CEILING REACHED] Free disk space {f_gb:.2f} GB < {min_free_gb:.2f} GB. "
                f"Saving final shard and halting."
            )

        shard_filename = f"shard_{tracker.next_shard_index:05d}.pt"
        tmp_path = out_dir_path / f"{shard_filename}.tmp"
        final_path = out_dir_path / shard_filename

        torch.save(current_shard_buffer, tmp_path)
        os.replace(tmp_path, final_path)

        shard_mb = final_path.stat().st_size / (1024**2)
        n_saved = len(current_shard_buffer)
        tracker.completed_shards.append(shard_filename)
        tracker.total_clips_saved += n_saved
        tracker.ingested_ids.update(current_shard_ids)
        tracker.next_shard_index += 1
        tracker.save()

        print(
            f"\n>>> [FLUSH] Wrote {shard_filename} ({n_saved} clips, {shard_mb:.2f} MiB). "
            f"Total progress: {tracker.total_clips_saved}/{max_clips} clips "
            f"({time.time()-t_start:.1f}s elapsed, {f_gb:.1f} GB free)",
            flush=True,
        )

        current_shard_buffer = []
        current_shard_ids = []

    for repo, config, split, ds in _iter_viable_candidates(probe_timeout_sec=probe_timeout):
        if tracker.total_clips_saved + len(current_shard_buffer) >= max_clips:
            break

        label = f"{repo}" + (f"/{config}" if config else "")
        print(f"\n[streaming] Active source: {label} [{split}]", flush=True)

        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                iterator = iter(ds)
                idx = 0
                for example in iterator:
                    if tracker.total_clips_saved + len(current_shard_buffer) >= max_clips:
                        break

                    scanned_total += 1
                    idx += 1

                    clip_id = _extract_clip_id(example, repo, idx)
                    if clip_id in tracker.ingested_ids or clip_id in current_shard_ids:
                        rejected["already_ingested"] += 1
                        continue

                    transcript = _extract_transcript(example)
                    if not transcript:
                        rejected["transcript"] += 1
                        continue
                    if not _has_khmer(transcript):
                        rejected["no_khmer"] += 1
                        continue

                    data, sr = _decode_audio(example.get("audio"))
                    if data is None or sr is None or len(data) == 0:
                        rejected["decode"] += 1
                        continue

                    duration = len(data) / float(sr)
                    if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
                        rejected["duration"] += 1
                        continue

                    # Resample, peak-normalize, and quantize to PCM16
                    wav_src = torch.from_numpy(data).unsqueeze(0)  # (1, T) float32
                    wav_16k_f32 = _peak_normalize(_resample_to(wav_src, sr, SR_16K))
                    wav_24k_f32 = _peak_normalize(_resample_to(wav_src, sr, SR_24K))

                    wav_16k_pcm16 = _quantize_pcm16(wav_16k_f32)[0]
                    wav_24k_pcm16 = _quantize_pcm16(wav_24k_f32)[0]

                    # 7-slot decomposition
                    batch: SyllableTensorBatch = decomposer.encode_sentence_to_batch(
                        transcript, device="cpu"
                    )
                    slot_tensors = {
                        slot: getattr(batch, slot)[0].to(torch.int16)
                        for slot in SLOT_NAMES
                    }

                    clip_record = {
                        "clip_id": clip_id,
                        "source_dataset": repo,
                        "transcript": transcript,
                        "wav_16k": wav_16k_pcm16,
                        "wav_24k": wav_24k_pcm16,
                        **slot_tensors,
                    }

                    current_shard_buffer.append(clip_record)
                    current_shard_ids.append(clip_id)

                    # Flush if shard full
                    if len(current_shard_buffer) >= shard_size:
                        _flush_shard()

                        # Check if disk limit reached
                        free_disk_gb = shutil.disk_usage(out_dir_path).free / (1024**3)
                        if free_disk_gb < min_free_gb:
                            print(f"[HALT] Min free disk limit reached ({free_disk_gb:.2f} GB).")
                            return 0

                    if (
                        len(current_shard_buffer) % 25 == 0
                        and len(current_shard_buffer) > 0
                    ):
                        print(
                            f"  [buffering] shard clip {len(current_shard_buffer):>3}/{shard_size} "
                            f"(total saved: {tracker.total_clips_saved}, scanned: {scanned_total})",
                            flush=True,
                        )

                # Completed iteration of this candidate without exception
                break

            except Exception as stream_err:
                retry_count += 1
                backoff = 2**retry_count
                print(
                    f"  [WARN] Stream hiccup on {label}: {type(stream_err).__name__}: {stream_err}. "
                    f"Retry {retry_count}/{max_retries} in {backoff}s...",
                    flush=True,
                )
                time.sleep(backoff)

        # After candidate loop, flush any partial shard if max_clips reached
        if tracker.total_clips_saved + len(current_shard_buffer) >= max_clips:
            _flush_shard()
            break

    # Final flush of remaining buffer
    if current_shard_buffer:
        _flush_shard()

    print("\n" + "=" * 78)
    print("INGESTION SUMMARY")
    print("=" * 78)
    print(f"Total clips saved : {tracker.total_clips_saved}")
    print(f"Total shards      : {len(tracker.completed_shards)}")
    print(f"Total scanned     : {scanned_total}")
    print(f"Rejected breakdown: {rejected}")
    print(f"Total time        : {time.time()-t_start:.1f}s")
    return 0


# ============================================================================
# Self-Audit Verification Routine
# ============================================================================
def verify_shards(out_dir: Path, expected_min_shards: int = 1) -> bool:
    """Rigorous audit gate verifying PCM16 audio, slot bounds, and shard size."""
    print("\n" + "-" * 78)
    print("STEP 2 SELF-AUDIT VERIFICATION GATE")
    print("-" * 78)

    shards = sorted(out_dir.glob("shard_*.pt"))
    assert len(shards) >= expected_min_shards, (
        f"Expected at least {expected_min_shards} shards, found {len(shards)}"
    )

    shard_0 = shards[0]
    size_mb = shard_0.stat().st_size / (1024**2)
    print(f"[AUDIT] Checking {shard_0.name} (file size: {size_mb:.2f} MiB)...")
    assert size_mb < 40.0, f"Shard file size {size_mb:.2f} MiB exceeds 40 MB ceiling"

    data = torch.load(shard_0, map_location="cpu", weights_only=False)
    assert isinstance(data, list) and len(data) > 0, "Shard data must be a non-empty list"
    print(f"[AUDIT] Shard {shard_0.name} contains {len(data)} clips.")

    for i, clip in enumerate(data):
        assert isinstance(clip, dict), f"Clip {i} is not a dict"
        assert "wav_16k" in clip and "wav_24k" in clip, f"Clip {i} missing audio tensors"
        assert clip["wav_16k"].dtype == torch.int16, f"Clip {i} wav_16k dtype != torch.int16"
        assert clip["wav_24k"].dtype == torch.int16, f"Clip {i} wav_24k dtype != torch.int16"
        assert clip["wav_16k"].ndim == 1, f"Clip {i} wav_16k must be 1-D mono"
        assert clip["wav_24k"].ndim == 1, f"Clip {i} wav_24k must be 1-D mono"

        assert isinstance(clip.get("transcript"), str) and len(clip["transcript"]) > 0, (
            f"Clip {i} transcript must be non-empty string"
        )
        assert _has_khmer(clip["transcript"]), f"Clip {i} transcript must contain Khmer text"
        assert isinstance(clip.get("source_dataset"), str), f"Clip {i} missing source_dataset"
        assert isinstance(clip.get("clip_id"), str), f"Clip {i} missing clip_id"

        # 7-slot tensor verification
        for slot in SLOT_NAMES:
            assert slot in clip, f"Clip {i} missing slot {slot}"
            t = clip[slot]
            assert isinstance(t, torch.Tensor), f"Clip {i} slot {slot} is not a Tensor"
            assert t.dtype in (torch.int16, torch.int64, torch.uint8), (
                f"Clip {i} slot {slot} has unexpected dtype {t.dtype}"
            )
            assert t.ndim == 1, f"Clip {i} slot {slot} must be 1-D"
            assert len(t) > 0, f"Clip {i} slot {slot} is empty"
            assert t.min().item() >= 0, f"Clip {i} slot {slot} has negative token ID {t.min().item()}"

    print(f"[AUDIT] All {len(data)} clips in {shard_0.name} PASSED all verification checks.")
    print("STEP 2 SELF-AUDIT VERIFICATION GATE: PASSED [100% OK]")
    print("-" * 78)
    return True


# ============================================================================
# CLI Entrypoint
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="STAGE 08 - Crash-proof, resumable overnight Khmer dataset sharder."
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "data" / "khmer_overnight_shards"),
        help="Target directory for output shards and progress tracker.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Number of valid clips per shard file (default: 500).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=15000,
        help="Total clips to collect across all candidate sources (default: 15000).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=15.0,
        help="Minimum free disk space in GB required to continue writing shards (default: 15.0).",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=25.0,
        help="Timeout in seconds for candidate dataset probe (default: 25.0).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run self-audit verification gate on existing shards in --out-dir.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.verify_only:
        verify_shards(out_dir)
        return 0

    ret = run_ingestion(
        out_dir_path=out_dir,
        shard_size=args.shard_size,
        max_clips=args.max_clips,
        min_free_gb=args.min_free_gb,
        probe_timeout=args.probe_timeout,
    )
    if ret != 0:
        return ret

    # Run verification on produced shards
    verify_shards(out_dir, expected_min_shards=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
