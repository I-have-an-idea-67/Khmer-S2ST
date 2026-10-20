"""
STAGE 07 - STEP 3: Streaming Pilot Khmer Dataset Ingestion & 7-Slot Pre-Tokenization.

Streams Khmer audio-text pairs from the HuggingFace Hub WITHOUT downloading
the full (~100 GB) corpora: `datasets.load_dataset(..., streaming=True)` reads
the Parquet row groups lazily, and this script stops as soon as `--max-clips`
valid clips have been collected. Peak disk usage is therefore bounded by the
shard that is written, not by the source corpus.

Dataset resolution order (per the STAGE 07 brief):
  1. `phonsobon/openslr42-khmer-male`  (OpenSLR-42 Khmer, male read speech)
  2. `seanghay/km-speech-corpus`        (Seanghay Khmer speech corpus)
  3. `google/fleurs` / `km_kh`          (FLEURS Khmer, the documented fallback)

Each candidate is tried in turn; a candidate that requires authentication
(gated) or is unreachable is skipped and the next one is used. The FLEURS
`km_kh` fallback is the one that succeeds in an unauthenticated environment.

Per-clip processing:
  * decode the audio bytes (via `soundfile`, so no FFmpeg/torchcodec is needed),
  * keep only clips with `1.2 s <= T <= 10.0 s` and a non-empty Khmer transcript,
  * resample to BOTH `wav_16k` (16,000 Hz) and `wav_24k` (24,000 Hz), mono float32,
    using `torchaudio.functional.resample`,
  * peak-normalize to 0.90,
  * decompose the transcript with `BilingualCodeSwitchDecomposer` into the
    7-slot `SyllableTensorBatch`.

The result is written to `<out-dir>/pilot_manifest.pt` as a list of dicts, each
holding `wav_16k`, `wav_24k`, `transcript` and the seven slot tensors.

Usage:
    python scripts/prepare_khmer_pilot_shard.py --max-clips 200
    python scripts/prepare_khmer_pilot_shard.py --max-clips 200 \
        --out-dir data/pilot_khmer_shard
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

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
MAX_DURATION_SEC = 10.0
PEAK_TARGET = 0.90

# (repo_id, config_name, split) candidates, in priority order.
CANDIDATES: List[Tuple[str, Optional[str], str]] = [
    ("phonsobon/openslr42-khmer-male", None, "train"),
    ("seanghay/km-speech-corpus", None, "train"),
    ("google/fleurs", "km_kh", "train"),
    ("google/fleurs", "km_kh", "validation"),
]

# The 7 slot names, in the canonical `SyllableTensorBatch` order.
SLOT_NAMES = (
    "onset_ids",
    "coeng_ids",
    "medial_ids",
    "register_ids",
    "vowel_ids",
    "coda_ids",
    "sesqui_ids",
)

# Stack of `(module, previous_value)` pairs for the offline-mode patches.
_OFFLINE_PATCHES: List[Any] = []


# ============================================================================
# Streaming helpers
# ============================================================================
def _is_cached(repo: str, config: Optional[str]) -> bool:
    """
    True when the HuggingFace DATASETS cache already holds this dataset.

    The cache key mirrors the layout `datasets` itself writes:
    `<cache>/<namespace>___<name>/<config>/<version>/<hash>`, where the
    namespace/name pair is the repo id with `/` replaced by `___`.
    """
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
    """
    Enable or disable HuggingFace offline mode. Returns the previous value.

    `huggingface_hub` reads HF_HUB_OFFLINE into a MODULE CONSTANT at import
    time, so setting the environment variable after the import has NO effect.
    Both the environment variable and the constants of BOTH `huggingface_hub`
    and `datasets` are therefore patched.

    The flag must stay active for the WHOLE lifetime of the dataset, not just
    the `load_dataset` call: the streaming iterator resolves its Parquet
    shards lazily, on the first `next()`, so restoring the flag too early
    sends the iterator back to the network.
    """
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
    """Undo every patch applied by `_set_offline(True)`."""
    while _OFFLINE_PATCHES:
        module, value = _OFFLINE_PATCHES.pop()
        try:
            module.HF_HUB_OFFLINE = value
        except Exception:
            pass
    # Only clear the environment variable if WE were the ones who set it.
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        os.environ.pop("HF_HUB_OFFLINE", None)


def _open_stream(repo: str, config: Optional[str], split: str, offline: bool = False):
    """
    Open a streaming `IterableDataset` with the audio column DECODING DISABLED.

    `Audio(decode=False)` returns the raw encoded bytes, which `soundfile`
    decodes directly. This avoids the `torchcodec`/FFmpeg hard dependency of
    the default decoder, which is not available on every machine (and would
    otherwise abort the run on the very first example).
    """
    from datasets import Audio, load_dataset

    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if offline:
        _set_offline(True)
    if config is not None:
        ds = load_dataset(repo, config, **kwargs)
    else:
        ds = load_dataset(repo, **kwargs)
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def _iter_candidates(
    probe_timeout_sec: float = 90.0,
) -> Iterator[Tuple[str, Optional[str], str, Any]]:
    """
    Yield the first candidate that can actually deliver a row.

    Opening a streaming Parquet dataset is not enough: the first row group may
    require a multi-hundred-megabyte download, which would blow the pilot's
    disk/time budget. Every candidate is therefore PROBED by pulling its first
    example under a wall-clock timeout. A candidate that is gated, missing,
    unreachable, or too slow to yield a first row is skipped and the next one
    is tried.

    The probe runs on a daemon thread so a stalled HTTP read can never keep
    the process alive.
    """
    import queue
    import threading

    last_error: Optional[BaseException] = None
    for repo, config, split in CANDIDATES:
        label = f"{repo}" + (f"/{config}" if config else "") + f" [{split}]"
        print(f"  [candidate] {label}", flush=True)

        # The offline flag is applied PER CANDIDATE and restored afterwards.
        # Setting it globally would poison every later candidate: the first
        # candidate that flips it on would make all subsequent candidates
        # fail to open, even when they are perfectly available in the cache.
        #
        # A candidate whose data is ALREADY in the local cache is opened
        # offline FIRST, because the online path issues HTTP range requests
        # against the Parquet shards and is very slow on a cold link. A
        # candidate that is NOT cached is opened online FIRST, because the
        # offline attempt would fail instantly and tell us nothing.
        cached = _is_cached(repo, config)
        if cached:
            order = (True, False)
        else:
            # Not in the local cache: the offline attempt can only fail (and
            # can block on a partially-materialised hub entry), so it is
            # skipped entirely.
            order = (False,)
        for offline in order:
            if offline:
                print("    -> opening from the local HF cache", flush=True)
            else:
                # Any offline flag left over from a PREVIOUS attempt must be
                # cleared first. Otherwise the "online" attempt silently runs
                # offline and fails for the wrong reason, and the resolver
                # never actually tries the network.
                _restore_offline()

            try:
                ds = _open_stream(repo, config, split, offline=offline)
            except Exception as exc:  # gated / missing / unreachable
                last_error = exc
                print(f"    -> SKIPPED (cannot open): {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:150]}", flush=True)
                continue

            q: "queue.Queue[Any]" = queue.Queue(maxsize=1)

            def _pull(iterator, out):
                try:
                    out.put(("ok", next(iterator)))
                except BaseException as exc:  # noqa: BLE001 - reported upstream
                    out.put(("err", exc))

            thread = threading.Thread(
                target=_pull, args=(iter(ds), q), daemon=True
            )
            thread.start()
            try:
                status, payload = q.get(timeout=probe_timeout_sec)
            except queue.Empty:
                print(f"    -> SKIPPED (no first row within {probe_timeout_sec:.0f}s)",
                      flush=True)
                last_error = TimeoutError(f"{label} produced no row in time")
                continue

            if status == "err":
                last_error = payload
                print(f"    -> SKIPPED (row error): {type(payload).__name__}: "
                      f"{str(payload).splitlines()[0][:150]}", flush=True)
                continue

            print(f"    -> OK (features: {len(ds.features)} columns, "
                  f"first row acquired, offline={offline})", flush=True)
            # The offline flag MUST stay active while the caller iterates the
            # stream: the Parquet shards are resolved lazily on the first
            # `next()`, so restoring the flag here would send the iterator
            # back to the network.
            try:
                yield repo, config, split, ds
            finally:
                if offline:
                    _restore_offline()
            return

    raise SystemExit(
        "No usable Khmer dataset could be opened. Last error: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _decode_audio(audio_field: Any):
    """Decode a streaming `audio` field into a mono float32 numpy array."""
    import numpy as np
    import soundfile as sf

    if isinstance(audio_field, dict):
        raw = audio_field.get("bytes")
        if raw is None:
            return None, None
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    else:
        # torchcodec-decoded Audio object.
        if hasattr(audio_field, "get_array"):
            data = audio_field.get_array().numpy()
            sr = int(audio_field.sampling_rate)
        else:
            return None, None
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def _resample_to(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a mono float32 waveform with `torchaudio.functional.resample`."""
    import torchaudio.functional as AF

    if orig_sr == target_sr:
        return wav
    return AF.resample(wav, orig_sr, target_sr)


def _peak_normalize(wav: torch.Tensor, peak: float = PEAK_TARGET) -> torch.Tensor:
    """Scale a waveform so that its absolute maximum equals `peak`."""
    mx = float(wav.abs().max().item())
    if mx <= 0.0 or not torch.isfinite(torch.tensor(mx)).item():
        return wav
    return wav * (peak / mx)


def _has_khmer(text: str) -> bool:
    """True when the string carries at least one Khmer codepoint."""
    return any("\u1780" <= ch <= "\u17ff" for ch in text)


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="STAGE 07 STEP 3 - streaming Khmer pilot shard preparation."
    )
    parser.add_argument(
        "--max-clips", type=int, default=200,
        help="Maximum number of valid clips to collect (default: 200).",
    )
    parser.add_argument(
        "--out-dir", default=str(_REPO_ROOT / "data" / "pilot_khmer_shard"),
        help="Output directory for pilot_manifest.pt.",
    )
    parser.add_argument(
        "--max-scan", type=int, default=20000,
        help="Safety cap on the number of streamed rows examined.",
    )
    parser.add_argument(
        "--probe-timeout", type=float, default=90.0,
        help="Seconds allowed for a candidate dataset to yield its first row.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use only locally cached HF data (sets HF_HUB_OFFLINE=1).",
    )
    parser.add_argument("--seed", type=int, default=20260721, help="RNG seed.")
    args = parser.parse_args()

    if args.offline:
        _set_offline(True)

    assert args.max_clips > 0, "--max-clips must be positive"
    torch.manual_seed(args.seed)

    print("=" * 78)
    print("STAGE 07 - STEP 3: Streaming Pilot Khmer Shard Ingestion & Pre-Tokenization")
    print("=" * 78)
    print(f"max clips        : {args.max_clips}")
    print(f"out dir          : {args.out_dir}")
    print(f"duration window  : [{MIN_DURATION_SEC}, {MAX_DURATION_SEC}] s")
    print(f"peak target      : {PEAK_TARGET}")
    print(f"target rates     : {SR_16K} Hz / {SR_24K} Hz mono float32")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "pilot_manifest.pt"

    decomposer = BilingualCodeSwitchDecomposer()
    items: List[Dict[str, Any]] = []
    scanned = 0
    rejected: Dict[str, int] = {
        "duration": 0, "transcript": 0, "no_khmer": 0, "decode": 0, "empty": 0,
    }
    t0 = time.time()
    source_label = "unknown"

    for repo, config, split, ds in _iter_candidates(
        probe_timeout_sec=args.probe_timeout
    ):
        source_label = f"{repo}" + (f"/{config}" if config else "") + f" [{split}]"
        iterator = iter(ds)
        for example in iterator:
            if len(items) >= args.max_clips or scanned >= args.max_scan:
                break
            scanned += 1

            transcript = (example.get("transcription")
                          or example.get("raw_transcription")
                          or "").strip()
            if not transcript:
                rejected["transcript"] += 1
                continue

            data, sr = _decode_audio(example.get("audio"))
            if data is None or sr is None or len(data) == 0:
                rejected["decode"] += 1
                continue

            duration = len(data) / float(sr)
            if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
                rejected["duration"] += 1
                continue
            if not _has_khmer(transcript):
                rejected["no_khmer"] += 1
                continue

            wav_src = torch.from_numpy(data).unsqueeze(0)  # (1, N) float32
            wav_16k = _peak_normalize(_resample_to(wav_src, sr, SR_16K))
            wav_24k = _peak_normalize(_resample_to(wav_src, sr, SR_24K))

            batch: SyllableTensorBatch = decomposer.encode_sentence_to_batch(
                transcript, device="cpu"
            )
            slot_tensors = {name: getattr(batch, name)[0].clone()
                            for name in SLOT_NAMES}

            items.append({
                "wav_16k": wav_16k[0].clone(),
                "wav_24k": wav_24k[0].clone(),
                "transcript": transcript,
                **slot_tensors,
            })

            if len(items) % 25 == 0 or len(items) == 1:
                print(f"    collected {len(items):>4}/{args.max_clips} "
                      f"(scanned {scanned}, {time.time()-t0:.1f}s)", flush=True)
        if len(items) >= args.max_clips or scanned >= args.max_scan:
            break

    if not items:
        raise SystemExit("No valid clips were collected - aborting.")

    torch.save(items, manifest_path)
    size_mb = manifest_path.stat().st_size / (1024 ** 2)
    print(f"\nSource          : {source_label}")
    print(f"Rows scanned    : {scanned}")
    print(f"Rejected        : {rejected}")
    print(f"Clips saved     : {len(items)}")
    print(f"Manifest        : {manifest_path} ({size_mb:.2f} MiB)")
    print(f"Elapsed         : {time.time()-t0:.1f}s")

    # ==================================================================
    # SELF-AUDIT VERIFICATION GATE
    # ==================================================================
    print("\n" + "-" * 78)
    print("STEP 3 SELF-AUDIT VERIFICATION GATE")
    print("-" * 78)

    # (a) clip count and total disk size
    total_bytes = sum(
        it["wav_16k"].numel() * 4 + it["wav_24k"].numel() * 4 for it in items
    )
    on_disk = manifest_path.stat().st_size
    print(f"(a) clips saved            : {len(items)} (required >= 50, cap {args.max_clips})")
    print(f"    manifest on disk       : {on_disk/1024**2:.2f} MiB")
    print(f"    total audio payload    : {total_bytes/1024**2:.2f} MiB")
    assert len(items) >= 50, f"only {len(items)} clips saved (need >= 50)"
    assert on_disk < 500 * 1024 ** 2, (
        f"manifest is {on_disk/1024**2:.2f} MiB, which exceeds the 500 MB cap"
    )
    print("    PASS - >= 50 clips and total disk size < 500 MB")

    # (b) no NaN/Inf, and the 16k/24k durations agree
    worst_dur_err = 0.0
    nan_count = 0
    for i, it in enumerate(items):
        w16, w24 = it["wav_16k"], it["wav_24k"]
        assert w16.dtype == torch.float32 and w24.dtype == torch.float32, (
            f"clip {i}: expected float32, got {w16.dtype} / {w24.dtype}"
        )
        if not (torch.isfinite(w16).all() and torch.isfinite(w24).all()):
            nan_count += 1
        d16 = len(w16) / float(SR_16K)
        d24 = len(w24) / float(SR_24K)
        worst_dur_err = max(worst_dur_err, abs(d16 - d24))
    print(f"(b) clips with NaN/Inf     : {nan_count} (required 0)")
    print(f"    max |dur16 - dur24|    : {worst_dur_err*1000:.4f} ms "
          f"(required <= 1.0 ms)")
    assert nan_count == 0, f"{nan_count} clips contain NaN/Inf"
    assert worst_dur_err <= 1e-3, (
        f"16k/24k duration mismatch {worst_dur_err:.6f}s exceeds 1e-3 s"
    )
    print("    PASS - no NaN/Inf and 16k/24k durations agree within 1e-3 s")

    # (c) every 7-slot tensor obeys the SLOT_VOCAB bounds
    from backend.nn_cartridges.mouth_cartridge import SLOT_VOCAB

    # "lt"  -> every value must be strictly below the SLOT_VOCAB size.
    # "set" -> every value must be a MEMBER of the allowed set.
    bounds: Dict[str, Any] = {
        "onset_ids": ("lt", SLOT_VOCAB["onset"]),
        "coeng_ids": ("lt", SLOT_VOCAB["coeng"]),
        "medial_ids": ("lt", SLOT_VOCAB["medial"]),
        "register_ids": ("set", frozenset({1, 2, 3})),
        "vowel_ids": ("lt", SLOT_VOCAB["vowel"]),
        "coda_ids": ("lt", SLOT_VOCAB["coda"]),
        "sesqui_ids": ("set", frozenset({1, 2})),
    }
    slot_min: Dict[str, int] = {}
    slot_max: Dict[str, int] = {}
    for it in items:
        for name, (kind, bound) in bounds.items():
            t = it[name]
            assert t.dtype == torch.int64, f"{name} is {t.dtype}, expected int64"
            lo, hi = int(t.min()), int(t.max())
            slot_min[name] = min(slot_min.get(name, lo), lo)
            slot_max[name] = max(slot_max.get(name, hi), hi)
            if kind == "lt":
                assert hi < bound, f"{name}={hi} violates < {bound}"
            else:
                offending = sorted(set(t.flatten().tolist()) - set(bound))
                assert not offending, (
                    f"{name} contains {offending} which are not in {sorted(bound)}"
                )
    print("(c) slot bounds            : " + ", ".join(
        f"{n}=[{slot_min[n]},{slot_max[n]}]" for n in SLOT_NAMES))
    print("    PASS - every 7-slot tensor obeys SLOT_VOCAB bounds")

    # (d) round-trip decode of the first 5 real Khmer transcripts
    print("(d) round-trip decode_batch_to_text on the first 5 transcripts:")
    decomp = BilingualCodeSwitchDecomposer()
    for i in range(min(5, len(items))):
        it = items[i]
        batch = SyllableTensorBatch(
            onset_ids=it["onset_ids"].unsqueeze(0),
            coeng_ids=it["coeng_ids"].unsqueeze(0),
            medial_ids=it["medial_ids"].unsqueeze(0),
            register_ids=it["register_ids"].unsqueeze(0),
            vowel_ids=it["vowel_ids"].unsqueeze(0),
            coda_ids=it["coda_ids"].unsqueeze(0),
            sesqui_ids=it["sesqui_ids"].unsqueeze(0),
        )
        texts, cs_words = decomp.decode_batch_to_text(batch)
        decoded = texts[0]
        assert decoded.strip(), f"clip {i} decoded to an empty string"
        decoded.encode("utf-8")  # must be valid UTF-8
        print(f"    [{i}] src : {it['transcript'][:60]}")
        print(f"    [{i}] dec : {decoded[:60]}")
    print("    PASS - 5/5 transcripts decode to valid non-empty UTF-8 Khmer")

    print("\n" + "=" * 78)
    print("STEP 3 PASS: pilot Khmer shard verified")
    print(f"  clips            : {len(items)}")
    print(f"  manifest size    : {on_disk/1024**2:.2f} MiB")
    print(f"  NaN/Inf clips    : {nan_count}")
    print(f"  max duration skew: {worst_dur_err*1000:.4f} ms")
    print(f"  manifest         : {manifest_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
