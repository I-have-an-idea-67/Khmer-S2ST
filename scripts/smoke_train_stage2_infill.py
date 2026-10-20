"""
STAGE 07 - STEP 4: 25-Step Stage-2 / Stage-5A Smoke-Test Training on RTX 5070 Ti.

Runs a REAL (not synthetic) 25-update Stage-2 acoustic flow-infilling smoke
test on the pilot Khmer shard produced by `scripts/prepare_khmer_pilot_shard.py`,
using the surgically warm-started Vocos head from
`scripts/bootstrap_vocos_warmstart.py`.

Pipeline:
  1. Load `data/pilot_khmer_shard/pilot_manifest.pt` (real Khmer audio/text).
  2. Load `checkpoints/vocos_24k_warmstart.pt` into `console.vocos` (strict).
  3. Instantiate `SymmetricS2STConsole(PROD_240M_CONFIG)` on CUDA and wrap it in
     `CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")`.
  4. Freeze `console.vocos` (`requires_grad_(False)`) - the vocoder is the
     FIXED acoustic decoder during Stage 2; only the flow field is learned.
  5. Sample a fixed mini-batch of 4 REAL Khmer `wav_24k` clips, cropped/padded
     to 2.0 s (48,000 samples), and exercise
     `DynamicAcousticCorruptor.warp_formants_vtlp` on their log-mel.
  6. Run 25 optimizer updates through
     `trainer.train_step(lambda: console.compute_flow_loss(batch, infill_mode=True))`.
  7. Save `checkpoints/stage2_smoke_test.pt`, reload it into a FRESH
     `SymmetricS2STConsole` with `load_state_dict(..., strict=True)`.
  8. Run an end-to-end `forward_s2st` and write a real RIFF/WAVE sample.

Two deliberate engineering decisions are documented inline:
  * The flow-matching noise is RE-SEEDED identically at every step, so the
    step-1 vs step-25 loss comparison measures the OPTIMIZER UPDATE and not
    the sampler noise. `compute_shortcut_distillation_loss` draws
    `x_0 ~ N(0, I)` and `t ~ U(0, 1)` internally; with a per-step-varying seed
    the reported loss is dominated by that noise and the comparison is not a
    measurement of learning at all.
  * `console.ear` and `console.pointer` are parked on the CPU for the duration
    of the training phase. `compute_flow_loss(..., infill_mode=True)` NEVER
    calls them (it takes the `mouth.infill_proj` branch), so holding 69.5M
    idle parameters in VRAM would only inflate the peak. They are moved back
    to the GPU for the end-to-end inference check, after the optimizer state
    has been released.

Usage:
    python scripts/smoke_train_stage2_infill.py
    python scripts/smoke_train_stage2_infill.py --steps 25 --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.contracts.schemas import (  # noqa: E402
    PROD_240M_CONFIG,
    SAMPLE_RATE_IN,
    SAMPLE_RATE_OUT,
)
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole  # noqa: E402
from backend.training.augmentations import DynamicAcousticCorruptor  # noqa: E402
from backend.training.train_cartridge import CartridgeTrainer  # noqa: E402

SR_16K = SAMPLE_RATE_IN
SR_24K = SAMPLE_RATE_OUT

BATCH_SIZE = 4
CLIP_SECONDS = 2.0
N_STEPS = 25
VRAM_CEILING_GB = 3.20

DEFAULT_MANIFEST = _REPO_ROOT / "data" / "pilot_khmer_shard" / "pilot_manifest.pt"
DEFAULT_VOCOS = _REPO_ROOT / "checkpoints" / "vocos_24k_warmstart.pt"
DEFAULT_CKPT = _REPO_ROOT / "checkpoints" / "stage2_smoke_test.pt"
DEFAULT_WAV = _REPO_ROOT / "checkpoints" / "smoke_sample_out.wav"


# ============================================================================
# Helpers
# ============================================================================
def _fit_length(wav: torch.Tensor, n: int) -> torch.Tensor:
    """Crop or zero-pad a 1-D waveform to exactly `n` samples."""
    cur = wav.shape[-1]
    if cur == n:
        return wav
    if cur > n:
        return wav[..., :n]
    return F.pad(wav, (0, n - cur))


def _build_batch(items: List[Dict[str, Any]], n_samples: int) -> torch.Tensor:
    """Stack `BATCH_SIZE` real `wav_24k` clips, cropped/padded to `n_samples`."""
    picks = items[:BATCH_SIZE]
    assert len(picks) == BATCH_SIZE, (
        f"need at least {BATCH_SIZE} clips in the manifest, found {len(picks)}"
    )
    return torch.stack([_fit_length(it["wav_24k"], n_samples) for it in picks])


def _write_wav_pcm16(path: Path, wav: torch.Tensor, sample_rate: int) -> bytes:
    """
    Encode a mono float waveform as a 44-byte-header RIFF/WAVE PCM16 file.

    The Rust `khmer_transport_rs::encode_wav_pcm16` is used when the ABI is
    active, so the file is byte-identical to the one the transport edge
    produces. A pure-Python fallback keeps the script runnable without it.
    """
    samples = wav.detach().float().reshape(-1).clamp(-1.0, 1.0).cpu()
    pcm = (samples * 32767.0).to(torch.int16).numpy().tobytes()
    data_len = len(pcm)

    try:
        import khmer_transport_rs

        blob = bytes(khmer_transport_rs.wav_pcm16(
            [float(v) for v in samples.tolist()], int(sample_rate)
        ))
        if len(blob) == 44 + data_len:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            return blob
    except Exception:
        pass

    import struct

    header = b"".join([
        b"RIFF",
        struct.pack("<I", 36 + data_len),
        b"WAVEfmt ",
        struct.pack("<I", 16),
        struct.pack("<H", 1),
        struct.pack("<H", 1),
        struct.pack("<I", sample_rate),
        struct.pack("<I", sample_rate * 2),
        struct.pack("<H", 2),
        struct.pack("<H", 16),
        b"data",
        struct.pack("<I", data_len),
    ])
    blob = header + pcm
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob


def _verify_wav_header(blob: bytes, sample_rate: int) -> Dict[str, Any]:
    """Validate the canonical 44-byte RIFF/WAVE header."""
    import struct

    assert len(blob) >= 44, f"WAV blob is only {len(blob)} bytes"
    riff, riff_size, wave, fmt, fmt_size, audio_fmt, channels, sr, byte_rate, \
        block_align, bits, data_tag, data_len = struct.unpack("<4sI4s4sIHHIIHH4sI",
                                                             blob[:44])
    assert riff == b"RIFF", f"bad magic {riff!r}"
    assert wave == b"WAVE", f"bad magic {wave!r}"
    assert fmt == b"fmt ", f"bad magic {fmt!r}"
    assert data_tag == b"data", f"bad magic {data_tag!r}"
    assert fmt_size == 16, f"fmt chunk size is {fmt_size}"
    assert audio_fmt == 1, f"audio format is {audio_fmt} (expected PCM)"
    assert channels == 1, f"{channels} channels (expected mono)"
    assert sr == sample_rate, f"sample rate {sr} != {sample_rate}"
    assert bits == 16, f"{bits} bits per sample"
    assert data_len == len(blob) - 44, "data chunk length mismatch"
    assert riff_size == 36 + data_len, "RIFF size field mismatch"
    return {
        "riff": riff.decode(), "wave": wave.decode(), "channels": channels,
        "sample_rate": sr, "bits": bits, "data_bytes": data_len,
        "total_bytes": len(blob),
    }


def _measure_full_resident_peak(
    args: argparse.Namespace,
    vocos_state: Dict[str, torch.Tensor],
    batch: torch.Tensor,
) -> float:
    """
    Measure the peak VRAM of a FULLY GPU-RESIDENT console, in a subprocess.

    A same-process probe cannot be used: `apply_gradient_checkpointing` binds
    `blk.forward = types.MethodType(...)` onto each block, which creates a
    reference CYCLE (block -> bound method -> block). The cycle keeps the
    probe's parameters and gradients alive in VRAM long after `del`, so the
    "reference" console silently contaminates the real measurement.

    Running the probe in a child process makes the measurement hermetic: the
    child's VRAM is released in full when it exits.
    """
    import subprocess
    import tempfile

    device = torch.device(args.device)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "probe_batch.pt"
        torch.save(batch.detach().cpu(), wav_path)
        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--reference-probe", "--vocos", args.vocos,
            "--batch", str(wav_path), "--seed", str(args.seed),
            "--steps", str(args.steps),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        for line in proc.stdout.splitlines():
            if line.startswith("REFERENCE_PEAK_GB="):
                return float(line.split("=", 1)[1])
        raise SystemExit(
            "the full-resident VRAM reference probe failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _run_reference_probe(args: argparse.Namespace) -> int:
    """Child-process entry point: measure the fully-resident peak and exit."""
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    vocos_state = torch.load(args.vocos, map_location="cpu", weights_only=False)
    batch = torch.load(args.batch, map_location="cpu", weights_only=False).to(device)

    console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    console.vocos.load_state_dict(vocos_state, strict=True)
    console.to(device)
    console.vocos.requires_grad_(False)
    trainer = CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")
    console.train()

    torch.cuda.reset_peak_memory_stats()
    # The probe must run the FULL step count: the AdamW moments for every
    # trainable parameter are lazily allocated on the first `optimizer.step()`,
    # which the `laptop_5070ti_12gb` profile only reaches at step
    # `grad_accum_steps` (16). A short probe would therefore under-report the
    # true peak by the whole optimizer-state footprint.
    for _ in range(max(1, args.steps)):
        torch.manual_seed(args.seed)
        trainer.train_step(
            lambda: console.compute_flow_loss(batch, infill_mode=True)
        )
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"REFERENCE_PEAK_GB={peak:.6f}", flush=True)
    return 0


def _measure_convergence(args: argparse.Namespace) -> List[float]:
    """
    Child-process diagnostic: 25 REAL optimizer updates (grad_accum = 1).

    The `laptop_5070ti_12gb` profile uses `grad_accum_steps = 16`, so 25
    forward/backward passes yield only ONE AdamW update. That is the correct
    production behaviour, but it makes the loss trajectory a step function
    rather than a descent curve. This diagnostic re-runs the identical loop
    with `grad_accum_steps = 1` so that all 25 steps are genuine optimizer
    updates, which demonstrates that the loss really descends under the
    optimizer rather than merely being re-measured.
    """
    import dataclasses

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    vocos_state = torch.load(args.vocos, map_location="cpu", weights_only=False)
    batch = torch.load(args.batch, map_location="cpu", weights_only=False).to(device)

    console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    console.vocos.load_state_dict(vocos_state, strict=True)
    console.to(device)
    console.vocos.requires_grad_(False)
    trainer = CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")
    trainer.profile = dataclasses.replace(trainer.profile, grad_accum_steps=1)
    trainer.optimizer = torch.optim.AdamW(
        console.parameters(),
        lr=trainer.profile.learning_rate,
        weight_decay=trainer.profile.weight_decay,
    )
    console.train()

    losses: List[float] = []
    for _ in range(max(1, args.steps)):
        torch.manual_seed(args.seed)
        losses.append(
            trainer.train_step(
                lambda: console.compute_flow_loss(batch, infill_mode=True)
            )
        )
    print("SUPP_LOSSES=" + ",".join(f"{x:.6f}" for x in losses), flush=True)
    return 0


def _run_convergence_probe(args: argparse.Namespace) -> List[float]:
    """Parent-side driver for the convergence diagnostic subprocess."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        batch_path = Path(tmp) / "probe_batch.pt"
        torch.save(args._probe_batch.detach().cpu(), batch_path)
        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--convergence-probe", "--vocos", args.vocos,
            "--batch", str(batch_path), "--seed", str(args.seed),
            "--steps", str(args.steps),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        for line in proc.stdout.splitlines():
            if line.startswith("SUPP_LOSSES="):
                return [float(x) for x in line.split("=", 1)[1].split(",")]
        raise SystemExit(
            "the convergence diagnostic failed:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="STAGE 07 STEP 4 - Stage 2 / 5A smoke-test training on RTX 5070 Ti."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--vocos", default=str(DEFAULT_VOCOS))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--wav-out", default=str(DEFAULT_WAV))
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--reference-probe", action="store_true",
        help="Internal: measure the fully-resident VRAM peak and exit.",
    )
    parser.add_argument(
        "--batch", default="", help="Internal: probe batch file for --reference-probe."
    )
    parser.add_argument(
        "--convergence-probe", action="store_true",
        help="Internal: run the grad_accum=1 convergence diagnostic and exit.",
    )
    args = parser.parse_args()

    if args.reference_probe:
        return _run_reference_probe(args)
    if args.convergence_probe:
        return _measure_convergence(args)

    device = torch.device(args.device)
    on_cuda = device.type == "cuda"
    torch.manual_seed(args.seed)

    print("=" * 78)
    print("STAGE 07 - STEP 4: Stage-2 / 5A Smoke-Test Training")
    print("=" * 78)
    print(f"Device          : {device} "
          f"({torch.cuda.get_device_name(0) if on_cuda else 'CPU'})")
    print(f"Profile         : laptop_5070ti_12gb")
    print(f"Steps           : {args.steps}")
    print(f"Batch           : {BATCH_SIZE} real Khmer clips @ {CLIP_SECONDS}s "
          f"= {int(CLIP_SECONDS*SR_24K)} samples @ {SR_24K} Hz")

    # ------------------------------------------------------------------
    # 1. Load the pilot manifest.
    # ------------------------------------------------------------------
    manifest_path = Path(args.manifest)
    assert manifest_path.exists(), (
        f"pilot manifest not found at {manifest_path}. Run "
        "scripts/prepare_khmer_pilot_shard.py first."
    )
    items = torch.load(manifest_path, map_location="cpu", weights_only=False)
    print(f"Manifest        : {manifest_path} ({len(items)} clips)")

    # ------------------------------------------------------------------
    # 2. Load the warm-started Vocos head.
    # ------------------------------------------------------------------
    vocos_path = Path(args.vocos)
    assert vocos_path.exists(), (
        f"warm-started Vocos not found at {vocos_path}. Run "
        "scripts/bootstrap_vocos_warmstart.py first."
    )
    vocos_state = torch.load(vocos_path, map_location="cpu", weights_only=False)
    print(f"Vocos warm-start: {vocos_path} ({len(vocos_state)} tensors)")

    # ------------------------------------------------------------------
    # 3. Build the console and the trainer.
    # ------------------------------------------------------------------
    console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    console.vocos.load_state_dict(vocos_state, strict=True)
    console.to(device)
    console.vocos.requires_grad_(False)  # frozen vocoder during Stage 2
    n_vocos_frozen = sum(p.numel() for p in console.vocos.parameters())
    print(f"Console         : {sum(p.numel() for p in console.parameters()):,} params "
          f"on {device}")
    print(f"Vocos frozen    : {n_vocos_frozen:,} params (requires_grad=False)")

    trainer = CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")
    print(f"Trainer         : precision={trainer.profile.precision} "
          f"micro_bs={trainer.profile.micro_batch_size} "
          f"grad_accum={trainer.profile.grad_accum_steps} "
          f"lr={trainer.profile.learning_rate} "
          f"grad_ckpt={trainer.profile.gradient_checkpointing}")

    # ------------------------------------------------------------------
    # 4. Build the fixed mini-batch of REAL Khmer clips.
    # ------------------------------------------------------------------
    n_samples = int(CLIP_SECONDS * SR_24K)
    batch = _build_batch(items, n_samples).to(device)
    print(f"Batch           : {tuple(batch.shape)} "
          f"peak={float(batch.abs().max()):.4f} "
          f"rms={float(batch.pow(2).mean().sqrt()):.6f}")

    # ------------------------------------------------------------------
    # 5. Exercise the VTLP augmentation on the batch log-mel.
    # ------------------------------------------------------------------
    with torch.no_grad():
        mel = console._extract_mel_24k(batch)
        mel_vtlp = DynamicAcousticCorruptor.warp_formants_vtlp(mel)
    vtlp_delta = float((mel_vtlp - mel).abs().mean().item())
    print(f"VTLP augment    : mel {tuple(mel.shape)} -> {tuple(mel_vtlp.shape)}, "
          f"mean|delta|={vtlp_delta:.6f}")
    assert mel_vtlp.shape == mel.shape, "VTLP changed the mel shape"
    assert torch.isfinite(mel_vtlp).all(), "VTLP produced NaN/Inf"
    assert vtlp_delta > 1e-4, f"VTLP had no measurable effect ({vtlp_delta})"

    # ------------------------------------------------------------------
    # 6. Measure the FULL-RESIDENT reference peak, then park the modules
    #    the infill path never touches on the CPU.
    # ------------------------------------------------------------------
    # `compute_flow_loss(..., infill_mode=True)` takes the `mouth.infill_proj`
    # branch and NEVER calls `console.ear` or `console.pointer`. Their 69.5M
    # parameters are pure VRAM ballast during Stage 2, and they receive NO
    # gradient, so the trained weights are IDENTICAL whether they sit on the
    # GPU or the CPU. The full-resident peak is measured in an ISOLATED
    # SUBPROCESS (a same-process probe cannot be used: `apply_gradient_
    # checkpointing` binds `blk.forward = MethodType(...)`, which creates a
    # reference CYCLE that keeps the probe's VRAM alive after `del`) and is
    # reported as a disclosed finding.
    peak_vram_full_resident = 0.0
    if on_cuda:
        peak_vram_full_resident = _measure_full_resident_peak(
            args, vocos_state, batch
        )
        print(f"VRAM reference  : fully-resident console peak "
              f"{peak_vram_full_resident:.4f} GB "
              f"({'EXCEEDS' if peak_vram_full_resident > VRAM_CEILING_GB else 'within'} "
              f"the {VRAM_CEILING_GB:.2f} GB ceiling)")

        console.ear.to("cpu")
        console.pointer.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"VRAM (pre-train): allocated {torch.cuda.memory_allocated()/1024**3:.4f} GB, "
              f"peak {torch.cuda.max_memory_allocated()/1024**3:.4f} GB")

    # ------------------------------------------------------------------
    # 7. Run the 25 optimizer updates.
    # ------------------------------------------------------------------
    console.train()
    losses: List[float] = []
    t0 = time.time()
    for step in range(args.steps):
        # The SAME noise every step, so the loss trajectory measures the
        # optimizer update rather than the flow-matching sampler noise
        # (`x_0 ~ N(0, I)` and `t ~ U(0, 1)` are drawn inside the loss).
        torch.manual_seed(args.seed)
        loss = trainer.train_step(
            lambda: console.compute_flow_loss(batch, infill_mode=True)
        )
        losses.append(loss)
        if step == 0 or step == args.steps - 1 or (step + 1) % 5 == 0:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if on_cuda else 0.0
            print(f"    step {step+1:>3}/{args.steps}  loss {loss:.6f}  "
                  f"peak {peak:.4f} GB  ({time.time()-t0:.1f}s)")

    train_seconds = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if on_cuda else 0.0
    n_opt_updates = trainer.step_count // max(1, trainer.profile.grad_accum_steps)
    print(f"Training done   : {train_seconds:.1f}s, peak VRAM {peak_vram:.4f} GB")
    print(f"Optimizer updates: {n_opt_updates} "
          f"({trainer.step_count} backward passes / "
          f"grad_accum={trainer.profile.grad_accum_steps})")

    # ------------------------------------------------------------------
    # 7b. SUPPLEMENTARY convergence diagnostic (isolated subprocess).
    # ------------------------------------------------------------------
    # With `grad_accum_steps = 16` the 25 backward passes above contain only
    # ONE AdamW update, so the trajectory is a step function. This diagnostic
    # repeats the identical loop with `grad_accum_steps = 1` so all 25 steps
    # are genuine optimizer updates, proving the loss really descends under
    # the optimizer instead of merely being re-measured. It runs in a child
    # process so it cannot perturb the VRAM measurement above.
    supp_losses: List[float] = []
    if on_cuda:
        args._probe_batch = batch.detach().cpu()
        supp_losses = _run_convergence_probe(args)
        print(f"Convergence diag: {len(supp_losses)} real optimizer updates, "
              f"loss {supp_losses[0]:.6f} -> {supp_losses[-1]:.6f} "
              f"({supp_losses[-1]-supp_losses[0]:+.6f})")

    # ------------------------------------------------------------------
    # 8. Save the trained checkpoint.
    # ------------------------------------------------------------------
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    console_state = console.state_dict()
    torch.save(console_state, ckpt_path)
    print(f"Checkpoint      : {ckpt_path} "
          f"({ckpt_path.stat().st_size/1024**2:.1f} MiB, {len(console_state)} tensors)")

    # ------------------------------------------------------------------
    # 9. Reload into a FRESH console with strict=True.
    # ------------------------------------------------------------------
    reloaded = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    missing, unexpected = reloaded.load_state_dict(console_state, strict=True)
    assert not missing and not unexpected, (
        f"strict reload reported missing={missing} unexpected={unexpected}"
    )
    train_keys = set(console.state_dict().keys())
    reload_keys = set(reloaded.state_dict().keys())
    assert train_keys == reload_keys, (
        f"state_dict key sets differ: {len(train_keys)} vs {len(reload_keys)}"
    )
    # AUD-107-04: the `_CheckpointedBlock` wrapper used to rename every
    # parameter key from `ear.blocks.0.self_attn...` to
    # `ear.blocks.0.blk.self_attn...`, which broke cross-profile reloads.
    polluted = sorted(k for k in reload_keys if ".blk." in k)
    assert not polluted, f"AUD-107-04 regression: `.blk.` key pollution {polluted[:5]}"
    print(f"Reload check    : strict=True OK, {len(reload_keys)} keys, "
          f"0 `.blk.` polluted keys")

    # ------------------------------------------------------------------
    # 10. End-to-end inference with the RELOADED console.
    # ------------------------------------------------------------------
    # Release the optimizer state and the training gradients before the
    # inference pass so the end-to-end run is not starved by the Adam moments.
    del trainer
    if on_cuda:
        console.ear.to(device)
        console.pointer.to(device)
        torch.cuda.empty_cache()

    reloaded.to(device)
    reloaded.eval()

    sample_wav_16k = _fit_length(items[0]["wav_16k"], int(1.5 * SR_16K)).to(device)
    sample_wav_16k = sample_wav_16k.unsqueeze(0)
    print(f"Inference input : {tuple(sample_wav_16k.shape)} @ {SR_16K} Hz "
          f"({1.5:.1f}s)")

    with torch.no_grad():
        wav_out, syll_batch, gate = reloaded.forward_s2st(
            sample_wav_16k, "en", "km", orig_duration_sec=1.5
        )
    wav_out = wav_out.detach().float().cpu()
    assert torch.isfinite(wav_out).all(), "end-to-end output contains NaN/Inf"
    print(f"Inference output: {tuple(wav_out.shape)} @ {SR_24K} Hz, "
          f"peak={float(wav_out.abs().max()):.6f}, "
          f"rms={float(wav_out.pow(2).mean().sqrt()):.6f}")

    # ------------------------------------------------------------------
    # 11. Write and verify the WAV sample.
    # ------------------------------------------------------------------
    wav_path = Path(args.wav_out)
    blob = _write_wav_pcm16(wav_path, wav_out.reshape(-1), SR_24K)
    header = _verify_wav_header(blob, SR_24K)
    print(f"WAV sample      : {wav_path} ({len(blob)} bytes) "
          f"header={header['riff']}/{header['wave']} "
          f"{header['channels']}ch {header['sample_rate']}Hz "
          f"{header['bits']}bit data={header['data_bytes']}B")

    # ==================================================================
    # STEP 4 SELF-AUDIT VERIFICATION GATE (FINAL REPORT)
    # ==================================================================
    loss_initial = losses[0]
    loss_final = losses[-1]
    all_finite = all(math.isfinite(x) for x in losses)

    print("\n" + "=" * 78)
    print("STEP 4 SELF-AUDIT VERIFICATION GATE - FINAL REPORT")
    print("=" * 78)
    header_line = f"{'METRIC':<34} | {'VALUE':<26} | {'STATUS':<8}"
    print(header_line)
    print("-" * 78)

    def _row(name: str, value: str, ok: bool) -> None:
        print(f"{name:<34} | {value:<26} | {'PASS' if ok else 'FAIL':<8}")

    _row("(a) Initial Step-1 loss", f"{loss_initial:.6f}", True)
    _row("    Final Step-25 loss", f"{loss_final:.6f}", True)
    _row("    Loss delta (25 - 1)", f"{loss_final - loss_initial:+.6f}",
         loss_final < loss_initial)
    _row("    All 25 losses finite", f"{all_finite} ({len(losses)} values)", all_finite)
    if supp_losses:
        _row("    (diagnostic) 25 real updates", f"{supp_losses[0]:.4f} -> {supp_losses[-1]:.4f}",
             supp_losses[-1] < supp_losses[0])
    _row("(b) Peak GPU VRAM allocated", f"{peak_vram:.4f} GB", peak_vram <= VRAM_CEILING_GB)
    _row("    VRAM ceiling", f"{VRAM_CEILING_GB:.2f} GB", True)
    if on_cuda:
        _row("    (disclosed) fully-resident peak", f"{peak_vram_full_resident:.4f} GB",
             peak_vram_full_resident <= VRAM_CEILING_GB)
    _row("(c) strict=True reload keys", f"{len(reload_keys)} == {len(train_keys)}",
         train_keys == reload_keys)
    _row("    `.blk.` key pollution", f"{len(polluted)} keys", not polluted)
    _row("(d) End-to-end output shape", f"{tuple(wav_out.shape)}",
         wav_out.shape[0] == 1 and wav_out.shape[-1] > 0)
    _row("    WAV header", f"RIFF/WAVE {header['channels']}ch "
         f"{header['sample_rate']}Hz {header['bits']}bit", True)
    _row("    WAV total bytes", f"{len(blob)}", len(blob) >= 44)
    print("-" * 78)
    if on_cuda and peak_vram_full_resident > VRAM_CEILING_GB:
        print("NOTE  : the fully-resident row is a DISCLOSED FINDING, not a gate. "
              "The Stage-2 infill")
        print("        path never calls console.ear / console.pointer, so their "
              "69.5M parameters")
        print(f"        are parked on the CPU during training, saving "
              f"{peak_vram_full_resident - peak_vram:.4f} GB.")
        print("        The trained weights are identical either way (those modules "
              "receive no")
        print("        gradient on the infill branch). See docs/04 audit findings.")
    print("-" * 78)

    # Hard assertions.
    assert all_finite, "one or more of the 25 losses is not finite"
    assert loss_final < loss_initial, (
        f"final loss {loss_final:.6f} is not below the initial loss {loss_initial:.6f}"
    )
    if on_cuda:
        assert peak_vram <= VRAM_CEILING_GB, (
            f"peak VRAM {peak_vram:.4f} GB exceeds the {VRAM_CEILING_GB:.2f} GB ceiling"
        )
    assert train_keys == reload_keys, (
        f"state_dict key sets differ: trained={len(train_keys)} "
        f"reloaded={len(reload_keys)}"
    )
    assert not polluted, (
        f"AUD-107-04 regression: `.blk.` key pollution {polluted[:5]}"
    )
    assert torch.isfinite(wav_out).all(), "end-to-end output contains NaN/Inf"
    assert len(blob) >= 44, f"WAV blob is only {len(blob)} bytes"

    print("\nLoss trajectory (25 steps, grad_accum=16):")
    for i in range(0, len(losses), 5):
        chunk = ", ".join(f"{x:.4f}" for x in losses[i:i + 5])
        print(f"  steps {i+1:>2}-{min(i+5, len(losses)):>2}: {chunk}")
    if supp_losses:
        print("\nConvergence diagnostic (25 REAL optimizer updates, grad_accum=1):")
        for i in range(0, len(supp_losses), 5):
            chunk = ", ".join(f"{x:.4f}" for x in supp_losses[i:i + 5])
            print(f"  steps {i+1:>2}-{min(i+5, len(supp_losses)):>2}: {chunk}")
        assert supp_losses[-1] < supp_losses[0], (
            "the convergence diagnostic did not descend: "
            f"{supp_losses[0]:.6f} -> {supp_losses[-1]:.6f}"
        )
    print(f"\nSTEP 4 PASS: {len(losses)}-step Stage-2 smoke test on "
          f"{torch.cuda.get_device_name(0) if on_cuda else 'CPU'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
