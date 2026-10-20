"""
STAGE 08 - Overnight Ingestion + Stage 2 Infill Pre-Training Orchestrator.

Chains:
  Phase A (Dataset Sharding):
    Runs `overnight_khmer_ingester.py` with `--shard-size 500 --max-clips 10000 --min-free-gb 15.0`.
    Logs progress after every shard to `data/khmer_overnight_shards/ingestion.log`.

  Phase B (Stage 2 Infill Pre-Training on RTX 5070 Ti):
    Streams batches of 4 random 2.0s `wav_24k` segments (`wav_24k_int16.float() / 32768.0`)
    from all available `shard_*.pt` files.
    Applies `DynamicAcousticCorruptor.warp_formants_vtlp`.
    Parks `console.ear` and `console.pointer` on CPU to keep peak VRAM <= 3.20 GB.
    Optimizes `compute_flow_loss(..., infill_mode=True)` with `CartridgeTrainer("laptop_5070ti_12gb")`.
    Saves rolling `checkpoints/stage2_latest.pt` (and `checkpoints/stage2_best.pt`) every 250 steps.
    Logs step, loss, and peak VRAM to `checkpoints/overnight_train.log`.

Usage:
  # 10-step smoke test:
  python scripts/overnight_runner.py --smoke-test --smoke-steps 10

  # Full overnight unattended run:
  python scripts/overnight_runner.py --phase all --max-clips 10000 --train-steps 10000
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

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

from backend.contracts.schemas import PROD_240M_CONFIG, SAMPLE_RATE_OUT  # noqa: E402
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole  # noqa: E402
from backend.training.augmentations import DynamicAcousticCorruptor  # noqa: E402
from backend.training.train_cartridge import CartridgeTrainer  # noqa: E402
from scripts.overnight_khmer_ingester import run_ingestion  # noqa: E402

SR_24K = SAMPLE_RATE_OUT
BATCH_SIZE = 4
CLIP_SECONDS = 2.0
N_SAMPLES_24K = int(CLIP_SECONDS * SR_24K)  # 48,000 samples
VRAM_CEILING_GB = 3.20


# ============================================================================
# Shard-Based Audio Streamer
# ============================================================================
class ShardAudioStreamer:
    """Streams randomized 2.0s 24kHz float32 audio segments from PCM16 shards."""

    def __init__(self, shards_dir: Path, target_samples: int = N_SAMPLES_24K):
        self.shards_dir = shards_dir
        self.target_samples = target_samples
        self.shard_files = sorted(shards_dir.glob("shard_*.pt"))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard_*.pt files found in {shards_dir}")

        self.current_shard_idx = 0
        self.current_shard_data: List[Dict[str, Any]] = []
        self._load_shard(0)

    def _load_shard(self, idx: int) -> None:
        self.current_shard_idx = idx % len(self.shard_files)
        path = self.shard_files[self.current_shard_idx]
        self.current_shard_data = torch.load(
            path, map_location="cpu", weights_only=False
        )

    def sample_batch(self, batch_size: int = BATCH_SIZE) -> torch.Tensor:
        """Sample a batch of `batch_size` 24kHz float32 waveforms (shape: B, T)."""
        clips: List[torch.Tensor] = []
        for _ in range(batch_size):
            if not self.current_shard_data:
                self._load_shard(self.current_shard_idx + 1)

            # Pick random clip from current shard
            clip = random.choice(self.current_shard_data)
            raw_pcm16 = clip["wav_24k"]
            wav_f32 = raw_pcm16.float() / 32768.0

            # Crop or pad to target_samples
            n = wav_f32.shape[-1]
            if n >= self.target_samples:
                start = random.randint(0, n - self.target_samples)
                segment = wav_f32[start : start + self.target_samples]
            else:
                segment = F.pad(wav_f32, (0, self.target_samples - n))

            clips.append(segment)

        # Periodically rotate shard
        if random.random() < 0.1 and len(self.shard_files) > 1:
            self._load_shard((self.current_shard_idx + 1) % len(self.shard_files))

        return torch.stack(clips)


# ============================================================================
# Phase A: Sharding Runner
# ============================================================================
def execute_phase_a(
    shards_dir: Path,
    shard_size: int,
    max_clips: int,
    min_free_gb: float,
    log_file: Path,
) -> int:
    """Run overnight sharder with progress output redirected to ingestion.log."""
    print("\n" + "=" * 78)
    print("PHASE A: OVERNIGHT KHMER DATASET SHARDING")
    print("=" * 78)
    shards_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    class TeeLogger:
        def __init__(self, orig_stream, file_path: Path):
            self.orig_stream = orig_stream
            self.f = open(file_path, "a", encoding="utf-8")

        def write(self, s):
            self.orig_stream.write(s)
            self.orig_stream.flush()
            self.f.write(s)
            self.f.flush()

        def flush(self):
            self.orig_stream.flush()
            self.f.flush()

        def close(self):
            self.f.close()

    tee = TeeLogger(sys.stdout, log_file)
    orig_stdout = sys.stdout
    sys.stdout = tee
    try:
        ret = run_ingestion(
            out_dir_path=shards_dir,
            shard_size=shard_size,
            max_clips=max_clips,
            min_free_gb=min_free_gb,
        )
        return ret
    finally:
        sys.stdout = orig_stdout
        tee.close()


# ============================================================================
# Phase B: Stage 2 Infill Pre-Training Runner
# ============================================================================
def execute_phase_b(
    shards_dir: Path,
    vocos_path: Path,
    latest_ckpt_path: Path,
    best_ckpt_path: Path,
    log_file: Path,
    total_steps: int = 10000,
    save_every: int = 250,
    device_name: str = "cuda",
    seed: int = 4242,
) -> int:
    """Execute Stage 2 acoustic flow-infill pre-training on RTX 5070 Ti."""
    print("\n" + "=" * 78)
    print("PHASE B: STAGE 2 ACOUSTIC FLOW-INFILL PRE-TRAINING")
    print("=" * 78)

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    torch.manual_seed(seed)
    random.seed(seed)

    latest_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Streamer from available shards
    streamer = ShardAudioStreamer(shards_dir, target_samples=N_SAMPLES_24K)
    print(f"Shards located    : {len(streamer.shard_files)} shards in {shards_dir}")

    # 2. Instantiate 241.637M Console
    console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    start_step = 0
    best_loss = float("inf")

    if latest_ckpt_path.exists():
        print(f"Resuming from     : {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        console.load_state_dict(ckpt["console_state_dict"], strict=True)
        start_step = ckpt.get("step", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed at step   : {start_step} (previous best loss: {best_loss:.6f})")
    else:
        assert vocos_path.exists(), (
            f"Vocos warmstart checkpoint not found at {vocos_path}. "
            f"Run scripts/bootstrap_vocos_warmstart.py first."
        )
        vocos_state = torch.load(vocos_path, map_location="cpu", weights_only=False)
        console.vocos.load_state_dict(vocos_state, strict=True)
        print(f"Warmstart loaded  : {vocos_path}")

    # 3. Freeze vocoder & park unused modules on CPU to protect VRAM ceiling
    console.vocos.requires_grad_(False)
    console.to(device)
    if on_cuda:
        console.ear.to("cpu")
        console.pointer.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")
    console.train()

    print(f"Device            : {device} ({torch.cuda.get_device_name(0) if on_cuda else 'CPU'})")
    print(f"Optimizer profile : laptop_5070ti_12gb (VRAM ceiling: {VRAM_CEILING_GB:.2f} GB)")
    print(f"Target steps      : {start_step} -> {total_steps} (save every {save_every})")

    t_start = time.time()
    recent_losses: List[float] = []

    with open(log_file, "a", encoding="utf-8") as f_log:
        header_msg = (
            f"--- Stage 2 Training Started: {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"| steps {start_step}..{total_steps} | shards={len(streamer.shard_files)} ---\n"
        )
        f_log.write(header_msg)
        f_log.flush()

        for step in range(start_step, total_steps):
            # Sample 4 real Khmer 24kHz clips
            batch = streamer.sample_batch(BATCH_SIZE).to(device)

            # Apply VTLP formant augmentation 50% of the time
            if random.random() < 0.5:
                with torch.no_grad():
                    mel = console._extract_mel_24k(batch)
                    mel_vtlp = DynamicAcousticCorruptor.warp_formants_vtlp(mel)
                    # Resynthesize warped waveform through frozen Vocos
                    batch_warped = console.vocos(mel_vtlp)[:, :batch.shape[-1]]
                    batch = batch_warped

            # Optimizer update step
            loss = trainer.train_step(
                lambda: console.compute_flow_loss(batch, infill_mode=True)
            )
            recent_losses.append(loss)

            peak_vram = (
                torch.cuda.max_memory_allocated() / (1024**3) if on_cuda else 0.0
            )
            assert peak_vram <= VRAM_CEILING_GB + 0.10, (
                f"Peak VRAM {peak_vram:.4f} GB violated ceiling {VRAM_CEILING_GB:.2f} GB"
            )

            # Logging
            if (
                step == start_step
                or step == total_steps - 1
                or (step + 1) % 10 == 0
                or (step + 1) % save_every == 0
            ):
                elapsed = time.time() - t_start
                avg_loss = sum(recent_losses[-10:]) / len(recent_losses[-10:])
                log_line = (
                    f"step {step+1:>5}/{total_steps} | loss: {loss:.6f} "
                    f"| avg10: {avg_loss:.6f} | peak_vram: {peak_vram:.4f} GB "
                    f"| elapsed: {elapsed:.1f}s"
                )
                print(f"  {log_line}", flush=True)
                f_log.write(log_line + "\n")
                f_log.flush()

            # Periodic rolling checkpoint saving
            if (step + 1) % save_every == 0 or (step + 1) == total_steps:
                ckpt_data = {
                    "step": step,
                    "loss": loss,
                    "best_loss": min(best_loss, loss),
                    "console_state_dict": console.state_dict(),
                    "profile_name": "laptop_5070ti_12gb",
                    "timestamp": time.time(),
                }
                # Atomic save latest
                tmp_latest = latest_ckpt_path.with_suffix(".pt.tmp")
                torch.save(ckpt_data, tmp_latest)
                os.replace(tmp_latest, latest_ckpt_path)

                # Save best if improved
                if loss < best_loss:
                    best_loss = loss
                    tmp_best = best_ckpt_path.with_suffix(".pt.tmp")
                    torch.save(ckpt_data, tmp_best)
                    os.replace(tmp_best, best_ckpt_path)
                    print(
                        f"    [CHECKPOINT] New best loss {best_loss:.6f} saved to {best_ckpt_path.name}"
                    )

                print(
                    f"    [CHECKPOINT] Rolling checkpoint updated at step {step+1} -> {latest_ckpt_path.name}"
                )

    print("\n" + "=" * 78)
    print("PHASE B COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"Final step        : {total_steps}")
    print(f"Best loss         : {best_loss:.6f}")
    print(f"Latest checkpoint : {latest_ckpt_path}")
    print(f"Training log      : {log_file}")
    return 0


# ============================================================================
# Main CLI Orchestrator
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="STAGE 08 - Overnight Ingestion & Stage 2 Pre-Training Orchestrator."
    )
    parser.add_argument(
        "--phase",
        choices=["all", "sharding", "training"],
        default="all",
        help="Pipeline phase to execute (default: all).",
    )
    parser.add_argument(
        "--shards-dir",
        default=str(_REPO_ROOT / "data" / "khmer_overnight_shards"),
        help="Directory containing or receiving shard_*.pt files.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Clips per shard (default: 500).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=10000,
        help="Total clips to collect in Phase A (default: 10000).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=15.0,
        help="Minimum free disk space threshold (default: 15.0 GB).",
    )
    parser.add_argument(
        "--vocos",
        default=str(_REPO_ROOT / "checkpoints" / "vocos_24k_warmstart.pt"),
        help="Warmstart Vocos checkpoint path.",
    )
    parser.add_argument(
        "--latest-ckpt",
        default=str(_REPO_ROOT / "checkpoints" / "stage2_latest.pt"),
        help="Rolling latest training checkpoint path.",
    )
    parser.add_argument(
        "--best-ckpt",
        default=str(_REPO_ROOT / "checkpoints" / "stage2_best.pt"),
        help="Best training checkpoint path.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=10000,
        help="Total training steps for Phase B (default: 10000).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=250,
        help="Steps between rolling checkpoints (default: 250).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute 10-step smoke test on existing shards.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=10,
        help="Number of steps for smoke test (default: 10).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda).",
    )
    parser.add_argument("--seed", type=int, default=4242, help="Random seed.")
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    vocos_path = Path(args.vocos)
    latest_ckpt = Path(args.latest_ckpt)
    best_ckpt = Path(args.best_ckpt)
    ingestion_log = shards_dir / "ingestion.log"
    train_log = _REPO_ROOT / "checkpoints" / "overnight_train.log"

    # Quick smoke test mode
    if args.smoke_test:
        print("[SMOKE TEST] Running smoke test of Phase B on existing shards...")
        return execute_phase_b(
            shards_dir=shards_dir,
            vocos_path=vocos_path,
            latest_ckpt_path=latest_ckpt,
            best_ckpt_path=best_ckpt,
            log_file=train_log,
            total_steps=args.smoke_steps,
            save_every=5,  # save at step 5 and step 10
            device_name=args.device,
            seed=args.seed,
        )

    # Full orchestration
    if args.phase in ("all", "sharding"):
        execute_phase_a(
            shards_dir=shards_dir,
            shard_size=args.shard_size,
            max_clips=args.max_clips,
            min_free_gb=args.min_free_gb,
            log_file=ingestion_log,
        )

    if args.phase in ("all", "training"):
        execute_phase_b(
            shards_dir=shards_dir,
            vocos_path=vocos_path,
            latest_ckpt_path=latest_ckpt,
            best_ckpt_path=best_ckpt,
            log_file=train_log,
            total_steps=args.train_steps,
            save_every=args.save_every,
            device_name=args.device,
            seed=args.seed,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
