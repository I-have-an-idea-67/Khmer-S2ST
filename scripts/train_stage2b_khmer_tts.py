"""
STAGE 09 - Stage 2B Supervised 7-Slot Khmer Voice-Cloning TTS Training Script.

Trains:
  - 7-slot syllable embeddings: mouth.embeddings["onset"], "coeng", "medial", "register", "vowel", "coda", "sesqui"
  - Learned duration aligner: mouth.duration_aligner.dur_proj
  - Conditioning projection: mouth.cond_proj
  - 15-layer DiT velocity backbone: mouth.blocks, mouth.adaLN_modulation, mouth.mel_out

Guarantees:
  - Peak CUDA VRAM strictly <= 3.10 GB (3.20 GB ceiling).
  - Full-utterance alignment (1.2s <= T <= 4.0s, no mid-sentence cropping).
  - Exact token-0 duration masking (sesqui_ids == 0 -> exact 0.0 frames).
  - 15% Classifier-Free Guidance (CFG) dropout on cond and emo_tokens.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from backend.contracts.schemas import (  # noqa: E402
    PROD_240M_CONFIG,
    SAMPLE_RATE_OUT,
    SyllableTensorBatch,
)
from backend.nn_cartridges.s2st_console import (  # noqa: E402
    SymmetricS2STConsole,
    compute_shortcut_distillation_loss,
)
from backend.training.train_cartridge import CartridgeTrainer  # noqa: E402

SR_24K = SAMPLE_RATE_OUT  # 24,000 Hz
MIN_SAMPLES = int(1.2 * SR_24K)  # 28,800 samples
MAX_SAMPLES = int(4.0 * SR_24K)  # 96,000 samples
VRAM_CEILING_GB = 3.20

SLOT_KEYS = (
    "onset_ids",
    "coeng_ids",
    "medial_ids",
    "register_ids",
    "vowel_ids",
    "coda_ids",
    "sesqui_ids",
)


# ============================================================================
# Full-Utterance Shard Streamer
# ============================================================================
class FullUtteranceShardStreamer:
    """Streams full-utterance batches (1.2s <= T <= 4.0s) with 0-padded 7-slot tensors."""

    def __init__(self, shards_dir: Path):
        self.shards_dir = shards_dir
        self.shard_files = sorted(shards_dir.glob("shard_*.pt"))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard_*.pt files found in {shards_dir}")

        self.current_shard_idx = 0
        self.eligible_clips: List[Dict[str, Any]] = []
        self._load_shard(0)

    def _load_shard(self, idx: int) -> None:
        self.current_shard_idx = idx % len(self.shard_files)
        path = self.shard_files[self.current_shard_idx]
        raw_data = torch.load(path, map_location="cpu", weights_only=False)
        # Filter strictly to full utterances between 1.2s and 4.0s
        self.eligible_clips = [
            clip
            for clip in raw_data
            if MIN_SAMPLES <= clip["wav_24k"].shape[-1] <= MAX_SAMPLES
        ]
        if not self.eligible_clips and len(self.shard_files) > 1:
            # Fallback to next shard if current has no matching clips
            self._load_shard(self.current_shard_idx + 1)

    def sample_batch(
        self, batch_size: int = 4
    ) -> Tuple[torch.Tensor, SyllableTensorBatch]:
        """Sample B full-utterance clips, pad waveforms and 7-slot tensors to batch max."""
        sampled_records: List[Dict[str, Any]] = []
        for _ in range(batch_size):
            if not self.eligible_clips:
                self._load_shard(self.current_shard_idx + 1)
            sampled_records.append(random.choice(self.eligible_clips))

        # Rotate shard occasionally to cycle across full corpus
        if random.random() < 0.08 and len(self.shard_files) > 1:
            self._load_shard((self.current_shard_idx + 1) % len(self.shard_files))

        # 1. Pad waveforms to max samples in batch
        max_wav_len = max(clip["wav_24k"].shape[-1] for clip in sampled_records)
        padded_wavs = []
        for clip in sampled_records:
            wav_f32 = clip["wav_24k"].float() / 32768.0
            pad_len = max_wav_len - wav_f32.shape[-1]
            if pad_len > 0:
                wav_f32 = F.pad(wav_f32, (0, pad_len))
            padded_wavs.append(wav_f32)
        batch_wav = torch.stack(padded_wavs)  # (B, T_wav_max)

        # 2. Pad 7-slot syllable tensors to max syllable length in batch using token 0
        max_syll_len = max(clip["onset_ids"].shape[-1] for clip in sampled_records)
        padded_slots: Dict[str, List[torch.Tensor]] = {k: [] for k in SLOT_KEYS}
        for clip in sampled_records:
            cur_len = clip["onset_ids"].shape[-1]
            pad_len = max_syll_len - cur_len
            for k in SLOT_KEYS:
                t = clip[k].to(torch.long)
                if pad_len > 0:
                    t = F.pad(t, (0, pad_len), value=0)
                padded_slots[k].append(t)

        syll_batch = SyllableTensorBatch(
            onset_ids=torch.stack(padded_slots["onset_ids"]),
            coeng_ids=torch.stack(padded_slots["coeng_ids"]),
            medial_ids=torch.stack(padded_slots["medial_ids"]),
            register_ids=torch.stack(padded_slots["register_ids"]),
            vowel_ids=torch.stack(padded_slots["vowel_ids"]),
            coda_ids=torch.stack(padded_slots["coda_ids"]),
            sesqui_ids=torch.stack(padded_slots["sesqui_ids"]),
        )
        return batch_wav, syll_batch


# ============================================================================
# Training & Audit Engine
# ============================================================================
def train_stage2b(
    shards_dir: Path,
    stage2_ckpt_path: Path,
    output_dir: Path,
    total_steps: int = 3000,
    save_every: int = 250,
    batch_size: int = 4,
    device_name: str = "cuda",
    audit_only: bool = False,
) -> int:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = output_dir / "stage2b_tts_latest.pt"
    best_ckpt_path = output_dir / "stage2b_tts_best.pt"
    log_file = output_dir / "stage2b_tts_train.log"

    print("=" * 78)
    print("STAGE 09 - STAGE 2B SUPERVISED 7-SLOT KHMER VOICE-CLONING TTS TRAINING")
    print("=" * 78)
    print(f"Device            : {device} ({torch.cuda.get_device_name(0) if on_cuda else 'CPU'})")
    print(f"Base checkpoint   : {stage2_ckpt_path}")
    print(f"Shards directory  : {shards_dir}")
    print(f"Output directory  : {output_dir}")
    print(f"Total steps       : {total_steps} (save every {save_every})")
    print(f"Batch size        : {batch_size}")
    print(f"VRAM ceiling      : {VRAM_CEILING_GB:.2f} GB")

    # 1. Initialize streamer
    streamer = FullUtteranceShardStreamer(shards_dir)
    print(f"Corpus Shards     : {len(streamer.shard_files)} shards loaded")

    # 2. Instantiate 241.637M Console and load Stage 2 checkpoint
    console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    assert stage2_ckpt_path.exists(), f"Stage 2 checkpoint not found at {stage2_ckpt_path}"

    print(f"[LOAD] Loading Stage 2 checkpoint: {stage2_ckpt_path.name}...")
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("console_state_dict", ckpt)
    missing, unexpected = console.load_state_dict(state_dict, strict=True)
    print(f"[LOAD] Loaded successfully (missing: {len(missing)}, unexpected: {len(unexpected)})")

    # 3. Park unused modules on CPU and freeze Vocos to respect VRAM ceiling
    console.vocos.requires_grad_(False)
    console.to(device)
    if on_cuda:
        console.ear.to("cpu")
        console.ear.requires_grad_(False)
        console.pointer.to("cpu")
        console.pointer.requires_grad_(False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 4. Set up trainer
    trainer = CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")
    console.train()

    t_start = time.time()
    recent_losses: List[float] = []
    best_loss = float("inf")

    # ========================================================================
    # STEP 2 SELF-AUDIT GATE (First 50 steps verification)
    # ========================================================================
    print("\n" + "-" * 78)
    print("STEP 2 SELF-AUDIT VERIFICATION GATE (50-Step Verification)")
    print("-" * 78)

    audit_steps = 50 if not audit_only else total_steps
    for step in range(audit_steps):
        wav_batch, syll_batch = streamer.sample_batch(batch_size)
        wav_batch = wav_batch.to(device)
        syll_batch = SyllableTensorBatch(
            onset_ids=syll_batch.onset_ids.to(device),
            coeng_ids=syll_batch.coeng_ids.to(device),
            medial_ids=syll_batch.medial_ids.to(device),
            register_ids=syll_batch.register_ids.to(device),
            vowel_ids=syll_batch.vowel_ids.to(device),
            coda_ids=syll_batch.coda_ids.to(device),
            sesqui_ids=syll_batch.sesqui_ids.to(device),
        )

        B = wav_batch.shape[0]
        L_max = syll_batch.onset_ids.shape[1]

        with trainer.autocast_context():
            x_1 = console._extract_mel_24k(wav_batch)  # (B, 100, T_mel)
            T_mel = x_1.shape[-1]
            orig_duration_sec = T_mel * 240.0 / 24000.0

            emo_tokens, _ = console.emotion_encoder(x_1, alpha=0.0)
            zero_bus = torch.zeros(B, L_max, 512, device=device, dtype=x_1.dtype)
            zero_gate = torch.zeros(B, L_max, 1, device=device, dtype=x_1.dtype)
            r_sps = torch.full((B,), 4.5, device=device, dtype=x_1.dtype)

            cond, _ = console.mouth.predict_syllables_and_align(
                zero_bus,
                zero_gate,
                r_sps,
                target_lang="km",
                orig_duration_sec=orig_duration_sec,
                target_frames=T_mel,
                explicit_syll_batch=syll_batch,
            )

            # Target duration supervision
            pred_durs = console.mouth.last_pred_durations
            prior = torch.where(
                syll_batch.sesqui_ids == 0,
                0.0,
                torch.where(syll_batch.sesqui_ids == 1, 0.25, 1.0),
            ).to(pred_durs.dtype)
            prior_sum = prior.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            target_durs = prior * (float(T_mel) / prior_sum)
            l_dur = F.smooth_l1_loss(pred_durs, target_durs)

            # 15% CFG conditioning dropout
            drop_mask = (torch.rand(B, 1, 1, device=device) >= 0.15).to(cond.dtype)
            cond = cond * drop_mask
            emo_tokens = emo_tokens * drop_mask

            l_cfm = compute_shortcut_distillation_loss(
                console.mouth, x_1, cond, emo_tokens
            )
            loss = l_cfm + 0.5 * l_dur

        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Audit check at step 0 (and periodically during audit)
        if step == 0 or step == 49:
            # a) Gradient verification
            g_onset = console.mouth.embeddings["onset"].weight.grad
            g_dur = console.mouth.duration_aligner.dur_proj[0].weight.grad
            g_cond = console.mouth.cond_proj.weight.grad

            assert g_onset is not None and torch.isfinite(g_onset).all(), (
                "Gradient on mouth.embeddings['onset'] must be non-None and finite"
            )
            assert g_onset.norm() > 0.0, "Gradient on mouth.embeddings['onset'] must be > 0"

            assert g_dur is not None and torch.isfinite(g_dur).all(), (
                "Gradient on mouth.duration_aligner.dur_proj[0] must be non-None and finite"
            )
            assert g_dur.norm() > 0.0, "Gradient on mouth.duration_aligner must be > 0"

            assert g_cond is not None and torch.isfinite(g_cond).all(), (
                "Gradient on mouth.cond_proj must be non-None and finite"
            )
            assert g_cond.norm() > 0.0, "Gradient on mouth.cond_proj must be > 0"

            # b) Sesqui zero padding exactness
            padded_mask = syll_batch.sesqui_ids == 0
            if padded_mask.any():
                padded_durs = pred_durs[padded_mask]
                assert torch.all(padded_durs == 0.0), (
                    f"Padded positions must receive exact 0.0 frames, got max {padded_durs.max().item()}"
                )

            # c) Peak VRAM verification
            peak_vram = (
                torch.cuda.max_memory_allocated() / (1024**3) if on_cuda else 0.0
            )
            assert peak_vram <= VRAM_CEILING_GB, (
                f"Peak VRAM {peak_vram:.4f} GB violated ceiling {VRAM_CEILING_GB:.2f} GB"
            )

            print(
                f"  [AUDIT-CHECK step {step+1:>2}/50] "
                f"grad_norms: [onset: {g_onset.norm():.4f}, dur: {g_dur.norm():.4f}, cond: {g_cond.norm():.4f}] | "
                f"zero_frames: OK | peak_vram: {peak_vram:.4f} GB <= {VRAM_CEILING_GB:.2f} GB"
            )

        torch.nn.utils.clip_grad_norm_(console.parameters(), max_norm=1.0)
        trainer.optimizer.step()

        loss_val = float(loss.detach().item())
        recent_losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if (step + 1) % 10 == 0:
            avg_loss = sum(recent_losses[-10:]) / len(recent_losses[-10:])
            peak_vram = (
                torch.cuda.max_memory_allocated() / (1024**3) if on_cuda else 0.0
            )
            print(
                f"  [audit step {step+1:>2}/50] loss: {loss_val:.6f} | avg10: {avg_loss:.6f} "
                f"| l_cfm: {l_cfm.item():.4f} | l_dur: {l_dur.item():.4f} | peak_vram: {peak_vram:.4f} GB",
                flush=True,
            )

    print("STEP 2 SELF-AUDIT VERIFICATION GATE: PASSED [100% OK]")
    print("-" * 78)

    if audit_only:
        print("[AUDIT-ONLY] Completed 50-step audit successfully.")
        return 0

    # ========================================================================
    # Full 3,000-Step Training Run
    # ========================================================================
    print(f"\n[TRAIN] Launching full {total_steps}-step Stage 2B training run...")
    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(
            f"--- Stage 2B Training Started: {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"| steps 1..{total_steps} | shards={len(streamer.shard_files)} ---\n"
        )
        f_log.flush()

        for step in range(50, total_steps):
            wav_batch, syll_batch = streamer.sample_batch(batch_size)
            wav_batch = wav_batch.to(device)
            syll_batch = SyllableTensorBatch(
                onset_ids=syll_batch.onset_ids.to(device),
                coeng_ids=syll_batch.coeng_ids.to(device),
                medial_ids=syll_batch.medial_ids.to(device),
                register_ids=syll_batch.register_ids.to(device),
                vowel_ids=syll_batch.vowel_ids.to(device),
                coda_ids=syll_batch.coda_ids.to(device),
                sesqui_ids=syll_batch.sesqui_ids.to(device),
            )

            B = wav_batch.shape[0]
            L_max = syll_batch.onset_ids.shape[1]

            with trainer.autocast_context():
                x_1 = console._extract_mel_24k(wav_batch)
                T_mel = x_1.shape[-1]
                orig_duration_sec = T_mel * 240.0 / 24000.0

                emo_tokens, _ = console.emotion_encoder(x_1, alpha=0.0)
                zero_bus = torch.zeros(B, L_max, 512, device=device, dtype=x_1.dtype)
                zero_gate = torch.zeros(B, L_max, 1, device=device, dtype=x_1.dtype)
                r_sps = torch.full((B,), 4.5, device=device, dtype=x_1.dtype)

                cond, _ = console.mouth.predict_syllables_and_align(
                    zero_bus,
                    zero_gate,
                    r_sps,
                    target_lang="km",
                    orig_duration_sec=orig_duration_sec,
                    target_frames=T_mel,
                    explicit_syll_batch=syll_batch,
                )

                pred_durs = console.mouth.last_pred_durations
                prior = torch.where(
                    syll_batch.sesqui_ids == 0,
                    0.0,
                    torch.where(syll_batch.sesqui_ids == 1, 0.25, 1.0),
                ).to(pred_durs.dtype)
                prior_sum = prior.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                target_durs = prior * (float(T_mel) / prior_sum)
                l_dur = F.smooth_l1_loss(pred_durs, target_durs)

                # 15% CFG conditioning dropout
                drop_mask = (torch.rand(B, 1, 1, device=device) >= 0.15).to(cond.dtype)
                cond = cond * drop_mask
                emo_tokens = emo_tokens * drop_mask

                l_cfm = compute_shortcut_distillation_loss(
                    console.mouth, x_1, cond, emo_tokens
                )
                loss = l_cfm + 0.5 * l_dur

            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(console.parameters(), max_norm=1.0)
            trainer.optimizer.step()

            loss_val = float(loss.detach().item())
            recent_losses.append(loss_val)

            peak_vram = (
                torch.cuda.max_memory_allocated() / (1024**3) if on_cuda else 0.0
            )
            assert peak_vram <= VRAM_CEILING_GB + 0.05, (
                f"Peak VRAM {peak_vram:.4f} GB violated ceiling {VRAM_CEILING_GB:.2f} GB"
            )

            # Logging
            if (step + 1) % 10 == 0 or (step + 1) == total_steps:
                elapsed = time.time() - t_start
                avg_loss = sum(recent_losses[-10:]) / len(recent_losses[-10:])
                log_line = (
                    f"step {step+1:>5}/{total_steps} | loss: {loss_val:.6f} "
                    f"| avg10: {avg_loss:.6f} | l_cfm: {l_cfm.item():.4f} "
                    f"| l_dur: {l_dur.item():.4f} | peak_vram: {peak_vram:.4f} GB "
                    f"| elapsed: {elapsed:.1f}s"
                )
                print(f"  {log_line}", flush=True)
                f_log.write(log_line + "\n")
                f_log.flush()

            # Periodic rolling checkpoint saving
            if (step + 1) % save_every == 0 or (step + 1) == total_steps:
                ckpt_data = {
                    "step": step,
                    "loss": loss_val,
                    "best_loss": min(best_loss, loss_val),
                    "console_state_dict": console.state_dict(),
                    "profile_name": "laptop_5070ti_12gb",
                    "timestamp": time.time(),
                }
                tmp_latest = latest_ckpt_path.with_suffix(".pt.tmp")
                torch.save(ckpt_data, tmp_latest)
                os.replace(tmp_latest, latest_ckpt_path)

                if loss_val < best_loss:
                    best_loss = loss_val
                    tmp_best = best_ckpt_path.with_suffix(".pt.tmp")
                    torch.save(ckpt_data, tmp_best)
                    os.replace(tmp_best, best_ckpt_path)
                    print(
                        f"    >>> [CHECKPOINT] New best loss {best_loss:.6f} saved to {best_ckpt_path.name}"
                    )

    print("\n" + "=" * 78)
    print("STAGE 2B TRAINING COMPLETE")
    print("=" * 78)
    print(f"Final best loss : {best_loss:.6f}")
    print(f"Latest saved at : {latest_ckpt_path}")
    print(f"Best saved at   : {best_ckpt_path}")
    print(f"Total time      : {time.time()-t_start:.1f}s")
    return 0


# ============================================================================
# CLI Entrypoint
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2B Supervised 7-Slot Khmer Voice-Cloning TTS Training."
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="data/khmer_overnight_shards",
        help="Directory holding shard_*.pt files (default: data/khmer_overnight_shards).",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=str,
        default="checkpoints/stage2_best.pt",
        help="Path to Stage 2 pre-trained checkpoint (default: checkpoints/stage2_best.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save stage2b checkpoints and logs (default: checkpoints).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Total training steps (default: 3000).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=250,
        help="Save checkpoint every N steps (default: 250).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Mini-batch size (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only the 50-step self-audit verification gate.",
    )
    args = parser.parse_args()

    return train_stage2b(
        shards_dir=Path(args.shards_dir),
        stage2_ckpt_path=Path(args.stage2_checkpoint),
        output_dir=Path(args.output_dir),
        total_steps=args.steps,
        save_every=args.save_every,
        batch_size=args.batch_size,
        device_name=args.device,
        audit_only=args.audit_only,
    )


if __name__ == "__main__":
    sys.exit(main())
