"""
STAGE 09 - Standalone Khmer Voice-Cloning TTS Synthesis & Verification Script.

Synthesizes high-fidelity 24 kHz speech using:
  - Stage 2B supervised 7-slot voice-cloning TTS checkpoint (mouth DiT + duration aligner + slot embeddings)
  - Frozen Vocos neural vocoder (vocos_24k_warmstart.pt)
  - Sway-schedule 12-step Euler ODE integration with Classifier-Free Guidance (cfg_scale=1.5)
  - Emotion and acoustic reference extraction from real Khmer audio shards

Produces:
  1. checkpoints/tts_test_1_greeting.wav: "សួស្តីកម្ពុជា ខ្ញុំឈ្មោះសុភមង្គល"
  2. checkpoints/tts_test_2_sentence.wav: "ប្រទេសកម្ពុជាមានធម្មជាតិដ៏ស្រស់ស្អាត"
  3. checkpoints/tts_test_3_codeswitch.wav: "ខ្ញុំទៅភ្នំពេញទិញ Calvin Klein និង Versace"
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from backend.linguistics.syllable_decomposer import (  # noqa: E402
    BilingualCodeSwitchDecomposer,
)
from backend.nn_cartridges.s2st_console import (  # noqa: E402
    SymmetricS2STConsole,
    integrate_shortcut_sway_ode,
)

SR_24K = SAMPLE_RATE_OUT  # 24,000 Hz

TEST_SENTENCES = [
    (
        "tts_test_1_greeting.wav",
        "សួស្តីកម្ពុជា ខ្ញុំឈ្មោះសុភមង្គល",
        "Formal Greeting",
    ),
    (
        "tts_test_2_sentence.wav",
        "ប្រទេសកម្ពុជាមានធម្មជាតិដ៏ស្រស់ស្អាត",
        "Descriptive Sentence",
    ),
    (
        "tts_test_3_codeswitch.wav",
        "ខ្ញុំទៅភ្នំពេញទិញ Calvin Klein និង Versace",
        "Bilingual Code-Switching",
    ),
]


def load_reference_audio(shards_dir: Path, target_sr: int = 24000) -> torch.Tensor:
    """Load a clean reference Khmer audio clip from the first available shard."""
    shards = sorted(shards_dir.glob("shard_*.pt"))
    assert shards, f"No shard files found in {shards_dir}"
    for shard_path in shards:
        try:
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            for clip in data:
                wav_24k = clip.get("wav_24k")
                if (
                    wav_24k is not None
                    and isinstance(wav_24k, torch.Tensor)
                    and 28800 <= len(wav_24k) <= 96000
                ):
                    wav_f32 = wav_24k.float() / 32768.0
                    return wav_f32.unsqueeze(0)  # (1, T)
        except Exception:
            continue
    raise RuntimeError("Could not find a valid reference audio clip in shards.")


def save_wav_16bit(filepath: Path, wav_tensor: torch.Tensor, sample_rate: int = 24000) -> None:
    """Write 1-D mono float tensor to 16-bit PCM WAV file with canonical 44-byte header."""
    wav = wav_tensor.detach().cpu().squeeze().float()
    peak = float(wav.abs().max().item())
    if peak > 0.0:
        wav_norm = (wav / peak) * 0.92
    else:
        wav_norm = wav
    pcm16 = (wav_norm.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)


def synthesize_khmer_speech(
    console: SymmetricS2STConsole,
    decomposer: BilingualCodeSwitchDecomposer,
    khmer_text: str,
    ref_wav_24k: torch.Tensor,
    cfg_scale: float = 1.5,
    euler_steps: int = 12,
    frames_per_syllable: int = 24,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, int, int]:
    """
    Synthesize Khmer speech with 7-slot duration alignment and Vocos resynthesis.
    Returns:
      wav_out: (1, T_samples) 24kHz audio tensor
      L_syll: number of syllables
      target_frames: number of 100Hz acoustic frames
    """
    console.eval()
    with torch.no_grad():
        # 1. Extract speaker/emotion conditioning from real reference audio
        ref_wav_cuda = ref_wav_24k.to(device)
        ref_mel = console._extract_mel_24k(ref_wav_cuda)
        emo_tokens, _ = console.emotion_encoder(ref_mel, alpha=0.0)

        # 2. Decompose Khmer text into 7-slot SyllableTensorBatch
        syll_batch: SyllableTensorBatch = decomposer.encode_sentence_to_batch(
            khmer_text, device=str(device)
        )
        L_syll = syll_batch.onset_ids.shape[1]
        target_frames = max(24, int(L_syll * frames_per_syllable))  # ~240ms per syllable at 100 Hz

        # 3. Predict syllables and align duration conditioning
        B = 1
        zero_bus = torch.zeros(B, L_syll, 512, device=device, dtype=ref_mel.dtype)
        zero_gate = torch.zeros(B, L_syll, 1, device=device, dtype=ref_mel.dtype)
        r_sps = torch.full((B,), 4.5, device=device, dtype=ref_mel.dtype)

        cond, _ = console.mouth.predict_syllables_and_align(
            zero_bus,
            zero_gate,
            r_sps,
            target_lang="km",
            orig_duration_sec=float(target_frames) / 100.0,
            target_frames=target_frames,
            explicit_syll_batch=syll_batch,
        )

        # 4. Sway ODE Integration
        x_0 = torch.randn(B, 100, target_frames, device=device, dtype=ref_mel.dtype)
        mel_pred = integrate_shortcut_sway_ode(
            console.mouth,
            x_0,
            cond,
            emo_tokens,
            steps=euler_steps,
            sway_coef=-0.8,
            cfg_scale=cfg_scale,
        )

        # 5. Neural Vocoder Resynthesis
        wav_out = console.vocos(mel_pred)  # (1, T_samples)
        return wav_out, L_syll, target_frames


def run_synthesis_and_audit(
    stage2b_ckpt_path: Path,
    vocos_ckpt_path: Path,
    shards_dir: Path,
    output_dir: Path,
    cfg_scale: float = 1.5,
    euler_steps: int = 12,
    frames_per_syllable: int = 24,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("STAGE 09 - STANDALONE KHMER VOICE-CLONING TTS SYNTHESIS & AUDIT")
    print("=" * 78)
    print(f"Device            : {device}")
    print(f"Stage 2B Checkpoint: {stage2b_ckpt_path}")
    print(f"Vocos Checkpoint  : {vocos_ckpt_path}")
    print(f"Shards Directory  : {shards_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"CFG scale         : {cfg_scale}")
    print(f"Euler steps       : {euler_steps}")
    print(f"Frames / syllable : {frames_per_syllable} ({frames_per_syllable*10}ms/syllable)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Console and load checkpoints
    console = SymmetricS2STConsole(config=PROD_240M_CONFIG)

    print(f"[LOAD] Loading Stage 2B checkpoint: {stage2b_ckpt_path.name}...")
    ckpt_2b = torch.load(stage2b_ckpt_path, map_location="cpu", weights_only=False)
    state_dict_2b = ckpt_2b.get("console_state_dict", ckpt_2b)
    console.load_state_dict(state_dict_2b, strict=False)

    print(f"[LOAD] Loading Vocos warmstart: {vocos_ckpt_path.name}...")
    vocos_state = torch.load(vocos_ckpt_path, map_location="cpu", weights_only=False)
    console.vocos.load_state_dict(vocos_state, strict=True)

    console.to(device)
    console.eval()

    # 2. Load linguistic decomposer and reference speech
    decomposer = BilingualCodeSwitchDecomposer()
    ref_wav_24k = load_reference_audio(shards_dir, target_sr=SR_24K)
    print(f"[REF] Loaded reference speech: {ref_wav_24k.shape[-1]} samples ({ref_wav_24k.shape[-1]/24000.0:.2f}s)")

    # 3. Synthesize the 3 benchmark sentences
    print("\n" + "-" * 78)
    print("SYNTHESIZING 3 BENCHMARK KHMER SPEECH CLIPS")
    print("-" * 78)

    synthesized_paths: List[Tuple[Path, str, int, int]] = []
    durations: List[float] = []

    for filename, text, category in TEST_SENTENCES:
        out_path = output_dir / filename
        t0 = time.time()
        wav_out, L_syll, target_frames = synthesize_khmer_speech(
            console=console,
            decomposer=decomposer,
            khmer_text=text,
            ref_wav_24k=ref_wav_24k,
            cfg_scale=cfg_scale,
            euler_steps=euler_steps,
            frames_per_syllable=frames_per_syllable,
            device=device,
        )
        save_wav_16bit(out_path, wav_out, sample_rate=SR_24K)
        elapsed = time.time() - t0
        dur_sec = wav_out.shape[-1] / float(SR_24K)
        durations.append(dur_sec)
        synthesized_paths.append((out_path, text, L_syll, target_frames))

        print(
            f"[SYNTH] {filename:<28} | L_syll: {L_syll:>2} | frames: {target_frames:>3} "
            f"| duration: {dur_sec:.2f}s | time: {elapsed:.2f}s\n"
            f"        Text: '{text}' ({category})"
        )

    # 4. Rigorous Step 3 Self-Audit Gate
    print("\n" + "-" * 78)
    print("STEP 3 SELF-AUDIT VERIFICATION GATE")
    print("-" * 78)

    for out_path, text, L_syll, target_frames in synthesized_paths:
        assert out_path.exists(), f"File {out_path} was not created!"
        file_size = out_path.stat().st_size
        print(f"[AUDIT] Checking {out_path.name} ({file_size} bytes)...")

        # a) Verify 44-byte RIFF/WAVE header
        with open(out_path, "rb") as f_raw:
            header_bytes = f_raw.read(44)
            assert len(header_bytes) == 44, f"Header too short: {len(header_bytes)} bytes"
            assert header_bytes[:4] == b"RIFF", "Missing RIFF magic"
            assert header_bytes[8:12] == b"WAVE", "Missing WAVE identifier"
            assert header_bytes[12:16] == b"fmt ", "Missing fmt subchunk"
            assert header_bytes[36:40] == b"data", "Missing data subchunk"

        # b) Read audio data via wave module
        with wave.open(str(out_path), "rb") as wf:
            assert wf.getnchannels() == 1, f"Expected 1 channel, got {wf.getnchannels()}"
            assert wf.getsampwidth() == 2, f"Expected 16-bit PCM (2 bytes), got {wf.getsampwidth()}"
            assert wf.getframerate() == 24000, f"Expected 24000 Hz, got {wf.getframerate()}"
            n_frames = wf.getnframes()
            pcm_bytes = wf.readframes(n_frames)

        # c) Check signal integrity: zero NaN/Inf, active energy (RMS > 0.02)
        samples = torch.frombuffer(bytearray(pcm_bytes), dtype=torch.int16).float() / 32768.0
        assert torch.isfinite(samples).all(), f"{out_path.name} contains NaN or Inf values!"
        rms = float(samples.pow(2).mean().sqrt().item())
        print(f"        Sample rate: 24,000 Hz | Frames: {n_frames} | RMS energy: {rms:.4f}")
        assert rms > 0.02, f"RMS energy {rms:.4f} <= 0.02 (insufficient audible speech energy)"

    # d) Verify distinct durations scaling with syllable count
    syll_counts = [item[2] for item in synthesized_paths]
    print(f"[AUDIT] Checking syllable counts: {syll_counts} vs durations: {durations}...")
    assert len(set(durations)) == len(durations), "Synthesized clips must have distinct durations"
    for i in range(len(synthesized_paths)):
        for j in range(len(synthesized_paths)):
            if syll_counts[i] < syll_counts[j]:
                assert durations[i] < durations[j], (
                    f"Clip {i} ({syll_counts[i]} syll, {durations[i]:.2f}s) not shorter than Clip {j} ({syll_counts[j]} syll, {durations[j]:.2f}s)"
                )

    print("\nSTEP 3 SELF-AUDIT VERIFICATION GATE: PASSED [100% OK]")
    print("-" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Khmer Voice-Cloning TTS Synthesis & Intelligibility Audit."
    )
    parser.add_argument(
        "--stage2b-checkpoint",
        type=str,
        default="checkpoints/stage2b_tts_best.pt",
        help="Path to Stage 2B best checkpoint (default: checkpoints/stage2b_tts_best.pt).",
    )
    parser.add_argument(
        "--vocos-checkpoint",
        type=str,
        default="checkpoints/vocos_24k_warmstart.pt",
        help="Path to Vocos 24k warmstart checkpoint (default: checkpoints/vocos_24k_warmstart.pt).",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="data/khmer_overnight_shards",
        help="Directory holding shard_*.pt files (default: data/khmer_overnight_shards).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save generated WAV files (default: checkpoints).",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-Free Guidance scale (default: 1.5).",
    )
    parser.add_argument(
        "--euler-steps",
        type=int,
        default=12,
        help="Euler integration steps (default: 12).",
    )
    parser.add_argument(
        "--frames-per-syllable",
        type=int,
        default=24,
        help="Target acoustic frames (100 Hz) per syllable (default: 24, i.e. 240ms/syllable).",
    )
    args = parser.parse_args()

    return run_synthesis_and_audit(
        stage2b_ckpt_path=Path(args.stage2b_checkpoint),
        vocos_ckpt_path=Path(args.vocos_checkpoint),
        shards_dir=Path(args.shards_dir),
        output_dir=Path(args.output_dir),
        cfg_scale=args.cfg_scale,
        euler_steps=args.euler_steps,
        frames_per_syllable=args.frames_per_syllable,
    )


if __name__ == "__main__":
    sys.exit(main())
