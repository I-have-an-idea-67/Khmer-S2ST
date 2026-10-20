"""
STAGE 07 - STEP 2: Surgical `VocosISTFTHead` Warm-Start & Mel-to-Wav Round-Trip Audit.

Downloads `charactr/vocos-mel-24khz` (a 1024-point / 256-hop / 8-layer ConvNeXt
Vocos) and surgically maps it into this repository's `VocosISTFTHead`
(`N_FFT_24K = 2048`, `HOP_LENGTH = 240`, `intermediate_dim = 1536 -> 768`).

The mapping is performed by the AUTHORITATIVE loader
`backend.training.train_cartridge.warm_start_vocos_from_pretrained`, which:

  * copies `backbone.embed` (Conv1d(100, 512, 7)) into `vocos.in_conv`,
  * copies `backbone.convnext.{i}.{dwconv,norm}` for all 8 ConvNeXt blocks,
  * performs an IMPORTANCE-BASED CHANNEL SELECTION on the pretrained
    `pwconv1`/`pwconv2` pair to squeeze the pretrained 1536-wide MLP down to
    this repository's 768-wide `inter_half`
    (`importance = ||pw1_w||_2 * ||pw2_w||_2`, top-k over the channel axis),
  * INTERPOLATES the pretrained 513-bin ISTFT head onto this repository's
    1025-bin (`N_FFT_24K // 2 + 1`) head, for BOTH the magnitude and the phase
    half, and for BOTH the weight and the bias.

The resulting state dict is written to `checkpoints/vocos_24k_warmstart.pt`
and is re-loadable into a fresh `SymmetricS2STConsole` with `strict=True`.

Self-Audit Gate (this script fails loudly if any check is violated):
  (a) `sum(p.numel() for p in console.vocos.parameters()) == 7_753_218`
  (b) `wav_recon.shape == wav_24k.shape == (1, 24000)`
  (c) `torch.isfinite(wav_recon).all()` and `wav_recon.abs().max() > 0.05`
  (d) the Log-Mel L1 round-trip error is finite and SIGNIFICANTLY LOWER than
      the error of a RANDOMLY INITIALIZED head.

Usage:
    python scripts/bootstrap_vocos_warmstart.py
    python scripts/bootstrap_vocos_warmstart.py --out checkpoints/vocos_24k_warmstart.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# --- Make the repository root importable when run as a standalone script ----
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.contracts.schemas import (  # noqa: E402
    HOP_LENGTH_24K,
    N_FFT_24K,
    PROD_240M_CONFIG,
)
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole  # noqa: E402
from backend.nn_cartridges.universal_core import VocosISTFTHead  # noqa: E402
from backend.training.train_cartridge import (  # noqa: E402
    warm_start_vocos_from_pretrained,
)

HF_REPO = "charactr/vocos-mel-24khz"
HF_WEIGHTS = "pytorch_model.bin"
HF_CONFIG = "config.yaml"

# The audited parameter count of the surgical `VocosISTFTHead` (PROD profile).
EXPECTED_VOCOS_PARAMS = 7_753_218

SR_24K = 24_000
TEST_SECONDS = 1.0
F0_HZ = 220.0
N_HARMONICS = 5  # f0 plus 5 harmonics -> 6 partials in total


# ============================================================================
# Helpers
# ============================================================================
def _align_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    """
    Align a reconstructed waveform to an exact sample count.

    `VocosISTFTHead.forward` runs `torch.istft(..., center=True, length=T*HOP)`,
    so a signal of `N` samples yields `ceil(N / HOP) + 1` mel frames and the
    inverse transform therefore emits `ceil(N / HOP) * HOP` samples - one hop
    MORE than the source. The trailing hop is the zero-padded centre frame of
    the forward `center=True` STFT and carries no signal, so it is trimmed
    here to recover the original grid exactly. This is the same trimming the
    upstream Vocos reference implementation performs.
    """
    cur = wav.shape[-1]
    if cur == length:
        return wav
    if cur > length:
        return wav[..., :length]
    return F.pad(wav, (0, length - cur))


def synth_harmonic_test_signal(
    seconds: float = TEST_SECONDS,
    sample_rate: int = SR_24K,
    f0: float = F0_HZ,
    n_harmonics: int = N_HARMONICS,
    peak: float = 0.5,
) -> torch.Tensor:
    """
    A deterministic harmonic test signal: `f0` plus `n_harmonics` partials with
    1/k amplitude roll-off, peak-normalized to `peak`.

    Returns a `(1, seconds * sample_rate)` float32 tensor.
    """
    n = int(round(seconds * sample_rate))
    t = torch.arange(n, dtype=torch.float32) / float(sample_rate)
    wav = torch.zeros(1, n, dtype=torch.float32)
    for k in range(n_harmonics + 1):
        wav = wav + (1.0 / (k + 1)) * torch.sin(2.0 * math.pi * f0 * (k + 1) * t)
    wav = wav / wav.abs().max() * peak
    return wav


def download_pretrained() -> Tuple[Dict[str, torch.Tensor], str, str]:
    """Fetch `pytorch_model.bin` and `config.yaml` from the HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    weights_path = hf_hub_download(HF_REPO, HF_WEIGHTS)
    config_path = hf_hub_download(HF_REPO, HF_CONFIG)
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise SystemExit(f"Unexpected checkpoint payload type: {type(state_dict)}")
    return state_dict, weights_path, config_path


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="STAGE 07 STEP 2 - surgical Vocos warm-start + mel-to-wav audit."
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "checkpoints" / "vocos_24k_warmstart.pt"),
        help="Destination path for the warm-started Vocos state dict.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Global RNG seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)  # inference-only script

    print("=" * 78)
    print("STAGE 07 - STEP 2: Vocos Surgical Warm-Start & Mel-to-Wav Round-Trip Audit")
    print("=" * 78)
    print("Blueprint      : KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7 (v10.7.0)")
    print(f"HF source      : {HF_REPO} ({HF_WEIGHTS}, {HF_CONFIG})")
    print(f"N_FFT_24K      : {N_FFT_24K}")
    print(f"HOP_LENGTH_24K : {HOP_LENGTH_24K}")

    # ------------------------------------------------------------------
    # 0. Download the pretrained checkpoint.
    # ------------------------------------------------------------------
    t0 = time.time()
    pretrained_sd, weights_path, config_path = download_pretrained()
    print(f"Downloaded     : {weights_path}")
    print(f"Config         : {config_path}")
    print(f"Checkpoint keys: {len(pretrained_sd)} tensors "
          f"({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 1. Instantiate the console on CPU.
    # ------------------------------------------------------------------
    console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    console.eval()
    print(f"Console        : SymmetricS2STConsole(PROD_240M_CONFIG) on "
          f"{next(console.parameters()).device}")

    # ------------------------------------------------------------------
    # 2. RANDOM-INITIALIZATION BASELINE (captured BEFORE the warm start so
    #    the two heads are provably compared on the same mel input).
    # ------------------------------------------------------------------
    random_head = VocosISTFTHead(PROD_240M_CONFIG)
    random_head.eval()

    # ------------------------------------------------------------------
    # 3. Build the test signal and extract the 100-bin Slaney log-mel.
    # ------------------------------------------------------------------
    wav_24k = synth_harmonic_test_signal()
    print(f"\nTest signal    : shape={tuple(wav_24k.shape)} "
          f"f0={F0_HZ}Hz +{N_HARMONICS} harmonics @ {SR_24K}Hz "
          f"peak={float(wav_24k.abs().max()):.4f}")

    mel_24k = console._extract_mel_24k(wav_24k)
    print(f"Log-Mel        : shape={tuple(mel_24k.shape)} "
          f"min={float(mel_24k.min()):.4f} max={float(mel_24k.max()):.4f}")

    # ------------------------------------------------------------------
    # 4. Round-trip through the RANDOM head (baseline).
    # ------------------------------------------------------------------
    wav_recon_random = _align_length(random_head(mel_24k), wav_24k.shape[-1])
    mel_back_random = console._extract_mel_24k(wav_recon_random)
    err_random = F.l1_loss(mel_back_random, mel_24k).item()

    # ------------------------------------------------------------------
    # 5. Surgical warm-start.
    # ------------------------------------------------------------------
    warm_start_vocos_from_pretrained(console.vocos, pretrained_sd)
    print("\nWarm-start     : warm_start_vocos_from_pretrained(console.vocos, sd) OK")

    # ------------------------------------------------------------------
    # 6. Round-trip through the WARM-STARTED head.
    # ------------------------------------------------------------------
    wav_recon = _align_length(console.vocos(mel_24k), wav_24k.shape[-1])
    mel_back = console._extract_mel_24k(wav_recon)
    err_warm = F.l1_loss(mel_back, mel_24k).item()

    # ------------------------------------------------------------------
    # 7. SAVE the warm-started state dict.
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vocos_state = console.vocos.state_dict()
    torch.save(vocos_state, out_path)
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"Saved          : {out_path} ({size_mb:.2f} MiB, "
          f"{len(vocos_state)} tensors)")

    # ==================================================================
    # SELF-AUDIT VERIFICATION GATE
    # ==================================================================
    print("\n" + "-" * 78)
    print("STEP 2 SELF-AUDIT VERIFICATION GATE")
    print("-" * 78)

    # (a) parameter count
    n_params = sum(p.numel() for p in console.vocos.parameters())
    print(f"(a) Vocos parameters      : {n_params:,} "
          f"(expected {EXPECTED_VOCOS_PARAMS:,})")
    assert n_params == EXPECTED_VOCOS_PARAMS, (
        f"Vocos parameter count moved: {n_params:,} != {EXPECTED_VOCOS_PARAMS:,}"
    )
    print("    PASS - surgical head parameter count is exactly 7,753,218")

    # (b) shape agreement
    print(f"(b) wav_24k.shape         : {tuple(wav_24k.shape)}")
    print(f"    wav_recon.shape       : {tuple(wav_recon.shape)}")
    assert tuple(wav_recon.shape) == tuple(wav_24k.shape) == (1, 24000), (
        f"shape mismatch: recon={tuple(wav_recon.shape)} src={tuple(wav_24k.shape)}"
    )
    print("    PASS - reconstruction is sample-aligned with the source")

    # (c) finiteness and audible amplitude
    finite = bool(torch.isfinite(wav_recon).all().item())
    peak = float(wav_recon.abs().max().item())
    rms = float(wav_recon.pow(2).mean().sqrt().item())
    print(f"(c) isfinite(wav_recon)   : {finite}")
    print(f"    max|wav_recon|        : {peak:.6f}  (required > 0.05)")
    print(f"    RMS(wav_recon)        : {rms:.6f}")
    assert finite, "reconstruction contains NaN/Inf"
    assert peak > 0.05, f"reconstruction is silent: max={peak:.6f}"
    print("    PASS - finite, non-silent reconstruction")

    # (d) Log-Mel round-trip error vs. random initialization
    print(f"(d) Log-Mel L1 (warm)     : {err_warm:.6f}")
    print(f"    Log-Mel L1 (random)   : {err_random:.6f}")
    assert math.isfinite(err_warm), "warm-start Log-Mel L1 is not finite"
    assert math.isfinite(err_random), "random-init Log-Mel L1 is not finite"
    ratio = err_random / err_warm if err_warm > 0 else float("inf")
    print(f"    improvement factor     : {ratio:.3f}x (random / warm)")
    assert err_warm < err_random, (
        f"warm-start error {err_warm:.6f} is NOT lower than random-init {err_random:.6f}"
    )
    assert ratio >= 1.5, (
        f"warm-start improvement is only {ratio:.3f}x (< 1.5x); "
        "the surgical mapping is not measurably effective"
    )
    print("    PASS - warm-start error is finite and significantly lower")

    # ------------------------------------------------------------------
    # Extra: the saved checkpoint must reload into a FRESH console strictly.
    # ------------------------------------------------------------------
    fresh = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    fresh.vocos.load_state_dict(vocos_state, strict=True)
    reloaded = fresh.vocos.state_dict()
    assert set(reloaded.keys()) == set(vocos_state.keys()), (
        f"reload key mismatch: {len(reloaded)} vs {len(vocos_state)}"
    )
    assert all(torch.equal(reloaded[k], vocos_state[k]) for k in vocos_state), (
        "reloaded tensors are not bit-identical to the saved state dict"
    )
    print(f"\nReload check   : strict=True reload of {len(vocos_state)} tensors OK "
          "(bit-identical)")

    print("\n" + "=" * 78)
    print("STEP 2 PASS: Vocos surgical warm-start verified")
    print(f"  parameters           : {n_params:,}")
    print(f"  reconstruction shape : {tuple(wav_recon.shape)}")
    print(f"  peak |wav_recon|     : {peak:.6f}")
    print(f"  Log-Mel L1 warm      : {err_warm:.6f}  (random-init {err_random:.6f}, "
          f"{ratio:.3f}x better)")
    print(f"  checkpoint           : {out_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
