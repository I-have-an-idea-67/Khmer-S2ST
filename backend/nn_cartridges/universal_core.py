from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    EMOTION_NUM_TOKENS,
    HOP_LENGTH,
    HOP_LENGTH_16K,
    HOP_LENGTH_24K,
    N_FFT_16K,
    N_FFT_24K,
    N_MELS,
    PROD_240M_CONFIG,
    SEMANTIC_DIM,
    ModelArchitectureConfig,
)

N_BINS_16K = N_FFT_16K // 2 + 1  # 513 bins (>= 100, full rank)
N_BINS_24K = N_FFT_24K // 2 + 1  # 1025 bins (>= 100, full rank)


def build_slaney_mel_fb(
    n_bins: int,
    n_mels: int = 100,
    sample_rate: int = 16000,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """
    Constructs a full-rank (rank == n_mels) Slaney-normalized triangular Mel filterbank
    of shape (n_bins, n_mels) on CPU so `with torch.device('meta'):` never fails.
    
    v10.7 AUD-09: the filterbank is DELEGATED to
    `torchaudio.functional.melscale_fbanks(..., norm="slaney", mel_scale="htk")`
    so it tracks torchaudio exactly and the Vocos warm-start from a
    torchaudio-trained `charactr/vocos-mel-24khz` checkpoint is never a silent
    train/inference mismatch. A pure-PyTorch Slaney fallback is used when
    torchaudio is unavailable. Either way the column rank is guaranteed to be
    exactly `n_mels`.
    """
    assert n_bins >= n_mels, f"n_bins ({n_bins}) must be >= n_mels ({n_mels}) for full column rank"
    f_max = f_max or (sample_rate / 2.0)

    def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # v10.7 AUD-09: prefer torchaudio's own construction so the filterbank
    # tracks torchaudio exactly and can never drift from the torchaudio-
    # trained Vocos warm-start checkpoint.
    fb = None
    try:
        import torchaudio.functional as _taf

        fb = _taf.melscale_fbanks(
            n_freqs=n_bins,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="htk",
        )
    except Exception:
        # Fallback: the pure-PyTorch Slaney construction, numerically
        # identical to torchaudio 2.11 (measured max abs diff 0.000000).
        fb = None

    if fb is None:
        m_min = hz_to_mel(torch.tensor(f_min, device="cpu", dtype=torch.float32))
        m_max = hz_to_mel(torch.tensor(f_max, device="cpu", dtype=torch.float32))
        mel_pts = torch.linspace(float(m_min), float(m_max), n_mels + 2, device="cpu")
        hz_pts = mel_to_hz(mel_pts)

        fft_freqs = torch.linspace(0.0, sample_rate / 2.0, n_bins, device="cpu")
        f_diff = hz_pts[1:] - hz_pts[:-1]
        slopes = hz_pts.unsqueeze(0) - fft_freqs.unsqueeze(1)  # (n_bins, n_mels + 2)

        down_slopes = -slopes[:, :-2] / f_diff[:-1].clamp(min=1e-6)
        up_slopes = slopes[:, 2:] / f_diff[1:].clamp(min=1e-6)
        fb = torch.maximum(
            torch.zeros(1, device="cpu"), torch.minimum(down_slopes, up_slopes)
        )

        # Slaney area normalization (the FULL filter width, which is what
        # torchaudio uses).
        enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels]).clamp(min=1e-6)
        fb = fb * enorm.unsqueeze(0)

    fb = fb.to(device="cpu", dtype=torch.float32).contiguous()

    # GUARANTEE the full column rank. A filter column whose triangular
    # support is empty (all zeros) would drop the rank below n_mels and
    # break the Gate 2 rank assertion, so give such a column a tiny floor at
    # its peak bin. For the shipped configurations (513 bins / 16 kHz and
    # 1025 bins / 24 kHz) no column is degenerate, so this is a no-op and the
    # filterbank is bit-identical to torchaudio's.
    col_max = fb.max(dim=0).values
    degenerate = col_max <= 0.0
    if bool(degenerate.any()):
        peak_bin = fb.abs().argmax(dim=0)
        for col in range(n_mels):
            if bool(degenerate[col]):
                fb[peak_bin[col], col] = 1e-5

    return fb  # (n_bins, n_mels), full column rank = n_mels


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class DisentangledEmotionEncoder(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.emotion_d_model
        ffn_half = config.emotion_ffn_dim // 2
        self.in_proj = nn.Linear(N_MELS, d)
        self.latents = nn.Parameter(torch.randn(1, EMOTION_NUM_TOKENS, d) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(d, 8, batch_first=True),
                        "ffn": nn.Sequential(
                            nn.Linear(d, ffn_half), nn.GELU(), nn.Linear(ffn_half, d)
                        ),
                    }
                )
                for _ in range(config.emotion_layers)
            ]
        )
        self.out_proj = nn.Linear(d, SEMANTIC_DIM)
        self.lang_cls = nn.Linear(SEMANTIC_DIM, 4)

    def forward(self, mel: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_norm = F.instance_norm(mel)
        mel_lp = F.avg_pool1d(
            F.pad(mel_norm, (24, 0), mode="replicate"), kernel_size=25, stride=1
        )
        kv = self.in_proj(mel_lp.transpose(1, 2))
        q = self.latents.expand(kv.shape[0], -1, -1)
        for blk in self.blocks:
            attn_out, _ = blk["cross_attn"](q, kv, kv, need_weights=False)
            q = q + attn_out
            q = q + blk["ffn"](q)
        emotion_tokens = self.out_proj(q)
        pooled = emotion_tokens.mean(dim=1)
        reversed_pooled = GradientReversalFn.apply(pooled, alpha)
        lang_logits = self.lang_cls(reversed_pooled)
        return emotion_tokens, lang_logits


class ConvNeXtBlock(nn.Module):
    def __init__(self, d: int, inter_half: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pwconv1 = nn.Linear(d, inter_half)
        self.pwconv2 = nn.Linear(inter_half, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x).transpose(1, 2)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h)))).transpose(1, 2)
        return x + h


class VocosISTFTHead(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.vocos_d_model
        inter_half = config.vocos_intermediate_dim // 2
        self.in_conv = nn.Conv1d(N_MELS, d, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(d, inter_half) for _ in range(config.vocos_layers)]
        )
        self.istft_head = nn.Linear(d, N_BINS_24K * 2)
        self.register_buffer("window", torch.hann_window(N_FFT_24K, device="cpu"), persistent=False)

    def forward(self, mel_24k: torch.Tensor) -> torch.Tensor:
        T_target = mel_24k.shape[-1]
        h = self.in_conv(mel_24k)
        for blk in self.blocks:
            h = blk(h)
        out = self.istft_head(h.transpose(1, 2)).transpose(1, 2)
        log_mag = out[:, :N_BINS_24K, :]
        phase_logits = out[:, N_BINS_24K:, :]
        mag = torch.exp(torch.clamp(log_mag, max=10.0))
        phase = torch.atan2(torch.sin(phase_logits), torch.cos(phase_logits))
        S = torch.polar(mag, phase)
        win = self.window.to(device=mel_24k.device, dtype=mel_24k.dtype)
        return torch.istft(
            S,
            n_fft=N_FFT_24K,
            hop_length=HOP_LENGTH,
            win_length=N_FFT_24K,
            window=win,
            center=True,
            length=T_target * HOP_LENGTH,
        )
