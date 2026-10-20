from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


class DynamicAcousticCorruptor:
    @staticmethod
    def warp_formants_vtlp(
        mel: torch.Tensor,
        alpha_min: float = 0.85,
        alpha_max: float = 1.15,
        pivot: float = 0.65,
    ) -> torch.Tensor:
        """0-Hz and Nyquist anchored piecewise-linear VTLP using 2D (x, y) grid_sample."""
        B, M, T = mel.shape
        device, dtype = mel.device, mel.dtype
        u = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype).view(1, M, 1).expand(B, M, T)
        alpha = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(alpha_min, alpha_max)
        boundary = torch.clamp(alpha * pivot, min=0.10, max=0.92)
        u_warped = torch.where(
            u <= boundary,
            u * (pivot / boundary),
            pivot + (u - boundary) * ((1.0 - pivot) / (1.0 - boundary)),
        )
        y_grid = 2.0 * u_warped - 1.0
        x_grid = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype).view(1, 1, T).expand(B, M, T)
        grid = torch.stack([x_grid, y_grid], dim=-1)  # (B, M, T, 2)
        return F.grid_sample(
            mel.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(1)

    @staticmethod
    def corrupt_youtube_mixture(
        clean_vocal_a: torch.Tensor,
        clean_vocal_b: Optional[torch.Tensor] = None,
        bgm_or_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mix = clean_vocal_a.clone()
        B, T = mix.shape
        device, dtype = mix.device, mix.dtype
        rms_a = mix.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)

        # 1. Co-host overlap (-10 dB to 0 dB) scaled relative to rms_a
        if clean_vocal_b is not None:
            rms_b = clean_vocal_b.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
            db = torch.empty(B, 1, device=device, dtype=dtype).uniform_(-10.0, 0.0)
            target_b_rms = rms_a * (10.0 ** (db / 20.0))
            scaled_b = clean_vocal_b * (target_b_rms / rms_b)
            mix = mix + scaled_b

        # 2. Environmental / BGM noise (-3 dB to +15 dB SNR) scaled relative to rms_a
        noise = bgm_or_noise if bgm_or_noise is not None else torch.randn_like(mix)
        rms_noise = noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        snr_db = torch.empty(B, 1, device=device, dtype=dtype).uniform_(-3.0, 15.0)
        target_noise_rms = rms_a * (10.0 ** (-snr_db / 20.0))
        scaled_noise = noise * (target_noise_rms / rms_noise)
        mix = mix + scaled_noise

        # 3. Decaying room echo tail
        t_tail = torch.linspace(0.0, 1.0, 64, device=device, dtype=dtype)
        fir = torch.exp(-3.0 * t_tail).unsqueeze(0).unsqueeze(0)
        fir = fir / fir.norm(p=2)
        mix = F.conv1d(F.pad(mix.unsqueeze(1), (63, 0)), fir).squeeze(1)

        # 4. Soft tanh saturation
        drive = 1.6
        return torch.tanh(mix * drive) / math.tanh(drive)
