from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    HOP_LENGTH,
    MAX_ELASTIC_PAUSE_BORROW_MS,
    MINOR_SYLLABLE_MAX_FRAMES,
    SAMPLE_RATE_OUT,
    SEMANTIC_DIM,
)


class LearnedDurationAligner(nn.Module):
    def __init__(self, d_model: int = SEMANTIC_DIM, sigma: float = 1.5) -> None:
        super().__init__()
        self.d_model = d_model
        self.sigma = sigma
        self.dur_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def compute_elastic_target_frames(
        orig_duration_sec: float,
        trailing_pause_ms: float,
        speech_rate_sps: float,
        playback_rate: float = 1.0,
    ) -> int:
        borrow_ms = 0.0
        if speech_rate_sps > 4.8 and trailing_pause_ms > 40.0:
            borrow_ms = min(MAX_ELASTIC_PAUSE_BORROW_MS, 0.65 * trailing_pause_ms)
        effective_sec = orig_duration_sec + (borrow_ms / 1000.0)
        safe_rate = max(0.5, min(2.5, float(playback_rate)))
        scaled_sec = effective_sec / safe_rate
        return max(6, int(round(scaled_sec * SAMPLE_RATE_OUT / HOP_LENGTH)))

    def forward(
        self,
        syll_hidden: torch.Tensor,
        sesqui_ids: torch.Tensor,
        target_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_dur = F.softplus(self.dur_proj(syll_hidden).squeeze(-1)) + 0.1
        # v10.7 AUD-107-01: sesqui_ids == 0 marks a DEFERRED (wait-gated)
        # syllable, which is not spoken in this clause and must consume
        # zero acoustic frames. Without this the deferred syllables still
        # received a 0.25 or 1.00 duration weight and the diffusion model
        # compressed the REAL syllables to make room for them.
        prior = torch.where(
            sesqui_ids == 0,
            0.0,
            torch.where(sesqui_ids == 1, 0.25, 1.00),
        ).to(raw_dur.dtype)
        weighted = raw_dur * prior
        norm_dur = weighted * (
            float(target_frames) / (weighted.sum(dim=-1, keepdim=True) + 1e-8)
        )

        is_minor = sesqui_ids == 1
        is_major = ~is_minor
        clamped_minor = torch.where(
            is_minor, torch.clamp(norm_dur, max=MINOR_SYLLABLE_MAX_FRAMES), norm_dur
        )
        excess = (norm_dur - clamped_minor).sum(dim=-1, keepdim=True)
        major_weights = clamped_minor * is_major.to(norm_dur.dtype)
        major_sum = major_weights.sum(dim=-1, keepdim=True)
        redistributed = torch.where(
            major_sum > 1e-6,
            clamped_minor + excess * (major_weights / (major_sum + 1e-8)),
            clamped_minor,
        )
        final_durations = torch.where(
            is_minor,
            torch.clamp(redistributed, max=MINOR_SYLLABLE_MAX_FRAMES),
            redistributed,
        )

        cum_dur = torch.cumsum(final_durations, dim=-1)
        centers = cum_dur - 0.5 * final_durations
        frame_grid = (
            torch.arange(
                target_frames, device=syll_hidden.device, dtype=syll_hidden.dtype
            ).view(1, target_frames, 1)
            + 0.5
        )
        attn_logits = -0.5 * ((frame_grid - centers.unsqueeze(1)) / self.sigma) ** 2
        # v10.7 AUD-107-09: mask the DEFERRED (sesqui_ids == 0) columns out
        # of the Gaussian attention so they cannot absorb attention mass on
        # the tail frames of the clause. A deferred syllable has a zero
        # duration, so its cumulative centre sits at target_frames, but it
        # still received softmax mass there and squeezed the REAL syllables.
        #
        # The has_active guard matters: a row in which EVERY syllable is
        # deferred has no active column, and masking it would leave the
        # softmax over an all-(-1e9) vector. The guard keeps that case
        # explicit and defined.
        active_mask = (sesqui_ids > 0).unsqueeze(1)  # (B, 1, L)
        has_active = active_mask.any(dim=-1, keepdim=True)  # (B, 1, 1)
        attn_logits = torch.where(
            has_active & (~active_mask),
            torch.full_like(attn_logits, -1e9),
            attn_logits,
        )
        attn_weights = F.softmax(attn_logits, dim=-1)
        aligned_hidden = torch.bmm(attn_weights, syll_hidden)
        return aligned_hidden, final_durations
