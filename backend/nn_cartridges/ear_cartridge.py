from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from backend.contracts.schemas import (
    HOP_LENGTH_16K as HOP_16K,
    LANG_TO_ID,
    N_FFT_16K,
    N_MELS,
    PROD_240M_CONFIG,
    SAMPLE_RATE_IN,
    SEMANTIC_DIM,
    CartridgeThreadState,
    LanguageCode,
    ModelArchitectureConfig,
)
from backend.nn_cartridges.universal_core import build_slaney_mel_fb

N_BINS_16K = N_FFT_16K // 2 + 1  # 513 bins -> guarantees rank 100


class FastConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dwconv = nn.Conv1d(d, d, kernel_size=31, padding=15, groups=d)
        # v10.7 P1-E: BatchNorm1d normalizes over the BATCH dimension, which is
        # size 1 during streaming inference, so its running statistics are
        # corrupted by batch-statistics estimation. GroupNorm normalizes per
        # sample and is correct at B=1. Verified parameter-neutral: both hold
        # 2*d trainable parameters (1,024 prod / 256 CI per block); the
        # BatchNorm running buffers are not counted by .parameters().
        self.bn = nn.GroupNorm(8, d)
        self.pw2 = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d))
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norms[0](x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norms[1](x)
        u, v = self.pw1(h).chunk(2, dim=-1)
        c = u * torch.sigmoid(v)
        c = self.dwconv(c.transpose(1, 2))
        c = torch.nn.functional.gelu(self.bn(c)).transpose(1, 2)
        x = x + self.pw2(c)

        x = x + self.ffn(self.norms[2](x))
        return x


class StreamingEarCartridge(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.ear_d_model
        self.register_buffer("window", torch.hann_window(N_FFT_16K, device="cpu"), persistent=False)
        self.register_buffer(
            "mel_fb",
            build_slaney_mel_fb(N_BINS_16K, N_MELS, sample_rate=SAMPLE_RATE_IN),
            persistent=False,
        )
        self.subsampler = nn.Sequential(
            nn.Conv1d(N_MELS, d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.lang_emb = nn.Embedding(4, d)
        self.blocks = nn.ModuleList(
            [
                FastConformerBlock(d, config.ear_heads, config.ear_ffn_dim)
                for _ in range(config.ear_layers)
            ]
        )
        self.bus_proj = nn.Linear(d, SEMANTIC_DIM)
        self.rate_head = nn.Linear(SEMANTIC_DIM, 1)
        # v10.7 P0-A: the per-call acoustic taps are published through a
        # thread-local container rather than as mutable module attributes, so
        # concurrent calls on different threads can never observe each other's
        # state. The last_l6_hidden / last_l8_hidden property bridges below
        # preserve 100% attribute compatibility for existing call sites.
        self._state = CartridgeThreadState()

    # v10.7 P0-A: property bridges onto the thread-local state container.
    # These keep the public attribute names (and therefore every existing
    # call site) working unchanged, while the underlying storage is now
    # strictly per-thread. Assignment and reads can no longer race between
    # the GPU worker thread and the event-loop thread.
    @property
    def last_bus_out(self) -> torch.Tensor | None:
        # v10.7 P2-B: property bridge onto the thread-local state container.
        return self._state.last_bus_out

    @last_bus_out.setter
    def last_bus_out(self, value: torch.Tensor | None) -> None:
        self._state.last_bus_out = value

    @property
    def last_l6_hidden(self) -> torch.Tensor | None:
        return self._state.l6_hidden

    @last_l6_hidden.setter
    def last_l6_hidden(self, value: torch.Tensor | None) -> None:
        self._state.l6_hidden = value

    @property
    def last_l8_hidden(self) -> torch.Tensor | None:
        return self._state.l8_hidden

    @last_l8_hidden.setter
    def last_l8_hidden(self, value: torch.Tensor | None) -> None:
        self._state.l8_hidden = value

    def log_mel(self, wav_16k: torch.Tensor) -> torch.Tensor:
        win = self.window.to(device=wav_16k.device, dtype=wav_16k.dtype)
        fb = self.mel_fb.to(device=wav_16k.device, dtype=wav_16k.dtype)
        spec = torch.stft(
            wav_16k,
            n_fft=N_FFT_16K,
            hop_length=HOP_16K,
            win_length=N_FFT_16K,
            window=win,
            center=True,
            return_complex=True,
        ).abs()
        mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)
        return torch.log(mel.clamp(min=1e-6))

    def forward(
        self, wav_16k: torch.Tensor, source_lang: LanguageCode = "en"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mel = self.log_mel(wav_16k)
        h = self.subsampler(mel).transpose(1, 2)
        lang_idx = torch.full(
            (h.shape[0],), LANG_TO_ID[source_lang], dtype=torch.long, device=h.device
        )
        h = h + self.lang_emb(lang_idx).unsqueeze(1)
        l6_idx = min(5, len(self.blocks) - 1)
        l8_idx = min(7, len(self.blocks) - 1)
        for idx, blk in enumerate(self.blocks):
            h = blk(h)
            if idx == l6_idx:
                self.last_l6_hidden = self.bus_proj(h)
            if idx == l8_idx:
                self.last_l8_hidden = self.bus_proj(h)
        bus = self.bus_proj(h)
        if self.last_l6_hidden is None:
            self.last_l6_hidden = bus
        if self.last_l8_hidden is None:
            self.last_l8_hidden = bus
        r_sps = (1.5 + 7.5 * torch.sigmoid(self.rate_head(bus).mean(dim=1))).squeeze(-1)
        # v10.7 P2-B: publish the bus-projected acoustic output so the
        # multi-teacher ear distiller can compare it against the teacher
        # features. Stored on the thread-local state, not as a shared
        # module attribute.
        self.last_bus_out = bus
        return bus, r_sps
