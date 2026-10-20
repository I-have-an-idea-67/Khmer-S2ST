from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    DOMINANCE_MIN_DB_RATIO,
    ERB_BANDS,
    MIC_EQ_TAPS_16K,
    MIC_EQ_TAPS_24K,
    N_MELS,
    PROD_240M_CONFIG,
    SAMPLE_RATE_IN,
    SILENCE_RMS_THRESHOLD,
    VELVET_RIR_TAPS_24K,
    ModelArchitectureConfig,
    SeparatedSceneSlice,
    VocalMode,
)
from backend.nn_cartridges.universal_core import build_slaney_mel_fb

# Causal separator STFT resolution (locked to 257 bins / 5.86M params)
N_FFT_SEP_16K = 512
HOP_SEP_16K = 160
N_BINS_SEP_16K = 257

# Mel filterbank resolution (locked to 513 bins / 100-rank Slaney)
N_FFT_MEL_16K = 1024
HOP_MEL_16K = 160
N_BINS_MEL_16K = 513


def filter_dominant_speakers(
    stem_a: torch.Tensor,
    stem_b: torch.Tensor,
    min_db_ratio: float = DOMINANCE_MIN_DB_RATIO,
    force_dual_speakers: bool = False,
) -> torch.Tensor:
    a_1d = stem_a.reshape(-1)
    b_1d = stem_b.reshape(-1)
    rms_a = torch.sqrt(torch.mean(a_1d.float() ** 2) + 1e-9)
    rms_b = torch.sqrt(torch.mean(b_1d.float() ** 2) + 1e-9)
    db_diff = 20.0 * torch.log10(rms_b / rms_a)
    if force_dual_speakers or (
        float(db_diff.item()) >= min_db_ratio and float(rms_b.item()) > SILENCE_RMS_THRESHOLD
    ):
        return torch.stack([a_1d, b_1d], dim=0)
    return a_1d.unsqueeze(0)


class Causal3StemSeparator(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.separator_d_model
        self.stft_in = nn.Linear(N_BINS_SEP_16K, d)
        self.film_proj = nn.Linear(512, 2 * d)
        self.gru = nn.GRU(d, d, num_layers=config.separator_layers, batch_first=True)
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
                for _ in range(2)
            ]
        )
        self.mask_head = nn.Linear(d, 3 * N_BINS_SEP_16K)
        self.erb_head = nn.Linear(d, ERB_BANDS)
        self.eq_head = nn.Linear(d, MIC_EQ_TAPS_16K)
        self.rt60_head = nn.Linear(d, 1)
        self.wet_head = nn.Linear(d, 1)
        self.register_buffer("window", torch.hann_window(N_FFT_SEP_16K, device="cpu"), persistent=False)

    def forward(
        self, wav_16k: torch.Tensor, primary_spk_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T_wav = wav_16k.shape
        win = self.window.to(device=wav_16k.device, dtype=wav_16k.dtype)
        spec = torch.stft(
            wav_16k,
            n_fft=N_FFT_SEP_16K,
            hop_length=HOP_SEP_16K,
            win_length=N_FFT_SEP_16K,
            window=win,
            center=True,
            return_complex=True,
        )
        mag = spec.abs().transpose(1, 2)  # (B, T_frames, 257)
        T_frames = mag.shape[1]
        h = self.stft_in(mag)
        if primary_spk_emb is not None:
            gamma, beta = self.film_proj(primary_spk_emb).chunk(2, dim=-1)
            h = h * (1.0 + torch.tanh(gamma.unsqueeze(1))) + beta.unsqueeze(1)
        h, _ = self.gru(h)
        for blk in self.mlp:
            h = h + blk(h)

        masks = (
            torch.sigmoid(self.mask_head(h))
            .view(B, T_frames, 3, N_BINS_SEP_16K)
            .permute(0, 2, 3, 1)
        )
        stems = []
        for s_idx in range(3):
            s_spec = spec * masks[:, s_idx]
            s_wav = torch.istft(
                s_spec,
                n_fft=N_FFT_SEP_16K,
                hop_length=HOP_SEP_16K,
                win_length=N_FFT_SEP_16K,
                window=win,
                center=True,
                length=T_wav,
            )
            stems.append(s_wav)
        stems_tensor = torch.stack(stems, dim=1)  # (B, 3, T_wav)

        erb_mask = torch.sigmoid(self.erb_head(h))  # (B, T_frames, 64)
        h_mean = h.mean(dim=1)
        mic_eq = 0.20 * torch.tanh(self.eq_head(h_mean)).unsqueeze(1)  # (B, 1, 64)
        rt60 = 0.05 + 0.45 * torch.sigmoid(self.rt60_head(h_mean))  # (B, 1)
        wet = 0.35 * torch.sigmoid(self.wet_head(h_mean))  # (B, 1)
        return stems_tensor, erb_mask, mic_eq, rt60, wet


class VocalModeGate(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.mode_gate_d_model
        self.in_proj = nn.Conv1d(N_MELS, d, kernel_size=1)
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d, d, kernel_size=3, padding=1),
                    nn.GroupNorm(8, d),
                    nn.GELU(),
                )
                for _ in range(config.mode_gate_layers)
            ]
        )
        self.cls_head = nn.Linear(d, 4)

    def forward(self, log_mel: torch.Tensor) -> Tuple[torch.Tensor, List[VocalMode]]:
        h = self.in_proj(log_mel)
        for conv in self.conv_layers:
            h = h + conv(h)
        logits = self.cls_head(h.transpose(1, 2))  # (B_spk, T_frames, 4)
        frame_modes = logits.argmax(dim=-1)  # (B_spk, T_frames)
        utt_modes = [VocalMode(int(m.item())) for m in logits.mean(dim=1).argmax(dim=-1)]
        return frame_modes, utt_modes


class SceneAcousticMatcher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        local_gen = torch.Generator(device="cpu").manual_seed(42)
        velvet = torch.zeros(1, VELVET_RIR_TAPS_24K, dtype=torch.float32, device="cpu")
        n_pulses = (VELVET_RIR_TAPS_24K - 24) // 24
        base_grid = torch.arange(1, n_pulses + 1, device="cpu") * 24
        jitter = torch.randint(0, 24, (n_pulses,), generator=local_gen, device="cpu")
        pulse_indices = (base_grid + jitter).clamp(max=VELVET_RIR_TAPS_24K - 1)
        signs = (
            torch.randint(
                0, 2, (1, pulse_indices.numel()), generator=local_gen, device="cpu"
            ).float()
            * 2.0
            - 1.0
        )
        velvet[:, pulse_indices] = signs
        self.register_buffer("velvet_grid", velvet, persistent=False)

    def apply_scene_glue(
        self,
        dry_khmer_wav_24k: torch.Tensor,
        mic_eq_residual_16k: torch.Tensor,
        rt60_sec: torch.Tensor,
        wet_gain: torch.Tensor,
        target_rms_scalar: torch.Tensor,
    ) -> torch.Tensor:
        B, T_len = dry_khmer_wav_24k.shape
        device = dry_khmer_wav_24k.device
        dtype = dry_khmer_wav_24k.dtype

        eq_res_24k = F.interpolate(
            mic_eq_residual_16k.to(dtype),
            size=MIC_EQ_TAPS_24K,
            mode="linear",
            align_corners=False,
        ).squeeze(1)  # (B, 96)
        eq_tail = F.pad(eq_res_24k[:, 1:], (1, 0))
        unit_impulse_96 = F.pad(torch.ones_like(eq_tail[:, :1]), (0, MIC_EQ_TAPS_24K - 1))
        eq_ir = eq_tail + unit_impulse_96
        eq_ir = eq_ir / (torch.norm(eq_ir, p=2, dim=-1, keepdim=True) + 1e-8)

        t_sec = torch.linspace(
            0.0, 0.25, VELVET_RIR_TAPS_24K, device=device, dtype=dtype
        ).unsqueeze(0)
        safe_rt60 = torch.clamp(rt60_sec.to(dtype), min=0.05)  # (B, 1)
        decay = torch.exp((-6.9078 / safe_rt60) * t_sec)  # (B, 6000)
        velvet = self.velvet_grid.to(device=device, dtype=dtype) * decay
        velvet_norm = velvet / (torch.norm(velvet, p=2, dim=-1, keepdim=True) + 1e-8)
        unit_impulse_6000 = F.pad(
            torch.ones_like(velvet_norm[:, :1]), (0, VELVET_RIR_TAPS_24K - 1)
        )
        rir = unit_impulse_6000 + torch.clamp(wet_gain.to(dtype), 0.0, 0.35) * velvet_norm

        n_fft = 1 << (T_len + MIC_EQ_TAPS_24K + VELVET_RIR_TAPS_24K - 2).bit_length()
        W_fft = torch.fft.rfft(dry_khmer_wav_24k, n=n_fft, dim=-1)
        EQ_fft = torch.fft.rfft(eq_ir, n=n_fft, dim=-1)
        RIR_fft = torch.fft.rfft(rir, n=n_fft, dim=-1)
        wet_wav = torch.fft.irfft(W_fft * EQ_fft * RIR_fft, n=n_fft, dim=-1)[:, :T_len]

        cur_rms = torch.sqrt(torch.mean(wet_wav ** 2, dim=-1, keepdim=True) + 1e-8)
        gain = torch.clamp(target_rms_scalar.to(dtype) / cur_rms, min=0.2, max=4.0)
        return torch.clamp(wet_wav * gain, min=-0.98, max=0.98)


class StemAndScenePipeline(nn.Module):
    def __init__(
        self, config: ModelArchitectureConfig = PROD_240M_CONFIG, n_mels: int = N_MELS
    ) -> None:
        super().__init__()
        self.config = config
        self.n_mels = n_mels
        self.separator = Causal3StemSeparator(config)
        self.mode_gate = VocalModeGate(config)
        self.scene_matcher = SceneAcousticMatcher()
        self.register_buffer("window_mel", torch.hann_window(N_FFT_MEL_16K, device="cpu"), persistent=False)
        self.register_buffer(
            "mel_fb",
            build_slaney_mel_fb(N_BINS_MEL_16K, n_mels, sample_rate=SAMPLE_RATE_IN),
            persistent=False,
        )

    def _compute_log_mel(self, wav_16k: torch.Tensor) -> torch.Tensor:
        win = self.window_mel.to(device=wav_16k.device, dtype=wav_16k.dtype)
        fb = self.mel_fb.to(device=wav_16k.device, dtype=wav_16k.dtype)
        spec = torch.stft(
            wav_16k,
            n_fft=N_FFT_MEL_16K,
            hop_length=HOP_MEL_16K,
            win_length=N_FFT_MEL_16K,
            window=win,
            center=True,
            return_complex=True,
        ).abs()
        mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)
        return torch.log(mel.clamp(min=1e-6))

    def separate_and_classify(
        self,
        wav_16k: torch.Tensor,
        utterance_id: int = 0,
        utterance_start_ms: float = 0.0,
        trailing_pause_ms: float = 0.0,
        primary_spk_emb: Optional[torch.Tensor] = None,
        force_dual_speakers: bool = False,
    ) -> SeparatedSceneSlice:
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)
        stems, erb_mask, mic_eq, rt60, wet = self.separator(wav_16k, primary_spk_emb)
        active_stems = filter_dominant_speakers(
            stems[0, 0], stems[0, 1], force_dual_speakers=force_dual_speakers
        )  # (B_spk, T_wav)
        B_spk = active_stems.shape[0]
        mel_stems = self._compute_log_mel(active_stems)  # (B_spk, 100, T_frames)
        frame_modes, dom_modes = self.mode_gate(mel_stems)
        rms_scalar = torch.sqrt(torch.mean(active_stems ** 2, dim=-1, keepdim=True) + 1e-8)

        return SeparatedSceneSlice(
            utterance_id=utterance_id,
            utterance_start_ms=utterance_start_ms,
            frame_vocal_modes=frame_modes,
            dominant_vocal_modes=dom_modes,
            active_speaker_stems=active_stems,
            speaker_ids=list(range(B_spk)),
            mic_eq_residual_16k=mic_eq.expand(B_spk, -1, -1),
            rt60_sec=rt60.expand(B_spk, -1),
            wet_gain=wet.expand(B_spk, -1),
            rms_scalar=rms_scalar,
            erb_suppression_mask=erb_mask[0],  # (T_frames, 64)
            trailing_pause_ms=trailing_pause_ms,
        )

    def apply_scene_glue(
        self,
        dry_khmer_wav_24k: torch.Tensor,
        mic_eq_residual_16k: torch.Tensor,
        rt60_sec: torch.Tensor,
        wet_gain: torch.Tensor,
        target_rms_scalar: torch.Tensor,
    ) -> torch.Tensor:
        return self.scene_matcher.apply_scene_glue(
            dry_khmer_wav_24k, mic_eq_residual_16k, rt60_sec, wet_gain, target_rms_scalar
        )
