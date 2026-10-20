from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    MINOR_SYLLABLE_MAX_FRAMES,
    HOP_LENGTH,
    LANG_TO_ID,
    N_FFT_24K,
    N_MELS,
    PROD_240M_CONFIG,
    SAMPLE_RATE_OUT,
    LanguageCode,
    ModelArchitectureConfig,
    PrecomputedEarFeatures,
    SyllableTensorBatch,
)
from backend.nn_cartridges.codeswitch_router import AcousticPointerCodeSwitchBridge
from backend.nn_cartridges.duration_aligner import LearnedDurationAligner
from backend.nn_cartridges.ear_cartridge import StreamingEarCartridge
from backend.nn_cartridges.mouth_cartridge import KhmerMouthCartridge
from backend.nn_cartridges.universal_core import (
    DisentangledEmotionEncoder,
    VocosISTFTHead,
    build_slaney_mel_fb,
)

N_BINS_24K = N_FFT_24K // 2 + 1  # 1025 bins -> guarantees rank 100


# v10.7 P1-C: the Sway coefficients are shared by the inference ODE and the
# training shortcut-distillation loss, so the trained step grid and the
# inference step grid are provably identical.
SWAY_COEF: float = -0.8
SC_DISTILL_STEPS: int = 12


def sway_time_schedule(
    steps: int, sway_coef: float = -0.8, device=None, dtype=torch.float32
) -> torch.Tensor:
    """
    v10.7 P1-C: the Sway time schedule, extracted so that the training
    shortcut-distillation intervals and the inference ODE use provably the
    SAME step grid.

    v10.6 hardcoded `dt_val = 0.125` in the distillation loss while the
    inference ODE produced steps spanning 0.0235 to 0.1211 at steps=12,
    sway_coef=-0.8. The trained step size was therefore never one the
    sampler actually visited. Sampling the training intervals from this very
    schedule removes the train/inference mismatch.
    """
    steps = max(1, int(steps))
    u = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
    return (u + sway_coef * (torch.cos(0.5 * math.pi * u) - 1.0 + u)).clamp(0.0, 1.0).to(dtype)


def integrate_shortcut_sway_ode(
    mouth: KhmerMouthCartridge,
    x_0: torch.Tensor,
    cond: torch.Tensor,
    emo_tokens: torch.Tensor,
    steps: int = 4,
    sway_coef: float = -0.8,
    # v10.7 P1-C: Classifier-Free Guidance scale. 1.0 disables guidance and
    # reproduces the v10.6 behaviour exactly. Training now applies the same
    # 15 percent conditioning dropout that makes the unconditional branch
    # learnable, so guidance is available at inference.
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    B = x_0.shape[0]
    device, dtype = x_0.device, x_0.dtype
    steps = max(1, int(steps))
    # v10.7 P1-C: shared with the training distillation loss, so the trained
    # step grid and the inference step grid are provably identical.
    t_schedule = sway_time_schedule(steps, sway_coef, device=device, dtype=dtype)

    x = x_0
    for k in range(steps):
        t_cur = t_schedule[k]
        dt_step = t_schedule[k + 1] - t_cur
        t_batch = torch.full((B,), float(t_cur.item()), device=device, dtype=dtype)
        dt_batch = torch.full((B,), float(dt_step.item()), device=device, dtype=dtype)
        # v10.7: the conditional velocity is computed UNCONDITIONALLY first.
        # v10.6 computed it only in the else branch and then referenced it
        # in the if branch, so any cfg_scale != 1.0 raised
        # UnboundLocalError: cannot access local variable 'v'.
        v_cond = mouth.predict_velocity(x, t_batch, cond, emo_tokens, dt=dt_batch)
        if cfg_scale != 1.0:
            v_uncond = mouth.predict_velocity(
                x, t_batch, torch.zeros_like(cond), torch.zeros_like(emo_tokens), dt=dt_batch
            )
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        x = x + dt_step * v
    return x


def compute_shortcut_distillation_loss(
    mouth: KhmerMouthCartridge,
    x_1: torch.Tensor,
    cond: torch.Tensor,
    emo_tokens: torch.Tensor,
    sc_ratio: float = 0.25,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    B = x_1.shape[0]
    device, dtype = x_1.device, x_1.dtype
    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, device=device, dtype=dtype)
    t_view = t.view(B, 1, 1)
    x_t = (1.0 - (1.0 - sigma_min) * t_view) * x_0 + t_view * x_1
    target_v = x_1 - (1.0 - sigma_min) * x_0

    # 1. Flow2GAN spectral energy-adaptive instantaneous OT-CFM velocity regression (dt = 0)
    pred_v = mouth.predict_velocity(x_t, t, cond, emo_tokens, dt=None)
    frame_energy = x_1.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=0.1)
    energy_weight = (1.0 / (frame_energy ** 0.3)).detach()
    l_fm = ((pred_v - target_v).pow(2) * energy_weight).mean()

    # 2. Step-conditioned Shortcut Self-Consistency, grounded in the Sway
    #    schedule (v10.7 P1-C).
    #
    #    v10.6 hardcoded dt_val = 0.125 and the teacher half-steps were each
    #    evaluated at that same dt, so the whole hierarchy was anchored to
    #    NOTHING: the two half-steps and the full step all used the same
    #    step-size conditioning, and the trained dt was never one the Sway ODE
    #    actually visits at inference (0.0235..0.1211 at steps=12).
    #
    #    v10.7 samples the coarse-to-fine step pairs DIRECTLY from the Sway
    #    schedule, so the training grid and the inference grid are provably
    #    identical. The two finest teacher sub-steps are evaluated with the
    #    instantaneous velocity network (dt=None, i.e. dt = 0.0), which
    #    anchors the entire hierarchy to the ground-truth OT-CFM vector field
    #    and cannot collapse to zero.
    B_sc = max(1, int(round(B * sc_ratio)))
    x_t_sc = x_t[:B_sc]
    cond_sc = cond[:B_sc]
    emo_sc = emo_tokens[:B_sc]

    # The Sway step grid, shared with integrate_shortcut_sway_ode.
    sched = sway_time_schedule(SC_DISTILL_STEPS, SWAY_COEF, device=device, dtype=dtype)
    dts = (sched[1:] - sched[:-1]).clamp(min=0.0)

    # Pick a random coarse step from the grid, then split it into two fine
    # sub-steps of equal length: dt_1 = dt_2 = dt_full / 2.
    idx = int(torch.randint(0, dts.numel(), (1,)).item())
    dt_full_val = float(dts[idx].item())
    dt_1 = dt_full_val * 0.5
    dt_2 = dt_full_val * 0.5
    t_cur_val = float(sched[idx].item())
    t_sc = torch.full((B_sc,), t_cur_val, device=device, dtype=dtype)

    with torch.no_grad():
        # The two FINEST teacher sub-steps use the instantaneous velocity
        # network (dt=None), which is the ground-truth anchor.
        v_step1 = mouth.predict_velocity(x_t_sc, t_sc, cond_sc, emo_sc, dt=None)
        x_mid = x_t_sc + dt_1 * v_step1
        v_step2 = mouth.predict_velocity(x_mid, t_sc + dt_1, cond_sc, emo_sc, dt=None)
        # Step-weighted average: (dt_1*v1 + dt_2*v2) / dt_full.
        v_sc_target = (dt_1 * v_step1 + dt_2 * v_step2) / max(dt_full_val, 1e-8)

    # The student predicts the FULL step, conditioned on the full dt.
    dt_full_t = torch.full((B_sc,), dt_full_val, device=device, dtype=dtype)
    v_sc_pred = mouth.predict_velocity(x_t_sc, t_sc, cond_sc, emo_sc, dt=dt_full_t)
    l_sc = F.mse_loss(v_sc_pred, v_sc_target)
    return l_fm + 0.5 * l_sc


class SymmetricS2STConsole(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.ear = StreamingEarCartridge(config)
        self.emotion_encoder = DisentangledEmotionEncoder(config)
        self.pointer = AcousticPointerCodeSwitchBridge(config)
        self.mouth = KhmerMouthCartridge(config)
        self.vocos = VocosISTFTHead(config)
        self.register_buffer("window_24k", torch.hann_window(N_FFT_24K, device="cpu"), persistent=False)
        self.register_buffer(
            "mel_fb_24k",
            build_slaney_mel_fb(N_BINS_24K, N_MELS, sample_rate=SAMPLE_RATE_OUT),
            persistent=False,
        )

    def _extract_mel_24k(self, wav_24k: torch.Tensor) -> torch.Tensor:
        win = self.window_24k.to(device=wav_24k.device, dtype=wav_24k.dtype)
        fb = self.mel_fb_24k.to(device=wav_24k.device, dtype=wav_24k.dtype)
        spec = torch.stft(
            wav_24k,
            n_fft=N_FFT_24K,
            hop_length=HOP_LENGTH,
            win_length=N_FFT_24K,
            window=win,
            center=True,
            return_complex=True,
        ).abs()
        mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)
        return torch.log(mel.clamp(min=1e-6))

    def forward_s2st(
        self,
        source_wav_16k: torch.Tensor,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        orig_duration_sec: float,
        trailing_pause_ms: float = 0.0,
        playback_rate: float = 1.0,
        explicit_syll_batch: Optional[SyllableTensorBatch] = None,
        past_clause_kv: Optional[torch.Tensor] = None,
        past_gate_kv: Optional[torch.Tensor] = None,
        deferred_syll_count: int = 0,
        is_hard_ceiling_slice: bool = False,
        # v10.7 P1-D: optional speaker-timbre vector, shape (B, 1, SEMANTIC_DIM).
        # When None it is derived from the layer-6 acoustic representation.
        speaker_timbre: Optional[torch.Tensor] = None,
        # v10.7 AUD-107-02: reuse an ear forward pass the caller already did.
        precomputed_ear: Optional[PrecomputedEarFeatures] = None,
    ) -> Tuple[torch.Tensor, SyllableTensorBatch, torch.Tensor]:
        # v10.7 AUD-107-02: the ear runs ONCE per audio slice.
        #
        # The full-duplex runner needs the layer-6 tap and the log-mel before it
        # can call forward_s2st, so it computed them itself. v10.7 then ran the
        # ear a SECOND time on the identical audio inside this method, which
        # doubled the 64.5M-parameter ear and the 8.7M-parameter emotion encoder
        # on every spoken utterance. When the caller supplies the bundle, the
        # ear and the log-mel are reused verbatim.
        if precomputed_ear is not None:
            bus = precomputed_ear.bus
            r_sps = precomputed_ear.r_sps
            mel_stems = precomputed_ear.mel_stems
            h_acoustic_l6 = precomputed_ear.h_acoustic_l6
        else:
            bus, r_sps = self.ear(source_wav_16k, source_lang)
            h_acoustic_l6 = (
                self.ear.last_l6_hidden if self.ear.last_l6_hidden is not None else bus
            )
            mel_stems = self.ear.log_mel(source_wav_16k)
        emo_tokens, _ = self.emotion_encoder(mel_stems)
        fused_bus, gate_g_t = self.pointer(bus, h_acoustic_l6)
        # v10.7 P1-D: when no explicit speaker timbre is supplied, derive it
        # from the L2-normalized mean-pooled layer-6 acoustic representation.
        # This preserves the speaker's vocal-tract identity in the acoustic
        # conditioning path, which F.instance_norm in the emotion encoder
        # removes. ZERO parameters.
        if speaker_timbre is None:
            speaker_timbre = F.normalize(
                h_acoustic_l6.mean(dim=1, keepdim=True), p=2, dim=-1
            )
        target_frames = LearnedDurationAligner.compute_elastic_target_frames(
            orig_duration_sec,
            trailing_pause_ms,
            float(r_sps.mean().item()),
            playback_rate,
        )
        cond, syll_batch = self.mouth.predict_syllables_and_align(
            fused_bus,
            gate_g_t,
            r_sps,
            target_lang,
            orig_duration_sec,
            target_frames,
            explicit_syll_batch=explicit_syll_batch,
            past_clause_kv=past_clause_kv,
            past_gate_kv=past_gate_kv,
            deferred_syll_count=deferred_syll_count,
            is_hard_ceiling_slice=is_hard_ceiling_slice,
            speaker_timbre=speaker_timbre,
        )
        B = source_wav_16k.shape[0]
        x_0 = torch.randn(B, N_MELS, target_frames, device=source_wav_16k.device, dtype=cond.dtype)
        x = integrate_shortcut_sway_ode(
            self.mouth,
            x_0,
            cond,
            emo_tokens,
            steps=max(1, self.config.euler_steps),
            sway_coef=SWAY_COEF,
        )
        wav_24k = self.vocos(x)
        return wav_24k, syll_batch, gate_g_t

    def compute_s2st_training_losses(
        self,
        source_wav_16k: torch.Tensor,
        target_wav_24k: torch.Tensor,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        target_syll_batch: SyllableTensorBatch,
        target_lang_id: torch.Tensor,
        cfg_drop_prob: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        x_1 = self._extract_mel_24k(target_wav_24k)
        B, _, T_mel = x_1.shape

        bus, r_sps = self.ear(source_wav_16k, source_lang)
        h_acoustic_l6 = self.ear.last_l6_hidden if self.ear.last_l6_hidden is not None else bus
        emo_tokens, lang_logits = self.emotion_encoder(self.ear.log_mel(source_wav_16k), alpha=1.0)
        fused_bus, gate_g_t = self.pointer(bus, h_acoustic_l6)

        L_syll = target_syll_batch.onset_ids.shape[1]
        lang_idx = torch.full((B,), LANG_TO_ID[target_lang], dtype=torch.long, device=x_1.device)
        bus_cond = fused_bus + self.mouth.target_lang_emb(lang_idx).unsqueeze(1)
        pos_q = self.mouth._build_sinusoidal_queries(B, L_syll, x_1.device, bus_cond.dtype)
        reordered, attn_w = self.mouth.reorder_attn(pos_q, bus_cond, bus_cond, need_weights=True)
        bus_l = self.mouth.reorder_norm(pos_q + reordered)
        gate_l = torch.bmm(attn_w, gate_g_t).squeeze(-1).clamp(1e-5, 1.0 - 1e-5)

        l_syll_ce = (
            F.cross_entropy(self.mouth.heads["onset"](bus_l).transpose(1, 2), target_syll_batch.onset_ids)
            + F.cross_entropy(self.mouth.heads["coeng"](bus_l).transpose(1, 2), target_syll_batch.coeng_ids)
            + F.cross_entropy(self.mouth.heads["medial"](bus_l).transpose(1, 2), target_syll_batch.medial_ids)
            + F.cross_entropy(self.mouth.heads["register"](bus_l).transpose(1, 2), target_syll_batch.register_ids)
            + F.cross_entropy(self.mouth.heads["vowel"](bus_l).transpose(1, 2), target_syll_batch.vowel_ids)
            + F.cross_entropy(self.mouth.heads["coda"](bus_l).transpose(1, 2), target_syll_batch.coda_ids)
            + F.cross_entropy(self.mouth.heads["sesqui"](bus_l).transpose(1, 2), target_syll_batch.sesqui_ids)
        ) / 7.0

        # Code-Switch Gate BCE supervision against Register-3 ground truth
        cs_target = (target_syll_batch.register_ids == 3).to(gate_l.dtype)
        l_gate_bce = F.binary_cross_entropy(gate_l, cs_target)

        cond, _ = self.mouth.predict_syllables_and_align(
            fused_bus,
            gate_g_t,
            r_sps,
            target_lang,
            orig_duration_sec=T_mel * HOP_LENGTH / float(SAMPLE_RATE_OUT),
            target_frames=T_mel,
            explicit_syll_batch=target_syll_batch,
        )

        # Duration prior supervision
        pred_durs = self.mouth.last_pred_durations
        # v10.7 P1-C: NON-CIRCULAR duration supervision. v10.6 supervised
        # pred_durs against a static prior built from the sesqui class alone
        # (0.25 / 1.00), which carries no acoustic information: the target was
        # a function of the label distribution, not of the audio. The target is
        # now derived from the monotonic attention mass of reorder_attn,
        # attn_w.sum(dim=-1) over the L_syll axis, scaled to T_mel and clamped
        # by the 6-frame minor-syllable cap.
        attn_mass = attn_w.sum(dim=-1)  # (B, L_syll) attention mass per syllable
        attn_mass = attn_mass.clamp(min=0.0)
        mass_total = attn_mass.sum(dim=-1, keepdim=True)
        dur_from_attn = attn_mass * (float(T_mel) / (mass_total + 1e-8))
        # Respect the minor-syllable cap: sesqui_id == 1 syllables may not
        # exceed MINOR_SYLLABLE_MAX_FRAMES; the excess is redistributed to
        # the major syllables exactly as LearnedDurationAligner does.
        is_minor_s = target_syll_batch.sesqui_ids == 1
        capped = torch.where(
            is_minor_s,
            torch.clamp(dur_from_attn, max=MINOR_SYLLABLE_MAX_FRAMES),
            dur_from_attn,
        )
        excess = (dur_from_attn - capped).sum(dim=-1, keepdim=True)
        major_w = capped * (~is_minor_s).to(capped.dtype)
        major_sum = major_w.sum(dim=-1, keepdim=True)
        prior_durs = torch.where(
            major_sum > 1e-6,
            capped + excess * (major_w / (major_sum + 1e-8)),
            capped,
        )
        l_dur = F.smooth_l1_loss(pred_durs, prior_durs) if pred_durs is not None else torch.tensor(0.0, device=x_1.device)

        # 15% Classifier-Free Guidance (CFG) conditioning dropout
        # v10.7 P1-C: v10.6 trained this dropout but NEVER applied guidance at
        # inference, so the unconditional branch was learned and then thrown
        # away. integrate_shortcut_sway_ode now evaluates the unconditional
        # branch (zero cond AND zero emo_tokens) and blends it with the
        # conditional one when cfg_scale != 1.0. The dropout below must
        # therefore zero EXACTLY the same two tensors the inference
        # unconditional branch zeroes, otherwise the guided direction points
        # at a branch the model was never trained on.
        if self.training and cfg_drop_prob > 0.0:
            drop_mask = (torch.rand(B, 1, 1, device=x_1.device) >= cfg_drop_prob).to(cond.dtype)
            cond = cond * drop_mask
            emo_tokens = emo_tokens * drop_mask

        l_cfm = compute_shortcut_distillation_loss(self.mouth, x_1, cond, emo_tokens)
        l_lang_grl = F.cross_entropy(lang_logits, target_lang_id)
        total = l_cfm + 1.0 * l_syll_ce + 0.5 * l_dur + 0.3 * l_gate_bce + 0.2 * l_lang_grl
        return {
            "total": total,
            "l_cfm": l_cfm,
            "l_syll_ce": l_syll_ce,
            "l_dur": l_dur,
            "l_gate_bce": l_gate_bce,
            "l_lang_grl": l_lang_grl,
        }

    def compute_flow_loss(
        self,
        target_wav_24k: torch.Tensor,
        syll_batch: Optional[SyllableTensorBatch] = None,
        infill_mode: bool = False,
        source_wav_16k: Optional[torch.Tensor] = None,
        source_lang: LanguageCode = "en",
    ) -> torch.Tensor:
        x_1 = self._extract_mel_24k(target_wav_24k)
        B, _, T_mel = x_1.shape
        emo_tokens, _ = self.emotion_encoder(x_1)

        if infill_mode or syll_batch is None:
            mask = torch.zeros_like(x_1)
            s = T_mel // 3
            e = max(s + 1, (2 * T_mel) // 3)
            mask[:, :, s:e] = 1.0
            cond = self.mouth.infill_proj(((1.0 - mask) * x_1).transpose(1, 2))
        else:
            if source_wav_16k is None:
                src_proxy = F.interpolate(
                    target_wav_24k.unsqueeze(1),
                    size=max(160, int(target_wav_24k.shape[-1] * 16000 / 24000)),
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            else:
                src_proxy = source_wav_16k

            bus, r_sps = self.ear(src_proxy, source_lang)
            h_acoustic_l6 = self.ear.last_l6_hidden if self.ear.last_l6_hidden is not None else bus
            fused_bus, gate_g_t = self.pointer(bus, h_acoustic_l6)
            cond, _ = self.mouth.predict_syllables_and_align(
                fused_bus, gate_g_t, r_sps, "km", 1.0, T_mel, syll_batch
            )

        return compute_shortcut_distillation_loss(self.mouth, x_1, cond, emo_tokens)
