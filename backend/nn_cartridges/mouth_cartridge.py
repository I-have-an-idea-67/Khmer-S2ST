from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.contracts.schemas import (
    CartridgeThreadState,
    LANG_TO_ID,
    N_MELS,
    PROD_240M_CONFIG,
    SEMANTIC_DIM,
    LanguageCode,
    ModelArchitectureConfig,
    SyllableTensorBatch,
)
from backend.nn_cartridges.duration_aligner import LearnedDurationAligner

def sinusoidal_time_embedding(val: torch.Tensor, dim: int) -> torch.Tensor:
    """
    v10.7 P1-C: parameter-free sinusoidal Fourier embedding of a continuous
    scalar (or a batch of scalars) into a `dim`-wide vector.

    The v10.6 code did `t.view(B, 1).expand(B, d_half)`, i.e. it repeated one
    rank-1 scalar across every column. `time_mlp` therefore received a vector
    that carried only two degrees of freedom (w1*t + w2*dt + b) and had no
    spectral resolution over the timestep at all.

    Here the value is embedded on log-spaced frequencies
    exp(-ln(10000) * i / (dim/2)), half sine and half cosine, which is the
    standard transformer positional encoding applied to a continuous value.

    Parameters:
        val: (B,) or (B, 1) tensor of values to embed.
        dim: output width. Must be even.

    Returns:
        (B, dim) tensor. Adds NO parameters.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal_time_embedding requires an even dim, got {dim}")
    v = val.reshape(-1, 1).to(dtype=torch.float32)
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=v.device, dtype=torch.float32) / half
    )
    ang = v * freq.unsqueeze(0)
    out = torch.zeros(v.shape[0], dim, device=v.device, dtype=torch.float32)
    out[:, 0::2] = torch.sin(ang)
    out[:, 1::2] = torch.cos(ang)
    return out


SLOT_VOCAB = {
    "onset": 64,
    "coeng": 64,
    "medial": 4,
    "register": 4,
    "vowel": 48,
    "coda": 32,
    "sesqui": 3,
}


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, d))
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d))

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor, emo: torch.Tensor) -> torch.Tensor:
        mod = (self.scale_shift_table.unsqueeze(0) + t_mod).unbind(dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            m.unsqueeze(1) for m in mod
        ]
        h = self.n1(x) * (1.0 + scale_msa) + shift_msa
        sa_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + (1.0 + gate_msa) * sa_out
        ca_out, _ = self.cross_attn(self.n2(x), emo, emo, need_weights=False)
        x = x + ca_out
        h_ffn = self.n3(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + (1.0 + gate_mlp) * self.ffn(h_ffn)
        return x


class KhmerMouthCartridge(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.mouth_d_model
        self.config = config

        # Non-Monotonic Cross-Attention Reordering Bridge (+1.05M params)
        self.reorder_attn = nn.MultiheadAttention(SEMANTIC_DIM, 8, batch_first=True)
        self.reorder_norm = nn.LayerNorm(SEMANTIC_DIM)

        self.heads = nn.ModuleDict(
            {k: nn.Linear(SEMANTIC_DIM, v) for k, v in SLOT_VOCAB.items()}
        )
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(v, SEMANTIC_DIM) for k, v in SLOT_VOCAB.items()}
        )
        self.target_lang_emb = nn.Embedding(4, SEMANTIC_DIM)
        self.duration_aligner = LearnedDurationAligner(SEMANTIC_DIM)
        self.cond_proj = nn.Linear(SEMANTIC_DIM, d)
        self.emo_proj = nn.Linear(SEMANTIC_DIM, d)
        self.mel_in = nn.Linear(N_MELS, d)
        self.infill_proj = nn.Linear(N_MELS, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(d, config.mouth_heads, config.mouth_ffn_dim)
                for _ in range(config.mouth_layers)
            ]
        )
        self.mel_out = nn.Linear(d, N_MELS)

        # Rolling Cross-Clause State & Training Telemetry Taps
        # v10.7 P0-A: the per-call telemetry is published through a thread-local
        # container rather than as mutable module attributes. The property
        # bridges below preserve the original attribute names, so every
        # existing call site keeps working unchanged while concurrent calls on
        # different threads can no longer observe each other's state.
        self._state = CartridgeThreadState()

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_next_past_kv(self) -> Optional[torch.Tensor]:
        return self._state.last_next_past_kv

    @last_next_past_kv.setter
    def last_next_past_kv(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_next_past_kv = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_next_past_gate(self) -> Optional[torch.Tensor]:
        return self._state.last_next_past_gate

    @last_next_past_gate.setter
    def last_next_past_gate(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_next_past_gate = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_deferred_syll_count(self) -> int:
        return self._state.last_deferred_syll_count

    @last_deferred_syll_count.setter
    def last_deferred_syll_count(self, value: int) -> None:
        self._state.last_deferred_syll_count = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_pred_durations(self) -> Optional[torch.Tensor]:
        return self._state.last_pred_durations

    @last_pred_durations.setter
    def last_pred_durations(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_pred_durations = value

    # v10.7 P0-A: property bridge onto the thread-local state container.
    @property
    def last_gate_l(self) -> Optional[torch.Tensor]:
        return self._state.last_gate_l

    @last_gate_l.setter
    def last_gate_l(self, value: Optional[torch.Tensor]) -> None:
        self._state.last_gate_l = value

    def _build_sinusoidal_queries(
        self, B: int, L_syll: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if L_syll > 1:
            pos = torch.linspace(0.0, 100.0, L_syll, device=device, dtype=torch.float32).unsqueeze(1)
        else:
            pos = torch.zeros(1, 1, device=device, dtype=torch.float32)
        div = torch.exp(
            torch.arange(0, SEMANTIC_DIM, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / SEMANTIC_DIM)
        )
        pe = torch.zeros(1, L_syll, SEMANTIC_DIM, device=device, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe.to(dtype=dtype).expand(B, -1, -1)

    def predict_syllables_and_align(
        self,
        fused_semantic_bus: torch.Tensor,
        gate_g_t: torch.Tensor,
        r_sps: torch.Tensor,
        target_lang: LanguageCode,
        orig_duration_sec: float,
        target_frames: int,
        explicit_syll_batch: Optional[SyllableTensorBatch] = None,
        past_clause_kv: Optional[torch.Tensor] = None,
        past_gate_kv: Optional[torch.Tensor] = None,
        deferred_syll_count: int = 0,
        is_hard_ceiling_slice: bool = False,
        entropy_wait_threshold: float = 1.85,
        # v10.7 P1-D: optional speaker-timbre vector, shape (B, 1, SEMANTIC_DIM).
        # Added to the aligned hidden state before cond_proj; no parameters.
        speaker_timbre: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SyllableTensorBatch]:
        B, T_cur, D = fused_semantic_bus.shape
        device = fused_semantic_bus.device
        lang_idx = torch.full((B,), LANG_TO_ID[target_lang], dtype=torch.long, device=device)
        bus_cond = fused_semantic_bus + self.target_lang_emb(lang_idx).unsqueeze(1)

        # 1. Prepend rolling past-clause KV cache with parameter-free negative sinusoidal offset
        # v10.7 P0-A: guard the B_spk transition. When the number of active
        # speakers changes between clauses (1 -> 2 or 2 -> 1) the cached KV
        # has a different batch dimension, and torch.cat on dim=1 would raise a
        # RuntimeError on the mismatched dim=0. The stale cache is discarded
        # instead: a KV cache built for a different speaker set carries no
        # valid context for this one.
        if past_clause_kv is not None and past_clause_kv.shape[0] != B:
            past_clause_kv = None
            past_gate_kv = None
        if past_clause_kv is not None and past_clause_kv.shape[1] > 0:
            T_past = past_clause_kv.shape[1]
            past_pe = -self._build_sinusoidal_queries(B, T_past, device, bus_cond.dtype)
            kv_mem = torch.cat([past_clause_kv.to(bus_cond.dtype) + past_pe, bus_cond], dim=1)
            gate_mem = (
                torch.cat([past_gate_kv.to(gate_g_t.dtype), gate_g_t], dim=1)
                if past_gate_kv is not None and past_gate_kv.shape[1] == T_past
                else F.pad(gate_g_t, (0, 0, T_past, 0))
            )
        else:
            kv_mem = bus_cond
            gate_mem = gate_g_t

        # 2. Cache trailing 20 frames (800ms) if sliced by Trigger #2 (1.60s hard ceiling)
        if is_hard_ceiling_slice:
            keep_frames = min(20, T_cur)
            self.last_next_past_kv = bus_cond[:, -keep_frames:, :].detach()
            self.last_next_past_gate = gate_g_t[:, -keep_frames:, :].detach()
        else:
            self.last_next_past_kv = None
            self.last_next_past_gate = None

        if explicit_syll_batch is None:
            mean_r_sps = float(r_sps.mean().item())
            gamma_contract = 0.78 if mean_r_sps > 5.5 else 1.00
            base_syll = max(1, int(round(orig_duration_sec * min(mean_r_sps, 6.5) * gamma_contract)))
            L_syll = base_syll + max(0, int(deferred_syll_count))

            pos_q = self._build_sinusoidal_queries(B, L_syll, device, bus_cond.dtype)
            reordered_bus, attn_w = self.reorder_attn(
                query=pos_q, key=kv_mem, value=kv_mem, need_weights=True
            )
            bus_l = self.reorder_norm(pos_q + reordered_bus)
            gate_l = torch.bmm(attn_w, gate_mem)
            self.last_gate_l = gate_l

            onset_logits = self.heads["onset"](bus_l)
            # 3. Hibiki-style predictability / entropy wait-gating on right boundary of Trigger #2 slices
            self.last_deferred_syll_count = 0
            # v10.7: wait_mask is hoisted so it can also be applied to the
            # ARGMAX result below, after the head projections.
            wait_mask = None
            if is_hard_ceiling_slice and L_syll > 3:
                tail_probs = F.softmax(onset_logits[:, -3:, :], dim=-1)
                tail_entropy = -(tail_probs * torch.log(tail_probs.clamp(min=1e-8))).sum(dim=-1)
                wait_mask = tail_entropy > entropy_wait_threshold
                # v10.7 P0-A: wait_mask has shape (B, 3). Indexing [0] counted
                # only the first speaker and silently ignored speakers 1..B-1.
                # Sum per speaker and take the maximum, so the conservative
                # (largest) deferral count is used for the whole batch.
                self.last_deferred_syll_count = int(
                    wait_mask.float().sum(dim=-1).max().item()
                )
                bus_l = torch.cat(
                    [
                        bus_l[:, :-3, :],
                        torch.where(wait_mask.unsqueeze(-1), torch.zeros_like(bus_l[:, -3:, :]), bus_l[:, -3:, :]),
                    ],
                    dim=1,
                )
                onset_logits = self.heads["onset"](bus_l)

            onset_ids = onset_logits.argmax(dim=-1)
            coeng_ids = self.heads["coeng"](bus_l).argmax(dim=-1)
            medial_ids = self.heads["medial"](bus_l).argmax(dim=-1)
            vowel_ids = self.heads["vowel"](bus_l).argmax(dim=-1)
            coda_ids = self.heads["coda"](bus_l).argmax(dim=-1)

            # v10.7: zero the DEFERRED tail syllables AFTER the argmax.
            #
            # Zeroing bus_l before the projection is not sufficient:
            # nn.Linear has a bias vector, so a zero input still produces a
            # non-zero logit vector and the argmax returns a real onset id.
            # The decoder skips a syllable only when the triplet is exactly
            # (0, 0, 0), so the mask must be applied to the ids themselves.
            if target_lang == "en":
                # v10.7 P1-B: a SYNTHESIZED English target syllable must land
                # inside the reserved English ranges, otherwise
                # decode_batch_to_text never reaches ENGLISH_SYLLABLE_TABLE
                # and falls through to the CVC pseudoword fallback, which
                # populates cs_words with synthetic brands.
                #
                #   onset : 34 + argmax over the 24 reserved onset logits
                #   coeng : 0 (the synthesized-syllable sentinel)
                #   vowel : 28 + argmax over the 17 reserved vowel logits
                #   coda  : 18 + argmax over the 14 reserved coda logits
                #
                # The ranges are disjoint from the Khmer ranges: onset
                # 1..33 / vowel 0..24 / coda 0..17.
                onset_ids = 34 + onset_logits[:, :, 34:58].argmax(dim=-1)
                coeng_ids = torch.zeros((B, L_syll), dtype=torch.long, device=device)
                medial_ids = self.heads["medial"](bus_l).argmax(dim=-1).clamp(0, 3)
                vowel_ids = 28 + self.heads["vowel"](bus_l)[:, :, 28:45].argmax(dim=-1)
                coda_ids = 18 + self.heads["coda"](bus_l)[:, :, 18:32].argmax(dim=-1)
                register_ids = torch.full((B, L_syll), 3, dtype=torch.long, device=device)
                sesqui_ids = torch.full((B, L_syll), 2, dtype=torch.long, device=device)
            else:
                reg_logits = self.heads["register"](bus_l)
                register_ids = 1 + reg_logits[:, :, 1:3].argmax(dim=-1)
                sesqui_ids = 1 + self.heads["sesqui"](bus_l)[:, :, 1:3].argmax(dim=-1)
                is_cs = gate_l.squeeze(-1) >= 0.5
                register_ids = torch.where(
                    is_cs, torch.full_like(register_ids, 3), register_ids
                )
                sesqui_ids = torch.where(is_cs, torch.full_like(sesqui_ids, 2), sesqui_ids)

            # v10.7 AUD-107-01: apply the DEFERRED no-op mask AFTER the
            # language branch, not before it.
            #
            # v10.7 applied it before the `if target_lang == "en"` branch, but
            # that branch OVERWRITES onset_ids, vowel_ids and coda_ids with the
            # reserved English ranges (34+, 28+, 18+). The deferred tail was
            # therefore never masked on a KM -> EN slice, so the tail syllables
            # were emitted in the current clause AND deferred to the next one,
            # duplicating the words.
            #
            # The mask also now covers coeng_ids and sesqui_ids. A deferred
            # syllable must be a complete no-op: sesqui_ids == 0 is the
            # zero-frame sentinel consumed by LearnedDurationAligner, so the
            # diffusion model stops allocating acoustic frames to a syllable
            # that is not spoken in this clause.
            if wait_mask is not None:
                zero_tail = torch.zeros_like(onset_ids[:, -3:])
                onset_ids = torch.cat(
                    [onset_ids[:, :-3], torch.where(wait_mask, zero_tail, onset_ids[:, -3:])],
                    dim=1,
                )
                coeng_ids = torch.cat(
                    [coeng_ids[:, :-3], torch.where(wait_mask, zero_tail, coeng_ids[:, -3:])],
                    dim=1,
                )
                vowel_ids = torch.cat(
                    [vowel_ids[:, :-3], torch.where(wait_mask, zero_tail, vowel_ids[:, -3:])],
                    dim=1,
                )
                coda_ids = torch.cat(
                    [coda_ids[:, :-3], torch.where(wait_mask, zero_tail, coda_ids[:, -3:])],
                    dim=1,
                )
                sesqui_ids = torch.cat(
                    [sesqui_ids[:, :-3], torch.where(wait_mask, zero_tail, sesqui_ids[:, -3:])],
                    dim=1,
                )

            syll_batch = SyllableTensorBatch(
                onset_ids=onset_ids,
                coeng_ids=coeng_ids,
                medial_ids=medial_ids,
                register_ids=register_ids,
                vowel_ids=vowel_ids,
                coda_ids=coda_ids,
                sesqui_ids=sesqui_ids,
            )
        else:
            syll_batch = explicit_syll_batch
            L_syll = syll_batch.onset_ids.shape[1]
            pos_q = self._build_sinusoidal_queries(B, L_syll, device, bus_cond.dtype)
            reordered_bus, attn_w = self.reorder_attn(
                query=pos_q, key=kv_mem, value=kv_mem, need_weights=True
            )
            bus_l = self.reorder_norm(pos_q + reordered_bus)
            self.last_gate_l = torch.bmm(attn_w, gate_mem)
            self.last_deferred_syll_count = 0

        slot_sum = (
            self.embeddings["onset"](syll_batch.onset_ids)
            + self.embeddings["coeng"](syll_batch.coeng_ids)
            + self.embeddings["medial"](syll_batch.medial_ids)
            + self.embeddings["register"](syll_batch.register_ids)
            + self.embeddings["vowel"](syll_batch.vowel_ids)
            + self.embeddings["coda"](syll_batch.coda_ids)
            + self.embeddings["sesqui"](syll_batch.sesqui_ids)
        )
        syll_hidden = bus_l + slot_sum
        aligned_hidden, pred_durs = self.duration_aligner(
            syll_hidden, syll_batch.sesqui_ids, target_frames
        )
        self.last_pred_durations = pred_durs
        # v10.7 P1-D: zero-parameter speaker-timbre conditioning. The speaker
        # registry needs a speaker-discriminative embedding, but the emotion
        # encoder applies F.instance_norm and avg_pool1d(k=25), which remove
        # the spectral envelope that carries vocal-tract identity. The
        # L2-normalized mean-pooled layer-6 acoustic representation is used
        # instead, and is added to the aligned hidden state BEFORE cond_proj.
        # This adds no parameters: the ledger is unchanged.
        if speaker_timbre is not None:
            aligned_hidden = aligned_hidden + speaker_timbre
        cond = self.cond_proj(aligned_hidden)
        return cond, syll_batch

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        emotion_tokens: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        d = self.config.mouth_d_model
        d_half = d // 2
        h = self.mel_in(x_t.transpose(1, 2)) + cond
        # v10.7 P1-C: parameter-free sinusoidal Fourier timestep embedding.
        # v10.6 expanded a rank-1 scalar across d_half columns, so time_mlp saw
        # only w1*t + w2*dt + b and had no spectral resolution over t at all.
        # The sinusoidal embedding restores full resolution with ZERO new
        # parameters: time_mlp keeps its Linear(d, d) shape.
        t_part = sinusoidal_time_embedding(t, d_half).to(
            dtype=h.dtype, device=h.device
        )
        if dt is None:
            dt_part = torch.zeros(B, d - d_half, dtype=h.dtype, device=h.device)
        else:
            dt_part = sinusoidal_time_embedding(dt, d - d_half).to(
                dtype=h.dtype, device=h.device
            )
        t_vec = torch.cat([t_part, dt_part], dim=-1)
        t_emb = self.time_mlp(t_vec)
        t_mod = self.adaLN_modulation(t_emb).view(B, 6, d)
        emo = self.emo_proj(emotion_tokens)
        for blk in self.blocks:
            h = blk(h, t_mod, emo)
        return self.mel_out(h).transpose(1, 2)
