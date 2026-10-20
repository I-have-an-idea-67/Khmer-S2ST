from __future__ import annotations

import base64
import io
import struct
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from backend.contracts.schemas import (
    CI_MICRO_CONFIG,
    SAMPLE_RATE_IN,
    SAMPLE_RATE_OUT,
    DownstreamEvent,
    PrecomputedEarFeatures,
    VocalMode,
)
from backend.linguistics.syllable_decomposer import BilingualCodeSwitchDecomposer
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole
from backend.nn_cartridges.stem_and_mode_gate import StemAndScenePipeline
from backend.transport.live_queue import LiveRequestQueue
from backend.transport.session import InMemorySession


def encode_wav_pcm16_b64(
    wav_24k: torch.Tensor | np.ndarray, sample_rate: int = SAMPLE_RATE_OUT
) -> str:
    """Pure 44-byte RIFF/WAVE PCM16 encoder (raw[:4] == b'RIFF' and raw[8:12] == b'WAVE')."""
    t = torch.as_tensor(wav_24k).detach().float().reshape(-1)
    pcm16 = (t.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu().numpy()
    data_bytes = pcm16.tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data_bytes)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data_bytes)))
    buf.write(data_bytes)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def energy_weighted_speaker_embedding(
    h_acoustic_l6: torch.Tensor,
    mel_stems: torch.Tensor,
) -> torch.Tensor:
    """
    v10.7 P1-D: extract a speaker-discriminative 512-D embedding.

    The SpeakerRegistry needs to tell speakers apart, so the embedding must
    retain the vocal-tract information. The DisentangledEmotionEncoder cannot
    supply it: it applies F.instance_norm (which removes each mel band's mean
    and variance over time) and avg_pool1d(k=25), and the resulting latent is
    deliberately pushed toward language invariance by the gradient-reversal
    branch. Matching on it conflates emotional prosody with speaker identity.

    The layer-6 acoustic representation carries the spectral envelope, so the
    embedding is an energy-weighted mean pool over the ACTIVE frames: frames
    whose mel energy is negligible contribute nothing, so silence does not drag
    the estimate toward the origin. The result is L2-normalized along the last
    dimension, matching the centroid convention used by SpeakerRegistry.

    Parameters:
        h_acoustic_l6: (B_spk, T_frames, SEMANTIC_DIM) acoustic representation.
        mel_stems: (B_spk, N_MELS, T_mel) log-mel of the separated stems.

    Returns:
        (B_spk, SEMANTIC_DIM) L2-normalized speaker embeddings.
        Adds NO parameters.
    """
    if h_acoustic_l6 is None:
        raise ValueError("energy_weighted_speaker_embedding requires the layer-6 tap")
    # Per-frame energy from the mel spectrogram, reduced onto the frame grid.
    mel_energy = mel_stems.pow(2).mean(dim=1)  # (B_spk, T_mel)
    if mel_energy.shape[-1] != h_acoustic_l6.shape[1]:
        mel_energy = F.interpolate(
            mel_energy.unsqueeze(1),
            size=h_acoustic_l6.shape[1],
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    weights = mel_energy.clamp(min=0.0)
    total = weights.sum(dim=-1, keepdim=True)
    # A fully silent stem falls back to an unweighted mean.
    weights = torch.where(total > 1e-8, weights, torch.ones_like(weights))
    pooled = (h_acoustic_l6 * weights.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.normalize(pooled, p=2, dim=-1)


class FullDuplexRunner:
    def __init__(
        self,
        pipeline: Optional[StemAndScenePipeline] = None,
        console: Optional[SymmetricS2STConsole] = None,
    ) -> None:
        self.pipeline = pipeline or StemAndScenePipeline(CI_MICRO_CONFIG)
        self.console = console or SymmetricS2STConsole(CI_MICRO_CONFIG)
        self.decomposer = BilingualCodeSwitchDecomposer()

    def process_utterance_slice(
        self,
        session: InMemorySession,
        queue: LiveRequestQueue,
        uid: int,
        start_ms: float,
        slice_pcm: np.ndarray,
        trailing_pause_ms: float,
    ) -> Optional[DownstreamEvent]:
        dur_ms = (slice_pcm.size / float(SAMPLE_RATE_IN)) * 1000.0
        pcm_crc = session.compute_pcm_crc32(slice_pcm)

        # 1. Check Rewind LRU Replay Cache first (< 0.1 ms fast-path)
        cached = session.lookup_cached_event(
            start_ms,
            duration_ms=dur_ms,
            pcm_crc32=pcm_crc,
            playback_rate=queue.playback_rate,
            target_lang="km",
        )
        if cached is not None:
            return cached

        # 2. Snapshot interrupt_epoch before GPU work (Epoch Fence)
        epoch_before = queue.interrupt_epoch

        # Retrieve primary host centroid E_0 (if registered) to prevent stem permutation
        primary_emb = (
            session.speaker_registry.centroids[0].unsqueeze(0)
            if session.speaker_registry.centroids
            else None
        )
        wav_16k = torch.from_numpy(slice_pcm).float().unsqueeze(0)
        scene = self.pipeline.separate_and_classify(
            wav_16k,
            utterance_id=uid,
            utterance_start_ms=start_ms,
            trailing_pause_ms=trailing_pause_ms,
            primary_spk_emb=primary_emb,
        )

        # Register / match separated speakers in SpeakerRegistry
        # v10.7 P1-D: run the ear over the SEPARATED STEMS so the layer-6 tap
        # is available per speaker. The emotion encoder is still used for the
        # emotional conditioning, but the SPEAKER identity now comes from the
        # acoustic representation, which is what actually carries vocal-tract
        # identity.
        stem_16k = scene.active_speaker_stems
        # v10.7 AUD-107-02: the ear runs ONCE per audio slice.
        #
        # v10.7 ran the ear here, then called log_mel TWICE more, then ran the
        # emotion encoder whose output was DISCARDED, and finally called
        # forward_s2st, which ran the ear and the emotion encoder a second time
        # on the identical audio. That doubled the 64.5M-parameter ear and the
        # 8.7M-parameter emotion encoder on every spoken utterance and
        # threatened the T_GPU <= 90 ms latency law.
        #
        # The bundle is computed once here and handed to forward_s2st, which
        # reuses it. The log-mel is computed ONCE and shared by the speaker
        # embedding and the emotion encoder.
        bus, r_sps = self.console.ear(stem_16k, "en")
        spk_l6 = self.console.ear.last_l6_hidden
        mel_stems = self.console.ear.log_mel(stem_16k)
        # v10.7 P1-D: the speaker identity must come from a SPEAKER-DISCRIMINATIVE
        # embedding. v10.6 used the DisentangledEmotionEncoder output, but that
        # encoder applies F.instance_norm and avg_pool1d(k=25) specifically to
        # REMOVE the spectral envelope that carries vocal-tract identity, so the
        # registry was matching on emotional prosody rather than on the speaker.
        # The layer-6 acoustic representation is used instead, mean-pooled over
        # the ACTIVE (energy-bearing) frames and L2-normalized. Zero parameters.
        spk_embs = energy_weighted_speaker_embedding(
            spk_l6, mel_stems
        )
        tracked_speaker_ids: list[int] = []
        for spk_i in range(spk_embs.shape[0]):
            slot_id, _ = session.speaker_registry.match_or_register_speaker(spk_embs[spk_i])
            tracked_speaker_ids.append(slot_id)

        modes = [int(m) for m in scene.dominant_vocal_modes]

        if all(m <= int(VocalMode.SINGING_OR_MUSIC) for m in modes):
            ev_type = (
                "singing_lyrics"
                if any(m == int(VocalMode.SINGING_OR_MUSIC) for m in modes)
                else "passthrough_vocal"
            )
            evt = DownstreamEvent(
                event_type=ev_type,
                utterance_id=uid,
                utterance_start_ms=start_ms,
                vocal_modes=modes,
                speaker_ids=tracked_speaker_ids,
                playback_rate=queue.playback_rate,
                interrupt_epoch=epoch_before,
            )
            if queue.interrupt_epoch != epoch_before:
                return None
            session.cache_dubbed_event(evt, duration_ms=dur_ms, pcm_crc32=pcm_crc)
            return evt

        orig_dur = float(slice_pcm.size) / float(SAMPLE_RATE_IN)
        # v10.7 P0-A: pass the rolling cross-clause KV state into the console.
        # v10.6 never passed these four arguments, so the rolling KV cache and
        # the Hibiki entropy wait-gating were DEAD CODE in production even
        # though the console accepted them.
        dry_24k, syll_batch, _ = self.console.forward_s2st(
            scene.active_speaker_stems,
            source_lang="en",
            target_lang="km",
            orig_duration_sec=orig_dur,
            trailing_pause_ms=trailing_pause_ms,
            playback_rate=queue.playback_rate,
            past_clause_kv=session.past_clause_kv,
            past_gate_kv=session.past_gate_kv,
            deferred_syll_count=session.deferred_syll_count,
            is_hard_ceiling_slice=session.last_slice_was_hard_ceiling,
            precomputed_ear=PrecomputedEarFeatures(
                bus=bus,
                r_sps=r_sps,
                mel_stems=mel_stems,
                h_acoustic_l6=spk_l6,
            ),
        )
        glued_24k = self.pipeline.apply_scene_glue(
            dry_24k,
            scene.mic_eq_residual_16k,
            scene.rt60_sec,
            scene.wet_gain,
            scene.rms_scalar,
        )

        # Decode batch to target text and extracted code-switch words
        target_texts, cs_words = self.decomposer.decode_batch_to_text(syll_batch)

        # Case C: Mixed-Mode Co-Host Re-Injection
        T_out = glued_24k.shape[-1]
        mixed = torch.zeros(T_out, dtype=glued_24k.dtype, device=glued_24k.device)
        for spk_idx, m in enumerate(modes):
            if m == int(VocalMode.SPOKEN_DIALOGUE):
                mixed = mixed + glued_24k[spk_idx]
            else:
                stem_16k = scene.active_speaker_stems[spk_idx : spk_idx + 1].unsqueeze(1)
                stem_24k = F.interpolate(
                    stem_16k, size=T_out, mode="linear", align_corners=False
                ).reshape(-1)
                mixed = mixed + stem_24k

        mixed = mixed.clamp(-0.98, 0.98)
        erb_np = scene.erb_suppression_mask.detach().cpu().numpy().astype(np.float32)
        erb_b64 = base64.b64encode(erb_np.tobytes()).decode("ascii")

        # 3. Verify Epoch Fence after GPU synthesis
        if queue.interrupt_epoch != epoch_before:
            return None

        evt = DownstreamEvent(
            event_type="audio_chunk",
            utterance_id=uid,
            utterance_start_ms=start_ms,
            vocal_modes=modes,
            speaker_ids=tracked_speaker_ids,
            dubbed_audio_b64=encode_wav_pcm16_b64(mixed),
            erb_suppression_mask_b64=erb_b64,
            erb_frame_count=int(erb_np.shape[0]),
            per_speaker_target_texts=target_texts,
            codeswitch_words=cs_words,
            playback_rate=queue.playback_rate,
            actual_duration_sec=float(T_out) / float(SAMPLE_RATE_OUT),
            interrupt_epoch=epoch_before,
        )
        # v10.7 P0-A: commit the NEW rolling cross-clause KV state to the
        # session, but ONLY after the post-synthesis epoch fence has passed.
        # An interrupted slice must not be able to poison the next clause with
        # a KV cache derived from audio the user has already seeked away from.
        with session._lock:
            session.past_clause_kv = self.console.mouth.last_next_past_kv
            session.past_gate_kv = self.console.mouth.last_next_past_gate
            session.deferred_syll_count = self.console.mouth.last_deferred_syll_count
        session.cache_dubbed_event(evt, duration_ms=dur_ms, pcm_crc32=pcm_crc)
        return evt
