from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch

# ============================================================================
# 1. LOCKED GLOBAL CONSTANTS
# ============================================================================
SAMPLE_RATE_IN: int = 16000
SAMPLE_RATE_OUT: int = 24000
SAMPLE_RATE_MNE: int = 48000
N_FFT_16K: int = 1024
HOP_LENGTH_16K: int = 160          # 10.0ms (100 Hz frame rate)
N_FFT_24K: int = 2048
HOP_LENGTH_24K: int = 240          # 10.0ms (100 Hz frame rate)
N_FFT_48K: int = 4096
HOP_LENGTH_48K: int = 480          # 10.0ms (100 Hz frame rate)
HOP_LENGTH: int = 240              # Backward-compatible 24k hop alias (100 Hz)
N_MELS: int = 100
ERB_BANDS: int = 64
ERB_MIN_FREQ_HZ: float = 120.0
ERB_MAX_FREQ_HZ: float = 8000.0
SEMANTIC_DIM: int = 512
EMOTION_NUM_TOKENS: int = 32
MIC_EQ_TAPS_16K: int = 64
MIC_EQ_TAPS_24K: int = 96
VELVET_RIR_TAPS_24K: int = 6000
MINOR_SYLLABLE_MAX_FRAMES: float = 6.0
MAX_ELASTIC_PAUSE_BORROW_MS: float = 150.0
DOMINANCE_MIN_DB_RATIO: float = -12.0
LOOKAHEAD_BUFFER_SEC: float = 2.0
MAX_CLAUSE_SLICE_SEC: float = 1.6
MIN_CLAUSE_SLICE_SEC: float = 1.2
SILENCE_RMS_THRESHOLD: float = 0.008
UTTERANCE_END_SILENCE_MS: float = 250.0
REPLAY_CACHE_MAX_ITEMS: int = 32
REPLAY_BUCKET_MS: float = 250.0

LanguageCode = Literal["en", "km", "th", "vi"]
LANG_TO_ID: Dict[str, int] = {"en": 0, "km": 1, "th": 2, "vi": 3}


class VocalMode(IntEnum):
    AMBIENT_OR_CLAP = 0
    NON_VERBAL_HUMAN = 1
    SINGING_OR_MUSIC = 2
    SPOKEN_DIALOGUE = 3


# ============================================================================
# 2. DUAL-PROFILE ARCHITECTURE CONFIGURATION
# ============================================================================
@dataclass(frozen=True)
class ModelArchitectureConfig:
    profile_name: Literal["prod_240m", "ci_micro"]
    separator_d_model: int
    separator_layers: int
    mode_gate_d_model: int
    mode_gate_layers: int
    ear_d_model: int
    ear_layers: int
    ear_heads: int
    ear_ffn_dim: int
    emotion_d_model: int
    emotion_layers: int
    emotion_ffn_dim: int
    pointer_d_model: int
    pointer_layers: int
    pointer_ffn_dim: int
    mouth_d_model: int
    mouth_layers: int
    mouth_heads: int
    mouth_ffn_dim: int
    euler_steps: int
    vocos_d_model: int
    vocos_layers: int
    vocos_intermediate_dim: int


PROD_240M_CONFIG = ModelArchitectureConfig(
    profile_name="prod_240m",
    separator_d_model=384,
    separator_layers=3,
    mode_gate_d_model=256,
    mode_gate_layers=4,
    ear_d_model=512,
    ear_layers=16,
    ear_heads=8,
    ear_ffn_dim=2048,
    emotion_d_model=512,
    emotion_layers=4,
    emotion_ffn_dim=2048,
    pointer_d_model=512,
    pointer_layers=2,
    pointer_ffn_dim=1024,
    mouth_d_model=768,
    mouth_layers=15,
    mouth_heads=12,
    mouth_ffn_dim=3072,
    euler_steps=12,
    vocos_d_model=512,
    vocos_layers=8,
    vocos_intermediate_dim=1536,
)

CI_MICRO_CONFIG = ModelArchitectureConfig(
    profile_name="ci_micro",
    separator_d_model=64,
    separator_layers=1,
    mode_gate_d_model=64,
    mode_gate_layers=2,
    ear_d_model=128,
    ear_layers=2,
    ear_heads=4,
    ear_ffn_dim=256,
    emotion_d_model=128,
    emotion_layers=1,
    emotion_ffn_dim=256,
    pointer_d_model=128,
    pointer_layers=1,
    pointer_ffn_dim=256,
    mouth_d_model=128,
    mouth_layers=2,
    mouth_heads=4,
    mouth_ffn_dim=256,
    euler_steps=2,
    vocos_d_model=128,
    vocos_layers=2,
    vocos_intermediate_dim=256,
)


# ============================================================================
# 3. FROZEN DATACLASSES & TENSOR SCHEMAS
# ============================================================================
@dataclass(frozen=True)
class KhmerSyllableToken:
    surface: str
    onset_id: int
    coeng_id: int
    medial_id: int
    register_id: int
    vowel_id: int
    coda_id: int
    sesqui_id: int


@dataclass(frozen=True)
class SyllableTensorBatch:
    onset_ids: torch.Tensor
    coeng_ids: torch.Tensor
    medial_ids: torch.Tensor
    register_ids: torch.Tensor
    vowel_ids: torch.Tensor
    coda_ids: torch.Tensor
    sesqui_ids: torch.Tensor


@dataclass
class CartridgeThreadState:
    """
    v10.7 P0-A: thread-local carrier for the per-call cartridge telemetry that
    v10.6 stored as mutable module attributes (last_l6_hidden, last_l8_hidden,
    last_bus_out, last_next_past_kv, last_next_past_gate,
    last_deferred_syll_count,
    last_pred_durations, last_gate_l).

    The GPU worker runs in asyncio.to_thread while the event loop thread owns
    the control plane, so a shared module attribute could be observed
    half-updated by the other thread. Each field is therefore stored in a
    threading.local() slot, and the owning module exposes property bridges
    under the original attribute names so every existing call site keeps
    working unchanged.

    This class holds NO parameters and no trainable state.
    """

    def __post_init__(self) -> None:
        import threading

        self._tls = threading.local()

    def _get(self, name: str):
        return getattr(self._tls, name, None)

    def _set(self, name: str, value) -> None:
        setattr(self._tls, name, value)

    @property
    def l6_hidden(self):
        return self._get("l6_hidden")

    @l6_hidden.setter
    def l6_hidden(self, value) -> None:
        self._set("l6_hidden", value)

    @property
    def l8_hidden(self):
        return self._get("l8_hidden")

    @l8_hidden.setter
    def l8_hidden(self, value) -> None:
        self._set("l8_hidden", value)

    @property
    def last_bus_out(self):
        """v10.7 P2-B: the bus-projected acoustic output of the ear.

        The multi-teacher distiller needs the student bus representation to
        compare against the teacher features, so it is published here.
        """
        return self._get("last_bus_out")

    @last_bus_out.setter
    def last_bus_out(self, value) -> None:
        self._set("last_bus_out", value)

    @property
    def last_next_past_kv(self):
        return self._get("last_next_past_kv")

    @last_next_past_kv.setter
    def last_next_past_kv(self, value) -> None:
        self._set("last_next_past_kv", value)

    @property
    def last_next_past_gate(self):
        return self._get("last_next_past_gate")

    @last_next_past_gate.setter
    def last_next_past_gate(self, value) -> None:
        self._set("last_next_past_gate", value)

    @property
    def last_deferred_syll_count(self):
        # v10.7: the deferred syllable count defaults to 0 when it has not
        # been set yet, so a reader never sees None.
        value = self._get("last_deferred_syll_count")
        return 0 if value is None else value

    @last_deferred_syll_count.setter
    def last_deferred_syll_count(self, value) -> None:
        self._set("last_deferred_syll_count", value)

    @property
    def last_pred_durations(self):
        return self._get("last_pred_durations")

    @last_pred_durations.setter
    def last_pred_durations(self, value) -> None:
        self._set("last_pred_durations", value)

    @property
    def last_gate_l(self):
        return self._get("last_gate_l")

    @last_gate_l.setter
    def last_gate_l(self, value) -> None:
        self._set("last_gate_l", value)


@dataclass(frozen=True)
class SeparatedSceneSlice:
    utterance_id: int
    utterance_start_ms: float
    frame_vocal_modes: torch.Tensor
    dominant_vocal_modes: List[VocalMode]
    active_speaker_stems: torch.Tensor
    speaker_ids: List[int]
    mic_eq_residual_16k: torch.Tensor
    rt60_sec: torch.Tensor
    wet_gain: torch.Tensor
    rms_scalar: torch.Tensor
    erb_suppression_mask: torch.Tensor
    trailing_pause_ms: float


@dataclass(frozen=True)
class LivePacket:
    kind: Literal["realtime_audio", "content"]
    timestamp_ms: float
    pcm_data: Optional[np.ndarray] = None
    payload: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DownstreamEvent:
    event_type: Literal[
        "audio_chunk", "passthrough_vocal", "singing_lyrics", "caption", "interrupted"
    ]
    utterance_id: int
    utterance_start_ms: float = 0.0
    vocal_modes: List[int] = field(default_factory=lambda: [3])
    speaker_ids: List[int] = field(default_factory=lambda: [0])
    source_lang: LanguageCode = "en"
    target_lang: LanguageCode = "km"
    dubbed_audio_b64: Optional[str] = None
    erb_suppression_mask_b64: Optional[str] = None
    erb_frame_count: int = 0
    per_speaker_source_texts: List[str] = field(default_factory=list)
    per_speaker_target_texts: List[str] = field(default_factory=list)
    codeswitch_words: List[str] = field(default_factory=list)
    playback_rate: float = 1.0
    actual_duration_sec: float = 0.0
    drift_ms: float = 0.0
    interrupt_epoch: int = 0


# ============================================================================
# 4. RUNTIME-CHECKABLE MULTI-AGENT PROTOCOLS
# ============================================================================
@runtime_checkable
class CodeSwitchDecomposerProtocol(Protocol):
    def normalize_khmer(self, text: str) -> str: ...
    def decompose_mixed_sentence(self, text: str) -> List[KhmerSyllableToken]: ...
    def encode_sentence_to_batch(
        self, text: str, device: str = "cpu"
    ) -> SyllableTensorBatch: ...
    def decode_batch_to_text(
        self, batch: SyllableTensorBatch
    ) -> Tuple[List[str], List[str]]: ...


@runtime_checkable
class StemAndSceneProtocol(Protocol):
    def separate_and_classify(
        self,
        wav_16k: torch.Tensor,
        utterance_id: int = 0,
        utterance_start_ms: float = 0.0,
        trailing_pause_ms: float = 0.0,
        primary_spk_emb: Optional[torch.Tensor] = None,
        force_dual_speakers: bool = False,
    ) -> SeparatedSceneSlice: ...

    def apply_scene_glue(
        self,
        dry_khmer_wav_24k: torch.Tensor,
        mic_eq_residual_16k: torch.Tensor,
        rt60_sec: torch.Tensor,
        wet_gain: torch.Tensor,
        target_rms_scalar: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class PrecomputedEarFeatures:
    """
    v10.7 AUD-107-02: the results of ONE StreamingEarCartridge forward pass.

    The full-duplex runner needs the layer-6 acoustic tap (for the speaker
    registry) and the log-mel (for the energy weighting) BEFORE it can call
    forward_s2st. v10.7 ran the ear twice and the log-mel three times on the
    same audio slice, doubling the 64.5M-parameter ear and the 8.7M-parameter
    emotion encoder on every spoken utterance and threatening the
    T_GPU <= 90 ms latency law.

    The runner computes this bundle ONCE and passes it to forward_s2st, which
    reuses it instead of re-running the ear. Adds NO parameters.
    """

    bus: torch.Tensor
    r_sps: torch.Tensor
    mel_stems: torch.Tensor
    h_acoustic_l6: torch.Tensor


@runtime_checkable
class S2STConsoleProtocol(Protocol):
    def forward_s2st(
        self,
        source_wav_16k: torch.Tensor,
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        orig_duration_sec: float,
        trailing_pause_ms: float = 0.0,
        playback_rate: float = 1.0,
        explicit_syll_batch: Optional[SyllableTensorBatch] = None,
        precomputed_ear: Optional[PrecomputedEarFeatures] = None,
    ) -> Tuple[torch.Tensor, SyllableTensorBatch, torch.Tensor]: ...


    def compute_flow_loss(
        self,
        target_wav_24k: torch.Tensor,
        syll_batch: Optional[SyllableTensorBatch] = None,
        infill_mode: bool = False,
        source_wav_16k: Optional[torch.Tensor] = None,
        source_lang: LanguageCode = "en",
    ) -> torch.Tensor: ...
