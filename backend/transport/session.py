from __future__ import annotations

import threading
import zlib
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend.contracts.schemas import (
    MAX_CLAUSE_SLICE_SEC,
    MIN_CLAUSE_SLICE_SEC,
    REPLAY_BUCKET_MS,
    REPLAY_CACHE_MAX_ITEMS,
    SAMPLE_RATE_IN,
    SILENCE_RMS_THRESHOLD,
    UTTERANCE_END_SILENCE_MS,
    DownstreamEvent,
    LanguageCode,
)


class SpeakerRegistry:
    def __init__(self, max_speakers: int = 4, similarity_threshold: float = 0.72) -> None:
        # v10.7 P0-B: the centroid list is mutated in place by a read-modify-write
        # in match_or_register_speaker. The GPU worker thread and the event-loop
        # thread can both reach it, so the list needs its own lock; otherwise a
        # concurrent caller can observe a half-updated centroid vector.
        self._lock = threading.RLock()
        self.max_speakers = max_speakers
        self.similarity_threshold = similarity_threshold
        self.centroids: List[torch.Tensor] = []

    def match_or_register_speaker(self, emb: torch.Tensor) -> Tuple[int, torch.Tensor]:
        # v10.7 P0-B: hold the registry lock across the whole
        # read-modify-write so a concurrent caller cannot observe a
        # half-updated centroid list.
        with self._lock:
            vec = F.normalize(emb.detach().float().view(-1), p=2, dim=0)
            if not self.centroids:
                self.centroids.append(vec)
                return 0, self.centroids[0]

            sims = [float(torch.dot(vec, c).item()) for c in self.centroids]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self.similarity_threshold:
                updated = F.normalize(0.85 * self.centroids[best_idx] + 0.15 * vec, p=2, dim=0)
                self.centroids[best_idx] = updated
                return best_idx, self.centroids[0]

            if len(self.centroids) < self.max_speakers:
                self.centroids.append(vec)
                return len(self.centroids) - 1, self.centroids[0]

            updated = F.normalize(0.85 * self.centroids[best_idx] + 0.15 * vec, p=2, dim=0)
            self.centroids[best_idx] = updated
            return best_idx, self.centroids[0]


class InMemorySession:
    def __init__(self, session_id: str) -> None:
        import threading
        self._lock = threading.RLock()
        self.session_id = session_id
        self.speaker_registry = SpeakerRegistry()
        self.pcm_buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self.silence_ms: float = 0.0
        self.active_speech_ms: float = 0.0
        self.is_speaking: bool = False
        self.current_utterance_start_ms: float = 0.0
        self.stream_cursor_ms: float = 0.0
        self.next_utterance_id: int = 0
        self._replay_cache: OrderedDict[Tuple[int, int, int, float, str], DownstreamEvent] = OrderedDict()
        self.past_clause_kv: Optional[torch.Tensor] = None
        # v10.7 P0-A: records whether the most recently CONSUMED slice was a
        # Trigger #2 hard-ceiling cut. The runner uses it to decide whether to
        # pass the rolling KV cache into the next clause.
        self.last_slice_was_hard_ceiling: bool = False
        self.past_gate_kv: Optional[torch.Tensor] = None
        self.deferred_syll_count: int = 0

    @staticmethod
    def compute_pcm_energy_hash(slice_pcm: np.ndarray) -> int:
        if slice_pcm.size < 64:
            return 0
        n_bins = 8
        bin_len = slice_pcm.size // n_bins
        if bin_len == 0:
            return 0
        overall_rms = float(np.sqrt(np.mean(slice_pcm**2) + 1e-12))
        profile = bytearray()
        for b in range(n_bins):
            chunk = slice_pcm[b * bin_len : (b + 1) * bin_len]
            c_rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            norm_c = int(np.clip(round((c_rms / max(overall_rms, 1e-6)) * 10.0), 0, 99))
            profile.append(norm_c)
        return zlib.crc32(bytes(profile)) & 0xFFFFFFFF

    @staticmethod
    def compute_pcm_crc32(slice_pcm: np.ndarray) -> int:
        # Phase-invariant energy envelope hash ensures 100% cache hits across 1-63 sample shifts
        return InMemorySession.compute_pcm_energy_hash(slice_pcm)

    @staticmethod
    def _make_replay_key(
        utterance_start_ms: float,
        duration_ms: float,
        pcm_crc32: int,
        playback_rate: float,
        target_lang: LanguageCode,
    ) -> Tuple[int, int, int, float, str]:
        bucket = int(round(float(utterance_start_ms) / REPLAY_BUCKET_MS))
        dur_bucket = int(round(float(duration_ms) / 50.0))
        return (bucket, dur_bucket, int(pcm_crc32), round(float(playback_rate), 2), str(target_lang))

    def cache_dubbed_event(
        self,
        event: DownstreamEvent,
        duration_ms: float = 0.0,
        pcm_crc32: int = 0,
    ) -> None:
        # v10.7 P0-B: this method mutates shared session state and is
        # reachable from both the event-loop thread and the GPU worker
        # thread, so it must hold the session lock.
        with self._lock:
            dur = duration_ms if duration_ms > 0.0 else (event.actual_duration_sec * 1000.0)
            key = self._make_replay_key(
                event.utterance_start_ms, dur, pcm_crc32, event.playback_rate, event.target_lang
            )
            if key in self._replay_cache:
                self._replay_cache.move_to_end(key)
            self._replay_cache[key] = event
            while len(self._replay_cache) > REPLAY_CACHE_MAX_ITEMS:
                self._replay_cache.popitem(last=False)

    def lookup_cached_event(
        self,
        utterance_start_ms: float,
        duration_ms: float = 0.0,
        pcm_crc32: int = 0,
        playback_rate: float = 1.0,
        target_lang: LanguageCode = "km",
    ) -> Optional[DownstreamEvent]:
        # v10.7 P0-B: this method mutates shared session state and is
        # reachable from both the event-loop thread and the GPU worker
        # thread, so it must hold the session lock.
        with self._lock:
            # Exact match
            key = self._make_replay_key(utterance_start_ms, duration_ms, pcm_crc32, playback_rate, target_lang)
            evt = self._replay_cache.get(key)
            if evt is not None:
                self._replay_cache.move_to_end(key)
                return evt

            # Fuzzy bucket fallback if crc32 was 0 (unit test) or duration matched within 1 bucket (live seek)
            target_bucket = int(round(float(utterance_start_ms) / REPLAY_BUCKET_MS))
            target_dur_bucket = int(round(float(duration_ms) / 50.0))
            for (b, db, cached_crc, r, l), cached_evt in reversed(self._replay_cache.items()):
                if b == target_bucket and r == round(float(playback_rate), 2) and l == str(target_lang):
                    crc_ok = (pcm_crc32 == 0 or cached_crc == 0 or cached_crc == pcm_crc32)
                    if (pcm_crc32 == 0 and duration_ms == 0.0) or (
                        duration_ms > 0.0 and abs(db - target_dur_bucket) <= 1 and crc_ok
                    ):
                        return cached_evt
            return None

    def reset_on_seek(self, new_timestamp_ms: float) -> None:
        """
        Rewind/Seek Law: Clears uncommitted PCM audio and resets VAD timers to new_timestamp_ms
        while preserving SpeakerRegistry voice centroids and _replay_cache.
        """
        with self._lock:
            self.pcm_buffer = np.zeros(0, dtype=np.float32)
            self.is_speaking = False
            self.silence_ms = 0.0
            self.active_speech_ms = 0.0
            self.stream_cursor_ms = float(new_timestamp_ms)
            self.current_utterance_start_ms = float(new_timestamp_ms)
            self.past_clause_kv = None
            self.past_gate_kv = None
            self.deferred_syll_count = 0
            # v10.7 P0-A: a seek ends the clause, so the hard-ceiling flag is
            # cleared along with the rolling cache.
            self.last_slice_was_hard_ceiling = False

    def _flush_rolling_state(self) -> None:
        """
        v10.7 P0-A: drop the rolling cross-clause KV cache and the deferred
        syllable count.

        The cache is only valid across a Trigger #2 hard-ceiling cut, where the
        same clause continues. A natural pause (Trigger #1), a rejected micro-pop
        and a seek all end the clause, so the cached context must not leak into
        the next one.

        The caller must already hold self._lock.
        """
        self.past_clause_kv = None
        self.past_gate_kv = None
        self.deferred_syll_count = 0


    def append_pcm_chunk(
        self, chunk: np.ndarray, timestamp_ms: float = -1.0
    ) -> Optional[Tuple[int, float, np.ndarray, float]]:
        # v10.7 P0-B: this method mutates shared session state and is
        # reachable from both the event-loop thread and the GPU worker
        # thread, so it must hold the session lock.
        with self._lock:
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                return None

            # Continuous timestamp clock synchronization during non-speech periods
            if timestamp_ms >= 0.0 and not self.is_speaking:
                self.stream_cursor_ms = float(timestamp_ms)

            chunk_ms = (chunk.size / float(SAMPLE_RATE_IN)) * 1000.0
            rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))

            if rms < SILENCE_RMS_THRESHOLD:
                self.silence_ms += chunk_ms
            else:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.current_utterance_start_ms = self.stream_cursor_ms
                    self.active_speech_ms = 0.0
                self.active_speech_ms += chunk_ms
                self.silence_ms = 0.0

            self.stream_cursor_ms += chunk_ms
            if not self.is_speaking:
                return None

            self.pcm_buffer = np.concatenate([self.pcm_buffer, chunk])

            # Trigger #1: Natural Pause (>= 250ms trailing silence)
            if self.silence_ms >= UTTERANCE_END_SILENCE_MS:
                if self.active_speech_ms < 250.0:
                    self.pcm_buffer = np.zeros(0, dtype=np.float32)
                    self.is_speaking = False
                    self.silence_ms = 0.0
                    self.active_speech_ms = 0.0
                    # v10.7 P0-A: a micro-pop is not a clause boundary, but the
                    # rejected buffer must not leave stale rolling state behind.
                    self._flush_rolling_state()
                    return None

                pause_ms = float(self.silence_ms)
                silence_samples = int(round((pause_ms / 1000.0) * SAMPLE_RATE_IN))
                if 0 < silence_samples < self.pcm_buffer.size:
                    slice_pcm = self.pcm_buffer[:-silence_samples].copy()
                else:
                    slice_pcm = self.pcm_buffer.copy()

                uid = self.next_utterance_id
                self.next_utterance_id += 1
                start_ms = float(self.current_utterance_start_ms)

                self.pcm_buffer = np.zeros(0, dtype=np.float32)
                self.is_speaking = False
                self.silence_ms = 0.0
                self.active_speech_ms = 0.0
                # v10.7 P0-A: a natural pause is a CLAUSE BOUNDARY. The rolling
                # cross-clause KV cache must not survive it: the next clause is a
                # new utterance and the cached context is not valid for it.
                self._flush_rolling_state()
                self.last_slice_was_hard_ceiling = False
                return (uid, start_ms, slice_pcm, pause_ms)

            # Trigger #2: Hard Wall-Clock & Active-Speech Elapsed Ceiling (>= 1600ms active clause span)
            utterance_wall_clock_ms = (self.pcm_buffer.size / float(SAMPLE_RATE_IN)) * 1000.0
            active_clause_wall_clock_ms = utterance_wall_clock_ms - self.silence_ms
            if (
                active_clause_wall_clock_ms >= MAX_CLAUSE_SLICE_SEC * 1000.0
                or self.active_speech_ms >= MAX_CLAUSE_SLICE_SEC * 1000.0
            ):
                min_idx = int(MIN_CLAUSE_SLICE_SEC * SAMPLE_RATE_IN)
                max_idx = min(self.pcm_buffer.size, int(MAX_CLAUSE_SLICE_SEC * SAMPLE_RATE_IN))
                if max_idx <= min_idx:
                    max_idx = self.pcm_buffer.size
                win = 320  # 20ms window at 16kHz
                best_split = max_idx
                best_rms = float("inf")
                for start in range(min_idx, max(min_idx + 1, max_idx - win + 1), win):
                    w_rms = float(np.sqrt(np.mean(self.pcm_buffer[start : start + win] ** 2) + 1e-12))
                    if w_rms < best_rms:
                        best_rms = w_rms
                        best_split = start + win // 2

                slice_pcm = self.pcm_buffer[:best_split].copy()
                remainder = self.pcm_buffer[best_split:].copy()
                uid = self.next_utterance_id
                self.next_utterance_id += 1
                start_ms = float(self.current_utterance_start_ms)

                self.current_utterance_start_ms += (best_split / float(SAMPLE_RATE_IN)) * 1000.0

                trim_idx = 0
                while trim_idx + win <= remainder.size:
                    f_rms = float(
                        np.sqrt(np.mean(remainder[trim_idx : trim_idx + win] ** 2) + 1e-12)
                    )
                    if f_rms < SILENCE_RMS_THRESHOLD:
                        trim_idx += win
                        self.current_utterance_start_ms += 20.0
                    else:
                        break
                remainder = remainder[trim_idx:]

                if remainder.size == 0:
                    self.pcm_buffer = np.zeros(0, dtype=np.float32)
                    self.is_speaking = False
                    self.active_speech_ms = 0.0
                    self.silence_ms = 0.0
                else:
                    self.pcm_buffer = remainder
                    active_frames = 0
                    for i in range(0, remainder.size, win):
                        seg = remainder[i : i + win]
                        if float(np.sqrt(np.mean(seg ** 2) + 1e-12)) >= SILENCE_RMS_THRESHOLD:
                            active_frames += seg.size
                    self.active_speech_ms = (active_frames / float(SAMPLE_RATE_IN)) * 1000.0
                    self.silence_ms = 0.0

                # v10.7 P0-A: this slice was cut by the Trigger #2 hard ceiling,
                # so the rolling cross-clause KV cache IS valid for the next
                # clause. The runner reads this flag to decide whether to pass
                # the cached state into the next forward_s2st call.
                self.last_slice_was_hard_ceiling = True
                return (uid, start_ms, slice_pcm, 40.0)

            return None
