import base64
import time

import numpy as np
import pytest
import torch

from backend.contracts.schemas import DownstreamEvent
from backend.transport.live_queue import LiveRequestQueue
from backend.transport.runner import encode_wav_pcm16_b64
from backend.transport.session import InMemorySession


def test_session_pop_rejection_natural_pause_and_hard_ceiling():
    session = InMemorySession("test-session")
    chunk_20ms_speech = np.full(320, 0.15, dtype=np.float32)
    chunk_20ms_silence = np.zeros(320, dtype=np.float32)

    # (a) 20ms burst + 300ms silence -> rejected as micro-pop
    assert session.append_pcm_chunk(chunk_20ms_speech, timestamp_ms=0.0) is None
    for _ in range(15):
        assert session.append_pcm_chunk(chunk_20ms_silence) is None
    assert session.pcm_buffer.size == 0

    # (b) 1.35s utterance (with 140ms internal micro-pause) + 260ms trailing silence -> Trigger #1
    for _ in range(30):
        assert session.append_pcm_chunk(chunk_20ms_speech) is None
    for _ in range(7):
        assert session.append_pcm_chunk(chunk_20ms_silence) is None
    for _ in range(38):
        assert session.append_pcm_chunk(chunk_20ms_speech) is None
    sliced = None
    for _ in range(13):
        out = session.append_pcm_chunk(chunk_20ms_silence)
        if out is not None:
            sliced = out
            break
    assert sliced is not None
    uid, start_ms, slice_pcm, pause_ms = sliced
    assert pause_ms >= 250.0
    assert 1.30 <= (slice_pcm.size / 16000.0) <= 1.55

    # (c) Continuous speech >= 1.60s -> Trigger #2 slices in [1.20s, 1.60s]
    ceil_slice = None
    for _ in range(85):
        out = session.append_pcm_chunk(chunk_20ms_speech)
        if out is not None:
            ceil_slice = out
            break
    assert ceil_slice is not None
    _, _, pcm_c, _ = ceil_slice
    assert 1.20 <= (pcm_c.size / 16000.0) <= 1.60


def test_rewind_seek_preserves_speakers_and_hits_lru_replay_cache():
    session = InMemorySession("rewind-session")
    emb_host_a = torch.ones(512)
    slot_before, _ = session.speaker_registry.match_or_register_speaker(emb_host_a)
    assert slot_before == 0

    # Cache a dubbed utterance at 45,000 ms
    evt = DownstreamEvent(
        event_type="audio_chunk",
        utterance_id=7,
        utterance_start_ms=45000.0,
        dubbed_audio_b64="UklGRg==",
        playback_rate=1.0,
    )
    session.cache_dubbed_event(evt)

    # Simulate partial buffered speech at 60,000 ms followed by user rewinding back to 45,000 ms
    session.append_pcm_chunk(np.full(320, 0.2, dtype=np.float32), timestamp_ms=60000.0)
    assert session.pcm_buffer.size > 0
    session.reset_on_seek(new_timestamp_ms=45000.0)

    assert session.pcm_buffer.size == 0
    assert not session.is_speaking
    assert session.stream_cursor_ms == 45000.0
    # Speaker identity must survive rewind
    slot_after, _ = session.speaker_registry.match_or_register_speaker(emb_host_a)
    assert slot_after == 0
    # Rewound timestamp hits LRU replay cache
    replayed = session.lookup_cached_event(45020.0, playback_rate=1.0, target_lang="km")
    assert replayed is not None and replayed.utterance_id == 7


@pytest.mark.asyncio
async def test_live_queue_5ms_flush_and_wav_riff_header():
    q = LiveRequestQueue()
    for i in range(50):
        await q.send_realtime(np.zeros(320, dtype=np.float32), timestamp_ms=float(i * 20))
    t0 = time.perf_counter()
    await q.send_content("interrupt", timestamp_ms=1000.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 5.0
    assert q.interrupt_epoch == 1
    pkt = await q.get_next()
    assert pkt.kind == "content" and pkt.payload["action"] == "interrupt"

    b64_wav = encode_wav_pcm16_b64(np.sin(np.linspace(0, 10, 2400, dtype=np.float32)))
    raw = base64.b64decode(b64_wav)
    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WAVE"
# ===========================================================================
# v10.7 additions. The v10.6 assertions above are retained VERBATIM.
# ===========================================================================


def test_v107_rolling_kv_lifecycle_across_hard_ceiling_and_pause():
    # v10.7 (P0-A): the rolling cross-clause KV cache must PERSIST across a
    # Trigger #2 hard-ceiling slice (the same clause continues) and be
    # FLUSHED on a Trigger #1 natural pause (a new clause begins). v10.6
    # never passed the cache into the console at all, so the whole feature
    # was dead code.
    session = InMemorySession("kv-lifecycle")

    chunk_speech = np.full(320, 0.15, dtype=np.float32)
    chunk_silence = np.zeros(320, dtype=np.float32)

    # Continuous speech past the 1.60 s ceiling triggers a hard-ceiling slice.
    ceil_slice = None
    for _ in range(85):
        out = session.append_pcm_chunk(chunk_speech)
        if out is not None:
            ceil_slice = out
            break
    assert ceil_slice is not None
    assert session.last_slice_was_hard_ceiling is True

    # A hard-ceiling slice arms the rolling cache.
    session.past_clause_kv = torch.randn(1, 20, 512)
    session.past_gate_kv = torch.randn(1, 20, 512)
    session.deferred_syll_count = 2

    # A natural pause (Trigger #1) must FLUSH it, because a new clause
    # begins and the cached context is not valid for it.
    for _ in range(40):
        session.append_pcm_chunk(chunk_speech)
    for _ in range(20):
        out = session.append_pcm_chunk(chunk_silence)
        if out is not None:
            break
    assert session.past_clause_kv is None
    assert session.past_gate_kv is None
    assert session.deferred_syll_count == 0
    assert session.last_slice_was_hard_ceiling is False


def test_v107_bspk_transition_does_not_raise():
    # v10.7 (P0-A): when the number of active speakers changes between
    # clauses the cached KV has a different batch dimension, and the v10.6
    # torch.cat on dim=1 raised a RuntimeError. The stale cache must be
    # discarded instead.
    #
    # The duration aligner requires L_syll == target_frames, so the syllable
    # count is pinned with an explicit batch.
    import torch

    from backend.contracts.schemas import CI_MICRO_CONFIG, SyllableTensorBatch
    from backend.nn_cartridges.mouth_cartridge import KhmerMouthCartridge

    torch.manual_seed(0)
    mouth = KhmerMouthCartridge(CI_MICRO_CONFIG)
    mouth.eval()

    B = 2
    T = 24
    L_SYLL = 2
    bus = torch.randn(B, T, 512)
    gate = torch.randn(B, T, 512)
    r_sps = torch.rand(B)

    t = lambda v: torch.full((B, L_SYLL), v, dtype=torch.long)
    explicit = SyllableTensorBatch(
        onset_ids=t(1),
        coeng_ids=t(0),
        medial_ids=t(0),
        register_ids=t(1),
        vowel_ids=t(0),
        coda_ids=t(0),
        sesqui_ids=t(2),
    )

    # A cache built for ONE speaker, presented to a TWO-speaker call.
    stale_kv = torch.randn(1, 20, 512)
    stale_gate = torch.randn(1, 20, 512)

    with torch.no_grad():
        cond, _ = mouth.predict_syllables_and_align(
            bus,
            gate,
            r_sps,
            "km",
            0.5,
            T,
            explicit_syll_batch=explicit,
            past_clause_kv=stale_kv,
            past_gate_kv=stale_gate,
            deferred_syll_count=0,
            is_hard_ceiling_slice=True,
        )
    assert cond.shape[0] == B


def test_v107_session_and_registry_are_thread_safe():
    # v10.7 (P0-B): every method that mutates shared session state must hold
    # the session lock, and the SpeakerRegistry must hold its own lock.
    import threading
    import traceback

    errors = []
    stop = threading.Event()

    def worker(seed):
        try:
            torch.manual_seed(seed)
            s = InMemorySession("threaded")
            emb = torch.randn(512)
            while not stop.is_set():
                s.append_pcm_chunk(np.zeros(320, dtype=np.float32), timestamp_ms=0.0)
                s.lookup_cached_event(0.0, duration_ms=10.0, pcm_crc32=1, playback_rate=1.0)
                s.cache_dubbed_event(
                    DownstreamEvent(event_type="caption", utterance_id=1),
                    duration_ms=10.0,
                    pcm_crc32=1,
                )
                s.speaker_registry.match_or_register_speaker(emb)
                s.reset_on_seek(0.0)
        except Exception:
            errors.append(traceback.format_exc())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    # The stop flag MUST be set BEFORE joining, otherwise the workers spin
    # until their join timeout expires.
    stop.set()
    for t in threads:
        t.join(timeout=15.0)
    assert not any(t.is_alive() for t in threads)
    assert not errors, errors[0]


def test_v107_epoch_fence_snapshot_precedes_the_gpu_semaphore():
    # v10.7 (P0-B): worker_loop must snapshot the interrupt epoch BEFORE
    # awaiting the GPU semaphore. v10.6 snapshotted it inside the worker
    # thread, i.e. after the semaphore wait, so a seek that landed during
    # the wait was invisible to the fence.
    import inspect

    import backend.transport.server as server_mod

    source = inspect.getsource(server_mod)
    body = source[source.index("async def worker_loop"):]

    snap = body.index("pkt_epoch = queue.interrupt_epoch")
    sem = body.index("async with GPU_SEMAPHORE")
    check = body.index("queue.interrupt_epoch != pkt_epoch")

    # The snapshot must precede the semaphore, and the re-check must be
    # inside it.
    assert snap < sem, "epoch snapshot must precede the GPU semaphore"
    assert check > sem, "the epoch re-check must be inside the semaphore"


def test_v107_deferred_tail_syllables_become_true_noop_tokens():
    # v10.7: a deferred (wait-gated) tail syllable must decode to the (0, 0, 0)
    # no-op triplet, so decode_batch_to_text SKIPS it.
    #
    # v10.7 zeroed bus_l BEFORE the head projections. That is not sufficient:
    # nn.Linear has a bias vector, so a zero input still produces a non-zero
    # logit vector and the argmax returns a real onset id. The decoder then
    # emitted the deferred syllables as if they were real content.
    import torch

    from backend.contracts.schemas import CI_MICRO_CONFIG
    from backend.linguistics.syllable_decomposer import BilingualCodeSwitchDecomposer
    from backend.nn_cartridges.mouth_cartridge import KhmerMouthCartridge

    torch.manual_seed(0)
    mouth = KhmerMouthCartridge(CI_MICRO_CONFIG)
    mouth.eval()

    B = 1
    T = 24
    bus = torch.randn(B, T, 512)
    gate = torch.randn(B, T, 1)
    r_sps = torch.full((B,), 8.0)  # -> L_syll = 5, so the tail of 3 is waited on

    with torch.no_grad():
        _, batch = mouth.predict_syllables_and_align(
            bus,
            gate,
            r_sps,
            "km",
            1.0,
            T,
            is_hard_ceiling_slice=True,
            entropy_wait_threshold=-1.0,  # force every tail syllable to wait
        )

    o = batch.onset_ids[0].tolist()
    v = batch.vowel_ids[0].tolist()
    cd = batch.coda_ids[0].tolist()
    triplets = list(zip(o, v, cd))

    assert len(triplets) == 5
    # The three deferred tail positions must be exact (0, 0, 0) no-ops.
    assert triplets[-3:] == [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    assert mouth.last_deferred_syll_count == 3

    # The decoder must therefore emit only the two REAL syllables.
    decomp = BilingualCodeSwitchDecomposer()
    texts, cs_words = decomp.decode_batch_to_text(batch)
    assert len(texts) == 1
    assert texts[0].strip() != ""
    # A deferred syllable that leaked through would show up as extra content.
    assert len(texts[0]) < 12
    assert cs_words == []


def test_v107_aud107_deferred_syllables_consume_zero_frames():
    # v10.7 AUD-107-01: a deferred (wait-gated) syllable must be a COMPLETE
    # no-op: the (0, 0, 0) triplet AND sesqui_ids == 0, so the aligner gives it
    # zero duration weight and the diffusion model spends no acoustic frames
    # on a syllable that is not spoken in this clause.
    import torch

    from backend.contracts.schemas import CI_MICRO_CONFIG
    from backend.nn_cartridges.mouth_cartridge import KhmerMouthCartridge

    torch.manual_seed(0)
    mouth = KhmerMouthCartridge(CI_MICRO_CONFIG)
    mouth.eval()

    B, T = 1, 24
    bus = torch.randn(B, T, 512)
    gate = torch.randn(B, T, 1)
    r_sps = torch.full((B,), 8.0)

    with torch.no_grad():
        _, batch = mouth.predict_syllables_and_align(
            bus, gate, r_sps, "km", 1.0, T,
            is_hard_ceiling_slice=True, entropy_wait_threshold=-1.0,
        )

    o = batch.onset_ids[0].tolist()
    v = batch.vowel_ids[0].tolist()
    cd = batch.coda_ids[0].tolist()
    sq = batch.sesqui_ids[0].tolist()
    triplets = list(zip(o, v, cd))

    assert triplets[-3:] == [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    assert sq[-3:] == [0, 0, 0]
    durs = mouth.last_pred_durations[0]
    assert float(durs[-3]) == 0.0
    assert float(durs[-2]) == 0.0
    assert float(durs[-1]) == 0.0
    # The REAL syllables must still receive frames.
    assert float(durs[0]) > 0.0


def test_v107_aud107_km_to_en_path_also_masks_the_deferred_tail():
    # v10.7 AUD-107-01: v10.7 applied the no-op mask BEFORE the
    # `if target_lang == "en"` branch, but that branch overwrites the ids with
    # the reserved English ranges, so the deferred tail was never masked on a
    # KM -> EN slice and the words were duplicated across the clause boundary.
    import torch

    from backend.contracts.schemas import CI_MICRO_CONFIG
    from backend.nn_cartridges.mouth_cartridge import KhmerMouthCartridge

    torch.manual_seed(0)
    mouth = KhmerMouthCartridge(CI_MICRO_CONFIG)
    mouth.eval()

    B, T = 1, 24
    with torch.no_grad():
        _, batch = mouth.predict_syllables_and_align(
            torch.randn(B, T, 512),
            torch.randn(B, T, 1),
            torch.full((B,), 8.0),
            "en",
            1.0,
            T,
            is_hard_ceiling_slice=True,
            entropy_wait_threshold=-1.0,
        )

    triplets = list(
        zip(
            batch.onset_ids[0].tolist(),
            batch.vowel_ids[0].tolist(),
            batch.coda_ids[0].tolist(),
        )
    )
    assert triplets[-3:] == [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    assert mouth.last_deferred_syll_count == 3
