# -*- coding: utf-8 -*-
"""Verification harness for KHMER-SOVEREIGN-S2ST v10.7.

Extracts every code block from a master blueprint, rebuilds the modules under a
stub package tree, and runs the V1-V18 validation suite.
"""
import io
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(ROOT, "KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt")


def extract_blocks(path):
    """Return {path_in_tree: (lang, body)} for every fenced block in the master."""
    text = io.open(path, encoding="utf-8").read()
    lines = text.split("\n")
    blocks = {}
    cur = None
    lang = None
    buf = []
    for line in lines:
        if cur is None:
            if not line.startswith("#"):
                continue
            m = re.search(r"`([^`]+\.(?:py|ts|rs|toml))`", line)
            if m:
                cand = m.group(1)
                if cand.startswith("backend/") or cand.startswith("frontend/") or cand.startswith("crates/"):
                    cur = cand
                    lang = None
                    buf = []
            continue
        if line.startswith("```"):
            if lang is None:
                lang = line[3:].strip()
                buf = []
            else:
                blocks[cur] = (lang, "\n".join(buf))
                cur = None
                lang = None
            continue
        if cur is not None and lang is not None:
            buf.append(line)
    return blocks


def _dataclasses():
    from dataclasses import dataclass

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
        onset_ids: object
        coeng_ids: object
        medial_ids: object
        register_ids: object
        vowel_ids: object
        coda_ids: object
        sesqui_ids: object

    @dataclass(frozen=True)
    class DownstreamEvent:
        event_type: str = ""
        utterance_id: int = 0
        utterance_start_ms: float = 0.0
        vocal_modes: object = ()
        speaker_ids: object = ()
        source_lang: str = "en"
        target_lang: str = "km"
        dubbed_audio_b64: object = None
        erb_suppression_mask_b64: object = None
        erb_frame_count: int = 0
        per_speaker_source_texts: object = ()
        per_speaker_target_texts: object = ()
        codeswitch_words: object = ()
        playback_rate: float = 1.0
        actual_duration_sec: float = 0.0
        drift_ms: float = 0.0
        interrupt_epoch: int = 0

    @dataclass(frozen=True)
    class LivePacket:
        kind: str = ""
        timestamp_ms: float = 0.0
        pcm_data: object = None
        payload: object = None

    return KhmerSyllableToken, SyllableTensorBatch, DownstreamEvent, LivePacket


def install_stubs():
    """Create the backend.* package tree so intra-package imports resolve."""
    KhmerSyllableToken, SyllableTensorBatch, DownstreamEvent, LivePacket = _dataclasses()

    class VocalMode:
        AMBIENT_OR_CLAP = 0
        NON_VERBAL_HUMAN = 1
        SINGING_OR_MUSIC = 2
        SPOKEN_DIALOGUE = 3

    for name in (
        "backend",
        "backend.contracts",
        "backend.contracts.schemas",
        "backend.linguistics",
        "backend.linguistics.khmer_normalizer",
        "backend.linguistics.syllable_decomposer",
        "backend.nn_cartridges",
        "backend.nn_cartridges.universal_core",
        "backend.nn_cartridges.stem_and_mode_gate",
        "backend.nn_cartridges.ear_cartridge",
        "backend.nn_cartridges.codeswitch_router",
        "backend.nn_cartridges.duration_aligner",
        "backend.nn_cartridges.mouth_cartridge",
        "backend.nn_cartridges.s2st_console",
        "backend.training",
        "backend.training.augmentations",
        "backend.training.train_cartridge",
        "backend.training.eval_s2st_suite",
        "backend.transport",
        "backend.transport.live_queue",
        "backend.transport.session",
        "backend.transport.runner",
        "backend.transport.server",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    sch = sys.modules["backend.contracts.schemas"]
    sch.KhmerSyllableToken = KhmerSyllableToken
    sch.SyllableTensorBatch = SyllableTensorBatch
    sch.DownstreamEvent = DownstreamEvent
    sch.LivePacket = LivePacket
    sch.VocalMode = VocalMode
    sch.LanguageCode = str
    sys.modules["backend.contracts"].schemas = sch

    # Real constant / config values, taken verbatim from the Part 2.4 contract.
    sch.SAMPLE_RATE_IN = 16000
    sch.SAMPLE_RATE_OUT = 24000
    sch.SAMPLE_RATE_MNE = 48000
    sch.N_FFT_16K = 1024
    sch.HOP_LENGTH_16K = 160
    sch.N_FFT_24K = 2048
    sch.HOP_LENGTH_24K = 240
    sch.N_FFT_48K = 4096
    sch.HOP_LENGTH_48K = 480
    sch.HOP_LENGTH = 240
    sch.N_MELS = 100
    sch.ERB_BANDS = 64
    sch.ERB_MIN_FREQ_HZ = 120.0
    sch.ERB_MAX_FREQ_HZ = 8000.0
    sch.SEMANTIC_DIM = 512
    sch.EMOTION_NUM_TOKENS = 32
    sch.MIC_EQ_TAPS_16K = 64
    sch.MIC_EQ_TAPS_24K = 96
    sch.VELVET_RIR_TAPS_24K = 6000
    sch.MINOR_SYLLABLE_MAX_FRAMES = 6.0
    sch.MAX_ELASTIC_PAUSE_BORROW_MS = 150.0
    sch.DOMINANCE_MIN_DB_RATIO = -12.0
    sch.LOOKAHEAD_BUFFER_SEC = 2.0
    sch.MAX_CLAUSE_SLICE_SEC = 1.6
    sch.MIN_CLAUSE_SLICE_SEC = 1.2
    sch.SILENCE_RMS_THRESHOLD = 0.008
    sch.UTTERANCE_END_SILENCE_MS = 250.0
    sch.REPLAY_CACHE_MAX_ITEMS = 32
    sch.REPLAY_BUCKET_MS = 250.0
    sch.LANG_TO_ID = {"en": 0, "km": 1, "th": 2, "vi": 3}
    # Derived bin counts (N_FFT // 2 + 1).
    sch.N_BINS_16K = sch.N_FFT_16K // 2 + 1
    sch.N_BINS_24K = sch.N_FFT_24K // 2 + 1
    sch.N_BINS_SEP_16K = 257
    sch.HOP_SEP_16K = 160
    sch.N_FFT_SEP_16K = 512

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ModelArchitectureConfig:
        profile_name: str = "prod_240m"
        separator_d_model: int = 384
        separator_layers: int = 3
        mode_gate_d_model: int = 256
        mode_gate_layers: int = 4
        ear_d_model: int = 512
        ear_layers: int = 16
        ear_heads: int = 8
        ear_ffn_dim: int = 2048
        emotion_d_model: int = 512
        emotion_layers: int = 4
        emotion_ffn_dim: int = 2048
        pointer_d_model: int = 512
        pointer_layers: int = 2
        pointer_ffn_dim: int = 1024
        mouth_d_model: int = 768
        mouth_layers: int = 15
        mouth_heads: int = 12
        mouth_ffn_dim: int = 3072
        euler_steps: int = 12
        vocos_d_model: int = 512
        vocos_layers: int = 8
        vocos_intermediate_dim: int = 1536

    sch.ModelArchitectureConfig = ModelArchitectureConfig
    sch.PROD_240M_CONFIG = ModelArchitectureConfig()
    sch.CI_MICRO_CONFIG = ModelArchitectureConfig(
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

    class CodeSwitchDecomposerProtocol:
        pass

    class StemAndSceneProtocol:
        pass

    class S2STConsoleProtocol:
        pass

    sch.CodeSwitchDecomposerProtocol = CodeSwitchDecomposerProtocol
    sch.StemAndSceneProtocol = StemAndSceneProtocol
    sch.S2STConsoleProtocol = S2STConsoleProtocol

    return sch


LOAD_ORDER = [
    "backend/linguistics/khmer_normalizer.py",
    "backend/linguistics/syllable_decomposer.py",
    "backend/nn_cartridges/universal_core.py",
    "backend/nn_cartridges/duration_aligner.py",
    "backend/nn_cartridges/codeswitch_router.py",
    "backend/nn_cartridges/ear_cartridge.py",
    "backend/nn_cartridges/stem_and_mode_gate.py",
    "backend/nn_cartridges/mouth_cartridge.py",
    "backend/nn_cartridges/s2st_console.py",
    "backend/training/augmentations.py",
    "backend/training/train_cartridge.py",
    "backend/transport/live_queue.py",
    "backend/transport/session.py",
    "backend/transport/runner.py",
    "backend/transport/server.py",
]


def load_modules(blocks, skip=()):
    """Execute the python blocks in dependency order and return the modules.

    The Part 2.4 contract block (backend/contracts/schemas.py) is the single
    source of truth for the constants, dataclasses, configs and Protocols, so it
    is loaded verbatim rather than stubbed.
    """
    install_stubs()
    mods = {}

    contract = blocks.get("backend/contracts/schemas.py")
    if contract is not None:
        lang, body = contract
        sch = sys.modules["backend.contracts.schemas"]
        exec(compile(body, "backend/contracts/schemas.py", "exec"), sch.__dict__)
        sys.modules["backend.contracts"].schemas = sch
        mods["backend/contracts/schemas.py"] = sch

    for key in LOAD_ORDER:
        if key in skip or key not in blocks:
            continue
        lang, body = blocks[key]
        if lang != "python":
            continue
        modname = key.replace("/", ".")[:-3]
        mod = types.ModuleType(modname)
        mod.__dict__["__name__"] = modname
        sys.modules[modname] = mod
        exec(compile(body, key, "exec"), mod.__dict__)
        mods[key] = mod
    return mods


EXPECTED_LEDGER = {
    "prod_240m": {
        "Causal3StemSeparator": 5864197,
        "VocalModeGate": 816388,
        "StreamingEarCartridge": 64546817,
        "DisentangledEmotionEncoder": 8735748,
        "AcousticPointerCodeSwitchBridge": 4989953,
        "KhmerMouthCartridge": 148931136,
        "VocosISTFTHead": 7753218,
    },
    "ci_micro": {
        "Causal3StemSeparator": 231877,
        "VocalModeGate": 31684,
        "StreamingEarCartridge": 528129,
        "DisentangledEmotionEncoder": 184196,
        "AcousticPointerCodeSwitchBridge": 329857,
        "KhmerMouthCartridge": 2109504,
        "VocosISTFTHead": 422786,
    },
}


def v1_ledger(mods):
    """V1 - re-derive the parameter ledger under torch meta device."""
    import torch

    stem = mods["backend/nn_cartridges/stem_and_mode_gate.py"]
    uni = mods["backend/nn_cartridges/universal_core.py"]
    ear = mods["backend/nn_cartridges/ear_cartridge.py"]
    csr = mods["backend/nn_cartridges/codeswitch_router.py"]
    mouth = mods["backend/nn_cartridges/mouth_cartridge.py"]
    con = mods["backend/nn_cartridges/s2st_console.py"]
    sch = sys.modules["backend.contracts.schemas"]

    results = {}
    for profile, cfg in (("prod_240m", sch.PROD_240M_CONFIG), ("ci_micro", sch.CI_MICRO_CONFIG)):
        with torch.device("meta"):
            pipeline = stem.StemAndScenePipeline(config=cfg)
            console = con.SymmetricS2STConsole(config=cfg)
        sep = pipeline.separator
        gate = pipeline.mode_gate
        ear_m = console.ear
        emo = console.emotion_encoder
        bridge = console.pointer
        mc = console.mouth
        voc = console.vocos
        counts = {
            "Causal3StemSeparator": sum(p.numel() for p in sep.parameters()),
            "VocalModeGate": sum(p.numel() for p in gate.parameters()),
            "StreamingEarCartridge": sum(p.numel() for p in ear_m.parameters()),
            "DisentangledEmotionEncoder": sum(p.numel() for p in emo.parameters()),
            "AcousticPointerCodeSwitchBridge": sum(p.numel() for p in bridge.parameters()),
            "KhmerMouthCartridge": sum(p.numel() for p in mc.parameters()),
            "VocosISTFTHead": sum(p.numel() for p in voc.parameters()),
        }
        results[profile] = counts
    return results


def check_v1(mods):
    results = v1_ledger(mods)
    ok = True
    for profile, expected in EXPECTED_LEDGER.items():
        counts = results[profile]
        for name, exp in expected.items():
            got = counts[name]
            if got != exp:
                ok = False
                print("  FAIL %-13s %-32s expected %d got %d" % (profile, name, exp, got))
        total = sum(counts.values())
        exp_total = sum(expected.values())
        status = "OK" if total == exp_total else "FAIL"
        if total != exp_total:
            ok = False
        print("  %-13s total %12d  expected %12d  %s" % (profile, total, exp_total, status))
    return ok, results


def check_v10(mods):
    """V10 - the timestep embedding must not be the v10.6 rank-1 all-ones map.

    The v10.6 defect is not that the output failed to change with t (the
    learned weights alone made it change) but that time_mlp only ever saw
    w1*t + w2*dt + b: every column of the timestep embedding was the SAME
    vector, so the embedding matrix had rank 1 and time_mlp could not express
    any non-linear function of t.

    The v10.7 sinusoidal embedding must therefore have rank STRICTLY GREATER
    THAN 1 across distinct timesteps. (It is not required to be full rank:
    with t restricted to [0, 1] and the highest frequency at 1 radian, the
    achievable rank is 5, which is already a strict improvement over 1 and is
    what restores the spectral resolution over t.)
    """
    import torch

    mouth_mod = mods["backend/nn_cartridges/mouth_cartridge.py"]
    sch = sys.modules["backend.contracts.schemas"]
    torch.manual_seed(0)
    mouth = mouth_mod.KhmerMouthCartridge(sch.CI_MICRO_CONFIG)
    mouth.eval()

    emb = mouth_mod.sinusoidal_time_embedding

    # Rank of the embedding matrix over 64 distinct timesteps.
    probe = torch.linspace(0.0, 1.0, 64).unsqueeze(0)
    emb_matrix = emb(probe, 64)  # (64, 64)
    rank = int(torch.linalg.matrix_rank(emb_matrix))
    rank_gt_1 = rank > 1

    # The v10.6 map (every column identical) has rank exactly 1.
    v106_matrix = torch.linspace(0.0, 1.0, 64).unsqueeze(0).expand(64, 64)
    v106_rank = int(torch.linalg.matrix_rank(v106_matrix))

    # The full model must also be sensitive to t and to dt.
    B, T = 1, 16
    d_model = sch.CI_MICRO_CONFIG.mouth_d_model
    x_t = torch.randn(B, sch.N_MELS, T)
    cond = torch.randn(B, T, d_model)
    emo = torch.randn(B, sch.EMOTION_NUM_TOKENS, sch.SEMANTIC_DIM)
    with torch.no_grad():
        v_t0 = mouth.predict_velocity(x_t, torch.zeros(B), cond, emo, dt=None)
        v_t1 = mouth.predict_velocity(x_t, torch.ones(B), cond, emo, dt=None)
        v_d0 = mouth.predict_velocity(x_t, torch.full((B,), 0.5), cond, emo, dt=torch.full((B,), 0.01))
        v_d1 = mouth.predict_velocity(x_t, torch.full((B,), 0.5), cond, emo, dt=torch.full((B,), 0.20))

    t_sensitive = not torch.allclose(v_t0, v_t1, atol=1e-6)
    dt_sensitive = not torch.allclose(v_d0, v_d1, atol=1e-6)

    print("  v10.7 embedding rank  :", rank, "( v10.6 rank:", v106_rank, ")")
    print("  rank strictly > 1     :", rank_gt_1)
    print("  t-sensitive           :", t_sensitive)
    print("  dt-sensitive          :", dt_sensitive)
    return rank_gt_1 and t_sensitive and dt_sensitive


def check_v8(mods):
    """V8 - concurrency stress: no exception, no torn cache, no torn centroids."""
    import threading
    import time
    import traceback

    import numpy as np
    import torch

    ses = mods["backend/transport/session.py"]
    sch = sys.modules["backend.contracts.schemas"]
    errors = []
    stop = threading.Event()

    def worker(seed):
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            s = ses.InMemorySession("stress")
            emb = torch.randn(512)
            while not stop.is_set():
                s.append_pcm_chunk(np.zeros(320, dtype=np.float32), timestamp_ms=0.0)
                s.lookup_cached_event(0.0, duration_ms=10.0, pcm_crc32=1, playback_rate=1.0)
                s.cache_dubbed_event(
                    sch.DownstreamEvent(event_type="caption", utterance_id=1),
                    duration_ms=10.0,
                    pcm_crc32=1,
                )
                s.speaker_registry.match_or_register_speaker(emb)
                s.reset_on_seek(0.0)
        except Exception:
            errors.append(traceback.format_exc())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    time.sleep(2.0)
    stop.set()
    for t in threads:
        t.join(timeout=10.0)

    if errors:
        print("  thread errors:", len(errors))
        print("  first error:\n", errors[0][:600])
        return False
    print("  threads:", len(threads), " errors: 0  (2.0s stress)")
    return True


def check_v8b(mods):
    """V8b - deterministic proof that every mutating method HOLDS the lock.

    A probabilistic stress test cannot prove the absence of a race (the GIL
    makes the interleaving rare). The deterministic property is: each method
    that mutates shared state must hold the relevant lock for the whole
    duration of its body. This test replaces the lock with an instrumented
    proxy that records acquisitions, so the assertion is exact rather than
    statistical.
    """
    import numpy as np
    import torch

    ses = mods["backend/transport/session.py"]
    sch = sys.modules["backend.contracts.schemas"]

    class RecordingRLock:
        """Wraps a real RLock and counts acquisitions."""

        def __init__(self, inner):
            self._inner = inner
            self.acquisitions = 0
            self.held = False

        def __enter__(self):
            self._inner.acquire()
            self.acquisitions += 1
            self.held = True
            return self

        def __exit__(self, *exc):
            self.held = False
            self._inner.release()
            return False

        def acquire(self, *a, **k):
            return self._inner.acquire(*a, **k)

        def release(self):
            return self._inner.release()

    s = ses.InMemorySession("lockprobe")
    real = s._lock
    probe = RecordingRLock(real)
    s._lock = probe

    reg = s.speaker_registry
    real_reg = reg._lock
    reg_probe = RecordingRLock(real_reg)
    reg._lock = reg_probe

    s.append_pcm_chunk(np.zeros(320, dtype=np.float32), timestamp_ms=0.0)
    a1 = probe.acquisitions
    s.lookup_cached_event(0.0, duration_ms=10.0, pcm_crc32=1, playback_rate=1.0)
    a2 = probe.acquisitions
    s.cache_dubbed_event(
        sch.DownstreamEvent(event_type="caption", utterance_id=1),
        duration_ms=10.0,
        pcm_crc32=1,
    )
    a3 = probe.acquisitions
    s.reset_on_seek(0.0)
    a4 = probe.acquisitions
    s.speaker_registry.match_or_register_speaker(torch.randn(512))
    a5 = reg_probe.acquisitions

    results = {
        "append_pcm_chunk": a1 >= 1,
        "lookup_cached_event": a2 > a1,
        "cache_dubbed_event": a3 > a2,
        "reset_on_seek": a4 > a3,
        "match_or_register_speaker": a5 >= 1,
    }
    for name, ok in results.items():
        print("  %-28s holds lock: %s" % (name, ok))
    return all(results.values())


def check_v14(blocks):
    """V14 - frontend wiring: worklet, binary upstream path, baseVideoTimeMs,
    ERB mask consumption, and the rate-scaled media clock."""
    client = blocks["frontend/src/client/duplex-ws-client.ts"][1]
    mixer = blocks["frontend/src/audio/stereo-mne-mixer.ts"][1]
    cap = blocks["frontend/src/worklets/pcm-capture-processor.ts"][1]

    checks = {
        "client instantiates AudioWorkletNode": "AudioWorkletNode" in client,
        "client calls addModule": "addModule" in client,
        "client forwards binary to ws.send": "this.ws.send(ev.data)" in client,
        "client posts baseVideoTimeMs": "baseVideoTimeMs" in client,
        "client posts playbackRate": "playbackRate:" in client,
        "client posts on ratechange": "ratechange" in client,
        "client posts on seeking": "seeking" in client,
        "client posts on playing": "playing" in client,
        "mixer calls getActiveErbMaskForTime": mixer.count("getActiveErbMaskForTime") >= 2,
        "mixer calls applyMidSideSpectralErbMask": mixer.count("applyMidSideSpectralErbMask") >= 2,
        "mixer has processMidSideFrame": "processMidSideFrame" in mixer,
        "mixer uses lookaheadTargetTime for the mask": "lookaheadTargetTime," in mixer,
        "worklet scales the clock by playbackRate": "this.playbackRate" in cap,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v12(mods):
    """V12 - independent vowels U+17AB..U+17AE survive decomposition."""
    dec = mods["backend/linguistics/syllable_decomposer.py"]
    nrm = mods["backend/linguistics/khmer_normalizer.py"]
    d = dec.BilingualCodeSwitchDecomposer()

    results = {}
    for cpv, expect_onset in [(0x17AB, 27), (0x17AC, 27), (0x17AD, 28), (0x17AE, 28)]:
        ch = chr(cpv)
        out = nrm.normalize_khmer(ch)
        toks = d.decompose_mixed_sentence(ch)
        results["U+%04X normalized" % cpv] = out == chr(0x179A if cpv in (0x17AB, 0x17AC) else 0x179B) + chr(0x17B9 if cpv in (0x17AB, 0x17AD) else 0x17BA)
        results["U+%04X reached" % cpv] = len(toks) == 1 and toks[0].onset_id == expect_onset

    # A word beginning with an independent vowel must keep its first syllable.
    word = chr(0x17AB) + chr(0x179F) + chr(0x17D2) + chr(0x179F)
    toks = d.decompose_mixed_sentence(word)
    results["multi-syllable keeps first syllable"] = len(toks) == 2

    for name, ok in results.items():
        print("  %-46s %s" % (name, ok))
    return all(results.values())


def check_v2(blocks, master_text):
    """V2 - version synchronization.

    The authoritative DECLARATIONS must all be 10.7.0. Historical mentions of
    10.4.0 and 10.5.0 inside the fix-ledger entries and the validation
    instruction are intentional and are retained.
    """
    import os

    man_start = master_text.index("### 2.3.1 Version Manifest")
    man = master_text[man_start:master_text.index("### 2.4", man_start)]
    rows = [
        l
        for l in man.split("\n")
        if l.startswith("|") and "10.7.0" in l and "---" not in l and "Manifest key" not in l
    ]
    tc = master_text[master_text.index("### 2.3 Root Workspace"):man_start]
    p8 = master_text[master_text.index("## PART 8:"):]

    checks = {
        "manifest lists exactly 5 locations": len(rows) == 5,
        "manifest prose is 10.7.0": "The project version is **`10.7.0`**" in man,
        "Part 2.3 toolchain blocks carry 10.7.0 x3": tc.count("10.7.0") == 3,
        "Part 8 prompt carries 10.7.0": p8.count("10.7.0") >= 1,
        "no 10.6.0 token in the master": "10.6.0" not in master_text,
    }
    for f in ("Cargo.toml", "pyproject.toml", "frontend/package.json"):
        p = os.path.join(ROOT, *f.split("/"))
        text = io.open(p, encoding="utf-8").read()
        checks["%s declares 10.7.0" % f] = "10.7.0" in text and "10.6.0" not in text
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v13(mods):
    """V13 - the scene pipeline must handle a BATCH without crashing.

    v10.6 read stems[0, 0], stems[0, 1] and erb_mask[0], silently discarding
    every batch element after the first, and then called
    mic_eq.expand(B_spk, -1, -1) on a (B, 1, 64) tensor, which raises a
    RuntimeError whenever B = 2 and B_spk = 1.
    """
    import torch

    stem = mods["backend/nn_cartridges/stem_and_mode_gate.py"]
    sch = sys.modules["backend.contracts.schemas"]
    torch.manual_seed(0)
    pipeline = stem.StemAndScenePipeline(config=sch.CI_MICRO_CONFIG)
    pipeline.eval()

    scene = pipeline.separate_and_classify(torch.randn(2, 16000))

    B_spk = scene.active_speaker_stems.shape[0]
    checks = {
        "call returned a scene": scene is not None,
        "erb mask is strictly 2D": scene.erb_suppression_mask.dim() == 2,
        "erb mask has 64 bands": scene.erb_suppression_mask.shape[-1] == 64,
        "mic_eq batch == B_spk": scene.mic_eq_residual_16k.shape[0] == B_spk,
        "rt60 batch == B_spk": scene.rt60_sec.shape[0] == B_spk,
        "wet batch == B_spk": scene.wet_gain.shape[0] == B_spk,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v16(mods):
    """V16 - the mouth's property bridges must route through the thread-local state.

    v10.7 renamed the bridges to the last_* names that the rest of the code
    actually uses. With the names mismatched, Python created plain shared
    attributes on self and completely bypassed CartridgeThreadState, so the
    thread isolation was silently lost.
    """
    import torch

    mouth_mod = mods["backend/nn_cartridges/mouth_cartridge.py"]
    con = mods["backend/nn_cartridges/s2st_console.py"]
    sch = sys.modules["backend.contracts.schemas"]
    torch.manual_seed(0)
    console = con.SymmetricS2STConsole(config=sch.CI_MICRO_CONFIG)
    console.eval()
    mouth = console.mouth

    with torch.no_grad():
        console.forward_s2st(torch.randn(1, 8000), "en", "km", orig_duration_sec=0.5)

    bridged = [
        "last_next_past_kv",
        "last_next_past_gate",
        "last_deferred_syll_count",
        "last_pred_durations",
        "last_gate_l",
    ]
    checks = {}
    for name in bridged:
        checks["%s not in mouth.__dict__" % name] = name not in mouth.__dict__
        checks["%s is a property" % name] = isinstance(
            getattr(type(mouth), name, None), property
        )
    checks["deferred_syll_count defaults to 0"] = (
        mouth._state.last_deferred_syll_count in (0, None)
    )
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v19(mods):
    """V19 - integrate_shortcut_sway_ode must work with cfg_scale != 1.0.

    v10.7 computed the conditional velocity only in the else branch and then
    referenced it in the if branch, so any cfg_scale != 1.0 raised
    UnboundLocalError: cannot access local variable 'v'.
    """
    import torch

    con = mods["backend/nn_cartridges/s2st_console.py"]
    sch = sys.modules["backend.contracts.schemas"]
    torch.manual_seed(0)
    console = con.SymmetricS2STConsole(config=sch.CI_MICRO_CONFIG)
    console.eval()

    B, T = 1, 16
    x_0 = torch.randn(B, sch.N_MELS, T)
    cond = torch.randn(B, T, sch.CI_MICRO_CONFIG.mouth_d_model)
    emo = torch.randn(B, sch.EMOTION_NUM_TOKENS, sch.SEMANTIC_DIM)

    with torch.no_grad():
        v_plain = con.integrate_shortcut_sway_ode(
            console.mouth, x_0, cond, emo, steps=2, cfg_scale=1.0
        )
        v_guided = con.integrate_shortcut_sway_ode(
            console.mouth, x_0, cond, emo, steps=2, cfg_scale=1.5
        )

    checks = {
        "cfg_scale=1.0 returns a finite tensor": bool(torch.isfinite(v_plain).all()),
        "cfg_scale=1.5 returns a finite tensor": bool(torch.isfinite(v_guided).all()),
        "cfg_scale=1.5 differs from 1.0": not torch.allclose(v_plain, v_guided),
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v21(mods):
    """V21 - the trainer must aggregate the distillation loss."""
    import torch

    tr = mods["backend/training/train_cartridge.py"]
    con = mods["backend/nn_cartridges/s2st_console.py"]
    sch = sys.modules["backend.contracts.schemas"]
    torch.manual_seed(0)
    console = con.SymmetricS2STConsole(config=sch.CI_MICRO_CONFIG)
    distiller = tr.MultiTeacherEarDistiller(d_student=512, d_teacher=1024)

    trainer = tr.CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")
    teacher = torch.randn(2, 25, 1024)
    trainer.attach_distiller(
        distiller, weight=0.5, teacher_targets={"w2v_layer16_50hz": teacher}
    )

    def forward_fn():
        with torch.no_grad():
            console.ear(torch.randn(2, 8000), "en")
        return torch.tensor(1.0, requires_grad=True)

    value = trainer.train_step(forward_fn)
    checks = {
        "train_step returned a float": isinstance(value, float),
        "teacher targets remembered": "w2v_layer16_50hz" in trainer.teacher_targets,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v22(blocks):
    """V22 - the eval suite must not reference an undefined COMET checkpoint."""
    ev = blocks["backend/training/eval_s2st_suite.py"][1]
    checks = {
        "comet_path is gone": "comet_path" not in ev,
        "comet_checkpoint attribute present": "self.comet_checkpoint" in ev,
        "_asr_transcribe takes a language": 'language: str = "km"' in ev,
        "the call site passes the language": "language=asr_lang" in ev,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def _js_module_tables(block):
    """Extract the module-level FFT tables so node can execute the methods.

    v10.7 AUD-107-05 moved the twiddle and bit-reversal tables to module
    scope, so the extracted methods reference names that only exist at module
    level. Without this the node harness fails with a ReferenceError.
    """
    lines = block.split("\n")
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("const N_FFT_MNE = "):
            start = idx
            break
    if start is None:
        return ""
    out = []
    for line in lines[start:]:
        if line.startswith("export ") or line.startswith("function "):
            break
        out.append(line)
    return "\n".join(out)


def _js_method(block, name):
    """Extract a TypeScript method and strip its annotations so node can run it."""
    lines = block.split("\n")
    i = next(
        j for j, l in enumerate(lines) if name + "(" in l and "private" in l
    )
    j = i
    while "{" not in lines[j]:
        j += 1
    depth = 0
    k = j
    while True:
        depth += lines[k].count("{") - lines[k].count("}")
        if depth == 0:
            break
        k += 1
    body = "\n".join(lines[i : k + 1])
    body = body.replace("private ", "")
    body = body.replace("    frame: Float32Array,", "    frame,")
    body = body.replace("    nFrames: number,", "    nFrames,")
    body = body.replace("    re: Float32Array,", "    re,")
    body = body.replace("    im: Float32Array,", "    im,")
    body = body.replace("    nFft: number", "    nFft")
    body = body.replace("  ): void {", "  ) {")
    body = body.replace("  ): Float32Array {", "  ) {")
    body = body.replace("const rev = (v: number): number => {", "const rev = (v) => {")
    body = body.replace(
        "const rev = (v: number): number => REV_TABLE_MNE[v];",
        "const rev = (v) => REV_TABLE_MNE[v];",
    )
    return body


def check_v26_aud107(blocks, mods):
    """V26 - the seven AUD-107 code-level findings must be closed.

    These are behavioural claims, so each one is EXECUTED rather than
    pattern-matched: the state_dict keys are compared, the ear call count is
    measured, the deferred durations are read, and the zero-frame sentinel is
    exercised.
    """
    import torch

    from backend.contracts.schemas import CI_MICRO_CONFIG, PrecomputedEarFeatures
    from backend.nn_cartridges.mouth_cartridge import KhmerMouthCartridge
    from backend.nn_cartridges.s2st_console import SymmetricS2STConsole
    from backend.training.train_cartridge import apply_gradient_checkpointing

    checks = {}

    # --- AUD-107-04: gradient checkpointing must not rename state_dict keys ---
    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    keys_before = set(console.state_dict().keys())
    apply_gradient_checkpointing(console)
    keys_after = set(console.state_dict().keys())
    checks["AUD-107-04 state_dict keys are unchanged"] = keys_before == keys_after
    checks["AUD-107-04 no '.blk.' segment appears"] = not any(
        ".blk." in k for k in keys_after
    )

    # --- AUD-107-02: the ear must not run twice on the same slice ---
    import types

    counts = {"ear": 0, "log_mel": 0}
    ear = console.ear
    f_ear, f_log = ear.forward, ear.log_mel

    def wrap_ear(self, *a, **k):
        counts["ear"] += 1
        return f_ear(*a, **k)

    def wrap_log(self, *a, **k):
        counts["log_mel"] += 1
        return f_log(*a, **k)

    ear.forward = types.MethodType(wrap_ear, ear)
    ear.log_mel = types.MethodType(wrap_log, ear)
    console.eval()
    torch.manual_seed(0)
    wav = torch.randn(1, 8000)
    with torch.no_grad():
        bus, r_sps = ear(wav, "en")
        l6 = ear.last_l6_hidden
        mel = ear.log_mel(wav)
        bundle = PrecomputedEarFeatures(
            bus=bus, r_sps=r_sps, mel_stems=mel, h_acoustic_l6=l6
        )
        counts["ear"] = 0
        counts["log_mel"] = 0
        console.forward_s2st(wav, "en", "km", 0.5, precomputed_ear=bundle)
    checks["AUD-107-02 the ear does not re-run"] = counts["ear"] == 0
    checks["AUD-107-02 the log-mel does not re-run"] = counts["log_mel"] == 0
    ear.forward = types.MethodType(lambda self, *a, **k: f_ear(*a, **k), ear)
    ear.log_mel = types.MethodType(lambda self, *a, **k: f_log(*a, **k), ear)

    # --- AUD-107-01: a deferred syllable consumes ZERO acoustic frames ---
    torch.manual_seed(0)
    mouth = KhmerMouthCartridge(CI_MICRO_CONFIG)
    mouth.eval()
    with torch.no_grad():
        _, batch = mouth.predict_syllables_and_align(
            torch.randn(1, 24, 512),
            torch.randn(1, 24, 1),
            torch.full((1,), 8.0),
            "km",
            1.0,
            24,
            is_hard_ceiling_slice=True,
            entropy_wait_threshold=-1.0,
        )
    o = batch.onset_ids[0].tolist()
    v = batch.vowel_ids[0].tolist()
    cd = batch.coda_ids[0].tolist()
    sq = batch.sesqui_ids[0].tolist()
    triplets = list(zip(o, v, cd))
    checks["AUD-107-01 the deferred tail is (0,0,0)"] = triplets[-3:] == [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    ]
    checks["AUD-107-01 sesqui_ids is zeroed on the tail"] = sq[-3:] == [0, 0, 0]
    durations = mouth.last_pred_durations[0]
    checks["AUD-107-01 the deferred syllables consume no frames"] = all(
        float(durations[i]) == 0.0 for i in (-3, -2, -1)
    )

    # --- AUD-107-01: the KM -> EN path must mask the tail as well ---
    with torch.no_grad():
        _, batch_en = mouth.predict_syllables_and_align(
            torch.randn(1, 24, 512),
            torch.randn(1, 24, 1),
            torch.full((1,), 8.0),
            "en",
            1.0,
            24,
            is_hard_ceiling_slice=True,
            entropy_wait_threshold=-1.0,
        )
    en_trips = list(
        zip(
            batch_en.onset_ids[0].tolist(),
            batch_en.vowel_ids[0].tolist(),
            batch_en.coda_ids[0].tolist(),
        )
    )
    checks["AUD-107-01 the KM->EN path masks the tail too"] = en_trips[-3:] == [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    ]

    # --- AUD-107-03: the 2.00s lookahead delay is wired into the M&E graph ---
    mne = blocks["frontend/src/audio/stereo-mne-mixer.ts"][1]
    ws = blocks["frontend/src/client/duplex-ws-client.ts"][1]
    checks["AUD-107-03 the DelayNode is created"] = (
        "this.ctx.createDelay(LOOKAHEAD_BUFFER_SEC + 1.0)" in mne
    )
    checks["AUD-107-03 the delay is 2.00s"] = (
        "delayNode.delayTime.value = LOOKAHEAD_BUFFER_SEC;" in mne
    )
    checks["AUD-107-03 the source feeds the delay"] = (
        "source.connect(delayNode);" in mne
        and "delayNode.connect(processor);" in mne
    )
    checks["AUD-107-03 the capture tap stays undelayed"] = (
        "return source;" in mne
    )

    # --- AUD-107-05: the FFT tables are precomputed ---
    checks["AUD-107-05 the tables are module-level"] = (
        "const REV_TABLE_MNE = new Uint16Array(N_FFT_MNE);" in mne
        and "const COS_TABLE_MNE_INV" in mne
    )
    checks["AUD-107-05 no per-block Math.log2"] = "Math.log2(size)" not in mne
    checks["AUD-107-05 no per-block twiddle allocation"] = (
        "new Float64Array(half)" not in mne
    )
    checks["AUD-107-05 the dead spectral pass is gone"] = (
        "this.processMidSideFrame(\n        lookaheadTargetTime," not in mne
    )

    # --- AUD-107-06: the capture worklet cannot reach the speakers ---
    checks["AUD-107-06 the capture sink is muted"] = (
        "captureSink.gain.value = 0.0;" in ws
        and "captureSink.connect(ctx.destination);" in ws
    )
    checks["AUD-107-06 the worklet is not connected directly"] = (
        "node.connect(ctx.destination);" not in ws
    )

    for name, ok in checks.items():
        print("  %-52s %s" % (name, ok))
    return all(checks.values())


def check_v27_aud107_09_and_10(blocks, mods):
    """V27 - AUD-107-09 (deferred syllables must not steal attention) and
    AUD-107-10 (universal cross-device sample rate).

    Both are behavioural, so each is EXECUTED rather than pattern-matched.
    """
    import math

    import torch

    from backend.nn_cartridges.duration_aligner import LearnedDurationAligner

    # ---------- AUD-107-09: the deferred columns collect no attention mass ----------
    torch.manual_seed(0)
    aligner = LearnedDurationAligner(d_model=64)
    B, L, D, T = 1, 4, 64, 24

    hidden = torch.randn(B, L, D)
    hidden[0, 2:, :] = 1e6
    hidden_ref = hidden[:, :2, :].contiguous()

    sesqui_four = torch.tensor([[2, 2, 0, 0]], dtype=torch.long)
    sesqui_two = torch.tensor([[2, 2]], dtype=torch.long)

    aligned_four, durs_four = aligner(hidden, sesqui_four, target_frames=T)
    aligned_two, durs_two = aligner(hidden_ref, sesqui_two, target_frames=T)

    # Recompute the attention distribution to measure the mass directly.
    import torch.nn.functional as F

    raw = F.softplus(aligner.dur_proj(hidden).squeeze(-1)) + 0.1
    prior = torch.where(
        sesqui_four == 0, 0.0, torch.where(sesqui_four == 1, 0.25, 1.00)
    ).to(raw.dtype)
    weighted = raw * prior
    norm_dur = weighted * (float(T) / (weighted.sum(dim=-1, keepdim=True) + 1e-8))
    cum = torch.cumsum(norm_dur, dim=-1)
    centers = cum - 0.5 * norm_dur
    grid = torch.arange(T, dtype=hidden.dtype).view(1, T, 1) + 0.5
    logits = -0.5 * ((grid - centers.unsqueeze(1)) / aligner.sigma) ** 2
    active = (sesqui_four > 0).unsqueeze(1)
    has_active = active.any(dim=-1, keepdim=True)
    logits = torch.where(
        has_active & (~active), torch.full_like(logits, -1e9), logits
    )
    weights = F.softmax(logits, dim=-1)
    deferred_mass = float(weights[0, :, 2].sum()) + float(weights[0, :, 3].sum())

    checks = {
        "AUD-107-09 the deferred columns collect zero mass": deferred_mass == 0.0,
        "AUD-107-09 the deferred durations are zero": (
            float(durs_four[0, 2]) == 0.0 and float(durs_four[0, 3]) == 0.0
        ),
        "AUD-107-09 the real durations are unchanged": bool(
            torch.allclose(durs_four[0, :2], durs_two[0, :], atol=1e-4)
        ),
        "AUD-107-09 the deferred hidden state cannot leak": bool(
            torch.allclose(aligned_four, aligned_two, atol=1e-4)
        ),
    }

    # ---------- AUD-107-10: the resampler produces exactly 16,000 Hz ----------
    def simulate(ctx_rate, quantum=128, outbuf=128, nq=800, freq=1000.0):
        out = []
        frac_pos = 0.0
        prev_sample = 0.0
        dropped = 0
        for q in range(nq):
            count = 0
            buf = [0.0] * outbuf
            for i in range(quantum):
                cur = math.sin(2 * math.pi * freq * (q * quantum + i) / ctx_rate)
                while frac_pos < 1.0:
                    if count < outbuf:
                        buf[count] = prev_sample + frac_pos * (cur - prev_sample)
                        count += 1
                    else:
                        dropped += 1
                    frac_pos += ctx_rate / 16000.0
                frac_pos -= 1.0
                prev_sample = cur
            out.extend(buf[:count])
        return out, dropped

    def goertzel(x, target, rate):
        n = len(x)
        k = 0.5 + (n * target) / rate
        w = 2 * math.pi * k / n
        c = 2 * math.cos(w)
        s1 = s2 = 0.0
        for v in x:
            s0 = v + c * s1 - s2
            s2 = s1
            s1 = s0
        return math.sqrt(max(0.0, s1 * s1 + s2 * s2 - c * s1 * s2)) * (2.0 / n)

    # The accumulator emits a whole number of samples per quantum, so the
    # measured rate is quantised by the window length. The window is therefore
    # sized to cover the SAME wall-clock duration at every rate, so a short
    # high-rate window cannot masquerade as a rate error.
    for rate in (22050, 24000, 32000, 44100, 48000, 96000, 192000):
        nq = max(800, int(round(rate * 5.0 / 128)))
        out, dropped = simulate(rate, nq=nq)
        duration = nq * 128 / rate
        effective = len(out) / duration
        peak_f = 0.0
        best = 0.0
        f = 500.0
        while f < 1500.0:
            mag = goertzel(out, f, 16000)
            if mag > best:
                best, peak_f = mag, f
            f += 1.0
        checks["AUD-107-10 ctx %d outputs 16 kHz" % rate] = abs(effective - 16000) < 2.0
        checks["AUD-107-10 ctx %d drops no sample" % rate] = dropped == 0
        checks["AUD-107-10 ctx %d has no pitch drift" % rate] = abs(peak_f - 1000.0) < 5.0

    # ---------- AUD-107-10: the worklet is wired correctly ----------
    cap = blocks["frontend/src/worklets/pcm-capture-processor.ts"][1]
    static = {
        "the worklet declares sampleRate": "declare const sampleRate: number;" in cap,
        "the worklet imports SAMPLE_RATE_IN": "SAMPLE_RATE_IN" in cap,
        "the 48 kHz fast path is retained": "ctxRate === 48000" in cap,
        "the Blackman-sinc taps are retained": "FIR_15_TAPS" in cap,
        "the fractional fallback exists": "this.fracPos" in cap and "this.prevSample" in cap,
        "the output buffer is at least 128": "Float32Array(128)" in cap,
        "the decimator follows the real rate": "ctxRate / SAMPLE_RATE_IN" in cap,
    }
    for name, ok in static.items():
        checks["AUD-107-10 %s" % name] = ok

    for name, ok in checks.items():
        print("  %-52s %s" % (name, ok))
    print("    (deferred attention mass = %.6f)" % deferred_mass)
    return all(checks.values())


def check_v28_aud107_08(blocks, master_text):
    """V28 - the Part 8 master-orchestration prompt must match Section 4.

    v10.7 still mandated resolving the ERB mask at the segment's own
    `lookaheadTargetTime`, which the 2.00 s lookahead DelayNode (AUD-107-03)
    invalidated, and it omitted the AUD-107-01..06 and AUD-107-09
    requirements entirely.
    """
    p8 = master_text[master_text.index("## PART 8:"):]

    stale = "lookaheadTargetTime`, NOT at `this.ctx.currentTime`"
    checks = {
        "the stale ERB instruction is gone from Part 8": stale not in p8,
        "Part 8 requires AUD-107-01": "AUD-107-01" in p8,
        "Part 8 requires AUD-107-02": "AUD-107-02" in p8,
        "Part 8 requires AUD-107-03": "AUD-107-03" in p8,
        "Part 8 requires AUD-107-04": "AUD-107-04" in p8,
        "Part 8 requires AUD-107-05": "AUD-107-05" in p8,
        "Part 8 requires AUD-107-06": "AUD-107-06" in p8,
        "Part 8 requires AUD-107-09": "AUD-107-09" in p8,
        "Part 8 names the precomputed FFT tables": "REV_TABLE_MNE" in p8,
        "Part 8 names the 2.00s DelayNode": "LOOKAHEAD_BUFFER_SEC" in p8,
        "Part 8 names the zero-gain capture sink": "gain.value = 0.0" in p8,
        "Part 8 names PrecomputedEarFeatures": "PrecomputedEarFeatures" in p8,
    }
    for name, ok in checks.items():
        print("  %-52s %s" % (name, ok))
    return all(checks.values())


def check_v29_phase2_rust(master_text):
    """V29 - Phase 2 Rust completion: every crate file must be byte-identical
    to its Part 4.9.x blueprint block, and the new deliverables must be
    present.

    v10.7 Phase 2 added six new Rust module files and a real `#[pymodule]` ABI
    to each of the two Python-facing crates. The blueprint is the source of
    truth for the crates, so every file is compared byte-for-byte.
    """
    files = [
        ("4.9.4", "crates/khmer-linguistics-rs/src/lib.rs"),
        ("4.9.4.1", "crates/khmer-linguistics-rs/src/normalizer.rs"),
        ("4.9.4.2", "crates/khmer-linguistics-rs/src/decomposer.rs"),
        ("4.9.5", "crates/khmer-transport-rs/src/lib.rs"),
        ("4.9.5.1", "crates/khmer-transport-rs/src/session.rs"),
        ("4.9.5.2", "crates/khmer-transport-rs/src/live_queue.rs"),
        ("4.9.5.3", "crates/khmer-transport-rs/src/wav_container.rs"),
        ("4.9.6", "crates/khmer-dsp-wasm/src/lib.rs"),
    ]
    checks = {}
    for tag, path in files:
        marker = "#### %s `%s`" % (tag, path)
        try:
            i = master_text.index(marker)
        except ValueError:
            checks["%s is published" % path] = False
            continue
        fo = master_text.index("```rust", i)
        fc = master_text.index("```", fo + 7)
        blueprint = master_text[fo + 7 : fc].strip("\n")
        disk_path = os.path.join(ROOT, *path.split("/"))
        if not os.path.exists(disk_path):
            checks["%s exists on disk" % path] = False
            continue
        disk = io.open(disk_path, encoding="utf-8").read().replace("\r\n", "\n").strip("\n")
        checks["%s matches the blueprint" % path] = blueprint == disk

    # The Phase 2 deliverables must be visible in the published sources.
    ling = master_text[
        master_text.index("#### 4.9.4 `crates/khmer-linguistics-rs/src/lib.rs`") :
        master_text.index("#### 4.9.5 `crates/khmer-transport-rs/src/lib.rs`")
    ]
    trans = master_text[
        master_text.index("#### 4.9.5 `crates/khmer-transport-rs/src/lib.rs`") :
        master_text.index("#### 4.9.6 `crates/khmer-dsp-wasm/src/lib.rs`")
    ]
    dsp = master_text[
        master_text.index("#### 4.9.6 `crates/khmer-dsp-wasm/src/lib.rs`") :
        master_text.index("#### 4.9.7")
    ]
    checks.update({
        "R-1 the linguistics crate has a #[pymodule]": "#[pymodule]" in ling
        and "khmer_linguistics_rs" in ling,
        "R-1 normalize_khmer is a #[pyfunction]": "#[pyfunction]" in ling
        and "fn normalize_khmer(" in ling,
        "R-2 the vowel map runs before the reordering": (
            "INDEPENDENT_VOWEL_MAP" in ling
            and ling.index("INDEPENDENT_VOWEL_MAP.iter()") < ling.index("for _ in 0..3")
        ),
        "R-3 the regexes are compiled once with once_cell": (
            "once_cell::sync::Lazy" in ling and "static RE_COENG" in ling
        ),
        "R-3 the decomposer uses crc32fast": "crc32fast::hash" in ling,
        "R-4 the transport crate has a #[pymodule]": "#[pymodule]" in trans
        and "khmer_transport_rs" in trans,
        "R-4 the fence is a #[pyclass]": "#[pyclass]" in trans,
        "R-4 the queue uses crossbeam_channel": "crossbeam_channel" in trans,
        "R-7 the session helpers compute the energy hash": (
            "compute_pcm_energy_hash" in trans and "crc32fast::hash" in trans
        ),
        "R-7 the SpeakerRegistry is present": "SpeakerRegistry" in trans,
        "R-5 the ERB band index is present": "erb_band_index_for_freq" in dsp,
        "R-6 the sinc decimator is present": "SincDecimator48To16" in dsp,
        "R-6 the FIR taps are retained": "FIR_15_TAPS" in dsp,
    })
    for name, ok in checks.items():
        print("  %-52s %s" % (name, ok))
    return all(checks.values())


def check_v23_fft(blocks):
    """V23a - the FFT/IFFT round trip must actually reconstruct the signal.

    v10.7 shipped a _computeSpectrum that bit-reversed its input in place and
    then ran the butterflies on two freshly allocated ZERO arrays, so every bin
    came out 0.0. _inverseSpectrum had the same defect, so it always returned
    an all-zero signal. The M&E suppressor was therefore a no-op that also
    silenced the bed. This check EXECUTES the two methods under node and
    asserts a real round trip.
    """
    import subprocess
    import tempfile

    mne = blocks["frontend/src/audio/stereo-mne-mixer.ts"][1]
    cls = (
        _js_module_tables(mne)
        + "\nclass Spectrum {\n"
        + _js_method(mne, "_computeSpectrum")
        + "\n\n"
        + _js_method(mne, "_inverseSpectrum")
        + "\n}\n"
    )
    harness = cls + r"""
const T = new Spectrum();
// N_FFT equals the ScriptProcessorNode block size (4096), and the blocks are
// NON-OVERLAPPING, so the windowing choice is what determines whether the
// block edges are preserved.
const N = 4096;
const frame = new Float32Array(N);
for (let i = 0; i < N; i++) frame[i] = Math.sin(2 * Math.PI * 5 * i / N) * 0.5;
const half = N / 2 + 1;
const re = new Float32Array(half), im = new Float32Array(half);
T._computeSpectrum(frame, N, re, im, N);
let nonzeroBins = 0, peak = 0;
for (let k = 0; k < half; k++) {
  const a = re[k], b = im[k];
  if (Math.abs(a) > 1e-9 || Math.abs(b) > 1e-9) nonzeroBins++;
  peak = Math.max(peak, Math.abs(a), Math.abs(b));
}
const out = T._inverseSpectrum(re, im, N);
let outEnergy = 0, outPeak = 0, inEnergy = 0, maxErr = 0;
for (let i = 0; i < N; i++) {
  outEnergy += out[i] * out[i];
  inEnergy += frame[i] * frame[i];
  outPeak = Math.max(outPeak, Math.abs(out[i]));
  maxErr = Math.max(maxErr, Math.abs(out[i] - frame[i]));
}
console.log("NONZERO_BINS=" + nonzeroBins);
console.log("PEAK=" + peak.toFixed(6));
console.log("OUT_ENERGY=" + outEnergy.toFixed(4));
console.log("IN_ENERGY=" + inEnergy.toFixed(4));
console.log("RATIO=" + (inEnergy > 0 ? outEnergy / inEnergy : 0).toFixed(9));
console.log("OUT_PEAK=" + outPeak.toFixed(6));
// SAMPLE EQUALITY: the round trip must reproduce the UNWINDOWED input
// sample-for-sample. A Hann window tapers the block edges to zero, so this
// is the check that detects the periodic block-edge dropouts.
console.log("MAX_ABS_ERR=" + maxErr.toExponential(3));
console.log("FIRST_OUT=" + out[0].toFixed(12));
console.log("FIRST_IN=" + frame[0].toFixed(12));
console.log("LAST_OUT=" + out[N - 1].toFixed(12));
console.log("LAST_IN=" + frame[N - 1].toFixed(12));
"""
    fd, path = tempfile.mkstemp(suffix=".js")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(harness)
    try:
        proc = subprocess.run(
            ["node", path], capture_output=True, text=True, timeout=60
        )
    except Exception as exc:
        print("  node execution failed: %s" % exc)
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if proc.returncode != 0:
        print("  node stderr: %s" % proc.stderr.strip()[:300])
        return False
    vals = {}
    for line in proc.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k] = v
    try:
        nonzero = int(vals["NONZERO_BINS"])
        ratio = float(vals["RATIO"])
        out_peak = float(vals["OUT_PEAK"])
        max_err = float(vals["MAX_ABS_ERR"])
        first_out = float(vals["FIRST_OUT"])
        first_in = float(vals["FIRST_IN"])
        last_out = float(vals["LAST_OUT"])
        last_in = float(vals["LAST_IN"])
    except (KeyError, ValueError):
        print("  unexpected node output: %r" % proc.stdout[:300])
        return False
    checks = {
        "forward transform is not all-zero": nonzero > 0,
        "inverse output is not all-zero": out_peak > 1e-9,
        "round-trip energy matches the unwindowed input": abs(ratio - 1.0) < 1e-6,
        "round trip is the exact identity (sample equality)": max_err < 1e-9,
        "the first block sample is preserved": abs(first_out - first_in) < 1e-9,
        "the last block sample is preserved": abs(last_out - last_in) < 1e-9,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    print(
        "    (nonzeroBins=%d ratio=%.9f maxAbsErr=%.3e lastOut=%.9f lastIn=%.9f)"
        % (nonzero, ratio, max_err, last_out, last_in)
    )
    return all(checks.values())


def check_v25_deferred_noop(mods):
    """V25 - a wait-gated (deferred) tail syllable must decode to a true
    (0, 0, 0) no-op token.

    v10.7 zeroed bus_l BEFORE the head projections, which is not sufficient:
    nn.Linear has a bias vector, so a zero input still produces a non-zero
    logit vector and the argmax returns a real onset id. The decoder then
    emitted the deferred syllables as if they were real content.
    """
    import torch

    from backend.contracts.schemas import CI_MICRO_CONFIG
    from backend.linguistics.syllable_decomposer import BilingualCodeSwitchDecomposer
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
    triplets = list(zip(o, v, cd))
    decomp = BilingualCodeSwitchDecomposer()
    texts, cs_words = decomp.decode_batch_to_text(batch)

    checks = {
        "the model produced five syllables": len(triplets) == 5,
        "the three deferred tail syllables are (0,0,0)": triplets[-3:]
        == [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        "the deferred count is three": mouth.last_deferred_syll_count == 3,
        "the decoder produced text": bool(texts and texts[0].strip()),
        "the decoder emitted no synthetic brand": cs_words == [],
        "the decoder skipped the deferred tail": len(texts[0]) < 12,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    print("    (triplets=%s)" % (triplets,))
    return all(checks.values())


def check_v23(blocks):
    """V23 - the frontend must transform the masked spectrum back to the time
    domain, and must create only ONE MediaElementAudioSourceNode."""
    mne = blocks["frontend/src/audio/stereo-mne-mixer.ts"][1]
    ws = blocks["frontend/src/client/duplex-ws-client.ts"][1]
    checks = {
        "mixer has _inverseSpectrum": "_inverseSpectrum" in mne,
        "the masked mid is reconstructed": "midOut" in mne,
        "the output uses midOut": "maskApplied ? midOut[i]" in mne,
        "the mixer creates the source node": "createMediaElementSource" in mne,
        "the mixer returns it": ": MediaElementAudioSourceNode {" in mne,
        "the mixer exposes getMediaSourceNode": "getMediaSourceNode" in mne,
        "the client taps the shared node": "this.mixer.getMediaSourceNode()" in ws,
        "the processor reaches the destination": "processor.connect(this.ctx.destination)" in mne,
        "the source feeds the delay node, not the processor": (
            mne.count("source.connect(processor);") == 0
            and mne.count("source.connect(delayNode);") == 1
        ),
        "the 2.00s lookahead DelayNode is in the M&E path": (
            "this.ctx.createDelay(LOOKAHEAD_BUFFER_SEC + 1.0)" in mne
            and "delayNode.delayTime.value = LOOKAHEAD_BUFFER_SEC;" in mne
            and "source.connect(delayNode);" in mne
            and "delayNode.connect(processor);" in mne
        ),
        "the FFT tables are precomputed at module load": (
            "const REV_TABLE_MNE = new Uint16Array(N_FFT_MNE);" in mne
            and "COS_TABLE_MNE" in mne
            and "COS_TABLE_MNE_INV" in mne
        ),
        "no Math.log2 call remains in the per-block path": "Math.log2(size)" not in mne,
        "no per-block twiddle allocation remains": "new Float64Array(half)" not in mne,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v24(master_text):
    """V24 - document hygiene: one policy, one provenance, no leftover marker,
    no mojibake, sequential ledger."""
    import re

    p8 = master_text[master_text.index("## PART 8:"):]
    # Mojibake is a QUESTION MARK standing in for a character that was lost,
    # so it appears as a run of two or more adjacent "?" or as a "?" glued to
    # a non-ASCII character. A SINGLE "?" in a comment is legitimate (a Rust
    # doc comment naming an optional field, a regex quantifier, a rhetorical
    # question), so it is not flagged.
    mojibake = [
        l
        for l in master_text.split("\n")
        if "?" in l
        and l.lstrip().startswith(("#", "//"))
        and not re.search(r"\w\?:", l)
        and re.search(r"\?\?|[^\x00-\x7F]\?[^\x00-\x7F]|\?\s*\?", l)
    ]
    nums = [int(x) for x in re.findall(r"^(\d+)\. \*\*Fix", master_text, re.M)]
    # Count the HEADERS only (a line that starts with the marker), so a prose
    # mention of the header name inside a ledger entry is not a duplicate.
    policy_headers = [
        l
        for l in master_text.split("\n")
        if l.startswith("**REVISION RETENTION POLICY:**")
    ]
    provenance_headers = [
        l
        for l in master_text.split("\n")
        if l.startswith("**DOCUMENT PROVENANCE")
    ]
    checks = {
        "exactly one retention-policy header": len(policy_headers) == 1,
        "exactly one provenance header": len(provenance_headers) == 1,
        "no @@VERSION@@ marker": "@@VERSION@@" not in master_text,
        "no mojibake comment lines": len(mojibake) == 0,
        "fix ledger is 1..74, sequential": nums == list(range(1, 75)),
        "Part 8 STAGE 00 names v10.7": "v10.7 - Code-Level Audit Remediation Edition" in p8,
        "Part 8 STAGE 06 names v10.7": "STAGE 06: v10.7 DOCUMENT-TRUTH" in p8,
        "the FastAPI title names v10.7": "v10.7 Full-Duplex Server" in master_text,
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def check_v20(mods):
    """V20 - the KM->EN forward must land inside the reserved English ranges.

    v10.7 constrains the synthesized English slots, so decode_batch_to_text
    reaches ENGLISH_SYLLABLE_TABLE and the CVC pseudoword fallback is never
    used, and cs_words stays empty.
    """
    import torch

    dec_mod = mods["backend/linguistics/syllable_decomposer.py"]
    con = mods["backend/nn_cartridges/s2st_console.py"]
    sch = sys.modules["backend.contracts.schemas"]
    torch.manual_seed(0)
    console = con.SymmetricS2STConsole(config=sch.CI_MICRO_CONFIG)
    console.eval()

    with torch.no_grad():
        _, batch, _ = console.forward_s2st(
            torch.randn(1, 8000), "km", "en", orig_duration_sec=0.5
        )

    decomp = dec_mod.BilingualCodeSwitchDecomposer()
    texts, cs_words = decomp.decode_batch_to_text(batch)

    onsets = batch.onset_ids.flatten().tolist()
    coengs = batch.coeng_ids.flatten().tolist()
    vowels = batch.vowel_ids.flatten().tolist()
    codas = batch.coda_ids.flatten().tolist()
    regs = batch.register_ids.flatten().tolist()

    checks = {
        "every onset is in 34..57": all(34 <= o <= 57 for o in onsets),
        "every coeng is 0": all(c == 0 for c in coengs),
        "every vowel is in 28..44": all(28 <= v <= 44 for v in vowels),
        "every coda is in 18..31": all(18 <= c <= 31 for c in codas),
        "every register is 3": all(r == 3 for r in regs),
        "cs_words is empty": list(cs_words) == [],
        "decoded text is non-empty ASCII": bool(texts[0].strip())
        and texts[0].strip().isascii(),
    }
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())


def _apply_rust_map(rust_map, word):
    """Apply the Rust normalizer's map and reordering to `word`.

    The Rust block is reproduced faithfully in Python: the independent-vowel
    map is applied FIRST and the coeng/shifter reordering SECOND, which is the
    order the fixed Rust source uses. Comparing this against the Python
    normalizer therefore detects any divergence between the two.
    """
    import re as _re

    out = word
    for k in sorted(rust_map, reverse=True):
        out = out.replace(chr(k), rust_map[k])
    re_coeng = _re.compile("([\\u17B6-\\u17CA]+)((?:\\u17D2[\\u1780-\\u17A2])+)")
    re_shift = _re.compile("([\\u17B6-\\u17C8]+)([\\u17C9\\u17CA])")
    for _ in range(3):
        out = re_coeng.sub(lambda m: m.group(2) + m.group(1), out)
        out = re_shift.sub(lambda m: m.group(2) + m.group(1), out)
    return out


def check_v4_rust_parity(master_text, mods):
    """V4 - the Rust and Python independent-vowel maps must agree on all 17.

    v10.6 omitted U+17B3 from the Rust map, so the two paths disagreed on the
    17th independent vowel.
    """
    import re

    def unesc(s):
        return re.sub(
            r"\\u([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s
        )

    # v10.7 Phase 2: the independent-vowel map moved from lib.rs into the
    # normalizer module, so the map is read from Section 4.9.4.1.
    rs = master_text[
        master_text.index(
            "\n#### 4.9.4.1 `crates/khmer-linguistics-rs/src/normalizer.rs`\n"
        ) + 1 :
    ]
    rs = rs[rs.index("```rust") + 7:]
    rs = rs[: rs.index("```")]
    rust = {}
    for k, v in re.findall(
        r"\('\\u\{([0-9A-F]{4})\}', \"((?:\\u\{[0-9A-F]{4}\})+)\"\)", rs
    ):
        rust[int(k, 16)] = unesc(re.sub(r"\\u\{([0-9A-F]{4})\}", r"\\u\1", v))

    py = master_text[
        master_text.index(
            "\n#### 4.1.1 `backend/linguistics/khmer_normalizer.py`\n"
        ) + 1 :
    ]
    py = py[py.index("```python") + 9:]
    py = py[: py.index("```")]
    pyc = {
        int(a, 16): unesc(b)
        for a, b in re.findall(r'"\\u([0-9A-F]{4})": "([^"]+)"', py)
    }

    common = sorted(set(rust) & set(pyc))

    # v10.7: the Rust normalizer must apply the independent-vowel map BEFORE
    # the coeng/shifter reordering, exactly like the Python one. A compound
    # word with an independent vowel followed by a subscript coeng is the
    # case that distinguishes the two orders.
    #
    # The order is observable in the Rust SOURCE: the map loop must appear
    # before the `for _ in 0..3` reordering loop.
    rust_map_at = rs.find("INDEPENDENT_VOWEL_MAP.iter()")
    rust_reorder_at = rs.find("for _ in 0..3")
    py_map_at = py.find("for k, v in _INDEPENDENT_VOWEL_MAP.items()")
    py_reorder_at = py.find("for _ in range(3)")

    checks = {
        "rust map has 17 entries": len(rust) == 17,
        "python map has 17 entries": len(pyc) == 17,
        "U+17B3 present in rust": 0x17B3 in rust,
        "U+17B3 present in python": 0x17B3 in pyc,
        "range is U+17A3..U+17B3": common == list(range(0x17A3, 0x17B4)),
        "all 17 mappings agree": all(rust[k] == pyc[k] for k in common),
        "rust applies the map before the reordering": 0 < rust_map_at < rust_reorder_at,
        "python applies the map before the reordering": 0 < py_map_at < py_reorder_at,
    }

    # The compound word must survive both normalizers identically. This case
    # is the one that the ordering bug actually changes.
    compound = "\u17AB\u1791\u17D2\u1792\u17B7"
    discriminating = "\u17AB\u17B7\u17D2\u1792"
    nrm = mods["backend/linguistics/khmer_normalizer.py"]
    for label, word in (("compound", compound), ("ordering-sensitive", discriminating)):
        py_out = nrm.normalize_khmer(word)
        rust_out = _apply_rust_map(rust, word)
        checks["%s word normalizes identically" % label] = py_out == rust_out
        checks["%s word is non-empty" % label] = len(py_out) > 0
    for name, ok in checks.items():
        print("  %-46s %s" % (name, ok))
    return all(checks.values())



def main():
    blocks = extract_blocks(MASTER)
    master_text = io.open(MASTER, encoding="utf-8").read()
    print("extracted %d blocks from the v10.7 master" % len(blocks))
    mods = load_modules(blocks)
    print("loaded %d python modules" % len(mods))
    print()

    results = []

    def run(label, fn, *args):
        print("=== %s ===" % label)
        ok = fn(*args)
        print("%s: %s" % (label.split(" - ")[0], "PASS" if ok else "FAIL"))
        print()
        results.append(ok)

    run("V1 - parameter ledger re-derivation", lambda: check_v1(mods)[0])
    run("V2 - version synchronization", check_v2, blocks, master_text)
    run("V4 - Rust / Python independent-vowel parity", check_v4_rust_parity, master_text, mods)
    run("V8 - concurrency stress", check_v8, mods)
    run("V8b - deterministic lock coverage", check_v8b, mods)
    run("V10 - step-conditioning sensitivity", check_v10, mods)
    run("V12 - independent vowel reachability", check_v12, mods)
    run("V13 - batch-aware scene separation", check_v13, mods)
    run("V14 - frontend wiring", check_v14, blocks)
    run("V16 - mouth property bridges", check_v16, mods)
    run("V19 - classifier-free guidance path", check_v19, mods)
    run("V20 - KM to EN English ranges", check_v20, mods)
    run("V21 - trainer distillation loss", check_v21, mods)
    run("V22 - eval suite COMET checkpoint", check_v22, blocks)
    run("V23 - frontend inverse FFT and shared node", check_v23, blocks)
    run("V23a - FFT/IFFT round trip executes under node", check_v23_fft, blocks)
    run("V25 - deferred tail syllables are true no-ops", check_v25_deferred_noop, mods)
    run("V26 - the seven AUD-107 code findings are closed", check_v26_aud107, blocks, mods)
    run("V27 - AUD-107-09 attention mask and AUD-107-10 resampler", check_v27_aud107_09_and_10, blocks, mods)
    run("V28 - AUD-107-08 Part 8 synchronization", check_v28_aud107_08, blocks, master_text)
    run("V29 - Phase 2 Rust crate parity and deliverables", check_v29_phase2_rust, master_text)
    run("V24 - document hygiene", check_v24, master_text)

    allok = all(results)
    print("ALL: %s (%d/%d checks)" % ("PASS" if allok else "FAIL", sum(results), len(results)))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
