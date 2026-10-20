## PART 5: STAGE 01 ADVERSARIAL CRITIC TDD SUITE BLUEPRINTS (`backend/tests/`)

### 5.1 Gate 1: `backend/tests/test_codeswitch_decomposer.py`
```python
import pytest

from backend.contracts.schemas import CodeSwitchDecomposerProtocol
from backend.linguistics.syllable_decomposer import BilingualCodeSwitchDecomposer

GROUND_TRUTH_CASES = [
    ("គណនា", ["គ", "ណ", "នា"], [1, 1, 2], [2, 1, 2], [0, 0, 0]),
    ("សុភមង្គល", ["សុ", "ភ", "មង", "គល"], [1, 1, 2, 2], [1, 2, 2, 2], [0, 0, 0, 0]),
    ("សង្គ្រាម", ["សង", "គ្រាម"], [2, 2], [1, 2], [0, 1]),
    ("កម្ពុជា", ["កម", "ពុ", "ជា"], [2, 1, 2], [1, 2, 2], [0, 0, 0]),
    ("បញ្ហា", ["បញ", "ហា"], [2, 2], [1, 1], [0, 0]),
    ("កន្លែង", ["កន", "លែង"], [2, 2], [1, 1], [0, 0]),
    ("ប្រជា", ["ប្រ", "ជា"], [1, 2], [1, 2], [1, 0]),
    ("ក្រសួង", ["ក្រ", "សួង"], [1, 2], [1, 1], [1, 0]),
    ("កក្រើក", ["ក", "ក្រើក"], [1, 2], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("word,exp_surfaces,exp_sesqui,exp_reg,exp_medial", GROUND_TRUTH_CASES)
def test_khmer_ground_truth_table(word, exp_surfaces, exp_sesqui, exp_reg, exp_medial):
    decomp = BilingualCodeSwitchDecomposer()
    assert isinstance(decomp, CodeSwitchDecomposerProtocol)
    tokens = decomp.decompose_mixed_sentence(word)
    assert [t.surface for t in tokens] == exp_surfaces
    assert [t.sesqui_id for t in tokens] == exp_sesqui
    assert [t.register_id for t in tokens] == exp_reg
    assert [t.medial_id for t in tokens] == exp_medial


def test_double_coeng_srey_and_foreign_brands():
    decomp = BilingualCodeSwitchDecomposer()
    srey = decomp.decompose_mixed_sentence("ស្ត្រី")
    assert len(srey) == 1
    assert srey[0].surface == "ស្ត្រី"
    assert srey[0].coeng_id > 0
    assert srey[0].medial_id == 1
    assert srey[0].register_id == 1
    assert srey[0].sesqui_id == 2
    assert 0 <= srey[0].coda_id <= 16

    mixed = "ខ្ញុំទៅ Calvin Klein និង Versace"
    tokens = decomp.decompose_mixed_sentence(mixed)
    brand_tokens = [t for t in tokens if t.register_id == 3]
    assert [t.surface.lower() for t in brand_tokens] == ["calvin", "klein", "versace"]
    assert all(t.sesqui_id == 2 for t in brand_tokens)
    assert all(18 <= t.coda_id <= 31 for t in brand_tokens)

    batch = decomp.encode_sentence_to_batch(mixed)
    assert batch.onset_ids.shape[0] == 1
    texts, cs_words = decomp.decode_batch_to_text(batch)
    assert len(texts) == 1 and len(texts[0]) > 0
    assert [w.lower() for w in cs_words] == ["calvin", "klein", "versace"]
# ===========================================================================
# v10.7 additions. The v10.6 assertions above are retained VERBATIM.
# ===========================================================================


def test_independent_vowels_are_canonicalized_and_reachable():
    # v10.7 (P1-A): U+17AB..U+17AE (RY / RYY / LY / LYY) used to map to
    # themselves, and _SYLLABLE_SCAN_RE begins at U+1780, so the decomposer
    # SKIPPED them entirely: a word beginning with one lost its first
    # syllable, and a standalone one fell back to the default onset id 1.
    from backend.linguistics.khmer_normalizer import normalize_khmer

    decomp = BilingualCodeSwitchDecomposer()

    # (codepoint, expected canonical form, expected onset id)
    cases = [
        (0x17AB, "\u179A\u17B9", 27),
        (0x17AC, "\u179A\u17BA", 27),
        (0x17AD, "\u179B\u17B9", 28),
        (0x17AE, "\u179B\u17BA", 28),
    ]
    for codepoint, expected, onset_id in cases:
        ch = chr(codepoint)
        assert normalize_khmer(ch) == expected
        tokens = decomp.decompose_mixed_sentence(ch)
        assert len(tokens) == 1
        assert tokens[0].onset_id == onset_id
        assert tokens[0].onset_id != 1  # not the default-onset fallback

    # A multi-syllable word beginning with an independent vowel must keep
    # its first syllable (v10.6 dropped it).
    word = chr(0x17AB) + chr(0x179F) + chr(0x17D2) + chr(0x179F)
    tokens = decomp.decompose_mixed_sentence(word)
    assert len(tokens) == 2


def test_robat_and_coda_membership():
    # v10.7: Robat (\u17CC) must map to vowel_id 24, and the coda for
    # \u1798 must be 17 in the plain form and 14 in the Bantoc form.
    # The coda is only derived when \u1798 is NOT the initial consonant
    # (a leading \u1798 is the onset, not a coda), so the test uses a
    # preceding consonant.
    decomp = BilingualCodeSwitchDecomposer()

    robat_tokens = decomp.decompose_mixed_sentence("\u1798\u17CC")
    assert robat_tokens
    assert robat_tokens[0].vowel_id == 24

    plain_mem = decomp.decompose_mixed_sentence("\u1780\u1798")
    assert plain_mem
    assert plain_mem[0].coda_id == 17

    bantoc_mem = decomp.decompose_mixed_sentence("\u1780\u1798\u17CB")
    assert bantoc_mem
    assert bantoc_mem[0].coda_id == 14


def test_km_to_en_syllable_decodes_to_latin_not_a_cvc_pseudoword():
    # v10.7 (P1-B): a synthesized English target syllable (register 3,
    # coeng 0, onset 34..57, vowel 28..44, coda 18..31) must decode to a
    # real Latin orthography. v10.6 fell through to a hardcoded CVC
    # fallback that ignored coeng_id and emitted text such as "kak".
    import torch

    from backend.contracts.schemas import SyllableTensorBatch

    decomp = BilingualCodeSwitchDecomposer()

    def make_batch(onset, medial, vowel, coda):
        t = lambda v: torch.tensor([[v]], dtype=torch.long)
        return SyllableTensorBatch(
            onset_ids=t(onset),
            coeng_ids=t(0),
            medial_ids=t(medial),
            register_ids=t(3),
            vowel_ids=t(vowel),
            coda_ids=t(coda),
            sesqui_ids=t(2),
        )

    # (onset, medial, vowel, coda) drawn from the reserved English ranges.
    for onset, medial, vowel, coda in [
        (34, 0, 28, 18),
        (40, 0, 30, 20),
        (57, 3, 44, 31),
    ]:
        texts, cs_words = decomp.decode_batch_to_text(
            make_batch(onset, medial, vowel, coda)
        )
        assert len(texts) == 1
        assert len(texts[0]) > 0
        assert texts[0].strip().isascii()
        assert texts[0].strip().isalpha()
        # A synthesized English syllable is NOT a code-switched brand.
        assert cs_words == []

    # The reserved ranges must be disjoint from the Khmer ranges.
    assert 34 > 33  # onset: Khmer 1..33, English 34..
    assert 28 > 24  # vowel: Khmer 0..24, reserved 25..27, English 28..


def test_brand_tokens_keep_one_slot_and_coda_in_range():
    # v10.7 (correction): a brand word occupies exactly ONE token, so the
    # Gate 1 surface assertions above remain valid, and every register-3
    # token keeps coda_id in 18..31.
    decomp = BilingualCodeSwitchDecomposer()

    mixed = "Calvin Klein Versace"
    tokens = decomp.decompose_mixed_sentence(mixed)
    brand_tokens = [t for t in tokens if t.register_id == 3]
    assert [t.surface.lower() for t in brand_tokens] == [
        "calvin",
        "klein",
        "versace",
    ]
    assert all(18 <= t.coda_id <= 31 for t in brand_tokens)
    assert all(t.sesqui_id == 2 for t in brand_tokens)
    # A multi-syllable brand is still ONE token.
    assert len(brand_tokens) == 3
```

### 5.2 Gate 2: `backend/tests/test_symmetric_s2st.py`
```python
import torch

from backend.contracts.schemas import (
    CI_MICRO_CONFIG,
    PROD_240M_CONFIG,
    S2STConsoleProtocol,
    StemAndSceneProtocol,
)
from backend.nn_cartridges.codeswitch_router import AcousticPointerCodeSwitchBridge
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole
from backend.nn_cartridges.stem_and_mode_gate import StemAndScenePipeline
from backend.nn_cartridges.universal_core import GradientReversalFn


def test_meta_device_240m_parameter_budget():
    with torch.device("meta"):
        pipeline = StemAndScenePipeline(config=PROD_240M_CONFIG)
        console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    total_params = sum(p.numel() for p in pipeline.parameters()) + sum(
        p.numel() for p in console.parameters()
    )
    assert 230_000_000 <= total_params <= 250_000_000, f"Got {total_params:,} params"


def test_protocols_bidirectional_grl_and_pointer():
    pipeline = StemAndScenePipeline(config=CI_MICRO_CONFIG)
    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    assert isinstance(pipeline, StemAndSceneProtocol)
    assert isinstance(console, S2STConsoleProtocol)

    # Gate 2 Slaney Mel full-rank check (Fix P0-1)
    assert torch.linalg.matrix_rank(console.ear.mel_fb) == 100
    assert torch.linalg.matrix_rank(console.mel_fb_24k) == 100

    wav_16k = torch.randn(1, 8000)
    wav_km, batch_km, _ = console.forward_s2st(wav_16k, "en", "km", orig_duration_sec=0.5)
    assert wav_km.shape[0] == 1 and wav_km.shape[-1] > 0

    wav_en, batch_en, _ = console.forward_s2st(wav_16k, "km", "en", orig_duration_sec=0.5)
    assert torch.all(batch_en.register_ids == 3)
    assert torch.all(batch_en.sesqui_ids == 2)

    x = torch.ones(2, 8, requires_grad=True)
    y = GradientReversalFn.apply(x, 0.5)
    (y * 2.0).sum().backward()
    assert torch.all(x.grad < 0.0)

    bridge = AcousticPointerCodeSwitchBridge(config=CI_MICRO_CONFIG)
    sem = torch.randn(1, 10, 512)
    src = torch.randn(1, 10, 512)
    fused_0, g0 = bridge(sem, src, force_gate=0.0)
    fused_1, g1 = bridge(sem, src, force_gate=1.0)
    assert torch.allclose(g0, torch.zeros_like(g0))
    assert torch.allclose(g1, torch.ones_like(g1))
    assert torch.allclose(fused_0, sem)
    assert not torch.allclose(fused_1, sem)
# ===========================================================================
# v10.7 additions. The v10.6 assertions above are retained VERBATIM.
# ===========================================================================


def test_v107_exact_parameter_ledger():
    # v10.7: the ledger must remain EXACTLY 241,637,457 (prod) and
    # 3,838,033 (CI). The v10.7 remediation is parameter-free by
    # construction, so these values must not move.
    with torch.device("meta"):
        pipeline = StemAndScenePipeline(config=PROD_240M_CONFIG)
        console = SymmetricS2STConsole(config=PROD_240M_CONFIG)
    prod_total = sum(p.numel() for p in pipeline.parameters()) + sum(
        p.numel() for p in console.parameters()
    )
    assert prod_total == 241_637_457, f"prod ledger moved: {prod_total:,}"

    with torch.device("meta"):
        pipeline = StemAndScenePipeline(config=CI_MICRO_CONFIG)
        console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    ci_total = sum(p.numel() for p in pipeline.parameters()) + sum(
        p.numel() for p in console.parameters()
    )
    assert ci_total == 3_838_033, f"CI ledger moved: {ci_total:,}"


def test_v107_step_conditioning_is_not_rank_degenerate():
    # v10.7 (P1-C): the v10.6 timestep embedding expanded a rank-1 scalar
    # across every column, so time_mlp only ever saw w1*t + w2*dt + b and
    # could not express any non-linear function of t. The embedding must now
    # have rank strictly greater than 1 across distinct timesteps.
    from backend.nn_cartridges.mouth_cartridge import sinusoidal_time_embedding

    emb = sinusoidal_time_embedding
    probe = torch.linspace(0.0, 1.0, 64).unsqueeze(0)
    rank = int(torch.linalg.matrix_rank(emb(probe, 64)))
    assert rank > 1, f"timestep embedding is still rank {rank}"

    # And the model must be sensitive to both t and dt.
    torch.manual_seed(0)
    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    console.eval()
    mouth = console.mouth
    B, T = 1, 16
    x_t = torch.randn(B, 100, T)
    cond = torch.randn(B, T, CI_MICRO_CONFIG.mouth_d_model)
    emo = torch.randn(B, 32, 512)
    with torch.no_grad():
        v_t0 = mouth.predict_velocity(x_t, torch.zeros(B), cond, emo, dt=None)
        v_t1 = mouth.predict_velocity(x_t, torch.ones(B), cond, emo, dt=None)
        v_d0 = mouth.predict_velocity(
            x_t, torch.full((B,), 0.5), cond, emo, dt=torch.full((B,), 0.01)
        )
        v_d1 = mouth.predict_velocity(
            x_t, torch.full((B,), 0.5), cond, emo, dt=torch.full((B,), 0.20)
        )
    assert not torch.allclose(v_t0, v_t1, atol=1e-6)
    assert not torch.allclose(v_d0, v_d1, atol=1e-6)


def test_v107_km_to_en_forward_decodes_to_latin():
    # v10.7 (P1-B): the KM to EN forward must produce a syllable batch that
    # decodes to non-empty Latin text, not synthetic CVC pseudowords.
    torch.manual_seed(0)
    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    console.eval()
    wav_16k = torch.randn(1, 8000)
    wav_en, batch_en, _ = console.forward_s2st(wav_16k, "km", "en", orig_duration_sec=0.5)
    assert wav_en.shape[0] == 1 and wav_en.shape[-1] > 0
    assert batch_en.register_ids.shape[1] > 0

    from backend.linguistics.syllable_decomposer import BilingualCodeSwitchDecomposer

    decomp = BilingualCodeSwitchDecomposer()
    texts, cs_words = decomp.decode_batch_to_text(batch_en)
    assert len(texts) == 1
    assert len(texts[0].strip()) > 0
    assert texts[0].strip().isascii()

    # v10.7 (P1-B): the synthesized English syllables must land inside the
    # reserved English ranges, so the decoder reaches ENGLISH_SYLLABLE_TABLE
    # and NEVER falls through to the CVC pseudoword fallback. A fallback
    # would also populate cs_words with synthetic brands.
    from backend.linguistics.syllable_decomposer import is_english_syllable_slot

    onsets = batch_en.onset_ids.flatten().tolist()
    coengs = batch_en.coeng_ids.flatten().tolist()
    vowels = batch_en.vowel_ids.flatten().tolist()
    codas = batch_en.coda_ids.flatten().tolist()
    assert all(34 <= o <= 57 for o in onsets)
    assert all(c == 0 for c in coengs)
    assert all(28 <= v <= 44 for v in vowels)
    assert all(18 <= c <= 31 for c in codas)
    assert all(
        is_english_syllable_slot(o, c, v, cd)
        for o, c, v, cd in zip(onsets, coengs, vowels, codas)
    )
    # No synthetic brand may leak into the code-switch word list.
    assert cs_words == []


def test_v107_classifier_free_guidance_path_is_usable():
    # v10.7 (P1-C): integrate_shortcut_sway_ode MUST work with cfg_scale != 1.0.
    # v10.6 computed the conditional velocity only in the else branch and then
    # referenced it in the if branch, so any cfg_scale != 1.0 raised
    # UnboundLocalError: cannot access local variable 'v'. Guidance was
    # therefore impossible to use at all.
    from backend.nn_cartridges.s2st_console import integrate_shortcut_sway_ode

    torch.manual_seed(0)
    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    console.eval()

    B, T = 1, 16
    x_0 = torch.randn(B, 100, T)
    cond = torch.randn(B, T, CI_MICRO_CONFIG.mouth_d_model)
    emo = torch.randn(B, 32, 512)

    with torch.no_grad():
        v_plain = integrate_shortcut_sway_ode(
            console.mouth, x_0, cond, emo, steps=2, cfg_scale=1.0
        )
        v_guided = integrate_shortcut_sway_ode(
            console.mouth, x_0, cond, emo, steps=2, cfg_scale=1.5
        )

    assert torch.isfinite(v_plain).all()
    assert torch.isfinite(v_guided).all()
    # A non-unit scale must actually change the trajectory.
    assert not torch.allclose(v_plain, v_guided)
```

### 5.3 Gate 3: `backend/tests/test_multispeaker_and_modes.py`
```python
import torch

from backend.contracts.schemas import CI_MICRO_CONFIG
from backend.nn_cartridges.stem_and_mode_gate import (
    StemAndScenePipeline,
    filter_dominant_speakers,
)


def test_dominance_gate_and_dual_speaker_modes():
    t = torch.linspace(0, 1, 16000)
    stem_a = 0.5 * torch.sin(2 * 3.14159 * 220 * t)
    stem_b_equal = 0.5 * torch.sin(2 * 3.14159 * 440 * t)
    stem_b_quiet = 0.02 * torch.sin(2 * 3.14159 * 440 * t)

    dual = filter_dominant_speakers(stem_a, stem_b_equal)
    single = filter_dominant_speakers(stem_a, stem_b_quiet)
    assert dual.shape == (2, 16000)
    assert single.shape == (1, 16000)

    pipeline = StemAndScenePipeline(config=CI_MICRO_CONFIG)
    scene = pipeline.separate_and_classify(
        stem_a.unsqueeze(0), utterance_id=1, force_dual_speakers=True
    )
    assert scene.frame_vocal_modes.shape[0] == 2
    T_frames = scene.frame_vocal_modes.shape[1]
    assert scene.erb_suppression_mask.shape == (T_frames, 64)
    assert float(scene.erb_suppression_mask.min()) >= 0.0
    assert float(scene.erb_suppression_mask.max()) <= 1.0
```

### 5.4 Gate 4: `backend/tests/test_fast_talker_and_glue.py`
```python
import torch

from backend.contracts.schemas import MINOR_SYLLABLE_MAX_FRAMES
from backend.nn_cartridges.duration_aligner import LearnedDurationAligner
from backend.nn_cartridges.stem_and_mode_gate import SceneAcousticMatcher


def test_scene_matcher_rng_isolation_and_autograd_backward():
    rng_before = torch.get_rng_state()
    matcher = SceneAcousticMatcher()
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after), "SceneAcousticMatcher mutated global RNG!"

    dry_wav = torch.randn(2, 4800, requires_grad=True)
    mic_eq = (torch.rand(2, 1, 64) * 0.1).requires_grad_(True)
    rt60 = torch.tensor([[0.15], [0.25]], requires_grad=True)
    wet = torch.tensor([[0.10], [0.20]], requires_grad=True)
    rms = torch.tensor([[0.12], [0.15]])

    glued = matcher.apply_scene_glue(dry_wav, mic_eq, rt60, wet, rms)
    assert glued.shape == (2, 4800)
    loss = glued.pow(2).mean()
    loss.backward()
    for tensor in (dry_wav, mic_eq, rt60, wet):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_duration_aligner_minor_cap_and_playback_scaling():
    aligner = LearnedDurationAligner(d_model=64)
    f_1x = aligner.compute_elastic_target_frames(
        1.0, trailing_pause_ms=100.0, speech_rate_sps=5.5, playback_rate=1.0
    )
    f_2x = aligner.compute_elastic_target_frames(
        1.0, trailing_pause_ms=100.0, speech_rate_sps=5.5, playback_rate=2.0
    )
    assert abs(f_2x - f_1x // 2) <= 1

    hidden = torch.randn(1, 4, 64)
    sesqui_ids = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
    aligned, durs = aligner(hidden, sesqui_ids, target_frames=400)
    assert aligned.shape == (1, 400, 64)
    assert float(durs[0, 0]) <= MINOR_SYLLABLE_MAX_FRAMES + 1e-5
    assert float(durs[0, 1]) <= MINOR_SYLLABLE_MAX_FRAMES + 1e-5
    assert torch.allclose(durs.sum(), torch.tensor(400.0), atol=1e-2)


def test_deferred_syllables_are_excluded_from_gaussian_attention():
    # v10.7 AUD-107-09: a deferred (sesqui_ids == 0) syllable has a zero
    # duration, so its cumulative centre sits at target_frames. Before the
    # fix it still participated in the Gaussian softmax and stole attention
    # mass from the tail frames of the clause (measured: 1.628554 of the
    # 24-frame mass, per deferred token), squeezing the REAL syllables.
    import torch

    from backend.nn_cartridges.duration_aligner import LearnedDurationAligner

    torch.manual_seed(0)
    aligner = LearnedDurationAligner(d_model=64)

    B, L, D, T = 1, 4, 64, 24

    # The hidden state for the four-token call. The first two rows are the
    # REAL syllables; the last two are the deferred ones, given an extreme
    # value so that ANY attention mass they receive is glaringly visible.
    hidden = torch.randn(B, L, D)
    hidden[0, 2:, :] = 1e6

    # The two-token reference REUSES the first two rows of the same tensor.
    # Using an independent randn here would compare two different signals and
    # would fail regardless of whether the attention mask is correct.
    hidden_ref = hidden[:, :2, :].contiguous()

    sesqui_four = torch.tensor([[2, 2, 0, 0]], dtype=torch.long)
    sesqui_two = torch.tensor([[2, 2]], dtype=torch.long)

    aligned_four, durs_four = aligner(hidden, sesqui_four, target_frames=T)
    aligned_two, durs_two = aligner(hidden_ref, sesqui_two, target_frames=T)

    # The deferred syllables consume no frames.
    assert float(durs_four[0, 2]) == 0.0
    assert float(durs_four[0, 3]) == 0.0

    # The real syllables are unaffected by the deferred ones.
    assert torch.allclose(durs_four[0, :2], durs_two[0, :], atol=1e-4)

    # THE REGRESSION ASSERTION: the deferred tokens' hidden state (1e6) must
    # not reach the aligned output. Before the fix this differs by ~1e6.
    assert torch.allclose(aligned_four, aligned_two, atol=1e-4)
```

### 5.5 Gate 5: `backend/tests/test_corruptor_and_infill.py`
```python
import torch

from backend.contracts.schemas import CI_MICRO_CONFIG
from backend.nn_cartridges.s2st_console import SymmetricS2STConsole
from backend.training.augmentations import DynamicAcousticCorruptor


def test_vtlp_0hz_and_nyquist_invariance_and_infill_loss():
    mel = torch.randn(2, 100, 50)
    warped = DynamicAcousticCorruptor.warp_formants_vtlp(
        mel, alpha_min=0.85, alpha_max=1.15, pivot=0.65
    )
    assert warped.shape == mel.shape
    assert torch.allclose(warped[:, 0, :], mel[:, 0, :], atol=1e-5), "0 Hz row mutated!"
    assert torch.allclose(warped[:, -1, :], mel[:, -1, :], atol=1e-5), "Nyquist row mutated!"

    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    target_wav = torch.randn(2, 4800)
    loss = console.compute_flow_loss(target_wav, infill_mode=True)
    loss.backward()
    grads = [p.grad for p in console.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
# ===========================================================================
# v10.7 additions. The v10.6 assertions above are retained VERBATIM.
# ===========================================================================


def test_multi_teacher_ear_distiller_produces_a_finite_backward_loss():
    # v10.7 (P2-B): the MultiTeacherEarDistiller must contribute a finite,
    # differentiable loss, and CartridgeTrainer must actually AGGREGATE it.
    # v10.6 initialised the distiller but never called it.
    from backend.training.train_cartridge import (
        CartridgeTrainer,
        MultiTeacherEarDistiller,
    )

    torch.manual_seed(0)
    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    distiller = MultiTeacherEarDistiller(d_student=512, d_teacher=1024)

    student_l8 = torch.randn(2, 25, 512, requires_grad=True)
    student_bus = torch.randn(2, 25, 512, requires_grad=True)
    teacher = torch.randn(2, 20, 1024)

    loss = distiller(student_l8, student_bus, w2v_layer16_50hz=teacher)
    assert torch.isfinite(loss)
    loss.backward()
    assert student_l8.grad is not None and torch.isfinite(student_l8.grad).all()

    # The trainer must aggregate the distiller loss.
    trainer = CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")
    trainer.attach_distiller(distiller, weight=0.5)
    assert trainer.distiller is distiller
    assert trainer.distiller_weight == 0.5


def test_v107_trainer_applies_amp_and_checkpointing():
    # v10.7 (P2-B): TRAINING_PROFILES declares precision and
    # gradient_checkpointing, but v10.6 never read either. The trainer must
    # honour both, and the backward pass must run OUTSIDE the autocast
    # block (wrapping backward in autocast is an anti-pattern).
    from backend.training.train_cartridge import (
        TRAINING_PROFILES,
        CartridgeTrainer,
    )

    profile = TRAINING_PROFILES["laptop_5070ti_12gb"]
    assert profile.precision is torch.bfloat16
    assert profile.gradient_checkpointing is True

    torch.manual_seed(0)
    console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
    trainer = CartridgeTrainer(console, profile_name="laptop_5070ti_12gb")

    # The profile precision must be readable from the trainer.
    assert trainer.profile.precision is torch.bfloat16
    assert trainer.profile.gradient_checkpointing is True

    # A pre-computed loss tensor must still work (v10.6 calling convention).
    loss = torch.tensor(1.0, requires_grad=True)
    value = trainer.train_step(loss)
    assert isinstance(value, float)

    # A callable must run the forward under autocast and backward outside it.
    seen = {}

    def forward_fn():
        seen["in_autocast"] = torch.is_autocast_enabled()
        return torch.tensor(2.0, requires_grad=True)

    trainer.train_step(forward_fn)
    # On a CPU-only host the autocast context is disabled, which is correct:
    # the contract is that the FORWARD is inside the context and the
    # BACKWARD is outside it, not that the context is always enabled.
    assert "in_autocast" in seen


def test_v107_aud107_gradient_checkpointing_preserves_state_dict_keys():
    # v10.7 AUD-107-04: v10.7 wrapped each block in a `_CheckpointedBlock`
    # nn.Module holding the real block as `self.blk`, which renamed every
    # parameter key in state_dict() from `ear.blocks.0.self_attn...` to
    # `ear.blocks.0.blk.self_attn...`. A checkpoint trained under
    # laptop_5070ti_12gb (gradient_checkpointing=True) could then not be
    # loaded under cloud_h100_80gb (gradient_checkpointing=False) or into an
    # inference model. Measured: 70 of the 178 keys were renamed.
    import torch

    from backend.contracts.schemas import CI_MICRO_CONFIG
    from backend.nn_cartridges.s2st_console import SymmetricS2STConsole
    from backend.training.train_cartridge import apply_gradient_checkpointing

    # A private RNG state, so this test does not disturb the global seed that
    # the other Gate 5 tests rely on.
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(0)
        console = SymmetricS2STConsole(config=CI_MICRO_CONFIG)
        console.eval()

        keys_before = set(console.state_dict().keys())
        assert keys_before, "the model must expose parameters"

        wav = torch.randn(1, 8000)

        def run(model):
            # Control the stochastic diffusion start (x_0) so the two runs
            # are comparable.
            torch.manual_seed(1234)
            with torch.no_grad():
                out = model.forward_s2st(wav, "en", "km", 0.5)
            return out[0].clone()

        # The SAME instance, before and after checkpointing.
        a = run(console)

        apply_gradient_checkpointing(console)
        keys_after = set(console.state_dict().keys())

        # The key set must be IDENTICAL, so a checkpoint is portable between
        # profiles.
        assert keys_after == keys_before
        assert not any(".blk." in k for k in keys_after)

        b = run(console)
        assert torch.allclose(a, b, atol=1e-5)
    finally:
        torch.random.set_rng_state(rng_state)
```

### 5.6 Gate 6: `backend/tests/test_live_queue.py`
```python
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
```

---

## PART 6: STAGE 02 GROUND-TRUTH ACCEPTANCE TARGETS

| Gate | Target Subsystem & Files | Ground-Truth Acceptance Criteria | Executable Verification Command |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Linguistics: Rust + Python)** | `crates/khmer-linguistics-rs/` & `backend/linguistics/` | Passes all 11 rows of the Khmer Phonological Ground-Truth Table (`"គណនា"` $\rightarrow$ `[1,1,2]`, `"សុភមង្គល"` $\rightarrow$ `[1,1,2,2]`, `"សង្គ្រាម"` $\rightarrow$ `[2,2]`, `"កម្ពុជា"` $\rightarrow$ `[2,1,2]`, `"បញ្ហា"` $\rightarrow$ `[2,2]`, `"កន្លែង"` $\rightarrow$ `[2,2]` with `[1,1]` register, `"ប្រជា"` $\rightarrow$ `[1,2]`, `"ក្រសួង"` $\rightarrow$ `[1,2]`, `"កក្រើក"` $\rightarrow$ `[1,2]`, `"ស្ត្រី"` $\rightarrow$ `coeng_id > 0, medial_id = 1`, and `"Calvin Klein"` / `"Versace"` $\rightarrow$ `register_id = 3` + reversible `decode_batch_to_text`). | `cargo test -p khmer-linguistics-rs && pytest backend/tests/test_codeswitch_decomposer.py` |
| **Gate 2 (Symmetric 241M S2ST & GRL)** | `backend/nn_cartridges/` (`ear`, `core`, `pointer`, `mouth`, `s2st_console`) | `with torch.device("meta"):` parameter sum is in `[230_000_000, 250_000_000]` (measured: `241,637,457` = `241.637M`; CI profile measured: `3,838,033` = `3.838M`). Matrix rank of `mel_fb` and `mel_fb_24k` is strictly `100`. `isinstance` protocol checks pass. `EN -> KM` and `KM -> EN` (`register_ids == 3`) forward passes succeed on `CI_MICRO_CONFIG`. `GradientReversalFn` negates gradients (`grad < 0`). `AcousticPointerCodeSwitchBridge` blends source features when `force_gate=1.0`. | `pytest backend/tests/test_symmetric_s2st.py` |
| **Gate 3 (Multi-Speaker & Mode Gate)** | `backend/nn_cartridges/stem_and_mode_gate.py` | `filter_dominant_speakers` returns `(2, T)` for equal-energy stems and `(1, T)` when Speaker B is `-20 dB` below Speaker A. `separate_and_classify(..., force_dual_speakers=True)` outputs `frame_vocal_modes` `(2, T_frames)` and `erb_suppression_mask` `(T_frames, 64)` bounded in `[0.0, 1.0]`. | `pytest backend/tests/test_multispeaker_and_modes.py` |
| **Gate 4 (Fast-Talker Aligner & FFT Glue)** | `duration_aligner.py` & `stem_and_mode_gate.py` | `SceneAcousticMatcher()` does not mutate `torch.get_rng_state()`. `.backward()` through `apply_scene_glue` (`B=2, T=4800`) produces finite non-zero gradients on `dry_wav`, `mic_eq`, `rt60`, and `wet`. `LearnedDurationAligner` caps `sesqui_id == 1` syllables at `<= 6.0` frames (`target_frames=400`) and halves frames at `playback_rate=2.0`. | `pytest backend/tests/test_fast_talker_and_glue.py` |
| **Gate 5 (Corruptor, 0-Hz VTLP & Infill)** | `backend/training/augmentations.py` & `s2st_console.py` | `warp_formants_vtlp` preserves exact equality (`torch.allclose`) on frequency row `0` (`0 Hz`) and row `M-1` (`Nyquist`). `console.compute_flow_loss(target_wav, infill_mode=True).backward()` produces finite gradients on `CI_MICRO_CONFIG`. | `pytest backend/tests/test_corruptor_and_infill.py` |
| **Gate 6 (Transport, Rewind/LRU, WAV & TS/WASM)** | `crates/khmer-transport-rs/`, `crates/khmer-dsp-wasm/`, `backend/transport/`, `frontend/` | Rejects `20ms` micro-pop + `300ms` silence; slices `1.35s` utterance + `260ms` silence via Trigger #1 (`1.30s <= len <= 1.55s`); slices continuous speech or active clause wall-clock elapsed `>= 1.60s` in `[1.20s, 1.60s]` via Trigger #2. `reset_on_seek` clears partial buffers while preserving `SpeakerRegistry` centroids and hitting the 32-entry phase-invariant energy envelope LRU replay cache. Queue flushes in `< 5.0 ms`. `raw[:4] == b"RIFF"` and `raw[8:12] == b"WAVE"`. Rust workspace tests and `tsc --noEmit` pass 100%. | `cargo test --workspace && pytest backend/tests/test_live_queue.py && npm --prefix frontend run typecheck` |

---

## PART 7: STAGE 03 & 04 MULTI-AGENT ORCHESTRATION PLAN (ZERO FILE OVERLAP)

1. **Master Orchestrator (`Stage 00 & Stage 01`):**
   * **Owned Files:** `Cargo.toml`, `pyproject.toml`, `README.md`, `frontend/package.json`, `frontend/tsconfig.json`, `backend/contracts/schemas.py`, `frontend/src/contracts/types.ts`, and all 6 test files in `backend/tests/test_*.py`.
   * **Responsibility:** Initializes the directory tree, writes the locked contracts and Red-Phase TDD test suite, verifies the initial Red failure, and locks `backend/contracts/` and `backend/tests/` as read-only before dispatching Specialists A, B, C, and D.
2. **Specialist A — PyTorch Neural Cartridges & Batched FFT Scene Glue:**
   * **Owned Files (Strict Isolation):** `backend/nn_cartridges/__init__.py`, `backend/nn_cartridges/stem_and_mode_gate.py`, `backend/nn_cartridges/ear_cartridge.py`, `backend/nn_cartridges/universal_core.py`, `backend/nn_cartridges/codeswitch_router.py`, `backend/nn_cartridges/duration_aligner.py`, `backend/nn_cartridges/mouth_cartridge.py`, `backend/nn_cartridges/s2st_console.py`.
3. **Specialist B — Rust + Python 7-Slot Khmer & Code-Switch Linguistic Engine:**
   * **Owned Files (Strict Isolation):** `crates/khmer-linguistics-rs/Cargo.toml`, `crates/khmer-linguistics-rs/src/lib.rs`, `crates/khmer-linguistics-rs/src/normalizer.rs`, `crates/khmer-linguistics-rs/src/decomposer.rs`, `backend/linguistics/__init__.py`, `backend/linguistics/khmer_normalizer.py`, `backend/linguistics/syllable_decomposer.py`. *(v10.7: the `#[pymodule]` ABI required by the Python bridge is DELIVERED in Sections 4.9.4 and 4.9.5, and Section 4.9.7 records the R-1 to R-8 acceptance checks; fix-ledger entries 72 to 74.)*
4. **Specialist C — Bad-YouTube Corruptor, 0-Hz VTLP & Dual-Profile Trainer:**
   * **Owned Files (Strict Isolation):** `backend/training/__init__.py`, `backend/training/augmentations.py`, `backend/training/train_cartridge.py`, `backend/training/eval_s2st_suite.py` *(v10.6: `eval_s2st_suite.py` was specified in Section 4.6.3 but was assigned to no owner in the v10.5 plan; it is now explicitly owned here.)*
5. **Specialist D — Rust Real-Time Transport, Rewind Cache, WASM 48kHz Audio DSP & TypeScript Client:**
   * **Owned Files (Strict Isolation):** `crates/khmer-transport-rs/Cargo.toml`, `crates/khmer-transport-rs/src/lib.rs`, `crates/khmer-transport-rs/src/session.rs`, `crates/khmer-transport-rs/src/live_queue.rs`, `crates/khmer-transport-rs/src/wav_container.rs`, `crates/khmer-dsp-wasm/Cargo.toml`, `crates/khmer-dsp-wasm/src/lib.rs`, `backend/transport/__init__.py`, `backend/transport/session.py`, `backend/transport/live_queue.py`, `backend/transport/runner.py`, `backend/transport/server.py`, `frontend/src/worklets/pcm-capture-processor.ts`, `frontend/src/audio/stereo-mne-mixer.ts`, `frontend/src/client/duplex-ws-client.ts`, `frontend/src/ui/caption-overlay.ts`.
6. **Adversarial Critic Agent (`Stage 04`):**
   * Runs `cargo test --workspace`, `pytest backend/tests/ -v`, and `npm --prefix frontend run typecheck`. Rejects any mock pass-through or assertion weakening and loops the owning specialist until every gate is 100% green.

---

## PART 8: STAGE 05 COPY-PASTEABLE MASTER ORCHESTRATION PROMPT

```text
ROLE & OPERATING MODE
You are the Master Orchestrator building KHMER-SOVEREIGN-S2ST (v10.7 - Code-Level Audit Remediation Edition). You operate under strict Red-to-Green Test-Driven Development (TDD) and Zero File Overlap across 4 isolated specialist tracks. Never invent mock pass-throughs or weaken test assertions to make tests pass.

STAGE 00: WORKSPACE & HYBRID TOOLCHAIN INITIALIZATION
1. Create the full directory hierarchy:
   - crates/khmer-linguistics-rs/src, crates/khmer-transport-rs/src, crates/khmer-dsp-wasm/src
   - backend/contracts, backend/linguistics, backend/nn_cartridges, backend/training, backend/transport, backend/tests
   - frontend/src/contracts, frontend/src/worklets, frontend/src/audio, frontend/src/client, frontend/src/ui
2. Generate root `Cargo.toml` (with workspace.dependencies for pyo3, version="10.7.0"), `pyproject.toml` (torch>=2.7.0, version="10.7.0", plus `sacrebleu`/`transformers`/`funasr` for the Section 4.6.3 eval suite), `frontend/package.json` (version="10.7.0"), and `frontend/tsconfig.json` exactly as specified in Part 2.3 of the v10.7 Blueprint. The authoritative version string is `10.7.0` and is enumerated in the Part 2.3.1 Version Manifest; use it verbatim and do not re-derive it.

STAGE 01: IMMUTABLE CONTRACTS & ADVERSARIAL TDD SUITE (RED PHASE)
1. Implement `backend/contracts/schemas.py` and `frontend/src/contracts/types.ts` strictly enforcing every constant (N_FFT_16K=1024, HOP_LENGTH_16K=160, N_FFT_24K=2048, HOP_LENGTH_24K=240, N_FFT_48K=4096, HOP_LENGTH_48K=480), enum, frozen dataclass (including `interrupt_epoch: int = 0`), `@runtime_checkable` protocol in Part 2.4, and locked submodule signature in Part 2.5.
2. Act as the Adversarial Critic Agent and write all 6 verification test files in `backend/tests/` (`test_codeswitch_decomposer.py`, `test_symmetric_s2st.py` with full-rank Slaney checks, `test_multispeaker_and_modes.py`, `test_fast_talker_and_glue.py`, `test_corruptor_and_infill.py`, `test_live_queue.py`) exactly as specified in Part 5.
3. Run `pytest backend/tests/` to confirm the Red Phase fails on missing specialist modules. Lock `backend/contracts/` and `backend/tests/` as read-only.

STAGE 02: PARALLEL SPECIALIST EXECUTION (GREEN PHASE - ZERO FILE OVERLAP)

Every subsystem below carries a v10.5, v10.6 or v10.7 remediation. The v10.7
items are marked **[v10.7]** and are all PARAMETER-FREE: the authoritative
ledger must remain `241,637,457` (`prod_240m`) and `3,838,033` (`ci_micro`).

- Specialist A (`backend/nn_cartridges/*`):
  1. Implement `stem_and_mode_gate.py` (keeping N_BINS=257 for the 3-stem separator mask
     and N_BINS_MEL=513 for mel_fb), `ear_cartridge.py` (N_BINS=513, rank 100),
     `universal_core.py` (with `build_slaney_mel_fb`, N_BINS_24K=1025, and VocosISTFTHead
     at 7.753M params), `codeswitch_router.py`, `duration_aligner.py`,
     `mouth_cartridge.py` (with `reorder_attn` and normalized sinusoidal queries), and
     `s2st_console.py` (N_BINS_24K=1025, rank 100, HOP_LENGTH=240). Use
     `torch.Generator(device="cpu").manual_seed(42)` in `SceneAcousticMatcher` with a
     jittered velvet grid, and use batched `torch.fft.rfft`/`irfft` in `apply_scene_glue`.
  2. **[v10.7] P1-E (normalization stability):** `FastConformerBlock` MUST use
     `nn.GroupNorm(8, d)` and MUST NOT use `nn.BatchNorm1d(d)`. BatchNorm
     normalizes over the batch axis, which has size 1 during streaming inference, so its
     running statistics are corrupted by batch-statistics estimation. GroupNorm
     normalizes per sample and is correct at B=1. The swap is PARAMETER-NEUTRAL: both
     hold 2*d trainable parameters (1,024 in prod, 256 in CI per block), and
     BatchNorm's running buffers are not counted by `.parameters()`.
  3. **[v10.7] P0-A (stateless cartridges):** `StreamingEarCartridge` and
     `KhmerMouthCartridge` MUST NOT expose the per-call taps
     (`last_l6_hidden`, `last_l8_hidden`, `last_next_past_kv`, `last_next_past_gate`,
     `last_deferred_syll_count`, `last_pred_durations`, `last_gate_l`) as mutable module
     attributes. Each MUST be a `@property` bridge onto the `CartridgeThreadState`
     container from `backend/contracts/schemas.py`, which is backed by
     `threading.local()`. The property bridges preserve 100% attribute
     compatibility for every existing call site, while the underlying storage is
     strictly per-thread.
  4. **[v10.7] P0-A (B_spk guard):** `predict_syllables_and_align` MUST discard a
     stale `past_clause_kv` whose `shape[0]` does not equal the current batch, instead
     of concatenating mismatched batch dimensions (which raises a `RuntimeError` when
     the number of active speakers changes between clauses).
  5. **[v10.7] P0-A (batch-safe deferral):** `deferred_syll_count` MUST be
     computed as `int(wait_mask.float().sum(dim=-1).max().item())`, not by indexing
     `wait_mask[0]`, which silently ignored speakers 1..B-1.
  6. **[v10.7] P1-C (step conditioning):** `predict_velocity` MUST embed the
     timestep and the step size with the parameter-free `sinusoidal_time_embedding`
     helper. The v10.6 code expanded a rank-1 scalar across every column, so
     `time_mlp` only ever saw `w1*t + w2*dt + b` and could not express any non-linear
     function of t. The embedding is parameter-free, so `time_mlp` keeps its
     `Linear(d, d)` shape and the ledger is unchanged.
  7. **[v10.7] P1-D (speaker timbre):** `forward_s2st` MUST add the L2-normalized,
     mean-pooled layer-6 acoustic representation to `aligned_hidden` before
     `self.cond_proj`, preserving vocal-tract identity in the acoustic conditioning
     path. This is PARAMETER-FREE.
  8. **[v10.7] P1-C (grounded distillation):** `compute_shortcut_distillation_loss`
     MUST sample its coarse-to-fine step pairs DIRECTLY from the Sway schedule
     (`SWAY_COEF = -0.8`, `SC_DISTILL_STEPS = 12`) using the shared
     `sway_time_schedule` helper, so the training grid and the inference grid are
     provably identical. The two finest teacher sub-steps MUST be evaluated with the
     instantaneous velocity network (`dt=None`, i.e. `dt = 0.0`), which anchors the
     whole hierarchy to the ground-truth OT-CFM vector field, and the teacher target
     MUST be the step-weighted average `(dt_1 * v_step1 + dt_2 * v_step2) / dt_full`.
  9. **[v10.7] P1-C (CFG consistency):** `integrate_shortcut_sway_ode` MUST accept a
     `cfg_scale` and blend the conditional and unconditional velocities, and the
     training-side CFG dropout MUST zero EXACTLY the tensors the inference
     unconditional branch zeros (`cond` and `emo_tokens`), so the guided direction
     points at a branch the model was actually trained on. `cfg_scale = 1.0`
     reproduces the v10.6 behaviour exactly.
  10. **[v10.7] P1-C (non-circular durations):** `compute_s2st_training_losses` MUST
      supervise `pred_durs` against the monotonic attention mass of `reorder_attn`
      (`attn_w.sum(dim=-1)`, scaled to `T_mel`, with the 6-frame minor-syllable cap and
      the excess redistributed to the major syllables), NOT against a static
      0.25 / 1.00 prior that carries no acoustic information.

  11. **[v10.7] AUD-107-01 (deferred wait-gating):** `predict_syllables_and_align` MUST
      apply the `wait_mask` (0, 0, 0) no-op masking AFTER the
      `if target_lang == "en":` branch, not before it. The v10.7 code masked
      first and the English branch then overwrote `onset_ids`, `vowel_ids` and
      `coda_ids` with the reserved English ranges, so on a KM to EN Trigger #2
      slice the deferred tail was never masked: the tail syllables were emitted
      in the current clause AND deferred to the next one, duplicating the words.
      The mask MUST also cover `coeng_ids` and `sesqui_ids`, so a deferred
      syllable is a COMPLETE no-op. `sesqui_ids == 0` is the zero-frame
      sentinel consumed by `LearnedDurationAligner.forward`, which MUST give it a
      0.0 duration prior weight, so the diffusion model stops allocating
      acoustic frames to a syllable that is not spoken in this clause.
  12. **[v10.7] AUD-107-09 (deferred syllables must not steal attention):**
      `LearnedDurationAligner.forward` MUST mask the `sesqui_ids == 0` columns to
      -1e9 in `attn_logits` BEFORE `F.softmax`. A deferred syllable has a zero
      duration, so its cumulative centre sits at `target_frames`, but it still
      received softmax mass there and squeezed the REAL syllables (measured:
      1.628554 of the 24-frame mass per deferred token). The mask MUST be
      applied only where the row has at least one active syllable
      (`has_active`), so a fully deferred row stays well defined.
  13. **[v10.7] AUD-107-02 (the ear runs once per slice):**
      `SymmetricS2STConsole.forward_s2st` MUST accept
      `precomputed_ear: Optional[PrecomputedEarFeatures] = None`, and
      `FullDuplexRunner.process_utterance_slice` MUST pass it, so that
      `StreamingEarCartridge` and `log_mel` execute only ONCE per utterance
      slice. The v10.7 runner ran the ear, then `log_mel` twice more, then the
      emotion encoder whose output was discarded, and then called `forward_s2st`,
      which ran the ear and the emotion encoder a second time on the identical
      audio, doubling the 64.5M-parameter ear and the 8.7M-parameter emotion
      encoder on every spoken utterance.

- Specialist B (`crates/khmer-linguistics-rs/*` & `backend/linguistics/*`):
  1. Implement the Rust normalizer and 7-slot syllable decomposer in
     `crates/khmer-linguistics-rs` and expose `BilingualCodeSwitchDecomposer` in
     `backend/linguistics/syllable_decomposer.py` preserving `\u200b`, integrating
     `khmer-nltk` for unspaced sub-spans, partitioning all 14 Khmer codas into 8
     phonological classes `[1..8]` via `CODA_GROUP_MAP`, encoding Bantoc codas into
     `[9..16]`, and using deterministic G2P phonetic approximation for Latin loanwords
     and foreign brands with 63-way `coeng_id` hash disambiguation and open-syllable
     fallback.
  2. **[v10.7] P1-A (independent vowels):** `khmer_normalizer.py` MUST canonicalize
     U+17AB..U+17AE to a consonant plus a dependent vowel (U+179A U+17B9, U+179A U+17BA,
     U+179B U+17B9, U+179B U+17BA) and MUST map U+17A8 to the open (vowel-final)
     form U+17A2 U+17BB U+1780. The v10.6 map sent these four codepoints to
     THEMSELVES, and because `_SYLLABLE_SCAN_RE` begins at U+1780 the decomposer
     SKIPPED them entirely, silently deleting the first syllable of any word that
     began with one. The Rust `normalize_khmer_pure` MUST apply the identical
     mapping so the two paths agree.
  3. **[v10.7] P1-B (English alphabet):** `syllable_decomposer.py` MUST reserve
     `coeng_id == 0` for SYNTHESIZED English target syllables and
     `coeng_id` in 1..63 for INPUT code-switched Latin brands, and MUST decode the
     former through `ENGLISH_SYLLABLE_TABLE` (onset ids 34..57, medial ids 0..3,
     vowel ids 28..44, coda ids 18..31) rather than through the hardcoded CVC
     fallback that ignored `coeng_id` and emitted text such as "kak".
  4. **[v10.7] P1-B (brand slot integrity):** `_foreign_to_token` MUST keep
     exactly ONE token per brand word, and every register-3 token MUST satisfy
     `18 <= coda_id <= 31`. Slot 18 is the OPEN (vowel-final) coda and 19..31 are the
     13 closed consonant codas.

- Specialist C (`backend/training/*`):
  1. Implement `DynamicAcousticCorruptor` (`corrupt_youtube_mixture` with
     primary-RMS-relative SNR and decaying room echo, plus 2D `(B, M, T, 2)`
     0-Hz/Nyquist-anchored `warp_formants_vtlp`) in `augmentations.py` and
     `CartridgeTrainer` in `train_cartridge.py`.
  2. **[v10.7] P2-B (mixed precision):** `CartridgeTrainer.__init__` MUST honour
     `self.profile.gradient_checkpointing` by applying activation checkpointing to the
     ear and mouth cartridge block stacks, and `train_step` MUST accept EITHER a
     pre-computed loss tensor OR a callable that performs the forward pass. The
     forward pass runs under `torch.amp.autocast`; the backward pass MUST run
     OUTSIDE the autocast block. Wrapping `.backward()` in autocast is a
     documented PyTorch anti-pattern.
  3. **[v10.7] P2-B (teacher distillation):** `CartridgeTrainer` MUST expose an
     `attach_distiller` method and MUST aggregate the `MultiTeacherEarDistiller`
     loss with the main loss inside the same autocast forward pass. The v10.6
     trainer initialised the distiller but never called it.
  4. **[v10.7] P2-C (benchmark entry point):** `eval_s2st_suite.py` MUST implement
     `evaluate()`, wiring ASR-BLEU, chrF, UTMOS, DNSMOS, speaker similarity,
     emotion similarity and mean turnaround latency into the existing
     `S2STEvalReport` dataclass, whose fields were otherwise never assigned.

  5. **[v10.7] AUD-107-04 (checkpointing must not rename state_dict keys):**
     `apply_gradient_checkpointing` MUST bind an instance-level `types.MethodType`
     `forward` wrapper (flagged `_v107_checkpointed`) on each block, rather than
     wrapping the block in a child `nn.Module`. The v10.7 `_CheckpointedBlock`
     held the real block as `self.blk`, which renamed every parameter key from
     `ear.blocks.0.self_attn...` to `ear.blocks.0.blk.self_attn...` and made a
     laptop-profile checkpoint unloadable under the cloud profile. The nn.Module
     hierarchy MUST be left untouched, preserving 100% of
     `model.state_dict().keys()`.

- Specialist D (`crates/khmer-transport-rs/*`, `crates/khmer-dsp-wasm/*`, `backend/transport/*` & `frontend/src/*`):
  1. Implement `InMemorySession` (with `reset_on_seek`, active clause wall-clock elapsed
     ceiling `utterance_wall_clock_ms - silence_ms >= 1600ms` on Trigger #2,
     $\pm 1$ duration-bucket fuzzy match with `cached_crc` validation on live seeks in
     `lookup_cached_event`, 8-bin normalized energy envelope replay cache, and
     continuous timestamp clock sync), Cosine `SpeakerRegistry`, Lock-Free
     `LiveRequestQueue`, and `FullDuplexRunner` (with in-flight epoch fencing and
     speaker centroids pass-through) in `backend/transport/`.
  2. Implement the TypeScript frontend in `frontend/src/` (`pcm-capture-processor.ts`
     with unity-gain 15-tap sinc decimator `sum = 1.0000`, `stereo-mne-mixer.ts` with
     4096-bin ERB suppressor mapping all 64 bands via `Math.floor(ERB_BANDS * norm)`
     and client-epoch fencing across `decodeAudioData`, `duplex-ws-client.ts` with
     non-destructive event listeners, and `caption-overlay.ts` with XSS-safe DOM node
     creation).
  3. **[v10.7] P0-B (lock coverage):** `InMemorySession.append_pcm_chunk`,
     `cache_dubbed_event` and `lookup_cached_event` MUST hold `self._lock` for their
     whole bodies, and `SpeakerRegistry` MUST own a `threading.RLock` held across the
     read-modify-write of `self.centroids`. The v10.6 code mutated all four
     without any lock, so the replay cache could be resized during iteration and the
     centroid list could be observed half-updated.
  4. **[v10.7] P0-A (KV lifecycle):** `InMemorySession.append_pcm_chunk` MUST flush
     `past_clause_kv`, `past_gate_kv` and `deferred_syll_count` on a Trigger #1
     natural pause and on a micro-pop rejection, and MUST track
     `last_slice_was_hard_ceiling`. `FullDuplexRunner.process_utterance_slice` MUST
     pass the session's rolling state and `is_hard_ceiling_slice` into
     `forward_s2st`, and MUST commit the new state back to the session ONLY after the
     post-synthesis epoch fence has passed. The v10.6 runner never passed the four
     rolling-state arguments at all, so the feature was dead code in production.
  5. **[v10.7] P1-D (registry identity):** `process_utterance_slice` MUST register
     speakers using an energy-weighted, L2-normalized mean pool of the layer-6
     acoustic representation, NOT the `DisentangledEmotionEncoder` output. The
     emotion encoder applies `F.instance_norm` and `avg_pool1d(k=25)`, which remove
     exactly the spectral envelope the registry needs to tell speakers apart.
  6. **[v10.7] P0-B (epoch fence):** `worker_loop` MUST snapshot
     `pkt_epoch = queue.interrupt_epoch` BEFORE awaiting the GPU semaphore and MUST
     re-check it inside the semaphore, aborting the slice when it changed. The
     v10.6 snapshot was taken inside the worker thread, i.e. after the semaphore
     wait, so a seek that landed during the wait was invisible to the fence and the
     stale pre-seek audio was emitted after the `interrupted` packet.
  7. **[v10.7] P1-E (upstream capture):** `duplex-ws-client.ts` MUST gain an
     `attachAudioCapture(ctx)` that creates the `AudioWorkletNode`, forwards the
     worklet's binary `ArrayBuffer` frames to `ws.send`, and posts
     `{ baseVideoTimeMs, playbackRate }` on connect, seeking, playing and ratechange.
     The v10.6 client never instantiated the worklet and never sent a binary frame,
     so the server's `upstream_reader` waited forever for bytes that never arrived
     and the full-duplex loop could not function.
  8. **[v10.7] P1-E (media clock):** `pcm-capture-processor.ts` MUST advance its
     media clock by the wall-clock delta scaled by the video `playbackRate`, so the
     upstream timestamps track the VIDEO timeline rather than the 1.0x AudioContext
     clock.
  9. **[v10.7] AUD-107-03 / AUD-107-05 / AUD-107-06 (frontend DSP, transport and
     real-time hygiene):** `StereoMneMixer` MUST precompute `COS_TABLE_MNE`,
     `SIN_TABLE_MNE`, `COS_TABLE_MNE_INV`, `SIN_TABLE_MNE_INV` and `REV_TABLE_MNE`
     ONCE at module load, rather than allocating the twiddle tables and calling
     `Math.log2(size)` inside the bit-reversal loop of every 4096-sample block
     (98,304 `Math.log2` calls per block, inside a real-time WebAudio callback
     where a GC pause is an audible dropout).
     `attachMediaElement` MUST insert a `DelayNode` of `LOOKAHEAD_BUFFER_SEC`
     (2.0 s) between the `source` (`MediaElementAudioSourceNode`) and the
     `processor` (`ScriptProcessorNode`), while still returning the UNDELAYED
     source node so `DuplexWsClient.attachAudioCapture` can tap the audio ahead
     of playback. Because the M&E audio stream is delayed by 2.0 s, the
     `onaudioprocess` callback queries
     `this.processMidSideFrame(this.ctx.currentTime, ...)` and reconstructs the
     time-domain Mid channel via `_inverseSpectrum`; `handleDownstreamEvent`
     MUST only push the decoded segment into `this.erbQueue` and MUST NOT run a
     redundant FFT pass. The v10.7 prompt resolved the mask at the segment's own
     `lookaheadTargetTime`, which the 2.0 s DelayNode invalidated.
     `DuplexWsClient.attachAudioCapture` MUST route the capture worklet into a
     `GainNode` with `gain.value = 0.0` connected to `ctx.destination`, so
     undelayed dry audio can never bleed to the speakers and bypass the M&E
     suppressor.
  10. **[v10.7] AUD-107-02 (the ear runs once per slice):**
      `FullDuplexRunner.process_utterance_slice` MUST pass a
      `PrecomputedEarFeatures` bundle into `console.forward_s2st`, so the
      `StreamingEarCartridge` runs once per slice instead of twice.

STAGE 03: ADVERSARIAL CRITIC VERIFICATION LOOP
Execute all verification gates:
1. `cargo test --workspace` (if `cargo` is installed in the environment)
2. `pytest backend/tests/ -v`
3. `npm --prefix frontend run typecheck`
If any check fails, route the exact traceback to the owning specialist track to fix their implementation file—never modify or weaken `backend/tests/` or `backend/contracts/`. Sign off only when all 6 gates pass 100%.

STAGE 06: v10.7 DOCUMENT-TRUTH VALIDATION (MANDATORY BEFORE SIGN-OFF)
1. Re-derive the Part 3.3 parameter ledger by instantiating every module under `torch.device("meta")` and summing `p.numel()`. The totals must equal `241,637,457` (`prod_240m`) and `3,838,033` (`ci_micro`) exactly. Do not accept a rounded or hand-written figure.
2. Sweep for stale version tokens: `10.7.0` must appear in exactly the five locations listed in the Part 2.3.1 Version Manifest, and no `10.4.0` or `10.5.0` version token may remain anywhere in the blueprint or the `docs/` volumes.
3. Confirm the `docs/` volumes regenerate from the master blueprint with zero content divergence, and that the line/token counts published in `docs/README.md` are re-derived from the regenerated files rather than carried over.
4. Confirm the ERB Band 63 lower edge is stated as `7603.5 Hz` (recompute from `hzToErbRate`; `7353 Hz` is the Band 62 edge).
5. Confirm every file in the Part 2.2 tree has exactly one owner in Part 7, and that the Phase 2 Rust ABI deliverables in Section 4.9.7 are recorded as **specifications** rather than silently dropped.
```
