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
