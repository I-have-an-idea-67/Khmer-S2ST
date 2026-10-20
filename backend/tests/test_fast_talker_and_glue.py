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
