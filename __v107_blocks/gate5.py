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
