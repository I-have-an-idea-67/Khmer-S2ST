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
