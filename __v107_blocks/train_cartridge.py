from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.nn_cartridges.universal_core import VocosISTFTHead


@dataclass(frozen=True)
class TrainingProfileConfig:
    profile_name: str
    precision: torch.dtype
    micro_batch_size: int
    grad_accum_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    weight_decay: float


TRAINING_PROFILES: Dict[str, TrainingProfileConfig] = {
    "laptop_5070ti_12gb": TrainingProfileConfig(
        profile_name="laptop_5070ti_12gb",
        precision=torch.bfloat16,
        micro_batch_size=4,
        grad_accum_steps=16,
        gradient_checkpointing=True,
        learning_rate=3e-4,
        weight_decay=0.01,
    ),
    "cloud_h100_80gb": TrainingProfileConfig(
        profile_name="cloud_h100_80gb",
        precision=torch.bfloat16,
        micro_batch_size=64,
        grad_accum_steps=1,
        gradient_checkpointing=False,
        learning_rate=3e-4,
        weight_decay=0.01,
    ),
}


def compute_total_multitask_loss(
    l_cfm: torch.Tensor,
    l_dur: torch.Tensor,
    l_lang_grl: torch.Tensor,
    l_stems: torch.Tensor,
) -> torch.Tensor:
    return l_cfm + 0.5 * l_dur + 0.2 * l_lang_grl + 0.3 * l_stems


class MultiTeacherEarDistiller(nn.Module):
    """
    Training-only wrapper distilling w2v-BERT 2.0 (50Hz, Layer 16 -> ear.last_l8_hidden),
    Canary-1B-v2 / Parakeet-TDT-0.6B-v3 (12.5Hz -> ear.bus_proj), and Qwen3-ASR-0.6B-Khmer
    into StreamingEarCartridge (25Hz) with zero inference parameter overhead.
    """

    def __init__(self, d_student: int = 512, d_teacher: int = 1024) -> None:
        super().__init__()
        self.proj_w2v_l8 = nn.Linear(d_student, d_teacher)
        self.proj_canary_l16 = nn.Linear(d_student, d_teacher)
        self.proj_qwen_km_l16 = nn.Linear(d_student, d_teacher)
        self.ln = nn.LayerNorm(d_teacher, elementwise_affine=False)

    @staticmethod
    def _align_temporal(teacher_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        if teacher_feat.shape[1] == target_len:
            return teacher_feat
        return F.interpolate(
            teacher_feat.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _pair_loss(self, stu_proj: torch.Tensor, tea_aligned: torch.Tensor) -> torch.Tensor:
        cos_loss = 1.0 - F.cosine_similarity(stu_proj, tea_aligned, dim=-1).mean()
        l1_loss = F.smooth_l1_loss(self.ln(stu_proj), self.ln(tea_aligned))
        return cos_loss + 0.5 * l1_loss

    def forward(
        self,
        student_l8_hidden: torch.Tensor,
        student_bus_out: torch.Tensor,
        w2v_layer16_50hz: Optional[torch.Tensor] = None,
        canary_top_12_5hz: Optional[torch.Tensor] = None,
        qwen_km_top: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T_ear = student_bus_out.shape[1]
        loss = torch.tensor(0.0, device=student_bus_out.device, dtype=student_bus_out.dtype)
        if w2v_layer16_50hz is not None:
            w2v_25hz = self._align_temporal(w2v_layer16_50hz.detach(), T_ear)
            loss = loss + self._pair_loss(self.proj_w2v_l8(student_l8_hidden), w2v_25hz)
        if canary_top_12_5hz is not None:
            can_25hz = self._align_temporal(canary_top_12_5hz.detach(), T_ear)
            loss = loss + self._pair_loss(self.proj_canary_l16(student_bus_out), can_25hz)
        if qwen_km_top is not None:
            qwen_25hz = self._align_temporal(qwen_km_top.detach(), T_ear)
            loss = loss + self._pair_loss(self.proj_qwen_km_l16(student_bus_out), qwen_25hz)
        return loss


def warm_start_vocos_from_pretrained(
    vocos_head: VocosISTFTHead, pretrained_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Surgical loader mapping charactr/vocos-mel-24khz (N_FFT=1024) into N_FFT_24K=2048."""
    with torch.no_grad():
        vocos_head.in_conv.weight.copy_(pretrained_state_dict["backbone.embed.weight"])
        vocos_head.in_conv.bias.copy_(pretrained_state_dict["backbone.embed.bias"])
        inter_half = vocos_head.blocks[0].pwconv1.out_features

        for i, blk in enumerate(vocos_head.blocks):
            prefix = f"backbone.convnext.{i}."
            blk.dwconv.weight.copy_(pretrained_state_dict[prefix + "dwconv.weight"])
            blk.dwconv.bias.copy_(pretrained_state_dict[prefix + "dwconv.bias"])
            blk.norm.weight.copy_(pretrained_state_dict[prefix + "norm.weight"])
            blk.norm.bias.copy_(pretrained_state_dict[prefix + "norm.bias"])

            pw1_w = pretrained_state_dict[prefix + "pwconv1.weight"]
            pw1_b = pretrained_state_dict[prefix + "pwconv1.bias"]
            pw2_w = pretrained_state_dict[prefix + "pwconv2.weight"]
            pw2_b = pretrained_state_dict[prefix + "pwconv2.bias"]

            importance = pw1_w.norm(p=2, dim=1) * pw2_w.norm(p=2, dim=0)
            top_idx = torch.topk(importance, k=inter_half).indices.sort().values

            blk.pwconv1.weight.copy_(pw1_w[top_idx, :])
            blk.pwconv1.bias.copy_(pw1_b[top_idx])
            blk.pwconv2.weight.copy_(pw2_w[:, top_idx])
            blk.pwconv2.bias.copy_(pw2_b)

        old_w = pretrained_state_dict["head.out.weight"]
        old_b = pretrained_state_dict["head.out.bias"]
        mag_w, phs_w = old_w[:513, :].T.unsqueeze(0), old_w[513:, :].T.unsqueeze(0)
        new_mag_w = F.interpolate(mag_w, size=1025, mode="linear", align_corners=True).squeeze(0).T
        new_phs_w = F.interpolate(phs_w, size=1025, mode="linear", align_corners=True).squeeze(0).T
        vocos_head.istft_head.weight.copy_(torch.cat([new_mag_w, new_phs_w], dim=0))

        mag_b = F.interpolate(old_b[:513].view(1, 1, 513), size=1025, mode="linear", align_corners=True).view(-1)
        phs_b = F.interpolate(old_b[513:].view(1, 1, 513), size=1025, mode="linear", align_corners=True).view(-1)
        vocos_head.istft_head.bias.copy_(torch.cat([mag_b, phs_b], dim=0))


def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    v10.7 P2-B: enable activation checkpointing on the two large cartridge
    stacks, so the laptop_5070ti_12gb profile can hold its 3.2 GB VRAM
    ceiling on a 12 GB RTX 5070 Ti.

    v10.7 AUD-107-04: v10.7 wrapped each block in a `_CheckpointedBlock`
    nn.Module that held the real block as `self.blk`. That RENAMED every
    parameter key in `model.state_dict()` from `ear.blocks.0.self_attn...` to
    `ear.blocks.0.blk.self_attn...`, so a checkpoint trained under the
    laptop profile (gradient_checkpointing=True) could not be loaded under the
    cloud profile (gradient_checkpointing=False) or into an inference model.
    70 of the 178 keys were affected.

    The fix replaces the module wrapper with an INSTANCE-LEVEL forward
    override. The nn.Module hierarchy is left completely untouched, so the
    state_dict() keys are bit-for-bit identical with and without
    checkpointing, and a checkpoint is portable between profiles.

    use_reentrant=False is used because the reentrant implementation does not
    support the non-tensor keyword arguments these blocks receive.

    Adds NO parameters.
    """
    import types

    from torch.utils.checkpoint import checkpoint

    def _make_checkpointed_forward(fn):
        def _checkpointed_forward(self, *args, **kwargs):
            return checkpoint(fn, *args, use_reentrant=False, **kwargs)

        return _checkpointed_forward

    for attr in ("ear", "mouth"):
        cartridge = getattr(model, attr, None)
        if cartridge is None:
            continue
        blocks = getattr(cartridge, "blocks", None)
        if blocks is None:
            continue
        if not isinstance(blocks, nn.ModuleList):
            continue
        for blk in blocks:
            if getattr(blk, "_v107_checkpointed", False):
                continue
            blk.forward = types.MethodType(
                _make_checkpointed_forward(blk.forward), blk
            )
            blk._v107_checkpointed = True
    return model


class CartridgeTrainer:
    def __init__(self, model: nn.Module, profile_name: str = "laptop_5070ti_12gb") -> None:
        if profile_name not in TRAINING_PROFILES:
            raise KeyError(f"Unknown profile: {profile_name}")
        self.profile = TRAINING_PROFILES[profile_name]
        self.model = model
        # v10.7 P2-B: TRAINING_PROFILES declares precision and
        # gradient_checkpointing, but v10.6 never read either of them, so the
        # laptop_5070ti_12gb profile could not hold its 3.2 GB VRAM ceiling on
        # a 12 GB RTX 5070 Ti. Gradient checkpointing is configured on the
        # model HERE, at construction time, because it must be in place BEFORE
        # the first forward pass.
        if self.profile.gradient_checkpointing:
            self.model = apply_gradient_checkpointing(self.model)
        # v10.7 P2-B: the MultiTeacherEarDistiller is training-only (it has
        # zero inference parameter overhead) but v10.6 never aggregated its
        # loss, so the three teacher streams contributed nothing to training.
        self.distiller: Optional[MultiTeacherEarDistiller] = None
        self.distiller_weight: float = 0.5
        # v10.7 P2-B: the teacher feature tensors, remembered so that
        # train_step can compute a non-zero distillation loss.
        self.teacher_targets: Dict[str, torch.Tensor] = {}
        self.step_count: int = 0
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.profile.learning_rate,
            weight_decay=self.profile.weight_decay,
        )

    def autocast_context(self):
        """
        v10.7 P2-B: the mixed-precision context for the FORWARD pass.

        Only the forward pass runs under autocast. The backward pass MUST run
        OUTSIDE the autocast block: PyTorch's autocast is not designed to
        wrap backward, and doing so is a documented anti-pattern that can
        silently produce wrong gradient scales.
        """
        return torch.autocast(
            device_type="cuda",
            dtype=self.profile.precision,
            enabled=torch.cuda.is_available(),
        )

    def attach_distiller(
        self,
        distiller: MultiTeacherEarDistiller,
        weight: float = 0.5,
        teacher_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Attach the training-only multi-teacher ear distiller.

        `teacher_targets` holds the teacher feature tensors, keyed by the
        distiller's parameter names: `w2v_layer16_50hz`, `canary_top_12_5hz`
        and `qwen_km_top`. They are remembered here and used as the default
        for every subsequent `train_step` call, so the distillation loss is
        non-zero whenever teacher features are available.
        """
        self.distiller = distiller
        self.distiller_weight = float(weight)
        self.teacher_targets = dict(teacher_targets or {})

    def train_step(
        self,
        loss: torch.Tensor | Callable[[], torch.Tensor],
        teacher_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> float:
        """
        v10.7 P2-B: accepts EITHER an already-computed loss tensor OR a
        callable that performs the forward pass.

        When a callable is given, the forward pass runs under the profile's
        autocast context and the backward pass runs OUTSIDE it. When a pre-
        computed tensor is given (the v10.6 calling convention), it is used
        as-is and the caller remains responsible for the forward pass.

        `teacher_targets` carries the multi-teacher feature tensors, keyed by
        the distiller's parameter names: `w2v_layer16_50hz`,
        `canary_top_12_5hz` and `qwen_km_top`. When they are supplied AND a
        distiller is attached, the distillation loss is computed against the
        student's layer-8 hidden state and bus output and is aggregated with
        the main loss.
        """
        if callable(loss):
            with self.autocast_context():
                loss_tensor = loss()
                # v10.7 P2-B: the multi-teacher ear distillation loss is
                # aggregated with the main loss inside the SAME autocast
                # forward pass, so the three teacher streams actually
                # contribute to training (v10.6 never aggregated them).
                if self.distiller is not None:
                    ear = getattr(self.model, "ear", None)
                    l8 = getattr(ear, "last_l8_hidden", None) if ear is not None else None
                    # v10.7 P2-B: the bus output is published by the ear on
                    # the thread-local state, so it is available here.
                    bus_out = getattr(ear, "last_bus_out", None) if ear is not None else None
                    if bus_out is None:
                        bus_out = getattr(self.model, "bus_out", None)
                    if l8 is not None and bus_out is not None:
                        kwargs = dict(teacher_targets or self.teacher_targets or {})
                        distil = self.distiller(l8, bus_out, **kwargs)
                        loss_tensor = loss_tensor + self.distiller_weight * distil
        else:
            loss_tensor = loss
        scaled_loss = loss_tensor / float(self.profile.grad_accum_steps)
        # Backward runs OUTSIDE the autocast block, by construction.
        scaled_loss.backward()
        self.step_count += 1
        if self.step_count % self.profile.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return float(loss_tensor.detach().item())
