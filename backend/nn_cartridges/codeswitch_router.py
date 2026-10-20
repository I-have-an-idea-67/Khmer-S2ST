from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from backend.contracts.schemas import (
    PROD_240M_CONFIG,
    SEMANTIC_DIM,
    ModelArchitectureConfig,
)


class AcousticPointerCodeSwitchBridge(nn.Module):
    def __init__(self, config: ModelArchitectureConfig = PROD_240M_CONFIG) -> None:
        super().__init__()
        d = config.pointer_d_model
        self.in_q = nn.Linear(SEMANTIC_DIM, d)
        self.in_kv = nn.Linear(SEMANTIC_DIM, d)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(d, 8, batch_first=True),
                        "ffn": nn.Sequential(
                            nn.Linear(d, config.pointer_ffn_dim),
                            nn.GELU(),
                            nn.Linear(config.pointer_ffn_dim, d),
                        ),
                    }
                )
                for _ in range(config.pointer_layers)
            ]
        )
        self.out_proj = nn.Linear(d, SEMANTIC_DIM)
        self.gate_head = nn.Linear(SEMANTIC_DIM, 1)

    def forward(
        self,
        semantic_bus: torch.Tensor,
        source_acoustic_features: torch.Tensor,
        force_gate: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.in_q(semantic_bus)
        kv = self.in_kv(source_acoustic_features)
        for blk in self.blocks:
            attn_out, _ = blk["attn"](q, kv, kv, need_weights=False)
            q = q + attn_out
            q = q + blk["ffn"](q)
        pointer_context = self.out_proj(q)
        gate = torch.sigmoid(self.gate_head(semantic_bus))
        if force_gate is not None:
            gate = torch.full_like(gate, float(force_gate))
        fused = (1.0 - gate) * semantic_bus + gate * pointer_context
        return fused, gate
