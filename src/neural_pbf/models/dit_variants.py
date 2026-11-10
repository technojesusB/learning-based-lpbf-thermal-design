"""DiT variants that live outside ``models/generative/fm``, whose files are part of the UQ pipeline's S4 hash.

``VelocityDiTAPE`` is DiT v1 (journal section 5.1): 4³ patches and an absolute position embedding learned as an
``nn.Parameter``. The original v1 code was never committed (the first DiT commit, 5744e48, already has the fixed
sinusoidal buffer), so this is rebuilt from the journal's spec. Initialization follows ViT/DiT practice (truncated
normal, std 0.02), and the embedding is added unscaled; the 0.02 scale in ``VelocityDiT.forward`` belongs to v2's
fixed sinusoidal embedding.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from neural_pbf.models.generative.fm.dit import VelocityDiT, sinusoidal_time_embedding


class VelocityDiTAPE(VelocityDiT):
    """VelocityDiT with a learned absolute position embedding instead of the fixed sinusoidal buffer."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        shape = self._buffers.pop("pos_embed").shape  # type: ignore[union-attr]  # (1, N_tokens, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty(shape))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def embed_tokens(self, x: Tensor) -> Tensor:
        """Patch tokens plus the learned position embedding: (B, N_tokens, embed_dim)."""
        return self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        B = x.shape[0]
        grid = [n // self.patch_size for n in x.shape[2:]]
        tokens = self.embed_tokens(x)
        c = self.time_embed(sinusoidal_time_embedding(t, self._time_emb_dim)) + self.cond_proj(cond)
        for block in self.blocks:
            tokens = block(tokens, c)
        tokens = tokens.transpose(1, 2).reshape(B, self.embed_dim, *grid)
        return self.final_layer(self.unpatchify(tokens))
