from __future__ import annotations

"""Transformer action-chunk critic (v3), per Haichen's suggestion.

Motivation. The v2 MultiModalCritic pools the policy's visual tokens into a
single vector before scoring; that pooling discards spatial layout, which we
identified as the likely reason QGF's *gradient* fails on viewpoint-type
perturbations ("排序对、梯度错"). This critic replaces the concat-then-MLP
fusion with self-attention over modality tokens, and optionally attends over
*un-pooled* visual tokens so the critic can see "what is where".

Two modes, chosen by ``visual_tokens``:
- ``visual_tokens=0`` (Stage A): use the existing pooled visual feature as ONE
  token. Runs on already-collected data (no re-collection). Validates the
  attention plumbing and the QGF gradient path.
- ``visual_tokens>0`` (Stage B): expect ``obs_features`` to carry N un-pooled
  visual tokens ([B, N, visual_dim]); attend over them. This is the real test
  of "finer visual handling helps generalization".

QGF compatibility: keeps the ``(obs_features, action_chunk, proprio,
task_features)`` call signature, and the action chunk flows through the network
differentiably, so ``guidance/qgf.py`` and ``policies/pi05_qgf.py`` are
untouched. Mapping (same as v2): obs_features->visual, proprio->state,
task_features->language.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerCriticConfig:
    action_dim: int
    action_horizon: int
    state_dim: int = 0
    visual_dim: int = 0          # per-token (Stage B) or pooled (Stage A) visual dim
    visual_tokens: int = 0       # 0 = pooled single token; >0 = N spatial tokens
    language_dim: int = 0
    embed_dim: int = 256         # token width
    n_heads: int = 4
    depth: int = 3               # transformer layers
    action_chunk_tokens: int = 8 # action horizon is chunked into this many tokens
    dropout: float = 0.1
    input_layernorm: bool = True


def _proj(in_dim: int, embed_dim: int, ln: bool):
    import torch.nn as nn

    layers = []
    if ln:
        layers.append(nn.LayerNorm(in_dim))
    layers.append(nn.Linear(in_dim, embed_dim))
    return nn.Sequential(*layers)


class TransformerCritic:
    """Tokenize each modality, self-attend, read out scalar Q from a CLS token."""

    def __init__(self, config: TransformerCriticConfig):
        import torch
        import torch.nn as nn

        self.config = config
        d = config.embed_dim
        modules: dict[str, nn.Module] = {}

        # per-modality projections into the shared token space
        if config.state_dim > 0:
            modules["state_proj"] = _proj(config.state_dim, d, config.input_layernorm)
        if config.visual_dim > 0:
            modules["visual_proj"] = _proj(config.visual_dim, d, config.input_layernorm)
        if config.language_dim > 0:
            modules["language_proj"] = _proj(config.language_dim, d, config.input_layernorm)

        # action chunk -> action_chunk_tokens tokens. Split the horizon into
        # contiguous groups, each group flattened and projected to one token.
        self._act_group = max(1, config.action_horizon // config.action_chunk_tokens)
        self._n_act_tok = (config.action_horizon + self._act_group - 1) // self._act_group
        modules["action_proj"] = _proj(
            config.action_dim * self._act_group, d, config.input_layernorm
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_heads, dim_feedforward=4 * d,
            dropout=config.dropout, batch_first=True, activation="gelu",
        )
        modules["encoder"] = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        modules["head"] = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        # learnable CLS token + modality type embeddings, kept INSIDE the module
        # (as a tiny nn.Module) so critic.module.to()/state_dict()/parameters()
        # cover them — no special plumbing needed in training/checkpoint code.
        n_types = 5  # cls, state, visual, language, action
        tokens_mod = nn.Module()
        tokens_mod.cls = nn.Parameter(torch.zeros(1, 1, d))
        tokens_mod.type_emb = nn.Parameter(torch.zeros(n_types, d))
        nn.init.normal_(tokens_mod.cls, std=0.02)
        nn.init.normal_(tokens_mod.type_emb, std=0.02)
        modules["tokens"] = tokens_mod

        self.module = nn.ModuleDict(modules)

    # nn.Module-like plumbing so training/checkpoint code works unchanged
    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, obs_features=None, action_chunk=None, proprio=None, task_features=None):
        """QGF-compatible. obs_features=visual, proprio=state, task_features=language."""
        import torch

        if action_chunk is None:
            raise ValueError("action_chunk is required.")
        B = action_chunk.shape[0]
        cls = self.module["tokens"].cls
        type_emb = self.module["tokens"].type_emb
        tokens = [cls.expand(B, -1, -1) + type_emb[0]]

        if "state_proj" in self.module:
            if proprio is None:
                raise ValueError("critic expects proprio (state).")
            t = self.module["state_proj"](proprio.reshape(B, -1)).unsqueeze(1)
            tokens.append(t + type_emb[1])

        if "visual_proj" in self.module and obs_features is not None:
            if self.config.visual_tokens > 0:
                v = obs_features.reshape(B, self.config.visual_tokens, self.config.visual_dim)
            else:
                v = obs_features.reshape(B, 1, -1)
            t = self.module["visual_proj"](v) + type_emb[2]
            tokens.append(t)

        if "language_proj" in self.module:
            if task_features is None:
                raise ValueError("critic expects language (task_features).")
            t = self.module["language_proj"](task_features.reshape(B, -1)).unsqueeze(1)
            tokens.append(t + type_emb[3])

        # action chunk -> tokens (this path carries the QGF gradient)
        a = action_chunk  # [B, horizon, action_dim]
        H = a.shape[1]
        g = self._act_group
        pad = (-H) % g
        if pad:
            a = torch.cat([a, a.new_zeros(B, pad, a.shape[-1])], dim=1)
        a = a.reshape(B, a.shape[1] // g, g * a.shape[-1])
        at = self.module["action_proj"](a) + type_emb[4]
        tokens.append(at)

        seq = torch.cat(tokens, dim=1)
        enc = self.module["encoder"](seq)
        q = self.module["head"](enc[:, 0])  # CLS readout
        return q.squeeze(-1)


def _as_param(t):
    import torch.nn as nn
    return t if isinstance(t, nn.Parameter) else nn.Parameter(t)
