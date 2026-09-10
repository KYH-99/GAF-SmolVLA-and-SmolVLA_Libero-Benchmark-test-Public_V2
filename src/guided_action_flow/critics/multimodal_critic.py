from __future__ import annotations

"""Multi-modal action-chunk critic (v2), designed per Haichen's guidance.

Haichen's brief (paraphrased): don't just improve one task — *design* the critic
so it generalizes. Give it richer inputs — **observation, language, and the
current video** — and explore how it *processes* those inputs (pooling vs
attention, tokens vs latents). Train it on a **task family** for generalization.

This critic reflects that directly:
- **observation** -> proprioceptive ``state`` (LIBERO pi05: 8-dim)
- **video**       -> a short temporal window of the policy's pooled image features
                     (``visual_frames`` recent frames; 1 = single current frame).
                     Encoded with a temporal conv, not a flattened bag.
- **language**    -> task-instruction features (hashed tokens or VLM text latent)
- **action**      -> the action chunk being scored, also temporally encoded
                     (this is the path QGF differentiates).

Each modality is encoded separately then fused (not concatenated raw) — the
"how it processes the inputs" part. Trained pooled over many tasks for
generalization; validate ranking with ``critics/validation.py`` (Phase 3).

QGF compatibility: kept the ``(obs_features, action_chunk, proprio,
task_features)`` call signature so ``guidance/qgf.py`` is untouched. Mapping:
**obs_features -> video/visual, proprio -> state, task_features -> language.**
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiModalCriticConfig:
    action_dim: int
    action_horizon: int
    state_dim: int = 0
    visual_dim: int = 0          # per-frame pooled visual feature dim
    visual_frames: int = 1       # video window: # recent frames (1 = single frame)
    language_dim: int = 0
    embed_dim: int = 256         # per-modality encoder output width
    hidden_dim: int = 512        # fusion head width
    depth: int = 2               # fusion head depth
    action_temporal: bool = True # temporal conv over the action horizon vs flatten
    input_layernorm: bool = True


def _mlp_encoder(input_dim: int, embed_dim: int, input_layernorm: bool):
    import torch.nn as nn

    layers: list[nn.Module] = []
    if input_layernorm:
        layers.append(nn.LayerNorm(input_dim))
    layers.append(nn.Linear(input_dim, embed_dim))
    layers.append(nn.SiLU())
    layers.append(nn.Linear(embed_dim, embed_dim))
    return nn.Sequential(*layers)


def _temporal_encoder(feature_dim: int, embed_dim: int):
    """1D-conv over a [B, T, feature_dim] sequence -> [B, embed_dim] (mean-pooled).

    Used for the action chunk (over its horizon) and for the video window (over
    recent frames) so the critic sees temporal structure, not a flattened bag.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(feature_dim, embed_dim, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
        nn.SiLU(),
    )


def _fusion_head(input_dim: int, hidden_dim: int, depth: int):
    import torch.nn as nn

    if depth < 1:
        raise ValueError("depth must be >= 1")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.SiLU())
        current = hidden_dim
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


class MultiModalCritic:
    """Per-modality encoders (state / video / language / action) then fusion."""

    def __init__(self, config: MultiModalCriticConfig):
        import torch.nn as nn

        self.config = config
        modules: dict[str, nn.Module] = {}
        self._temporal_keys = set()

        if config.state_dim > 0:
            modules["state_enc"] = _mlp_encoder(
                config.state_dim, config.embed_dim, config.input_layernorm
            )
        if config.visual_dim > 0:
            if config.visual_frames > 1:
                # video: temporal conv over the recent-frame window
                modules["visual_enc"] = _temporal_encoder(config.visual_dim, config.embed_dim)
                self._temporal_keys.add("visual_enc")
            else:
                modules["visual_enc"] = _mlp_encoder(
                    config.visual_dim, config.embed_dim, config.input_layernorm
                )
        if config.language_dim > 0:
            modules["language_enc"] = _mlp_encoder(
                config.language_dim, config.embed_dim, config.input_layernorm
            )

        if config.action_temporal:
            modules["action_enc"] = _temporal_encoder(config.action_dim, config.embed_dim)
            self._temporal_keys.add("action_enc")
        else:
            modules["action_enc"] = _mlp_encoder(
                config.action_dim * config.action_horizon,
                config.embed_dim,
                config.input_layernorm,
            )

        n_modalities = sum(
            1 for k in ("state_enc", "visual_enc", "language_enc", "action_enc") if k in modules
        )
        modules["head"] = _fusion_head(
            config.embed_dim * n_modalities, config.hidden_dim, config.depth
        )
        self.module = nn.ModuleDict(modules)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def _encode_temporal(self, key, seq):
        # seq: [B, T, feature_dim] -> conv wants [B, feature_dim, T] -> mean-pool over T
        x = seq.transpose(1, 2)
        x = self.module[key](x)
        return x.mean(dim=-1)

    def _encode_visual(self, obs_features, batch_size):
        if "visual_enc" not in self.module:
            return None
        if obs_features is None:
            raise ValueError("critic expects visual/video (obs_features) but got None.")
        if "visual_enc" in self._temporal_keys:
            # video window: [B, frames, visual_dim]
            seq = obs_features.reshape(batch_size, self.config.visual_frames, self.config.visual_dim)
            return self._encode_temporal("visual_enc", seq)
        return self.module["visual_enc"](obs_features.reshape(batch_size, -1))

    def _encode_action(self, action_chunk):
        if "action_enc" in self._temporal_keys:
            # action chunk: [B, horizon, action_dim]
            return self._encode_temporal("action_enc", action_chunk)
        return self.module["action_enc"](action_chunk.reshape(action_chunk.shape[0], -1))

    def __call__(self, obs_features=None, action_chunk=None, proprio=None, task_features=None):
        """QGF-compatible. obs_features=video/visual, proprio=state, task_features=language."""
        import torch

        if action_chunk is None:
            raise ValueError("action_chunk is required.")
        batch_size = action_chunk.shape[0]

        parts = []
        if "state_enc" in self.module:
            if proprio is None:
                raise ValueError("critic expects proprio (state) but got None.")
            parts.append(self.module["state_enc"](proprio.reshape(batch_size, -1)))
        visual = self._encode_visual(obs_features, batch_size)
        if visual is not None:
            parts.append(visual)
        if "language_enc" in self.module:
            if task_features is None:
                raise ValueError("critic expects language (task_features) but got None.")
            parts.append(self.module["language_enc"](task_features.reshape(batch_size, -1)))

        parts.append(self._encode_action(action_chunk))

        fused = torch.cat(parts, dim=-1)
        return self.module["head"](fused).squeeze(-1)
