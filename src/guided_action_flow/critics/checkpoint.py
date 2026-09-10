from __future__ import annotations


def load_action_chunk_critic(checkpoint_path, *, device="cpu"):
    """Load a trained critic, dispatching on the stored ``critic_arch``.

    Returns ``(critic, checkpoint)``. Supports the multi-modal v2 critic and the
    legacy concat critic; older checkpoints without ``critic_arch`` are treated
    as concat for backward compatibility.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = checkpoint.get("critic_arch", "concat")

    if arch == "transformer":
        from guided_action_flow.critics.transformer_critic import (
            TransformerCritic,
            TransformerCriticConfig,
        )

        config = TransformerCriticConfig(**checkpoint["critic_config"])
        critic = TransformerCritic(config)
    elif arch == "multimodal":
        from guided_action_flow.critics.multimodal_critic import (
            MultiModalCritic,
            MultiModalCriticConfig,
        )

        config = MultiModalCriticConfig(**checkpoint["critic_config"])
        critic = MultiModalCritic(config)
    else:
        from guided_action_flow.critics.action_chunk_critic import (
            ActionChunkCritic,
            ActionChunkCriticConfig,
        )

        config = ActionChunkCriticConfig(**checkpoint["critic_config"])
        critic = ActionChunkCritic(config)

    critic.load_state_dict(checkpoint["model_state_dict"])
    critic.module.to(device)
    critic.module.eval()
    return critic, checkpoint
