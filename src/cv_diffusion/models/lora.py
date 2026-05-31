"""LoRA configuration and helpers built on top of the PEFT library.

We use parameter-efficient fine-tuning (LoRA) instead of full fine-tuning of
the UNet for three reasons:

1. Compute — LoRA adds ~1-5 M trainable parameters vs ~860 M for the SD 1.5
   UNet, so training fits on a single 12-16 GB consumer GPU.
2. Storage — adapters are tens of MB rather than several GB.
3. Mixability — multiple disaster categories can be trained as separate
   adapters and blended at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class LoRAConfig:
    """Hyperparameters for the diffusers/PEFT LoRA recipe.

    Defaults follow the official ``train_text_to_image_lora.py`` script in
    diffusers and the values typically reported for small-domain fine-tunes.
    """

    rank: int = 8
    alpha: int = 8  # alpha == rank is the diffusers/PEFT default for LoRA
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    )
    bias: str = "none"
    init_lora_weights: bool | str = True
    train_text_encoder: bool = False
    text_encoder_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")
    text_encoder_lr: float = 5e-6
    unet_lr: float = 1e-4
    extra: dict = field(default_factory=dict)


def apply_lora_to_unet(unet, config: LoRAConfig):
    """Attach LoRA adapters to a UNet using PEFT.

    Returns the UNet with LoRA enabled in-place plus the list of trainable
    parameters — useful for plugging into an optimizer.
    """

    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias=config.bias,
        init_lora_weights=config.init_lora_weights,
    )
    unet = get_peft_model(unet, peft_cfg)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    return unet, trainable


def apply_lora_to_text_encoder(text_encoder, config: LoRAConfig):
    """Attach LoRA adapters to the CLIP text encoder (optional).

    Enabled only when ``config.train_text_encoder`` is True. Significantly
    increases trainable parameter count but can help when the new category
    names are far from natural language seen during SD pre-training.
    """

    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.text_encoder_target_modules),
        bias=config.bias,
    )
    text_encoder = get_peft_model(text_encoder, peft_cfg)
    trainable = [p for p in text_encoder.parameters() if p.requires_grad]
    return text_encoder, trainable


def count_trainable_parameters(parameters: Iterable) -> int:
    return sum(p.numel() for p in parameters)
