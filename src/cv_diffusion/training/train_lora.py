"""LoRA fine-tuning loop for Stable Diffusion.

This is a compact, readable adaptation of the official
``examples/text_to_image/train_text_to_image_lora.py`` recipe from the
diffusers repository, kept intentionally simple for university-project use:

* only the UNet attention projections receive LoRA adapters by default
  (text-encoder LoRA is opt-in),
* mixed-precision is delegated to ``accelerate`` for portability across
  CPU/CUDA/MPS,
* the script writes adapter weights, training config and the loss curve
  to ``output_dir`` so the run is fully reproducible from disk.

Compared to the upstream script we drop:
* SNR-weighted loss, EMA, prior preservation — none required for the
  evaluation protocol in the project brief,
* ``--push_to_hub`` — not needed for an academic project.

If you need full feature parity (CLI flags, push-to-hub, etc.) use the
upstream script directly; this one is here to keep the training logic
visible and editable for the report.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from cv_diffusion.models.lora import LoRAConfig, count_trainable_parameters
from cv_diffusion.utils.logging import get_logger
from cv_diffusion.utils.seed import seed_everything

logger = get_logger(__name__)


@dataclass
class TrainLoRAConfig:
    """All knobs needed to launch a LoRA fine-tune.

    Values default to the recipe recommended for SD 1.5 + small disaster
    dataset on a single 12–16 GB consumer GPU.
    """

    pretrained_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: Optional[str] = None
    output_dir: str = "outputs/lora"
    train_data_dir: Optional[str] = None
    train_manifest_csv: Optional[str] = None
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    train_batch_size: int = 1
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = 4000
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    seed: int = 42
    log_every: int = 25
    checkpointing_steps: int = 500
    validation_prompt: Optional[str] = (
        "a satellite image of severe flooding, brown water covering buildings, "
        "post-disaster remote sensing"
    )
    num_validation_images: int = 4
    lora: LoRAConfig = field(default_factory=LoRAConfig)


def _build_dataloader(config: TrainLoRAConfig, tokenizer):
    """Build the image+caption dataloader using our DisasterImageDataset."""

    import torch
    from torchvision import transforms

    from cv_diffusion.preprocessing.dataset import DisasterImageDataset

    image_transforms = transforms.Compose(
        [
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
            transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1] for the VAE
        ]
    )

    dataset = DisasterImageDataset(
        root=config.train_data_dir,
        manifest_csv=config.train_manifest_csv,
        transform=image_transforms,
    )

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        captions = [b["caption"] for b in batch]
        tokenized = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids,
            "captions": captions,
            "categories": [b["category"] for b in batch],
        }

    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.train_batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    ), dataset


def train_lora(
    config: TrainLoRAConfig,
    *,
    progress_callback: Optional[Callable[[int, dict], None]] = None,
) -> Path:
    """Run a LoRA fine-tune. Returns the output directory containing weights."""

    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig as PeftLoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))

    seed_everything(config.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
        log_with="tensorboard",
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("cv-diffusion-lora")

    logger.info("Loading components from %s", config.pretrained_model_id)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet_lora_cfg = PeftLoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        init_lora_weights=config.lora.init_lora_weights,
        target_modules=list(config.lora.target_modules),
    )
    unet.add_adapter(unet_lora_cfg)

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    logger.info("Trainable LoRA parameters: %d", count_trainable_parameters(trainable_params))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    train_dataloader, dataset = _build_dataloader(config, tokenizer)
    logger.info("Dataset size: %d images", len(dataset))

    num_update_steps_per_epoch = max(
        1, math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    loss_history: list[dict] = []
    progress_log_path = output_dir / "training_log.jsonl"

    for epoch in range(config.num_train_epochs):
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                step_loss = float(loss.detach().item())
                if global_step % config.log_every == 0 and accelerator.is_main_process:
                    logger.info(
                        "step=%d epoch=%d loss=%.4f lr=%.2e",
                        global_step, epoch, step_loss, lr_scheduler.get_last_lr()[0],
                    )
                    loss_history.append({"step": global_step, "loss": step_loss, "lr": lr_scheduler.get_last_lr()[0]})
                    with progress_log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(loss_history[-1]) + "\n")
                    accelerator.log({"train_loss": step_loss}, step=global_step)
                    if progress_callback is not None:
                        progress_callback(global_step, loss_history[-1])

                if global_step % config.checkpointing_steps == 0 and accelerator.is_main_process:
                    _save_lora_adapter(accelerator, unet, output_dir / f"checkpoint-{global_step}")

                if global_step >= config.max_train_steps:
                    break
        if global_step >= config.max_train_steps:
            break

    if accelerator.is_main_process:
        _save_lora_adapter(accelerator, unet, output_dir)
        accelerator.end_training()
        logger.info("LoRA training complete: %s", output_dir)

    return output_dir


def _save_lora_adapter(accelerator, unet, target_dir: Path) -> None:
    """Persist LoRA weights using the diffusers convention."""

    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict
    from diffusers import StableDiffusionPipeline

    target_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(unet)
    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(target_dir),
        unet_lora_layers=state_dict,
        safe_serialization=True,
    )
    logger.info("Saved LoRA weights -> %s", target_dir)
