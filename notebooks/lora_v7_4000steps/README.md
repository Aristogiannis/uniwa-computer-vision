# LoRA v7 — 4000-step fine-tune on EuroSAT proxy

Frozen artifact from Kaggle kernel run **v7** of
`aristogiannis/uniwa-cv-diffusion-proxy-pipeline`.

## Recipe

- **Base model**: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- **Training resolution**: 256×256 (upsampled from EuroSAT's 64×64)
- **Training data**: 2400 images — 800 each from EuroSAT `forest`,
  `residential_buildings`, `river`
- **LoRA**: rank 8, alpha 8, target modules
  `to_q, to_k, to_v, to_out.0` (UNet cross-attention only)
- **Optimiser**: AdamW(β₁=0.9, β₂=0.999, wd=1e-2), lr=1e-4 with cosine
  schedule
- **Steps**: 4000 (effective batch 4 via 1×4 grad-accum) ≈ 6.7 epochs
- **Mixed precision**: fp16, gradient checkpointing on

## Loss curve

| step | loss   | lr      |
|-----:|-------:|--------:|
|   25 | 0.0729 | 1.0e-04 |
|  500 | 0.0050 | 9.6e-05 |
| 1000 | 0.0012 | 8.5e-05 |
| 1500 | 0.3496 | 6.9e-05 |
| 2000 | 0.0602 | 5.0e-05 |
| 2500 | 0.0414 | 3.1e-05 |
| 3000 | 0.8892 | 1.5e-05 |
| 3500 | 0.0119 | 3.8e-06 |
| 4000 | 0.0039 | 0       |

## Files

- `pytorch_lora_weights.safetensors` — diffusers-compatible adapter
  weights, 6.4 MB
- `config.json` — PEFT LoraConfig
- `training_log.jsonl` — per-25-step loss and lr

## Loading

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe.load_lora_weights("notebooks/lora_v7_4000steps")
```
