# Video-Free-LoRA-Hyvideo1.5_I2V

Experimental finetuning recipe for HunyuanVideo-1.5 i2v that uses image-only data and an identity loss to reduce motion-induced identity drift during character-domain adaptation.

Motivation: when the character distribution shifts from general to a narrow class (for example, Pokémon), adding motion to a static reference image often causes identity drift. In the same time, a potential issue of enforcing identity consistency too hard is motion collapse: the model learns the shortcut of keeping frames near-static to preserve identity suppresing motion, background changes, and pose variation. This approach addresses both issues with:
- Teacher → student distillation to preserve the base model’s motion / dynamics priors.
- Identity regularization using DINOv3 embeddings with a domain-tuned projection head, explicitly penalizing identity drift while allowing style and pose variation. 

The projection head is trained with supervised contrastive loss so same-character images stay close even across large style shifts, while different characters (including hard negatives) are pushed apart.

### Example: Pokémon domain adaption

I implemented this approach for Pokémon character-domain adaptation and the results are decent.  
- For sample side-by-side comparison against the baseline HunyuanVideo-1.5 i2v model and sample generations and, see: [Video-Free-Pokemon-LoRA-Hyvideo1.5_I2V](https://huggingface.co/Kev0208/Video-Free-Pokemon-LoRA-Hyvideo1.5_I2V)
- For the Pokémon identity embedding (DINOv3 projection head checkpoint), see: [DINOv3-PokeCon-Head](https://huggingface.co/Kev0208/DINOv3-PokeCon-Head)

### Beyond character: concept-alignment domain adaptation 

Although I’ve only tested on Pokémon so far, this approach could potentially have broader application than “large-scale character adaptation.”

- Teacher → student distillation is a prior-preservation mechanism: it helps the LoRA adapt without forgetting behaviors you want to keep (motion priors, camera dynamics, background dynamics, general temporal coherence, etc.).
- The identity loss can be viewed more generally as a concept alignment loss: it encourages the generated output to stay aligned with a reference image that embodies the concept you care about. 

In other words: as long as you can represent the target concept in an image and define a robust embedding that captures that concept (could be a DINOv3 projection head), this approach could work.

## Repo structure

```
.
├── build_prompt.py
├── checkpoints-download.md
├── export_lora_from_dcp.py
├── generate.py
├── hyvideo/                    # Upstream HunyuanVideo pipeline, models, schedulers, utils
├── identity_loss/
│   └── train_projection_head.py
├── train.py
├── requirements.txt
├── LICENSE
└── NOTICE
```

## Finetuning approach (train.py)

This section is a code-aligned overview of what `train.py` currently does. If any description elsewhere differs, the code behavior here is the source of truth.

### 1) Dataset format (WebDataset)

`WDS_ROOT` layout:

```
WDS_ROOT/
├── data/
│   ├── train/
│   │   ├── train-000000.tar
│   │   └── ...
│   └── val/
│       └── val-000000.tar
└── manifest/
    ├── train_manifest.csv
    └── val_manifest.csv
```

Each shard sample includes:
- `<image_id>.<ext>` (jpg/jpeg/png/webp) raw bytes
- `<image_id>.json` metadata JSON

Metadata JSON must contain a character identifier field (config: `--metadata_key`), for example:

```json
{ "pokemon_species": "mega rayquaza" }
```

Manifest CSV schema:
- Column `<metadata_key>` contains the canonical character string (lowercase, includes form).
- Column `count` is the number of images for that character.

Important: images should be preprocessed to a consistent size/resolution within the dataset. The train loader stacks images into a batch and caches secondary reference images by tensor; mismatched sizes will break batching or secondary reference usage.

Note: `train.py` uses manifest counts for inverse-frequency rejection sampling to reduce imbalance. It does not explicitly sample identities uniformly.

### 2) Dataloader requirements (WebDataset + identity-aware sampling)

2.1 Train loader sampling policy (code behavior)
- Train shards are read with `wds.ResampledShards`.
- Each sample is accepted or rejected using inverse-frequency rejection:
  `p = min(1, min_count / count(character))`.
- Optional secondary reference sampling uses a per-worker LRU cache:
  - `--p_secondary` controls the probability of attaching a previous sample of the same character.
  - `--secondary_cache_max_size` bounds the cache size.

2.2 Prompt construction
- There is no prompt stored in the dataset.
- `train.py` calls `build_prompt.build_prompt(character)`.
-  `build_prompt` is only a minimal, rough template and you should implement your own prompt builder.

2.3 Validation loader
- `val` uses `wds.SimpleShardList` and the same manifest-based rejection sampling.
- No secondary references or augmentation in validation.
- Validation runs on rank 0 only.

2.4 Optional light augmentation
- `--enable_augmentation` enables a light long-side crop jitter (`crop_long_side_only` in `train.py`).
- `--augment_r` and `--augment_j` control crop ratio and jitter.

### 3) Teacher vs student definition (distillation setup)

- Teacher: a frozen base transformer loaded from the pipeline (no LoRA), `eval()` + `no_grad()`.
- Student: pipeline transformer with LoRA injected; only LoRA parameters are trainable.

The teacher and student are separate instances to guarantee correct separation of weights.

### 4) Training objective (loss)

Total loss:

```
L_total = L_distill + lambda_id * L_id
```

4.1 Distillation loss (always on)
- Teacher rollout caches `(x_t, t, y_T)` at a small set of timesteps.
- Student predicts `y_S` on the same cached inputs.
- MSE is computed with uniform weighting (no SNR weighting in current code).

4.2 Identity loss (low-noise timesteps only)
- DINOv3 backbone + projection head is loaded from local paths.
- A low-noise timestep `t_deep` is randomly chosen near the end of the rollout, sampled from
  `--id_timestep_frac_low` to `--id_timestep_frac_high` (defaults: 0.60–0.85 of the rollout steps).
- `x0` is estimated using one-step Euler in code:
  `x0 = x_t - s * pred` with `s = t / num_train_timesteps`.
- `x0` is decoded by the video VAE, a few frames are sampled, and DINO embeddings are compared with an anchor image.
- Cosine loss:

```
L_id = mean_k (1 - cos(g(frame_k), g(anchor)))
```

Caution: Be careful setting id_timestep_frac_* too close to 1.0 (i.e., too near the clean end of the rollout). The identity loss is a global constraint; if you only apply it at the very last steps, it can over-emphasize final texture/detail cleanup near `\hat{x}_0` rather than enforcing identity consistency under the harder motion/pose changes earlier in denoising.

Anchor policy:
- If a secondary reference was sampled, it is used as the anchor.
- Otherwise, the primary reference image is used.

### 5) On-the-fly trajectory distillation (deep rollout + harvesting)

Per batch:
1. Sample noise latent `x_T`.
2. Run the teacher sampler from `T` down to `t_deep`.
3. Cache a few (latents, timestep, teacher_pred) pairs for distillation.

Harvesting behavior (code defaults):
- `--harvest_scheme bracket` uses bracketed indices near 50% and 25% of the schedule with jitter.
- `--harvest_count` limits the number of cached timesteps.
- `--num_inference_steps` controls the rollout schedule length.

### 6) Frame sampling for identity loss

Decoding all frames is too expensive, so a few frames are sampled:
- Bucketed random sampling biased toward early and mid frames.
- Controlled by `--id_num_frames` (default 5 in CLI).

### 7) Lambda and warmup schedule for identity loss

- `--lambda_id` scales the identity loss (default 0.6 in CLI).
- Optional warmup via `--lambda_id_schedule {constant, linear, cosine}` and
  `--lambda_id_warmup_steps`.
- Identity loss can be computed every N steps via `--id_every_steps`.

### 8) Optimizer / LR scheduler

- Optimizer: Muon by default (`--use_muon true`), or AdamW if disabled.
- Scheduler: cosine warmup (`diffusers.get_scheduler("cosine")`).
- Only LoRA parameters are trainable.

### 9) Validation and logging (rank 0 only)

Metrics:
1. Teacher deviation: distill MSE on val batches.
2. DINO cosine mean/std on sampled frames.
3. Motion metric: mean absolute frame-to-frame difference on decoded preview.

Validation is gated by `--enable_validation` and runs on rank 0 only.

### 10) Checkpointing and resume

- Multi-rank FSDP: DCP checkpoint is saved under `output_dir/checkpoint-<step>/dcp`.
- Single-rank (non-FSDP): only optimizer/scheduler state + step is saved as `training_state.pt`.
- LoRA adapters are not saved by `train.py` directly. Use `export_lora_from_dcp.py`
  to extract LoRA adapters from a DCP checkpoint.
- Resume via `--resume_from_checkpoint`. If a `lora/` directory exists inside the checkpoint,
  it is loaded automatically for non-FSDP runs.

### 11) LoRA target modules (defaults)

Default `lora_target_modules`:

```
MMDoubleStreamBlock:
  img_attn_q, img_attn_k, img_attn_v, img_attn_proj
  txt_attn_q, txt_attn_k, txt_attn_v, txt_attn_proj
MMSingleStreamBlock:
  linear1_q, linear1_k, linear1_v, linear2.fc, linear1_mlp
Modulation / gating:
  img_mod.linear, txt_mod.linear, modulation.linear, adaLN_modulation.1
```

### 12) Distributed training

- Uses the repo's default distributed setup with sequence parallel (`--sp_size`) and FSDP2.
- WebDataset uses `split_by_node` and `split_by_worker`.
- Validation is rank 0 only.

## Key training arguments (train.py)

| Arguments | Purpose | Notes |
| --- | --- | --- |
| `--pipeline_dir`, `--transformer_version` | Model pipeline to load | Includes text/vision encoders; default version is `480p_i2v`. |
| `--wds_root`, `--metadata_key` | Dataset location and identity key | Must match JSON metadata and manifest column. |
| `--batch_size`, `--num_workers`, `--wds_shuffle_buf` | Data throughput | Train uses resampled shards + shuffle. |
| `--p_secondary`, `--secondary_cache_max_size` | Secondary ref sampling | Enables same-identity anchor via per-worker cache. |
| `--enable_augmentation`, `--augment_r`, `--augment_j` | Light crop jitter | Optional; applied to reference images. |
| `--train_target_resolution`, `--train_video_length` | Training size and length | `train_video_length` must be `4n+1`. |
| `--num_inference_steps` | Teacher rollout length | Affects distillation and identity timestep bands. |
| `--harvest_count`, `--harvest_scheme`, `--harvest_jitter_frac` | Distill target harvesting | `bracket` or `random` scheme. |
| `--id_timestep_frac_low`, `--id_timestep_frac_high` | Identity timestep band | Fraction of the rollout schedule. |
| `--id_num_frames`, `--id_every_steps` | Identity frame sampling and frequency | Controls compute cost. |
| `--lambda_id`, `--lambda_id_schedule`, `--lambda_id_warmup_steps` | Identity loss weight | Warmup can be linear or cosine. |
| `--id_decode_downsample_mode`, `--id_decode_scale` | Identity decode cost control | Downsample x0 before VAE decode. |
| `--use_lora`, `--lora_r`, `--lora_alpha`, `--lora_target_modules`, `--pretrained_lora_path` | LoRA config | Only LoRA params are trainable. |
| `--learning_rate`, `--use_muon`, `--weight_decay`, `--warmup_steps`, `--max_train_steps`, `--gradient_accumulation_steps` | Optimization controls | Scheduler is cosine warmup by default. |
| `--enable_validation`, `--val_every_steps`, `--val_num_samples`, `--val_video_length` | Validation cadence | Runs on rank 0 only. |
| `--enable_fsdp`, `--enable_gradient_checkpointing`, `--sp_size`, `--resume_from_checkpoint` | Distributed and resume | Multi-rank saves DCP checkpoints. |

## How to run

### 0) Download base checkpoints

Read `checkpoints-download.md` for the repo-specific download steps, and use:
https://huggingface.co/tencent/HunyuanVideo-1.5 for the base model.

### 1) Prepare the DINOv3 identity projection head

Train a projection head with supervised contrastive loss using `identity_loss/train_projection_head.py`.
It expects a WebDataset where the JSON metadata contains an integer identity ID (`--id-key`).

Required WebDataset format (from the script docstring):

```
WDS_ROOT/
├── data/
│   ├── train/
│   │   ├── train-000000.tar
│   │   └── ...
│   └── val/
│       └── val-000000.tar
└── manifest/
    ├── train_manifest.csv
    └── val_manifest.csv
```

Each sample inside a shard:
- `<sample_id>.<ext>` (jpg/png/webp)
- `<sample_id>.json` metadata containing `{ "<id_key>": <int> }`

Manifest CSVs must include a column named `<id_key>` listing identity IDs. Use the same `--id-key` for JSON and CSV.
Ensure each validation identity has at least `--pk-instances` samples for the P*K sampler.
Shard patterns are passed via `--train-shards` and `--val-shards` and are resolved under `WDS_ROOT/data/<split>/`.

Example:

```bash
accelerate launch identity_loss/train_projection_head.py \
  --model-dir /path/to/dinov3 \
  --wds-root /path/to/identity_wds \
  --output-dir /path/to/identity_head \
  --id-key identity_id \
  --train-shards "train-*.tar" \
  --val-shards "val-*.tar" \
  --proj-dims 512 256 \
  --pk-identities 32 \
  --pk-instances 2 \
  --total-steps 4000 \
  --val-every 250 \
  --save-every 250 \
  --model-dtype bf16
```

This produces a projection head checkpoint used by `train.py` via `--dino_head_path`.

### 2) Finetune LoRA with distillation + identity loss

Example:

```bash
torchrun --nproc_per_node=4 train.py \
  --pipeline_dir /path/to/hunyuanvideo_pipeline \
  --wds_root /path/to/wds_root \
  --metadata_key pokemon_species \
  --dino_model_dir /path/to/dinov3 \
  --dino_head_path /path/to/identity_head.pt \
  --output_dir ./outputs \
  --enable_validation true
```

Notes:
- `build_prompt.py` must be populated with a prompt bank; otherwise prompts are empty.
- `train_video_length` must be `4n+1` for the VAE (default 17).
- `--p_secondary` and `--secondary_cache_max_size` control optional same-identity anchors.

### 3) Run inference

Base model (t2v):

```bash
torchrun --nproc_per_node=1 generate.py \
  --prompt "your_prompt" \
  --resolution 480p \
  --model_path /path/to/hunyuanvideo_pipeline \
  --output_path ./outputs/base.mp4
```

LoRA adapter:

```bash
torchrun --nproc_per_node=1 generate.py \
  --prompt "your_prompt" \
  --image_path /path/to/ref.png \
  --resolution 480p \
  --model_path /path/to/hunyuanvideo_pipeline \
  --lora_path /path/to/lora_adapter \
  --lora_r 16 --lora_alpha 16 --lora_scale 0.5 \
  --output_path ./outputs/lora.mp4
```

DCP checkpoint (full transformer):

```bash
torchrun --nproc_per_node=4 generate.py \
  --prompt "your prompt" \
  --image_path /path/to/ref.png \
  --resolution 480p \
  --model_path /path/to/hunyuanvideo_pipeline \
  --checkpoint_path /path/to/checkpoint-1000 \
  --use_lora true --lora_r 16 --lora_alpha 16 --lora_scale 0.7 \
  --output_path ./outputs/dcp.mp4
```

If your DCP checkpoint includes LoRA weights, you must enable LoRA injection and pass the matching LoRA config
(`--use_lora`, `--lora_r`, `--lora_alpha`, and optionally `--lora_target_modules`). Use `--lora_scale` to control strength.

### 4) Extract PEFT adapters from a DCP checkpoint

If training used FSDP and saved DCP checkpoints, extract the LoRA adapter with:

```bash
torchrun --nproc_per_node=1 export_lora_from_dcp.py \
  --model_path /path/to/hunyuanvideo_pipeline \
  --checkpoint_path /path/to/checkpoint-1000 \
  --output_dir ./outputs/lora_adapter \
  --transformer_version 480p_i2v \
  --lora_r 16 --lora_alpha 16
```

## Notes

- `hyvideo/` is vendor code from Tencent HunyuanVideo and includes pipelines, models, schedulers, and utilities.
- Read `LICENSE` and `NOTICE` for usage terms.
