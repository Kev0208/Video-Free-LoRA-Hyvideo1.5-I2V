# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""
HunyuanVideo-1.5 training script for image-to-video distillation with identity guidance.

This script trains a student transformer using a frozen teacher via short rollouts.
Key steps:
- Sample timesteps along the teacher trajectory using bracketed buckets with jitter.
- Harvest teacher predictions at those timesteps for a distillation MSE loss.
- Choose a deep timestep for identity guidance, estimate x0, decode with the 3D VAE,
  sample frames, and compute an identity loss with a frozen DINO encoder.

WebDataset layout:
  <wds_root>/
    data/{train,val}/{split}-*.tar
    manifest/{split}_manifest.csv

Each shard sample must include:
  - <id>.<ext> (jpg/jpeg/png/webp) image bytes
  - <id>.json metadata

The metadata JSON must include a character identifier field, provided via --metadata_key.
Character strings are converted into prompts via build_prompt.build_prompt.
The manifest CSV must contain the same column name and a "count" column:
  <metadata_key>,count
These counts are used for inverse-frequency accept/reject sampling to reduce
character imbalance.

Checkpoints:
- Multi-rank FSDP saves sharded weights using torch.distributed.checkpoint (DCP) under
  <output_dir>/checkpoint-<step>/dcp.
- Single-rank runs save the LoRA adapter (if enabled) and optimizer/scheduler state.

Example usage:
  torchrun --nproc_per_node=4 train.py \
    --pipeline_dir /your/path/to/hunyuanvideo_pipeline \
    --wds_root /your/path/to/wds_root \
    --metadata_key character_id \
    --dino_model_dir /your/path/to/dino_model \
    --dino_head_path /your/path/to/dino_head.pt \
    --output_dir ./outputs

Resume training:
  torchrun --nproc_per_node=4 train.py \
    --pipeline_dir /your/path/to/hunyuanvideo_pipeline \
    --wds_root /your/path/to/wds_root \
    --metadata_key character_id \
    --dino_model_dir /your/path/to/dino_model \
    --dino_head_path /your/path/to/dino_head.pt \
    --output_dir ./outputs \
    --resume_from_checkpoint /your/path/to/outputs/checkpoint-2500
"""

import argparse
import csv
import io
import json
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import einops
import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from diffusers.optimization import get_scheduler
from hyvideo.commons.parallel_states import get_parallel_state, initialize_parallel_state
from hyvideo.optim.muon import get_muon_optimizer
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer

from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

import build_prompt

# --- WebDataset required (no manual tarfile I/O) ---
try:
    import webdataset as wds
except Exception as e:
    raise RuntimeError(
        "webdataset is required for this training script. "
        "Please `pip install webdataset` in your environment."
    ) from e

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def save_video(video: torch.Tensor, path: str):
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c")
    imageio.mimwrite(path, vid.cpu().numpy(), fps=24)


def crop_long_side_only(
    img: Image.Image, r: float = 0.92, j: float = 0.04, rng: Optional[random.Random] = None
) -> Image.Image:
    """Lightweight augmentation: crop along long side with small jitter."""
    if rng is None:
        rng = random
    img = ImageOps.exif_transpose(img)
    W, H = img.size

    if W >= H:
        Cw = int(round(W * r))
        Cw = max(1, min(Cw, W))

        dx = rng.uniform(-j * Cw, j * Cw)
        cx = (W / 2.0) + dx
        x0 = int(round(cx - Cw / 2.0))
        x0 = max(0, min(x0, W - Cw))

        return img.crop((x0, 0, x0 + Cw, H))
    else:
        Ch = int(round(H * r))
        Ch = max(1, min(Ch, H))

        dy = rng.uniform(-j * Ch, j * Ch)
        cy = (H / 2.0) + dy
        y0 = int(round(cy - Ch / 2.0))
        y0 = max(0, min(y0, H - Ch))

        return img.crop((0, y0, W, y0 + Ch))


DEFAULT_LORA_TARGETS = [
    # MMDoubleStreamBlock
    "img_attn_q",
    "img_attn_k",
    "img_attn_v",
    "img_attn_proj",
    "txt_attn_q",
    "txt_attn_k",
    "txt_attn_v",
    "txt_attn_proj",
    # MMSingleStreamBlock
    "linear1_q",
    "linear1_k",
    "linear1_v",
    "linear2.fc",
    "linear1_mlp",
    # Modulation / gating
    "img_mod.linear",
    "txt_mod.linear",
    "modulation.linear",
    "adaLN_modulation.1",
]


@dataclass
class TrainingConfig:
    # Model paths
    pipeline_dir: str
    transformer_version: str = "480p_i2v"

    # Data (WebDataset)
    wds_root: str = ""
    wds_shuffle_buf: int = 4096
    batch_size: int = 1
    num_workers: int = 4
    p_secondary: float = 0.05
    secondary_cache_max_size: int = 64
    metadata_key: str = ""
    enable_augmentation: bool = False
    augment_r: float = 0.92
    augment_j: float = 0.04
    train_target_resolution: str = "480p"
    train_video_length: int = 17  # must be 4n+1 for VAE

    # Distillation / identity
    num_inference_steps: int = 14
    harvest_count: int = 2
    harvest_scheme: str = "bracket"
    harvest_jitter_frac: float = 0.05
    id_timestep_frac_low: float = 0.7
    id_timestep_frac_high: float = 0.9
    id_num_frames: int = 3
    lambda_id: float = 0.5
    lambda_id_schedule: str = "constant"  # constant|linear|cosine
    lambda_id_warmup_steps: int = 500
    id_every_steps: int = 1
    id_decode_downsample_mode: str = "false"  # false|area|bicubic
    id_decode_scale: float = 1.0

    # Optim
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_muon: bool = True
    max_train_steps: int = 10000
    warmup_steps: int = 500

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None
    pretrained_lora_path: Optional[str] = None

    # Identity model
    dino_model_dir: str = ""
    dino_head_path: str = ""

    # Output / logging
    output_dir: str = "./outputs"
    save_every_steps: int = 1000
    log_every_steps: int = 10
    dtype: str = "bf16"  # bf16|fp32
    seed: int = 42

    # Validation
    enable_validation: bool = False
    val_every_steps: int = 1000
    val_num_samples: int = 16
    val_video_length: int = 121

    # Distributed / misc
    enable_fsdp: bool = True
    enable_gradient_checkpointing: bool = True
    sp_size: int = 8
    resume_from_checkpoint: Optional[str] = None


class ProjectionHead(nn.Module):
    """Projection head used for DINO identity embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.05,
        use_layernorm: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for i, d in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, d))
            is_last = i == len(hidden_dims) - 1
            if not is_last:
                if use_layernorm:
                    layers.append(nn.LayerNorm(d))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev = d
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_projection_head_state(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "head_state_dict" in obj:
        return obj["head_state_dict"], obj.get("config", None)
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj, None
    raise RuntimeError(f"Unexpected projection head checkpoint format at {path}")


class DINOIdentity(nn.Module):
    """Frozen DINOv3 backbone + projection head."""

    def __init__(
        self,
        model_dir: str,
        head_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.backbone = AutoModel.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=dtype
        ).to(device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        size_cfg = self.processor.size
        if isinstance(size_cfg, dict):
            height = size_cfg.get("height") or size_cfg.get("shortest_edge") or 224
            width = size_cfg.get("width") or size_cfg.get("shortest_edge") or height
            self.target_size = (int(height), int(width))
        else:
            self.target_size = (int(size_cfg), int(size_cfg))
        self.image_mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1).to(device)
        self.image_std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1).to(device)

        head_sd, cfg = load_projection_head_state(head_path)
        proj_dims = cfg.get("proj_dims", [512, 256]) if isinstance(cfg, dict) else [512, 256]
        dropout = float(cfg.get("dropout", 0.05)) if isinstance(cfg, dict) else 0.05
        use_layernorm = bool(cfg.get("use_layernorm", True)) if isinstance(cfg, dict) else True

        self.head = ProjectionHead(
            in_dim=self.backbone.config.hidden_size,
            hidden_dims=proj_dims,
            dropout=dropout,
            use_layernorm=use_layernorm,
        ).to(device=device, dtype=dtype)
        self.head.load_state_dict(head_sd, strict=True)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad = False

        self.device = device
        self.dtype = dtype

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: float tensor in [0, 1], shape [B, 3, H, W]
        Returns:
            L2-normalized embeddings [B, D]
        """
        images = images.to(self.device)
        if images.shape[-2:] != self.target_size:
            images = F.interpolate(images, size=self.target_size, mode="bilinear", align_corners=False)
        images = (images - self.image_mean) / self.image_std
        with torch.cuda.amp.autocast(enabled=self.dtype == torch.bfloat16, dtype=self.dtype):
            out = self.backbone(pixel_values=images.to(self.dtype))
            pooled = out.pooler_output
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]
            proj = self.head(pooled.to(self.dtype))
            proj = F.normalize(proj, dim=-1)
        return proj


# ------------------------- WebDataset pipeline helpers -------------------------


def _load_manifest_counts(path: str, metadata_key: str) -> Tuple[Dict[str, int], int]:
    """
    Manifest schema:
      <metadata_key>,count
    """
    counts: Dict[str, int] = {}
    if not os.path.exists(path):
        logger.warning(f"Manifest not found at {path}; inverse-frequency sampling will accept all samples.")
        return counts, 1

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get(metadata_key)
            if key is None:
                continue
            k = str(key).strip().lower()
            try:
                c = int(row.get("count", 0))
            except Exception:
                c = 0
            if c > 0:
                counts[k] = c

    min_count = min(counts.values()) if counts else 1
    return counts, max(1, int(min_count))


def _wds_worker_seed(base_seed: int) -> int:
    wi = torch.utils.data.get_worker_info()
    worker_id = wi.id if wi is not None else 0
    rank = int(os.environ.get("RANK", "0"))
    # Spread seeds across ranks/workers
    return int(base_seed + 10007 * rank + 37 * worker_id)


def _pil_to_tensor_pm1(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W] in [0,1]
    return t * 2.0 - 1.0  # [-1,1]


_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _raw_pil_handler(key: str, data: Any) -> Optional[Image.Image]:
    ext = key.rsplit(".", 1)[-1].lower()
    if ext not in _IMAGE_EXTENSIONS or not isinstance(data, (bytes, bytearray)):
        return None
    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
    return img


def _ensure_rgb(img: Image.Image, bg: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Convert any PIL image to RGB while correctly handling palette transparency.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        bg_img = Image.new("RGBA", rgba.size, (*bg, 255))
        rgba = Image.alpha_composite(bg_img, rgba)
        return rgba.convert("RGB")
    return img.convert("RGB")


def _make_wds_preprocess_fn(
    enable_augmentation: bool,
    augment_r: float,
    augment_j: float,
    base_seed: int,
    metadata_key: str,
):
    """
    Input: (pil_img, meta_dict)
    Output: dict with fields expected by identity_collate / trainer.
    """
    def _fn(tup):
        img_pil, meta = tup
        if not isinstance(meta, dict):
            meta = {}

        character = meta.get(metadata_key)
        if character is None:
            character = "unknown"
        character = str(character).strip().lower()

        rng = random.Random(_wds_worker_seed(base_seed))

        img_pil = _ensure_rgb(img_pil, bg=(0, 0, 0))
        if enable_augmentation:
            img_pil = crop_long_side_only(img_pil, r=augment_r, j=augment_j, rng=rng)

        return {
            "pixel_values": _pil_to_tensor_pm1(img_pil),
            "text": character,
            "character": character,
            "data_type": "image",
            "primary_pil": img_pil,
            "meta": meta,
        }

    return _fn


def _make_inverse_freq_reject_fn(
    character_counts: Dict[str, int],
    min_count: int,
    base_seed: int,
    metadata_key: str,
):
    """
    Inverse-frequency rejection sampling:
      accept with p = min(1, min_count / count(character))
    """
    def _accept(sample: Dict[str, Any]) -> bool:
        character = sample.get("character", None)
        if character is None:
            meta = sample.get("meta", {})
            character = meta.get(metadata_key) or "unknown"
            character = str(character).strip().lower()

        c = int(character_counts.get(character, 0))
        if c <= 0:
            return True

        p = float(min_count) / float(c)
        if p >= 1.0:
            return True

        rng = random.Random(_wds_worker_seed(base_seed) + 1337)
        return rng.random() < p

    return _accept


def _make_secondary_mapper(
    p_secondary: float,
    base_seed: int,
    enable_secondary: bool = True,
    max_cache_size: int = 64,
):
    """
    Streaming-friendly same-character "secondary" sampling:
      keep a small per-worker cache: character -> last seen sample (pil/tensor).
      With probability p_secondary, attach the previous cached sample for that character.
    """
    if (not enable_secondary) or p_secondary <= 0 or max_cache_size <= 0:
        def _no_secondary(sample: Dict[str, Any]) -> Dict[str, Any]:
            sample["secondary_pixel_values"] = None
            sample["secondary_pil"] = None
            sample["secondary_mask"] = False
            return sample

        return _no_secondary

    # LRU cache to cap memory usage when many unique characters are present.
    cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    rng = random.Random(_wds_worker_seed(base_seed) + 4242)

    def _fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        character = sample["character"]

        sec = cache.get(character, None) if rng.random() < p_secondary else None

        if sec is not None:
            sample["secondary_pixel_values"] = sec
            sample["secondary_pil"] = None
            sample["secondary_mask"] = True
        else:
            sample["secondary_pixel_values"] = None
            sample["secondary_pil"] = None
            sample["secondary_mask"] = False

        # update cache with current; enforce LRU size limit to prevent unbounded growth
        cache[character] = sample["pixel_values"]
        cache.move_to_end(character)
        if len(cache) > max_cache_size:
            cache.popitem(last=False)
        return sample

    return _fn


def identity_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    primary = torch.stack([b["pixel_values"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    characters = [b["character"] for b in batch]
    primary_pils = [b["primary_pil"] for b in batch]

    sec_mask = []
    sec_images = []
    sec_pils = []
    for b in batch:
        if b.get("secondary_pixel_values", None) is None:
            sec_mask.append(False)
            sec_images.append(torch.zeros_like(primary[0]))
            sec_pils.append(None)
        else:
            sec_mask.append(True)
            sec_images.append(b["secondary_pixel_values"])
            sec_pils.append(b.get("secondary_pil", None))

    secondary = torch.stack(sec_images, dim=0)
    return {
        "pixel_values": primary,
        "text": texts,
        "character": characters,
        "data_type": "image",
        "primary_pils": primary_pils,
        "secondary_pixel_values": secondary,
        "secondary_mask": torch.tensor(sec_mask, dtype=torch.bool),
        "secondary_pils": sec_pils,
    }


def sync_tensor_for_sp(tensor, sp_group):
    """
    Sync tensor within sequence parallel group.
    Ensures all ranks in the SP group have the same tensor values.
    """
    if sp_group is None:
        return tensor
    if not isinstance(tensor, torch.Tensor):
        obj_list = [tensor]
        dist.broadcast_object_list(obj_list, src=0, group=sp_group)
        return obj_list[0]
    dist.broadcast(tensor, src=0, group=sp_group)
    return tensor


class HunyuanVideoTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True

        if config.sp_size > self.world_size:
            raise ValueError(f"sp_size ({config.sp_size}) cannot be greater than world_size ({self.world_size})")
        if self.world_size % config.sp_size != 0:
            raise ValueError(
                f"sp_size ({config.sp_size}) must evenly divide world_size ({self.world_size}). "
                f"world_size % sp_size = {self.world_size % config.sp_size}"
            )

        initialize_parallel_state(sp=config.sp_size)
        torch.cuda.set_device(self.local_rank)
        self.parallel_state = get_parallel_state()
        self.dp_rank = self.parallel_state.world_mesh["dp"].get_local_rank()
        self.dp_size = self.parallel_state.world_mesh["dp"].size()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

        self.model_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float32
        self._set_seed(config.seed + self.dp_rank)

        self.pipeline = self._build_pipeline()
        self.student_transformer = self.pipeline.transformer
        self.teacher_transformer = self._load_teacher()

        if self.config.use_lora:
            self._apply_lora()

        if self.config.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing()

        if self.config.enable_fsdp and self.world_size > 1:
            self._apply_fsdp()

        if self.is_main_process:
            logger.info("building optimizer...")
        self._build_optimizer()
        if self.is_main_process:
            logger.info("loading DINO...")
        self.dino = DINOIdentity(
            model_dir=self.config.dino_model_dir,
            head_path=self.config.dino_head_path,
            dtype=self.model_dtype,
            device=self.device,
        )

        self.global_step = 0
        if self.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)

        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_pipeline(self):
        pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=self.config.pipeline_dir,
            transformer_version=self.config.transformer_version,
            transformer_dtype=self.model_dtype,
            enable_offloading=False,
            enable_group_offloading=False,
            overlap_group_offloading=False,
            create_sr_pipeline=False,
            flow_shift=1.0,
            device=self.device,
        )
        text_encoder, text_encoder_2 = HunyuanVideo_1_5_Pipeline._load_text_encoders(
            self.config.pipeline_dir, device=self.device
        )
        pipeline.text_encoder = text_encoder
        pipeline.text_encoder_2 = text_encoder_2
        byt5_kwargs, prompt_format = HunyuanVideo_1_5_Pipeline._load_byt5(
            self.config.pipeline_dir,
            glyph_byT5_v2=pipeline.config.glyph_byT5_v2,
            byt5_max_length=pipeline.config.byt5_max_length if hasattr(pipeline.config, "byt5_max_length") else 256,
            device=self.device,
        )
        pipeline.byt5_model = byt5_kwargs["byt5_model"]
        pipeline.byt5_tokenizer = byt5_kwargs["byt5_tokenizer"]
        pipeline.prompt_format = prompt_format
        pipeline.scheduler = FlowMatchDiscreteScheduler.from_pretrained(os.path.join(self.config.pipeline_dir, "scheduler"))
        pipeline.target_dtype = self.model_dtype
        pipeline.execution_device = self.device
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.text_encoder_2 = pipeline.text_encoder_2
        self.vision_encoder = pipeline.vision_encoder
        self.byt5_kwargs = {"byt5_model": pipeline.byt5_model, "byt5_tokenizer": pipeline.byt5_tokenizer}
        self.scheduler_config = pipeline.scheduler.config
        return pipeline

    def _load_teacher(self):
        transformer_path = os.path.join(self.config.pipeline_dir, "transformer", self.config.transformer_version)
        teacher = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            transformer_path, low_cpu_mem_usage=True, torch_dtype=self.model_dtype
        ).to(self.device)
        teacher.eval()
        teacher.requires_grad_(False)
        return teacher

    def _apply_lora(self):
        if self.is_main_process:
            logger.info("Applying LoRA adapters to student transformer")
        from peft import LoraConfig

        target_modules = self.config.lora_target_modules or DEFAULT_LORA_TARGETS
        self.student_transformer.requires_grad_(False)

        if self.config.pretrained_lora_path:
            self.student_transformer.load_lora_adapter(
                pretrained_model_name_or_path_or_dict=self.config.pretrained_lora_path,
                adapter_name="default",
                use_safetensors=True,
            )
        else:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.student_transformer.add_adapter(lora_config, adapter_name="default")

        for name, param in self.student_transformer.named_parameters():
            if "lora_" in name or "adapter" in name:
                param.requires_grad = True

        if self.is_main_process:
            total_params = sum(p.numel() for p in self.student_transformer.parameters())
            trainable_params = sum(p.numel() for p in self.student_transformer.parameters() if p.requires_grad)
            logger.info(f"LoRA trainable params: {trainable_params:,} / {total_params:,}")

    def _apply_fsdp(self):
        if self.is_main_process:
            logger.info("Applying FSDP2 to student transformer...")

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.model_dtype,
            reduce_dtype=torch.float32,
        )

        fsdp_config = {"mp_policy": mp_policy}
        if self.world_size > 1:
            try:
                fsdp_config["mesh"] = get_parallel_state().fsdp_mesh
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Could not create DeviceMesh: {e}. FSDP will use process group instead.")

        for block in list(self.student_transformer.double_blocks) + list(self.student_transformer.single_blocks):
            if block is not None:
                fully_shard(block, **fsdp_config)

        fully_shard(self.student_transformer, **fsdp_config)

    def _apply_gradient_checkpointing(self):
        no_split_module_type = None
        for block in self.student_transformer.double_blocks:
            if block is not None:
                no_split_module_type = type(block)
                break
        if no_split_module_type is None:
            for block in self.student_transformer.single_blocks:
                if block is not None:
                    no_split_module_type = type(block)
                    break
        if no_split_module_type is None:
            logger.warning("Could not find block type for gradient checkpointing. Using fallback.")
            if hasattr(self.student_transformer, "gradient_checkpointing_enable"):
                self.student_transformer.gradient_checkpointing_enable()
            return

        def non_reentrant_wrapper(module):
            return checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

        def selective_checkpointing(submodule):
            return isinstance(submodule, no_split_module_type)

        apply_activation_checkpointing(
            self.student_transformer,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=selective_checkpointing,
        )

    def _build_optimizer(self):
        params = [p for p in self.student_transformer.parameters() if p.requires_grad]
        if self.config.use_muon:
            self.optimizer = get_muon_optimizer(
                model=self.student_transformer,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            )
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_train_steps,
        )

    def encode_text(self, prompts: List[str], data_type: str = "video"):
        text_inputs = self.text_encoder.text2tokens(prompts, data_type=data_type)
        text_outputs = self.text_encoder.encode(text_inputs, data_type=data_type, device=self.device)
        text_emb = text_outputs.hidden_state
        text_mask = text_outputs.attention_mask

        text_emb_2 = None
        text_mask_2 = None
        if self.text_encoder_2 is not None:
            text_inputs_2 = self.text_encoder_2.text2tokens(prompts)
            text_outputs_2 = self.text_encoder_2.encode(text_inputs_2, device=self.device)
            text_emb_2 = text_outputs_2.hidden_state
            text_mask_2 = text_outputs_2.attention_mask
        return text_emb, text_mask, text_emb_2, text_mask_2

    def _prepare_conditions(self, batch: Dict[str, Any], video_length: int):
        prompts = [build_prompt.build_prompt(c) for c in batch["character"]]
        text_emb, text_mask, text_emb_2, text_mask_2 = self.encode_text(prompts, data_type="video")
        extra_kwargs = {}
        if getattr(self.pipeline.config, "glyph_byT5_v2", False):
            with torch.no_grad():
                extra_kwargs = self.pipeline._prepare_byt5_embeddings(prompts, device=self.device)

        batch_size = batch["pixel_values"].shape[0]
        primary_pils: List[Image.Image] = batch["primary_pils"]
        height, width = self.pipeline.get_closest_resolution_given_reference_image(
            primary_pils[0], self.config.train_target_resolution
        )
        latent_target_length, latent_height, latent_width = self.pipeline.get_latent_size(
            video_length, height, width
        )

        latents = self.pipeline.prepare_latents(
            batch_size,
            self.student_transformer.config.in_channels,
            latent_height,
            latent_width,
            latent_target_length,
            self.model_dtype,
            self.device,
            generator=None,
        )

        cond_latents_list = []
        for pil in primary_pils:
            cond_latents_list.append(self.pipeline.get_image_condition_latents("i2v", pil, height, width))
        cond_latents = torch.cat(cond_latents_list, dim=0)
        multitask_mask = self.pipeline.get_task_mask("i2v", latent_target_length)
        cond_latents = self.pipeline._prepare_cond_latents(
            "i2v", cond_latents.to(self.device), latents, multitask_mask.to(self.device)
        )

        vision_states = None
        if self.vision_encoder is not None:
            vs_list = []
            for pil in primary_pils:
                img_np = np.array(pil)
                vs = self.vision_encoder.encode_images(img_np)
                # vs.last_hidden_state is typically [1, T, D]
                vs_list.append(vs.last_hidden_state.to(device=self.device, dtype=self.model_dtype))
            vision_states = torch.cat(vs_list, dim=0)  # [B, T, D]

        return {
            "latents": latents,
            "cond_latents": cond_latents,
            "vision_states": vision_states,
            "text_emb": text_emb,
            "text_mask": text_mask,
            "text_emb_2": text_emb_2,
            "text_mask_2": text_mask_2,
            "extra_kwargs": extra_kwargs,
            "prompts": prompts,
            "height": height,
            "width": width,
            "latent_target_length": latent_target_length,
        }

    def _new_scheduler(self):
        return FlowMatchDiscreteScheduler.from_config(self.scheduler_config)

    def _select_harvest_indices(self, num_steps: int) -> Tuple[List[int], int]:
        if num_steps <= 0:
            return [], 0
        indices: List[int] = []
        if self.config.harvest_scheme == "random":
            indices = sorted(random.sample(range(num_steps), k=min(self.config.harvest_count, num_steps)))
        else:
            base_fracs = [0.5, 0.25]
            jitter = max(1, int(num_steps * self.config.harvest_jitter_frac))
            for frac in base_fracs:
                idx = int(round(frac * (num_steps - 1)))
                idx += random.randint(-jitter, jitter)
                idx = min(max(idx, 0), num_steps - 1)
                indices.append(idx)
            indices = sorted(set(indices))
            if self.config.harvest_count < len(indices):
                indices = indices[: self.config.harvest_count]

        low = int(self.config.id_timestep_frac_low * (num_steps - 1))
        high = int(self.config.id_timestep_frac_high * (num_steps - 1))
        id_idx = min(num_steps - 1, max(low, random.randint(low, max(low, high))))
        if id_idx not in indices:
            indices.append(id_idx)
            indices = sorted(set(indices))
        return indices, id_idx

    def _teacher_rollout(
        self,
        latents: torch.Tensor,
        cond_latents: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        text_emb_2: Optional[torch.Tensor],
        text_mask_2: Optional[torch.Tensor],
        vision_states: Optional[torch.Tensor],
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Roll teacher from T down to t_deep (id_idx) only.
        Harvest along the way (<= id_idx), and stop at id_idx (do not continue to 0).
        """
        scheduler = self._new_scheduler()
        scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        harvest_indices, id_idx = self._select_harvest_indices(len(timesteps))
        harvest_indices = sorted([i for i in harvest_indices if i <= id_idx])

        cached = []
        identity_latents = None
        identity_t = None

        for idx, t in enumerate(timesteps):
            if idx > id_idx:
                break

            latents_input = torch.cat([latents, cond_latents], dim=1)
            t_expand = t.repeat(latents_input.shape[0])
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=self.model_dtype, enabled=self.model_dtype == torch.bfloat16
            ):
                teacher_pred = self.teacher_transformer(
                    latents_input,
                    t_expand,
                    text_states=text_emb.to(self.model_dtype),
                    text_states_2=text_emb_2.to(self.model_dtype) if text_emb_2 is not None else None,
                    encoder_attention_mask=text_mask.to(self.model_dtype),
                    vision_states=vision_states.to(self.model_dtype) if vision_states is not None else None,
                    mask_type="i2v",
                    extra_kwargs=extra_kwargs,
                    return_dict=False,
                )[0]

            if idx in harvest_indices:
                cached.append((latents.detach(), teacher_pred.detach(), t_expand.detach()))

            if idx == id_idx:
                identity_latents = latents.detach()
                identity_t = t.detach()
                break

            latents = scheduler.step(teacher_pred, t_expand, latents, return_dict=False)[0]

        return cached, identity_latents, identity_t, timesteps

    def _distill_loss(
        self,
        cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cond_latents: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        text_emb_2: Optional[torch.Tensor],
        text_mask_2: Optional[torch.Tensor],
        vision_states: Optional[torch.Tensor],
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not cache:
            return torch.tensor(0.0, device=self.device)

        latents_inputs = []
        timesteps = []
        teacher_preds = []
        for lat, teacher_pred, ts in cache:
            latents_inputs.append(torch.cat([lat, cond_latents], dim=1))
            timesteps.append(ts)
            teacher_preds.append(teacher_pred)

        latents_batch = torch.cat(latents_inputs, dim=0)
        timesteps_batch = torch.cat(timesteps, dim=0)
        teacher_batch = torch.cat(teacher_preds, dim=0)

        rep = latents_batch.shape[0] // text_emb.shape[0]
        text_emb_exp = text_emb.repeat_interleave(rep, dim=0)
        text_mask_exp = text_mask.repeat_interleave(rep, dim=0) if text_mask is not None else None
        text_emb_2_exp = text_emb_2.repeat_interleave(rep, dim=0) if text_emb_2 is not None else None
        text_mask_2_exp = text_mask_2.repeat_interleave(rep, dim=0) if text_mask_2 is not None else None
        vision_states_exp = vision_states.repeat_interleave(rep, dim=0) if vision_states is not None else None
        extra_kwargs_exp = extra_kwargs
        if extra_kwargs is not None and "byt5_text_states" in extra_kwargs:
            extra_kwargs_exp = {
                "byt5_text_states": extra_kwargs["byt5_text_states"].repeat_interleave(rep, dim=0),
                "byt5_text_mask": extra_kwargs["byt5_text_mask"].repeat_interleave(rep, dim=0),
            }

        with torch.autocast(device_type="cuda", dtype=self.model_dtype, enabled=self.model_dtype == torch.bfloat16):
            student_pred = self.student_transformer(
                latents_batch.to(self.model_dtype),
                timesteps_batch,
                text_states=text_emb_exp.to(self.model_dtype),
                text_states_2=text_emb_2_exp.to(self.model_dtype) if text_emb_2_exp is not None else None,
                encoder_attention_mask=text_mask_exp.to(self.model_dtype) if text_mask_exp is not None else None,
                vision_states=vision_states_exp.to(self.model_dtype) if vision_states_exp is not None else None,
                mask_type="i2v",
                extra_kwargs=extra_kwargs_exp,
                return_dict=False,
            )[0]

        mse = ((student_pred - teacher_batch.to(student_pred.dtype)) ** 2).mean()
        return mse

    def _downsample_latents_hw(self, latents: torch.Tensor, scale: float, mode: str) -> torch.Tensor:
        mode = mode.lower()
        if mode == "false" or scale is None or scale >= 1.0:
            return latents
        if scale <= 0:
            raise ValueError(f"id_decode_scale must be > 0, got {scale}")

        orig_dtype = latents.dtype
        if latents.ndim == 5:
            b, c, t, h, w = latents.shape
            latents_4d = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            has_time = True
        elif latents.ndim == 4:
            latents_4d = latents
            has_time = False
        else:
            return latents

        if latents_4d.dtype not in (torch.float16, torch.float32):
            latents_4d = latents_4d.float()

        if mode == "area":
            latents_4d = F.interpolate(latents_4d, scale_factor=scale, mode="area")
        elif mode == "bicubic":
            latents_4d = F.interpolate(
                latents_4d,
                scale_factor=scale,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        else:
            raise ValueError(f"Unsupported id_decode_downsample_mode: {mode}")

        if latents_4d.dtype != orig_dtype:
            latents_4d = latents_4d.to(orig_dtype)

        if has_time:
            h2, w2 = latents_4d.shape[-2:]
            latents_4d = latents_4d.reshape(b, t, c, h2, w2).permute(0, 2, 1, 3, 4)
        return latents_4d

    def _decode_latents(
        self,
        latents: torch.Tensor,
        downsample_mode: Optional[str] = None,
        downsample_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if downsample_mode is not None and downsample_mode != "false":
            latents = self._downsample_latents_hw(latents, downsample_scale, downsample_mode)
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            self.vae.enable_tiling()
            video = self.vae.decode(latents, return_dict=False)[0]
            self.vae.disable_tiling()
        return (video / 2 + 0.5).clamp(0, 1)

    def _sample_frame_indices(self, num_frames: int) -> List[int]:
        if num_frames <= 1:
            return [0]
        rng = random
        buckets = [
            (1, max(1, int(0.10 * num_frames))),
            (int(0.10 * num_frames), max(1, int(0.30 * num_frames))),
            (int(0.30 * num_frames), max(1, int(0.60 * num_frames))),
            (int(0.60 * num_frames), num_frames - 1),
            (1, max(1, int(0.20 * num_frames))),
        ]
        indices = []
        for start, end in buckets[: self.config.id_num_frames]:
            start = min(start, num_frames - 1)
            end = max(start + 1, end)
            idx = rng.randint(start, end - 1)
            if idx not in indices:
                indices.append(idx)
        while len(indices) < self.config.id_num_frames:
            idx = rng.randint(0, num_frames - 1)
            if idx not in indices:
                indices.append(idx)
        return sorted(indices)

    def _identity_cosine_stats(
        self,
        identity_latents: torch.Tensor,
        cond_latents: torch.Tensor,
        identity_t: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        text_emb_2: Optional[torch.Tensor],
        text_mask_2: Optional[torch.Tensor],
        vision_states: Optional[torch.Tensor],
        anchors: torch.Tensor,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          cos_mean: scalar tensor
          cos_std:  scalar tensor
        """
        scheduler = self._new_scheduler()
        scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)

        latents = identity_latents
        latents_input = torch.cat([latents, cond_latents], dim=1)
        t_expand = identity_t.repeat(latents_input.shape[0])

        with torch.autocast(device_type="cuda", dtype=self.model_dtype, enabled=self.model_dtype == torch.bfloat16):
            pred = self.student_transformer(
                latents_input,
                t_expand,
                text_states=text_emb.to(self.model_dtype),
                text_states_2=text_emb_2.to(self.model_dtype) if text_emb_2 is not None else None,
                encoder_attention_mask=text_mask.to(self.model_dtype) if text_mask is not None else None,
                vision_states=vision_states.to(self.model_dtype) if vision_states is not None else None,
                mask_type="i2v",
                extra_kwargs=extra_kwargs,
                return_dict=False,
            )[0]

        # Recover x0 estimate using linear blend formula: x0 = x_t - s * pred
        s = (identity_t.float() / float(self.scheduler_config.num_train_timesteps)).view(
            -1, *([1] * (latents.dim() - 1))
        )
        latents_x0 = latents - s * pred.to(latents.dtype)

        video = self._decode_latents(
            latents_x0,
            downsample_mode=self.config.id_decode_downsample_mode,
            downsample_scale=self.config.id_decode_scale,
        )
        frame_indices = self._sample_frame_indices(video.shape[2])

        all_cos = []
        for b in range(video.shape[0]):
            frames = video[b, :, frame_indices, :, :]                # [3, K, H, W]
            frames = frames.permute(1, 0, 2, 3).contiguous()         # [K, 3, H, W]

            anchor = anchors[b : b + 1]                              # [1, 3, H, W]
            emb_anchor = self.dino(anchor)
            emb_frames = self.dino(frames)

            cos = (emb_frames * emb_anchor).sum(dim=-1)              # [K]
            all_cos.append(cos)

        cos_all = torch.cat(all_cos, dim=0) if len(all_cos) > 0 else torch.tensor([0.0], device=self.device)
        return cos_all.mean(), cos_all.std(unbiased=False)

    def _identity_loss(
        self,
        identity_latents: torch.Tensor,
        cond_latents: torch.Tensor,
        identity_t: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        text_emb_2: Optional[torch.Tensor],
        text_mask_2: Optional[torch.Tensor],
        vision_states: Optional[torch.Tensor],
        anchors: torch.Tensor,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        cos_mean, _ = self._identity_cosine_stats(
            identity_latents=identity_latents,
            cond_latents=cond_latents,
            identity_t=identity_t,
            text_emb=text_emb,
            text_mask=text_mask,
            text_emb_2=text_emb_2,
            text_mask_2=text_mask_2,
            vision_states=vision_states,
            anchors=anchors,
            extra_kwargs=extra_kwargs,
        )
        return (1.0 - cos_mean).clamp(min=0.0, max=2.0)

    def _lambda_id_weight(self, step: int) -> float:
        base = self.config.lambda_id
        if self.config.lambda_id_schedule == "constant":
            return base
        progress = min(1.0, float(step) / max(1, self.config.lambda_id_warmup_steps))
        if self.config.lambda_id_schedule == "linear":
            return base * progress
        if self.config.lambda_id_schedule == "cosine":
            return base * 0.5 * (1 - math.cos(math.pi * progress))
        return base

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.student_transformer.train()
        setup = self._prepare_conditions(batch, self.config.train_video_length)
        cache, identity_latents, identity_t, _timesteps = self._teacher_rollout(
            setup["latents"],
            setup["cond_latents"],
            setup["text_emb"],
            setup["text_mask"],
            setup["text_emb_2"],
            setup["text_mask_2"],
            setup["vision_states"],
            setup["extra_kwargs"],
        )

        distill_loss = self._distill_loss(
            cache,
            setup["cond_latents"],
            setup["text_emb"],
            setup["text_mask"],
            setup["text_emb_2"],
            setup["text_mask_2"],
            setup["vision_states"],
            setup["extra_kwargs"],
        )

        if identity_latents is not None and identity_t is not None and (self.global_step % self.config.id_every_steps == 0):
            anchors = torch.where(
                batch["secondary_mask"].view(-1, 1, 1, 1),
                batch["secondary_pixel_values"],
                batch["pixel_values"],
            )
            anchors = ((anchors.to(self.device)) + 1) / 2.0
            id_loss = self._identity_loss(
                identity_latents,
                setup["cond_latents"],
                identity_t,
                setup["text_emb"],
                setup["text_mask"],
                setup["text_emb_2"],
                setup["text_mask_2"],
                setup["vision_states"],
                anchors,
                setup["extra_kwargs"],
            )
        else:
            id_loss = torch.tensor(0.0, device=self.device)

        lambda_id = self._lambda_id_weight(self.global_step)
        total_loss = distill_loss + lambda_id * id_loss
        total_loss = total_loss / self.config.gradient_accumulation_steps
        total_loss.backward()

        grad_norm = torch.tensor(0.0)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.student_transformer.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {
            "loss_total": total_loss.item() * self.config.gradient_accumulation_steps,
            "loss_distill": distill_loss.item(),
            "loss_id": id_loss.item(),
            "lambda_id": lambda_id,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

    def validate(self, val_loader):
        """
        Metrics (rank 0 only), aligned with your spec:
          1) DINO similarity over time: mean/std of cosine(anchor, sampled-frame-embeddings)
          2) motion metric (simple): mean absolute frame-to-frame difference
          3) teacher deviation (distill MSE) on held-out prompts (val split)
        """
        if not self.config.enable_validation or not self.is_main_process:
            return
        if val_loader is None:
            logger.warning("Validation enabled but no val_loader was provided.")
            return

        self.student_transformer.eval()
        metrics = {
            "distill_mse": [],
            "dino_cos_mean": [],
            "dino_cos_std": [],
            "motion": [],
        }

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= self.config.val_num_samples:
                    break

                setup = self._prepare_conditions(batch, self.config.val_video_length)

                cache, identity_latents, identity_t, _timesteps = self._teacher_rollout(
                    setup["latents"],
                    setup["cond_latents"],
                    setup["text_emb"],
                    setup["text_mask"],
                    setup["text_emb_2"],
                    setup["text_mask_2"],
                    setup["vision_states"],
                    setup["extra_kwargs"],
                )

                distill_loss = self._distill_loss(
                    cache,
                    setup["cond_latents"],
                    setup["text_emb"],
                    setup["text_mask"],
                    setup["text_emb_2"],
                    setup["text_mask_2"],
                    setup["vision_states"],
                    setup["extra_kwargs"],
                )
                metrics["distill_mse"].append(distill_loss.item())

                anchors = ((batch["pixel_values"].to(self.device)) + 1) / 2.0
                if identity_latents is not None and identity_t is not None:
                    cos_mean, cos_std = self._identity_cosine_stats(
                        identity_latents=identity_latents,
                        cond_latents=setup["cond_latents"],
                        identity_t=identity_t,
                        text_emb=setup["text_emb"],
                        text_mask=setup["text_mask"],
                        text_emb_2=setup["text_emb_2"],
                        text_mask_2=setup["text_mask_2"],
                        vision_states=setup["vision_states"],
                        anchors=anchors,
                        extra_kwargs=setup["extra_kwargs"],
                    )
                    metrics["dino_cos_mean"].append(float(cos_mean.item()))
                    metrics["dino_cos_std"].append(float(cos_std.item()))

                    # motion metric on student x0 preview
                    # (reuse the same decode path as identity metrics)
                    scheduler = self._new_scheduler()
                    scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)
                    latents_input = torch.cat([identity_latents, setup["cond_latents"]], dim=1)
                    t_expand = identity_t.repeat(latents_input.shape[0])
                    with torch.autocast(device_type="cuda", dtype=self.model_dtype, enabled=self.model_dtype == torch.bfloat16):
                        pred = self.student_transformer(
                            latents_input,
                            t_expand,
                        text_states=setup["text_emb"].to(self.model_dtype),
                        text_states_2=setup["text_emb_2"].to(self.model_dtype) if setup["text_emb_2"] is not None else None,
                        encoder_attention_mask=setup["text_mask"].to(self.model_dtype) if setup["text_mask"] is not None else None,
                        vision_states=setup["vision_states"].to(self.model_dtype) if setup["vision_states"] is not None else None,
                        mask_type="i2v",
                        extra_kwargs=setup["extra_kwargs"],
                        return_dict=False,
                    )[0]
                    s = (identity_t.float() / float(self.scheduler_config.num_train_timesteps)).view(
                        -1, *([1] * (identity_latents.dim() - 1))
                    )
                    latents_x0 = identity_latents - s * pred.to(identity_latents.dtype)
                    video = self._decode_latents(
                        latents_x0,
                        downsample_mode=self.config.id_decode_downsample_mode,
                        downsample_scale=self.config.id_decode_scale,
                    )

                    if video.shape[2] > 1:
                        motion = (video[:, :, 1:] - video[:, :, :-1]).abs().mean().item()
                        metrics["motion"].append(motion)

        def _mean(xs):
            return sum(xs) / max(1, len(xs))

        logger.info(
            f"[val] teacher_dev(mse)={_mean(metrics['distill_mse']):.6f} | "
            f"dino_cos_mean={_mean(metrics['dino_cos_mean']):.4f} | "
            f"dino_cos_std={_mean(metrics['dino_cos_std']):.4f} | "
            f"motion={_mean(metrics['motion']):.6f}"
        )
        self.student_transformer.train()

    def _save_checkpoint_dcp(self, checkpoint_dir: str):
        """
        Robust checkpointing under composable FSDP:
          - save sharded model + optimizer with torch.distributed.checkpoint (DCP)
        """
        try:
            from torch.distributed.checkpoint import save as dcp_save
            from torch.distributed.checkpoint import FileSystemWriter
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"DCP not available; skipping sharded checkpoint save. Error: {e}")
            return

        writer = FileSystemWriter(os.path.join(checkpoint_dir, "dcp"))
        state = {
            "model": self.student_transformer,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "global_step": torch.tensor([self.global_step], device="cpu"),
        }
        dcp_save(state, writer)

    def _load_checkpoint_dcp(self, checkpoint_dir: str):
        try:
            from torch.distributed.checkpoint import load as dcp_load
            from torch.distributed.checkpoint import FileSystemReader
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"DCP not available; skipping sharded checkpoint load. Error: {e}")
            return False

        dcp_path = os.path.join(checkpoint_dir, "dcp")
        if not os.path.exists(dcp_path):
            return False

        reader = FileSystemReader(dcp_path)
        state = {
            "model": self.student_transformer,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
            "global_step": torch.tensor([0], device="cpu"),
        }
        dcp_load(state, reader)
        self.global_step = int(state["global_step"].item())
        return True

    def save_checkpoint(self, step: int):
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Use DCP for multi-rank FSDP checkpoints.
        if self.config.enable_fsdp and self.world_size > 1:
            self._save_checkpoint_dcp(checkpoint_dir)
            # Only rank0 writes small training_state pointer for convenience
            if self.is_main_process:
                torch.save(
                    {"global_step": step},
                    os.path.join(checkpoint_dir, "training_state.pt"),
                )
            return

        # Non-FSDP (or single-rank): save training state only
        if self.is_main_process:
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "global_step": step,
                },
                os.path.join(checkpoint_dir, "training_state.pt"),
            )

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # FSDP checkpoint load via DCP.
        if self.config.enable_fsdp and self.world_size > 1:
            ok = self._load_checkpoint_dcp(checkpoint_path)
            if ok and self.is_main_process:
                logger.info(f"Resumed (DCP) from {checkpoint_path} at step {self.global_step}")
            return

        # Non-FSDP path: adapter (if present) + optimizer state.
        lora_dir = os.path.join(checkpoint_path, "lora")
        if os.path.exists(lora_dir) and self.config.use_lora:
            self.student_transformer.load_lora_adapter(
                pretrained_model_name_or_path_or_dict=lora_dir,
                adapter_name="default",
                use_safetensors=True,
            )
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state.get("optimizer", {}))
            self.lr_scheduler.load_state_dict(state.get("lr_scheduler", {}))
            self.global_step = state.get("global_step", 0)
            if self.is_main_process:
                logger.info(f"Resumed from {checkpoint_path} at step {self.global_step}")

    def train(self, train_loader, val_loader=None):
        if self.is_main_process:
            logger.info("Starting training")
        while self.global_step < self.config.max_train_steps:
            pbar = None
            if self.is_main_process and tqdm is not None:
                pbar = tqdm(
                    total=self.config.max_train_steps,
                    initial=self.global_step,
                    desc="train",
                    dynamic_ncols=True,
                    ascii=True,
                )
            try:
                for batch in train_loader:
                    if self.global_step >= self.config.max_train_steps:
                        break
                    metrics = self.train_step(batch)
                    if self.global_step % self.config.log_every_steps == 0 and self.is_main_process:
                        logger.info(
                            f"step {self.global_step} | loss={metrics['loss_total']:.4f} "
                            f"(distill={metrics['loss_distill']:.4f}, id={metrics['loss_id']:.4f}, "
                            f"lambda_id={metrics['lambda_id']:.3f}) grad={metrics['grad_norm']:.4f} "
                            f"lr={metrics['lr']:.2e}"
                        )
                    if self.config.enable_validation and self.global_step % self.config.val_every_steps == 0:
                        self.validate(val_loader)
                    self.global_step += 1
                    if pbar is not None:
                        pbar.update(1)
                    if self.global_step % self.config.save_every_steps == 0:
                        self.save_checkpoint(self.global_step)
            finally:
                if pbar is not None:
                    pbar.close()
        if self.config.enable_fsdp and self.world_size > 1:
            self.save_checkpoint(self.global_step)
        elif self.is_main_process:
            self.save_checkpoint(self.global_step)


def _build_wds_dataset(
    root: str,
    split: str,
    metadata_key: str,
    batch_size: int,
    shuffle_buf: int,
    seed: int,
    p_secondary: float,
    enable_secondary: bool,
    secondary_cache_max_size: int,
    enable_augmentation: bool,
    augment_r: float,
    augment_j: float,
    reject_using_manifest: bool = True,
) -> wds.DataPipeline:
    shards = sorted(glob(os.path.join(root, "data", split, f"{split}-*.tar")))
    if not shards:
        raise FileNotFoundError(f"No shards found under {root}/data/{split}/{split}-*.tar")

    manifest_path = os.path.join(root, "manifest", f"{split}_manifest.csv")
    character_counts, min_count = _load_manifest_counts(manifest_path, metadata_key)

    preprocess = _make_wds_preprocess_fn(
        enable_augmentation=enable_augmentation,
        augment_r=augment_r,
        augment_j=augment_j,
        base_seed=seed,
        metadata_key=metadata_key,
    )
    reject_fn = _make_inverse_freq_reject_fn(
        character_counts=character_counts,
        min_count=min_count,
        base_seed=seed,
        metadata_key=metadata_key,
    )
    secondary_mapper = _make_secondary_mapper(
        p_secondary=p_secondary,
        base_seed=seed,
        enable_secondary=enable_secondary,
        max_cache_size=secondary_cache_max_size,
    )

    if split == "train":
        shard_source = wds.ResampledShards(shards, seed=seed)
    else:
        shard_source = wds.SimpleShardList(shards)

    pipeline = [
        shard_source,
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode(_raw_pil_handler, "pil"),
        wds.map_dict(json=lambda b: json.loads(b) if isinstance(b, (bytes, bytearray, str)) else b),
        wds.to_tuple("jpg;jpeg;png;webp", "json"),
        wds.map(preprocess),
    ]

    if reject_using_manifest:
        pipeline.append(wds.select(reject_fn))

    pipeline.append(wds.map(secondary_mapper))

    if split == "train" and shuffle_buf and shuffle_buf > 0:
        pipeline.append(wds.shuffle(shuffle_buf, initial=shuffle_buf))

    pipeline.append(wds.batched(batch_size, collation_fn=identity_collate, partial=False))
    return wds.DataPipeline(*pipeline)


def create_dataloaders(config: TrainingConfig):
    train_ds = _build_wds_dataset(
        root=config.wds_root,
        split="train",
        metadata_key=config.metadata_key,
        batch_size=config.batch_size,
        shuffle_buf=config.wds_shuffle_buf,
        seed=config.seed,
        p_secondary=config.p_secondary,
        enable_secondary=True,
        secondary_cache_max_size=config.secondary_cache_max_size,
        enable_augmentation=config.enable_augmentation,
        augment_r=config.augment_r,
        augment_j=config.augment_j,
        reject_using_manifest=True,
    )
    train_loader = wds.WebLoader(
        train_ds,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(config.num_workers > 0),
    )

    val_loader = None
    if config.enable_validation:
        val_ds = _build_wds_dataset(
            root=config.wds_root,
            split="val",
            metadata_key=config.metadata_key,
            batch_size=config.batch_size,
            shuffle_buf=config.wds_shuffle_buf,
            seed=config.seed + 999,
            p_secondary=0.0,
            enable_secondary=False,
            secondary_cache_max_size=config.secondary_cache_max_size,
            enable_augmentation=False,
            augment_r=config.augment_r,
            augment_j=config.augment_j,
            reject_using_manifest=True,
        )
        val_loader = wds.WebLoader(
            val_ds,
            batch_size=None,
            num_workers=max(1, config.num_workers // 2),
            pin_memory=True,
            persistent_workers=(max(1, config.num_workers // 2) > 0),
        )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Distillation + DINO identity training for HunyuanVideo-1.5")
    parser.add_argument(
        "--pipeline_dir",
        type=str,
        required=True,
        help="Path containing the full pipeline (including text encoders and tokenizer).",
    )
    parser.add_argument("--transformer_version", type=str, default="480p_i2v", help="Transformer version to load")
    parser.add_argument(
        "--wds_root",
        type=str,
        required=True,
        help="WebDataset root directory containing data/ and manifest/.",
    )
    parser.add_argument(
        "--metadata_key",
        type=str,
        required=True,
        help="Metadata key in WebDataset JSON and manifest CSV for character identifier.",
    )
    parser.add_argument("--wds_shuffle_buf", type=int, default=128, help="WebDataset shuffle buffer size")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--p_secondary", type=float, default=0.10)
    parser.add_argument("--secondary_cache_max_size", type=int, default=128)
    parser.add_argument("--enable_augmentation", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--augment_r", type=float, default=0.92)
    parser.add_argument("--augment_j", type=float, default=0.04)
    parser.add_argument("--train_target_resolution", type=str, default="480p")
    parser.add_argument("--train_video_length", type=int, default=17)

    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--harvest_count", type=int, default=2)
    parser.add_argument("--harvest_scheme", type=str, default="bracket")
    parser.add_argument("--harvest_jitter_frac", type=float, default=0.05)
    parser.add_argument("--id_timestep_frac_low", type=float, default=0.6)
    parser.add_argument("--id_timestep_frac_high", type=float, default=0.85)
    parser.add_argument("--id_num_frames", type=int, default=5)
    parser.add_argument("--lambda_id", type=float, default=0.6)
    parser.add_argument("--lambda_id_schedule", type=str, default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--lambda_id_warmup_steps", type=int, default=500)
    parser.add_argument("--id_every_steps", type=int, default=1, help="Compute identity loss every N steps (default 1).")
    parser.add_argument(
        "--id_decode_downsample_mode",
        type=str,
        default="false",
        choices=["false", "area", "bicubic"],
        help="Downsample x0 latents in H/W before VAE decode for identity loss.",
    )
    parser.add_argument(
        "--id_decode_scale",
        type=float,
        default=0.5,
        help="Scale factor for H/W downsampling in identity decode (e.g., 0.5).",
    )

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_muon", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=500)

    parser.add_argument("--use_lora", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--pretrained_lora_path", type=str, default=None)

    parser.add_argument(
        "--dino_model_dir",
        type=str,
        required=True,
        help="Path to the local DINO backbone checkpoint directory.",
    )
    parser.add_argument(
        "--dino_head_path",
        type=str,
        required=True,
        help="Path to the projection head checkpoint for identity embeddings.",
    )

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--enable_validation", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--val_every_steps", type=int, default=1000)
    parser.add_argument("--val_num_samples", type=int, default=4)
    parser.add_argument("--val_video_length", type=int, default=60)

    parser.add_argument("--enable_fsdp", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--enable_gradient_checkpointing", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--sp_size", type=int, default=8)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig(
        pipeline_dir=args.pipeline_dir,
        transformer_version=args.transformer_version,
        wds_root=args.wds_root,
        wds_shuffle_buf=args.wds_shuffle_buf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        p_secondary=args.p_secondary,
        secondary_cache_max_size=args.secondary_cache_max_size,
        metadata_key=args.metadata_key,
        enable_augmentation=args.enable_augmentation,
        augment_r=args.augment_r,
        augment_j=args.augment_j,
        train_target_resolution=args.train_target_resolution,
        train_video_length=args.train_video_length,
        num_inference_steps=args.num_inference_steps,
        harvest_count=args.harvest_count,
        harvest_scheme=args.harvest_scheme,
        harvest_jitter_frac=args.harvest_jitter_frac,
        id_timestep_frac_low=args.id_timestep_frac_low,
        id_timestep_frac_high=args.id_timestep_frac_high,
        id_num_frames=args.id_num_frames,
        lambda_id=args.lambda_id,
        lambda_id_schedule=args.lambda_id_schedule,
        lambda_id_warmup_steps=args.lambda_id_warmup_steps,
        id_every_steps=args.id_every_steps,
        id_decode_downsample_mode=args.id_decode_downsample_mode,
        id_decode_scale=args.id_decode_scale,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_muon=args.use_muon,
        max_train_steps=args.max_train_steps,
        warmup_steps=args.warmup_steps,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.pretrained_lora_path,
        dino_model_dir=args.dino_model_dir,
        dino_head_path=args.dino_head_path,
        output_dir=args.output_dir,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        dtype=args.dtype,
        seed=args.seed,
        enable_validation=args.enable_validation,
        val_every_steps=args.val_every_steps,
        val_num_samples=args.val_num_samples,
        val_video_length=args.val_video_length,
        enable_fsdp=args.enable_fsdp,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        sp_size=args.sp_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer = HunyuanVideoTrainer(config)
    train_loader, val_loader = create_dataloaders(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
