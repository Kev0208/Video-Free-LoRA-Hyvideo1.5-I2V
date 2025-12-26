"""
Train a projection head on top of a frozen DINOv3 backbone for identity embeddings.
These embeddings are later used by the main training script to compute identity loss.

This script streams WebDataset shards and uses a P*K sampler (pk-identities x
pk-instances) to build batches for supervised contrastive loss.

Minimum WebDataset layout:
  /your/wds_root/
    data/
      train/
        train-000000.tar
        train-000001.tar
      val/
        val-000000.tar
    manifest/
      train_manifest.csv
      val_manifest.csv

Each sample in a shard:
  - <sample_id>.<ext> (jpg/png/webp)
  - <sample_id>.json metadata with at least:
    { "<id_key>": <int> }

Manifest CSVs must include a column named <id_key> listing identity IDs
used for the P*K sampler. Use the same <id_key> in JSON and manifest via --id-key.
Ensure each validation identity has at least pk-instances samples.

Example:
  accelerate launch identity_loss/train_projection_head.py \
    --model-dir /your/path/to/dinov3 \
    --wds-root /your/path/to/identity_wds \
    --output-dir /your/path/to/output \
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
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from accelerate import Accelerator
from accelerate.utils import set_seed, tqdm
from braceexpand import braceexpand
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel


def load_manifest_ids(manifest_path: Path, id_key: str) -> Optional[List[int]]:
    if not manifest_path.is_file():
        return None

    ids: set[int] = set()
    key = id_key.strip().lower()
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lowered = {k.lower(): v for k, v in row.items() if k}
            value = lowered.get(key)
            if value is None:
                continue
            try:
                identity_id = int(value)
            except ValueError:
                continue
            ids.add(identity_id)

    return sorted(ids) if ids else None


def parse_metadata(meta_blob) -> Dict:
    if isinstance(meta_blob, dict):
        return meta_blob
    if isinstance(meta_blob, bytes):
        meta_blob = meta_blob.decode("utf-8")
    if isinstance(meta_blob, str):
        return json.loads(meta_blob)
    raise TypeError(f"Unsupported metadata type: {type(meta_blob)}")


class RandomLongSideJitter:
    def __init__(self, keep_ratio: float = 0.9, jitter: float = 0.06):
        self.keep_ratio = keep_ratio
        self.jitter = jitter

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        rng = random.random

        if width >= height:
            crop_w = max(1, int(round(width * self.keep_ratio)))
            jitter_span = int(round(self.jitter * crop_w))
            dx = rng() * (2 * jitter_span) - jitter_span
            center_x = width / 2.0 + dx
            x0 = int(round(center_x - crop_w / 2.0))
            x0 = max(0, min(x0, width - crop_w))
            box = (x0, 0, x0 + crop_w, height)
        else:
            crop_h = max(1, int(round(height * self.keep_ratio)))
            jitter_span = int(round(self.jitter * crop_h))
            dy = rng() * (2 * jitter_span) - jitter_span
            center_y = height / 2.0 + dy
            y0 = int(round(center_y - crop_h / 2.0))
            y0 = max(0, min(y0, height - crop_h))
            box = (0, y0, width, y0 + crop_h)

        return image.crop(box)


def preprocess_sample(sample, cropper: Optional[RandomLongSideJitter], id_key: str) -> Dict:
    image, meta = sample
    label = meta.get(id_key)
    if label is None:
        raise KeyError(f"Metadata missing `{id_key}`.")

    if not isinstance(image, Image.Image):
        raise ValueError("Decoded sample does not contain a PIL image.")

    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and ("transparency" in image.info):
        image = image.convert("RGBA")

    image = image.convert("RGB")
    if cropper is not None:
        image = cropper(image)

    return {"image": image, "label": int(label)}


def build_webdataset(
    root: Path,
    split: str,
    shard_pattern: str,
    *,
    training: bool,
    shuffle_buffer: int,
    seed: int,
    cropper: Optional[RandomLongSideJitter],
    id_key: str,
    handler=wds.handlers.warn_and_continue,
) -> wds.DataPipeline:
    data_dir = root / "data" / split
    pattern = str(data_dir / shard_pattern)

    shards = list(braceexpand(pattern)) if "{" in pattern else sorted(
        str(p) for p in data_dir.glob(shard_pattern)
    )
    if not shards:
        raise FileNotFoundError(f"No shards found for pattern: {pattern}")

    if training:
        shard_source = wds.ResampledShards(shards, seed=seed)
    else:
        shard_source = wds.SimpleShardList(shards)

    pipeline = [
        shard_source,
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=handler),
    ]
    if training and shuffle_buffer > 0:
        pipeline.append(wds.shuffle(shuffle_buffer, initial=shuffle_buffer))
    pipeline.extend(
        [
            wds.decode("pil"),
            wds.map_dict(json=parse_metadata),
            wds.to_tuple("jpg;jpeg;png;webp", "json"),
            wds.map(lambda sample: preprocess_sample(sample, cropper, id_key)),
        ]
    )

    return wds.DataPipeline(*pipeline)


@dataclass
class PKConfig:
    identities: int
    instances: int
    max_per_identity: int


class PKBatchBuilder:
    def __init__(
        self,
        config: PKConfig,
        allowed_ids: Optional[Sequence[int]] = None,
        seed: int = 0,
    ):
        self.config = config
        self.allowed = set(allowed_ids) if allowed_ids is not None else None
        if self.config.max_per_identity < self.config.instances:
            raise ValueError("max_per_identity must be >= pk_instances.")
        self.buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.config.max_per_identity))
        self.ready_ids: set[int] = set()
        self.rng = random.Random(seed)

    def reset(self):
        self.buffers.clear()
        self.ready_ids.clear()

    def add(self, sample: Dict) -> List[List[Dict]]:
        label = int(sample["label"])
        if self.allowed is not None and label not in self.allowed:
            return []

        buf = self.buffers[label]
        if len(buf) >= self.config.max_per_identity:
            buf.popleft()
        buf.append(sample)

        if len(buf) >= self.config.instances:
            self.ready_ids.add(label)
        elif label in self.ready_ids and len(buf) < self.config.instances:
            self.ready_ids.discard(label)

        collected: List[List[Dict]] = []
        while len(self.ready_ids) >= self.config.identities:
            chosen = self.rng.sample(list(self.ready_ids), self.config.identities)
            batch: List[Dict] = []
            for cid in chosen:
                buf = self.buffers[cid]
                if len(buf) < self.config.instances:
                    self.ready_ids.discard(cid)
                    continue
                idxs = self.rng.sample(range(len(buf)), self.config.instances)
                idx_set = set(idxs)
                buf_list = list(buf)
                batch.extend(buf_list[idx] for idx in idxs)
                remaining = [s for idx, s in enumerate(buf_list) if idx not in idx_set]
                self.buffers[cid] = deque(remaining, maxlen=self.config.max_per_identity)
                if len(self.buffers[cid]) < self.config.instances:
                    self.ready_ids.discard(cid)
            collected.append(batch)

        return collected


class PKBatchStream:
    def __init__(self, dataset: wds.DataPipeline, builder: PKBatchBuilder, processor: AutoImageProcessor):
        self.dataset = dataset
        self.builder = builder
        self.processor = processor
        self.iterator: Optional[Iterator] = None
        self.pending: deque[List[Dict]] = deque()

    def reset(self):
        self.builder.reset()
        self.pending.clear()
        self.iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.pending:
            return collate_batch(self.pending.popleft(), self.processor)

        if self.iterator is None:
            self.iterator = iter(self.dataset)

        while True:
            try:
                sample = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                continue

            ready = self.builder.add(sample)
            if not ready:
                continue
            self.pending.extend(ready)
            return collate_batch(self.pending.popleft(), self.processor)


def collate_batch(samples: List[Dict], processor: AutoImageProcessor) -> Dict[str, torch.Tensor]:
    images = [entry["image"] for entry in samples]
    labels = torch.tensor([entry["label"] for entry in samples], dtype=torch.long)
    encoded = processor(images=images, return_tensors="pt")
    encoded["labels"] = labels
    return encoded


class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if activation.lower() == "gelu":
            act_layer = nn.GELU()
        elif activation.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        prev_dim = in_dim
        total_layers = len(hidden_dims)
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            is_last = idx == (total_layers - 1)
            if not is_last:
                if use_layernorm:
                    layers.append(nn.LayerNorm(dim))
                layers.append(act_layer)
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityProjectionModel(nn.Module):
    def __init__(self, backbone: AutoModel, head: ProjectionHead):
        super().__init__()
        self.backbone = backbone
        self.projection_head = head
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        dtype = next(self.projection_head.parameters()).dtype
        outputs = self.backbone(pixel_values=pixel_values.to(self.backbone.dtype))
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = pooled.to(dtype)
        projected = self.projection_head(pooled)
        normalized = F.normalize(projected, dim=-1)
        return {"projected": projected, "normalized": normalized}


def supervised_contrastive_loss_ddp(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    accelerator: Accelerator,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    features = F.normalize(embeddings, dim=-1)
    labels = labels.view(-1)

    B = features.size(0)
    rank = accelerator.process_index

    all_features = accelerator.gather(features.detach())
    all_labels = accelerator.gather(labels)

    logits = (features @ all_features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    mask = torch.ones_like(logits, dtype=torch.bool)
    self_idx = rank * B + torch.arange(B, device=logits.device)
    mask[torch.arange(B, device=logits.device), self_idx] = False

    positives = (labels.unsqueeze(1) == all_labels.unsqueeze(0)) & mask
    pos_counts = positives.sum(dim=1)

    exp_logits = torch.exp(logits) * mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    mean_log_prob_pos = (positives.float() * log_prob).sum(dim=1) / pos_counts.clamp(min=1)

    valid = pos_counts > 0
    if not torch.any(valid):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    return (-mean_log_prob_pos[valid]).mean()


def supervised_contrastive_loss_local(embeddings, labels, temperature=0.1, eps=1e-8):
    features = F.normalize(embeddings, dim=-1)
    labels = labels.view(-1)

    logits = (features @ features.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    logits_mask = torch.ones_like(logits, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)

    positives = (labels[:, None] == labels[None, :]) & logits_mask
    pos_counts = positives.sum(dim=1)

    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    mean_log_prob_pos = (positives.float() * log_prob).sum(dim=1) / pos_counts.clamp(min=1)
    valid = pos_counts > 0
    if not valid.any():
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    return -(mean_log_prob_pos[valid]).mean()


def compute_cosine_stats(embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    n = embeddings.shape[0]
    if n < 2:
        return {}
    normalized = F.normalize(embeddings, dim=-1)
    sims = normalized @ normalized.t()
    mask = torch.eye(n, dtype=torch.bool, device=sims.device)
    sims = sims.masked_fill(mask, 0.0)
    labels = labels.view(-1)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~mask)
    diff = (~same) & (~mask)

    stats: Dict[str, float] = {}
    if same.any():
        pos_vals = sims[same]
        stats["pos_mean"] = float(pos_vals.mean().item())
        stats["pos_median"] = float(pos_vals.median().item())
    if diff.any():
        neg_vals = sims[diff]
        stats["neg_mean"] = float(neg_vals.mean().item())
        stats["neg_median"] = float(neg_vals.median().item())
    return stats


def move_batch_to_device(batch: Dict[str, torch.Tensor], accelerator: Accelerator) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(accelerator.device, non_blocking=True)
        else:
            out[key] = value
    return out


def evaluate(
    model: IdentityProjectionModel,
    batch_stream: PKBatchStream,
    args,
    accelerator: Accelerator,
) -> Dict[str, float]:
    assert accelerator.is_main_process
    model = accelerator.unwrap_model(model)

    model.eval()
    batch_stream.reset()

    pos_mean_list, pos_median_list = [], []
    neg_mean_list, neg_median_list = [], []
    losses: List[float] = []

    for _ in range(args.val_batches):
        batch = next(batch_stream)
        batch = move_batch_to_device(batch, accelerator)
        with torch.no_grad():
            outputs = model(batch["pixel_values"])
            loss = supervised_contrastive_loss_local(
                outputs["projected"],
                batch["labels"],
                temperature=args.temperature,
            )
        losses.append(float(loss.detach().cpu().item()))
        stats = compute_cosine_stats(
            outputs["normalized"].detach().cpu(),
            batch["labels"].detach().cpu(),
        )
        if "pos_mean" in stats:
            pos_mean_list.append(stats["pos_mean"])
        if "pos_median" in stats:
            pos_median_list.append(stats["pos_median"])
        if "neg_mean" in stats:
            neg_mean_list.append(stats["neg_mean"])
        if "neg_median" in stats:
            neg_median_list.append(stats["neg_median"])

    return {
        "val_loss": float(mean(losses)) if losses else 0.0,
        "pos_cos_mean": float(mean(pos_mean_list)) if pos_mean_list else 0.0,
        "pos_cos_median": float(mean(pos_median_list)) if pos_median_list else 0.0,
        "neg_cos_mean": float(mean(neg_mean_list)) if neg_mean_list else 0.0,
        "neg_cos_median": float(mean(neg_median_list)) if neg_mean_list else 0.0,
    }


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    accelerator: Accelerator,
    model: IdentityProjectionModel,
    optimizer: torch.optim.Optimizer,
    args,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    ckpt = {
        "step": step,
        "head_state_dict": unwrapped.projection_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(args),
    }
    save_path = output_dir / f"projection_head_step{step}.pt"
    torch.save(ckpt, save_path)
    accelerator.print(f"[checkpoint] Saved projection head to {save_path}")


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a projection head on top of DINOv3.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the DINOv3 model directory.")
    parser.add_argument("--wds-root", type=str, required=True, help="WebDataset root containing data/ and manifest/.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save projection head checkpoints.")
    parser.add_argument(
        "--id-key",
        type=str,
        required=True,
        help="Metadata/manifest key containing identity IDs (used for P*K sampling).",
    )
    parser.add_argument("--train-manifest", type=str, default="manifest/train_manifest.csv")
    parser.add_argument("--val-manifest", type=str, default="manifest/val_manifest.csv")
    parser.add_argument("--train-shards", type=str, default="train-*.tar")
    parser.add_argument("--val-shards", type=str, default="val-*.tar")
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--pk-identities", type=int, default=16)
    parser.add_argument("--pk-instances", type=int, default=4)
    parser.add_argument("--max-per-identity", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu"])
    parser.add_argument("--use-layernorm", action="store_true", default=True)
    parser.add_argument("--no-layernorm", dest="use_layernorm", action="store_false")
    parser.add_argument("--proj-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--crop-ratio", type=float, default=0.9)
    parser.add_argument("--crop-jitter", type=float, default=0.06)
    parser.add_argument("--val-crop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)

    process_seed = args.seed + accelerator.process_index
    set_seed(process_seed)
    accelerator.print(f"Running with {accelerator.num_processes} processes.")

    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model_dtype = parse_dtype(args.model_dtype)
    backbone = AutoModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=model_dtype,
    )

    proj_head = ProjectionHead(
        in_dim=backbone.config.hidden_size,
        hidden_dims=args.proj_dims,
        activation=args.activation,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
    )
    model = IdentityProjectionModel(backbone, proj_head)

    optimizer = torch.optim.AdamW(
        model.projection_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.total_steps, args.warmup_ratio)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    root = Path(args.wds_root)
    train_manifest = root / args.train_manifest
    val_manifest = root / args.val_manifest

    train_ids = load_manifest_ids(train_manifest, args.id_key)
    val_ids = load_manifest_ids(val_manifest, args.id_key)

    accelerator.print(
        f"Loaded {len(train_ids) if train_ids else 'N/A'} train identities and "
        f"{len(val_ids) if val_ids else 'N/A'} val identities from manifests."
    )

    if args.val_every > 0:
        if val_ids is None:
            raise RuntimeError(
                "Validation is enabled but no validation manifest was found. "
                "Provide manifest/val_manifest.csv or disable validation via --val-every 0."
            )
        if len(val_ids) < args.pk_identities:
            raise ValueError(
                f"Validation manifest only contains {len(val_ids)} identities but pk-identities={args.pk_identities}."
            )

    cropper = RandomLongSideJitter(args.crop_ratio, args.crop_jitter)
    val_cropper = cropper if args.val_crop else None

    train_dataset = build_webdataset(
        root,
        "train",
        args.train_shards,
        training=True,
        shuffle_buffer=args.shuffle_buffer,
        seed=process_seed,
        cropper=cropper,
        id_key=args.id_key,
    )
    val_dataset = build_webdataset(
        root,
        "val",
        args.val_shards,
        training=False,
        shuffle_buffer=args.shuffle_buffer,
        seed=process_seed + 1,
        cropper=val_cropper,
        id_key=args.id_key,
    )

    pk_config = PKConfig(
        identities=args.pk_identities,
        instances=args.pk_instances,
        max_per_identity=args.max_per_identity,
    )
    train_builder = PKBatchBuilder(pk_config, allowed_ids=train_ids, seed=process_seed)
    val_builder = PKBatchBuilder(pk_config, allowed_ids=val_ids, seed=process_seed + 1000)

    train_stream = PKBatchStream(train_dataset, train_builder, processor)
    val_stream = PKBatchStream(val_dataset, val_builder, processor)

    global_step = 0
    train_stream.reset()
    model.train()

    accelerator.print(f"Training for {args.total_steps} optimizer steps.")
    progress_bar = tqdm(total=args.total_steps, initial=0, disable=not accelerator.is_main_process)
    progress_bar.set_description("training")

    while global_step < args.total_steps:
        batch = next(train_stream)
        batch = move_batch_to_device(batch, accelerator)

        with accelerator.accumulate(model):
            outputs = model(batch["pixel_values"])
            loss = supervised_contrastive_loss_ddp(
                outputs["projected"],
                batch["labels"],
                accelerator=accelerator,
                temperature=args.temperature,
            )
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

            if global_step % args.log_every == 0:
                accelerator.print(
                    f"[step {global_step}] loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(accelerator, model, optimizer, args, global_step)

            if args.val_every > 0 and global_step % args.val_every == 0:
                accelerator.wait_for_everyone()
                metrics = None
                if accelerator.is_main_process:
                    metrics = evaluate(model, val_stream, args, accelerator)
                accelerator.wait_for_everyone()
                model.train()
                if accelerator.is_main_process:
                    accelerator.print(
                        f"[val] step={global_step} loss={metrics['val_loss']:.4f} "
                        f"pos_mean={metrics['pos_cos_mean']:.4f} pos_median={metrics['pos_cos_median']:.4f} "
                        f"neg_mean={metrics['neg_cos_mean']:.4f} neg_median={metrics['neg_cos_median']:.4f}"
                    )

    progress_bar.close()

    if args.val_every > 0 and (args.total_steps % args.val_every) != 0:
        accelerator.wait_for_everyone()
        metrics = None
        if accelerator.is_main_process:
            metrics = evaluate(model, val_stream, args, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print(
                f"[val] step={global_step} loss={metrics['val_loss']:.4f} "
                f"pos_mean={metrics['pos_cos_mean']:.4f} pos_median={metrics['pos_cos_median']:.4f} "
                f"neg_mean={metrics['neg_cos_mean']:.4f} neg_median={metrics['neg_cos_median']:.4f}"
            )

    save_checkpoint(accelerator, model, optimizer, args, global_step)

    accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
