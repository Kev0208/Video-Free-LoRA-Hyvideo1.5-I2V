"""
Export a LoRA adapter from a sharded DCP checkpoint.

This script:
- loads the base transformer from a local pipeline,
- injects LoRA modules with the provided config,
- loads DCP checkpoint weights into the transformer, and
- saves the LoRA adapter to disk in PEFT format.

Example usage:
  torchrun --nproc_per_node=1 export_lora_from_dcp.py \
    --model_path /your/path/to/hunyuanvideo_pipeline \
    --checkpoint_path /your/path/to/checkpoint\
    --output_dir ./outputs/lora_adapter \
    --transformer_version 480p_i2v \
    --lora_r 16 --lora_alpha 16
"""

import argparse
import os

import torch
import torch.distributed as dist
from loguru import logger
from peft import LoraConfig

from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)


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


def _resolve_dcp_path(path: str) -> str:
    if path is None:
        raise ValueError("checkpoint_path is required")
    if os.path.basename(path) == "dcp":
        return path
    dcp_path = os.path.join(path, "dcp")
    if os.path.exists(dcp_path):
        return dcp_path
    return path


def _init_dist():
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


def export_lora(args):
    _init_dist()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    transformer_path = os.path.join(args.model_path, "transformer", args.transformer_version)
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        transformer_path, low_cpu_mem_usage=True, torch_dtype=dtype
    ).to(device)

    target_modules = args.lora_target_modules or DEFAULT_LORA_TARGETS
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    transformer.add_adapter(lora_config, adapter_name="default")

    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint import FileSystemReader

    dcp_path = _resolve_dcp_path(args.checkpoint_path)
    reader = FileSystemReader(dcp_path)
    state = {"model": transformer}
    dcp_load(state, reader)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        transformer.save_lora_adapter(
            save_directory=args.output_dir,
            adapter_name="default",
            safe_serialization=True,
        )
        logger.info(f"Saved LoRA adapter to: {args.output_dir}")

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Export LoRA adapter from a DCP checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model directory")
    parser.add_argument(
        "--transformer_version", type=str, default="480p_i2v", help="Transformer version (default: 480p_i2v)"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint or checkpoint/dcp")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save LoRA adapter")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)

    args = parser.parse_args()
    export_lora(args)


if __name__ == "__main__":
    main()
