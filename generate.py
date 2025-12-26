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
HunyuanVideo-1.5 generation script (text-to-video or image-to-video).

This script loads a base pipeline and optionally applies:
- DCP checkpoints (full transformer weights), or
- a LoRA adapter checkpoint.

Example usage:
  Base model (t2v):
    torchrun --nproc_per_node=1 generate.py \
      --prompt "A subject running across a snowy field." \
      --resolution 480p \
      --model_path /your/path/to/hunyuanvideo_pipeline \
      --output_path ./outputs/base.mp4

  DCP checkpoint (i2v):
    torchrun --nproc_per_node=4 generate.py \
      --prompt "Same subject, moving in wind." \
      --image_path /your/path/to/ref.png \
      --resolution 480p \
      --model_path /your/path/to/hunyuanvideo_pipeline \
      --checkpoint_path /your/path/to/checkpoint\
      --output_path ./outputs/dcp.mp4

  LoRA adapter:
    torchrun --nproc_per_node=1 generate.py \
      --prompt "Stylized motion, soft lighting." \
      --resolution 480p \
      --model_path /your/path/to/hunyuanvideo_pipeline \
      --lora_path /your/path/to/lora_adapter \
      --lora_r 16 --lora_alpha 16 --lora_scale 0.7 \
      --output_path ./outputs/lora.mp4
"""

import os

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import datetime
import json

import loguru
import torch
import argparse
import einops
import imageio
from torch import distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

_ = initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

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

def save_video(video, path, fps=24):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=fps)

def rank0_log(message, level):
    if int(os.environ.get('RANK', '0')) == 0:
        loguru.logger.log(level, message)

def save_config(args, output_path, task, transformer_version):
    arguments = {}
    for key, value in vars(args).items():
        if not key.startswith('_') and not callable(value):
            try:
                json.dumps(value)
                arguments[key] = value
            except (TypeError, ValueError):
                arguments[key] = str(value)
    
    config = {
        'timestamp': datetime.datetime.now().isoformat(),
        'task': task,
        'transformer_version': transformer_version,
        'output_path': output_path,
        'arguments': arguments
    }
    
    base_path, _ = os.path.splitext(output_path)
    config_path = f"{base_path}_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    rank0_log(f"Saved generation config to: {config_path}", "INFO")
    return config_path

def str_to_bool(value):
    """Convert string to boolean, supporting true/false, 1/0, yes/no.
    If value is None (when flag is provided without value), returns True."""
    if value is None:
        return True  
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")

def _resolve_dcp_path(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        if os.path.exists(os.path.join(checkpoint_path, ".metadata")):
            return checkpoint_path
        dcp_dir = os.path.join(checkpoint_path, "dcp")
        if os.path.exists(os.path.join(dcp_dir, ".metadata")):
            return dcp_dir
    return checkpoint_path


def load_checkpoint_to_transformer(pipe, checkpoint_path):
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    checkpoint_path = _resolve_dcp_path(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Transformer checkpoint directory not found: {checkpoint_path}")
    
    rank0_log(f"Loading checkpoint from {checkpoint_path}", "INFO")
    
    try:
        dcp.load(
            state_dict={"model": pipe.transformer},
            checkpoint_id=checkpoint_path,
        )
        rank0_log("Transformer model state loaded successfully", "INFO")
    except Exception as e:
        rank0_log(
            f"Strict DCP load failed ({e}). Retrying with allow_partial_load=True.",
            "WARNING",
        )
        try:
            dcp.load(
                state_dict={"model": pipe.transformer},
                checkpoint_id=checkpoint_path,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
            rank0_log(
                "Transformer model state loaded with allow_partial_load=True. "
                "Missing keys were left at base model defaults.",
                "WARNING",
            )
        except Exception:
            rank0_log(f"Error loading checkpoint: {e}", "ERROR")
            raise

def load_lora_adapter(pipe, lora_path):
    rank0_log(f"Loading LoRA adapter from {lora_path}", "INFO")
    try:
        pipe.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=lora_path,
            prefix=None,
            adapter_name="default",
            use_safetensors=True,
            hotswap=False,
        )
        rank0_log("LoRA adapter loaded successfully", "INFO")
    except Exception as e:
        rank0_log(f"Error loading LoRA adapter: {e}", "ERROR")
        raise


def apply_lora_to_transformer(pipe, args):
    if not args.use_lora:
        return
    from peft import LoraConfig

    target_modules = args.lora_target_modules or DEFAULT_LORA_TARGETS
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    pipe.transformer.add_adapter(lora_config, adapter_name="default")


def apply_lora_scale(pipe, scale):
    if scale is None:
        return
    try:
        from peft.utils import other as peft_other
        if hasattr(peft_other, "scale_lora_layers"):
            peft_other.scale_lora_layers(pipe.transformer, scale)
            rank0_log(f"Applied LoRA scale via peft.utils.other.scale_lora_layers: {scale}", "INFO")
            return
    except Exception:
        pass

    try:
        from peft.tuners.lora import LoraLayer
    except Exception:
        LoraLayer = None

    def _set_layer_scale(layer, adapter_name="default"):
        scaling = getattr(layer, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            r = layer.r.get(adapter_name) if isinstance(getattr(layer, "r", None), dict) else getattr(layer, "r", None)
            alpha = (
                layer.lora_alpha.get(adapter_name)
                if isinstance(getattr(layer, "lora_alpha", None), dict)
                else getattr(layer, "lora_alpha", None)
            )
            if r:
                scaling[adapter_name] = float(alpha) / float(r) * float(scale)
            else:
                scaling[adapter_name] = float(scaling[adapter_name]) * float(scale)
            return 1
        if scaling is not None:
            try:
                layer.scaling = float(scaling) * float(scale)
                return 1
            except Exception:
                return 0
        return 0

    updated = 0
    for module in pipe.transformer.modules():
        if LoraLayer is not None and isinstance(module, LoraLayer):
            updated += _set_layer_scale(module)
            continue
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "scaling"):
            updated += _set_layer_scale(module)

    if updated == 0:
        rank0_log(
            "LoRA scale requested but no LoRA layers were updated; adapter scaling may be unsupported.",
            "WARNING",
        )
    

def generate_video(args):

    infer_state = initialize_infer_state(args)

    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn and use_sageattn cannot be enabled simultaneously. Please enable only one of them.")
    
    if args.enable_step_distill and args.enable_cache:
        raise ValueError("Enabling both step distilled model and cache will lead to performance degradation.")
    
    task = 'i2v' if args.image_path else 't2v'
    
    enable_sr = args.sr
    
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(args.resolution, task, args.cfg_distilled, args.enable_step_distill, args.sparse_attn)
    
    if args.dtype == 'bf16':
        transformer_dtype = torch.bfloat16
    elif args.dtype == 'fp32':
        transformer_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Must be 'bf16' or 'fp32'")
    
    enable_offloading = args.offloading
    if args.group_offloading is None:
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config['enable_group_offloading']
    else:
        enable_group_offloading = args.group_offloading
    
    overlap_group_offloading = args.overlap_group_offloading
    
    if enable_offloading:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    if enable_group_offloading:
        transformer_init_device = torch.device('cpu')
    else:
        transformer_init_device = device
    
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=enable_sr,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )

    apply_lora_to_transformer(pipe, args)
    
    loguru.logger.info(f"{enable_offloading=} {enable_group_offloading=} {overlap_group_offloading=}")
    
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=overlap_group_offloading,
    )
    
    if args.checkpoint_path:
        load_checkpoint_to_transformer(pipe, args.checkpoint_path)
    
    if args.lora_path:
        load_lora_adapter(pipe, args.lora_path)

    if args.lora_scale is not None and (args.use_lora or args.lora_path):
        apply_lora_scale(pipe, args.lora_scale)

    if enable_sr and hasattr(pipe, 'sr_pipeline'):
        sr_infer_state = copy.deepcopy(infer_state)
        sr_infer_state.enable_cache = False # SR pipeline does not require cache optimization yet
        pipe.sr_pipeline.apply_infer_optimization(
            infer_state=sr_infer_state,
            enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=overlap_group_offloading,
        )

    extra_kwargs = {}
    if task == 'i2v':
        extra_kwargs['reference_image'] = args.image_path

    if args.video_length != 121:
        rank0_log(f"Warning: 121 frames is the optimal value for best quality. "
                  f"Attempting to generate {args.video_length} frames...", "WARNING")

    enable_rewrite = args.rewrite
    if not args.rewrite:
        rank0_log("Warning: Prompt rewriting is disabled. This may affect the quality of generated videos.", "WARNING")

    out = pipe(
        enable_sr=enable_sr,
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.num_inference_steps,
        sr_num_inference_steps=None,
        video_length=args.video_length,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        output_type="pt",
        prompt_rewrite=enable_rewrite,
        return_pre_sr_video=args.save_pre_sr_video,
        **extra_kwargs,
    )
    
    if int(os.environ.get('RANK', '0')) == 0:
        output_path = args.output_path
        if output_path is None:
            now = f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}'
            output_path = f'./outputs/output_{transformer_version}_{now}.mp4'
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        original_path = None
        if enable_sr and hasattr(out, 'sr_videos'):
            save_video(out.sr_videos, output_path, fps=args.fps)
            rank0_log(f"Saved SR video to: {output_path}", "INFO")
            
            if args.save_pre_sr_video:
                base_path, ext = os.path.splitext(output_path)
                original_path = f"{base_path}_before_sr{ext}"
                save_video(out.videos, original_path, fps=args.fps)
                rank0_log(f"Saved original video (before SR) to: {original_path}", "INFO")
        else:
            save_video(out.videos, output_path, fps=args.fps)
            rank0_log(f"Saved video to: {output_path}", "INFO")
        
        if args.save_generation_config:
            try:
                save_config(args, output_path, task, transformer_version)
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanVideo-1.5')

    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for video generation (default: empty string)'
    )
    parser.add_argument(
        '--resolution', type=str, required=True, choices=['480p', '720p'],
        help='Video resolution (480p or 720p)'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--aspect_ratio', type=str, default='16:9',
        help='Aspect ratio (default: 16:9)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=None,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=None,
        help='Classifier-free guidance scale (default: model config).'
    )
    parser.add_argument(
        '--video_length', type=int, default=121,
        help='Total number of frames to generate (default: 121)'
    )
    parser.add_argument(
        '--num_frames', type=int, default=None,
        help='Alias for --video_length'
    )
    parser.add_argument(
        '--sr', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable super resolution (default: true). '
             'Use --sr or --sr true/1 to enable, --sr false/0 to disable'
    )
    parser.add_argument(
        '--save_pre_sr_video', type=str_to_bool, nargs='?', const=True, default=False,
        help='Save original video before super resolution (default: false). '
             'Use --save_pre_sr_video or --save_pre_sr_video true/1 to enable, '
             '--save_pre_sr_video false/0 to disable'
    )
    parser.add_argument(
        '--rewrite', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable prompt rewriting (default: false). '
             'Use --rewrite or --rewrite true/1 to enable, --rewrite false/0 to disable'
    )
    parser.add_argument(
        '--cfg_distilled', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable CFG distilled model (default: false). '
             'Use --cfg_distilled or --cfg_distilled true/1 to enable, '
             '--cfg_distilled false/0 to disable'
    )
    parser.add_argument(
        '--enable_step_distill', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable step distilled model (default: false). '
             'Use --enable_step_distill or --enable_step_distill true/1 to enable, '
             '--enable_step_distill false/0 to disable'
    )
    parser.add_argument(
        '--sparse_attn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sparse attention (default: false). '
             'Use --sparse_attn or --sparse_attn true/1 to enable, '
             '--sparse_attn false/0 to disable'
    )
    parser.add_argument(
        '--offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable offloading (default: true). '
             'Use --offloading or --offloading true/1 to enable, '
             '--offloading false/0 to disable'
    )
    parser.add_argument(
        '--group_offloading', type=str_to_bool, nargs='?', const=True, default=None,
        help='Enable group offloading (default: None, automatically enabled if offloading is enabled). '
             'Use --group_offloading or --group_offloading true/1 to enable, '
             '--group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--overlap_group_offloading', type=str_to_bool, nargs='?', const=True, default=True,
        help='Enable overlap group offloading (default: true). '
             'Significantly increases CPU memory usage but speeds up inference. '
             'Use --overlap_group_offloading or --overlap_group_offloading true/1 to enable, '
             '--overlap_group_offloading false/0 to disable'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for transformer (default: bf16). '
             'bf16: faster, lower memory; fp32: better quality, slower, higher memory'
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='Random seed (default: 123)'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided, uses i2v mode)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Output file path for generated video (if not provided, saves to ./outputs/output.mp4)'
    )
    parser.add_argument(
        '--fps', type=int, default=24,
        help='Frames per second for saved video (default: 24)'
    )
    parser.add_argument(
        '--use_sageattn', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable sageattn (default: false). '
             'Use --use_sageattn or --use_sageattn true/1 to enable, '
             '--use_sageattn false/0 to disable'
    )
    parser.add_argument(
        '--sage_blocks_range', type=str, default="0-53",
        help='Sageattn blocks range (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--enable_torch_compile', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable torch compile for transformer (default: false). '
             'Use --enable_torch_compile or --enable_torch_compile true/1 to enable, '
             '--enable_torch_compile false/0 to disable'
    )
    parser.add_argument(
        '--enable_cache', type=str_to_bool, nargs='?', const=True, default=False,
        help='Enable cache for transformer (default: false). '
             'Use --enable_cache or --enable_cache true/1 to enable, '
             '--enable_cache false/0 to disable'
    )
    parser.add_argument(
        '--cache_type', type=str, default="deepcache",
        help='Cache type for transformer (e.g., deepcache, teacache, taylorcache)'
    )
    parser.add_argument(
        '--no_cache_block_id', type=str, default="53",
        help='Blocks to exclude from deepcache (e.g., 0-5 or 0,1,2,3,4,5)'
    )
    parser.add_argument(
        '--cache_start_step', type=int, default=11,
        help='Start step to skip when using cache (default: 11)'
    )
    parser.add_argument(
        '--cache_end_step', type=int, default=45,
        help='End step to skip when using cache (default: 45)'
    )
    parser.add_argument(
        '--total_steps', type=int, default=50,
        help='Total inference steps (default: 50)'
    )
    parser.add_argument(
        '--cache_step_interval', type=int, default=4,
        help='Step interval to skip when using cache (default: 4)'
    )
    parser.add_argument(
        '--save_generation_config', type=str_to_bool, nargs='?', const=True, default=True,
        help='Save generation config file (default: true). '
             'Use --save_generation_config or --save_generation_config true/1 to enable, '
             '--save_generation_config false/0 to disable'
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Path to DCP checkpoint directory or its parent (e.g., ./outputs/checkpoint-1000 or ./outputs/checkpoint-1000/dcp). '
             'If provided, the transformer model weights will be loaded from this checkpoint.'
    )
    parser.add_argument(
        '--lora_path', type=str, default=None,
        help='Path to LoRA adapter directory or checkpoint directory containing LoRA adapter. '
             'If provided, the LoRA adapter will be loaded to the transformer model.'
    )
    parser.add_argument(
        '--use_lora', type=str_to_bool, nargs='?', const=True, default=False,
        help='Inject LoRA modules into the transformer before loading a DCP checkpoint '
             '(default: false). Enable this if your checkpoint was trained with LoRA.'
    )
    parser.add_argument(
        '--lora_r', type=int, default=16,
        help='LoRA rank (default: 8)'
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=16,
        help='LoRA alpha (default: 16)'
    )
    parser.add_argument(
        '--lora_dropout', type=float, default=0.0,
        help='LoRA dropout (default: 0.0)'
    )
    parser.add_argument(
        '--lora_target_modules', type=str, nargs='+', default=None,
        help='Target modules for LoRA injection. If omitted, uses the default list from training.'
    )
    parser.add_argument(
        '--lora_scale', type=float, default=0.5,
        help='LoRA adapter strength (default: 0.5). Requires --use_lora true or --lora_path.'
    )

    args = parser.parse_args()

    if args.num_frames is not None:
        if args.video_length != args.num_frames:
            rank0_log(
                f"Overriding --video_length ({args.video_length}) with --num_frames ({args.num_frames}).",
                "INFO",
            )
        args.video_length = args.num_frames

    if args.image_path is not None and args.image_path.lower().strip() == 'none':
        args.image_path = None
    
    
    generate_video(args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
