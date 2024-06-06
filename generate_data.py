'''
Copyright (c) Haowei Zhu, 2024
'''

import argparse
import copy
import gc
import importlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from huggingface_hub import create_repo, model_info, upload_folder
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.utils.torch_utils import randn_tensor
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from tqdm import tqdm as tqdm_bar
from dataloader import StandardDataLoader, SDDataset, extract_prototypes_with_encoder

import copy
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torchvision.utils import save_image, _log_api_usage_once, make_grid
from torch import nn
from torch.utils.checkpoint import checkpoint
from datetime import datetime
import random
import json
from PIL import Image
import time
import open_clip
from torchvision.transforms.functional import to_pil_image
from torch.autograd import Variable
from model_utils import create_model

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def vae_tensor_to_pil_image(tensor, denormalize=False):
    ims = []
    bs = tensor.shape[0]
    if denormalize:
        tensor = (tensor / 2 + 0.5).clamp(0, 1)

    for i in range(bs):
        grid = make_grid(tensor[i: i+1])
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        ims.append(im)
    return ims


def simple_preprocess(pil_images):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor_list = [transform(image) for image in pil_images]
    batch_tensor = torch.stack(tensor_list, dim=0)

    return batch_tensor



def denoise_one_step(latents, noise_scheduler, t, unet, prompt_embeds, class_labels):
    latent_model_input = torch.cat([latents] * 2) if args.do_classifier_free_guidance else latents
    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
    noise_pred = unet(latent_model_input, t, prompt_embeds, class_labels=class_labels, return_dict=False)[0]

    # classifier free guidance
    if args.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

    ddim_output = noise_scheduler.step(noise_pred, t, latents, return_dict=True)
    latents, x_0 = ddim_output["prev_sample"], ddim_output["pred_original_sample"]
    return latents, x_0


def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]
    return res


def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    # elif model_class == "RobertaSeriesModelWithTransformation":
    #     from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default='caltech-101',
        help=(
            "The name of the downstream dataset."
        ),
    )
    parser.add_argument(
        "--arch", "-a",
        type=str,
        default='open_clip_vit_b32',
        help=(
            "image encoder."
        ),
    )

    parser.add_argument(
        "--encoder_weight_path",
        type=str,
        default=None,
        help=(
            "image encoder weight path."
        ),
    )
    parser.add_argument(
        "--guidance_type",
        default=None,
        help="Training free guidance type: transform_guidance.",
    )
    parser.add_argument('--constraint_value', default=0.8, type=float)
    parser.add_argument('--steps', default=50, type=int)
    parser.add_argument('--K', default=3, type=int, help='number of local prototypes per class.')
    parser.add_argument('--guidance_step', default=1, type=int, help='index for start guidance.')
    parser.add_argument('--guidance_period', default=1, type=int, help='guidance period.')
    parser.add_argument('--total_split', default=8, type=int)
    parser.add_argument('--split', default=0, type=int, help='Dividing classes into 5 parts, the index of which parts')
    parser.add_argument('--num_images_per_prompt', default=4, type=int, help='generate image per prompt')
    parser.add_argument('--first_image_index', default=0, type=int,
                        help='first generated image_id, resume from this id.')
    parser.add_argument('--optimize_targets', default=None, type=str,
                        # choices=['global_prototype', 'local_prototype'],
                        help='global_prototype, local_prototype')
    parser.add_argument("--rho", type=float, default=10.0, help="learning rate for optimization")
    parser.add_argument("--gs", type=float, default=1.0, help="scale for global score.")
    parser.add_argument("--ls", type=float, default=1.0, help="scale for local score.")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.9,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--language_enhance", "-le",
        action="store_true",
        help="whether to use language enhancement.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='poloclub/diffusiondb',
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default='large_random_1k',
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="prompt",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="dog",
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of sks dog",
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )

    parser.add_argument(
        "--text_to_img",
        default=False,
        action="store_true",
        help="Use text prompts to generate image.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_expand",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=1, help="Batch size (per device) for the val dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=400,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        # default=50,
        help="Classifier free guidance scale.",
    )

    parser.add_argument(
        "--do_classifier_free_guidance",
        type=bool,
        default=True,
        help="Whether or not to do classifier free guidance scale.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--skip_save_text_encoder", action="store_true", required=False, help="Set to not save text encoder"
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args


def collate_fn(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    uncond_inputs_ids = [example["uncond_prompt_ids"] for example in examples]
    image_latents = [example["image_latents"] for example in examples]

    targets = [example["targets"] for example in examples]
    class_names = [example["class_names"] for example in examples]
    image_paths = [example["image_paths"] for example in examples]
    pil_images = [example["pil_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]
        uncond_attention_mask = [example["uncond_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    uncond_inputs_ids = torch.cat(uncond_inputs_ids, dim=0)
    image_latents = torch.cat(image_latents, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "uncond_inputs_ids": uncond_inputs_ids,
        "class_names": class_names,
        "targets": targets,
        "image_paths": image_paths,
        "image_latents": image_latents,
        "pil_images": pil_images,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

        uncond_attention_mask = torch.cat(uncond_attention_mask, dim=0)
        batch["uncond_attention_mask"] = uncond_attention_mask

    return batch


def transform_guidance(latents, batch, sub_timesteps, noise_scheduler, unet, prompt_embeds, class_labels,
                       vae, image_encoder, image_processor, weight_dtype, generator,
                       total_global_proto, total_local_proto):
    bs = latents.shape[0]
    channel_noise_dim = latents.shape[1]
    channel_noise = Variable(torch.rand([bs, channel_noise_dim, 1, 1]), requires_grad=True).cuda().to(
        dtype=weight_dtype)
    channel_noise_bias = Variable(torch.zeros([bs, channel_noise_dim, 1, 1]).data.normal_(0, 1),
                                  requires_grad=True).cuda().to(dtype=weight_dtype)
    x_dec_noisy = latents * (1 + channel_noise) + channel_noise_bias

    score = 0.
    for temp_t in sub_timesteps:
        x_dec_noisy, pred_x0 = denoise_one_step(x_dec_noisy, noise_scheduler, temp_t, unet, prompt_embeds, class_labels)
        D_x0_t = vae.decode(pred_x0 / vae.config.scaling_factor, return_dict=False, generator=generator)[0]

        D_x0_t = image_processor.postprocess(D_x0_t, output_type="pt", do_denormalize=[False] * D_x0_t.shape[0])
        D_x0_t = torch.nn.functional.interpolate(D_x0_t, size=(224, 224), mode='bicubic')
        image_features = image_encoder.encode_image(D_x0_t).float()

        if total_global_proto is not None:
            global_proto = total_global_proto[batch["targets"]]
            global_distance = torch.norm(image_features - global_proto.detach(), dim=1, p=2).mean()
            score += global_distance * args.gs

        if total_local_proto is not None:
            local_proto = total_local_proto[batch["targets"]]
            target_cluster_index = torch.argmax(torch.bmm(image_features.unsqueeze(1), local_proto.permute(0, 2, 1)), -1)
            local_proto = local_proto[torch.arange(local_proto.size(0)), target_cluster_index.squeeze()]
            local_distance = torch.norm(image_features - local_proto.detach(), dim=1, p=2).mean()
            score += local_distance * args.ls

    score = score / args.guidance_period

    (channel_noise_grad, channel_noise_bias_grad) = torch.autograd.grad(score, [channel_noise, channel_noise_bias])

    channel_noise.data.add_(-args.rho * channel_noise_grad)
    channel_noise_bias.data.add_(-args.rho * channel_noise_bias_grad)

    x_dec_temp = latents.clone()
    latents = latents * (1 + channel_noise) + channel_noise_bias
    linfball_proj(x_dec_temp, args.constraint_value, latents, in_place=True)

    del channel_noise, channel_noise_bias, x_dec_temp
    latents = latents.detach()
    return latents, score


def direct_guidance(latents, batch, t_i, noise_scheduler, unet, prompt_embeds, class_labels,
                    vae, image_encoder, image_processor, weight_dtype, generator,
                    total_global_proto, total_local_proto):
    score = 0.
    latents.requires_grad = True

    x_dec_next, x_0 = denoise_one_step(latents, noise_scheduler, t_i, unet, prompt_embeds, class_labels)

    D_x0_t = vae.decode(x_0 / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
    D_x0_t = image_processor.postprocess(D_x0_t, output_type="pt", do_denormalize=[False] * D_x0_t.shape[0])
    D_x0_t = torch.nn.functional.interpolate(D_x0_t, size=(224, 224), mode='bicubic')
    image_features = image_encoder.encode_image(D_x0_t).float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    if total_global_proto is not None:
        global_proto = total_global_proto[batch["targets"]]
        global_distance = torch.norm(image_features - global_proto.detach(), dim=1, p=2).mean()
        score += global_distance * args.gs

    if total_local_proto is not None:
        local_proto = total_local_proto[batch["targets"]]
        target_cluster_index = torch.argmax(torch.bmm(image_features.unsqueeze(1), local_proto.permute(0, 2, 1)), -1)
        local_proto = local_proto[torch.arange(local_proto.size(0)), target_cluster_index.squeeze()]
        local_distance = torch.norm(image_features - local_proto.detach(), dim=1, p=2).mean()
        score += local_distance * args.ls

    x_dec_grad = torch.autograd.grad(score, latents)[0]
    x_dec_next = x_dec_next - args.rho * x_dec_grad
    latents = x_dec_next

    x_0 = x_0.detach()
    latents = latents.detach()
    return latents, x_0, score


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None,
                  negative_input_ids=None, negative_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    # print(text_input_ids.shape)
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    # get unconditional embeddings for classifier free guidance
    negative_prompt_embeds = text_encoder(
        negative_input_ids.to(text_encoder.device),
        attention_mask=negative_attention_mask,
        return_dict=False
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    return prompt_embeds, negative_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.logging_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    cache_dir=args.cache_dir)

    # torch_dtype = torch.float32
    torch_dtype = torch.float16
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        revision=args.revision,
        cache_dir=args.cache_dir,
        variant=args.variant
    ).to(accelerator.device)
    image_processor = copy.deepcopy(pipeline.image_processor)
    del pipeline

    if accelerator.is_main_process:
        # # 获取当前时间
        # now = datetime.now()
        # # 格式化时间为字符串，例如：2023-03-29_14-13-52
        # time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        if args.output_dir is not None:
            # args.output_dir = f"{args.output_dir}_scale_{args.guidance_scale}_{args.num_images_per_prompt}x/"
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.logging_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            cache_dir=args.cache_dir,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )

    try:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
            revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model, type(unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if vae is not None:
        vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(f"Unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}")

    if args.train_text_encoder and unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {unwrap_model(text_encoder).dtype}." f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    original_train_dataset = SDDataset(args, tokenizer, text_encoder, vae, size=512, center_crop=False)
    total_data_number = len(original_train_dataset)
    number_per_split = math.ceil(total_data_number / args.total_split)
    if args.split == (args.total_split - 1) and total_data_number < number_per_split * (args.split + 1):
        mask = list(range(number_per_split * args.split, total_data_number))
    else:
        mask = list(range(number_per_split * args.split, number_per_split * (args.split + 1)))

    train_dataset = torch.utils.data.Subset(original_train_dataset, mask)

    data_number = len(train_dataset)
    # indices = np.arange(0, data_number)
    # expanded_number_per_sample = args.expanded_number_per_sample
    # expanded_number = expanded_number_per_sample * data_number

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )

    # # Scheduler and math around the number of training steps.
    # overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    weight_dtype = torch.float16
    # weight_dtype = torch.float32

    # 4. Prepare timesteps
    num_inference_steps = 50
    timesteps, num_inference_steps = retrieve_timesteps(noise_scheduler, num_inference_steps, None)

    if unet is not None:
        unet.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable(use_reentrant=False)

    # Prepare everything with our `accelerator`.
    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)
    # Move vae and text_encoder to device and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    if not args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if unet is not None:
        unet.to(accelerator.device, dtype=weight_dtype)

    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth", config=tracker_config)

    # Train!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    progress_bar = tqdm(
        range(0, len(train_dataloader) * args.num_images_per_prompt),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    image_encoder = create_model(args.arch, pretrained=False,
                                 num_classes=len(original_train_dataset.class_names),
                                 class_names=original_train_dataset.class_names,
                                 cache_dir=args.cache_dir, dataset_name=args.dataset, weight_path=args.encoder_weight_path).cuda()
    global_prototypes_numpy, local_prototypes_numpy = extract_prototypes_with_encoder(args, image_encoder)

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.optimize_targets is not None:
        assert os.path.exists(args.encoder_weight_path)
        args.optimize_targets = args.optimize_targets.split("-")

    print(f"optimize strategy: {args.guidance_type}, target: {args.optimize_targets}, learning rate: {args.rho}")

    if args.optimize_targets is not None:
        if "global_prototype" in args.optimize_targets:
            total_global_proto = torch.from_numpy(global_prototypes_numpy).cuda()
            total_global_proto = total_global_proto / total_global_proto.norm(dim=-1, keepdim=True)
        else:
            total_global_proto = None

        if "local_prototype" in args.optimize_targets:
            total_local_proto = torch.from_numpy(local_prototypes_numpy).cuda()
            total_local_proto = total_local_proto / total_local_proto.norm(dim=-1, keepdim=True)
            print("local prototype shape:", total_local_proto.shape)
        else:
            total_local_proto = None
    else:
        total_global_proto = total_local_proto = None

    unet.eval()
    for step, batch in enumerate(train_dataloader):
        for image_i in range(args.first_image_index, args.num_images_per_prompt):
            batch_image_exists = True
            for i in range(len(batch["image_paths"])):
                image_file_path = os.path.basename(batch["image_paths"][i]).split(".")[0]
                path = f'{args.output_dir}/{batch["class_names"][i]}/{image_file_path}_expand_{image_i}.png'
                if not os.path.exists(path):
                    batch_image_exists = False
                else:
                    print(f"File {path} exists, so skipped.")
            if batch_image_exists:
                progress_bar.update(1)
                global_step += 1
                continue

            with (accelerator.accumulate([unet])):
                # Get the text embedding for conditioning
                prompt_embeds = batch["input_ids"].to(dtype=weight_dtype)
                negative_prompt_embeds = batch["uncond_inputs_ids"].to(dtype=weight_dtype)

                if args.text_to_img:
                    batch_size = prompt_embeds.shape[0]
                    num_channels_latents = unet.config.in_channels
                    shape = (batch_size, num_channels_latents, unet.sample_size, unet.sample_size)

                    latents = randn_tensor(shape, generator=generator, device=unet.device, dtype=weight_dtype)

                    # scale the initial noise by the standard deviation required by the scheduler
                    noisy_model_input = latents * noise_scheduler.init_noise_sigma

                else:
                    model_input = batch["image_latents"].to(dtype=weight_dtype)

                    # Sample noise that we'll add to the model input
                    if args.offset_noise:
                        noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                            model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device,
                            dtype=weight_dtype
                        )
                    else:
                        noise = torch.randn_like(model_input).to(dtype=weight_dtype)
                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)

                    start_index = int((1 - args.strength) * len(timesteps))
                    t_enc = timesteps[start_index]    # strength=1.0 means full destruction original image.
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, t_enc).to(dtype=weight_dtype)

                guide_timesteps = timesteps[len(timesteps) - args.guidance_step: len(timesteps) - args.guidance_step + args.guidance_period].tolist()
                assert len(guide_timesteps) == args.guidance_period
                assert args.guidance_step >= 1      # start from 1

                bsz, channels, height, width = noisy_model_input.shape
                if args.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                if unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(
                    args.seed)
                latents = noisy_model_input.to(dtype=weight_dtype)

                if accelerator.is_main_process: logging.info("Guidance timesteps: %s", ", ".join([str(x) for x in guide_timesteps]))
                for i, t in tqdm_bar(enumerate(timesteps[start_index:]),
                                     total=len(timesteps[start_index:]), desc="Denoise steps",
                                     disable=not accelerator.is_local_main_process):

                    if t == guide_timesteps[0] and args.guidance_type == "transform_guidance":
                        latents, score = transform_guidance(latents, batch, guide_timesteps, noise_scheduler, unet, prompt_embeds,
                                                     class_labels, vae, image_encoder, image_processor, weight_dtype,
                                                     generator, total_global_proto, total_local_proto)
                        latents, x_0 = denoise_one_step(latents, noise_scheduler, t, unet, prompt_embeds, class_labels)
                        if accelerator.is_main_process: logging.info(
                            f"transform guidance in {t} step for {args.guidance_period} steps period, score: {score.item():.4f}")
                    elif t in guide_timesteps and args.guidance_type == "direct_guidance":
                        latents, x_0, score = direct_guidance(latents, batch, t, noise_scheduler, unet,
                                                       prompt_embeds, class_labels, vae, image_encoder, image_processor,
                                                       weight_dtype, generator, total_global_proto, total_local_proto)
                        if accelerator.is_main_process: logging.info(
                            f"start direct guidance in {t}/[{args.guidance_step} - "
                            f"{args.guidance_step - args.guidance_period}] step, score: {score.item():.4f}")
                    else:
                        latents, x_0 = denoise_one_step(latents, noise_scheduler, t, unet, prompt_embeds, class_labels)

                # 1. generate images with original pipelines
                with torch.no_grad():
                    decoded_original_latent = \
                        vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[0]

                    if accelerator.is_main_process:
                        image = decoded_original_latent
                        image = image_processor.postprocess(image, output_type="pt",
                                                            do_denormalize=[True] * image.shape[0])

                        for i in range(len(batch["image_paths"])):
                            image_file_path = os.path.basename(batch["image_paths"][i]).split(".")[0]
                            path = f'{args.output_dir}/{batch["class_names"][i]}/{image_file_path}_expand_{image_i}.png'
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            save_image([image[i]], path)

                torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
