# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
MODIFIED for unconditional training and selectable attention mechanism.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from sliding_window_models import DiT_models # Imports the factory functions
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params: # Ensure the param exists in EMA model
             ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        # Original code didn't check if param exists, which is fine if models are identical
        # ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:  # Check if DDP is initialized
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2
        # Add attention type to folder name
        attn_str = f"attn-{args.attention_type}"
        if args.attention_type == 'sliding':
             attn_str += f"-win{args.window_size}"
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{attn_str}"  # Create specific experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Training args: {args}") # Log arguments
    else:
        logger = create_logger(None) # Dummy logger for other ranks

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    # Pass necessary arguments for unconditional model and attention type
    model_kwargs = {
         'input_size': latent_size,
         'num_classes': 0, # Set num_classes to 0 for unconditional
         'class_dropout_prob': 0.0, # Not used if num_classes is 0, but set explicitly
         'attention_type': args.attention_type,
         'window_size': args.window_size,
    }
    model = DiT_models[args.model](**model_kwargs)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # Move model to device before DDP wrapper
    model = model.to(device)
    model = DDP(model, device_ids=[device]) # Use device instead of rank for device_ids
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # Load VAE **after** setting device to prevent memory allocation on default device
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters ({args.attention_type} attn): {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # Assumes landscape dataset is structured like ImageFolder (e.g., data_path/class_name/image.jpg)
    # If it's just flat files (data_path/image.jpg), ImageFolder might need a dummy class structure
    # or use a different Dataset class (e.g., torchvision.datasets.DatasetFolder or custom)
    try:
        dataset = ImageFolder(args.data_path, transform=transform)
    except Exception as e:
        logger.error(f"Failed to load dataset using ImageFolder from {args.data_path}. "
                     f"Ensure it has a subdirectory structure (e.g., data_path/images/*.jpg). Error: {e}")
        raise e

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance (if used)
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # Use _ for the label as it's not needed for unconditional training
        for x, _ in loader:
            x = x.to(device)
            # y = y.to(device) # No label needed

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                # VAE requires NCHW format
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            # For unconditional model, model_kwargs is empty or doesn't contain 'y'
            # However, our modified DiT forward needs 'y=None' explicitly if not provided
            # The diffusion loss function might internally call model(**model_kwargs)
            # Let's pass empty dict, assuming diffusion library handles it or
            # DiT forward's default y=None works.
            # Check diffusion.training_losses source if issues arise.
            # It likely unpacks kwargs: model(x, t, **model_kwargs).
            # Since our model is unconditional, we don't pass 'y'.
            model_kwargs = {}
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args # Save args to restore model correctly
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier() # Ensure all ranks wait for rank 0 to save checkpoint

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    # This script focuses on training. Use evaluate.py for sampling/FID.

    logger.info("Training finished!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to landscape dataset (ImageFolder structure)")
    parser.add_argument("--results-dir", type=str, default="results_unconditional", help="Directory to save training results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/8", help="Which DiT model configuration to use") # Smaller default for faster testing
    parser.add_argument("--attention-type", type=str, choices=['full', 'sliding'], default='full', help="Type of attention mechanism")
    parser.add_argument("--window-size", type=int, default=8, help="Window size for sliding attention (if used)")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Image size for training")
    # No num_classes argument needed, will be set to 0 internally
    # parser.add_argument("--num-classes", type=int, default=1000) # REMOVED
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs") # Reduced default for faster testing
    parser.add_argument("--global-batch-size", type=int, default=64, help="Total batch size across all GPUs") # Reduced default
    parser.add_argument("--global-seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema", help="VAE checkpoint variant (ema/mse)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--log-every", type=int, default=100, help="Log training stats every N steps")
    parser.add_argument("--ckpt-every", type=int, default=10_000, help="Save checkpoint every N steps") # Reduced default
    args = parser.parse_args()

    # Validate window size if sliding attention is used
    if args.attention_type == 'sliding' and args.window_size <= 0:
        raise ValueError("Window size must be positive for sliding attention.")
    if args.attention_type == 'sliding' and args.window_size % 2 != 0:
         print(f"Warning: Sliding window size {args.window_size} is odd. Padding might be asymmetric.")


    main(args)