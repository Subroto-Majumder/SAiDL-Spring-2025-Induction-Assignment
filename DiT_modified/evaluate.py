# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to sample from a trained DiT model and calculate FID.
Handles unconditional models trained with either original train.py (using models.py)
or sliding_window_train.py (using sliding_window_models.py).
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
# from models import DiT_models # Import dynamically below
from diffusion import create_diffusion
import math
from tqdm import tqdm
import numpy as np
import importlib # Needed for dynamic imports

# Assume pytorch-fid is installed: pip install pytorch-fid
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    fid_available = True
except ImportError:
    print("Warning: pytorch-fid not installed. FID calculation will be skipped. Install with: pip install pytorch-fid")
    fid_available = False

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: No CUDA device found, using CPU. Sampling will be very slow.")

    if args.ckpt is None:
        raise ValueError("Must specify a checkpoint path with --ckpt")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt}")

    # --- Dynamic Model Import ---
    try:
        # Prioritize the modified models file if it exists
        model_module = importlib.import_module("sliding_window_models")
        DiT_models = model_module.DiT_models
        using_sliding_window_models = True
        print("Using model definitions from sliding_window_models.py")
    except ImportError:
        try:
            # Fallback to the original models file
            model_module = importlib.import_module("models")
            DiT_models = model_module.DiT_models
            using_sliding_window_models = False
            print("Using model definitions from models.py (sliding_window_models.py not found)")
        except ImportError:
            raise ImportError("Could not import DiT_models from either 'sliding_window_models.py' or 'models.py'. Ensure one of them is in the Python path.")
    # --- End Dynamic Model Import ---


    # Load checkpointed args
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if "args" not in checkpoint:
        # Try loading older checkpoints that might only have model state
        print("Warning: Checkpoint does not contain 'args'. Attempting to load model based on command-line arguments.")
        # In this case, user *must* provide correct --model, --image-size etc. matching the checkpoint
        ckpt_args = args # Use command-line args as fallback
        ckpt_args.attention_type = args.attention_type # Need to explicitly set this
        ckpt_args.window_size = args.window_size
        # VAE needs to be specified via command line if not in checkpoint
        if not hasattr(ckpt_args, 'vae'):
             ckpt_args.vae = args.vae # Assume command line VAE is correct
    else:
        ckpt_args = checkpoint["args"]
        print(f"Restoring model using args from checkpoint: {ckpt_args}")
        # Ensure VAE attribute exists, default if necessary
        if not hasattr(ckpt_args, 'vae'):
            print(f"Warning: 'vae' not found in checkpoint args. Using command-line VAE '{args.vae}'.")
            ckpt_args.vae = args.vae


    # Create Diffusion process
    # Use diffusion steps from command line args, as it's a sampling parameter
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Load VAE based on checkpoint args (or command line if missing)
    try:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{ckpt_args.vae}").to(device)
    except Exception as e:
        print(f"Error loading VAE 'stabilityai/sd-vae-ft-{ckpt_args.vae}': {e}")
        print("Ensure the VAE name is correct and you have internet access / cached files.")
        raise e

    # Determine model parameters from checkpoint args
    latent_size = ckpt_args.image_size // 8
    model_name = ckpt_args.model # e.g., 'DiT-S/8'

    # Prepare model kwargs based on which model definition is being used
    model_kwargs = {
        'input_size': latent_size,
        'num_classes': 0, # Always 0 for unconditional specified in the task
    }

    if using_sliding_window_models:
        # sliding_window_models.py expects attention_type and window_size
        # Default to 'full' if not present in older checkpoints trained with original models.py
        model_kwargs['attention_type'] = getattr(ckpt_args, 'attention_type', 'full')
        # Default window_size if sliding attention is specified but size is missing (unlikely but safe)
        if model_kwargs['attention_type'] == 'sliding':
             model_kwargs['window_size'] = getattr(ckpt_args, 'window_size', 8) # Default to 8 if missing
        print(f"Instantiating model with: {model_kwargs}")
    else:
        # Original models.py does not expect attention_type or window_size
        print(f"Instantiating model with: {model_kwargs}")
        # Check if checkpoint args have sliding window info - warn if incompatible
        if hasattr(ckpt_args, 'attention_type') and ckpt_args.attention_type == 'sliding':
             print("Warning: Checkpoint args indicate sliding window attention, but using original 'models.py'. "
                   "Ensure the checkpoint was actually trained with full attention.")

    # Create Model using the dynamically imported factory
    model = DiT_models[model_name](**model_kwargs).to(device)

    # Load model weights
    if "ema" in checkpoint:
        print("Loading EMA model weights.")
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        print("Loading standard model weights (EMA not found).")
        state_dict = checkpoint["model"]
    else:
        raise ValueError("Checkpoint does not contain 'ema' or 'model' state_dict.")

    # Handle potential mismatch in state_dict keys (e.g., DDP prefix)
    # If the checkpoint was saved with DDP, keys might start with 'module.'
    # If loading into a non-DDP model, remove the prefix.
    # Since we load directly into the base model here (not DDP wrapped), we might need to strip 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False) # Use strict=False for robustness
    if missing_keys:
        print("Warning: Missing keys when loading state_dict:", missing_keys)
    if unexpected_keys:
        print("Warning: Unexpected keys when loading state_dict:", unexpected_keys)


    model.eval()  # important!

    # Create output directory
    # Use the checkpoint's directory structure to save samples nearby
    ckpt_dir = os.path.dirname(args.ckpt)
    experiment_dir = os.path.dirname(ckpt_dir) # Go up one level from /checkpoints
    samples_dir = os.path.join(experiment_dir, f"samples_step{args.num_sampling_steps}")
    if args.suffix:
        samples_dir += f"_{args.suffix}"
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Saving samples to {samples_dir}")

    # Figure out how many samples we need to generate on this process
    samples_needed = args.num_fid_samples
    samples_generated = 0

    print(f"Generating {samples_needed} samples...")
    pbar = tqdm(total=samples_needed)
    while samples_generated < samples_needed:
        n = min(args.batch_size, samples_needed - samples_generated)

        # Create random noise
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)

        # Setup classifier-free guidance: Not applicable for unconditional
        model_kwargs_sample = {}

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs_sample, progress=False, device=device
        )

        # Decode samples
        samples = vae.decode(samples / 0.18215).sample

        # Save samples to disk
        for i, sample in enumerate(samples):
            img_path = os.path.join(samples_dir, f"{samples_generated + i:06d}.png")
            save_image(sample, img_path, normalize=True, value_range=(-1, 1))

        samples_generated += n
        pbar.update(n)

    pbar.close()
    print("Finished generating samples.")

    # --- FID Calculation ---
    if not fid_available:
        print("Skipping FID calculation because pytorch-fid is not installed.")
        return

    if args.fid_stats_path is None and args.fid_real_path is None:
        print("Skipping FID calculation: Neither --fid-stats-path nor --fid-real-path provided.")
        return

    fid_value = -1.0
    try:
        if args.fid_stats_path:
            if not os.path.exists(args.fid_stats_path):
                 raise FileNotFoundError(f"FID stats file not found: {args.fid_stats_path}")
            print(f"Calculating FID against precomputed stats: {args.fid_stats_path}")
            fid_value = calculate_fid_given_paths(
                [samples_dir, args.fid_stats_path],
                batch_size=args.fid_batch_size,
                device=device,
                dims=2048, # Standard FID dimension
                num_workers=args.num_workers
            )
        elif args.fid_real_path:
            if not os.path.exists(args.fid_real_path):
                raise FileNotFoundError(f"Real image path for FID not found: {args.fid_real_path}")
            print(f"Calculating FID against real images: {args.fid_real_path}")
            fid_value = calculate_fid_given_paths(
                [samples_dir, args.fid_real_path],
                batch_size=args.fid_batch_size,
                device=device,
                dims=2048,
                num_workers=args.num_workers
            )

        print(f"FID score: {fid_value:.4f}")
        # Save FID score to a file in the experiment directory
        fid_file_path = os.path.join(experiment_dir, f"fid_score_{os.path.basename(samples_dir)}.txt")
        with open(fid_file_path, 'w') as f:
             f.write(f"FID: {fid_value:.4f}\n")
             f.write(f"Checkpoint: {args.ckpt}\n")
             f.write(f"Generated Samples Path: {samples_dir}\n")
             if args.fid_stats_path:
                 f.write(f"Reference Stats Path: {args.fid_stats_path}\n")
             if args.fid_real_path:
                 f.write(f"Reference Real Path: {args.fid_real_path}\n")
             f.write(f"Sampling Steps: {args.num_sampling_steps}\n")
        print(f"Saved FID score to {fid_file_path}")

    except Exception as e:
        print(f"Could not calculate FID: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the DiT checkpoint (.pt file)")
    parser.add_argument("--num-fid-samples", type=int, default=10000, help="Number of samples to generate for FID calculation")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for sampling")
    parser.add_argument("--num-sampling-steps", type=int, default=250, help="Number of DDIM sampling steps")
    parser.add_argument("--fid-real-path", type=str, default=None, help="Path to directory of real images for FID calculation")
    parser.add_argument("--fid-stats-path", type=str, default=None, help="Path to precomputed FID stats file (.npz) for real images")
    parser.add_argument("--fid-batch-size", type=int, default=64, help="Batch size for FID calculation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for FID calculation")
    parser.add_argument("--suffix", type=str, default=None, help="Optional suffix for samples directory name")

    # Add fallback arguments in case checkpoint doesn't contain 'args'
    parser.add_argument("--model", type=str, default="DiT-S/8", help="Fallback: Model architecture, if not in ckpt args")
    parser.add_argument("--image-size", type=int, default=256, help="Fallback: Image size, if not in ckpt args")
    parser.add_argument("--attention-type", type=str, choices=['full', 'sliding'], default='full', help="Fallback: Attention type, if not in ckpt args")
    parser.add_argument("--window-size", type=int, default=8, help="Fallback: Window size, if not in ckpt args")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema", help="Fallback: VAE checkpoint variant, if not in ckpt args")


    args = parser.parse_args()

    if args.fid_real_path and args.fid_stats_path:
        print("Warning: Both --fid-real-path and --fid-stats-path provided. Using --fid-stats-path.")
        args.fid_real_path = None

    main(args)