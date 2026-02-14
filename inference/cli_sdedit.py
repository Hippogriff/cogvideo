import argparse
import logging
import math
import os
from typing import Optional, List, Union

import torch
import torch.nn.functional as F
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_video, load_image
import decord
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO)

def get_video_frames(video_path: str, width: int, height: int, max_num_frames: int) -> torch.FloatTensor:
    """
    Loads video frames from a file, resizes them, and normalizes them to [-1, 1].
    """
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        
        # Simple frame sampling for now
        indices = list(range(0, min(video_num_frames, max_num_frames)))
        
        frames = video_reader.get_batch(indices=indices)
        frames = frames[:max_num_frames].float()

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        selected_num_frames = frames.size(0)
        remainder = (3 + selected_num_frames) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        assert frames.size(0) % 4 == 1

        # Normalize the frames
        transform = T.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]

def encode_video_frames(vae, video_frames: torch.FloatTensor) -> torch.FloatTensor:
    """
    Encodes video frames into latents using the VAE.
    """
    video_frames = video_frames.to(device=vae.device, dtype=vae.dtype)
    video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    latent_dist = vae.encode(x=video_frames).latent_dist.sample().transpose(1, 2)
    return latent_dist * vae.config.scaling_factor

def sdedit(
    prompt: str,
    video_path: str,
    model_path: str,
    output_path: str,
    strength: float = 0.8,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    width: int = 720,
    height: int = 480,
    max_num_frames: int = 81,
    fps: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    device: str = "cuda",
):
    # 1. Load Pipeline
    # We use CogVideoXPipeline because it has the Transformer and VAE.
    # We will implement the loop manually.
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    
    # Enable optimizations
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # 2. Load and Encode Input Video
    logging.info(f"Loading video from {video_path}...")
    try:
        video_frames = get_video_frames(video_path, width, height, max_num_frames)
    except Exception as e:
        logging.error(f"Error loading video: {e}")
        return

    logging.info(f"Encoding video frames... Shape: {video_frames.shape}")
    with torch.no_grad():
        latents = encode_video_frames(pipe.vae, video_frames)
    
    # 3. Add Noise (Forward Diffusion)
    # Schedule setup
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # Calculate start timestep based on strength
    init_timestep_len = int(num_inference_steps * strength)
    t_start_idx = max(num_inference_steps - init_timestep_len, 0)
    
    # Handle edge case where we don't do any steps
    if t_start_idx >= num_inference_steps:
         logging.info("Strength is too low, no denoising will be performed.")
         return 

    timestep = timesteps[t_start_idx]
    remaining_timesteps = timesteps[t_start_idx:]
    
    logging.info(f"Strength {strength} -> Starting from timestep {timestep.item()} (index {t_start_idx}). Total steps: {len(remaining_timesteps)}")

    # Add noise
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    
    latents_noisy = pipe.scheduler.add_noise(latents, noise, timestep.unsqueeze(0))
    
    # 4. Denoise (Reverse Diffusion)
    
    # Encode prompt
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # Decode loop
    latents = latents_noisy
    
    # Prepare rotary embeds
    image_rotary_emb = (
        pipe._prepare_rotary_positional_embeddings(
            height=latents.size(3) * pipe.vae_scale_factor_spatial,
            width=latents.size(4) * pipe.vae_scale_factor_spatial,
            num_frames=latents.size(1),
            device=device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    logging.info("Starting denoising loop...")
    with torch.no_grad():
        for i, t in enumerate(remaining_timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            # Broadcast timestep
            timestep_tensor = t.expand(latent_model_input.shape[0])
            
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_tensor,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            
            # Guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = latents.to(dtype)

    # 5. Decode Latents
    logging.info("Decoding latents...")
    with torch.no_grad():
        video = pipe.decode_latents(latents)
        frames = pipe.video_processor.postprocess_video(video=video, output_type="pil")
    
    export_to_video(video_frames=frames[0], output_video_path=output_path, fps=fps)
    logging.info(f"Saved video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video for SDEdit")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX1.5-5B", help="Hugging Face model path")
    parser.add_argument("--output_path", type=str, default="output_sdedit.mp4", help="Output video path")
    parser.add_argument("--strength", type=float, default=0.8, help="Denoising strength (0.0 to 1.0)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Total number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--width", type=int, default=720, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--fps", type=int, default=8, help="Output FPS")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_num_frames", type=int, default=81, help="Max frames to process")
    
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    sdedit(
        prompt=args.prompt,
        video_path=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        fps=args.fps,
        dtype=dtype,
        seed=args.seed,
        max_num_frames=args.max_num_frames
    )
