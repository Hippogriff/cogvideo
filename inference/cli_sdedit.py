import os
os.environ["HF_HOME"] = "/group-volume/gopalsharma/"
import argparse
import logging
import math
from typing import Optional, List, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import decord
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_video, load_image

# Set HF_HOME if not already set, but user explicitly added this so we keep it.
os.environ["HF_HOME"] = "/group-volume/gopalsharma/"

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
    
    if hasattr(vae.config, "invert_scale_latents") and vae.config.invert_scale_latents:
         return latent_dist / vae.config.scaling_factor
    else:
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
    debug_sdedit: bool = False,
    image_cond_path: Optional[str] = None,
):
    # 1. Load Pipeline
    is_i2v = "i2v" in model_path.lower()
    if is_i2v:
        logging.info("Detected Image-to-Video model.")
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    else:
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Enable optimizations
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # Calculate required frames including padding for patch_size_t
    num_frames = max_num_frames
    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = getattr(pipe.transformer.config, "patch_size_t", None)
    additional_frames = 0
    latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * pipe.vae_scale_factor_temporal

    # 2. Load and Encode Input Video
    logging.info(f"Loading video from {video_path}...")
    try:
        video_frames = get_video_frames(video_path, width, height, num_frames)
    except Exception as e:
        logging.error(f"Error loading video: {e}")
        return

    logging.info(f"Encoding video frames... Shape: {video_frames.shape}")
    
    with torch.no_grad():
        latents = encode_video_frames(pipe.vae, video_frames)
        # Pad latents if they have an odd number of frames (specific user workaround)
        if latents.shape[1] % 2 == 1:
            add_latents = torch.randn(
                (1, latents.shape[1] % 2, latents.shape[2], latents.shape[3], latents.shape[4]), 
                generator=generator, 
                device=device, 
                dtype=latents.dtype
            )
            latents = torch.cat([latents, add_latents], 1)

    # Handle I2V Image Conditioning
    image_latents = None
    if is_i2v:
        if image_cond_path is None:
            raise ValueError("image_cond_path is required for I2V models")
        
        logging.info(f"Loading conditioning image from {image_cond_path}...")
        image_cond = load_image(image_cond_path)
        
        # Preprocess image
        image_cond = image_cond.resize((width, height))
        
        # Convert PIL to tensor
        image_cond_tensor = T.ToTensor()(image_cond) # [C, H, W] -> [0,1]
        
        # Normalize to [-1, 1]
        image_cond_tensor = image_cond_tensor * 2.0 - 1.0
        
        # encode_video_frames expects [F, C, H, W]
        # We have [C, H, W], so unsqueeze(0) to get [1, C, H, W]
        image_cond_tensor = image_cond_tensor.unsqueeze(0)
        
        image_cond_tensor = image_cond_tensor.to(device=pipe.device, dtype=pipe.dtype)
        
        with torch.no_grad():
            image_latents = encode_video_frames(pipe.vae, image_cond_tensor) 

            # Match padding logic from pipeline
            padding_shape = (
                latents.shape[0], # Batch
                latents.shape[1] - 1, # Num frames - 1
                latents.shape[2], # Channels
                latents.shape[3], # Height
                latents.shape[4], # Width
            )
            
            latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
            image_latents = torch.cat([image_latents, latent_padding], dim=1) # Concatenate along frames

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
         t_start_idx = num_inference_steps - 1

    timestep = timesteps[t_start_idx]
    remaining_timesteps = timesteps[t_start_idx:]
    
    logging.info(f"Strength {strength} -> Starting from timestep {timestep.item()} (index {t_start_idx}). Total steps: {len(remaining_timesteps)}")

    # Add noise
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
    latents_noisy = pipe.scheduler.add_noise(latents, noise, timestep.unsqueeze(0))

    if debug_sdedit:
        logging.info("Saving debug video (noisy input)...")
        with torch.no_grad():
            debug_video = pipe.decode_latents(latents_noisy)
            debug_frames = pipe.video_processor.postprocess_video(video=debug_video, output_type="pil")
        debug_output_path = output_path.replace(".mp4", "_debug_noisy.mp4")
        export_to_video(video_frames=debug_frames[0], output_video_path=debug_output_path, fps=fps)
        logging.info(f"Saved debug video to {debug_output_path}")

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

    # Create ofs embeds if required
    ofs_emb = None
    if pipe.transformer.config.ofs_embed_dim is not None:
        ofs_emb = latents.new_full((1,), fill_value=2.0)

    logging.info("Starting denoising loop...")
    with torch.no_grad():
        for i, t in enumerate(remaining_timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Handle I2V Concatenation
            if is_i2v and image_latents is not None:
                # image_latents: [B, F, C, H, W]
                # latent_model_input: [2B, F, C, H, W] (if CFG)
                image_latents_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                # Concatenate along CHANNEL dimension (dim=2)
                latent_model_input = torch.cat([latent_model_input, image_latents_input], dim=2)

            # Predict noise
            # Broadcast timestep
            timestep_tensor = t.expand(latent_model_input.shape[0])
            
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep_tensor,
                image_rotary_emb=image_rotary_emb,
                ofs=ofs_emb,
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
    parser.add_argument("--debug_sdedit", action="store_true", help="Save debug videos (noisy input)")
    parser.add_argument("--image_cond_path", type=str, default=None, help="Path to conditioning image for I2V")
    
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
        max_num_frames=args.max_num_frames,
        debug_sdedit=args.debug_sdedit,
        image_cond_path=args.image_cond_path
    )