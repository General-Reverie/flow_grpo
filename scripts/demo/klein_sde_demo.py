"""
SDE verification demo for FLUX.2 Klein-base-4B.

Tests that the ODE-to-SDE conversion produces valid images at various noise levels.
Step 2 from the "How to Support Other Models" guide:
  Set noise_level=0 to verify deterministic ODE sampling produces normal images.
  Increasing noise levels should produce valid stochastic samples.

Usage:
    python scripts/demo/klein_sde_demo.py
"""

import torch
from diffusers import Flux2KleinPipeline
from flow_grpo.diffusers_patch.klein_pipeline_with_logprob import pipeline_with_logprob
from PIL import Image
import os

def main():
    model_id = "black-forest-labs/FLUX.2-klein-base-4B"
    output_dir = "outputs/klein_sde_demo"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_id}...")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    prompt = "A professional photograph of a comedian on stage holding a microphone, spotlight, dark background"
    seed = 42

    # Print model info for debugging
    print(f"Transformer in_channels: {pipe.transformer.config.in_channels}")
    print(f"Transformer num_layers (dual): {pipe.transformer.config.num_layers}")
    print(f"Transformer num_single_layers: {pipe.transformer.config.num_single_layers}")
    print(f"Transformer guidance_embeds: {pipe.transformer.config.guidance_embeds}")
    print(f"VAE config keys: {list(pipe.vae.config.keys()) if hasattr(pipe.vae.config, 'keys') else dir(pipe.vae.config)}")
    if hasattr(pipe.vae.config, 'scaling_factor'):
        print(f"VAE scaling_factor: {pipe.vae.config.scaling_factor}")
    if hasattr(pipe.vae.config, 'shift_factor'):
        print(f"VAE shift_factor: {pipe.vae.config.shift_factor}")

    # Print transformer module names for LoRA target identification
    print("\n--- Transformer module names (for LoRA targeting) ---")
    for name, _ in pipe.transformer.named_modules():
        if any(k in name for k in ['attn', 'ff', 'proj', 'norm']):
            print(f"  {name}")
    print("--- End module names ---\n")

    for noise_level in [0, 0.5, 0.7, 0.8, 0.9, 1.0]:
        print(f"Generating with noise_level={noise_level}...")
        generator = torch.Generator(device="cuda").manual_seed(seed)

        images, latents, image_ids, text_ids, log_probs = pipeline_with_logprob(
            pipe,
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=4.0,
            output_type="pil",
            height=512,
            width=512,
            generator=generator,
            noise_level=noise_level,
        )

        # Save image
        img_path = os.path.join(output_dir, f"noise_{noise_level:.1f}.jpg")
        images[0].save(img_path, quality=95)
        print(f"  Saved: {img_path}")
        print(f"  Num denoising steps: {len(log_probs)}")
        if noise_level > 0:
            print(f"  Log probs range: [{min(lp.min().item() for lp in log_probs):.2f}, {max(lp.max().item() for lp in log_probs):.2f}]")

    print(f"\nAll images saved to {output_dir}/")
    print("Verify: noise_level=0 should produce deterministic ODE samples.")
    print("Higher noise levels should produce valid stochastic variations.")


if __name__ == "__main__":
    main()
