# Adapted from flux_pipeline_with_logprob.py for FLUX.2 Klein-base-4B
# Key differences from FLUX.1:
#   - Single text encoder (Qwen3ForCausalLM) instead of dual (CLIP + T5)
#   - No pooled_prompt_embeds (no pooled_projections in transformer call)
#   - 4D position IDs (T, H, W, L) with batch dimension
#   - guidance=None (no embedded guidance; CFG handled externally if needed)
#   - in_channels=128 (vs FLUX.1's 64)
#   - Uses compute_empirical_mu for timestep shifting
#   - AutoencoderKLFlux2 VAE with different unpack/decode path

from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 4.0,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple = (9, 18, 27),
    noise_level: float = 0.7,
):
    """
    Klein pipeline with SDE sampling and log-probability computation for Flow-GRPO.

    Returns:
        (images, all_latents, latent_image_ids, text_ids, all_log_probs)
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Encode prompt — single Qwen3 encoder returns (prompt_embeds, text_ids)
    if prompt_embeds is None:
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
    else:
        # When prompt_embeds are pre-computed, generate text_ids from shape
        # Klein's _prepare_text_ids takes the embeddings tensor directly
        text_ids = self._prepare_text_ids(prompt_embeds)

    # 3. Prepare latent variables
    # Klein: in_channels=128, packing factor=4 → num_channels_latents=32
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 4. Prepare timesteps with empirical mu shift
    image_seq_len = latents.shape[1]
    mu = self.compute_empirical_mu(image_seq_len, num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 5. Denoising loop with SDE sampling and log-prob collection
    all_latents = [latents]
    all_log_probs = []

    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Klein transformer: no pooled_projections, guidance=None
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.attention_kwargs if hasattr(self, 'attention_kwargs') else attention_kwargs,
                return_dict=False,
            )[0]

            latents_dtype = latents.dtype
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0).repeat(latents.shape[0]),
                latents.float(),
                noise_level=noise_level,
            )
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    # 6. Decode latents to images
    # Klein: unpack sequence → spatial, batch norm denormalize, unpatchify, VAE decode
    latents = self._unpack_latents_with_ids(latents, latent_image_ids)

    # Batch norm denormalization (Klein-specific, replaces FLUX.1's scaling_factor)
    latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(
        self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * latents_bn_std + latents_bn_mean

    # Unpatchify: reverse 2x2 patch packing
    latents = self._unpatchify_latents(latents)

    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    self.maybe_free_model_hooks()

    return image, all_latents, latent_image_ids, text_ids, all_log_probs
