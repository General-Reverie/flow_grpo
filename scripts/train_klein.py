"""
Flow-GRPO training script for FLUX.2 Klein-base-4B.

Adapted from train_flux.py. Key differences:
  - Uses Flux2KleinPipeline (single Qwen3 text encoder, no pooled_projections)
  - No guidance embeddings in transformer call
  - 4D position IDs with batch dimension
  - LoRA target modules auto-discovered from Flux2Transformer2DModel

Usage (single node):
    accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml \
        --num_processes=4 --main_process_port 29503 \
        scripts/train_klein.py --config config/grpo.py:pickscore_klein_base_4b
"""

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import Flux2KleinPipeline
from diffusers.utils.torch_utils import is_compiled_module
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.klein_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], {}


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_image_per_prompt, seed=0, rank=0, num_replicas=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = batch_size * num_replicas // (batch_size * num_replicas // num_image_per_prompt) if num_image_per_prompt > 0 else 1
        self.m = batch_size * num_replicas // self.k
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, pipeline, max_sequence_length, text_encoder_out_layers, device):
    """
    Klein: single Qwen3 encoder, returns (prompt_embeds, text_ids).
    No pooled_prompt_embeds.
    """
    with torch.no_grad():
        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
    return prompt_embeds, text_ids


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def compute_log_prob(transformer, pipeline, sample, j, config):
    """
    Klein-adapted log-prob computation.
    Key differences from FLUX.1:
      - No pooled_projections
      - No guidance tensor (guidance=None)
      - text_ids stored in sample with batch dimension
    """
    packed_noisy_model_input = sample["latents"][:, j]
    device = packed_noisy_model_input.device
    dtype = packed_noisy_model_input.dtype

    # Klein: no guidance embeddings, no pooled_projections
    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        timestep=sample["timesteps"][:, j] / 1000,
        guidance=None,
        encoder_hidden_states=sample["prompt_embeds"],
        txt_ids=sample["text_ids"],
        img_ids=sample["image_ids"],
        return_dict=False,
    )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def discover_lora_targets(transformer):
    """
    Auto-discover LoRA target modules from the Klein transformer architecture.
    Targets attention projections and feed-forward layers, matching the FLUX.1 pattern.
    """
    target_patterns = []
    seen_patterns = set()

    for name, module in transformer.named_modules():
        # Skip top-level
        if '.' not in name:
            continue
        # Get the relative pattern (e.g., "attn.to_k" from "transformer_blocks.0.attn.to_k")
        parts = name.split('.')
        # Find attention and FF patterns
        for i, part in enumerate(parts):
            if part in ('attn', 'ff', 'ff_context'):
                pattern = '.'.join(parts[i:])
                if pattern not in seen_patterns and hasattr(module, 'weight'):
                    seen_patterns.add(pattern)
                    target_patterns.append(pattern)

    if not target_patterns:
        # Fallback: use standard FLUX-like targets
        logger.warning("Could not auto-discover LoRA targets, using FLUX.1 defaults")
        target_patterns = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2",
            "ff_context.net.0.proj", "ff_context.net.2",
        ]

    return target_patterns


def eval(pipeline, test_dataloader, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, text_ids = compute_text_embeddings(
            prompts, pipeline,
            max_sequence_length=128,
            text_encoder_out_layers=(9, 18, 27),
            device=accelerator.device,
        )
        with autocast():
            with torch.no_grad():
                images, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,  # Deterministic eval
                )

        rewards, reward_metadata = reward_fn(images, prompts, prompt_metadata, only_strict=True)
        for key, value in rewards.items():
            all_rewards[key].append(torch.as_tensor(value, device=accelerator.device).float())

    all_rewards = {key: torch.cat(value) for key, value in all_rewards.items()}
    gathered_rewards = {key: accelerator.gather(value) for key, value in all_rewards.items()}
    gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

    if accelerator.is_main_process:
        wandb.log(
            {
                "eval_epoch": global_step,
                **{f"eval_reward_{key}": value.mean() for key, value in gathered_rewards.items()
                   if '_strict_accuracy' not in key and '_accuracy' not in key},
            },
            step=global_step,
        )

    if config.train.ema:
        ema.restore(transformer_trainable_parameters)


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    if config.use_lora:
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(os.path.join(save_path, "lora"))
    if config.train.ema:
        ema.restore(transformer_trainable_parameters)


def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(project="flow_grpo")
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # Load Klein pipeline
    pipeline = Flux2KleinPipeline.from_pretrained(
        config.pretrained.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    # Freeze non-trainable components
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    # Klein has single text encoder (no text_encoder_2)
    pipeline.transformer.requires_grad_(not config.use_lora)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move to device
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Auto-discover LoRA targets from Klein transformer
        target_modules = discover_lora_targets(pipeline.transformer)
        logger.info(f"LoRA target modules: {target_modules}")

        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # Dataset and dataloader
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, split='train')
        test_dataset = TextPromptDataset(config.dataset, split='test')
    else:
        # geneval or other prompt functions
        train_dataset = TextPromptDataset(config.dataset, split='train')
        test_dataset = TextPromptDataset(config.dataset, split='test')

    train_sampler = BatchSampler(
        train_dataset,
        batch_size=config.sample.train_batch_size,
        num_image_per_prompt=config.sample.num_image_per_prompt,
        seed=config.seed,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=lambda x: list(zip(*x)),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: list(zip(*x)),
    )

    # Setup reward functions
    reward_fn = flow_grpo.rewards.multi_reward_fn(config.reward_fn, config=config, accelerator=accelerator)
    eval_reward_fn = flow_grpo.rewards.multi_reward_fn(config.reward_fn, config=config, accelerator=accelerator)

    # Per-prompt stat tracking
    stat_tracker = PerPromptStatTracker(
        config.per_prompt_stat_tracking_buffer_size,
        config.per_prompt_stat_tracking_min_count,
    ) if config.per_prompt_stat_tracking else None

    # Prepare with accelerator
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    executor = futures.ThreadPoolExecutor(max_workers=2)
    autocast = contextlib.nullcontext if config.mixed_precision == "no" else accelerator.autocast

    samples_per_epoch = (
        config.sample.num_batches_per_epoch
        * config.sample.train_batch_size
        * accelerator.num_processes
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running Klein Flow-GRPO training *****")
    logger.info(f"  Model: {config.pretrained.model}")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size = {total_train_batch_size}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            # Klein: single encoder text embeddings
            prompt_embeds, text_ids = compute_text_embeddings(
                prompts, pipeline,
                max_sequence_length=128,
                text_encoder_out_layers=(9, 18, 27),
                device=accelerator.device,
            )

            # Tokenize for per-prompt stat tracking (using Klein's single tokenizer)
            prompt_ids = pipeline.tokenizer(
                list(prompts),
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # Sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, image_ids, ret_text_ids, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                    )

            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )

            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "text_ids": text_ids,  # Klein: stored for training log-prob
                    "image_ids": image_ids,  # Klein: 4D with batch dim
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # Wait for rewards
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # Collate samples
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # Log images
        if epoch % 10 == 0 and accelerator.is_main_process:
            pass  # TODO: add image logging like train_flux.py

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()
                       if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # Per-prompt stat tracking
        if config.per_prompt_stat_tracking and stat_tracker is not None:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)
            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                transformer, pipeline, sample, j, config
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                            transformer, pipeline, sample, j, config
                                        )

                        # GRPO loss
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float())
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1
        if epoch >= config.num_epochs:
            break


if __name__ == "__main__":
    app.run(main)
