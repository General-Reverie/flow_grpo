---
agents_plan_schema: '1.0'
archive_reason: stale plan queue purge
created: '2026-03-27'
date: '2026-03-27'
plan_id: flow-grpo-klein-base-4b
project: petridish/flow_grpo
provenance:
  created_by: claude-code
  created_from: legacy_plan_metadata
revision: 1
source: claude-code
stale_review_decision: archived
stale_review_resolved: '2026-05-02'
status: archived
tags:
- flow-grpo
- klein
- flux2
- rl
- diffusion
- training
- research
title: 'Flow-GRPO + Klein-base-4B: Test Project Plan'
updated: '2026-05-02'
---

# Flow-GRPO + Klein-base-4B: Test Project Plan


## Changelog

### 2026-05-02
- Backfilled provenance metadata from existing plan frontmatter and filesystem metadata.

## Context

Apply Flow-GRPO (online RL for flow matching models) to FLUX.2 Klein-base-4B to improve prompt adherence, compositional accuracy, and text rendering for the Slugworth Studios comedy generation pipeline. Going straight to Klein rather than SD3.5-M — accepting the porting risk to test the actual target model.

**Key constraint**: Klein-base-4B (non-distilled) is required — the distilled 4-step Klein 4B has too few denoising steps for Flow-GRPO's trajectory-based RL to work.

## Research Summary

| Item | Detail |
|------|--------|
| Paper | [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) — NeurIPS 2025 |
| Repo | [yifan123/flow_grpo](https://github.com/yifan123/flow_grpo) |
| Target model | [FLUX.2-klein-base-4B](https://hf.co/black-forest-labs/FLUX.2-klein-base-4B) (Apache 2.0) |
| Architecture gap | FLUX.2 (Qwen VL encoder, new VAE, different block ratios) vs supported FLUX.1 (T5+CLIP) |
| VRAM (LoRA) | Est. 24-48GB — fits 2x RTX 5090 or cloud A100s |
| Must use bf16 | fp16 breaks FLUX inference |

## Implementation Steps

1. Fork yifan123/flow_grpo to generalreverie, clone to ~/dev/petridish/flow_grpo
2. Study existing FLUX.1 integration (model loading, text encoding, noise schedule, configs)
3. Port to Flux2KleinPipeline (Qwen VL encoder, FLUX.2 VAE, adjusted transformer config)
4. Local smoke test on RTX 5090 (forward pass, LoRA attachment, ODE-to-SDE, VRAM check)
5. Cloud training via dtrain on 4x A100 (PickScore reward, LoRA rank 32, 500-1000 steps)
6. Evaluate: comparison grids, comedy-relevant prompts, PickScore delta

## Key Risk: Step Distillation

Klein 4B (distilled) uses ~4 inference steps — too few for Flow-GRPO's trajectory-based RL. Must use Klein-base-4B (non-distilled, full denoising trajectory).

## Sources

- [Flow-GRPO Paper](https://arxiv.org/abs/2505.05470)
- [Flow-GRPO GitHub](https://github.com/yifan123/flow_grpo)
- [FLUX.2 Klein Base 4B](https://hf.co/black-forest-labs/FLUX.2-klein-base-4B)