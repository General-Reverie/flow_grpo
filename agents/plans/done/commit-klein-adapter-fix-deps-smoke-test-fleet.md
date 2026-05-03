---
agents_plan_schema: '1.0'
checkboxes: 12/12
created: '2026-03-27'
date: '2026-03-27'
last_swept: '2026-05-02T08:42:12Z'
plan_id: commit-klein-adapter-fix-deps-smoke-test-fleet
project: petridish/flow_grpo
provenance:
  created_by: claude-code
  created_from: legacy_plan_metadata
revision: 1
source: claude-code
status: completed
tags:
- inference
- infrastructure
- training
title: 'Next: Commit Klein adapter, fix deps, smoke test on fleet'
updated: '2026-05-02'
---

# Next: Commit Klein adapter, fix deps, smoke test on fleet


## Changelog

### 2026-05-02
- Backfilled provenance metadata from existing plan frontmatter and filesystem metadata.

## Where We Are
> Project: flow_grpo (Klein-base-4B port) | Location: `~/dev/petridish/flow_grpo`
> Fork: `General-Reverie/flow_grpo` from `yifan123/flow_grpo`

Research complete, 4 adapter files written (pipeline, training script, config, SDE demo), but not yet committed or tested. A critical dependency issue blocks the smoke test: `setup.py` pins `diffusers==0.33.1` but `Flux2KleinPipeline` requires `diffusers>=0.37.1`.

## Carry-Forward Context
- `setup.py` pins `diffusers==0.33.1` — Klein needs `>=0.37.1`
- `transformers==4.40.0` — Qwen3ForCausalLM likely needs `>=4.44.0` (Bagel notes confirm this)
- `peft==0.10.0` — may be too old for Flux2Transformer2DModel LoRA support
- Must use bf16 (fp16 breaks FLUX family inference)
- Fleet nodes: babylon/atlantis/olympus (RTX 5090 32GB) for smoke test
- Klein-base-4B model (~8GB bf16) needs to be pre-downloaded to fleet before testing

## Proposed Work

### 1. Commit Klein adapter files [DONE]
**Do:**
- [x] Stage the 4 new files + config/grpo.py changes
- [x] Commit with descriptive message
- [x] Push to `General-Reverie/flow_grpo`

**Files:**
- `flow_grpo/diffusers_patch/klein_pipeline_with_logprob.py` — new
- `scripts/train_klein.py` — new
- `scripts/demo/klein_sde_demo.py` — new
- `scripts/single_node/grpo_klein.sh` — new
- `config/grpo.py` — modified (3 new config functions)

**Verify:**
- `git log --oneline -1` shows the commit

### 2. Update dependencies for FLUX.2 Klein support [DONE]
**Do:**
- [x] Bump `diffusers>=0.37.1` in setup.py (resolved to 0.37.1)
- [x] Bump `transformers>=4.44.0` in setup.py (resolved to 5.4.0)
- [x] Bump `peft>=0.13.0` in setup.py (resolved to 0.18.1)
- Add `qwen-vl-utils` if needed by the text encoder
- Keep all other deps pinned (don't break SD3/FLUX.1 compatibility)
- Commit the dependency update separately

**Files:**
- `setup.py` — version bumps

**Verify:**
- `pip install -e .` succeeds in a fresh env
- `python -c "from diffusers import Flux2KleinPipeline"` works

### 3. Deploy to fleet node for smoke test [DONE]
**Note:** RTX 5090 (sm_120) incompatible with torch 2.6.0 — used delphi (RTX 4090 24GB).
**Do:**
- [x] Push updated branch to remote
- [x] Clone on delphi (RTX 4090)
- [x] Install deps (diffusers 0.37.1, transformers 5.4.0, peft 0.18.1)
- [x] Run klein_sde_demo.py — ALL 6 noise levels passed
- [x] Module names captured for LoRA targeting
- [x] LoRA targets hardcoded in train_klein.py
- Pull the repo
- Create Python env and install deps
- Pre-download Klein-base-4B model from HuggingFace
- Pre-download PickScore reward model
- Run `klein_sde_demo.py` — this does TWO things:
  1. Verifies SDE sampling produces valid images at noise levels 0-1.0
  2. Prints all transformer module names for LoRA target verification

**Files:**
- Remote: `~/dev/petridish/flow_grpo/` on fleet node

**Verify:**
- SDE demo generates valid images at all noise levels
- `noise_level=0` produces deterministic ODE samples
- Module names printed — verify `discover_lora_targets()` finds the right ones
- VRAM usage reported (validates whether 2x 5090 or cloud needed)

### 4. Fix LoRA targets based on module inspection
**Do:**
- Compare auto-discovered module names from SDE demo output against the targets in `discover_lora_targets()`
- If patterns differ from FLUX.1 (likely), update the fallback list
- Optionally hardcode the verified targets for Klein in `train_klein.py` instead of relying on auto-discovery
- Run on-policy consistency check per README Step 3:
  - Set `num_batches_per_epoch=1` and `gradient_accumulation_steps=1`
  - Verify ratio stays exactly 1.0

**Files:**
- `scripts/train_klein.py` — update LoRA targets if needed

**Verify:**
- LoRA attaches to correct modules
- Trainable parameter count is reasonable (~2-5% of total params)
- On-policy ratio = 1.0

### 5. Create dtrain job configuration
**Do:**
- Create a dtrain-compatible training job config for Klein Flow-GRPO
- Set up the cloud training environment (Prime Intellect 4x A100-80GB)
- Configure model/reward model download as part of job setup
- Set up wandb logging
- Prepare BS=1 trial config (single training step) per AGENTS.md rules

**Files:**
- New dtrain job config (location TBD based on dtrain structure)
- May need a `requirements.txt` or Dockerfile amendments

**Verify:**
- dtrain job submits successfully
- BS=1 trial completes, loss is finite, reward is computed
- wandb logs appear

### 6. Run full training (500-1000 steps)
**Do:**
- After BS=1 trial passes, launch full training run
- Monitor via wandb: reward curve, ratio distribution, clip fraction
- Watch for reward hacking (proxy score rises but image quality drops)
- Save checkpoints every 30 epochs

**Verify:**
- Reward curve increases monotonically for first 200+ steps
- Ratio stays centered near 1.0 (not diverging)
- Clip fraction < 0.3 (not too aggressive)
- Generated images look reasonable at checkpoint intervals

### 7. Evaluate and compare
**Do:**
- Load best checkpoint
- Generate comparison grid: vanilla Klein-base-4B vs Flow-GRPO-tuned
- Comedy-pipeline test prompts:
  - Multi-character: "two comedians at a table, one standing with microphone"
  - Spatial: "person on the left holding a drink, person on the right gesturing"
  - Text: "neon sign reading KILL TONY"
  - Attributes: "tall man in red shirt, short woman in blue dress"
- Compute PickScore delta (average improvement)
- Run through batch generation pipeline to test real-world integration

**Files:**
- New eval script or notebook

**Verify:**
- PickScore improves by measurable margin (>5% relative)
- Visual inspection confirms better compositional accuracy
- No mode collapse or quality degradation

## Implementation Order
1. Commit + push (save work immediately)
2. Dependency updates (unblock Klein import)
3. Fleet deploy + SDE demo (validate pipeline on real GPU)
4. LoRA target fix (based on demo output)
5. dtrain job config (prepare cloud training)
6. Training run (the actual experiment)
7. Evaluation (determine if it was worth it)

Steps 1-4 are the critical path before any training can happen. Step 3 is the first real GPU test.

## Verification
End-to-end success means: Klein-base-4B with Flow-GRPO LoRA generates images with measurably better prompt adherence than vanilla Klein-base-4B, as measured by PickScore delta and visual inspection on comedy-relevant prompts.