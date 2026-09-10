# RL Pipeline Notes

Working notes on what actually happens inside a VeRL GRPO update. **Every claim in this file must
be traceable to the checked-out source at the recorded commit** — see
[`../manifests/versions.txt`](../manifests/versions.txt). Where a statement has not yet been
verified against source, it is marked `UNVERIFIED` and must not be cited.

## The loop we are trying to understand

```
rollout
  -> reward
  -> group reward statistics (mean, std over the G samples of one prompt)
  -> advantage (group-normalised)
  -> old logprob
  -> current logprob
  -> importance ratio
  -> clipping
  -> KL
  -> entropy
  -> gradient
  -> optimizer update
  -> next-policy rollout
```

The questions this project has to answer from real code and real runs, not from memory, are
enumerated in [`../analysis/FINAL_RL_ENGINEERING_REPORT.md`](../analysis/FINAL_RL_ENGINEERING_REPORT.md).

## The official baseline we start from

`examples/grpo_trainer/run_qwen3_8b_fsdp.sh` in the VeRL checkout — GRPO, Qwen3-8B, FSDP training,
vLLM rollout, NVIDIA GPUs. Its defaults, read directly from the script:

| Knob | Default | Where |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` | `DATA` array |
| `algorithm.use_kl_in_reward` | `False` | `DATA` array |
| `data.train_batch_size` | 1024 | `TRAIN_BATCH_SIZE` |
| `data.max_prompt_length` | 1024 | `MAX_PROMPT_LENGTH` |
| `data.max_response_length` | 2048 | `MAX_RESPONSE_LENGTH` |
| `data.filter_overlong_prompts` | `True` | `DATA` array |
| `data.truncation` | `error` | `DATA` array |
| `actor.ppo_mini_batch_size` | 256 | `PPO_MINI_BATCH_SIZE` |
| `actor.use_dynamic_bsz` | `True` | `ACTOR` array |
| `actor.ppo_max_token_len_per_gpu` | 24576 | `PPO_MAX_TOKEN_LEN_PER_GPU` |
| `actor.optim.lr` | `1e-6` | `ACTOR_LR` |
| `actor.use_kl_loss` | `True` | `ACTOR` array |
| `actor.kl_loss_coef` | `0.001` | `KL_LOSS_COEF` |
| `actor.kl_loss_type` | `low_var_kl` | `ACTOR` array |
| `actor.entropy_coeff` | `0` | `ENTROPY_COEFF` |
| `actor.fsdp_config.param_offload` | `False` (GPU branch) | derived defaults |
| `actor.fsdp_config.optimizer_offload` | `False` (GPU branch) | derived defaults |
| `model.use_remove_padding` | `True` | `MODEL` array |
| `model.enable_gradient_checkpointing` | `True` | `MODEL` array |
| `rollout.name` | `vllm` | `INFER_BACKEND` |
| `rollout.tensor_model_parallel_size` | 2 (GPU branch) | derived defaults |
| `rollout.gpu_memory_utilization` | 0.6 (GPU branch) | derived defaults |
| `rollout.n` | 5 | `ROLLOUT_N` |
| `rollout.log_prob_use_dynamic_bsz` | `True` | `ROLLOUT` array |
| `ref.fsdp_config.param_offload` | `True` | `REF` array |
| `trainer.balance_batch` | `True` | `TRAINER` array |

Two structural facts already visible in the script, both worth internalising:

1. **KL is applied as a loss term, not baked into the reward.** `use_kl_loss=True` with
   `use_kl_in_reward=False`. So KL shapes the gradient directly, and the reward stays a clean
   correctness signal. That matters for interpreting the reward curve.
2. **The reference policy is offloaded to CPU by default** (`ref.fsdp_config.param_offload=True`)
   while the actor is not. The reference model only needs forward passes for its logprobs, so it
   trades host-transfer time for VRAM — which is why reference-logprob wall time is tracked as its
   own metric.

### Launcher

The script launches through the project's uv lock rather than ambient python:

```
uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo ...
```

and pins Ray workers to the same interpreter via
`ray_kwargs.ray_init.runtime_env.py_executable`. Setting `VERL_USE_UV=0` falls back to system
python. We use the uv path so the ambient torch 2.8.0+cu128 install is never touched.

## Dependency contract of the current checkout

Read from `pyproject.toml` at the recorded commit:

- `requires-python = ">=3.10,<3.13"`, and the uv `environments` list targets
  `python_full_version >= '3.12'`. Node python is **3.12.3** — inside the contract.
- `torch==2.11.0`, `torchvision==0.26.0`, `torchaudio==2.11.0`, routed to the `pytorch-cu130`
  index — i.e. a **CUDA 13.0** world.
- `vllm==0.24.0` (PyPI wheel is already a cu130 / torch-2.11 abi3 build).
- `transformers==5.9.0`, applied project-wide via `override-dependencies`.
- `flash-attn==2.8.3` under the `fsdp` extra, served prebuilt from the `verl-wheelhouse` index
  (no source build) — this is the pin that INC-001 tripped over.
- `ray[default]>=2.41.0`; lock resolves `ray==2.55.1`.
- `mbridge` and `TransferQueue` are git-sourced — the pins that INC-002 tripped over.

**Consequence:** the ambient `torch 2.8.0+cu128` does *not* satisfy the current VeRL contract, and
the node driver (580.105.08, advertising CUDA 13.0) does support the cu130 wheels. The correct
move is therefore the isolated uv venv, **not** `pip install -U torch` into the base env.

`UNVERIFIED` until the smoke test runs: whether FSDP or FSDP2 is the effective sharding strategy
for this example, and the exact point at which trained weights are synced back into the vLLM
rollout engine. Both are to be read from source and confirmed at R0, then written into
[`../analysis/verl_config_explained.md`](../analysis/verl_config_explained.md).
