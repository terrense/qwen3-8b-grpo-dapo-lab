# Pre-RL Gate Report

**Date:** 2026-09-10 · **Node:** AutoDL single node, 4 x NVIDIA H20 96GB (host identifiers redacted)

**FINAL DECISION: `READY_FOR_VERIFIER_AND_ROLLOUT`**

Every claim below was produced by a command actually executed on this node. Anything not executed
is marked **NOT RUN** and must not be cited as a result.

---

## 1. Hardware gate — **PASS**

## 2. GPU — **4 x NVIDIA H20**, 97871 MiB each (~95.6 GiB; torch reports 95.07 GiB), CC 9.0 (sm90), 500 W, all idle at start

## 3. NVLink — **PASS**
Full **NV18 mesh**: every GPU pair bonded by 18 NVLinks at 26.562 GB/s, no PCIe/SYS hop.
NUMA split: GPU0/1 -> node 0 (CPU 0-63), GPU2/3 -> node 1 (CPU 64-127).

## 4. NCCL correctness — **PASS**
4-rank `all_reduce`, result 10.0 == expected 10.0 on all ranks.

## 5. NCCL bandwidth — **PASS**
bfloat16 all-reduce, 8 warmup + 25 timed iters, busBW = algBW x 2(N-1)/N:

| payload | ambient torch 2.8.0+cu128 | **VeRL venv torch 2.11.0+cu130** |
|---|---|---|
| 64 MB | 309.4 GB/s | 306.7 GB/s |
| 256 MB | 331.8 GB/s | 337.2 GB/s |
| 1024 MB | **351.1 GB/s** | **349.8 GB/s** |
| rank spread @1GB | 0.01 % | 0.04 % |

**Reasonable:** ~78 % of the H20 NVLink unidirectional ceiling, and the two CUDA worlds agree to
within 0.4 % — so the cu130 stack drives NVLink exactly as well as the validated cu128 baseline.
A degraded/PCIe-fallback node would read single-digit GB/s here.

## 6. RAM — total ~1.2 TiB, available ~1.2 TiB, no swap
(`free` shows ~229 GiB "free" with ~962 GiB reclaimable page cache; **available** is the real number.)

## 7. `/dev/shm` — **300 GB**. No Ray object-store or vLLM workaround needed.

## 8. Storage

| Mount | FS | Total | Used | Avail |
|---|---|---|---|---|
| `/root/autodl-tmp` (lab) | xfs on `/dev/vdb`, rw, prjquota | **1.3 TB** | 39 GB | **1.2 TB** |
| `/` (system) | overlay | **30 GB** | **358 MB (2 %)** | 30 GB |

System disk is at **2 %** — every cache (HF, pip, uv, triton, torch-ext, Ray, wandb, vLLM) is
redirected to the data disk. `/autodl-pub` is mounted **read-only**, so it cannot be mistaken for
the private disk. Data-disk sequential write (2 GiB, `oflag=direct`): **226 MB/s**.

## 9. Environment — isolated, ambient untouched

| | |
|---|---|
| Python | 3.12.3 |
| **Torch** | **2.11.0+cu130** |
| **torch CUDA** | **13.0** |
| Driver | 580.105.08 (advertises CUDA 13.0) |
| NCCL | 2.28.9 |
| **vLLM** | **0.24.0** |
| transformers | 5.9.0 |
| Ray | 2.55.1 |
| flash-attn | 2.8.3 |
| flashinfer-python | 0.6.12 |
| triton | 3.6.0 |
| tensordict | 0.10.0 |
| verl | 0.10.0.dev0 |
| **VeRL commit** | **`1252cc71aa5bd82e5604322064d69bfe6454c660`** (main, v0.9.0 + 113) |
| verl-recipe commit | `7f14b203934a981664b295696e930f9276825bf0` |
| DAPO commit | `33fe3176f0bb212588e84fc8ccf50dd554975144` |

Built with `uv sync --frozen --all-packages --extra vllm --extra fsdp` into
`repos/verl/.venv` (11 GB, data disk). **The ambient conda torch 2.8.0+cu128 was never modified** —
it still reports 2.8.0+cu128. Full package list: [`../manifests/pip_freeze.txt`](../manifests/pip_freeze.txt) (256 packages).

### CU130 runtime validation

| Gate | Result |
|---|---|
| A. torch 2.11+cu130 sees 4 GPUs, BF16 4096x4096 matmul | **PASS** |
| B. import torch / transformers / vllm / ray / flash_attn / verl | **PASS** — no ABI or undefined-symbol errors |
| C. FlashAttention **kernel launch** on sm90 | **PASS** — bf16 + fp16, seqlen 128/2048/8192, max abs diff vs SDPA reference **0.0078** |
| D. 4-rank NCCL all-reduce under cu130 | **PASS** — 349.8 GB/s busBW |
| E. vLLM Qwen3-8B bf16 load + generate | **PASS** — load 118.3 s, 4 prompts x 64 tokens in 0.54 s, coherent output |

**`CU130_STACK_VALIDATED = PASS`.** Peak GPU memory during the vLLM test: **60.11 GiB on GPU0**
(`gpu_memory_utilization=0.60`, `max_model_len=2048`, TP=1), 40.0 GiB of it KV cache
(291,248 tokens). GPUs returned to 0 MiB after the test — clean teardown.

## 10. Qwen3-8B — **downloaded**
Path `$LAB/models/Qwen3-8B`, 16 GB, 5 safetensors shards, post-trained release (not `-Base`, not
VL). Details: [`../manifests/model.md`](../manifests/model.md).

## 11. DAPO-Math-17k — **downloaded, schema confirmed**
`$LAB/data/raw/DAPO-Math-17k/data/dapo-math-17k.parquet`, 299,363,855 bytes,
sha256 `534375d6bb8630d2...`.

**1,791,700 rows but only 17,917 unique prompts — every prompt repeated exactly 100x.**
Schema: `data_source`, `prompt` (chat list), `ability`, `reward_model{ground_truth, style}`,
`extra_info{index}`. Details and consequences: [`../manifests/dataset.md`](../manifests/dataset.md).

## 12. AIME 2024 — **NOT PREPARED**

## 13. Official Qwen3-8B VeRL GRPO example — **found**
`repos/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh`. Defaults transcribed in
[`../docs/rl_pipeline_notes.md`](../docs/rl_pipeline_notes.md): `adv_estimator=grpo`,
`train_batch_size=1024`, `ppo_mini_batch_size=256`, `max_prompt_length=1024`,
`max_response_length=2048`, `rollout.n=5`, `rollout_tp=2`, `gpu_memory_utilization=0.6`,
`actor lr=1e-6`, `use_kl_loss=True` + `kl_loss_coef=0.001` (`low_var_kl`),
`use_kl_in_reward=False`, `entropy_coeff=0`, `use_dynamic_bsz=True`,
`ppo_max_token_len_per_gpu=24576`, actor offload off / ref `param_offload=True`.

---

## 14. Risks and unresolved problems

1. **Dataset 100x repetition (highest priority).** Epoch and batch semantics are off by 100x if
   the raw file is used naively, and duplicate prompts inside one GRPO batch silently double-weight
   those prompts. Must be resolved in data prep before R1.
2. **AIME 2024 not prepared.** No validation set yet; G2 is therefore only half-cleared.
3. **`uv.lock` is locally modified** — PyPI URLs retargeted to the Aliyun mirror (INC-003). Versions
   and all 1413 sha256 hashes are unchanged and still verified; upstream copy saved at
   `system/uv.lock.upstream.bak` and revertible via `git checkout uv.lock`. This must be stated in
   any reproducibility claim.
4. **Network is split-routed and the proxy is flaky** (INC-001, INC-002). Any future download step
   needs the right routing or it will fail in a way that looks like a dependency error.
5. **Data-disk write is 226 MB/s.** Full checkpoints will cost minutes; the 2-checkpoint policy is
   not optional.
6. **`verl` reports `0.10.0.dev0` from a moving `main`.** The exact commit is pinned in the
   manifest, but `main` is not a release tag — noted deliberately, not accidentally.
7. **Not yet verified in source** (`UNVERIFIED`, to be settled at R0): FSDP vs FSDP2 for this
   example, and exactly when trained weights are synced into the vLLM rollout engine.

## 15. Final decision

**`READY_FOR_VERIFIER_AND_ROLLOUT`**

Hardware, interconnect, storage and the full CUDA-13 software stack are validated by execution, not
by inspection. Model and training data are on disk with confirmed schemas. The immediate next steps
are AIME 2024 preparation, prompt-level dedup handling for DAPO-Math-17k, and the verifier unit
test — **no long GRPO run may start before the verifier gate passes.**

Still **NOT RUN**: verifier unit test, fixed-policy reward distribution, R0, R1, R2, R3.
