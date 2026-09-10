# Qwen3-8B GRPO & DAPO Lab

Hands-on reinforcement learning systems experiments with Qwen3-8B, VeRL, GRPO and DAPO.

This repository documents a hands-on reinforcement learning systems study using Qwen3-8B, VeRL,
GRPO, and DAPO. The objective is not only to improve benchmark scores, but to understand the
actual training dynamics and failure modes of large-language-model reinforcement learning.

> **Scope note.** Everything here is a **GRPO systems pilot / rehearsal**. It starts from the
> released post-trained `Qwen/Qwen3-8B` and applies GRPO directly. It is **not** the formal `M3`
> checkpoint of the author's medical paper, whose design is
> `Qwen3-8B -> SFT (M1) -> DPO (M2) -> GRPO (M3)`. No result in this repository may be presented
> as that `M3`.

---

## Setup

| | |
|---|---|
| **Model** | `Qwen/Qwen3-8B` (post-trained release, not `-Base`, not VL) |
| **Framework** | [VeRL](https://github.com/verl-project/verl) |
| **Training data** | `BytedTsinghua-SIA/DAPO-Math-17k` |
| **Validation** | AIME 2024 |
| **Hardware** | 4 x NVIDIA H20 96GB, full NVLink mesh |

Full hardware and environment records: [`docs/hardware_environment.md`](docs/hardware_environment.md),
[`manifests/hardware.txt`](manifests/hardware.txt), [`manifests/versions.txt`](manifests/versions.txt).

---

## Experiment sequence

| Run | Purpose |
|---|---|
| **R0** — `experiments/R0_grpo_smoke` | GRPO smoke test: prove the end-to-end RL pipeline, 2-5 optimizer updates |
| **R1** — `experiments/R1_grpo_baseline` | Vanilla GRPO baseline, long enough to read the dynamics |
| **R2** — `experiments/R2_dapo` | DAPO matched-budget experiment vs. R1 |
| **R3** — `experiments/R3_truncation` | Failure injection: truncation contamination |
| **R3** — `experiments/R3_high_lr` | Failure injection: excessive learning rate |
| **R3** — `experiments/R3_verifier_timeout` | Failure injection: verifier timeout contamination |

---

## Primary goal

The primary goal is to build practical RL systems intuition by observing:

- rollout behavior
- reward distribution
- group-relative advantages
- zero-variance groups
- old-policy log probabilities
- current-policy log probabilities
- importance ratios
- clipping
- KL divergence
- entropy
- gradient norms
- effective batch size
- dynamic sampling
- response-length dynamics
- KV-cache pressure
- throughput bottlenecks
- checkpoint/resume behavior
- verifier failures
- reward contamination
- reward hacking
- infrastructure failures

**A successful run is not defined only by a higher reward. The experiment must explain *why* the
reward changes.**

A large fraction of what looks like "the model failed to train" is in reality a systems, data, or
verifier fault. Separating those two categories is the point of this lab, and is why
[`analysis/incident_log.md`](analysis/incident_log.md) is treated as a primary deliverable rather
than a scratch file.

---

## Current Status

| Stage | Status |
|---|---|
| Hardware preflight | **PASS** |
| GPU | **4 x NVIDIA H20 96GB** |
| NVLink | **PASS** |
| BF16 | **PASS** |
| NCCL correctness | **PASS** |
| NCCL bandwidth | **PASS** (351 GB/s busBW @ 1GB all-reduce) |
| VeRL environment | **IN PROGRESS** |
| Dataset | **IN PROGRESS** |
| R0 GRPO smoke | **NOT RUN** |
| R1 GRPO baseline | **NOT RUN** |
| R2 DAPO | **NOT RUN** |
| R3 failure injection | **NOT RUN** |

---

## Repository layout

```
docs/          experiment plan, RL pipeline notes, debugging playbook, hardware record
configs/       grpo/ dapo/ failure_injection/ -- resolved training configurations
scripts/       preflight/ data/ training/ evaluation/ metrics/ plotting/
experiments/   one directory per run: README + resolved config + results summary
analysis/      incident log, verifier unit test, reward distribution, R1-vs-R2, final report
figures/       generated/ -- all figures regenerable from the committed metric files
manifests/     hardware, versions, dataset, model records
```

Raw datasets, model weights, optimizer checkpoints, HuggingFace cache and large rollout dumps are
**not** committed. They are referenced by run ID, step, and path/hash in the manifests.

---

## Reproducibility contract

Every run records: run id, VeRL git commit, model revision, dataset revision, seed, the fully
resolved config, a hardware snapshot, the environment lock, checkpoint IDs, and restart
boundaries. Restarted runs are **never** stitched into a single continuous curve — a restart
always leaves a visible boundary in the metrics.
