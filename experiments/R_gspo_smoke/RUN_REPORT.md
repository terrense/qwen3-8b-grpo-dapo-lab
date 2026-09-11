# Run Report — R_gspo_smoke

**Run UID:** `R_gspo_smoke-20260911-085513`
**Outcome:** **FAILED (exit 1)**
**Duration:** 0.3 min
**Optimizer updates recorded:** 0
**Generated:** 2026-09-11T08:55:28

## Configuration

| | |
|---|---|
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `69df5954856c09e9a66a4863de191488a078292c` |
| Seed | 20260910 |
| Started | 2026-09-11T08:55:13 |
| Exit code | `1` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The trainer exited non-zero (`1`). The exit code is preserved verbatim by the launcher and is NOT swallowed.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | unavailable |
| learning rate | unavailable |
| grad norm | unavailable |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | unavailable |
| reward std | unavailable |
| mixed group ratio | unavailable |
| all-correct group ratio | unavailable |
| all-wrong group ratio | unavailable |
| zero-std group ratio | unavailable |
| effective signal fraction | unavailable |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | unavailable |
| entropy | unavailable |
| PPO objective clip fraction | unavailable |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | unavailable |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | unavailable |
| response length max | unavailable |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | unavailable |
| verifier | unavailable |
| logprob | unavailable |
| actor update | unavailable |
| weight sync | unavailable |
| other | unavailable |

| metric | trajectory |
|---|---|
| tokens/sec | unavailable |
| step wall time (s) | unavailable |
| peak VRAM GPU0 (GiB) | unavailable |

**Policy provenance / staleness:** unavailable. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

None detected by the flight recorder during this run.

## Root Causes

Not applicable — no incidents.

## Recovery / Fixes

Not applicable.

## Representative Trajectories

0 sampled trajectories in `trajectories/audit.jsonl`
(high-reward / low-reward / mixed-group / longest per sampled update).

## What Changed During the Run

No restarts. Metrics are never stitched across a restart.

## What We Learned

TO BE COMPLETED BY A HUMAN — cite steps, metrics, figures and incident IDs.

## Open Questions

TO BE COMPLETED BY A HUMAN.

## Next Experiment

TO BE DECIDED.

---

### Figures

- `figures/07_gpu_memory_util.png`

### Metric availability note

Fields that the current VeRL build does not emit are recorded as empty in
`metrics/training_metrics.csv` and rendered as `unavailable` here. They are never
substituted with `0`. See `docs/observability/metric_dictionary.md`.
