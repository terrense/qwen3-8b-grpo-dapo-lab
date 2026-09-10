# Run Report — R0_grpo_smoke

**Run UID:** `R0_grpo_smoke-20260910-132022`
**Outcome:** **COMPLETED**
**Duration:** 7.1 min
**Optimizer updates recorded:** 2
**Generated:** 2026-09-10T13:27:28

## Configuration

| | |
|---|---|
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `78b21d80815e9380710950b42a08401bd9b060fe` |
| Seed | 20260910 |
| Started | 2026-09-10T13:20:23 |
| Exit code | `0` |

Full resolved config: `resolved_config.yaml`. Launch command: `command.txt`.

## Outcome

The trainer exited cleanly.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0262 -> last 0.0284 (min 0.0262, max 0.0284, mean 0.0273) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2547 -> last 0.2123 (min 0.2123, max 0.2547, mean 0.2335) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.1875 -> last -0.3750 (min -0.3750, max -0.1875, mean -0.2812) |
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
| KL (`actor/ppo_kl`) | first 0.00000 -> last 0.00000 (min 0.00000, max 0.00000, mean 0.00000) |
| entropy | first 0.3430 -> last 0.3417 (min 0.3417, max 0.3430, mean 0.3424) |
| clip fraction (total) | first 0.0000 -> last 0.0000 (min 0.0000, max 0.0000, mean 0.0000) |
| clip fraction lower | first 0.0000 -> last 0.0000 (min 0.0000, max 0.0000, mean 0.0000) |
| clip fraction upper | first 0.0000 -> last 0.0000 (min 0.0000, max 0.0000, mean 0.0000) |
| advantage mean | first -0.0262 -> last -0.0284 (min -0.0284, max -0.0262, mean -0.0273) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1532.2 -> last 2040.7 (min 1532.2, max 2040.7, mean 1786.5) |
| response length max | first 3509 -> last 4096 (min 3509, max 4096, mean 3802) |
| truncation rate | first 0.0312 -> last 0.0938 (min 0.0312, max 0.0938, mean 0.0625) |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 40.8 % |
| verifier | 0.0 % |
| logprob | 26.6 % |
| actor update | 17.8 % |
| weight sync | 7.9 % |
| other | 6.9 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 207 -> last 418 (min 207, max 418, mean 312) |
| step wall time (s) | first 64.8 -> last 41.8 (min 41.8, max 64.8, mean 53.3) |
| peak VRAM GPU0 (GiB) | first 86.4 -> last 86.4 (min 86.4, max 86.4, mean 86.4) |

**Policy provenance / staleness:** observed values [0, 1] over 2 updates. For synchronous GRPO this should
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

- `figures/01_reward_validation.png`
- `figures/02_policy_dynamics.png`
- `figures/03_ratio_grad.png`
- `figures/04_group_signal.png`
- `figures/05_length_dynamics.png`
- `figures/06_system_timing.png`
- `figures/07_gpu_memory_util.png`

### Metric availability note

Fields that the current VeRL build does not emit are recorded as empty in
`metrics/training_metrics.csv` and rendered as `unavailable` here. They are never
substituted with `0`. See `docs/observability/metric_dictionary.md`.
