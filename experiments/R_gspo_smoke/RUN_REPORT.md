# Run Report — R_gspo_smoke

**Run UID:** `R_gspo_smoke-20260911-085807`
**Outcome:** **COMPLETED**
**Duration:** 8.0 min
**Optimizer updates recorded:** 3
**Generated:** 2026-09-11T09:06:04

## Configuration

| | |
|---|---|
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `e891fad9e38228b11b25ca718bab73b8e889c2e2` |
| Seed | 20260910 |
| Started | 2026-09-11T08:58:07 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first -0.0000 -> last -0.0000 (min -0.0000, max 0.0000, mean -0.0000) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2647 -> last 0.2611 (min 0.2611, max 0.3603, mean 0.2954) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.3281 -> last -0.4062 (min -0.4062, max -0.3281, mean -0.3594) |
| reward std | first 0.9446 -> last 0.9138 (min 0.9138, max 0.9446, mean 0.9325) |
| mixed group ratio | first 0.625 -> last 0.500 (min 0.500, max 0.688, mean 0.604) |
| all-correct group ratio | first 0.000 -> last 0.062 (min 0.000, max 0.062, mean 0.021) |
| all-wrong group ratio | first 0.375 -> last 0.438 (min 0.312, max 0.438, mean 0.375) |
| zero-std group ratio | first 0.375 -> last 0.500 (min 0.312, max 0.500, mean 0.396) |
| effective signal fraction | first 0.625 -> last 0.500 (min 0.500, max 0.688, mean 0.604) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first 0.00001 -> last -0.00000 (min -0.00000, max 0.00005, mean 0.00002) |
| entropy | first 0.3576 -> last 0.4437 (min 0.3460, max 0.4437, mean 0.3824) |
| PPO objective clip fraction | first 0.0000 -> last 0.0000 (min 0.0000, max 0.0000, mean 0.0000) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first 0.0078 -> last -0.0028 (min -0.0246, max 0.0078, mean -0.0065) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1556.3 -> last 1809.5 (min 1556.3, max 1809.5, mean 1707.5) |
| response length max | first 5064 -> last 5383 (min 5064, max 7747, mean 6065) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 38.4 % |
| verifier | unavailable |
| logprob | 22.9 % |
| actor update | 31.8 % |
| weight sync | 4.1 % |
| other | 2.9 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 503 -> last 689 (min 503, max 689, mean 600) |
| step wall time (s) | first 108.7 -> last 92.3 (min 92.3, max 108.7, mean 100.6) |
| peak VRAM GPU0 (GiB) | first 87.0 -> last 87.0 (min 87.0, max 87.0, mean 87.0) |

**Policy provenance / staleness:** observed values [0] over 3 updates. For synchronous GRPO this should
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

12 sampled trajectories in `trajectories/audit.jsonl`
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
