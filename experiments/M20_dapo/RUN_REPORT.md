# Run Report — M20_dapo

**Run UID:** `M20_dapo-20260911-111356`
**Outcome:** **COMPLETED**
**Duration:** 45.6 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T15:48:26

## Configuration

| | |
|---|---|
| Algorithm | DAPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `b72e90e6daef15977d1ae2d84753fcba77eb1e40` |
| Seed | 20260910 |
| Started | 2026-09-11T11:13:56 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0476 -> last 0.0681 (min -0.0007, max 0.1141, mean 0.0528) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.3478 -> last 0.2464 (min 0.2391, max 0.3718, mean 0.2934) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.1781 -> last -0.1329 (min -0.3281, max 0.1562, mean -0.0308) |
| reward std | first 0.9851 -> last 0.9999 (min 0.9332, max 1.0000, mean 0.9805) |
| mixed group ratio | first 1.000 -> last 1.000 (min 0.875, max 1.000, mean 0.981) |
| all-correct group ratio | first 0.000 -> last 0.000 (min 0.000, max 0.062, mean 0.003) |
| all-wrong group ratio | first 0.000 -> last 0.000 (min 0.000, max 0.125, mean 0.016) |
| zero-std group ratio | first 0.000 -> last 0.000 (min 0.000, max 0.125, mean 0.019) |
| effective signal fraction | first 1.000 -> last 1.000 (min 0.875, max 1.000, mean 0.981) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00001 -> last -0.00004 (min -0.00007, max 0.00009, mean -0.00001) |
| entropy | first 0.3562 -> last 0.3145 (min 0.2413, max 0.4423, mean 0.3334) |
| PPO objective clip fraction | first 0.0001 -> last 0.0001 (min 0.0001, max 0.0002, mean 0.0001) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0503 -> last -0.0751 (min -0.1125, max 0.0030, mean -0.0540) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1522.2 -> last 2052.9 (min 1262.9, max 2156.3, mean 1731.2) |
| response length max | first 7773 -> last 8192 (min 3417, max 8192, mean 5891) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 62.3 % |
| verifier | 0.0 % |
| logprob | unavailable |
| actor update | 26.7 % |
| weight sync | 3.3 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 429 -> last 398 (min 398, max 595, mean 501) |
| step wall time (s) | first 125.3 -> last 176.6 (min 86.4, max 176.6, mean 122.1) |
| peak VRAM GPU0 (GiB) | unavailable |

**Policy provenance / staleness:** unavailable. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-114303-T_STEP_ROBUST_Z` — see `incidents/INC-20260911-114303-T_STEP_ROBUST_Z/incident.md`

## Root Causes

PENDING — each incident bundle carries a `Root cause: PENDING` field that a human must complete after investigation. The detector deliberately does not guess.

## Recovery / Fixes

PENDING

## Representative Trajectories

76 sampled trajectories in `trajectories/audit.jsonl`
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
