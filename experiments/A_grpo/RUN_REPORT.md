# Run Report — A_grpo

**Run UID:** `A_grpo-20260911-155511`
**Outcome:** **COMPLETED**
**Duration:** 53.6 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T16:48:45

## Configuration

| | |
|---|---|
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `47bea07aa92ba89be41f895def9b3c27c13c89a0` |
| Seed | 20260910 |
| Started | 2026-09-11T15:55:11 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0247 -> last 0.0135 (min -0.0163, max 0.0503, mean 0.0130) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2020 -> last 0.2504 (min 0.1325, max 0.3267, mean 0.2395) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.3750 -> last -0.1562 (min -0.4688, max 0.0938, mean -0.1336) |
| reward std | first 0.9270 -> last 0.9877 (min 0.8833, max 0.9999, mean 0.9764) |
| mixed group ratio | first 0.562 -> last 0.750 (min 0.188, max 0.875, mean 0.622) |
| all-correct group ratio | first 0.000 -> last 0.125 (min 0.000, max 0.375, mean 0.116) |
| all-wrong group ratio | first 0.438 -> last 0.125 (min 0.000, max 0.500, mean 0.263) |
| zero-std group ratio | first 0.438 -> last 0.250 (min 0.125, max 0.812, mean 0.378) |
| effective signal fraction | first 0.562 -> last 0.750 (min 0.188, max 0.875, mean 0.622) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00002 -> last -0.00003 (min -0.00006, max 0.00004, mean -0.00001) |
| entropy | first 0.3428 -> last 0.3272 (min 0.2604, max 0.4025, mean 0.3332) |
| PPO objective clip fraction | first 0.0001 -> last 0.0002 (min 0.0001, max 0.0002, mean 0.0001) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0385 -> last -0.0145 (min -0.0804, max 0.0241, mean -0.0296) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1713.7 -> last 2353.4 (min 1597.6, max 2578.6, mean 1895.2) |
| response length max | first 4409 -> last 8192 (min 4409, max 8192, mean 6650) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 41.6 % |
| verifier | unavailable |
| logprob | 19.5 % |
| actor update | 34.4 % |
| weight sync | 4.0 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 533 -> last 617 (min 533, max 718, mean 650) |
| step wall time (s) | first 112.9 -> last 129.5 (min 80.8, max 135.8, mean 101.8) |
| peak VRAM GPU0 (GiB) | first 87.4 -> last 87.4 (min 87.4, max 87.4, mean 87.4) |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-162918-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-162918-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-163058-EFFECTIVE_SIGNAL_LOW` — see `incidents/INC-20260911-163058-EFFECTIVE_SIGNAL_LOW/incident.md`
- `INC-20260911-163058-ZERO_STD_GROUP_RATIO_ABOVE_HEURISTIC` — see `incidents/INC-20260911-163058-ZERO_STD_GROUP_RATIO_ABOVE_HEURISTIC/incident.md`
- `INC-20260911-163458-EFFECTIVE_SIGNAL_LOW` — see `incidents/INC-20260911-163458-EFFECTIVE_SIGNAL_LOW/incident.md`
- `INC-20260911-163458-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-163458-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-163458-ZERO_STD_GROUP_RATIO_ABOVE_HEURISTIC` — see `incidents/INC-20260911-163458-ZERO_STD_GROUP_RATIO_ABOVE_HEURISTIC/incident.md`
- `INC-20260911-164818-TRAINING_STALL` — see `incidents/INC-20260911-164818-TRAINING_STALL/incident.md`
- `INC-20260911-164838-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-164838-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`

## Root Causes

PENDING — each incident bundle carries a `Root cause: PENDING` field that a human must complete after investigation. The detector deliberately does not guess.

## Recovery / Fixes

PENDING

## Representative Trajectories

80 sampled trajectories in `trajectories/audit.jsonl`
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
