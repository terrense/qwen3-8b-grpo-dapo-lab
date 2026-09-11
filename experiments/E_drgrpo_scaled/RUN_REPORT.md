# Run Report — E_drgrpo_scaled

**Run UID:** `E_drgrpo_scaled-20260911-193309`
**Outcome:** **COMPLETED**
**Duration:** 52.7 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T20:25:51

## Configuration

| | |
|---|---|
| Algorithm | DrGRPO_fixed |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `1bfb09dc4d2508e3ed763dca0d4e54685dcaf2a8` |
| Seed | 20260910 |
| Started | 2026-09-11T19:33:09 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0705 -> last 0.0455 (min -0.0178, max 0.1213, mean 0.0308) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.1633 -> last 0.1981 (min 0.1263, max 0.2159, mean 0.1685) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.3438 -> last -0.3125 (min -0.4688, max 0.0781, mean -0.1656) |
| reward std | first 0.9391 -> last 0.9499 (min 0.8833, max 0.9999, mean 0.9734) |
| mixed group ratio | first 0.562 -> last 0.750 (min 0.375, max 0.875, mean 0.594) |
| all-correct group ratio | first 0.062 -> last 0.000 (min 0.000, max 0.250, mean 0.119) |
| all-wrong group ratio | first 0.375 -> last 0.250 (min 0.062, max 0.500, mean 0.287) |
| zero-std group ratio | first 0.438 -> last 0.250 (min 0.125, max 0.625, mean 0.406) |
| effective signal fraction | first 0.562 -> last 0.750 (min 0.375, max 0.875, mean 0.594) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00002 -> last 0.00003 (min -0.00006, max 0.00004, mean -0.00001) |
| entropy | first 0.3272 -> last 0.3244 (min 0.2210, max 0.3891, mean 0.3018) |
| PPO objective clip fraction | first 0.0001 -> last 0.0001 (min 0.0000, max 0.0002, mean 0.0001) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0758 -> last -0.0410 (min -0.1138, max 0.0165, mean -0.0319) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1767.9 -> last 2102.7 (min 1587.1, max 2266.7, mean 1845.7) |
| response length max | first 7971 -> last 8192 (min 4477, max 8192, mean 7288) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 43.2 % |
| verifier | unavailable |
| logprob | 18.8 % |
| actor update | 33.5 % |
| weight sync | 4.0 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 474 -> last 620 (min 474, max 721, mean 626) |
| step wall time (s) | first 130.5 -> last 115.9 (min 82.3, max 130.5, mean 103.2) |
| peak VRAM GPU0 (GiB) | first 87.2 -> last 87.2 (min 87.2, max 87.2, mean 87.2) |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-194754-KL_ROBUST_Z` — see `incidents/INC-20260911-194754-KL_ROBUST_Z/incident.md`
- `INC-20260911-194954-T_ROLLOUT_ROBUST_Z` — see `incidents/INC-20260911-194954-T_ROLLOUT_ROBUST_Z/incident.md`

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
