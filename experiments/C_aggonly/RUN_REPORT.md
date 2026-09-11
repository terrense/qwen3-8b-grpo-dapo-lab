# Run Report — C_aggonly

**Run UID:** `C_aggonly-20260911-174244`
**Outcome:** **COMPLETED**
**Duration:** 55.0 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T18:37:41

## Configuration

| | |
|---|---|
| Algorithm | GRPO_seqagg |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `728221140a318aa293a9805169b76578a7609e41` |
| Seed | 20260910 |
| Started | 2026-09-11T17:42:44 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0000 -> last 0.0000 (min -0.0000, max 0.0001, mean 0.0000) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.1986 -> last 0.1908 (min 0.1755, max 0.3546, mean 0.2439) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.2500 -> last -0.1094 (min -0.3750, max 0.2031, mean -0.1070) |
| reward std | first 0.9682 -> last 0.9940 (min 0.9270, max 0.9999, mean 0.9814) |
| mixed group ratio | first 0.500 -> last 0.562 (min 0.438, max 0.875, mean 0.625) |
| all-correct group ratio | first 0.125 -> last 0.188 (min 0.000, max 0.250, mean 0.138) |
| all-wrong group ratio | first 0.375 -> last 0.250 (min 0.000, max 0.500, mean 0.237) |
| zero-std group ratio | first 0.500 -> last 0.438 (min 0.125, max 0.562, mean 0.375) |
| effective signal fraction | first 0.500 -> last 0.562 (min 0.438, max 0.875, mean 0.625) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00001 -> last -0.00002 (min -0.00005, max 0.00005, mean 0.00000) |
| entropy | first 0.3550 -> last 0.3124 (min 0.2569, max 0.3738, mean 0.3177) |
| PPO objective clip fraction | first 0.0001 -> last 0.0001 (min 0.0001, max 0.0003, mean 0.0001) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0195 -> last 0.0036 (min -0.0740, max 0.0418, mean -0.0183) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1659.2 -> last 2520.0 (min 1497.7, max 2755.6, mean 1945.3) |
| response length max | first 4381 -> last 8192 (min 4381, max 8192, mean 6876) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 42.1 % |
| verifier | unavailable |
| logprob | 19.3 % |
| actor update | 34.1 % |
| weight sync | 3.9 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 525 -> last 637 (min 525, max 706, mean 643) |
| step wall time (s) | first 111.2 -> last 133.9 (min 79.5, max 144.6, mean 105.0) |
| peak VRAM GPU0 (GiB) | first 87.7 -> last 87.7 (min 87.7, max 87.7, mean 87.7) |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-181729-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-181729-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-182329-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-182329-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-183649-TRAINING_STALL` — see `incidents/INC-20260911-183649-TRAINING_STALL/incident.md`

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
