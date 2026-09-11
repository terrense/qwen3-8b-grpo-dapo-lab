# Run Report — M20_gspo

**Run UID:** `M20_gspo-20260911-103728`
**Outcome:** **COMPLETED**
**Duration:** 36.4 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T11:13:49

## Configuration

| | |
|---|---|
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `d851cf95e830aeb35b55370a01328d4bbb1eb3cb` |
| Seed | 20260910 |
| Started | 2026-09-11T10:37:28 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first -0.0000 -> last -0.0000 (min -0.0000, max 0.0000, mean -0.0000) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2726 -> last 0.2338 (min 0.1639, max 0.4253, mean 0.2760) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.3594 -> last 0.4375 (min -0.4531, max 0.4375, mean -0.0414) |
| reward std | first 0.9332 -> last 0.8992 (min 0.8914, max 0.9995, mean 0.9708) |
| mixed group ratio | first 0.562 -> last 0.375 (min 0.375, max 0.812, mean 0.594) |
| all-correct group ratio | first 0.062 -> last 0.500 (min 0.000, max 0.500, mean 0.172) |
| all-wrong group ratio | first 0.375 -> last 0.125 (min 0.062, max 0.438, mean 0.234) |
| zero-std group ratio | first 0.438 -> last 0.625 (min 0.188, max 0.625, mean 0.406) |
| effective signal fraction | first 0.562 -> last 0.375 (min 0.375, max 0.812, mean 0.594) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00002 -> last 0.00002 (min -0.00005, max 0.00005, mean -0.00000) |
| entropy | first 0.3618 -> last 0.2759 (min 0.2488, max 0.4387, mean 0.3107) |
| PPO objective clip fraction | first 0.0000 -> last 0.0000 (min 0.0000, max 0.0000, mean 0.0000) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0234 -> last -0.0091 (min -0.0541, max 0.0220, mean -0.0165) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1578.3 -> last 1540.9 (min 1345.3, max 2256.9, mean 1685.1) |
| response length max | first 5797 -> last 5105 (min 3639, max 8192, mean 6104) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 41.3 % |
| verifier | unavailable |
| logprob | 19.4 % |
| actor update | 34.0 % |
| weight sync | 4.8 % |
| other | 0.6 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 479 -> last 672 (min 479, max 717, mean 642) |
| step wall time (s) | first 115.4 -> last 80.5 (min 68.5, max 118.9, mean 92.2) |
| peak VRAM GPU0 (GiB) | first 87.3 -> last 87.3 (min 87.3, max 87.3, mean 87.3) |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-105734-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-105734-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-111054-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-111054-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`

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
