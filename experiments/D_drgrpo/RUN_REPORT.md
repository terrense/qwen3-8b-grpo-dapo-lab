# Run Report — D_drgrpo

**Run UID:** `D_drgrpo-20260911-183747`
**Outcome:** **COMPLETED**
**Duration:** 52.8 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T19:30:32

## Configuration

| | |
|---|---|
| Algorithm | DrGRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `0e82a1c26e505ebe4306730e249def01448c9ba7` |
| Seed | 20260910 |
| Started | 2026-09-11T18:37:47 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0175 -> last 0.0086 (min -0.0025, max 0.0282, mean 0.0056) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.0420 -> last 0.0640 (min 0.0274, max 0.1232, mean 0.0457) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.3281 -> last -0.1719 (min -0.4531, max 0.1406, mean -0.1180) |
| reward std | first 0.9446 -> last 0.9851 (min 0.8914, max 0.9999, mean 0.9795) |
| mixed group ratio | first 0.625 -> last 0.688 (min 0.312, max 0.938, mean 0.622) |
| all-correct group ratio | first 0.062 -> last 0.062 (min 0.000, max 0.250, mean 0.122) |
| all-wrong group ratio | first 0.312 -> last 0.250 (min 0.000, max 0.500, mean 0.256) |
| zero-std group ratio | first 0.375 -> last 0.312 (min 0.062, max 0.688, mean 0.378) |
| effective signal fraction | first 0.625 -> last 0.688 (min 0.312, max 0.938, mean 0.622) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00001 -> last -0.00006 (min -0.00006, max 0.00005, mean 0.00001) |
| entropy | first 0.0719 -> last 0.1057 (min 0.0588, max 0.1208, mean 0.0794) |
| PPO objective clip fraction | first 0.0001 -> last 0.0002 (min 0.0000, max 0.0003, mean 0.0002) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0824 -> last -0.0296 (min -0.1222, max 0.0115, mean -0.0234) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1738.8 -> last 2363.2 (min 1596.9, max 2494.9, mean 1906.9) |
| response length max | first 6164 -> last 8192 (min 4452, max 8192, mean 6619) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 41.7 % |
| verifier | unavailable |
| logprob | 19.4 % |
| actor update | 34.3 % |
| weight sync | 4.0 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 505 -> last 632 (min 505, max 717, mean 650) |
| step wall time (s) | first 120.7 -> last 126.9 (min 83.5, max 131.6, mean 102.3) |
| peak VRAM GPU0 (GiB) | first 87.4 -> last 87.4 (min 87.4, max 87.4, mean 87.4) |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-190913-T_ROLLOUT_ROBUST_Z` — see `incidents/INC-20260911-190913-T_ROLLOUT_ROBUST_Z/incident.md`
- `INC-20260911-191133-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-191133-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-191133-T_ROLLOUT_ROBUST_Z` — see `incidents/INC-20260911-191133-T_ROLLOUT_ROBUST_Z/incident.md`
- `INC-20260911-191653-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-191653-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`

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
