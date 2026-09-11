# Run Report — B_gspo

**Run UID:** `B_gspo-20260911-164853`
**Outcome:** **COMPLETED**
**Duration:** 53.8 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T17:42:38

## Configuration

| | |
|---|---|
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `317ffd8665bdaef99c26dc0d1a02347d8ff64d0b` |
| Seed | 20260910 |
| Started | 2026-09-11T16:48:53 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0000 -> last -0.0000 (min -0.0000, max 0.0000, mean -0.0000) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2982 -> last 0.2679 (min 0.1547, max 0.3514, mean 0.2469) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.2812 -> last -0.1875 (min -0.4219, max 0.1250, mean -0.1062) |
| reward std | first 0.9596 -> last 0.9823 (min 0.9067, max 1.0000, mean 0.9803) |
| mixed group ratio | first 0.625 -> last 0.688 (min 0.375, max 1.000, mean 0.591) |
| all-correct group ratio | first 0.000 -> last 0.062 (min 0.000, max 0.312, mean 0.141) |
| all-wrong group ratio | first 0.375 -> last 0.250 (min 0.000, max 0.500, mean 0.269) |
| zero-std group ratio | first 0.375 -> last 0.312 (min 0.000, max 0.625, mean 0.409) |
| effective signal fraction | first 0.625 -> last 0.688 (min 0.375, max 1.000, mean 0.591) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first 0.00003 -> last 0.00004 (min -0.00007, max 0.00004, mean -0.00001) |
| entropy | first 0.3596 -> last 0.3189 (min 0.2678, max 0.3903, mean 0.3211) |
| PPO objective clip fraction | first 0.0000 -> last 0.0000 (min 0.0000, max 0.0000, mean 0.0000) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0483 -> last -0.0179 (min -0.0918, max 0.0128, mean -0.0322) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1715.5 -> last 2415.2 (min 1564.8, max 2552.3, mean 1879.3) |
| response length max | first 5667 -> last 8192 (min 4724, max 8192, mean 7088) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 42.9 % |
| verifier | unavailable |
| logprob | 19.0 % |
| actor update | 33.5 % |
| weight sync | 4.0 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 510 -> last 637 (min 510, max 724, mean 631) |
| step wall time (s) | first 118.0 -> last 128.6 (min 81.2, max 130.8, mean 103.8) |
| peak VRAM GPU0 (GiB) | first 87.4 -> last 87.4 (min 87.4, max 87.4, mean 87.4) |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-172239-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-172239-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-172239-T_STEP_ROBUST_Z` — see `incidents/INC-20260911-172239-T_STEP_ROBUST_Z/incident.md`
- `INC-20260911-172819-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-172819-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-172819-T_STEP_ROBUST_Z` — see `incidents/INC-20260911-172819-T_STEP_ROBUST_Z/incident.md`
- `INC-20260911-174139-TRAINING_STALL` — see `incidents/INC-20260911-174139-TRAINING_STALL/incident.md`

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
