# Run Report — M20_grpo

**Run UID:** `M20_grpo-20260911-100052`
**Outcome:** **COMPLETED**
**Duration:** 36.5 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-11T15:48:23

## Configuration

| | |
|---|---|
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `94f366fa8ec86c29fde074cd8824451a2034a085` |
| Seed | 20260910 |
| Started | 2026-09-11T10:00:52 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0082 -> last 0.0258 (min -0.0185, max 0.0621, mean 0.0098) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2370 -> last 0.2736 (min 0.1772, max 0.3862, mean 0.2528) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.4688 -> last 0.3281 (min -0.5156, max 0.3281, mean -0.0266) |
| reward std | first 0.8833 -> last 0.9446 (min 0.8568, max 1.0000, mean 0.9696) |
| mixed group ratio | first 0.625 -> last 0.562 (min 0.438, max 0.875, mean 0.609) |
| all-correct group ratio | first 0.000 -> last 0.312 (min 0.000, max 0.312, mean 0.163) |
| all-wrong group ratio | first 0.375 -> last 0.125 (min 0.062, max 0.438, mean 0.228) |
| zero-std group ratio | first 0.375 -> last 0.438 (min 0.125, max 0.562, mean 0.391) |
| effective signal fraction | first 0.625 -> last 0.562 (min 0.438, max 0.875, mean 0.609) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00000 -> last 0.00000 (min -0.00005, max 0.00004, mean 0.00000) |
| entropy | first 0.3527 -> last 0.2986 (min 0.2358, max 0.4283, mean 0.3171) |
| PPO objective clip fraction | first 0.0001 -> last 0.0002 (min 0.0001, max 0.0003, mean 0.0002) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first 0.0050 -> last -0.0318 (min -0.0935, max 0.0436, mean -0.0176) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1666.9 -> last 1429.8 (min 1279.2, max 2299.9, mean 1677.1) |
| response length max | first 5980 -> last 4820 (min 4734, max 8192, mean 6468) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 42.4 % |
| verifier | unavailable |
| logprob | 19.1 % |
| actor update | 33.6 % |
| weight sync | 4.3 % |
| other | 0.6 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 492 -> last 690 (min 492, max 698, mean 635) |
| step wall time (s) | first 118.3 -> last 73.3 (min 70.1, max 122.4, mean 93.1) |
| peak VRAM GPU0 (GiB) | unavailable |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260911-102058-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-102058-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260911-103438-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260911-103438-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`

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
