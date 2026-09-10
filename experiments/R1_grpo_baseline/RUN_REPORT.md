# Run Report — R1_grpo_baseline

**Run UID:** `R1_grpo_baseline-20260910-134534`
**Outcome:** **COMPLETED**
**Duration:** 36.4 min
**Optimizer updates recorded:** 20
**Generated:** 2026-09-10T14:21:57

## Configuration

| | |
|---|---|
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `4382441ca545481b6a141449eab73b2093c2a686` |
| Seed | 20260910 |
| Started | 2026-09-10T13:45:35 |
| Exit code | `0` |

Full resolved config: `resolved_config.yaml`. Launch command: `command.txt`.

## Outcome

The trainer exited cleanly.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first -0.0138 -> last 0.0374 (min -0.0212, max 0.0401, mean 0.0101) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2714 -> last 0.2783 (min 0.2011, max 0.5185, mean 0.2776) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.5000 -> last 0.2656 (min -0.5000, max 0.3438, mean -0.0641) |
| reward std | first 0.8660 -> last 0.9641 (min 0.8660, max 1.0000, mean 0.9667) |
| mixed group ratio | first 0.625 -> last 0.562 (min 0.500, max 0.812, mean 0.637) |
| all-correct group ratio | first 0.000 -> last 0.250 (min 0.000, max 0.375, mean 0.147) |
| all-wrong group ratio | first 0.375 -> last 0.188 (min 0.062, max 0.375, mean 0.216) |
| zero-std group ratio | first 0.375 -> last 0.438 (min 0.188, max 0.500, mean 0.362) |
| effective signal fraction | first 0.625 -> last 0.562 (min 0.500, max 0.812, mean 0.637) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00002 -> last 0.00004 (min -0.00005, max 0.00004, mean -0.00001) |
| entropy | first 0.3585 -> last 0.3384 (min 0.2255, max 0.4432, mean 0.3307) |
| clip fraction (total) | first 0.0001 -> last 0.0001 (min 0.0001, max 0.0003, mean 0.0002) |
| clip fraction lower | first 0.0000 -> last 0.0000 (min 0.0000, max 0.0000, mean 0.0000) |
| clip fraction upper | first 0.0001 -> last 0.0001 (min 0.0001, max 0.0003, mean 0.0002) |
| advantage mean | first -0.0434 -> last -0.0778 (min -0.0778, max 0.0350, mean -0.0173) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1683.8 -> last 1508.3 (min 1192.2, max 2188.1, mean 1674.4) |
| response length max | first 7068 -> last 8192 (min 3820, max 8192, mean 6027) |
| truncation rate | first 0.0078 -> last 0.0078 (min 0.0078, max 0.0234, mean 0.0086) |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 41.0 % |
| verifier | 0.0 % |
| logprob | 19.6 % |
| actor update | 34.3 % |
| weight sync | 4.7 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 463 -> last 563 (min 463, max 718, mean 650) |
| step wall time (s) | first 126.9 -> last 94.3 (min 64.0, max 126.9, mean 91.2) |
| peak VRAM GPU0 (GiB) | first 87.3 -> last 87.3 (min 87.3, max 87.3, mean 87.3) |

**Policy provenance / staleness:** observed values [0] over 20 updates. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

- `INC-20260910-140540-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260910-140540-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260910-141140-RESPONSE_LENGTH_MEAN_ROBUST_Z` — see `incidents/INC-20260910-141140-RESPONSE_LENGTH_MEAN_ROBUST_Z/incident.md`
- `INC-20260910-141700-GRAD_NORM_ROBUST_Z` — see `incidents/INC-20260910-141700-GRAD_NORM_ROBUST_Z/incident.md`

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
