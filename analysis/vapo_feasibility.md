# VAPO Implementation Feasibility Report

**Verdict: `DO NOT IMPLEMENT FROM SCRATCH ON PAID H20 TIME — YET.`**
VAPO is **NOT IMPLEMENTED** in the current stack. This report inventories what would be
required and what VeRL already supplies, so the decision is made from evidence rather than
from enthusiasm. No VAPO training run is scheduled.

Paper: *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks*,
[arXiv 2504.05118](https://arxiv.org/pdf/2504.05118).

## Availability audit (verl `1252cc71`, verl-recipe @ `7f14b203`)

| searched | result |
|---|---|
| `grep -rni "vapo"` in `repos/verl` (py/yaml/md) | **1 hit**, `README.md:76` — a citation of the paper in the news list |
| `grep -rni "vapo"` in `repos/verl-recipe` | **0 hits** |
| VAPO advantage estimator in `AdvantageEstimator` | **absent** |
| VAPO policy loss in the `@register_policy_loss` registry | **absent** |
| VAPO recipe directory | **absent** |

The registry contains `gae, grpo, grpo_passk, grpo_vectorized, rloo, rloo_vectorized, opo,
gdpo, gpg, remax, reinforce_plus_plus, reinforce_plus_plus_baseline,
optimal_token_baseline, tir_optimal_token_baseline` and policy losses
`vanilla, gspo, cispo, clip_cov, kl_cov, geo_mean, gpg, dro, sapo, dppo_kl, dppo_tv,
bypass_mode`. **No VAPO in either.**

So VAPO would be a **from-scratch implementation**, not a config change — categorically
different from Dr.GRPO (a two-key config change), DAPO (an existing recipe) and GSPO (an
already-implemented policy loss).

## What VeRL already provides

The critic path is real and maintained — VAPO does not start from nothing:

| VAPO component | VeRL status | Where |
|---|---|---|
| Critic architecture | **available** — full critic worker, FSDP + Megatron | `verl/workers/critic/`, `critic.*` config tree |
| Critic initialization | **available** — `critic.model.path` (typically the actor init) | `_generated_ppo_trainer.yaml` critic block |
| Value loss | **available** | `compute_value_loss`, `core_algos.py` |
| Value clipping | **available** — `critic.cliprange_value` | critic config |
| Value loss normalization | **available**, and unit-tested across all 4 `loss_agg_mode`s | `tests/trainer/ppo/test_value_loss_normalization_on_cpu.py` |
| GAE | **available** — `algorithm.adv_estimator=gae`, `gamma`, `lam` | `compute_gae_advantage_return` |
| Explained variance | **available** — `critic/vf_explained_var` | `metric_utils.py` |
| Value metrics | **available** — `critic/values/{mean,max,min}`, `critic/returns/{...}` | `metric_utils.py` |
| **Length-adaptive GAE** | **ABSENT** | — |
| **Value pretraining / warmup phase** | **ABSENT** as a distinct VAPO stage | — |
| Decoupled / dual clipping | partially — `clip_ratio_low/high`, `clip_ratio_c` exist | actor config |
| Token-level PG loss | **available** — `loss_agg_mode=token-mean` | shared with DAPO |
| Positive-example LM loss | **ABSENT** | — |

**The genuinely missing piece is length-adaptive GAE**, VAPO's core claim: `lambda` adapted
to sequence length so that credit assignment does not decay across very long chains of
thought. Everything else is either present or a modest variation on present code.

## Cost estimate

| item | estimate |
|---|---|
| Actor peak VRAM (measured, R0) | **40.52 GiB of 95.07** |
| Critic (8B, bf16, Adam) | roughly comparable to the actor — the binding constraint is optimizer state, not activations |
| Expected total peak | plausibly fits on 4×H20 at reduced batch, but **unverified** |
| Implementation effort | length-adaptive GAE + value warmup + config plumbing + tests |
| Risk | a subtly wrong GAE is **silent** — it produces plausible curves and wrong credit assignment |

That last row is the decisive one. A broken group-advantage shows up immediately as an
absurd advantage range (R0 caught ±1.5 and confirmed it against the source arithmetic). A
broken GAE does not: it yields smooth, believable curves that are simply wrong. Debugging
it costs far more GPU-hours than implementing it.

## Recommendation

**Phase 0 (no GPU).** Keep VAPO as a documented, unimplemented arm. Done — this report.

**Phase 1 (cheap, high value).** Run **plain PPO with GAE and a critic** using VeRL's
existing, maintained path (`adv_estimator=gae` + critic enabled), as a 2-update smoke test
under the flight recorder. This delivers most of the cross-paradigm insight the suite wants
— critic memory cost, value fwd/bwd time, explained variance, actor–critic coupling — with
**zero new algorithm code**, and it establishes whether an 8B actor + 8B critic even fits
this node. It also validates every value-side metric before any VAPO-specific code exists.

**Phase 2 (only if Phase 1 is healthy and the suite still wants it).** Implement
length-adaptive GAE as a registered advantage estimator, with CPU unit tests against
hand-computed values *before* any GPU run — the same discipline that INC-004 taught: the
expensive stage must never be the first place a code path executes.

Attempting Phase 2 first would violate the project's own stage-gate rule and risks burning
hours on a silent correctness bug.

## Observability extension required for any value-based arm

The flight recorder is critic-blind today. Before a value-based run, the collector must
carry (VeRL keys that already exist marked ✓):

- `value_mean`, `value_std` ✓ (`critic/values/{mean,max,min}`)
- `value_target_mean`, `value_target_std`, `returns_mean`, `returns_std` ✓ (`critic/returns/*`)
- `value_loss` ✓, `value_clip_fraction` (verify emission), `explained_variance` ✓ (`critic/vf_explained_var`)
- TD-error statistics — **derive**, not currently emitted
- GAE advantage `mean, std, p01, p50, p99` — partially available (`critic/advantages/{mean,max,min}`); quantiles need derivation
- **Per-length-bucket** (short / medium / long) value error, advantage magnitude and return
  — **derive from the rollout dump**; this is the bucket split that would actually expose
  value bias and reward-signal decay on long CoT, which is the whole point of the arm.
- Systems: critic memory, actor memory, total peak VRAM, value fwd/bwd time, actor fwd/bwd
  time, communication overhead.

Anything not emitted stays `unavailable` — never `0`.
