# Algorithm Matrix

Five policy-optimization methods, compared as **interventions on specific layers of the RL
pipeline** — not as five trainers racing for the highest reward.

The question for each is: *which layer does it modify, and what does that modification do
to the observable training dynamics?*

Every configuration key below was read from the checked-out source at
**verl `1252cc71`** (or from VeRL's own `docs/algo/grpo.md`), not from blog summaries.
Anything not verifiable in this checkout is marked **NOT IMPLEMENTED**.

> **Status: all five are PLANNED except vanilla GRPO, which has completed R0 only
> (2 optimizer updates). No comparative result exists yet.**

---

## The matrix

| # | Dimension | **GRPO** (control) | **Dr.GRPO** | **DAPO** | **GSPO** | **VAPO** |
|---|---|---|---|---|---|---|
| 1 | Parent / lineage | PPO without critic | GRPO | GRPO | GRPO | PPO (critic restored) |
| 2 | Advantage estimator | group-normalised outcome | group-**mean-only** outcome | group-normalised outcome | group-normalised outcome | **GAE from a value function** |
| 3 | Reward normalization | `(r − mean) / (std + ε)` | `r − mean` (**no std**) | `(r − mean) / (std + ε)` | `(r − mean) / (std + ε)` | value-baseline, not group |
| 4 | Importance-ratio granularity | **token** | token | token | **sequence** (length-normalised) | token |
| 5 | Clipping | symmetric `[1−0.2, 1+0.2]` | symmetric | **asymmetric** (clip-higher) | symmetric, on the **sequence** ratio | symmetric (+ decoupled variants) |
| 6 | Loss aggregation | `token-mean` | **`seq-mean-token-sum-norm`** (constant divisor) | `token-mean` (token-level PG) | `seq-mean-token-mean` | token-mean |
| 7 | Critic required | no | no | no | no | **yes** |
| 8 | Value function | none | none | none | none | **yes, trained** |
| 9 | Dynamic sampling | no | no | **yes** (`filter_groups`) | no | not intrinsic |
| 10 | Treatment of response length | length-averaged loss (implicit length bias) | **length bias removed** by constant normaliser | overlong **reward shaping** (soft penalty) | length enters via the `1/\|y\|` ratio exponent | length-adaptive GAE (paper) |
| 11 | Claimed failure mode addressed | — (baseline) | **optimization bias inflating response length**, esp. for wrong answers | zero-variance groups, entropy collapse, truncation noise | **token-ratio variance** destabilising long sequences | poor credit assignment on long CoT |
| 12 | Expected systems overhead | baseline | ≈ baseline (loss-math only) | **higher** — resampling regenerates discarded groups | ≈ baseline (ratio reshaped, not recomputed) | **highest** — critic fwd/bwd + memory |
| 13 | Expected observable signature | reward ↑ slowly, response length ↑, high zero-std ratio | **shorter wrong answers**, flatter length growth, larger advantage spread on easy/hard groups | mixed-group ratio ↑, effective batch ↑, generated tokens ↑, upper-clip ↑ | token-ratio tails wide but **sequence-ratio tails narrow**; lower clipfrac volatility | value/return calibration, explained variance, actor–critic coupling |

---

## How each is configured in this checkout

### GRPO — control

```
algorithm.adv_estimator=grpo
algorithm.norm_adv_by_std_in_grpo=True          # default
actor_rollout_ref.actor.loss_agg_mode=token-mean # default
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.2
actor_rollout_ref.actor.use_kl_loss=True kl_loss_coef=0.001 kl_loss_type=low_var_kl
algorithm.use_kl_in_reward=False
```

Source: `compute_grpo_outcome_advantage`, `core_algos.py:304-322`. Uses `torch.std`
(**unbiased, n−1**) — measured and confirmed in R0, where a 1-correct-of-4 group produced
advantages of exactly ±1.5.

### Dr.GRPO — bias correction

Paper: *Understanding R1-Zero-Like Training: A Critical Perspective*,
[arXiv 2503.20783](https://arxiv.org/pdf/2503.20783).
Config per VeRL's own `docs/algo/grpo.md:53-62`:

```
algorithm.norm_adv_by_std_in_grpo=False                       # removes the std/difficulty bias
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm # removes the length bias
actor_rollout_ref.actor.loss_scale_factor=<constant>          # optional; e.g. max_response_length
actor_rollout_ref.actor.use_kl_loss=False
```

**Two distinct biases, removed independently — which is why this is one experiment with two
falsifiable claims:**

1. **Std normalisation (difficulty bias).** Dividing by the group std up-weights groups
   whose reward variance happens to be small. With binary ±1 rewards and G samples, the
   std is a deterministic function of how many samples were correct, so dividing by it
   silently re-weights prompts by difficulty. R0 measured this concretely: a 1-of-4 group
   yields ±1.5 while a 2-of-2 split yields ±0.866 — a 1.7× difference in gradient
   magnitude arising purely from group composition, not from learning signal.
2. **Length normalisation (length bias).** `token-mean` divides each sequence's loss by its
   own token count, so a long wrong answer receives a *smaller per-token* penalty than a
   short wrong answer. `seq-mean-token-sum-norm` divides by a constant instead, removing
   the incentive to pad incorrect responses.

**The diagnostic question:** how much of the response-length growth observed under GRPO is
optimization bias rather than genuine reasoning improvement? R0 already showed
`response_length/mean` rising 1532 → 2041 across two updates, so the phenomenon is present
and measurable from the first steps.

**Key metric split for this comparison:** response length must be reported
**separately for correct and incorrect answers**. The Dr.GRPO claim is specifically about
*incorrect* responses lengthening, and an aggregate mean would hide it.

### DAPO — group-signal and clipping engineering

Recipe: `repos/verl-recipe/dapo/` (`main_dapo.py`, `dapo_ray_trainer.py`).
Defaults from `run_dapo_qwen2.5_32b.sh`:

```
algorithm.adv_estimator=grpo
actor_rollout_ref.actor.clip_ratio_low=0.2   clip_ratio_high=0.28   clip_ratio_c=10.0
algorithm.filter_groups.enable=True  metric=acc  max_num_gen_batches=10
actor_rollout_ref.actor.loss_agg_mode=token-mean
reward_model.overlong_buffer.enable=True  len=4096  penalty_factor=1.0
use_kl_in_reward=False  kl_coef=0.0  use_kl_loss=False  kl_loss_coef=0.0
train_prompt_bsz=512  n_resp_per_prompt=16  train_prompt_mini_bsz=32
max_response_length=20480  temperature=1.0  top_p=1.0
```

Four mechanisms, all present in this checkout:

1. **Clip-Higher** — `clip_ratio_high=0.28 > clip_ratio_low=0.2`. Asymmetry gives
   low-probability tokens more room to grow, countering entropy collapse.
2. **Dynamic Sampling** — `algorithm.filter_groups` (native, `_generated_ppo_trainer.yaml:863`)
   drops zero-variance groups and regenerates up to `max_num_gen_batches` times. This is
   the direct attack on the metric R0 measured at 37.5–50%.
3. **Token-Level PG Loss** — `loss_agg_mode=token-mean`, so long sequences contribute
   proportionally to their token count.
4. **Overlong Reward Shaping** — `reward_model.overlong_buffer` applies a *graded* penalty
   inside a soft buffer before the hard cap (`reward_manager/dapo.py`:
   `expected_len = max_resp_len − overlong_buffer_len`, penalty scaled by `exceed_len`),
   instead of a cliff at truncation.

Note `use_kl_loss=False` and `kl_coef=0.0` — **DAPO removes the KL term entirely**. Any
GRPO↔DAPO comparison must state whether KL was matched or removed, since that alone moves
the dynamics.

### GSPO — sequence-level policy optimization

Paper: [arXiv 2507.18071](https://arxiv.org/pdf/2507.18071). **Natively implemented** as
`compute_policy_loss_gspo`, `core_algos.py:1546-1610`.

```
actor_rollout_ref.actor.policy_loss.loss_mode=gspo
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean   # recommended in the docstring
```

The implementation, read directly:

```python
seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
negative_approx_kl_seq = (negative_approx_kl * response_mask).sum(-1) / seq_lengths
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
```

So the sequence ratio is the **length-normalised geometric mean** of token ratios,
`s_i(θ) = (π_θ(y_i|x)/π_old(y_i|x))^(1/|y_i|)`, and the stop-gradient construction
(`log_prob − log_prob.detach() + sg[log s_i]`) keeps a per-token gradient path while the
*clipping decision* is made on the sequence quantity. **Clipping therefore accepts or
rejects a whole sequence rather than individual tokens.**

**The diagnostic question:** do a few extreme token ratios dominate clipping and gradient
behaviour under GRPO, and does the sequence-level ratio actually suppress that? This is
answered by plotting the two ratio distributions side by side — token p01/p05/p50/p95/p99/max
against sequence p01/p05/p50/p95/p99/max.

> **Hard prerequisite, established by R0.** With `ppo_mini_batch_size == train_batch_size`
> and `ppo_epochs=1`, ρ ≡ 1 exactly and every clip fraction is 0. **A GSPO-vs-GRPO ratio
> comparison run in that configuration would be meaningless.** All ratio experiments must
> set `ppo_mini_batch_size < train_batch_size` (DAPO's own recipe uses 32 vs 512).

> **更正（2026-09-11）**：上面这段对 `mini == train` 的判断是**对的** —— 那时 ρ 恰好等于 1。但由此外推出的「`mini < train` 时 ρ 仍然约等于 1，所以 GSPO 也测不出来」**是错的，已撤回**。在 `mini=8 / train=16`（R1、R2 用的配置）下实测：|ρ−1| 的 p99 = 0.055、极值 0.42、**8% 的 token 偏离超过 1%**。分布是尖峰厚尾，不是窄。错在拿 `ppo_kl`（带符号均值）当分布宽窄的判据 —— 它在连续三步里符号是 +、−、+ 翻转的，正是正负抵消的证据。**GSPO 不需要先改配置就能测。**见 [`../analysis/ratio_distribution.md`](../analysis/ratio_distribution.md)。


### VAPO — value-based cross-paradigm comparison

Paper: [arXiv 2504.05118](https://arxiv.org/pdf/2504.05118), referenced in
`repos/verl/README.md:76`.

**NOT IMPLEMENTED** — no VAPO module, recipe or config exists in verl `1252cc71` or in
verl-recipe. See [`../analysis/vapo_feasibility.md`](../analysis/vapo_feasibility.md).
No VAPO training run is planned until that report concludes it is safe; the pieces VeRL
*does* provide (`adv_estimator=gae`, a trainable critic, value clipping, value loss
normalization) are inventoried there.

VAPO is deliberately included because Dr.GRPO, DAPO and GSPO are all **critic-free,
group-based** methods — variations within one family. VAPO takes the other road and brings
the value function back, which makes it the only genuinely **cross-paradigm** arm.

---

## Matched comparison protocol

For **GRPO / Dr.GRPO / DAPO / GSPO**, hold constant: Qwen3-8B initialization, dataset
subset, validation set, seed set, prompt sampling, RLVR verifier, `max_response_length`,
and decoding settings.

**Match on generated-token budget, not only on optimizer updates.** Dynamic sampling and
`rollout.n` change the real cost of an update, so "100 updates" is not a comparable unit
across arms. Every report states:

optimizer updates · training prompts consumed · rollouts generated · **tokens generated** ·
valid/effective trajectories · GPU-hours · wall-clock.

**VAPO is not forced into compute-equivalence.** It is reported from two angles —
(1) matched generated-token budget, (2) actual resource cost — with critic memory, actor
memory, total peak VRAM, value fwd/bwd time, actor fwd/bwd time and communication overhead
recorded separately.

The comparison axes are therefore: **performance, sample efficiency, compute efficiency,
memory cost, training stability, diagnostic complexity** — not reward alone.

---

## Budget and order

| Phase | Scope |
|---|---|
| 1 | 2-update smoke test per algorithm |
| 2 | 20-update short run, only after its smoke test passes |
| 3 | 50–150 updates, only for algorithms whose dynamics prove worth the GPU-hours |

Priority: **1. GRPO (done: R0)  2. DAPO  3. Dr.GRPO  4. GSPO  5. VAPO** — VAPO last because
it is the heaviest and currently unimplemented.

---

## What the final comparison must answer

Not "DAPO is best." Specifically:

- **Dr.GRPO** — which bias does it remove, and what is the measured evidence? Concretely:
  does incorrect-response length grow more slowly than under GRPO?
- **DAPO** — what do dynamic sampling and asymmetric clipping actually change? Quantified as
  effective-signal fraction and discarded-group / resampling counts against extra tokens spent.
- **GSPO** — does the sequence-level ratio change stability? Quantified as token-ratio vs
  sequence-ratio tail behaviour and clip-fraction volatility.
- **VAPO** — does a critic give better credit assignment on long CoT, and at what cost in
  memory, time and diagnostic complexity?
