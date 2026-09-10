# GRPO — the control arm

Group Relative Policy Optimization. Critic-free: the baseline comes from *other samples of
the same prompt* instead of from a learned value function.

Implementation in this checkout: `verl/trainer/ppo/core_algos.py:272-322`
(`compute_grpo_outcome_advantage`), verl `1252cc71`.

---

## Objective

For prompt $x$, sample a group of $G$ responses $\{y_1,\dots,y_G\}$ from $\pi_{\theta_{\text{old}}}$:

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{x,\{y_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\Big(\rho_{i,t}(\theta)\,\hat{A}_i,\ \mathrm{clip}\big(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\big)\hat{A}_i\Big)\right]
$$

with the **token-level** importance ratio

$$
\rho_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}
$$

and the **group-normalised outcome advantage**, identical for every token of a response:

$$
\hat{A}_i=\frac{r_i-\mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})+\varepsilon},\qquad \mathbf{r}=(r_1,\dots,r_G),\ \varepsilon=10^{-6}
$$

---

## Data flow

```mermaid
flowchart LR
    X["prompt x"] --> S["sample G responses<br/>from π_old"]
    S --> R["RLVR verifier<br/>r_i ∈ {+1, −1}"]
    R --> M["group mean & std"]
    M --> A["Â_i = (r_i − mean) / (std + ε)"]
    A --> T["broadcast to every<br/>response token"]
    T --> L["clipped surrogate<br/>token-mean aggregation"]
    L --> G["gradient"]
    style A fill:#ddf4ff,stroke:#0969da
    style R fill:#dafbe1,stroke:#1a7f37
```

---

## The `std` is the unbiased (n−1) estimator — measured, not assumed

`core_algos.py:312` calls `torch.std(scores_tensor)`, whose PyTorch default is
`correction=1`. With binary rewards $r_i\in\{+1,-1\}$ and $k$ correct out of $G=4$:

| $k$ (correct of 4) | mean | std (n−1) | $\lvert\hat{A}\rvert$ |
|---|---|---|---|
| 1 | −0.5 | 1.000 | **1.500** |
| 2 | 0.0 | 1.155 | **0.866** |
| 3 | +0.5 | 1.000 | **1.500** |
| 0 or 4 | ±1 | 0 | **0** (no signal) |

R0 measured `critic/advantages/max = +1.4999985` and `min = −1.4999985` — matching the
$k\in\{1,3\}$ row exactly, including the $\varepsilon$ offset. The pipeline arithmetic is confirmed.

**Two consequences that motivate the other arms:**

1. A $\sqrt{3}\approx 1.73\times$ difference in gradient magnitude arises purely from *how many
   samples happened to be correct* — i.e. from group composition, not from learning signal.
   This is the term **Dr.GRPO** argues is a bias.
2. When all samples agree, $\mathrm{std}=0$ and $r_i-\mathrm{mean}=0$, so
   $\hat{A}_i=0/\varepsilon=0$: **that prompt contributes no gradient at all**. Not a weak
   push — silence. This is what **DAPO's dynamic sampling** attacks.

The fraction of the batch that survives is tracked as `effective_signal_fraction`.
R0 measured 50.0% and 62.5% on its two updates.

---

## The degenerate case R0 hit

With `ppo_mini_batch_size == train_batch_size` and `ppo_epochs = 1`, exactly one gradient
step is taken per rollout batch, using the same weights that generated it. Then
$\theta=\theta_{\text{old}}$, so

$$
\rho_{i,t}=\exp(0)=1\quad\Longrightarrow\quad \text{clipfrac}=0,\quad \text{KL}_{\text{ppo}}=0
$$

R0 measured exactly this: `actor/ppo_kl = 0.0`, `pg_clipfrac = 0.0` on both updates. **This
is correct behaviour and is evidence that old-logprob bookkeeping is right** — a non-zero
value here would be the bug.

It also means: **any experiment about ratios or clipping (GSPO, clip-higher) is vacuous
unless `ppo_mini_batch_size < train_batch_size` or `ppo_epochs > 1`.** R1 therefore uses
`8 < 16`, giving 2 gradient steps per update.

---

## KL placement

The official Qwen3-8B example applies KL as a **loss term**, not inside the reward:

$$
\mathcal{L}=-\mathcal{J}_{\text{GRPO}}+\beta\,\mathbb{D}_{\text{KL}}\big[\pi_\theta\,\|\,\pi_{\text{ref}}\big],\qquad \beta=0.001
$$

`use_kl_loss=True`, `kl_loss_type=low_var_kl`, `use_kl_in_reward=False`. The reward stays a
clean correctness signal, which matters when reading the reward curve.

Note **DAPO removes this term entirely** — any GRPO↔DAPO comparison must say whether KL was
matched or dropped.

---

## Config

```
algorithm.adv_estimator=grpo
algorithm.norm_adv_by_std_in_grpo=True
actor_rollout_ref.actor.loss_agg_mode=token-mean
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.2
actor_rollout_ref.actor.use_kl_loss=True  kl_loss_coef=0.001  kl_loss_type=low_var_kl
algorithm.use_kl_in_reward=False
algorithm.filter_groups.enable=False
```

Runs: [`R0`](../../analysis/R0_smoke_report.md) (2 updates, PASS) ·
[`R1`](../../analysis/R1_baseline_report.md) (20 updates, PASS)

### What R1 measured

| quantity | R1, 20 updates |
|---|---|
| `actor/pg_clipfrac` | **5.07e-5 – 2.61e-4** (mean 1.54e-4) — nonzero, but ~0.015% of tokens |
| `actor/ppo_kl` | **−5.44e-5 … +4.02e-5**, signed |
| `actor/kl_loss` | monotone rise **1e-4 → 2.4e-3** — real drift from the reference |
| `critic/advantages/max` | **2.4749** — exactly $1.75/\sqrt{0.5}$, the $G{=}8$, $k{=}1$ case |
| `response_length/mean` | oscillates **1192 – 2188**, no monotone trend |
| `entropy` | **0.2255 – 0.4432**, no collapse |
| staleness / rollout↔train corr | **0** / **0.9994 – 0.9997** |

With $G=8$ the advantage extrema become a **direct readout of group composition**, because
the unbiased std is a deterministic function of $k$:

| $k$ correct of 8 | mean | std ($n{-}1$) | $\lvert\hat A
vert$ | seen in R1 |
|---|---|---|---|---|
| 1 or 7 | −0.75 | 0.7071 | **2.4749** | most updates |
| 2 or 6 | −0.50 | 0.9258 | **1.6202** | steps 5, 12 |
| 3 or 5 | −0.25 | 1.0351 | **1.2076** | steps 19, 20 |
| 4 | 0 | 1.0690 | 0.9354 | — |
| 0 or 8 | ±1 | 0 | **0** (no signal) | — |

Reading `advantages/max = 1.6202` therefore tells you the batch's most extreme group was
2-of-8, with no extra instrumentation.

**The important negative finding:** clipping is technically active but only ~0.015% of tokens
clip, because at lr 1e-6 two mini-batch steps move the policy very little, so
$
hopprox1$ still. **A GSPO-vs-GRPO ratio comparison at this LR would still be close to
vacuous** — that arm needs a larger `train/mini` ratio or a higher LR to have anything to
measure. R0 established that $
ho\equiv1$ is fatal; R1 shows $
hopprox1$ is nearly as bad.
