# GSPO vs GRPO：ratio 分布的直接对比

**第一次把 token ratio 和 sequence ratio 的分布摆在一起量。GSPO 的机制确实是按设计工作的，
而且效果比我预期的强得多。**

两边配置完全相同：Qwen3-8B、去重数据集、seed 20260910、`train_batch_size=16`、
`rollout.n=8`、`ppo_mini_batch_size=8`（2 个梯度步）、`max_response_length=8192`、
关 thinking、`lr=1e-6`、温度 1 / top-p 1、TP=2、4×H20、各 3 个 update。

唯一的差别：`actor.policy_loss.loss_mode` = `vanilla` vs `gspo`，
`loss_agg_mode` = `token-mean` vs `seq-mean-token-mean`。

指标来自 [`../scripts/patches/patch_ratio_stats.py`](../scripts/patches/patch_ratio_stats.py)，
同一套观测代码同时打在两个 policy loss 上（前缀分别是 `actor/ratio` 和 `actor/seqratio`）。

---

## 数据

### GRPO —— token-level ratio

| step | \|ρ−1\| p95 | \|ρ−1\| p99 | \|ρ−1\| max | ρ 范围 | 偏离 >1% | clipfrac |
|---|---|---|---|---|---|---|
| 1 | 0.0303 | 0.0555 | **0.301** | 0.778 – 1.301 | **7.99%** | 1.27e-4 |
| 2 | 0.0279 | 0.0557 | **0.367** | 0.742 – 1.354 | **7.25%** | 1.87e-4 |
| 3 | 0.0317 | 0.0568 | **0.416** | 0.730 – 1.416 | **8.72%** | 3.76e-5 |

### GSPO —— sequence-level ratio

| step | \|s−1\| p95 | \|s−1\| p99 | \|s−1\| max | s 范围 | 偏离 >1% | clipfrac |
|---|---|---|---|---|---|---|
| 1 | 5.41e-4 | 5.41e-4 | **1.08e-3** | 0.9991 – 1.0011 | **0** | **0** |
| 2 | 4.40e-4 | 4.40e-4 | **8.79e-4** | 0.9992 – 1.0009 | **0** | **0** |
| 3 | 3.73e-4 | 3.73e-4 | **7.46e-4** | 0.9994 – 1.0007 | **0** | **0** |

### 对比

| | GRPO (token) | GSPO (sequence) | 倍数 |
|---|---|---|---|
| p99 偏离 | 0.0555 | 5.41e-4 | **紧 103 倍** |
| 最大偏离 | 0.301 | 1.08e-3 | **紧 278 倍** |
| 偏离 >1% 的 token | 8.0% | **0** | — |
| clipfrac | ~1e-4 | **恰好 0** | — |
| 三步内尾巴走向 | **0.301 → 0.367 → 0.416（变宽）** | **1.08e-3 → 7.46e-4（收窄）** | 方向相反 |

---

## 为什么会这样 —— 算术上是必然的

sequence ratio 是长度归一化的几何平均：

$$
s_i(\theta)=\left(\prod_{t=1}^{\lvert y_i\rvert}\rho_{i,t}\right)^{1/\lvert y_i\rvert}
$$

response 平均 ~1600 token。一条序列里就算有一个 token 的 ρ 冲到 1.3，
其余都贴着 1：

$$
s_i \approx (1.3 \times 1^{1599})^{1/1600} = 1.3^{1/1600} \approx 1.00016
$$

**那个 30% 的离群值被 1/|y| 这个指数压成了 0.016%。** 这正是 GSPO 论文的论点，
只不过之前只是纸上的论证，现在是自己机器上的数。

所以 clipfrac 恰好为 0 也就顺理成章 —— sequence ratio 压根没离开过 [0.8, 1.2]，
甚至没离开过 [0.999, 1.001]。

---

## 两个必须说清楚的限制

**1. 这是在比两个不同的量，不是同一个量在两种算法下的表现。**

GRPO 的 token ratio 和 GSPO 的 sequence ratio 是不同的随机变量。
sequence ratio 是 ~1600 个 token ratio 的几何平均，**由集中不等式，它必然更窄** ——
这不需要 GSPO 做对什么，是数学上的必然。

所以这份数据证明的是：**GSPO 的机制按设计在工作，离群 token 确实被抹平了**。
它**不能**直接推出「GSPO 训得更好」。后者要看 reward / validation，
而 3 个 update 根本不够。

**2. GSPO 那几个分位数是粗的。** p95 和 p99 完全相等（5.41e-4），
因为这批只有 128 条序列（16 prompt × 8），而 sequence ratio 是**每条序列一个值**，
一条序列里所有 token 共享同一个 s_i。按 token 加权算分位数，
底层只有 128 个不同取值，所以分位数会撞在一起。

要更细的分布得按序列统计而不是按 token。这个观测钩子目前是 token 加权的，
以后要比较精细的 sequence ratio 形状的话得改。

---

## 这回答了什么

`docs/algorithms/gspo.md` 里写的诊断问题是：

> **GRPO 下是不是少数极端的 token ratio 主导了 clipping 和梯度行为，
> sequence-level ratio 是不是真的压住了这种敏感性？**

**答案：是，而且是压得非常彻底。**

- GRPO 下有 8% 的 token 偏离 >1%，极值到 ±30~42%，而且**三步内在持续变宽**
- GSPO 下 0% 的 token 偏离 >1%，极值只有 0.1%，而且**三步内在收窄**

尾巴走向相反这一点值得单独拎出来 —— GRPO 的 max 偏离 0.301 → 0.367 → 0.416 在涨，
GSPO 的 1.08e-3 → 8.79e-4 → 7.46e-4 在降。20 个 update 的运行里这可能是比 reward
更早的稳定性信号。

## 代价

| | GRPO | GSPO |
|---|---|---|
| `timing_s/step` | ~90 s | 108.7 / 100.9 s |
| reward（3 步，**不可比**） | — | −0.328 / −0.344 |
| grad_norm | 0.27 – 0.31 | 0.265 / 0.360 |

step 时间贵约 10~20%。reward 在 3 个 update 上没有任何意义，不做解读。

---

## 下一步

GSPO 这个臂**已经确认可跑、机制确认生效**，可以进 20-update 的正式对比。
要注意的是：既然 GSPO 的 clipfrac 恒为 0，**clip-higher 这类改动在 GSPO 上更没意义** ——
它的 trust region 根本没被触碰过。这也意味着 GSPO 和 DAPO 的组合需要重新想清楚
到底要比什么。
