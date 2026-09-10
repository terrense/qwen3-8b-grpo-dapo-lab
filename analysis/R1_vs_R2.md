# R1 (vanilla GRPO) vs R2 (DAPO)

**先把话说清楚：这不是一个公平的性能对比，别拿这里的 reward 数字说事。**

R1 是 20 个 update 的正式 baseline，R2 目前只是 2 个 update 的 smoke test。而且 DAPO 的
dynamic sampling 会把 unanimous 的 group 丢掉重新生成，所以两边**训的 prompt 子集都不一样**。

这份文档能站得住的是**机制层面的证据** —— 哪个开关真的生效了、代价是多少。
性能对比要等 R2 跑到 20 个 update、matched budget 之后再说。

| | R1 | R2 |
|---|---|---|
| 算法 | vanilla GRPO | DAPO |
| update 数 | **20** | **2**（smoke） |
| 入口 | `verl.trainer.main_ppo` | `recipe.dapo.main_dapo` |
| 状态 | PASS, exit 0 | PASS, exit 0 |

配置上刻意对齐的：Qwen3-8B、去重后的 `train.parquet`、seed 20260910、
`train_batch_size=16`、`rollout.n=8`、`ppo_mini_batch_size=8`、`max_response_length=8192`、
关 thinking、`lr=1e-6`、temperature 1 / top-p 1、TP=2、4×H20。

不一样的就是 DAPO 那四个机制 + 它把 KL 整个去掉。

---

## 1. Dynamic Sampling —— 生效了，代价正好 2 倍

这是这次最实的一条。

```
num_prompt_in_batch=10 < prompt_bsz=16
Keep generating
```

`train/num_gen_batches` 两个 update 都是 **2**。

翻译一下：生成 16 个 prompt 的 group，只有 **10 个**通过了「不能全对也不能全错」的筛选，
不够 16 个，于是又生成了一批。

10/16 = **62.5%**。R1 20 个 update 量出来的 effective signal 是 **63.75%**。
**两个数字几乎一样** —— 说明这个筛选率不是偶然，是这个 policy 在这个数据集上的稳定属性。

代价直接反映在 rollout 时间上：

| | R1 u1 | R1 u2 | R2 u1 | R2 u2 |
|---|---|---|---|---|
| `timing_s/gen` | 42.4 s | 34.4 s | **73.1 s** | **71.0 s** |

rollout 生成时间基本翻倍，跟 `num_gen_batches=2` 完全对得上。

**所以 DAPO 的账是这么算的**：花 2 倍的生成开销，换来一个 100% 都带梯度的 batch。
R1 那边名义 batch 是 16，实际只有约 10 个 group 在贡献梯度；R2 花双倍 token 把这 16 个填满。

单位有效 group 的成本其实差不多 —— **DAPO 在这里买到的不是效率，是 batch 的确定性**。
每一步都是满的 16 个有效 group，而不是 10 ± 波动。这对梯度噪声有意义，但不是免费的。

---

## 2. 去掉 KL 省下了一整趟 reference 前向

这个我事先没预料到，是从指标缺失里看出来的。

R2 的指标里**完全没有**这几个字段，R1 有：

```
actor/kl_loss
actor/kl_coef
timing_s/ref
timing_per_token_ms/ref
```

DAPO 的官方配置是 `use_kl_loss=False` + `kl_coef=0.0`，KL 项整个不要。
既然不算 KL，就**不需要 reference model 的前向**，于是那趟 forward 直接没了。

省下的东西是实的：

| | R1 | R2 |
|---|---|---|
| `actor/perf/max_memory_allocated_gb` | 48.94 / 49.14 GB | **46.12 / 46.12 GB** |
| `timing_s/ref` | 21.1 s (u1) → 2.7 s (u2) | **不存在** |

峰值显存少了约 3 GB，而且 R1 第一个 update 那 21 秒的 reference 加载开销也没有了
（R1 的 reference 是 CPU offload 的，第一次调用要付 host→device 传输）。

**所以 dynamic sampling 的翻倍开销被 KL 的省掉部分抵掉了一些**，
总的 `timing_s/step` 并没有翻倍：

| | R1 u1 | R1 u2 | R2 u1 | R2 u2 |
|---|---|---|---|---|
| `timing_s/step` | 126.9 s | 85.7 s | 119.4 s | 112.4 s |

R1 稳态是 86 秒，R2 是 112 秒，贵了约 **30%**，不是 100%。

**但这里有个必须说明的事**：KL 去掉不只是省了算力，它同时**改变了算法本身** ——
没有 KL 锚，policy 可以自由地漂离 reference。所以任何 GRPO↔DAPO 的对比都必须交代
KL 是对齐了还是去掉了。这次是**按官方 recipe 去掉的**。

---

## 3. Clip-Higher —— 一点效果都没有，跟我预测的一样

R1 跑完我就在 [`docs/algorithms/gspo.md`](../docs/algorithms/gspo.md) 和
[`docs/algorithm_matrix.md`](../docs/algorithm_matrix.md) 里写过：R1 的 `pg_clipfrac`
只有 1.5e-4，说明 lr 1e-6 下 ρ ≈ 1，clipping 几乎没有发挥空间，所以 clip-higher 应该测不出东西。

实测：

| | R1 u1 | R1 u2 | R2 u1 | R2 u2 |
|---|---|---|---|---|
| `clip_ratio_low` / `high` | 0.2 / **0.2** | 0.2 / 0.2 | 0.2 / **0.28** | 0.2 / 0.28 |
| `actor/pg_clipfrac` | 5.21e-5 | 1.53e-4 | 7.27e-5 | 1.65e-4 |
| `actor/pg_clipfrac_lower` | 0 | 0 | 0 | 0 |
| `actor/ppo_kl` | −2.18e-5 | −4.40e-5 | −2.03e-5 | −6.71e-5 |

上界从 0.2 放到 0.28，**clipfrac 基本没动**（都在 1e-4 量级）。
`pg_clipfrac_lower`（dual-clip 分支，`clip_ratio_c=10.0`）两边全是 0，从来没触发。

原因很简单：clip 只在 ρ 跑出 `[1−0.2, 1+0.28]` 这个区间时才起作用。
现在 ρ 离 1 的距离是 1e-5 量级，区间边界是 0.2 —— 差了四个数量级。
**把上界从 1.2 放到 1.28 对一个恒在 1.00002 附近的 ρ 来说毫无意义。**

**结论**：clip-higher 在这个配置下**不可测**，不是「没效果」。
要测它必须先让 ρ 真的动起来 —— 拉大 `train/mini` 比例，或者把 LR 提上去。
DAPO 官方 recipe 用的是 `train_prompt_bsz=512 / mini_bsz=32`，也就是**每个 update 16 个梯度步**，
我这里只有 2 个。这是个真实的实验设计缺陷，不是 DAPO 的问题。

同理，**GSPO 那组照这个配置跑也是白跑** —— 它整个论点就是 token ratio vs sequence ratio
的分布差异，而现在两个 ratio 都恒等于约 1。

---

## 4. 顺手确认的两件事

**verifier 的开销可以彻底忽略。** R2 有 `timing_s/reward` 这个字段：

```
timing_s/reward           u1 = 9.05e-05 s     u2 = 1.03e-04 s
timing_s/agent_loop/compute_score/mean  u1 = 5.7e-03 s
```

**90 微秒**。rollout 是 71 秒。差了六个数量级。
之前单元测试量的是 0.017 ms/样本，这里是端到端的确认。以后任何吞吐问题都不用看 verifier。

**advantage 还是那套 G=8 的算术。** 两边 `critic/advantages/max` 都是 **2.4749**，
`min` 都是 **−2.4749** —— 就是 1-of-8 那个 case（$1.75/\sqrt{0.5}$）。
DAPO 没有改 advantage 的算法，只改了哪些 group 能进 batch，这一点从数字上得到确认。

---

## 5. Overlong Reward Shaping —— 这次没机会触发

配的是 `overlong_buffer.len=2048`、`penalty_factor=1.0`、`max_resp_len=8192`，
也就是超过 6144 token 才开始扣分。

实际 `response_length/max` 是 6260 和 5658，`mean` 是 1743 和 1571。
只有 u1 那个 6260 稍微越过了 6144 一点，扣分幅度约 −0.06，基本可以忽略。

`reward_extra_info` 里的 `overlong` / `overlong_reward` 字段没进到 metric sink 里，
所以**具体有多少条被扣了分，现在拿不到**。要看得从 rollout dump 里自己统计。
20 update 的正式 R2 之前得把这个补上。

---

## 目前能下的结论

| DAPO 机制 | 生效了吗 | 证据 | 代价 |
|---|---|---|---|
| Dynamic Sampling | **是** | `num_gen_batches=2`，`10 < 16` 触发重采样 | rollout 生成时间 ×2 |
| 去掉 KL | **是** | `kl_loss` / `timing_s/ref` 字段整个消失 | 省 3 GB 显存 + 一趟 forward；但 policy 没有锚了 |
| Token-level PG loss | 配置生效（`token-mean`） | 2 个 update 看不出行为差异 | — |
| Clip-Higher | **不可测** | clipfrac 跟 R1 一样在 1e-4 量级 | — |
| Overlong shaping | 几乎没触发 | max 长度只有 6260，刚过 6144 | — |

**净开销**：稳态每个 update 贵约 30%（86 s → 112 s），
因为 dynamic sampling 的翻倍被去掉 KL 的节省抵掉了一部分。

---

## 下一步必须先解决的问题

**R2 跑 20 update 之前，配置得改。** 现在这个 `mini=8 / train=16`（2 个梯度步）
让 ρ ≈ 1，直接导致：

- clip-higher 测不出来
- 未来的 GSPO 也测不出来
- clipfrac / ppo_kl 这两个观测面基本是死的

建议按 DAPO 官方的比例来 —— 它是 `512 / 32` = 16 个梯度步。在我们这个规模上对应
`train_batch_size=32 / ppo_mini_batch_size=4`，也是 8 个梯度步。
但这样就跟现在的 R1 baseline 不 matched 了，所以 **R1 也得用新配置重跑一遍**。

这是个要花钱的决定（两个 20-update run，约 1.5 小时），得先确认再动。

另外要补的：
1. 从 rollout dump 里统计 overlong 扣分的实际条数和幅度
2. 记录 discarded group 数和 resampling 轮数（`num_gen_batches` 有了，被丢掉的 group 数还没有）
3. matched budget 得按**生成 token 数**对齐，不能按 update 数 —— R2 每个 update 的 token
   是 R1 的 2 倍，「20 个 update」在两边不是一回事
