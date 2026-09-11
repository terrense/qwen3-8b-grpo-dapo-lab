# RL 工程总报告

这个项目最重要的产出不是 checkpoint，是这份报告。

**范围声明**：这是在**已发布的后训练版 `Qwen/Qwen3-8B`** 上直接做 GRPO 的
**系统工程预演（systems pilot / rehearsal）**，不是医学论文里的正式 M3
（论文设计是 `Qwen3-8B → SFT M1 → DPO M2 → GRPO M3`）。这里的任何结果都不得当作 M3。

每条结论都标注了证据来源。**没跑过的一律写 NOT RUN，不用教科书描述填充。**

---

## 贯穿全篇的一条结论

这个项目反复验证了同一件事，而且我自己在 4 个不同的地方栽在上面：

> **一个标量不能代表一个分布；两个非随机分组的均值之差不是效应；
> 而在比较任何东西之前，先得知道同一件事重复做两次会差多少。**

具体代价见 [`BUG_LOG.md`](BUG_LOG.md) 的第 21、23、24 条。

---

## 1. 这套 VeRL GRPO pipeline 的真实数据流是什么？

源码路径全部来自 verl `1252cc71`：

```
dataset row (data_source="math_dapo")
  → RLHFDataset 套 chat template（我们关了 thinking：
      +data.apply_chat_template_kwargs.enable_thinking=False）
  → vLLM rollout，每个 prompt 采 G=rollout.n 条
  → NaiveRewardManager.run_single()  (reward_loop/reward_manager/naive.py:34-99)
      valid_response_length = attention_mask[-response_length:].sum()   (L39)
      只 decode response，prompt 不进 verifier                            (L54-56)
  → default_compute_score() 按 data_source 分派  (reward_score/__init__.py:59)
      "math_dapo" 和任何 startswith("aime") → math_dapo.compute_score
  → compute_grpo_outcome_advantage()  (core_algos.py:304-322)
      组内 mean/std → 归一化 → * response_mask 广播到 response token
  → compute_policy_loss_vanilla()    (core_algos.py:1286-1440)
      ratio → clip → agg_loss(loss_agg_mode)
  → FSDP actor update → weight sync → 下一轮 rollout
```

完整带行号的追踪：[`reward_pipeline_trace.md`](reward_pipeline_trace.md)。

## 2. rollout policy 与 training policy 什么时候同步？

**每个 optimizer update 之后同步一次，而且是严格同步的。** 实测证据（R0）：

- `training/off_policy/trajectory_staleness` 的 mean/max/min **全部 = 0**
- `training/rollout_actor_probs_pearson_corr` = **0.99958 / 0.99957**
- `rollout_corr/kl` = 4.89e-4 / 3.73e-4

残留的 max per-token 差 0.1436 来自 bf16 和 vLLM/FSDP 不同的 kernel 路径，不是同步故障。

**这是量出来的，不是从架构推的。**

## 3. old logprob 是什么时候产生/重算的？

每个 update 里，rollout 之后**单独跑一次前向重算**（`timing_s/old_log_prob`），
不是从 rollout engine 直接拿。R0 实测这一步 4.22s / 3.22s。

关键推论（见第 13 条）：`ppo_mini_batch_size == train_batch_size` 且 `ppo_epochs=1` 时，
old logprob 和当前 logprob 由**同一份权重**算出，所以 ρ ≡ 1 精确成立。

## 4. response token mask 到底怎么构造？

`valid_response_length = attention_mask[-response_length:].sum()`（`naive.py:39`），
然后 advantage 通过 `scores.unsqueeze(-1) * response_mask` 施加（`core_algos.py:322`）。

**是这个 mask 把 advantage 限制在 response token 上的**，不是靠别的机制。

## 5. prompt / tool / environment token 为什么不能进入 policy loss？

因为 policy gradient 的对象是**模型自己的动作**。prompt 是环境给的，模型没有选择它的自由，
对它求梯度等于在训练模型去"生成它本来就被给定的输入"。

实测确认：`prompt_length/mean` = 142.8 和 `response_length/mean` = 1532.2 分开统计，
verifier 只拿到 response（`naive.py:54-56`）。

本次数学任务没有 tool/environment token。将来做 Agent RL 时，
工具返回的内容必须进 context 但不能当作 policy action 训练 —— 同样的道理。

## 6. GRPO group advantage 在当前代码里怎么算？

`core_algos.py:304-322`：


$$
\hat A_i=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)+\varepsilon},\quad \varepsilon=10^{-6}
$$

**`torch.std` 是无偏 (n−1) 估计**，这一点很重要，因为它让 advantage 的极值成为
**group 组成的直接读数**。G=8、r∈{+1,−1} 时：

| k 个答对 | 理论 \|A\| | 实测出现 |
|---|---|---|
| 1 或 7 | **2.4749** | R1/M20 大量步 |
| 2 或 6 | **1.6202** | M20 step 5, 12 |
| 3 或 5 | **1.2076** | M20 step 19, 20 |
| 0 或 8 | **0** | 零梯度 |

**看到 `advantages/max = 1.6202` 就知道这批最极端的组是 2-of-8，不需要额外埋点。**

全组一致时分子分母同时为 0，结果是 0/ε = 0 —— 不会数值爆炸，但也完全没有信号。

## 7. zero-variance group 实际出现了多少？

| 场景 | zero-std 占比 |
|---|---|
| 预训练前 rollout（thinking @4096） | **79.69%** |
| R0 update 1 / 2 | 50.0% / 37.5% |
| M20_grpo 全程 | 平均约 **42%** |
| M20_dapo（开 dynamic sampling） | **0~6%** |

**而且 GRPO 的废组不随训练减少，只是换形式**（M20_grpo）：

| step | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| 全对（太简单） | 0 | 3 | 2 | 3 | **5** |
| 全错（太难） | **6** | 6 | 5 | 2 | 2 |
| 浪费比例 | 37.5% | 56% | 43.8% | 31.3% | 43.8% |

模型变强后全错组从 6 降到 2，但全对组从 0 涨到 5，总浪费基本不变。
**固定难度的数据集上，GRPO 的有效信号比例会长期卡住，不会自己变好。**

## 8. dynamic sampling 对 effective batch 造成了什么影响？

DAPO 每个 batch **100% 是混合组**（20 步里 19 步 16/16，1 步 15/16），GRPO 平均 58%。

代价：`train/num_gen_batches` 平均 2.1 轮，**rollout 时间 13.2 → 25.5 分钟，1.93 倍**。

换算单位算力产出：

| | 每 update 有效组 | 有效组/rollout分钟 |
|---|---|---|
| GRPO | 9.3 | **14.0** |
| DAPO | 15.8 | **12.4** |

**DAPO 买到的不是效率，是 batch 组成的确定性。**

**一个观测陷阱**：`perf/total_num_tokens` 三臂只差 3.4%，但 rollout 时间差近一倍 ——
被丢弃的 rollout 烧了 GPU 但 token 不进训练。**有 dynamic sampling 时该看 rollout wall time。**

## 9. rollout.n 从工程上改变了什么？

`rollout.n` = G 决定 advantage 的**取值格点**。R0 用 G=4，advantage 极值 ±1.5；
R1 起用 G=8，极值 ±2.4749。这不是超参调整，是**分母变了**：
G 越大，group mean/std 估计越稳，极端 advantage 越大（因为单个样本偏离组均值更远）。

同时 G 直接乘在 rollout 成本上：G=8、16 个 prompt = 每 update 128 条序列。

## 10. max response length 对 KV cache / throughput / truncation 有什么影响？

**这是本项目最大的一个坑**（[`pre_rl_reward_distribution.md`](pre_rl_reward_distribution.md)）。

Qwen3-8B 的 chat template **默认开 thinking**。实测：

| 配置 | 截断率 | 准确率 | 每样本 token |
|---|---|---|---|
| thinking @ 4096 | **87.11%** | **12.70%** | — |
| thinking @ 16384 | 18.75% | **56.25%** | 8820 |
| **关 thinking @ 4096** | 1.56% | 31.25% | **1477** |

**同一份权重，只改长度预算，准确率 12.7% → 56.25%。**

vLLM 侧：`gpu_memory_utilization=0.6` 给出 40.0 GiB KV cache = 291,248 token。
关 thinking 之后正式运行的 cap 命中率降到 **5/2560 = 0.2%**。

## 11. KL 上升的时候 reward 发生了什么？

**这个问题在当前配置下无法回答，而且过程中我答错过一次。**

`actor/ppo_kl` 全程在 ±1e-5 量级且**符号在步间翻转**（+5.1e-5 / −4.9e-5 / +3.1e-5）。
它是**带符号均值**（`core_algos.py:1339`），正负偏差会抵消。

我曾据此断定"ρ≈1，策略几乎没动"。**错的。** 打了观测补丁后量到真实分布：

| | 值 |
|---|---|
| \|ρ−1\| 中位数 | 3.9e-06 |
| \|ρ−1\| **p99** | **0.055** |
| \|ρ−1\| **最大** | **0.30 ~ 0.42** |
| **偏离 >1% 的 token** | **8%** |

**尖峰厚尾。** 而且第二个臂证明 `ppo_kl` 对分布宽度**根本不敏感** ——
梯度步 2→8 时分布宽度翻倍（p99 0.056→0.099），`ppo_kl` 纹丝不动。

**结论：`ppo_kl` 不能当策略偏移的判据。要看 `ratio_absdev_p99` 和 `ratio_frac_gt_*`。**

## 12. entropy 下降意味着什么，是否伴随性能上升？

M20_grpo entropy 0.353 → 0.299（降 15%），同期 validation 48.0% → 47.0%（降）。
**entropy 下降没有伴随性能上升。**

但更重要的是两条方法论警告：

1. **`actor/entropy` 的数值随 `loss_agg_mode` 变**。D_drgrpo 报 0.106 vs 其他 0.32，
   看着像崩塌，实际是 `seq-mean-token-sum-norm` 用常数 8192 做除数造成的归一化差异
   （实测比值 0.2375 vs 预测 \|y\|/8192 = 0.2328，误差 3.6%）。
   **跨不同聚合方式的臂直接比 entropy 数值是没有意义的。**
2. entropy 的变化幅度（~15%）远小于运行间方差对其他指标的影响，n=1 下不足以下结论。

## 13. clipfrac 高到底意味着什么？

本项目里 clipfrac **从来没高过**，最大 2.7e-4。但这个"没高过"本身产生了三条有价值的结论：

**(a) 精确为 0 不是 bug。** R0 的 `ppo_kl` 和 `pg_clipfrac` 都恰好是 0，因为
`ppo_mini_batch_size == train_batch_size` 且 `ppo_epochs=1` → 一个梯度步 → θ = θ_old →
ρ = exp(0) = 1。**恰好为 0 反而证明 old-logprob 记账是对的。**

**(b) 语义我一开始映射错了。** `pg_clipfrac` 本来就是**双向**的
（`gt(pg_losses2, pg_losses1)`，`core_algos.py:1271`）；
`pg_clipfrac_lower` 是 `clip_ratio_c` 的 **dual-clip 分支**、只在 advantage<0 时有效
（`L1275`）。我把两个加起来当"总 clip 比例"是重复计数。

**(c) clip-higher 的效果方向对但幅度小。** 上界 0.2→0.28 让 clipfrac 从 1.74e-4 降到
1.26e-4（−28%）。幅度小是因为偏离 >20% 的 token 只占 0.05%，而 (1.2, 1.28] 是其中一小片。

**(d) GSPO 的 clipfrac 恒为 0（连续两次实验共 23 个 update）**，说明它的 trust region
从未被触碰 —— sequence-level clipping 在这个配置下是 **inert 机制**。

## 14. excessive LR 的实际症状是什么？

**NOT RUN.** R3-B 失败注入没有执行。

## 15. verifier failure 怎样污染 reward？

**注入实验 NOT RUN**（R3-C）。但 verifier 本身的特性量清楚了
（[`verifier_semantics.md`](verifier_semantics.md)、[`verifier_unit_test.md`](verifier_unit_test.md)）：

- reward 是 **+1.0 / −1.0**，不是 +1/0
- **parse failure 和答错拿到完全相同的 −1.0**，从 score 上分不开，
  只能靠 `pred == "[INVALID]"` 区分。**这就是污染的入口**
- 只看回答**最后 300 个字符**（boxed 回退只看 100）。构造的 40 个"答案正确但后面有废话"
  的 case **一条都没过**；真实 rollout 里 6.05% 是这个原因
- `Answer: 34.`（结尾句号）判错。目前这个模型不这么写，但 RL 会改输出分布
- 延迟 0.017 ms、0% 异常。**verifier 永远不是吞吐瓶颈**（实测 `timing_s/reward` = 9e-5 秒，
  比 rollout 小六个数量级）

## 16. checkpoint/resume 有什么坑？

**NOT RUN.** 所有运行都用 `save_freq=-1`，没有做 save/resume 测试。

但发现了两个相邻的 provenance 坑（[`BUG_LOG.md`](BUG_LOG.md) #8、#9）：
`resolved_config.yaml` 从头到尾是占位符（launcher 压根没写）；
manifest 里的数据集 hash 指向 `data/raw/`，而所有训练用的都是去重后的文件。
**provenance 记错比不记更糟 —— 不记你知道自己不知道，记错了你以为自己知道。**

## 17. GPU 利用率低的时候瓶颈在哪里？

R0 update 2（稳态）的分解：

| 阶段 | 占比 |
|---|---|
| **rollout** | **53.4%** |
| logprob (old + ref) | 14.2% |
| actor update | 22.8% |
| weight sync | 9.5% |

**rollout 主导。** update 1 的 logprob 占 39.1% 是一次性的 —— reference model 是
CPU offload 的（`ref.fsdp_config.param_offload=True`），第一次调用要付 host→device 传输
（`timing_s/ref` 21.11s → 2.70s）。

verifier 占比 **不可测**（`timing_s/reward` 在主 trainer 里不发），
但独立量到 0.02 ms/样本，可以直接排除。

## 18. rollout 和 actor update 各占多少 wall time？

见上。另外 DAPO 因为 dynamic sampling，rollout 占比从 ~43% 升到 ~63%
（13.2/31.0 vs 25.5/40.7）。

**DAPO 去掉 KL 还省了一整趟 reference 前向** —— R2 的指标里
`timing_s/ref`、`actor/kl_loss`、`actor/kl_coef` 整个字段不存在，
峰值显存 48.9 → 46.1 GB。所以 dynamic sampling 的翻倍被抵消了一部分，
稳态 step 只贵约 30% 而不是 100%。

## 19. vanilla GRPO → DAPO 后最明显的 dynamics change 是什么？

**batch 组成，而且它导致 reward 不可比。**

| | GRPO s20 | DAPO s20 |
|---|---|---|
| 全对组（reward +1） | **5** | **0** |
| 全错组 | 2 | **0** |
| 混合组 | 9 | **16** |
| batch 均值 | **+0.328** | **+0.016** |

GRPO 到后期攒了 5 个全对组把均值拉高，DAPO 按定义把它们全扔了。

**推论：开了 dynamic sampling 的实验，reward 均值不能跟没开的比。**
这一条不看 batch 组成就会得出"DAPO 训得差"的完全错误结论。

## 20. 哪些所谓"模型训练失败"实际上是系统/数据/verifier 故障？

这是本项目收获最大的一条。完整 24 条见 [`BUG_LOG.md`](BUG_LOG.md)，最典型的：

| 现象 | 看起来像 | 实际是 |
|---|---|---|
| reward −0.746，准确率 12.7%，79.7% 零梯度组 | 模型很差 | **512 条里只有 2 条真答错**，87% 被长度上限截断 |
| `flash-attn` 装不上，timeout | 依赖版本问题 | 网络分裂路由（wheel 重定向到被墙的 github.com） |
| `MPClient: engine core exited unexpectedly` | vLLM 崩了 | 我自己脚本的 `TypeError`，引擎是跟着父进程死的 |
| 装依赖 0.1 MB/s | 机器网络差 | PyPI 特慢 + `uv --frozen` 无视 index 覆盖（模型同时在以 30 MB/s 下载） |
| D 臂 entropy 0.106 vs 0.32 | Dr.GRPO 导致 entropy 崩塌 | metric 随 `loss_agg_mode` 归一化，底层 entropy 没变 |
| 答错的回答比答对长 60% | Dr.GRPO 说的优化偏差 | **85% 是难度混杂**，同题内配对后只剩 9% |
| 算法 A 的 reward 比 B 高 | A 更好 | **同一算法两次运行 reward 就能符号翻转** |

**共同点：报错信息指向的地方，和真正的问题，经常不是一回事。**

---

## 最后一条：这次实验设计本身的教训

四臂消融跑完之后，我用两次**同一算法**的独立运行量了运行间方差：

| | 同算法两次运行 | 四个不同算法之间 |
|---|---|---|
| reward | **符号翻转** | — |
| 回答长度 | **±65%** | ±7% |
| validation | — | 最大 0.84 个标准误 |
| ratio p99 | **±1.0%** | GSPO 差 100 倍（真实差异） |
| advantage max | **±2.7%** | Dr.GRPO 差 28%（真实差异） |

**结果类指标的运行间方差，比算法间差异大一个量级。机制类指标复现到 5% 以内。**

所以这个项目在 n=1、20 update 的预算下：

- ✅ **能验证机制**（GSPO 压缩 ratio、DAPO 填满 batch、Dr.GRPO 改 advantage 尺度）
- ❌ **不能比较效果**（谁的 validation 更高全在噪声里）

**下一步真正需要的不是更多算法臂，而是每臂 3+ seed 和 100+ updates。
不做这两件事，再加多少算法都只是在测噪声。**

---

## 状态清单

| 项 | 状态 |
|---|---|
| 硬件 / CUDA13 / NCCL / verifier gate | **PASS** |
| R0 smoke · R1 baseline · R2 DAPO · GSPO smoke | **PASS** |
| ratio 分布观测补丁 | **PASS**（58 行插入 0 行删除，CPU 自测） |
| M20 三臂 matched（200 题验证集） | **PASS** |
| 四臂消融（1500 题验证集） | **PASS** |
| E_drgrpo_scaled（修正 loss_scale_factor） | 见 git 历史 |
| **R3-A 截断污染注入** | **NOT RUN**（但在预训练前 rollout 里意外复现了） |
| **R3-B 超大 LR** | **NOT RUN** |
| **R3-C verifier timeout 污染** | **NOT RUN** |
| **checkpoint / resume 测试** | **NOT RUN** |
| **VAPO** | **上游未实现**，只有可行性报告 |
| **多 seed 重复** | **NOT RUN** ← 当前最大的缺口 |
