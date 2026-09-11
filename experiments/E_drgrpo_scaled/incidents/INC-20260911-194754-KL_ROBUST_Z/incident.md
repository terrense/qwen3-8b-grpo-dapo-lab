# INC-20260911-194754-KL_ROBUST_Z

**Level:** YELLOW
**Rule:** `kl_robust_z`
**Metric:** `kl`
**Step:** 7
**Timestamp:** 2026-09-11T19:47:55

## Observed symptom

kl=2.2035671804587764e-05 is +5.0 robust-z from the rolling median of the last 6 updates

Value at detection: `2.2035671804587764e-05`
Robust z-score vs rolling median: `4.96`


## Detection rule

`kl_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=2 | reward_mean=-0.28125 | kl=-4.831214255318628e-07 | entropy=0.33847203850746155 | clip_fraction=7.768667046548217e-05 | grad_norm=0.1514272838830948 | response_length_mean=2044.828125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=44.35742050409317 | t_actor_update=37.32672253251076
global_step=3 | reward_mean=-0.46875 | kl=-1.824102332648181e-05 | entropy=0.3553938567638397 | clip_fraction=5.771448240921018e-05 | grad_norm=0.16950258612632751 | response_length_mean=1955.0078125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=44.36041585728526 | t_actor_update=37.392739068716764
global_step=4 | reward_mean=-0.09375 | kl=2.0949772533640498e-05 | entropy=0.24460768699645996 | clip_fraction=9.196723704008036e-05 | grad_norm=0.1634281426668167 | response_length_mean=1666.359375 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=46.371352173388004 | t_actor_update=31.39823703095317
global_step=5 | reward_mean=-0.015625 | kl=-2.149262945749797e-05 | entropy=0.26476800441741943 | clip_fraction=0.00018846634702640586 | grad_norm=0.20192071050405502 | response_length_mean=1704.1171875 | zero_std_group_ratio=0.125 | effective_signal_fraction=0.875 | t_rollout=30.34441814571619 | t_actor_update=30.967726048082113
global_step=6 | reward_mean=-0.3125 | kl=-1.451880848435394e-05 | entropy=0.2818373441696167 | clip_fraction=0.00015817550320207374 | grad_norm=0.20302365720272064 | response_length_mean=1680.3203125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=46.3612158857286 | t_actor_update=32.23388513177633
```

## Metrics at the event

```
global_step=7 | reward_mean=0.078125 | kl=2.2035671804587764e-05 | entropy=0.26395779848098755 | clip_fraction=0.0001316406301157258 | grad_norm=0.17521021887660027 | response_length_mean=1848.328125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=46.373244162648916 | t_actor_update=33.70860982686281
```

## Initial hypotheses

Work the checklist in [`docs/observability/diagnosis_playbook.md`](../../../../docs/observability/diagnosis_playbook.md)
in order — environment/verifier, data batch composition, truncation, policy-version
freshness, old-logprob correctness, token masking, reward normalisation, optimizer,
LR, and only then KL/clipping.

## Evidence collected

- `metric_window.csv` — 7 updates around the event
- `system_window.csv` — system telemetry, last 600s
- `last_log_lines.txt` — trainer log tail (200 lines)
- `nvidia_smi.txt`, `process_snapshot.txt`
- `resolved_config.yaml`, `run_manifest_snapshot.json` (when available)

## Root cause

检测器误报，不是训练问题。

`actor/ppo_kl` 是 (old_logprob - logprob) 的**带符号均值**（core_algos.py:1339），
天然在 0 附近正负翻转。本次窗口的历史值：

    [-1.5e-05, -0.0, -1.8e-05, +2.1e-05, -2.1e-05, -1.5e-05]

中位数 -1.49e-05、MAD 5.0e-06，当前值 +2.2e-05 只是又一次符号翻转，
但按 robust-z 算出来是 +5.0。

对一个零中心、会正负翻转的量做 robust-z 本身就没有意义 ——
"相对中位数偏离 248%" 这种数字在中位数接近 0 时不可解释。

## Fix

把 `kl` 从 incident_detector.py 的 ADAPTIVE 规则里移除，并在代码里写明原因。
判断策略偏移改用 `actor/ratio_absdev_p99` 和 `actor/ratio_frac_gt_*`
（见 analysis/ratio_distribution.md：ppo_kl 对分布宽度根本不敏感，
梯度步 2->8 时分布宽度翻倍而 ppo_kl 纹丝不动）。

## Post-fix evidence

用真实历史回放确认：修改后 `kl` 已不在 ADAPTIVE 中，同样的输入不再触发。
