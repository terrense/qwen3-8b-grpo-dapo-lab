"""CPU 自测：花 GPU 之前先确认钩子算得对。INC-004 的教训。"""
import torch
from verl.trainer.ppo.core_algos import _verl_lab_ratio_stats

# 构造一个已知分布：一半 token 是 1.0，一半是 1.5
mask = torch.ones(2, 100, dtype=torch.bool)
ratio = torch.ones(2, 100)
ratio[:, 50:] = 1.5
s = _verl_lab_ratio_stats(ratio, mask, "t")
print("== 已知分布 (一半1.0 一半1.5) ==")
for k in ["t_p50","t_p95","t_max","t_mean","t_absdev_p99","t_frac_gt_1pct","t_frac_gt_20pct","t_n_tokens"]:
    print(f"  {k:20s} {s[k]}")
assert s["t_n_tokens"] == 200, s["t_n_tokens"]
assert abs(s["t_mean"] - 1.25) < 1e-5, s["t_mean"]
assert abs(s["t_max"] - 1.5) < 1e-5
assert abs(s["t_frac_gt_20pct"] - 0.5) < 1e-5, s["t_frac_gt_20pct"]

# mask 生效？只算前 50 列 -> 全是 1.0
mask2 = torch.zeros(2, 100, dtype=torch.bool); mask2[:, :50] = True
s2 = _verl_lab_ratio_stats(ratio, mask2, "t")
assert s2["t_n_tokens"] == 100 and abs(s2["t_max"] - 1.0) < 1e-6, s2
print("== mask 生效 ==  n_tokens=", s2["t_n_tokens"], " max=", s2["t_max"])

# 空 mask 不该炸
s3 = _verl_lab_ratio_stats(ratio, torch.zeros(2,100,dtype=torch.bool), "t")
assert s3 == {}, s3
print("== 空 mask 安全 ==")

# 不该有梯度泄漏
r = torch.ones(2,10, requires_grad=True)*1.2
s4 = _verl_lab_ratio_stats(r, torch.ones(2,10,dtype=torch.bool), "t")
assert all(not torch.is_tensor(v) for v in s4.values()), "返回值必须是 python 标量"
print("== 返回纯标量，无梯度泄漏 ==")
print("RATIO_HOOK_SELFTEST_OK")
