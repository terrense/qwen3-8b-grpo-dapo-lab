# 踩坑记录

这个文件就是流水账，专门记这个项目里所有把我坑过的东西。按踩到的顺序排，不分大小。

写这个的原因很简单：这些坑里有一半以上，报错信息指的地方跟真正的问题**根本不是一回事**。下次再遇到，直接翻这里，别再花一遍时间。

正式的事故分析在 [`incident_log.md`](incident_log.md)，格式规整、带证据链。这个文件更随便，什么都往里扔。

---

## 环境和网络

### 1. 三个主机，三种走法，互相冲突

这台机器最烦的地方。同一条 `uv sync` 命令要同时访问的主机，走法是矛盾的：

| 主机 | 直连 | 走代理 |
|---|---|---|
| `github.com` / `huggingface.co` | ✗ | ✓ |
| `download.pytorch.org` | ✓ | **✗ 被代理挡** |
| `files.pythonhosted.org` | 40 KB/s | 40 KB/s |
| `mirrors.aliyun.com` | **63 MB/s** | — |

所以既不能全开代理，也不能全关。解法是开代理 + 把 `download.pytorch.org` 塞进 `no_proxy`。

报错长这样，看着像 flash-attn 这个包有问题：

```
x Failed to download `flash-attn==2.8.3`
`-> operation timed out
```

实际跟包一点关系没有 —— wheelhouse 的索引页在 `github.io` 上能通，但 wheel 本身重定向到 `github.com/releases/download/...`，那个不走代理进不去。

**教训**：在墙内的机器上装东西失败，先把每个主机的连通性表打出来，再动版本号。差点就手贱去删 `flash-attn` 这个 pin 了 —— 那等于用改训练栈的方式绕一个网络问题。

### 2. uv 起的 git 子进程不认环境变量里的代理

修完上面那个，`uv sync` 又死在别的地方：

```
fatal: unable to access 'https://github.com/ISEEKYAN/mbridge.git/':
Recv failure: Connection reset by peer
```

但我手动 `git ls-remote` 同一个仓库，连着试三次全成功。

所以不是仓库不通，是 uv spawn 出来的 `git` 没继承到代理。**给 git 自己配，别指望环境变量**：

```bash
git config --global http.https://github.com/.proxy http://<代理>
```

按 host 配，这样 `download.pytorch.org` 不受影响。

### 3. `uv sync --frozen` 会无视 `UV_DEFAULT_INDEX`

这个坑最贵，卡了我最久。

装依赖慢到离谱 —— 60 秒才 6 MB，剩下还有 5 GB，照这速度环境要装一个半小时。先怀疑机器网络差，但模型那边刚从 hf-mirror 拉了 16 GB，跑到 30 MB/s，所以带宽没问题。

量了一下同一个 vllm wheel 在不同源上的速度：

| 源 | 速度 |
|---|---|
| `mirrors.aliyun.com` | **63.01 MB/s** |
| `mirrors.bfsu.edu.cn` | 48.14 MB/s |
| `repo.huaweicloud.com` | 8.95 MB/s |
| `pypi.tuna.tsinghua.edu.cn` | 6.57 MB/s |
| `mirrors.ustc.edu.cn` | 0.83 MB/s |
| `files.pythonhosted.org` | **0.04 MB/s** |

差 1500 倍。而且镜像之间自己就差 75 倍 —— 要是不测直接凭印象选，很可能就选了 tuna 或者 ustc，白慢十倍。

然后设了 `UV_DEFAULT_INDEX` 指向 aliyun，**一点用没有**，还是 1 MB/s 左右。

怎么确认的：把 uv 进程 `/proc/<pid>/fd` 里的 socket inode 抓出来，跟 `/proc/net/tcp` 对一遍，看它到底连的是谁。结果连的是 `151.101.64.223` —— Fastly，也就是 `files.pythonhosted.org`。

**根因**：`uv.lock` 里每个包记的是**完整 URL**，`--frozen` 的意思就是「完全照 lock 装」，包括主机名。index 覆盖只影响 resolution，而 `--frozen` 压根跳过 resolution。

解法是直接把 lock 里的 URL 换主机（aliyun 的路径结构跟 PyPI 一模一样）：

```bash
sed -i 's|https://files\.pythonhosted\.org/|https://mirrors.aliyun.com/pypi/|g' uv.lock
```

**版本一个没动，1413 个 sha256 一个没动，uv 照样逐个校验**。0.1 MB/s → 46.6 MB/s，快了 460 倍。

**教训**：知道你加的那个 flag 到底关掉了什么。`UV_DEFAULT_INDEX` 看着就是标准答案，实际是个空操作，我在它上面浪费了两轮。看进程真正连的是谁，比猜配置快。

---

## 自己写的代码坑自己

### 4. 贵的那一步跑完了，才在后处理炸

长度预算探测脚本，A 组（thinking 模式、16384 上限）在 GPU 上跑满整整 12 分钟，生成全做完了，然后：

```
TypeError: dict() got multiple values for keyword argument 'state'
```

一个结果都没存下来。

原因蠢得很：`score()` 返回的 dict 里已经有 `state` 这个 key，我又写了 `dict(..., state=..., **s)`，等于传了两遍。这个错**必然在第一条打分的样本上就触发**，一秒钟就能测出来。

而且日志最后一行是：

```
MPClient: engine core exited unexpectedly; starting cleanup
```

看着像 vLLM 崩了。**不是**。那是父进程死了之后引擎跟着关，是结果不是原因。真正的 traceback 在它上面几行。

**两个教训**：

1. **看 traceback，不看最吓人的那一行。** 差点往 vLLM/CUDA 方向查过去。
2. **任何跑在贵阶段之后的代码，必须先用假数据过一遍。** 后处理代码天生就有这个毛病 —— 它只有在贵的部分做完之后才会执行。现在脚本里加了个 `_selftest()`，模型都不加载就先跑。

### 5. 两个 metric 我映射错了，而且错得很像对的

这个最值得记，因为它**正好是这个项目要研究的那类错误**，只不过发生在我自己的监控代码里。

**（a）`response_length/clip_ratio` 不是截断率。**

我一直当它是「多少条撞到 `max_response_length`」。看 `metric_utils.py:460`：

```python
max_response_length = batch.batch["responses"].shape[-1]
```

是**padded tensor 的宽度**，不是配置里的上限。所以这个数只是「有多少条正好等于这批里最长的那条长度」，几乎必然非零。

R0 自己的数就能证伪：step 1 报 `clip_ratio=0.03125`，但 `response_length/max` 是 **3509**，上限是 **4096** —— 根本没有任何一条到过上限。

**（b）`pg_clipfrac` / `pg_clipfrac_lower` 不是上侧/下侧 clip。**

`core_algos.py:1271`：

```python
pg_clipfrac = masked_mean(gt(pg_losses2, pg_losses1))   # 双向都算
pg_clipfrac_lower = masked_mean(gt(clip_pg_losses1, pg_losses3) * (advantages < 0))  # dual-clip 分支
```

前者本来就是双向的，后者是 `clip_ratio_c` 那个 dual-clip 分支、只在 advantage 为负时才管。我把两个加起来当「总 clip 比例」，纯粹是重复计数。

**影响范围我写清楚了**：R0 报告里「截断从 3.13% 涨到 9.38%」这句作废，R1 启动脚本注释里引用同一个数的地方也作废。但**预训练前 rollout 那个 87.11% 截断率没事** —— 那是我自己脚本里用 vLLM 的 `finish_reason == "length"` 量的，信号是对的。

改法是在 R0 报告里**加一段更正**，不是把数字悄悄改掉。

### 6. 自己算的指标跟框架量的对不上，那就是自己错

`policy_staleness_steps` 我是拿相邻 step 号相减算的，得到 `0, 1`。但 verl 自己就有 `training/off_policy/trajectory_staleness`，量出来是 `0, 0`。

同步 GRPO 本来就该恒为 0，所以是我算错了。直接删掉自己那套推导，改用框架的原生指标。

**教训**：框架已经量了的东西别自己再推一遍。那是观测层开始编事实的起点。

### 7. observer 和 trainer 各写各的路径，还不报错

R1 跑完发现 group 统计全空。原因：

- launcher 里设了 `DUMP_DIR="$LAB/logs/$RUN_UID/rollout_dump"` 但**没 export**
- 训练脚本自己的默认值是 `$LAB/logs/R1_rollout_dump`

于是 trainer 往一个地方写，observer 去另一个地方读，**两边都不报错，就是静默返回空**。

这种坑最阴 —— 要是没人核对，R1 的 group signal 分析就整段是空的，还看不出来。

### 8. `resolved_config.yaml` 一直是个占位符

报告里引用它，每个 incident bundle 里都拷它，结果它从头到尾都是 `# Written by the trainer at launch time. NOT RUN yet.`。launcher 压根没写过。

Hydra 其实是会输出的（`outputs/<日期>/<时间>/.hydra/config.yaml`），加个函数从那儿拷出来就行。

### 9. manifest 里记的数据集 hash 指的是没用过的文件

launcher 里 `DATA_PARQUET` 默认值写的是 `data/raw/...`，但所有训练用的都是去重后的 `data/dapo_math_17k/train.parquet`。所以 manifest 记了个跟这次运行完全无关的文件的 hash。

provenance 记错比不记更糟 —— 不记你知道自己不知道，记错了你以为自己知道。

---

## 数据集

### 10. DAPO-Math-17k 每条重复了正好 100 遍

原始 parquet **1,791,700 行，但只有 17,917 个不同 prompt**，按 `extra_info.index` 数，每条正好出现 100 次（min = max = 100）。

要是直接拿原文件训：

- `total_epochs=1` 实际是把同一批 prompt 过 100 遍
- 打乱之后**同一个 batch 里会出现重复 prompt**，GRPO 里这等于给那道题偷偷加了双倍权重

已经去重了，训练一律用 `data/dapo_math_17k/` 下的 split。

---

## verifier

### 11. 官方 verifier 只看回答的最后 300 个字符

`math_dapo.compute_score` 第一行就是：

```python
solution_str = solution_str[-300:]
```

boxed 那条回退路径更狠，只看最后 100 个。

所以答案对了、但后面跟了一段废话，就判错。构造的 40 个这种 case **一条都没过**，真实 rollout 里 6.05% 是这个原因。

顺带：`Answer: 34.`（结尾带个句号）也判错。`SUBSTITUTIONS` 里那条 `(".$", "$")` 只去掉 `$` 前面的句号，光秃秃一个句号留在那儿，字符串精确比较就挂了。目前这个模型不这么写，但 RL 是会改输出分布的，随时可能踩上。

还有一点：**parse failure 和答错拿到的 reward 一样是 −1**，从 score 上分不开，只能靠 `pred == "[INVALID]"` 自己区分。

---

## 长度预算 —— 这个项目最大的一个坑

### 12. 看着像模型很差，其实是被截断

预训练前的固定策略 rollout，64 prompt × 8 样本：

- reward 平均 **−0.746**
- 准确率 **12.7%**
- zero-std group **79.69%**

任何人看这三个数都会说模型不行。

但 512 条里**只有 2 条是真的答错**。

- 512/512 全部以 `<think>` 开头（Qwen3-8B 的 chat template 默认开思考）
- 长度中位数**正好卡在 4096 上限**
- 截断的样本是在推理中间被切断的，既没有 `</think>`，也没有 `Answer:`，也没有 `\boxed{}`

只改长度预算、权重一个字节没动：

| 配置 | 截断率 | 准确率 | 每样本 token |
|---|---|---|---|
| thinking @ 4096 | 87.11% | 12.70% | — |
| thinking @ 16384 | 18.75% | **56.25%** | 8820 |
| 关 thinking @ 4096 | 1.56% | 31.25% | **1477** |

12.7% → 56.25%。

关 thinking 那组虽然准确率不是最高，但**每个 token 能换到的梯度信号最多**（mixed group 43.75%，token 数只有 1/6），所以后面正式跑用的是关 thinking。

关的方法：`+data.apply_chat_template_kwargs.enable_thinking=False`

---

## 训练配置

### 13. `ppo_kl` 和 `clipfrac` 恰好为 0 不是 bug

R0 两个 update 的 `ppo_kl` 和 `pg_clipfrac` 都是**精确的 0**。第一反应是坏了。

其实是对的：`ppo_mini_batch_size == train_batch_size` 且 `ppo_epochs=1` 的时候，一个 rollout batch 只有一个梯度步，而且用的就是生成这批数据的那份权重，所以 `old_logprob ≡ current_logprob`，ρ = exp(0) = 1，clip 永远不可能触发。

**恰好为 0 反而说明 old-logprob 记账是对的** —— 这个配置下要是非 0，那才是 bug。

但后果很实在：**任何关于 ratio 或者 clipping 的实验，在这个配置下都是空的**。R1 改成 `mini=8 < train=16`，ρ 确实不等于 1 了，可 clipfrac 只有 **1.5e-4** —— lr 1e-6 下两个 mini 步几乎推不动策略，ρ ≈ 1 还是成立。

所以 GSPO 那组要是照这个 LR 跑，比的是两个都约等于 1 的 ratio，还是白跑。得把 `train/mini` 比例拉大或者 LR 提上去。

---

## 工具链杂七杂八

### 14. `.gitignore` 里后面的规则会翻掉前面的

`__pycache__/` 明明写在第 40 行，`.pyc` 还是被追踪了。因为后面有一条 `!scripts/**` 把 `scripts/` 下面所有东西重新包含回来了，包括 `__pycache__`。

**gitignore 里最后一条匹配的规则赢**。重新 ignore 的那几行必须放文件末尾。

### 15. `.gitignore` 的行内注释会被当成模式的一部分

我写了：

```
system/preflight.txt          # raw: 含 hostname，改用 manifests/ 里脱敏的那份
```

gitignore **不支持行尾注释**，整行连注释一起被当成文件名模式，于是什么都没匹配上，`preflight.txt`（里面有 hostname 和 GPU UUID）差点就进了公开仓库。

`env.sh` 那条也是同样的问题。第一次 commit 之前抓出来了。

### 16. ssh heredoc 里带撇号会炸

`ssh host 'cat > file' <<EOF` 这个写法，内容里只要有单引号（比如英文的 `Update 1's`）就报 `unexpected EOF while looking for matching`。改成本地写文件再 `scp` 过去。

写这份日志的时候又踩了一次，同一个坑，第四回。

### 17. GitHub 的 KaTeX 不认 `\operatorname`

README 里公式全是红框：

```
The following macros are not allowed: operatorname
```

而且一个宏挂了整块公式都不渲染，直接把 `$$...$$` 原样吐出来。

换成 `\mathrm{}` 就好了。`\lvert` `\rvert` `\big` `\Big` `\qquad` `\mathbf` `\mathcal` `\mathbb` `\underbrace` 这些都没问题。

### 18. vLLM 的进度条前期 ETA 严重虚高

64 个 prompt × 8 样本的 rollout，进度条一开始报 **ETA 1 小时 20 分**，吞吐 222 tok/s。差点就去砍任务规模了。

实际上 512 个请求是**并发跑的，按批次成组完成**，进度条只在请求结束时才跳。等批次填满之后吞吐涨到 **2709 tok/s**，全程 12.5 分钟跑完。

**别信前期 ETA，看 GPU 利用率和输出文件的行数。**

### 19. DAPO recipe 的 hydra searchpath 是相对路径

`dapo_trainer.yaml` 里写的是：

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config
```

相对路径，按**当前工作目录**解析。官方是从 verl 仓库根目录跑 `python3 -m recipe.dapo.main_dapo`，那个路径才存在。我在 recipe 目录里直接跑 `main_dapo.py`，于是：

```
In 'dapo_trainer': Could not load 'ppo_trainer'.
```

解法：cd 到 verl 根目录再跑。

### 20. Ray 的 uv hook 对着 project 外的脚本会挂

修完 19 之后换了个错：

```
File "ray/_private/runtime_env/uv_runtime_env_hook.py", line 34, in _is_path
TypeError: path_or_uri must be a string, got <class 'NoneType'>.
```

Ray 检测到是 `uv run` 起的，就想从 uv project 里推算脚本路径，推不出来就 None。R0/R1 用 `main_ppo` 没这问题，因为那个模块本来就在 verl 包里。

把 recipe 软链到 `repos/verl/recipe/dapo` 再用 `-m recipe.dapo.main_dapo` 还是一样的错，说明软链没骗过它。

拿 `HYDRA_FULL_ERROR=1` 打出完整调用链才看清：

```
verl/trainer/main_ppo.py:74      ray.init(**OmegaConf.to_container(ray_init_kwargs))
ray/_private/worker.py:1396      return hook(runtime_env)
ray/.../uv_runtime_env_hook.py:411   if _is_path(working_dir):
```

真凶在 verl 自己这边，`verl/trainer/constants_ppo.py:118-121`：

```python
runtime_env = {
    "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
    **({"working_dir": None} if working_dir is None else {}),
}
```

verl **故意**把 `working_dir` 塞成 `None`，意思是「别上传工作目录」。而 Ray 那个 hook 的逻辑是：

```python
if "working_dir" not in runtime_env:
    runtime_env["working_dir"] = os.getcwd()
working_dir = runtime_env["working_dir"]
if _is_path(working_dir):     # None 进来就炸
```

键**存在但是 None**，所以既躲过了默认值那一步，又过不了类型检查。
说白了就是 verl 和 Ray 2.55.1 对「不要 working_dir」这件事的表达方式不兼容。

**解法**：`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`。

这个 hook 干的事就是告诉 Ray worker 用哪个解释器 —— 而我本来就显式传了
`ray_kwargs.ray_init.runtime_env.py_executable`，完全重复，关掉没有副作用。
关掉之后 `DAPOTaskRunner` 正常起来了。

**还没搞清楚的**：R0/R1 用完全一样的 `uv run` + `py_executable` 组合走
`-m verl.trainer.main_ppo`，**没碰到这个错**。`RAY_ENABLE_UV_RUN_RUNTIME_ENV`
默认是开的，两边都该触发 hook。为什么只有 `-m recipe.dapo.main_dapo` 会挂，
目前没有解释。先记下来，不编。


### 21. 又一次拿均值推断分布，而且是在自己写下警告之后

R1 和 R2 的 `actor/ppo_kl` 都在 1e-5 量级，`pg_clipfrac` 在 1e-4 量级。
我据此写了「ρ ≈ 1，所以 clip-higher 和 GSPO 在这个配置下都测不出东西」，
还写进了 README、algorithm_matrix 和三份算法笔记。

**错的。**

`ppo_kl` 是**带符号均值**（`core_algos.py:1339`）：

```python
negative_approx_kl = log_prob - old_log_prob
ppo_kl = masked_mean(-negative_approx_kl, response_mask)
```

均值接近 0 既可能是分布窄，也可能是正负偏差抵消。而 `pg_clipfrac` 只数越过 ±0.2 的尾巴，
对边界以内的形状一无所知 —— 偏偏 1%~5% 那一段才是 GSPO 关心的地方。

加了个纯观测钩子把分布量出来，同样的配置（mini=8，2 个梯度步，lr 1e-6）：

| | 值 |
|---|---|
| \|ρ−1\| 中位数 | 3.9e-06 |
| \|ρ−1\| p99 | **0.055** |
| \|ρ−1\| 最大 | **0.30 → 0.37 → 0.42**（三步内在变宽） |
| ρ 范围 | **0.730 ~ 1.416** |
| 偏离 >1% 的 token | **8%** |
| 偏离 >5% 的 token | **3%** |

尖峰厚尾，不是窄。均值 1.0001 只是把 0.73 和 1.42 平均了一下。

**抵消的直接证据**：三个 update 的 `ppo_kl` 符号是 **+、−、+** 翻转的。
真正窄的分布不会这样来回翻，它会稳定贴着 0。

结论修正：
- **GSPO 有东西可测**，不需要先改配置 —— 撤回原说法
- **clip-higher 作用小这条保留**，但理由从「分布窄」换成「(1.2, 1.28] 这个窄带里
  token 太少」（>20% 偏离的只占 0.05%）

最难受的是：我在自己写的 `docs/algorithms/gspo.md` 里明明写了
「不要只记录 mean，因为 mean ratio ≈ 1 不能排除 heavy tails」，然后自己就这么干了。

**这是这个项目第二次栽在同一件事上。** 第一次是 pre-RL rollout —— reward 均值 −0.746
看着像模型差，实际 512 条里只有 2 条真答错。

**均值不是分布。** 在下「有没有东西可测」这种结论之前，先把分布量出来，
成本就是几行观测代码。


### 22. GSPO 的配置键名我是按注册名猜的，猜错了

`@register_policy_loss("gspo")` 注册的名字是 `gspo`，我就顺手写成：

```
actor_rollout_ref.actor.policy_loss.policy_loss_mode=gspo
```

Hydra 直接拒了：

```
Key 'policy_loss_mode' is not in struct
    full_key: actor_rollout_ref.actor.policy_loss.policy_loss_mode
```

真实的键是 **`actor_rollout_ref.actor.policy_loss.loss_mode`**。
`policy_loss_mode` 这个名字确实存在，但在完全无关的
`distillation.distillation_loss.policy_loss_mode` 下面。

怎么查的 —— 直接把 generated config 当字典遍历，比 grep 靠谱：

```python
import yaml
d = yaml.safe_load(open("verl/trainer/config/_generated_ppo_trainer.yaml"))
print(d["actor_rollout_ref"]["actor"]["policy_loss"])
# {'_target_': ..., 'loss_mode': 'vanilla', 'clip_cov_ratio': ..., ...}
```

**打脸的地方**：项目一开始我就给自己定过规矩「任何 config key 都必须通过当前源码确认，
不准凭空创造」。结果这次从函数的注册名反推键名，没验就写进了脚本、configs/、
还有两份文档。四个文件一起错。

好在 Hydra 是 fail-fast 的，模型都没加载就挂了，没烧 GPU。
但如果这个键恰好存在于别处、只是语义不对，那就会静默跑出错误的实验 —— 那才是真的贵。

**规矩补一条**：验配置键不要 grep 名字，要把 config 当字典按完整路径查。
grep 会在无关的段落里给你假阳性 —— 这次 `policy_loss_mode` 就真的在
distillation 下面存在。

---

## 一句话总结

这 20 条里，**报错信息直接指向真凶的只有个别几条**。`flash-attn` 那个其实是网络路由，`mbridge` 那个是子进程不继承环境变量，`MPClient` 那个是我自己的 TypeError，「模型很差」那个是长度上限。

所以这个项目的核心那句话 —— **reward 不是 ground truth，它是一整条流水线的输出** —— 不光对训练成立，对工具链也一样成立。
