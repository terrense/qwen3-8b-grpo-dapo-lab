# Dataset manifest

## Training: BytedTsinghua-SIA/DAPO-Math-17k

| | |
|---|---|
| Repository | `BytedTsinghua-SIA/DAPO-Math-17k` (HF, `repo_type=dataset`) |
| Downloaded | 2026-09-10 via `hf download`, endpoint `https://hf-mirror.com` |
| Local path | `$LAB/data/raw/DAPO-Math-17k/data/dapo-math-17k.parquet` |
| File size | 299363855 bytes (286M) |
| sha256 | `534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941` |
| **Total rows** | **1,791,700** |
| **Unique prompts** | **17,917** |
| **Repeat factor** | **exactly 100x per prompt** (min repeats = max repeats = 100) |
| `data_source` | `math_dapo` (single value) |
| `ability` | `MATH` (single value) |

### The row count is a trap — read this before setting batch sizes

The file ships **1,791,700 rows, but only 17,917 distinct problems**: every prompt appears
**exactly 100 times**, keyed by `extra_info.index` (a UUID). Verified by counting distinct
`extra_info.index` values and their multiplicities — `min = max = 100`.

Consequences that must be handled before R1:

- `trainer.total_epochs=1` over the raw file is **100 passes over the same 17,917 prompts**, not
  one. Any "epoch" reasoning based on the file length is wrong by 100x.
- Validation/train contamination and prompt-level dedup must be done on `extra_info.index`, not
  on row identity.
- A shuffled batch drawn from the raw file will contain **duplicate prompts within the same
  batch** with high probability. For GRPO that is not neutral: the same prompt appearing twice in
  one optimizer step gets two independent groups, silently double-weighting it.

The 100x repetition is how the upstream DAPO recipe feeds a fixed prompt set for many steps. It is
intentional upstream, but it must be an explicit choice here, not an accident.

### Schema (read from the real file, not assumed)

```
data_source  : string                                    -> "math_dapo"
prompt       : list<struct<content: string, role: string>>  -> chat-format, single user turn
ability      : string                                    -> "MATH"
reward_model : struct<ground_truth: string, style: string>
                 .ground_truth -> the reference answer, as a string
                 .style        -> "rule-lighteval/MATH_v2"
extra_info   : struct<index: string>                     -> UUID, stable prompt identity
```

Field mapping for VeRL: **prompt** = `prompt` (chat list), **ground truth** =
`reward_model.ground_truth`, **data source** = `data_source`, **reward metadata** =
`reward_model.style`.

### Sample records

```
sample 0
  data_source  : "math_dapo"
  prompt       : [{role: "user", content: "Solve the following math problem step by step.
                  The last line of your response should be of the form Answer: \ ..."}]
  ability      : "MATH"
  reward_model : {ground_truth: "34", style: "rule-lighteval/MATH_v2"}
  extra_info   : {index: "9a9b6eb4-a1cb-49d1-8c1e-62eaf2f74079"}

sample 1   reward_model.ground_truth = "113"   index = b426d104-244d-4831-a2c4-cd756b61700a
sample 2   reward_model.ground_truth = "-3"    index = 6ff0b17f-7e5c-4ae9-b5e9-63ebecd2b9f7
```

The prompt carries its own answer-format instruction (`Answer: $Answer` on the last line), which
is what the rule-based verifier will parse. Format-following is therefore part of the reward, and
`format success rate` is tracked as a separate metric from answer correctness.

## Validation: AIME 2024

**NOT PREPARED YET.** To be added with source repo, revision, count and schema once downloaded.
