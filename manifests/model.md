# Model manifest

| | |
|---|---|
| Repository | `Qwen/Qwen3-8B` (HuggingFace) |
| Variant | **post-trained release** — NOT `Qwen3-8B-Base`, NOT a VL variant |
| Downloaded | 2026-09-10 via `hf download`, endpoint `https://hf-mirror.com` |
| Local path | `$LAB/models/Qwen3-8B` (data disk; system disk untouched) |
| Total size | 16 GB |
| Shards | 5 x `model-0000N-of-00005.safetensors` |
| config.json sha256 (first 16) | `f7c4eadfbbf52247` |

## Files

```
        1570  .gitattributes
       11343  LICENSE
       16660  README.md
         728  config.json
         239  generation_config.json
     1671853  merges.txt
  3996250744  model-00001-of-00005.safetensors
  3993160032  model-00002-of-00005.safetensors
  3959604768  model-00003-of-00005.safetensors
  3187841392  model-00004-of-00005.safetensors
  1244659840  model-00005-of-00005.safetensors
       32878  model.safetensors.index.json
    11422654  tokenizer.json
        9732  tokenizer_config.json
     2776833  vocab.json
```

## Why this checkpoint

The paper design is `Qwen3-8B -> SFT M1 -> DPO M2 -> GRPO M3`. This pilot applies GRPO directly to
the released post-trained model, so its results are a **GRPO systems pilot / rehearsal** and must
never be reported as M3. Using `-Base` instead would change the starting policy entirely and is
explicitly out of scope; no switch to `-Base` may happen without an explicit decision.

## Verified

vLLM 0.24.0 loads this checkpoint in bfloat16 and generates coherent text (gate E,
[../analysis/PRE_RL_GATE_REPORT.md](../analysis/PRE_RL_GATE_REPORT.md)). Load 118.3 s,
40.0 GiB KV cache at `gpu_memory_utilization=0.60`, peak 60.11 GiB on one H20.
