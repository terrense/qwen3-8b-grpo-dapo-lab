"""Fixed-policy rollout on Qwen3-8B: 64 prompts x G=8, scored with the SAME
official math_dapo verifier the trainer will use.

Purpose is NOT benchmarking. It answers two gating questions before R0/R1:
  1. Is the reward distribution usable for GRPO? (zero-variance groups produce
     no advantage signal at all)
  2. How often is a mathematically-correct answer marked wrong purely because
     of an output-format / extraction-window mismatch? (format false negative)

Nothing here modifies reward semantics. The verifier state classification is
instrumentation layered on the official return value.
"""

import json
import os
import random
import re
import statistics
import time
from collections import Counter

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score

MODEL = "/root/autodl-tmp/rl_lab/models/Qwen3-8B"
PARQUET = "/root/autodl-tmp/rl_lab/data/raw/DAPO-Math-17k/data/dapo-math-17k.parquet"
OUTDIR = "/root/autodl-tmp/rl_lab/analysis"
DUMP = os.path.join(OUTDIR, "pre_rl_rollouts.jsonl")

N_PROMPTS = 64
G = 8
SEED = 20260910
MAX_PROMPT_LEN = 2048
MAX_RESPONSE_LEN = 4096
TEMPERATURE = 1.0
TOP_P = 1.0
DATA_SOURCE = "math_dapo"

INVALID = (None, "[INVALID]", "")


def classify(result, exc):
    if exc is not None:
        return "VERIFIER_EXCEPTION"
    pred = result.get("pred") if isinstance(result, dict) else None
    if result and result.get("acc"):
        return "OK_CORRECT"
    if pred in INVALID:
        return "PARSE_FAILURE"
    return "OK_INCORRECT"


def score(sol, gt):
    t0 = time.perf_counter()
    exc, res = None, None
    try:
        res = default_compute_score(data_source=DATA_SOURCE, solution_str=sol, ground_truth=gt)
    except Exception as e:  # noqa: BLE001
        exc = f"{type(e).__name__}: {e}"
    lat = (time.perf_counter() - t0) * 1e3
    st = classify(res, exc)
    return dict(score=(res or {}).get("score") if isinstance(res, dict) else res,
                acc=bool((res or {}).get("acc")) if isinstance(res, dict) else None,
                pred=(res or {}).get("pred") if isinstance(res, dict) else None,
                state=st, exception=exc, latency_ms=lat)


# ---- format-false-negative probes -------------------------------------------------
ANSWER_RE = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def format_forensics(text, gt):
    """Detect: the model DID produce the right integer somewhere, but the official
    verifier could not credit it. Distinguishes the *reason*."""
    gt_norm = gt.strip()
    tail300 = text[-300:]

    # every integer the model emitted anywhere
    all_ints = set(re.findall(r"-?\d[\d,]*", text.replace(",", "")))
    said_gt_anywhere = gt_norm in {s.replace(",", "") for s in all_ints}

    ans_matches = ANSWER_RE.findall(text)
    ans_in_tail = ANSWER_RE.findall(tail300)
    boxed_all = BOXED_RE.findall(text)
    boxed_tail = BOXED_RE.findall(tail300)

    last_ans = ans_matches[-1].strip() if ans_matches else None
    last_boxed = boxed_all[-1].strip() if boxed_all else None

    def eq(x):
        return x is not None and x.replace(",", "").rstrip(".").strip("$ ") == gt_norm

    reason = None
    if not said_gt_anywhere:
        reason = "model_did_not_produce_gt"
    elif (eq(last_ans) or eq(last_boxed)) and not ans_in_tail and not boxed_tail:
        reason = "answer_outside_300char_window"
    elif eq(last_ans) and last_ans != gt_norm:
        reason = f"formatting_mismatch:{last_ans[:20]!r}"
    elif not ans_matches and not boxed_all:
        reason = "no_answer_marker_emitted"
    return dict(said_gt_anywhere=said_gt_anywhere,
                n_answer_markers=len(ans_matches),
                answer_marker_in_tail300=bool(ans_in_tail),
                n_boxed=len(boxed_all),
                boxed_in_tail300=bool(boxed_tail),
                last_answer_text=last_ans, last_boxed_text=last_boxed,
                forensic_reason=reason)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    random.seed(SEED)

    tbl = pq.read_table(PARQUET, columns=["reward_model", "extra_info", "prompt"])
    rm = tbl.column("reward_model").to_pylist()
    ei = tbl.column("extra_info").to_pylist()
    pr = tbl.column("prompt").to_pylist()
    seen = {}
    for r, e, p in zip(rm, ei, pr):
        k = e["index"]
        if k not in seen:
            seen[k] = (r["ground_truth"], p)
    keys = sorted(seen)
    picked = random.sample(keys, N_PROMPTS)

    tok = AutoTokenizer.from_pretrained(MODEL)
    # Match how VeRL builds prompts: tokenizer chat template over the dataset's chat list.
    chat_texts, gts, kept = [], [], []
    for k in picked:
        gt, chat = seen[k]
        chat = [dict(role=m["role"], content=m["content"]) for m in chat]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if len(tok(text)["input_ids"]) > MAX_PROMPT_LEN:
            continue
        chat_texts.append(text)
        gts.append(gt)
        kept.append(k)
    print(f"[rollout] prompts kept {len(kept)}/{N_PROMPTS} after max_prompt_length={MAX_PROMPT_LEN}")
    print(f"[rollout] chat template applied; sample head:\n{chat_texts[0][:300]}\n---")

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 256,
              tensor_parallel_size=1, seed=SEED)
    sp = SamplingParams(n=G, temperature=TEMPERATURE, top_p=TOP_P,
                        max_tokens=MAX_RESPONSE_LEN, seed=SEED)

    t0 = time.time()
    outs = llm.generate(chat_texts, sp)
    gen_s = time.time() - t0

    rows, groups = [], []
    total_out_tokens = 0
    for k, gt, o in zip(kept, gts, outs):
        grp = []
        for j, comp in enumerate(o.outputs):
            text = comp.text
            ntok = len(comp.token_ids)
            total_out_tokens += ntok
            truncated = comp.finish_reason == "length"
            s = score(text, gt)
            fx = format_forensics(text, gt)
            state = "TRUNCATED" if truncated and s["state"] == "PARSE_FAILURE" else s["state"]
            row = dict(prompt_index=k, sample=j, ground_truth=gt,
                       finish_reason=comp.finish_reason, truncated=truncated,
                       n_tokens=ntok, state_final=state,
                       response_head=text[:200], response_tail=text[-300:], **s, **fx)
            rows.append(row)
            grp.append(row)
        groups.append(dict(prompt_index=k, ground_truth=gt, rows=grp))

    with open(DUMP, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- group statistics ----------------
    n_groups = len(groups)
    all_correct = all_wrong = mixed = zero_std = 0
    stds = []
    for g in groups:
        sc = [r["score"] for r in g["rows"]]
        sd = statistics.pstdev(sc)
        stds.append(sd)
        ncor = sum(1 for x in sc if x == 1.0)
        if ncor == len(sc):
            all_correct += 1
        elif ncor == 0:
            all_wrong += 1
        else:
            mixed += 1
        if sd == 0.0:
            zero_std += 1

    n = len(rows)
    states = Counter(r["state_final"] for r in rows)
    trunc = sum(1 for r in rows if r["truncated"])
    lens = [r["n_tokens"] for r in rows]
    lens_sorted = sorted(lens)
    p95 = lens_sorted[int(0.95 * (len(lens_sorted) - 1))]
    lat = [r["latency_ms"] for r in rows]
    lat_s = sorted(lat)

    # format false negatives: model produced gt but scored -1
    fn = [r for r in rows if r["score"] == -1.0 and r["said_gt_anywhere"]]
    fn_window = [r for r in fn if r["forensic_reason"] == "answer_outside_300char_window"]
    fn_fmt = [r for r in fn if str(r["forensic_reason"]).startswith("formatting_mismatch")]
    fn_nomark = [r for r in fn if r["forensic_reason"] == "no_answer_marker_emitted"]

    print("=" * 78)
    print(f"ROLLOUT: {n_groups} prompts x G={G} = {n} samples, temp={TEMPERATURE}, "
          f"max_resp={MAX_RESPONSE_LEN}, seed={SEED}")
    print(f"generation wall time {gen_s:.1f}s  output tokens {total_out_tokens}  "
          f"{total_out_tokens / gen_s:.0f} tok/s")
    print("=" * 78)
    print(f"reward mean {statistics.mean([r['score'] for r in rows]):+.4f}   "
          f"accuracy {sum(1 for r in rows if r['score'] == 1.0) / n:.2%}")
    print()
    print(f"all_correct_group_ratio  {all_correct / n_groups:6.2%}  ({all_correct}/{n_groups})")
    print(f"all_wrong_group_ratio    {all_wrong / n_groups:6.2%}  ({all_wrong}/{n_groups})")
    print(f"mixed_group_ratio        {mixed / n_groups:6.2%}  ({mixed}/{n_groups})")
    print(f"ZERO_STD_GROUP_RATIO     {zero_std / n_groups:6.2%}  ({zero_std}/{n_groups})   <-- GRPO signal killer")
    print(f"mean group std           {statistics.mean(stds):.4f}")
    print()
    print("verifier states:")
    for s, c in states.most_common():
        print(f"   {s:22s} {c:5d}  {c / n:6.2%}")
    print()
    print(f"truncation_rate          {trunc / n:6.2%}  ({trunc}/{n})")
    print(f"EOS/stop rate            {sum(1 for r in rows if r['finish_reason'] == 'stop') / n:6.2%}")
    print(f"response tokens  mean {statistics.mean(lens):.0f}  median {statistics.median(lens):.0f}  "
          f"p95 {p95}  max {max(lens)}")
    print()
    print(f"verifier latency mean {statistics.mean(lat):.4f} ms  median {statistics.median(lat):.4f} ms  "
          f"p95 {lat_s[int(0.95 * (len(lat_s) - 1))]:.4f} ms  max {max(lat):.4f} ms")
    print()
    print("--- FORMAT FALSE NEGATIVES (model emitted the ground-truth integer but scored -1) ---")
    print(f"  total                              {len(fn):5d}  {len(fn) / n:6.2%}")
    print(f"    answer outside 300-char window   {len(fn_window):5d}  {len(fn_window) / n:6.2%}")
    print(f"    formatting mismatch              {len(fn_fmt):5d}  {len(fn_fmt) / n:6.2%}")
    print(f"    no answer marker emitted         {len(fn_nomark):5d}  {len(fn_nomark) / n:6.2%}")
    print("  NOTE: 'emitted gt somewhere' is an UPPER BOUND on true false negatives -- a long")
    print("        chain of thought may mention the right number incidentally without concluding it.")
    print()
    ex = Counter(str(r["forensic_reason"]).split(":")[0] for r in fn)
    for k2, v in ex.most_common():
        print(f"    reason {k2:36s} {v}")
    print(f"\nwrote {DUMP}")

    with open(os.path.join(OUTDIR, "pre_rl_rollout_stats.json"), "w") as f:
        json.dump(dict(
            n_prompts=n_groups, G=G, n_samples=n, temperature=TEMPERATURE,
            max_response_length=MAX_RESPONSE_LEN, seed=SEED,
            gen_wall_s=gen_s, total_out_tokens=total_out_tokens,
            reward_mean=statistics.mean([r["score"] for r in rows]),
            accuracy=sum(1 for r in rows if r["score"] == 1.0) / n,
            all_correct_group_ratio=all_correct / n_groups,
            all_wrong_group_ratio=all_wrong / n_groups,
            mixed_group_ratio=mixed / n_groups,
            zero_std_group_ratio=zero_std / n_groups,
            states={k2: v for k2, v in states.items()},
            truncation_rate=trunc / n,
            resp_tokens=dict(mean=statistics.mean(lens), median=statistics.median(lens),
                             p95=p95, max=max(lens)),
            verifier_latency_ms=dict(mean=statistics.mean(lat), median=statistics.median(lat),
                                     p95=lat_s[int(0.95 * (len(lat_s) - 1))], max=max(lat)),
            format_false_negative_upper_bound=len(fn) / n,
            fn_outside_window=len(fn_window) / n,
            fn_formatting=len(fn_fmt) / n,
            fn_no_marker=len(fn_nomark) / n,
        ), f, indent=2)
    print("PRE_RL_ROLLOUT_DONE")


if __name__ == "__main__":
    main()
