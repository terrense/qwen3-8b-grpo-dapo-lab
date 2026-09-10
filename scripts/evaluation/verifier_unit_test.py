"""Verifier unit test for the OFFICIAL VeRL math_dapo reward path.

Calls verl.utils.reward_score.default_compute_score with the real dataset
data_source, i.e. exactly the function the trainer will call. Baseline reward
semantics are NOT modified -- the extra state classification here is pure
instrumentation layered on top of the official return value, so that
"verifier failure" is never silently read as "model was wrong".

States:
  OK_CORRECT         official acc truthy
  OK_INCORRECT       official acc falsy AND an answer was successfully extracted
  PARSE_FAILURE      no answer could be extracted (pred is None / "[INVALID]")
  VERIFIER_EXCEPTION compute_score raised
  TRUNCATED          rollout-level flag (set by the caller, not inferable here)
"""

import json
import os
import random
import re
import statistics
import sys
import time
import traceback

import pyarrow.parquet as pq

from verl.utils.reward_score import default_compute_score

PARQUET = "/root/autodl-tmp/rl_lab/data/raw/DAPO-Math-17k/data/dapo-math-17k.parquet"
OUT_JSONL = "/root/autodl-tmp/rl_lab/analysis/verifier_cases.jsonl"
DATA_SOURCE = "math_dapo"
SEED = 20260910

INVALID_MARKERS = (None, "[INVALID]", "")


def classify(result, exc):
    """Layer diagnostic states on top of the official result. Does not alter reward."""
    if exc is not None:
        return "VERIFIER_EXCEPTION"
    pred = result.get("pred") if isinstance(result, dict) else None
    acc = result.get("acc") if isinstance(result, dict) else None
    if acc:
        return "OK_CORRECT"
    if pred in INVALID_MARKERS:
        return "PARSE_FAILURE"
    return "OK_INCORRECT"


def score_one(solution_str, ground_truth):
    t0 = time.perf_counter()
    exc = None
    result = None
    try:
        result = default_compute_score(
            data_source=DATA_SOURCE,
            solution_str=solution_str,
            ground_truth=ground_truth,
        )
    except Exception as e:  # noqa: BLE001 -- deliberately catching everything for diagnosis
        exc = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)
    latency_ms = (time.perf_counter() - t0) * 1e3
    state = classify(result, exc)
    if isinstance(result, dict):
        score, acc, pred = result.get("score"), result.get("acc"), result.get("pred")
    else:
        score, acc, pred = result, None, None
    return dict(score=score, acc=bool(acc) if acc is not None else None,
                pred=pred, state=state, exception=exc, latency_ms=latency_ms)


# --------------------------------------------------------------------------
# Response builders. The DAPO prompt instructs:
#   "The last line of your response should be of the form Answer: $Answer"
# so the canonical shape is a final `Answer: <int>` line.
# Every ground truth in DAPO-Math-17k is a plain integer (verified: 17917/17917),
# so the meaningful equivalence classes are integer FORMATTING, not symbolic
# algebra. We do not invent \frac/\sqrt cases the dataset never contains.
# --------------------------------------------------------------------------
SHORT_COT = (
    "Let me work through this step by step.\n"
    "First I set up the relevant equation and simplify.\n"
    "Then I solve for the requested quantity.\n"
)

LONG_COT = (
    "Let me think about this carefully.\n\n"
    + ("I will consider the constraints one at a time and verify each algebraic step "
       "before moving on, since a sign error early on would propagate.\n") * 40
)

# Trailing text long enough to push the answer line outside compute_score's
# solution_str[-300:] window.
TRAILING_CHATTER = (
    "\n\nI hope this helps. Let me restate the reasoning once more for clarity, "
    "because it is worth double checking the arithmetic in the final step and "
    "confirming that no constraint was overlooked anywhere in the derivation "
    "above, and that the final value is consistent with the problem statement.\n"
) * 3


def variants(gt):
    """Return list of (case_id, category, expectation, response_text)."""
    try:
        gt_int = int(gt)
    except ValueError:
        gt_int = None
    wrong = str(gt_int + 1) if gt_int is not None else gt + "0"

    v = [
        ("A_canonical_correct", "A", "correct",
         f"{SHORT_COT}\nAnswer: {gt}"),

        ("B1_boxed_correct", "B", "correct",
         f"{SHORT_COT}\nThe final answer is $\\boxed{{{gt}}}$."),
        ("B2_dollar_wrapped", "B", "correct",
         f"{SHORT_COT}\nAnswer: ${gt}$"),
        ("B3_trailing_period", "B", "correct",
         f"{SHORT_COT}\nAnswer: {gt}."),
        ("B4_bold_latex", "B", "correct",
         f"{SHORT_COT}\nAnswer: \\textbf{{{gt}}}"),
        ("B5_spaces", "B", "correct",
         f"{SHORT_COT}\nAnswer:   {gt}   "),
        ("B6_decimal_form", "B", "correct?",
         f"{SHORT_COT}\nAnswer: {gt}.0"),
        ("B7_plus_sign", "B", "correct?",
         f"{SHORT_COT}\nAnswer: +{gt}"),
        ("B8_comma_grouped", "B", "correct?",
         f"{SHORT_COT}\nAnswer: {int(gt):,}" if gt_int is not None and abs(gt_int) >= 1000
         else f"{SHORT_COT}\nAnswer: {gt}"),
        ("B9_text_wrapped", "B", "correct",
         f"{SHORT_COT}\nAnswer: \\text{{{gt}}}"),

        ("C1_clearly_wrong", "C", "incorrect",
         f"{SHORT_COT}\nAnswer: {wrong}"),
        ("C2_wrong_boxed", "C", "incorrect",
         f"{SHORT_COT}\nThe final answer is $\\boxed{{{wrong}}}$."),

        ("D1_no_final_answer", "D", "parse_failure",
         f"{SHORT_COT}\nSo the quantity is determined by the constraints above."),
        ("D2_empty", "D", "parse_failure", ""),
        ("D3_whitespace_only", "D", "parse_failure", "   \n  \n "),
        ("D4_unbalanced_boxed", "D", "parse_failure",
         f"{SHORT_COT}\nThe final answer is $\\boxed{{{gt}"),
        ("D5_answer_label_no_value", "D", "parse_failure",
         f"{SHORT_COT}\nAnswer:"),

        ("E_long_cot_correct", "E", "correct",
         f"{LONG_COT}\nAnswer: {gt}"),
        ("F_long_cot_wrong", "F", "incorrect",
         f"{LONG_COT}\nAnswer: {wrong}"),

        # Truncation-window stress: correct answer present, but pushed outside
        # the last 300 characters that compute_score actually inspects.
        ("G_correct_but_outside_300char_window", "G", "correct_but_at_risk",
         f"{SHORT_COT}\nAnswer: {gt}{TRAILING_CHATTER}"),
        # Same, with a boxed answer instead.
        ("H_boxed_outside_window", "G", "correct_but_at_risk",
         f"{SHORT_COT}\n$\\boxed{{{gt}}}${TRAILING_CHATTER}"),
        # Qwen3 thinking-style block, answer at the very end (the healthy shape).
        ("I_think_block_then_answer", "E", "correct",
         f"<think>\n{LONG_COT}\n</think>\n\nAnswer: {gt}"),
    ]
    return v


def main():
    random.seed(SEED)
    tbl = pq.read_table(PARQUET, columns=["reward_model", "extra_info", "prompt"])
    rm = tbl.column("reward_model").to_pylist()
    ei = tbl.column("extra_info").to_pylist()
    pr = tbl.column("prompt").to_pylist()

    # dedupe to unique prompts (dataset repeats each exactly 100x)
    seen = {}
    for r, e, p in zip(rm, ei, pr):
        k = e["index"]
        if k not in seen:
            seen[k] = (r["ground_truth"], p[0]["content"] if p else "")
    keys = sorted(seen)
    picked = random.sample(keys, 20)

    rows = []
    for k in picked:
        gt, prompt = seen[k]
        for case_id, cat, expectation, resp in variants(gt):
            r = score_one(resp, gt)
            rows.append(dict(
                prompt_index=k, ground_truth=gt, case_id=case_id, category=cat,
                expectation=expectation, response_len_chars=len(resp),
                response_tail=resp[-120:], **r,
            ))

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- aggregate ----------------
    lat = [r["latency_ms"] for r in rows]
    lat_sorted = sorted(lat)
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]

    def rate(pred_fn, over_fn):
        num = sum(1 for r in rows if over_fn(r) and pred_fn(r))
        den = sum(1 for r in rows if over_fn(r))
        return (num / den if den else float("nan")), num, den

    is_correct_case = lambda r: r["expectation"] in ("correct",)
    is_wrong_case = lambda r: r["expectation"] == "incorrect"
    is_parse_case = lambda r: r["expectation"] == "parse_failure"
    is_risk_case = lambda r: r["expectation"] == "correct_but_at_risk"

    acc_rate, an, ad = rate(lambda r: r["state"] == "OK_CORRECT", is_correct_case)
    rej_rate, rn, rd = rate(lambda r: r["state"] == "OK_INCORRECT", is_wrong_case)
    pf_detect, pn, pd = rate(lambda r: r["state"] == "PARSE_FAILURE", is_parse_case)
    risk_ok, kn, kd = rate(lambda r: r["state"] == "OK_CORRECT", is_risk_case)

    n = len(rows)
    parse_failure_rate = sum(1 for r in rows if r["state"] == "PARSE_FAILURE") / n
    exception_rate = sum(1 for r in rows if r["state"] == "VERIFIER_EXCEPTION") / n

    print("=" * 78)
    print(f"cases: {n}  ({len(picked)} problems x {n // len(picked)} variants)")
    print("=" * 78)
    print(f"CORRECT_ACCEPT_RATE      {acc_rate:6.2%}   ({an}/{ad} canonical+equivalent-correct accepted)")
    print(f"WRONG_REJECT_RATE        {rej_rate:6.2%}   ({rn}/{rd} clearly-wrong rejected)")
    print(f"PARSE_FAILURE_DETECTED   {pf_detect:6.2%}   ({pn}/{pd} malformed cases flagged PARSE_FAILURE)")
    print(f"TRUNC_WINDOW_SURVIVED    {risk_ok:6.2%}   ({kn}/{kd} answers outside the 300-char window still scored correct)")
    print(f"PARSE_FAILURE_RATE(all)  {parse_failure_rate:6.2%}")
    print(f"VERIFIER_EXCEPTION_RATE  {exception_rate:6.2%}")
    print(f"LATENCY mean {statistics.mean(lat):.3f} ms  median {statistics.median(lat):.3f} ms  "
          f"p95 {p95:.3f} ms  max {max(lat):.3f} ms")

    print("\n--- state x expectation matrix ---")
    from collections import Counter
    m = Counter((r["expectation"], r["state"]) for r in rows)
    for (e, s), c in sorted(m.items()):
        print(f"  {e:22s} -> {s:20s} {c:4d}")

    print("\n--- per-case-id outcome (score, state) ---")
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)
    for cid in sorted(by_case):
        rs = by_case[cid]
        sc = Counter(r["score"] for r in rs)
        st = Counter(r["state"] for r in rs)
        print(f"  {cid:42s} score={dict(sc)}  state={dict(st)}")

    print(f"\nwrote {OUT_JSONL}")


if __name__ == "__main__":
    main()
