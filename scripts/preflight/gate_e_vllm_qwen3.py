import os, time, torch
from vllm import LLM, SamplingParams

MODEL = "/root/autodl-tmp/rl_lab/models/Qwen3-8B"
t0 = time.time()
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.60,
          max_model_len=2048, tensor_parallel_size=1, seed=0)
load_s = time.time() - t0
print(f"[gate5] model load: {load_s:.1f}s")

prompts = [
    "What is 17 * 23? Answer with just the number.",
    "Solve: if 3x + 7 = 22, what is x?",
    "Name the capital of France in one word.",
    "What is the derivative of x^2?",
]
sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=64, seed=0)
t1 = time.time()
outs = llm.generate(prompts, sp)
gen_s = time.time() - t1

ntok = 0
for i, o in enumerate(outs):
    txt = o.outputs[0].text
    ntok += len(o.outputs[0].token_ids)
    print(f"[gate5] --- prompt {i}: {prompts[i][:45]}")
    print(f"[gate5]     finish={o.outputs[0].finish_reason} ntok={len(o.outputs[0].token_ids)}")
    print(f"[gate5]     text={txt[:160]!r}")
print(f"[gate5] generate: {gen_s:.2f}s  total_out_tokens={ntok}")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"[gate5] gpu{i} used={(total-free)/1024**3:.2f} GiB / {total/1024**3:.2f} GiB")
print("[gate5] VLLM_QWEN3_8B_INFERENCE PASS")
