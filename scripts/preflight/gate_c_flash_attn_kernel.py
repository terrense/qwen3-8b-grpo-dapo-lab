import torch, flash_attn
from flash_attn import flash_attn_func
print("flash_attn", flash_attn.__version__)
dev = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0))
ok = True
for dtype in (torch.bfloat16, torch.float16):
    for seqlen in (128, 2048, 8192):
        B, H, D = 2, 8, 128
        q = torch.randn(B, seqlen, H, D, device=dev, dtype=dtype)
        k = torch.randn(B, seqlen, H, D, device=dev, dtype=dtype)
        v = torch.randn(B, seqlen, H, D, device=dev, dtype=dtype)
        try:
            o = flash_attn_func(q, k, v, causal=True)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(o).all())
            print(f"  {str(dtype):18s} seqlen={seqlen:5d} out={tuple(o.shape)} finite={finite} PASS")
            ok &= finite
        except Exception as e:
            print(f"  {str(dtype):18s} seqlen={seqlen:5d} FAIL {e!r}")
            ok = False

# correctness cross-check vs SDPA reference at one config
B,H,S,D = 1,4,512,128
q = torch.randn(B,S,H,D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B,S,H,D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B,S,H,D, device=dev, dtype=torch.bfloat16)
fa = flash_attn_func(q,k,v,causal=True)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1,2).float(), k.transpose(1,2).float(), v.transpose(1,2).float(), is_causal=True
).transpose(1,2).to(torch.bfloat16)
md = (fa-ref).abs().max().item()
print(f"  max|flash-attn - sdpa| = {md:.4f}  (bf16 tolerance ~<0.1)")
ok &= md < 0.1
print("FLASH_ATTENTION_KERNEL", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
