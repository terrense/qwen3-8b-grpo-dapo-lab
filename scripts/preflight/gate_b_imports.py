mods = ["torch","transformers","vllm","ray","flash_attn","verl"]
fail = False
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:14s} PASS  {getattr(mod,'__version__','version unavailable')}")
    except Exception as e:
        print(f"{m:14s} FAIL  {e!r}")
        fail = True
raise SystemExit(1 if fail else 0)
