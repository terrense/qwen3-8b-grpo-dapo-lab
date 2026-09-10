import sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("NCCL:", ".".join(map(str, torch.cuda.nccl.version())))
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
for i in range(4):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i} {p.name} {p.total_memory/1024**3:.2f} GiB sm{p.major}{p.minor}")
x = torch.randn(4096,4096,device="cuda:0",dtype=torch.bfloat16)
y = torch.randn(4096,4096,device="cuda:0",dtype=torch.bfloat16)
z = x @ y
torch.cuda.synchronize()
assert torch.isfinite(z).all()
print("CU130 BF16 MATMUL PASS", tuple(z.shape))
