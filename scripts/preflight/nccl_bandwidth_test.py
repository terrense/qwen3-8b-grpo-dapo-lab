"""NCCL all_reduce bandwidth sanity check for the 4x H20 NVLink node.

Run:  torchrun --standalone --nproc_per_node=4 nccl_bandwidth_test.py
Goal is NOT a publication benchmark -- only to confirm NVLink/NCCL is not
silently degraded to PCIe-class (a few GB/s) speeds before we burn GPU hours.
"""
import os
import time
import json
import torch
import torch.distributed as dist

SIZES_MB = [64, 256, 1024]
WARMUP = 8
ITERS = 25


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    # pass device_id so NCCL binds the comm eagerly -> no "No device id provided" warning
    dist.init_process_group(backend="nccl", device_id=dev)
    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        print(f"world_size={world}  torch={torch.__version__}  "
              f"nccl={'.'.join(map(str, torch.cuda.nccl.version()))}", flush=True)
        print(f"{'size':>10} {'iters':>6} {'avg_ms':>10} {'algBW_GB/s':>12} "
              f"{'busBW_GB/s':>12} {'max_rank_ms':>12} {'spread_%':>9}", flush=True)

    results = []
    for mb in SIZES_MB:
        numel = mb * 1024 * 1024 // 2          # bfloat16 = 2 bytes
        nbytes = numel * 2
        buf = torch.ones(numel, dtype=torch.bfloat16, device=dev)

        for _ in range(WARMUP):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        dist.barrier()

        local_ms = (t1 - t0) / ITERS * 1e3
        # gather each rank's timing to confirm no straggler GPU
        times = torch.tensor([local_ms], dtype=torch.float64, device=dev)
        allt = [torch.zeros_like(times) for _ in range(world)]
        dist.all_gather(allt, times)
        per_rank = [float(x.item()) for x in allt]

        if rank == 0:
            avg_ms = sum(per_rank) / world
            mx, mn = max(per_rank), min(per_rank)
            algbw = nbytes / (avg_ms / 1e3) / 1e9
            busbw = algbw * 2 * (world - 1) / world     # ring all_reduce correction
            spread = (mx - mn) / mn * 100.0
            print(f"{str(mb)+'MB':>10} {ITERS:>6} {avg_ms:>10.3f} {algbw:>12.2f} "
                  f"{busbw:>12.2f} {mx:>12.3f} {spread:>9.2f}", flush=True)
            results.append(dict(size_mb=mb, bytes=nbytes, iters=ITERS, avg_ms=avg_ms,
                                algbw_gbs=algbw, busbw_gbs=busbw,
                                per_rank_ms=per_rank, spread_pct=spread))
        del buf
        torch.cuda.empty_cache()

    if rank == 0:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "nccl_bandwidth.json"), "w") as f:
            json.dump(results, f, indent=2)
        print("\nReference: H20 NVLink busBW is expected in the ~200-400 GB/s band at 1GB.")
        print("Anything in the single-digit GB/s range means NVLink is NOT being used.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
