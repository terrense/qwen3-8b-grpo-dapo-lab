# Hardware & Environment

Raw, unedited preflight output: [`../manifests/hardware.txt`](../manifests/hardware.txt)
(hostname, container id and GPU UUIDs redacted; nothing else altered).

Preflight date: **2026-09-10**. Single AutoDL node, containerised (LXC over KVM), Ubuntu 22.04.5,
kernel 5.15.0-130-generic.

## GPU

| Property | Value |
|---|---|
| Model | NVIDIA H20 |
| Count | 4 |
| VRAM per GPU | 97871 MiB (~95.6 GiB), torch reports 95.07 GiB usable |
| Total VRAM | ~384 GB |
| Compute capability | 9.0 (Hopper, `sm90`) |
| Power limit | 500 W per GPU |
| Idle temperature | 31-34 C |
| Occupancy at preflight | all 4 idle, 0 MiB used, no processes |

## Interconnect

`nvidia-smi topo -m` reports a **full NV18 mesh** — every GPU pair is connected by 18 bonded
NVLink links, no PCIe/SYS hop anywhere:

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity   NUMA
GPU0     X      NV18    NV18    NV18     0-63           0
GPU1    NV18     X      NV18    NV18     0-63           0
GPU2    NV18    NV18     X      NV18    64-127          1
GPU3    NV18    NV18    NV18     X      64-127          1
```

`nvidia-smi nvlink -s`: 18 active links per GPU, 26.562 GB/s each.

**NUMA note that matters for rollout throughput:** GPU0/GPU1 are on NUMA node 0, GPU2/GPU3 on
NUMA node 1. Host-side dataloader / reward-verifier threads that land on the wrong node will pay
cross-socket memory latency. `numactl` is not installed in this container.

### Measured NCCL all-reduce bandwidth

`torchrun --standalone --nproc_per_node=4 scripts/preflight/nccl_bandwidth_test.py`,
bfloat16 payloads, 8 warmup + 25 timed iterations, ring-corrected
busBW = algBW x 2(N-1)/N:

| payload | avg latency | algBW | busBW | rank spread |
|---|---|---|---|---|
| 64 MB | 0.325 ms | 206.3 GB/s | 309.4 GB/s | 0.25 % |
| 256 MB | 1.214 ms | 221.2 GB/s | 331.8 GB/s | 0.07 % |
| 1024 MB | 4.587 ms | 234.1 GB/s | 351.1 GB/s | 0.01 % |

**Verdict: PASS.** 351 GB/s busBW is roughly 78 % of the H20 NVLink unidirectional ceiling, and
the per-rank spread of 0.01 % at 1 GB means no straggler GPU and no silent PCIe fallback. A
degraded node would have shown single-digit GB/s here.

## CPU / memory

| Property | Value |
|---|---|
| CPU | AMD EPYC 9K84 96-Core (virtualised, KVM) |
| Logical CPUs | 128 (2 sockets x 32 cores x 2 threads) |
| NUMA nodes | 2 — node0 = CPU 0-63, node1 = CPU 64-127 |
| L3 cache | 256 MiB (8 instances) |
| RAM visible | ~1.2 TiB total, ~1.2 TiB available |
| Swap | none |

`free -h` shows only ~229 GiB "free" but ~962 GiB in reclaimable page cache — **available** is the
number that matters and it is ~1.2 TiB. This is not a memory-pressure signal.

## Shared memory

`/dev/shm` = **300 GB**. Comfortably large, so no Ray object-store or vLLM workaround is needed
(`--shm-size` starvation is a classic cause of Ray plasma failures on rented nodes; not an issue
here).

## Storage

| Mount | FS | Size | Used | Purpose |
|---|---|---|---|---|
| `/` | overlay | 30 GB | ~0.1 GB | **system disk — must stay empty** |
| `/root/autodl-tmp` | xfs on `/dev/vdb`, rw, prjquota | 1.3 TB | ~1.6 GB | all lab data |
| `/autodl-pub` | xfs, **read-only** | 8.8 TB | 82 % | shared public mount, not ours |
| `/dev/shm` | tmpfs | 300 GB | 0 | |

Sequential write to the data disk, 2 GiB with `oflag=direct`: **226 MB/s**. Modest — a full
8B-parameter bf16 checkpoint plus optimizer state is on the order of 100 GB, so a naive full
checkpoint costs minutes, not seconds. This is why the checkpoint policy keeps at most 2 resumable
checkpoints.

`ulimit -n` = 1048576, `max locked memory` unlimited, `stack` 8 MB. No blocker for Ray/NCCL.

`/autodl-pub` is mounted read-only, so there is no risk of accidentally treating it as the private
experiment disk.

## Software baseline (ambient, untouched)

The node ships a conda base env that we deliberately **do not modify**:

| | |
|---|---|
| Python | 3.12.3 (Anaconda) |
| PyTorch | 2.8.0+cu128 |
| `torch.version.cuda` | 12.8 |
| NCCL | 2.27.3 |
| Driver | 580.105.08 (advertises CUDA 13.0) |
| `torch.cuda.device_count()` | 4 |
| BF16 4096x4096 matmul | PASS |
| 4-rank NCCL all-reduce correctness | PASS (result 10.0 == expected 10.0) |

The `No device id is provided via init_process_group` message seen in the first preflight is a
warning from a test script that did not pass `device_id`, **not** an NCCL failure. The bandwidth
script in this repo passes `device_id=` explicitly and the warning is gone.

VeRL runs in a **separate uv-managed virtualenv** (CUDA 13.0 / torch 2.11.0 world) so the ambient
torch 2.8.0+cu128 install is never disturbed. See [`../manifests/versions.txt`](../manifests/versions.txt).

## Network topology (affects every download in this repo)

This node cannot reach `github.com` or `huggingface.co` directly. It has an academic HTTP proxy,
but the proxy itself blocks `download.pytorch.org`. The resulting split routing is documented in
[`debugging_playbook.md`](debugging_playbook.md) and
[`../analysis/incident_log.md`](../analysis/incident_log.md) (INC-001, INC-002) because it caused
two real install failures.

| Host | direct | via proxy |
|---|---|---|
| `github.com` | blocked | OK |
| `huggingface.co` | blocked | OK |
| `hf-mirror.com` | OK | — |
| `download.pytorch.org` | **OK** | **blocked** |
| `pypi.org` | OK | OK |
