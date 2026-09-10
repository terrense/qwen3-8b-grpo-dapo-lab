#!/usr/bin/env bash
# uv sync with bounded retries.
# Rationale: this node reaches github.com only through a flaky academic HTTP proxy,
# while download.pytorch.org is reachable ONLY when bypassing that proxy. uv's parallel
# fetches trip the proxy into "Connection reset by peer" at random. Every retry resumes
# from UV_CACHE_DIR, so attempts accumulate progress instead of restarting the download.
set -uo pipefail
source /root/autodl-tmp/rl_lab/env.sh
lab_proxy_on >/dev/null
cd "$LAB/repos/verl"

MAX=${MAX:-15}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-180}
export UV_LINK_MODE=copy

for i in $(seq 1 "$MAX"); do
    echo "================ uv sync attempt $i/$MAX  $(date -Is) ================"
    if uv sync --frozen --all-packages --extra vllm --extra fsdp; then
        echo "================ UV_SYNC_SUCCESS on attempt $i  $(date -Is) ================"
        exit 0
    fi
    echo "---- attempt $i failed, retrying in 10s ----"
    sleep 10
done
echo "================ UV_SYNC_FAILED after $MAX attempts ================"
exit 1
