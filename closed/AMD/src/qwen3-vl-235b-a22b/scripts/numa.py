"""NUMA-aware CPU pinning for the multi-replica (8xTP1) serving topology."""
from __future__ import annotations

import glob
import re
import shutil
import subprocess

_RESERVE_PER_NODE = 8


def _node_ranges() -> dict[int, list[tuple[int, int]]]:
    """{numa_node: [(lo,hi), ...]} from /sys cpulist files (e.g. '0-63,128-191')."""
    nodes: dict[int, list[tuple[int, int]]] = {}
    for p in glob.glob("/sys/devices/system/node/node*/cpulist"):
        m = re.search(r"node(\d+)", p)
        if not m:
            continue
        try:
            ranges = []
            for part in open(p).read().strip().split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    a, b = part.split("-")
                    ranges.append((int(a), int(b)))
                else:
                    ranges.append((int(part), int(part)))
            if ranges:
                nodes[int(m.group(1))] = ranges
        except Exception:
            pass
    return nodes


def _gpu_to_numa(num_gpus: int, node_ids: list[int]) -> dict[int, int]:
    """GPU index -> NUMA node. Prefer rocm-smi --showtoponuma; else even split."""
    m: dict[int, int] = {}
    try:
        out = subprocess.run(
            ["rocm-smi", "--showtoponuma"], capture_output=True, text=True, timeout=30
        ).stdout
        for line in out.splitlines():
            g = re.search(r"GPU\[(\d+)\].*Numa Node:\s*(\d+)", line)
            if g:
                m[int(g.group(1))] = int(g.group(2))
    except Exception:
        pass
    if all(i in m for i in range(num_gpus)):
        return {i: m[i] for i in range(num_gpus)}
    # Fallback: split GPUs evenly across nodes in order (typical 4-GPU-per-socket).
    nl = sorted(node_ids)
    per = max(1, -(-num_gpus // len(nl)))  # ceil
    return {i: nl[min(i // per, len(nl) - 1)] for i in range(num_gpus)}


def _fmt(cpus: list[int]) -> str:
    """Compress a sorted int list to a taskset/numactl cpu-list string."""
    cpus = sorted(set(cpus))
    if not cpus:
        return ""
    out, lo, prev = [], cpus[0], cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        out.append(f"{lo}-{prev}" if lo != prev else f"{lo}")
        lo = prev = c
    out.append(f"{lo}-{prev}" if lo != prev else f"{lo}")
    return ",".join(out)


def compute_plan(num_workers: int, enabled: bool = True, gpu_ids: list[int] | None = None) -> dict | None:
    """Partition cores: each worker -> its GPU-local node cores (minus a reserved
    tail); the load generator -> the reserved tail across all nodes.

    ``gpu_ids`` (from the server.gpu_ids knob) is the physical GPU id each worker runs on
    (worker i -> gpu_ids[i]); defaults to worker i -> GPU i, so the pin follows the actual GPU.

    Returns dict {worker_cores:{i:str}, worker_node:{i:int}, loadgen_cores:str}, or
    None when binding is disabled / there is a single worker / there are <2 NUMA nodes /
    topology is unreadable (in all of which cases binding is a no-op and the prefix helpers
    return []).
    """
    if not enabled:
        return None
    # Only use when nreplicas>1 else no numa binding is needed
    if num_workers < 2:
        return None
    nodes = _node_ranges()
    if len(nodes) < 2:
        return None
    if gpu_ids is None:
        gpu_ids = list(range(num_workers))
    phys_to_node = _gpu_to_numa(max(gpu_ids) + 1, list(nodes))
    g2n = {i: phys_to_node[gpu_ids[i]] for i in range(num_workers)}
    node_worker_cpus: dict[int, list[int]] = {}
    reserved: list[int] = []
    for nid, ranges in nodes.items():
        wk: list[int] = []
        for (a, b) in ranges:
            r = list(range(a, b + 1))
            if 0 < _RESERVE_PER_NODE < len(r):
                reserved += r[-_RESERVE_PER_NODE:]
                wk += r[:-_RESERVE_PER_NODE]
            else:
                wk += r
        node_worker_cpus[nid] = wk
    if not reserved:  # nothing reserved -> nothing to isolate
        return None
    worker_cores = {i: _fmt(node_worker_cpus[g2n[i]]) for i in range(num_workers)}
    return {
        "worker_cores": worker_cores,
        "worker_node": {i: g2n[i] for i in range(num_workers)},
        "loadgen_cores": _fmt(reserved),
    }


def visible_gpu(env: dict) -> int:
    """The GPU index a single worker will use, from HIP_VISIBLE_DEVICES (default 0)."""
    for k in ("HIP_VISIBLE_DEVICES",):
        v = env.get(k)
        if v and v.split(",")[0].strip().isdigit():
            return int(v.split(",")[0])
    return 0


def _has_numactl() -> bool:
    return shutil.which("numactl") is not None


# TODO: the *worker* half could be delegated to vLLM's native NUMA binding
# (`vllm/utils/numa_utils.py`): launch each worker with `--numa-bind --numa-bind-cpus
# <this worker's subset>`, which also binds memory (`--membind`), is DP-shard aware, and
# is upstream-maintained. Requires the `numactl` binary (vLLM re-execs workers under it;
# not installed in our image) -- our taskset path is CPU-affinity only. If adopted, numa.py
# shrinks to: compute the partition (worker subsets + reserved loadgen set) and pin the
# CLIENT to the reserved cores -- vLLM has no concept of the external load generator, so
# that reservation/client-pin always stays here.
def worker_prefix(plan: dict | None, worker_index: int) -> list[str]:
    """Command prefix to pin a worker to its GPU-local node cores (+ local memory)."""
    if not plan:
        return []
    cores = plan["worker_cores"].get(worker_index, "")
    if not cores:
        return []
    if _has_numactl():
        node = plan["worker_node"][worker_index]
        return ["numactl", f"--physcpubind={cores}", f"--membind={node}"]
    return ["taskset", "-c", cores]


def loadgen_prefix(plan: dict | None) -> list[str]:
    """Command prefix to pin the load generator (client / HAProxy) to the reserved cores."""
    if not plan:
        return []
    cores = plan.get("loadgen_cores", "")
    if not cores:
        return []
    if _has_numactl():
        return ["numactl", f"--physcpubind={cores}", "--interleave=all"]
    return ["taskset", "-c", cores]
