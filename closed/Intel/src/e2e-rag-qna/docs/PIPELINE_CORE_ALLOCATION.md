# Pipeline core allocation & framework choice (this host)

Deliverable for `TASK.md` on branch `e2e-rag/integrate-numa-pipeline`: right-size the
NUMA pipeline for **this** machine, baseline the OpenVINO INT8 path, add a vLLM
embedding-serving path, and decide which framework to ship.

**TL;DR — ship OpenVINO INT8, in-process, one embedder per physical core.** It builds
the full DB in **119.6 s vs 206.8 s** for vLLM (1.73× faster) on identical output.

All numbers below were measured inside container `mk-vllm-embedding` on the full
FRAMES corpus: **2515 documents → 108,790 passages**, 768-char chunks / 32 overlap,
e5-base-v2 (768-dim), FAISS HNSW. Both DBs were validated to `ntotal=108790`.

---

## The host

Xeon 6787P — 2 sockets, **4 NUMA nodes × 43 physical cores = 172 physical cores**
(344 logical with hyperthreading). The committed pipeline defaults were tuned for a
6-NUMA / 256-core Xeon 6980P (`EMBED_PER_NODE=34`, `PARSE_PER_NODE=8`,
`PIPELINE_EMBED_INSTANCES=204`) and had to be re-sized here. Physical cores only; no
HT siblings used for compute.

---

## Chosen configuration

### OpenVINO INT8 (shipped) — `config.sh`
```
INGESTION_EMBED_RUNTIME=openvino
INGESTION_PIPELINE_EMBED_INSTANCES=0   # auto-derive from topology
INGESTION_NUMA_BIND=all                # numactl --interleave=all
EMBED_PER_NODE=37
PARSE_PER_NODE=5
```
Derived layout across 4 nodes: **148 embed + 20 parse + 1 index** (1 physical core
each). One shared INT8 IR is quantized once (~23 s) and mmap'd by all 148 embed
workers; the single global FAISS indexer needs only 1 core (confirmed — it is never
the bottleneck, Stage-3 blocks index in <0.2 s each).

### vLLM (implemented, not shipped) — standalone server + HTTP clients
```
scripts/servers/launch_server_embedding_vllm.sh
  --runner pooling --enforce-eager --dtype bfloat16
  --data-parallel-size 4                       # one replica per NUMA node
  VLLM_CPU_OMP_THREADS_BIND=0-40|43-83|86-126|129-169   # ~41 cores/replica
  --max-model-len 512                          # headroom; see fix #3
INGESTION_EMBED_RUNTIME=vllm                   # embed workers become HTTP clients
```
Here the embed "workers" are thin clients (tokenize + POST on their pinned core); the
vLLM server owns the compute cores via its own OMP binding. Parse (20) + index (1)
run in the pipeline as before.

---

## Results (full corpus, MLPerf LoadGen Offline, VALID)

| Metric | **OpenVINO INT8 (in-process)** | vLLM (HTTP server) |
|---|---|---|
| **Throughput** | **21.02 samples/sec** | 12.16 samples/sec |
| **Wall time (max latency)** | **119.6 s** | 206.8 s |
| Mean latency | 67.2 s | 120.8 s |
| Passages indexed | 108,790 | 108,790 |
| Embed errors | — | 0 (after fixes) |

**OpenVINO INT8 is 1.73× faster.**

---

## Why OpenVINO wins here

1. **Perfect parallelism, zero coordination.** OV runs 148 independent 1-core INT8
   embedders. There is no server, no request queue, no cross-process batching — every
   core stays saturated. vLLM funnels 148 HTTP clients into 4 replicas that batch then
   drain ("Running: 0, Waiting: N" bursts on the CPU backend), leaving cores idle
   between batches.

2. **The pipeline is parse-bound, not embed-bound.** Measured compute split on this
   corpus is ~8:1 embed:parse (8453 s embed vs 1064 s parse summed across workers).
   Because embedding is cheap per-core with INT8, the win came from *rebalancing toward
   parse*: 160E/8P → **148E/20P** cut wall time 164.7 s → 119.8 s (27% faster). A faster
   embed backend can't help a parse-bound pipeline — and vLLM's is not faster here
   anyway.

3. **Inference-level sanity check.** TASK.md's standalone `ov-int8/accurate` table is
   ~16–17 passages/sec/core, near-flat across batch → throughput scales ~linearly with
   cores, which is exactly the "1 embedder per physical core" design. vLLM CPU pooling
   does not beat that per-core on this box, and adds serving overhead on top.

The colleague's "vLLM may be faster" hypothesis (from `taran/vllm-reranker`) did not
hold for **CPU** embedding serving on this hardware. It may still win on a GPU/XPU
backend, where a single served replica can out-throughput many small CPU instances —
untested here.

---

## vLLM path: defects found & fixed to get a valid number

The vLLM path was newly wired end-to-end; three real bugs had to be fixed before its
numbers were trustworthy:

1. **`main_ingestion.py`** — `vllm` was missing from the `--embed_runtime` argparse
   choices, so the CLI rejected it even though the SUT handled it.
2. **`config.sh`** — `config.default.sh` sets `SERVER_EMBED_MODEL` to a bare HF id, and
   the launcher prefers it over `INGESTION_RETRIEVER_MODEL`; `vllm serve` then couldn't
   find the model. Added the host's on-disk path override.
3. **Silent passage loss (correctness).** vLLM rejects any prompt over `--max-model-len`
   with HTTP 400, and the embed worker's exception handler drops the **entire batch**
   (up to 8 passages) — a first run lost ~1920 passages this way. Fixed with client-side
   `truncate_prompt_tokens=256` (matching the OV INT8 calibration length) **plus** raising
   the server's `--max-model-len` to 512 so the tokenizer's CLS/SEP specials on top of
   256 content tokens can't trip the limit. Verified **0 errors** and full 108,790-passage
   parity with the OV DB.

Note: the DP-4 server startup "hang" seen in earlier testing was **container contention**
from a concurrent job, not a fundamental CPU-DP problem — on the idle box DP-4 comes up
cleanly (all 4 replicas "Application startup complete"), it is simply slower end-to-end.

---

## Files touched

- `config.sh` (host-local) — OV core knobs (37E/5P/auto), `SERVER_EMBED_MODEL` path fix.
- `sut/SUT_ingestion_pipelined.py` — `load_vllm_http_embedder()` + `embed_runtime=="vllm"`
  branch; client-side `truncate_prompt_tokens`.
- `scripts/servers/launch_server_embedding_vllm.sh` (new) — vLLM `/v1/embeddings` server.
- `main_ingestion.py` — add `vllm` to `--embed_runtime` choices.
