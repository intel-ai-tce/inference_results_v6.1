#!/bin/bash
#SBATCH --job-name=trt-l31-fp8
#SBATCH --partition=kempner_h200_priority
#SBATCH --qos=kemp_mlcommons_gpu16
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/n/holylfs06/LABS/kempner_dev/Lab/nkhoshnevis/ai_projects/mlcommon/inference/attempted/trt/pipe_%j.out

# TensorRT-LLM 1.0.0 FP8 llama3.1-8b Offline: quantize -> build -> serve -> benchmark.
# MPI note: container OpenMPI needs a PMIx server; --cleanenv strips host SLURM env so
# MPI_Init does a clean singleton (proven). All heavy artifacts on scratch.
set -uo pipefail
echo "=== START $(date) on $(hostname) ==="
nvidia-smi -L

BASE=/n/holylfs06/LABS/kempner_dev/Lab/nkhoshnevis/ai_projects/mlcommon/inference
export SCRATCH=/n/netscratch/kempner_dev/nkhoshnevis/mlcommon-scratch
SIF="$SCRATCH/containers/trtllm-release-1.0.0.sif"
MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models
export HF_HOME="$SCRATCH/hf"
WORK="$SCRATCH/trtllm-work"; mkdir -p "$WORK"
CKPT="$WORK/l31-8b-fp8-ckpt"
ENGINE="$WORK/l31-8b-fp8-engine"
# Port must stay BELOW /proc/sys/net/ipv4/ip_local_port_range (32768 here): a port inside the
# ephemeral range can be transiently held by any process's outbound connection. The old formula
# A listener placed inside that window can lose its port to another process's outbound
# connection, surfacing as "[Errno 98] address already in use" at server start.
PORT=$(( 20000 + SLURM_JOB_ID % 12000 ))
for _ in $(seq 1 200); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; PORT=$((PORT+1)); done
echo "[port] using $PORT (below ephemeral range start 32768)"
REPORT="$BASE/attempted/trt/report_${SLURM_JOB_ID}"
SERVE_LOG="$BASE/attempted/trt/serve_${SLURM_JOB_ID}.log"
BENCH_YAML="$WORK/l31_8b_offline_trt_${SLURM_JOB_ID}.yaml"
SRC_YAML="$BASE/endpoints/examples/05_Llama3.1-8B_Example/submission_llama3_8b_offline.yaml"
ROUGE1_GATE=38.391408

# Container exec wrapper: --cleanenv (fix MPI singleton) + re-inject needed env + binds.
# Pass CUDA_VISIBLE_DEVICES through (cleanenv strips it; cgroup already isolates the 1 GPU).
CT() { singularity exec --cleanenv --nv \
  --env PYTHONNOUSERSITE=1 --env HF_HOME=/hf --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --bind "$MODELS":/models:ro --bind "$WORK":/work --bind "$HF_HOME":/hf \
  "$SIF" "$@"; }

# MLPerf-official calibration set (998 unique ids from calibration-list.txt, built offline-verified).
# Dir name has NO 'cnn_dailymail' -> modelopt get_calib_dataloader takes the os.path.isdir/'text' branch.
CALIB_DIR_HOST="$WORK/mlperf_l31_calib"
if [ ! -s "$CALIB_DIR_HOST/train.jsonl" ]; then
  echo "CALIB_MISSING: $CALIB_DIR_HOST/train.jsonl not found — run build_official_calib.py first"; exit 5
fi

########################## STAGE 1: FP8 quantize (ModelOpt PTQ) ##########################
if [ -f "$CKPT/config.json" ] || ls "$CKPT"/*.safetensors >/dev/null 2>&1; then
  echo "[stage1] checkpoint exists, skipping quantize"
else
  echo "[stage1] $(date) FP8 PTQ quantize (calib=MLPerf-official /work/mlperf_l31_calib size=512)"
  CT python /app/tensorrt_llm/examples/quantization/quantize.py \
      --model_dir /models/Llama-3.1-8B-Instruct \
      --dtype bfloat16 --qformat fp8 --kv_cache_dtype fp8 \
      --calib_dataset /work/mlperf_l31_calib --calib_size 512 \
      --output_dir /work/l31-8b-fp8-ckpt
  rc=$?; if [ $rc -ne 0 ]; then echo "QUANTIZE_FAILED rc=$rc"; exit 10; fi
fi

########################## STAGE 2: build engine ##########################
if ls "$ENGINE"/*.engine >/dev/null 2>&1; then
  echo "[stage2] engine exists, skipping build"
else
  echo "[stage2] $(date) trtllm-build FP8 engine"
  CT trtllm-build \
      --checkpoint_dir /work/l31-8b-fp8-ckpt \
      --output_dir /work/l31-8b-fp8-engine \
      --gemm_plugin auto \
      --max_seq_len 4096 \
      --max_batch_size 512 \
      --max_num_tokens 8192 \
      --use_paged_context_fmha enable \
      --multiple_profiles enable \
      --workers 1
  rc=$?; if [ $rc -ne 0 ]; then echo "BUILD_FAILED rc=$rc"; exit 20; fi
fi
ls -lh "$ENGINE" || { echo "ENGINE_MISSING"; exit 21; }

########################## STAGE 3: serve ##########################
# MPI SPAWN FIX: bare trtllm-serve uses MpiPoolSession -> mpi4py COMM_SELF.Spawn -> MPI_ERR_SPAWN
# (no process manager in a plain container proc). Official fix: run under `mpirun -n 1` +
# `trtllm-llmapi-launch`, which sets TLLM_SPAWN_PROXY_PROCESS=1 and starts an mgmn comm-server so
# the LLM API BINDS to the existing world (create_mpi_comm_session) instead of spawning. Verified:
# mpirun -n1 under --cleanenv yields OMPI_COMM_WORLD_SIZE=1; wrapper + leader/worker modules present.
# ============================ MLPERF COMPLIANCE: KV-CACHE BLOCK REUSE OFF ============================
# TRT-LLM 1.0.0 defaults BOTH of these to True (verified in the container:
# /usr/local/lib/python3.12/dist-packages/tensorrt_llm/llmapi/llm_args.py:927 enable_block_reuse=True,
# :972 enable_partial_reuse=True). Block reuse is prefix-KV sharing across requests, which
# inference_rules.adoc 5.1 and 8.2 forbid, so all three reuse flags are set false below.
# The result this script produces is measured with reuse disabled.
# PLUMBING: free_gpu_memory_fraction moves INTO this yaml and the CLI flag is dropped, so the fraction has
# exactly one source of truth and cannot silently differ between invocations.
KVCFG="${WORK:-/tmp}/kvcfg_noreuse_${SLURM_JOB_ID}.yaml"
printf 'kv_cache_config:\n  enable_block_reuse: false\n  enable_partial_reuse: false\n  copy_on_partial_reuse: false\n  free_gpu_memory_fraction: 0.9\n' > "$KVCFG"
echo "[stage3] KV-cache config (block reuse DISABLED for MLPerf compliance):"; cat "$KVCFG"
echo "[stage3] $(date) trtllm-serve (mpirun -n1 + trtllm-llmapi-launch) on port $PORT"
CT mpirun -n 1 trtllm-llmapi-launch \
    trtllm-serve serve /work/l31-8b-fp8-engine \
    --tokenizer /models/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port "$PORT" \
    --backend trt \
    --max_batch_size 512 --max_num_tokens 8192 --max_seq_len 4096 \
    --tp_size 1 \
    --extra_llm_api_options "/work/$(basename "$KVCFG")" \
    > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
cleanup() { echo "[cleanup] stopping serve tree ($SERVE_PID)"; kill "$SERVE_PID" 2>/dev/null;
  # SCOPED to this job's port. An unscoped `pkill -f trtllm-serve` kills a CO-LOCATED job
  # belonging to the same user: TRT-LLM binds a hardcoded MPI-proxy port (127.0.0.1:10012),
  # so two trtllm-serve processes cannot share a node -- run them on different nodes or sequentially.
  pkill -f "trtllm-serve.*--port $PORT" 2>/dev/null;
  pkill -f "mgmn_leader_node.*$PORT" 2>/dev/null; pkill -f "mgmn_worker_node.*$PORT" 2>/dev/null;
  sleep 3; kill -9 "$SERVE_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[stage3] waiting for readiness (up to 600s)..."
ready=0
for i in $(seq 1 120); do
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "SERVE_DIED early; tail:"; tail -30 "$SERVE_LOG"; exit 30; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ $ready -ne 1 ]; then echo "SERVE_NOT_READY; tail:"; tail -40 "$SERVE_LOG"; exit 31; fi
echo "[stage3] server READY"

MODEL_ID=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "[stage3] advertised model id: $MODEL_ID"
# quick sanity generation
curl -s "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}" | head -c 400; echo

########################## STAGE 4: patch yaml + benchmark ##########################
echo "[stage4] $(date) patch yaml + run harness"
"$BASE/endpoints/.venv/bin/python" - "$SRC_YAML" "$BENCH_YAML" "$MODEL_ID" "$PORT" "$REPORT" "$MODELS/Llama-3.1-8B-Instruct" <<'PY'
import sys, yaml
src,dst,mid,port,report = sys.argv[1:6]
tok = sys.argv[6]
d=yaml.safe_load(open(src))
d['model_params']['name']=mid
# FIX 2026-07-27: model_params.name is the ENGINE id, which has no tokenizer on the HF hub, so the
# client cannot tokenize responses and result_summary.json comes back with tps=None and
# output_sequence_lengths={}. Setting tokenizer_name makes tps a measured field rather than a
# derivation. It overrides the client-side metrics
# tokenizer to the real model (schema.py: "Overrides model name for client-side token metrics
# (ISL/OSL/TPOT)"), exactly as the Server scripts already do for the Server row.
d['model_params']['tokenizer_name']=tok
# Offline terminates on n_samples_to_issue, so one epoch of 13,368 completes well inside the declared
# min_duration_ms. Set EPOCHS>1 to issue multiple epochs, which extends wall time and amortizes
# ramp/drain dilution.
import os
_ep = int(os.environ.get("EPOCHS","1"))
if _ep > 1:
    _b = d["settings"]["runtime"].get("n_samples_to_issue")
    if _b: d["settings"]["runtime"]["n_samples_to_issue"] = int(_b)*_ep
# TRT-LLM asserts topK>=0; the yaml's vLLM greedy convention top_k=-1 makes EVERY request fail
# (samplingConfig.cpp:285). temperature=0.0 already forces greedy, so top_k=1 = identical argmax output.
d['model_params']['top_k']=1
d['endpoint_config']={'endpoints':[f'http://localhost:{port}'],'api_key':None}
# CONNECTION POOL BOUND (added 2026-07-31). With max_connections=-1 the client sets the pool
# ceiling to the ENTIRE ephemeral port budget (28,232 here) and auto-warmup pre-establishes 25%
# of it, holding those ports for the whole run. At one epoch that fits; at three epochs the
# multi-epoch max_throughput burst can exhaust the range, surfacing as
# "OSError: [Errno 99] Cannot assign requested address". The server's
# max_batch_size is 512, so a 2,048 ceiling is 4x the useful concurrency and cannot throttle us.
d['settings'].setdefault('client', {})
d['settings']['client']['max_connections'] = 2048
d['settings']['client']['warmup_connections'] = 256
d['report_dir']=report
yaml.safe_dump(d,open(dst,'w'),sort_keys=False)
print('patched ->',dst)
PY

cd "$BASE/endpoints"
echo "[stage4] launching benchmark"
./.venv/bin/inference-endpoint benchmark from-config -c "$BENCH_YAML" --timeout 2400
BRC=$?
echo "[stage4] benchmark rc=$BRC"

########################## STAGE 5: results ##########################
echo "=== RESULTS ==="
PERF="$REPORT/performance/result_summary.json"
ACC="$REPORT/accuracy/accuracy_results.json"
python3 - "$PERF" "$ACC" "$ROUGE1_GATE" <<'PY'
import sys, json, os
perf,acc,gate = sys.argv[1], sys.argv[2], float(sys.argv[3])
tps=None; rouge1=None
try:
    d=json.load(open(perf)); tps=d.get('tps')
    if isinstance(tps,dict): tps=tps.get('mean') or tps.get('value') or list(tps.values())[0]
except Exception as e: print("perf read err:",e)
try:
    a=json.load(open(acc))
    def find(o):
        if isinstance(o,dict):
            for k,v in o.items():
                if k.lower() in ('rouge1','rouge-1','rouge_1'): return v
                r=find(v)
                if r is not None: return r
        return None
    rouge1=find(a)
except Exception as e: print("acc read err:",e)
print(f"TPS(tok/s, 1 GPU)   = {tps}")
print(f"ROUGE1              = {rouge1}")
print(f"ROUGE1 gate         = {gate}")
if rouge1 is not None:
    try: print("ROUGE_OK" if float(rouge1)>=gate else "ROUGE_FAIL", f"({rouge1} vs {gate})")
    except: pass
PY
echo "=== DONE $(date) ==="
