#!/bin/bash
#SBATCH --job-name=trt-l31-server
#SBATCH --partition=kempner_h200_priority
#SBATCH --qos=kemp_mlcommons_gpu16
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --output=/n/holylfs06/LABS/kempner_dev/Lab/nkhoshnevis/ai_projects/mlcommon/inference/attempted/trt/l31srv_%j.out

# llama3.1-8b SERVER scenario on the CACHED TRT engine (l31-8b-fp8-engine, the 7,987 Offline winner).
# NO rebuild. Serve once (mpirun -n1 + trtllm-llmapi-launch + --backend trt), then sweep target_qps to
# find the MAX VALID within the llama3.1 Server gates: TTFT p99 < 2000ms AND TPOT p99 < 100ms.
# PERFORMANCE-only target_qps sweep. Exploratory: this script did NOT produce the submitted
# Server row. The submitted row came from l31_8b_server_acc.sbatch, which runs accuracy and
# performance in one online pass.
# top_k -1 (endpoints default) is rejected by TRT -> patch model_params.top_k=1 (greedy temp=0 => identical).
set -uo pipefail
echo "=== START $(date) on $(hostname) ==="; nvidia-smi -L

BASE=/n/holylfs06/LABS/kempner_dev/Lab/nkhoshnevis/ai_projects/mlcommon/inference
export SCRATCH=/n/netscratch/kempner_dev/nkhoshnevis/mlcommon-scratch
SIF="$SCRATCH/containers/trtllm-release-1.0.0.sif"
MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models
export HF_HOME="$SCRATCH/hf"
WORK="$SCRATCH/trtllm-work"
ENGINE="$WORK/l31-8b-fp8-engine"; ENGINE_CT="/work/l31-8b-fp8-engine"
SRC_YAML="$BASE/endpoints/examples/05_Llama3.1-8B_Example/submission_llama3_8b_server_q58.yaml"
# Port must stay BELOW /proc/sys/net/ipv4/ip_local_port_range (32768 here): a port inside the
# ephemeral range can be transiently held by any process's outbound connection. The old formula
# A listener placed inside that window can lose its port to another process's outbound
# connection, surfacing as "[Errno 98] address already in use" at server start.
PORT=$(( 20000 + SLURM_JOB_ID % 12000 ))
for _ in $(seq 1 200); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; PORT=$((PORT+1)); done
echo "[port] using $PORT (below ephemeral range start 32768)"
RDIR="$BASE/attempted/trt/l31_server_report_${SLURM_JOB_ID}"; mkdir -p "$RDIR"
SERVE_LOG="$BASE/attempted/trt/l31srv_serve_${SLURM_JOB_ID}.log"
QPS_LIST="51 52 53 54 55 56 58"
TTFT_GATE_MS=2000; TPOT_GATE_MS=100

ls "$ENGINE"/*.engine >/dev/null 2>&1 || { echo "ENGINE_MISSING $ENGINE"; exit 5; }

CT() { singularity exec --cleanenv --nv \
  --env PYTHONNOUSERSITE=1 --env HF_HOME=/hf --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --bind "$MODELS":/models:ro --bind "$WORK":/work --bind "$HF_HOME":/hf \
  "$SIF" "$@"; }

########################## serve the cached engine (once) ##########################
# ============================ MLPERF COMPLIANCE: KV-CACHE BLOCK REUSE OFF ============================
# See l31_8b_fp8_pipeline.sbatch for the full rationale. Short form: TRT-LLM 1.0.0 defaults
# enable_block_reuse=True / enable_partial_reuse=True (llmapi/llm_args.py:927,972); that is prefix-KV
# sharing across requests, forbidden by inference_rules.adoc 5.1 and 8.2. All three reuse flags are
# set false below, so the result this script produces is measured with reuse disabled.
# Server is the MORE reuse-exposed scenario: requests arrive Poisson over a 600s+ window, so later requests
# can hit blocks left by earlier ones for the whole run -- unlike Offline, where one pass sees each prompt once.
# free_gpu_memory_fraction moves into the yaml; the CLI flag is dropped (single source of truth).
KVCFG="$WORK/kvcfg_noreuse_srv_${SLURM_JOB_ID}.yaml"
printf 'kv_cache_config:\n  enable_block_reuse: false\n  enable_partial_reuse: false\n  copy_on_partial_reuse: false\n  free_gpu_memory_fraction: 0.9\n' > "$KVCFG"
echo "[serve] KV-cache config (block reuse DISABLED for MLPerf compliance):"; cat "$KVCFG"
echo "[serve] $(date) trtllm-serve TP1 (mpirun -n1 + llmapi-launch) on port $PORT"
CUDA_VISIBLE_DEVICES=0 CT mpirun -n 1 trtllm-llmapi-launch \
    trtllm-serve serve "$ENGINE_CT" \
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

echo "[serve] waiting for readiness (up to 900s)..."
ready=0
for i in $(seq 1 90); do
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "SERVE_DIED early; tail:"; tail -40 "$SERVE_LOG"; exit 30; fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
[ $ready -ne 1 ] && { echo "SERVE_NOT_READY; tail:"; tail -40 "$SERVE_LOG"; exit 31; }
MODEL_ID=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "[serve] READY; model id: $MODEL_ID"

########################## QPS sweep (performance-only) ##########################
cd "$BASE/endpoints"
for QPS in $QPS_LIST; do
  QY="$WORK/l31_server_q${QPS}_${SLURM_JOB_ID}.yaml"
  QOUT="$RDIR/q${QPS}"
  # patch: model id, top_k=1 (greedy), endpoint port, target_qps, report_dir, PERFORMANCE-only dataset
  "$BASE/endpoints/.venv/bin/python" - "$SRC_YAML" "$QY" "$MODEL_ID" "$PORT" "$QPS" "$QOUT" "$MODELS/Llama-3.1-8B-Instruct" <<'PY'
import sys, yaml
src,dst,mid,port,qps,rep,tok = sys.argv[1:8]
d=yaml.safe_load(open(src))
d['model_params']['name']=mid
# CRITICAL: name=engine id has no tokenizer -> OSL/TPOT/tps come out empty. tokenizer_name overrides
# the client-side metrics tokenizer to the real HF model (schema.py: "Overrides model name for
# client-side token metrics (ISL/OSL/TPOT)"), so tps computes exactly like E11.
d['model_params']['tokenizer_name']=tok
d['model_params']['top_k']=1                      # TRT rejects -1; greedy => identical output
d['endpoint_config']={'endpoints':[f'http://localhost:{port}'],'api_key':None}
d['settings']['load_pattern']={'type':'poisson','target_qps':int(qps)}
d['datasets']=[x for x in d['datasets'] if x.get('type')=='performance']  # perf-only; accuracy reused from Offline
# CONNECTION POOL BOUND (added 2026-07-31). With max_connections=-1 the client sets the pool
# ceiling to the ENTIRE ephemeral port budget (28,232 here) and auto-warmup pre-establishes 25%
# of it, holding those ports for the whole run. At one epoch that fits; at three epochs the
# multi-epoch max_throughput burst can exhaust the range, surfacing as
# "OSError: [Errno 99] Cannot assign requested address". The server's
# max_batch_size is 512, so a 2,048 ceiling is 4x the useful concurrency and cannot throttle us.
d['settings'].setdefault('client', {})
d['settings']['client']['max_connections'] = 2048
d['settings']['client']['warmup_connections'] = 256
d['report_dir']=rep
yaml.safe_dump(d,open(dst,'w'),sort_keys=False)
print('patched qps',qps,'->',dst)
PY
  echo "[sweep] $(date) target_qps=$QPS"
  ./.venv/bin/inference-endpoint benchmark from-config -c "$QY" --timeout 2400 2>&1 | tail -2
  # parse
  python3 - "$QOUT/performance/result_summary.json" "$QPS" "$TTFT_GATE_MS" "$TPOT_GATE_MS" <<'PY'
import sys, json
f,qps,tg,pg = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
try:
    d=json.load(open(f))
    eqps=d.get('qps'); osl=d.get('output_sequence_lengths') or {}; mean_out=osl.get('avg')
    tps=d.get('tps'); src='native'                        # E11 uses this field (= eff_qps x mean_output)
    if tps is None:                                       # fallback if tokenizer still didn't attach
        if eqps and mean_out: tps=eqps*mean_out; src='qps*osl_avg'
        elif eqps: tps=eqps*128.0; src='qps*128cap_approx'
    def p99(m):
        try: return d[m]['percentiles']['99.0']/1e6       # ns->ms
        except Exception: return None
    ttft=p99('ttft'); tpot=p99('tpot')
    ttft_ok = ttft is not None and ttft<tg
    tpot_ok = (tpot is None) or (tpot<pg)                 # don't hard-fail if tpot unmeasured (flag)
    valid = ttft_ok and tpot_ok and d.get('state')=='complete'
    ts = f"{tps:.1f}" if tps is not None else "NA"
    tt = f"{ttft:.1f}" if ttft is not None else "NA"
    tp = f"{tpot:.2f}" if tpot is not None else "NA_unmeasured"
    print(f"CAMPROW l31-server qps={qps} tps_per_gpu={ts}({src}) mean_out={mean_out} eff_qps={eqps:.2f} ttft_p99_ms={tt}(gate<{tg:.0f}) tpot_p99_ms={tp}(gate<{pg:.0f}) {'VALID' if valid else 'INVALID'}")
except Exception as e:
    print(f"CAMPROW l31-server qps={qps} PARSE_FAIL {e}")
PY
done
echo "=== DONE $(date) ==="
