#!/bin/bash
#SBATCH --job-name=trt-l31-server-acc
#SBATCH --partition=kempner_h200_priority
#SBATCH --qos=kemp_mlcommons_gpu16
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --output=/n/holylfs06/LABS/kempner_dev/Lab/nkhoshnevis/ai_projects/mlcommon/inference/attempted/audit_fixes/logs/fix2_%j.out

# llama3.1-8b SERVER scenario on the CACHED TRT engine (l31-8b-fp8-engine). NO rebuild.
# Serve once (mpirun -n1 + trtllm-llmapi-launch + --backend trt), then run ONE online pass at
# target_qps=58 carrying BOTH the accuracy and performance datasets, so the Server row's accuracy
# and throughput come from the same measured run in the same scenario.
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
QPS="${QPS:-58}"
# Port must stay BELOW /proc/sys/net/ipv4/ip_local_port_range (32768 here): a port inside the
# ephemeral range can be transiently held by any process's outbound connection.
PORT=$(( 20000 + SLURM_JOB_ID % 12000 ))
for _ in $(seq 1 200); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; PORT=$((PORT+1)); done
echo "[port] using $PORT (below ephemeral range start 32768)"
RDIR="$BASE/attempted/audit_fixes/l31_server_acc_${SLURM_JOB_ID}"; mkdir -p "$RDIR"
SERVE_LOG="$BASE/attempted/audit_fixes/logs/fix2_serve_${SLURM_JOB_ID}.log"
TTFT_GATE_MS=2000; TPOT_GATE_MS=100; ROUGE1_GATE=38.391408

ls "$ENGINE"/*.engine >/dev/null 2>&1 || { echo "ENGINE_MISSING $ENGINE"; exit 5; }

CT() { singularity exec --cleanenv --nv \
  --env PYTHONNOUSERSITE=1 --env HF_HOME=/hf --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --bind "$MODELS":/models:ro --bind "$WORK":/work --bind "$HF_HOME":/hf \
  "$SIF" "$@"; }

KVCFG="$WORK/kvcfg_noreuse_fix2_${SLURM_JOB_ID}.yaml"
printf 'kv_cache_config:\n  enable_block_reuse: false\n  enable_partial_reuse: false\n  copy_on_partial_reuse: false\n  free_gpu_memory_fraction: 0.9\n' > "$KVCFG"
echo "[serve] KV-cache config (block reuse DISABLED, all three flags):"; cat "$KVCFG"
echo "[serve] $(date) trtllm-serve TP1 on port $PORT"
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
  # Scoped to this job's port: TRT-LLM binds a fixed MPI-proxy port (127.0.0.1:10012), so two
  # trtllm-serve processes cannot share a node. An unscoped pkill would kill the co-located job.
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

########################## Server-scenario pass: accuracy + performance ##########################
QY="$WORK/l31_server_acc_q${QPS}_${SLURM_JOB_ID}.yaml"
"$BASE/endpoints/.venv/bin/python" - "$SRC_YAML" "$QY" "$MODEL_ID" "$PORT" "$QPS" "$RDIR" "$MODELS/Llama-3.1-8B-Instruct" <<'PY'
import sys, yaml
src,dst,mid,port,qps,rep,tok = sys.argv[1:8]
d=yaml.safe_load(open(src))
d['name']='server-llama3.1-8b-cnn-submission'
d['type']='online'                                  # SERVER scenario, streaming on
d['model_params']['name']=mid
d['model_params']['tokenizer_name']=tok             # engine id has no HF tokenizer -> tps/OSL empty without this
d['model_params']['top_k']=1                        # TRT rejects -1; greedy at temperature 0 => identical argmax
d['endpoint_config']={'endpoints':[f'http://localhost:{port}'],'api_key':None}
d['settings']['load_pattern']={'type':'poisson','target_qps':int(qps)}
d['settings']['client']={'num_workers':16}
# Keep both datasets: accuracy is measured in the Server scenario, in the same pass as the
# throughput it is reported alongside.
names=[(x.get('type'),x.get('samples')) for x in d['datasets']]
assert any(t=='accuracy' for t,_ in names), 'accuracy dataset missing from source yaml'
assert any(t=='performance' for t,_ in names), 'performance dataset missing from source yaml'
# Bound the client connection pool. With max_connections=-1 the ceiling is the whole ephemeral
# port budget and warmup pre-opens a quarter of it, which exhausts the range on multi-epoch runs
# ("OSError: [Errno 99] Cannot assign requested address"). 2048 is 4x the server's max_batch_size.
d['settings'].setdefault('client', {})
d['settings']['client']['max_connections'] = 2048
d['settings']['client']['warmup_connections'] = 256
d['report_dir']=rep
yaml.safe_dump(d,open(dst,'w'),sort_keys=False)
print('patched ->',dst,'datasets:',names)
PY

cd "$BASE/endpoints"
echo "[run] $(date) Server-scenario accuracy+performance at target_qps=$QPS"
./.venv/bin/inference-endpoint benchmark from-config -c "$QY" --timeout 3600
echo "[run] rc=$?"

########################## verdict ##########################
python3 - "$RDIR" "$QPS" "$TTFT_GATE_MS" "$TPOT_GATE_MS" "$ROUGE1_GATE" <<'PY'
import sys, json, os, glob
rdir,qps,tg,pg,rg = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
print("=== RESULT: llama3.1-8b Server, accuracy measured in the Server scenario ===")
p=os.path.join(rdir,"performance","result_summary.json")
if os.path.exists(p):
    d=json.load(open(p))
    def pc(m,k='percentiles'):
        try: return d[m][k]['99.0']/1e6
        except Exception: return None
    ttft, tpot = pc('ttft'), pc('tpot')
    ttft_es = pc('ttft','early_stopping_percentiles'); tpot_es = pc('tpot','early_stopping_percentiles')
    dur = (d.get('duration_ns') or 0)/1e9
    print(f"  tps={d.get('tps')}  qps={d.get('qps')}  duration_s={dur:.1f}  state={d.get('state')}")
    print(f"  samples issued={d.get('n_samples_issued')} completed={d.get('n_samples_completed')} failed={d.get('n_samples_failed')}")
    print(f"  TTFT p99 raw={ttft} ms  early_stopping={ttft_es} ms  gate<{tg:.0f}")
    print(f"  TPOT p99 raw={tpot} ms  early_stopping={tpot_es} ms  gate<{pg:.0f}")
    binding = ttft_es if ttft_es is not None else ttft
    print(f"  TTFT VERDICT on binding figure: {'PASS' if binding is not None and binding<tg else 'FAIL'} (margin {tg-binding:.2f} ms)" if binding else "  TTFT VERDICT: UNMEASURED")
    print(f"  MIN DURATION 600s: {'OK' if dur>=600 else 'SHORT'} ({dur:.1f}s)")
else: print("  NO performance/result_summary.json")
a=os.path.join(rdir,"accuracy","accuracy_results.json")
if os.path.exists(a):
    d=json.load(open(a))
    for e in d.get("accuracy_scores",[]):
        print(f"  ACC {e.get('dataset_name')}: score={e.get('score')} n={e.get('unit_samples')} resp={e.get('response_counts')}")
    print(f"  (ROUGE1 gate {rg})")
else: print("  NO accuracy/accuracy_results.json")
print("=== END RESULT ===")
PY
echo "REPORT_DIR=$RDIR"
echo "=== DONE $(date) ==="
