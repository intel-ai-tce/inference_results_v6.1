
"""
Patch vLLM core.py to make engine_id unique per DP rank for non-MoE models.

Bug: For non-MoE models (like Llama2 70B), vLLM treats each DP rank as
independent (data_parallel_size=1) and does NOT modify the NIXL engine_id.
All 8 DP ranks share the same engine_id. The engine_id is only made unique
inside _init_data_parallel(), which is only called for MoE models via
DPEngineCoreProc.

Impact: The decode worker (MI350X) uses engine_id as the key for NIXL
handshakes. When all prefill DP ranks share the same engine_id, the decode
worker handshakes with the first DP rank it encounters and reuses that
connection for ALL requests — even those whose KV blocks live on other DP
ranks. This causes:
  - "remote index out of range" / "transfer_setup_failed" (wrong memory)
  - "unrecognized request" notifications on the wrong DP rank
  - "Releasing expired KV blocks ... 0 decode worker(s)" (correct rank
    never gets notified)

Fix: Append _dp{local_dp_rank} to engine_id in the non-MoE branch of
run_engine_core(), matching what _init_data_parallel() already does for MoE.

Run INSIDE the H200 container:
    python3 <submission-root>/scripts/patch_dp_engine_id.py
"""

import os
import shutil
import sys


core_file = None
try:
    import vllm
    candidate = os.path.join(os.path.dirname(vllm.__file__), "v1", "engine", "core.py")
    if os.path.exists(candidate):
        core_file = candidate
except ImportError:
    pass

if core_file is None:
    print("ERROR: Could not find vllm/v1/engine/core.py through the installed package")
    sys.exit(1)

print(f"Found: {core_file}")

with open(core_file, "r") as f:
    content = f.read()

MARKER = "# PATCH: dp_engine_id unique per non-MoE DP rank"
MARKER_V2 = "_base_eid"
if MARKER in content:
    if MARKER_V2 in content:
        print("Patch v2 (accumulation-safe) already applied.")
        sys.exit(0)
    else:
        print("Old patch (v1) detected — restoring from backup to re-apply v2...")
        backup = core_file + ".bak_dp_engine_id"
        if os.path.exists(backup):
            shutil.copy2(backup, core_file)
            with open(core_file, "r") as f:
                content = f.read()
            print("Restored from backup.")
        else:
            print("ERROR: backup not found, cannot upgrade patch.")
            sys.exit(1)


OLD = """\
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)"""

NEW = """\
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                # PATCH: dp_engine_id unique per non-MoE DP rank
                # Save base engine_id on first iteration to avoid accumulation
                # across loop iterations (vllm_config is shared).
                if vllm_config.kv_transfer_config is not None:
                    if not hasattr(vllm_config.kv_transfer_config, '_base_eid'):
                        vllm_config.kv_transfer_config._base_eid = (
                            vllm_config.kv_transfer_config.engine_id)
                    vllm_config.kv_transfer_config.engine_id = (
                        f"{vllm_config.kv_transfer_config._base_eid}"
                        f"_dp{local_dp_rank}"
                    )
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)"""

if OLD not in content:
    print("ERROR: Could not find the target code block in core.py.")
    print("The file may have been modified or is a different vLLM version.")
    print()
    print("Expected to find:")
    for line in OLD.split("\n")[:4]:
        print(f"  {line}")
    sys.exit(1)

backup = core_file + ".bak_dp_engine_id"
if not os.path.exists(backup):
    shutil.copy2(core_file, backup)
    print(f"Backup: {backup}")

content = content.replace(OLD, NEW, 1)

with open(core_file, "w") as f:
    f.write(content)

print("Patched: engine_id now unique per non-MoE DP rank.")
print(f"Written to {core_file}")
