
"""
GPU synchronization patch for NIXL KV cache transfers.

In cross-vendor PD (H200 prefill → MI350X decode), RDMA reads from the
decode side pull KV cache data directly from the prefill GPU's memory.
If the prefill GPU's CUDA kernels (which wrote the KV cache during the
forward pass) haven't fully flushed to device memory before the decode
side's RDMA NIC reads, the NIC may fetch stale/partial data, causing
garbage output.

This patch adds torch.cuda.synchronize() barriers at two critical points:

  PREFILL side — NixlConnector.wait_for_save()
    Called after every forward pass, before get_finished().
    Ensures all KV-cache writes from attention kernels are visible to the
    RDMA NIC before the response (with kv_transfer_params) leaves the engine.

  DECODE side — NixlConnectorWorker.get_finished()
    Called every step to poll RDMA completion.
    After RDMA transfers complete, ensures the DMA-written data is visible
    to subsequent GPU compute kernels (cache coherence barrier).

Performance: adds one torch.cuda.synchronize() per engine step on each
side.  In practice this overlaps with the natural pipeline idle gap and
adds < 1 ms.

Run INSIDE the container (both prefill and decode):
    python3 <submission-root>/src/patches/cross_platform/nixl_gpu_sync.py
"""

import os
import sys


def find_file(candidates, fallback_subpath=None, label="file"):
    for c in candidates:
        if os.path.exists(c):
            return c
    if fallback_subpath:
        try:
            import vllm
            vllm_dir = os.path.dirname(vllm.__file__)
            p = os.path.join(vllm_dir, *fallback_subpath)
            if os.path.exists(p):
                return p
        except ImportError:
            pass
    return None


NIXL_FILE = find_file(
    [
    ],
    fallback_subpath=["distributed", "kv_transfer", "kv_connector", "v1",
                       "nixl_connector.py"],
    label="nixl_connector.py",
)
NIXL_LAYOUT = "flat"
if NIXL_FILE is None:
    NIXL_FILE = find_file(
        [
        ],
        fallback_subpath=["distributed", "kv_transfer", "kv_connector", "v1",
                          "nixl", "__init__.py"],
        label="nixl/__init__.py",
    )
    NIXL_LAYOUT = "package"

if NIXL_FILE is None:
    print("[GPU-SYNC] ERROR: Could not find NIXL connector module/package")
    sys.exit(1)

print("=" * 60)
print("NIXL GPU Sync Patch")
print("=" * 60)
print(f"\n  NIXL patch target ({NIXL_LAYOUT}): {NIXL_FILE}")

PATCH_DIR = os.path.dirname(NIXL_FILE)
PATCH_FILE = os.path.join(PATCH_DIR, "_nixl_gpu_sync.py")

PATCH_CODE = r'''"""GPU synchronization barriers for NIXL KV cache transfers.

PREFILL: torch.cuda.synchronize() after forward pass, before response is
sent. Guarantees KV cache writes from attention are visible to RDMA NIC.

DECODE: torch.cuda.synchronize() after RDMA transfers complete, before
data is used by attention. Cache coherence barrier for DMA writes.
"""

import os
import logging
import torch

logger = logging.getLogger(__name__)

_ENABLE = os.environ.get("NIXL_GPU_SYNC", "1") != "0"
_LOG_FIRST = True


def _install():
    if not _ENABLE:
        logger.info("[GPU-SYNC] Disabled via NIXL_GPU_SYNC=0")
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import nixl_connector
        connector_cls = nixl_connector.NixlConnector
        worker_cls = nixl_connector.NixlConnectorWorker
    except (ImportError, AttributeError):
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
                NixlConnector as connector_cls,
            )
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
                NixlConnectorWorker as worker_cls,
            )
        except (ImportError, AttributeError):
            print("[GPU-SYNC] NixlConnector not found — skipping", flush=True)
            return

    # ---- PREFILL side: wrap wait_for_save ----
    if hasattr(connector_cls.wait_for_save, '_gpu_sync_wrapped'):
        return

    _orig_wait_for_save = connector_cls.wait_for_save

    def _synced_wait_for_save(self):
        _orig_wait_for_save(self)
        if (hasattr(self, '_connector_metadata')
                and self._connector_metadata is not None
                and hasattr(self._connector_metadata, 'reqs_to_save')
                and self._connector_metadata.reqs_to_save):
            torch.cuda.synchronize()
            global _LOG_FIRST
            if _LOG_FIRST:
                print(f"[GPU-SYNC] Prefill: torch.cuda.synchronize() "
                      f"after wait_for_save ({len(self._connector_metadata.reqs_to_save)} reqs)",
                      flush=True)
                _LOG_FIRST = False

    _synced_wait_for_save._gpu_sync_wrapped = True
    connector_cls.wait_for_save = _synced_wait_for_save

    # ---- DECODE side: wrap get_finished on the worker ----
    _orig_get_finished = worker_cls.get_finished

    def _synced_get_finished(self):
        done_sending, done_recving = _orig_get_finished(self)
        if done_recving:
            torch.cuda.synchronize()
        return done_sending, done_recving

    _synced_get_finished._gpu_sync_wrapped = True
    worker_cls.get_finished = _synced_get_finished

    print("[GPU-SYNC] Patched wait_for_save (prefill) + "
          "get_finished (decode)", flush=True)


_install()
'''

print("\n--- Creating _nixl_gpu_sync.py ---")

with open(PATCH_FILE, "w") as f:
    f.write(PATCH_CODE)
print(f"  Created: {PATCH_FILE}")

print("  Syntax check...")
try:
    compile(open(PATCH_FILE).read(), PATCH_FILE, "exec")
    print("  _nixl_gpu_sync.py: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    sys.exit(1)


print("\n--- Injecting import into nixl_connector.py ---")

with open(NIXL_FILE, "r") as f:
    nixl_content = f.read()

if "_nixl_gpu_sync" in nixl_content:
    print("  Import already present in NIXL patch target")
else:
    import_line = (
        "\ntry:\n"
        "    from . import _nixl_gpu_sync  # GPU sync barriers for RDMA\n"
        "except Exception:\n"
        "    pass\n"
    )
    nixl_content = nixl_content.rstrip() + "\n" + import_line
    with open(NIXL_FILE, "w") as f:
        f.write(nixl_content)
    print("  Added _nixl_gpu_sync import to NIXL patch target")


nixl_cache = os.path.join(PATCH_DIR, "__pycache__")
if os.path.isdir(nixl_cache):
    for fn in os.listdir(nixl_cache):
        if "nixl_connector" in fn or "worker" in fn or "_nixl_gpu_sync" in fn:
            os.remove(os.path.join(nixl_cache, fn))
            print(f"  Cleared: __pycache__/{fn}")

print(f"""
{'=' * 60}
Done! GPU sync patch applied.
{'=' * 60}

The patch adds torch.cuda.synchronize() at two points:
  1. PREFILL: after forward pass KV writes (wait_for_save)
  2. DECODE: after RDMA transfer completion (get_finished)

To disable at runtime without removing the patch:
  export NIXL_GPU_SYNC=0
""")
