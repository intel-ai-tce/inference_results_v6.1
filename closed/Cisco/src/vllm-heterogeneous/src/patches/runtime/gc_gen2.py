
"""
Disable Python GC gen2 collections to prevent multi-second pauses.

After vLLM calls gc.freeze() to exclude static objects from collection,
this patch sets gc.set_threshold(700, 10, 0) to disable gen2 collections
entirely.  Gen0/gen1 collections continue normally (~0.1ms each), but gen2
(which scans the entire unfrozen heap and can take 1-3 seconds) is disabled.

This prevents the periodic 2-second pauses that appear on random DP ranks.

Usage: python3 scripts/patch_gc_gen2.py
  (run before starting the decode worker)
"""

import os
import sys
import shutil


def find_file(candidates, fallback_subpath=None):
    for c in candidates:
        if os.path.exists(c):
            return c
    if fallback_subpath:
        try:
            import vllm
            p = os.path.join(os.path.dirname(vllm.__file__), *fallback_subpath)
            if os.path.exists(p):
                return p
        except ImportError:
            pass
    return None


CORE_FILE = find_file(
    [
    ],
    fallback_subpath=["v1", "engine", "core.py"],
)

if CORE_FILE is None:
    print("[GC-GEN2] ERROR: Could not find engine/core.py")
    sys.exit(1)

print("=" * 60)
print("Disable GC Gen2 Collections Patch")
print("=" * 60)
print(f"\n  Target: {CORE_FILE}")

backup = CORE_FILE + ".gc_gen2_bak"
if not os.path.exists(backup):
    shutil.copy2(CORE_FILE, backup)
    print(f"  Backup: {backup}")

with open(CORE_FILE, "r") as f:
    content = f.read()

if "_gc_gen2_patched" in content:
    print("\n  [GC-GEN2] Already applied, skipping.")
    sys.exit(0)

OLD_FREEZE = "        freeze_gc_heap()"
if OLD_FREEZE not in content:
    print(f"\n  ERROR: anchor not found: {OLD_FREEZE!r}")
    sys.exit(1)

NEW_FREEZE = """        freeze_gc_heap()
        # --- GC-GEN2: disable gen2 collections to prevent multi-second pauses ---
        import gc as _gc_gen2_mod
        _gc_gen2_patched = True
        _old_thresh = _gc_gen2_mod.get_threshold()
        _gc_gen2_mod.set_threshold(_old_thresh[0], _old_thresh[1], 0)
        logger.info("GC gen2 disabled (was threshold=%s, now=%s)",
                     _old_thresh, _gc_gen2_mod.get_threshold())
        # --- end GC-GEN2 ---"""

content = content.replace(OLD_FREEZE, NEW_FREEZE, 1)
print("  Injected gc.set_threshold(..., 0) after freeze_gc_heap()")

with open(CORE_FILE, "w") as f:
    f.write(content)

print(f"\n  Syntax check...")
try:
    compile(open(CORE_FILE).read(), CORE_FILE, "exec")
    print(f"  {os.path.basename(CORE_FILE)}: OK")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    shutil.copy2(backup, CORE_FILE)
    print("  Restored from backup.")
    sys.exit(1)

print("\n  [GC-GEN2] Patch applied successfully.")
print("  Gen0/Gen1 collections continue normally (~0.1ms each).")
print("  Gen2 collections disabled (prevents 1-3 second pauses).\n")
