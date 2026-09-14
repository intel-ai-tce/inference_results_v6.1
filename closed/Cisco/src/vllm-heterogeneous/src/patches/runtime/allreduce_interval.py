
"""
Increase the engine loop's DP all-reduce check interval from 32 to 512 steps.

The _has_global_unfinished_reqs() function performs a dist.all_reduce every
32 steps to determine if any DP rank still has work. This acts as a barrier:
if any rank experiences a pause (GC, kernel jitter), all other ranks block.

During benchmark runs there are always unfinished requests, so this check
always returns True. Increasing the interval from 32 to 512 reduces the
number of synchronization points by 16x while having no effect on correctness.

Usage: python3 scripts/patch_allreduce_interval.py
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
    print("[AR-INTERVAL] ERROR: Could not find engine/core.py")
    sys.exit(1)

print("=" * 60)
print("All-Reduce Interval Patch (32 -> 512)")
print("=" * 60)
print(f"\n  Target: {CORE_FILE}")

backup = CORE_FILE + ".ar_interval_bak"
if not os.path.exists(backup):
    shutil.copy2(CORE_FILE, backup)
    print(f"  Backup: {backup}")

with open(CORE_FILE, "r") as f:
    content = f.read()

if "_ar_interval_patched" in content:
    print("\n  [AR-INTERVAL] Already applied, skipping.")
    sys.exit(0)

OLD_CHECK = "        if self.step_counter % 32 != 0:\n            return True"
if OLD_CHECK not in content:
    print(f"\n  ERROR: anchor not found")
    sys.exit(1)

NEW_CHECK = "        _ar_interval_patched = True  # sentinel\n        if self.step_counter % 512 != 0:\n            return True"

content = content.replace(OLD_CHECK, NEW_CHECK, 1)
print("  Changed all-reduce check interval: 32 -> 512")

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

print("\n  [AR-INTERVAL] Patch applied successfully.")
print("  Ranks now synchronize every 512 steps instead of 32.\n")
