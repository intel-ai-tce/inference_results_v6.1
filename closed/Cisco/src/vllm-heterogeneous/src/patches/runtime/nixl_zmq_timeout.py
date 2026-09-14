
"""
Patch NIXL connector to increase ZMQ handshake timeout.

The default ZMQ RCVTIMEO on the NIXL handshake socket is too short.
When the prefill side is busy with JIT compilation or inference, it
cannot respond to handshake requests in time, causing:
  zmq.error.Again: Resource temporarily unavailable

This patch increases the ZMQ receive timeout for NIXL handshake sockets
from the default (typically 5-10s) to 120 seconds.

Usage (inside container):
    python <submission-root>/src/patches/runtime/nixl_zmq_timeout.py
"""

import importlib
import importlib.util
import os
import re
import sys

TARGET_MODULES = [
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker",
]
PATCH_SENTINEL = "# [PATCH] Increased NIXL ZMQ handshake timeout"
TIMEOUT_MS = 600_000  


def find_source():
    for module in TARGET_MODULES:
        spec = importlib.util.find_spec(module)
        if spec is not None and spec.origin is not None:
            return spec.origin
    print(
        "ERROR: Cannot find any NIXL connector module: "
        + ", ".join(TARGET_MODULES),
        file=sys.stderr,
    )
    sys.exit(1)


def patch(path: str):
    with open(path, "r") as f:
        src = f.read()

    if PATCH_SENTINEL in src:
        if str(TIMEOUT_MS) in src:
            print(f"NIXL ZMQ timeout: already patched ({path})")
            return
        else:
            print(f"NIXL ZMQ timeout: updating timeout to {TIMEOUT_MS}ms ({path})")
            src = src.replace(PATCH_SENTINEL, "")
            src = re.sub(r'sock\.setsockopt\(zmq\.RCVTIMEO,\s*\d+\)', '', src)
            src = re.sub(r'sock\.setsockopt\(zmq\.SNDTIMEO,\s*\d+\)', '', src)
            src = re.sub(r'\n\s*\n\s*\n', '\n\n', src)

    original = src
    patched = False

    
    rcvtimeo_pattern = re.compile(
        r'(sock\.setsockopt\s*\(\s*zmq\.RCVTIMEO\s*,\s*)(\d+)(\s*\))'
    )
    if rcvtimeo_pattern.search(src):
        src = rcvtimeo_pattern.sub(
            rf'\g<1>{TIMEOUT_MS}\3  {PATCH_SENTINEL}', src
        )
        patched = True

    
    if not patched:
        
        
        handshake_socket_pattern = re.compile(
            r'([ \t]+)(sock\s*=\s*\w+\.socket\s*\(\s*zmq\.REQ\s*\))'
        )
        match = handshake_socket_pattern.search(src)
        if match:
            indent = match.group(1)
            socket_line = match.group(2)
            replacement = (
                f"{indent}{socket_line}\n"
                f"{indent}{PATCH_SENTINEL}\n"
                f"{indent}sock.setsockopt(zmq.RCVTIMEO, {TIMEOUT_MS})\n"
                f"{indent}sock.setsockopt(zmq.SNDTIMEO, {TIMEOUT_MS})"
            )
            src = src[:match.start()] + replacement + src[match.end():]
            patched = True

    
    if not patched:
        
        hs_func_match = re.search(r'def _nixl_handshake\b', src)
        if hs_func_match:
            
            connect_pattern = re.compile(r'([ \t]+)(sock\.connect\s*\()')
            connect_match = connect_pattern.search(src, hs_func_match.end())
            if connect_match:
                indent = connect_match.group(1)
                insert_point = connect_match.start()
                timeout_lines = (
                    f"{indent}{PATCH_SENTINEL}\n"
                    f"{indent}sock.setsockopt(zmq.RCVTIMEO, {TIMEOUT_MS})\n"
                    f"{indent}sock.setsockopt(zmq.SNDTIMEO, {TIMEOUT_MS})\n"
                )
                src = src[:insert_point] + timeout_lines + src[insert_point:]
                patched = True

    if not patched:
        print(f"WARNING: Could not find ZMQ socket pattern to patch in {path}")
        print("  Attempting global RCVTIMEO injection on all zmq.REQ sockets...")

        
        req_pattern = re.compile(
            r'([ \t]+)(\w+\s*=\s*\w+\.socket\s*\(\s*zmq\.REQ\s*\))'
        )
        matches = list(req_pattern.finditer(src))
        if matches:
            for match in reversed(matches):
                indent = match.group(1)
                var_name = match.group(2).split("=")[0].strip()
                insert_after = match.end()
                timeout_lines = (
                    f"\n{indent}{PATCH_SENTINEL}\n"
                    f"{indent}{var_name}.setsockopt(zmq.RCVTIMEO, {TIMEOUT_MS})\n"
                    f"{indent}{var_name}.setsockopt(zmq.SNDTIMEO, {TIMEOUT_MS})"
                )
                src = src[:insert_after] + timeout_lines + src[insert_after:]
            patched = True

    if not patched:
        print(f"ERROR: Could not patch {path} - no recognizable ZMQ socket pattern found")
        print("  Please manually add: sock.setsockopt(zmq.RCVTIMEO, 120000)")
        sys.exit(1)

    backup = path + ".bak.zmq"
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(path, backup)

    with open(path, "w") as f:
        f.write(src)

    print(f"NIXL ZMQ timeout: patched (RCVTIMEO={TIMEOUT_MS}ms) ({path})")


if __name__ == "__main__":
    path = find_source()
    patch(path)
