


from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

MARKER = "ENGINE-CORE-ENV"
BACKUP_SUFFIX = ".engine_core_env_bak"


def _vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if not spec or not spec.origin:
        raise RuntimeError("Could not locate installed vLLM package")
    return Path(spec.origin).parent


def main() -> None:
    target = _vllm_root() / "v1" / "engine" / "core.py"
    if not target.exists():
        raise FileNotFoundError(target)

    text = target.read_text()
    if MARKER in text:
        print(f"{MARKER}: already patched ({target})")
        return

    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(text)

    anchor = "        # Ensure we can serialize transformer config after spawning\n"
    insert = '        # ENGINE-CORE-ENV: vLLM EngineCore can lose selected ROCm/PyTorch env vars.\n        if os.environ.get("VLLM_ENGINE_CORE_DISABLE_TUNABLEOP", "1") != "0":\n            os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"\n        os.environ.setdefault(\n            "GCN_ARCH_NAME",\n            os.environ.get("AITER_FORCE_GCN_ARCH_NAME", "gfx950"),\n        )\n\n'
    if anchor not in text:
        raise RuntimeError("run_engine_core anchor not found")

    text = text.replace(anchor, insert + anchor, 1)

    ready_anchor = (
        '                assert addresses.coordinator_input is not None\n'
        '                logger.info("Waiting for READY message from DP Coordinator...")\n'
    )
    ready_insert = (
        '                # ENGINE-CORE-ENV: local DP=1 has no coordinator socket.\n'
        '                if addresses.coordinator_input is None:\n'
        '                    logger.info("Waiting for local input socket READY message...")\n'
        '                else:\n'
        '                    logger.info("Waiting for READY message from DP Coordinator...")\n'
    )
    if ready_anchor not in text:
        raise RuntimeError("EngineCore READY wait anchor not found")
    text = text.replace(ready_anchor, ready_insert, 1)

    target.write_text(text)
    py_compile.compile(str(target), doraise=True)
    print(f"{MARKER}: patched {target}")


if __name__ == "__main__":
    main()
