"""Compare a system description JSON against the local machine.

Intended to be run inside the wan-harness container on the submission host,
e.g.::

    python3 -m tools.verify_system_desc systems/8xMI355X_2xEPYC_9575F.json

Exit code 0 when all checked fields match; 1 when mismatches are found.
Fields that cannot be read locally are reported as ``skipped``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger("verify_system_desc")


@dataclass(frozen=True)
class CheckResult:
    field: str
    expected: str
    actual: str
    status: str  # ok | mismatch | skipped


def _read_os_pretty_name() -> str | None:
    path = Path("/etc/os-release")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def _read_lscpu() -> dict[str, str]:
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    data: dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip().lower()] = value.strip()
    return data


def _read_dmi(field: str) -> str | None:
    path = Path(f"/sys/class/dmi/id/{field}")
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _gpu_count() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showid"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return len(re.findall(r"^GPU\[\d+\]", out, flags=re.MULTILINE))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _gpu_model() -> str | None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _storage_capacity_tb() -> str | None:
    try:
        out = subprocess.check_output(
            ["lsblk", "-d", "-b", "-o", "NAME,SIZE,TYPE"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    total_bytes = 0
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3 or parts[2] != "disk":
            continue
        if parts[0].startswith("loop"):
            continue
        total_bytes += int(parts[1])
    if total_bytes <= 0:
        return None
    return f"{total_bytes / 1_000_000_000_000:.1f}TB"


def _framework_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import torch

        versions["pytorch"] = torch.__version__
    except Exception:
        pass
    rocm = Path("/opt/rocm/.info/version")
    if rocm.is_file():
        versions["rocm"] = rocm.read_text(encoding="utf-8").strip()
    try:
        import importlib.metadata as md

        versions["xfuser"] = md.version("xfuser")
        versions["amd-aiter"] = md.version("amd-aiter")
    except Exception:
        pass
    try:
        import xfuser

        xdit_root = Path(xfuser.__file__).resolve().parent.parent
        sha = subprocess.check_output(
            ["git", "-C", str(xdit_root), "rev-parse", "--short=7", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        versions["xdit_git"] = sha
    except Exception:
        pass
    return versions


def _normalize_system_name(name: str) -> str:
    return re.sub(r"[\s-]+", "", name.lower().replace("supermicro", "supermicro"))


def verify_system_desc(payload: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    lscpu = _read_lscpu()

    checks: list[tuple[str, str | None, str | None, str]] = []

    gpu_n = _gpu_count()
    if gpu_n is not None:
        checks.append(
            (
                "accelerators_per_node",
                str(payload.get("accelerators_per_node")),
                str(gpu_n),
                "exact",
            )
        )

    gpu_model = _gpu_model()
    if gpu_model is not None:
        checks.append(
            (
                "accelerator_model_name",
                str(payload.get("accelerator_model_name")),
                gpu_model,
                "prefix",
            )
        )

    if "model name" in lscpu:
        checks.append(
            (
                "host_processor_model_name",
                str(payload.get("host_processor_model_name")),
                lscpu["model name"],
                "contains",
            )
        )
    if "socket(s)" in lscpu:
        checks.append(
            (
                "host_processors_per_node",
                str(payload.get("host_processors_per_node")),
                lscpu["socket(s)"],
                "exact",
            )
        )
    if "core(s) per socket" in lscpu:
        checks.append(
            (
                "host_processor_core_count",
                str(payload.get("host_processor_core_count")),
                lscpu["core(s) per socket"],
                "exact",
            )
        )

    os_name = _read_os_pretty_name()
    if os_name is not None:
        checks.append(("operating_system", str(payload.get("operating_system")), os_name, "os"))

    vendor = _read_dmi("sys_vendor")
    product = _read_dmi("product_name")
    if vendor and product:
        actual_system = f"{vendor} {product}".strip()
        expected = str(payload.get("system_name"))
        status = (
            "ok"
            if _normalize_system_name(actual_system) == _normalize_system_name(expected)
            else "mismatch"
        )
        results.append(
            CheckResult("system_name", expected, actual_system, status)
        )

    storage = _storage_capacity_tb()
    if storage is not None:
        checks.append(
            ("host_storage_capacity", str(payload.get("host_storage_capacity")), storage, "storage")
        )

    for field, expected, actual, mode in checks:
        if expected is None or actual is None:
            results.append(CheckResult(field, expected or "", actual or "", "skipped"))
            continue
        exp_norm = expected.strip().lower()
        act_norm = actual.strip().lower()
        if mode == "prefix":
            ok = act_norm in exp_norm or exp_norm.startswith(act_norm)
        elif mode == "contains":
            ok = exp_norm in act_norm or act_norm in exp_norm
        elif mode == "storage":
            ok = exp_norm.replace("tb", "") in act_norm.replace("tb", "") or act_norm.replace("tb", "") in exp_norm.replace("tb", "")
        elif mode == "os":
            ok = exp_norm.split()[0:2] == act_norm.split()[0:2] or exp_norm in act_norm
        else:
            ok = exp_norm == act_norm
        results.append(CheckResult(field, expected, actual, "ok" if ok else "mismatch"))

    fw = str(payload.get("framework", ""))
    versions = _framework_versions()
    for key, label in (
        ("pytorch", "PyTorch"),
        ("rocm", "ROCm"),
        ("xfuser", "xfuser"),
        ("xdit_git", "xDiT"),
    ):
        value = versions.get(key)
        if not value:
            results.append(CheckResult(f"framework/{key}", fw, "", "skipped"))
            continue
        token = value if key != "xdit_git" else f"@{value}"
        status = "ok" if token.lower() in fw.lower() else "mismatch"
        results.append(CheckResult(f"framework/{key}", token, value, status))

    aiter = versions.get("amd-aiter")
    if aiter:
        expected = str(payload.get("other_software_stack", ""))
        status = "ok" if aiter.lower() in expected.lower() else "mismatch"
        results.append(
            CheckResult("other_software_stack", expected, aiter, status)
        )

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify system description JSON against this host.")
    parser.add_argument("system_desc", type=Path, help="Path to system description JSON.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    path = args.system_desc.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = verify_system_desc(payload)

    mismatches = 0
    for item in results:
        if item.status == "ok":
            _log.info("OK   %-28s expected=%r actual=%r", item.field, item.expected, item.actual)
        elif item.status == "skipped":
            _log.info("SKIP %-28s (could not read local value)", item.field)
        else:
            mismatches += 1
            _log.error(
                "FAIL %-28s expected=%r actual=%r",
                item.field,
                item.expected,
                item.actual,
            )

    if mismatches:
        _log.error("%d mismatch(es) in %s", mismatches, path)
        return 1
    _log.info("all checked fields match for %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
