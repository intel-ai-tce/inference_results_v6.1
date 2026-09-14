#!/bin/bash

parse_servers() {
    local yaml="$1" role="$2" addr_var="$3" count_var="$4" hardware="${5:-}" scenario="${6:-}" profile="${7:-}"
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    eval "$(REPO_ROOT="${repo_root}" python3 - "$yaml" "$role" "$hardware" "$scenario" "$profile" "$addr_var" "$count_var" <<'PYTHON'
import os
import re
import shlex
import sys
import yaml

path, role, hardware, scenario, profile, addr_var, count_var = sys.argv[1:]
pattern = re.compile(r"\$\{oc\.env:([^},]+)(?:,[^}]*)?\}")
def replace(match):
    variable = match.group(1)
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"Required deployment variable {variable} is unset")
    return value
try:
    with open(path, encoding="utf-8") as handle:
        model = yaml.safe_load(pattern.sub(replace, handle.read()))
    sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "src"))
    from config_helpers import resolve_standalone_endpoints, select_model_profile
    model = select_model_profile(model, profile or None, scenario or None)
    if role == "standalone":
        endpoints = resolve_standalone_endpoints(
            model, hardware=hardware or None, scenario=scenario or None
        )
    else:
        endpoints = (model.get("servers") or {}).get(role, [])
        if isinstance(endpoints, str):
            endpoints = endpoints.split()
except (OSError, ValueError, yaml.YAMLError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
addresses = ",".join(str(host).rsplit(":", 1)[0] for host in endpoints) if endpoints else ""
print(f"{addr_var}={shlex.quote(addresses)}")
print(f"{count_var}={shlex.quote(str(len(endpoints)))}")
PYTHON
)"
    export "${addr_var}" "${count_var}"
}
