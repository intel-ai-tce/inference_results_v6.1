#!/bin/bash
# Restart servers = stop_servers.sh then launch_servers.sh, same targets.
# Counterpart to stop_servers.sh / launch_servers.sh.
#
# Usage:
#   bash restart_servers.sh <target> [target ...]   (no args -> prints usage)
# Targets: 20b 120b judge embed rerank cpu all
#   cpu: the CPU-container stack (20b judge embed rerank)
#   all: cpu stack + 120b (120b normally runs on the GPU container)

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: bash restart_servers.sh <target> [target ...]"
    echo "Targets: 20b 120b judge embed rerank cpu all"
    echo "  cpu: CPU-container stack (20b judge embed rerank)"
    echo "  all: cpu stack + 120b"
}

if [ $# -eq 0 ]; then
    usage; exit 1
fi

# stop_servers.sh has no 'cpu' target; expand it to the stack it covers so the
# stop phase matches the launch phase.
stop_targets=()
for t in "$@"; do
    case "$t" in
        cpu) stop_targets+=(20b judge embed rerank) ;;
        20b|120b|judge|embed|rerank|all) stop_targets+=("$t") ;;
        *) echo "unknown target: $t (use: 20b 120b judge embed rerank cpu all)"; exit 1 ;;
    esac
done

echo "== stopping: ${stop_targets[*]} =="
bash "${_DIR}/stop_servers.sh" "${stop_targets[@]}"

echo "== launching: $* =="
bash "${_DIR}/launch_servers.sh" "$@"
