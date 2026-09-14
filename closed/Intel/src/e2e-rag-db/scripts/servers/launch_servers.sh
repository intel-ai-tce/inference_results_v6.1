#!/bin/bash
# Launch servers by delegating to the individual launch_server_*.sh scripts.
# Counterpart to stop_servers.sh (same target names).
#
# Usage:
#   bash launch_servers.sh <target> [target ...]   (no args -> prints usage)
# Targets: 20b 120b judge embed rerank cpu all
#   cpu: the CPU-container stack (20b judge embed rerank)
#   all: cpu stack + 120b (120b normally runs on the GPU container)

_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Map a target to its launcher script (judge -> CPU 8B judge).
launcher_for() {
    case "$1" in
        20b)    echo "launch_server_20b.sh" ;;
        120b)   echo "launch_server_120b.sh" ;;
        judge)  echo "launch_server_8b_judge.sh" ;;
        embed)  echo "launch_server_embedding.sh" ;;
        rerank) echo "launch_server_rerank.sh" ;;
        *)      echo "" ;;
    esac
}

launch_one() {
    local t="$1" script; script="$(launcher_for "$t")"
    if [ -z "$script" ]; then
        echo "unknown target: $t (use: 20b 120b judge embed rerank cpu all)"; return 1
    fi
    echo "[$t] launching ${script} ..."
    bash "${_DIR}/${script}"
}

usage() {
    echo "Usage: bash launch_servers.sh <target> [target ...]"
    echo "Targets: 20b 120b judge embed rerank cpu all"
    echo "  cpu: CPU-container stack (20b judge embed rerank)"
    echo "  all: cpu stack + 120b"
}

if [ $# -eq 0 ]; then
    usage; exit 1
fi

for t in "$@"; do
    case "$t" in
        cpu) for s in 20b judge embed rerank; do launch_one "$s"; done ;;
        all) for s in 20b judge embed rerank 120b; do launch_one "$s"; done ;;
        20b|120b|judge|embed|rerank) launch_one "$t" ;;
        *) echo "unknown target: $t (use: 20b 120b judge embed rerank cpu all)"; exit 1 ;;
    esac
done
