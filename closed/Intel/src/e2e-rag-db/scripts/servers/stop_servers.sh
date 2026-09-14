#!/bin/bash
# Stop unattended (nohup'd) servers started by launch_server_*.sh.
#
# Usage:
#   bash stop_servers.sh <target> [target ...]   (no args -> prints usage)
# Targets: 20b 120b judge embed rerank all

# Match a server to its process-name pattern.
pattern_for() {
    case "$1" in
        20b)    echo "vllm serve .*gpt-oss-20b|vllm serve .*/gpt-oss-20b" ;;
        120b)   echo "vllm.entrypoints.openai.api_server.*gpt-oss-120b" ;;
        judge)  echo "vllm serve .*Llama-3.1-8B|vllm serve .*[Ll]lama|vllm.entrypoints.openai.api_server.*Llama-3.1-8B|vllm.entrypoints.openai.api_server.*[Ll]lama" ;;
        embed)  echo "servers.embed_search_server|embed_search_server.py" ;;
        rerank) echo "servers.rerank_server|rerank_server.py" ;;
        *)      echo "" ;;
    esac
}

# Kill every PID matching $1 plus all their descendants; TERM then KILL.
kill_tree() {
    local pat="$1" label="$2"
    local roots; roots=$(pgrep -f "$pat" 2>/dev/null)
    if [ -z "$roots" ]; then
        echo "[$label] nothing running"
        return 0
    fi
    # collect roots + descendants
    local all=""
    for p in $roots; do
        all="$all $p $(pgrep -P "$p" 2>/dev/null)"
        # grandchildren (pool workers spawn under the front-end)
        for c in $(pgrep -P "$p" 2>/dev/null); do
            all="$all $(pgrep -P "$c" 2>/dev/null)"
        done
    done
    all=$(echo "$all" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | grep -vx "$$")
    echo "[$label] killing: $(echo $all | tr '\n' ' ')"
    kill $all 2>/dev/null
    sleep 3
    # re-scan and SIGKILL survivors (workers can ignore TERM)
    local survivors; survivors=$(pgrep -f "$pat" 2>/dev/null)
    if [ -n "$survivors" ]; then
        for p in $survivors; do
            kill -9 "$p" $(pgrep -P "$p" 2>/dev/null) 2>/dev/null
        done
        sleep 1
    fi
    pgrep -f "$pat" >/dev/null 2>&1 && echo "[$label] WARNING: still alive (perms? try as root)" \
                                     || echo "[$label] stopped"
}

usage() {
    echo "Usage: bash stop_servers.sh <target> [target ...]"
    echo "Targets: 20b 120b judge embed rerank all"
}

if [ $# -eq 0 ]; then
    usage; exit 1
fi

for t in "$@"; do
    case "$t" in
        all)      for s in 20b 120b judge embed rerank; do kill_tree "$(pattern_for $s)" "$s"; done ;;
        20b|120b|judge|embed|rerank) kill_tree "$(pattern_for $t)" "$t" ;;
        *) echo "unknown target: $t (use: 20b 120b judge embed rerank all)"; exit 1 ;;
    esac
done
