#!/usr/bin/env bash

# Generate one vLLM OpenMP CPU binding group per NUMA node. Keep only one
# logical CPU for each physical core and leave the requested number of cores
# free on every node for vLLM/runtime housekeeping.
set -euo pipefail

reserved_per_numa="${1:-2}"
if ! [[ "${reserved_per_numa}" =~ ^[0-9]+$ ]]; then
    echo "reserved cores per NUMA node must be a non-negative integer" >&2
    exit 2
fi

lscpu -p=CPU,CORE,NODE,ONLINE | awk -F, -v reserve="${reserved_per_numa}" '
    /^#/ { next }
    $4 == "Y" && $3 >= 0 {
        key = $3 ":" $2
        if (!(key in seen_core)) {
            seen_core[key] = 1
            count[$3]++
            cpu[$3, count[$3]] = $1 + 0
            if ($3 > max_node) {
                max_node = $3
            }
        }
    }

    function append_range(group, first, last) {
        if (group != "") {
            group = group ","
        }
        if (first == last) {
            return group first
        }
        return group first "-" last
    }

    END {
        rank_separator = ""
        for (node = 0; node <= max_node; node++) {
            if (!(node in count)) {
                continue
            }

            usable = count[node] - reserve
            if (usable <= 0) {
                printf "NUMA node %d has %d physical cores; cannot reserve %d\n", \
                    node, count[node], reserve > "/dev/stderr"
                exit 2
            }

            group = ""
            range_start = cpu[node, 1]
            previous = range_start
            for (cpu_index = 2; cpu_index <= usable; cpu_index++) {
                current = cpu[node, cpu_index]
                if (current == previous + 1) {
                    previous = current
                    continue
                }
                group = append_range(group, range_start, previous)
                range_start = current
                previous = current
            }
            group = append_range(group, range_start, previous)

            printf "%s%s", rank_separator, group
            rank_separator = "|"
        }
        print ""
    }
'
