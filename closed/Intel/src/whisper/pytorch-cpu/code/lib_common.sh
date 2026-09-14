#!/bin/bash

# resolve_host_dir VAR_NAME SUBDIR
# Returns the value of VAR_NAME if set; otherwise WORKLOAD_DIR/SUBDIR if
# WORKLOAD_DIR is set; otherwise PWD/SUBDIR.
resolve_host_dir() {
    local var_name="$1"
    local subdir="$2"

    if [ "${!var_name+x}" = "x" ]; then
        printf '%s\n' "${!var_name}"
    elif [ -n "${WORKLOAD_DIR:-}" ]; then
        printf '%s/%s\n' "${WORKLOAD_DIR}" "${subdir}"
    else
        printf '%s/%s\n' "${PWD}" "${subdir}"
    fi
}
