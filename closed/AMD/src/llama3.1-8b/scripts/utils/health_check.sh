#!/bin/bash

FAIL_PATTERN="EngineCore failed to start|Memory access fault|torch.OutOfMemoryError"
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-""}

check_zombie_processes() {
    local defunct=$(ps aux | grep '[p]ython' | grep 'defunct')
    if [[ -n "$defunct" ]]; then
        if [ $found_zombie -eq 1 ]; then
            return 1
        fi
        found_zombie=1
    else
        found_zombie=0
    fi
    return 0
}

check_error_in_log() {
    local LOGFILE=$1
    if grep -E "$FAIL_PATTERN" "$LOGFILE" > /dev/null; then
        return 1
    fi
    return 0
}

get_model_timeout() {
    case "$1" in
    *llama3.1-405b*)
        echo "900"
        ;;
    *)
        echo "60"
        ;;
    esac
}

health_checker() {
    local PIPE=$1
    local LOGFILE=$2
    local MODEL_NAME=$3

    if [ $RUN_HEALTHCHECK -lt 1 ]; then
        echo "[health_check] Skip health-check"
        exit 1
    fi

    if [ ! -p $PIPE ]; then
        echo "[health_check] Pipe doesn't exist: $PIPE"
        exit 1
    fi

    if [[ -z $HEALTH_CHECK_TIMEOUT ]]; then
        # Allow to set different timeout for models
        HEALTH_CHECK_TIMEOUT=$(get_model_timeout $MODEL_NAME)
    fi

    # Check 2 subsequent iterations whether there are zombie processes
    found_zombie=0
    while true; do
        check_zombie_processes $found_zombie
        zombie_status=$?
        check_error_in_log $LOGFILE
        log_status=$?

        if [ $zombie_status -ne 0 ] || [ $log_status -ne 0 ]; then
            echo "[health_check] Terminate the run"
            # Kill python processes
            pkill python
            # Kill hanging vllm processes
            ps aux | grep -i "vllm" | grep -v grep | awk '{print $2}' | xargs kill -9
            echo 1 > $PIPE
            break
        fi
        sleep $HEALTH_CHECK_TIMEOUT
    done
}
