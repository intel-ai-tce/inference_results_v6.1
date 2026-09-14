#!/bin/bash
# Shared health-check helper for launch_server_*.sh. Source this file, set
# PORT, then call health_check_v1.

function health_check_v1() {
    echo "Waiting for servers to start and checking health..."
    local no_proxy_bak="$no_proxy" NO_PROXY_bak="$NO_PROXY"
    export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
    for port in "${PORT}"; do
        echo "Checking server at: $port"
        RETRY_COUNT=0
        MAX_RETRIES=100
        while [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${port}/v1/models 2>/dev/null)" != "200" ]; do
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -gt $MAX_RETRIES ]; then
                echo "Server failed to start"
                exit 1
            fi
            sleep 5
        done
        echo "Server ready"
    done
    export no_proxy="$no_proxy_bak" NO_PROXY="$NO_PROXY_bak"
}
