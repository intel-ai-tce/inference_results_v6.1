#!/bin/bash
set -u

if [ "$#" -ne 3 ]; then
    echo "Insufficient arguments: $0 <model> <gpu name> <scenario>"
    exit 1
fi

MODEL=$1
GPU_NAME=$2
SCENARIO=$3

# Llama2-70b - Offline
llama2-70b_offline() {
    if [[ "$GPU_NAME" == "mi325x" ]]; then
        sudo rocm-smi --setperfdeterminism 1700
        sudo amd-smi set --soc-pstate 0 -g all
    elif [[ "$GPU_NAME" == "mi300x" ]]; then
        sudo rocm-smi --setperfdeterminism 1400
        sudo amd-smi set --soc-pstate 1 -g all
    elif [[ "$GPU_NAME" == "mi355x" ]]; then
        echo "$0 no power setting found for: $GPU_NAME"
    else
        echo "$0 unknown GPU: $GPU_NAME"
    fi
}

# Llama2-70b - Server
llama2-70b_server() {
    if [[ "$GPU_NAME" == "mi325x" ]]; then
        sudo rocm-smi --setperfdeterminism 1600
        sudo amd-smi set --soc-pstate 0 -g all
    elif [[ "$GPU_NAME" == "mi300x" ]]; then
        sudo rocm-smi --setperfdeterminism 1300
        sudo amd-smi set --soc-pstate 1 -g all
    elif [[ "$GPU_NAME" == "mi355x" ]]; then
        echo "$0 no power setting found for: $GPU_NAME"
    else
        echo "$0 unknown GPU: $GPU_NAME"
    fi
}

# Llama2-70b
llama2-70b() {
    if [[ "$SCENARIO" == "offline" ]]; then
        llama2-70b_offline
    elif [[ "$SCENARIO" == "server" ]]; then
        llama2-70b_server
    elif [[ "$SCENARIO" == "interactive" ]]; then
        llama2-70b_interactive
    else
        echo "$0 unknown scenario: $SCENARIO"
    fi
}

# Llama2-70b - Interactive
llama2-70b_interactive() {
    if [[ "$GPU_NAME" == "mi325x" ]]; then
        sudo rocm-smi --setperfdeterminism 1700
        sudo amd-smi set --soc-pstate 0 -g all
    elif [[ "$GPU_NAME" == "mi300x" ]]; then
        sudo rocm-smi --setperfdeterminism 2000
        sudo amd-smi set --soc-pstate 0 -g all
    elif [[ "$GPU_NAME" == "mi355x" ]]; then
        echo "$0 no power setting found for: $GPU_NAME"
    else
        echo "$0 unknown GPU: $GPU_NAME"
    fi
}

# Mixtral 8x7b
mixtral-8x7b() {
    sudo rocm-smi --setperfdeterminism 1500
    sudo amd-smi set --soc-pstate 0 -g all
}

# Main
if [[ "$MODEL" == "llama2-70b" || "$MODEL" == "llama2-70b-99" || "$MODEL" == "llama2-70b-99.9" ]]; then
    llama2-70b
elif [[ "$MODEL" == "mixtral-8x7b" ]]; then
    mixtral-8x7b
elif [[ "$GPU_NAME" != "mi355x" ]]; then
    sudo rocm-smi --resetperfdeterminism
    sudo amd-smi set --soc-pstate 0 -g all
fi
