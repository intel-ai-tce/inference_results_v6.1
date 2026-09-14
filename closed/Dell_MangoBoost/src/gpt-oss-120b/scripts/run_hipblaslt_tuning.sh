#!/bin/bash

set -xeu

INPUT_FILE=${INPUT_FILE:-hipblaslt.log}
ITERS=${ITERS:-1000}
COLD_ITERS=${COLD_ITERS:-1000}
ROTATING=${ROTATING:-512}
ALGO_METHOD=${ALGO_METHOD:-all}
N=${N:-0}
TUNING_FILE=${TUNING_FILE:-tuning}

START_TIME=$(date +%m%d-%H%M%S)
LOG_FILE_NAME=${TUNING_FILE}_${START_TIME}
LOG_DIR=/lab-mlperf-inference/code/hipblaslt_tuning
mkdir -p ${LOG_DIR}
export HIPBLASLT_TUNING_FILE=${LOG_DIR}/${LOG_FILE_NAME}.txt

remove_leading_number() {
    local line="$1"
    echo "$line" | sed 's/^[[:space:]]*[0-9]\+ //'
}

remove_algo_method() {
    local line="$1"
    echo "$line" | sed 's/--algo_method index//'
}

remove_solution_index() {
    local line="$1"
    echo "$line" | sed 's/--solution_index \([0-9]\+\).*//'
}

add_executable_prefix_and_bin_path() {
    local line="$1"
    echo "$line" | sed 's|^|./bin/|'
}

change_n() {
    local line="$1"
    local n_value="$2"
    echo "$line" | sed 's/-n [0-9]\+/-n '${n_value}' /'
}

clean_and_prepare_command() {
    local line="$1"
    line=$(remove_leading_number "$line")
    line=$(remove_algo_method "$line")
    line=$(remove_solution_index "$line")
    line=$(add_executable_prefix_and_bin_path "$line")
    echo "$line"
}

add_iters() {
    local line="$1"
    echo "$line" | sed 's|$| -i '${ITERS}'|' 
}

add_cold_iters() {
    local line="$1"
    echo "$line" | sed 's|$| -j '${COLD_ITERS}'|' 
}

add_rotating() {
    local line="$1"
    echo "$line" | sed 's|$| --rotating '${ROTATING}'|'
}

add_algo_method() {
    local line="$1"
    echo "$line" | sed 's|$| --algo_method '${ALGO_METHOD}'|'
}

add_parameters_to_command() {
    local line="$1"
    line=$(add_iters "$line")
    line=$(add_cold_iters "$line")
    line=$(add_rotating "$line")
    line=$(add_algo_method "$line")
    echo "$line"
}

if [[ ${N} -eq 0 ]]; then

    while IFS= read -r line; do
        command=$line
        command=$(clean_and_prepare_command "$command")
        command=$(add_parameters_to_command "$command")

        eval $command
    done < "$INPUT_FILE"

else

    while IFS= read -r line; do
        for ((i = 1; i <= ${N}; i++)); do
            command=$line
            command=$(clean_and_prepare_command "$command")
            command=$(change_n "$command" ${i})
            command=$(add_parameters_to_command "$command")

            eval $command
        done    
    done < "$INPUT_FILE"

fi
