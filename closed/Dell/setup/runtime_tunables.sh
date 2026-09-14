#!/bin/bash

SUCCESS_COUNT=0
TOTAL_COMMANDS=8

STATUS_CODE=0

execute_and_validate() {
    local command="$1"
    local description="$2"
    
    echo "Executing: $description"
    if eval "$command"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ Success: $description"
    else
        local exit_code=$?
        echo "✗ Failed: $description"
        echo "Exit code: $exit_code"
        STATUS_CODE=1
    fi
}

execute_and_validate "echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null" "Drop caches"
execute_and_validate "sudo cpupower idle-set -d 2" "Set CPU idle state"
execute_and_validate "sudo cpupower frequency-set -g performance" "Set CPU frequency governor"
execute_and_validate "echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog > /dev/null" "Disable NMI watchdog"
execute_and_validate "echo 0 | sudo tee /proc/sys/kernel/numa_balancing > /dev/null" "Disable NUMA balancing"
execute_and_validate "echo 0 | sudo tee /proc/sys/kernel/randomize_va_space > /dev/null" "Disable ASLR"
execute_and_validate "echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null" "Enable transparent hugepages"
execute_and_validate "echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/defrag > /dev/null" "Set hugepage defrag"

echo "========================================="
echo "Completed: $SUCCESS_COUNT/$TOTAL_COMMANDS commands"
echo "========================================="

exit $STATUS_CODE