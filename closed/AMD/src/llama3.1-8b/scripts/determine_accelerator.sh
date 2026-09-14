#!/bin/bash
set -u

ACCELERATOR=""
if command -v rocminfo &> /dev/null; then
    GFX=$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')
    POWER=$(rocm-smi --showmaxpower | grep "(W)" | awk 'NR==1' | awk '{print $8}')
    case "$GFX-$POWER" in
        gfx942-750.0) ACCELERATOR="MI300X" ;;
        gfx942-1000.0) ACCELERATOR="MI325X" ;;
        gfx950-1000.0) ACCELERATOR="MI350X" ;;
        gfx950-1400.0) ACCELERATOR="MI355X" ;;
    esac
elif command -v nvidia-smi &> /dev/null; then
    ACCELERATOR=$(nvidia-smi --query-gpu=name --format=csv | awk 'NR==2' | awk '{print $2}')
fi

if [ -z $ACCELERATOR ]; then
    echo "Can't determine accelerator!"
    exit 1
fi

echo "$ACCELERATOR" | tr '[:upper:]' '[:lower:]'
