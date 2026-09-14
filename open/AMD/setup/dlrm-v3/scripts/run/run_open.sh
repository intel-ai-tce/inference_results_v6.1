#!/usr/bin/env bash
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-}"
case "$PROFILE" in
  Server|ServerAccuracy|Offline|OfflineAccuracy) ;;
  *) echo "usage: bash $0 {Server|ServerAccuracy|Offline|OfflineAccuracy}" >&2; exit 64 ;;
esac

export WINDOW=1
export MAX_ATTN_LEN=1024
export BATCH=64
export CONF=user_mi355x8_nve_b64_qps16200_PROD10min_C1on_v61.conf
export FUSE_EPILOGUE=1
export INFLIGHT=32
export LASTLAYER_TARGETS_ONLY=0
export ATTN_FASTMASK=0
export ATTN_FULLGRID=0
export ATTN_OCCTUNE=0
export GATE_POLY=1
export GATE_POLY_DEG=5

case "$PROFILE" in
  Server)
    export SCENARIO=Server MODE=performance
    export TAG="${TAG:-mi355x8_C1on_v61seed_b64_q16200_SERVER10min}"
    export WAIT="${WAIT:-2400}"
    exec bash "${SELF}/run_gold.sh"
    ;;
  ServerAccuracy)
    export SCENARIO=Server
    export TAG="${TAG:-mi355x8_C1on_v61seed_b64_SERVER_accuracy}"
    export WAIT="${WAIT:-7200}"
    exec bash "${SELF}/run_accuracy.sh"
    ;;
  Offline)
    export SCENARIO=Offline MODE=performance
    export TAG="${TAG:-mi355x8_C1on_v61seed_b64_q17000_OFFLINE10min}"
    export WAIT="${WAIT:-2400}"
    exec bash "${SELF}/run_gold.sh"
    ;;
  OfflineAccuracy)
    export SCENARIO=Offline
    export TAG="${TAG:-mi355x8_C1on_v61seed_b64_OFFLINE_accuracy}"
    export WAIT="${WAIT:-7200}"
    exec bash "${SELF}/run_accuracy.sh"
    ;;
esac
