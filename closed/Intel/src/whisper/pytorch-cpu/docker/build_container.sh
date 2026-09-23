#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-redhat/mlperf:mlperf-inference-6.1-whisper_cpu}"

echo "Building RHAI 3.5 Whisper CPU image:"
echo "  Dockerfile: docker/Dockerfile.redhat"
echo "  Image:      ${IMAGE_NAME}"

docker build \
    -f docker/Dockerfile.redhat \
    --build-arg http_proxy="${http_proxy:-}" \
    --build-arg HTTP_PROXY="${http_proxy:-}" \
    --build-arg https_proxy="${https_proxy:-}" \
    --build-arg HTTPS_PROXY="${https_proxy:-}" \
    --build-arg no_proxy="${no_proxy:-}" \
    --build-arg NO_PROXY="${no_proxy:-}" \
    -t "${IMAGE_NAME}" .
