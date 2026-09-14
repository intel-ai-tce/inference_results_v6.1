#!/bin/bash
# Pre-stage ALL VBench scorer model weights for wan22 accuracy runs (offline).
#
# The mlcommons endpoints `-vbench` image ships the VBench venv but NONE of the 6
# MLPerf wan22 dimension weights (verified by inspecting the published image; its
# /opt/vbench_cache is an empty dir). VBench would otherwise download them at
# evaluation time, which FAILS on compute nodes that run offline (HF_HUB_OFFLINE=1,
# no wget/unzip in the client image). Run this once on a login node (needs
# curl/wget + unzip + git + network) so every dimension scores offline.
#
# This host CACHE_DIR (default build/vbench_cache) sits under WORK_DIR, which is
# mounted at /work in-container -- so it is reachable at /work/build/vbench_cache
# with no extra bind-mount (perf-only / audit runs that never stage it are
# unaffected). The accuracy template (wan22_videogen_endpoints_submission.yaml)
# exports:
#   VBENCH_CACHE_DIR=/work/build/vbench_cache          (amt/raft/caption)
#   TORCH_HOME=/work/build/vbench_cache/torch_home     (DINO torch.hub cache)
# and endpoint_env_vbench.yaml exports:
#   HOME=/work/build/vbench_cache/home                 (CLIP -> $HOME/.cache/clip)
# This script populates all three subtrees so those paths resolve offline.
#
# Usage: prepare_vbench_cache.sh [CACHE_DIR]   (default: build/vbench_cache)
set -euo pipefail

CACHE_DIR=${1:-build/vbench_cache}
mkdir -p "$CACHE_DIR"/{amt_model,raft_model,caption_model,torch_home/hub/checkpoints,home/.cache/clip}
cd "$CACHE_DIR"

dl() {  # dl <url> <dest-file>
    [ -f "$2" ] && { echo "  have $2"; return; }
    if command -v curl >/dev/null; then curl -fsSL -o "$2" "$1"
    else wget -q -O "$2" "$1"; fi
}

echo "== 3 wget-based dims (motion_smoothness, dynamic_degree, scene) =="
# motion_smoothness — AMT-S checkpoint
dl https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth amt_model/amt-s.pth
# dynamic_degree — RAFT models (VBench expects raft_model/models/raft-things.pth)
if [ ! -f raft_model/models/raft-things.pth ]; then
    dl https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip raft_model/models.zip
    unzip -q -o -d raft_model/ raft_model/models.zip && rm -f raft_model/models.zip
fi
# scene — Tag2Text checkpoint
dl https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth \
   caption_model/tag2text_swin_14m.pth

echo "== subject_consistency — DINO (torch.hub github checkout + weights) =="
# VBench (local=False) calls torch.hub.load('facebookresearch/dino:main',
# 'dino_vitb16', source='github'); with the repo pre-cloned under
# $TORCH_HOME/hub and validation bypassed by vbench_runner, it loads offline.
if [ ! -d torch_home/hub/facebookresearch_dino_main ]; then
    git clone --depth 1 https://github.com/facebookresearch/dino \
        torch_home/hub/facebookresearch_dino_main
fi
dl https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth \
   torch_home/hub/checkpoints/dino_vitbase16_pretrain.pth

echo "== background_consistency + appearance_style — CLIP ViT-B/32 =="
# VBench (local=False) calls clip.load('ViT-B/32'), which resolves to
# $HOME/.cache/clip/ViT-B-32.pt. sha256 is verified by the clip loader.
dl "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
   home/.cache/clip/ViT-B-32.pt

echo "VBench cache ready at $(pwd):"
du -sh amt_model raft_model caption_model torch_home home 2>/dev/null || true
