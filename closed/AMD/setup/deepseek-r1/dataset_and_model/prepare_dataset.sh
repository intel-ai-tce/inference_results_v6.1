#!/bin/bash

DOWNLOAD_DIR="/data/deepseek-r1"

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
-d ${DOWNLOAD_DIR} https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri
