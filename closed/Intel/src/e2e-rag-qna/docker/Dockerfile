# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG BASE_IMAGE=vllm/vllm-openai-cpu
ARG BASE_TAG=v0.23.0-x86_64
FROM ${BASE_IMAGE}:${BASE_TAG} AS dev-base

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    apt-get install -y --no-install-recommends \
                    vim \
                    sox \
                    dmidecode \
                    pciutils && \
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --upgrade --python /opt/venv/bin/python \
                   librosa \
                   pandas \
                   sox \
                   more-itertools \
                   py-libnuma \
                   jaraco.context \
                   setuptools \
                   wheel

RUN uv venv /opt/calibration-venv --system-site-packages && \
    uv pip install --python /opt/calibration-venv/bin/python \
                   llmcompressor==0.6.0 \
                   huggingface_hub
ENV CALIBRATION_PYTHON=/opt/calibration-venv/bin/python

# Pin the inference checkout so Docker can reuse this layer.
WORKDIR /opt
RUN git clone https://github.com/mlcommons/inference.git mlperf-inference && \
    cd mlperf-inference/loadgen && \
    uv pip install --python /opt/venv/bin/python .

RUN uv pip install --python /opt/venv/bin/python --no-deps openai-whisper==20250625

RUN test -f /opt/mlperf-inference/compliance/TEST01/run_verification.py && \
    test -f /opt/mlperf-inference/compliance/TEST01/whisper/audit.config && \
    python /opt/mlperf-inference/tools/submission/truncate_accuracy_log.py --help >/dev/null && \
    python /opt/mlperf-inference/tools/submission/submission_checker/main.py --help >/dev/null
 
COPY ./ /workspace/
WORKDIR /workspace

RUN ln -s /opt/mlperf-inference /workspace/third_party/mlperf-inference && \
    cp /opt/mlperf-inference/loadgen/mlperf.conf /workspace/

RUN uv cache clean && \
    rm -rf /workspace/docker

ENV PYTHONPATH=/workspace/third_party/mlperf-inference/tools/submission
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so
ENTRYPOINT []
CMD ["/bin/bash"]
