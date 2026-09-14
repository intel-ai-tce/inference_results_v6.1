apt update \
&& apt install -y \
   libfmt-dev \
   libsqlite3-dev \
   numactl \
   sqlite3 \
   zip \
   nano \
&& rm -rf /var/lib/apt/lists/*
pip install \
   absl-py==2.1.0 \
   datasets==2.20.0 \
   nltk==3.8.1 \
   numpy==1.26.4 \
   py-libnuma==1.2 \
   rouge_score==0.1.2 \
   omegaconf==2.3.0 \
   hydra-core==1.3.2 \
   optuna==4.1.0
cd /app
git clone https://github.com/mlcommons/inference.git mlperf_inference \
&& cd mlperf_inference/loadgen \
&& git checkout 76f61013a987800cad246eca315e3f0405b67d9e \
&& git submodule update --init --recursive \
&& CFLAGS="-std=c++14 -O3" python -m pip install .
cd /app
git clone https://github.com/ROCm/rocm_bandwidth_test --depth 1 rocm_bandwidth_test \
&& cd rocm_bandwidth_test \
&& mkdir build \
&& cd build \
&& cmake -DCMAKE_MODULE_PATH="/app/rocm_bandwidth_test/cmake_modules" -DCMAKE_PREFIX_PATH="/opt/rocm/" .. \
&& make \
&& make install
# RPD
cd /app
git clone https://github.com/ROCm/rocmprofiledata --depth 1 rocm_profile_data \
&& cd rocm_profile_data \
&& make \
&& make install
