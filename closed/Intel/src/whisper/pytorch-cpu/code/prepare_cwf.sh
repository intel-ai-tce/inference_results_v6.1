pip uninstall -y vllm
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

git clone -b feat/avx2_w8a8 https://github.com/tianmu-li/vllm.git vllm_local
cd vllm_local
pip install -r requirements/cpu.txt
VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel
cd /workspace
pip install vllm_local/dist/*.whl
