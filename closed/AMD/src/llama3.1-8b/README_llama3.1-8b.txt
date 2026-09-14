Here is a README.txt file on how to run it:  (1) get the dataset according to the following directions:
 
cnn_eval.json dataset:
 
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
 
calibration dataset:
 
curl -OL https://raw.githubusercontent.com/mlcommons/inference/v4.0/calibration/CNNDailyMail/calibration-list.txt

uv run python download_cnndm.py --save-dir data --calibration-ids-file calibration-list.txt --split train
 
 
(2) download the model from huggingface:

amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ (hf download amd/Llama-3.1-8B-Instruct-MXFP4-W4A4-MLCAL-C1000-GPTQ --

local-dir=./)
 
 
(3) docker pull this docker: vllm/vllm-openai-rocm:v0.22.0
 
 
(4) start  the docker:

cd src/llama3.1-8b/setup
 
change dkrun.sh according to your dataset and model file locations.
 
cd ..
 
to start the docker, do the following:
 
bash setup/start.sh vllm/vllm-openai-rocm:v0.22.0 &
 
docker ps
 
docker exec -it containerID bash
 
cd /lab-mlperf-inference/code
 
to install loadgen related mlperf stuff, do the following:
 
bash build_docker_mlperf_minimal.sh
 
after this, do the following:

pip install matplotlib
 
(5) start the benchmark run
 
in run.sh, you can choose to run benchmarks for offline, server or interactive by choose the running line and commenting ou
t the other two, by default, it will run Offline for Performance.

Please note that we use SPX model for Offline with --config-name offline_mi355x

To set up DPX mode:
# Set compute partition to DPX (2 compute partitions per GPU)
sudo rocm-smi --setcomputepartition DPX --autorespond y

# Set memory partition to NPS2 (matches what the benchmark uses: NPS2, DPX)
sudo rocm-smi --setmemorypartition NPS2 --autorespond y

For Server, in run.sh, we need to use DPX mode with --config-name server_mi355x_dpx

python3 main.py --config-path llama3.1-8b --config-name server_mi355x_dpx --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_1_8b_server_performance

 For Interactive, in run.sh, we need to use DPX mode with --config-name interactive_mi355x_dpx
python3 main.py --config-path llama3.1-8b --config-name interactive_mi355x_dpx --backend vllm test_mode=performance harness_config.output_log_dir=results/llama3_1_8b_interactive_performance


To switch back to SPX mode from DPX mode:
Stop your docker
sudo rocm-smi --setmemorypartition NPS1 --autorespond y 
sudo rocm-smi --setcomputepartition SPX  --autorespond y 
 
(6) check the accuracy
 
in run.sh, change test_mode=accuracy

also, in corresponding yaml file (offline_mi355x.yaml, server_mi355x.yaml, interactive_mi355x.yaml), change test_mode: accu
racy
 
At the end of the run, look for mlperf_log_accuracy.json in code/results/llama3_1_8b_offline_accuracy
 
do the following:
 
cp llama3_1_8b_offline_accuracy ../../scripts/llama3.1-8b/
 
cd ../../
 
cd scripts/llama3.1-8b
 
bash run_accuracy.sh mlperf_log_accuracy.json >accuracy.txt
 
the accuracy results will be in this accuracy.txt file
 
(7)
Reference accuracy scores:
 
Reference score (Accuracy target of 99%):

{ 'rouge1’: 38.7792, 'rouge2': 15.9075, 'rougeL': 24.4957, 'rougeLsum': 35.793, 'gen_len': 8167644,gen_num': 13368}
endpoints/examples/05_Llama3.1-8B_Example at main · mlcommons/endpoints
MLCommons Inference Endpoints repository. Contribute to mlcommons/endpoints development by creating an account on GitHub.
 

 
