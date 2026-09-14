# MLPerf Inference v6.1 Implementations
This is a repository of Fujitsu Limited (Fsas Technologies) servers using optimized implementations for [MLPerf Inference Benchmark ](https://github.com/mlcommons/inference).

# Implementations
### Benchmarks
**Please refer to /closed/NVIDIA for detailed instructions for NVIDIA GPU & Triton submissions, including performace guides, and instructions on how to run with new systems.**


## Getting Started
First, specify the dataset location by setting the environment variable MLPERF_SCRATCH_PATH:

 `export MLPERF_SCRATCH_PATH=your_dataset_folder `

After launching the NVIDIA Docker container, execute the following installation command within the container:

 `pip install -e ".[llm]" `

## Launching the TRT-LLM Server
Once the installation is complete, you can launch the TRT-LLM server. Please select the appropriate mode for the --scenarios option, depending on whether you are conducting a "Server" or "Offline" measurement.

Launch Command:

 `make run_llm_server --benchmarks=llama2-70b --scenarios={Server|Offline} `

Note: After executing the command, please allow several minutes (up to 10 minutes) for the server to initialize. You can monitor the startup status by checking the logs inside the Docker container:

 `tail -f build/logs/<latest_date>/trtllm_serve_0.log `

# Running Benchmarks
After executing `export SUBMITTER=Fujitsu`, please run the appropriate command below according to the scenario you wish to measure.


### Offline Scenario
#### Run harness (Performance)
 `make run_harness --benchmarks=llama2-70b --scenarios=Offline --test_mode=PerformanceOnly `

#### Run harness (High accuracy test - 99.9% target)
 `make run_harness --benchmarks=llama2-70b --scenarios=Offline --accuracy_target=.999 --test_mode=AccuracyOnly `

### Server Scenario
#### Run harness (Performance)
 `make run_harness --benchmarks=llama2-70b --scenarios=Server --test_mode=PerformanceOnly `

#### Run harness (High accuracy test - 99.9% target)
 `make run_harness --benchmarks=llama2-70b --scenarios=Server --accuracy_target=.999 --test_mode=AccuracyOnly `

# Run Compliance Test
 `make run_audit_test06 RUN_ARGS="--benchmarks=llama2-70b --scenarios={Server|Offline} --test_mode={PerformanceOnly|AccuracyOnly}" `

## Important Note
We recommend using the make run_harness command for all executions. If you run the benchmarks directly via nv-mlpinf run_harness, the results will be overwritten with each execution because the system will not create a new date-stamped folder under the logs directory.
