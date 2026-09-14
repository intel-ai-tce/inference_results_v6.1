import os
import sys
import glob
import hashlib
from pathlib import Path

data = {
    "llama2-70b": {
        "model_path" : ["/model/llama2-70b-chat-hf/fp4_quantized", "/model/llama2-70b-chat-hf/fp4_quantized_gptq", "/model/llama2-70b-chat-hf/fp8_quantized"],
        "safetensor_md5sum" : ["4d0cbae6b8bccbdfc8fdd73aa2829917", "332a949d2829dd1b96fb2c162075c993", "b7aa0389b68d204fe6fca1c6a600db61", "7c50bf7529f3fdbc308576871e66571c", "797d4235f78a00032db48219ae2bf5ee"],
        "dataset_path" : "/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl",
        "dataset_md5sum" : "5fe8be0a7ce5c3c9a028674fd24b00d5",
    },
    "llama2-70b-interactive": {
        "model_path" : ["/model/llama2-70b-chat-hf/fp4_quantized", "/model/llama2-70b-chat-hf/fp4_quantized_gptq", "/model/llama2-70b-chat-hf/fp8_quantized"],
        "safetensor_md5sum" : ["4d0cbae6b8bccbdfc8fdd73aa2829917", "332a949d2829dd1b96fb2c162075c993", "b7aa0389b68d204fe6fca1c6a600db61", "7c50bf7529f3fdbc308576871e66571c", "797d4235f78a00032db48219ae2bf5ee"],
        "dataset_path" : "/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl",
        "dataset_md5sum" : "5fe8be0a7ce5c3c9a028674fd24b00d5",
    },
    "llama3_1-405b": {
        "model_path" : ["/model/llama3.1-405b/fp4_quantized", "/model/llama3.1-405b/fp8_quantized", "/model/llama3.1-405b/fp4_quantized/pruned_59_84"],
        "safetensor_md5sum" : ["7dccd6047bbf888c31bce62f588b9dbb", "9cf165d1e925964ff9936902d6db3613", "fd1d753bf4e8a98ce6cc7e39a9d49295"],
        "dataset_path" : "/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl",
        "dataset_md5sum" : "25d9f2085bbcacb07554cd7f4af303c9",
    },
    "llama3_1-405b-interactive": {
        "model_path" : ["/model/llama3.1-405b/fp4_quantized", "/model/llama3.1-405b/fp8_quantized", "/model/llama3.1-405b/fp4_quantized/pruned_59_84"],
        "safetensor_md5sum" : ["7dccd6047bbf888c31bce62f588b9dbb", "9cf165d1e925964ff9936902d6db3613", "fd1d753bf4e8a98ce6cc7e39a9d49295"],
        "dataset_path" : "/data/llama3.1-405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl",
        "dataset_md5sum" : "25d9f2085bbcacb07554cd7f4af303c9",
    },
    "deepseek-r1": {
        "model_path" : ["/model/deepseek-r1/orig", "/model/deepseek-r1/fp4_quantized"],
        "safetensor_md5sum" : ["694a092b87b7547fc1db086e1ad89518", "746616546d8cb2db0a39d57dee3c46af"],
        "dataset_path" : "/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl",
        "dataset_md5sum" : "994a30e4787831b8c8f6e44da7de67fe",
    },
    "gpt-oss-120b": {
        "model_path" : ["/model/gpt-oss-120b", "/model/gpt-oss-120b/orig", "/model/gpt-oss-120b/fp4_quantized", "/model/gpt-oss-120b/original", "/model/original/gpt-oss-120b"],
        "safetensor_md5sum" : ["302c08bef24e406ac5bae73e768dc205", "7348601da33911a5ca9ded5eb59f2f4c", "6755d7dada79aeaddfc207652cb62be8", "01020c7ecfc9cec468f7e82038e6aa61"],
        "dataset_path" : ["/data/gptoss-dataset/perf/perf_eval_ref.parquet", "/data/gpt-oss-120b/perf_eval_ref.parquet", "/data/gpt-oss-120b/acc_eval_ref.parquet", "/data/gptoss-dataset/acc/acc_eval_ref.parquet" ,"/data/gpt-oss-120b/acc_eval_compliance_gpqa.parquet"],
        "dataset_md5sum" : ["e4cd6cef6dd975f3e50c85b3279b358b", "29d35424fc8d31461e73a7766446480e", "e35e6bc6d7343df1930501bb61e61260"],
    },
}


class ResourceChecker:


    def check(self, configs):
        result = []
        benchmark_name = configs["benchmark_name"]
        scenario = configs["scenario"]
        benchmark = data[benchmark_name]
        if isinstance(benchmark.get(scenario), dict):
            benchmark = benchmark.get(scenario)
        output_log_dir = configs["harness_config"]["output_log_dir"]

        invalid_file_path = output_log_dir + "/INVALID"
        if os.path.exists(invalid_file_path):
            os.remove(invalid_file_path)

        #validate model
        current_model_path = configs["llm_config"].get("model", configs["llm_config"].get("model_path", None))
        expected_model_paths = benchmark["model_path"]
        expected_model_md5sums = benchmark["safetensor_md5sum"]
        if current_model_path not in expected_model_paths:
            result.append("The model path is NOT matching with the default settings \n"
                          f"  default: {', '.join(expected_model_paths)} \n"
                          f"  current: {current_model_path}")
        else:
            folder_path = Path(current_model_path)
            if folder_path.exists() and folder_path.is_dir():
                matching_files = glob.glob(f"{folder_path}/model-00001*.safetensors")
                if matching_files:
                    md5sum = self.get_md5sum(matching_files[0])
                    if md5sum not in expected_model_md5sums:
                        result.append("The model's MD5 checksum does not match the expected value \n"
                                      f"  expected: {', '.join(expected_model_md5sums)} \n"
                                      f"  current:  {md5sum}")

        #validate dataset
        current_dataset_path = configs["harness_config"]["dataset_path"]
        expected_dataset_paths = benchmark["dataset_path"]
        expected_dataset_md5sums = benchmark["dataset_md5sum"]
        if current_dataset_path not in expected_dataset_paths:
            result.append("The dataset path is NOT matching with the default settings \n"
                          f"  default: {', '.join(expected_dataset_paths)} \n"
                          f"  current: {current_dataset_path}")
        else:
            dataset = Path(current_dataset_path)
            if dataset.exists() and dataset.is_file():
                md5sum = self.get_md5sum(dataset)
                if md5sum not in expected_dataset_md5sums:
                    result.append("The dataset's MD5 checksum does not match the expected value \n"
                                    f"  expected: {', '.join(expected_model_md5sums)} \n"
                                    f"  current:  {md5sum}")

        if result:
            print("Resource validation error:", file=sys.stderr)

            for value in result:
                print(value + "\n", file=sys.stderr)

            with open(invalid_file_path, "w") as file:
                for value in result:
                    file.write(value + "\n")

            if configs["harness_config"]["resource_checker_abort_on_failure"]:
                sys.exit(1)

    def get_md5sum(self, file):
        hash_md5 = hashlib.md5()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()
