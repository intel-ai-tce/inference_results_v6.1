from vllm import LLM, SamplingParams
import multiprocessing as mp
import os

env_config= {
    # ENV VARIABLES
}

llm_config = {
    # MODEL PARAMETERS
}

sampling_params_config = {
    # SAMPLING PARAMETERS
}

def test():
    for env, val in env_config.items():
        os.environ[env] = str(val)
    llm = LLM(**llm_config)

    prompts = ["Some test"]
    response = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(**sampling_params_config),
        use_tqdm=False,
    )

    print(response[0].outputs[0].text)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    test()
