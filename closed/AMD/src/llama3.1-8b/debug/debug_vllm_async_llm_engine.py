from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs
import multiprocessing as mp
import os, asyncio
from transformers import AutoTokenizer

env_config= {
    # ENV VARIABLES
}

llm_config = {
    # MODEL PARAMETERS
}

sampling_params_config = {
    # SAMPLING PARAMETERS
}

llm = None

def create_engine():
    for env, val in env_config.items():
        os.environ[env] = str(val)
    
    engine_args = AsyncEngineArgs(**llm_config)
    global llm
    llm = AsyncLLMEngine.from_engine_args(
            engine_args = engine_args, 
            start_engine_loop = True
    ) 

async def test():
    tokenizer = AutoTokenizer.from_pretrained(llm_config['model'])
    prompt_token_ids = tokenizer.encode("Some test")
    response_generator = llm.generate(
        request_id ="1",
        prompt = {'prompt_token_ids': prompt_token_ids},
        sampling_params = SamplingParams(**sampling_params_config),
    )

    async for response in response_generator:
         final_response = response
    print(tokenizer.decode(final_response.outputs[0].token_ids))
    llm.shutdown_background_loop()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    create_engine()
    asyncio.new_event_loop().run_until_complete(test())
