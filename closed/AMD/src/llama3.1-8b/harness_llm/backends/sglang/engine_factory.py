import sglang
from sglang.srt.server_args import ServerArgs
from harness_llm.common.container_utils import remove_none_from_dict

def create_from(llm_config: dict):
    server_args = ServerArgs(**remove_none_from_dict(llm_config))
    return sglang.Engine(server_args=server_args)