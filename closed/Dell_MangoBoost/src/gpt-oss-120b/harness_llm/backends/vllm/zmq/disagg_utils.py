from typing import Optional

from vllm.config import KVTransferConfig

if __name__ == "__main__":
    raise Exception("Cannot be run as an entry-point!")


DEFAULT_KV_BUFFER_SIZE = int(8e9)


def setup_kv_transfer_config(
    kv_connector: str,
    kv_role: str,
    kv_port: int,
    kv_rank: int,
    kv_buffer_size: Optional[float] = None,
) -> KVTransferConfig:
    # KVTransferConfig expects an integer number of bytes; YAML scientific
    # notation (e.g. "4.0e10") is parsed as float so we coerce here.
    effective_kv_buffer_size = (
        int(kv_buffer_size) if kv_buffer_size is not None else DEFAULT_KV_BUFFER_SIZE
    )

    kv_transfer_config_dict = {
        "ExampleConnector": KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_load_failure_policy="fail",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
            kv_role=kv_role,
            kv_rank=kv_rank,
        ),
        "P2pNcclConnector": KVTransferConfig(
            kv_connector="P2pNcclConnector",
            kv_load_failure_policy="fail",
            kv_role=kv_role,
            kv_rank=kv_rank,
            kv_port=kv_port,
            kv_buffer_size=effective_kv_buffer_size,
            kv_connector_extra_config={
                "send_type": "PUT",
                "nccl_num_channels": "16",
            },
        ),
    }

    if kv_connector not in kv_transfer_config_dict:
        supported_connectors = ", ".join(sorted(kv_transfer_config_dict))
        raise ValueError(
            f"Unsupported kv_connector '{kv_connector}'. "
            f"Supported connectors: {supported_connectors}"
        )
    return kv_transfer_config_dict[kv_connector]


def format_request_id(request_id: str, prefill_addr: str, decode_addr: str) -> str:
    return f"___prefill_addr_{prefill_addr}___decode_addr_{decode_addr}_{request_id}"


# ---------------------------------------------------------------------------
# SUT-driven disagg protocol (server and offline scenarios)
#
# The disagg path makes the SUT own all routing. Workers are scenario-agnostic
# engine runners; the SUT assigns each worker a globally unique kv_rank at
# registration time and pairs prefills/decodes itself. The helpers below are
# the single source of truth for the SUT<->worker wire protocol so both sides
# cannot drift.
# ---------------------------------------------------------------------------

# Role labels embedded as the prefix of each worker's ZMQ identity.
ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"

# Worker -> SUT control message tags (first element of a list payload).
TAG_REGISTER = "REGISTER"        # pre-engine registration request
TAG_KVADDR = "KVADDR"            # post-engine kv address report
TAG_PREFILL_DONE = "PREFILL_DONE"  # prefill finished; SUT may release to decode


def detect_local_address() -> str:
    """Resolve this host's IP exactly the way vLLM's P2pNcclConnector does.

    The connector advertises/binds ``zmq_address = f"{get_ip()}:{kv_port}"``,
    so the address the SUT embeds in the request_id markers must come from the
    same resolver or the KV rendezvous will not match. `get_ip()` honors
    ``VLLM_HOST_IP`` (vLLM's documented multi-NIC override) then falls back to
    a UDP-route probe.
    """
    from vllm.utils.network_utils import get_ip

    return get_ip()


def kv_port_for_rank(base_port: int, global_kv_rank: int) -> int:
    """KV listen port for a worker, derived from its SUT-assigned global rank.

    Layout (server-disagg): ``base + 0`` and ``base + 1`` are the SUT router
    ports; KV ports start at ``base + 2`` indexed by global kv_rank in
    ``[0, N_p + N_d)`` (prefill ranks first, then decode ranks).
    """
    return base_port + 2 + global_kv_rank


def make_register_msg(identity: bytes, role: str, local_address: str) -> list:
    return [TAG_REGISTER, identity, role, local_address]


def make_kvaddr_msg(identity: bytes, kv_addr: str) -> list:
    return [TAG_KVADDR, identity, kv_addr]


def make_prefill_done_msg(request_id: str) -> list:
    return [TAG_PREFILL_DONE, request_id]


def message_tag(msg) -> Optional[str]:
    """Return the control tag of a worker->SUT message, or None for data.

    Decode token outputs are ``[sample_id, token_ids_or_None]`` whose first
    element is an int/str sample id (never one of the tag constants), and the
    shutdown sentinel is ``None`` -- both yield ``None`` here.
    """
    if isinstance(msg, list) and msg and isinstance(msg[0], str) and msg[0] in (
        TAG_REGISTER,
        TAG_KVADDR,
        TAG_PREFILL_DONE,
    ):
        return msg[0]
    return None
