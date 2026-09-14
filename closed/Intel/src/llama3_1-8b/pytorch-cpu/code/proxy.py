# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import logging
import os
import random
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Running total of prompt length per prefill for least-loaded routing
    app.state.prefill_prompt_len = [0] * len(app.state.prefill_clients)

    # Static mapping: prefill index -> decode index
    m = len(app.state.prefill_clients)
    n = len(app.state.decode_clients)
    app.state.prefill_to_decode = {}
    if n > 0:
        chunk_size = m // n
        for decode_idx in range(n):
            for i in range(chunk_size):
                app.state.prefill_to_decode[decode_idx * chunk_size + i] = decode_idx

    app.state.prefill_errors = 0
    app.state.decode_errors = 0
    app.state.requests_completed = 0

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )
    print(f"Prefill-to-decode mapping: {app.state.prefill_to_decode}")

    yield

    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8192)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default="localhost",
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default="localhost"
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    prefill = [args.prefiller_hosts] * len(args.prefiller_ports)
    decode = [args.decoder_hosts] * len(args.decoder_ports)

    args.prefiller_instances = list(zip(prefill, args.prefiller_ports))
    args.decoder_instances = list(zip(decode, args.decoder_ports))

    return args


def get_least_loaded_prefill(app, prompt_len: int):
    prefill_idx = min(
        range(len(app.state.prefill_clients)),
        key=lambda i: app.state.prefill_prompt_len[i],
    )
    app.state.prefill_prompt_len[prefill_idx] += prompt_len
    return app.state.prefill_clients[prefill_idx], prefill_idx


def get_decode_client_for_prefill(app, prefill_idx: int):
    decode_idx = app.state.prefill_to_decode[prefill_idx]
    return app.state.decode_clients[decode_idx]


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()
    await response.aread()

    req_data["min_tokens"] = min_tokens
    req_data["min_completion_tokens"] = min_completion_tokens

    return response


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


def _get_prompt_len(req_data: dict) -> int:
    if "prompt" in req_data:
        prompt = req_data["prompt"]
        if isinstance(prompt, str):
            return len(prompt)
        elif isinstance(prompt, list):
            if len(prompt) > 0 and isinstance(prompt[0], int):
                return len(prompt)
            return sum(len(p) for p in prompt if isinstance(p, str))
    elif "messages" in req_data:
        total = 0
        for msg in req_data["messages"]:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += len(part["text"])
        return total
    return 0


async def _handle_completions(api: str, request: Request):
    req_data = await request.json()
    request_id = f"{random.getrandbits(64):016x}"

    prompt_len = _get_prompt_len(req_data)

    prefill_client_info, prefill_idx = get_least_loaded_prefill(
        request.app, prompt_len
    )

    try:
        response = await send_request_to_service(
            prefill_client_info, api, req_data, request_id
        )
    except Exception as e:
        request.app.state.prefill_errors += 1
        request.app.state.prefill_prompt_len[prefill_idx] -= prompt_len
        raise

    request.app.state.prefill_prompt_len[prefill_idx] -= prompt_len

    response_json = response.json()
    await response.aclose()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    # Extract first token from prefill response to stream immediately
    first_token_chunk = None
    choices = response_json.get("choices", [])
    if choices:
        choice = choices[0]
        first_token_data = {"choices": [{"token_ids": choice.get("token_ids", [])}]}
        first_token_chunk = f"data: {json.dumps(first_token_data)}\n\n".encode()

    decode_client_info = get_decode_client_for_prefill(request.app, prefill_idx)

    async def generate_stream():
        # Yield prefill's first token immediately — client gets TTFT before decode starts
        if first_token_chunk:
            yield first_token_chunk

        try:
            async for chunk in stream_service_response(
                decode_client_info, api, req_data, request_id=request_id
            ):
                yield chunk
            request.app.state.requests_completed += 1
        except Exception as e:
            request.app.state.decode_errors += 1

    return StreamingResponse(generate_stream(), media_type="application/json")


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
        "requests_completed": app.state.requests_completed,
        "prefill_errors": app.state.prefill_errors,
        "decode_errors": app.state.decode_errors,
    }


if __name__ == "__main__":
    import resource

    import uvicorn

    global_args = parse_args()

    # Raise file descriptor limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = max(65536, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except ValueError:
        pass

    uvicorn.run(
        app,
        host=global_args.host,
        port=global_args.port,
        backlog=16384,
        timeout_keep_alive=5,
        loop="uvloop",
        http="httptools",
    )
