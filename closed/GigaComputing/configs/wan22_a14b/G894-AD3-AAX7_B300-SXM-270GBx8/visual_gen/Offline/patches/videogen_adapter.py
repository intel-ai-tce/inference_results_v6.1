# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter for trtllm-serve's POST /v1/videos/generations endpoint."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inference_endpoint.core.types import (
    Query,
    QueryResult,
    StreamChunk,
    TextModelOutput,
)
from inference_endpoint.dataset_manager.transforms import ColumnFilter
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from .types import VideoPathRequest, VideoPathResponse, VideoPayloadResponse

_BINARY_FALLBACK_ENV = "INFERENCE_ENDPOINT_VIDEOGEN_FALLBACK_DIR"

if TYPE_CHECKING:
    from inference_endpoint.config.schema import ModelParams
    from inference_endpoint.dataset_manager.transforms import Transform


class VideoGenAdapter(HttpRequestAdapter):
    """Adapter for trtllm-serve POST /v1/videos/generations.

    `response_format` is read from `query.data` (default "video_path") and
    is *not* derived from BenchmarkConfig.benchmark_mode. Callers that want
    accuracy-mode bytes must inject `response_format="video_bytes"` into the
    dataset rows — typically via an `AddStaticColumns` transform.
    """

    @classmethod
    def dataset_transforms(cls, model_params: "ModelParams") -> "list[Transform]":
        # ColumnFilter rejects unknown columns at dataset-load time so typos
        # (e.g. "negitive_prompt") fail loud instead of silently falling back
        # to server-side defaults.
        request_fields = list(VideoPathRequest.model_fields.keys())
        return [
            ColumnFilter(
                required_columns=["prompt"],
                optional_columns=[f for f in request_fields if f != "prompt"],
            ),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        """Serialise query.data to VideoPathRequest JSON bytes.

        Only `prompt` is required. All other fields fall back to defaults on
        VideoPathRequest but can be overridden via query.data. Streaming is
        not supported — `stream=True` raises.
        """
        data = query.data
        if "prompt" not in data:
            raise KeyError(
                f"'prompt' not found in query.data keys: {list(data.keys())}"
            )
        if data.get("stream"):
            raise ValueError(
                "VideoGenAdapter is non-streaming; remove `stream` from query.data."
            )
        known = VideoPathRequest.model_fields.keys()
        req = VideoPathRequest.model_validate({k: data[k] for k in known if k in data})
        # exclude_none so optional fields with value None fall back to
        # server-side defaults; fields explicitly set in query.data
        # (e.g. negative_prompt from the bundled JSONL) are forwarded.
        payload = req.model_dump(exclude_none=True)

        # The Jul10 TRT-LLM video server follows the newer OpenAI-shaped
        # schema: url/b64_json transport, "format" for encoder selection, and
        # model-specific Wan knobs under extra_params.
        response_format = payload.get("response_format")
        if response_format == "video_path":
            payload["response_format"] = "url"
        elif response_format == "video_bytes":
            payload["response_format"] = "b64_json"

        payload.pop("output_format", None)
        payload["format"] = "mp4"

        extra_params = payload.setdefault("extra_params", {})
        for field in ("guidance_scale_2", "boundary_ratio"):
            if field in payload:
                extra_params[field] = payload.pop(field)
        if not extra_params:
            payload.pop("extra_params", None)

        return json.dumps(payload, separators=(",", ":")).encode()

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        """Deserialise trtllm-serve response to QueryResult.

        Dispatches by sniffing the first byte:
        - JSON (`{` / `[`): parse as VideoPath/VideoPayload response.
        - Otherwise: treat as raw video bytes. Some trtllm-serve builds return
          binary `video/mp4` regardless of `response_format=video_path`. The
          adapter persists the bytes itself to `$INFERENCE_ENDPOINT_VIDEOGEN_FALLBACK_DIR`
          (must be set; expected to be a shared filesystem path) and returns
          a QueryResult carrying only that path so the QueryResult stays small
          enough to fit in the IPC transport frame.
        """
        # mp4 files carry an ISO BMFF `ftyp` box at offset 4. AVI files start
        # with `RIFF`. The Jul10 server returns a FileResponse for
        # response_format=url, so persist binary video responses locally.
        if cls._looks_like_binary_video(response_bytes):
            return cls._decode_binary_response(response_bytes, query_id)
        return cls._decode_json_response(response_bytes, query_id)

    @classmethod
    def _looks_like_binary_video(cls, response_bytes: bytes) -> bool:
        return (len(response_bytes) >= 8 and response_bytes[4:8] == b"ftyp") or (
            len(response_bytes) >= 12
            and response_bytes[0:4] == b"RIFF"
            and response_bytes[8:12] == b"AVI "
        )

    @classmethod
    def _binary_video_suffix(cls, response_bytes: bytes) -> str:
        if (
            len(response_bytes) >= 12
            and response_bytes[0:4] == b"RIFF"
            and response_bytes[8:12] == b"AVI "
        ):
            return ".avi"
        return ".mp4"

    @classmethod
    def _decode_json_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        raw = json.loads(response_bytes)
        if isinstance(raw.get("b64_json"), str):
            return QueryResult(
                id=query_id,
                metadata={
                    "video_id": raw.get("id", query_id),
                    "video_bytes": raw["b64_json"],
                },
            )
        if isinstance(raw.get("url"), str):
            return QueryResult(
                id=query_id,
                response_output=TextModelOutput(output=raw["url"]),
                metadata={
                    "video_id": raw.get("id", query_id),
                    "video_path": raw["url"],
                },
            )
        # Truthiness check, not key presence: a server that returns
        # `"video_bytes": null` belongs in the video_path branch.
        if isinstance(raw.get("video_bytes"), str):
            resp_bytes = VideoPayloadResponse.model_validate(raw)
            return QueryResult(
                id=query_id,
                metadata={
                    "video_id": resp_bytes.video_id,
                    "video_bytes": resp_bytes.video_bytes,
                },
            )
        resp_path = VideoPathResponse.model_validate(raw)
        # Mirror video_path into response_output so the event log carries it
        # to the accuracy scorer (VBench reads videos by path).
        return QueryResult(
            id=query_id,
            response_output=TextModelOutput(output=resp_path.video_path),
            metadata={
                "video_id": resp_path.video_id,
                "video_path": resp_path.video_path,
            },
        )

    @classmethod
    def _decode_binary_response(
        cls, response_bytes: bytes, query_id: str
    ) -> QueryResult:
        fallback_dir = os.environ.get(_BINARY_FALLBACK_ENV)
        if not fallback_dir:
            raise RuntimeError(
                f"trtllm-serve returned binary video bytes "
                f"({len(response_bytes)} B), but {_BINARY_FALLBACK_ENV} is not "
                "set. Point it at a shared-filesystem directory so the adapter "
                "can persist responses for downstream scoring."
            )
        out_path = Path(fallback_dir) / f"{query_id}{cls._binary_video_suffix(response_bytes)}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response_bytes)
        return QueryResult(
            id=query_id,
            response_output=TextModelOutput(output=str(out_path)),
            metadata={"video_id": query_id, "video_path": str(out_path)},
        )

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> str:
        raise NotImplementedError("VideoGenAdapter does not use SSE streaming")


class VideoGenAccumulator:
    """SSE accumulator stub for HTTPClientConfig contract.

    Video generation requests are non-streaming HTTP, so this class should
    never be exercised. `get_final_output` raises rather than returning an
    empty `QueryResult`, because the worker's SSE path swallows the
    `NotImplementedError` from `decode_sse_message` and would otherwise
    surface zero-output queries as successful.
    """

    def __init__(self, query_id: str, stream_all_chunks: bool) -> None:
        self.query_id = query_id
        # stream_all_chunks is intentionally ignored: non-streaming endpoint.

    def add_chunk(self, delta: Any) -> StreamChunk | None:
        return None

    def get_final_output(self) -> QueryResult:
        raise RuntimeError(
            "VideoGenAccumulator.get_final_output called: video generation is "
            "non-streaming — check HTTPClientConfig.streaming and query.data['stream']."
        )
