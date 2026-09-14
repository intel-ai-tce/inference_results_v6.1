# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

import logging
import os
import sys
import traceback
import atexit
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from generative_recommenders.common import fx_unwrap_optional_tensor, HammerModule
from generative_recommenders.modules.positional_encoder import HSTUPositionalEncoder
from generative_recommenders.modules.postprocessors import (
    L2NormPostprocessor,
    OutputPostprocessor,
)
from generative_recommenders.modules.preprocessors import InputPreprocessor
from generative_recommenders.modules.stu import STU
from generative_recommenders.ops.jagged_tensors import split_2D_jagged

from torch.profiler import record_function

logger: logging.Logger = logging.getLogger(__name__)
torch.fx.wrap("len")
# C1-off last-layer target-only lever. This MUST mirror stu.py's default exactly: when
# DLRM_HSTU_LASTLAYER_TARGETS_ONLY=1, stu.py defaults RETURN_TARGETS_ONLY on and the final
# STU layer returns *trimmed* target rows — so _postprocess must skip the embedding split,
# or it splits an already-trimmed tensor (row-count AssertionError in split_2D_jagged).
# Previously this constant hard-defaulted to "0" and drifted from stu.py, which crashed the
# documented one-flag command in warmup. With the lever OFF (the certified C1-on path),
# _lastlayer_default is "0" → this stays False → the split runs unchanged, so the C1-on
# path is byte-for-byte untouched.
_LASTLAYER_TARGETS_ONLY: bool = (
    os.environ.get("DLRM_HSTU_LASTLAYER_TARGETS_ONLY", "0") == "1"
)
_lastlayer_default: str = "1" if _LASTLAYER_TARGETS_ONLY else "0"
_LASTLAYER_RETURN_TARGETS_ONLY: bool = (
    os.environ.get("DLRM_HSTU_LASTLAYER_RETURN_TARGETS_ONLY", _lastlayer_default) == "1"
)
_STU_GRAPH_ENABLED: bool = os.environ.get("DLRM_HSTU_STU_GRAPH", "0") == "1"
_STU_GRAPH_VERBOSE: bool = os.environ.get("DLRM_HSTU_STU_GRAPH_VERBOSE", "0") == "1"
_STU_GRAPH_L_GRAN: int = int(os.environ.get("DLRM_HSTU_STU_GRAPH_L_GRAN", "16384"))
_STU_GRAPH_N_GRAN: int = int(os.environ.get("DLRM_HSTU_STU_GRAPH_N_GRAN", "1"))
_STU_GRAPH_MAX_BUCKETS: int = int(os.environ.get("DLRM_HSTU_STU_GRAPH_MAX_BUCKETS", "16"))
_STU_GRAPH_MAX_ROWS: int = int(os.environ.get("DLRM_HSTU_STU_GRAPH_MAX_ROWS", "0"))
_STU_GRAPH_CAPTURE_ERROR_MODE: str = os.environ.get(
    "DLRM_HSTU_STU_GRAPH_CAPTURE_ERROR_MODE", "thread_local"
)
_STU_GRAPH_DISABLE_ON_FAIL: bool = (
    os.environ.get("DLRM_HSTU_STU_GRAPH_DISABLE_ON_FAIL", "1") == "1"
)
# Plan 58 Item A3: share ONE CUDA graph mempool across all STU buckets instead of
# letting each capture allocate a private pool. The buckets are never replayed
# concurrently (one batch at a time per worker), so a shared pool is the documented-safe
# pattern and it sizes the capture reservation to the LARGEST bucket's transient peak
# instead of the SUM of <=16 private pools -- removing the fragmentation that made the
# 557056-row capture fail under the loaded worker. Bit-exact (trust-after-replay still
# gates each bucket) and PROVEN to make 557056 capture TRUSTED under load (2026-07-05).
# DEFAULT OFF: Item A was BANKED after the q11,100 PROD10min A/B showed that capturing the
# large bucket (cap raised to 573440) does NOT move the tail -- it slightly REGRESSED it
# (p99 61.7->65.2, p99.9 66.3->87.4 vs the private-pool cap-540672 baseline). The dominant
# large band is 589824 (not 557056) and is un-coverable anyway (max_seq_len proliferation +
# greedy warmup ordering). Kept as an opt-in flag for future work; GOLD uses private pools.
# Set DLRM_HSTU_STU_GRAPH_SHARED_POOL=1 to re-enable (needs its own knee A/B before shipping).
_STU_GRAPH_SHARED_POOL: bool = (
    os.environ.get("DLRM_HSTU_STU_GRAPH_SHARED_POOL", "0") == "1"
)
_STU_GRAPH_STATS: bool = os.environ.get("DLRM_HSTU_STU_GRAPH_STATS", "0") == "1"
_STU_GRAPH_STATS_TOPK: int = int(os.environ.get("DLRM_HSTU_STU_GRAPH_STATS_TOPK", "24"))
_STU_GRAPH_DEFER_CAPTURE: bool = (
    os.environ.get("DLRM_HSTU_STU_GRAPH_DEFER_CAPTURE", "0") == "1"
)
_STU_GRAPH_FREEZE_CAPTURE: bool = False
_STU_GRAPH_QUARANTINE = []


def _graph_log(msg: str) -> None:
    if _STU_GRAPH_VERBOSE:
        print(f"[hstu_stu_graph] {msg}", flush=True, file=sys.stderr)


def set_stu_graph_defer_capture(enabled: bool) -> None:
    global _STU_GRAPH_DEFER_CAPTURE
    _STU_GRAPH_DEFER_CAPTURE = bool(enabled)
    _graph_log(f"defer_capture={_STU_GRAPH_DEFER_CAPTURE}")


def set_stu_graph_freeze_capture(enabled: bool) -> None:
    global _STU_GRAPH_FREEZE_CAPTURE
    _STU_GRAPH_FREEZE_CAPTURE = bool(enabled)
    _graph_log(f"freeze_capture={_STU_GRAPH_FREEZE_CAPTURE}")


def _ceil_to(x: int, gran: int) -> int:
    return ((int(x) + gran - 1) // gran) * gran


class _STUGraphBucket:
    __slots__ = (
        "key",
        "cap_rows",
        "x",
        "lengths",
        "offsets",
        "targets",
        "graph",
        "out",
        "trusted",
        "eager_only",
    )

    def __init__(self, key, cap_rows: int) -> None:
        self.key = key
        self.cap_rows = cap_rows
        self.x = None
        self.lengths = None
        self.offsets = None
        self.targets = None
        self.graph = None
        self.out = None
        self.trusted = False
        self.eager_only = False


class _STUGraphRunner:
    """Static-buffer graph runner for the dense STUStack boundary."""

    def __init__(self) -> None:
        self.buckets = {}
        self.n_graphs = 0
        self.disabled = False
        # Plan 58 A3: one shared mempool handle for every bucket's capture (lazily
        # created on first use, after CUDA is up). None => private pool per graph
        # (the pre-Plan-58 behavior, kept for A/B via DLRM_HSTU_STU_GRAPH_SHARED_POOL=0).
        self._pool = None
        self._stats = defaultdict(
            lambda: {
                "seen": 0,
                "replay": 0,
                "eager_over_cap": 0,
                "eager_only": 0,
                "eager_no_slot": 0,
                "eager_deferred": 0,
                "eager_frozen": 0,
                "capture_attempt": 0,
                "capture_trusted": 0,
                "capture_rejected": 0,
                "capture_failed": 0,
            }
        )
        if _STU_GRAPH_STATS:
            atexit.register(self._dump_stats)

    def _graph_pool(self):
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        return self._pool

    def _log_capture_mem(self, bucket: "_STUGraphBucket", when: str) -> None:
        # Plan 58 A1: log the real capture-point memory state so a piggybacked run
        # shows whether the 557056 capture is bounded by headroom (driver_free < needed)
        # or by fragmentation (driver_free >> needed but a big contiguous block is not
        # available because torch's caching allocator is holding reserved-but-unused VRAM
        # that hipBLASLt's own hipMalloc cannot use). Verbose-gated; costs nothing in GOLD.
        if not _STU_GRAPH_VERBOSE:
            return
        try:
            dev = torch.cuda.current_device()
            free_b, total_b = torch.cuda.mem_get_info(dev)
            reserved = torch.cuda.memory_reserved(dev)
            allocated = torch.cuda.memory_allocated(dev)
            gib = float(1024 ** 3)
            # Largest free (inactive) block inside torch's caching-allocator segments;
            # this is the biggest contiguous region torch can hand out without a new
            # segment. hipBLASLt uses raw hipMalloc, so it competes with driver_free too.
            largest_free = 0
            try:
                for seg in torch.cuda.memory_snapshot():
                    if seg.get("device", dev) != dev:
                        continue
                    for blk in seg.get("blocks", ()):  # 'inactive' == free block
                        if blk.get("state") == "inactive" and blk.get("size", 0) > largest_free:
                            largest_free = blk["size"]
            except Exception:  # noqa: BLE001
                largest_free = -1
            lf = (largest_free / gib) if largest_free >= 0 else float("nan")
            _graph_log(
                f"bucket {bucket.key} mem@{when}: driver_free={free_b / gib:.2f}G "
                f"total={total_b / gib:.2f}G torch_reserved={reserved / gib:.2f}G "
                f"torch_alloc={allocated / gib:.2f}G "
                f"torch_overhead={(reserved - allocated) / gib:.2f}G "
                f"largest_torch_free_block={lf:.2f}G "
                f"shared_pool={_STU_GRAPH_SHARED_POOL} cap_rows={bucket.cap_rows}"
            )
        except Exception as exc:  # noqa: BLE001
            _graph_log(f"bucket {bucket.key}: mem log @{when} failed: {exc!r}")

    def _key(self, max_seq_len: int, x: torch.Tensor, seq_lengths: torch.Tensor,
             num_targets: torch.Tensor, targets_are_uniform: Optional[bool],
             tgt_per_seq: Optional[int]):
        cap_rows = _ceil_to(int(x.shape[0]), _STU_GRAPH_L_GRAN)
        cap_max_seq_len = _ceil_to(int(max_seq_len), _STU_GRAPH_N_GRAN)
        return (
            cap_max_seq_len,
            cap_rows,
            int(x.shape[1]),
            str(x.dtype),
            int(seq_lengths.numel()),
            int(num_targets.numel()) if num_targets is not None else 0,
            bool(targets_are_uniform) if targets_are_uniform is not None else None,
            int(tgt_per_seq) if tgt_per_seq is not None else None,
        )

    def _bump(self, key, field: str, amount: int = 1) -> None:
        if _STU_GRAPH_STATS:
            self._stats[key][field] += amount

    def _dump_stats(self) -> None:
        if not _STU_GRAPH_STATS or not self._stats:
            return
        rows = []
        for key, stats in self._stats.items():
            total = int(stats.get("seen", 0))
            rows.append((total, key, stats))
        rows.sort(reverse=True, key=lambda row: row[0])
        output_dir = os.environ.get("DLRM_OUTPUT_DIR", "").strip()
        rank = os.environ.get("DLRM_MPI_LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "unknown"))
        if output_dir:
            try:
                out_path = Path(output_dir) / f"stu_graph_stats_rank{rank}.jsonl"
                with out_path.open("w") as f:
                    header = {
                        "record": "summary",
                        "rank": rank,
                        "total_buckets": len(rows),
                        "trusted_graphs": self.n_graphs,
                        "disabled": self.disabled,
                        "max_rows": _STU_GRAPH_MAX_ROWS,
                        "max_buckets": _STU_GRAPH_MAX_BUCKETS,
                    }
                    f.write(json.dumps(header, sort_keys=True) + "\n")
                    for _, key, stats in rows:
                        rec = {
                            "record": "bucket",
                            "rank": rank,
                            "key": list(key),
                            "max_seq_len": int(key[0]),
                            "cap_rows": int(key[1]),
                            "dim": int(key[2]),
                            "dtype": str(key[3]),
                            "batch_or_lengths": int(key[4]),
                            "num_targets": int(key[5]),
                            "targets_uniform": key[6],
                            "targets_per_seq": key[7],
                        }
                        rec.update({k: int(v) for k, v in stats.items()})
                        f.write(json.dumps(rec, sort_keys=True) + "\n")
                _graph_log(f"stats wrote {out_path}")
            except Exception as exc:  # noqa: BLE001
                _graph_log(f"stats file write failed: {exc!r}")
        _graph_log(
            f"stats total_buckets={len(rows)} trusted_graphs={self.n_graphs} "
            f"disabled={self.disabled} max_rows={_STU_GRAPH_MAX_ROWS} "
            f"max_buckets={_STU_GRAPH_MAX_BUCKETS}"
        )
        for total, key, stats in rows[:_STU_GRAPH_STATS_TOPK]:
            _graph_log(
                "stats bucket "
                f"key={key} seen={total} replay={stats.get('replay', 0)} "
                f"over_cap={stats.get('eager_over_cap', 0)} "
                f"eager_only={stats.get('eager_only', 0)} "
                f"no_slot={stats.get('eager_no_slot', 0)} "
                f"deferred={stats.get('eager_deferred', 0)} "
                f"frozen={stats.get('eager_frozen', 0)} "
                f"capture_attempt={stats.get('capture_attempt', 0)} "
                f"trusted={stats.get('capture_trusted', 0)} "
                f"rejected={stats.get('capture_rejected', 0)} "
                f"failed={stats.get('capture_failed', 0)}"
            )

    def run(self, stu_module: STU, *, max_seq_len: int, x: torch.Tensor,
            x_lengths: torch.Tensor, x_offsets: torch.Tensor, num_targets: torch.Tensor,
            targets_are_uniform: Optional[bool], tgt_per_seq: Optional[int]) -> torch.Tensor:
        def eager() -> torch.Tensor:
            return stu_module(
                max_seq_len=max_seq_len,
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                num_targets=num_targets,
                targets_are_uniform=targets_are_uniform,
                tgt_per_seq=tgt_per_seq,
            )

        if self.disabled or (
            hasattr(torch.cuda, "is_current_stream_capturing")
            and torch.cuda.is_current_stream_capturing()
        ):
            return eager()
        if x.dim() != 2 or x_lengths.dim() != 1 or x_offsets.dim() != 1:
            return eager()
        if num_targets is None or num_targets.dim() != 1:
            return eager()

        key = self._key(max_seq_len, x, x_lengths, num_targets,
                        targets_are_uniform, tgt_per_seq)
        self._bump(key, "seen")
        bucket = self.buckets.get(key)
        if _STU_GRAPH_MAX_ROWS > 0 and int(key[1]) > _STU_GRAPH_MAX_ROWS:
            self._bump(key, "eager_over_cap")
            if bucket is None:
                bucket = _STUGraphBucket(key, key[1])
                bucket.eager_only = True
                self.buckets[key] = bucket
                _graph_log(
                    f"bucket {key}: skip capture because cap_rows>{_STU_GRAPH_MAX_ROWS}"
                )
            return eager()
        if bucket is not None and bucket.trusted and not bucket.eager_only:
            self._bump(key, "replay")
            self._copy_in(bucket, x, x_lengths, x_offsets, num_targets)
            bucket.graph.replay()
            return bucket.out
        if bucket is not None and bucket.eager_only:
            self._bump(key, "eager_only")
            return eager()
        if _STU_GRAPH_DEFER_CAPTURE:
            self._bump(key, "eager_deferred")
            return eager()
        if _STU_GRAPH_FREEZE_CAPTURE:
            self._bump(key, "eager_frozen")
            return eager()

        eager_out = eager()
        if bucket is None:
            if self.n_graphs >= _STU_GRAPH_MAX_BUCKETS:
                self._bump(key, "eager_no_slot")
                return eager_out
            bucket = _STUGraphBucket(key, key[1])
            self.buckets[key] = bucket
        try:
            self._bump(key, "capture_attempt")
            self._capture(bucket, stu_module, key[0], x, x_lengths, x_offsets,
                          num_targets, targets_are_uniform, tgt_per_seq)
            self._copy_in(bucket, x, x_lengths, x_offsets, num_targets)
            bucket.graph.replay()
            torch.cuda.synchronize()
            n = min(bucket.out.numel(), eager_out.numel())
            delta = (
                bucket.out.reshape(-1)[:n].float() - eager_out.reshape(-1)[:n].float()
            ).abs().max().item() if n > 0 else 0.0
            shape_ok = tuple(bucket.out.shape) == tuple(eager_out.shape)
            bucket.trusted = shape_ok and delta == 0.0
            bucket.eager_only = not bucket.trusted
            self.n_graphs += 1
            self._bump(key, "capture_trusted" if bucket.trusted else "capture_rejected")
            _graph_log(
                f"bucket {key}: capture {'TRUSTED' if bucket.trusted else 'REJECTED'} "
                f"shape_ok={shape_ok} max|delta|={delta:.3e} graphs={self.n_graphs}"
            )
        except Exception as exc:  # noqa: BLE001
            self._bump(key, "capture_failed")
            bucket.trusted = False
            bucket.eager_only = True
            bucket.graph = None
            bucket.out = None
            if _STU_GRAPH_DISABLE_ON_FAIL:
                self.disabled = True
            _graph_log(f"bucket {key}: capture FAILED -> eager-only: {exc!r}\n{traceback.format_exc()}")
            try:
                torch.cuda.synchronize()
            except Exception as sync_exc:  # noqa: BLE001
                _graph_log(f"bucket {key}: post-failure synchronize also failed: {sync_exc!r}")
        return eager_out

    def _capture(self, bucket: _STUGraphBucket, stu_module: STU, max_seq_len: int,
                 x: torch.Tensor, x_lengths: torch.Tensor, x_offsets: torch.Tensor,
                 num_targets: torch.Tensor, targets_are_uniform: Optional[bool],
                 tgt_per_seq: Optional[int]) -> None:
        graph = None
        capture_stream = None
        try:
            bucket.x = torch.zeros(
                (bucket.cap_rows, x.shape[1]), dtype=x.dtype, device=x.device
            )
            bucket.lengths = torch.empty_like(x_lengths)
            bucket.offsets = torch.empty_like(x_offsets)
            bucket.targets = torch.empty_like(num_targets)
            self._copy_in(bucket, x, x_lengths, x_offsets, num_targets)

            def static_forward() -> torch.Tensor:
                return stu_module(
                    max_seq_len=max_seq_len,
                    x=bucket.x,
                    x_lengths=bucket.lengths,
                    x_offsets=bucket.offsets,
                    num_targets=bucket.targets,
                    targets_are_uniform=targets_are_uniform,
                    tgt_per_seq=tgt_per_seq,
                )

            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    static_forward()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            self._log_capture_mem(bucket, "pre_capture")
            graph_kwargs = dict(
                stream=capture_stream,
                capture_error_mode=_STU_GRAPH_CAPTURE_ERROR_MODE,
            )
            if _STU_GRAPH_SHARED_POOL:
                graph_kwargs["pool"] = self._graph_pool()
            with torch.cuda.graph(graph, **graph_kwargs):
                out = static_forward()
            torch.cuda.current_stream().wait_stream(capture_stream)
            bucket.graph = graph
            bucket.out = out
            torch.cuda.synchronize()
            self._log_capture_mem(bucket, "post_capture")
        except Exception:
            # ROCm can leave a failed graph object/stream in an invalidated
            # capture state. Keep references alive so their destructors do not
            # run while the worker is trying to return to eager fallback.
            if graph is not None:
                _STU_GRAPH_QUARANTINE.append(graph)
            if capture_stream is not None:
                _STU_GRAPH_QUARANTINE.append(capture_stream)
            raise

    def _copy_in(self, bucket: _STUGraphBucket, x: torch.Tensor, x_lengths: torch.Tensor,
                 x_offsets: torch.Tensor, num_targets: torch.Tensor) -> None:
        if x.shape[0] > bucket.cap_rows:
            raise RuntimeError(f"x rows {x.shape[0]} exceed graph cap {bucket.cap_rows}")
        if x.shape[0] < bucket.cap_rows:
            bucket.x[x.shape[0]:].zero_()
        bucket.x[: x.shape[0]].copy_(x)
        bucket.lengths.copy_(x_lengths)
        bucket.offsets.copy_(x_offsets)
        bucket.targets.copy_(num_targets)

try:
    import fbgemm_gpu  # noqa: F401
except ImportError:
    pass


@torch.fx.wrap
def default_seq_payload(
    seq_payloads: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if seq_payloads is None:
        return {}
    else:
        return torch.jit._unwrap_optional(seq_payloads)


class HSTUTransducer(HammerModule):
    def __init__(
        self,
        stu_module: STU,
        input_preprocessor: InputPreprocessor,
        output_postprocessor: Optional[OutputPostprocessor] = None,
        input_dropout_ratio: float = 0.0,
        positional_encoder: Optional[HSTUPositionalEncoder] = None,
        is_inference: bool = True,
        return_full_embeddings: bool = False,
        listwise: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_module = stu_module
        self._input_preprocessor: InputPreprocessor = input_preprocessor
        self._output_postprocessor: OutputPostprocessor = (
            output_postprocessor
            if output_postprocessor is not None
            else L2NormPostprocessor(is_inference=is_inference)
        )
        assert (
            self._is_inference == self._input_preprocessor._is_inference
        ), f"input_preprocessor must have the same mode; self: {self._is_inference} vs input_preprocessor {self._input_preprocessor._is_inference}"
        self._positional_encoder: Optional[HSTUPositionalEncoder] = positional_encoder
        self._input_dropout_ratio: float = input_dropout_ratio
        self._return_full_embeddings: bool = return_full_embeddings
        self._listwise_training: bool = listwise and self.is_train
        self._stu_graph_runner: Optional[_STUGraphRunner] = (
            _STUGraphRunner() if _STU_GRAPH_ENABLED else None
        )

        for name, m in self.named_modules():
            if "_stu_module" in name:
                continue
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, torch.nn.LayerNorm):
                if m.weight.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None and m.bias.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.bias)

    def _preprocess(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        seq_payloads = default_seq_payload(seq_payloads)

        with record_function("hstu_input_preprocessor"):
            (
                output_max_seq_len,
                output_total_uih_len,
                output_total_targets,
                output_seq_lengths,
                output_seq_offsets,
                output_seq_timestamps,
                output_seq_embeddings,
                output_num_targets,
                output_seq_payloads,
            ) = self._input_preprocessor(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_lengths=seq_lengths,
                seq_timestamps=seq_timestamps,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                seq_payloads=seq_payloads,
            )

        with record_function("hstu_positional_encoder"):
            if self._positional_encoder is not None:
                output_seq_embeddings = self._positional_encoder(
                    max_seq_len=output_max_seq_len,
                    seq_lengths=output_seq_lengths,
                    seq_offsets=output_seq_offsets,
                    seq_timestamps=output_seq_timestamps,
                    seq_embeddings=output_seq_embeddings,
                    num_targets=(
                        None if self._listwise_training else output_num_targets
                    ),
                )

        output_seq_embeddings = torch.nn.functional.dropout(
            output_seq_embeddings,
            p=self._input_dropout_ratio,
            training=self.training,
        )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            output_seq_payloads,
        )

    def _hstu_compute(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        targets_are_uniform: Optional[bool] = None,
        tgt_per_seq: Optional[int] = None,
    ) -> torch.Tensor:
        with record_function("hstu"):
            graph_num_targets = None if self._listwise_training else num_targets
            if self._stu_graph_runner is not None and graph_num_targets is not None:
                seq_embeddings = self._stu_graph_runner.run(
                    self._stu_module,
                    max_seq_len=max_seq_len,
                    x=seq_embeddings,
                    x_lengths=seq_lengths,
                    x_offsets=seq_offsets,
                    num_targets=graph_num_targets,
                    targets_are_uniform=targets_are_uniform,
                    tgt_per_seq=tgt_per_seq,
                )
            else:
                seq_embeddings = self._stu_module(
                    max_seq_len=max_seq_len,
                    x=seq_embeddings,
                    x_lengths=seq_lengths,
                    x_offsets=seq_offsets,
                    num_targets=graph_num_targets,
                    targets_are_uniform=targets_are_uniform,
                    tgt_per_seq=tgt_per_seq,
                )
        return seq_embeddings

    def _postprocess(
        self,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        with record_function("hstu_output_postprocessor"):
            if self._return_full_embeddings:
                seq_embeddings = self._output_postprocessor(
                    seq_embeddings=seq_embeddings,
                    seq_timestamps=seq_timestamps,
                    seq_payloads=seq_payloads,
                )
            uih_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                seq_lengths - num_targets
            )
            candidates_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_targets
            )
            # C1-off lever: skip the history/candidate split only when the final STU layer
            # actually returned target rows only. The `size(0) == total_targets` guard makes
            # this robust to flag drift between this module and stu.py — if the embeddings are
            # still full-length we fall back to the split instead of asserting. With the lever
            # off, _LASTLAYER_RETURN_TARGETS_ONLY is False → the split runs exactly as the
            # certified C1-on path (the guard is not reached).
            if (
                _LASTLAYER_RETURN_TARGETS_ONLY
                and not self._return_full_embeddings
                and seq_embeddings.size(0) == total_targets
            ):
                candidate_embeddings = seq_embeddings
            else:
                _, candidate_embeddings = split_2D_jagged(
                    values=seq_embeddings,
                    max_seq_len=max_seq_len,
                    total_len_left=total_uih_len,
                    total_len_right=total_targets,
                    offsets_left=uih_offsets,
                    offsets_right=candidates_offsets,
                    kernel=self.hammer_kernel(),
                )
            interleave_targets: bool = self._input_preprocessor.interleave_targets()
            if interleave_targets:
                candidate_embeddings = candidate_embeddings.view(
                    -1, 2, candidate_embeddings.size(-1)
                )[:, 0, :]
            if not self._return_full_embeddings:
                _, candidate_timestamps = split_2D_jagged(
                    values=seq_timestamps.unsqueeze(-1),
                    max_seq_len=max_seq_len,
                    total_len_left=total_uih_len,
                    total_len_right=total_targets,
                    offsets_left=uih_offsets,
                    offsets_right=candidates_offsets,
                    kernel=self.hammer_kernel(),
                )
                candidate_timestamps = candidate_timestamps.squeeze(-1)
                if interleave_targets:
                    candidate_timestamps = candidate_timestamps.view(-1, 2)[
                        :, 0]
                candidate_embeddings = self._output_postprocessor(
                    seq_embeddings=candidate_embeddings,
                    seq_timestamps=candidate_timestamps,
                    seq_payloads=seq_payloads,
                )

            return (
                seq_embeddings if self._return_full_embeddings else None,
                candidate_embeddings,
            )

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        targets_are_uniform: Optional[bool] = None,
        tgt_per_seq: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        orig_dtype = seq_embeddings.dtype
        if not self._is_inference:
            seq_embeddings = seq_embeddings.to(self._training_dtype)

        (
            max_seq_len,
            total_uih_len,
            total_targets,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
            seq_payloads,
        ) = self._preprocess(
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

        encoded_embeddings = self._hstu_compute(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            targets_are_uniform=targets_are_uniform,
            tgt_per_seq=tgt_per_seq,
        )

        encoded_embeddings, encoded_candidate_embeddings = self._postprocess(
            max_seq_len=max_seq_len,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

        if not self._is_inference:
            encoded_candidate_embeddings = encoded_candidate_embeddings.to(
                orig_dtype)
            if self._return_full_embeddings:
                encoded_embeddings = fx_unwrap_optional_tensor(encoded_embeddings).to(
                    orig_dtype
                )
        return (
            encoded_candidate_embeddings,
            encoded_embeddings,
        )
