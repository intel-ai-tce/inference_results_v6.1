import zmq
import pickle
import time
import queue
import threading
from typing import Dict

from harness_llm.backends.common.server_utils import DeviceSelector
from harness_llm.backends.common.constants import WarmUp
from harness_llm.backends.common.utils import (
    create_response_and_send_complete,
    create_response_and_send_first_token,
)
from harness_llm.backends.vllm.zmq.disagg_utils import (
    format_request_id,
    message_tag,
    TAG_REGISTER,
    TAG_KVADDR,
    TAG_PREFILL_DONE,
    ROLE_PREFILL,
    ROLE_DECODE,
)
from harness_llm.backends.vllm.zmq.distributed_server_sut import (
    DistributedServerSUT,
    QueryInfo,
    _ACK_SEND_RETRIES,
    _ACK_SEND_DELAY_S,
    _REQUEST_SEND_RETRIES,
    _REQUEST_SEND_DELAY_S,
)


class DistributedDisaggSUT(DistributedServerSUT):
    """SUT-driven disaggregated prefill/decode SUT serving BOTH the server and
    offline MLPerf scenarios from a single class. Server vs offline differ only
    in self.report_first_token (and the matching worker stream_output)."""

    _ALLOWED_ALGOS = (
        "shortest_queue_with_tokens",
        "shortest_queue",
        "round_robin",
    )

    # Sentinel enqueued to tell the dedicated sender thread to exit.
    _SENDER_STOP = object()

    def __init__(self, config):
        # The parent asserts on harness_config["schedule_algo"]; the disagg SUT
        # uses its own per-role selectors, so satisfy the assert defensively.
        if "schedule_algo" not in config["harness_config"]:
            config["harness_config"]["schedule_algo"] = "round_robin"

        super().__init__(config)

        self.report_first_token = (config["scenario"].lower() == "server")

        # Global totals default to this node's local counts (single-node). For
        # multi-node, the *_disagg_mn.yaml sets the global totals explicitly.
        # When derived, a multi-node deployment (workers from >1 host) is a
        # misconfiguration and is rejected at registration time.
        self._totals_derived = (
            "total_prefills_count" not in self.harness_config
            or "total_decodes_count" not in self.harness_config
        )
        self.total_prefills_count = self.harness_config.get(
            "total_prefills_count", self.harness_config["prefills_count"]
        )
        self.total_decodes_count = self.harness_config.get(
            "total_decodes_count", self.harness_config["decodes_count"]
        )

        # Role-local pools (int idx -> QueryInfo) and identity routing table.
        self.prefills: Dict[int, QueryInfo] = {}
        self.decodes: Dict[int, QueryInfo] = {}
        self.identity_to_pool: Dict[bytes, tuple] = {}

        # Per-role arrival-order rank counters.
        self._prefill_rank_counter = 0
        self._decode_rank_counter = 0

        # formatted_request_id -> (item_tuple, decode_idx), gated on prefill-done.
        self.pending: Dict[str, tuple] = {}

        self.benchmark = config["benchmark_name"]

        # Per-role load balancing.
        prefill_algo = self.harness_config.get("prefill_schedule_algo", "round_robin")
        decode_algo = self.harness_config.get("decode_schedule_algo", "round_robin")
        assert (
            prefill_algo in self._ALLOWED_ALGOS
        ), f"Unsupported prefill schedule algo: {prefill_algo}"
        assert (
            decode_algo in self._ALLOWED_ALGOS
        ), f"Unsupported decode schedule algo: {decode_algo}"
        self.prefill_schedule_algo = prefill_algo
        self.decode_schedule_algo = decode_algo

        self.prefill_selector = DeviceSelector(self.prefills)
        self.decode_selector = DeviceSelector(self.decodes)
        self.get_next_prefill = self._make_selector_fn(
            self.prefill_selector, self.prefills, prefill_algo
        )
        self.get_next_decode = self._make_selector_fn(
            self.decode_selector, self.decodes, decode_algo
        )

        # Warmup / cross-thread sync state.
        self.warming = False
        self._warmup_done = 0
        self._warmup_cond = threading.Condition()
        # Set by the recv thread once every worker has reported its KV address.
        self._registered = threading.Event()
        # Distinct worker hosts seen during registration, and an error raised by
        # the recv thread (propagated to the main thread via _registered).
        self._registered_hosts: set = set()
        self._startup_error = None
        # self.sender (port ROUTER) is written from BOTH the LoadGen issue
        # thread(s) (prefill dispatch) and the recv thread (decode release +
        # handshake replies). A ZMQ socket must be owned by exactly ONE thread,
        # so instead of sharing it we funnel every send through this queue,
        # drained by a single dedicated sender thread (see _sender_loop).
        self.send_queue: queue.Queue = queue.Queue()
        # Number of workers still expected to send their shutdown sentinel.
        self._drain_remaining = 0

    def _make_selector_fn(self, selector, pool, algo):
        """Build a get_next_* callable mirroring the parent's algo mapping but
        scoped to a single role pool."""
        if algo == "shortest_queue_with_tokens":
            return lambda: selector.next_best_device_id_with_tokens(
                len(pool), self.harness_config["load_balance_token_weight"]
            )
        elif algo == "shortest_queue":
            return lambda: selector.next_best_device_id(len(pool))
        else:
            return selector.next_device_id

    def start(self):
        self.context = zmq.Context()
        self.context.set(zmq.MAX_SOCKETS, 1024)

        # Dedicated sender thread exclusively owns the port ROUTER socket
        # (SUT -> workers). Every send_data call from any thread is enqueued and
        # this thread performs the actual socket I/O, satisfying ZMQ's
        # one-socket-per-thread rule.
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.sender_thread.start()

        # The recv thread registers the workers (register_servers) and then
        # collects their outputs, mirroring DistributedServerSUT.recv_outputs.
        self.output_collector_thread = threading.Thread(
            target=self.recv_outputs, daemon=True
        )
        self.output_collector_thread.start()

        self.wait_for_servers_ready()
        if self.harness_config["enable_warmup"]:
            self.run_warmup()
        self.log(
            f"Disagg server started with {self.total_prefills_count} prefills, "
            f"{self.total_decodes_count} decodes"
        )

    def wait_for_servers_ready(self):
        """Block until the recv thread has registered both pools, then release
        every worker with a go-ahead ack. Mirrors
        DistributedServerSUT.wait_for_servers_ready; the registration itself is
        the disagg two-round handshake driven by register_servers (the per-worker
        rank reply must be sent there, since each worker blocks on it before
        building its engine and reporting its KV address)."""
        self._registered.wait()
        if self._startup_error is not None:
            raise self._startup_error
        # All engines are bound; release every worker with a go-ahead ack (long
        # retry budget so a slow-joining worker doesn't miss it and hang).
        for identity in self.identity_to_pool:
            self.send_data(
                identity, identity,
                retries=_ACK_SEND_RETRIES, delay=_ACK_SEND_DELAY_S,
            )

    def _sender_loop(self):
        # Exclusively owns self.sender (port ROUTER, SUT -> workers). All sends
        # are funneled here via self.send_queue so the socket is touched by only
        # this thread.
        self.sender = self.context.socket(zmq.ROUTER)
        # Fail loudly (EHOSTUNREACH) on an unroutable send so _routed_send can
        # retry the slow-joiner race instead of silently dropping the frame.
        self.sender.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.sender.bind(f"tcp://*:{self.port}")
        while True:
            item = self.send_queue.get()
            if item is self._SENDER_STOP:
                break
            identity, data, retries, delay = item
            try:
                self._routed_send(identity, data, retries=retries, delay=delay)
            except RuntimeError as exc:
                # A dead/unreachable worker must not kill the sender thread (e.g.
                # shutdown Nones to an already-exited worker). Log and continue.
                self.log(str(exc))
        self.sender.close()

    def send_data(self, identity, data, retries=_REQUEST_SEND_RETRIES, delay=_REQUEST_SEND_DELAY_S):
        # Hand off to the dedicated sender thread; never touch self.sender from
        # the calling thread (issue thread, recv thread, warmup, ...). The retry
        # budget rides along so the sender thread applies the same slow-joiner-safe
        # ROUTER_MANDATORY retry (long for acks, short for steady-state requests).
        self.send_queue.put((identity, data, retries, delay))

    def register_servers(self, socket):
        """Registration handshake for the separate prefill and decode pools.

        Mirrors DistributedServerSUT.register_servers (a discrete phase run at
        the head of recv_outputs that populates the pools before steady state),
        but performs the disagg two-round exchange: each worker REGISTERs with
        its role + address and is immediately replied [rank, N_p, N_d] (the
        worker blocks on this before building its engine), then reports its KV
        address. Populates self.prefills / self.decodes / self.identity_to_pool
        and sets self._registered once every worker's KV address is in (or on a
        startup error, with self._startup_error set)."""
        total = self.total_prefills_count + self.total_decodes_count
        self.log(
            f"Waiting for {total} workers to register "
            f"({self.total_prefills_count} prefill + {self.total_decodes_count} decode)..."
        )
        registered = 0
        while registered < total:
            identity, raw = socket.recv_multipart()
            msg = pickle.loads(raw)
            tag = message_tag(msg)
            if tag == TAG_REGISTER:
                role = msg[2]
                local_address = msg[3]
                # Reject a multi-node deployment when the global totals were
                # only derived from this node's local counts: the SUT would
                # otherwise wait for the wrong number of workers.
                self._registered_hosts.add(local_address)
                if self._totals_derived and len(self._registered_hosts) > 1:
                    self._startup_error = RuntimeError(
                        "Multi-node disagg detected (workers from hosts "
                        f"{sorted(self._registered_hosts)}) but "
                        "total_prefills_count/total_decodes_count are not set. "
                        "Use the *_disagg_mn.yaml config (or set the totals "
                        "explicitly)."
                    )
                    self._registered.set()
                    return
                if role == ROLE_PREFILL:
                    idx = self._prefill_rank_counter
                    rank = idx
                    self._prefill_rank_counter += 1
                    self.prefills[idx] = QueryInfo(identity=identity)
                    self.identity_to_pool[identity] = (ROLE_PREFILL, idx)
                else:
                    idx = self._decode_rank_counter
                    rank = self.total_prefills_count + idx
                    self._decode_rank_counter += 1
                    self.decodes[idx] = QueryInfo(identity=identity)
                    self.identity_to_pool[identity] = (ROLE_DECODE, idx)
                self.log(
                    f"Registered {role} identity={identity} rank={rank} "
                    f"local_address={local_address}"
                )
                self.send_data(
                    identity,
                    [rank, self.total_prefills_count, self.total_decodes_count],
                    retries=_ACK_SEND_RETRIES, delay=_ACK_SEND_DELAY_S,
                )
            elif tag == TAG_KVADDR:
                k_identity = msg[1]
                kv_addr = msg[2]
                role, idx = self.identity_to_pool[k_identity]
                pool = self.prefills if role == ROLE_PREFILL else self.decodes
                pool[idx].kv_addr = kv_addr
                registered += 1
                self.log(
                    f"KV addr for {role}[{idx}] = {kv_addr} ({registered}/{total})"
                )
        self._drain_remaining = total
        self._registered.set()

    def recv_outputs(self):
        # ZMQ requires a dedicated context per thread.
        ctx = zmq.Context()
        receiver = ctx.socket(zmq.ROUTER)
        receiver.bind(f"tcp://*:{int(self.port) + 1}")
        receiver.setsockopt(zmq.LINGER, 0)

        # Registration phase first, then steady state (mirrors
        # DistributedServerSUT.recv_outputs). The go-ahead acks that release the
        # registered workers are sent from wait_for_servers_ready (main thread).
        self.register_servers(receiver)
        if self._startup_error is not None:
            receiver.close()
            ctx.term()
            return

        # --- Steady state: prefill-done releases + decode token streams. ---
        self.log("Collecting outputs started...")
        while True:
            identity, raw = receiver.recv_multipart()
            response = pickle.loads(raw)
            if response is None:
                self.log(f"{identity} exited")
                self._drain_remaining -= 1
                if self._drain_remaining <= 0:
                    break
                continue
            tag = message_tag(response)
            if tag == TAG_PREFILL_DONE:
                formatted_id = response[1]
                _, p_idx = self.identity_to_pool[identity]
                self.prefills[p_idx].increment_finished()
                item, decode_idx = self.pending.pop(formatted_id)
                # Prefill is done and the KV cache is staged; release to decode.
                self.send_data(self.decodes[decode_idx].identity, [item])
            else:
                decode_idx = self.identity_to_pool[identity][1]
                self.post_proc(response, decode_idx)
            if not self.stopped and self.debug_toolkit.debug_print_finished:
                self.print_finished()
        self.log("Collecting outputs finished...")
        receiver.close()
        ctx.term()

    def _dispatch(self, sample_id, prompt_token_ids, stop_ids):
        p_idx = self.get_next_prefill()
        d_idx = self.get_next_decode()
        p = self.prefills[p_idx]
        d = self.decodes[d_idx]
        formatted_id = format_request_id(str(sample_id), p.kv_addr, d.kv_addr)
        item = (formatted_id, str(sample_id), prompt_token_ids, stop_ids)
        # Gate the decode dispatch until the prefill reports PREFILL_DONE.
        self.pending[formatted_id] = (item, d_idx)

        p.increment_sent()
        if self.prefill_schedule_algo == "shortest_queue_with_tokens":
            window_size = self.harness_config["load_balance_window_size"]
            if len(p.tokens_in) > window_size:
                p.tokens_in.pop(0)
            p.tokens_in.append(len(prompt_token_ids))
        # Commit-time decode accounting so shortest_queue LB sees the in-flight
        # request immediately, even though the work is sent only after prefill.
        d.increment_sent()

        # Eager dispatch to the prefill only; decode is released later.
        self.send_data(p.identity, [item])

    def send_sample(self, sample):
        prompt_token_ids = self.data_object.input_ids[sample.index]
        stop_ids = (
            self.data_object.stop_ids[sample.index]
            if self.data_object.stop_ids
            else None
        )
        self._dispatch(sample.id, prompt_token_ids, stop_ids)

    def post_proc(self, response, decode_idx):
        sample_id = int(response[0])
        token_ids = response[1]
        finished = token_ids is None

        if self.warming:
            # Warmup traffic never reaches LoadGen; just count completions.
            if finished:
                with self._warmup_cond:
                    self._warmup_done += 1
                    self._warmup_cond.notify()
                self.response_buffer.pop(sample_id, None)
            return

        if finished:
            if self.harness_config["debug_dump_model_output"]:
                self.debug_toolkit.dump([self.response_buffer[sample_id]])
            create_response_and_send_complete(
                sample_id, self.response_buffer[sample_id]
            )
            del self.response_buffer[sample_id]
            self.n_finished += 1
            self.decodes[decode_idx].increment_finished()
            if not self.report_first_token:
                # Offline scenario never streams first tokens, so the
                # carriage-return progress line (print_finished) carries no
                # useful signal. Log the completion count like the offline SUT
                # so response progress is visible without debug_print_finished.
                self.log(
                    f"Processed prompts: {self.n_finished}/"
                    f"{self.harness_config['total_sample_count']}"
                )
        elif sample_id not in self.response_buffer:
            self.response_buffer[sample_id] = list(token_ids)
            if self.report_first_token:
                create_response_and_send_first_token(sample_id, token_ids)
                self.n_finished_first += 1
        else:
            self.response_buffer[sample_id].extend(token_ids)

        if self.report_first_token and self.harness_config["debug_record_sample_latencies"]:
            self.debug_toolkit.record_sample_latencies(sample_id, token_ids)

    def run_warmup(self):
        prompt = WarmUp.ENCODED_SAMPLES.get(self.benchmark)
        if prompt is None:
            self.log(f"No warmup samples for benchmark={self.benchmark}; skipping")
            return

        self.warming = True
        n = max(self.total_prefills_count, self.total_decodes_count) * 2
        self._warmup_done = 0
        # Negative sentinel ids so warmup can never collide with real samples.
        for i in range(n):
            self._dispatch(-(i + 1), prompt, None)

        with self._warmup_cond:
            while self._warmup_done < n:
                self._warmup_cond.wait()

        # Reset all state warmup touched so it does not skew load balancing.
        self.warming = False
        self.pending.clear()
        self.response_buffer.clear()
        for entry in list(self.prefills.values()) + list(self.decodes.values()):
            entry.sent = 0
            entry.finished = 0
            entry.tokens_in = []
        self.log("Warmup done")

    def print_finished(self):
        # Use the same progress print as the regular server SUT
        # (DebugToolkit.print_server_progress): a single carriage-return status
        # line "Processed prompts: N first tokens: M  <dev>:sent/finished q:queue".
        # Decodes own completions and first tokens, so report the decode pool.
        self.debug_toolkit.print_server_progress(
            self.n_finished, self.n_finished_first, self.decodes, len(self.decodes)
        )

    def stop(self):
        for entry in self.prefills.values():
            self.send_data(entry.identity, None)
        for entry in self.decodes.values():
            self.send_data(entry.identity, None)
        self.stopped = True
        # Workers exit after receiving the shutdown sentinel and echo a final
        # None back; the recv thread drains those and returns.
        self.output_collector_thread.join()
        # Stop the sender thread only after the shutdown sentinels are flushed.
        self.send_queue.put(self._SENDER_STOP)
        self.sender_thread.join()
        self.context.term()
        time.sleep(10)
