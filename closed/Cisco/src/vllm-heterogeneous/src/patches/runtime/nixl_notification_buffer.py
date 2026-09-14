
"""
Patch vLLM's NIXL worker for two independent notification failure modes.

The legacy patch buffers completion notifications that arrive before scheduler
metadata. The v0.24 patch optionally chunks decode heartbeat notifications so
UCCL never receives a notification larger than its 256-byte hard limit.

Heartbeat chunking is deliberately opt-in. Set
VLLM_NIXL_HEARTBEAT_MAX_BYTES=256 (or a smaller positive wire-size limit) on
decode workers. The patch reserves 32 bytes for NIXL/UCCL framing. With the
variable unset or zero, vLLM keeps its original single-message heartbeat
behavior, so normal PD launches are unchanged.

Mixed topologies can also opt into a longer KV lease by setting
VLLM_NIXL_KV_LEASE_DURATION_SECONDS to a positive integer. With the variable
unset or zero, the installed lease sources are not touched.

Usage (inside container):
    python <submission-root>/src/patches/runtime/nixl_notification_buffer.py

This modifies the installed NIXL worker source in-place and creates backups.
"""

import importlib
import os
import shutil
import sys


def find_nixl_connector():
    module_names = [
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker",
    ]
    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
            return mod.__file__
        except ImportError:
            pass

    for sp in sys.path:
        for rel in (
            "vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py",
            "vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py",
        ):
            candidate = os.path.join(sp, rel)
            if os.path.isfile(candidate):
                return candidate

    return None


def find_nixl_base_worker():
    module_name = (
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker"
    )
    try:
        mod = importlib.import_module(module_name)
        return mod.__file__
    except ImportError:
        pass

    for sp in sys.path:
        candidate = os.path.join(
            sp,
            "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py",
        )
        if os.path.isfile(candidate):
            return candidate

    return None


def find_nixl_base_scheduler():
    module_name = (
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler"
    )
    try:
        mod = importlib.import_module(module_name)
        return mod.__file__
    except ImportError:
        pass

    for sp in sys.path:
        candidate = os.path.join(
            sp,
            "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py",
        )
        if os.path.isfile(candidate):
            return candidate

    return None


INIT_MARKER = "self._pending_notifs"

HEARTBEAT_MARKER = "VLLM_NIXL_HEARTBEAT_MAX_BYTES"
HEARTBEAT_PATCH_VERSION = "VLLM_NIXL_HEARTBEAT_PATCH_V2"
HEARTBEAT_WIRE_OVERHEAD_BYTES = 32
LEASE_ENV = "VLLM_NIXL_KV_LEASE_DURATION_SECONDS"
LEASE_PATCH_MARKER = "VLLM_NIXL_KV_LEASE_DURATION_PATCH"

NEW_SEND_HEARTBEATS = '''\
    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        """
        Send heartbeat notifications to remote engines, extending lease on KV
        blocks. Optionally split the request IDs across bounded notifications;
        UCCL's notification payload limit is 256 bytes.
        """
        # VLLM_NIXL_HEARTBEAT_PATCH_V2: the configured limit is the maximum
        # wire notification size, so reserve space for NIXL/UCCL framing.
        raw_max_bytes = os.environ.get(
            "VLLM_NIXL_HEARTBEAT_MAX_BYTES", "0"
        ).strip()
        try:
            heartbeat_max_bytes = int(raw_max_bytes or "0")
        except ValueError:
            logger.error(
                "Invalid VLLM_NIXL_HEARTBEAT_MAX_BYTES=%r; "
                "heartbeat chunking disabled",
                raw_max_bytes,
            )
            heartbeat_max_bytes = 0

        heartbeat_payload_max_bytes = 0
        if heartbeat_max_bytes > 0:
            heartbeat_wire_max_bytes = min(heartbeat_max_bytes, 256)
            heartbeat_payload_max_bytes = max(
                heartbeat_wire_max_bytes - 32, 3
            )

        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake (this request may still be in waiting queue) so
            # the next heartbeat for this remote can go through.
            if (
                self._ensure_handshake(
                    engine_id, hb_info.host, hb_info.port, hb_info.tp_size
                )
                is not None
            ):
                continue  # handshake is still pending

            heartbeat_messages: list[bytes]
            if heartbeat_payload_max_bytes <= 0:
                # Preserve stock vLLM behavior unless explicitly enabled.
                heartbeat_messages = [
                    ("HB:" + ",".join(hb_info.req_ids)).encode()
                ]
            else:
                prefix = b"HB:"
                heartbeat_messages = []
                current = prefix
                for req_id in hb_info.req_ids:
                    encoded_req_id = req_id.encode("utf-8")
                    separator = b"" if current == prefix else b","
                    if (
                        len(current) + len(separator) + len(encoded_req_id)
                        > heartbeat_payload_max_bytes
                    ):
                        if current != prefix:
                            heartbeat_messages.append(current)
                            current = prefix
                            separator = b""
                        if (
                            len(prefix) + len(encoded_req_id)
                            > heartbeat_payload_max_bytes
                        ):
                            logger.error(
                                "NIXL heartbeat request ID is too large for "
                                "%d-byte notification limit: %r",
                                heartbeat_payload_max_bytes,
                                req_id,
                            )
                            continue
                    current += separator + encoded_req_id

                if current != prefix or not hb_info.req_ids:
                    heartbeat_messages.append(current)

                if len(heartbeat_messages) > 1:
                    logger.debug(
                        "Split NIXL heartbeat for engine %s into %d "
                        "notifications (payload_max_bytes=%d)",
                        engine_id,
                        len(heartbeat_messages),
                        heartbeat_payload_max_bytes,
                    )

            for agent_name in self._remote_agents[engine_id].values():
                for hb_msg in heartbeat_messages:
                    try:
                        self.nixl_wrapper.send_notif(
                            agent_name, notif_msg=hb_msg
                        )
                    except Exception:
                        logger.debug(
                            "Failed to send heartbeat to engine %s",
                            engine_id,
                            exc_info=True,
                        )
'''

NEW_GET_NEW_NOTIFS = '''\
    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.

        Notifications that arrive before their metadata has been processed
        (RDMA faster than scheduler->worker IPC) are buffered and retried
        on the next call rather than discarded.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()

        all_notifs: list[bytes] = list(self._pending_notifs)
        self._pending_notifs = []
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            all_notifs.extend(notifs)

        still_pending: list[bytes] = []
        for notif in all_notifs:
            msg = notif.decode("utf-8")

            # Handle heartbeat messages from D-side.
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            req_id, tp_size = msg.rsplit(":", 1)
            if (
                req_id not in self._reqs_to_send
                and req_id not in self._reqs_to_process
            ):
                still_pending.append(notif)
                continue

            # NOTE: `tp_ratio` is the opposite when swapping local<>remote
            n_consumers = int(tp_size)
            tp_ratio = self.transfer_topo.tp_ratio(n_consumers)

            # Number of reads *per producer* to wait for.
            # When remote D TP > local P TP we expect `tp_ratio` reads.
            consumers_per_producer = (
                -tp_ratio if n_consumers > self.world_size else 1
            )

            self.consumer_notification_counts_by_req[req_id] += 1
            # Wait all consumers (D) to be done reading before freeing.
            if (
                self.consumer_notification_counts_by_req[req_id]
                == consumers_per_producer
            ):
                notified_req_ids.add(req_id)
                del self.consumer_notification_counts_by_req[req_id]
                self._reqs_to_process.remove(req_id)
                self._reqs_to_send.pop(req_id, None)

        if still_pending:
            logger.debug(
                "Buffered %d early NIXL notification(s) for next step",
                len(still_pending),
            )
            self._pending_notifs = still_pending
        return notified_req_ids'''


def patch_heartbeat_chunking():
    path = find_nixl_base_worker()
    if path is None:
        print("NIXL base_worker.py not found; heartbeat patch skipped.")
        return

    backup = path + ".heartbeat_chunking.bak"
    with open(path) as f:
        src = f.read()

    if HEARTBEAT_PATCH_VERSION in src:
        print(f"NIXL heartbeat chunking v2 patch already present: {path}")
        return

    if HEARTBEAT_MARKER in src:
        if not os.path.exists(backup):
            print(
                "ERROR: Cannot upgrade NIXL heartbeat chunking patch; "
                f"pristine backup is missing: {backup}"
            )
            return
        with open(backup) as f:
            src = f.read()
        print(f"Upgrading NIXL heartbeat chunking patch from pristine: {backup}")

    method_start = src.find("    def _send_heartbeats(")
    if method_start < 0:
        print("ERROR: Could not find _send_heartbeats in NIXL base_worker.py.")
        return
    next_method = src.find("\n    def ", method_start + 1)
    if next_method < 0:
        print("ERROR: Could not find end of NIXL _send_heartbeats method.")
        return

    patched_src = (
        src[:method_start] + NEW_SEND_HEARTBEATS + src[next_method:]
    )
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    with open(path, "w") as f:
        f.write(patched_src)

    print(
        "Patched NIXL _send_heartbeats with opt-in 256-byte wire limit "
        f"({HEARTBEAT_WIRE_OVERHEAD_BYTES}-byte framing reserve; "
        f"enable with {HEARTBEAT_MARKER}=256): {path}"
    )


def patch_kv_lease_duration():
    raw_seconds = os.environ.get(LEASE_ENV, "0").strip()
    try:
        lease_seconds = int(raw_seconds or "0")
    except ValueError:
        print(f"ERROR: Invalid {LEASE_ENV}={raw_seconds!r}; lease patch skipped.")
        return

    if lease_seconds <= 0:
        return

    scheduler_path = find_nixl_base_scheduler()
    worker_path = find_nixl_base_worker()
    if scheduler_path is None or worker_path is None:
        print("ERROR: NIXL lease source not found; lease patch skipped.")
        return

    scheduler_backup = scheduler_path + ".kv_lease_duration.bak"
    worker_backup = worker_path + ".kv_lease_duration.bak"

    def pristine_source(path, backup):
        with open(path) as f:
            source = f.read()
        if LEASE_PATCH_MARKER in source:
            if not os.path.exists(backup):
                raise RuntimeError(
                    f"cannot update lease patch; pristine backup missing: {backup}"
                )
            with open(backup) as f:
                source = f.read()
        return source

    try:
        scheduler_src = pristine_source(scheduler_path, scheduler_backup)
        worker_src = pristine_source(worker_path, worker_backup)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    scheduler_anchor = (
        "        self._kv_lease_duration: int = (\n"
        "            vllm_config.kv_transfer_config.get_from_extra_config(\n"
        '                "kv_lease_duration", 30\n'
        "            )\n"
        "        )\n"
    )
    worker_anchor = (
        "        kv_lease_duration: int = "
        "vllm_config.kv_transfer_config.get_from_extra_config(\n"
        '            "kv_lease_duration", 30\n'
        "        )\n"
    )
    if scheduler_anchor not in scheduler_src:
        print("ERROR: Could not find NIXL scheduler KV lease anchor.")
        return
    if worker_anchor not in worker_src:
        print("ERROR: Could not find NIXL worker lease-extension anchor.")
        return

    scheduler_patched = scheduler_src.replace(
        scheduler_anchor,
        f"        # {LEASE_PATCH_MARKER}: mixed-topology opt-in.\n"
        "        self._kv_lease_duration: int = (\n"
        "            vllm_config.kv_transfer_config.get_from_extra_config(\n"
        f'                "kv_lease_duration", {lease_seconds}\n'
        "            )\n"
        "        )\n",
        1,
    )
    worker_patched = worker_src.replace(
        worker_anchor,
        f"        # {LEASE_PATCH_MARKER}: mixed-topology opt-in.\n"
        "        kv_lease_duration: int = "
        "vllm_config.kv_transfer_config.get_from_extra_config(\n"
        f'            "kv_lease_duration", {lease_seconds}\n'
        "        )\n",
        1,
    )

    if not os.path.exists(scheduler_backup):
        shutil.copy2(scheduler_path, scheduler_backup)
    if not os.path.exists(worker_backup):
        shutil.copy2(worker_path, worker_backup)
    with open(scheduler_path, "w") as f:
        f.write(scheduler_patched)
    with open(worker_path, "w") as f:
        f.write(worker_patched)

    print(
        f"Patched NIXL KV lease to {lease_seconds}s via {LEASE_ENV}: "
        f"{scheduler_path}, {worker_path}"
    )


def main():
    patch_kv_lease_duration()
    patch_heartbeat_chunking()

    path = find_nixl_connector()
    if path is None:
        print("ERROR: Could not find nixl_connector.py in the vLLM installation.")
        sys.exit(1)

    print(f"Found nixl_connector.py at: {path}")

    with open(path) as f:
        src = f.read()

    if INIT_MARKER in src:
        print("Patch already applied (found _pending_notifs). Nothing to do.")
        return

    init_anchor = "        self._reqs_to_process: set[ReqId] = set()\n"
    if init_anchor not in src:
        print("ERROR: Could not find _reqs_to_process init anchor.")
        print("       The vLLM version may differ from expected. Manual patch required.")
        sys.exit(1)
    src = src.replace(
        init_anchor,
        init_anchor
        + "\n"
        + "        # Buffer for NIXL notifications that arrived before metadata was\n"
        + "        # processed (race: RDMA completes faster than scheduler->worker IPC).\n"
        + "        self._pending_notifs: list[bytes] = []\n",
        1,
    )
    print("Patched __init__ to add _pending_notifs buffer.")

    method_start = src.find("    def _get_new_notifs(self) -> set[str]:\n")
    if method_start < 0:
        print("ERROR: Could not find expected _get_new_notifs method.")
        print("       The vLLM version may differ from expected. Manual patch required.")
        sys.exit(1)
    next_method = src.find("\n    def ", method_start + 1)
    if next_method < 0:
        print("ERROR: Could not find end of _get_new_notifs method.")
        print("       The vLLM version may differ from expected. Manual patch required.")
        sys.exit(1)
    src = src[:method_start] + NEW_GET_NEW_NOTIFS + src[next_method:]

    print("Patched _get_new_notifs to buffer early notifications.")

    backup = path + ".bak"
    shutil.copy2(path, backup)
    print(f"Backup saved to: {backup}")

    with open(path, "w") as f:
        f.write(src)

    print("Patch applied successfully.")
    print("Restart the prefill server(s) for changes to take effect.")


if __name__ == "__main__":
    main()
