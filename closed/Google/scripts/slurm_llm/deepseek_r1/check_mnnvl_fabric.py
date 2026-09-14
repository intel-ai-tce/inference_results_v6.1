#!/usr/bin/env python3
"""Validate that all Slurm ranks belong to one healthy MNNVL fabric."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess

from mpi4py import MPI


def _field(report: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.*?)\s*$", report, re.MULTILINE)
    return match.group(1) if match else ""


def _fabric_report() -> str:
    filtered = subprocess.run(
        ["nvidia-smi", "-q", "-d", "FABRIC"],
        capture_output=True,
        text=True,
        check=False,
    )
    if filtered.returncode == 0:
        return filtered.stdout

    # Some driver branches expose fabric fields in the full query but do not
    # accept FABRIC as a -d selector. Fall back without weakening validation.
    full_report = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    if "Fabric" not in full_report:
        raise RuntimeError(
            "nvidia-smi did not expose fabric information; "
            f"filtered query error: {filtered.stderr.strip()}"
        )
    return full_report


def main() -> None:
    comm = MPI.COMM_WORLD
    report = _fabric_report()
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()

    row = {
        "rank": comm.rank,
        "host": socket.gethostname(),
        "gpu_uuid": gpu_uuid,
        "state": _field(report, "State"),
        "status": _field(report, "Status"),
        "clique_id": _field(report, "Clique ID"),
        "cluster_uuid": _field(report, "Cluster UUID"),
    }
    print("MNNVL_RANK " + json.dumps(row, sort_keys=True), flush=True)

    rows = comm.gather(row, root=0)
    error = ""
    if comm.rank == 0:
        expected_ranks = int(os.environ.get("EXPECTED_RANKS", "8"))
        expected_nodes = int(os.environ.get("EXPECTED_NODES", "2"))
        expected_ranks_per_node = int(os.environ.get("EXPECTED_RANKS_PER_NODE", "4"))

        host_counts: dict[str, int] = {}
        for item in rows:
            host_counts[item["host"]] = host_counts.get(item["host"], 0) + 1

        problems = []
        if len(rows) != expected_ranks:
            problems.append(f"expected {expected_ranks} ranks, found {len(rows)}")
        if len(host_counts) != expected_nodes:
            problems.append(f"expected {expected_nodes} nodes, found {len(host_counts)}")
        if any(count != expected_ranks_per_node for count in host_counts.values()):
            problems.append(
                f"expected {expected_ranks_per_node} ranks per node, found {host_counts}"
            )

        gpu_uuids = [item["gpu_uuid"] for item in rows]
        if any(not uuid for uuid in gpu_uuids):
            problems.append("one or more ranks did not report a GPU UUID")
        elif len(set(gpu_uuids)) != expected_ranks:
            problems.append(
                f"expected {expected_ranks} distinct GPU UUIDs, found {len(set(gpu_uuids))}"
            )

        host_gpu_uuids: dict[str, set[str]] = {}
        for item in rows:
            host_gpu_uuids.setdefault(item["host"], set()).add(item["gpu_uuid"])
        if any(
            len(uuids) != expected_ranks_per_node
            for uuids in host_gpu_uuids.values()
        ):
            problems.append(
                f"expected {expected_ranks_per_node} distinct GPUs per node, "
                f"found {dict((host, len(uuids)) for host, uuids in host_gpu_uuids.items())}"
            )

        bad_fabric = [
            item
            for item in rows
            if item["state"].lower() != "completed" or item["status"].lower() != "success"
        ]
        if bad_fabric:
            problems.append(f"fabric not ready on ranks {[item['rank'] for item in bad_fabric]}")

        reported_cluster_uuids = [item["cluster_uuid"] for item in rows if item["cluster_uuid"]]
        cluster_uuids = set(reported_cluster_uuids)
        if reported_cluster_uuids and len(reported_cluster_uuids) != len(rows):
            problems.append("MNNVL cluster UUID was reported for only some ranks")
        elif len(cluster_uuids) > 1:
            problems.append(f"expected one MNNVL cluster UUID, found {sorted(cluster_uuids)}")

        reported_clique_ids = [item["clique_id"] for item in rows if item["clique_id"]]
        clique_ids = set(reported_clique_ids)
        if reported_clique_ids and len(reported_clique_ids) != len(rows):
            problems.append("MNNVL clique ID was reported for only some ranks")
        elif len(clique_ids) > 1:
            problems.append(f"expected one MNNVL clique ID, found {sorted(clique_ids)}")

        if problems:
            error = "; ".join(problems)
        else:
            print(
                "PASS: healthy MNNVL fabric across "
                f"{expected_nodes} nodes and {expected_ranks} GPUs; "
                f"cluster_uuid={next(iter(cluster_uuids), 'not-reported')}; "
                f"clique_id={next(iter(clique_ids), 'not-reported')}",
                flush=True,
            )

    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
