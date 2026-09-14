#!/usr/bin/env python3
"""mpi_log_parser.py

Split a combined Slurm/PMIx MPI log into separate per-rank log files.

Each line that starts with a pattern like

    " 5: message text..."

is interpreted as coming from MPI rank 5.  The characters up to and
including the first colon are removed, the remainder of the line is
written to a file named ``rank_<rank>.log`` in the output directory.

Lines that do not match the pattern are ignored (or could optionally be
written to an ``unranked.log`` file if desired).

Usage
-----
    python mpi_log_parser.py --log_file path/to/combined.log \
                             --output_directory ./mpi_logs
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, TextIO

# Regex to match lines of the form " 12: actual text"
# Captures the rank number and the rest of the line.
_MPI_LINE_RE = re.compile(r"^\s*(\d+):\s*(.*)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a combined MPI log into per-rank logs.")
    parser.add_argument("--log_file", required=True, help="Path to the combined log file to split.")
    parser.add_argument(
        "--output_directory",
        "-o",
        required=True,
        help="Directory to write per-rank logs into (created if missing).",
    )
    return parser.parse_args()


def split_log(log_path: Path, out_dir: Path) -> None:
    """Read *log_path* and write per-rank logs into *out_dir*.

    Parameters
    ----------
    log_path: Path
        Path to the combined MPI log.
    out_dir: Path
        Directory where per-rank files will be created.
    """
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a dict of open file handles so we do not reopen per line.
    open_files: Dict[int, TextIO] = {}

    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as infile:
            for line in infile:
                match = _MPI_LINE_RE.match(line)
                if match is None:
                    # Skip lines that do not contain a rank prefix.
                    continue

                rank = int(match.group(1))
                message = match.group(2)

                # Lazily open the output file for this rank.
                if rank not in open_files:
                    fname = out_dir / f"rank_{rank}.log"
                    open_files[rank] = fname.open("w", encoding="utf-8")

                open_files[rank].write(message + "\n")
    finally:
        # Ensure all files are closed.
        for f in open_files.values():
            f.close()


def main() -> None:
    args = _parse_args()
    split_log(Path(args.log_file).expanduser(), Path(args.output_directory).expanduser())
    print(f"Logs written to {args.output_directory}")


if __name__ == "__main__":
    main()
