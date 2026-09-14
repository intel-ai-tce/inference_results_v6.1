# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared virtual environment utilities for accuracy checkers and compliance tests."""

from __future__ import annotations

import hashlib
import fcntl
import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_venv_ready(venv_path: Path, requirements_file: Path) -> Path:
    """Ensure a virtual environment exists with required dependencies installed.

    The venv is considered ready if:
    1. The venv python3 binary exists
    2. A marker file exists with the hash of the requirements file (to detect changes)

    Args:
        venv_path: Path where the venv should be created/verified.
        requirements_file: Path to requirements.txt file for pip install.

    Returns:
        Path to the venv directory.

    Raises:
        FileNotFoundError: If requirements_file doesn't exist.
        RuntimeError: If venv creation or pip install fails.
    """
    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    venv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = venv_path.parent / f".{venv_path.name}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return _ensure_venv_ready_locked(venv_path, requirements_file)


def _ensure_venv_ready_locked(venv_path: Path, requirements_file: Path) -> Path:
    """Validate or create a venv while its inter-process lock is held."""

    venv_python = venv_path / "bin" / "python3"
    marker_file = venv_path / ".requirements_hash"
    requirements_hash = _environment_hash(requirements_file)

    # Check if venv is ready
    if venv_path.exists():
        if not venv_python.exists():
            logging.warning(f"Venv at {venv_path} is corrupted (python not found). Recreating...")
            shutil.rmtree(venv_path)
        elif not marker_file.exists() or marker_file.read_text().strip() != requirements_hash:
            logging.warning(f"Requirements have changed. Recreating venv at {venv_path}...")
            shutil.rmtree(venv_path)
        else:
            logging.info(f"Venv ready at {venv_path}")
            return venv_path

    # Create new venv
    _create_venv(venv_path, requirements_file, requirements_hash)
    return venv_path


def _environment_hash(requirements_file: Path) -> str:
    """Hash requirements and copied local projects used by the venv."""

    sha256 = hashlib.sha256()
    sha256.update(requirements_file.read_bytes())
    for source in _editable_sources(requirements_file):
        resolved = source.resolve()
        sha256.update(source.name.encode())
        for path in sorted(resolved.rglob("*")):
            relative = path.relative_to(resolved)
            if any(part in {".git", "__pycache__"} for part in relative.parts):
                continue
            if not path.is_file() or path.suffix == ".pyc":
                continue
            sha256.update(relative.as_posix().encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    sha256.update(chunk)
    return sha256.hexdigest()


def _editable_sources(requirements_file: Path) -> tuple[Path, ...]:
    sources = []
    for line in requirements_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = shlex.split(stripped)
        if len(parts) == 2 and parts[0] in {"-e", "--editable"}:
            source = Path(parts[1])
            if not source.is_dir():
                raise FileNotFoundError(
                    f"Editable accuracy dependency not found: {source}"
                )
            sources.append(source)
    return tuple(sources)


def _create_venv(venv_path: Path, requirements_file: Path, requirements_hash: str) -> None:
    """Create a new venv and install requirements."""
    logging.info(f"Creating venv at {venv_path}...")
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_path)], check=True)

    private_requirements, local_projects = _materialize_private_requirements(
        venv_path, requirements_file, requirements_hash
    )
    logging.info(f"Installing requirements from {private_requirements}...")
    pip_path = venv_path / "bin" / "pip"
    result = subprocess.run(
        [str(pip_path), "install", "-r", str(private_requirements)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.error(f"Failed to install requirements: {result.stderr}")
        raise RuntimeError(
            f"Failed to install requirements from {requirements_file}. "
            f"Please manually run: {pip_path} install -r {requirements_file}\n"
            f"Error: {result.stderr}"
        )

    if local_projects:
        logging.info("Installing copied local evaluator projects without dependencies...")
        result = subprocess.run(
            [str(pip_path), "install", "--no-deps", *(str(path) for path in local_projects)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.error(f"Failed to install local evaluator projects: {result.stderr}")
            raise RuntimeError(
                "Failed to install copied local evaluator projects with --no-deps. "
                f"Error: {result.stderr}"
            )

    # Write marker file with requirements hash
    marker_file = venv_path / ".requirements_hash"
    marker_file.write_text(requirements_hash)
    logging.info(f"Successfully created venv at {venv_path}")


def _materialize_private_requirements(
    venv_path: Path,
    requirements_file: Path,
    requirements_hash: str,
) -> tuple[Path, tuple[Path, ...]]:
    """Copy editable projects and exclude their optional runtime dependencies."""

    source_root = venv_path.parent / f".{venv_path.name}-sources-{requirements_hash[:12]}"
    generated = venv_path.parent / f".{venv_path.name}-requirements-{requirements_hash[:12]}.txt"
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True)

    rendered = []
    local_projects = []
    for index, line in enumerate(requirements_file.read_text().splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            rendered.append(line)
            continue
        parts = shlex.split(stripped)
        if len(parts) != 2 or parts[0] not in {"-e", "--editable"}:
            rendered.append(line)
            continue

        source = Path(parts[1])
        if not source.is_dir():
            raise FileNotFoundError(f"Editable accuracy dependency not found: {source}")
        destination = source_root / f"{index:02d}-{source.name}"
        shutil.copytree(source, destination)
        if source.name == "prm800k":
            setup_py = destination / "setup.py"
            if setup_py.is_file():
                setup_py.write_text(
                    "".join(
                        line
                        for line in setup_py.read_text().splitlines(keepends=True)
                        if line.strip() != "import numpy"
                    )
                )
        rendered.append(f"# installed separately with --no-deps: {source.name}")
        local_projects.append(destination)

    generated.write_text("\n".join(rendered) + "\n")
    return generated, tuple(local_projects)
