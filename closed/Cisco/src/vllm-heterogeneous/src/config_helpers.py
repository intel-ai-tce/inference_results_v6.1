"""Shared helpers for resolving model-level configuration.

Single source of truth for any logic that has to agree across:

  * ``src/resolve_config.py``    -- exports ``MP_ENDPOINTS`` for
                                    ``start_server.sh`` (which builds the
                                    GPU slab geometry and fan-out).
  * ``scripts/lib/servers.sh``   -- exports ``STANDALONE_ADDR`` /
                                    ``STANDALONE_N`` for ``run.sh``'s
                                    banner and SUT count.
  * ``src/sut/standalone.py``    -- the harness-side SUT, which has to
                                    open one ZMQ connection per worker.

Putting the resolution in three places will eventually drift; putting it
here lets all three callers agree by construction.
"""

from __future__ import annotations

import copy
from typing import List


def select_model_profile(
    model_cfg: dict,
    profile: str | None = None,
    scenario: str | None = None,
) -> dict:
    if not profile:
        return model_cfg
    profiles = model_cfg.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("model config has no profiles section")
    selected = profiles.get(profile)
    if not isinstance(selected, dict):
        raise ValueError(f"model config has no {profile!r} profile")
    scenario_key = str(scenario or "").lower()
    if scenario_key and isinstance(selected.get(scenario_key), dict):
        selected = selected[scenario_key]
    if not isinstance(selected, dict) or "benchmark_name" not in selected:
        if scenario_key:
            raise ValueError(
                f"profile {profile!r} has no complete {scenario_key!r} configuration"
            )
        raise ValueError(f"profile {profile!r} requires an MLPerf scenario")
    return copy.deepcopy(selected)


def resolve_standalone_endpoints(
    model_cfg: dict,
    hardware: str | None = None,
    scenario: str | None = None,
) -> List[str]:
    """Return the resolved list of ``"host:port"`` endpoints for the
    ``standalone`` backend, auto-expanding a single-entry base when
    present.

    Resolution rules
    ----------------

    1. **Multi-entry list:** if ``servers.standalone`` has more than one
       entry, return it as-is. This preserves backward compatibility
       with explicit multi-host setups (we don't currently use them for
       standalone, but the schema allows it).

    2. **Single-entry base:** if the list has exactly one entry, treat
       that as a base ``host:port``. Compute::

           n_engines = harness_config.device_count // (tp_size * dp_size)

       and generate ``n_engines`` endpoints with ports incrementing from
       the base port. Example: base ``worker-host:8200`` with
       ``tp_size=2 dp_size=1 device_count=8`` yields::

           ["worker-host:8200", "worker-host:8201",
            "worker-host:8202", "worker-host:8203"]

    3. **Empty list:** error. The base is required even when
       auto-expansion is desired.

    Parameters come from these locations in the model YAML. If ``hardware``
    is provided, ``standalone.<hardware>.tp_size`` and
    ``standalone.<hardware>.dp_size`` override the top-level standalone
    values. If ``scenario`` is provided, its nested hardware override is
    applied as well, so endpoint expansion matches ``src/resolve_config.py``.

    ===============================  ==========================
    YAML key                         Default
    ===============================  ==========================
    ``standalone.tp_size``           ``1``
    ``standalone.dp_size``           ``1``
    ``harness_config.device_count``  ``8``
    ``servers.standalone``           required, list[str]
    ===============================  ==========================

    Raises ``ValueError`` (with a user-actionable message) on:

    * empty ``servers.standalone``
    * base entry not ``host:port`` shape
    * ``device_count`` not divisible by ``tp_size * dp_size``
    """
    servers = (model_cfg.get("servers") or {}).get("standalone", []) or []
    if isinstance(servers, str):
        servers = servers.split()
    servers = [str(e).strip() for e in servers if str(e) and str(e).strip()]

    if not servers:
        raise ValueError(
            "servers.standalone is empty. Add at least one host:port "
            "entry (e.g. 'worker-host:8200'); the engine count auto-resolves "
            "to harness_config.device_count // (standalone.tp_size * "
            "standalone.dp_size)."
        )

    if len(servers) > 1:
        return servers

    base = servers[0]
    if ":" not in base:
        raise ValueError(
            f"servers.standalone base must be 'host:port', got {base!r}."
        )
    host, port_str = base.rsplit(":", 1)
    try:
        port_base = int(port_str)
    except ValueError as exc:
        raise ValueError(
            f"servers.standalone[0] port must be an integer, got "
            f"{port_str!r}."
        ) from exc

    sect = model_cfg.get("standalone") or {}
    hw_sect = {}
    if hardware:
        maybe_hw = sect.get(hardware)
        if isinstance(maybe_hw, dict):
            hw_sect = maybe_hw
    if scenario:
        scenario_overrides = hw_sect.get(scenario.lower())
        if isinstance(scenario_overrides, dict):
            hw_sect = {
                key: value for key, value in hw_sect.items()
                if not isinstance(value, dict)
            }
            hw_sect.update(scenario_overrides)
    try:
        tp = int(hw_sect.get("tp_size", sect.get("tp_size", 1)))
        dp = int(hw_sect.get("dp_size", sect.get("dp_size", 1)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"standalone.tp_size / dp_size must be integers; got "
            f"tp_size={sect.get('tp_size')!r} dp_size={sect.get('dp_size')!r}."
        ) from exc
    slab = max(1, tp * dp)

    hc = model_cfg.get("harness_config") or {}
    try:
        device_count = int(hc.get("device_count", 8))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"harness_config.device_count must be an integer, got "
            f"{hc.get('device_count')!r}."
        ) from exc

    if device_count % slab != 0:
        raise ValueError(
            f"harness_config.device_count={device_count} is not divisible "
            f"by tp_size*dp_size={slab}. Adjust standalone.tp_size / "
            f"dp_size, or change harness_config.device_count."
        )

    n = device_count // slab
    return [f"{host}:{port_base + i}" for i in range(n)]
