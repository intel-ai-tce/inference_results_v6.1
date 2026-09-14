# Copyright (c) 2025 Intel Corporation
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
# =============================================================================

"""Shared configuration printer for the MLPerf run entrypoints.

Prints the resolved run configuration and per-component endpoints, highlighting
values that differ from the argparse default (bold/cyan). Color is emitted when
stdout is a TTY or FORCE_COLOR is set; disable with NO_COLOR (so tee'd run.log
stays clean)."""

import os
import sys

_BOLD = "\033[1m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _use_color():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def arg_defaults(parser):
    """Map of dest -> default value for every argparse action."""
    return {a.dest: a.default for a in parser._actions if a.dest != "help"}


def _fmt(value, is_default, color):
    if value is None or value == "":
        value = "(unset)"
    if is_default or not color:
        return str(value)
    return f"{_BOLD}{_CYAN}{value}{_RESET}"


def print_config(args, parser, title="Run configuration", endpoints=None,
                 fields=None, extra=None):
    """Print args as key=value, bolding non-defaults.

    endpoints: list of (label, url_attr, model_attr) tuples printed in a
               dedicated section with their resolved URL + model.
    fields:    optional ordered list of arg dests to show (default: all).
    extra:     optional list of (label, value) pairs for values that are not
               argparse args (e.g. env-derived settings like TRACE); printed
               after the args, before the endpoints section. None/"" -> (unset).
    """
    color = _use_color()
    defaults = arg_defaults(parser)
    vals = vars(args)
    keys = fields if fields is not None else sorted(vals)

    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

    # Strategy-specific knobs only matter for their strategy; hide otherwise.
    strategy = vals.get("retrieval_strategy")
    skip = set()
    if strategy != "top_p":
        skip.add("retrieval_top_p")
    if strategy != "relative":
        skip.add("relative_ratio")

    for k in keys:
        if k not in vals or k in skip:
            continue
        v = vals[k]
        is_default = defaults.get(k, object()) == v
        tag = "" if is_default else "  *"
        print(f"  {k:<26} {_fmt(v, is_default, color)}{tag}")

    if extra:
        for label, value in extra:
            # env-derived, so treat non-empty as a set (non-default) value
            print(f"  {label:<26} {_fmt(value, value in (None, ''), color)}")

    if endpoints:
        print("-" * 72)
        print("  Endpoints")
        for label, url_attr, model_attr in endpoints:
            url = vals.get(url_attr)
            model = vals.get(model_attr) if model_attr else None
            u_def = defaults.get(url_attr, object()) == url
            m_def = defaults.get(model_attr, object()) == model
            line = f"  {label:<14} {_fmt(url, u_def, color)}"
            if model is not None:
                line += f"  model={_fmt(model, m_def, color)}"
            print(line)

    if color:
        print(f"{_DIM}  (* / bold = non-default value){_RESET}")
    else:
        print("  (* = non-default value)")
    print("=" * 72 + "\n")

    # Flush so the config appears BEFORE the SUT's (stderr) logging under `tee`
    sys.stdout.flush()
