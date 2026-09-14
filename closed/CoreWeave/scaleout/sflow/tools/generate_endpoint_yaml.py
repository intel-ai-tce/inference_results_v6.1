#!/usr/bin/env python3
"""
Render endpoint.yaml with a dynamic list of frontend endpoint URLs.

Reads an existing endpoint.yaml template, replaces ``endpoint_config.endpoints``
with the provided URL list, and writes the result. Other ``${VAR}`` placeholders
in the template are preserved verbatim so the endpoints repo's own env-var
interpolation (resolve_env_vars in inference_endpoint.config.utils) still runs
at ``benchmark from-config`` load time.

Usage:
    python3 generate_endpoint_yaml.py \\
        --input  /work/.../endpoint.yaml \\
        --frontend-urls 10.0.0.1:8000,10.0.0.2:8000 \\
        --output /tmp/endpoint_resolved.yaml \\
        [--scheme http]

The comma-separated --frontend-urls form matches generate_master_yaml.py's
--ctx-urls / --gen-urls convention, so it composes naturally with the
``paste -sd ','`` pattern used to flatten the FRONTEND_URLS artifact.
"""
import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Source endpoint.yaml template")
    p.add_argument("--frontend-urls", required=True,
                   help="Comma-separated frontend addresses, e.g. 10.0.0.1:8000,10.0.0.2:8000. "
                        "Each entry may include a scheme; otherwise --scheme is prepended.")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination endpoint.yaml")
    p.add_argument("--scheme", default="http", choices=["http", "https"],
                   help="Scheme to prepend to bare ip:port entries (default: http)")
    return p.parse_args()


def parse_urls(raw: str, scheme: str) -> list[str]:
    """Split CSV, strip blanks, prepend scheme if absent."""
    urls = []
    for entry in raw.split(","):
        e = entry.strip()
        if not e:
            continue
        if not (e.startswith("http://") or e.startswith("https://")):
            e = f"{scheme}://{e}"
        urls.append(e)
    if not urls:
        print("ERROR: no URLs provided in --frontend-urls", file=sys.stderr)
        sys.exit(1)
    return urls


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        print(f"ERROR: expected YAML mapping at root, got {type(config).__name__}", file=sys.stderr)
        sys.exit(1)

    ep_cfg = config.get("endpoint_config")
    if not isinstance(ep_cfg, dict):
        print("ERROR: input YAML is missing 'endpoint_config' mapping", file=sys.stderr)
        sys.exit(1)

    ep_cfg["endpoints"] = parse_urls(args.frontend_urls, args.scheme)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Written: {args.output}")
    print(f"endpoints ({len(ep_cfg['endpoints'])}):")
    for u in ep_cfg["endpoints"]:
        print(f"  - {u}")


if __name__ == "__main__":
    main()
