"""Standalone DLRM-v3 accuracy scorer (streaming, print-based output).

Mirrors benchmarks/accuracy.py but (1) uses the installed module layout
(`configs`/`utils` under mlcommons-inference/recommendation/dlrm_v3) instead of the
stale `generative_recommenders.dlrm_v3.*` paths, (2) streams the accuracy log
line-by-line (LoadGen writes one JSON object per line) to avoid loading the whole
multi-GB array into memory, and (3) prints results (torch/torchrec set the root
logger to WARNING, which suppresses logging.info output).
"""

import argparse
import json
import sys

import numpy as np
import torch
from configs import get_hstu_configs
from utils import MetricsLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to mlperf_log_accuracy.json")
    args = parser.parse_args()

    print(f"[score] parsing {args.path}", flush=True)
    hstu_config = get_hstu_configs(dataset="sampled-streaming-100b")
    metrics = MetricsLogger(
        multitask_configs=hstu_config.multitask_configs,
        batch_size=1,
        window_size=3000,
        device=torch.device("cpu"),
        rank=0,
    )

    n = 0
    with open(args.path, "r") as f:
        for raw in f:
            line = raw.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            result = json.loads(line)
            data = np.frombuffer(bytes.fromhex(result["data"]), np.float32)
            num_candidates = int(data[-1])
            assert len(data) == 3 + num_candidates * 3, (len(data), num_candidates)
            preds = torch.from_numpy(data[2:2 + num_candidates].copy())
            labels = torch.from_numpy(data[2 + num_candidates: 2 + num_candidates * 2].copy())
            weights = torch.from_numpy(data[2 + num_candidates * 2: 2 + num_candidates * 3].copy())
            metrics.update(
                predictions=preds.view(1, -1),
                labels=labels.view(1, -1),
                weights=weights.view(1, -1),
                num_candidates=torch.tensor([num_candidates]),
            )
            n += 1
            if n % 50000 == 0:
                print(f"[score] processed {n} entries", flush=True)

    print(f"[score] total entries: {n}", flush=True)
    print("==== ACCURACY METRICS ====", flush=True)
    for k, v in metrics.compute().items():
        try:
            val = v.item() if hasattr(v, "item") else v
        except Exception:
            val = v
        print(f"{k}: {val}", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
