from inference_harness.backends.base import DLRMBackend
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from typing import Dict, List, Optional, Tuple

import os
import torch

from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference

# this implementation couples the embedding table config and hstu config to the backend
# so user cannot switch between using different embedding config and hstu config.


class GenerativeRecommenderBackend(DLRMBackend):
    def __init__(self, model_name: str, perf_mode: str, device_id: List[int]):
        super().__init__(model_name=model_name)
        self.backend = "GR"
        # One GPU per MPI worker: use a single local device index.
        self.device_id = device_id if device_id else [0]
        self.perf_mode = perf_mode

    def shutdown(self) -> None:
        impl = getattr(self, "model_impl", None)
        if impl is None:
            return
        if hasattr(impl, "shutdown"):
            impl.shutdown()
        else:
            impl.predict(None)
        self.model_impl = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def initialize(self, hstu_config: DlrmHSTUConfig, embedding_table_config: Dict[str, EmbeddingConfig]):
        set_is_inference(is_inference=True if self.perf_mode == "performance" else False)
        os.environ["WORLD_SIZE"] = str(len(self.device_id))
        mpi_rank = os.environ.get("DLRM_MPI_LOCAL_RANK", "")
        mpi_world = os.environ.get("DLRM_MPI_WORKER_WORLD", "")
        if mpi_rank != "":
            os.environ.setdefault("DLRM_SPARSE_RANK", mpi_rank)
        if mpi_world != "":
            os.environ.setdefault("DLRM_SPARSE_WORLD", mpi_world)
        if len(self.device_id) == 1:
            gpu_id = self.device_id[0]
            torch.cuda.set_device(gpu_id)
            if os.environ.get("DLRM_SPARSE_GPU", "0") == "1":
                os.environ.setdefault("DLRM_SPARSE_DEVICE", f"cuda:{gpu_id}")
        else:
            gpu_id = 0
        self.model_impl = HSTUModelFamily(
            hstu_config=hstu_config,
            table_config=embedding_table_config,
            # profiler currently turned off
            output_trace=False,
            sparse_quant=False,
        )
        # Plan 04: HSTUModelFamily holds a `dense` sub-object (typically
        # `ModelFamilyDenseSingleWorker`) whose __init__ hardcodes
        #     self.device = torch.device("cuda:0")
        #     torch.cuda.set_device(self.device)
        # That set_device(0) wins over the set_device(gpu_id) above, AND
        # the subsequent `.dense.load(model_path)` does `.to(self.device)`,
        # so for any rank != 0 the dense model loads onto cuda:0 (= the
        # wrong physical GPU under HIP_VISIBLE_DEVICES remap). The sparse
        # sub-object (`ModelFamilySparseDist`) doesn't set a device but
        # inherits from torch.cuda.current_device(), so it follows
        # whichever set_device() was most recent. Fix all of:
        #   * model_impl.device           -> rank's actual cuda:N  (cosmetic)
        #   * model_impl.dense.device     -> rank's actual cuda:N  (REAL FIX)
        #   * torch.cuda.set_device(N)    -> restore current device for
        #                                    sparse and any future allocs
        target_device = torch.device(f"cuda:{gpu_id}")
        self.model_impl.device = target_device
        if hasattr(self.model_impl, "dense") and hasattr(self.model_impl.dense, "device"):
            self.model_impl.dense.device = target_device
        torch.cuda.set_device(gpu_id)

    def load_model(self, checkpoint_path: str):
        if not checkpoint_path:
            return
        self.model_impl.load(checkpoint_path)

    def predict(self, feed: Samples):
        out = self.model_impl.predict(feed)
        if out is None:
            return None, None, None
        preds, labels, weights, _, _ = out
        return preds, labels, weights

    def predict_dummy(self, feed: Samples):
        return self.predict(feed)
