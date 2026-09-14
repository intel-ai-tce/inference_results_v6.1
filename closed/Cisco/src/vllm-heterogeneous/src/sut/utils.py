import array
import logging
import os
from datetime import datetime

import numpy as np
import mlperf_loadgen as lg

log = logging.getLogger(__name__)


def get_visible_device_indices(device_count=8):
    for name in ("HARNESS_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                 "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.getenv(name)
        if val is not None:
            return tuple(int(x) for x in val.split(",") if x.strip())
    return tuple(range(device_count))


def check_parallelism_configuration(instance_count, dp, tp, pp, dc):
    if (instance_count * dp * tp * pp) != dc:
        msg = (f"EDP={instance_count} x DP={dp} x TP={tp} x PP={pp} "
               f"!= {dc} GPUs")
        raise ValueError(msg)


def _send_response(sample_id, token_ids, first_token):
    response_array = array.array("B", np.array(token_ids, np.int32).tobytes())
    bi = response_array.buffer_info()
    response = [lg.QuerySampleResponse(sample_id, bi[0], bi[1], len(token_ids))]
    if first_token:
        lg.FirstTokenComplete(response)
    else:
        lg.QuerySamplesComplete(response)


def create_response_and_send_complete(sample_id, token_ids):
    _send_response(sample_id, token_ids, False)


def create_response_and_send_first_token(sample_id, token_ids):
    _send_response(sample_id, token_ids, True)
