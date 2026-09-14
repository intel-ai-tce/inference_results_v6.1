import logging
from datetime import datetime
import numpy as np
import array
import os
import mlperf_loadgen as lg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

def check_parallelism_configuration(instance_count, dp, tp, pp, dc):
    if (instance_count * dp * tp * pp) != dc:
        error_message = f"EDP={instance_count} x DP={dp} x TP={tp} x PP={pp} are not compatible with {dc} GPUs"
        log.error(error_message)
        raise ValueError(error_message)

def get_visible_device_indices(device_count=8):
    visible_devices = None # means: all devices visible
    for name in ("HARNESS_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if (val := os.getenv(name)) is not None:
            visible_devices = tuple(int(x) for x in val.split(",") if x.strip())
            break

    if visible_devices:
        if len(visible_devices) < device_count:
            raise ValueError(f"{visible_devices=} < {device_count=}")
    else:
        visible_devices = tuple(range(device_count))
    return visible_devices

def create_response_array(sample_id, token_ids, is_first_token):
    response_array = array.array(
        "B", np.array(token_ids, np.int32).tobytes()
    )
    bi = response_array.buffer_info()
    response = [
        lg.QuerySampleResponse(
            sample_id, bi[0], bi[1], len(token_ids)
        )
    ]

    if is_first_token:
        lg.FirstTokenComplete(response)
    else:
        lg.QuerySamplesComplete(response)

def create_response_and_send_complete(sample_id, token_ids):
    create_response_array(sample_id, token_ids, False)

def create_response_and_send_first_token(sample_id, token_ids):
    create_response_array(sample_id, token_ids, True)

class ProgressReporter:
    @staticmethod
    def format_timestamp():
        now = datetime.now()
        now_mon = f"{now.month:02d}"
        now_day = f"{now.day:02d}"
        now_hr = f"{now.hour:02d}"
        now_min = f"{now.minute:02d}"
        now_sec = f"{now.second:02d}"
        return f"{now.year}-{now_mon}-{now_day} {now_hr}:{now_min}:{now_sec} INFO     SUT - "
