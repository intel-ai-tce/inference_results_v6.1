# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from nvmitten.configurator import Field

from code.resnet50.tensorrt.fields import disable_beta1_smallk


__doc__ = """Retinanet Fields"""


nms_type = Field(
    "nms_type",
    description="Select EfficientNMS/NMSOptPlugin/RetinaNetNMSPVATRT plugin for RetinaNet NMS layer",
    argparse_opts={"choices": ['efficientnms', 'nmsopt', 'nmspva']})
