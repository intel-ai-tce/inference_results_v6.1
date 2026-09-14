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
from code.ops.harness import LWISExecutableHarness

from .builder import (ResNet50EngineBuilder,
                      ResNet50BackboneEngineBuilder,
                      ResNet50TopKEngineBuilder,
                      ResNet50PreRes2EngineBuilder,
                      ResNet50PreRes3EngineBuilder,
                      ResNet50Res2_3EngineBuilder,
                      ResNet50Res3EngineBuilder,
                      ResNet50PostRes3EngineBuilder)
from .constants import ResNet50Component as Component


COMPONENT_MAP = {
    Component.ResNet50: ResNet50EngineBuilder,
    Component.Backbone: ResNet50BackboneEngineBuilder,
    Component.TopK: ResNet50TopKEngineBuilder,
    Component.PreRes2: ResNet50PreRes2EngineBuilder,
    Component.PreRes3: ResNet50PreRes3EngineBuilder,
    Component.Res2Res3: ResNet50Res2_3EngineBuilder,
    Component.Res3: ResNet50Res3EngineBuilder,
    Component.PostRes3: ResNet50PostRes3EngineBuilder,
}


VALID_COMPONENT_SETS = {"gpu": [{Component.ResNet50},
                                {Component.PreRes2, Component.Res2Res3, Component.Res3},
                                {Component.PreRes3, Component.Res3, Component.PostRes3}],
                        "dla": [{Component.ResNet50},
                                {Component.Backbone, Component.TopK}]}


BenchmarkHarnessOp = LWISExecutableHarness
