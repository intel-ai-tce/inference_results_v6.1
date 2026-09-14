# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

file(GLOB SOURCE_FILES
     *.h
     *.cpp
     *.cu
     *.cuh
)

pybind11_add_module(nve ${SOURCE_FILES})

if(NVE_ROCM)
  file(GLOB BINDING_CU *.cu)
  set_source_files_properties(${BINDING_CU} PROPERTIES LANGUAGE HIP)
  # nve-common already brings in hip::device/hip::host (HIP driver API).
  target_link_libraries(nve PUBLIC nve-common hip::device)
else()
  target_link_libraries(nve PUBLIC nve-common -lcuda)
endif()
