/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <distributed.hpp>
#include <common.hpp>
#include <cuda_support.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>  // getpid/close/syscall (was pulled in transitively by CUDA headers)
#include <cstring>   // std::strerror
#include <filesystem>
#include <cstdlib>   // std::getenv / std::atoll (Plan 14.7e setAccess chunking)
#include <algorithm> // std::min (Plan 14.7e)

#define ERRNO_CHECK(_expr_) \
do { \
  auto res = (_expr_); \
  if (res != 0) { \
    std::ostringstream oss; \
    oss << "Error (" << std::strerror(res) << ") at " << __FILE__ << ":" << __LINE__ << " :: " #_expr_ << std::endl; \
    throw std::runtime_error(oss.str());\
  } \
} while (0);

#define ROUND_UP(n, multiple) \
    (((n) + ((multiple)-1)) - (((n) + ((multiple)-1)) % (multiple)))


// Wrapper for pidfd_open syscall
static int pidfd_open(pid_t pid, unsigned int flags) {
    return static_cast<int>(syscall(SYS_pidfd_open, pid, flags));
}

// Wrapper for pidfd_getfd syscall
static int pidfd_getfd(int pidfd, int targetfd, unsigned int flags) {
    return static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, targetfd, flags));
}

namespace nve {

CUDADistributedBuffer::CUDADistributedBuffer(uint64_t size, std::shared_ptr<DistributedEnv> dist_env) : env_(dist_env) {
  NVE_CHECK_(dist_env != nullptr);
  single_host_ = env_->single_host();
  if (single_host_) {
    init_single_host(size);
  } else {
    init_multi_host(size);
  }
  env_->barrier();
  NVE_IF_DEBUG_(
    std::cout << __FUNCTION__
      << " size " << size
      << ", rank " << env_->rank()
      << ", local_device " << env_->local_device()
      << ", world_size " << env_->world_size()
      << ", num_shards_ " << num_shards_
      << ", shard_size_ " << shard_size_
      << ", total_size_ " << total_size_
      << std::endl
  );
}

CUDADistributedBuffer::~CUDADistributedBuffer() {
  // Plan 14.8 §5 teardown race fix: drain THIS rank's GPU before the collective
  // barrier. The barrier alone is host-side only — it proves every rank reached
  // the destructor, but a peer rank's in-flight gather / cache-fill kernel can
  // still be reading a shard over xGMI when another rank cuMemUnmap()s it, which
  // faults ("Memory access fault by GPU node-N" during teardown). Syncing the
  // device on every rank *before* the barrier guarantees no kernel anywhere is
  // touching any peer shard once unmapping begins. cudaDeviceSynchronize ->
  // hipDeviceSynchronize via nve_hip_compat.hpp (catch-all across all streams).
  NVE_CHECK_(cudaDeviceSynchronize());

  // Make sure all processes got here
  env_->barrier();

  // Unmap each allocation (non-participant ranks mapped nothing — skip)
  if (maps_buffer_) {
    for (size_t i=0 ; i<num_shards_ ; i++) {
      const CUdeviceptr buf_start = reinterpret_cast<CUdeviceptr>(buffer_ + (i * shard_size_));
      NVE_CHECK_(cuMemUnmap(buf_start, shard_size_));
    }
  }

  // Release all allocations

  for (size_t i=0 ; i<all_alloc_handles_.size() ; i++) {
    NVE_CHECK_(cuMemRelease(all_alloc_handles_[i]));
  }

  if (!single_host_ && (all_devices_[env_->rank()] >= 0)) {
    NVE_CHECK_(cuMemRelease(alloc_handle_));
  }

  // Make sure all processes got here
  env_->barrier();

  // Release buffer reservation (non-participant ranks reserved nothing — skip)
  if (maps_buffer_) {
    NVE_CHECK_(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(buffer_), total_size_));
  }
}

size_t CUDADistributedBuffer::get_device_granularity(CUmemAllocationProp prop) {
  size_t granularity = 0;
  prop.location.id = 0;
  NVE_CHECK_(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  // Verify all devices use the same granularity
  for (size_t i=1 ; i<env_->device_count() ; i++) {
    size_t dev_granularity;
    prop.location.id = static_cast<int>(i);
    NVE_CHECK_(cuMemGetAllocationGranularity(&dev_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    NVE_CHECK_(granularity == dev_granularity);
  }

  return granularity;
}

void CUDADistributedBuffer::init_single_host(uint64_t size) {
  const bool root_proc = (env_->rank() == 0);
  NVE_CHECK_(env_->single_host());
  num_shards_ = collect_devices(all_devices_);
  const size_t world_size = env_->world_size();

  // Plan 14.7d (ROCm 9-rank harness): a rank that joins the collective but owns
  // no shard (all_devices_[rank] < 0 — e.g. the LoadGen rank in `mpirun -n 9` =
  // 8 NVE workers + 1 LoadGen) must still drive every MPI collective below (the
  // all_gather/broadcasts/barriers are over COMM_WORLD and would deadlock the
  // workers otherwise) but must NOT do any local VMM: it never reads embeddings,
  // and on ROCm a non-owning context calling cuMemSetAccess for peer devices it
  // never set returns hipErrorInvalidValue ("invalid argument"), which aborts
  // construction mid-flight and wedges the GPU driver. Gate all reserve/import/
  // map/setAccess (+ teardown) on this flag; participants behave exactly as
  // before (so the np=8 repro is unchanged).
  maps_buffer_ = (all_devices_[env_->rank()] >= 0);
  const bool maps_buffer = maps_buffer_;

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = get_device_granularity(prop); // we assume all GPUs will have the same granularity requirements

  // Round shard to multiple of num_shards_ * granularity (need every shard to be aligned to granularity)
  total_size_ = ROUND_UP(size, num_shards_ * granularity);
  shard_size_ = total_size_ / num_shards_;

  // Rank 0 should do all cuMemCreate and export
  std::vector<int> shareable_handles(world_size, -1);
  if (root_proc) {
    // Plan 14.7c (ROCm): on ROCm, hipMemCreate places the physical allocation on
    // the *current* device and does NOT honor prop.location.id the way CUDA's
    // cuMemCreate does. Without setting the active device per shard, rank 0 would
    // back every shard on its own device 0 (the full table on one GPU -> OOM at
    // the 1e9-row item_id table; the tiny standalone repro never exposed this
    // because all shards fit on one GPU). Set the device to all_devices_[i] before
    // each create and restore it afterward, which is also a no-op on CUDA.
    int saved_device = 0;
    NVE_CHECK_(cudaGetDevice(&saved_device));
    for (size_t i=0 ; i<world_size ; i++) {
      if (all_devices_[i] >= 0) {
        // Allocate & export handle
        CUmemGenericAllocationHandle handle;
        prop.location.id = all_devices_[i];
        NVE_CHECK_(cudaSetDevice(all_devices_[i]));
        NVE_CHECK_(cuMemCreate(&handle, shard_size_, &prop, 0 /*flags*/));
        NVE_CHECK_(cuMemExportToShareableHandle(&shareable_handles.at(i), handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
        all_alloc_handles_.push_back(handle);
      }
    }
    NVE_CHECK_(cudaSetDevice(saved_device));
  }

  // Broadcast the shareable handles to all procs
  env_->broadcast(reinterpret_cast<uintptr_t>(shareable_handles.data()), sizeof(shareable_handles[0]) * shareable_handles.size(), 0);

  // Broadcast root pid so we can map FDs
  pid_t root_pid = root_proc ? getpid() : -1;
  env_->broadcast(reinterpret_cast<uintptr_t>(&root_pid), sizeof(root_pid), 0);

  // Map FD's in all non root procs (skip on non-participant ranks: they never
  // import a shard, so they never need the root's exported FDs).
  if (!root_proc && maps_buffer) {
    int root_pidfd = pidfd_open(root_pid, 0);
    NVE_CHECK_(root_pidfd != -1);
    for (size_t i=0 ; i<world_size; i++) {
      if (all_devices_[i] >= 0) {
        int local_fd = pidfd_getfd(root_pidfd, shareable_handles[i], 0);
        NVE_CHECK_(local_fd != -1, "pidfd_getfd() failed, make sure SYS_PTRACE is available!");
        shareable_handles[i] = local_fd;
      }
    }
    // Now close the root pidfd - it's no longer used
    ERRNO_CHECK(close(root_pidfd));
  }

  // Participant ranks reserve, import and map the unified buffer. A non-shard
  // rank (maps_buffer == false) skips all of this — it keeps buffer_ == nullptr
  // and never touches the GPU VMM — but still reaches the collective barrier
  // below so the participants don't hang.
  if (maps_buffer) {
    // Reserve virtual address range for the unified buffer
    NVE_CHECK_(cuMemAddressReserve((CUdeviceptr *) &buffer_, total_size_, 0, 0 /*baseVA*/, 0 /*flags*/));

    for (size_t i=0, shard_id=0 ; i<world_size ; i++) {
      if (all_devices_[i] >= 0) {
        // Import
        CUmemGenericAllocationHandle handle;
        NVE_CHECK_(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(shareable_handles.at(i)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

        // Map
        const CUdeviceptr buf_start = reinterpret_cast<CUdeviceptr>(buffer_ + (shard_id * shard_size_));
        NVE_CHECK_(cuMemMap(buf_start, shard_size_, 0 /*offset*/, handle, 0 /*flags*/));

        // Release the handled we imported (not needed anymore)
        NVE_CHECK_(cuMemRelease(handle));
        shard_id++;
      }
    }

    // Set access. ROOT CAUSE (Plan 18, cuts E3/E3g, 2026-05-31): grant THIS rank's
    // OWN device only (count=1), exactly as init_multi_host already does below
    // (cuMemSetAccess(..., &desc, 1)). Each rank's buffer_ is a PROCESS-LOCAL virtual
    // reservation (cuMemAddressReserve above), so only this rank's own GPU ever
    // dereferences it. The legacy code granted the full N-way device mesh (every
    // all_devices_[i]) over the whole ~1 TB range, which made every GPU the peer-map
    // target of N ranks x N shards (= N^2 distinct-physical peer mappings, ~8 TB/device
    // at production scale). Under concurrent rank init that blows the amdgpu kernel's
    // per-GPU validate_and_fence budget -> the grant fails with hipErrorInvalidValue
    // (-ENOMEM underneath) and wedges the box. The cross-device descs were pure
    // overhead: with only its OWN device granted, the rank still maps all shards and
    // reads every shard directly over xGMI -- the NVE peer-read / lockstep-escape path
    // is unchanged. Own-device-only drops each GPU to N shards (~1 TB/device = the
    // single-process level proven safe), removing the size-dependent failure with no
    // chunking. This supersedes the Plan 14.7e NVE_SETACCESS_CHUNK_SHARDS workaround,
    // which targeted the wrong layer (the #2516 validator, not the kernel -ENOMEM).
    // A/B proof: scripts/debug/plan18_e3_grantmode.cpp (grant=mesh FAILS, grant=own
    // PASSES at identical N/footprint). Set NVE_GRANT_FULL_MESH=1 to restore the legacy
    // full-mesh grant for debugging/validation only.
    CUmemAccessDesc own_desc;
    own_desc.location.id = all_devices_[env_->rank()];  // this rank's GPU
    own_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    own_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    const char* full_mesh_env = std::getenv("NVE_GRANT_FULL_MESH");
    const bool full_mesh = (full_mesh_env && std::atoll(full_mesh_env) > 0);
    if (!full_mesh) {
      NVE_CHECK_(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(buffer_), total_size_, &own_desc, 1 /*count*/));
    } else {
      // Legacy full-mesh grant (debugging only) -- reproduces the -ENOMEM wedge at scale.
      std::vector<CUmemAccessDesc> access_descs;
      for (size_t i=0 ; i<world_size ; i++) {
        if (all_devices_[i] >= 0) {
          CUmemAccessDesc desc;
          desc.location.id = all_devices_[i];
          desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
          desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          access_descs.push_back(desc);
        }
      }
      NVE_CHECK_(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(buffer_), total_size_, access_descs.data(), access_descs.size()));
    }

    // ROCm: the own-device cuMemSetAccess grant above marks this rank's GPU as
    // *allowed* to touch the unified buffer, but on this HIP/amdgpu stack that
    // alone does NOT establish the xGMI routing needed to actually READ a peer
    // rank's shard — those reads silently return garbage (NaN/Inf downstream),
    // verified by the 2-rank wire test. Enable peer access from this rank's
    // device to every other participating device explicitly. This is the classic
    // O(N) per-device routing enablement (a fixed peer link per ordinal), NOT the
    // per-allocation N*N cuMemSetAccess mesh that blows the kernel peer-map budget
    // and wedges the driver at TB scale (that is what NVE_GRANT_FULL_MESH does).
    // So we keep the wedge-safe own-device grant AND get correct peer reads.
    if (!full_mesh) {
      const int my_dev = all_devices_[env_->rank()];
      int saved = 0;
      NVE_CHECK_(cudaGetDevice(&saved));
      NVE_CHECK_(cudaSetDevice(my_dev));
      for (size_t i=0 ; i<world_size ; i++) {
        const int peer = all_devices_[i];
        if (peer < 0 || peer == my_dev) continue;
        int can_access = 0;
        NVE_CHECK_(hipDeviceCanAccessPeer(&can_access, my_dev, peer));
        if (can_access) {
          hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
          // torch's caching allocator may have already enabled P2P between all
          // pairs; that returns hipErrorPeerAccessAlreadyEnabled which is benign
          // (peer routing is what we want), but it sets the sticky per-thread HIP
          // last-error that a later torch C10_HIP_CHECK (e.g. dist.barrier) would
          // rethrow. Swallow it AND clear the error state.
          if (e == hipErrorPeerAccessAlreadyEnabled) {
            (void)hipGetLastError();
          } else if (e != hipSuccess) {
            NVE_CHECK_(e);
          }
        }
      }
      NVE_CHECK_(cudaSetDevice(saved));
    }
  }

  // Now we can close all shareable handles
  env_->barrier();
  if (maps_buffer) {
    for (size_t i=0 ; i<world_size; i++) {
      if (all_devices_[i] >= 0) {
        ERRNO_CHECK(close(shareable_handles.at(i)));
      }
    }
  }
}

bool CUDADistributedBuffer::check_imex() {
  const std::string imex_root("/dev/nvidia-caps-imex-channels");
  std::filesystem::path fs_path(imex_root);
  bool found_imex = false;
  try {
    for ([[maybe_unused]] const auto& it : std::filesystem::directory_iterator(fs_path)) {
      NVE_IF_DEBUG_(std::cout << "Found IMEX channel: " << it.path().string() << std::endl);
      found_imex = true;
    }
  } catch (const std::filesystem::filesystem_error& e) {
    return false;
  }
  return found_imex;
}

void CUDADistributedBuffer::init_multi_host(uint64_t size) {
#ifdef NVE_ROCM
  // Multi-host uses CU_MEM_HANDLE_TYPE_FABRIC / IMEX, which has no ROCm equivalent.
  // Plan 14 is single-node 8xMI355 only (POSIX-FD path); fail loudly if ever reached.
  (void)size;
  throw std::runtime_error(
      "CUDADistributedBuffer multi-host (FABRIC/IMEX) path is not supported on ROCm; "
      "single-node only");
#else
  // Check for IMEX channels
  NVE_CHECK_(check_imex(), "Failed to locate IMEX channel");

  const auto local_device = env_->local_device();
  num_shards_ = collect_devices(all_devices_);
  const size_t world_size = env_->world_size();

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(std::max(local_device, 0)); // Assuming required granularity of all devices is the same (so using local device or 0 to query)
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = get_device_granularity(prop);

  // Round shard to multiple of num_shards_ * granularity (need every shard to be aligned to granularity)
  total_size_ = ROUND_UP(size, num_shards_ * granularity);
  shard_size_ = total_size_ / num_shards_;

  // Allocate & export handle
  CUmemFabricHandle fabric_handle = {};
  if (local_device >= 0) {
    NVE_CHECK_(cuMemCreate(&alloc_handle_, shard_size_, &prop, 0 /*flags*/));
    NVE_CHECK_(cuMemExportToShareableHandle(&fabric_handle, alloc_handle_, CU_MEM_HANDLE_TYPE_FABRIC, 0 /*flags*/));
  }

  // Gather all exported handles
  std::vector<CUmemFabricHandle> all_fabric_handles(world_size);
  env_->all_gather(reinterpret_cast<uintptr_t>(&fabric_handle), reinterpret_cast<uintptr_t>(&all_fabric_handles[0]), sizeof(fabric_handle));

  // Import all exported handles
  for (size_t i=0 ; i<world_size ; i++) {
    if (all_devices_[i] >= 0) {
      CUmemGenericAllocationHandle handle = {};
      NVE_CHECK_(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(&(all_fabric_handles.at(i))), CU_MEM_HANDLE_TYPE_FABRIC));
      all_alloc_handles_.push_back(handle);
    }
  }
  NVE_CHECK_(all_alloc_handles_.size() == num_shards_);

  // Reserve virtual address range for the unified buffer
  NVE_CHECK_(cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(&buffer_), total_size_, 0, 0 /*baseVA*/, 0 /*flags*/));

  // Finally map each shared handle to its part of the reserved bufer
  for (size_t i=0 ; i<num_shards_ ; i++) {
    const CUdeviceptr buf_start = reinterpret_cast<CUdeviceptr>(buffer_ + (i * shard_size_));
    NVE_CHECK_(cuMemMap(buf_start, shard_size_, 0 /*offset*/, all_alloc_handles_.at(i), 0 /*flags*/));
  }

  // And set the access for the buffer
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = static_cast<int>(std::max(local_device, 0));
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVE_CHECK_(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(buffer_), total_size_, &desc, 1 /*count*/));
#endif  // NVE_ROCM
}

uint64_t CUDADistributedBuffer::collect_devices(std::vector<int>& all_devices) {
  int local_device = env_->local_device();
  const auto world_size = env_->world_size();

  all_devices.resize(world_size);
  env_->all_gather(reinterpret_cast<uintptr_t>(&local_device), reinterpret_cast<uintptr_t>(&all_devices[0]), sizeof(local_device));

  uint64_t num_devices = 0;
  for (auto& d : all_devices) {
    if (d >= 0) {
      num_devices++;
    }
  }
  return num_devices;
}

} // namespace nve
