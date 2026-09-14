// Upstream reproducer for ROCm/rocm-systems#2516:
//   hipMemSetAccess fails with hipErrorInvalidValue for some allocation sizes
//   despite correct arguments (size-dependent sub-buffer-coverage validator bug).
// Fixed by PR #2451 / commit 5cda2a4 (SWDEV-568260), backported here into a
// container-local libamdhip64.so (Plan 18 / Plan 19).
//
// Single-process, single-GPU, NO MPI, ~16 MB total mapped -> trivially killable,
// clean ordered teardown (unmap -> addressFree). On STOCK 7.2.3 it must FAIL at
// the hipMemSetAccess line; under the PATCHED libamdhip64.so it must print SUCCESS.
//
// Build:  hipcc -o /tmp/repro_2516 repro_2516.cpp
// Run:    ./ /tmp/repro_2516           (lib chosen via LD_LIBRARY_PATH)
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define HIP_CHECK(fn) { hipError_t err = fn; if(err != hipSuccess){ \
    fprintf(stderr, "Error: %s: %s at line %d\n", hipGetErrorName(err), \
            hipGetErrorString(err), __LINE__); exit(1);} }

int main() {
  size_t granularity;
  hipMemAllocationProp alloc_prop = {};
  alloc_prop.type = hipMemAllocationTypePinned;
  alloc_prop.location.type = hipMemLocationTypeDevice;
  alloc_prop.location.id = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &alloc_prop,
                                           hipMemAllocationGranularityRecommended));
  printf("Device recommended granularity %zu\n", granularity);

  constexpr size_t maxSize = 1ull << 35;  // 32 GB VA reservation
  hipDeviceptr_t pool_addr = 0;
  HIP_CHECK(hipMemAddressReserve(&pool_addr, maxSize, 0, 0, 0));
  printf("Reserved virtual memory pool at %p\n", (void*)pool_addr);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;

  size_t regions[] = {2379776, 7864320, 6533120};
  const size_t nregions = sizeof(regions) / sizeof(size_t);

  for (size_t i = 0; i < nregions; ++i) {
    if (regions[i] % granularity != 0) {
      size_t rounded = granularity * ((regions[i] / granularity) + 1);
      printf("Rounding allocation of %zu bytes to granularity: %zu bytes\n",
             regions[i], rounded);
      regions[i] = rounded;
    }
  }

  size_t pool_size = 0;
  for (size_t i = 0; i < nregions; ++i) {
    printf("Allocating %zu bytes at %p\n", regions[i],
           (void*)(static_cast<char*>(pool_addr) + pool_size));
    fflush(stdout);
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, regions[i], &prop, 0));
    HIP_CHECK(hipMemMap(static_cast<char*>(pool_addr) + pool_size, regions[i], 0, handle, 0));
    HIP_CHECK(hipMemRelease(handle));

    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = 0;
    access.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(static_cast<char*>(pool_addr) + pool_size, regions[i], &access, 1));

    pool_size += regions[i];
  }

  // Clean ordered teardown so a (patched) success leaves nothing pinned/mapped.
  pool_size = 0;
  for (size_t i = 0; i < nregions; ++i) {
    HIP_CHECK(hipMemUnmap(static_cast<char*>(pool_addr) + pool_size, regions[i]));
    pool_size += regions[i];
  }
  printf("Freeing virtual space %zu at %p\n", maxSize, (void*)pool_addr);
  HIP_CHECK(hipMemAddressFree(pool_addr, maxSize));

  printf("SUCCESS: all hipMemSetAccess calls passed\n");
  return 0;
}
