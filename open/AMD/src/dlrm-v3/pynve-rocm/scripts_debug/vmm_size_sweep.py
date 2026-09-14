"""Single-process, single-GPU VMM size sweep — localize the 128 GB-shard
``RuntimeError: invalid argument`` that the multi-GPU NVE harness hits in
``CUDADistributedBuffer::init_single_host`` (Plan 14.7e).

The harness allocates the full ``item_id`` table (1e9 x 512 x fp16 = 1.024 TB)
sharded as **128 GB per GPU**. ``cuMemCreate(128 GB)`` SUCCEEDS (rocm-smi shows
128 GB used on every GPU) but the subsequent VA phase — ``cuMemAddressReserve`` /
``cuMemMap`` / ``cuMemSetAccess`` — fails with hipErrorInvalidValue. The
standalone 8/9-rank repros only used ~0.13 GB shards (rows=2M), so they never
exercised a large mapping.

This script reproduces the *exact* VMM sequence on a SINGLE GPU (no MPI, no
peer FDs) for a sweep of single-allocation sizes, so we can find the threshold
and which call fails — and it is trivially killable (no mpirun, no zombies, no
cross-rank teardown), so it will not wedge the driver the way the harness does.

Run inside the container (one GPU), e.g.:
    HIP_VISIBLE_DEVICES=0 python3 scripts_debug/vmm_size_sweep.py 16 32 64 128

Each arg is a size in GiB. Prints, per size, which of {reserve, create, map,
setAccess} raised, or "OK" if the full sequence completed (then frees).
"""
import sys
import ctypes

# Use HIP's driver API through the same libamdhip64 the .so links. We go through
# ctypes so this test has no dependency on the nve binding (keeps it isolated).
hip = ctypes.CDLL("libamdhip64.so")

# hipError_t hipMemGetAllocationGranularity(size_t*, const hipMemAllocationProp*, flags)
# We mirror nve's CUmemAllocationProp via the HIP struct layout.

GiB = 1024 * 1024 * 1024

# hipMemAllocationProp layout (hip/hip_runtime_api.h):
#   struct { hipMemAllocationType type; hipMemAllocationHandleType requestedHandleType;
#            hipMemLocation location { hipMemLocationType type; int id };
#            void* win32HandleMetaData; struct { unsigned char compressionType;
#            unsigned char gpuDirectRDMACapable; unsigned short usage; ... } allocFlags; }
class hipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class hipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", hipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags_compressionType", ctypes.c_ubyte),
        ("allocFlags_gpuDirectRDMACapable", ctypes.c_ubyte),
        ("allocFlags_usage", ctypes.c_ushort),
        ("allocFlags_reserved", ctypes.c_ubyte * 4),
    ]

class hipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", hipMemLocation), ("flags", ctypes.c_int)]

HIP_MEM_ALLOCATION_TYPE_PINNED = 0x1
HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
HIP_MEM_LOCATION_TYPE_DEVICE = 0x1
HIP_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
HIP_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3


def chk(name, rc):
    if rc != 0:
        raise RuntimeError(f"{name} -> hipError {rc}")


def make_prop(dev=0):
    p = hipMemAllocationProp()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.type = HIP_MEM_ALLOCATION_TYPE_PINNED
    p.requestedHandleType = HIP_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    p.location.type = HIP_MEM_LOCATION_TYPE_DEVICE
    p.location.id = dev
    return p


def granularity(dev=0):
    g = ctypes.c_size_t(0)
    p = make_prop(dev)
    chk("hipMemGetAllocationGranularity",
        hip.hipMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(p),
                                           HIP_MEM_ALLOC_GRANULARITY_RECOMMENDED))
    return g.value


def try_size(size_bytes, dev=0):
    """Run reserve+create+map+setAccess for one allocation; return failing step or 'OK'."""
    g = granularity(dev)
    size = ((size_bytes + g - 1) // g) * g
    handle = ctypes.c_uint64(0)
    ptr = ctypes.c_void_p(0)
    p = make_prop(dev)
    created = mapped = reserved = False
    try:
        rc = hip.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(p), ctypes.c_ulonglong(0))
        if rc != 0:
            return f"create FAIL (hipError {rc}) size={size/GiB:.1f}GiB gran={g}"
        created = True

        rc = hip.hipMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_ulonglong(0))
        if rc != 0:
            return f"reserve FAIL (hipError {rc}) size={size/GiB:.1f}GiB"
        reserved = True

        rc = hip.hipMemMap(ptr, ctypes.c_size_t(size), ctypes.c_size_t(0), handle, ctypes.c_ulonglong(0))
        if rc != 0:
            return f"map FAIL (hipError {rc}) size={size/GiB:.1f}GiB"
        mapped = True

        desc = hipMemAccessDesc()
        desc.location.type = HIP_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = dev
        desc.flags = HIP_MEM_ACCESS_FLAGS_PROT_READWRITE
        rc = hip.hipMemSetAccess(ptr, ctypes.c_size_t(size), ctypes.byref(desc), ctypes.c_size_t(1))
        if rc != 0:
            return f"setAccess FAIL (hipError {rc}) size={size/GiB:.1f}GiB"
        return f"OK size={size/GiB:.1f}GiB"
    finally:
        if mapped:
            hip.hipMemUnmap(ptr, ctypes.c_size_t(size))
        if reserved:
            hip.hipMemAddressFree(ptr, ctypes.c_size_t(size))
        if created:
            hip.hipMemRelease(handle)


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 128]
    chk("hipSetDevice", hip.hipSetDevice(0))
    print(f"granularity(dev0) = {granularity(0)} bytes")
    for s in sizes:
        print(f"[{s:>4} GiB] {try_size(s * GiB)}", flush=True)
