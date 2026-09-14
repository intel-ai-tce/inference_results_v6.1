"""Multi-handle whole-range hipMemSetAccess test (the EXACT production pattern of
CUDADistributedBuffer::init_single_host): reserve one VA range, create N separate
physical handles, map them contiguously, then grant access ONCE over the whole
union -- and read access back per shard with hipMemGetAccess.

This is the scenario the buggy ROCm 7.2.3 sub-buffer-coverage validator rejects
(multiple independently-mapped sub-buffers under one reservation). On the STOCK
lib the single whole-range grant should FAIL (hipErrorInvalidValue); under the
PATCHED libamdhip64.so it should PASS and every shard should read back RW.

Single process, single GPU, NO MPI. Clean ordered teardown (unmap -> addressFree
-> release). Default 8 handles x 2 GiB = 16 GiB (safe on one MI355X, ~288 GB HBM).

    HIP_VISIBLE_DEVICES=0 python3 vmm_multihandle_grant.py [n_handles] [gib_per_handle]
"""
import sys
import ctypes

hip = ctypes.CDLL("libamdhip64.so")
GiB = 1024 * 1024 * 1024


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
HIP_MEM_ACCESS_FLAGS_PROT_NONE = 0x0
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


def run(n_handles, gib_per_handle, dev=0):
    g = granularity(dev)
    shard = ((gib_per_handle * GiB + g - 1) // g) * g
    total = shard * n_handles
    print(f"granularity={g}  n_handles={n_handles}  shard={shard/GiB:.1f}GiB  total={total/GiB:.1f}GiB")

    p = make_prop(dev)
    base = ctypes.c_void_p(0)
    chk("hipMemAddressReserve",
        hip.hipMemAddressReserve(ctypes.byref(base), ctypes.c_size_t(total),
                                 ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_ulonglong(0)))
    handles = []
    mapped = []
    try:
        for i in range(n_handles):
            h = ctypes.c_uint64(0)
            chk(f"hipMemCreate[{i}]",
                hip.hipMemCreate(ctypes.byref(h), ctypes.c_size_t(shard),
                                 ctypes.byref(p), ctypes.c_ulonglong(0)))
            handles.append(h)
            seg = ctypes.c_void_p(base.value + i * shard)
            chk(f"hipMemMap[{i}]",
                hip.hipMemMap(seg, ctypes.c_size_t(shard), ctypes.c_size_t(0),
                              h, ctypes.c_ulonglong(0)))
            mapped.append(seg)
        print(f"  mapped {n_handles} handles contiguously into one reservation")

        # THE production call: ONE whole-range grant over all N sub-buffers.
        desc = hipMemAccessDesc()
        desc.location.type = HIP_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = dev
        desc.flags = HIP_MEM_ACCESS_FLAGS_PROT_READWRITE
        rc = hip.hipMemSetAccess(base, ctypes.c_size_t(total), ctypes.byref(desc), ctypes.c_size_t(1))
        if rc != 0:
            print(f"  whole-range hipMemSetAccess(total={total/GiB:.1f}GiB) -> hipError {rc}  FAIL")
            return False
        print(f"  whole-range hipMemSetAccess(total={total/GiB:.1f}GiB) -> OK")

        # Read access back per shard to confirm ALL sub-buffers got the grant
        # (Step 3 end-to-end: ROCr SetMemAccess covered the whole range).
        all_rw = True
        for i in range(n_handles):
            flags = ctypes.c_int(-1)
            loc = hipMemLocation()
            loc.type = HIP_MEM_LOCATION_TYPE_DEVICE
            loc.id = dev
            rc = hip.hipMemGetAccess(ctypes.byref(flags), ctypes.byref(loc),
                                     ctypes.c_void_p(base.value + i * shard))
            ok = (rc == 0 and flags.value == HIP_MEM_ACCESS_FLAGS_PROT_READWRITE)
            all_rw = all_rw and ok
            print(f"    shard[{i}] hipMemGetAccess rc={rc} flags={flags.value} "
                  f"{'RW' if ok else 'NOT-RW <--'}")
        return all_rw
    finally:
        for seg in mapped:
            hip.hipMemUnmap(seg, ctypes.c_size_t(shard))
        if base.value:
            hip.hipMemAddressFree(base, ctypes.c_size_t(total))
        for h in handles:
            hip.hipMemRelease(h)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    gib = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    chk("hipSetDevice", hip.hipSetDevice(0))
    ok = run(n, gib)
    print("RESULT:", "PASS (whole-range grant + all shards RW)" if ok else "FAIL")
    sys.exit(0 if ok else 1)
