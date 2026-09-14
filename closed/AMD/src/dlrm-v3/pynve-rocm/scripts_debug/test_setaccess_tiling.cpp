// CPU-only unit test for the Plan 14.7e NVE_SETACCESS_CHUNK_SHARDS tiling loop in
// CUDADistributedBuffer::init_single_host (src/distributed.cpp). NO HIP/GPU: it
// replicates the pure integer segment arithmetic and proves the emitted
// cuMemSetAccess segments exactly tile [0, num_shards*shard_size) with no gap /
// no overlap, covering whole sub-buffers, and that chunk_shards>=num_shards
// reduces to the single legacy whole-range grant.
//
// Build + run (host CPU, no HIP):
//   g++ -std=c++17 -O2 -Wall -Wextra scripts_debug/test_setaccess_tiling.cpp -o /tmp/test_setaccess_tiling && /tmp/test_setaccess_tiling
//
// Exits 0 and prints "ALL TESTS PASSED" on success; aborts (assert) / exits 1 on failure.

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace {

struct Seg { uint64_t start; uint64_t size; };

// Exact mirror of the init_single_host loop (Plan 14.7e). `chunk_shards_env` is
// the *effective* chunk size after env parsing: in distributed.cpp it defaults
// to 1 and is only overwritten when NVE_SETACCESS_CHUNK_SHARDS parses to > 0, so
// "unset / non-positive env" is modeled here as the default 1.
std::vector<Seg> emit_segments(uint64_t num_shards, uint64_t shard_size, long long chunk_shards_env) {
  size_t chunk_shards = 1;                       // default (env unset or <= 0)
  if (chunk_shards_env > 0) chunk_shards = static_cast<size_t>(chunk_shards_env);
  if (chunk_shards > num_shards) chunk_shards = static_cast<size_t>(num_shards);

  std::vector<Seg> segs;
  for (size_t shard_id = 0; shard_id < num_shards; shard_id += chunk_shards) {
    const size_t n_shards = std::min(chunk_shards, static_cast<size_t>(num_shards - shard_id));
    const uint64_t seg_start = static_cast<uint64_t>(shard_id) * shard_size;
    const uint64_t seg_size  = static_cast<uint64_t>(n_shards) * shard_size;
    segs.push_back(Seg{seg_start, seg_size});
  }
  return segs;
}

uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
  std::printf("FAIL: %s  [num_shards=%llu shard_size=%llu chunk=%lld]\n", (msg), \
    (unsigned long long)ns, (unsigned long long)ss, (long long)chunk); \
  ++g_failures; } } while (0)

void verify(uint64_t ns, uint64_t ss, long long chunk) {
  const uint64_t total = ns * ss;
  std::vector<Seg> segs = emit_segments(ns, ss, chunk);

  // effective chunk after default+clamp, for the call-count expectation
  uint64_t eff = (chunk > 0) ? static_cast<uint64_t>(chunk) : 1;
  if (eff > ns) eff = ns;

  CHECK(!segs.empty(), "no segments emitted");
  CHECK(segs.size() == ceil_div(ns, eff), "call count != ceil(num_shards/chunk_shards)");

  // tiling: contiguous, ordered, no gap, no overlap, whole-sub-buffer coverage
  uint64_t cursor = 0;
  uint64_t sum = 0;
  std::vector<int> shard_cover(static_cast<size_t>(ns), 0);
  for (size_t i = 0; i < segs.size(); ++i) {
    CHECK(segs[i].start == cursor, "segment start not contiguous (gap or overlap)");
    CHECK(segs[i].size > 0, "zero-size segment");
    CHECK(segs[i].size % ss == 0, "segment not a whole number of shards (partial sub-buffer)");
    const uint64_t nsh = segs[i].size / ss;
    CHECK(nsh >= 1 && nsh <= eff, "segment shard-count out of [1, chunk] range");
    // mark covered shards exactly once
    for (uint64_t k = 0; k < nsh; ++k) {
      const uint64_t shard_idx = (segs[i].start / ss) + k;
      CHECK(shard_idx < ns, "covered shard index out of range");
      if (shard_idx < ns) shard_cover[static_cast<size_t>(shard_idx)]++;
    }
    cursor += segs[i].size;
    sum    += segs[i].size;
  }
  CHECK(cursor == total, "segments do not reach total (under/over-tile)");
  CHECK(sum == total, "sum of segment sizes != total");
  for (uint64_t k = 0; k < ns; ++k)
    CHECK(shard_cover[static_cast<size_t>(k)] == 1, "a shard is covered != exactly once");

  // legacy equivalence: chunk_shards >= num_shards => exactly one whole-range call
  if (chunk > 0 && static_cast<uint64_t>(chunk) >= ns) {
    CHECK(segs.size() == 1, "chunk>=num_shards must be a single call");
    CHECK(segs.size() == 1 && segs[0].start == 0 && segs[0].size == total,
          "single call must equal the whole-range [0,total) grant");
  }
}

} // namespace

int main() {
  const uint64_t GiB = 1024ULL * 1024ULL * 1024ULL;
  const uint64_t big_shard   = 128ULL * GiB;   // production: 128 GiB/shard (overflow-prone in 32-bit)
  const uint64_t small_shard = 4096ULL;        // tiny, to exercise small-size arithmetic

  int cases = 0;
  for (uint64_t shard_size : {big_shard, small_shard}) {
    for (uint64_t ns = 1; ns <= 16; ++ns) {
      // default (unset env) modeled as chunk = 0 -> defaults to 1
      verify(ns, shard_size, 0);          ++cases;
      // every valid explicit chunk_shards 1..num_shards
      for (long long chunk = 1; chunk <= static_cast<long long>(ns); ++chunk) {
        verify(ns, shard_size, chunk);    ++cases;
      }
      // chunk_shards > num_shards (clamped to one whole-range call) + the legacy restore
      verify(ns, shard_size, static_cast<long long>(ns));      ++cases;
      verify(ns, shard_size, static_cast<long long>(ns) + 1);  ++cases;
      verify(ns, shard_size, 1000);                            ++cases;
    }
  }

  // overflow sanity: 16 x 128 GiB = 2 TiB total fits in uint64 and tiles exactly
  {
    const uint64_t ns = 16, ss = big_shard, total = ns * ss;
    long long chunk = 1;
    auto segs = emit_segments(ns, ss, 3);  // 3-shard chunks over 16 shards -> 6 calls (5x3 + 1x1)
    uint64_t sum = 0; for (auto& s : segs) sum += s.size;
    if (sum != total) { std::printf("FAIL: 2TiB overflow tiling sum mismatch\n"); ++g_failures; }
    if (segs.size() != ceil_div(ns, 3)) { std::printf("FAIL: 2TiB chunk=3 call count\n"); ++g_failures; }
    (void)chunk;
  }

  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED (%d cases: num_shards 1..16 x chunk_shards {unset,1..N,>=N,1000} x {128GiB,4KiB} shard)\n", cases);
    return 0;
  }
  std::printf("TESTS FAILED: %d check(s)\n", g_failures);
  return 1;
}

