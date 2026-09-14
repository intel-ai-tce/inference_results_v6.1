// Standalone (no-torch) hipBLASLt fp8 GEMM with a PINNED Tensile solution index,
// exposed via a C ABI for ctypes. Built with hipcc (like the probes), so it never
// includes torch headers -> avoids the rocThrust/cub include that breaks torch's
// cpp_extension build in this image.
//
// Motivation (DLRM-v3 GEMM tuning): for the two HSTU bf16-output dense projections
// (UVQK m=2048,k=512 and OUTPROJ m=512,k=1536; TN e4m3 via torch._scaled_mm) the
// default hipBLASLt heuristic does NOT pick the fastest *existing* stock solution,
// leaving ~10-27% per-GEMM at real token counts. This forces the better index
// (hipblaslt_ext::getAlgosFromIndex), validating it for the actual shape and falling
// back to the heuristic if unsupported. Math is the same fp8 GEMM (numerically
// equivalent; different kernel schedule).
//
// Build: build_lib.sh   ->   libfp8tuned.so
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <mutex>
#include <map>
#include <tuple>
#include <vector>

namespace {
hipblasLtHandle_t g_handle = nullptr;
std::once_flag g_handle_once;
hipblasLtHandle_t lt_handle() {
  std::call_once(g_handle_once, [] { hipblasLtCreate(&g_handle); });
  return g_handle;
}

// Reusable workspace (most pinned solutions need 0; grow on demand).
void*  g_ws = nullptr;
size_t g_ws_cap = 0;
std::mutex g_ws_mtx;
void ensure_ws(size_t need) {
  std::lock_guard<std::mutex> lk(g_ws_mtx);
  if (need > g_ws_cap) {
    if (g_ws) hipFree(g_ws);
    hipMalloc(&g_ws, need);
    g_ws_cap = need;
  }
}

struct AlgoKey {
  long m, k, pin; int ta, tb, has_bias, out_bf16, has_c;
  bool operator<(const AlgoKey& o) const {
    return std::tie(m,k,pin,ta,tb,has_bias,out_bf16,has_c) <
           std::tie(o.m,o.k,o.pin,o.ta,o.tb,o.has_bias,o.out_bf16,o.has_c);
  }
};
std::map<AlgoKey, hipblasLtMatmulAlgo_t> g_cache;
std::mutex g_cache_mtx;
}  // namespace

// Returns 0 on success, nonzero on error. d is the caller-allocated output buffer
// (column-major [m,n] ld=m == row-major [n,m]); bias may be null. bias_dtype:
// 0=bf16, 1=fp16, 2=fp32 (ignored if bias==null).
//
// c: optional residual/C matrix for a fused beta*C epilogue. When non-null the op
// computes D = alpha*op(A)*op(B) + beta*C with beta=1 (alpha=1), where C has the SAME
// layout/dtype as D (column-major [m,n] ld=m == row-major [n,m]). This folds an
// out-of-GEMM residual add (e.g. the HSTU OUTPROJ `out + x` skip) into the matmul
// epilogue. When c==null, beta=0 and C aliases D (original behaviour). bias and c are
// independent (hipBLASLt applies bias as a vector broadcast, beta*C as a full matrix).
extern "C" int fp8_scaled_mm_tuned(
    const void* a, const void* b, const void* scale_a, const void* scale_b,
    void* d, const void* bias, int bias_dtype, const void* c,
    long m, long n, long k, long lda, long ldb,
    int trans_a, int trans_b, int out_bf16, long pin_index, void* stream) {
  hipblasLtHandle_t h = lt_handle();
  hipDataType cdt = out_bf16 ? HIP_R_16BF : HIP_R_8F_E4M3;

  hipblasOperation_t opA = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t opB = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  long aRows = trans_a ? k : m, aCols = trans_a ? m : k;
  long bRows = trans_b ? n : k, bCols = trans_b ? k : n;

  hipblasLtMatrixLayout_t lA, lB, lC, lD;
  hipblasLtMatrixLayoutCreate(&lA, HIP_R_8F_E4M3, aRows, aCols, lda);
  hipblasLtMatrixLayoutCreate(&lB, HIP_R_8F_E4M3, bRows, bCols, ldb);
  hipblasLtMatrixLayoutCreate(&lC, cdt, m, n, m);
  hipblasLtMatrixLayoutCreate(&lD, cdt, m, n, m);

  hipblasLtMatmulDesc_t mm;
  hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
  hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
  hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a));
  hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b));

  hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_DEFAULT;
  if (bias) {
    epi = HIPBLASLT_EPILOGUE_BIAS;
    hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    int32_t bdt = (bias_dtype==0)?(int32_t)HIP_R_16BF:(bias_dtype==1)?(int32_t)HIP_R_16F:(int32_t)HIP_R_32F;
    hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bdt, sizeof(bdt));
  }
  hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));

  const float alpha = 1.f;
  const float beta = c ? 1.f : 0.f;  // c != null => fused D = A*B + 1.0*C residual
  const AlgoKey key{m, k, pin_index, trans_a, trans_b, bias?1:0, out_bf16, c?1:0};
  hipblasLtMatmulAlgo_t algo; bool have = false; size_t ws = 0;
  {
    std::lock_guard<std::mutex> lk(g_cache_mtx);
    auto it = g_cache.find(key);
    if (it != g_cache.end()) { algo = it->second; have = true; }
  }
  if (have) {
    if (hipblaslt_ext::matmulIsAlgoSupported(h, mm, &alpha, lA, lB, &beta, lC, lD, algo, ws)
        != HIPBLAS_STATUS_SUCCESS) have = false;
  }
  if (!have && pin_index >= 0) {
    std::vector<int> idx{(int)pin_index};
    std::vector<hipblasLtMatmulHeuristicResult_t> got;
    if (hipblaslt_ext::getAlgosFromIndex(h, idx, got) == HIPBLAS_STATUS_SUCCESS && !got.empty()) {
      if (hipblaslt_ext::matmulIsAlgoSupported(h, mm, &alpha, lA, lB, &beta, lC, lD, got[0].algo, ws)
          == HIPBLAS_STATUS_SUCCESS) {
        algo = got[0].algo; have = true;
        std::lock_guard<std::mutex> lk(g_cache_mtx); g_cache[key] = algo;
      }
    }
  }
  if (!have) {
    hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
    size_t ws_max = 256ull*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_max, sizeof(ws_max));
    hipblasLtMatmulHeuristicResult_t res[1]; int ng = 0;
    hipblasLtMatmulAlgoGetHeuristic(h, mm, lA, lB, lC, lD, pref, 1, res, &ng);
    hipblasLtMatmulPreferenceDestroy(pref);
    if (ng <= 0) { hipblasLtMatmulDescDestroy(mm);
      hipblasLtMatrixLayoutDestroy(lA); hipblasLtMatrixLayoutDestroy(lB);
      hipblasLtMatrixLayoutDestroy(lC); hipblasLtMatrixLayoutDestroy(lD); return 10; }
    algo = res[0].algo; ws = res[0].workspaceSize;
    if (pin_index < 0) { std::lock_guard<std::mutex> lk(g_cache_mtx); g_cache[key] = algo; }
  }

  if (ws) ensure_ws(ws);
  // C operand = caller's residual matrix when fused (beta=1); else alias D (beta=0).
  const void* cmat = c ? c : d;
  hipblasStatus_t st = hipblasLtMatmul(h, mm, &alpha, a, lA, b, lB, &beta,
                                       cmat, lC, d, lD, &algo,
                                       ws ? g_ws : nullptr, ws, (hipStream_t)stream);
  hipblasLtMatmulDescDestroy(mm);
  hipblasLtMatrixLayoutDestroy(lA); hipblasLtMatrixLayoutDestroy(lB);
  hipblasLtMatrixLayoutDestroy(lC); hipblasLtMatrixLayoutDestroy(lD);
  return st == HIPBLAS_STATUS_SUCCESS ? 0 : (int)st;
}
