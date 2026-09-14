# rocprofv3 counter breakdown — prod vs clean attention kernels (Plan 45 M1 validation)

**What.** Dynamic hardware-counter breakdown of the dominant `_hstu_attn_fwd` attention forward, comparing
**prod** (autotuned Triton, pinned to best config BM32/BN64/nw4/ns2) against the two clean-room kernels
from Plan 44/45 — **tri** (`plan45_m2_triton._hstu_tri`, cooperative Triton DSL, BLK32/ns3/weu3) and
**gluon** (`plan44_m3_lds._hstu_fp8_lds`, wave-independent Gluon, LDS-staged + poly). Matched microbench
shape SEQ=8192, H=1, fp8 e4m3, poly gate. Collected with `rocprofv3 --pmc` (4-pass counter file
`scripts/profile/plan45_counters.txt`), median over the matched-kernel dispatches, isolated via
`--kernel-include-regex`. Driver: `scripts/profile/plan45_rocprof_driver.py`; parser:
`scripts/profile/plan45_rocprof_parse.py`. Container `dlrmv3-e2e723`, 1× MI355X (gfx950).

## Breakdown (median per dispatch)

| metric | prod | tri (Triton clean) | gluon (Gluon clean) |
|---|---:|---:|---:|
| **duration µs** | **173.8** | 214.3 (1.23×) | 364.9 (2.10×) |
| VGPR / AccVGPR | 120 / 0 | 76 / 0 | 192 / 0 |
| Scratch (spill) | 0 | 0 | 0 |
| SQ_WAVES | 1024 | 1024 | **256** |
| **issued-instruction mix** | | | |
| VALU % | 69.8 | **86.0** | **87.0** |
| MFMA % | **6.1** | 1.5 | 2.3 |
| LDS % | 15.4 | 6.2 | 7.5 |
| VMEM % | 2.1 | 1.6 | 1.2 |
| SALU % | 6.7 | 4.7 | 2.0 |
| **VALU : MFMA ratio** | **11.4** | **55.6** | 37.9 |
| VALUBusy % | 18.1 | 26.5 | **5.5** |
| MemUnitStalled % | **0.0** | 0.0 | 0.0 |

> (Some raw-cycle derived ratios rocprofv3 reports — e.g. `SQ_VALU_MFMA_BUSY_CYCLES/SQ_BUSY_CYCLES` —
> exceed 100% because the raw counters accumulate across SE/SIMD instances at different scopes; they are
> omitted as unreliable. The metrics above are either already-normalized derived %s or instance-summed
> instruction counts, which are sound.)

## Read

1. **Confirms M1 (static) on real hardware: the clean kernels are VALU-bound.** VALU is **86–87%** of the
   issued-instruction stream in both clean kernels vs **70%** for prod, and the **VALU:MFMA ratio is
   3.3–4.9× worse** (tri 55.6, gluon 37.9 vs prod **11.4**). The matrix engine is starved by VALU — the
   poly gate + fp8 conversions + index math — exactly the static finding (clean loop valu=1106 vs prod
   209). Prod's instruction stream is far more MFMA-dense (MFMA 6.1% vs 1.5–2.3%).
2. **Not memory-bound.** `MemUnitStalled ≈ 0%` for all three — re-confirms the floor doc ("memory units
   <1% stalled; latency-bound not bandwidth-bound"). No bandwidth lever exists here.
3. **The two clean kernels fail for *different* reasons** — which is why neither beats prod:
   - **tri (Triton):** good occupancy (1024 waves, VALUBusy 26.5%) but the *worst* VALU:MFMA ratio
     (55.6) — it is genuinely doing too much VALU per MFMA (poly gate at BLK=32). Still 1.23× prod
     because the autoscheduler + full occupancy hide it.
   - **gluon (Gluon):** only **256 waves** (4× lower occupancy, from the wave-independent `[NW,1]`
     mapping) and VALUBusy just **5.5%** (heavily stalled on barriers/latency, not doing useful VALU).
     Its VALU:MFMA (37.9) is *better* than tri's, yet it is slowest (2.10×) — occupancy/stall, not
     instruction mix, is its killer.
4. **Prod wins by having both** at once: ~1024 waves **and** a low VALU:MFMA ratio (11.4) **and** 3× MFMA
   share — the product of its 48-config autotune (asymmetric BM32/BN64, mi16, kpack2, ns2) feeding the
   mature scheduler. The clean kernels each get one of these, never all.

**Consistency with Plan 45 close:** the counters independently reproduce the wall-clock ranking
(prod 174µs < tri 214µs < gluon 365µs) and the mechanism (VALU-density + occupancy), reinforcing the
NO-GO — neither clean path simultaneously matches prod's MFMA density and occupancy.
