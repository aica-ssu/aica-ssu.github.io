# 미선정 / Drop / 흡수 아이디어 로그 (2026-04-28 vlm-edge-layerwise-context)

> Phase 1 staging 13 candidate (AI-opt 5 + HW-PIM 4 + legacy-system 4) 중 Phase 1' / Phase 2 / Phase 1'' 과정에서 4 drop + 3 흡수 = **7 미선정**. R10-α.3 형식: GAP / 시도한 overview / 미선정 사유 / 재방문 조건.

---

## A. 명시적 Drop (4)

### A.1 CARILLON — 3-Phase SM Map + Layer-wise Concurrent Batching + Bandwidth-Token Coupled Scheduling

- **Source**: legacy-system-expert Phase 1 staging (Idea 12)
- **Mechanism overview**:
  - M1 (3PSM): VLM encode-prefill-decode 3-phase 각각에 Green Context partition 동적 reconfig (encode 64 SM / prefill 48 SM / decode 32×2 SM)
  - M2 (LCB): HOT/COLD layer 별 batch size sweet spot 분리, layer-wise concurrent batching
  - M3 (BTCS): bandwidth-token coupled scheduling, GDDR7 1.79 TB/s envelope 안 admission control
- **GAP 시도**: Sarathi-Serve (chunked prefill) + DistServe (multi-GPU disagg) + Nexus (intra-GPU PD disagg) + DuetServe (adaptive SM multiplexing) 와의 차별 axis 로 "VLM 3-phase + layer-wise" 강조
- **미선정 사유**:
  - **Critical scoop**: Nova ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301), 2025) 가 single-GPU 3-phase pipelining + elastic GPU spatial partitioning + Pareto-optimal latency-throughput trade-off 모두 cover. M1 (3PSM) 은 Nova 의 핵심 mechanism 그대로 — **80%+ overlap**
  - DuetServe (arXiv:2511.04791, 2025) 도 SM partition prefill/decode 50% overlap
  - PD-Multiplexing (LMSYS 2025-09) 도 GreenContext stage 분리 cover
  - R56.2 published-ratio 30% 미달
  - novelty review 3.5/10, differentiation 6.5/10 — Nova 와 critical scoop 으로 Tier-1 진입 불가
- **재방문 조건**: 만약 Nova 가 datacenter A100 한정이고 본 idea 가 Jetson Thor LPDDR5x UMA 273 GB/s bandwidth-tight envelope 에서 BTCS 가 Nova 보다 substantially 다른 trade-off 를 보여준다면 reconsider 가능. 그러나 그것만으로 Tier-1 충분치 않음 → DAC/DATE 6p 으로 reframing

### A.2 BREAKWATER-T-R — single Jetson Thor MIG 7-partition reframe

- **Source**: ai-optimization-expert Phase 1 staging (Idea 3)
- **Mechanism overview**:
  - M1 (MIVP): Jetson Thor MIG 7-partition 으로 GPU 를 (1) ViT-instance (compute-heavy, SM 적음) + (2) LLM-instance (memory-heavy, SM 많음) 분리, CPU pre-process pipeline
  - M2: 4-bit channel-wise DeepStack tap activation compression (BREAKWATER-T M2 그대로 유지)
  - M3: NVFP4 ViT + INT4-AWQ LLM cross-precision pipeline
- **GAP 시도**: InternVL3.5 DvD (multi-server ViT/LLM split) 의 single-device intra-GPU partition 으로 reframe
- **미선정 사유**:
  - **Feasibility critical fail**: NVIDIA 공식 Jetson Thor T5000 MIG 가 2026-04 현재 single-partition only ([HotHardware Aug 2025 review](https://hothardware.com/reviews/nvidia-jetson-agx-thor-developer-kit-hands-on?page=2): "MIG is supported on T5000, but currently it can only create a single partition"). software update 가 "coming soon" 이지만 W4-W7 prototype window 안에 unverified
  - reframing to Green Context 만 사용 시 differentiation 이 Nexus / DuetServe 와 collapse
  - Nova + InternVL3.5 DvD 와 65-70% overlap (novelty review 5.0/10)
  - differentiation review 3.5/10 (M1 infeasible, 4-bit DeepStack tap activation 만 retain 시 incremental over LightVLM/SparseVILA)
- **재방문 조건**: NVIDIA 가 Jetson Thor MIG 의 multi-partition support 를 2026-Q3+ 에 release 하면 reconsider. 그 전까지는 단순 Green Context SM partition 으로 BREAKWATER-T 의 dual-Jetson 기여를 single-Thor 로 reframe 시 PRISM-FOG-FX 의 M1 LayerClassifier + M2 4:8 sparsity 와 mechanism overlap 발생 → 흡수 권고

### A.3 TIDEGATE — UMA bandwidth gating + L2 carveout duty cycle + nvpmodel hopper

- **Source**: hw-pim-accelerator-expert Phase 1 staging (Idea 7)
- **Mechanism overview**:
  - M1 (SBC): vision-encode/prefill/decode stage 별 cudaMemcpyAsync stream priority 부여 + bandwidth classifier
  - M2 (LCDC): L2 carveout duty cycle (prefill 75% / decode 50% / vision 25%)
  - M3 (NSH): nvpmodel mode 동적 switch (60W/130W/90W per stage), 50ms switching overhead
- **GAP 시도**: Asynchronous KV Cache Prefetch + LMCache tiered KV + SparseDVFS + Nova 와 차별
- **미선정 사유**:
  - Nova + PD-Multiplexing 와 70%+ overlap (M1 stream priority + M3 nvpmodel 모두 well-known engineering)
  - M3 nvpmodel switching 50ms overhead 가 short request (<1s) 에 비효율 — gain ceiling 작음
  - R56.2 published-ratio 30% 미달 (DistServe OSDI 2024 1편 외 모두 arXiv 또는 industry blog)
  - novelty review 4.0/10, differentiation 4.5/10
  - **R56.2 30% 미달 + Idea 1 (PRISM-FOG-FX) 의 M3 KSF-VCD 에 nvpmodel 일부 흡수 가능** — standalone idea 로는 약함
- **흡수 trace**: M3 nvpmodel hopper 의 일부 (decode-phase frequency slip) 는 PRISM-FOG-FX 의 M3 KSF-VCD (KV-Scale-Freeze + Visual-Context DVFS) 로 흡수
- **재방문 조건**: Jetson Thor 의 nvpmodel sub-ms switching support (Blackwell GB10 의 fast DVFS) 가 production-grade 로 검증되고, BTCS 가 Sarathi-Serve chunked prefill 의 bandwidth 버전으로 차별 강화되면 ISLPED 6p 으로 reconsider

### A.4 HARBOR-DLA — Jetson Orin AGX DLA offload of ViT patch-embed + KV value-projection

- **Source**: hw-pim-accelerator-expert Phase 1 staging (Idea 8)
- **Mechanism overview**:
  - M1 (DEF): DLA-eligible layer (ViT Conv2D + Linear FP16) automatic partitioning, LayerNorm/GELU/Softmax/RoPE 는 GPU 유지
  - M2 (DESB): DLA output → GPU input cast cudaMemcpyAsync zero-copy UMA, pipeline depth=2
  - M3 (DTAF): DLA frequency 와 GPU clock inverse 조정 (LPDDR5 bandwidth contention 회피)
- **GAP 시도**: NVIDIA Jetson DLA Tutorial + ME-ViT FPGA + ASPLOS 2025 NPU 와 차별로 "VLM 의 ViT patch-embed + KV value-projection 동시 DLA offload + GPU pipeline depth=2" 강조
- **미선정 사유**:
  - **Platform mismatch (R20-γ)**: 본 세션 4-axis platform set (Jetson Thor T5000 / RTX 5090 / Jetson Orin NX 16GB) 과 mismatch. AGX Orin (Ampere) 한정 — Jetson Thor 의 DLA-NextGen spec 미공개, RTX 5090 은 DLA 부재
  - **DLA 2.0 LayerNorm 미지원** — modern post-LN ViT 의 30% 미만만 offloadable, gain ceiling 14-22% 에서 LPDDR5 bandwidth contention 으로 +14% 까지 erode
  - AGX Orin 은 Jetson Thor 의 sunset 단계 (NVIDIA roadmap 2026)
  - R56.2 published 30% 미달 (ASPLOS 2025 NPU 1편 외)
  - novelty review 5.5/10, differentiation 5.5/10, impact 5.5/10
- **재방문 조건**: Jetson Thor DLA-NextGen spec 이 2026-Q4+ 에 공개되고 ViT LayerNorm 지원 추가되면 platform set 에 Jetson Thor + DLA 추가 후 reconsider. 그러나 현 시점 R20-γ.3 platform constraint 명시 위반

---

## B. 흡수 (Absorbed, 3)

### B.1 ATRIUM-R(AI variant) — ATRIUM v3 with NVFP4-native + Semantic KV Eviction

- **Source**: ai-optimization-expert Phase 1 staging (Idea 2)
- **Mechanism overview**:
  - M1: LayerClassifier + Modality-Asymmetric NVFP4/INT4 Assignment (HOT NVFP4 / COLD INT4-AWQ)
  - M2: Green Context SM partition + L2 carveout + DeepStack alloc skip (ATRIUM v2 그대로)
  - M3: Semantic Cluster Centroid KV Eviction (SC-CKV)
- **흡수 trace**:
  - **M1** → PRISM-FOG-FX 의 M1 DALMP + M4 LayerClassifier 로 흡수 (visual_attn_ratio HOT/COLD 자동 분류 + DeepStack-anchor NVFP4)
  - **M3 (SC-CKV)** → BIVOUAC-SLATE-R 의 M1 HSCV 로 흡수 (HOT layer K-means cluster centroid + 1-bit membership mask)
  - **M2 Green Context SM partition** → PRISM-VL-R 의 M3 GC-LSH 와 OBELISK-5090-R 의 M1 SPVL 로 흡수
- **사유**: 단일 idea 로 retain 시 PRISM-FOG-FX + BIVOUAC-SLATE-R 와 mechanism 80%+ overlap → 분할 흡수가 cleaner. R56.1 in-session evolution 명시.

### B.2 ATRIUM-R(System variant) — ATRIUM v3 with KV layer-aware page color allocation

- **Source**: legacy-system-expert Phase 1 staging (Idea 13)
- **Mechanism overview**:
  - M1: LayerClassifier + KVTuner-style sensitivity 통합
  - M2: Green Context SM partition (ATRIUM 유지)
  - M3: Layer-aware KV page color allocation (HOT layer KV → fast page color, COLD → slow)
- **흡수 trace**: **M3 page-color affinity** → STRATA-K-R 의 M3 PCA-UBP (Page-Color Affinity UMA Bank Partitioning) 로 흡수. M1 + M2 는 ATRIUM-R(AI) 의 흡수 결과와 동일하므로 중복.
- **사유**: differentiation review 5.0/10 (LOW), STRATA-K-R 와 axis 중복. drop or merge into STRATA-K-R 권고 (review consensus).

### B.3 PRISM-FX (HW variant) — Layer-wise NVFP4/FP8/INT4 + 4:8 sparsity + Green Context partition

- **Source**: hw-pim-accelerator-expert Phase 1 staging (Idea 6)
- **Mechanism overview**:
  - M1 (LSP): LayerSensitivityProbe (SQNR-based per-layer precision plan)
  - M2 (TCIR): TensorCoreItineraryRouter (4:8 NVFP4 sparsity + 2:4 FP8 dispatch)
  - M3 (DSFR): DynamicScalingFactorReuse (NVFP4 scale freeze across prefill→decode)
- **흡수 trace**: 본 idea 는 PRISMATIC-FOG (AI variant Idea 1) 와 axis 70%+ overlap → 단일 PRISM-FOG-FX 로 merge. M1 LSP + M2 TCIR (4:8 sparsity) + M3 DSFR (KV scale freeze) 모두 PRISM-FOG-FX 의 4 mechanism 안에 통합.
- **사유**: review-novelty Cluster A merge 권고 (PRISMATIC-FOG + PRISM-FX + 부분 ATRIUM-R(AI) M1) → 단일 idea PRISM-FOG-FX 로 통합 후 score +1~+1.5 격상.

---

## C. 미선정 종합 (R56.1 / R56.2 / R45 audit)

| 항목 | 결과 |
|------|------|
| Drop 4 (CARILLON / BREAKWATER-T-R / TIDEGATE / HARBOR-DLA) 모두 외부 published reference 만 인용 | ✅ R56.1 위반 0 |
| Drop 4 평균 R56.2 published 비율 | 32.5% (모두 65% 미달) |
| Drop 4 평균 R45 risk 점수 | 6.5/10 (BREAKWATER-T-R 8/10 critical, HARBOR-DLA 7/10 platform mismatch) |
| 흡수 3 의 R56.1 in-session evolution 명시 | ✅ 모두 명시 |
| 흡수 후 mechanism overlap 검증 | ✅ ATRIUM-R(AI) M3 SC-CKV → BIVOUAC-SLATE-R M1 HSCV 로 흡수, ATRIUM-R(Sys) M3 page-color → STRATA-K-R M3 PCA-UBP 로 흡수, PRISM-FX 전체 → PRISM-FOG-FX 로 merge |

## D. 향후 재방문 조건 정리

| 미선정 idea | 재방문 trigger | 예상 venue | 시기 |
|-----------|--------------|------------|------|
| CARILLON | Jetson Thor 의 BTCS bandwidth-tight envelope 차별 검증 | DAC/DATE 6p | 2026-Q3 |
| BREAKWATER-T-R | NVIDIA Jetson Thor MIG multi-partition release | MICRO/ASPLOS | 2026-Q4+ |
| TIDEGATE | Jetson Thor sub-ms nvpmodel switching production-grade 검증 | ISLPED 6p | 2026-Q3 |
| HARBOR-DLA | Jetson Thor DLA-NextGen spec 공개 + ViT LayerNorm 지원 | IEEE CAL | 2026-Q4+ |
| ATRIUM-R(AI) | (흡수 완료, 단독 재방문 없음) | — | — |
| ATRIUM-R(Sys) | (흡수 완료, 단독 재방문 없음) | — | — |
| PRISM-FX (HW) | (흡수 완료 — PRISM-FOG-FX 로 merge) | — | — |

---

## E. 외부 검색 출처 (재현용)

- [arXiv:2509.21301 Nova](https://arxiv.org/abs/2509.21301) — Real-Time Agentic VLM Serving (2025)
- [HotHardware Jetson AGX Thor Developer Kit Review (Aug 2025)](https://hothardware.com/reviews/nvidia-jetson-agx-thor-developer-kit-hands-on?page=2) — MIG single-partition only confirmation
- [arXiv:2508.18265 InternVL3.5 DvD](https://arxiv.org/abs/2508.18265) — Decoupled Vision-Language Deployment (2025)
- [arXiv:2511.04791 DuetServe](https://arxiv.org/abs/2511.04791) (2025)
- [arXiv:2507.06608 Nexus](https://arxiv.org/abs/2507.06608) (2025)
- [PD-Multiplexing LMSYS 2025-09](https://lmsys.org/blog/2025-09-28-pdmux/)
- [NVIDIA Jetson DLA Tutorial](https://github.com/NVIDIA-AI-IOT/jetson_dla_tutorial)
- [arXiv:2402.09709 ME-ViT FPGA](https://arxiv.org/abs/2402.09709)
- [Fast On-device LLM NPU ASPLOS 2025](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf)
- [arXiv:2504.06319 Async KV Cache Prefetching](https://arxiv.org/abs/2504.06319)
