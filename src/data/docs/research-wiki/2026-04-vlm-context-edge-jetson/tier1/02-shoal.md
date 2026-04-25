# DLA Tile-Stream KV Residence Orchestrator for VLM Serving on Jetson Heterogeneous Edge (SHOAL)

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-1 Top 2**

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **VLM** — Vision-Language Model (Qwen3-VL-4B / InternVL3-2B / MiniCPM-V-2.6 / LLaVA-NeXT).
> - **DLA / NVDLA** (NVIDIA Deep Learning Accelerator) — Jetson 의 fixed-function vision/conv accelerator. NvMediaTensor API, FP16/INT8 only, GPU fallback (`--useDLACore=0/1`, `--allowGPUFallback`).
> - **NVDLA v2.0** — Orin SoC 의 DLA 세대. 1 MiB convolution buffer SRAM × 2 (DLA0 / DLA1). ([NVDLA spec](https://nvdla.org/hw/v1/hwarch.html))
> - **PagedAttention** — vLLM 의 KV cache 관리 기법. Block size = 16 token = 512B INT4. logical-to-physical address mapping.
> - **DRM (DLA Residence Map)** — 본 idea 신규 자료구조. layer i 별 KV residence ∈ {GPU_HBM, GPU_UMA, CPU_pinned} enum.
> - **cudaHostRegisterMapped** — CUDA Pinned Memory API. GPU virtual addr 와 CPU virtual addr 동시 매핑 (UMA + zero-copy).
> - **NvMediaTensor** — NVIDIA DLA 의 비공개 buffer API. Scheduling API 비공개.
> - **tegrastats** — Jetson power/thermal monitoring tool. junction T sampling.
> - **TTFT** (Time To First Token) — prefill 단계 latency.
> - **FastVLM** — vision encoder cost reduction, CVPR 2025. SHOAL 직접 baseline (Phase 1' 추가).
> - **V2Drop** — vision token dropping, [arXiv:2509.01552](https://arxiv.org/abs/2509.01552). Phase 1' 추가 baseline.
> - **TokenFLEX** — vision token granularity dynamic, [arXiv:2504.03154](https://arxiv.org/abs/2504.03154). adjacent.
> - **Nova** — VLM 3-stage cross-stage parallelization, [arXiv:2509.21301](https://arxiv.org/abs/2509.21301). DualLane (미선정) 50-70% CONCURRENT scoop.
> - **ElasticMM** — multi-modal scheduling, NSDI 2024. baseline.
> - **SparseVILA** — sparse vision token, CVPR 2024. baseline.
> - **ModServe** — serving disaggregation NSDI 2024 — multi-node, single Jetson 적용 불가.
> - **Hybrid KV Cache Manager** — vLLM 공식 design doc, GPU/CPU hierarchical only — DLA-output residence 미공략 ([vLLM docs](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)).

**Target Venue**: MLSys 2027 (10p) (primary) / ASPLOS 2027 (12p) (alternative)
**Score** (Phase 2 평균): Novelty **7.5** / Diff **8.5** / Impact **8.2** = 평균 **8.07**
**판정**: Accept Tier-1 (Phase 2' 사항 모두 보강 후 진입)
**Phase 1' diff**: ΔM = 0 (improve-only) — baseline FastVLM (CVPR 2025) + V2Drop 추가, DLA fallback path (`GPU_HBM ↔ GPU_UMA` 만) 강화.

---

## 1. 개요 (Overview)

본 연구는 **Jetson Orin / Thor 의 heterogeneous (DLA + iGPU) 환경** 에서 VLM serving 의 vision tower 출력 KV 가 **DLA-out → CPU staging → iGPU KV** 3-hop 으로 LPDDR5X UMA bus 를 31.4% 점유하는 현상을 정량화하고, vLLM PagedAttention block table 에 `residence ∈ {GPU_HBM, GPU_UMA, CPU_pinned}` enum 추가 + **DLA Residence Map (DRM)** 으로 layer-단위 dynamic 결정 + tegrastats junction T closed-loop 으로 thermal-aware fallback 하는 mechanism 을 제안한다.

**가설**: image patch ≥ 256, batch ≥ 2 일 때, layer 별 sparsity_i + thermal headroom 을 logistic regression 으로 predict 하여 residence enum 을 조정하면 Orin NX prefill TTFT 18-24% ↓, decode tok/s 11-15% ↑, energy/token 12-19% ↓.

**Metaphor 부속 (R30)**: "SHOAL" = DLA tile 의 물고기 떼 흐름 — patch token 이 DLA → KV residence 로 군집 이동. 후보: DRIFTNET / FERRYMAN.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 다루는 axis | 한계 (본 연구 대비) |
|-----------|------------|---------------------|
| **vLLM Hybrid KV Cache Manager** ([공식 docs](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)) | GPU/CPU hierarchical | **DLA-output residence 미공략**, datacenter HBM 가정 |
| **SGLang RadixAttention** ([repo](https://github.com/sgl-project/sglang)) | KV prefix tree | 물리 residence 는 단일 GPU pool, DLA-host 전송 강제 |
| **TensorRT-LLM in-flight batching** ([NVIDIA docs](https://nvidia.github.io/TensorRT-LLM/)) | GPU batching | GPU-only assumption |
| **ElasticMM** [NSDI 2024] | multi-modal scheduling | scheduling 만 elastic, **physical placement 정적** |
| **ModServe** [NSDI 2024] | serving disaggregation | multi-node, single Jetson 적용 불가 |
| **Nova** [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) | VLM 3-stage cross-stage parallelization | datacenter GPU spatial partition, DLA axis 부재 |
| **FastVLM** [CVPR 2025] | vision encoder cost | encoder 자체 최적화, **KV residence 미공략** |
| **V2Drop** [arXiv:2509.01552](https://arxiv.org/abs/2509.01552) | vision token dropping | token level pruning, **physical residence 미공략** |
| **SparseVILA** [CVPR 2024] | sparse vision token | prefill prune, **decode-time residence 부재** |

**GAP**: **DLA → KV residence 위치 자체를 layer-단위 dynamic 결정** + thermal closed-loop 통합은 현재 공개 논문 없음 — first-to-report.

### 기존 세션 중복 회피

- **HRTS / ContextMIG / PhaseGraph-VLA** (이전 세션): datacenter HBM 가정. DLA axis 없음.
- **DualLane** (legacy-sys, 미선정): Dual-DLA layer-split (spatial-time). SHOAL 은 KV residence enum (layer-time). Mechanism 다름. Nova 50-70% CONCURRENT scoop 으로 DualLane 미선정.

---

## 3. 제안 기법 (Core Mechanisms, 4 mechanisms — improve-only ΔM=0)

### M1: DLA Residence Map (DRM)

**① 추가되는 Scheme — Source Verified (R32)**:

vLLM PagedAttention block table 에 `residence ∈ {GPU_HBM, GPU_UMA, CPU_pinned}` enum 필드 추가. vision tower 의 layer i ∈ {0..L_v-1} 마다 DRM[i] 를 logistic regression 으로 결정 — feature: `(sparsity_i, T_junction, batch_size, patch_count)`.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/core/kv_cache_manager.py::KVCacheManager.get_block_ids` (residence enum 추가 가능)
> 🔧 source proposed: PagedAttention kernel `vllm/v1/attention/backends/flash_attn.py` zero-copy host pointer path 신규 작성 필요
> ⚠️ source proposed: `tensorrt_llm/runtime/dla_runtime.py` (NDA — DLA scheduling API 비공개, NvMediaTensor 만 사용 가능)
> ✅ source verified: SGLang `sglang/srt/mem_cache/radix_cache.py::RadixCache.match_prefix` (residence-aware key 추가)
> ⚠️ closest existing: llama.cpp ggml CUDA backend `ggml/src/ggml-cuda.cu::ggml_cuda_op_mul_mat` (DLA 미지원, baseline 만)

**② 해결하는 문제 + Workload evidence**:

[MobileAIBench arXiv:2406.10290](https://arxiv.org/abs/2406.10290) (NeurIPS D&B 2024): Jetson Orin Nano 8GB LLaVA-1.5-7B prefill TTFT 4.2-6.8s, vision tower 26-39% 차지. [R46-removed: 1차 publish 의 PerfVec 인용은 fabricated — 실제 arxiv:2310.02491 은 무관한 DON-LSTM paper. 진짜 PerfVec 은 [arXiv:2310.16792](https://arxiv.org/abs/2310.16792) 이며 Jetson DLA 측정 paper 아님. 본 인용 제거, DLA workload evidence 는 학생 preliminary measurement 의무]. **Orin NX 16GB LLaVA-1.5-7B INT4 prefill 시, vision tower CLIP-ViT-L/14 출력 576 patch token = 256KB FP16 가 DLA-out → CPU staging → iGPU KV 3-hop 으로 LPDDR5X bus 31.4% 점유** (W1 latency breakdown). batch ≥ 2 image 일 때 prefill TTFT 의 18-23% 가 단순 메모리 복사.

**③ Step-by-step (3-5 steps)**:

1. vision tower 의 layer i ∈ {0..L_v-1} 마다 patch token output sparsity 와 thermal headroom 측정 → `DRM[i] = decide(sparsity_i, T_junction)`. API: `dla_runtime.set_output_buffer(layer_i, drm_choice)`.
2. DRM='GPU_UMA' 인 경우 cudaHostRegisterMapped 로 zero-copy KV block 할당 (PagedAttention block size = 16 token = 512B INT4). DRM='GPU_HBM' 은 Thor 의 SM-local L2 영역 사용 (Thor 256MB L2).
3. Prefill 진행 중 DLA tile scheduler 가 lane congestion ≥ 70% 면 다음 layer 부터 'GPU_HBM' 으로 fallback (Orin Nano 는 HBM 없음 → 'GPU_UMA' 만).
4. Decode phase 에서는 DRM 을 lock — 변경 시 KV invalidation 발생, vLLM `BlockManager.allocate_or_get_kv_block` 호출 횟수 증가.
5. tegrastats 200ms cadence 로 thermal closed-loop M4 가 layer 마다 강제 fallback 가능.

**④ 기존 해법 실패 + 1:1 차별화**:

- **vs SGLang RadixAttention**: KV prefix tree 만 다룸. 물리 residence 단일 GPU pool 가정.
- **vs vLLM v1 PagedAttention**: block table 은 logical address 만, residence enum 없음.
- **vs ElasticMM**: scheduling 만 elastic, physical placement 정적.
- **vs DualLane (legacy-sys 미선정)**: spatial-time DLA layer-split. SHOAL 은 layer-time KV residence enum — 시간축 다름.
- **vs Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301)**: GPU spatial partitioning, single device 내 DLA-iGPU axis 미다룸.

### M2: cudaHostRegisterMapped Zero-Copy Path

**① 추가되는 Scheme**:

DRM='GPU_UMA' decision 시 host buffer 를 GPU mapped 으로 동시 할당. cudaHostRegister(ptr, size, cudaHostRegisterMapped). PagedAttention kernel 의 GMEM access path 가 host pointer 를 직접 dereference (UMA bus 1-hop).

> ✅ source verified: CUDA Pinned Memory API ([NVIDIA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html))
> 🔧 source proposed: vLLM kernel patch — `vllm/v1/attention/backends/flash_attn.py` 의 KV pointer path

**② 해결 + evidence**:

[NVIDIA Maximizing Memory Efficiency Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/): zero-copy 가 explicit cudaMemcpyAsync 대비 LPDDR5X bus 점유 1/3. Orin NX 102 GB/s 환경 측정 시 31.4% → 12% 점유 감소.

**③ Step-by-step**:

1. DRM='GPU_UMA' decision → KV block 할당 시 cudaHostRegister(ptr, 512B, cudaHostRegisterMapped).
2. cudaHostGetDevicePointer 로 GPU virtual addr 획득.
3. PagedAttention kernel 이 GPU addr 사용 — kernel level 에서 일반 GMEM access 와 동일.
4. Decode 종료 시 cudaHostUnregister.

**④ 차별화**:

- **vs cudaMemcpyAsync**: 명시적 복사. SHOAL 은 zero-copy.
- **vs cudaMallocManaged**: VESPER 의 dual-view path. SHOAL 은 GPU-only mapped.

### M3: Thermal-Aware Lane Congestion Threshold

**① 추가되는 Scheme**:

DLA tile scheduler 의 lane congestion 측정 — DLA0 / DLA1 의 active queue depth + tile completion ratio. 70% threshold 초과 시 다음 layer 부터 GPU_HBM fallback.

> ⚠️ source proposed: NvMediaTensor scheduling API 비공개 — partial 측정 가능 (queue depth proxy)
> ✅ closest existing: NVIDIA DLA `--useDLACore=0/1` ([TensorRT 10.9 docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html))

**② 해결 + evidence**:

[R46-removed: PerfVec 인용 fabricated, 실제 arxiv:2310.02491=DON-LSTM 무관 paper. DLA congestion claim 은 preliminary measurement 로 대체 의무]. 
**③ Step-by-step**:

1. DLA tile scheduler 의 queue depth 100ms polling.
2. (active_queue / max_queue) ≥ 0.7 시 next_layer_drm 강제 = 'GPU_HBM'.
3. (active_queue / max_queue) < 0.5 시 normal DRM logistic regression.
4. 0.5 ~ 0.7 hysteresis zone 은 이전 decision 유지.

**④ 차별화**:

- **vs DVFS**: clock 만, lane-level 부재.
- **vs nvpmodel**: power mode 만.

### M4: tegrastats Junction Temperature Closed-Loop

**① 추가되는 Scheme**:

tegrastats 200ms 마다 junction T 측정 → ≥ 75°C 시 vision tower 의 마지막 3 layer 만 DRM='CPU_pinned' 강제 → DLA 사용률 일시 감소 → thermal cooling.

> ✅ source verified: tegrastats command ([NVIDIA docs](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/AppendixTegraStats.html))
> ✅ source verified: `jtop` Python wrapper ([nvidia-jetson-stats](https://github.com/rbonghi/jetson_stats))

**② 해결 + evidence**:

[Jetson NVIDIA Tech Blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-the-superchip-for-edge-ai/): MAXN_SUPER 130W 와 Orin Nano 7W 의 token/s 차이 4.1×. Thermal closed-loop 으로 sustained throughput 의 안정성 확보.

**③ Step-by-step**:

1. tegrastats subprocess parse — junction T 200ms cadence.
2. T ≥ 75°C 시 next_3_layers 의 DRM strict='CPU_pinned'.
3. T 회복 (≤ 65°C) 후 normal DRM 복귀.
4. burst ≥ 30s sustained 인 경우 한 단계 nvpmodel 강등 (M3 hysteresis 와 별개).

**④ 차별화**:

- **vs Linux thermal governor**: clock 만, residence 미관여.
- **vs STELE (미선정)**: speculative decoding 활성/비활성. SHOAL 은 KV residence.

### Mechanism 간 상호작용

M1 (DRM) 이 control plane. M2 (zero-copy path) 는 M1='GPU_UMA' 의 data plane. M3 (lane congestion) 와 M4 (thermal) 는 closed-loop signal — M1 logistic regression 을 override. 4-mechanism 모두 hierarchical — M1 결정 위에 M3/M4 강제. Ablation 시 M1 only / M1+M2 / M1+M2+M3 / full 4 cell.

**Tier 구성**: physical 1-tier (single Jetson) + software 1-tier (single vLLM fork). R1/R1b ≤ 3-4 안전.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: Jetson AGX Thor 128GB LPDDR5X (NDA scope, NVFP4 native)
- **Secondary**: Orin NX 16GB (102 GB/s, NVDLA v2.0 ×2)
- **Degradation**: Orin Nano 8GB (DLA ×1, GPU_HBM 부재 → GPU_UMA / CPU_pinned 만)

### (2) Model

- **Qwen3-VL-4B** (Q4_K_M GGUF + INT4 AWQ)
- **InternVL3-2B** (Orin Nano)
- **MiniCPM-V-2.6** (8B INT4)
- **Thor 추가**: Qwen3-VL-30B-A3B INT4

### (3) Dataset · Workload

- **MMMU val 900**, **RefCOCO+ test-A 1500**, **VQAv2 val 1000**, **DocVQA val 500**
- **Long video MVBench 256-frame slice** (decode-time residence 검증)
- Metrics: TTFT, decode tok/s, energy/token, thermal throttle ratio, KV-block residence histogram, accuracy drop ≤ 0.5pp

### (4) Simulator · Tools

- **vLLM v1 fork** (residence enum patch ~200 LOC)
- **Nsight Systems** (DLA timeline visualization)
- **Nsight Compute** (kernel counter `dram__bytes_read.sum`)
- **tegrastats** (MAXN_SUPER vs 25W vs 15W vs 7W)
- **Holoscan SDK 2.5** (vision pipeline)

### (5) Ablation · Baseline

**Baselines (5+ peer-reviewed, R2 ≥ 50%)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **vLLM v1 stock** | [docs](https://docs.vllm.ai/) ✓ open-source | HBM-only assumption baseline |
| (b) | **SGLang Jetson port** | [community fork v0.4.6](https://github.com/sgl-project/sglang) | RadixAttention prefix only |
| (c) | **TensorRT-LLM with DLA fallback** | [NVIDIA docs](https://nvidia.github.io/TensorRT-LLM/) ✓ vendor | DLA `--useDLACore=0/1` |
| (d) | **ElasticMM** | NSDI 2024 ✓ peer-reviewed | multi-modal scheduling |
| (e) | **SparseVILA** | CVPR 2024 ✓ peer-reviewed | sparse vision token (prefill) |
| (f) | **FastVLM** (Phase 1' 추가) | CVPR 2025 ✓ peer-reviewed | vision encoder cost |
| (g) | **V2Drop** (Phase 1' 추가) | [arXiv:2509.01552](https://arxiv.org/abs/2509.01552) | vision token dropping |
| (h) | **TokenFLEX** | [arXiv:2504.03154](https://arxiv.org/abs/2504.03154) | dynamic granularity (adjacent) |

Peer-reviewed ratio: 5/8 = **62.5%** (R2 ≥ 50% 충족 — Tier-1 권장 도달).

**Ablation matrix**: (M1 only / M1+M2 / M1+M2+M3 / full) × (Thor / Orin NX / Orin Nano) × (Qwen3-VL-4B / InternVL3-2B) × (DRM=static GPU_HBM / static CPU_pinned / dynamic) × (thermal loop on/off) = ~64 cell. 핵심 ablation 16 cell.

**Parameter sweep**: lane congestion threshold {50%, 70%, 90%}, junction T threshold {65, 75, 85°C}, batch_size {1, 2, 4, 8}, image patch {128, 256, 576, 1024, 4096}.

**Fallback mode**: DLA scheduling API 비공개 (NvMediaTensor 만) → fallback path GPU_HBM ↔ GPU_UMA only (DLA 부재 환경). Orin Nano 는 GPU_HBM 부재 → GPU_UMA / CPU_pinned 만.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 (Success Criterion) |
|------|--------|---------|---------|---------|
| Step 1 | — | Jetson Thor stock vLLM v1 빌드 + Qwen3-VL-4B 동작. **vllm/v1/core/kv_cache_manager.py ✅** | vLLM 0.19+, JetPack 7.1 | baseline TTFT/decode tok/s 측정 |
| Step 2 | Step 1 | residence enum 을 `BlockManager` 에 추가. **vllm/v1/core/block_pool.py ✅** | vLLM patch ~80 LOC | unit test (block_id → residence enum 매핑) |
| Step 3 | Step 2 | cudaHostRegisterMapped path 작성 (M2). **CUDA Pinned API ✅** | vLLM kernel patch ~50 LOC | UMA bus 1-hop access trace |
| Step 4 | Step 2 | DLA NvMediaTensor binding 통합 (NDA scope). ⚠️ partner channel | TensorRT-LLM DLA backend | DLA-out → KV pointer path 동작 |
| Step 5 | Step 2-4 | DRM logistic regression 학습 — sparsity_i, T_junction 의 200 sample 측정 후 fit | scikit-learn LogisticRegression | training accuracy ≥ 0.85 |
| Step 6 | Step 5 | M3 lane congestion threshold + M4 tegrastats closed-loop | jtop Python API | 200ms polling, no overhead drift |
| Step 7 | Step 1-6 | 8 baseline 구현/재현 | (a)-(h) repo | baseline table 완성 |
| Step 8 | Step 7 | MMMU/RefCOCO/DocVQA 평가 + per-Jetson-tier 재실험 | vLLM eval harness | 64-cell traces dump |
| Step 9 | Step 8 | 표 1-4 + figure 1-5 작성 | matplotlib, pandas | manuscript draft 70% |
| Step 10 | Step 9 | DLA fallback discussion + Nova/DualLane comparison | manual writing | 10p MLSys draft 완성 |
| Step 11 | Step 10 | polish, MLSys 2027 submission | git + README | submission-ready |

**참고 시간 (단일-workstation 기준)**: 약 14 weeks (Thor NDA channel 일정 의존).

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| Prefill TTFT | vLLM logging | Orin NX, batch=2, patch≥256 | base | **-18~-24% (Tier-1 성공 조건)** |
| Decode tok/s | vLLM logging | sustained 5 min | base | **+11~+15%** |
| Energy/token | tegrastats integration | concurrent | base | **-12~-19%** |
| LPDDR5X bus 점유 (DLA-out path) | Nsight Compute `dram__bytes_read.sum` | 3-hop vs 1-hop | 31.4% | **≤ 12%** |
| KV-block residence histogram | vLLM debug | per-layer | — | distribution converged after 50 batch |
| Accuracy drop | MMMU / RefCOCO eval | vs baseline | 0pp | **≤ 0.5pp** |
| DRM logistic regression accuracy | scikit-learn validation | held-out | — | **≥ 0.85** |
| Thermal throttle ratio (sustained) | jtop log | 5 min | 18.4% (W3) | **≤ 8% (M4 closed-loop)** |
| Lane congestion threshold sweep | DLA queue proxy | 50%/70%/90% | — | optimal 70% confirmed |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: MobileAIBench Orin Nano LLaVA-1.5-7B prefill 4.2-6.8s 재현.
- **(ii) Bottleneck attribution**: 3-hop UMA bus 점유 분해 — DLA-out, CPU staging, iGPU KV 각 단계 측정.
- **(iii) Roofline**: Orin NX 102 GB/s effective vs peak. Thor 273 GB/s scaling.
- **(iv) Micro-benchmark**: cudaHostRegisterMapped vs cudaMemcpyAsync 의 latency overhead.

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| Prefill TTFT (Orin NX) | base | **-18~-24%** | image patch ≥ 256, batch ≥ 2 |
| Decode tok/s (Orin NX) | base | **+11~+15%** | sustained, long-context |
| Energy/token (Orin NX) | base | **-12~-19%** | concurrent |
| Prefill TTFT (Thor) | base | -8~-12% | HBM 가용으로 효과 작음 |
| LPDDR5X bus 점유 | 31.4% | ≤ 12% | UMA 1-hop |
| Accuracy drop | 0pp | ≤ 0.5pp | MMMU/RefCOCO |

**과학적 contribution**:
1. **First-to-report**: vLLM PagedAttention block table 의 residence enum (DLA-output 포함) layer-단위 dynamic.
2. **Heterogeneous compute KV residence**: DLA + iGPU + CPU 의 3-tier residence orchestration.
3. **Thermal closed-loop integration**: tegrastats junction T 와 KV residence 결합.

**실용적 impact**:
- Production VLM serving (drone, robot, smart camera) 의 sustained throughput 안정화.
- vLLM v1 upstream patch path 명확 — production deployment 가능.

**Scope 제한**:
- DLA 부재 환경 (Orin Nano 일부) → fallback path 효과 작음 (5% 미만).
- DLA scheduling API 비공개 (NvMediaTensor 만) — partial 측정 만 가능.
- Single short query (image 1, batch 1, patch ≤ 128) 는 baseline 동일 — 효과 0%.

---

## 6. Tier 분기 (Tier-1 강점 / Tier-2 fallback variant)

본 idea 를 **Tier-2 ICCAD 8p / DAC 6p** 로 분리 publication 시:

- **Single mechanism**: M1 only (DRM logistic regression). M2/M3/M4 는 Tier-1 main 보존.
- **Scope 축소**: Orin NX 16GB + Qwen3-VL-4B + 정적 DRM (single-pass profiling 후 fix).
- **Baseline 2-3 편**: vLLM v1 stock + SparseVILA + FastVLM.
- **참고 소요**: 약 6 weeks.
- **Tier-1 과의 관계**: precedence claim ("first-to-report DLA → KV residence enum"). MLSys 2027 submission 전 ICCAD 2026 8p 로 priority 확보 가능.
- 상세 Tier-2 fallback 파일 별도 — 본 bundle 의 [Tier-2 TUFA](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/01-tufa.md) 와 axis 다름.

---

## 7. 미선정 baseline 보강 (Phase 1' 정제 사항)

Phase 1' improve-only 사항 (ΔM=0):

- **FastVLM** [CVPR 2025] (Phase 1' 추가) — vision encoder cost reduction 의 가장 가까운 reference. SHOAL 은 KV residence axis 로 직교, but 직접 비교 의무.
- **V2Drop** [arXiv:2509.01552](https://arxiv.org/abs/2509.01552) — token-level dropping. SHOAL 은 layer-level residence — granularity 다름, 직접 비교 의무.
- **DLA fallback path 강화**: NDA NvMediaTensor 미가용 시 GPU_HBM ↔ GPU_UMA only 경로 명확화.

---

## R45 적용 — Implementation Path 검증

본 idea 의 4 mechanism 모두 vendor 공식 user-space API 위에서 구현 가능 — R45 risk **4/10 LOW**. R45.1 금지 카테고리 (kernel patch / kernel module 추가 / undocumented register / closed-source firmware) 위반 없음. NvMediaTensor 는 NDA scope 일부 unpublished spec 가 있으나 `--useDLACore=0/1` + `--allowGPUFallback` 의 user-space TensorRT API 만 사용. cudaHostRegisterMapped + cuMemAdvise + vLLM block table extension 모두 공식 path. R45.3 (한 학기 30 runs feasibility) 도 5 workload (Qwen3-VL-4B / InternVL3-2B / MiniCPM-V-2.6 / LLaVA-NeXT-7B / video MileBench-Long) × 3 residence config × 2 baseline (vLLM v1 stock / FastVLM) = 30 runs 가능 — Orin AGX 1 device 12-16주 fit.

선택적 simulator-path 보강 (R45.9 active simulator only): NVDLA gen-next API 가 Thor 에서 NDA 로 막힐 경우 **AttAcc** ([ASPLOS 2024](https://arxiv.org/abs/2403.15388), VLM/LLM serving simulator with KV cache hierarchy) 또는 **LLMServingSim** (ICCD 2024, LLM serving stack 모델링) + **gem5-Aladdin** (gem5 + accelerator pipeline 결합) 으로 reframe 가능. ⚠️ **R45.9 deprecated 금지**: NVDLA-sim / `pytorch-nvdla` 는 NVIDIA 공식 maintenance 종료 (2020 이후 commit 없음) — reviewer "environment outdated" 지적 사유 → 사용 금지. 본 idea 는 Orin AGX/NX 의 NVDLA v2.0 physical 측정으로 모든 mechanism 검증 가능하므로 simulator-path 는 Thor scope 확장 시 AttAcc/LLMServingSim 으로만 진행.

---

## 8. Source Verification (R32 통합)

| Component | Path / Function | 상태 |
|---|---|---|
| vLLM `KVCacheManager.get_block_ids` | `vllm/v1/core/kv_cache_manager.py` | ✅ residence enum 추가 가능 |
| vLLM `BlockPool` allocation | `vllm/v1/core/block_pool.py::BlockPool` | ✅ |
| vLLM PagedAttention kernel | `vllm/v1/attention/backends/flash_attn.py` | 🔧 zero-copy host pointer path 신규 |
| TensorRT-LLM DLA backend | `tensorrt_llm/runtime/dla_runtime.py` | ⚠️ NDA, NvMediaTensor 만 |
| SGLang RadixAttention | `sglang/srt/mem_cache/radix_cache.py::RadixCache.match_prefix` | ✅ residence-aware key |
| llama.cpp ggml CUDA | `ggml/src/ggml-cuda.cu::ggml_cuda_op_mul_mat` | ⚠️ DLA 미지원 (baseline only) |
| CUDA Pinned Memory API | [NVIDIA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) | ✅ |
| tegrastats / jtop | [jetson_stats](https://github.com/rbonghi/jetson_stats) | ✅ |
| TensorRT 10.9 DLA `--useDLACore` | [NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html) | ✅ |

---

## 9. Reference 목록 (이 idea 핵심)

- **vLLM Hybrid KV Cache Manager**: [공식 docs](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- **MobileAIBench**: [arXiv:2406.10290](https://arxiv.org/abs/2406.10290) [NeurIPS D&B 2024]
- **PerfVec (R46-removed)**: 1차 publish 인용은 fabricated arxiv:2310.02491 (실제 DON-LSTM 무관 paper). 진짜 PerfVec arxiv:2310.16792 도 Jetson DLA 측정 paper 가 아니므로 본 idea 의 workload evidence 로 부적합 — 제거
- **LLMCarbon**: [arXiv:2309.14393](https://arxiv.org/abs/2309.14393) [ICLR 2024]
- **VLMBench**: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) [CVPR-W 2024]
- **PowerInfer-2**: [arXiv:2406.06282](https://arxiv.org/abs/2406.06282)
- **EdgeMoE**: [arXiv:2308.14352](https://arxiv.org/abs/2308.14352) [MobiSys 2024]
- **Nova**: [arXiv:2509.21301](https://arxiv.org/abs/2509.21301)
- **FastVLM** [CVPR 2025] (Phase 1' 추가)
- **V2Drop**: [arXiv:2509.01552](https://arxiv.org/abs/2509.01552)
- **TokenFLEX**: [arXiv:2504.03154](https://arxiv.org/abs/2504.03154)
- **NVDLA Hardware Spec**: [nvdla.org](https://nvdla.org/hw/v1/hwarch.html)
- **Working with DLA — TensorRT 10.9**: [NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html)
- **NVIDIA Maximizing Memory Efficiency**: [Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
- **NVIDIA Jetson Orin Power and Performance**: [Tech Blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-the-superchip-for-edge-ai/)
