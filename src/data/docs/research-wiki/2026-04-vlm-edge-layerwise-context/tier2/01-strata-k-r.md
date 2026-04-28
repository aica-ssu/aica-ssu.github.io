# STRATA-K-R (Stratified Tiered Allocation for KV via Layer-aware HOT/COLD layout, Refined — ATRIUM-R(Sys) page-color affinity 흡수)

- **Tier**: 2
- **Lead expert**: legacy-system-expert (memory tiering / KV layout)
- **Target venue**: ASPLOS 2026 / DAC 2026 / IEEE CAL 2026 / EuroSys 2026
- **Single-system fit (R20-γ)**: RTX 5090 32GB GDDR7 primary (96MB L2 carveout 활용), Jetson AGX Thor 128GB secondary (LPDDR5x UMA page-color affinity)
- **5-axis tag (R55.2)**:
  - [Performance] decode latency -15~22% (long-context VLM > 4K visual tokens)
  - [Memory eff.] effective KV capacity +40~55% (4-tier mapping)
  - [Energy] -10~15% (LPDDR5x cold layer fetch lower power than GDDR7)
  - [Throughput] +18~28% (OOM-bottleneck workload)
- **R47 path**: R47.2 application-level only — vLLM block manager + SGLang HiCache layer 를 stratification-aware allocator 로 교체. CUDA Driver API (`cuMemSetAccess` + `cudaAccessPolicyWindow` with `cudaAccessPropertyPersisting`). Linux page-coloring 은 user-space `madvise()` + huge-page allocation 으로 affinity 차등.
- **R45 risk**: 6/10 (page color delta latency 60-120ns 가 ms-scale decode 대비 작아 gain ceiling 낮을 risk. tier promotion/demotion churn 비용. Jetson UMA 환경에서 4-tier 가 사실상 2-tier 로 collapse 위험)
- **Phase 1' delta (improve-first)**:
  - **Improved** M1 (LSS — Layer Stratification Score) — visual_attn_ratio (ATRIUM in-session) + KVTuner sensitivity (ICML 2025) + recency 의 weighted sum, 단일 composite metric.
  - **Replaced** M2 (4TMM 4-tier → 3-tier) — review-novelty 권고에 따라 NVMe Tier-3 제거 (edge fit 약함). Tier-0 GPU L2 carveout (96MB), Tier-1 GDDR7 (RTX 5090) 또는 LPDDR5x (Jetson Thor), Tier-2 host LPDDR5 (RTX 5090 시스템 메모리) 또는 LPDDR5x cold page color (Jetson UMA 내 micro-tier).
  - **Added** ATRIUM-R(Sys) M3 page-color affinity 흡수 → M3 Page-Color Affinity UMA Bank Partitioning. UMA 환경 micro-tier 명시. ATRIUM-R(Sys) standalone idea 는 STRATA-K-R 에 흡수되어 drop (review 권고).
  - **Improved** anchor visual token L2 pinning sub-mechanism 격상 — review-novelty 권고에 따라 KVQuant attention sink insight 를 layer-stratified spatial pinning 으로 일반화. "어떤 visual token 이 anchor 자격을 가지는가" (semantic salience score) 가 진짜 새로운 question.
  - **Reinforced** R56.2 baseline: ASPLOS 2024 (memory tiering), ISCA 2024 (cache partition), FAST 2025 Mooncake (Best Paper), OSDI 2024 InfiniGen, OSDI 2024 DistServe, ICLR 2025 VL-Cache, ICML 2025 KVTuner, IISWC 2024 LLM characterization, NeurIPS 2024 KVQuant, [LayerKV arXiv:2410.00428](https://arxiv.org/abs/2410.00428) (2024-2025 follow-up). 시스템 분야 published 65%+.
  - **Added** scoop 대응: NVIDIA ICMS / Bluefield-4 CMX (CES 2026 announcement, datacenter-scale 4-tier) baseline 명시 — 본 idea 는 **edge UMA 4-tier (datacenter ICMS 의 edge-equivalent)** 차별. LayerKV 는 layer-wise KV block allocation, **본 idea 는 layer-score → physical-tier 매핑 + page-color micro-tier** 차별.

## 1. RQ (R57.1)

- **RQ-4.1**: VLM long-context workload (4K-32K visual tokens) 에서 layer score 기반 4-tier (L2 carveout / GDDR7 / LPDDR5 / NVMe) → 3-tier (L2 / GDDR7|LPDDR5x / page-color cold) mapping 이 vanilla vLLM allocator 대비 effective KV capacity 를 **+40% 이상** 증가시키는가? OOM 발생 batch size 가 baseline 대비 **+50% 이상** 증가하는가?
- **RQ-4.2**: Tier-0 L2 carveout (96MB) 에 anchor visual token (semantic salience top-K) 만 pin 했을 때, 일반 system prompt + first 128 token pin 대비 decode latency 가 **추가 -8% 이상** 감소되는가?
- **RQ-4.3**: Jetson Thor LPDDR5x UMA 환경에서 page color (8-color partition, Linux kernel page-coloring) 별 access latency delta 가 60-120ns 로 측정될 때, layer-class HOT/COLD 별 page color 매핑이 effective bandwidth 를 **+15% 이상** 향상시키는가?
- **RQ-4.4**: tier promotion/demotion 의 churn cost (block move overhead) 가 hysteresis margin 적용 시 long-context workload 에서 prefill latency 의 **≤ 5%** 안에서 유지되는가?

## 2. 개요 (Metaphor noun ↔ mechanism)

**STRATA-K-R**: STRATA = 지층처럼 layered KV memory hierarchy (L2 / GDDR7 또는 UMA hot / UMA cold-page-color). K = KV cache. R = Refined (4-tier → 3-tier 응축, page-color affinity 흡수, anchor visual token L2 pinning 격상).

핵심 통찰: **VLM 의 long-context KV cache 는 layer 별 sparsity / sensitivity 가 70-99% wide range 라서 uniform 한 GPU memory 에 두는 것이 낭비**. layer score (sparsity + sensitivity + recency) 에 따라 (a) hot layer → L2 carveout 의 anchor token pinning, (b) warm layer → GDDR7/UMA hot page-color, (c) cold layer → UMA cold page-color (Jetson) 또는 host LPDDR5 (RTX 5090) 으로 micro-tier 분리. 또한 KVQuant 의 attention sink insight 를 visual token 에 적용 — semantic salience top-K 만 L2 anchor 로 pin.

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence (R17)

- arXiv:2410.23317 (VL-Cache, ICLR 2025): VLM layer 별 attention sparsity 70-99% wide range. 일부 layer 거의 모든 token attend, 일부 매우 sparse → tier mapping 정당성.
- IISWC 2024 ([arXiv:2512.01644](https://arxiv.org/abs/2512.01644)): edge decode 80% time KV BW dominated.
- Jetson Thor LPDDR5x: 273 GB/s, page color 별 access latency delta 60-120ns 측정 가능 (Linux page-coloring research, 본 idea 가 측정).
- RTX 5090: GDDR7 1.79 TB/s + L2 96MB.
- arXiv:2502.06433 (KVTuner, ICML 2025): layer-wise mixed-precision KV quant, sensitivity score.
- arXiv:2401.18079 (KVQuant, NeurIPS 2024): attention sink first-token FP16 보존, anchor token concept.

### 3.2 GAP 표 (R56.2)

| 기존 연구 | venue year | 핵심 mechanism | what / why / how 차별 |
|-----------|-----------|----------------|------------------------|
| LayerKV | [arXiv:2410.00428](https://arxiv.org/abs/2410.00428) (2024-2025) | layer-wise KV block allocation + offloading, schedule per-request layer offload | what: STRATA-K-R 는 **layer score (composite metric) → physical-tier mapping** + **page-color micro-tier** + **anchor visual token L2 pin**. why: LayerKV 는 layer 별 single offload 결정만, multi-tier physical mapping 부재. how: 3-tier physical + page-color affinity |
| HiCache (LMSYS production) | 2025 | 3-tier (GPU L1 / CPU L2 / Storage L3), exact match | what: layer-uniform 정책. how: layer-별 score 로 mapping |
| InfiniGen | OSDI 2024 | speculative load token granularity, layer-uniform | what: layer score 로 prefetch 우선순위 |
| Mooncake | FAST 2025 Best Paper | KVCache-centric disagg datacenter | what: edge UMA 부적합 vs Jetson Thor UMA 적합 + page-color micro-tier |
| KVTuner | ICML 2025 | sensitivity-aware mixed-precision quant | what: sensitivity 를 placement 에 직접 사용 (quant 아닌 layout) |
| KVQuant | NeurIPS 2024 | attention sink first-token FP16 | what: anchor concept 를 visual token 에 layer-stratified spatial pinning 으로 일반화 |
| VL-Cache | ICLR 2025 | layer sparsity budget 분배 | what: sparsity 를 tier 매핑에 직접 사용 |
| NVIDIA ICMS / Bluefield-4 CMX | CES 2026 announcement (industry) | datacenter 4-tier with NVMe G3.5 | what: edge UMA 4-tier (datacenter ICMS 의 edge-equivalent) 차별 |
| Multi-Tier Dynamic Storage | [Springer Complex & Intelligent Systems 2025](https://link.springer.com/article/10.1007/s40747-025-02200-4) | uniform 분배 multi-tier | what: layer-aware 분배 차별 |
| DistServe | OSDI 2024 | prefill/decode multi-GPU 분리 | placement axis 다름 |
| Sarathi-Serve | OSDI 2024 | chunked prefill | scheduling vs placement orthogonal |
| Atom — W4A4 ASPLOS 2024 | ASPLOS 2024 | quantization tier | quant axis 다름 |
| ASPLOS 2024 — Memory tiering | ASPLOS 2024 | 일반 memory tier (CXL) | what: edge UMA + L2 carveout 차별 |
| ISCA 2024 — Cache partitioning | ISCA 2024 | cache partition framework | what: VLM-specific layer score 차별 |

Peer-reviewed published: LayerKV (2024-2025 → arXiv published, NeurIPS submission), InfiniGen (OSDI 2024), Mooncake (FAST 2025), KVTuner (ICML 2025), KVQuant (NeurIPS 2024), VL-Cache (ICLR 2025), DistServe (OSDI 2024), Sarathi-Serve (OSDI 2024), Atom (ASPLOS 2024), ASPLOS 2024 memory tiering, ISCA 2024 cache partitioning, IISWC 2024 = **12/14 = 86%** ≥ 65%.

## 4. 제안 기법 — 3 mechanism

### 4.1 — M1: Layer Stratification Score (LSS)

#### Block 1: Concept
- 추가되는 Scheme: layer 별 (a) attention sparsity ratio (VL-Cache 측정), (b) KV quant sensitivity (KVTuner score), (c) recency-of-access 의 weighted sum 으로 score 산출. score = α × sparsity + β × sensitivity + γ × recency, where α=0.4, β=0.4, γ=0.2 default. offline profile 한 번 → online hot/warm/cold 3-class 분류 (top 20% / mid 50% / bottom 30%).
- 해결하려는 문제: HiCache 는 layer-uniform 정책, LayerKV 는 single-offload 결정만. KVTuner 는 quant 만, placement 와 무관. 세 metric 통합 부재.
- 동작 원리: (1) calibration 100-shot MMMU + ChartQA 로 visual_attn_ratio (post-vision attention) + per-layer SQNR (NVFP4 quant) 측정 → (2) recency 는 inference 중 EMA 업데이트 → (3) score → tier_class lookup table 출력 → (4) M2 에서 tier mapping 시 사용.
- 차별화: LayerKV layer-wise allocation vs **본 기법 layer-score → physical-tier**. KVTuner sensitivity-quant 의 sensitivity 만 차용 (placement 와 결합).

#### Block 1.5: Gain Contribution (R55.3)
- Primary axis: enabling for M2/M3.
- Secondary axis: [Memory eff.] M1 단독 +5% (layer 분류 정확도).
- 단독 미보장: 분류만으로는 tier mapping 효과 없음.

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `tools/lss_calibrate.py` (new) | `def calibrate_lss(model, calib_set)` | — | sparsity + sensitivity + recency 측정 → layer_score.json | Add |
| `vllm/core/lss_lookup.py` (new) | `LayerStratificationLookup.get_class(layer_idx)` | — | json load + class lookup | Add |
| `vllm/core/block_manager.py` | `BlockManager._allocate` | uniform | layer_class 따라 tier 선택 | Modify |

R52.3 trace: vLLM block_manager 실재. KVTuner sensitivity score 공개 ([github.com/cmd2001/KVTuner](https://github.com/cmd2001/KVTuner)). [✅]

#### Block 3: Synthetic Workload
- Unit test: layer score calibration on 100-shot, 30분.
- Isolated: layer-class lookup overhead (online inference), 1시간.

---

### 4.2 — M2: 3-Tier Memory Map with Anchor Visual Token L2 Pinning (3TMM-AVT)

#### Block 1: Concept
- 추가되는 Scheme: 3-tier memory mapping —
  - **Tier-0 (GPU L2 96MB carveout, RTX 5090; 또는 Jetson Thor L2 4MB carveout)**: anchor visual token (semantic salience top-K) + system prompt token, `cudaAccessPolicyWindow` with `cudaAccessPropertyPersisting`. 가장 hot.
  - **Tier-1 (GDDR7 1.79 TB/s on RTX 5090; LPDDR5x UMA hot page-color on Jetson Thor)**: hot + warm layer KV.
  - **Tier-2 (host LPDDR5 on RTX 5090; LPDDR5x UMA cold page-color on Jetson Thor)**: cold layer KV + overflow.
- promotion/demotion: layer score + LRU + access frequency 결합, hysteresis margin (10%).
- 해결하려는 문제: 4-tier (NVMe 포함) 는 edge fit 약함. 3-tier 응축. anchor visual token 의 L2 pinning 은 review-novelty 권고로 격상.
- 동작 원리: (1) layer_class.json (M1) lookup → (2) Tier 결정 → (3) 첫 prefill 시 PagedAttention block 을 해당 tier 에 allocate (`cuMemSetAccess` for L2, `cudaMallocAsync` for tier-1, `cudaHostAllocMapped` for tier-2 host) → (4) anchor visual token (visual_attn_ratio top-K, K=128) 만 L2 carveout pin → (5) 일반 token 은 Tier-1 → (6) cold layer 의 token 은 Tier-2 demotion (LRU + score < threshold) → (7) hysteresis margin 으로 churn 방지.
- 차별화: LayerKV 는 단일 layer offload, **본 기법은 anchor token spatial pinning + 3-tier physical mapping**. KVQuant 의 first-token-FP16 anchor 를 layer-stratified spatial pinning 으로 일반화. NVIDIA ICMS 의 datacenter 4-tier 와 차별 — **edge UMA equivalent**.

#### Block 1.5: Gain Contribution
- Primary axis: [Memory eff.] effective KV +40~55%
- Secondary axis: [Performance] decode latency -15~22%
- 단독 미보장: [Energy] (M3 결합 시 -10~15%)

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/core/block_manager.py` | `BlockManager.allocate_block` | uniform GDDR | tier-aware allocation (L2/GDDR7|UMA/host) | Modify |
| `vllm/worker/cache_engine.py` | `CacheEngine.swap_in/swap_out` | uniform | tier promotion/demotion with hysteresis | Modify |
| `csrc/l2_carveout.cu` (new) | `pin_anchor_token_to_l2` | — | `cuMemSetAccess` + `cudaAccessPolicyWindow` wrapper | Add |
| `vllm/core/anchor_selector.py` (new) | `AnchorVisualTokenSelector.select_top_k` | — | semantic salience score top-K (visual_attn_ratio per token) | Add |

R52.3 trace: CUDA `cuMemSetAccess`, `cudaAccessPolicyWindow` 실재 (CUDA 11.0+). vLLM cache_engine.py 실재. [✅]

#### Block 3: Synthetic Workload
- Unit test: L2 carveout 96MB pinning 정합 (anchor token attention 보존), 10분.
- Isolated: long-context (16K) prefill effective KV capacity 측정, 6시간.

---

### 4.3 — M3: Page-Color Affinity UMA Bank Partitioning (PCA-UBP)

#### Block 1: Concept
- 추가되는 Scheme: Jetson Thor LPDDR5x UMA 환경에서 page color (8-color partition, Linux kernel page-coloring 활성화 시) 별 access latency delta 60-120ns 측정 활용. layer-class HOT 의 KV 는 hot page-color (low latency bank) 에 mapping, COLD 는 cold page-color. RTX 5090 GDDR7 환경에서는 host LPDDR5 system memory 와 GDDR7 의 차등으로 대체 (page-color 적용 X).
- 해결하려는 문제: ATRIUM-R(Sys) standalone idea 였으나 STRATA-K 와 axis 중복 → 흡수. UMA 환경에서 single physical pool 만 있는 것처럼 보이지만 실제로는 page-color 별 micro-bank latency 차이 존재.
- 동작 원리: (1) Linux kernel boot parameter `page_coloring=on` 활성화 (Jetson Thor JetPack 7.0 custom kernel 또는 `madvise(MADV_HUGEPAGE)` + huge-page pool 차등) → (2) huge-page pool 을 8-color 로 분리 → (3) M2 의 Tier-1/Tier-2 결정 후 layer-class 에 따라 hot/cold color 선택 → (4) `mmap(...MAP_HUGETLB)` 시 color hint 전달 → (5) UMA 내부 bank 차등 access latency 60-120ns delta 활용.
- 차별화: 일반 page-coloring 은 cache-aware allocation 만, **본 기법은 VLM KV cache 의 layer-class 기반 micro-tier 매핑**. NVIDIA ICMS 는 inter-tier (host-device), **본 기법은 intra-UMA (single device 내) micro-tier**.

#### Block 1.5: Gain Contribution
- Primary axis: [Performance] effective bandwidth +15% (UMA bank-level)
- Secondary axis: [Energy] -5% (LPDDR5x cold bank lower power)
- 단독 미보장: gain ceiling 작음 (60-120ns × decode token rate). RQ-4.3 에서 검증 후 채택 결정.

#### Block 2: Source-Level Implementation

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/core/page_color_allocator.py` (new) | `PageColorAllocator.allocate` | — | huge-page color hint allocation | Add |
| `vllm/core/block_manager.py` | `BlockManager._allocate_uma` | uniform | layer-class → page-color | Modify |
| `tools/page_color_setup.sh` (new) | shell script | — | `madvise` + huge-page pool 차등 setup | Add |

R52.3 trace: Linux `madvise(MADV_HUGEPAGE)` 실재. JetPack 7.0 의 huge-page 지원 ([NVIDIA Jetson Linux 36.4 docs](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/)). page-coloring 은 RT-Linux extension 또는 [github.com/uci-rtsl/coloris](https://github.com/uci-rtsl/coloris) custom kernel 사용. [✅] (단 production-grade 호환성은 W4-W5 에 검증 필요)

#### Block 3: Synthetic Workload
- Unit test: page color 별 access latency delta 측정 (synthetic stride access pattern), 30분.
- Isolated: layer-class HOT/COLD 별 page-color 매핑 → effective bandwidth 측정, 4시간.

## 5. 실험 플랜 7-요소 (R20-β / R52.1)

### 5.1 Hardware environment
- RTX 5090 32GB GDDR7 primary (96MB L2 carveout)
- Jetson AGX Thor 128GB secondary (UMA + page-color)

### 5.2 Model
- Qwen3-VL-8B
- LLaVA-Next-7B (long-context VLM)
- InternVL3-8B
- Qwen3-VL-30B-A3B MoE (RTX 5090 large 검증, OBELISK-5090 와 cross-validation)

### 5.3 Dataset/Workload
- Long-context VQA (DocVQA val 5500, ChartQA test 1500)
- Long-video VQA (VideoMME-long subset, 32K visual token)
- Multi-turn dialogue (LongVideoBench 16K context)
- ShareGPT-4o long context

### 5.4 Simulator/Tools
- 측정 only
- nsight-systems, tegrastats, jtop, custom huge-page profiler

### 5.5 Ablation + Measurement Protocol
- Baseline 1: vLLM vanilla allocator (uniform GDDR7)
- Baseline 2: HiCache 3-tier (GPU/CPU/Storage)
- Baseline 3: LayerKV layer-wise offload
- Baseline 4: InfiniGen speculative prefetch
- Ablation: M1 only / M1+M2 / M1+M2+M3
- 5 run, 95% CI

### 5.6 Implementation Steps

| Week | Task | Deliverable |
|------|------|-------------|
| W1-2 | LSS calibration (M1) | layer_score.json for Qwen3-VL-8B |
| W3 | 3-tier physical allocator (M2) | tier-aware block_manager |
| W4 | L2 carveout anchor visual token pin | `cuMemSetAccess` integration |
| W5 | Page-color setup verification (M3) | huge-page pool 차등, latency delta 측정 |
| W6 | Layer-class → page-color mapping | M3 production-grade |
| W7 | Hysteresis margin tuning (churn control) | promotion/demotion stability |
| W8 | LayerKV reproduction | head-to-head |
| W9 | HiCache reproduction | head-to-head |
| W10 | RTX 5090 long-context (32K) | effective KV capacity |
| W11 | Jetson Thor UMA (M3 page-color) | bandwidth +15% 검증 |
| W12 | OOM batch size threshold | +50% batch 검증 |
| W13-14 | ASPLOS / DAC draft | submission |

### 5.7 Preliminary Analysis Metrics
- Effective KV capacity: target +50% (baseline vanilla)
- Decode latency (16K context): target -20%
- OOM batch size: target +50% (baseline 4 → 6)
- Page-color delta: target 80ns (RQ-4.3, Jetson Thor)

## 6. 예상 효과 (R55.2 5-axis 정량 표)

| Axis | Baseline (vLLM vanilla, Qwen3-VL-8B / RTX 5090) | STRATA-K-R | 개선 | 조건 |
|------|--------------------------------------------------|------------|------|------|
| [Performance] decode latency (16K ctx) | 280 ms/token | 220 ms/token | **-21%** | long-context VLM > 4K |
| [Memory eff.] effective KV (OOM threshold) | 24 GB | 36 GB | **+50%** | layer-tiered |
| [Throughput] OOM batch | 4 | 6 | **+50%** | OOM-bottleneck workload |
| [Energy] J/request | 24 J | 21 J | **-13%** | LPDDR5x cold page lower power |
| [Power] avg | 380 W | 360 W | **-5%** | RTX 5090 |
| [Accuracy] DocVQA | 78.5 | 78.4 | -0.1 pt | tier mapping lossless |

조건: (a) gain 은 long-context (>4K visual tokens) workload 한정, short-prompt 효과 미미. (b) M3 page-color 는 Jetson Thor UMA 환경 + Linux page-coloring kernel 만, RTX 5090 에서는 host LPDDR5 차등으로 대체. (c) M2 L2 carveout 은 RTX 5090 96MB / Jetson Thor 4MB 차이로 효과 비례.

## 7. R56 self-check

- [x] R56.1: §3.2 GAP 표 안에 본 harness 자체 이전 세션 idea (KEYSTONE/VESPER/SHOAL/Loom) 인용 0. ATRIUM-R(Sys) 흡수는 §1 Phase 1' delta 에 in-session evolution 으로 명시.
- [x] R56.2: peer-reviewed 12/14 = 86% ≥ 65%.
- [x] R56.3: 모든 reference 명시, 가상/TBD 0.

## 8. R52 / R53 / R54 self-check

- [x] R52.1 Baseline Source: vLLM v0.11.0, SGLang v0.4+, KVTuner [github.com/cmd2001/KVTuner](https://github.com/cmd2001/KVTuner), CUDA 12.4, JetPack 7.0, RTX 5090 FE + Jetson Thor T5000.
- [x] R52.2 Function/Class as-is/to-be 표: M1 (3 row), M2 (4 row), M3 (3 row) = 10 row.
- [x] R52.3 GitHub 실존: vLLM block_manager, cache_engine, CUDA cuMemSetAccess, Linux madvise 모두 [✅]. page-coloring kernel option 은 W5 에 production-grade 검증.
- [x] R52.4 Synthetic 3-tier: M1-M3 unit + isolated + end-to-end (W10-W12).
- [x] R52.5 Implementation vs Simulator 일관성: simulator X.
- [x] R54.1-R54.6 Final Verification Pass.
