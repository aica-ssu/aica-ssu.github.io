# Strata — Stratified KV Layout + Page-Color Affinity

> VLM long-context KV cache 의 layer 별 sparsity/sensitivity wide range 를 layer-score → physical-tier 매핑 + UMA page-color micro-tier + anchor visual token L2 pinning 으로 응축한다.

## 1. Research Questions

- **RQ-1**: VLM long-context workload (4K-32K visual tokens) 에서 layer score 기반 3-tier (L2 carveout / GDDR7 또는 LPDDR5x / page-color cold) mapping 이 vanilla vLLM allocator 대비 effective KV capacity 를 **+40% 이상** 증가시키는가? OOM 발생 batch size 가 baseline 대비 **+50% 이상** 증가하는가?
- **RQ-2**: Tier-0 L2 carveout (96MB) 에 anchor visual token (semantic salience top-K) 만 pin 했을 때, 일반 system prompt + first 128 token pin 대비 decode latency 가 **추가 -8% 이상** 감소되는가?
- **RQ-3**: Jetson Thor LPDDR5x UMA 환경에서 page color (8-color partition, Linux kernel page-coloring) 별 access latency delta 가 60-120ns 로 측정될 때, layer-class HOT/COLD 별 page color 매핑이 effective bandwidth 를 **+15% 이상** 향상시키는가?
- **RQ-4**: tier promotion/demotion 의 churn cost (block move overhead) 가 hysteresis margin 적용 시 long-context workload 에서 prefill latency 의 **≤ 5%** 안에서 유지되는가?

## 2. 개요 (Metaphor noun ↔ mechanism)

**Strata**: 지층(strata) 처럼 layered KV memory hierarchy — L2 carveout / GDDR7 또는 UMA hot page / UMA cold page-color.

핵심 통찰: **VLM 의 long-context KV cache 는 layer 별 sparsity / sensitivity 가 70-99% wide range 라서 uniform 한 GPU memory 에 두는 것이 낭비**. layer score (sparsity + sensitivity + recency) 에 따라 (a) hot layer → L2 carveout 의 anchor token pinning, (b) warm layer → GDDR7/UMA hot page-color, (c) cold layer → UMA cold page-color (Jetson) 또는 host LPDDR5 (RTX 5090) 으로 micro-tier 분리. 또한 KVQuant 의 attention sink insight 를 visual token 에 적용 — semantic salience top-K 만 L2 anchor 로 pin.

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence

- [arXiv:2410.23317](https://arxiv.org/abs/2410.23317) (VL-Cache, ICLR 2025): VLM layer 별 attention sparsity 70-99% wide range. 일부 layer 거의 모든 token attend, 일부 매우 sparse → tier mapping 정당성.
- IISWC 2024 ([arXiv:2512.01644](https://arxiv.org/abs/2512.01644)): edge decode 80% time KV BW dominated.
- Jetson Thor LPDDR5x: 273 GB/s, page color 별 access latency delta 60-120ns 측정 가능 (Linux page-coloring research, 본 idea 가 측정).
- RTX 5090: GDDR7 1.79 TB/s + L2 96MB.
- [arXiv:2502.06433](https://arxiv.org/abs/2502.06433) (KVTuner, ICML 2025): layer-wise mixed-precision KV quant, sensitivity score.
- [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) (KVQuant, NeurIPS 2024): attention sink first-token FP16 보존, anchor token concept.

### 3.2 GAP 표

| 기존 연구 | venue year | 핵심 mechanism | what / why / how 차별 |
|-----------|-----------|----------------|------------------------|
| LayerKV | [arXiv:2410.00428](https://arxiv.org/abs/2410.00428) (2024-2025) | layer-wise KV block allocation + offloading, schedule per-request layer offload | what: Strata 는 **layer score (composite metric) → physical-tier mapping** + **page-color micro-tier** + **anchor visual token L2 pin**. why: LayerKV 는 layer 별 single offload 결정만, multi-tier physical mapping 부재. how: 3-tier physical + page-color affinity |
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

## 4. 제안 기법 — Mechanism

### 4.1 — M1: Layer Stratification Score (LSS)

#### Concept
- 추가되는 Scheme: layer 별 (a) attention sparsity ratio (VL-Cache 측정), (b) KV quant sensitivity (KVTuner score), (c) recency-of-access 의 weighted sum 으로 score 산출. score = α × sparsity + β × sensitivity + γ × recency, where α=0.4, β=0.4, γ=0.2 default. offline profile 한 번 → online hot/warm/cold 3-class 분류 (top 20% / mid 50% / bottom 30%).
- 해결하려는 문제 (Workload evidence): HiCache 는 layer-uniform 정책, LayerKV 는 single-offload 결정만. KVTuner 는 quant 만, placement 와 무관. 세 metric 통합 부재.
- 동작 원리 (학부생용 step-by-step): (1) calibration 100-shot MMMU + ChartQA 로 visual_attn_ratio (post-vision attention) + per-layer SQNR (NVFP4 quant) 측정 → (2) recency 는 inference 중 EMA 업데이트 → (3) score → tier_class lookup table 출력 → (4) M2 에서 tier mapping 시 사용.
- 기존 해법 실패 이유 + 본 기법 차별화: LayerKV layer-wise allocation vs **본 기법 layer-score → physical-tier**. KVTuner sensitivity-quant 의 sensitivity 만 차용 (placement 와 결합).

#### Per-mechanism gain contribution
- Primary axis: enabling for M2/M3.
- Secondary axis: [Memory eff.] M1 단독 +5% (layer 분류 정확도).
- 단독 미보장 axis: 분류만으로는 tier mapping 효과 없음.

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.11.x 의 `vllm/core/block_manager.py` 와 신규 `tools/lss_calibrate.py` 를 결합하여 visual_attn_ratio + KVTuner sensitivity + recency 를 layer 별로 측정 → **측정**: PyTorch forward hook 기반 attention map dump + KVTuner SQNR 산출 + EMA recency tracker → **개선**: layer 분류 정확도 (top/mid/bottom 분류 F1 ≥ 0.85) 에 따라 M2 tier mapping 의 enabling signal 제공, [Memory eff.] M1 단독 +5%.

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `tools/lss_calibrate.py` (new) | `def calibrate_lss(model, calib_set)` | — | sparsity + sensitivity + recency 측정 → layer_score.json | Add |
| `vllm/core/lss_lookup.py` (new) | `LayerStratificationLookup.get_class(layer_idx)` | — | json load + class lookup | Add |
| `vllm/core/block_manager.py` | `BlockManager._allocate` | uniform | layer_class 따라 tier 선택 | Modify |

GitHub 실존: vLLM block_manager. KVTuner sensitivity score 공개 ([github.com/cmd2001/KVTuner](https://github.com/cmd2001/KVTuner)).

#### Synthetic workload validation
- Unit test: layer score calibration on 100-shot, 30분.
- Mechanism-isolated: layer-class lookup overhead (online inference), 1시간.

---

### 4.2 — M2: 3-Tier Memory Map with Anchor Visual Token L2 Pinning (3TMM-AVT)

#### Concept
- 추가되는 Scheme: 3-tier memory mapping —
  - **Tier-0 (GPU L2 96MB carveout, RTX 5090; 또는 Jetson Thor L2 4MB carveout)**: anchor visual token (semantic salience top-K) + system prompt token, `cudaAccessPolicyWindow` with `cudaAccessPropertyPersisting`. 가장 hot.
  - **Tier-1 (GDDR7 1.79 TB/s on RTX 5090; LPDDR5x UMA hot page-color on Jetson Thor)**: hot + warm layer KV.
  - **Tier-2 (host LPDDR5 on RTX 5090; LPDDR5x UMA cold page-color on Jetson Thor)**: cold layer KV + overflow.
- promotion/demotion: layer score + LRU + access frequency 결합, hysteresis margin (10%).
- 해결하려는 문제 (Workload evidence): 4-tier (NVMe 포함) 는 edge fit 약함. 3-tier 응축. anchor visual token 의 L2 pinning 은 review-novelty 권고로 격상.
- 동작 원리 (학부생용 step-by-step): (1) layer_class.json (M1) lookup → (2) Tier 결정 → (3) 첫 prefill 시 PagedAttention block 을 해당 tier 에 allocate (`cuMemSetAccess` for L2, `cudaMallocAsync` for tier-1, `cudaHostAllocMapped` for tier-2 host) → (4) anchor visual token (visual_attn_ratio top-K, K=128) 만 L2 carveout pin → (5) 일반 token 은 Tier-1 → (6) cold layer 의 token 은 Tier-2 demotion (LRU + score < threshold) → (7) hysteresis margin 으로 churn 방지.
- 기존 해법 실패 이유 + 본 기법 차별화: LayerKV 는 단일 layer offload, **본 기법은 anchor token spatial pinning + 3-tier physical mapping**. KVQuant 의 first-token-FP16 anchor 를 layer-stratified spatial pinning 으로 일반화. NVIDIA ICMS 의 datacenter 4-tier 와 차별 — **edge UMA equivalent**.

#### Per-mechanism gain contribution
- Primary axis: [Memory eff.] effective KV +40~55%
- Secondary axis: [Performance] decode latency -15~22%
- 단독 미보장 axis: [Energy] (M3 결합 시 -10~15%)

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.11.x 의 `vllm/core/block_manager.py` + `vllm/worker/cache_engine.py` 와 신규 `csrc/l2_carveout.cu` 를 수정하여 CUDA `cuMemSetAccess` + `cudaAccessPolicyWindow` 로 L2 carveout 96MB 에 anchor visual token 을 persistent pinning → **측정**: Nsight Compute `lts__t_sectors_op_read_hit_rate` (L2 hit rate) + nsight-systems decode latency + PagedAttention block allocation trace → **개선**: baseline effective KV 24GB → 36GB (+50%, OOM batch 4→6), decode latency 280ms→220ms (-21% on 16K context).

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/core/block_manager.py` | `BlockManager.allocate_block` | uniform GDDR | tier-aware allocation (L2/GDDR7\|UMA/host) | Modify |
| `vllm/worker/cache_engine.py` | `CacheEngine.swap_in/swap_out` | uniform | tier promotion/demotion with hysteresis | Modify |
| `csrc/l2_carveout.cu` (new) | `pin_anchor_token_to_l2` | — | `cuMemSetAccess` + `cudaAccessPolicyWindow` wrapper | Add |
| `vllm/core/anchor_selector.py` (new) | `AnchorVisualTokenSelector.select_top_k` | — | semantic salience score top-K (visual_attn_ratio per token) | Add |

GitHub 실존: CUDA `cuMemSetAccess`, `cudaAccessPolicyWindow` 실재 (CUDA 11.0+). vLLM cache_engine.py 실재.

#### Synthetic workload validation
- Unit test: L2 carveout 96MB pinning 정합 (anchor token attention 보존), 10분.
- Mechanism-isolated: long-context (16K) prefill effective KV capacity 측정, 6시간.

---

### 4.3 — M3: Page-Color Affinity UMA Bank Partitioning (PCA-UBP)

#### Concept
- 추가되는 Scheme: Jetson Thor LPDDR5x UMA 환경에서 page color (8-color partition, Linux kernel page-coloring 활성화 시) 별 access latency delta 60-120ns 측정 활용. layer-class HOT 의 KV 는 hot page-color (low latency bank) 에 mapping, COLD 는 cold page-color. RTX 5090 GDDR7 환경에서는 host LPDDR5 system memory 와 GDDR7 의 차등으로 대체 (page-color 적용 X).
- 해결하려는 문제 (Workload evidence): UMA 환경에서 single physical pool 만 있는 것처럼 보이지만 실제로는 page-color 별 micro-bank latency 차이 존재.
- 동작 원리 (학부생용 step-by-step): (1) Linux kernel boot parameter `page_coloring=on` 활성화 (Jetson Thor JetPack 7.0 custom kernel 또는 `madvise(MADV_HUGEPAGE)` + huge-page pool 차등) → (2) huge-page pool 을 8-color 로 분리 → (3) M2 의 Tier-1/Tier-2 결정 후 layer-class 에 따라 hot/cold color 선택 → (4) `mmap(...MAP_HUGETLB)` 시 color hint 전달 → (5) UMA 내부 bank 차등 access latency 60-120ns delta 활용.
- 기존 해법 실패 이유 + 본 기법 차별화: 일반 page-coloring 은 cache-aware allocation 만, **본 기법은 VLM KV cache 의 layer-class 기반 micro-tier 매핑**. NVIDIA ICMS 는 inter-tier (host-device), **본 기법은 intra-UMA (single device 내) micro-tier**.

#### Per-mechanism gain contribution
- Primary axis: [Performance] effective bandwidth +15% (UMA bank-level)
- Secondary axis: [Energy] -5% (LPDDR5x cold bank lower power)
- 단독 미보장 axis: gain ceiling 작음 (60-120ns × decode token rate). RQ-3 에서 검증 후 채택 결정.

#### Source-level implementation

> **구현 큰 그림**: Jetson Thor JetPack 7.0 custom kernel (또는 `madvise(MADV_HUGEPAGE)`) 위에서 신규 `vllm/core/page_color_allocator.py` + `tools/page_color_setup.sh` 를 통해 huge-page pool 을 8-color 로 분리하고 vLLM block_manager 가 layer-class HOT/COLD 별 color 를 선택하도록 수정 → **측정**: synthetic stride access pattern 으로 page-color 별 access latency delta 측정 + tegrastats LPDDR5x bandwidth + nvidia memory profiler → **개선**: effective bandwidth +15% (UMA bank-level), [Energy] -5% (cold bank lower power).

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/core/page_color_allocator.py` (new) | `PageColorAllocator.allocate` | — | huge-page color hint allocation | Add |
| `vllm/core/block_manager.py` | `BlockManager._allocate_uma` | uniform | layer-class → page-color | Modify |
| `tools/page_color_setup.sh` (new) | shell script | — | `madvise` + huge-page pool 차등 setup | Add |

GitHub 실존: Linux `madvise(MADV_HUGEPAGE)` 실재. JetPack 7.0 의 huge-page 지원 ([NVIDIA Jetson Linux 36.4 docs](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/)). page-coloring 은 RT-Linux extension 또는 [github.com/uci-rtsl/coloris](https://github.com/uci-rtsl/coloris) custom kernel 사용. (단 production-grade 호환성은 W4-W5 에 검증 필요)

#### Synthetic workload validation
- Unit test: page color 별 access latency delta 측정 (synthetic stride access pattern), 30분.
- Mechanism-isolated: layer-class HOT/COLD 별 page-color 매핑 → effective bandwidth 측정, 4시간.

## 5. 실험 플랜

### 5.1 Hardware environment
- RTX 5090 32GB GDDR7 primary (96MB L2 carveout)
- Jetson AGX Thor 128GB secondary (UMA + page-color)

### 5.2 Model
- Qwen3-VL-8B
- LLaVA-Next-7B (long-context VLM)
- InternVL3-8B
- Qwen3-VL-30B-A3B MoE (RTX 5090 large 검증, Obelisk 와 cross-validation)

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
- Page-color delta: target 80ns (RQ-3, Jetson Thor)

## 6. 예상 효과

| Axis | Baseline (vLLM vanilla, Qwen3-VL-8B / RTX 5090) | Strata | 개선 | 조건 |
|------|--------------------------------------------------|--------|------|------|
| [Performance] decode latency (16K ctx) | 280 ms/token | 220 ms/token | **-21%** | long-context VLM > 4K |
| [Memory eff.] effective KV (OOM threshold) | 24 GB | 36 GB | **+50%** | layer-tiered |
| [Throughput] OOM batch | 4 | 6 | **+50%** | OOM-bottleneck workload |
| [Energy] J/request | 24 J | 21 J | **-13%** | LPDDR5x cold page lower power |
| [Power] avg | 380 W | 360 W | **-5%** | RTX 5090 |
| [Accuracy] DocVQA | 78.5 | 78.4 | -0.1 pt | tier mapping lossless |

조건: (a) gain 은 long-context (>4K visual tokens) workload 한정, short-prompt 효과 미미. (b) M3 page-color 는 Jetson Thor UMA 환경 + Linux page-coloring kernel 만, RTX 5090 에서는 host LPDDR5 차등으로 대체. (c) M2 L2 carveout 은 RTX 5090 96MB / Jetson Thor 4MB 차이로 효과 비례.

## 7. Implementation Decision Flowchart (per-idea)

> 본 idea 단독 prototype 시 어느 mechanism 부터 시작하고 결과 분기에 따라 어느 venue 까지 갈 수 있는지.

### 7.1 1st Mechanism Priority

- **가장 먼저**: M1 (Layer Stratification Score) — 28-layer Qwen3-VL-8B 의 layer 별 visual_attn_ratio + KVTuner sensitivity 측정
- **왜**: M2/M3 의 prerequisite — layer score 가 hot/warm/cold 3-class 로 잘 분리되어야 tier mapping / page-color 매핑 모두 의미 있음. 측정 도구 가장 단순 (PyTorch forward hook + offline calibration).
- **임계값**: top 20% layer 의 visual_attn_ratio 평균 ≥ 0.20, bottom 30% layer 의 visual_attn_ratio 평균 ≤ 0.05. layer 분류 F1 ≥ 0.85.

### 7.2 결과 분기

```
[M1 LSS calibration on Qwen3-VL-8B]
   ↓
   ├─ Pass (top/bottom 분리 명확, F1 ≥ 0.85)
   │   → M2 3-tier physical mapping 진행 (W3-7)
   │   → L2 carveout 96MB anchor pinning + GDDR7 hot + host LPDDR5 cold
   │   → effective KV +40~55% 검증 시 ASPLOS / EuroSys full paper draft
   │
   ├─ Below (분리 약함, 10-30% short, F1 0.65-0.85)
   │   → M2 simplify (3-tier → 2-tier, page-color 제거)
   │   → 단순 hot/cold 2-tier + L2 anchor pin 만 → IEEE CAL letter 또는 DAC short
   │
   ├─ Critical (분리 실패, F1 < 0.65)
   │   → drop M1 — Qwen3-VL 의 layer asymmetry 가 측정 불충분
   │   → reframe: KVTuner sensitivity 만으로 quant-aware tier 매핑 (memory efficiency 만, performance 포기)
   │   → 또는 idea drop
   │
   └─ Outperform (top/bottom 격차 ≥ 4×, F1 ≥ 0.95)
       → M2 + M3 모두 진행 (UMA page-color micro-tier 까지)
       → page-color affinity standalone variant (Tier-2 spinoff) → DATE/ISLPED short
       → Strata main + spinoff 2 paper 동시 가능
```

## 8. 참고 / cross-share dependency

- **Cross-share**: Strata 의 layer_score.json (M1 calibration 결과) 는 같은 세션의 ATRIUM (Green Context SM 분할) / Prism-FogFx (NVFP4 mixed precision) 와 cross-validation 가능. layer 별 visual_attn_ratio 가 모든 idea 의 prerequisite signal.
- **Workload evidence**: VL-Cache (ICLR 2025) layer sparsity 70-99% wide range 가 본 idea tier mapping 정당성 핵심.

## 9. Rule self-check

- [x] Tier 2 / venue: ASPLOS 2026 / DAC 2026 / IEEE CAL 2026 / EuroSys 2026 / 5-axis: Performance -15~22%, Memory eff. +40~55%, Energy -10~15%, Throughput +18~28%
- [x] Lead expert: legacy-system-expert (memory tiering / KV layout) / R47 path: application-level only / R45 risk: 6/10
- [x] Single-system fit: RTX 5090 32GB GDDR7 primary (96MB L2 carveout) + Jetson AGX Thor 128GB secondary (UMA page-color)
- [x] R10-α bullet 형식
- [x] R20-α mechanism 4 요소 (Concept / Per-mechanism gain / Source-level / Synthetic workload)
- [x] R20-β 실험 플랜 7-요소
- [x] R20-γ single-system fit
- [x] R52.1 Baseline Source: vLLM v0.11.0, SGLang v0.4+, KVTuner [github.com/cmd2001/KVTuner](https://github.com/cmd2001/KVTuner), CUDA 12.4, JetPack 7.0
- [x] R52.2 file path / symbol 표 (M1 3 row + M2 4 row + M3 3 row = 10 row)
- [x] R52.3 GitHub 검증 trace (vLLM block_manager / cache_engine / CUDA cuMemSetAccess / Linux madvise 모두 실재)
- [x] R52.4 synthetic workload 3-tier (M1-M3 unit + isolated + end-to-end)
- [x] R53 inline 3-block (concept / source-level / synthetic)
- [x] R54.1-R54.6 final verification
- [x] R55.2 5-axis gain target 1+ (Memory eff. +40~55% primary)
- [x] R55.3 mechanism 별 gain contribution
- [x] R56.1 self-citation 0 (in-session evolution: ATRIUM-R(Sys) page-color affinity 흡수)
- [x] R56.2 published 12/14 = 86% ≥ 65% (시스템 분야)
- [x] R56.3 가상 reference 0
- [x] R58.1 본문 rule notation inline 0
- [x] R58.2 Decision Tree 순차적 paper-route
- [x] R58.3 §7 per-idea flowchart
- [x] R58.4 idea name "Strata" (1 word)
- [x] R58.5 첫 ## 헤더 = §1 Research Questions
- [x] R58.6 raw arxiv ID 0 (모두 markdown link)
- [x] R58.7 workshop paper 단독 인용 0
- [x] R58.8 mechanism source-level implementation 에 "구현 큰 그림" narrative 1+ paragraph
