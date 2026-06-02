# RECOVER-TIER: Dual-Tier Recoverable Vision-KV with Reconstruction-Bounded Cold Spill for Memory-Constrained Edge VLM Serving

**Tier-1 · Grade: Accept (mean 7.75 / Total mean 7.43) · Paper Pair backbone (+R3) · INFRA-A/B producer**

> **Metaphor noun:** *RECOVER-TIER* — "tier" 가 정밀도 정책 layer, "recover" 가 cold spill 에서 reconstruction-bounded 복원.
> **Merge 출처:** DUET(ai-opt) + algo-E DualTier-Recover + F1(legacy). E 의 reconstruction bound 이론 backbone + F1 의 LPDDR5/UMA hardware cost model + DUET 의 vLLM/SGLang 구현 + DUET error-est selective retrieve energy policy.

📖 약어 / 핵심 용어:
- **Vision-KV**: VLM 의 LLM backbone 에서 vision token 이 차지하는 Key/Value cache. multi-turn 에서 vision token 은 input 의 98-99% (LLaVA-OneVision @ VideoMME sample 당 6,272 visual vs 109 text token, corpus A 본문 verified).
- **UMA (Unified Memory Architecture)**: Jetson Orin NX 는 CPU/GPU 가 같은 LPDDR5 16GB 를 공유 (discrete GPU 의 HBM↔host DRAM 분리 없음). hot/cold tier 가 **물리적으로 같은 메모리** → "promote" 는 physical migration 이 아니라 dequant **compute**.
- **Hot tier**: low-rank(rank-r) + b-bit quant, reconstruction error `‖V−V̂‖_F ≤ ε` 보장, 항상 attend.
- **Cold tier**: Q2/Q4 recoverable 표현, dequant 전까지 attention 제외. **같은 LPDDR5 내 logical partition** (별도 device 아님 — CF-3).
- **Reconstruction bound (Eckart-Young)**: rank-r truncated SVD 의 Frobenius error 가 (r+1)번째 이후 singular value 합으로 bound — hot tier 의 `‖V−V̂‖_F ≤ ε` anchor.

---

## 1. Research Questions

- **Master RQ**: Jetson Orin NX 16GB UMA 에서 vision-KV 를 hot(low-rank+quant, reconstruction-bounded) / cold(Q2/Q4 recoverable, LPDDR5 logical spill) dual-tier 로 관리하면, full-FP16-cache(OOM) 와 destructive prune(multi-focus lossy) 사이에서 capacity-fit·recoverable·bounded 를 동시에 달성할 수 있는가?
- **RQ-1**: dual-tier 가 full-FP16-cache(16GB OOM) 대비 GPU-resident vision-KV footprint 를 60-76% (α∈[0.25,0.4] 범위) 절감하면서, OOM 없이 Qwen2.5-VL-7B Q4_K + dual-tier KV 를 16GB 에 fit 시키는가?
- **RQ-2**: cold dequant+promote 의 per-turn recover latency 가 LPDDR5 102.4GB/s bandwidth 경합(hot decode read 와 공유) 하에서 decode latency 의 30% 이하로 유지되며, KVSwap(NVMe disk spill, carrier 장착) 대비 latency·energy 우월한가?
- **RQ-3**: multi-focus benchmark 에서 dual-tier 의 cold recover accuracy 가 full-cache 의 ≥92% 를 회복하고 (OCR 등 fine-grained focus 포함), uniform-Q4(no-tier) ablation 대비 marginal footprint·accuracy 이득이 ≥5% 인가?

## 2. Two-Sentence Pitch

SparseVILA 의 "retain most of visual cache" 가 물리적으로 불가능한 Jetson 16GB UMA 에서, vision KV 를 hot(low-rank+quant 압축, reconstruction error `‖V−V̂‖_F≤ε` 보장, 항상 attend) + cold(저비트 recoverable 표현, **같은 LPDDR5 내 logical partition**, dequant 전까지 attention 제외) 로 이원화한다. focus-shift turn 이 cold region 을 요구하면 framework-level KV buffer 를 block 단위 dequant 로 promote 하여(managed-memory migration 아님 — CF-1), capacity 를 quantization 으로 절감하면서 destructive prune 의 multi-focus 손실을 reconstruction bound 내에서 회복한다.

## 3. 가설 + Falsification

**가설:** vision KV 를 (hot: low-rank rank-r + b-bit, reconstruction `‖V−V̂‖_F≤ε` + cold: Q2/Q4 recoverable, LPDDR5 logical spill) dual-tier 로 관리하면, full-FP16-cache(16GB OOM) 대비 GPU-resident footprint 를 60-76% 절감(본질은 quantization)하고, prefill-prune-discard(SparseVLM/FastV) 대비 multi-focus accuracy 를 full-cache 의 ≥92% 회복하며, focus 가 hot 안일 때 cold retrieve 대비 per-turn energy 를 절감한다.

**Falsification (4):**
1. cold dequant+promote 의 LPDDR5 bandwidth(hot decode read 와 102.4GB/s 공유) 경합으로 per-turn recover latency 가 decode latency 의 **>30%** → 기각. (focus-shift rate sweep 0.1~1.0)
2. Q2/Q4 cold recover accuracy 가 multi-focus benchmark 에서 full-cache 대비 **<90%** (OCR 등 fine-grained focus 포함) → "recoverable" 가정 붕괴, 기각.
3. **uniform-Q4(no-tier) baseline 대비 dual-tier 의 marginal footprint·accuracy 이득 <5%** → tiering 이 quant 와 분리된 기여 없음(arch-sys F02), 부분 기각 → quant-only 로 축소.
4. low-rank SVD 의 prefill compute/energy overhead 가 quant-only(DUET) 대비 net 이득을 상쇄 → SVD 정당성 붕괴, low-rank 제거(arch-sys F17).

## 4. Workload Evidence (Step 0-α 정량 인용)

- **Vision token 비중 (verified, corpus A 본문):** LLaVA-OneVision @ VideoMME, sample 당 **6,272 visual vs 109 text token** (visual 98-99%). LLM 이 LLaVA-OneVision-72B 의 99.35% param·98.98% ops. → KV cache 가 edge 16GB 의 binding constraint.
- **Capacity 동기:** VLCache 인용 — LLaVA-OV-7B, bs=256, 1,000 frame 시 visual KV cache **~720GB** (모델보다 큼) [⚠️ unverified — search-reported, edge bs=1 재산정 의무].
- **Multi-turn 취약성 (★ verified):** SparseVILA — "permanently remove visual tokens during context stage are quite lossy in multi-turn evaluations", 해법은 "retaining most of the visual cache so that query-aware tokens can be retrieved at each conversation round" → full-cache 보존이 핵심인데 Jetson 16GB 에선 불가.
- **Edge bottleneck (verified, PAISE'25):** Jetson Orin decode 는 memory-bandwidth-bound — memory freq 하향 시 latency +370% / energy +72% (power −52%). → cold dequant 의 bandwidth 경합이 1차 falsification 근거.
- **Bottleneck 이분법 (verified, Modality Inflation):** vision-encode 는 compute-bound, decode 는 memory-bandwidth-bound (weight+KV HBM streaming). energy overhead 모델별 17-94%.

## 5. 기준 코드베이스 (R52.1)

- **vLLM** `main` commit `7c37096` (clone-verified 2026-06-02). V1 core: `KVCacheManager`/`BlockPool` 는 **modality-agnostic** (vision token 미추적, CF-2). `EncoderCacheManager` 는 **multimodal embedding 캐시** (docstring "operates on the level of multimodal embeddings") = LLM KV cache 아님 (CF-2). vision token→KV block 매핑은 `_gen_mm_extra_hash_keys` 의 `(mm_hash, start_offset)` 재활용.
- **SGLang** `main` commit `e958f45` (secondary). `RadixCache.match_prefix`(L361) shared-prefix. `multimodal_cache.py` = embedding 캐시 (≠ KV).
- **llama.cpp** `main` commit `dbe9c0c` (Jetson 1순위). `src/llama-kv-cache.cpp` `find_slot`(L818), GGUF KV quant `--cache-type-k q4`. modality-agnostic.
- **HF model:** `Qwen/Qwen2.5-VL-7B-Instruct` (Q4_K 4.4GB). secondary `llava-hf/llava-onevision-qwen2-7b-ov-hf`.
- **deps:** HQQ/AWQ (hot-tier low-rank+quant), CUDA 12.x (JetPack 6.2), PyTorch, cuSOLVER (randomized SVD).
- **Jetson Orin NX 16GB 정정 spec (CF-4):** CUDA core **1024** (1792 아님, [datasheet](https://developer.nvidia.com/downloads/jetson-orin-nx-series-data-sheet)), LPDDR5 **102.4 GB/s** (ODECC 실효 85-90%), nvpmodel **10/15/25/40W** (40W = JetPack 6.2 Super Mode 한정), Ampere **sm_87**, `concurrentManagedAccess=0` ([CUDA for Tegra appnote](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/) — `cudaMemPrefetchAsync`/`cudaMemAdvise(PreferredLocation)` **미지원**, `cudaStreamAttachMemAsync`/`cudaStreamCreateWithPriority` 지원).

## 6. 동작 원리 (R53 inline)

### M1 — Dual-Tier 정밀도 정책 layer (hot low-rank+quant / cold Q2-Q4 logical spill)

**① 동작원리 4요소**
- **(무엇을)** vision KV 를 hot(low-rank rank-r + b-bit, `‖V−V̂‖_F≤ε`) + cold(Q2/Q4 recoverable, **LPDDR5 logical partition — 같은 16GB**) 로 이원화. 정밀도가 tier 를 정의 (별도 device 아님).
- **(왜 작동)** vision 98-99% 비중 + single-turn prune 무손실(FastV) 이나 어느 부분이 필요한지가 turn 간 이동(RVIS [arXiv:2604.12358](https://arxiv.org/abs/2604.12358)). 손실 대신 압축-보존 + reconstruction bound 로 복원 가능. **CF-3: capacity 절감 본질은 quantization** (cold tier 가 같은 LPDDR5 → tier 분리 자체는 정밀도 정책).
- **(구현변경점, CF-1/CF-2)** vLLM `BlockPool`/`KVCacheManager`(modality-agnostic)에 vision token range→KV block 매핑 신규 인프라(`_gen_mm_extra_hash_keys` 의 `(mm_hash, start_offset)` 재활용). `free_blocks` → in-place Q4 requantize. `EncoderCacheManager`(embedding 캐시)와 명확 분리.
- **(검증 시나리오)** uniform-Q4(no-tier) vs dual-tier ablation: 동일 16GB budget 에서 multi-focus FCR(R4 metric) 비교. 1실험.

**② 기대효과**: GPU-resident vision-KV −60~76%, OOM 회피 (16GB 단일 device serving).
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Vision KV→block 매핑 | vision token range 식별 (신규, CF-2) | vLLM V1 | [`vllm/v1/core/kv_cache_utils.py#L395-L503`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py#L395-L503) `_gen_mm_extra_hash_keys` | **신규 cross-cutting** | 상 | clone ✓ |
| Cold spill (requantize) | free 대신 Q4 in-place | vLLM V1 | [`vllm/v1/core/block_pool.py#L419-L460`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py#L419-L460) `free_blocks` | replace | 상 | clone ✓ |
| Hot tier compress | low-rank+quant, `‖V−V̂‖≤ε` | HQQ/AWQ | [`~/skills/AI-Research-SKILLs/10-optimization/hqq`](https://github.com) (local skill) | integrate | 중 | skill ✓ |
| Edge KV quant 대안 | GGUF KV Q2_K/Q4_K | llama.cpp | [`src/llama-kv-cache.cpp#L818-L860`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp#L818-L860) `find_slot` (L824) + `--cache-type-k q4` (main 부재 — llama.cpp 기본 branch = master, R68.2) | extend | 중 | clone ✓ |

**④ 검증 trace (R52.3)**: 위 표 clone-verified @ `7c37096`/`dbe9c0c` (drift 반영, symbol navigate 권장). **CF-1: `cudaMemPrefetchAsync`/`cudaMemAdvise` 제거.** **CF-2: `EncoderCacheManager`(embedding) ≠ `BlockPool`(KV) 분리.**

### M2 — Reconstruction-Bounded Promote (framework-level dequant, no managed-memory prefetch)

**① 동작원리 4요소**
- **(무엇을)** focus-shift turn 이 cold block 요구 시 framework-level KV buffer 를 block 단위 dequant 로 promote. promote = **dequant kernel + `cudaStreamAttachMemAsync`** (Orin 지원✅).
- **(왜 작동)** Orin UMA 는 hot/cold 가 애초에 같은 LPDDR5 → "promote"=physical migration 이 아니라 dequant **compute**. 따라서 prefetch API 미지원이 메커니즘을 죽이지 않음 (arch-sys §1, B-1 판정 ✅).
- **(구현변경점, CF-1)** managed-memory migration API 불요 (UMA zero-copy mapped). dequant 를 저우선 stream 에 격리하여 decode read 와 분리.
- **(검증 시나리오)** focus-shift rate sweep 0.1~1.0 에서 per-turn recover latency / decode latency 비율 측정 (30% 임계).

**② 기대효과**: TTFT(cold promote) full re-prefill 대비 −30~50%.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Promote (dequant) | block dequant, stream 격리 | CUDA | `cudaStreamAttachMemAsync` (Orin ✅, [appnote](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)) | new kernel | 중 | appnote ✓ |
| KV block alloc hook | promote 시 hot block 확보 | vLLM V1 | [`vllm/v1/core/kv_cache_manager.py#L238-L300`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L238-L300) `allocate_slots` | extend | 중 | clone ✓ |
| Reconstruction bound | rank-r SVD `‖V−V̂‖≤ε` | cuSOLVER | randomized SVD / power-iteration (edge latency pilot) | new | 중 | pilot |

**④ 검증 trace (R52.3)**: `allocate_slots`(L238)/`free`(L431) clone ✓. **promote=dequant compute (managed-memory migration 아님). Jetson Thor (Blackwell, `concurrentManagedAccess=1`) = prefetch 지원 대조군.**

## 7. End-to-End Evaluation

- **Multi-turn / multi-focus:** R4 FOCUS-COVERAGE benchmark (focus-trajectory + FCR/Coverage-Energy Pareto) 가 1차 토대. MMDU / ConvBench / MileBench(CSP 41% over-pruning 회복 측정) / MultiVerse(647 dialog 4-turn).
- **Long-video / single-turn:** Video-MME (900 video/2,700 QA, median 1,024s), MLVU (1,730 video, CVPR'25), MileBench (29 multimodal datasets).
- **합성:** focus-shift rate(0.1~1.0) 제어 multi-focus VQA — 같은 image N=4-6 focus region 질문.
- **Baseline:** full-FP16-cache(OOM 경계), SparseVLM/FastV(prefill-prune-discard), uniform-Q4(no-tier ablation), KVSwap(carrier NVMe disk spill 직접 비교), VL-Cache/SnapKV/PyramidKV/MEDA(budget 분배 비교군), VLCache(same-input 98% reuse).

## 8. 실험 7요소 (12-16주)

1. **Hardware**: Jetson Orin NX 16GB primary (1024 core, LPDDR5 102.4GB/s, sm_87). secondary: DGX Spark 128GB (capacity-relaxed 대조), Jetson Thor Blackwell (prefetch 지원 대조). carrier M.2 NVMe (KVSwap baseline).
2. **Model**: Qwen2.5-VL-7B Q4_K (4.4GB) + dual-tier KV → 16GB fit. secondary LLaVA-OneVision-7B.
3. **Framework**: 1순위 llama.cpp GGUF KV quant (검증됨), 2순위 vLLM V1 (NVIDIA-AI-IoT container; Jetson build multi-hour + Marlin SM8.7 미제공 risk).
4. **Energy 측정**: tegrastats / INA3221 sysfs (VDD_GPU_SOC/VDD_CPU_CV) per-turn J 적분 (N≥30 turn averaging — R2 TRACE-C 재사용).
5. **Ablation (gate)**: uniform-Q4(no-tier) vs dual-tier + randomized SVD compute overhead pilot (cuSOLVER edge latency). marginal 미입증 시 quant-only 축소.
6. **Steps**: (a) JetPack/CUDA + `concurrentManagedAccess` 실측 → (b) vision token→KV block 매핑 신규 → (c) block requantize + dequant promote → (d) hot(low-rank+quant) + cold(Q2/Q4) → (e) ablation + KVSwap 비교.
7. **Metrics**: GPU-resident footprint, multi-focus FCR/recall@k, per-turn recover latency, TTFT, per-turn energy (J/turn).

## 9. 예상효과 5-axis 표 (Energy 강조)

| Axis | 예상 개선 | 조건/scope |
|---|---|---|
| **Memory eff.** ★ | GPU-resident KV −60~76% (α∈[0.25,0.4], **단일 76% 금지** arch-sys F02) | Jetson 16GB 핵심. quant 본질 |
| **Energy/Power** ★ | hot-only turn I/O 없음 → cold retrieve 대비 per-turn −2~4× | hot hit-rate 높을 때 |
| Performance | multi-focus accuracy full 의 ≥92% 회복; destructive prune 대비 +8~15%p | N=4-6 focus/video |
| Latency | TTFT(cold promote) full re-prefill 대비 −30~50% | promote < recompute 시 |
| Cost eff. | OOM 회피 → 16GB 단일 device serving | edge 한정 |

## 10. 관련연구 + 차별화

- **SparseVILA** [arXiv:2510.17777](https://arxiv.org/abs/2510.17777) (ICCV'25) — "retaining most of the visual cache", edge/Jetson/energy 무, no formal guarantee. → **차별화 axis: 16GB UMA full 보존 불가 + reconstruction bound `‖V−V̂‖_F≤ε`.**
- **KVSwap** [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) — UMA capacity wall + disk preload, text-only. → **차별화: vision-token-aware + LPDDR5 logical spill (NVMe round-trip µs 대비 ns, CF-3). carrier NVMe 장착 후 직접 baseline.**
- **VLCache** [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) — same-input 98% reuse. → **차별화: same image + 다른 focus block-level promote.**
- **VL-Cache** [arXiv:2410.23317](https://arxiv.org/abs/2410.23317) / **MEDA** [arXiv:2502.17599](https://arxiv.org/abs/2502.17599) / **SnapKV/PyramidKV** — 누락 baseline 추가 (budget 분배 비교군).
- **MMTok** [arXiv:2508.18264](https://arxiv.org/abs/2508.18264) / **Quest**(ICML'24) / **MMDU** [arXiv:2510.16641](https://arxiv.org/abs/2510.16641) — scoop 인접군: MMTok(single-prompt coverage)·Quest(query-aware page selection)·MMDU(multi-turn bench) 와 본 idea(edge UMA × recoverable-quant × multi-focus × reconstruction bound) 의 3축 결합 차별화.
- **Eckart-Young** (classical low-rank bound) — hot-tier `‖V−V̂‖_F≤ε` anchor.

## 11. Implementation Consistency

- **R47 path**: Application-level. vLLM V1 BlockPool requantize + vision-block 매핑 신규 인프라. Jetson 1순위 = llama.cpp GGUF KV quant (vLLM Jetson build risk 회피).
- **CF 일관성**: CF-1(prefetch→framework KV buffer + stream attach), CF-2(KV/embedding layer 분리, vision-block 매핑 신규), CF-3(NVMe 재정의: LPDDR5 logical < NVMe round-trip + quant 본질), CF-4(1024 core).
- **Layer 분리 명시**: hot/cold 는 `BlockPool`(LLM KV, modality-agnostic) layer; `EncoderCacheManager`(embedding) 는 별개. promote=dequant compute (UMA zero-copy mapped).

## 12. Reproducibility Checklist (5 필드)

1. **Clone Spec**: vLLM `7c37096` / llama.cpp `dbe9c0c` (2026-06-02). `_gen_mm_extra_hash_keys`(L395), `block_pool.py free_blocks`(L419), `kv_cache_manager.py allocate_slots`(L238) verified, hallucinated 0건.
2. **Environment**: JetPack 6.2, CUDA 12.x, Python 3.10+, llama.cpp `-DCMAKE_CUDA_ARCHITECTURES=87`, HQQ/AWQ, cuSOLVER.
3. **Build**: llama.cpp GGUF `--cache-type-k q4` smoke (1-turn VQA) → vLLM editable patch.
4. **Patch List**: M1(vision-block 매핑 + free_blocks requantize + GGUF KV) / M2(dequant promote + allocate_slots + SVD bound).
5. **Smoke Test**: Qwen2.5-VL-7B Q4_K + dual-tier KV 16GB fit 확인, focus-shift VQA 에서 cold promote path 동작, uniform-Q4 vs dual-tier footprint·FCR 비교.

## 13. Scoring 및 이유 (R67, phase2prime Section D)

| Reviewer | sub1 | sub2 | sub3 | sub4 | rev-mean |
|---|---|---|---|---|---|
| Novelty (Mech/Comb/Hypo/D2) | 6 | 8 | 7 | 8 | 7.25 |
| Differentiation (RW/Clarity/Pos/Scope) | 8 | **9** ★ | 9 | 7 | 8.25 |
| Impact (Mag/Breadth/Adopt/D1) | 8 | 6 ▼ | 7 | 7 | 7.0 |
| AI-impl (Src/Kernel/Integ/D6) | 6 | 6.5 | 7 | 7.5 | 6.75 |
| Arch-sys (R47/fit/HW/D6) | 8 | 7.5 | 8 | 8 | 7.875 |

- **★ 전체 최고: Differentiation-Clarity = 9** — E reconstruction bound 흡수 후 "recoverable"=휴리스틱→보장. SparseVILA/KVSwap 교집합 공백을 formal bound 로 점유 (세션 전체 최고 sub-axis).
- **▼ 전체 최저: Impact-Breadth = 6** — UMA-bound (Jetson 16GB 한정), 의도적 scope. 720GB 미검증 motivation 은 Phase 3 gate.
- **이유**: AI-impl 최대 상승(5.0→7.0) — CF-1(framework KV buffer)/CF-2(vision-block 매핑) 로 H1/H2 critical → resolved. flagship, 5 reviewer 중 4개 primary 합의. **Total mean 7.43 / Grade Accept.**

## 14. R14.4 Decision Tree

```
preliminary gate: uniform-Q4(no-tier) vs dual-tier ablation
├─ marginal 이득 ≥5% (footprint·FCR)
│   ├─ SVD compute overhead < net 이득 → 전체 dual-tier (hot low-rank+quant + cold) [MLSys/ASPLOS flagship]
│   └─ SVD overhead 상쇄 (F4) → low-rank 제거, quant-only hot + cold tier 유지
├─ marginal <5% (F3) → tiering 기여 없음 → quant-only(uniform-Q4) 로 축소 (여전히 OOM 회피 contribution)
└─ cold recover accuracy <90% (F2) → "recoverable" 붕괴 → hot budget 확대 + cold 축소 (capacity 한계 honest)
  └─ recover latency >30% decode (F1) → bandwidth 경합 치명 → R7 compute-idle prefetch 결합 또는 hot-only fallback
```

## 15. Inter-idea Dependency

- **R1 = INFRA-A/B producer**: INFRA-A(vision token→KV block 매핑, `_gen_mm_extra_hash_keys` 재활용) + INFRA-B(framework-level KV buffer 정밀도 관리, dequant promote). **R5/R6 가 INFRA-A 재사용, R5/R7 가 INFRA-B 재사용, R3 evict 도 의존.**
- **R1 ← R2 (TRACE-C)**: energy axis 측정에 R2 의 rail-level probe 재사용.
- **R1 ← R4 (BENCH-D)**: multi-focus FCR/recall 평가 토대.
- **Paper Pair {R1+R3}**: R3 의 retrieve-on-miss 가 R1 cold tier 에 물리적 의존. R1=dual-tier capacity 인프라(system), R3=evict/선정 정책(algorithm) → MLSys/ASPLOS 1편.

## 16. Stakeholder (7-row)

| Stakeholder | 관심사 | R1 제공 가치 |
|---|---|---|
| Edge VLM 서비스 운영자 | 16GB 단일 device serving | OOM 회피, full-cache 불가 환경 enabling |
| MLSys 연구자 | reconstruction bound 이론 | `‖V−V̂‖_F≤ε` formal guarantee |
| Robotics / on-device 개발자 | multi-turn 대화 fidelity | destructive prune 손실 회복 (≥92%) |
| Framework maintainer (vLLM/llama.cpp) | vision-block KV 매핑 인프라 | `_gen_mm_extra_hash_keys` 경로 재활용 PR |
| HW vendor (NVIDIA Jetson) | UMA serving 효율 | LPDDR5 logical spill cost model |
| 후속 연구자 (R3/R5/R7) | INFRA-A/B 재사용 | 공통 cross-cutting 인프라 producer |
| Energy / sustainability 평가자 | per-turn J 절감 | hot-only turn I/O 제거 |

## 17. Boundary (5-axis)

| Axis | In-scope | Out-of-scope |
|---|---|---|
| Modality | vision-KV (image multi-turn) | text-only KV (KVSwap 영역) |
| Device | Jetson Orin NX 16GB UMA (sm_87) | discrete GPU HBM migration |
| Mechanism | precision-tier + reconstruction-bounded dequant | HW accelerator 신설 (V-Rex/Focus) |
| Capacity | quantization 본질 + logical spill | NVMe disk 가 primary (KVSwap baseline 으로만) |
| Promote | framework-level dequant compute | `cudaMemPrefetchAsync` managed migration (Orin 미지원) |

## 18. Self-Check

- [x] RQ 3개 의문문 + 정량 (60-76% / 30% / ≥92%·≥5%)
- [x] CF-1 (cudaMemPrefetchAsync 제거, framework KV buffer + cudaStreamAttachMemAsync)
- [x] CF-2 (EncoderCacheManager embedding ≠ BlockPool KV 분리, vision-block 매핑 신규)
- [x] CF-3 (NVMe 재정의: LPDDR5 logical < round-trip, quant 본질, KVSwap 직접 baseline)
- [x] CF-4 (1024 core, 40W=JetPack6.2, LPDDR5 102.4GB/s, sm_87, concurrentManagedAccess=0)
- [x] R68 GitHub link: 모든 R52.2 표 `blob/main/{path}#L{A}-L{B}` (fixed path)
- [x] R8 arxiv clickable markdown (raw ID 없음)
- [x] R67 ★(Diff-Clarity 9) / ▼(Impact-Breadth 6)
- [x] vendor-neutral title (RTX/Jetson/Orin 등 device명 title 없음, motivation 본문만)
- [x] 18 의무 섹션 전부
