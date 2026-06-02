# VKV-SKIP: Non-Destructive Vision-Aware Cache-Resident Attention Block Skipping for Memory-Bound Edge VLM Decode

**Tier-2 (T1 조건부) · Grade: Conditional Accept borderline (mean 6.20 / Total mean 6.23) · 전체 최저 sub-axis (AI-Kernel ▼4.5)**

> **Metaphor noun:** *VKV-SKIP* — "VKV" 가 vision-KV, "skip" 이 non-destructive softmax-threshold block skip (selection 아닌 skip).
> **Merge 출처:** VISION-BLASST(ai-opt) + F6(legacy). F6 의 modality-aware λ + Ampere kernel + VISION-BLASST 의 llama.cpp mtmd 경로. (VISION-BLASST = F6 부분집합 → 통합.) **Quest/MInference scoop 차별화 필수.**

📖 약어 / 핵심 용어:
- **BLASST**: training-free dynamic blocked attention sparsity via softmax thresholding (MLSys'26). FlashAttention online-softmax running max 재사용, block-local max 가 running max 보다 ln(λ) 이상 작으면(post-softmax≈0) exp/V-load/PV-matmul skip. decode 1.48×@73.2% (B200).
- **Non-destructive**: cache 보존, query 별 V-load 만 동적 skip — cache 파괴 안 함 → 다음 focus 동일 cache 재skip (multi-focus 안전). Quest(selection) 와 달리 softmax-threshold(skip).
- **Vision-aware λ**: vision block 에 더 aggressive threshold (λ_vision < λ_text). 근거 = vision vs text post-softmax redundancy 분포 차이 (KL/Wasserstein 으로 분리).
- **SM87 kernel**: Ampere block-skip variant 가 llama.cpp `fattn-tile.cu` 에 **부재** → from-scratch SM87 kernel (전체 최저 sub-axis 의 원인).

---

## 1. Research Questions

- **Master RQ**: vision KV block 의 spatial/semantic redundancy 가 text 대비 높다면, BLASST 의 softmax-threshold block-skip 을 vision-aware λ 로 적용해 cache 보존(multi-focus 무손실)한 채 decode V-load skip 율을 높이면서, Quest/MInference 대비 vision-modality 차별화된 non-destructive 이득을 얻을 수 있는가?
- **RQ-1**: vision KV block 의 post-softmax redundancy 분포가 text 대비 통계적으로 높은가 (KL/Wasserstein 으로 분리 가능한가)?
- **RQ-2**: vision-block-aware λ 가 BLASST-uniform 대비 decode V-load skip 율을 **+15%p** 높이고 Jetson decode TPOT 를 **15-25%** 절감하면서 accuracy degradation **<2%p** 를 유지하는가?
- **RQ-3**: Ampere(SM87, 1024 core) from-scratch kernel 의 skip-check overhead 가 절감을 상쇄하지 않고 (0% sparsity 1.0× 초과 안 함), Quest/MInference 대비 vision-aware non-destructive 의 marginal gain 이 입증되는가?

## 2. Two-Sentence Pitch

모든 vision token pruning 은 destructive(cache 파괴)라 multi-focus 에서 손실되나, BLASST 의 online-softmax threshold block-skip 은 cache 비파괴로 query 별 V-load 만 동적 skip 한다 — 단 vision token redundancy 와 multi-focus 를 미활용. 본 idea 는 BLASST 를 vision-aware(vision block 에 더 aggressive λ) + Jetson Ampere kernel + multi-focus 무손실 평가로 확장하여, Quest/MInference 의 query-aware page selection 과 달리 vision-modality redundancy 분포 차이를 명시 활용한 non-destructive skip 을 제공한다.

## 3. 가설 + Falsification

**가설:** vision KV block 은 text 대비 spatial/semantic redundancy 가 통계적으로 높아(KL/Wasserstein 으로 분리), BLASST softmax-threshold 를 vision-block-aware λ 로 적용하면 cache 보존(multi-focus 무손실)한 채 decode V-load skip 율을 BLASST-uniform 대비 +15%p 높이고 Jetson decode TPOT 를 15-25% 절감하면서 accuracy degradation <2%p 를 유지한다.

**Falsification (5):**
1. **vision block 의 post-softmax redundancy 분포가 text 와 통계적으로 차이 없음** (KL/Wasserstein) → vision-aware λ 무의미 → "BLASST 재현 + vision tag" 격하 (novelty F04: extension cliff, 이 분리를 1급 empirical claim 으로).
2. vision-aware skip 이 multi-focus turn 의 새 focus block 잘못 skip → accuracy **>2%p** 하락 → non-destructive 가정 충돌.
3. **Ampere(SM87, 1024 core, CF-4) kernel skip-check overhead 가 절감 상쇄** (0% sparsity 1.0× 초과) → BLASST Blackwell 측정치 transfer 불가, kernel 비효율 부분 기각.
4. **Quest/MInference 대비 vision-aware non-destructive 의 marginal gain 미입증** → engineering 격하, scoop (diff F08).
5. vision KV > 16GB → cache 보존 전제 위반 → R1 dual-tier 선행 필수.

## 4. Workload Evidence (Step 0-α 정량 인용)

- **Non-destructive 선례 (verified, BLASST corpus B):** decode kernel 이 "memory-bound 에서 V block 의 HBM load 를 skip" — Jetson Orin dominant bottleneck (memory bandwidth) 과 정확히 일치. decode 1.48×@73.2% sparsity (B200). KV 보존, query 별 동적 skip = cache 비파괴.
- **Edge bottleneck (verified, PAISE'25):** Jetson Orin decode memory-bandwidth-bound (memory freq↓ → latency +370%). → V-load skip = bandwidth(=energy) 절감.
- **Vision redundancy 근거 (verified):** "When Token Pruning is Worse than Random" [arXiv:2512.07580](https://arxiv.org/abs/2512.07580) — layer-20 이후 prune = random, OCR horizon 깊음 → vision-aware threshold 가 layer/task 의존. corpus A Focus — vision token 98-99% + spatial redundancy.
- **CSP:** vision/text modality 분포 차이로 over-pruning → vision-aware λ 의 근거.

## 5. 기준 코드베이스 (R52.1)

- **llama.cpp** `main` commit `dbe9c0c` (1순위 — decode kernel path). `ggml/src/ggml-cuda/fattn-tile.cu`/`fattn-vec.cuh`/`fattn-mma-f16.cuh` 에 **block-skip variant 부재** → from-scratch SM87 kernel (ai-impl F07/H8). `tools/mtmd/mtmd.cpp` `mtmd_image_tokens`(L44) → llama_batch → kv_cache plumbing **신규** (CF-2, F08). `tools/mtmd/clip.cpp` `clip_ctx`(L142). MoE decode hang [#19219](https://github.com/ggml-org/llama.cpp/issues/19219) 회피 (dense backbone).
- **vLLM** `7c37096` / **SGLang** `e958f45` (보조).
- **HF model:** `Qwen/Qwen2.5-VL-7B-Instruct` (**dense backbone, non-MoE — SM87 MoE decode hang 회피**).
- **deps:** FlashInfer/Triton (Ampere backend), KL/Wasserstein (redundancy 분포).
- **Jetson Orin NX 16GB 정정 spec (CF-4):** CUDA core **1024**, **32 Tensor core** (1792 아님), LPDDR5 **102.4 GB/s**, sm_87, `concurrentManagedAccess=0`. llama.cpp Ampere FA 존재 (`-DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_FA_ALL_QUANTS=ON`) 하나 **block-skip variant 부재**. Ampere shared-mem 164KB opt-in (`cudaFuncAttributeMaxDynamicSharedMemorySize`), occupancy 재산정. **dev GPU = sm_86 (RTX 3090) 또는 Orin 직접 — RTX 4090=Ada sm_89 부적합** (F30).

## 6. 동작 원리 (R53 inline)

### M1 — Vision vs Text Redundancy 분포 분리 (1급 empirical claim)

**① 동작원리 4요소**
- **(무엇을)** vision KV block 과 text block 의 post-softmax redundancy 분포 차이를 KL/Wasserstein 으로 측정 — vision-aware λ 의 근거.
- **(왜 작동)** vision 98-99% 비중 + spatial redundancy → 대부분 block post-softmax≈0. 분포 분리되면 λ_vision < λ_text 정당.
- **(구현변경점)** GPU pilot (accuracy/skip-rate 만, kernel 과 분리). vision token 식별 = `mtmd_image_tokens` → kv_cache plumbing 신규 (CF-2).
- **(검증 시나리오)** vision vs text block redundancy 분포 차이 pilot (분리 작으면 Tier-2 강등).

**② 기대효과**: 분포 분리 입증 → vision-aware λ 1급 근거.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Vision range plumbing | mtmd→kv_cache token range (CF-2) | llama.cpp | [`tools/mtmd/mtmd.cpp#L44-L80`](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd.cpp#L44-L80) `mtmd_image_tokens` (main 부재 — 기본 branch master, R68.2) | new plumbing | 상 | clone ✓ |
| Vision embed offset | image token 위치 | llama.cpp mtmd | [`tools/mtmd/clip.cpp#L142-L180`](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip.cpp#L142-L180) `clip_ctx` (main 부재 — master, R68.2) | read | 중 | clone ✓ |

**④ 검증 trace (R52.3)**: `mtmd_image_tokens`(L44)/`clip_ctx`(L142) clone ✓. vision range plumbing 신규 (CF-2).

### M2 — Non-Destructive Vision-Aware Softmax-Threshold Block Skip (from-scratch SM87 kernel)

**① 동작원리 4요소**
- **(무엇을)** decode attention kernel 에서 vision KV block 에 vision-aware softmax threshold (λ_vision < λ_text) 로 V-load/PV-matmul skip. cache 보존, query 변경 시 skip pattern 재평가.
- **(왜 작동)** memory-bound decode 에서 V-load skip = bandwidth(=energy) 절감. skip 매 step 재평가 → cache 비파괴 → 다음 focus 동일 cache 재skip (multi-focus 안전). **Quest 와 달리 vision-modality 분포 차이 명시.**
- **(구현변경점, CF)** llama.cpp `fattn-tile.cu` 에 running-max threshold/block-skip 로직 **신규 작성** (부재 확인, "port BLASST"=from-scratch SM87). Ampere shared-mem 164KB opt-in, 1024 core occupancy 재산정 (CF-4).
- **(검증 시나리오)** accuracy-first (PyTorch/Triton ref 로 비파괴성 먼저) + Quest/MInference marginal gain + 0% sparsity 1.0× 확인.

**② 기대효과**: Jetson decode TPOT −15~25%, decode V-load skip → LPDDR5 read −20~30% (energy 비례).
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Softmax-threshold skip | V-load skip kernel (신규) | ggml-cuda | [`ggml/src/ggml-cuda`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda) `fattn-tile.cu` (running-max **부재**, main 부재 — master, R68.2) | **from-scratch SM87** | 상 | clone ✓ |
| GPU ref / accuracy-first | sm_86 kernel dev (CF-4) | FlashInfer/Triton | RTX 3090 (sm_86) 또는 Orin 직접 (4090=Ada 부적합) | port | 상 | spec ✓ |

**④ 검증 trace (R52.3)**: `fattn-tile.cu` block-skip **부재 (신규 kernel, F07)**. **CF-4: dev GPU=sm_86 (4090=Ada sm_89 아님, F30). dense backbone (MoE decode hang 회피).**

## 7. End-to-End Evaluation

- **Accuracy-first:** PyTorch/Triton reference 로 비파괴성 + multi-focus 무손실 먼저 검증 (speedup 분리).
- **Multi-turn / multi-focus:** R4 FOCUS-COVERAGE (cache 보존 시 multi-focus 무손실 검증) + MileBench / MMDU.
- **Long-vision:** Video-MME / MLVU (long-vision context decode V-load skip).
- **Baseline:** BLASST-uniform (vision 미구분), VisionZip(destructive merge), Quest(query-aware page selection, modality-agnostic), MInference(prefill sparse), Focus(HW destructive), "When Token Pruning is Worse than Random".

## 8. 실험 7요소 (12-16주)

1. **Hardware**: Jetson Orin NX 16GB (1024 core, 32 Tensor, sm_87). dev = RTX 3090 (sm_86) 또는 Orin 직접 (RTX 4090=Ada 부적합).
2. **Model**: Qwen2.5-VL-7B (**dense backbone, MoE decode hang 회피**).
3. **Framework**: llama.cpp ggml-cuda decode kernel (가장 무거운 path) + mtmd vision plumbing. FlashInfer Ampere backend.
4. **Energy 측정**: TRACE-C(R2) 재사용 — decode V-load skip LPDDR5 read 절감 (J).
5. **Gate**: vision vs text redundancy 분포 분리 pilot (GPU, kernel 분리) + Quest/MInference marginal gain. 분리 작거나 marginal 없으면 Tier-2 강등 또는 DROP.
6. **Steps**: (a) vision vs text redundancy 분포 pilot (accuracy/skip-rate, kernel 분리) → (b) 분리 입증 시 SM87 block-skip kernel (accuracy-first PyTorch ref 먼저) → (c) Ampere occupancy 재산정 → (d) Jetson decode TPOT.
7. **Metrics**: KL/Wasserstein 분포 거리, V-load skip 율 (vs BLASST-uniform), decode TPOT, accuracy degradation, 0% sparsity overhead.

## 9. 예상효과 5-axis 표 (Energy 강조)

| Axis | 예상 개선 | 조건/scope |
|---|---|---|
| **Energy/Power** | decode V-load skip → LPDDR5 read −20~30% = energy 비례 | memory-bound decode |
| Performance | Jetson decode TPOT −15~25% (Ampere baseline 선측정, B200 transfer 금지) | long-vision context |
| Memory eff. | cache 보존 (절감 아님) — multi-focus 안전이 본질 | — |
| Cost eff. | training-free, drop-in | Ampere kernel porting 비용 |
| Latency | decode latency −20~30% | memory-bound |

## 10. 관련연구 + 차별화

- **BLASST** [arXiv:2512.12087](https://arxiv.org/abs/2512.12087) (MLSys'26) — training-free online-softmax V-load skip, decode 1.48×@73.2%, text-LLM, vision 미구분, Blackwell/Hopper kernel. → **차별화: vision-aware λ + Ampere kernel + multi-focus.**
- **Quest** (ICML'24, query-aware page selection without eviction) — **CF: "query-aware cache-resident without eviction" = F6 핵심과 매우 인접 (diff F08 scoop high, R6 borderline 의 핵심 이유).** → **차별화: Quest 는 page-level top-k selection (modality-agnostic), 본 idea 는 vision-modality redundancy 분포 차이 명시 + multi-focus 무손실 + non-destructive softmax-threshold (selection 아닌 skip).**
- **MInference** (NeurIPS'24, dynamic sparse attention) — 누락 baseline 추가. → **차별화: prefill sparse pattern vs decode non-destructive vision skip.**
- **VisionZip** [arXiv:2412.04467](https://arxiv.org/abs/2412.04467) (CVPR'25) — dominant token + merge, "multi-turn underperform". → **차별화: destructive merge → non-destructive skip.**
- **When Token Pruning is Worse than Random** [arXiv:2512.07580](https://arxiv.org/abs/2512.07580) (CVPR'26) — layer-20 이후 prune = random. → vision-aware threshold layer/task 의존 근거.
- **Focus** [arXiv:2512.14661](https://arxiv.org/abs/2512.14661) (HPCA'26) — dedicated HW semantic prune (destructive). → **차별화: commodity Jetson, 비파괴.**

## 11. Implementation Consistency

- **R47 path**: Application-level. llama.cpp ggml-cuda decode kernel (가장 무거운 path) + mtmd vision plumbing. accuracy-first: PyTorch/Triton reference 로 비파괴성 먼저, speedup 분리.
- **CF 일관성**: "port BLASST"=from-scratch SM87 kernel (F07). vision range plumbing 신규 (CF-2). 1024 core/sm_86 dev GPU (CF-4, 4090=Ada). MoE decode hang 회피 (dense backbone). cache 보존 전제 위반 시 (vision KV>16GB) R1 dual-tier 선행.
- **R1 의존**: vision KV>16GB 시 R1 dual-tier 결합 (cache 보존 전제 유지).

## 12. Reproducibility Checklist (5 필드)

1. **Clone Spec**: llama.cpp `dbe9c0c`. `fattn-tile.cu` block-skip 부재 (신규), `mtmd_image_tokens`(L44), `clip_ctx`(L142) verified, hallucinated 0건.
2. **Environment**: JetPack 6.2, CUDA 12.x, llama.cpp `-DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_FA_ALL_QUANTS=ON`, FlashInfer/Triton, dev = RTX 3090 sm_86 또는 Orin.
3. **Build**: vision range plumbing + SM87 block-skip kernel (Ampere 164KB shared-mem opt-in). dense backbone (Qwen2.5-VL-7B non-MoE).
4. **Patch List**: M1(mtmd plumbing + clip_ctx read + 분포 pilot) / M2(fattn-tile.cu from-scratch SM87 skip + Triton ref).
5. **Smoke Test**: vision vs text 분포 분리 (KL/Wasserstein), 비파괴성 (multi-focus 동일 cache 재skip), 0% sparsity 1.0× 초과 안 함, BLASST-uniform 대비 skip 율.

## 13. Scoring 및 이유 (R67, phase2prime Section D)

| Reviewer | sub1 | sub2 | sub3 | sub4 | rev-mean |
|---|---|---|---|---|---|
| Novelty (Mech/Comb/Hypo/D2) | 6 | 6 | 6 | 7 | 6.25 |
| Differentiation (RW/Clarity/Pos/Scope) | 8 | **8** ★ | 7 | 6 | 7.25 |
| Impact (Mag/Breadth/Adopt/D1) | 6 | 6 | 6 | 7 | 6.25 |
| AI-impl (Src/Kernel/Integ/D6) | 5.5 | **4.5** ▼ | 5 | 7 | 5.5 |
| Arch-sys (R47/fit/HW/D6) | 5.5 | 5.5 | 5.5 | 7 | 5.875 |

- **★ 전체 최고: Diff-Clarity = 8** — cache 비파괴=multi-focus 안전의 정공법, 명확한 차별화.
- **▼ 전체 최저: AI-Kernel = 4.5 (세션 전체 최저 sub-axis)** — BLASST online-softmax block-skip 의 from-scratch SM87 Ampere kernel. fix 가 명확화했으나 난이도 본질 불변, R6 borderline 의 단일 최대 risk.
- **이유**: non-destructive = multi-focus 안전 정공법. Quest 차별화 + Ampere accuracy-first 입증 시 T1 조건부 승격. 미입증/kernel 실패 시 engineering 격하 또는 DROP. **Total mean 6.23 / Grade Conditional Accept (borderline).**

## 14. R14.4 Decision Tree

```
preliminary gate: vision vs text redundancy 분포 분리 pilot + Quest/MInference marginal gain
├─ 분포 분리 입증 (KL/Wasserstein) + Quest 대비 marginal gain 있음
│   ├─ SM87 kernel skip-check overhead < 절감 (0% sparsity 1.0× 이내) → vision-aware kernel [DAC/CAL T1 조건부]
│   └─ kernel overhead 상쇄 (F3) → accuracy-only contribution (분포 분리 paper, FlashInfer Ampere 별도)
├─ 분포 차이 없음 (F1) → "BLASST 재현 + vision tag" 격하 (extension cliff)
├─ Quest marginal 미입증 (F4) → engineering 격하 또는 DROP (scoop)
├─ multi-focus 새 focus block 잘못 skip (F2) → λ_vision 보수화, accuracy 우선
└─ vision KV > 16GB (F5) → R1 dual-tier 선행 결합 필수
```

## 15. Inter-idea Dependency

- **R6 ← R1 (INFRA-A)**: vision token range plumbing (mtmd→kv_cache) = R1 INFRA-A 와 동일 인프라 (재사용). vision KV>16GB 시 R1 dual-tier 선행.
- **R6 ← R4 (BENCH-D)**: multi-focus 무손실 (cache 보존) recall 평가 토대.
- **R6 ← R2 (TRACE-C)**: decode V-load skip LPDDR5 read 절감 energy 측정.
- **독립 paper unit (조건부)**: R6 = Ampere kernel DAC/CAL (Quest 차별화·accuracy-first 입증 시).

## 16. Stakeholder (7-row)

| Stakeholder | 관심사 | R6 제공 가치 |
|---|---|---|
| Edge VLM decode 최적화 | memory-bound decode 가속 | V-load skip TPOT −15~25% |
| Kernel 엔지니어 | Ampere SM87 sparse attention | from-scratch block-skip kernel |
| Multi-focus 안전 연구자 | cache 비파괴 | 다음 focus 동일 cache 재skip |
| R1 INFRA 소비자 | vision range plumbing | INFRA-A 재사용 |
| 측정 커뮤니티 | edge decode energy | LPDDR5 read −20~30% |
| 분포 분석 연구자 | vision vs text redundancy | KL/Wasserstein 1급 claim |
| 후속 (Quest 차별화) | non-destructive vs selection | softmax-threshold skip |

## 17. Boundary (5-axis)

| Axis | In-scope | Out-of-scope |
|---|---|---|
| Modality | vision-aware λ (분포 차이) | modality-agnostic (Quest/BLASST-uniform) |
| Mechanism | non-destructive softmax-threshold skip | destructive prune/merge (VisionZip), selection (Quest) |
| Device | Jetson Ampere SM87 kernel | Blackwell/Hopper (BLASST), dedicated HW (Focus) |
| Memory | cache-resident (≤16GB) | vision KV>16GB (R1 dual-tier 선행) |
| Phase | decode V-load skip | prefill sparse (MInference) |

## 18. Self-Check

- [x] RQ 3개 의문문 + 정량 (분포 분리 / +15%p·15-25%·<2%p / kernel overhead·Quest marginal)
- [x] CF-1 (cache-resident skip, prefetch API 무관)
- [x] CF-2 (mtmd→kv_cache vision range plumbing 신규, encoder/KV 분리)
- [x] CF-3 (cache 보존 ≤16GB, 초과 시 R1 LPDDR5 logical/NVMe)
- [x] CF-4 (1024 core/32 Tensor, sm_87, dev=sm_86 RTX 3090, 4090=Ada 부적합, MoE hang 회피)
- [x] R68 GitHub link `blob/main/{path}#L{A}-L{B}` + `tree/main/{dir}` (fattn-tile.cu 신규 디렉토리)
- [x] R8 arxiv clickable markdown (BLASST/VisionZip/Worse-than-Random/Focus)
- [x] R67 ★(Diff-Clarity 8) / ▼(AI-Kernel 4.5, 전체 최저)
- [x] vendor-neutral title (device명 title 없음 — "Edge VLM" 일반명)
- [x] 18 의무 섹션 전부
