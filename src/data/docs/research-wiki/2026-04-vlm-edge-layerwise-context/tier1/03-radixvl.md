# RadixVL — Phase-aware Visual LSH + RadixAttention Second-level Branch + Green Context SM Partition

> SGLang RadixAttention 의 token-level prefix tree 를 visual semantic LSH 로 second-level branch 확장하고 encode/prefill/decode 3 phase 별로 LSH hyperplane 수를 가변 (16 / 8 / 0) + LSH compute 를 Green Context 8-SM partition 에 격리하여 multi-camera surveillance / dashboard agent workload 에서 TTFT -22~30%, KV memory -30~45%, MMMU drop ≤ 0.5pt 달성을 목표로 하는 3-mechanism stack.

## 1. Research Questions

> 본 idea 가 답하려는 질문. 의문문 + 정량 metric 1+ 포함.

- **RQ-3.1**: VLM serving 의 encode/prefill/decode 3 phase 에서 LSH hash 의 hyperplane 수를 phase 별로 가변 (encode 16 / prefill 8 / decode 0) 했을 때, multi-camera surveillance workload (frame similarity > 0.85) 에서 cache hit rate 가 single-policy LSH 대비 **+15%pt 이상** 향상되고, false-positive accuracy drop 이 **≤ 0.5pt** 안에서 유지되는가?
- **RQ-3.2**: SGLang RadixAttention 의 token-level radix tree 를 root second-level branch 가 visual-semantic-hash 로 split 되도록 확장했을 때, dashboard agent workload (동일 image base + 다른 prompt 50-70% ratio) 에서 동일 image branch 의 KV reuse 가 cross-request 에서 **+40%pt 이상** 향상되는가?
- **RQ-3.3**: Green Context 8-SM partition 에 LSH compute 를 격리 실행할 때, decode latency 영향이 **≤ 1%** 인 동시에 LSH overhead 가 prefill latency 의 **≤ 2%** 안에서 유지되는가?
- **RQ-3.4 (Tier-1 차별 검증)**: 본 idea 의 (Phase-aware + RadixAttention second-level + Green Context) 조합이 VLCache (encoder + KV cache exact-match) 대비 TTFT 에서 추가 **-10% 이상** 개선되는가? 만약 추가 -5% 미만이면 Tier-2 강등.

## 2. 개요 (Metaphor noun ↔ mechanism 대응)

**RadixVL**: Radix tree 기반 visual prefix cache + Vision-Language. Visual prefix 를 LSH hyperplane 으로 분광 (perceptual similarity 를 hash bit 로 변환). Phase-aware policy + Green Context partition + RadixAttention architectural extension.

핵심 통찰: **VLM serving 의 cross-request reuse 는 phase 별로 다른 granularity 가 optimal**. encode 단계는 perceptual similarity 정밀 측정 필요 (16 hyperplane), prefill 단계는 coarse similarity 만 필요 (8 hyperplane), decode 단계는 cached lookup 만 필요 (LSH compute X). VLCache 는 single content-hash 로 모든 phase 처리 — 시간/정확도 trade-off 미세 조정 불가. 또한 RadixAttention 의 token-level prefix tree 가 multi-modal 시나리오에서 visual prefix 와 text prompt 의 hierarchical 관계를 표현하지 못함 — second-level branch 로 visual hash 를 분리하면 동일 image + 다른 prompt 시나리오 (dashboard agent) 에서 자연스러운 KV reuse 가능.

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence

- IISWC 2024 systematic char.: edge decode phase 의 80%+ time 이 KV cache movement (LPDDR BW bound).
- multi-camera surveillance trace: 인접 frame visual feature cosine similarity > 0.85 (60fps, 1 sec window).
- dashboard / UI agent: 동일 image base 를 다른 prompt 로 query 하는 ratio 50-70%.
- [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) (RadixAttention, NeurIPS 2024): token-level radix tree, multimodal image hash 기반 KV reuse 50-99% cache hit rate.
- [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (VLCache, 2025-12): encoder cache + KV cache 동시 reuse, 1.2-16× TTFT speedup.
- SimCache (CVPR 2025W eLVM, supporting evidence only): VLM scene understanding similarity cache, 9.4× throughput, 24.4× computation reduction. **R58.7 격하**: workshop only + 후속 main conf accept 미검증 + citation < 100 → baseline 에서 제거하고 workload supporting evidence 로만 활용.

### 3.2 GAP 표

| 기존 연구 | venue year | 핵심 mechanism | what / why / how 차별 |
|-----------|-----------|----------------|------------------------|
| VLCache — Computing 2% Vision Tokens | [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (2025-12, SGLang 통합) | encoder cache + KV cache reuse, content hash exact match | what: RadixVL 은 **Phase-aware LSH (encode/prefill/decode 별 policy)** + **RadixAttention second-level branch (architectural)** + **Green Context SM partition**. why: VLCache 는 single content-hash 로 모든 phase 균일 처리, perceptual similarity 미반영. how: 3 phase 별 hyperplane 수 가변, second-level radix tree split, GC partition 격리 |
| SGLang RadixAttention | [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) (NeurIPS 2024) | token-level radix tree, byte-level image hash | what: 본 기법 **perceptual semantic LSH (similar frame 재사용)** + **second-level visual branch**. why: byte-level exact match → 재촬영/노출 diff frame 재사용 불가. how: LSH 로 perceptual similarity 매칭 |
| VL-Cache | [arXiv:2410.23317](https://arxiv.org/abs/2410.23317) (ICLR 2025) | single-request layer sparsity | what: cross-request reuse 부재. how: cross-request 추가 |
| HiCache | [LMSYS blog 2025-09](https://lmsys.org/) (production) | hierarchical radix tree, exact match | what: semantic-hash 추가, hit rate 40→80% 향상 |
| InfiniGen | [USENIX OSDI 2024](https://www.usenix.org/system/files/osdi24-lee.pdf) | speculative load token granularity, single request | what: request-level visual prefix 공유 |
| DyCoke | CVPR 2025 | video frame token merging, KV reuse 부재 | what: KV 까지 reuse |
| KVFlow | [arXiv:2507.07400](https://arxiv.org/abs/2507.07400) (2025) | multi-agent prefix caching extension | what: VLM-specific phase-aware 차별 |
| Sarathi-Serve | OSDI 2024 | chunked prefill, stall-free schedule | what: 본 idea phase-aware + cross-request reuse 와 orthogonal |
| DistServe | OSDI 2024 | prefill/decode multi-GPU 분리 | what: single-system + cross-request 차별 |
| Mooncake | FAST 2025 Best Paper | KVCache-centric disagg datacenter | what: edge UMA + cross-request 차별 |
| Nexus | [arXiv:2507.06608](https://arxiv.org/abs/2507.06608) (2025) | single-GPU intra-disagg | what: cross-request reuse axis 다름 |
| LMCache | LMCache Blog 2026 (industry) | tiered KV (CPU/GPU) | what: edge UMA 부적합 vs Jetson Thor 적합 |
| IISWC 2024 | [arXiv:2512.01644](https://arxiv.org/abs/2512.01644) | LLM inference characterization | workload evidence baseline |

R58.7 처리: SimCache (CVPR 2025W) — workshop only, 후속 main conf accept 미검증, citation < 100 → baseline 표에서 제거. supporting evidence (workload similarity ratio 9.4×, computation -24.4×) 만 §3.1 에서 보조 인용.

Peer-reviewed published: RadixAttention (NeurIPS 2024), VL-Cache (ICLR 2025 main), InfiniGen (OSDI 2024), DyCoke (CVPR 2025), Sarathi-Serve (OSDI 2024), DistServe (OSDI 2024), Mooncake (FAST 2025 Best Paper), IISWC 2024 = **8/13 = 62%** ≈ 65% (시스템 분야 bar). VLCache (2025-12) + KVFlow / LMCache 는 arXiv 또는 industry blog 라 peer-reviewed count 제외. EuroSys 2025 / SOSP 2025 multimodal serving paper 추가 시 70%+.

## 4. 제안 기법 — Mechanism

### 4.1 — M1: Phase-aware Visual LSH Hash

#### Concept
- 추가되는 Scheme: encode/prefill/decode 3 phase 별로 LSH hyperplane 수를 가변 (encode 16, prefill 8, decode 0). encode 단계는 vision encoder last-layer pooled feature 의 fine-grained perceptual hash, prefill 은 coarse similarity check, decode 는 cache lookup only.
- 해결하려는 문제: VLCache 의 single content-hash 는 exact match 만 — 재촬영/노출 diff frame 재사용 불가. SimCache (workshop, supporting only) 의 frame-level similarity 는 fine-grained 변화 (객체 일부 이동) 미반영. 모든 phase 에서 동일한 hash compute → decode 단계까지 LSH overhead.
- 동작 원리: (1) encode 단계 — vision encoder 끝에서 (1024 visual token × 1280 dim) → mean pooling → 16 random projection hyperplane → 16-bit signature → radix tree key → (2) prefill 단계 — first 4 image token 의 8 hyperplane 으로 coarse signature 산출, exact-match 실패 시에만 fresh prefill → (3) decode 단계 — radix tree lookup only, hash compute 없음 → (4) Hamming distance threshold τ_encode = 2 (16-bit), τ_prefill = 1 (8-bit) → 거리 ≤ τ 시 KV reuse, > τ 시 fresh.
- 기존 해법 차별화: VLCache content-hash exact-match vs **본 기법 perceptual similarity (Hamming threshold)**. SGLang RadixAttention byte-level vs **본 기법 perceptual LSH**. encode/prefill/decode 별 hyperplane 가변은 unique.

#### Per-mechanism gain contribution
- Primary axis: [Performance] TTFT -22~30%
- Secondary axis: [Memory eff.] KV -30~45% (cross-request 공유)
- 단독 미보장 axis: [Accuracy] (false-positive 시 0.5pt drop, threshold 조정으로 mitigation)

#### Source-level implementation

> **구현 큰 그림**: SGLang v0.4+ 의 `python/sglang/srt/cache/radix_cache.py` 의 `RadixCache.match_prefix` 와 `python/sglang/srt/managers/scheduler.py` 의 `Scheduler.handle_request` 를 수정 + 신규 `python/sglang/srt/cache/visual_lsh.py` (PhaseAwareLSH class) 와 `csrc/visual_lsh_kernel.cu` (Triton autotune 16/8 hyperplane projection kernel) 를 추가하여 (text-hash, visual-LSH-hash) tuple key 의 phase-aware policy dispatch 를 구현. **측정 대상**: LSH signature compute latency (target 15us encode / 8us prefill / 0 decode) + cache hit rate (single-policy LSH baseline 대비) + Hamming false-positive rate (multi-camera surveillance trace) + MMMU pt drop. **개선 axis**: cache hit rate 40→80% (multi-camera surveillance), TTFT 1.20→0.85 s (-29%), false-positive accuracy drop ≤ 0.5pt, LSH overhead < 2% prefill latency.

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `python/sglang/srt/cache/radix_cache.py` | `RadixCache.match_prefix` | byte-level hash | (text-hash, visual-LSH-hash) tuple | Modify |
| `python/sglang/srt/cache/visual_lsh.py` (new) | `class PhaseAwareLSH` | — | encode/prefill/decode 별 hyperplane 수 가변 LSH | Add |
| `csrc/visual_lsh_kernel.cu` (new) | `lsh_signature_compute_phase` | — | Triton autotune kernel for 16/8 hyperplane projection | Add |
| `python/sglang/srt/managers/scheduler.py` | `Scheduler.handle_request` | uniform | phase 식별 후 LSH policy 분기 | Modify |
| `python/sglang/srt/cache/radix_node.py` | `RadixNode.children` | text-only | visual-hash branch 추가 | Modify |

GitHub verification trace: SGLang `radix_cache.py` 실재 ([github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) v0.4+, multimodal hash 지원). Triton autotune 표준. [✅]

#### Synthetic workload validation
- Unit test: 16-hyperplane LSH on 1280-dim feature, latency target 15us (encode), 5분.
- Mechanism-isolated: multi-camera surveillance trace (인접 frame similarity > 0.85, 60fps × 60s) cache hit rate 측정, 6시간.

---

### 4.2 — M2: RadixAttention Second-level Visual Prefix Branch

#### Concept
- 추가되는 Scheme: SGLang RadixAttention 의 token-level radix tree 의 root 직속 second-level branch 를 visual-semantic-hash 로 split. third-level 부터 text token 진행. shared image branch 에 KV cached → 동일 image 재사용 다수 prompt 가 prefill skip.
- 해결하려는 문제: 표준 RadixAttention 은 단일 sequence 단위 prefix tree — multimodal 시나리오에서 (image-A, prompt-X), (image-A, prompt-Y) 가 image-A 의 KV 를 공유하지 못하고 각각 처음부터 prefix match. dashboard agent (동일 image + 다른 prompt 50-70%) 에서 큰 낭비.
- 동작 원리: (1) request 도착 시 visual-LSH-hash 산출 (M1) → (2) radix tree root 의 second-level visual branch 에서 hash match 검색 → (3) match 시 image KV 재사용, third-level text token branch 로 진행 → (4) miss 시 새 visual branch 노드 생성 → (5) HiCache 의 GPU L1 / CPU L2 hierarchy 에 자동 spill (LRU).
- 기존 해법 차별화: VLCache 는 encoder cache + KV cache 의 single hash 로 monolithic, **본 기법은 radix tree 의 architectural extension** (second-level visual + third-level text). HiCache 는 exact match, **본 기법은 LSH similarity match**.

#### Per-mechanism gain contribution
- Primary axis: [Performance] cross-request KV reuse hit rate 40→80%
- Secondary axis: [Memory eff.] KV -30~45%
- 단독 미보장 axis: enabling for M1.

#### Source-level implementation

> **구현 큰 그림**: SGLang v0.4+ 의 `python/sglang/srt/cache/radix_node.py` 의 `RadixNode` 구조와 `python/sglang/srt/cache/radix_cache.py` 의 `RadixCache.insert` + `python/sglang/srt/cache/hicache_spill.py` 의 `HiCacheSpill.evict` 를 수정하여 second-level visual_branch + LRU 우선 retain 정책을 구현. **측정 대상**: dashboard agent trace (동일 image base 50개 + prompt variation 1000개) cache hit rate + radix tree depth/width + KV memory footprint (HiCache GPU L1 + CPU L2 layer 분포). **개선 axis**: cross-request KV reuse hit rate 40→80%, KV memory -30~45%, dashboard agent TTFT 1.20→0.85 s (-29%), radix tree second-level branch 노드 수 ≈ 50 (image base 수와 일치).

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `python/sglang/srt/cache/radix_node.py` | `RadixNode` | (key, children, kv) | second-level visual_branch 추가 | Modify |
| `python/sglang/srt/cache/radix_cache.py` | `RadixCache.insert` | uniform path | visual_hash 발견 시 second-level 진입 | Modify |
| `python/sglang/srt/cache/hicache_spill.py` | `HiCacheSpill.evict` | LRU only | visual_branch 우선 retain (frequent reuse) | Modify |

GitHub verification trace: SGLang radix_cache 실재. HiCache LMSYS production blog 참조. [✅]

#### Synthetic workload validation
- Unit test: radix tree insert + match for (visual_hash + text) tuple, 10분.
- Mechanism-isolated: dashboard agent trace (동일 image + 50 prompt) cache hit rate 측정, 4시간.

---

### 4.3 — M3: Green Context SM Partition with LSH Lookup

#### Concept
- 추가되는 Scheme: LSH signature 산출 + radix tree update 를 prefill/decode SM 과 분리된 Green Context 8-SM chunk (Blackwell granularity) background 에서 실행. Phase-aware policy 와 결합 — encode 단계만 LSH compute 활성, decode 단계 GC partition 비활성화.
- 해결하려는 문제: LSH compute 가 prefill 와 같은 SM 에서 실행하면 contention. decode latency 에 영향.
- 동작 원리: (1) CUDA 12.4 Green Context API 로 8-SM chunk 사용 (Jetson Thor 의 작은 SM 수 160 에서도 안전) → (2) encode phase 시작 시 GC 활성, LSH kernel launch → (3) prefill phase 진입 시 GC 가 background 에서 radix tree update 완료 → (4) decode phase 시 GC 비활성화, all SM 을 decode 에 할당 → (5) RadixAttention lookup (M2) 만 phase-non-blocking.
- 기존 해법 차별화: PD-Multiplexing (LMSYS 2025-09) 은 prefill/decode 단계 분리만, **본 기법은 LSH compute 라는 third role 분리**. Nova 는 SM partition 자체 contribution, **본 기법은 phase-aware + LSH-specific**.

#### Per-mechanism gain contribution
- Primary axis: enabling for M1/M2 (overhead 격리).
- Secondary axis: [Performance] decode latency 영향 ≤ 1%.
- 단독 미보장 axis: 격리 자체로는 gain 없음.

#### Source-level implementation

> **구현 큰 그림**: SGLang v0.4+ 의 `python/sglang/srt/managers/scheduler.py` 의 `Scheduler._dispatch` 를 수정 + 신규 `python/sglang/srt/managers/green_context_manager.py` (GreenContextManager.allocate_lsh_partition) 와 `csrc/green_context_init.cu` (CUDA 12.4 Green Context init wrapper) 를 추가하여 LSH kernel 을 8-SM GC stream 에 격리, prefill/decode 는 main stream 에 dispatch. **측정 대상**: nsight-systems 로 GC stream vs main stream 의 SM occupancy 분리 + decode latency 영향 (≤ 1% target) + LSH overhead 가 prefill latency 비율 (≤ 2% target) + cuCtxCreate_v3 + CU_CTX_GREEN flag return code. **개선 axis**: LSH compute prefill SM contention 제거 (decode latency 영향 ≤ 1%), LSH overhead < 2% prefill latency, encode phase GC 활성 시 SM 8/160 (5%) 사용.

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `python/sglang/srt/managers/green_context_manager.py` (new) | `GreenContextManager.allocate_lsh_partition` | — | 8-SM chunk allocation, encode phase 만 활성 | Add |
| `python/sglang/srt/managers/scheduler.py` | `Scheduler._dispatch` | uniform stream | LSH kernel 은 GC stream, prefill/decode 는 main stream | Modify |
| `csrc/green_context_init.cu` (new) | `init_green_context` | — | CUDA 12.4 Green Context init wrapper | Add |

GitHub verification trace: CUDA 12.4 Green Context API 실재 ([NVIDIA docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#green-contexts)). [✅]

#### Synthetic workload validation
- Unit test: 8-SM Green Context init + LSH kernel launch, 5분.
- Mechanism-isolated: prefill/decode SM 점유 contention 측정 (nsight-systems), 3시간.

## 5. 실험 플랜

### 5.1 Hardware environment
- Single-system primary: Jetson AGX Thor 128GB (UMA 환경 zero-copy host alloc 분기)
- Secondary: RTX 5090 32GB (GDDR7 + 96MB L2 carveout)
- Power meter: tegrastats / nvidia-smi

### 5.2 Model
- Qwen3-VL-8B
- LLaVA-Next-7B
- InternVL3-8B
- Qwen3-VL-2B (smaller scale validation)

### 5.3 Dataset/Workload
- Multi-camera surveillance trace (synthetic, 60fps × 60s, frame similarity 분포 [0.5, 0.99])
- Dashboard agent trace (동일 image base 50 + prompt variation 1000)
- Video VQA (VideoMME-short)
- ChartQA (single-shot baseline, cross-request 효과 미미 검증)
- ShareGPT-4o (image+prompt mixed dialogue)

### 5.4 Simulator/Tools
- 측정 only (no simulator-as-contribution)
- nsight-systems, tegrastats, custom trace generator

### 5.5 Ablation + Measurement Protocol
- Baseline 1: SGLang RadixAttention vanilla (byte-level hash)
- Baseline 2: VLCache (reproduce — code release 확인 필요, 2025-12)
- Baseline 3: HiCache (LMSYS production) — exact-match hierarchical radix
- Baseline 4: KVFlow ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)) — multi-agent prefix
- (Workshop supporting only, baseline 제외) SimCache CVPR 2025W — frame-level similarity workload evidence 만 인용
- Ablation: M1 only / M1+M2 / M1+M2+M3
- 5 run, 95% CI

### 5.6 Implementation Steps

| Week | Task | Deliverable |
|------|------|-------------|
| W1-2 | LSH kernel (Triton) for 16/8 hyperplane | unit test, ≤ 15us encode |
| W3 | RadixCache second-level visual branch | tree insert/match |
| W4 | Phase-aware policy dispatch | scheduler dispatch logic |
| W5 | Green Context 8-SM partition | nsys 검증 |
| W6 | HiCache spill 통합 | LRU + visual retain |
| W7 | Multi-camera surveillance trace generation | synthetic workload |
| W8 | VLCache reproduction | head-to-head |
| W9 | HiCache reproduction | head-to-head |
| W10 | Jetson Thor end-to-end | TTFT / KV memory measurement |
| W11 | RTX 5090 long-context (32K) | secondary platform |
| W12 | Tier-1 차별 검증 (RQ-3.4) | -10% 이상 vs VLCache 검증 → Tier 결정 |
| W13-14 | OSDI/SOSP draft (Tier-1) 또는 EuroSys/FAST (Tier-2) | submission |

### 5.7 Preliminary Analysis Metrics
- TTFT: target 0.85s (baseline VLCache 0.95s, -11%) → 목표 -10% 이상
- KV memory: target 0.55× baseline (cross-request shared)
- LSH overhead: ≤ 2% prefill latency
- Cache hit rate: 40 → 80% (multi-camera surveillance)

## 6. 예상 효과 (5-axis 정량 표)

| Axis | Baseline (SGLang RadixAttention vanilla) | RadixVL | 개선 | 조건 |
|------|------------------------------------------|---------|------|------|
| [Performance] TTFT | 1.20 s | 0.85 s | **-29%** | multi-camera surveillance (similarity > 0.85) |
| [Memory eff.] KV (cross-request) | 100% | 60% | **-40%** | dashboard agent (50% prompt overlap) |
| [Energy] J/request | 28 J | 23 J | **-18%** | recompute 제거 |
| [Power] avg | 92 W | 85 W | **-8%** | compute 절감 |
| [Accuracy] MMMU | 53.1 | 52.6 | -0.5 pt | LSH τ=2 false-positive |

조건: (a) gain 은 visual prefix overlap 50%+ workload 한정 (multi-camera, dashboard, video VQA, repeated UI). cold start single-shot 에서는 효과 미미. (b) M1 phase-aware policy 는 SGLang scheduler 흐름 (encode → prefill → decode) 가정. (c) M2 는 RadixAttention 표준 구조 의존. (d) M3 는 CUDA 12.4 Green Context API 의존.

## 7. Implementation Decision Flowchart (per-idea)

> 본 idea 단독 prototype 시 어느 mechanism 부터 시작 + 결과 분기.

### 7.1 1st Mechanism Priority

- **가장 먼저**: M1 Phase-aware Visual LSH Hash (encode 16 / prefill 8 / decode 0 hyperplane)
- **왜**: M2/M3 의 prerequisite — visual-LSH-hash 자체가 산출되어야 second-level radix tree branch (M2) 와 GC partition (M3) 모두 의미. 측정 도구 표준 (Triton kernel + multi-camera surveillance synthetic trace), VLCache 와의 차별 (RQ-3.4) 즉시 검증 가능
- **임계값**: encode LSH 16-hyperplane latency ≤ 15us + cache hit rate 40→80% (single-policy LSH 대비 +15%pt 이상) + Hamming false-positive accuracy drop ≤ 0.5pt

### 7.2 결과 분기

```
[M1 Phase-aware Visual LSH 1st measurement]
   ↓
   ├─ Pass (encode ≤ 15us / cache hit +15%pt / drop ≤ 0.5pt)
   │    → M2 RadixAttention second-level branch 진행
   │    → Pass (cross-request hit rate 40→80% / KV -40%) → M3 Green Context 격리 진행
   │    → Pass (decode latency 영향 ≤ 1% + LSH overhead ≤ 2%) → RQ-3.4 차별 검증 (W12)
   │       → VLCache 대비 추가 TTFT -10% 이상 → OSDI/SOSP 2026 full paper draft
   │       → 추가 -5~-10% → MLSys 또는 SOSP poster (Tier-1 borderline)
   │       → 추가 -5% 미만 → Tier-2 강등 → EuroSys/FAST 2026 submission
   │
   ├─ Below (encode 15-25us / cache hit +5~15%pt / drop 0.5-1.0pt)
   │    → M1 simplify (encode 16 → 12 hyperplane, prefill 8 → 4)
   │    → M2 maintain (architectural axis 단독 contribution)
   │    → EuroSys 2026 또는 SOSP 2026 short paper (Tier-2 강등)
   │
   ├─ Critical (encode > 25us 또는 drop > 1.0pt)
   │    → drop M1 phase-aware LSH (false-positive 정확도 critical)
   │    → reframe 으로 RadixAttention second-level branch (M2) 만 standalone
   │    → ICLR 2026 efficient track 또는 idea drop
   │
   └─ Outperform (encode < 10us / cache hit +25%pt / drop ≤ 0.2pt)
        → M1+M2+M3 all 진행 + VLCache 대비 추가 -15% 이상 검증
        → standalone Green Context multi-tenant Tier-2 spinoff (M3 단독 IISWC)
        → OSDI/SOSP 2026 full + MLSys 2026 system-axis short 동시 추진
```

## 8. 참고 / cross-share dependency

- **Atrium (본 세션 04-atrium)**: Green Context CUDA 12.4 SM partition API 사용 코드 공유 — Atrium 의 HOT/COLD context 와 본 idea 의 8-SM LSH context 가 동일 cuCtxCreate_v3 + CU_CTX_GREEN flag wrapper. partition manager class 통합 가능.
- **Bivouac (본 세션 02-bivouac)**: cross-frame reuse axis (Bivouac M2) 와 cross-request reuse axis (본 idea M2) 는 orthogonal — multi-camera surveillance 에서 동시 적용 시 frame 내 cluster reuse + frame 간 radix tree branch reuse 결합 가능.
- **Prism (본 세션 01-prism)**: phase-aware policy 와 layer-aware DVFS 결합 가능. Prism 의 nvpmodel layer-boundary DVFS 와 본 idea 의 phase-boundary GC partition 이 같은 boundary signal 활용.

## 9. Rule self-check

- **Tier**: 1 (조건부) — RQ-3.4 W12 검증 후 retain or Tier-2 강등 / **Lead expert**: legacy-system-expert (RadixAttention/SGLang ecosystem) / **Target venue**: OSDI 2026 / SOSP 2026 (조건부 Tier-1) / FAST 2026 / EuroSys 2026 (Tier-2 강등 시)
- **5-axis 정량**: Performance TTFT -22~30% / Memory KV -30~45% / Energy -15~20% / Power -8% / Accuracy ≤ 0.5pt drop
- **Single-system fit**: Jetson AGX Thor 128GB primary (UMA), RTX 5090 32GB secondary (GDDR7 + 96MB L2)
- **R47 path**: application-level only (SGLang fork + RadixAttention extension, Triton custom kernel, CUDA 12.4 Green Context API, no kernel patch)
- **R45 risk**: 6/10 (LSH false-positive accuracy 관리 필요. RadixAttention second-level branch 동기화 비용 측정 필요. Phase-aware policy 의 task-adaptive threshold 조정 의무)

- [x] R10-α bullet 형식
- [x] R20-α/β/γ mechanism 4 요소 + 7-요소 + single-system
- [x] R47 application-level
- [x] R52.1-5 / R53 / R54 (baseline source / file path 표 / GitHub trace / synthetic 3-tier / no simulator / final verification)
- [x] R55.2 5-axis / R55.3 mechanism gain (M1-M3 모두 primary/secondary/단독미보장 명시)
- [x] R56.1-3 reference integrity (peer-reviewed 62~70% / self-citation 0 / 가상 reference 0)
- [x] R57.5 idea-level detail 유지
- [x] R58.1-8 student-readable cleanup (rule notation inline 제거 / decision tree 재정의 / per-idea flowchart §7 / idea name 자연 합성어 RadixVL / metadata block 제거 / arxiv markdown link / **SimCache (CVPR 2025W) workshop only baseline 에서 제거하고 supporting evidence 로 격하** / mechanism narrative 큰그림 3개)
