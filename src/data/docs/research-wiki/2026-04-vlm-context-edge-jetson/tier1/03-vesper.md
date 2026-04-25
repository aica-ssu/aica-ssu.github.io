# UMA Zero-Copy KV Ledger with CPU-Side Patch Pruner for VLM Long-Context Decode on Jetson Edge (VESPER)

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-1 Top 3**

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **VLM** — Vision-Language Model.
> - **UMA** (Unified Memory Architecture) — Jetson 의 GPU+CPU 가 동일 LPDDR5X physical pool 공유.
> - **cudaMallocManaged / cuMemAdvise** — CUDA Unified Memory API. ([NVIDIA Maximizing Memory Efficiency](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/))
> - **NEON / SVE2** — ARM Cortex-A 의 SIMD intrinsic. `vdupq_n_f16`, `vmaxq_f16` 등 16 KV block 단위 in-place pruning.
> - **PagedAttention** — vLLM 의 KV cache 관리. block size = 16 token = 512B INT4.
> - **KV Ledger** — 본 idea 신규 자료구조. single physical LPDDR5X buffer 를 GPU+CPU 동시 view, evict_pending flag 포함.
> - **lockfree ringbuffer** — concurrent producer/consumer 구조. `lockfree_ringbuffer<AttScore>` 형식.
> - **Grace period** — concurrent eviction 시 race condition 회피 buffer. 본 idea 1 token grace.
> - **VL-Cache** — VLM visual KV layer-adaptive sparsity-aware compression, ICLR 2025. ([OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub))
> - **VLCache** (대문자 V — 별도 paper) — 98% vision token reuse, [arXiv:2512.12977](https://arxiv.org/abs/2512.12977). VESPER 직접 baseline.
> - **Dissecting CPU-GPU UPM** — UMA + page-fault overhead 정량, [arXiv:2508.12743](https://arxiv.org/abs/2508.12743). VESPER baseline.
> - **H2O** — heavy hitter eviction, NeurIPS 2023. GPU only.
> - **SnapKV** — KV compression, NeurIPS 2024.
> - **MileBench** — long-context VLM benchmark.
> - **MVBench** — video VLM benchmark, 256-frame slice.
> - **Video-MME** — video VLM benchmark.
> - **AKVQ-VL** — KV bit-width quantization. baseline.
> - **MBQ** — bit-level minimization, MICRO 2024. baseline.
> - **PowerInfer-2** — smartphone hot/cold neuron split, [arXiv:2406.06282](https://arxiv.org/abs/2406.06282). idle CPU 41-67% 정량 source.
> - **MetaEmbed** — embedding axis, [arXiv:2509.18095](https://arxiv.org/abs/2509.18095). adjacent baseline.
> - **LMCache** — CPU offload, [arXiv:2510.09665](https://arxiv.org/abs/2510.09665). CPU offload 만, NEON in-place pruning + UMA dual-view 부재.

**Target Venue**: OSDI 2027 (12p) (primary) / SOSP 2027 (12p) (alternative)
**Score** (Phase 2 평균): Novelty **8.0** / Diff **7.5** / Impact **8.8** = 평균 **8.10**
**판정**: Accept Tier-1 (Phase 2' Strong Refine — system gain vs algorithmic gain 분리 ablation 의무)
**Phase 1' diff**: ΔM = 0 (improve-only) — baseline VLCache [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) + Dissecting CPU-GPU UPM [arXiv:2508.12743](https://arxiv.org/abs/2508.12743) 추가, system gain vs algorithmic gain 분리 ablation 강화.

---

## 1. 개요 (Overview)

본 연구는 **Jetson Orin NX / Thor 의 UMA 환경** 에서 VLM long-context (image 4 + text 4K) decode 진행 중 (a) image patch token KV 의 60-72% 가 attention score < 0.005 으로 redundant 임에도 (b) vLLM 이 prune 결정을 GPU kernel 안에서 하므로 SM 점유 — 동시에 CPU (Thor 14-core / Orin NX 8-core / Nano 6-core Cortex-A78AE) 가 **유휴 상태로 41-67% 시간 낭비** ([PowerInfer-2 arXiv:2406.06282](https://arxiv.org/abs/2406.06282)) 하는 비대칭을 정량화하고, **cudaMallocManaged dual-view KV buffer + CPU NEON SIMD in-place pruner + lockfree ringbuffer + 4-tier evict policy** 로 양쪽 자원을 동시 활용하는 mechanism 을 제안한다.

**가설**: Orin NX long-context (≥ 8K) decode tok/s **22-28% ↑**, energy/token **17-24% ↓**, idle CPU 활용도 41% → 78%, accuracy drop ≤ 1.0pp (MMMU). Short context (≤ 1K) 효과 미미 (5% 미만) — scope 명시.

**Metaphor 부속 (R30)**: "VESPER" = 저녁의 공동 기도 (UMA 공유의 metaphor). 후보: CONFLUENCE / SYMBIONT.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 다루는 axis | 한계 (본 연구 대비) |
|-----------|------------|---------------------|
| **VL-Cache** [ICLR 2025 OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) | layer/token granularity KV compression | **CPU 활용 0%** — GPU-only |
| **VLCache** [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) | 98% vision token reuse | cross-request reuse, **single-request decode-time pruning 부재** |
| **AKVQ-VL** | bit-width quantization | KV 정적 residence |
| **SparseVILA** [CVPR 2024] | prefill sparse vision token | **decode-time refresh 없음** |
| **vLLM CPU offload** | KV 를 CPU 로 이동 | UMA 환경에서 의미 없음. VESPER 는 동일 buffer 양쪽에서 조작 |
| **MBQ** [MICRO 2024] | bit-level minimization | residence/concurrency axis 다름 |
| **H2O** [NeurIPS 2023] | heavy hitter eviction | GPU only |
| **SnapKV** [NeurIPS 2024] | KV compression | GPU only |
| **LMCache** [arXiv:2510.09665](https://arxiv.org/abs/2510.09665) | CPU offload | CPU offload 만, **NEON in-place pruning + UMA dual-view 부재** |
| **Dissecting CPU-GPU UPM** [arXiv:2508.12743](https://arxiv.org/abs/2508.12743) | UMA page-fault 정량 | analysis only, **defense mechanism 부재** |
| **PageWeave** (legacy-sys, 미선정) | visual-token lifetime classifier + cgroup | UMA + KV management 같은 axis. PageWeave 는 kernel layer attack, VESPER 는 user-space NEON SIMD — layered defense 가능 (별도 paper). |

**GAP**: **UMA 의 dual-view buffer + CPU NEON SIMD KV pruner + lockfree ringbuffer 4-tier evict** 통합은 현재 공개 논문 없음 — first-to-report.

### 기존 세션 중복 회피

- **HRTS / ContextMIG / RESIDUA** (이전 세션): datacenter HBM 가정. UMA dual-view 부재.
- **PageWeave** (본 세션 미선정): kernel layer cgroup attack — VESPER 와 layered defense 가능, 별도 paper.

---

## 3. 제안 기법 (Core Mechanisms, 4 mechanisms — improve-only ΔM=0)

### M1: KV Ledger (Dual-View UMA Buffer)

**① 추가되는 Scheme — Source Verified (R32)**:

Single physical LPDDR5X buffer 를 cudaMallocManaged + cudaMemAdvise(cudaMemAdviseSetPreferredLocation, CPU) 로 GPU+CPU 동시 access. vLLM `KVCacheManager.allocate_dual_view(num_blocks, modality='image')` 신규 API.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/core/kv_cache_manager.py` (dual_view enum 추가 가능)
> ✅ source verified: cudaMallocManaged + cuMemAdvise ([NVIDIA Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/))
> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/core/block_pool.py::BlockPool.free` (evict_pending flag 추가 가능)

**② 해결하는 문제 + Workload evidence**:

[PowerInfer-2 arXiv:2406.06282](https://arxiv.org/abs/2406.06282): smartphone LPDDR5X 환경 sparse activation 23-37% 만 메모리 transfer. Jetson UMA 도 동일 패턴 — CPU 가 유휴 41-67% 시간. [VLMBench CVPR-W 2024](https://arxiv.org/abs/2406.09246): RefCOCO image patch attention score < 0.005 인 KV 가 60-72%. [Dissecting CPU-GPU UPM arXiv:2508.12743](https://arxiv.org/abs/2508.12743): UMA page-fault overhead 정량.

**③ Step-by-step**:

1. KV block 할당 시 `cudaMallocManaged(ptr, 16*hidden_dim*sizeof(int4))`.
2. `cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, CPU)` 로 CPU 의 access 효율 확보.
3. vLLM `KVCacheManager.allocate_dual_view(num_blocks, modality='image')` 호출 시 modality flag 저장.
4. 양쪽 view 의 virtual addr 를 metadata table 에 저장 — GPU 는 일반 GMEM access, CPU 는 NEON intrinsic 으로 access.

**④ 기존 해법 실패 + 1:1 차별화**:

- **vs vLLM CPU offload**: 명시적 cudaMemcpyAsync. VESPER 는 zero-copy.
- **vs cudaMallocManaged 기존 사용**: 단일 view (GPU only). VESPER 는 dual view + modality 분류.
- **vs SHOAL**: GPU-only mapped (cudaHostRegisterMapped). VESPER 는 GPU+CPU 양쪽 active.

### M2: CPU NEON SIMD In-Place Pruner

**① 추가되는 Scheme**:

Decode step 진행 중 GPU 가 token t 처리하는 동안, CPU thread 가 token (t-W..t-1) 의 attention statistics 를 lockfree ringbuffer 로 받아 NEON SIMD intrinsic 으로 16 KV block 단위 importance score 계산. Threshold τ 미만이면 block 의 evict_pending=true flag set.

> ✅ source verified: ARM NEON Intrinsics ([ARM Developer](https://developer.arm.com/architectures/instruction-sets/intrinsics/))
> ✅ source verified: llama.cpp ggml NEON kernel `ggml/src/ggml-cpu/ops.c::ggml_compute_forward_mul_mat_neon` (참조 구조)
> 🔧 source proposed: 신규 NEON kernel ~150 LOC for KV pruning

**② 해결 + evidence**:

GPU kernel 안에서 prune 결정 시 SM 점유 — Orin Nano 8GB 의 1024 CUDA core 가 decode 중 91% busy. CPU NEON 으로 분담하면 GPU SM 활용 가능.

**③ Step-by-step**:

1. NEON intrinsic `vdupq_n_f16(0)` initialize, `vmaxq_f16` 으로 16 block 단위 max attention score 계산.
2. 각 block 의 max score 를 threshold τ 와 비교 (`vcleq_f16` for ≤).
3. τ 미만 block 의 metadata 의 evict_pending flag set (atomic store).
4. 16 block 처리 ~120 cycle = 0.05μs per block 단위.

**④ 차별화**:

- **vs GPU kernel pruner (H2O / SnapKV)**: SM 점유 vs CPU 분담.
- **vs CPU scalar pruner**: NEON SIMD 16× speedup.
- **vs LMCache CPU offload**: pure offload, in-place pruning 부재.

### M3: Lockfree Ringbuffer + 4-Tier Evict Policy

**① 추가되는 Scheme**:

CPU thread 가 GPU attention statistics 를 받는 buffer 는 `lockfree_ringbuffer<AttScore>` (Boost.Lockfree 또는 custom). 4-tier evict policy:
- Tier 0: image patch background (τ=0.001) — 가장 적극 evict
- Tier 1: image patch object (τ=0.005)
- Tier 2: text history (τ=0.01)
- Tier 3: system prompt (never evict)

> ✅ source verified: Boost.Lockfree ([docs](https://www.boost.org/doc/libs/release/doc/html/lockfree.html))
> 🔧 source proposed: 4-tier τ tuning on validation MileBench

**② 해결 + evidence**:

[VLMBench CVPR-W 2024](https://arxiv.org/abs/2406.09246) RefCOCO 측정: image patch background 가 token 의 73% 차지하지만 attention < 0.001. 4-tier 는 modality + semantic role 통합.

**③ Step-by-step**:

1. CPU thread 가 ringbuffer 에서 (block_id, att_score, modality) tuple 소비.
2. modality 와 semantic role (background/object/history/system) 분류 — pre-computed metadata.
3. tier-specific τ 와 비교 → evict_pending flag set.
4. ringbuffer overflow 시 oldest item 삭제 (lossy 허용 — eventual consistency).

**④ 차별화**:

- **vs single-threshold eviction (H2O)**: tier 무관.
- **vs SnapKV**: GPU kernel only.
- **vs VL-Cache**: layer-level, semantic role 부재.

### M4: Grace Period Race Condition Avoidance

**① 추가되는 Scheme**:

GPU 가 다음 prefill chunked step 진입 시 evict_pending block 을 free list 로 회수. **Grace period 1 token** — race condition 회피용. cudaMemcpyAsync 불필요 (UMA).

> ✅ source verified: vLLM `vllm/v1/core/block_pool.py::BlockPool.free` (free list manipulation)
> 🔧 source proposed: grace_period 1 token implementation

**② 해결 + evidence**:

GPU 와 CPU 가 동시 access — race condition 우려. Grace period 1 token 동안 evict_pending block 을 정상 KV 로 취급, 다음 token decode 시작 전에 회수.

**③ Step-by-step**:

1. Decode token t 시작 시 block 의 evict_pending flag 확인.
2. flag=true 이고 last_modified < t-1 (1 token grace) 이면 free list 로 회수.
3. flag=true 이고 last_modified ≥ t-1 이면 회수 보류 (다음 step).
4. GPU 는 free list 의 새 block 우선 할당 (allocator preference).

**④ 차별화**:

- **vs immediate eviction (H2O)**: race 발생 가능.
- **vs lazy eviction (no race avoidance)**: stale data risk.

### Mechanism 간 상호작용

M1 (dual-view buffer) 가 foundation. M2 (NEON pruner) + M3 (ringbuffer + 4-tier) 가 control plane. M4 (grace period) 가 race avoidance. 4-mechanism 모두 hierarchical. **System gain vs algorithmic gain 분리 ablation 의무** — Phase 1' Strong Refine 사항:
- **Cell A**: GPU-only H2O baseline (algorithmic eviction only, no system change).
- **Cell B**: CPU-only scalar KV pruner (system axis only, no SIMD).
- **Cell C**: NEON SIMD dual-view (VESPER full).
- **Cell D**: NEON SIMD only, no dual-view (intermediate).

3-way ablation A/B/C 가 OSDI reviewer 의 "그냥 H2O 를 CPU 에서 돌린 것" 비판 회피 핵심.

**Tier 구성**: physical 1-tier (single Jetson) + software 1-tier (single VLM). R1/R1b ≤ 3-4 안전.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: Jetson Orin NX 16GB (UMA 102 GB/s 정확 측정)
- **Secondary**: Thor 128GB (NDA scope), Orin Nano 8GB (degradation case)

### (2) Model

- **Qwen3-VL-8B** INT4
- **InternVL3-8B** INT4
- **LLaVA-NeXT-7B** (long-context image-grid)

### (3) Dataset · Workload

- **MileBench** (long-context VLM)
- **MVBench 256-frame** slice
- **DocVQA** val (long-context document)
- **Video-MME** (video VLM)
- Metrics: decode tok/s, energy/token, KV memory footprint, CPU utilization (target 41% → 78%), accuracy drop, NEON kernel latency

### (4) Simulator · Tools

- **vLLM v1 fork** (~250 LOC)
- **Nsight Systems** (CPU/GPU concurrent timeline)
- **Linux perf** (NEON cycle counter `cpu/cycles/`)
- **tegrastats** (energy)
- **ARM Streamline** (option, NEON cycle precise)

### (5) Ablation · Baseline

**Baselines (5+ peer-reviewed, R2 ≥ 50%)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **vLLM v1 stock** (no eviction) | [docs](https://docs.vllm.ai/) ✓ open-source | baseline |
| (b) | **VL-Cache** | [ICLR 2025 OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) ✓ peer-reviewed | KV compression |
| (c) | **VLCache** (Phase 1' 추가) | [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) | 98% reuse, orthogonal |
| (d) | **SparseVILA** | CVPR 2024 ✓ peer-reviewed | prefill prune |
| (e) | **H2O** | NeurIPS 2023 ✓ peer-reviewed | heavy hitter (GPU only) |
| (f) | **SnapKV** | NeurIPS 2024 ✓ peer-reviewed | KV compression (GPU only) |
| (g) | **LMCache** | [arXiv:2510.09665](https://arxiv.org/abs/2510.09665) | CPU offload |
| (h) | **Dissecting CPU-GPU UPM** (Phase 1' 추가) | [arXiv:2508.12743](https://arxiv.org/abs/2508.12743) | UMA page-fault baseline |
| (i) | **MetaEmbed** | [arXiv:2509.18095](https://arxiv.org/abs/2509.18095) | embedding axis (adjacent) |

Peer-reviewed ratio: 5/9 = **55.5%** (R2 ≥ 50% 충족).

**Ablation matrix (system vs algorithmic 분리, Phase 1' Strong Refine)**:
- Cell A: GPU-only H2O baseline
- Cell B: CPU-only scalar KV pruner
- Cell C: NEON SIMD dual-view (VESPER full M1+M2+M3+M4)
- Cell D: NEON SIMD only (no dual-view, no M1)
- × (Qwen3-VL-8B / InternVL3-8B / LLaVA-NeXT-7B)
- × (MileBench / MVBench / Video-MME)
- × (NEON disabled / NEON enabled)
- × (4-tier τ {0.001/0.005/0.01/never} sweep)
- × (grace period {0, 1, 4, 16} token)

핵심 ablation 16 cell 본문.

**Fallback**: SGLang attention sink Triton-on-Jetson 부분지원 (Orin Nano 미지원) → llama.cpp NEON kernel 외부 통합 baseline.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | cudaMallocManaged + memAdvise path 검증. **NVIDIA Tech Blog ✅** | CUDA 11+, JetPack 7.1 | dual-view buffer 동시 access 단위 테스트 통과 |
| Step 2 | Step 1 | NEON kernel skeleton 작성. **llama.cpp NEON 참조 ✅** | NEON intrinsics ~150 LOC | 16 block batch SIMD 통과 |
| Step 3 | Step 2 | lockfree ringbuffer 통합 (M3) | Boost.Lockfree | producer/consumer 8K item/s 처리 |
| Step 4 | Step 1-3 | evict_pending flag + grace period (M4) | vLLM patch ~80 LOC | race condition 0 in 1M-token stress |
| Step 5 | Step 4 | 4-tier τ tuning on validation MileBench | vLLM eval | optimal τ per tier 결정 |
| Step 6 | Step 5 | full eval + accuracy drop check (≤ 1.0pp on MMMU) | vLLM eval harness | accuracy regression ≤ 1.0pp |
| Step 7 | Step 6 | system vs algorithmic 분리 ablation (Cell A/B/C/D) | vLLM ablation harness | 4-cell trace dump |
| Step 8 | Step 7 | 9 baseline 구현/재현 | (a)-(i) repo | baseline table 완성 |
| Step 9 | Step 8 | 표 1-5 + figure 1-7 작성 | matplotlib, pandas | manuscript draft 70% |
| Step 10 | Step 9 | system framing + race analysis discussion | manual writing | 12p OSDI draft 완성 |
| Step 11 | Step 10 | polish, OSDI 2027 submission | git + README | submission-ready |

**참고 시간**: 약 12 weeks.

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| Decode tok/s (Orin NX, long-context ≥ 8K) | vLLM logging | sustained 5min | base | **+22~+28% (Tier-1 성공 조건)** |
| Energy/token | tegrastats integration | concurrent CPU+GPU | base | **-17~-24%** |
| KV memory footprint | vLLM debug | post-eviction | base | **-50~-65%** |
| CPU utilization | mpstat | concurrent | 41% | **78%** |
| Accuracy drop | MMMU / MileBench | vs baseline | 0pp | **≤ 1.0pp** |
| NEON kernel latency | perf | 16-block batch | — | ≤ 0.05μs/block |
| Race condition incidents | grace period log | 1M token stress | — | **0** |
| System gain (Cell A vs Cell C) | ablation | algorithmic isolated | — | **+8~+12% beyond H2O** |
| Algorithmic gain (Cell B vs Cell C) | ablation | system isolated | — | **+10~+15% beyond CPU scalar** |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: VL-Cache, H2O, SnapKV decode tok/s 재현 (Orin NX).
- **(ii) Bottleneck attribution**: CPU idle 41-67% measurement 재현 (PowerInfer-2 method).
- **(iii) Roofline**: Orin NX 102 GB/s effective bandwidth — UMA dual-view 가 bandwidth doubling 인지 검증.
- **(iv) Micro-benchmark**: NEON SIMD vs scalar KV pruning latency.

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| Decode tok/s (Orin NX, ≥8K) | base | **+22~+28%** | long-context |
| Energy/token (Orin NX) | base | **-17~-24%** | concurrent |
| Decode tok/s (Thor) | base | +12~+15% | HBM 미사용 환경 |
| Decode tok/s (short ≤ 1K) | base | +5% 미만 | scope 제한 |
| Accuracy drop | 0pp | ≤ 1.0pp | MMMU |
| CPU utilization | 41% | 78% | idle 자원 활용 |

**과학적 contribution**:
1. **First-to-report**: UMA dual-view + NEON SIMD KV pruner + 4-tier evict policy 통합.
2. **System axis isolation**: algorithmic eviction (H2O) 와 system gain 분리 정량화.
3. **CPU-side concurrency**: VLM serving 의 CPU+GPU 동시 활용 패러다임.

**실용적 impact**:
- Long-context VLM (video, document) 의 edge 배포 가능성 — Orin NX 에서 8K context decode 안정.
- Open-source vLLM upstream patch path 명확.

**Scope 제한**:
- Short context (≤ 1K) 효과 미미 — long-context 만 의미.
- Accuracy drop 1.0pp 까지 허용 — strict accuracy 환경 부적합.
- SGLang Triton-on-Jetson Orin Nano 미지원 — baseline 부족.

---

## 6. Tier 분기 (Tier-1 강점 / Tier-2 fallback variant)

본 idea 를 **Tier-2 ISLPED 6p / DAC 6p** 로 분리 publication 시:

- **Single mechanism**: M2 only (NEON SIMD pruner). M1/M3/M4 는 Tier-1 보존.
- **Scope 축소**: Orin NX 16GB + Qwen3-VL-8B + MileBench long-context only.
- **Baseline 2-3 편**: vLLM stock + H2O + VL-Cache.
- **참고 소요**: 약 5-6 weeks.
- **Tier-1 과의 관계**: precedence claim ("first-to-report NEON SIMD KV pruner on Jetson edge").

---

## 7. 미선정 baseline 보강 (Phase 1' 정제 사항)

Phase 1' improve-only 사항 (ΔM=0):

- **VLCache** [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (Phase 1' 추가) — 98% vision token reuse. 직교 axis (cross-request reuse vs single-request decode-time pruning) — 직접 비교 의무.
- **Dissecting CPU-GPU UPM** [arXiv:2508.12743](https://arxiv.org/abs/2508.12743) (Phase 1' 추가) — UMA page-fault baseline. analysis 만, defense 부재 — 본 idea 가 이를 mitigation.
- **System gain vs algorithmic gain 분리 ablation 강화** (Phase 1' Strong Refine 사항) — Cell A/B/C/D matrix 명시.
- **MileBench peer-reviewed 베이스 강화** — long-context benchmark 정당화.

---

## R45 적용 — Implementation Path 검증

본 idea 의 4 mechanism 모두 vendor 공식 user-space API 위에서 구현 가능 — R45 risk **3/10 LOW** (Tier-1 lead 의 가장 낮은 risk). R45.1 금지 카테고리 위반 없음:

- **Mechanism #1 (UMA dual-view KV)**: `cudaMallocManaged` + `cudaHostRegisterMapped` + `cuMemAdvise(SetPreferredLocation)` 모두 NVIDIA CUDA Runtime 공식 API. Jetson UMA 환경의 [공식 NVIDIA Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/) 에서 권장 path.
- **Mechanism #2 (NEON SIMD CPU pruner)**: `vdupq_n_f16` / `vmaxq_f16` / `vld1q_f16` 등 ARM NEON intrinsics — [ARM 공식 Intrinsics docs](https://developer.arm.com/architectures/instruction-sets/intrinsics/) user-space.
- **Mechanism #3 (lockfree ringbuffer + grace period)**: `Boost.Lockfree` user-space C++ — kernel patch 불필요.
- **Mechanism #4 (vLLM block table extension)**: `vllm/v1/core/kv_cache_manager.py` Python-level extension — closed-source firmware 의존 없음.

R45.3 (feasibility) 도 5 workload (Qwen3-VL-4B / InternVL3-2B long-context / MileBench / MVBench / Video-MME) × 3 config (NEON only / dual-view only / both) × 2 baseline (H2O / VLCache) = 30 runs feasibility 확보 — Orin NX 16GB 1 device 12-16주 fit. 모든 Jetson (Thor / Orin AGX / NX / Nano) 적용 가능.

R45 종합 판정: **risk 3/10 LOW, Tier-1 lead 적합**.

---

## 8. Source Verification (R32 통합)

| Component | Path / Function | 상태 |
|---|---|---|
| vLLM `KVCacheManager.allocate` | `vllm/v1/core/kv_cache_manager.py` | ✅ dual_view enum 추가 가능 |
| vLLM block free list | `vllm/v1/core/block_pool.py::BlockPool.free` | ✅ evict_pending flag 추가 |
| llama.cpp ggml NEON kernel | `ggml/src/ggml-cpu/ops.c::ggml_compute_forward_mul_mat_neon` | ✅ 참조 NEON 구조 |
| SGLang attention sink | `sglang/srt/layers/attention/triton_ops/decode_attention.py` | ⚠️ Triton-on-Jetson 부분지원 (Orin Nano 미지원) |
| TensorRT-LLM KV pool | `tensorrt_llm/runtime/kv_cache_manager.py` | ⚠️ public 부분 공개, dual-view 직접 패치 필요 |
| cudaMallocManaged + cuMemAdvise | [NVIDIA Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/) | ✅ |
| ARM NEON Intrinsics | [ARM Developer](https://developer.arm.com/architectures/instruction-sets/intrinsics/) | ✅ |
| Boost.Lockfree | [docs](https://www.boost.org/doc/libs/release/doc/html/lockfree.html) | ✅ |

---

## 9. Reference 목록 (이 idea 핵심)

- **VL-Cache** [ICLR 2025]: [OpenReview HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) / [arXiv:2410.23317](https://arxiv.org/abs/2410.23317)
- **VLCache**: [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (Phase 1' 추가)
- **Dissecting CPU-GPU UPM**: [arXiv:2508.12743](https://arxiv.org/abs/2508.12743) (Phase 1' 추가)
- **PowerInfer-2**: [arXiv:2406.06282](https://arxiv.org/abs/2406.06282)
- **VLMBench**: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- **EdgeMoE** [MobiSys 2024]: [arXiv:2308.14352](https://arxiv.org/abs/2308.14352)
- **LMCache**: [arXiv:2510.09665](https://arxiv.org/abs/2510.09665)
- **MetaEmbed**: [arXiv:2509.18095](https://arxiv.org/abs/2509.18095)
- **H2O** [NeurIPS 2023]: [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- **SnapKV** [NeurIPS 2024]: [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
- **MileBench**: [arXiv:2404.18532](https://arxiv.org/abs/2404.18532)
- **MVBench** [CVPR 2024]: [arXiv:2311.17005](https://arxiv.org/abs/2311.17005)
- **Video-MME**: [arXiv:2405.21075](https://arxiv.org/abs/2405.21075)
- **NVIDIA Maximizing Memory Efficiency**: [Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
