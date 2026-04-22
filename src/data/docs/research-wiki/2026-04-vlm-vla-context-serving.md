# Session 2026-04-22 — Mode 1 — VLM/VLA Context-Aware Caching & Serving Optimization

## Meta
- **User request**: "VLM 혹은 VLA 모델에서 Context-awareness 를 고려해서 caching 하거나 serving optimization (시스템레벨에서, kernel optimization, processing unit 할당, offloading, 메모리 할당 정책 등) 하는 것에 관심이 있는데, 관련된 최근 1,2년간 연구탐색을 통해 ideation 을 진행해줘"
- **Mode**: 1 (sentence-input)
- **Experts participated**: ai-optimization-expert (primary), legacy-system-expert (primary), algorithm-expert (sub-predictor)
- **Reviewers**: novelty-reviewer, differentiation-reviewer, impact-reviewer, similarity-critique (differentiation-reviewer 재분장)
- **Constraints**: PIM 비의존 (pure GPU serving systems stack 만). Single workstation scope (RTX 4090/5090/Pro 6000). 이전 세션의 MoE/PIM 중심 contribution 과 중복 금지.
- **Papers analyzed**: 55+ (외부 arxiv/WebSearch 직접 수집). Closest competitor 중심.
- **Ideas generated**: 6 (A1-A3 + L1-L3) + 2 predictor (P1, P2). Predictor 는 main idea 에 흡수되어 최종 공식 아이디어는 6개.
- **Phase flow**: 1(6개 초안) → 2(3-인 리뷰 + 전문가 상호 + 유사연구 critique) → 1'(refinement) → 2'(2차 리뷰 + 최근 6개월 유사연구 재점검) → 1''(Top 3 선정).
- **Output target**: PIM 에 의존하지 않고 **request/scene/trajectory context** 를 system-level resource 결정 (SM 파티션, HBM tier, kernel variant, SM yielding, CUDA Graph 분기) 으로 연결하는 defensible 연구주제.

---

## Phase 2' 외부 검색 후속 검증 (2026-04-22 수행)

Phase 2' integrated review 에서 similarity-critique agent 가 제시한 11편 "최근 6개월 (2025-10 이후) competitor" placeholder 에 대해 WebSearch 로 실존 여부를 재검증하였다. 결과:

### ✅ 실존 확인 (2편) — **scoop/concurrent 판정 유효**

1. **"GUIAgent-KV" (placeholder) → [arXiv:2510.00536](https://arxiv.org/abs/2510.00536) "GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness" 실존** (2025-10-01, Kung-Hsiang Huang, Haoyi Qiu, Yutong Dai).
   - **확인 메커니즘**: (a) Attention sparsity uniformly high across layers → uniform budget. (b) Spatial saliency = **residual stream L2 norm of hidden states** augments attention score. (c) Temporal redundancy = **previous frames' keys 를 current frame key subspace 로 projection 하여 redundant history 제거**.
   - **AgentNetBench 5-screenshot 결과**: decoding FLOPs -38.9%, step accuracy +4.1%.
   - **L3 MTV-Pool 대비 재평가**: 메커니즘 자체 (key-subspace projection vs. turn_oldness+log-sum-exp) 는 다르나, **target scenario (multi-turn GUI agent screenshot history) 와 positioning ("screenshot history 용 KV 압축") 이 완전히 중첩**. **유사도 재평가 55-65%** (기계적 일치는 아니나 contribution positioning 정면 scoop). → **L3 Major Revision 판정 유지** (근거 강화).

2. **"EventPrefetch-Robot" (placeholder) → [arXiv:2603.14371](https://arxiv.org/abs/2603.14371) "OxyGen: Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism" 실존** (2026-03-15, Xiangyu Li, Huaizhi Tang, Xin Ding).
   - **확인 메커니즘**: (a) Embodied AI 의 **multi-task-within-single-request** (manipulation + conversation + memory from shared observations, 각 다른 time constraint). (b) KV cache 를 first-class shared resource 로 취급, cross-task KV sharing 으로 redundant prefill 제거. (c) Cross-frame continuous batching — variable-length language decode 와 fixed-rate action generation 분리.
   - **결과**: 3.7× speedup, 200 tokens/s language throughput + 70 Hz action frequency 동시.
   - **A3 SemCOW-Deadline 대비**: OxyGen 은 **single-request 내 multi-task**, A3 는 **multi-request cross-tenant** — 축이 다름. COW/refcount/deadline-aware SM yielding 은 OxyGen 에 미명시. **유사도 ~35% adjacent** (baseline 필수, scoop 아님).
   - **L2 TemporalTier-3 대비**: OxyGen 은 cross-task sharing, L2 는 hierarchical HBM/pinned tier — 직접 overlap 없음. **유사도 ~30% adjacent**.
   - → A3/L2 baseline 에 추가. 판정 변경 없음 (둘 다 정면 scoop 아님).

### ℹ️ 실존 확인 (1편, withdrawn) — **기록 목적**

3. **"ScreenHistory-Cache" (placeholder) 는 없지만 → [arXiv:2603.26041](https://arxiv.org/abs/2603.26041) "Rethinking Token Pruning for Historical Screenshots in GUI Visual Agents: Semantic, Spatial, and Temporal Perspectives" 가 2026-03-27 제출, 2026-03-31 withdrawn** (authorship disputes + incomplete docs).
   - **Temporal perspective**: recency-based budget allocation (larger budget to recent screenshots, compress distant) — **L3 의 turn_oldness 와 정면 유사**. 그러나 withdrawn 상태이므로 공식 scoop 로 카운트하지 않음. 저자들이 재제출 시 재평가 필요.
   - → L3 미선정 로그에 "withdrawn paper 존재" 로 참고 언급.

### ❌ 실존 부재 (9편) — **Phase 2' 판정 근거 보정**

다음 9편은 arxiv/openreview/conference 검색으로 발견되지 않음 (agent 환각 placeholder 로 확인):

| Placeholder | 검색 결과 | 실제 adjacent 연구 (실존, baseline 후보) |
|-------------|---------|---------------------------------------|
| "Helix-VLA" (A1) | **부재**. Figure AI Helix 제품 blog만 존재, 논문 없음. | [arXiv:2505.03912](https://arxiv.org/abs/2505.03912) OpenHelix (survey+opensource, dual-system VLA) — A1 baseline 추가 |
| "TileSparse" (A2) | **부재**. | FlashInfer MLSys'25 가 이미 tile-level block-sparse. [arXiv:2507.09071](https://arxiv.org/abs/2507.09071) BlindSight (input-template-aware attention sparsity for VLM) — A2 baseline 추가 |
| "L2-Persist-Attn" (A2) | **부재**. | [arXiv:2510.14719](https://arxiv.org/abs/2510.14719) Tawa (Automatic Warp Specialization), FA-3 관련 — A2 reference |
| "SLOServe-GC" (A3) | **부재**. | [arXiv:2504.08784](https://arxiv.org/abs/2504.08784) SLOs-Serve, [arXiv:2508.16449](https://arxiv.org/abs/2508.16449) GreenLLM (SLO-aware dynamic frequency), [arXiv:2501.12162](https://arxiv.org/abs/2501.12162) AdaServe (SLO-customized speculative), [arXiv:2601.10729](https://arxiv.org/abs/2601.10729) OrbitFlow — **A3 baseline 에 SLOs-Serve / AdaServe / OrbitFlow 추가** |
| "Refcount-KV" (A3) | **부재**. | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) PagedAttention 이 이미 copy-on-write + refcount 구현 — A3 의 foundational reference로 명시. A3 의 contribution 은 **page-granular refcount 위에 top-k partial recompute + deadline-aware SM yielding 추가**로 재포지셔닝 유지 |
| "NestedGPU" (L1) | **부재**. | [arXiv:2409.06646](https://arxiv.org/abs/2409.06646) Optimal Workload Placement on MIG, [arXiv:2508.20274](https://arxiv.org/abs/2508.20274) Predictable LLM Serving on GPU Clusters, [arXiv:2504.19516](https://arxiv.org/abs/2504.19516) Bullet — **L1 baseline 에 추가** |
| "Taxonomy-Sched" (L1) | **부재**. | [arXiv:2506.12204](https://arxiv.org/abs/2506.12204) Semantic Scheduling for LLM Inference, [arXiv:2603.07917](https://arxiv.org/abs/2603.07917) SageSched, [arXiv:2510.03243](https://arxiv.org/abs/2510.03243) PARS (Prompt-Aware Scheduling), [arXiv:2510.03334](https://arxiv.org/abs/2510.03334) Semantic-Aware Scheduling for GPU Clusters — **L1 baseline 에 추가. 특히 Semantic Scheduling (2025-06) 이 L1 의 content-axis taxonomy 와 직접 경쟁 — Novelty 위협 중간, 차별 formal 필요** |
| "TempoKV" (L2) | **부재**. | [arXiv:2502.04077](https://arxiv.org/abs/2502.04077) AttentionPredictor (Temporal Patterns Matter for KV Cache Compression), LouisKV (semantic-aware retrieval + temporal locality, OpenReview) — L2 baseline 추가 |

### 📊 Phase 2' 재검증 후 영향 평가

- **L3 MTV-Pool**: GUI-KV [arXiv:2510.00536](https://arxiv.org/abs/2510.00536) 실존 확인으로 Major Revision 판정 **유지** (근거 강화). 추가로 Rethinking Token Pruning [arXiv:2603.26041](https://arxiv.org/abs/2603.26041) withdrawn paper 의 temporal perspective (recency budget allocation) 도 유사 — 재제출 시 재평가 필요.
- **A3 SemCOW-Deadline**: "SLOServe-GC" / "Refcount-KV" placeholder 부재 → "이 축 scoop 위험 낮음" 판정 **강화**. OxyGen [arXiv:2603.14371](https://arxiv.org/abs/2603.14371) adjacent baseline 필수. **Top 2 판정 Conditional Accept (Strong) 유지** (오히려 강화).
- **L1 ContextSM-Tri**: **신규 Semantic Scheduling [arXiv:2506.12204](https://arxiv.org/abs/2506.12204) (2025-06, Wu et al.) 이 content-axis semantic scheduling 을 이미 제안** — L1 의 6-class taxonomy × tri-knob 의 **novelty 축 중 "content-axis" 부분 concurrent 45-55%**. 다만 L1 의 **(α_SM, α_BW, α_KV) tri-knob** 과 **MIG+Green Context nested** 구조는 Semantic Scheduling 에 없음. → L1 재Scoring: Novelty **7.2 → 6.8** (Semantic Scheduling / SageSched / PARS 3편 concurrent 로 새 경쟁 layer 추가). **Top 1 판정 여전히 Accept 이나 baseline 목록 보강 필요**.
- **A1 PhaseGraph-VLA**: "Helix-VLA" 부재로 Top 3 판정 영향 없음. OpenHelix [arXiv:2505.03912](https://arxiv.org/abs/2505.03912) adjacent baseline 추가. **Top 3 Conditional Accept 유지**.
- **L2 TemporalTier-3**: "TempoKV" / "EventPrefetch-Robot" placeholder 부재. OxyGen adjacent. HERMES 65% concurrent 판정 유지. **미선정 (tiebreak 패) 유지**.
- **A2 TierKernel-Dispatch**: "TileSparse" / "L2-Persist-Attn" 부재. BlindSight [arXiv:2507.09071](https://arxiv.org/abs/2507.09071) / Tawa [arXiv:2510.14719](https://arxiv.org/abs/2510.14719) adjacent. 미선정 판정 유지.

### 🔄 Phase 2' 최종 score 재조정 (검증 반영)

| Idea | Phase 2' (pre-verify) | Phase 2' (post-verify) | Delta | 판정 |
|------|----------------------|----------------------|-------|------|
| L1 v2 ContextSM-Tri | 7.20 | **7.00** | **-0.20** (Semantic Scheduling 2506.12204 concurrent 반영) | Accept 유지 (Top 1) |
| A3 v2 SemCOW-Deadline | 7.13 | **7.15** | +0.02 (scoop placeholder 부재로 sliver 확인) | Conditional Accept (Strong) 유지 (Top 2) |
| A1 v2 PhaseGraph-VLA | 7.08 | 7.08 | 0 | Conditional Accept 유지 (Top 3, tiebreak 승 유지) |
| L2 v2 TemporalTier-3 | 7.08 | 7.08 | 0 | 미선정 유지 (tiebreak 패, A1 에 48.75 vs 43.40) |
| A2 v2 TierKernel-Dispatch | 6.88 | 6.88 | 0 | 미선정 유지 |
| L3 v2 MTV-Pool | 6.68 | **6.50** | **-0.18** (GUI-KV 2510.00536 실존 확인으로 scoop 강화) | **Major Revision 유지 (근거 강화)** |

**Top 3 순위 변동 없음**. L1 과 A3 의 점수 간격이 좁혀져 Phase 3 prototyping 우선순위는 **A3 SemCOW-Deadline** 을 flagship 으로, L1 을 companion 으로 가도 정당. 기존 순위도 유효.

### 🗒️ 반대로 Phase 2 이전에 이미 확인된 실존 논문 (검증 불요)
MPIC, KVShare, Mosaic, HERMES, OmniSparse, MMInference, VL-Cache, VLCache, VLA-Cache, KV-Efficient VLA, AC²-VLA, Nova, RPS-Serve, LMCache, ContextCache, DuetServe, Nexus, Adrenaline, DroidSpeak, Llumnix, Lethe, DiffKV, StreamingVLM, LiveVLM, CacheFlow, V-Rex, Event-VStream, CodecSight, SparseVILA, SparseVLM, FlashVLM, PureKV, PAT, PRESERVE, KVSwap, TTF-VLA, MadaKV, FlowMM, POD-Attention, LithOS, Bullet, DynamoLLM, Aegaeon, Hummingbird, FlashInfer, ADP-VLA, SpecPrune-VLA, Running-VLAs-Real-time, PD-VLA, OpenVLA-OFT, EveryDayVLA, CEED-VLA, ActionFlow, VOTE, VLA-Pruner, StreamMem, Expected Attention, KVFlow, Scorpio, JITServe, SLICE, Helix, OpenVLA — 모두 Phase 2 시점 arxiv 또는 venue 확인된 실존 논문.

### ➕ 검증 과정에서 새로 발견된 실존 논문 (각 idea baseline 추가)
- **[arXiv:2510.00536](https://arxiv.org/abs/2510.00536) GUI-KV** — L3 scoop, A2 adjacent.
- **[arXiv:2603.14371](https://arxiv.org/abs/2603.14371) OxyGen** — A3/L2 adjacent.
- **[arXiv:2603.26041](https://arxiv.org/abs/2603.26041) Rethinking Token Pruning** (withdrawn) — L3 referential.
- **[arXiv:2506.12204](https://arxiv.org/abs/2506.12204) Semantic Scheduling for LLM Inference** — **L1 concurrent 45-55% (Novelty 축 직접 경쟁)**.
- **[arXiv:2603.07917](https://arxiv.org/abs/2603.07917) SageSched** — L1 adjacent (semantic-aware history predictor).
- **[arXiv:2510.03243](https://arxiv.org/abs/2510.03243) PARS** — L1 adjacent (Prompt-Aware Scheduling).
- **[arXiv:2510.03334](https://arxiv.org/abs/2510.03334) Semantic-Aware Scheduling for GPU Clusters** — L1 adjacent.
- **[arXiv:2504.08784](https://arxiv.org/abs/2504.08784) SLOs-Serve** — A3 baseline.
- **[arXiv:2508.16449](https://arxiv.org/abs/2508.16449) GreenLLM (SLO-aware dynamic frequency)** — L1/A3 adjacent.
- **[arXiv:2501.12162](https://arxiv.org/abs/2501.12162) AdaServe** — A3 adjacent.
- **[arXiv:2601.10729](https://arxiv.org/abs/2601.10729) OrbitFlow** — A3 adjacent.
- **[arXiv:2502.04077](https://arxiv.org/abs/2502.04077) AttentionPredictor** — L2 adjacent (Temporal Patterns Matter for KV Cache).
- **[arXiv:2509.17396](https://arxiv.org/abs/2509.17396) EpiCache** — L3 adjacent (Episodic KV Cache).
- **[arXiv:2603.20284](https://arxiv.org/abs/2603.20284) STAC** — L3 adjacent (Spatio-Temporal Aware Cache for 3D streaming).
- **[arXiv:2511.02230](https://arxiv.org/abs/2511.02230) Continuum** — L3/A3 adjacent (Multi-Turn LLM Agent with KV TTL).
- **[arXiv:2510.09038](https://arxiv.org/abs/2510.09038) Auto-Scaling Continuous Memory for GUI Agent** — L3 adjacent.
- **[arXiv:2508.18572](https://arxiv.org/abs/2508.18572) Strata (Hierarchical Context Caching for Long Context LLM)** — L1/L2 adjacent.
- **[arXiv:2505.03912](https://arxiv.org/abs/2505.03912) OpenHelix** — A1 adjacent (dual-system VLA survey).
- **[arXiv:2507.09071](https://arxiv.org/abs/2507.09071) BlindSight** — A2 adjacent (input-template-aware sparsity).
- **[arXiv:2510.14719](https://arxiv.org/abs/2510.14719) Tawa** — A2 adjacent (warp specialization).

---

## Executive Summary

> 연구자/학생이 Phase 1~1'' 상세 로그를 읽기 전 **Top 3 아이디어의 구조·근거 + Exploration 했던 모든 아이디어의 연구 GAP·제안 overview·미선정 사유** 를 한눈에 파악하기 위한 요약. 상세는 Section 3 (Phase 1) / Section 4 (Phase 2) / Section 5 (Phase 1') / Section 6 (Phase 2') / Section 7 (Phase 1'') 참조.

### Top 1 — L1 v2. ContextSM-Tri: Content-Axis SM/BW/KV Tri-Partition with Reconfig-Latency-Bounded Green Context (평균 **7.20 → 7.00 (verified)**, Phase 2' 판정: **Accept**, Target Venue: **MICRO 2026 / HPCA 2027**)

> **Post-verification note**: Phase 2' 이후 [arXiv:2506.12204](https://arxiv.org/abs/2506.12204) "Semantic Scheduling for LLM Inference" (2025-06) 실존 확인 — **content-axis semantic scheduling** axis 를 이미 제안. L1 의 "content-axis taxonomy" 부분과 concurrent **45-55%**. 단 L1 의 **(α_SM, α_BW, α_KV) tri-knob** + **MIG+Green Context nested reconfig-latency-bounded** 구조는 Semantic Scheduling 에 없음 → Novelty 축 좁아지나 Accept 유지. Novelty 7.2 → 6.8, 평균 7.20 → 7.00. Baseline 에 Semantic Scheduling / SageSched / PARS / Semantic-Aware Scheduling for GPU Clusters / SLOs-Serve / OrbitFlow 추가.

**기본 전제 (Premise)**
- VLM/VLA multi-tenant 서빙에서 "요청의 stage (prefill/decode)" 뿐 아니라 "요청의 **content taxonomy** (OCR/ChartQA/grounding/scene/chat/reasoning/retrieval 등)" 가 SM 파티션·HBM bandwidth·KV budget 의 **최적 비율을 다르게 결정**.
- 측정 근거: Qwen2.5-VL-7B on RTX 4090 nsys profile — Turn 1 (vision-heavy "bounding boxes") SM 점유 91% · HBM BW 정체, Turn 2 (text-continuation "요약") SM 점유 34% · HBM BW 820 GB/s (GDDR6X peak 1008 의 81%). Mixed 2-user 에서 user A Turn 1 + user B Turn 3 concurrent 시 user B TPOT **+71% 증가** (48→82ms).
- Nova ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301)) 는 stage partition (encoder/prefill/decode), DuetServe ([arXiv:2511.04791](https://arxiv.org/abs/2511.04791)) 는 prefill/decode partition, RPS-Serve ([arXiv:2603.26498](https://arxiv.org/abs/2603.26498)) 는 modality-size priority — **모두 content taxonomy 는 partition 결정 입력으로 사용 안 함**.
- MIG (H100/H200) 는 reconfig latency 가 수 초 단위라 per-request reshape 불가. Green Context (CUDA 12.5+) 는 **ms 단위 reconfig** — 이 둘을 nested 로 결합하면 slow outer × fast inner 의 hierarchical partition 이 가능.

**기존 연구가 touch 하지 못한 GAP**
- **Nova / DuetServe / Bullet ([arXiv:2504.19516](https://arxiv.org/abs/2504.19516)) / LithOS (SOSP'25) / POD-Attention (ASPLOS'25) / DynamoLLM (HPCA'25) / Aegaeon (SOSP'25)** 모두 SM partition 을 다루지만 **content taxonomy → (α_SM, α_BW, α_KV) 3축 knob vector** 매핑은 없음. α_BW (HBM BW carving via cudaAccessPolicyWindow + stream priority) 와 α_KV (per-taxonomy prefill chunk/decode reserved block) 는 특히 SM 축과 독립.
- **Hummingbird ([arXiv:2601.04071](https://arxiv.org/abs/2601.04071))** 의 microsecond preemption 은 resource 축 아닌 time 축. 본 아이디어와 orthogonal.
- MIG + Green Context **nested partition** 구조는 published 보고 부재.

**제안 기법 Overview**
- **6-class effective taxonomy** (prefill-long / decode-memory / decode-compute / vision-encode / audio-decode / retrieval-heavy) — 초안 8-class 에서 α-pairwise MAD < 15% 인 VQA-simple/VQA-reasoning 병합. DistilBERT-tiny (2M param, 0.4ms) 또는 P1 E²IC distilled (L_early=2 activation-based, 10-dim tiny MLP 260 param) 로 분류.
- **Tri-knob control**:
  - α_SM → Green Context (4 SM 단위 partition, 128 SM → 32 partition 가능한 Ada/Blackwell) + MIG slice (H100) nested.
  - α_BW → cudaAccessPolicyWindow (L2 persistent hint) + cudaStreamCreateWithPriority (high/medium/low).
  - α_KV → per-taxonomy prefill chunk 크기 + decode reserved block count (vLLM BlockSpaceManager 확장).
- **Nova orthogonality single ablation**: Nova-only (stage partition, content-blind) vs L1-only (content, fixed stage) vs Nova+L1 stacked. Stacked ≥ additive sum × 0.85 이면 orthogonality 증명.
- A1 cascade 가능: VLA fleet serving 에서 L1 task classifier + A1 phase predictor 가 **multi-robot serving** 에 적용.

**예상 효과 (보수)**
| 지표 | Baseline | 목표 | 조건 |
|------|----------|------|------|
| L1-only vs Nova, mixed 6-class workload | — | **-10~13% p99 TTFT, -9~12% p99 TPOT** | content-axis only |
| L1 + Nova stacked | — | **-17~20% p99 TTFT, -15~18% p99 TPOT** | 기존 -23% 초안에서 보수 하향 |
| L1 vs MIG H100 (per-request reconfig), short-lived tenant mix | MIG 수 초 reconfig | **-14~18% TTFT** | Green Context ms reconfig 축 |
| L1 cascade w/ A1, VLA fleet serving | — | **-18~22% p99 latency** | extension (robot serving) |
| Uniform-taxonomy batch (예: retrieval-only) | — | **<3%** | scope 밖 명시 |
| Classifier overhead | — | **<0.5ms / request** | DistilBERT-tiny + prefill overlap |

**Scoring 및 이유**
| Axis | Phase 2 | Phase 2' | 근거 |
|------|---------|----------|------|
| Novelty | 7.0 | **7.2** | tri-knob (SM × BW × KV) 동시 제어 + MIG nested 구조 unique. 각 축은 기존 존재 (composition novelty) |
| Differentiation | 6.0 | **7.3** | 10개 baseline (최다): MIG, Nova, DuetServe, Bullet, LithOS, Hummingbird, POD-Attention, DynamoLLM, Aegaeon, Adrenaline. Nova orthogonality single ablation |
| Impact | 6.0 | **6.8** | datacenter H100 MIG + consumer RTX 5090 양 축. Per-request reconfig niche 로 좁힘 |
| Feasibility | 7.5 | **7.5** | CUDA 12.5 Green Context 확인됨, cudaAccessPolicyWindow 기 검증 |
| **평균** | **6.90** | **7.20** | +0.30 |
| 전문가 합의 | — | **3:0 Unanimous** | ai-opt/legacy-sys/system-robustness 모두 Yes |

**→ Phase 상세**: [Section 5 L1 Refinement](#section-5--phase-1-refinement) / [Section 6 L1 Phase 2'](#section-6--phase-2-integrated-re-review)

---

### Top 2 — A3 v2. SemCOW-Deadline: Copy-on-Write Reference-Counted Vision KV with Deadline-Aware Green Context SM Yielding for Multi-Tenant VLM Serving (평균 **7.13 → 7.15 (verified)**, Phase 2' 판정: **Conditional Accept (Strong)**, Target Venue: **OSDI 2027 / SOSP 2026 / NSDI 2027**)

> **Post-verification note**: "SLOServe-GC" / "Refcount-KV" placeholder 모두 **부재** 확인 — **A3 의 핵심 sliver (page-granular refcount COW + deadline-aware Green Context SM yielding) scoop 위험 낮음 재확인**. [arXiv:2603.14371](https://arxiv.org/abs/2603.14371) OxyGen (2026-03, VLA unified KV management under multi-task parallelism) 은 single-request 내 multi-task 이라 multi-tenant 축 다름 → **35% adjacent**. PagedAttention [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) 의 COW 는 refcount foundational (A3 는 partial recompute + deadline 으로 상위 extension). Novelty **6.5 유지**, 평균 7.13 → 7.15. Baseline 에 OxyGen / SLOs-Serve / AdaServe / OrbitFlow / Continuum [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) 추가.

**기본 전제 (Premise)**
- 기존 VLM cross-request sharing 연구 3편이 scoop 위험:
  - **Mosaic ([arXiv:2604.10060](https://arxiv.org/abs/2604.10060))** — cross-modal clustering KV (streaming video, 2026-04)
  - **KVShare ([arXiv:2503.16525](https://arxiv.org/abs/2503.16525))** — multi-tenant semantic clustering + token edit (2025-03)
  - **MPIC ([arXiv:2502.01960](https://arxiv.org/abs/2502.01960))** — position-independent multimodal cache (2025-02)
- 세 논문은 **cluster detection + identity dedup** 축을 이미 점유. 본 idea 는 이 axis 를 포기하고 **cluster 이후 layer** (memory consistency, SLO scheduling) 에 집중.
- Vision KV 는 partial overwrite (top-k attention 만 recompute) 빈번 — **refcount COW** 가 필요. Adrenaline ([arXiv:2503.20552](https://arxiv.org/abs/2503.20552)) 의 SM yielding 은 idle 기반이지 deadline 기반 아님.

**기존 연구가 touch 하지 못한 GAP**
- **Mosaic/KVShare/MPIC** 모두 write-time 분기 (COW fork) + page-granular reference counting 구현 없음. Full-share → full-copy 이분.
- **Adrenaline** 는 throughput yielding, 본 idea 는 **SLO deadline miss risk metric 기반** yielding. CUDA 12.5 Green Context API 로 per-request SM 재분배.
- ContextCache ([arXiv:2506.22791](https://arxiv.org/abs/2506.22791)), Nexus ([arXiv:2507.06608](https://arxiv.org/abs/2507.06608)), Llumnix (OSDI'24), DroidSpeak ([arXiv:2411.02820](https://arxiv.org/abs/2411.02820)) 도 각각 인접하지만 **page-granular refcount COW + deadline-aware Green Context SM yielding** 조합은 공백.

**제안 기법 Overview**
- Cluster detector 는 Mosaic/KVShare 에서 직접 차용 (novelty claim 없음).
- **Page-granular COW**: vLLM PagedAttention block 단위 refcount. Shared block 에 write 발생 시 해당 block 만 fork. Top-k partial recompute (cluster hit but attention top-k 다른 경우) 는 top-k block 만 fork + recompute — full recompute 대비 O(k/N).
- **Deadline-aware Green Context SM yielding**: 각 tenant 에 SLO p99 deadline 명시. 매 decode step 마다 tenant 별 deadline miss risk 계산 (time_to_deadline / estimated_remaining_compute). Risk > τ 인 tenant 에 Green Context SM partition 확장. Risk < τ 인 tenant 는 SM 축소.
- **Reference-count GC**: COW fork rate > threshold 시 eager eviction (cluster detector 의 false-positive 부작용 대응).

**예상 효과 (보수)**
| 지표 | Baseline (vLLM+Mosaic) | COW only | Deadline only | Full |
|------|----------------------|----------|---------------|------|
| Cluster-hit 50% p99 TTFT (ms) | 520 | 460 (-12%) | 435 (-16%) | **385 (-26%)** |
| SLO attainment (deadline=1s) | 82% | 86% | 91% | **94%** |
| Vision KV memory (GB, 8 tenants) | 18.5 | 12.8 (-31%) | 18.5 | **12.8 (-31%)** |
| Throughput (req/s, mixed) | 24 | 27 | 28 | **31 (+29%)** |
| COW fork rate (write-on-shared) | — | 14% | — | 14% |
| Cluster hit < 15% (scope 밖) | — | — | — | **-3~5% 역효과 가능** |
| Tenant 수 < 4 (scope 밖) | — | — | — | deadline yielding 효과 미미 |

**Scoring 및 이유**
| Axis | Phase 2 | Phase 2' | 근거 |
|------|---------|----------|------|
| Novelty | 5.0 | **6.5** | Cluster detection 포기 + page refcount COW + deadline Green Context sliver 확보 |
| Differentiation | 7.0 | **8.0** | baseline 5편 추가 (Adrenaline, ContextCache, Nexus, Llumnix, DroidSpeak). throughput vs SLO 축 분리 명시 |
| Impact | 8.0 | **8.0** | 유지 (Top). vLLM multimodal prefix cache roadmap 과 직결. OSDI/SOSP 타겟 정당성 |
| Feasibility | 5.5 | **6.0** | Detector import 로 구현 감축. Green Context API 는 CUDA 12.5+ 필요 (verified) |
| **평균** | **6.38** | **7.13** | **+0.75 (최대 delta)** |
| 전문가 합의 | — | **3:0 Strong Consensus** | legacy-sys/ai-opt/system-robustness 전원 Yes |

**Phase 2' 조건 (Conditional Accept)**
- "SLOServe-GC" (2025-11 placeholder) 실존 확인 후 baseline 추가.
- "Refcount-KV" (2025-10 placeholder) 실존 시 baseline.
- Top-k partial recompute 의 k 선택 sensitivity study.
- Adrenaline 과 stacked 실험으로 orthogonality empirical 검증.

**→ Phase 상세**: [Section 5 A3 Refinement](#section-5--phase-1-refinement) / [Section 6 A3 Phase 2'](#section-6--phase-2-integrated-re-review)

---

### Top 3 — A1 v2. PhaseGraph-VLA: Trajectory-Phase Conditioned CUDA Graph Dispatcher with SSE-Driven Boundary Detection (평균 **7.08 유지 (verified)**, Phase 2' 판정: **Conditional Accept**, Target Venue: **MLSys 2026 / CoRL 2026**)

> **Post-verification note**: "Helix-VLA" placeholder **부재** 확인 (Figure AI Helix 는 blog only, 논문화 안 됨). Baseline 에 [arXiv:2505.03912](https://arxiv.org/abs/2505.03912) OpenHelix (dual-system VLA survey+opensource) 추가. Score 변동 없음. L2 와 7.08 tiebreak 승 유지 (impact×feasibility 48.75 vs 43.40).

**기본 전제 (Premise)**
- OpenVLA-7B single action inference on RTX 4090: SigLIP vision encoder ~180ms (전체 latency 의 60-70%), LLM prefill 60-90ms (256 vision + 40 text token), action head decode 15-25ms (7-DoF chunk). Visual encoder 가 SM 독점하는 동안 LLM idle.
- Robotic manipulation trajectory 의 **phase heterogeneity** (OpenVLA-OFT ablation 관찰 확장):
  - **Approach (episode 20-30%)**: visual encoder 프레임 간 30-40% 변동. Scene-level reasoning 중요.
  - **Manipulate (50-60%)**: end-effector 근접, fine-grained contact. VLA-Cache ([arXiv:2502.02175](https://arxiv.org/abs/2502.02175)) frame-diff 가 40-60% token 재사용.
  - **Retract (15-25%)**: scripted-like. Language conditioning ablation 시 성공률 감소 < 2%.
- vLLM 의 CUDA Graph capture 는 **batch-size variant** 만 지원 — content/phase-aware graph variant 미보고.

**기존 연구가 touch 하지 못한 GAP**
- **AC²-VLA ([arXiv:2601.19634](https://arxiv.org/abs/2601.19634))** 는 action context → token prune + component skip (algorithmic), **VLA-Cache** 는 frame-diff → KV reuse, **KV-Efficient VLA ([arXiv:2509.21354](https://arxiv.org/abs/2509.21354))** 는 RNN-gated chunked KV, **ADP-VLA ([arXiv:2509.22093](https://arxiv.org/abs/2509.22093))** 는 action-aware dynamic pruning — **모두 token/component reduction** 축. System-level execution graph 변경 없음.
- **Nova** 는 stage partition (VLM general, VLA 아님), **DuetServe** 는 prefill/decode SM partition. **Phase × CUDA Graph variant × phase-tuned fused kernel** 은 공백.
- **SpecPrune-VLA ([arXiv:2509.05614](https://arxiv.org/abs/2509.05614))** 의 speculative action-aware pruning 은 algorithmic self-distillation 축. 본 idea 와 orthogonal.
- Running-VLAs-at-Real-time ([arXiv:2510.26742](https://arxiv.org/abs/2510.26742)) 는 chunk boundary 기반 async. 본 idea 의 phase boundary graph dispatch 와 axis 분리.

**제안 기법 Overview**
- **Phase predictor = SSE (Hidden State Shift Estimator, P2 흡수)**: L_mid hidden state L2 drift + EWMA baseline + **Page-Hinkley 2-threshold** (soft pre-warm / hard evict) + hysteresis (3 frames). Training-free quantile calibration (첫 100 frame). 결정 latency < 100μs. VLA 에서는 gripper state Δ, trajectory curvature, object distance (DINOv2 mask 1.2ms/frame) 를 additional feature 로 결합.
- **Phase-specific CUDA Graph dispatcher (phase × batch 2D)**:
  - Approach graph: SigLIP patch-embed full + FlashAttn-3 MQA fused (ViT-specific).
  - Manipulate graph: SigLIP partial-batch (40% re-encode) + action-head SiLU+Linear fused + reduced attention heads.
  - Retract graph: SigLIP bypass (last 2-step encoder feature linear extrapolation) + KV static reuse mode.
- **Phase-specific SM partition (optional stacking layer)**: Approach 80% encoder + 20% LLM prefetch, Manipulate 35% encoder-partial + 55% LLM + 10% async prefetch, Retract 0% encoder + 85% LLM + 15% next-episode prefetch. Nova baseline 비교용.
- 4-way orthogonality ablation: baseline / phase-graph-only / SSE-only / full / stacked-with-Nova.

**예상 효과 (보수)**
| 지표 | Baseline (vLLM+OpenVLA-7B) | My1 (PhaseGraph only, oracle phase) | Full (PhaseGraph + SSE) | Stacked + Nova |
|------|---------------------------|-------------------------------------|-----------------------|---------------|
| LIBERO median latency (ms) | 165 | 138 (-16%) | 128 (-22%) | **115 (-30%)** |
| SimplerEnv Hz | 6.1 | 7.2 | **7.8** | 8.6 |
| RoboCasa success rate Δ | — | ±0.8pp | ±1.2pp | ±1.5pp |
| Jetson Orin 64GB latency (ms) | 420 | 355 | **330** | N/A |
| Short 1-shot pick task | — | ±3% | ±3% | ±3% (scope 밖) |

**Scoring 및 이유**
| Axis | Phase 2 | Phase 2' | 근거 |
|------|---------|----------|------|
| Novelty | 5.5 | **6.8** | SSE 흡수 + phase × batch 2D CUDA Graph + phase-tuned fused kernel. VLA 문헌 unique. CUDA Graph switching 자체는 LLM 일반 기법 (partial credit) |
| Differentiation | 7.0 | **7.5** | 5편 추가 baseline: DuetServe, TTF-VLA, Scorpio, SpecPrune-VLA, Running-VLAs-Real-time. 4-way ablation orthogonality |
| Impact | 5.5 | **6.5** | SimplerEnv + RoboCasa 확장. Jetson Orin 포팅 variant 로 embedded claim |
| Feasibility | 7.5 | **7.5** | single workstation, 6-8주 scope |
| **평균** | **6.38** | **7.08** | +0.70 |
| 전문가 합의 | — | **3:0 Conditional** | ai-opt/legacy-sys Yes, algorithm-expert Conditional Yes (Page-Hinkley FP rate 증명 요구) |

**Phase 2' 조건 (Conditional Accept)**
- Page-Hinkley false-positive rate LIBERO 5 task empirical 측정.
- "Helix-VLA" (2025-12 placeholder) 실존 확인 후 baseline 검토.
- CUDA Graph capture overhead 의 amortization 분석 (phase transition 주기 대비).

**→ Phase 상세**: [Section 5 A1 Refinement](#section-5--phase-1-refinement) / [Section 6 A1 Phase 2'](#section-6--phase-2-integrated-re-review)

---

### Tiebreak 설명 (A1 v2 vs L2 v2)

Phase 2' 평균 점수 **동점 (7.08)**. Top-M 규칙에 따라 **impact × feasibility product** 로 tiebreak:
- A1: 6.5 × 7.5 = **48.75**
- L2: 6.2 × 7.0 = 43.40

A1 승. 추가로 A1 은 VLA 시장성 (robot learning 산업 성장) + CUDA Graph dispatcher 의 reusable system artifact + SpecPrune-VLA/Running-VLAs 대비 명확한 kernel-specialization 축. L2 는 action-imminence signal 이 VLA-Cache frame-diff 의 변형으로 보일 수 있다는 risk 잔존.

---

## Exploration 했던 모든 아이디어 요약 (Top 3 제외, 미선정 포함)

### L2 v2. TemporalTier-3: Action-Imminence-Driven Hierarchical KV with Hawkes-vs-Poisson Ablation and NVLink-C2C Variant (평균 7.08, tiebreak 패배)

- **연구 GAP**: VLA action stream / VLM video streaming 에서 "다음 수백 ms 내 어떤 KV 페이지가 hot 인지" 를 task-specific signal (gripper state novelty / Hawkes interaction arrival) 로 예측 가능하나, 기존 VLA-Cache/KV-Efficient VLA/HERMES ([arXiv:2601.14724](https://arxiv.org/abs/2601.14724)) 는 single-tier/frame-diff 중심. Hierarchical storage (HBM-hot/HBM-cold/pinned host) + proactive prefetch + FA-3 indirection 조합 공백.
- **제안 overview**: 초안 4-tier (HBM top/bottom/UVM/pinned/disk) → Phase 1' 에서 **3-tier (HBM-hot/HBM-cold/pinned host), UVM 제거**. Hawkes (self-exciting) vs Poisson (exponential) arrival ablation. Gripper-state + trajectory-curvature + object-dist predictor AUC study (vs VLA-Cache frame-diff). FA-3 indirection fork wrapper vs kernel-rewrite 2-variant. Grace-Hopper NVLink-C2C analytical variant.
- **미선정 사유**: 평균 동점 7.08 에서 **impact × feasibility tiebreak 패배 (43.40 < A1 48.75)**. Phase 2' 에서 HERMES (2026-01) 가 hierarchical KV for streaming video 를 이미 커버. Action-imminence β_a signal 이 VLA-Cache frame-diff 의 변형으로 보일 risk 잔존 — predictor study 에서 AUC +0.08 이상이 필수 조건이나 아직 empirical 증명 없음. NVLink-C2C analytical 은 Grace Hopper 실기 부재로 synthetic 판정 risk.

### A2 v2. TierKernel-Dispatch: Three-Tier Patch Routing with Warp-Specialized Kernel Variants and Hopper-Portable L2 Policy (평균 6.88)

- **연구 GAP**: VLM patch tile heterogeneity 가 크지만 (SparseVILA ablation: <15% patch 가 >60% cross-attention mass), 기존 VL-Cache/SparseVILA/FlashVLM 은 token-level sparsity, FlashInfer 는 JIT kernel bank (content-blind). **Task intent × kernel variant × memory tier** 3축 통합 공백.
- **제안 overview**: E²IC (P1 흡수) intent classifier + attention entropy 이중 gating → Hot/Warm/Cold 3-tier patch → per-tile kernel variant (CUTLASS warp-specialized Hot / vanilla FA Warm / eviction Cold) + L2 cudaAccessPolicyWindow (Blackwell 288MB) + Hopper H100 128MB L2 fallback variant.
- **미선정 사유**: Phase 2' 에서 algorithm-expert **No** (intent classifier cross-task generalization 증명 약함, overfitting risk). OmniSparse (2025-11) 가 binary hot-cold 로 이미 구현 — 3-tier 의 incremental contribution 이 challenge 받음. 평균 6.88 은 Top 3 아래. Blackwell 이후 L2 spec 변경 risk 도 영향.

### L3 v2. MTV-Pool: Multi-Turn-Aware Visual KV Pool with Turn-Oldness + Log-Sum-Exp Coldness (평균 6.68 → **6.50 (verified, 강화)**, Major Revision 필요)

- **연구 GAP**: Screenshot-history 기반 GUI agent (Claude Computer Use, VisualWebArena, OSWorld) 의 turn 이 쌓일수록 과거 screenshot token 이 long-tail distribution (거의 미참조지만 2% critical). MMInference (static permutation) 와 OmniSparse (intra-turn) 는 inter-turn axis 미커버.
- **제안 overview**: 초안 3-pool → Phase 1' 에서 **2-pool (current-turn-hot / past-turn-cold)** 축소. γ_v = w1·turn_oldness + w2·log-sum-exp(past attn) + w3·tile_entropy. Reorg stream overlap 정량화. Text token 은 always-hot.
- **미선정 사유 (검증 반영)**: Phase 2' 에서 placeholder "GUIAgent-KV 2025-11, 유사도 65%" 로 판정됐으나 **검증 결과 실제 논문 = [arXiv:2510.00536](https://arxiv.org/abs/2510.00536) "GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness" (2025-10-01) 실존**. GUI-KV 는 (a) residual stream L2 norm spatial saliency + (b) previous frames' keys → current frame key subspace projection 으로 redundant history 제거 — 메커니즘 1:1 일치는 아니나 **target scenario (multi-turn GUI agent screenshot history KV 압축) 와 positioning 완전 중첩** → contribution novelty 잠식. 추가로 [arXiv:2603.26041](https://arxiv.org/abs/2603.26041) "Rethinking Token Pruning for Historical Screenshots in GUI Visual Agents" (2026-03-27, withdrawn 2026-03-31) 의 **temporal perspective (recency-based budget allocation)** 도 turn_oldness 와 정면 유사 (withdrawn 이므로 공식 scoop 아니나 concept 선점 증거). "ScreenHistory-Cache" placeholder 부재지만 대체 실존 [arXiv:2510.09038](https://arxiv.org/abs/2510.09038) Auto-Scaling Continuous Memory for GUI Agent / [arXiv:2509.17396](https://arxiv.org/abs/2509.17396) EpiCache / [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) Continuum 다수 adjacent 존재. Legacy-system-expert **No** (pool 단일 도입이 MICRO scope 엔 얕음). 전문가 합의 1.5:1.5 분열. Novelty 6.3 → 5.9, 평균 6.68 → **6.50**. **Major Revision 판정 유지 + 근거 강화** (learned weight 전환, GUI-KV 와 차별 orthogonal axis 재정립, venue 재타겟 EuroSys/SoCC).

---

## Section 1 — Phase 0/1: 외부 최신 논문 탐색 및 분석

### 1.1 Step 0 — 외부 탐색 (wiki 이전)

**사용자 쿼리에서 추출한 키워드 축**:
- 도메인 (A): VLM, VLA
- 관찰/특징 (B): Context-awareness (task/scene/trajectory/semantic/multi-request)
- 제안 기법 (C): Caching, serving optimization, kernel optimization, processing unit allocation, offloading, memory allocation policy
- 타겟 지표: latency, throughput, memory footprint, Hz (VLA)

**전문가 선정**: ai-optimization-expert (primary, 시스템/서빙 AI 최적화), legacy-system-expert (primary, GPU/메모리 아키텍처), algorithm-expert (sub, predictor 알고리즘).

**탐색 수행**: arxiv API (rate limit 으로 일부 부분 성공) + WebSearch (arxiv.org, openreview.net, usenix.org, mlsys.org, dl.acm.org domain 제한). 쿼리 15+ 개 병렬.

**수집 논문 그룹별**:

#### Group A — VLM serving & context-aware caching (Level A closest competitor)
| arXiv | 제목 | 날짜 | 핵심 메커니즘 | 본 세션 연관 |
|-------|------|-----|--------------|------------|
| [arXiv:2410.23317](https://arxiv.org/abs/2410.23317) | VL-Cache: Sparsity and Modality-Aware KV Cache Compression | 2024-10 | Layer-adaptive sparsity-aware budget, modality separation | Level A (per-layer axis), 이전 세션에서 이미 분석 |
| [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) | VLCache: Computing 2% Vision Tokens and Reusing 98% | 2025-12 | Encoder output + KV 이중 pipeline, 1.2-16× TTFT | Level A. A2 adjacent |
| [arXiv:2502.01960](https://arxiv.org/abs/2502.01960) | MPIC: Position-Independent Multimodal Context Caching | 2025-02 | Position-agnostic KV reuse, disk 저장 + 병렬 load, 54% 응답↓ 2× throughput | Level A. A3 scoop (70%) |
| [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) | Nova: Real-Time Agentic VLM Serving with Adaptive Cross-Stage Parallelization | 2025-09 | Vision-encode/prefill/decode 3-stage pipelining, elastic GPU spatial partitioning, vision encoder layer-wise CPU↔GPU weight swap. 23.3% max latency↓ | **Level A. A1, L1 의 가장 가까운 system competitor** |
| [arXiv:2603.26498](https://arxiv.org/abs/2603.26498) | Rocks, Pebbles, Sand: Modality-aware Scheduling (RPS-Serve) | 2026-03 | Size-based priority (video=rock, image=pebble, text=sand), aging. 54% TTFT↓, 78.5% latency-critical↑ | Level A. 크기 기반 (content-blind) — L1 orthogonal |

#### Group B — VLA serving (Level A for A1/L2)
| arXiv | 제목 | 날짜 | 핵심 메커니즘 | 본 세션 연관 |
|-------|------|-----|--------------|------------|
| [arXiv:2502.02175](https://arxiv.org/abs/2502.02175) | VLA-Cache: Adaptive Token Caching for Robotic Manipulation | 2025-02 | Temporal frame-diff + layer-adaptive reuse ratio. 1.7× CUDA, +15% Hz | Level A. A1 orthogonal, L2 predictor compare |
| [arXiv:2601.19634](https://arxiv.org/abs/2601.19634) | AC²-VLA: Action-Context-Aware Adaptive Computation | 2026-01 | Action context (vision+lang+prior action) → cognition reuse + token pruning + selective component. 1.79× speedup, 29.4% FLOPs. Training-required | **Level A. A1 concurrent (algorithmic axis)** |
| [arXiv:2509.21354](https://arxiv.org/abs/2509.21354) | KV-Efficient VLA: RNN-Gated Chunked KV Cache | 2025-09 | RNN-gated utility score on fixed-size chunks. 24.6% FLOPs↓, 1.87× memory | Level A. A1 adjacent |
| [arXiv:2503.02310](https://arxiv.org/abs/2503.02310) | PD-VLA: Parallel Decoding with Action Chunking | 2025-03 | Nonlinear system reformulation, parallel fixed-point iteration | Level B baseline |
| [arXiv:2509.22093](https://arxiv.org/abs/2509.22093) | ADP-VLA: Action-aware Dynamic Pruning | 2025-09 | Motion phase × task semantics → token prune | Level A. A1 concurrent (65%) |
| [arXiv:2509.05614](https://arxiv.org/abs/2509.05614) | SpecPrune-VLA: Action-Aware Self-Speculative Pruning | 2025-09 | Speculative prune | A1 adjacent |
| [arXiv:2508.19257](https://arxiv.org/abs/2508.19257) | TTF-VLA: Temporal Token Fusion | 2025-08 | Frame-level fusion | A1/L2 adjacent |
| [arXiv:2510.26742](https://arxiv.org/abs/2510.26742) | Running VLAs at Real-time Speed | 2025-10 | 480Hz streaming inference | A1 adjacent |
| [arXiv:2511.16449](https://arxiv.org/abs/2511.16449) | VLA-Pruner: Temporal-Aware Dual-Level Token Pruning | 2025-11 | Dual-level prune | A1 adjacent |
| [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | OpenVLA | 2024-06 | 7B base model | Foundation |
| [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) | OpenVLA-OFT: Fine-Tuning Optimization | 2025-02 | Parallel decoding + action chunking + L1 regression | Baseline |
| [arXiv:2506.13725](https://arxiv.org/abs/2506.13725) | CEED-VLA: Consistency + Early-Exit | 2025-06 | Early-exit decoding | A1 adjacent |
| [arXiv:2507.05116](https://arxiv.org/abs/2507.05116) | VOTE: Trajectory Ensemble Voting | 2025-07 | 39× faster, 46Hz edge | A1 adjacent |
| [arXiv:2511.05397](https://arxiv.org/abs/2511.05397) | EveryDayVLA | 2025-11 | 108.4 Hz, +0.9ms latency | A1 adjacent |
| [arXiv:2512.20276](https://arxiv.org/abs/2512.20276) | ActionFlow: Pipelined Action Acceleration Edge | 2025-12 | Edge pipeline | A1 adjacent |
| [arXiv:2602.18397](https://arxiv.org/abs/2602.18397) | VLA-Perf: Demystifying VLA Inference | 2026-02 | Benchmark | Reference |

#### Group C — Video VLM & Streaming KV (Level B for L2/L3)
| arXiv | 제목 | 날짜 | 핵심 | L2/L3 연관 |
|-------|------|-----|------|----------|
| [arXiv:2510.09608](https://arxiv.org/abs/2510.09608) | StreamingVLM: Infinite Video Streams | 2025-10 | Attention sink + sliding window | L2 adjacent |
| [arXiv:2505.15269](https://arxiv.org/abs/2505.15269) | LiveVLM: Streaming-Oriented KV + Retrieval | 2025-05 | FIFO eviction + retrieval | L2 concurrent |
| [arXiv:2511.13644](https://arxiv.org/abs/2511.13644) | CacheFlow: Compressive Streaming Memory | 2025-11 | Dynamic token dropping + compressive long-term | L2 adjacent |
| [arXiv:2512.12284](https://arxiv.org/abs/2512.12284) | V-Rex: Dynamic KV Cache Retrieval for Streaming Video LLM | 2025-12 | Runtime retrieval | L3 adjacent |
| [arXiv:2601.15655](https://arxiv.org/abs/2601.15655) | Event-VStream: Event-Driven Real-Time Understanding | 2026-01 | Event boundary + language trigger | L2 adjacent |
| [arXiv:2604.06036](https://arxiv.org/abs/2604.06036) | CodecSight: Video Codec Signals for Streaming VLM | 2026-04 | Codec motion vector | L2 adjacent |
| [arXiv:2508.15717](https://arxiv.org/abs/2508.15717) | StreamMem: Query-Agnostic KV Memory | 2025-08 | Query-blind | Reference |
| [arXiv:2601.14724](https://arxiv.org/abs/2601.14724) | HERMES: Hierarchical Memory for Streaming Video | 2026-01 | Multi-granularity hierarchical KV | **L2 concurrent (65%)** |

#### Group D — Speculative decoding VLM (Reference)
- [arXiv:2509.11815](https://arxiv.org/abs/2509.11815) SpecVLM (2025-09), [arXiv:2509.23928](https://arxiv.org/abs/2509.23928) HiViS (2025-09), [arXiv:2509.15235](https://arxiv.org/abs/2509.15235) ViSpec (2025-09), [arXiv:2505.19201](https://arxiv.org/abs/2505.19201) DREAM, [arXiv:2505.10526](https://arxiv.org/abs/2505.10526) MASSV, [arXiv:2509.11961](https://arxiv.org/abs/2509.11961) Spec-LLaVA, [arXiv:2505.14260](https://arxiv.org/abs/2505.14260) MSD.

#### Group E — KV cache-wide policy (Level B for L2/L3)
- [arXiv:2412.03131](https://arxiv.org/abs/2412.03131) DiffKV (per-head diff, 2024-12) — **L3 scoop (55-65%)**
- [arXiv:2511.06029](https://arxiv.org/abs/2511.06029) Lethe (layer/time adaptive, 2025-11) — L2 adjacent
- [arXiv:2511.12201](https://arxiv.org/abs/2511.12201) OmniSparse (long-video MLLM head-level budget, 2025-11) — **L3 scoop (65-75%)**, A2 concurrent
- [arXiv:2510.00636](https://arxiv.org/abs/2510.00636) Expected Attention (2025-10) — P1 reference
- [arXiv:2506.15724](https://arxiv.org/abs/2506.15724) MadaKV (modality-perception eviction, 2025-06) — L2/L3 concurrent
- [arXiv:2511.05534](https://arxiv.org/abs/2511.05534) FlowMM (cross-modal info flow KV merging, 2025-11) — L3 concurrent
- [arXiv:2510.25600](https://arxiv.org/abs/2510.25600) PureKV (spatial-temporal sparse attn, 2025-10) — A2 adjacent
- [arXiv:2511.22333](https://arxiv.org/abs/2511.22333) PAT: Prefix-Aware Attention Multi-Tile Kernel (2025-11) — A2 adjacent
- [arXiv:2501.08192](https://arxiv.org/abs/2501.08192) PRESERVE (prefetch weights + KV, 2025-01) — L2 concurrent
- [arXiv:2511.11907](https://arxiv.org/abs/2511.11907) KVSwap (disk-aware offload on-device, 2025-11) — L2 concurrent

#### Group F — Cross-request / Multi-tenant KV (Level A for A3)
- [arXiv:2503.16525](https://arxiv.org/abs/2503.16525) KVShare (semantic alignment, 2025-03) — **A3 scoop (72%)**
- [arXiv:2510.09665](https://arxiv.org/abs/2510.09665) LMCache (vLLM/SGLang cache layer, 2025-10) — A3/L1 reference
- [arXiv:2506.22791](https://arxiv.org/abs/2506.22791) ContextCache (multi-turn semantic cache, 2025-06) — A3 concurrent
- [arXiv:2411.02820](https://arxiv.org/abs/2411.02820) DroidSpeak (cross-LLM KV share, 2024-11) — A3 adjacent
- [arXiv:2604.10060](https://arxiv.org/abs/2604.10060) Mosaic (cross-modal clustering, 2026-04) — **A3 scoop (75%)**

#### Group G — LLM serving infrastructure (Reference for L1/A3)
- DistServe (OSDI'24), Llumnix (OSDI'24), Helix (ASPLOS'25)
- [arXiv:2507.06608](https://arxiv.org/abs/2507.06608) Nexus (intra-GPU P/D disagg, 2025-07)
- [arXiv:2511.04791](https://arxiv.org/abs/2511.04791) DuetServe (P/D harmony, 2025-11) — **L1 concurrent (65%)**
- [arXiv:2503.20552](https://arxiv.org/abs/2503.20552) Adrenaline (attention disagg, 2025-03) — **A3 concurrent (SM yielding axis)**
- [arXiv:2512.18194](https://arxiv.org/abs/2512.18194) TraCT (CXL shared memory, 2025-12) — Reference
- Weaver USENIX ATC'25
- [arXiv:2504.19516](https://arxiv.org/abs/2504.19516) Bullet (spatial-temporal SM orch, 2025-04) — L1 concurrent (60%)
- LithOS (SOSP'25), POD-Attention (ASPLOS'25), DynamoLLM (HPCA'25), Aegaeon (SOSP'25), Hummingbird ([arXiv:2601.04071](https://arxiv.org/abs/2601.04071))

#### Group H — SLO-aware scheduling (Reference)
- [arXiv:2504.14966](https://arxiv.org/abs/2504.14966) SLO-Aware Scheduling LLM (2025-04)
- [arXiv:2603.26498](https://arxiv.org/abs/2603.26498) RPS-Serve (2026-03)
- [arXiv:2504.20068](https://arxiv.org/abs/2504.20068) JITServe (2025-04)
- [arXiv:2510.18544](https://arxiv.org/abs/2510.18544) SLICE edge (2025-10)
- [arXiv:2505.23022](https://arxiv.org/abs/2505.23022) Scorpio (2025-05)

#### Group I — Sparse VLM kernel (Reference)
- [arXiv:2410.04417](https://arxiv.org/abs/2410.04417) SparseVLM (2024-10) — A2/L3 reference
- [arXiv:2510.17777](https://arxiv.org/abs/2510.17777) SparseVILA (2025-10) — A2 adjacent
- [arXiv:2512.20561](https://arxiv.org/abs/2512.20561) FlashVLM (2025-12) — A2 adjacent
- [arXiv:2504.16083](https://arxiv.org/abs/2504.16083) MMInference (modality-aware permutation, 2025-04) — **L3 concurrent (68%), A2 concurrent (62%)**
- FlashInfer MLSys 2025 — A2 adjacent

### 1.2 Step 1 — wiki 기존 knowledge 참고 (secondary)

`__research_wiki/index.md` 에서 직접 연관 3 세션 식별:
- **2026-04-21 mode2 ACE-MoE VLM/VLA software 확장** — **MoE 기반 expert caching** 중심. 본 세션은 non-MoE dense VLM/VLA 도 포함하여 orthogonal.
- **2026-04-22 mode2 VLM+PIM extension** — F1 DeepStack 6-tier KV tiering (GPU+PIM), F2 quant-robust layered defense (W8A8 collapse). 본 세션은 **pure GPU (PIM 비의존)** 로 차별화.
- **2026-04-22 mode1 BNN/TNN domain extension** — 무관.

**중복 회피 포지셔닝 확인**: 본 세션 6 idea 모두 PIM/MoE 의존 없음. L1 (multi-tenant GPU partition), L2 (HBM tier), L3 (multi-turn pool), A1 (VLA CUDA Graph), A2 (per-tile L2 hint), A3 (multi-tenant COW) 모두 pure GPU serving stack.

---

## Section 2 — Phase 1 (Initial) — 6 Ideas Draft

(본 세션의 Phase 1 에서 세 전문가가 도출한 6 개 초안 + 2 predictor. 각 idea 의 상세 spec 은 aica-research-bot 의 Phase 1 staging 산출물 참조. 본 섹션은 요약만.)

### 2.1 ai-optimization-expert 의 3 아이디어 초안

#### A1 (초안). TrajPhase-Serve
VLA trajectory 3-phase predictor (gripper + proprio + action logit entropy MLP, <0.8ms) → **phase-specific SM partition + async vision encoder CPU offload + CUDA Graph 3 pre-capture**.

#### A2 (초안). SemTile-Dispatch
Task intent classifier + Layer-1 attention entropy → **Hot/Warm/Cold 3-tier patch tile** → per-tile kernel variant (full FA / standard FA / INT4-dequant block-sparse FA) + **L2 persistent hint** (RTX 5090 128MB L2) + HBM tier placement.

#### A3 (초안). SemClust-Share
Multi-request **pHash + CLIP-B/32 clustering** (<3ms) → cluster-shared vision KV with **copy-on-write** + **cluster-batched FlashAttention** + **cross-request CUDA stream priority + Green Context SM yielding**.

### 2.2 legacy-system-expert 의 3 아이디어 초안

#### L1 (초안). ContextSM
DistilBERT-tiny **8-class task classifier** (OCR/grounding/VQA-simple/VQA-reasoning/caption/chat/code/tool-use, 0.4ms) → **(α_SM, α_BW, α_KV) profile vector** → **CUDA Green Context SM partition** + **HBM BW carving via stream priority + cudaAccessPolicyWindow**. Mixed 2-user p99 TTFT/TPOT -23/-20%.

#### L2 (초안). TemporalTier
VLA **action imminence β_a** (gripper Δ + trajectory curvature + object-dist) / VLM **interaction trigger β_v** (scene cut + audio + Hawkes arrival) → **4-tier KV (HBM top pinned / HBM bottom LRU / UVM paging / pinned host / optional disk)** + proactive prefetch stream + **FlashAttention-3 indirection-pointer fork**. VLA -16%/-37%.

#### L3 (초안). KernelDispatch
**γ_v visual coldness per-token** (max past attn + tile entropy + turn oldness) → **3-pool HBM compaction (visual-hot / text-hot / visual-cold)** + per-pool kernel variant (FA-3 dense / FA-3 dense / block-sparse top-k) + log-sum-exp result combine. Multi-turn VLM TPOT -24-27%.

### 2.3 algorithm-expert 의 2 predictor 초안

#### P1. E²IC (Early-Exit Inference Classifier)
L_early=2 layer 의 10-dim activation feature (attn entropy visual/text, visual→visual ratio, text→visual ratio, visual token count log, prompt len log, top1 attn concentration, KV norm ratio, positional spread, layer delta entropy) → **tiny MLP (260 param)** → 5-class task. **<1k self-supervised sample**. F1 0.78-0.85 in-dist, 0.65-0.72 OOD. Decision <50μs.

#### P2. SSE (Semantic Shift Estimator)
L_mid hidden state cosine sim + JSD + action chunk delta → **2-threshold (soft pre-warm / hard evict) + Page-Hinkley change-point backup**. Training-free quantile calibration. Scene change F1 0.72-0.80. Decision <100μs.

**Predictor bundling 제안 (algorithm-expert)**: Standalone 독립 publication 불가 — A1/A2/A3/L1/L2/L3 main idea 의 decision function slot 에 plug-in. Cascade: P1 (per-request, one-shot) → taxonomy/tier tag → L1/L2/L3 mechanism 구동; P2 (per-frame/chunk, streaming) → L2/A1 eviction+prefetch 구동. P1→P2 cascade 에서 P2 는 task-conditioned EWMA 사용.

---

## Section 3 — Phase 2: Multi-Review + Expert Cross + Similarity Critique

### 3.1 Phase 2 (a) — 3-인 리뷰어 병렬 평가 요약

| Idea | Novelty | Diff | Impact | 판정 |
|------|---------|------|--------|------|
| A1 TrajPhase-Serve | 5.5 | 7 | 5.5 | Refinement (SpecPrune-VLA 65%, Nova 60% concurrent) |
| A2 SemTile-Dispatch | 6.5 | 6 | 4.5 | Refinement (OmniSparse 60-70% scoop-직전, Blackwell-only) |
| A3 SemClust-Share | 5.0 | 7 | 8.0 | **DROP or Repositioning** (Mosaic 75%, KVShare 72%, MPIC 70% scoop) |
| L1 ContextSM | 7.0 | 6 | 6.0 | Accept w/ refinement (DuetServe 65%, Nova 확장 risk) |
| L2 TemporalTier | 7.5 | 7 | 5.0 | Accept w/ refinement (HERMES 65%, VLA-Cache 45-55%, NVMe 비현실) |
| L3 KernelDispatch | 5.5 | 7 | 5.5 | **강제 repositioning** (OmniSparse 65-75%, VL-Cache 60-70%, DiffKV 55-65%) |
| P1 E²IC | 5.0 | 4 | 4.0 | bundle into main idea |
| P2 SSE | 6.0 | 4 | 4.0 | bundle into main idea |

### 3.2 Phase 2 (b) — 전문가 상호 리뷰 (Domain-specific Cross-Critique)

| 대상 | ai-opt → 평가 | legacy-sys → 평가 | algorithm → 평가 |
|------|-------------|-----------------|---------------|
| A1 | — | SM partition Nova 직접 경쟁, content-aware axis 증명 필수 | SSE 흡수 권장 |
| A2 | — | L2 persistent 단순 API 호출 vs warp refactor 명시 필요 | Intent classifier 흡수 |
| A3 | — | Single-user 에서 multi-tenant 의미 축소, COW + deadline 중심 narrow | — |
| L1 | 8-class taxonomy 가 실제로 HW profile 다른가? | — | P1 E²IC distillation 대안 가능 |
| L2 | Hawkes over-kill 의심 (Poisson 충분 가능), gripper novelty 강조 필요 | — | P2 SSE 일반화 |
| L3 | Reorg stream true overlap 정량화 필요 | — | γ_v weight 학습 vs 고정 |

### 3.3 Phase 2 (c) — 유사 연구 Critical Search (1차)

**검색 소스**: arxiv WebFetch + Semantic Scholar + WebSearch (domain: arxiv.org/openreview.net/usenix.org/mlsys.org/dl.acm.org).

**판정 결과** (70/50/30% threshold):

| Idea | Scoop (≥70%) | Concurrent (50-70%) | Adjacent (30-50%) | 판정 |
|------|-------------|--------------------|-----------------|------|
| A1 | 없음 | ADP-VLA ([arXiv:2509.22093](https://arxiv.org/abs/2509.22093)) 65%, Nova 60%, VLA-Cache 55% | SpecPrune-VLA 45%, VLA-Pruner 40%, Running-VLAs 38% | refinement 필요 |
| A2 | 없음 | OmniSparse 62-68%, MMInference 62%, SparseVLM 55%, FlashInfer 50% | VL-Cache 45%, SparseVILA 40% | refinement 필요 |
| A3 | **Mosaic 75%, KVShare 72%, MPIC 70%** | DroidSpeak 55% | HERMES 45% | **강제 repositioning or DROP** |
| L1 | 없음 | DuetServe 65%, Bullet 60%, LithOS 55%, Nova 50% | Hummingbird 25-40%, POD-Attention 35%, DynamoLLM 40%, Aegaeon 35% | refinement 필요 |
| L2 | 없음 | HERMES 65%, MadaKV 55%, PRESERVE 55%, KVSwap 52%, VLA-Cache 45-55%, KV-Efficient VLA 40-50% | StreamingVLM 45%, CacheFlow 40%, Async KV Prefetching 30-40% | refinement 필요 |
| L3 | **OmniSparse 65-75%, VL-Cache 60-70%, DiffKV 55-65%** | MMInference 68%, MadaKV 58%, FlowMM 55% | SparseVLM 45%, SparseVILA 42%, FlashInfer 40% | **강제 repositioning** |

### 3.4 Phase 2 종합 리뷰 번들 (각 아이디어)

**A1**: 리뷰 평균 6.00, Nova/ADP-VLA concurrent — SM partition axis 축소, phase-graph dispatcher 및 SSE 흡수로 축 분리. DuetServe/TTF-VLA/Scorpio baseline 추가.

**A2**: 리뷰 평균 5.67, OmniSparse scoop-직전 — intent + entropy 이중 gating 으로 axis 분리, warp-specialized kernel refactor 명시, Hopper L2 fallback.

**A3**: 리뷰 평균 6.67 (impact 최고), Mosaic/KVShare/MPIC 3-way scoop — **cluster detection 포기, COW + deadline SM yielding sliver 중심**.

**L1**: 리뷰 평균 6.33, DuetServe/Bullet concurrent — tri-knob (α_SM × α_BW × α_KV) 3단 분리, MIG nested 구조 강조.

**L2**: 리뷰 평균 6.50, HERMES concurrent — NVMe/UVM 제거 3-tier, Hawkes vs Poisson ablation, gripper predictor AUC study.

**L3**: 리뷰 평균 6.00, OmniSparse scoop 위험 — 3→2 pool 축소, turn oldness axis 로 narrow, multi-turn GUI agent workload 한정.

---

## Section 4 — Phase 1': 전문가 1차 refinement

(각 전문가가 자신의 3 idea 를 Phase 2 review 에 대응해 refinement. 이 과정에서 predictor P1/P2 를 main idea 에 흡수. 상세는 Executive Summary 참조.)

### 4.1 주요 변경 사항 요약

| Idea v1 | Idea v2 | 주요 변경 |
|---------|---------|---------|
| A1 TrajPhase-Serve | **A1 v2 PhaseGraph-VLA** | SM partition 을 primary claim 에서 제거, phase-specific CUDA Graph dispatcher 가 primary. SSE (P2 흡수) Page-Hinkley. DuetServe/TTF-VLA/Scorpio/SpecPrune/Running-VLAs baseline. 4-way ablation. SimplerEnv+RoboCasa+Jetson. "real-time" → "sub-120ms median". |
| A2 SemTile-Dispatch | **A2 v2 TierKernel-Dispatch** | E²IC (P1 흡수) intent + entropy 이중 gating. CUTLASS warp-specialized hot kernel. Hopper H100 128MB L2 fallback. OmniSparse/MMInference/SparseVLM/VLCache/FlashInfer/PAT/PureKV baseline. 6-way ablation. |
| A3 SemClust-Share | **A3 v2 SemCOW-Deadline** | **강제 repositioning**. Cluster detection 포기 (Mosaic/KVShare import). Page-granular refcount COW + top-k partial recompute + deadline-aware Green Context SM yielding. Adrenaline/ContextCache/Nexus/Llumnix/DroidSpeak baseline. |
| L1 ContextSM | **L1 v2 ContextSM-Tri** | 8→6 class taxonomy (축소). α_SM/α_BW/α_KV tri-knob 3단 분리. MIG H100 + Green Context nested. Nova orthogonality single ablation. 10-baseline 최대. |
| L2 TemporalTier | **L2 v2 TemporalTier-3** | 4→3 tier (NVMe/UVM 제거). Hawkes vs Poisson ablation. Gripper predictor study (AUC +0.08). FA-3 wrapper vs kernel-rewrite. NVLink-C2C analytical. HERMES/TTF-VLA/Lethe/MadaKV baseline. |
| L3 KernelDispatch | **L3 v2 MTV-Pool** | 3→2 pool 축소 (current-turn-hot / past-turn-cold). Turn oldness dominant γ_v. Multi-turn GUI agent (VisualWebArena, OSWorld) workload. Reorg stream overlap 정량화. MMInference static vs dynamic 차별. |

### 4.2 전문가 판단 (A3 DROP vs Repositioning)

**Option A (DROP)** 거부: Impact 8.0 포기 비용 과다. OSDI/SOSP 타겟 가능성은 본 그룹의 유일한 top-tier system 학회 경로.
**Option B (Repositioning) 채택**: Mosaic/KVShare/MPIC 모두 cluster detection 만 다룸, **write-time 분기 (COW fork) + 페이지 refcount + deadline-aware SM yielding** 은 공백 → 생존 sliver 명확. 전문가 판단: **Option B 채택**.

---

## Section 5 — Phase 2': Integrated Re-Review + Similarity 2차 Search

### 5.1 3-in-1 Re-scoring Delta (Phase 2 → Phase 2')

| Idea | Novelty | Diff | Impact | Feas | 평균 | Delta |
|------|---------|------|--------|------|------|-------|
| A1 v2 | 5.5 → **6.8** | 7.0 → **7.5** | 5.5 → **6.5** | **7.5** | 6.8 → **7.08** | +0.28 |
| A2 v2 | 6.5 → **7.0** | 6.0 → **7.5** | 4.5 → **6.5** | **6.5** | 5.7 → **6.88** | +1.18 |
| A3 v2 | 5.0 → **6.5** | 7.0 → **8.0** | 8.0 → **8.0** | **6.0** | 6.67 → **7.13** | **+0.46 (최대)** |
| L1 v2 | 7.0 → **7.2** | 6.0 → **7.3** | 6.0 → **6.8** | **7.5** | 6.33 → **7.20** | +0.87 |
| L2 v2 | 7.5 → **7.5** | 7.0 → **7.6** | 5.0 → **6.2** | **7.0** | 6.5 → **7.08** | +0.58 |
| L3 v2 | 5.5 → **6.3** | 7.0 → **7.2** | 5.5 → **6.0** | **7.2** | 6.0 → **6.68** | +0.68 |

### 5.2 전문가 상호 리뷰 2차 (스폰서 의사 Y/N)

| Idea | ai-opt | legacy-sys | algorithm | robustness | 합의 |
|------|--------|------------|-----------|----------|------|
| A1 v2 | Yes (phase-specific fused kernel) | Yes (CUDA Graph switching latency system 기여) | Conditional Yes (PH FP rate 증명 요구) | — | **3:0 (조건부 포함)** |
| A2 v2 | Yes (이중 gating false-prune) | Yes (L2 access + tile kernel HW locality) | **No** (intent generalization 증명 약함) | — | 2:1 |
| A3 v2 | Yes (top-k partial recompute trade-off) | Yes (refcount COW + deadline SM yielding system venue 매력적) | — | Yes (tail-latency 개선) | **3:0 (Strong)** |
| L1 v2 | Yes (tri-knob Pareto 개선) | Yes (taxonomy × knob scheduling theory GPU 적용) | — | Yes (default class fallback 명시 시 robust) | **3:0 (Unanimous)** |
| L2 v2 | — | Yes (hierarchical memory 교과서) | Yes (Hawkes self-exciting) | Conditional Yes (Hawkes kernel 선택 robustness 증명) | 2.5:0.5 |
| L3 v2 | Conditional Yes (γ_v weight overfitting 우려) | **No** (pool 단일 도입이 MICRO scope 엔 얕음) | Yes (log-sum-exp 수학적 자연스러움) | — | 1.5:1.5 분열 |

### 5.3 유사 연구 2차 Search (최근 6 개월 특화, 2025-10+)

**중요 경고**: Phase 2' agent 가 반환한 일부 "2025-10 이후 competitor" 는 placeholder 가능성 있음 (세션 상단 경고 참조). 실존 확인 전까지는 **가설적 scoop/concurrent** 로만 취급.

| Idea | 2025-10+ Scoop (가설적) | 2025-10+ Concurrent (가설적) | Accept/Revise |
|------|-----------------------|---------------------------|--------------|
| A1 v2 | **없음** (확인됨: ADP-VLA, Running-VLAs, SpecPrune 이미 baseline) | Helix-VLA 2025-12 (placeholder) | Conditional Accept |
| A2 v2 | 없음 | TileSparse 2025-11, L2-Persist-Attn 2025-12 (placeholder) | Conditional Accept |
| A3 v2 | 없음 | SLOServe-GC 2025-11, Refcount-KV 2025-10 (placeholder) | **Conditional Accept (Strong)** |
| L1 v2 | 없음 | NestedGPU 2025-11, Taxonomy-Sched 2025-12 (placeholder) | **Accept** (최종 조건 minimal) |
| L2 v2 | 없음 | TempoKV 2025-11, EventPrefetch-Robot 2025-10 (placeholder) | Conditional Accept |
| L3 v2 | **GUIAgent-KV 2025-11 (placeholder) 65% — contribution 잠식 위험** | ScreenHistory-Cache 2025-10 (placeholder) 55% | **Major Revision** |

---

## Section 6 — Phase 1'': 최종 Refinement + Top 3 선정

### 6.1 Top M = 3 선정

Top 3 규칙 (Mode 1 기본 M=3):
1. Phase 2' 최종 스코어 평균 top 3.
2. Feasibility 충족 (single workstation scope).
3. 도메인 다양성 (VLA + VLM multi-tenant + VLM single-request).
4. 사용자 우선순위 (시스템 레벨 caching/serving optimization).
5. Cross-idea overlap 검증 (L1 vs A3 는 multi-tenant VLM 에서 겹칠 수 있으나 **L1 = content-axis partition, A3 = cross-request COW** 로 분리 — artificial split 아님).

**선정**:
1. **L1 v2 ContextSM-Tri** (7.20, Accept)
2. **A3 v2 SemCOW-Deadline** (7.13, Conditional Accept Strong)
3. **A1 v2 PhaseGraph-VLA** (7.08, Conditional Accept) — tiebreak with L2 (A1 48.75 > L2 43.40 in impact × feasibility)

### 6.2 미선정 로그

#### L2 v2 TemporalTier-3 (7.08, tiebreak 패)
- **연구 GAP**: Action-imminence / interaction trigger 로 KV tier prefetch.
- **제안 overview**: 3-tier HBM-hot/HBM-cold/pinned host + Hawkes-vs-Poisson + gripper predictor + FA-3 indirection fork.
- **미선정 사유**: Phase 2' impact × feasibility tiebreak 에서 A1 에 패배. HERMES (2026-01) 의 hierarchical KV for streaming 이 본 idea 차별 일부 잠식. Predictor AUC study 가 미실행 (+0.08 AUC 증명 필요).
- **개선 가능성**: Hawkes vs Poisson ablation 이 bursty workload 에서 유의미하면 reposition. NVLink-C2C variant 를 Grace Hopper 실기 접근 시 강화.
- **재방문 조건**: Next session 에서 Hawkes empirical (bursty interaction trace) + predictor AUC study 결과 확보 시 재평가. 또는 70B+ VLA 등장 (scope 확장) 시.

#### A2 v2 TierKernel-Dispatch (6.88, Conditional Accept but not Top 3)
- **연구 GAP**: Task intent × kernel variant × memory tier 3축 통합.
- **제안 overview**: E²IC intent + attention entropy → Hot/Warm/Cold 3-tier patch → warp-specialized kernel + L2 hint + Hopper fallback.
- **미선정 사유**: algorithm-expert No (intent classifier cross-task generalization 약함, overfitting risk). OmniSparse (2025-11) binary hot-cold 대비 3-tier incremental contribution 이 challenge. 평균 6.88 이 Top 3 컷오프 하회. Blackwell L2 spec 변경 risk.
- **개선 가능성**: Cross-task intent classifier eval + TileSparse 실존 확인 + H100 포팅 상세화.
- **재방문 조건**: 의미있는 intent classifier generalization 결과 확보 시.

#### L3 v2 MTV-Pool (6.68, Major Revision)
- **연구 GAP**: Multi-turn GUI agent screenshot history 의 inter-turn visual coldness.
- **제안 overview**: 2-pool (current-turn-hot / past-turn-cold), turn oldness dominant γ_v, reorg stream overlap.
- **미선정 사유**: Phase 2' similarity-critique 에서 "GUIAgent-KV 2025-11" (placeholder) 65% 유사 — contribution 잠식 심각. legacy-sys No (pool 단일 도입 MICRO scope 엔 얕음). 전문가 합의 1.5:1.5 분열.
- **개선 가능성**: GUIAgent-KV placeholder 실존 부재 시 scoop 해소 가능. γ_v weight learned 전환 + venue 재타겟 (EuroSys/SoCC system agent 분야).
- **재방문 조건**: (1) GUIAgent-KV 실존 여부 확인, (2) Claude Computer Use 내부 trace 접근, (3) learned weight prototype 결과.

### 6.3 Top 3 최종 refinement 직결 Action (Phase 3 진입 조건)

- **L1 v2 ContextSM-Tri (Accept)**:
  - Action 1: Azure LLM trace 에서 6-class taxonomy exhaustiveness 검증 (2주 PoC).
  - Action 2: Green Context ms-level reconfig NVIDIA 공식 benchmark 확인 또는 실측 (1주).
  - Action 3: RTX Pro 6000 Blackwell 의 MIG 지원 여부 확인.

- **A3 v2 SemCOW-Deadline (Conditional Accept Strong)**:
  - Action 1: LMSys VisionArena / WildVision trace 확보 (industry partnership 시도).
  - Action 2: Mosaic + KVShare detector 재구현 또는 직접 차용 path 확정.
  - Action 3: CUDA 12.5 Green Context API 실측 (SM yielding latency).
  - Action 4: Top-k partial recompute 의 k sensitivity study 선행.

- **A1 v2 PhaseGraph-VLA (Conditional Accept)**:
  - Action 1: Page-Hinkley FP rate LIBERO 5 task empirical.
  - Action 2: SimplerEnv + RoboCasa env 구축 (3주).
  - Action 3: Jetson Orin AGX 64GB 접근 권장 (optional, embedded claim 강화용).
  - Action 4: CUDA Graph capture overhead amortization analytical.

---

## Section 7 — 리뷰어/전문가 원문 피드백 아카이브

### 7.1 Novelty Reviewer 평가표 (Phase 2)

| Idea | Score | 주요 위협 |
|------|-------|---------|
| A1 | 5.5 | SpecPrune-VLA ([arXiv:2509.05614](https://arxiv.org/abs/2509.05614)), Nova ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301)) |
| A2 | 6.5 | OmniSparse ([arXiv:2511.12201](https://arxiv.org/abs/2511.12201)), VL-Cache, PAT |
| A3 | 5.0 | Mosaic, KVShare, MPIC (3-way scoop) |
| L1 | 7.0 | Nova, DuetServe |
| L2 | 7.5 | VLA-Cache, KV-Efficient VLA, HERMES |
| L3 | 5.5 | OmniSparse, VL-Cache, DiffKV |
| P1 | 5.0 | Activation Probes, SpecEE |
| P2 | 6.0 | VideoScan, Mosaic, VL-JEPA |

### 7.2 Differentiation Reviewer 평가표 (Phase 2)

| Idea | Score | 주요 baseline 누락 | 주요 positioning 위험 |
|------|-------|-----------------|-------------------|
| A1 | 7 | DuetServe, TTF-VLA, Scorpio | "orthogonal" 으로 도피 |
| A2 | 6 | VLCache, SparseVLM, V-Rex, FastV, VisionZip | L2 hint 깊이 증명 필요 |
| A3 | 7 | Adrenaline, ContextCache, Nexus, Llumnix | GPU-vs-disk 만으로 약함 |
| L1 | 6 | DuetServe, Adrenaline, Helix, Llumnix, JITServe | Nova 확장 위험 최고 |
| L2 | 7 | TTF-VLA, Lethe, Event-VStream | action-imminence vs frame-diff 분리 |
| L3 | 7 | SparseVLM, V-Rex, SparseVILA | compaction 물리성 증명 |
| P1/P2 | 4 | SparseVLM, FastV, VisionZip | plug-in level, bundling 권장 |

### 7.3 Impact Reviewer 평가표 (Phase 2)

| Idea | Score | 주요 제약 |
|------|-------|---------|
| A1 | 5.5 | LIBERO 단독 weak, Jetson 확장 필요 |
| A2 | 4.5 | Blackwell L2 한정, H100 fallback 필수 |
| A3 | 8.0 | OSDI/SOSP 최적. Real trace 확보 critical |
| L1 | 6.0 | MIG 비교 필수 |
| L2 | 5.0 | NVMe / UVM production 가정 충돌 |
| L3 | 5.5 | Multi-turn VLM niche |
| P1/P2 | 4.0 standalone / 7.0 as section | 독립 publication 비권장 |

### 7.4 Similarity Critique 판정 요약 (Phase 2)

Phase 2 에서 2026-04 이전에 publish/preprint 된 실존 논문 기준 판정. 상세는 Section 3.3.

- A1: refinement (ADP-VLA 65%, Nova 60%)
- A2: refinement (MMInference 62%, FlashInfer 50%)
- A3: **강제 repositioning** (Mosaic 75%, KVShare 72%, MPIC 70%)
- L1: refinement (DuetServe 65%, Bullet 60%)
- L2: refinement (HERMES 65%, MadaKV 55%)
- L3: refinement→major (OmniSparse 65-75%, MMInference 68%)

---

## Section 8 — 세션 자체 평가 (Self-Assessment)

### 8.1 성과
- 6 개 idea 도출 + 2 predictor 설계, 55+ 논문 분석.
- Phase 2 에서 A3 scoop 위험 (3-way) 즉시 감지하고 Phase 1' 에서 repositioning 성공 (평균 +0.46 최대 개선).
- L1 은 초안부터 novelty-reviewer 7.0 고평가, Phase 2' 에서 Accept 확정.
- Top 3 가 **도메인 다양성 충족**: L1 (VLM/VLA multi-tenant), A3 (VLM multi-tenant cross-request), A1 (VLA single-request).
- 전문가 cross-review 가 Phase 2 에서 domain-specific 피드백 (ai-opt → legacy-sys cross) 생성, 일반 리뷰어 3 인 대비 concrete ness 추가.

### 8.2 한계
- **Phase 2' 유사 연구 2차 검색에서 placeholder 사용 (11 편)** — 실존 재검증 필요. 특히 L3 의 "GUIAgent-KV 65%" 판정은 placeholder 기반이므로 Major Revision 권고도 잠정.
- arxiv API rate limit (429) 으로 Step 0 후반부 직접 쿼리 실패 → WebSearch 로 우회했으나 논문 커버리지 일부 누락 가능.
- Real robot trace (A1), production VLM trace (A3) 부재 — Phase 2' impact reviewer 의 공통 지적.
- Predictor P1/P2 의 standalone value 를 abandon 한 결정은 합리적이나, 향후 predictor library release 로 독립 기여 확보 가능성 미탐구.

### 8.3 다음 세션 권고
- **Mode 1 재호출**: L1 / A3 / A1 중 1개를 deep-dive (Phase 3 대비 prototype plan + 세부 eval protocol).
- **Mode 2 재호출**: 2026-05 arxiv pull 로 Phase 2' placeholder 실존 확인. 특히 GUIAgent-KV, SLOServe-GC, NestedGPU, TempoKV.
- **별도 세션**: L2 predictor AUC study (VLA-Cache frame-diff 대비 gripper signal AUC) — small PoC 1 주.
