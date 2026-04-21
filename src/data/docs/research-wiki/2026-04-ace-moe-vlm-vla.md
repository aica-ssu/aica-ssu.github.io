# ACE-MoE 확장 연구 주제 Top 3 — VLM/VLA Software 방향

**작성일**: 2026-04-21
**Mode**: 2 (local PDF 분석) | **참여 전문가**: ai-optimization (main), algorithm (sub) | **리뷰어**: novelty / differentiation / impact
**도출 출처**: `aica-research-bot` harness — ACE-MoE ICCAD 제출본 + Qwen3-VL software 측정 데이터. PIM 측면 제외, software 확장만.

> 본 문서는 세 개의 연구 주제 초안입니다. 모든 예상 개선치는 **보수적**으로 제시되었으며, 단일 워크스테이션(RTX 4090/5090/RTX Pro 6000) 환경에서 구현 및 exploration 가능한 scope로 설계했습니다. Phase 2 리뷰어 3인 평가 후 최신 arxiv 관련연구 60+편 재검색(2024-2026)으로 positioning을 검증했습니다.

---

## 배경 — 왜 VLM/VLA로 확장하는가

ACE-MoE(ICCAD'26 submission)의 결론부에는 다음이 명시되어 있습니다:

> "We expect ACE-MoE to further optimize the trade-off between performance, accuracy, and memory efficiency in emerging **Vision-Language or Vision-Language-Action models**, where inference involves significantly higher token counts."

이 future direction은 (a) VLM의 visual token이 sequence의 86.2%를 차지하면서 attention 기여는 11.4%에 그치는 극단적 비대칭, (b) VLM prefill이 LLM 대비 6-22× 길어지는 TTFT explosion, (c) VLA의 closed-loop 30-100Hz 실시간 제약 등 **ACE-MoE의 메커니즘(ACE caching, time-budgeted skipping/prefetching, fused kernel)이 직접 가치를 창출할 영역**이 뚜렷하게 존재한다는 관찰에서 나왔습니다.

최근 6개월 arxiv 조사 결과:
- **VLA serving 최적화** 관심이 2025-09 이후 집중 연구 분야로 부상 (ActionFlow, HyperVLA, CogVLA NeurIPS'25, A1, SemanticVLA 등)
- **MoE edge inference**가 budget/precision co-design 방향으로 선회 (DyMoE, Dynamic Expert Quantization, MoE-SpeQ 등)
- **VLA + MoE architectural fusion**이 2026-04부터 본격화 (HEX, HY-Embodied-0.5, Qwen3.5-Omni)
- **Modality-aware vs modality-agnostic routing** 논쟁 가시화 (AlignMamba-2 vs ERNIE 5.0)

이 배경 위에서 Top 3 주제를 선정했습니다.

---

## Top 1. ACE-VLA — Real-Time Action Decoding under Latency Budget for Vision-Language-Action Models

- **제안자**: ai-optimization-expert (main), algorithm-expert (sub)
- **리뷰 점수 (refined, post-arxiv augmentation)**: Novelty 8 · Differentiation 9 · Impact 9 · **평균 8.5 / 10**

### 핵심 가설
VLA 모델은 robotics control loop의 **hard real-time(30-100 Hz)** 제약을 가집니다. ACE-MoE의 time-budgeted skipping을 **action chunk별 hard latency budget**(예: 33 ms = 30 Hz)으로 재정의하고 visual context와 action prediction expert에 modality-aware ACE를 적용하면, action decoding latency를 30-50% 단축하면서 task success rate를 95% 이상 유지할 수 있습니다.

### 접근법
1. **Hard budget = `1/control_freq`** 명시적 설정 (예: 33 ms).
2. **Action chunk별 ACE update**: action sequence가 일관된 sub-episode 단위로 ACE 갱신.
3. **Visual freshness scoring**: 환경 motion(optical flow magnitude) 기반 visual KV importance reweighting. 정적 환경 = 높은 reuse, 동적 환경 = freshness penalty.
4. **Safety-aware skip whitelist**: collision avoidance 같은 safety-critical task는 minimal expert subset 강제 활성화.
5. **Streaming visual prefill**: camera frame 수신 즉시 incremental ViT + ACE importance 업데이트(full prefill 대기 없이).
6. **VLA-MoE retrofit recipe**: OpenVLA-7B의 FFN을 expert로 분해하는 간단한 변환 방법을 contribution에 포함.
7. **Schedulability analysis**: WCET(Worst-Case Execution Time) 모델로 probabilistic real-time guarantee 제공.

### 예상 개선 (보수적)
| 지표 | Baseline | 목표 | 적용 조건 |
|------|---------|------|---------|
| Per-action latency | OpenVLA+ACE-MoE direct port ≈ 50 ms | **25-35 ms** | RTX 4090, 7B class |
| Task success rate (SimplerEnv) | direct port 대비 −5% ~ −10% | **−1% ~ −3%** | manipulation |
| Throughput (actions/sec/GPU) | 1.0× | **1.4-2.0×** | mixed batch |
| Memory footprint | 100% | **50-65%** | 50% expert cache |

**적용 범위**: VLA (OpenVLA, RT-2-X, Octo, Pi-0) + MoE 변형. Manipulation/navigation.
**미적용**: pure simulation only, language-only robots.
**불확실 영역**: real robot eval은 협력 lab 필요 — 우선은 simulator.

### 실험 설계
- **Server**: #1(RTX 4090) 또는 #3(RTX 5090)
- **Framework**: HuggingFace + custom OpenVLA inference patch
- **Models**: OpenVLA-7B, Pi-0, (가용 시) OpenVLA-MoE retrofit
- **Datasets**: SimplerEnv (Bridge / RoboCasa), LIBERO
- **Metrics**: end-to-end latency, control frequency 달성도, task success rate, action chunk consistency
- **Baselines**: vanilla OpenVLA, ACE-MoE direct port, FlexGen-style offloading
- **Expected runtime**: ≈6 days (sim 평가 포함)
- **Fallback**: simulator only, real robot은 향후 협력

### 최신 관련 연구 (Closest competitors)
| Paper | arXiv / Venue | Date | 관계 | 차별점 |
|-------|-------------|------|------|------|
| **AdaMoE-VLA** | [2510.14300](https://arxiv.org/abs/2510.14300) · [OpenReview](https://openreview.net/forum?id=cNZ5W1f4tE) · [GitHub](https://github.com/swjTheDad/AdaMoE-VLA) | 2025-10 | **가장 직접적 VLA-MoE 선례** (training-time) | Dense VLA의 FFN을 sparse MoE로 대체. **Scale Adapter**로 expert selection/weighting decouple → "collaborative" utilization. 4 routed experts example, top-K=1-N 설정 가능. LIBERO +1.8%, RoboTwin +9.3%, real-world +21.5%. Base: OpenPI(Pi-0) 기반. **Inference serving 최적화는 미제시** — ACE-VLA와 orthogonal하게 결합. 상세는 아래 Implementation Roadmap 섹션 참조. |
| **ActionFlow** | 2512.20276 | 2025-12 | strong competitor (pipeline) | Cross-Request Pipelining으로 2.55× FPS on OpenVLA-7B. **Expert-level skip/cache 없음**. ACE-VLA는 expert axis 추가 |
| **HyperVLA** | 2510.04898 | 2025-10 | alternative | Hypernetwork로 90× param, 120× speedup. 아키텍처 변경 vs runtime scheduling — orthogonal |
| **CogVLA** | 2508.21046 | 2025-08 (**NeurIPS'25**) | related | Instruction-driven sparsification 2.8× latency. ACE-VLA는 runtime expert scheduling — 결합 가능 |
| **A1** | 2604.05672 | 2026-04 | budget-aware baseline | 72% per-episode latency reduction via budget-aware truncated flow matching. Episode-level vs chunk-level budget 차별 |
| **HEX** | 2604.07993 | 2026-04 | humanoid MoE | Mixture-of-Experts Unified Proprioceptive Predictor for humanoid. 전체 VLA stack이 아닌 proprioceptive 전용 MoE |
| **HY-Embodied-0.5** | 2604.07430 | 2026-04 | trend 근거 | Tencent 2B/32B embodied foundation, Mixture-of-Transformers |
| **SemanticVLA** | 2511.10518 | 2025-11 | visual-side pruning | 2.7× latency reduction. Orthogonal |
| **Action-aware Dynamic Pruning** | 2509.22093 | 2025-09 | token pruning | action stage별 pruning, 1.35× |
| **Adaptive Action Chunking** | 2604.04161 | 2026-04 | chunk 관련 | action entropy 기반 chunk size. ACE-VLA의 chunk boundary 결정에 채택 검토 |
| **DA-PTQ / QVLA / HBVLA** | 2602-2604 | 2026 | VLA quantization | weight quantization, ACE-VLA와 orthogonal |

> **보강 관찰 (2026-04-21)**: 실제 VLA+MoE 연구는 공개된 것이 매우 드뭅니다(AdaMoE-VLA, HEX, HY-Embodied 정도). 대부분 dense architecture이거나 MoE를 "judgment 용도로 곁가지로 붙이는" 수준에 머물러 있습니다. 이는 **ACE-VLA의 blue-ocean 특성을 유지하면서 AdaMoE-VLA를 base training recipe로 활용**하는 positioning을 가능하게 합니다. AdaMoE-VLA는 training-time architecture, ACE-VLA는 inference-time serving — 두 연구가 **strictly complementary** 합니다.

### Implementation Roadmap — AdaMoE-VLA base 활용 재현 전략

AdaMoE-VLA 공식 checkpoint 미공개 상태이므로, 본 lab에서 직접 훈련 후 ACE-VLA inference 최적화를 얹는 2단계 계획을 권장합니다.

#### Step A. AdaMoE-VLA base 재현 (2-4주 추정)

**Starting point 선택지 (권장 순)**:
1. **공식 GitHub + Pi-0 base** (권장):
   - `git clone --recurse-submodules https://github.com/swjTheDad/AdaMoE-VLA.git`
   - `GIT_LFS_SKIP_SMUDGE=1 uv sync`
   - Pi-0 pretrained weight 로드 (Physical Intelligence 공개 checkpoint)
   - `finetune.sh`를 LIBERO 데이터로 수정하여 실행
2. **OpenVLA-7B 기반 retrofit** (alternative):
   - AdaMoE-VLA의 core 메커니즘(FFN → MoE + Scale Adapter)을 OpenVLA의 Llama2-7B backbone에 porting
   - Pi-0보다 model/codebase가 널리 알려져 있어 디버깅 용이
   - 단, paper 결과와 직접 비교 불가 — 방법론 복제 검증 차원

**Base 모델 크기 & 자원**:
| 옵션 | Base 파라미터 | Active 추정 | Training 가능 서버 |
|------|-------------|-----------|-----------------|
| Pi-0 + AdaMoE 4 experts | ≈3B dense → ≈6-8B total | ≈3B | #5 (RTX Pro 6000 96GB) 단일 GPU + LoRA |
| Pi-0 + AdaMoE 8 experts | ≈3B dense → ≈12-15B total | ≈3-4B | #5 with 4-bit optimizer state |
| OpenVLA-7B + AdaMoE 4 experts | ≈7B dense → ≈14-18B total | ≈7B | #4 (4090×2) with FSDP, 또는 #5 with LoRA |

**Training time 추정** (LIBERO 기준, full finetune):
- 논문이 H100 기준 시간 미공개 — 유사 VLA 훈련(OpenVLA LIBERO finetune = 8×H100, 1-2일 기준)을 기준으로 scaling:
- **#5 단일 RTX Pro 6000 96GB + LoRA**: 5-10일 (보수적)
- **#4 4090×2 + FSDP + gradient checkpointing**: 10-15일
- **#5 full finetune (no LoRA)**: 8-12일
- *위험 요소*: RoboTwin 데이터는 LeRobot format 변환 비용 추가 (-2-3일 engineering)

**재현 시 우선 검증 metric**:
- LIBERO Goal/Object/Spatial task success rate가 AdaMoE-VLA paper 수치(+1.8% over dense baseline) 수준에 도달하는지
- Scale Adapter 제거 시 accuracy degradation 재현 (ablation)

#### Step B. ACE-VLA 적용 (2-3주 추정)

AdaMoE-VLA base 위에 ACE-VLA의 7가지 기법 구현:
1. Hard budget 33ms 설정 + measurement infrastructure (1주)
2. Expert-level cumulative score + ACE caching integration (1주)
3. Visual freshness scoring + safety whitelist + schedulability analysis (0.5주)
4. SimplerEnv/LIBERO inference latency/success rate 측정 (0.5주)

**총 소요 (Step A + B)**: 보수적으로 5-7주 (1인 집중 연구 기준).

#### 구현 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| AdaMoE-VLA checkpoint 미공개 | High — 재현 비용 | Pi-0 또는 OpenVLA base로 방법론만 복제. Paper 결과와 가까워지는지를 검증 기준으로 삼음 |
| LIBERO + RoboTwin 데이터 license/접근 | Mid | LIBERO는 공개, RoboTwin만 별도 확인 필요 |
| Scale Adapter + ACE caching 상호작용 | Mid | Scale Adapter는 expert weighting 신호를 제공 — ACE의 cumulative score 계산에 scale value를 이미 반영할 수 있어 **오히려 ACE-VLA를 단순화**할 가능성 |
| Training 기간 중 컴퓨팅 unavailability | Mid | Step A와 Step B를 분리하여 Step B는 AdaMoE-VLA 공식 checkpoint가 혹시 공개되면 (OpenReview decision 이후 가능성) fast-track |
| Pi-0 vs OpenVLA 양쪽 검증 부담 | Mid | Pi-0 하나로 집중, OpenVLA는 ACE-VLA의 generality 실험에서만 ablation |

#### Checkpoint/Dataset 확보 체크리스트

- [ ] Pi-0 공개 pretrained weight 확인 (Physical Intelligence 페이지)
- [ ] LIBERO 데이터셋 다운로드 (LeRobot hub)
- [ ] RoboTwin 데이터셋 접근 권한 확인 (필요 시)
- [ ] SimplerEnv 설치 (real robot eval 미지원 대안)
- [ ] AdaMoE-VLA OpenReview decision 모니터링 — accept 시 공식 checkpoint 공개 기대

### 리스크 (전체)
| 리스크 | 완화 방안 |
|--------|---------|
| VLA-MoE 공개 모델 부재 | **AdaMoE-VLA 재현** (Pi-0 base)을 첫 step으로 편입. Ablation으로 contribution 명확화 |
| Hard real-time guarantee | Probabilistic SLO(P99 < budget)로 완화 |
| Safety-critical skip 실패 | Whitelisted "always-on" expert 명시 |
| Real robot eval 부재 | Sim-only로 PoC, real robot은 future |
| AdaMoE-VLA 결과 재현 실패 | OpenVLA base로 스위치 + 저자 연락 (GitHub issue) |

---

## Top 2. Joint Token-Expert Budget Allocator — Cross-Modal Resource Scheduling for VLM Inference

- **제안자**: ai-optimization-expert (main), algorithm-expert (sub)
- **리뷰 점수 (refined, post-arxiv augmentation)**: Novelty 7.5 · Differentiation 8 · Impact 8.5 · **평균 8.0 / 10**

### 핵심 가설
VLM-MoE에서 **visual token skip**과 **expert skip**은 공유된 PCIe/HBM bandwidth budget을 두고 경쟁합니다. 두 axis를 joint하게 layer-wise budget allocation으로 풀면, 단일 axis 최적화 대비 latency를 추가 20-30% 단축하면서 iso-accuracy Pareto 15-25% gain을 얻을 수 있습니다.

### 접근법
1. **Layer별 (token attention compute + expert transfer)** 두 cost 모델링, 총 layer budget `T_layer`.
2. **Per-layer optimizer**(closed-form 또는 DP)로 token retention `r_t`와 expert prefetch 비율 `r_e` 결정.
3. **Modality-conditional**: Dense layer(L17-21)는 token 우선, sparse layer(L0-7)는 expert 우선.
4. **Online adaptation**: 첫 N requests로 cost 재추정(EWMA).
5. **Differential quantization**: Sparse layer KV는 INT8, dense layer는 FP16.
6. **vLLM scheduler API design** — first-class scheduler로 공개(PR-ready).

### 예상 개선 (보수적)
| 지표 | Baseline | 목표 | 적용 조건 |
|------|---------|------|---------|
| TTFT (FHD VLM-MoE) | ACE-MoE 단일 ≈ 800 ms | **500-650 ms** | BS=4-8 |
| Decode throughput | 단일 ≈ 600 tok/s | **700-800 tok/s** | BS=1 |
| Accuracy (MMMU / ChartQA) | baseline | **-0.5--1.0%** | 50% combined retention |
| Pareto gain vs naive I1+I2 | — | **15-25%** | iso-accuracy |

**적용 범위**: VLM-MoE + dense VLM에서 ACE-MoE-style 동시 적용 시나리오.
**미적용**: 단일 modality LLM(불필요), VLA closed-loop(별도 budget).

### 실험 설계
- **Server**: #5 (RTX Pro 6000 96GB)
- **Framework**: vLLM + custom joint budget scheduler
- **Models**: Qwen3-VL-4B/8B, LLaVA-Mixtral, MoE-LLaVA
- **Datasets**: MMMU, ChartQA, MS-COCO captioning
- **Metrics**: latency-accuracy Pareto curve vs 각 baseline
- **Baselines**: I1 alone, I2 alone, I1+I2 naive stack, ACE-MoE original, SARATHI
- **Expected runtime**: ≈5 days (Pareto 측정 포함)

### 최신 관련 연구 (Closest competitors)
| Paper | arXiv / Venue | Date | 관계 | 차별점 |
|-------|-------------|------|------|------|
| **DyMoE** | 2603.19172 | 2026-03 | strongest competitor | depth-adaptive + importance + mixed-precision + look-ahead prefetching. **Edge TTFT 3.44-22.7× 개선**. **Token axis 미포함** — 본 idea의 차별 포인트 |
| **Dynamic Expert Quantization** | 2511.15015 | 2025-11 | budget-constrained formulation 선례 | "online budget-constrained precision allocation" 정식화. Single-axis (precision only) |
| **Context-Aware MoE on CXL-NDP** | 2512.04476 | 2025-12 | per-expert 1-4 bit | 8.7× decoding. CXL-NDP specific |
| **SliceMoE** | 2512.12990 | 2025-12 | bit-sliced caching | slice-level granularity, 2.37-2.85× energy |
| **MoE-SpeQ** | 2511.14102 | 2025-11 | spec + prefetch + quantize | 2.34× speedup. I3는 token axis 추가 |
| **SpecMoE** | 2604.10152 | 2026-04 | speculative MoE | 4.30× throughput. Token generation axis — orthogonal |
| **QUOKA** | 2602.08722 | 2026-02 | token axis 관련 | 3× TTFT + 5× attention, 88% KV reduction. Text-only |
| **LAPS** | 2601.11589 | 2026-01 | length-aware serving | long/short prefill disaggregation |
| **FlowPrefill** | 2602.16603 | 2026-02 | chunked prefill follow-up | operator-level preemption |
| **BuddyMoE** | 2511.10054 | 2025-11 | redundancy fallback | I3에 fallback 통합 검토 |

### 리스크
| 리스크 | 완화 방안 |
|--------|---------|
| Optimizer 복잡도 | Layer당 cost가 단순 함수 → closed-form 가능 |
| Cost profiling 부정확 | Online EWMA adaptation |
| Quantization 차등이 accuracy 손상 | Per-layer sensitivity gating, layer 0/inject layer는 FP16 강제 |

---

## Top 3. Modality-Aware ACE for VLM-MoE (with Hierarchical Injection Layer Boundary)

- **제안자**: ai-optimization-expert (main), algorithm-expert (sub)
- **리뷰 점수 (refined, post-arxiv augmentation)**: Novelty 7 · Differentiation 7.5 · Impact 8 · **평균 7.5 / 10**

### 핵심 가설
ACE-MoE의 cumulative expert score를 **modality dimension(visual/text)**으로 분할하여 두 stream으로 유지하고, modality-conditional variance normalization을 적용하면, VLM-MoE 추론에서 **동일 accuracy 달성 cache ratio를 50% → 25-35%로 절반 감소** 가능합니다. DeepStack 같은 hierarchical injection layer는 **boundary group**으로 별도 처리합니다.

### 접근법
1. **Token별 modality tag 부여**. DeepStack injection 이후 token은 visual mass에 weight fading.
2. **Layer-wise expert score를 visual/text 두 stream으로 누적**.
3. **Modality-conditional variance normalization**.
4. **Global selection**: weighted sum `α` — entropy-based dynamic(token modality entropy 기반).
5. **ACE slot reservation**: 30%는 visual-critical 전용.
6. **Modality-aware time-budgeted prefetch**.
7. **Hierarchical Injection Layer (HIL) boundary handling**: Inject layer ±2는 별도 ACE pool, inject layer 자체는 quantized GPU resident.

### 예상 개선 (보수적)
| 지표 | Baseline | 목표 | 적용 조건 |
|------|---------|------|---------|
| Cache ratio (iso-accuracy) | 50% (ACE-MoE) | **25-35%** | VLM-MoE, batch ≥ 4 |
| Prefill latency vs ACE-MoE single | 1.0× | **0.85-0.95×** | cache ratio matched |
| Memory footprint | 100% | **60-75%** | 50% cache ratio |
| Accuracy (MMMU / VQAv2) | baseline | **-0.5--1.0%** | 50% cache |

**적용 범위**: VLM-MoE (MoE-LLaVA, Uni-MoE, DeepSeek-VL2, LLaVA-Mixtral, future Qwen3-VL-MoE), batch ≥ 4.
**미적용**: Dense VLM (다른 idea), 단일 modality LLM.
**불확실 영역**: VLM-MoE 공개 모델 부족 → MoE-LLaVA-7B로 우선 검증.

### 실험 설계
- **Server**: #5 (RTX Pro 6000 96GB)
- **Framework**: vLLM + ACE-MoE patch + modality tag
- **Models**: MoE-LLaVA-7B, Uni-MoE-8B, LLaVA-Mixtral
- **Datasets**: MMMU, ScienceQA, VQAv2, ChartQA
- **Metrics**: cache ratio별 accuracy curve, TTFT, per-layer expert miss rate
- **Baselines**: ACE-MoE direct port, MoE-LLaVA original serving, Edge-MoE, vLLM Mixtral default
- **Expected runtime**: ≈5-7 days

### 최신 관련 연구 (Closest competitors)
| Paper | arXiv / Venue | Date | 관계 | 차별점 |
|-------|-------------|------|------|------|
| **AlignMamba-2** | 2603.18462 | 2026-03 | philosophy 선례 | **Modality-specific + modality-shared experts**. Multimodal sentiment 특화. Serving 최적화 미제시 |
| **ERNIE 5.0** | 2602.04705 | 2026-02 | **counter-design** | Ultra-sparse MoE with **modality-agnostic routing**. I1과 반대 방향, 직접 비교 실험 필수 |
| **MoE-LLaVA** | 2401.15947 | 2024-01 | foundational VLM-MoE | 아키텍처, serving 최적화 없음 — I1 평가 target |
| **Qwen3.5-Omni** | 2604.15804 | 2026-04 | SOTA VLM-MoE | Hybrid Attention MoE, 256k context. 가용 시 evaluation target |
| **MoTE** | 2506.14435 | 2025-06 | ternary experts for multimodal | precision allocation — orthogonal |
| **QMoP** | 2603.21232 | 2026-03 | query-guided MoE projector | Visual projector level — I1과 다른 레벨 |
| **Efficient MoE Quantization** | 2604.06515 | 2026-04 | expert-wise mixed precision | I1 + quantization 결합 baseline |
| **VL-Cache** | 2410.23317 | 2024-10 | visual KV 측면 cousin | KV level, I1은 expert level — 결합 가능 |
| **AirCache** | 2503.23956 | 2025-03 | inter-modal KV | 10% KV, 29-66% latency |
| **Omni-C** | 2603.05528 | 2026-02 | dense alternative | MoE 대신 dense encoder — positioning 언급 필요 |

### 리스크
| 리스크 | 완화 방안 |
|--------|---------|
| VLM-MoE 공개 모델 부재 | MoE-LLaVA/Uni-MoE 우선, 또는 Mixtral+LLaVA adapter |
| Modality tag 부여 cost | Tokenizer 단계에서 tagging, runtime overhead <1% |
| Two-stream score noise | 첫 5 requests calibration window |
| ERNIE 5.0의 modality-agnostic 정당성 | 직접 비교 실험에서 modality-aware가 더 낫다는 근거 필수 |

---

## 미선정 아이디어 (참고)

- **Visual Token ACT (Accuracy-Critical Token)** — ACE 원칙의 visual KV 전이. KV compression 분야(VL-Cache, PyramidKV, SnapKV, H2O, Quest, Look-M 등)가 crowded하여 standalone으로는 차별성 약함. I3 또는 I1의 sub-component로 흡수 시 가치 있음.
- **Hierarchical Injection-Aware Routing** — DeepStack 한정 시 적용 범위 좁음. HIL framework로 일반화 후 **I1에 흡수**됨.
- **Image-Level Cross-Request Cache** (KV + expert profile) — vLLM prefix caching의 visual extension. Novelty incremental, vLLM/SGLang PR 형태로 권장.

---

## 연구 진행 제안 우선순위

| 우선순위 | Idea | 근거 |
|---------|------|------|
| 1 | **Top 1 (ACE-VLA)** | Novelty/Impact 가장 높음, blue ocean. VLA-MoE retrofit recipe가 별도 contribution이 될 수 있음 |
| 2 | **Top 2 (Joint Budget)** | System-level contribution, vLLM PR로 실질 기여 가능. DyMoE가 가까운 선례지만 token axis 추가가 명확한 차별 |
| 3 | **Top 3 (Modality-Aware ACE)** | ACE-MoE 직계 후속. AlignMamba-2/ERNIE 5.0로 인해 positioning 정교화 필요. MoE-LLaVA로 빠른 검증 가능 |

**결합 가능성**:
- Top 1(ACE-VLA)은 Top 2/3의 메커니즘을 내부적으로 차용 가능 (visual freshness + modality-aware caching).
- Top 2 + Top 3: VLM-MoE serving의 통합 framework로 합쳐 single paper로 제출 가능.

---

## 참고 자료

- **본 연구의 출발점**: ACE-MoE ICCAD'26 submission (본 연구실 제출본)
- **Motivation measurements**: Qwen3-VL-4B 기반 TTFT / layer-wise attention / visual KV asymmetry 분석 (Minsik Choi, 2026-04-07 internal report의 software 측면만)
- **관련 논문 60+편**: 2024-01 ~ 2026-04 arxiv API 재검색 결과 (이 문서의 Closest competitors 표에 핵심 반영)

---

## 문의

각 주제에 대한 진행 희망, 수정 제안, 혹은 PoC 구현 합류 의사는 공영호 교수에게 전달 바랍니다. 본 문서는 hidden link 형태로 공유되며, 홈페이지 일반 네비게이션에는 노출되지 않습니다.
