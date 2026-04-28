# Edge VLM Layer-Wise + Context-Semantic Optimization on Single-GPU / Jetson — Session Overview

> **한 문장 hook**: 최신 VLM (Qwen3-VL / LLaVA-Next / InternVL3) 의 single-GPU (RTX 5090) 또는 Jetson Thor / Orin NX edge inference 에서 **layer-wise + Context/Semantic** 특성을 활용해 Performance + Energy 동시 최적화. ATRIUM/BREAKWATER-T evolution + 신규 6 idea Top-M 선정.

> **CTRL+F 안내** (Mac: Cmd+F): 본 문서에서 모르는 용어 발견 시 **§9 Glossary** 에서 검색하세요. 도메인 약어, polysemous term (channel/block/layer), idea metaphor 풀이 모두 §9 에 통합.

---

## 1. Research Questions (이 연구가 답하려는 질문, R57.1)

### 1.1 전체 주제 RQ (overarching, 1-3 문장 개조식)

- **RQ-Master-1**: 최신 VLM 의 transformer layer 별 visual attention 비대칭 (HOT L17-21 = 24.5% / COLD L0-7 = 2.6%) 을 활용해 Jetson Thor 273 GB/s LPDDR5x UMA (60-130W) 또는 RTX 5090 32GB GDDR7 (575W) 전력 envelope 안에서 decode TPS 를 **+18% 이상** + Energy/token 을 **-20% 이상** 동시 달성하는 system-level mechanism 은 무엇인가?
- **RQ-Master-2**: Context/Semantic 특징 (visual token semantic similarity / cross-frame redundancy / DeepStack tap point) 을 KV cache management 와 결합했을 때 long-form video VLM 의 KV memory footprint 를 **-50% 이상** 줄이면서 accuracy drop **≤ 1.5pp** 유지 가능한가?
- **RQ-Master-3**: Layer-wise mixed precision (NVFP4/FP8/INT4) 와 power-envelope adaptive DVFS 를 결합한 single-system VLM serving 이 single Jetson Orin NX 16GB 환경 (10-25W) 에서 production-grade latency 보장 + thermal envelope 위반 0 으로 동작 가능한가?

### 1.2 각 idea 별 RQ (Tier-1 Top 4 + Tier-2 독립 Top 3)

#### 🥇 PRISM-FOG-FX
- **RQ-1.1**: Blackwell native NVFP4 (E2M1 16-block) 의 transformer layer 별 sensitivity 분포가 어떻게 visual attention ratio 와 상관되는가? DeepStack inject layer (L8/L16/L24) 에 NVFP4 anchor 보존 + 비-inject layer INT4-AWQ 압축 시 MMMU drop 이 uniform NVFP4 대비 **≤ 0.5pt** 안에서 prefill throughput **+18% 이상** 달성되는가?
- **RQ-1.2**: 4:8 structured sparsity 를 HOT layer (visual_attn_ratio > 0.15) MLP block 에만 적용 시 tensor core utilization 격상이 decode TPS 에 미치는 effect 는 baseline 대비 **+8% 이상** 인가? MMMU drop 은 **≤ 1.0pt** 이내 유지되는가?
- **RQ-1.3**: Visual-context confidence 가 낮은 decode step 에서 GPU SM clock 을 1500→900 MHz 로 down-slip + KV cache scale freeze 시, Energy/token 이 **-22% 이상** 줄고 latency 영향 **≤ 5%** 인 sweet-spot 은 어디에 있는가?
- **RQ-1.4**: LayerClassifier (visual_attn_ratio HOT/COLD/MEDIUM 자동 분류) 를 100-shot calibration 만으로 구축할 때 online classifier overhead 가 prefill latency **≤ 2%** 안에서 제한되는가?

#### 🥈 BIVOUAC-SLATE-R
- **RQ-2.1**: VLM HOT layer (Qwen3-VL L17-21) 의 visual KV token 1024개를 K-means k=16 cluster centroid 로 압축할 때 cluster-level attention broadcast 만으로 MMMU drop **≤ 1.0pt**, decode latency **-30% 이상** 감소되는가?
- **RQ-2.2**: 연속 video frame (또는 multi-turn image dialogue) 의 cluster centroid cosine distance ≥ 0.95 일 때 prev-frame KV 재사용 시, VideoMME-short 8-frame workload 에서 prefill **-40% 이상** 단축, KV memory **-50% 이상** 절감? Sali-Cache (arXiv:2602.14236) 의 frame-level reuse 대비 cluster-level reuse 가 fine-grained false-negative 를 **30% 이상** 줄이는가?
- **RQ-2.3**: layer 별 cluster k 를 (visual_attn_ratio + PyramidDrop convergence-layer signal) weighted sum 으로 가변 (shallow k=64 / mid k=16 / deep k=8) 했을 때, fine-grained reasoning task (DocVQA / OCR) 정확도 drop **≤ 1.5pt** 안에서 KV memory 추가 **-15%** 절감 가능한가?

#### 🥉 PRISM-VL-R
- **RQ-3.1**: 3 phase (encode/prefill/decode) 별 LSH hyperplane 가변 (encode 16 / prefill 8 / decode 0) 시, multi-camera surveillance workload (frame similarity > 0.85) 에서 cache hit rate 가 single-policy LSH 대비 **+15%pt 이상** 향상, false-positive accuracy drop **≤ 0.5pt** 안에서 유지되는가?
- **RQ-3.2**: SGLang RadixAttention 의 token-level radix tree 를 root second-level branch 가 visual-semantic-hash 로 split 되도록 확장했을 때, dashboard agent workload (동일 image base + 다른 prompt 50-70%) 에서 cross-request KV reuse 가 **+40%pt 이상** 향상되는가?
- **RQ-3.3**: Green Context 8-SM partition 에 LSH compute 격리 실행 시 decode latency 영향 **≤ 1%**, LSH overhead prefill latency **≤ 2%** 안에 유지되는가?
- **RQ-3.4 (Tier-1 차별 검증)**: 본 idea 의 (Phase-aware + RadixAttention second-level + Green Context) 조합이 VLCache (encoder + KV cache exact-match) 대비 TTFT 추가 **-10% 이상** 개선? **-5% 미만이면 Tier-2 강등**.

#### 4️⃣ ATRIUM (2026-04-28 retain, v2-r55 origin)
- **RQ-A.1**: Qwen3-VL 의 layer-wise visual_attn_ratio 비대칭 (L17-21 = 24.5% / L0-7 = 2.6% / L22-35 = 9.8%) 을 system-level (Green Context CUDA 12.4 SM 분할) 로 활용 시, COLD layer 의 LPDDR5x 273 GB/s 대역폭 86% waste 가 HOT layer 로 재배치되어 decode TPS **+14% 이상** 달성되는가?
- **RQ-A.2**: GPU L2 SLC carveout (cudaCacheConfigure persistent) 을 HOT layer KV access pattern 에 맞춰 적용 시, L2 hit rate 가 vanilla vLLM 대비 **+25%pt 이상** 향상되고 LPDDR5x bandwidth pressure 가 **-30% 이상** 줄어드는가?
- **RQ-A.3**: DeepStack 미주입 layer (LLM L0-3) 의 visual KV alloc skip 시, KV memory footprint 가 baseline 대비 **-9% 이상** 절감되며 MMMU accuracy drop 이 **0pt** 유지되는가?
- **RQ-A.4 (cross-share)**: ATRIUM 의 LayerClassifier (visual_attn_ratio HOT/COLD/MEDIUM 분류) 를 PRISM-FOG-FX M4 의 prerequisite 신호로 cross-share 시, sequential development path (ATRIUM W1-12 prototype → PRISM-FOG-FX 추가 W13-24) 가 학생 1명 12-16주 budget 안에 fit 가능한가?

#### T1 STRATA-K-R
- **RQ-4.1**: VLM long-context workload (4K-32K visual tokens) 에서 layer score 기반 3-tier (L2 carveout / GDDR7|LPDDR5x / page-color cold) mapping 이 vanilla vLLM allocator 대비 effective KV capacity **+40% 이상** 증가? OOM batch size **+50% 이상** 증가?
- **RQ-4.2**: Tier-0 L2 carveout (RTX 5090 96MB / Jetson Thor 4MB) 에 anchor visual token (semantic salience top-K=128) pin 시, system prompt + first 128 token pin 대비 decode latency 추가 **-8% 이상** 감소?
- **RQ-4.3**: Jetson Thor LPDDR5x UMA 환경에서 page color (8-color partition) 별 access latency delta 60-120ns 측정 시, layer-class HOT/COLD 별 page color 매핑이 effective bandwidth **+15% 이상** 향상시키는가?
- **RQ-4.4**: tier promotion/demotion churn cost 가 hysteresis margin (10%) 적용 시 long-context workload prefill latency **≤ 5%** 안에서 유지되는가?

#### T2 HARBINGER-CLOVER-R
- **RQ-5.1**: ViT CLS embedding 또는 DeepStack inject signal entropy 가 낮을 때 (τ_low = 0.3) speculative draft length γ=8, 높을 때 (τ_high = 0.7) γ=2 동적 조절 시, 정적 γ=4 baseline 대비 acceptance rate **+18%pt 이상**, decode throughput **+20% 이상** 향상?
- **RQ-5.2**: middle layer (Qwen3-VL-2B L8-12 convergence layer) visual cluster heat (top-3 cluster attention 합산) ≥ 0.85 일 때 early-exit 적용 시 average layer skip **28% 이상**, MMMU drop **≤ 1.5pt** 이내? OCR/ChartQA 같은 visually ambiguous task 에서 task-adaptive threshold (0.92) 로 drop **≤ 2.5pt** 유지?
- **RQ-5.3**: Orin NX nvpmodel state (10W/15W/25W) sub-ms switching reliability 가 1000회 측정에서 failure rate **< 1%**, layer-boundary frequency lock 적용 시 J/token 정적 25W mode 대비 **-22% 이상** 감소?
- **RQ-5.4**: 3 mechanism 결합 시 Orin NX 25W envelope peak power 베이스 **-18% 이상**, 동시 throughput **+32% 이상** 향상?

#### T3 OBELISK-5090-R
- **RQ-6.1**: RTX 5090 32GB GDDR7 single-GPU 에서 Qwen3-VL-30B-A3B MoE NVFP4 quantization local serving 시, vanilla vLLM (uniform expert placement) 대비 expert routing layer-aware placement 가 decode tok/s **+24% 이상** 향상?
- **RQ-6.2**: Green Context CUDA 12.4 로 21760 CUDA core 를 vision (5632) + LLM (16128) 분할 후 vision-encode 종료 시 동적 합류 시, vision phase 길이 ≥ 100ms workload 에서 release+recreate 30ms overhead amortize 되어 prefill latency **-15% 이상** 감소?
- **RQ-6.3**: per-stage power cap (vision 350W, prefill 575W, decode 450W) 적용 시 nvidia-smi -pl 200ms overhead batch 단위 가정에서 average power 540W → 470W (-13% avg), peak 575W → 510W (-11%) 달성, throughput drop **≤ 5%** 이내?
- **RQ-6.4**: 32GB GDDR7 fit 검증 — Qwen3-VL-30B-A3B NVFP4 weight (~22GB) + 활성 KV cache + activation buffer 가 32GB 안에 들어가는가? long-context (16K+) OOM threshold 는?

## 2. Essential Reading (R57.2 — 본 연구 진입을 위한 핵심 5편)

> 본 ideation 의 baseline / closest competitor / scoop 위험 paper 5편. 학생 / AI agent 가 본 연구를 시작하기 전 반드시 읽어야 할 minimum set. 세부 ideation 별 추가 paper 는 각 tier1/tier2 idea 파일 §3.2 GAP 표 참조.

1. **vLLM PagedAttention** — Kwon et al., [arXiv:2309.06180](https://arxiv.org/abs/2309.06180), **SOSP 2023**
   - **읽어야 하는 이유**: KV cache block manager 의 16-token block paged allocation 이 본 세션 6 idea 모두의 application-level emulation 진입점. block table 자료 구조와 swap_in/swap_out semantics 이해 필수
   - **분담**: 6 idea 모두의 R52.1 Baseline Source

2. **Qwen3-VL Tech Report** — Alibaba, [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) — DeepStack architecture
   - **읽어야 하는 이유**: ViT intermediate output (default index [8, 16, 24]) → LLM L8/L16/L24 inject 메커니즘. 본 세션의 모든 VLM idea architecture 이해 필수. visual_attn_ratio 비대칭의 root cause 가 DeepStack 의 inject geometry
   - **분담**: PRISM-FOG-FX (DeepStack-aware NVFP4 anchor), BIVOUAC-SLATE-R (visual KV cluster), HARBINGER-CLOVER-R (visual entropy signal)

3. **NVFP4 NVIDIA Tech Blog** — [NVIDIA NVFP4 Tech Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
   - **읽어야 하는 이유**: Blackwell native NVFP4 (E2M1, 16-element block scaling, FP8 micro-block scale + FP32 second-level scale) format. 4-bit 4:8 sparsity 와의 결합. PRISM-FOG-FX / OBELISK-5090-R 의 baseline format
   - **분담**: PRISM-FOG-FX, OBELISK-5090-R

4. **VL-Cache** — Tu et al., [arXiv:2410.23317](https://arxiv.org/abs/2410.23317), **ICLR 2025**
   - **읽어야 하는 이유**: VLM KV cache eviction 의 published baseline. layer-adaptive sparsity budget allocation. 10% memory budget 으로 full cache 동등 정확도 + 7.08× decode speedup. BIVOUAC-SLATE-R 의 closest competitor
   - **분담**: BIVOUAC-SLATE-R, PRISM-VL-R, STRATA-K-R

5. **Sarathi-Serve** — Agrawal et al., [USENIX OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal)
   - **읽어야 하는 이유**: Continuous batching + chunked prefill 의 published baseline. Mistral-7B 단일 A100 2.6× serving capacity. STRATA-K-R / PRISM-VL-R / OBELISK-5090-R 의 system baseline
   - **분담**: STRATA-K-R, PRISM-VL-R, OBELISK-5090-R

## 3. 연구 개요 + 기존 GAP outline (R57.3, 개조식)

### 3.1 연구 개요 (5-10 bullet)

- **본 세션 도메인**: 최신 VLM (Qwen3-VL / LLaVA-Next / InternVL3 / DeepSeek-VL2 / Cosmos-Reason2) 의 edge inference on single-GPU
- **타겟 workload**: Qwen3-VL-2B/4B/8B/30B-A3B-Instruct (DeepStack inject [8,16,24]), LLaVA-Next-7B, InternVL3-8B, MMMU/DocVQA/ChartQA/VideoMME benchmark
- **타겟 platform**: RTX 5090 (32GB GDDR7, 575W TDP, 96MB L2) / Jetson Thor T5000 (LPDDR5x UMA 273 GB/s, 60-130W, 2070 FP4 TFLOPS) / Jetson Orin NX 16GB (LPDDR5 102-204 GB/s, 10-25W)
- **핵심 측정 metric**: decode TPS / TTFT / Energy/token / Memory footprint / Power envelope (avg + peak)
- **본 세션의 hypothesis**: VLM 의 layer-wise visual attention 비대칭 (HOT L17-21 = 24.5% / COLD L0-7 = 2.6%) 과 visual token semantic similarity (cluster-level cosine, frame-cross redundancy) 를 system-level (SM partition / KV layout / DVFS / mixed precision) 에서 활용하면 Performance + Energy 동시 large gain 가능
- **이전 세션 evolution base**: 2026-04-27 vlm-llm-asym-dual-jetson-v2-r55 의 ATRIUM (Green Context SM partition + L2 carveout + DeepStack alloc skip) + BREAKWATER-T (ViT/LLM split + tap activation 4-bit channel-wise) — 본 세션은 단일 system 환경 (dual-Jetson 배제) 으로 reframe + Blackwell NVFP4 native HW 활용 추가
- **기여 범위**: lab single-system implementation + Jetson Thor / RTX 5090 / Orin NX dev kit 1 unit, multi-tenant cluster scope 는 R20-γ.3 검증 후만 보조

### 3.2 기존 연구의 한계 (5-10 bullet, 외부 published / arxiv reference 의무, R56.1 자체 인용 금지)

- **GAP-1 (Layer-uniform precision)**: vLLM ([SOSP 2023, arXiv:2309.06180](https://arxiv.org/abs/2309.06180)), TensorRT-LLM 모두 layer-uniform precision (BF16/FP16) 가정 — VLM 의 layer-wise visual attention 비대칭 미반영. NVFP4 mixed precision (FGMP [arXiv:2504.14152](https://arxiv.org/abs/2504.14152)) 도 LLM-only block-level, **VLM DeepStack inject geometry 미고려**. MicroMix ([arXiv:2508.02343](https://arxiv.org/abs/2508.02343)) 는 channel-mix LLM-only. NVFP4-QAD ([arXiv:2601.20088](https://arxiv.org/abs/2601.20088)) 는 uniform NVFP4 distillation
- **GAP-2 (KV cache layer-uniform)**: VL-Cache ([ICLR 2025, arXiv:2410.23317](https://arxiv.org/abs/2410.23317)) 는 token-level sparsity-based eviction 이지만 **video/multi-frame VLM 의 cross-frame redundancy 미활용**. ClusterKV ([arXiv:2412.03213](https://arxiv.org/abs/2412.03213)) 는 LLM-only layer-uniform cluster, **VLM-specific layer adaptation 부재**. Mosaic ([arXiv:2604.10060](https://arxiv.org/abs/2604.10060), 2026-04 concurrent) 는 full-layer cluster (HOT-only 미차별) + datacenter
- **GAP-3 (Power-envelope blind)**: SGLang RadixAttention ([NeurIPS 2024, arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) 은 prefix sharing 만, **edge power envelope adaptive 미반영**. Spec-LLaVA ([arXiv:2509.11961](https://arxiv.org/abs/2509.11961)) 는 speculative decode 만, DVFS 부재. CLONE ([arXiv:2506.02847](https://arxiv.org/abs/2506.02847), USENIX ATC 2025) 은 LLM-only layer-boundary DVFS, **VLM speculative + early-exit 결합 부재**
- **GAP-4 (Phase-uniform caching)**: SGLang HiCache (LMSYS 2025-09 production) 와 VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977)) 모두 **single-policy caching, encode/prefill/decode 별 다른 policy 부재**. SimCache (CVPR 2025W) 는 frame-level similarity, token-level radix tree integration 없음
- **GAP-5 (Static KV layout)**: PagedAttention ([SOSP 2023, arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) 은 single-tier KV, **GPU L2 carveout / GDDR7 / LPDDR5x UMA 의 multi-tier 미활용**. LayerKV ([arXiv:2410.00428](https://arxiv.org/abs/2410.00428)) 는 layer-wise allocation 만, physical-tier mapping 부재. AttAcc (ASPLOS 2024) 은 PIM 만
- **GAP-6 (Single-GPU large MoE)**: Qwen3-VL-30B-A3B MoE 는 cluster 가정 — RTX 5090 32GB GDDR7 single-GPU local serving 미타겟. DynaExq ([arXiv:2511.15015](https://arxiv.org/abs/2511.15015), 2025-11 concurrent) 는 **precision allocation 만, physical GDDR placement 부재 + VLM (Qwen3-VL) 미평가**. Nova ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301)) 는 datacenter A100 만

### 3.3 본 세션 idea 가 위 GAP 에 어떻게 대응하는가 (mapping 표)

| GAP | Tier-1 idea 대응 | Tier-2 idea 대응 |
|-----|----------------|----------------|
| GAP-1 (Layer-uniform precision) | 🥇 PRISM-FOG-FX (M1 DALMP / M2 4:8 sparsity / M3 KSF-VCD / M4 LayerClassifier) + 4️⃣ ATRIUM (M1 LayerClassifier — visual_attn_ratio HOT/COLD/MEDIUM cross-share) | T3 OBELISK-5090-R (per-stage power cap + NVFP4 + MoE expert placement) |
| GAP-2 (KV cache layer-uniform) | 🥈 BIVOUAC-SLATE-R (M1 HSCV HOT-only cluster / M2 CFCR cosine cross-frame / M3 LACB layer-adaptive k) + 4️⃣ ATRIUM (M3 DeepStack L0-3 alloc skip — visual KV memory eff.) | T1 STRATA-K-R (M1 LSS / M2 3TMM-AVT) |
| GAP-3 (Power-envelope blind) | 🥇 PRISM-FOG-FX (M3 visual-context DVFS slip) | T2 HARBINGER-CLOVER-R (M3 PEAFL power-envelope locking) |
| GAP-4 (Phase-uniform caching) | 🥉 PRISM-VL-R (M1 phase-aware LSH 16/8/0) | — |
| GAP-7 (Layer-uniform SM/L2 partition) | 4️⃣ ATRIUM (M2 Green Context SM 1500/1060 split + L2 SLC carveout — system-level layer asymmetry 활용) | — |
| GAP-5 (Static KV layout) | — | T1 STRATA-K-R (M1-M3 stratified + page-color UMA bank micro-tier) |
| GAP-6 (Single-GPU large MoE) | — | T3 OBELISK-5090-R (M1 SPVL Green Context / M2 MERL expert placement / M3 DPCS) |

## 4. Implementation-Priority Decision Tree (R14.4)

> **양식 A markdown native ASCII flowchart default + 양식 C 4-branch 액션 표 첨부**. 각 idea 의 실험 단계 + 분기 (Pass / Below / Critical fail / Outperform) + inter-idea dependency. 학생이 12-16주 내 어느 idea 를 먼저 prototype 하고 결과에 따라 어떻게 자원 재배치할지에 대한 의사결정 트리.

### 4.1 양식 A — ASCII Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Week 1-3: Preliminary (모든 idea 공통)                                    │
│  - vLLM v0.11.0 fork + Jetson Thor / RTX 5090 / Orin NX 환경 setup       │
│  - Qwen3-VL-8B / LLaVA-Next-7B / InternVL3-8B HF model 검증               │
│  - 100-shot calibration set 준비 (MMMU + ChartQA + DocVQA)                │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Week 4-7: MVP Prototype (병렬 6 idea)                                     │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├─ 🥇 Tier-1 #1 PRISM-FOG-FX (Jetson Thor)
        │      ├─ Pass (decode +14-18% / Energy -18-22%) ──→ W8-12 full eval
        │      ├─ Below (10-30% short) ──→ Tier-2 강등 (M2 4:8 drop, 3-mech)
        │      ├─ Critical (>30% short) ──→ M3 DVFS slip drop, M1+M4 retain
        │      └─ Outperform (>20% better) ──→ Tier-1 retain + Tier-2 spinoff
        │
        ├─ 🥈 Tier-1 #2 BIVOUAC-SLATE-R (Jetson Thor)
        │      ├─ Pass (decode +30-35% / KV mem -50-65%) ──→ W8-12 full eval
        │      ├─ Below (Mosaic 차별 axis 5개 중 ≤3 verified) ──→ Tier-2 강등
        │      ├─ Critical (Mosaic >65% overlap 추가 발견) ──→ drop 또는 M1+M3 만
        │      │                                                  retain (CFCR drop)
        │      └─ Outperform (Mosaic 대비 +20% 우위) ──→ Tier-1 retain + ICML spinoff
        │
        ├─ 🥉 Tier-1 #3 PRISM-VL-R (RTX 5090)
        │      ├─ Pass (TTFT -22-30% vs vanilla SGLang) ──→ RQ-3.4 W12 검증
        │      │       ├─ vs VLCache 추가 -10%↑ ──→ Tier-1 retain (OSDI/SOSP)
        │      │       ├─ vs VLCache -5~-10% ──→ Tier-1 borderline (MLSys poster)
        │      │       └─ vs VLCache -5% 미만 ──→ Tier-2 강등 (EuroSys/FAST)
        │      ├─ Below (TTFT -10-20%) ──→ Tier-2 강등 (M2 RA branch drop)
        │      └─ Outperform (TTFT -35%↑) ──→ Tier-1 retain + workshop spinoff
        │
        ├─ 4️⃣ Tier-1 #4 ATRIUM (Jetson Thor 128GB, 2026-04-28 retain)
        │      ├─ Pass (decode +14% / Energy -12% / DeepStack L0-3 alloc skip 0pt drop) ──→ HPCA 2027 retain
        │      ├─ Below (Green Context SM partition overhead > +5% prefill) ──→ libsmctrl secondary path
        │      ├─ Critical (visual_attn_ratio measurement noise > σ=10%pp drift) ──→ M1 LayerClassifier 만 retain,
        │      │                                                                       PRISM-FOG-FX M4 enabler 로만 활용
        │      └─ Outperform (decode +18%↑ + L2 hit +30%pp) ──→ HPCA 2027 retain +
        │                                                       PRISM-FOG-FX 와 sequential development path 권장
        │
        ├─ T1 STRATA-K-R (RTX 5090 + Jetson Thor)
        │      ├─ Pass (effective KV +40-55%) ──→ W8-12 full eval
        │      ├─ Below (RQ-4.3 page-color delta < 60ns) ──→ M3 PCA-UBP drop,
        │      │                                              M1+M2 만 retain
        │      ├─ Critical (UMA 4-tier collapse) ──→ drop → CARILLON 흡수 시도
        │      └─ Outperform (anchor visual L2 pin gain >20%) ──→ Tier-1 격상 시도
        │
        ├─ T2 HARBINGER-CLOVER-R (Jetson Orin NX 16GB)
        │      ├─ Pass (decode +30-32% / Energy -22-28%) ──→ W8-12 full eval
        │      ├─ Below (visually ambiguous task accuracy drop > 3pt) ──→ M2 LCEH
        │      │                                                       drop, M1+M3 만
        │      ├─ Critical (nvpmodel sub-ms switching failure rate >5%) ──→
        │      │                                       M3 PEAFL drop, ISLPED short
        │      └─ Outperform (peak power -25%↑) ──→ Tier-1 격상 시도
        │
        └─ T3 OBELISK-5090-R (RTX 5090 single)
               ├─ Pass (decode +24% / power -13%) ──→ W8-12 full eval
               ├─ Below (DynaExq 추가 35-45% overlap 의 차별 axis 입증 실패) ──→
               │                                       M2 MERL drop, M1+M3 만 retain
               ├─ Critical (575W TDP thermal runaway 발생) ──→ M3 DPCS 보강,
               │                              vision/decode 만 cap, prefill 무관
               └─ Outperform (+30% throughput) ──→ Tier-1 격상 + MLSys 진입
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Week 8-12: Full Evaluation + Tier 확정                                    │
│  - Baseline reproduction (FGMP / MicroMix / NVFP4-QAD / VL-Cache /        │
│    ClusterKV / Mosaic / VLCache / SimCache / Spec-LLaVA / DynaExq)        │
│  - 5-axis 측정 (Performance / Energy / Memory / Power / Accuracy)          │
│  - Cross-platform validation (Jetson Thor ↔ RTX 5090 ↔ Orin NX)           │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Week 13-16: Paper draft + camera-ready                                    │
│  - Tier-1: MLSys / ASPLOS / NeurIPS / OSDI / SOSP submission              │
│  - Tier-2: DAC / DATE / EuroSys / FAST / ISLPED submission                │
│  - Inter-idea dependency 정리: PRISM-FOG-FX 의 LayerClassifier (M4)         │
│    가 BIVOUAC-SLATE-R / STRATA-K-R / HARBINGER-CLOVER-R 의 layer-class      │
│    입력으로 cross-share                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 양식 C — 4-Branch Action 표 (6 idea × 4 branch × action)

| Idea | Pass (목표 달성) | Below (10-30% short) | Critical (>30% short 또는 feasibility 실패) | Outperform (>20% better) |
|------|----------------|---------------------|-------------------------------------------|------------------------|
| 🥇 PRISM-FOG-FX | W8-12 full eval Jetson Thor + RTX 5090 secondary, MLSys submission | Tier-2 강등, M2 4:8 sparsity drop, 3-mech (M1+M3+M4) 로 축소 | M3 visual-context DVFS slip drop, M1 DALMP + M4 LayerClassifier 만 retain (FGMP/MicroMix 와 head-to-head 만) | Tier-1 retain + standalone variant Tier-2 spinoff (KSF-VCD 단독으로 ISLPED) |
| 🥈 BIVOUAC-SLATE-R | W7-8 Mosaic reproduction 후 차별 5 axis (HOT-only / edge Jetson / 1-bit mask + 4-bit delta / RAFT-cosine cost / layer-adaptive k) ablation NeurIPS submission | Mosaic 차별 axis 5개 중 ≤ 3 verified → Tier-2 강등 (ICML workshop) | Mosaic > 65% overlap 추가 발견 → drop 또는 M1+M3 만 retain (CFCR drop, single-shot only) | Mosaic 대비 +20% 우위 → Tier-1 retain + ICML 2026 spinoff (cluster centroid theory) |
| 🥉 PRISM-VL-R | RQ-3.4 W12 검증 시 vs VLCache 추가 -10%↑ → Tier-1 retain OSDI/SOSP | vs VLCache -5~-10% → Tier-1 borderline MLSys poster, RA-2VPB 만 강조 | vs VLCache -5% 미만 → Tier-2 강등 EuroSys/FAST, Phase-aware policy 단독 contribution | vs VLCache -20%↑ → Tier-1 retain + workshop on similarity caching |
| T1 STRATA-K-R | RQ-4.3 page-color delta 60-120ns 측정 → effective bandwidth +15% 달성 → ASPLOS/DAC submission | RQ-4.3 page-color delta < 60ns → M3 PCA-UBP drop, M1+M2 (3-tier physical + L2 anchor pin) 만 retain | UMA 4-tier collapse → drop, CARILLON 흡수 시도 (이미 W3 후보 미선정 idea) | Anchor visual L2 pin gain > 20% → Tier-1 격상 시도, KVQuant 의 spatial pinning generalization 으로 NeurIPS spinoff |
| T2 HARBINGER-CLOVER-R | Orin NX 25W envelope decode +32% / Energy -28% / peak -18% 달성 → DATE/ISLPED submission | visually ambiguous task accuracy drop > 3pt → M2 LCEH drop, M1+M3 (speculative + DVFS) 만 retain → IEEE CAL | nvpmodel sub-ms switching failure rate > 5% → M3 PEAFL drop, M1+M2 만 EMNLP findings | Peak power -25%↑ → Tier-1 격상 시도 ISLPED 풀논문 |
| T3 OBELISK-5090-R | RTX 5090 + Qwen3-VL-30B-A3B decode +24% / power -13% 달성 → MLSys/DAC submission | DynaExq 추가 차별 axis (Green Context vision/LLM 분할 + per-stage power cap) 입증 실패 → M2 MERL drop, M1+M3 만 retain | 575W TDP thermal runaway 발생 → M3 DPCS 보강 (vision/decode 만 cap, prefill 무관) | +30% throughput → Tier-1 격상 + MLSys 풀논문 진입, MoE-on-consumer-GPU 단독 axis |

## 5. Tier-1 Top 4 (요약 + contribution bullet R15-β) — 2026-04-28 ATRIUM retain 포함

| Rank | Title | Score | 5-axis | 링크 |
|------|-------|-------|--------|------|
| 🥇 | PRISM-FOG-FX | nov 7.5 / diff 7.5 / impact 9.0 | Performance +18-25% / Energy -22% / Memory -32% / Power -9% | [tier1/01-prism-fog-fx.md](tier1/01-prism-fog-fx.md) |
| 🥈 | BIVOUAC-SLATE-R | nov 6.5 / diff 8.0 / impact 8.0 | decode +35% / KV memory -65% / Energy -18% | [tier1/02-bivouac-slate-r.md](tier1/02-bivouac-slate-r.md) |
| 🥉 | PRISM-VL-R | nov 6.0 / diff 8.5 / impact 9.0 | TTFT -22-30% / KV -30-45% / Energy -15-20% | [tier1/03-prism-vl-r.md](tier1/03-prism-vl-r.md) |
| 4️⃣ | ATRIUM (retain) | nov 7.7 / diff 7.5 / impact 8.0 | Performance +14% decode / Energy -12% / Memory eff. (DeepStack L0-3 alloc skip) | [tier1/04-atrium.md](tier1/04-atrium.md) |

### 5.1 🥇 PRISM-FOG-FX — contribution bullet (R15-β, 5 항목)

- **C1 (Mechanism unique)**: DeepStack inject layer (Qwen3-VL L8/L16/L24) 를 quantization-critical anchor 로 framing — FGMP block-level (LLM-only) / MicroMix channel-mix / NVFP4-QAD uniform 모두 미고려한 VLM-specific layer geometry 활용. **DALMP (DeepStack-aware NVFP4/FP8/INT4 Layer-wise Mixed Precision)** 4-class routing
- **C2 (Multi-axis gain)**: Performance +18-25% (NVFP4 native FLOPS + 4:8 sparsity HOT layer dispatch) + Energy -22% (decode-phase memory-bound DVFS slip + KV cache scale freeze) + Memory -32% (layer-mix INT4 cold + NVFP4 hot anchor) **동시** 달성. CLONE LLM-only DVFS 와 차별
- **C3 (Implementation feasibility)**: vLLM v0.11.0 NVFP4 native + TensorRT-LLM Edge build + nvpmodel CLI Python wrapper + CUTLASS 3.5 sparse epilogue — 모두 application-level (kernel patch 0). Jetson Thor T5000 dev kit 한 대로 W1-W12 prototype 가능
- **C4 (Cross-share enabler)**: M4 LayerClassifier (visual_attn_ratio HOT/COLD/MEDIUM 자동 분류) 가 BIVOUAC-SLATE-R / STRATA-K-R / HARBINGER-CLOVER-R 의 layer-class 입력으로 cross-share — 본 세션 6 idea 중 4 idea 의 enabler
- **C5 (Venue alignment)**: MLSys 2026 / ASPLOS 2026 (primary), NeurIPS 2026 efficient ML track (secondary). R56.2 published 77% 충족 (FGMP/MicroMix/NVFP4-QAD 등 closest competitor 명시 + Atom ASPLOS 2024 / KVQuant NeurIPS 2024 / SparseVLM ICML 2025 baseline)

### 5.2 🥈 BIVOUAC-SLATE-R — contribution bullet (5 항목)

- **C1 (Mechanism unique vs Mosaic concurrent)**: HOT layer (visual_attn_ratio > 0.15) **만** cluster centroid 보유 (Mosaic 2026-04-11 = full-layer cluster) + 1-bit membership mask + 4-bit delta vector lossy 압축 + layer-adaptive cluster k (shallow 64 / mid 16 / deep 8). 5 axis 차별
- **C2 (Cost-explicit cross-frame reuse)**: cluster centroid cosine (80us, k=16 × 1280-dim matmul) 가 RAFT optical flow (10ms, Sali-Cache arXiv:2602.14236) 대비 **125× faster** — edge-fit 명시. CacheFlow frame-level 보다 fine-grained false-negative 30% 감축 가능
- **C3 (Memory + Performance dual gain)**: visual KV memory -65% (HOT-only cluster + cross-frame delta) + decode tok/s +35% (8-frame video) — Sali-Cache 의 2.20× memory compression 와 동급, 그러나 RAFT 비용 없음. ClusterKV 의 LLM-only 80% reduction 보다 VLM-specific
- **C4 (Edge platform 검증)**: Jetson Thor T5000 primary (UMA 273 GB/s) + RTX 5090 secondary (long-context) — Mosaic 의 H800/A40 datacenter 와 platform 차별. R56.2 published 74% 충족
- **C5 (Venue alignment)**: NeurIPS 2026 efficient VLM track (primary) / ICML 2026 (secondary) / MICRO 2026 (hardware integration sub-claim 강화 시). W7-W8 Mosaic reproduction + 차별 5 axis 정량 ablation 의무

### 5.3 🥉 PRISM-VL-R — contribution bullet (5 항목)

- **C1 (Phase-aware policy unique)**: encode/prefill/decode 3 phase 별 LSH hyperplane 가변 (encode 16 / prefill 8 / decode 0) — VLCache (single content-hash exact match) 와 차별. SimCache (frame-level similarity) 와 다름 (token-level radix tree integration)
- **C2 (Architectural extension)**: SGLang RadixAttention 의 token-level radix tree 를 root second-level branch 가 visual-semantic-hash 로 split 되도록 확장 — dashboard agent (동일 image base + 다른 prompt 50-70% ratio) 에서 cross-request KV reuse 가 자연스러움. KVFlow / HiCache 와 차별
- **C3 (Workload-bounded gain)**: TTFT -22~30% / Memory -30~45% (visual prefix overlap 50%+ workload 한정 — multi-camera surveillance / dashboard agent / video VQA / repeated UI). cold start single-shot 효과 미미 명시
- **C4 (System-level integration)**: Green Context 8-SM partition 에 LSH compute 격리 + decode latency 영향 ≤ 1% — PD-Multiplexing (LMSYS 2025-09) 의 prefill/decode 분리에 third role (LSH compute) 추가
- **C5 (Tier-1 차별 검증 분기)**: §9 decision tree 에서 RQ-3.4 (W12) 결과에 따라 Tier-1 (vs VLCache 추가 -10%↑) / Tier-2 (vs VLCache -5% 미만 → EuroSys/FAST) 분기. R56.2 published 64% 경계 (EuroSys 2025 / SOSP 2025 multimodal serving 1편 추가 시 70%+)

### 5.4 4️⃣ ATRIUM — contribution bullet (5 항목, 2026-04-28 retain)

> **Retain note**: 본 idea 는 2026-04-27 v2-r55 세션의 Tier-1 lead idea 로 도출. 사용자 명시 요청에 따라 본 2026-04-28 세션의 Tier-1 4번째 idea 로 retain. mechanism 변경 없음 (v2-r55 그대로). 상세는 [tier1/04-atrium.md](tier1/04-atrium.md).

- **C1 (Mechanism unique — system-level layer asymmetry)**: VLM 의 layer-wise visual_attn_ratio 비대칭 (L17-21 = 24.5% HOT / L0-7 = 2.6% COLD / L22-35 = 9.8% MEDIUM) 을 algorithm-level (Q Cache decode skip) 이 아닌 **system-level SM partition + L2 carveout** 으로 활용. KVTuner (per-layer KV quant only) / MIG (정적 partition) 와 axis 직교
- **C2 (Multi-axis gain — Performance + Energy + Memory)**: Performance +14% decode (Green Context SM 1500/1060 split, HOT layer 우선 SM allocation) + Energy -12% (COLD layer SM down-sized) + Memory eff. (DeepStack L0-3 alloc skip — visual context 미주입 layer 의 visual KV 자체 alloc 안 함) 동시 달성
- **C3 (Implementation feasibility — single AGX Thor)**: vLLM 0.7.x source modification + Green Context CUDA 12.4 공식 SM partition API + libsmctrl R45.2 secondary fallback (sweet-spot exploration). **Simulator 미사용** (실기 only). single AGX Thor 128GB dev kit 한 대로 12-16주 prototype
- **C4 (Cross-share with PRISM-FOG-FX)**: ATRIUM 의 M1 LayerClassifier (HOT/COLD visual_attn_ratio 자동 분류) 가 본 세션 PRISM-FOG-FX M4 LayerClassifier 와 cross-share dependency — 동일 visual_attn_ratio measurement infrastructure 공유 (`tools/atrium_calibrate.py::measure_layer_attn`). 학생이 ATRIUM prototype 후 PRISM-FOG-FX 의 NVFP4 mixed precision routing 을 추가 적용 가능 — sequential development path
- **C5 (Venue alignment)**: HPCA 2027 (primary) / MICRO 2027 / ASPLOS 2027. Green Context CUDA 12.4 공식 API 활용 + 비-PIM solution + AGX Thor edge platform (자동차/robotics 산업 정합)

## 6. Tier-2 독립 Top 3 (R15-β 동일 형식)

| Rank | Title | Score | 5-axis | 링크 |
|------|-------|-------|--------|------|
| T1 | STRATA-K-R | nov 6.5 / diff 7.5 / impact 7.0 | Performance -15-22% latency / Memory eff. +40-55% / Energy -10-15% | [tier2/01-strata-k-r.md](tier2/01-strata-k-r.md) |
| T2 | HARBINGER-CLOVER-R | nov 7.0 / diff 7.5 / impact 6.5 | decode +32% / Energy -28% / Peak power -18% | [tier2/02-harbinger-clover-r.md](tier2/02-harbinger-clover-r.md) |
| T3 | OBELISK-5090-R | nov 5.5 / diff 6.5 / impact 7.5 | decode +24% / Power -13% / TTFT -15% | [tier2/03-obelisk-5090-r.md](tier2/03-obelisk-5090-r.md) |

### 6.1 T1 STRATA-K-R — contribution bullet (4 항목)

- **C1 (Page-color UMA bank micro-tier unique)**: Jetson Thor LPDDR5x UMA 환경에서 page color (8-color partition, Linux kernel page-coloring) 별 access latency delta 60-120ns 활용 — VLM 도메인 검색 범위 내 unique. NVIDIA ICMS / Bluefield-4 CMX 의 datacenter 4-tier 와 edge-equivalent 차별
- **C2 (Anchor visual token L2 pinning)**: KVQuant 의 attention sink first-token-FP16 anchor concept 을 visual token 에 layer-stratified spatial pinning 으로 일반화. RTX 5090 96MB L2 carveout (Jetson Thor 4MB) 에 semantic salience top-K=128 token pin
- **C3 (Long-context workload 한정)**: long-context VLM (>4K visual tokens) workload 에서 effective KV capacity +40-55% / OOM batch size +50% — short-prompt 효과 미미 명시. R56.2 published 86% (LayerKV / InfiniGen / Mooncake / KVTuner / KVQuant / VL-Cache / Atom ASPLOS 2024)
- **C4 (Venue alignment)**: ASPLOS 2026 / DAC 2026 / EuroSys 2026 / IEEE CAL. R45 risk 6/10 (page-color delta gain ceiling 낮을 risk → RQ-4.3 검증 필수)

### 6.2 T2 HARBINGER-CLOVER-R — contribution bullet (4 항목)

- **C1 (Visual-cluster-heat early-exit unique)**: middle layer (Qwen3-VL-2B L8-12 convergence layer, VisionZip 측정) visual cluster heat (top-3 cluster attention 합산 ≥ 0.85) signal 로 early-exit — FREE early-exit (confidence-only) 와 차별. PyramidDrop / VisionZip 의 convergence-layer insight 를 layer-skip signal 로 reuse
- **C2 (Vision-grounded speculative draft length)**: ViT CLS embedding entropy + DeepStack inject signal entropy 로 draft length γ 동적 조절 (γ=8 if visually unambiguous / γ=2 if ambiguous) — Spec-LLaVA (draft model confidence-only) 와 차별. acceptance rate +18%pt 향상
- **C3 (Power-envelope locking)**: Orin NX nvpmodel state (10W/15W/25W) sub-ms switching reliability 활용 + early-exit triggered DVFS — CLONE (LLM-only layer-boundary DVFS) 보다 power-envelope-aware. peak power -18% / J/token -28%
- **C4 (Edge tier specific)**: Orin NX 16GB power-limited tier 한정 약점 명시 (큰 모델은 Jetson Thor / RTX 5090). DATE 2026 / ISLPED 2026 / IEEE CAL / EMNLP 2026 efficient track. R45 risk 6/10 (visually ambiguous task drop, task-adaptive threshold W5 calibration 의무)

### 6.3 T3 OBELISK-5090-R — contribution bullet (4 항목)

- **C1 (RTX 5090 single-GPU large MoE inflection point)**: Qwen3-VL-30B-A3B (active 3B per token) MoE 가 RTX 5090 32GB GDDR7 에 NVFP4 quantization 으로 fit (~22GB weight + active expert 1GB + 16K KV 4-8GB = 27-31GB ≤ 32GB) — consumer GPU 가 datacenter-class large MoE 를 local serving 가능한 inflection point
- **C2 (MoE expert layer-aware GDDR7 placement)**: layer 별 expert routing histogram 기반 active expert 를 GDDR7 hot-zone (first 16GB), inactive expert cold-zone (second 16GB) — DynaExq (precision allocation) 와 orthogonal axis. Ban&Pick (routing-only) vs physical placement 차별
- **C3 (Per-stage power cap)**: vision-encode (250-300W) → 350W cap, prefill (450-520W) → 575W full, decode (380-420W) → 450W cap — nvidia-smi -pl 200ms overhead 감안 batch 단위 적용. PCIe 12VHPWR 600W envelope 안 thermal margin 확보
- **C4 (Concurrent work navigation)**: DynaExq (arXiv:2511.15015, 2025-11) 와 35-45% adjacent zone overlap 이지만 axis nature 가 orthogonal (precision vs physical placement). DynaExq 추가 baseline 의무 + R56.2 published 47% → 65%+ 보강 (ASPLOS 2025 NPU + ISCA 2024 추가 W7). 575W TDP thermal margin 검증 필수

## 7. 미선정 아이디어 요약

본 세션 Phase 1 → 2 → 1' → 2' 과정에서 13 candidate 중 6 개 선정 + 4 개 명시적 drop + 3 개 흡수.

- **CARILLON** (3-Phase SM Map + Layer-wise Concurrent Batching) — Nova ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301)) 와 80%+ mechanism overlap. Critical scoop, drop
- **BREAKWATER-T-R** (single Jetson Thor MIG 7-partition) — Jetson Thor MIG T5000 single-partition only (2026-04 시점 NVIDIA 공식 confirmation), infeasible. drop
- **TIDEGATE** (UMA bandwidth gating + L2 carveout duty cycle + nvpmodel hopper) — Nova + PD-Multiplexing 와 80%+ overlap, R56.2 30% 미달, gain ceiling 작음. drop, M3 nvpmodel hopper 일부는 PRISM-FOG-FX M3 KSF-VCD 로 흡수
- **HARBOR-DLA** (Jetson Orin AGX DLA offload of ViT patch-embed + KV value-projection) — DLA 2.0 LayerNorm 미지원으로 ViT 30% 미만만 offloadable, AGX Orin (Ampere) 한정 → Jetson Thor (Blackwell) 의 본 세션 4-axis platform set 과 mismatch (R20-γ). drop

상세는 [unselected.md](unselected.md) 참조.

## 8. 참고 / 관련 자료

- 상세 Phase 로그: [../../sessions/2026-04-28-mode1-vlm-edge-layerwise-context.md](/research-wiki/2026-04/vlm-edge-layerwise-context)
- ATRIUM/BREAKWATER-T (이전 세션 evolution base): [../2026-04-27-mode1-vlm-llm-asym-dual-jetson-v2-r55/](../2026-04-27-mode1-vlm-llm-asym-dual-jetson-v2-r55/)
- Phase 1 staging: `/tmp/2026-04-28-vlm-edge/staging-{ai-optimization,hw-pim,legacy-system}.md`
- Phase 2 review: `/tmp/2026-04-28-vlm-edge/review-{novelty,differentiation,impact,phase2prime}.md`

## 9. 약어 / 핵심 용어 풀이 (Glossary, R57.4 — 맨 뒤)

> 본 Summary 에서 사용된 도메인 약어 / Polysemous term / Idea Metaphor 통합 풀이. 본문에서 모르는 용어 발견 시 **CTRL+F (Mac: Cmd+F)** 로 본 섹션에서 검색.

### 9.1 도메인 약어

- **DeepStack** — Qwen3-VL ([arXiv:2406.04334](https://arxiv.org/abs/2406.04334), NeurIPS 2024) ViT intermediate output (default index [8, 16, 24]) 을 LLM 여러 layer 에 inject 하는 architecture
- **NVFP4** — Blackwell native E2M1 4-bit float format (16-element block scaling, FP8 micro-block scale + FP32 second-level scale). [NVIDIA Tech Blog 2025](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- **PagedAttention** — vLLM ([SOSP 2023, arXiv:2309.06180](https://arxiv.org/abs/2309.06180)) KV cache 16-token block 단위 paged allocation
- **DVFS** — Dynamic Voltage and Frequency Scaling
- **SM partition** — Streaming Multiprocessor partition (NVIDIA Green Context CUDA 12.4 + libsmctrl)
- **LSH** — Locality-Sensitive Hashing (random projection hyperplane → bit signature)
- **DLA** — Deep Learning Accelerator (Jetson 의 NVDLA 2.0)
- **MIG** — Multi-Instance GPU (NVIDIA Ampere/Hopper/Blackwell datacenter GPU 분할 기능, Jetson Thor T5000 은 2026-04 현재 single-partition only)
- **MMMU** — Massive Multi-discipline Multimodal Understanding ([arXiv:2311.16502](https://arxiv.org/abs/2311.16502)) — VLM 종합 benchmark
- **TPS** — Tokens Per Second
- **TTFT** — Time To First Token
- **UMA** — Unified Memory Architecture (CPU+GPU 가 같은 physical memory pool 공유, Jetson Thor LPDDR5x 273 GB/s)
- **GDDR7** — RTX 5090 의 graphics memory (28Gbps per pin × 512-bit bus = 1.79 TB/s)
- **L2 carveout** — GPU L2 cache 의 일부를 specific block 에 persistently 할당 (`cudaAccessPolicyWindow`, RTX 5090 96MB / Jetson Thor 4MB)
- **MoE** — Mixture of Experts (Qwen3-VL-30B-A3B = 30B total, active 3B per token)
- **ViT** — Vision Transformer (image patch embedding + transformer encoder)
- **KV cache** — Key-Value cache for transformer attention (decode 단계 의 main memory pressure)
- **PCIe 12VHPWR** — RTX 5090 의 power connector (600W envelope, 일부 unit thermal runaway 보고)
- **RadixAttention** — SGLang ([NeurIPS 2024, arXiv:2312.07104](https://arxiv.org/abs/2312.07104)) token-level prefix tree KV cache reuse
- **CUTLASS** — NVIDIA CUDA Templates for Linear Algebra Subroutines (3.5+ sparse epilogue 지원)
- **NeoVerse-V3AE** — Jetson Thor 의 Arm CPU core (14-core)

### 9.2 Polysemous Term Disambiguation (R51-α 통합)

- **channel** (본 세션 의미) — **GDDR7/HBM3 pseudo-channel** 또는 **vLLM PagedAttention block channel** 또는 **MicroMix 의 quantization channel-mix unit**. CXL channel / CNN feature channel 과 무관
- **block** (본 세션 의미) — **vLLM PagedAttention block** (16 token, 64KB BF16 / 32KB NVFP4) 또는 **NVFP4 micro-block** (16 element). DRAM block / CUDA thread block 과 무관
- **layer** (본 세션 의미) — **transformer layer L_i** (Qwen3-VL-8B = 32 LLM layer + 24 ViT layer). Network OSI / cache hierarchy layer / DLA layer 와 무관
- **cluster** (본 세션 의미) — **K-means visual KV cluster centroid** (BIVOUAC-SLATE-R) 또는 **HOT/COLD layer class** (PRISM-FOG-FX LayerClassifier). Datacenter cluster (multi-node) 와 무관
- **scale** (본 세션 의미) — **NVFP4 quantization scale** (per-block FP8 + per-tensor FP32) 또는 **scale-freeze (KSF-VCD M3)**. SI unit scale 과 무관
- **tier** (본 세션 의미) — **STRATA-K-R 의 memory tier** (L2 / GDDR7|UMA / page-color cold) 또는 **본 세션 idea Tier-1/Tier-2** (venue priority). Container tier 와 무관

### 9.3 Idea Metaphor Noun → Mechanism 대응

- **PRISM-FOG-FX** — 빛 (NVFP4 precision) 이 layer 별로 differential 로 분광 (mixed precision) + visual context fog (DVFS slip) 가 idle decode step 에 GPU clock 을 낮춤 + FX (FleX tensor-core) 가 4:8 sparse 와 dense 를 layer 마다 다른 itinerary 로 dispatch
- **BIVOUAC-SLATE** — visual KV 가 cluster centroid (slate, 검은 슬레이트 위에 핵심 token 만 기록) 로 hierarchical 하게 정리 + cross-frame bivouac (캠프) 처럼 reuse
- **PRISM-VL-R** — visual prefix 의 LSH semantic hash 가 RadixAttention prism 처럼 분광되어 second-level branch
- **STRATA-K-R** — KV layout 이 stratified (지층, L2 / GDDR7|UMA / page-color cold) layer 별 매핑
- **HARBINGER-CLOVER-R** — visual confidence harbinger (전조) 가 speculative draft length + cluster heat (4-leaf clover lucky pattern) 가 early-exit
- **OBELISK-5090-R** — RTX 5090 single-GPU 가 large MoE obelisk (단독 monument, 거대한 inference stack 의 monolith) 로 local serving
- **ATRIUM** (2026-04-28 retain, v2-r55 origin) — 빛 (LPDDR5x bandwidth GB/s) 이 layer (transformer L_i) 별로 차등 들어오는 중정 (atrium, 천장 채광창에서 빛이 차등 입사). 중정의 **천장 채광창** = SM partition (Green Context CUDA 12.4 SM 1500/1060 split, HOT layer 우선), **벽면 단열재** = L2 carveout (cudaCacheConfigure persistent), **빈 채광창** = layer 0-3 alloc skip (DeepStack 미주입 layer)
