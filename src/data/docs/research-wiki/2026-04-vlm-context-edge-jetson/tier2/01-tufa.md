# Spatial-Entropy-Gated Vision Tower Early-Exit on Jetson Orin Nano 7W (TUFA) — Tier-2 Single Mechanism

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-2 독립 #1**

> ## 약어 풀이 (R35, 핵심만)
>
> - **CLIP-ViT-L/14** — Vision Transformer Large 14-patch. VLM vision tower 표준 — 24-layer encoder.
> - **Vision tower / Vision encoder** — VLM 의 image → token 변환 모듈.
> - **Early-exit** — 모델의 일부 layer 만 통과 후 출력 — DynamicViT / A-ViT / V2Drop / LayerSkip.
> - **Spatial entropy** — image patch 의 spatial complexity 측정. 단순 background = low, object dense = high.
> - **Confidence head** — early-exit 가능성 판정 linear probe (1-2 layer MLP).
> - **DynamicViT** — token-pruning ViT. 본 idea adjacent baseline.
> - **A-ViT** — adaptive ViT. baseline.
> - **V2Drop** — vision token dropping, [arXiv:2509.01552](https://arxiv.org/abs/2509.01552).
> - **TokenFLEX** — vision token granularity dynamic, [arXiv:2504.03154](https://arxiv.org/abs/2504.03154).
> - **FastVLM** — vision encoder cost reduction CVPR 2025. 본 idea adjacent.
> - **LayerSkip** [ICML 2024] — LLM decoder early-exit.
> - **EdgeViT++** [SSRN 5319357] — dynamic token pruning.
> - **OmniVLM** [arXiv:2412.11475](https://arxiv.org/abs/2412.11475) — small VLM deployment baseline.
> - **NanoVLM** [arXiv:2503.07920](https://arxiv.org/abs/2503.07920) — small VLM.
> - **MobileVLM** — mobile VLM.
> - **Phi-3-Vision** — Microsoft small VLM, deployment 비교.
> - **mtmd-cli** — llama.cpp 의 multimodal CLI tool. vision encoder pipeline.

**Target Venue**: IEEE CAL 4p (primary) / IEEE ESL 4p (alternative)
**Score** (Tier-2 rubric): Novelty **6.5** / Diff **6.5** / Impact **6.4** = 평균 **6.47**
**판정**: Accept Tier-2 (single-mechanism + Orin Nano 7W first-to-report).
**Phase 1' diff**: ΔM = 0 — trivialty 회피 명시 (Orin Nano-specific 하드웨어 제약 + first-to-report 명시).

---

## 1. 개요

본 연구는 **Jetson Orin Nano 8GB 의 7W power mode** (single HW) 에서 InternVL3-2B INT4 GGUF (single workload) 의 **CLIP-ViT-L/14 24-layer vision tower 를 spatial entropy 로 12-24 layer 동적 선택** (single mechanism) 하는 single-mechanism Tier-2 study. 7W constraint thermal envelope 측정은 **first-to-report**.

기존 vision early-exit (DynamicViT / A-ViT / V2Drop / LayerSkip) 은 datacenter only — Orin Nano 7W mode 의 thermal budget 에서 측정한 paper 부재.

**Metaphor 부속 (R30)**: "TUFA" = 응회암 (가벼운 화산암). 후보: PUMICE / FROTH.

---

## 2. 기존 연구의 한계 / GAP

| 기존 | 본 TUFA 와 차별 |
|------|-----------------|
| DynamicViT | datacenter, token pruning 만 |
| A-ViT | datacenter |
| V2Drop [arXiv:2509.01552](https://arxiv.org/abs/2509.01552) | token-level granularity |
| LayerSkip [ICML 2024] | LLM decoder skip |
| FastVLM [CVPR 2025] | encoder 자체 최적화, layer dynamic 부재 |
| EdgeViT++ [SSRN 5319357] | mobile, **Orin Nano 7W thermal scope 부재** |
| OmniVLM [arXiv:2412.11475](https://arxiv.org/abs/2412.11475) | small VLM, layer dynamic 부재 |
| NanoVLM [arXiv:2503.07920](https://arxiv.org/abs/2503.07920) | small VLM scope |

**GAP**: Vision tower layer-level dynamic + Orin Nano 7W thermal envelope 측정 은 first-to-report.

### Trivialty 회피 (R24/R25)

- Vision early-exit 자체는 trivial 변형 가능 — 그러나 본 idea 는 **Orin Nano 7W constraint** 의 thermal budget 측정과 spatial entropy gating 통합으로 trivialty 회피.
- "단순 layer 수 줄이기" 가 아니라 **spatial entropy → confidence head → exit decision** 의 closed-loop.
- First-to-report claim: Orin Nano 7W mode 의 vision encoder thermal 분석 publication 부재.

---

## 3. 제안 기법 (Tier-1 의 1 mechanism — single mechanism)

### M1: Spatial-Entropy-Gated Layer Exit

**① Scheme — Source Verified (R32)**:

CLIP-ViT-L/14 vision tower 의 24 layer 중 입력 이미지의 spatial entropy 추정으로 12-24 layer 중 동적 선택. Confidence head (linear probe) 가 layer 12, 16, 20 출력에서 task-aligned similarity score 산출, threshold 통과 시 exit. ggml/llama.cpp 의 vision encoder pipeline 에 hook 추가.

> ✅ source verified: llama.cpp `mtmd-cli` ([repo](https://github.com/ggml-org/llama.cpp/tree/master/examples/llava))
> 🔧 source proposed: vision encoder layer-wise hook (~80 LOC)
> ✅ source verified: vLLM vision tower (model_executor 분리)

**② 문제 + evidence**:

[MobileAIBench arXiv:2406.10290](https://arxiv.org/abs/2406.10290): Orin Nano 8GB LLaVA-1.5-7B prefill TTFT 4.2-6.8s, vision tower 26-39% 차지. [VLMBench](https://arxiv.org/abs/2406.09246): RefCOCO image patch 91% 가 ≤ 576 patches — simple 이미지 다수. Simple 이미지는 12-layer 만 통과해도 충분.

**③ Step-by-step**:

1. 입력 image patch 의 spatial entropy 계산 (CPU side, ~0.1ms).
2. Vision tower layer 12 forward pass 종료 후 confidence head linear probe — task-aligned similarity score.
3. score ≥ τ_high 시 exit. score < τ_low 시 continue. τ_low ≤ score < τ_high 면 다음 confidence layer (16) 까지 진행.
4. Exit layer 출력을 LLM projector 에 전달.

**④ 차별화**:

- **vs DynamicViT**: token granularity, layer 부재.
- **vs LayerSkip**: LLM decoder.
- **vs V2Drop**: token dropping, layer 동적 부재.
- **vs FastVLM**: encoder 자체 최적화, layer dynamic 부재.

---

## 4. 평가 (R27-β, scope 축소)

### (1) Hardware

- **Single HW**: Jetson Orin Nano 8GB at **7W mode** (first-to-report scope).
- 비교: Orin NX 25W (selectivity 검증).

### (2) Model

- **InternVL3-2B INT4 GGUF** (single workload).

### (3) Dataset

- **VQAv2 val** (simple VQA — entropy gating 효과 큼)
- **RefCOCO+** (grounding — accuracy drop 검증)

### (4) Tools

- **llama.cpp** mtmd-cli + custom hook
- **tegrastats** (7W mode thermal)
- **Nsight Compute** (vision encoder layer-wise latency)

### (5) Baseline (3 stated, Phase 1' 추가 보강)

| # | Baseline | 출처 | 역할 |
|---|----------|------|------|
| (a) | **24-layer full forward** | stock InternVL3 | upper bound |
| (b) | **Static 16-layer** | manual truncation | simple lower bound |
| (c) | **DynamicViT** | ICCV 2021 ✓ peer-reviewed | token pruning |
| (d) | **FastVLM** (Phase 1' 추가) | CVPR 2025 ✓ peer-reviewed | encoder cost |
| (e) | **V2Drop** (Phase 1' 추가) | [arXiv:2509.01552](https://arxiv.org/abs/2509.01552) | token dropping |
| (f) | **LayerSkip** (Phase 1' 추가) | ICML 2024 ✓ peer-reviewed | LLM decoder skip variant |

Peer-reviewed ratio: 3/6 = **50%** (R2 ≥ 25% 충족, Tier-2 권장).

### (6) Implementation Steps

| Step | 의존성 | Component | 완료 판정 |
|------|--------|---------|---------|
| Step 1 | — | llama.cpp mtmd-cli 빌드 + InternVL3-2B baseline | TTFT 측정 |
| Step 2 | Step 1 | Vision encoder layer-wise hook | layer 12/16/20 출력 capture |
| Step 3 | Step 2 | Spatial entropy + confidence head 학습 (validation 200 image) | confidence accuracy ≥ 0.8 |
| Step 4 | Step 3 | 7W mode thermal sustained 측정 | tegrastats junction T 안정 |
| Step 5 | Step 4 | 6 baseline 재현 + accuracy regression | manuscript draft |

**참고 소요**: 약 5 weeks.

### (7) Preliminary Analysis

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| Vision encoder latency | base | **-31~-44%** | simple image 91% |
| Total TTFT | base | **-18~-22%** | 7W mode |
| Accuracy drop (VQAv2) | 0pp | ≤ 0.8pp | simple VQA |
| Accuracy drop (RefCOCO+) | 0pp | ≤ 1.5pp | grounding |
| 7W mode thermal sustained | throttle | -40% throttle entry | sustained 5min |

---

## 5. 예상 효과 + 차별화 강조

- Vision encoder latency **31-44% ↓**, total TTFT **18-22% ↓**, accuracy drop ≤ 0.8pp on simple VQA, ≤ 1.5pp on RefCOCO grounding.
- **First-to-report**: Orin Nano 7W mode 의 vision tower thermal envelope 측정 + spatial entropy gating 통합.
- 4-6p IEEE CAL / ESL 분량.

---

## 6. Tier-1 Tier 분기 가능성

본 Tier-2 idea 는 Tier-1 [SHOAL](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/02-shoal.md) (DLA tile-stream KV residence) 과 axis 직교 — 본 idea 는 vision tower layer 자체 단축, SHOAL 은 DLA 출력 KV residence. 동시 적용 가능. 단 본 Tier-2 idea 는 single mechanism scope 만 유지 — Tier-1 격상 시 confidence head + spatial entropy + thermal closed-loop 4-mechanism 으로 확장 가능 (별도 paper).

---

## 7. Source Verification (R32)

| Component | Path / Function | 상태 |
|---|---|---|
| llama.cpp mtmd-cli | [examples/llava](https://github.com/ggml-org/llama.cpp/tree/master/examples/llava) | ✅ |
| Vision encoder hook | 신규 ~80 LOC | 🔧 |
| tegrastats / jtop | [jetson_stats](https://github.com/rbonghi/jetson_stats) | ✅ |
| CLIP-ViT-L/14 weights | HuggingFace `openai/clip-vit-large-patch14` | ✅ |
| InternVL3-2B GGUF | HuggingFace `OpenGVLab/InternVL3-2B-GGUF` | ✅ |

---

## 8. Reference

- **CLIP** [ICML 2021]: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- **MobileAIBench**: [arXiv:2406.10290](https://arxiv.org/abs/2406.10290)
- **VLMBench**: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- **DynamicViT** [ICCV 2021]: [arXiv:2106.02034](https://arxiv.org/abs/2106.02034)
- **A-ViT** [CVPR 2022]: [arXiv:2112.07658](https://arxiv.org/abs/2112.07658)
- **V2Drop**: [arXiv:2509.01552](https://arxiv.org/abs/2509.01552)
- **TokenFLEX**: [arXiv:2504.03154](https://arxiv.org/abs/2504.03154)
- **FastVLM** [CVPR 2025]
- **LayerSkip** [ICML 2024]
- **EdgeViT++**: [SSRN 5319357](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5319357)
- **OmniVLM**: [arXiv:2412.11475](https://arxiv.org/abs/2412.11475)
- **NanoVLM**: [arXiv:2503.07920](https://arxiv.org/abs/2503.07920)
