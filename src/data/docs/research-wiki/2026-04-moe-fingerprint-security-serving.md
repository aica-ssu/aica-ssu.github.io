# MoE Expert-Activation Fingerprint — Security & Serving Research Ideation (Summary)

**Date**: 2026-04-24 · **Mode**: 1 (sentence-input) · **Session type**: research ideation (blog-style summary)
**Input 재료**: 이방산 학생 (SSU AICA Lab) 의 `260424 MoE fingerprinting` 실험 폴더 (summary for presentation + raw data)
**실험 규모**: 4 MoE 모델 (Mixtral-8x7B, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, Qwen3-30B-A3B) × 608 config × ~5,900 runs, 모델 수정 없이 forward 1회 (`max_tokens=1`) fingerprint 추출
**핵심 측정**: MMLU 4-cat **96.2%** (Qwen3 k-NN), MMLU 57-way **89.9%**, WildJailbreak 2-way **94.0%**, 4-way **94.0%**, BoolQ/TrueFalse 포함
**사용자 요구**: (1) AI 보안 또는 모델 서빙 최적화 방향, (2) 추가 실험 최소 (+α ≤ 기존의 작은 분량), (3) 보안 전문가 관점에서 novelty 엄밀 검증

---

## 0. Executive Summary

### 0.1 사용자 insight 엄밀 검증 결과 (보안 전문가 관점)

사용자 제시 insight ("no finetune + no extra guard forward + multi-task unified 이면 paper-worthy novelty") 를 S&P/USENIX Security/NDSS 급 novelty-reviewer 관점에서 엄밀 검증한 결과:

- **"No finetune"** 자체는 novelty **아님** — FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558), EMNLP 2025 Findings), HSF ([arXiv:2409.03788](https://arxiv.org/abs/2409.03788)), HiddenDetect ([arXiv:2502.14744](https://arxiv.org/abs/2502.14744)), "Do Internal Layers..." ([arXiv:2510.06594](https://arxiv.org/abs/2510.06594)) 등 training-free dense LLM detection 다수.
- **"No extra guard forward"** 도 novelty **아님** — OmniGuard ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) 가 internal representation classifier 로 120x faster, FJD 는 first-token logit 로 사실상 추가 forward 0.
- **"Multi-task unified"** 도 약함 — MultiTaskGuard/UniGuard ([arXiv:2504.19333](https://arxiv.org/abs/2504.19333)) multi-task guard 14x + LoRA-shared. **vLLM Semantic Router v0.1 Iris (2026-01 merge)** 이 jailbreak+domain+PII+fact-check pluggable classifier 를 production 에 merge 완료.

**진짜 paper-worthy novelty 는 아래 3 축 중 하나 이상** 이어야:

1. **MoE discrete routing 의 adaptive-adversarial robustness** (continuous hidden-state probe 는 Obfuscated Activations [arXiv:2412.09565](https://arxiv.org/abs/2412.09565) 에 의해 recall 100→0 붕괴; discrete top-k 는 combinatorial non-differentiable 이라 질적 차이 기대). → **DISCRETE-VEIL'** (Tier-1 S&P).
2. **Token × layer 2D early-exit MoE fingerprint framework** (fingerprint convergence 를 token prefix length k* 와 layer depth L_k 두 축에서 Pareto 최적화). → **LOOM'** (Tier-1 MLSys/ASPLOS).
3. **Information-theoretic shared-substrate Pareto proof** — fingerprint Fisher information 을 N task 에 분배하는 systems-theory 축. → **LOOM'** 내부 Section.

### 0.2 Tier-1 Top 3 (Tier-1 main + 1 Tier-2 pair)

| Rank | Idea | 전제 | GAP | 제안 overview | 예상 효과 | Score |
|------|------|------|-----|---------------|-----------|-------|
| 🥇 | **DISCRETE-VEIL'** (S&P 2027 13p) | MoE router top-k 는 discrete argmax, continuous h 와 질적으로 다른 attack surface | Obfuscated Activations (2412.09565) 는 dense hidden probe 만 타격, MoE routing 은 미측정 | DRO-Attack (embedding-space PGD + joint routing-pattern loss via Gumbel-softmax surrogate) + Entropy-Sharpen-KS Tripwire 로 adaptive adversary robustness 정량화 | Clean WildJailbreak 94% → DRO-Attack 후 ≥55% 유지 (목표 하한), hidden-state probe 는 0-20% 붕괴 대조 | **8.00** |
| 🥈 | **LOOM'** (MLSys 2027 18p, LOOM+EMBER+TALLY+THRESHOLD 통합) | MoE forward 1회 pool `(L,E)` 가 detection + residency + prefetch 3 consumer 에 동시 공급 가능 | vLLM Semantic Router Iris 는 LoRA-shared classifier heads 만, serving cache/prefetch consumer 미포함; MoE-Infinity/DuoServe 는 detection 미포함 | 4 mechanism: (M1) end-of-prefill pooled tap, (M2) token×layer 2D early-exit Pareto, (M3) multi-consumer fan-out + admission, (M4) LEAP+QIVF compressed index | decode p50 -15%, expert miss rate -20-25%, detection F1 93%+ @ +<10ms lookup | **7.38** |
| 🥉 | **BEACON-GUARD-Lite** (USENIX ATC 2027 12p 또는 DATE 2027 6p; LOOM' 와 paper pair) | 단일 (L, 4E) feature bank 에 FAISS IVF 로 training-free multi-task (domain + safety) guard | MultiTaskGuard/UniGuard + vLLM Semantic Router Iris 는 LoRA classifier heads (본 연구는 k-NN only, 완전 no-train) | Unified Routing-Feature Bank (URFB) + Multi-signal (discrete + continuous + integer) fingerprint fusion geometric analysis | WildGuard-7B 대비 latency 50-100x ↓, memory 28GB → <50MB, 2-task accuracy 유지 (96% / 94%) | **7.25** (Tier-2 rubric) |

### 0.3 Tier-2 독립 Top 3

| Rank | Idea | Venue | 특이점 |
|------|------|-------|--------|
| T1 | **DISCRETE-VEIL-Lite** | IEEE CAL 4p 또는 DSN practical 6p | Qwen3 + PAIR 공격 200 prompts 단일 scope. 8주 부담 없는 short paper. |
| T2 | **TALLY-Spinoff** | DATE 2027 4p WIP | LEAP 의 MoE-specific interpretability (expert-j-at-layer-i = safety-critical SAFEx-style label) 축 독립. LOOM' 의 Section 7 과 중복되지 않게 positioning. |
| T3 | **BEACON-GUARD-Lite** (Tier-2 fallback) | DATE 2027 6p | Tier-1 ATC reject 시 fallback. Multi-signal Fusion 축만 유지. |

### 0.4 Exploration 했던 모든 아이디어 (미선정 포함)

| ID | 이름 | 판정 | GAP (한 줄) | 제안 overview (2-3 줄) | 미선정/변경 사유 (한 줄) |
|----|------|------|-------------|-------------------------|---------------------------|
| A1 | **DISCRETE-VEIL** (원안) | **→ DISCRETE-VEIL'** | hidden probe 가 adaptive attack 에 깨지는데 MoE discrete 은 어떤가 | DRO-Attack (PGD joint loss) + Routing-Entropy Tripwire, 4 모델 × 3 공격 | scope 축소 (2 모델 × 2 공격 + Entropy-Sharpen-KS 재정의) |
| A2 | **BEACON-GUARD** (원안) | **→ BEACON-GUARD-Lite Tier-2** | no-finetune multi-task unified guard | `(L, 4E)` FAISS IVF + 3-head k-NN (domain+safety+OOD) | Tier-1 USENIX Security 은 MultiTaskGuard + vLLM Semantic Router Iris concurrent. OOD 축은 ASE 2025 NIER scoop |
| A3 | **THRESHOLD** (원안) | **DROP (LOOM' 흡수)** | streaming + drift-aware Crescendo 방어 | k*-token prefix convergence + decode 32-step drift re-check | EMBER 와 early-exit axis 중복; LOOM' M2 token-axis 로 병합 |
| A4 | **LOOM** (원안) | **→ LOOM'** | shared observability substrate | 3 consumer fan-out (detection + residency + prefetch) | vLLM Semantic Router Iris scoop; EMBER+TALLY+THRESHOLD 흡수하여 4 mechanism 로 재편 + systems-theory proof 추가 |
| A5 | **EMBER** (원안) | **MERGED (LOOM' M2)** | mid-prefill early-layer jailbreak reject | L_k=2-5 partial fingerprint + adaptive layer budget + batch-packed early exit | LOOM' 과 vLLM fork 공유; artificial split 위험; LOOM' M2 layer-axis 로 흡수 |
| A6 | **TALLY** (원안) | **DOWNGRADED (LOOM' M4 + T2 spinoff)** | fingerprint index 경량화 | LEAP + QIVF + FSE | 단독 NeurIPS Systems 는 FAISS IVF-PQ 의 incremental novelty 부족; LOOM' 의 M4 + DATE 4p WIP spinoff |

**핵심**: 원안 6 아이디어 → 최종 3 파이프라인 (+ Tier-2 spinoff 3개). improve-first + merge 로 mechanism 합계 15 → 8 개 압축 (-7). Triviality / Tiering / Naming / Peer-review-mix 규율 전부 준수.

---

## 1. 연구 진행 Meta

### 1.1 사용자 쿼리 원문 (한국어)

> "학생 이방산이 260424 MoE fingerprinting 폴더에 정리한 실험 데이터 (summary for presentation + raw data) 를 기반으로 연구 ideation 진행. 가능하면 지금 수준에서 +α 로 실험을 너무 많이 하지 않고 얻을 수 있는 novelty. AI 보안 혹은 모델 서빙 최적화 방향. finetuning-free + no-extra-guard-forward + multi-task 장점이 paper-worthy novelty 인지 보안 전문가 입장에서 엄밀 검증 + 가능한 ideation 방향."

### 1.2 주요 키워드 (4-8개)

- **도메인**: MoE (Mixtral / Qwen1.5-MoE / DeepSeek-V2-Lite / Qwen3-30B-A3B), LLM safety, training-free classifier
- **관찰**: MoE router top-k 선택이 domain/safety signature 를 96.2%/94% 정확도로 인코딩, 모델 수정 없음
- **기법**: expert-activation fingerprint (L, E) matrix, k-NN (k=1,5,15,51) / k-means / NC / Rank, FAISS IVF
- **타겟 지표**: (보안) adaptive-attack robustness + jailbreak detection F1 + latency; (서빙) token budget, FLOPs, expert miss rate, TTFT

### 1.3 중점적으로 고려한 축

1. **모델 수정 없음 (training-free)** — 사용자 핵심 요구. 모든 아이디어가 no-finetune + forward 1 회 유지.
2. **Adaptive adversarial robustness** — 보안 novelty 의 진짜 축으로 판정 (DISCRETE-VEIL).
3. **Serving stack integration** — 단일 fingerprint read 로 detection + residency + prefetch (LOOM').
4. **기존 실험 재활용 ≥80%** — 사용자 "+α 최소" 요구 반영. BEACON-GUARD-Lite 는 100% 재활용.
5. **Tier-1 + Tier-2 paper pair 전략** — 동일 core idea 를 Top-tier 13p + Tier-2 4-6p 로 분리 제출 가능성 검토.

### 1.4 의도적으로 제외한 축 + 이유

| 제외 축 | 이유 |
|---------|------|
| **Hardware accelerator / PIM MoE 가속** | 사용자 직접 fingerprint 결과 기반 ideation. HW co-design 은 본 실험 데이터와 거리 멀고 추가 FPGA/ASIC 자원 필요. 이전 PRISM/TernVLM 세션에서 별도 다룸. |
| **Fine-tuning-based safety alignment (RASA / SAFEx)** | 사용자 core premise 가 "no finetune". Finetune 방향은 별개 연구로 분리. |
| **Side-channel MoE attack (MoEcho, ExpertEcho)** | 2026-04-21 이전 세션이 이미 다룸 (FARD-C / ZMSP / PhantomRoute). 본 세션은 **model-owner in-worker fingerprint** 로 성격 다름. |
| **Cross-model fingerprint alignment / Procrustes** | 사용자 실험 데이터 재활용 축 밖; 각 모델별 classifier 를 별도 학습하는 것이 실용적. 단 미래 확장 vector 로 summary appendix 에 기록. |
| **VLM/VLA fingerprinting** | 본 실험은 text LLM 4 모델 한정. VLM/VLA 는 2026-04-22 이전 세션이 다룸. |
| **RLHF / Constitutional alignment 결합** | 사용자 "no finetune" premise 와 상충. |
| **Multi-turn dialogue safety (Crescendo / Skeleton-Key) 방어 단독** | THRESHOLD 가 covered, LOOM' 의 M3 admission extension 으로 흡수. 단독 paper 는 본 세션 범위 밖. |

### 1.5 검색 쿼리 전략

- Step 0-α workload characterization: IISWC 2024-2025 accepted papers, ISPASS 2024-2025, MLPerf MoE benchmarks, vLLM/SGLang/llama.cpp 공식 blog + PR.
- Step 0 Phase: 5 축 병렬 WebSearch:
  1. training-free LLM jailbreak detection internal activation probing 2025
  2. MoE router expert activation fingerprint safety jailbreak classifier 2025
  3. LLM hidden state linear probe jailbreak detection arxiv 2024 2025
  4. MoE serving cache prefetch router activation profiling arxiv 2025 ISCA MICRO
  5. arxiv 2025 LLM domain classification routing signature task identification zero-shot probe
- Phase 2 novelty reviewer 추가 search 7 쿼리 + WebFetch 4 회 (Task-Cond Routing 2603.11114 peer-review 상태 확인 등).

### 1.6 사용된 전문가 에이전트

- **system-robustness-expert** (메인, 보안 novelty 엄밀 검증 1차): DISCRETE-VEIL / BEACON-GUARD / THRESHOLD 도출
- **ai-optimization-expert** (서빙 최적화): LOOM / EMBER / TALLY 도출
- (sub) **novelty-reviewer / differentiation-reviewer / impact-reviewer** 3인 병렬 리뷰

---

## 2. Tier-1 Top 3 — 상세

### 2.1 🥇 DISCRETE-VEIL' (IEEE S&P 2027 13p)

> **Mechanism-Metaphor**: "얇은 veil 이 adaptive adversary 의 latent-space manipulation 에 어떤 불규칙성을 주는가" — discrete routing 은 얇지만 combinatorial 구조가 공격을 차단.

#### 2.1.1 개요

Obfuscated Activations ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)) 는 dense LLM 의 hidden-state linear/MLP probe + SAE + representation-probing + latent OOD 모두 adaptive PGD 공격에 recall **100% → 0%** 로 붕괴함을 보였다. **MoE router 의 top-k selection 은 discrete argmax 연산** 이므로, embedding-space PGD 가 (a) jailbreak 의도를 유지하면서 (b) routing pattern 을 benign pattern 에 맞추려면 **combinatorial non-differentiable objective** 를 풀어야 한다. 이 질적 차이가 MoE routing fingerprint 를 hidden-state probe 대비 **adaptive-attack robust** 하게 만드는지를 정량화하는 것이 DISCRETE-VEIL' 의 핵심 질문.

#### 2.1.2 기존 연구 한계·GAP

| 기존 연구 | 공격/방어 축 | Signal | 한계 |
|-----------|------------|--------|------|
| Obfuscated Activations ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)) | embedding-PGD attack | dense hidden state | MoE routing 미다룸, discrete signal 공격 안 함 |
| V-MoE Adversarial Robustness ([OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp)) | image PGD | V-MoE router | **이미지 공격 공간**; text LLM embedding-PGD 와 구분 |
| GateBreaker ([arXiv:2512.21008](https://arxiv.org/abs/2512.21008)) | weight-level neuron ablation | MoE safety expert | 3% neuron disable 로 ASR 7.4→64.9%; **embedding 공격 아님** |
| Expert Selections Reveal... ([arXiv:2602.04105](https://arxiv.org/abs/2602.04105)) | text reconstruction attack | clean routing indices | **privacy 공격** 방향, detection 공격 아님 |
| L³ ([arXiv:2602.08741](https://arxiv.org/abs/2602.08741)) | refusal-expert silencing | routing pattern for attack | attack 방향이지 detection 공격 아님 |
| FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558)) | — (no adversarial eval) | first-token logit | dense LLM 가정, adaptive attack 평가 없음 |
| OmniGuard ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) | limited (non-adaptive) | dense internal repr | adaptive attack 평가 미비 |

**GAP**: **MoE routing classifier 에 대한 embedding-space PGD attack 과 그 robustness 정량화 는 현재 공개 논문 없음**.

#### 2.1.3 제안 기법 (3-mechanism)

##### Mechanism M1 — DRO-Attack (Discrete Routing Obfuscation)

① **어떤 scheme 을 어디에 추가하나**: `vllm/model_executor/models/qwen3_moe.py` + `mixtral.py` 의 `FusedMoE.forward` 에 **hook** 을 등록해 prompt embedding 에 PGD 를 적용하면서 fingerprint 재추출 파이프라인을 그대로 사용. Attack module: `attacks/dro_attack.py` (~200 LOC, PyTorch 2.5 PGD + Gumbel-softmax surrogate).

② **해결하는 문제 + Workload evidence**: 사용자 실측 WildJailbreak 2-way **94.0%** (Qwen3 k-NN, balanced 8K) 는 **clean setting 수치**. Obfuscated Activations 는 dense hidden probe 를 recall 100 → 0 으로 무력화했다. **MoE discrete routing 이 이 공격 공간에서 얼마만큼 방어력을 가지는지** 는 미측정. S&P reviewer 는 training-free detection 의 adaptive robustness 를 반드시 요구.

③ **Step-by-step 구현 절차**:
1. PAIR (harmbench) + GCG (nanoGCG) 로 clean jailbreak prompts **200 + 200 = 400** 생성 (Qwen3 + Mixtral 2 모델).
2. 각 prompt token embedding `e_t` 에 perturbation `δ_t` (L∞ ≤ 8/255, L2 ≤ 0.3) 를 할당, PGD 20 iteration.
3. Joint loss: `L = CE(jailbreak_generation) + λ · MSE(topk_index_continuous_surrogate(e+δ), benign_target_pattern)`. `topk_index_continuous_surrogate` 은 Gumbel-softmax relaxation (temperature τ=0.5) 으로 discrete argmax 의 미분 가능 버전.
4. 공격 후 perturbed input 으로 fingerprint 재추출 (기존 vLLM fork 파이프라인 그대로).
5. k-NN classifier 로 detection recall 재측정.

④ **기존 접근 실패 + 본 mechanism 차별화**: (a) hidden-state linear probe 는 continuous h 에 gradient 가 직접 적용 → recall 0%. (b) MoE routing 은 argmax 이라 differentiable surrogate 없이 gradient 안 먹음 → Gumbel-softmax surrogate 가 필수. (c) 본 연구는 처음으로 "surrogate 를 통한 PGD 가 실제 discrete argmax 의 shift 를 얼마나 유도하는가" 를 WildJailbreak 에서 측정. V-MoE PGD (image) 와는 modality 구분, GateBreaker (weight ablation) 와는 attack surface 구분 — Venn diagram 으로 명시.

##### Mechanism M2 — Entropy-Sharpen-KS Tripwire

① **어디에**: `detection/tripwire.py` (~80 LOC). 기존 `softmax_scores` fingerprint 에서 per-layer Shannon entropy `H_l` 를 계산 → (L,) vector.

② **해결하는 문제 + evidence**: Mechanism M1 이 PGD 로 routing 을 조작하면 **softmax distribution 이 sharpen** (entropy 감소). 자연 jailbreak 의 entropy 분포와 adaptive-attacked jailbreak 의 entropy 분포 간 KS-test 로 구분. ASE 2025 NIER 의 "routing entropy OOD" 와 구분점: 단순 entropy threshold 가 아니라 **distribution KS-test** (PGD-specific anomaly detection).

③ **Step-by-step**:
1. Clean WildJailbreak 8K 와 DRO-Attack post WildJailbreak 800 두 set 의 per-prompt layer-entropy vector 수집 (기존 데이터 재활용).
2. 두 set 의 entropy 분포 간 layer-wise KS statistic 계산.
3. KS p < 0.01 인 layer L_KS 를 tripwire 로 선정 (예상: 큰 모델은 중~후반 layer).
4. feature = concat(topk_weights, tripwire_layer_entropy) 로 k-NN 재학습.
5. Clean vs DRO-attacked detection recall + AUC 측정.

④ **기존 실패 + 차별화**: HSF / HiddenDetect 는 hidden state 단일 통계. RASA 는 supervised finetune. ASE 2025 NIER 의 entropy 는 OOD 축 (natural 다른 분포) 이지만 본 Tripwire 는 **adversarial-specific entropy sharpen** 을 KS 로 구분 → *adaptive-attack forensics* 축.

##### (통합 포지셔닝)

Mechanism M1 + M2 = **attack (DRO) + defense (Entropy-Sharpen-KS)** 의 짝. Scoop 재검증: V-MoE PGD 는 image, GateBreaker 는 weight, Expert Selections 2602.04105 는 privacy leakage — 본 연구와 **Venn diagram 에서 non-overlapping**. Scoop 판정 **adjacent (35-45%)**.

#### 2.1.4 평가 · 실험 플랜 (7 요소)

##### (1) Hardware environment
- Primary: **RTX Pro 6000 96GB** × 1 (Qwen3-30B-A3B BF16 ≈ 60GB, 여유 36GB 로 adversarial 계산).
- Secondary: **RTX 5090 32GB** × 1 (Mixtral-8x7B-Instruct AWQ-4bit ≈ 24GB).
- Storage: NVMe 4TB (8K+800 prompts × 48-layer × 128-expert × bf16 ≈ 2GB raw trace per model × 2 = 4GB).
- Network/multi-node: single-workstation scope, no multi-node.

##### (2) Model
- **Qwen3-30B-A3B** (HuggingFace: `Qwen/Qwen3-30B-A3B`) BF16, fingerprint dim 48×128=6,144.
- **Mixtral-8x7B-Instruct-v0.1** (`mistralai/Mixtral-8x7B-Instruct-v0.1`) AWQ 4-bit, fingerprint dim 32×8=256.
- Precision: router stays BF16 even under AWQ (verified from vLLM FusedMoE source).
- Checkpoint source: 사용자 기존 `expert_traces_*` 재활용.
- Inference: vLLM fork (사용자 버전 0.19.0 + custom capture).
- Robustness model: 동일 2 모델 (adversarial PGD 는 transfer 를 체크하기 위해).

##### (3) Dataset · Workload
- **Benchmark + subset**: WildJailbreak 8K balanced eval (사용자 v2 샘플 재활용), JailbreakBench 100 behaviors, HarmBench 200 val (OOD 체크).
- **Adaptive attack corpus**: PAIR 200 prompts + GCG 200 prompts = 400 prompts × 2 모델 = 800 prompts. Clean baseline attack corpus 는 WildJailbreak 에서 직접 샘플링.
- **Real vs synthetic**: 모두 real public data.
- **Scale**: 8K clean + 800 attacked = 8.8K test samples per model.
- **Metrics**: detection recall (clean), recall-under-attack (DRO), AUC, balanced_acc, per-class confusion matrix.

##### (4) Simulator · Tools
- vLLM fork (사용자 existing version + DRO hook 100 LOC).
- PAIR: `harmbench/attacks/pair.py`, GCG: `nanoGCG` 0.2, Obfuscated Activations baseline 재현은 original authors 의 `obfuscated-activations` repo 재사용.
- PyTorch 2.5 + Gumbel-softmax implementation (`torch.nn.functional.gumbel_softmax` hard=True) + custom PGD loop (~150 LOC).
- Profiling: Nsight Compute counters (`lts__t_sectors`, `sm__warps_active`) for fingerprint overhead.
- External libraries: FAISS-GPU 1.10 (k-NN), Nsight Systems, PyTorch profiler.

##### (5) Ablation · Baseline
- **Baselines (8 종)**: (a) **FJD** ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558) [EMNLP 2025 Findings]), (b) **OmniGuard** ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856) [preprint]), (c) **WildGuard-7B** ([arXiv:2406.18495](https://arxiv.org/abs/2406.18495) [NeurIPS 2024]), (d) **HSF** ([arXiv:2409.03788](https://arxiv.org/abs/2409.03788)), (e) **LlamaGuard-3** [Meta], (f) **hidden-state linear probe** (Obfuscated Activations 재현), (g) **Task-Cond. Routing Sig.** ([arXiv:2603.11114](https://arxiv.org/abs/2603.11114) [arXiv preprint]) — domain axis only, (h) **V-MoE PGD text-adaptation** (novelty-reviewer 권고).
- **Ablation matrix**: (Clean vs DRO-Attack) × (Mech M1 only vs M1+M2) × (Qwen3 vs Mixtral) × (PAIR vs GCG) = 2^5 = 32 runs. 충분한 factorial.
- **Parameter sweep**: PGD ε ∈ {4/255, 8/255, 16/255}, Gumbel τ ∈ {0.1, 0.5, 1.0}, KS threshold p ∈ {0.001, 0.01, 0.05}.
- **Peer-reviewed baseline ratio**: WildGuard (NeurIPS 2024) + LlamaGuard + FJD (EMNLP 2025 Findings) + V-MoE (ICLR) = **4/8 = 50%**, 규율 R2 (25%+) 충족.
- **Expected runtime**: 10주 (W1-W10, 아래 Implementation Steps 참조).
- **Fallback mode**: 만약 DRO-Attack 이 Qwen3 에서 recall 을 20% 미만으로 떨어뜨리면 (=hidden probe 수준 붕괴), negative result 를 **"MoE routing 도 adaptive adversary 에 취약"** 로 positioning 하여 ACSAC/DSN tier paper 로 전환. Mech M2 entropy tripwire 가 어느 정도 복구하면 **defense-in-depth 논문** 으로 재구성.

##### (6) Implementation Steps (week-level)

| Week | Task | Deliverable |
|------|------|-------------|
| W1 | vLLM fork 검증 + 기존 fingerprint 94% 재현 + V-MoE PGD / GateBreaker Venn 작성 | baseline reproduction 표 |
| W2 | PAIR + GCG 공격 400 prompts 생성 (Qwen3 + Mixtral) | attack corpus |
| W3 | DRO-Attack (Mech M1) PGD + Gumbel-softmax 구현 | attack module 100% ready |
| W4 | 2 모델 × 2 공격 × clean/DRO fingerprint 수집 (약 1,600 runs, 기존 pipeline 재사용) | perturbed fingerprint dataset |
| W5 | Mech M2 entropy tripwire + k-NN 재학습 + KS-test analysis | Mech 평가 표 |
| W6 | 8 baseline 구현/재현 (FJD + OmniGuard + WildGuard + HSF + LlamaGuard-3 + hidden probe + Task-Cond. Routing + V-MoE PGD) | baseline table |
| W7 | 표 1-4, figure 1-5 작성 (Venn, attack recall, ablation, entropy dist) | figures |
| W8-9 | draft + adaptive-attack discussion section + mechanistic interpretability (SAFEx-style expert attribution for Entropy-Sharpen-KS) | 13p draft |
| W10 | polish, artifact evaluation 준비, OpenReview 게시 | submission-ready |

##### (7) Preliminary Analysis Metrics

**4단계 preliminary study**:

- **Stage 1 Baseline reproduction**: 기존 사용자 실험의 Qwen3 WildJailbreak 2-way **94.0%** ± 1% 재현. Mixtral 90.1% ± 1% 재현.
- **Stage 2 Bottleneck attribution**: DRO-Attack 후 recall drop 분해. 목표:
  - Qwen3 clean 94% → DRO-attack 후 target ≥55% (하한 목표, Qwen1.5 noisy 수준 유지 시 success).
  - Mixtral clean 90% → DRO-attack 후 target ≥45%.
  - Hidden-state linear probe (Obfuscated Activations 재현): 100% → 0-20% (예상).
  - **차이 > 30%p 면 discrete-robust 가설 성립**.
- **Stage 3 Roofline**: PGD iteration 횟수 별 recall drop curve. 20 iter 이후 converge 여부. Attack budget 증가 시 (ε=8→16/255) 추가 drop 이 asymptote 에 수렴하는지.
- **Stage 4 Micro-benchmark**: Mech M2 Entropy-Sharpen-KS 단독 (Mech M1 제외) recall 복구 quantify. 목표: 복구 ≥10%p.

#### 2.1.5 예상 효과

- **과학적**: MoE discrete routing 과 dense hidden state 의 adaptive-attack robustness 를 **공식적으로 구분**한 첫 결과. Venn diagram 표준 포맷으로 V-MoE PGD / GateBreaker / Expert Selections / DISCRETE-VEIL 4 attack 공간 경계 확정.
- **실용적**: 만약 recall ≥55% 유지 시 → MoE serving 에 fingerprint-based guard 를 "adaptive-adversary certified" 로 deploy 가능. NeMo Guardrails / vLLM / NVIDIA AI Enterprise 에 인용 근거.
- **Impact 리스크**: MoE 가 전체 LLM 배포의 30-40% 이라는 scope 제한. 단 reviewer 는 scope 자체가 differentiation 역할.

---

### 2.2 🥈 LOOM' (MLSys 2027 또는 ASPLOS 2027 18p, LOOM+EMBER+TALLY+THRESHOLD merged)

> **Mechanism-Metaphor**: "실 하나(routing fingerprint)가 베틀(LOOM)에서 여러 가닥(consumer)으로 엮인다"

#### 2.2.1 개요

MoE forward 1회에서 얻는 `(L, E)` router fingerprint 를 **shared observability substrate** 로 정의하고, 세 consumer (jailbreak detection + domain-aware expert residency + next-layer prefetch hint) 가 공유 read 하는 serving stack. 4 mechanism (end-of-prefill pooled tap + token×layer 2D early-exit + multi-consumer fan-out with admission + LEAP+QIVF compressed index) + **information-theoretic Fisher information Pareto proof** 로 engineering 조합이 아닌 systems-theory contribution 확보.

#### 2.2.2 기존 연구 한계·GAP

| 기존 연구 | 다루는 축 | 한계 (LOOM' 대비) |
|-----------|-----------|-------------------|
| MoE-Infinity ([arXiv:2401.14361](https://arxiv.org/abs/2401.14361) [ATC 2024]) | expert residency + prefetch | detection consumer 미포함, trace 가 batch level |
| PreScope ([arXiv:2509.23638](https://arxiv.org/abs/2509.23638)) | cross-layer prefetch | detection 미포함, best-effort async (miss 시 stall) |
| DuoServe-MoE ([arXiv:2509.07379](https://arxiv.org/abs/2509.07379)) | affinity routing + cache | safety-aware admission 없음 |
| BuddyMoE ([arXiv:2511.10054](https://arxiv.org/abs/2511.10054)) | miss fallback substitute | 품질 손실, detection 없음 |
| Gimbal ([arXiv:2602.21626](https://arxiv.org/abs/2602.21626)) | prefix+KV+stickiness dispatcher | expert footprint signal 미사용 |
| vLLM Semantic Router Iris v0.1 (2026-01) | LoRA-shared classifier heads (jailbreak+domain+PII+fact-check) | serving cache/prefetch consumer 없음, LoRA 학습 필요 |
| FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558) [EMNLP 2025 Findings]) | dense first-token logit | MoE-specific 아님, multi-consumer 아님 |
| OmniGuard ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) | internal representation safety classifier | safety 단일, MoE routing 미사용, serving integration 없음 |

**GAP**: **MoE router fingerprint 를 shared observability substrate 로 정의하고 Fisher information 분배 Pareto proof 로 systems-theory 하는 연구는 공개 없음**. vLLM Semantic Router Iris 는 framing 유사하지만 serving cache/prefetch consumer 가 없고 LoRA 학습이 필요.

#### 2.2.3 제안 기법 (4-mechanism)

##### Mechanism M1 — End-of-Prefill Pooled Router Tap (EPRT)

① `vllm/v1/worker/gpu_model_runner.py` 의 `_execute_model_forward` 말미에 MoE layer 마다 forward hook 등록. `router_logits` + `topk_weights` + `activation_count` 를 prefill token 축에 대해 mean pool → `(L, E)` tensor. `SchedulerOutput` 에 `fingerprint_tensor` 필드 추가.
② **문제 + evidence**: vLLM PR #6734 ("router logit telemetry") 는 per-token overhead 이유로 merge 되지 않음. Per-prompt pooled (~4KB per prompt, L=27 E=160 기준) 은 이 overhead 우려를 우회. MoE-Infinity 는 batch-level trace, per-prompt 정렬 어려움.
③ **step**: (a) FusedMoE.forward 내 이미 계산된 topk_weights 를 per-prompt buffer 로 add, (b) prefill 종료 시 flush, (c) non-blocking queue publish.
④ **기존 실패 + 차별화**: MoE-Infinity 는 scheduler 레벨 only, model runner 와 decouple. PR #6734 per-token overhead 문제 우회.

##### Mechanism M2 — Token × Layer 2D Early-Exit Pareto (ELPT + K-token merged)

① `vllm/v1/engine/core.py` per-request state + `vllm/v1/worker/gpu_model_runner.py` layer-wise hook. Per-layer 마다 partial fingerprint margin 계산; margin 이 high-confidence threshold m_high 초과 시 early-exit, low-confidence m_low 미만이면 추가 layer 진행. K-token (prefix length) 과 L_k (layer depth) 두 축의 joint Pareto frontier 탐색.
② **문제 + evidence**: 고정 L_k 는 easy case 낭비 / hard case 부족. 기존 실험: Qwen3 L22/48, DeepSeek L22/27, Qwen1.5 L02/24 로 모델별 optimal 다름. FJD 는 token k=1 only, HSF 는 final layer only — 2D joint 는 미개척.
③ **step**: (a) token position × layer depth 2D grid 에서 margin 측정 (기존 trace 재활용 + token-position 별 재추출), (b) 각 (k, L_k) 쌍의 F1 / FLOPs / latency 계산, (c) Pareto frontier 도출, (d) runtime 에서 adaptive layer budget 으로 구현.
④ **기존 실패 + 차별화**: FJD (k=1 dense logit), HSF (layer, dense), HiddenDetect (layer, dense VLM) 모두 1D axis. **2D joint 는 공개 없음**.

##### Mechanism M3 — Multi-Consumer Fan-Out + Safety-Aware Admission (SRMC + ACP merged)

① `vllm/v1/engine/core.py` 에 `FingerprintBus` 추가. 3 consumer: (A) `JailbreakDetector` (k-NN on 253K pool), (B) `ExpertResidencyPolicy` (cluster → replica pin), (C) `PrefetchHintProducer` (next-layer top-8 expert prior). Admission controller 가 (A) 결과로 reject/sandbox/full_decode 결정.
② **문제 + evidence**: vLLM Semantic Router Iris 는 classifier heads 만 공유 (LoRA-shared), serving-level cache/prefetch consumer 미포함. 세 consumer 가 각자 forward 돌리면 3배 cost.
③ **step**: (a) fingerprint ring buffer, (b) 3 consumer lock-free read, (c) RequestMetadata union, (d) admission + cluster replica routing.
④ **기존 실패 + 차별화**: Semantic Parallelism ([arXiv:2503.04398](https://arxiv.org/abs/2503.04398)) 는 within-replica reorder only. Gimbal 은 prefix+KV+stickiness. vLLM Semantic Router Iris 는 classifier head shared but serving consumer 없음. 본 연구는 **detection + serving 3-consumer joint + safety-aware admission** first.

##### Mechanism M4 — Compressed Index (LEAP + QIVF)

① Offline: train pool variance + label mutual info 분석 → top-D=256 dim 만 유지 (LEAP). Online: FAISS IVF-PQ 256-dim × 32 subvector × 8-bit = 32 byte/vector. 253K pool → ~8MB.
② **문제 + evidence**: 기존 k-NN 은 6144-dim × 253K × fp32 = ~6GB. QIVF 로 ~8MB, query latency 40ms → ≤3ms.
③ **step**: (a) per-dim variance + MI, (b) top-D selection, (c) FAISS IVF-PQ (nlist=256, PQ=32×8bit), (d) classifier F1 retention check.
④ **기존 실패 + 차별화**: FAISS IVF-PQ 는 general (MoE-unspecific). LEAP 의 **MoE-specific layer-expert block structure** 는 PCA 가 놓치는 interpretable axis 보존 (e.g., "L22 expert 47 is safety-critical" SAFEx-style label 유지).

##### (통합 novelty: Information-Theoretic Shared-Substrate Pareto)

Fisher information 분배 정리: fingerprint F 의 N 개 task 에 대한 분배 Pareto 상한은 `F^{-1}(task_i) · N` 에 bound. 단일 read 로 3 consumer 가 동시에 최적일 수 있는 조건 → **task correlation matrix 의 eigenvalue 분포** 에 의존. 구체 수식 + proof 는 appendix. 이 proof 가 vLLM Semantic Router Iris 대비 systems-theory contribution.

#### 2.2.4 평가 · 실험 플랜 (7 요소)

##### (1) HW
- Primary: **RTX Pro 6000 96GB** + **RTX 5090 32GB** (2-replica mini-cluster).
- Multi-replica extrapolation: trace replay on single-GPU + simulator (Orca / vLLM-v1 simulator 공개 버전).

##### (2) Model
- Primary: Qwen3-30B-A3B BF16, DeepSeek-V2-Lite BF16. Secondary: Mixtral-8x7B AWQ-4bit. Robustness: Qwen1.5-MoE (edge case).

##### (3) Dataset · Workload
- Mixed workload: WildJailbreak 2-way (balanced 8K) + MMLU 57-way (14K) + WildChat subset (10K) + LMSys-Chat (5K) Poisson arrival λ=2-10 req/s. Jailbreak injection rate 1%/5%/10% (production-realistic).
- Metrics: detection F1, expert miss rate, p50/p99 decode latency, throughput req/s, SLO attainment (TTFT ≤500ms, TPOT ≤50ms).

##### (4) Simulator · Tools
- vLLM fork (existing + M1-M3 integration).
- FAISS-GPU 1.10 + Nsight Compute counters (`lts__*`, `dram__*`).
- Profiler: vLLM benchmark_serving.py, NVIDIA NsightSystems.

##### (5) Ablation · Baseline
- Baselines: vLLM v1 vanilla, vLLM+OmniGuard, vLLM+MoE-Infinity, vLLM+PreScope, vLLM+DuoServe-MoE, vLLM+Gimbal, vLLM+Semantic Router Iris, vLLM+FJD, vLLM+WildGuard, ProMoE, BuddyMoE.
- Ablation: (M1 only) / (M1+M2) / (M1+M3) / (M1+M2+M3+M4). K ∈ {5, 15, 50} cluster, k-NN K ∈ {1, 15}. Token k ∈ {1, 4, 16, 64, full}, Layer L_k ∈ {2, 5, 10, 22, full}.
- Peer-reviewed baseline ratio: MoE-Infinity (ATC 2024), PreScope (NeurIPS 2025 accepted), Gimbal (MLSys 2026), vLLM (SOSP 2023) = **4/11 ≈ 36%**, 규율 R2 충족.
- Expected runtime: 12 weeks.
- Fallback: M4 LEAP+QIVF F1 degrade 2%+ 시 full-precision re-ranker fallback (classifier 의 top-M re-score).

##### (6) Implementation Steps (week-level)

| Week | Task | Deliverable |
|------|------|-------------|
| W1 | vLLM fork EPRT (M1) hook 구현 | (L,E) tensor in SchedulerOutput |
| W2 | token×layer 2D (M2) — 기존 trace 재분석 + k/L_k grid | Pareto frontier 1st draft |
| W3 | Multi-consumer fan-out (M3) + admission | 3-consumer prototype |
| W4 | LEAP + QIVF index (M4) | 8MB index + 3ms query |
| W5 | Mixed workload benchmark (WildJailbreak + MMLU + WildChat) | throughput/latency 표 |
| W6 | Baseline 재현 (11 baseline) | baseline table |
| W7 | Information-theoretic proof + figure | systems-theory section |
| W8 | Jailbreak injection 1%/5%/10% 실험 | safety-aware admission evaluation |
| W9 | Mixtral / Qwen1.5 generalization | cross-model validation |
| W10-11 | draft (18p) + polish | MLSys submission |
| W12 | artifact evaluation 준비 | reproducible artifact |

##### (7) Preliminary Analysis Metrics

- Stage 1 Baseline reproduction: MoE-Infinity miss rate 15-28% (decoder 512 tokens) 재현. vLLM vanilla throughput reproduction.
- Stage 2 Bottleneck attribution: forward hook overhead vs k-NN lookup overhead 분해. 예상: hook <0.5ms, lookup ≤3ms.
- Stage 3 Roofline: 2-replica throughput 이론적 상한 (SLO breach 없는 λ_max) 대비 본 방식 달성치.
- Stage 4 Micro-benchmark: 각 mechanism 단독 gain (M1 only, M2 only, etc.) 의 additivity 확인 (예상: additive within 10%).

#### 2.2.5 예상 효과

- **Detection F1 93-94%** (Qwen3 k-NN baseline 94% 근사, token×layer 2D early-exit 로 3%p 미만 손실)
- **Expert cache miss rate -20-25%** (cluster-aware admission)
- **Decode p50 latency -15%** (early-exit 로 쉬운 prompt 의 token budget 절약)
- **Memory cost +10MB** (QIVF index) vs **+28GB** (WildGuard-7B 대안)
- **vLLM upstream PR path**: M1 EPRT 는 PR 후보. M2 는 실험적 feature, M3 는 plugin stack. 산업 impact 경로 분명.

---

### 2.3 🥉 BEACON-GUARD-Lite (USENIX ATC 2027 12p 또는 DATE 2027 6p — LOOM' 과 paper pair)

> **Mechanism-Metaphor**: "하나의 beacon(routing fingerprint)이 domain + safety 두 방향을 동시에 비춘다" (원안의 3-task 에서 OOD drop → 2-task)

#### 2.3.1 개요

`(L, 4E)` 통합 Unified Routing-Feature Bank (URFB) + FAISS IVF 로 training-free multi-task (domain + safety) guard. Multi-signal (discrete topk_index + continuous softmax_scores + integer activation_count) fingerprint fusion 의 geometric property 를 mechanism-interpretability 축으로 정립. 원안 Tier-1 USENIX Security 는 novelty 약해서 **Tier-2 ATC systems / DATE 로 강등** (deployment cost 차트 중심 pivot).

#### 2.3.2 기존 연구 한계·GAP

| 기존 연구 | 한계 (BEACON-GUARD-Lite 대비) |
|-----------|-------------------------------|
| WildGuard ([arXiv:2406.18495](https://arxiv.org/abs/2406.18495)) [NeurIPS 2024] | safety only, 별도 Mistral-7B forward +200ms +28GB |
| LlamaGuard-3 [Meta] | safety only, +100ms +16GB |
| OmniGuard ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) | safety only, 120x faster 이지만 multi-task 아님 |
| FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558)) [EMNLP 2025 Findings] | safety only, first-token logit |
| Task-Cond. Routing Sig. ([arXiv:2603.11114](https://arxiv.org/abs/2603.11114)) | domain only, OLMoE 단일, preprint |
| MultiTaskGuard/UniGuard ([arXiv:2504.19333](https://arxiv.org/abs/2504.19333)) | multi-task + LoRA-shared (training required) |
| vLLM Semantic Router Iris v0.1 (2026-01) | LoRA-shared classifier heads, training required |

**GAP**: **training-free + no-finetune + no-LoRA + multi-task (domain+safety) MoE guard** 는 vLLM Semantic Router Iris + MultiTaskGuard 가 **LoRA training** 을 요구하는 점에서 차별화. Deployment cost 축이 main claim.

#### 2.3.3 제안 기법 (2-mechanism)

##### Mechanism M1 — Unified Routing-Feature Bank (URFB)

① `vllm/v1/worker/gpu_model_runner.py` 의 forward 말미에 4 signal (router_logits, softmax_scores, topk_weights, activation_count) concat → `(L, 4E)` bank. FAISS IVF IndexIVFFlat(nlist=64). Domain + safety 2 head k-NN 조회.
② 기존 stack: 3 task × 200ms = 600ms + 수 GB 메모리. 본: 1 forward + kNN <5ms.
③ step: (기존 5,900 runs 데이터 재활용 100%), FAISS 빌드, 2-head 학습, 병렬 lookup latency 측정.
④ 기존 실패: OmniGuard / Task-Cond Routing 은 각 task 단독. MultiTaskGuard / Semantic Router Iris 는 LoRA training 요구. 본 은 **완전 no-train k-NN**.

##### Mechanism M2 — Multi-Signal Fingerprint Fusion Interpretability

① `analysis/fusion_geometry.py` (~150 LOC). topk_index (discrete L×E×K) + softmax_scores (continuous L×E) + activation_count (integer L×E) 3 signal 각각의 per-class centroid 에 대해 MMD + JS divergence 계산. Signal-specific discriminability 분석.
② 기존은 single signal 만 사용 (예: OmniGuard hidden state, FJD logit). Multi-signal fusion 의 geometric property 는 미탐구. "어떤 signal 이 어떤 task 에 dominant" 가 mechanism-interpretability 축.
③ step: (a) 3 signal per-task centroid 계산, (b) MMD / JS matrix, (c) signal-weighted sum ensemble 의 optimal weight 탐색 (grid search), (d) signal-specific interpretability plot.
④ 기존 실패 + 차별화: OmniGuard 은 hidden state single. Task-Cond Routing 은 routing signatures 수치 없이 single signal. 본 연구는 **3 signal MoE geometric analysis** + LoRA-free.

#### 2.3.4 평가 · 실험 플랜 (7 요소) (간소)

##### (1) HW: RTX Pro 6000 96GB × 1 (ATC 기준 충분), 보조 RTX 5090 32GB (Mixtral).
##### (2) Model: 4 모델 (기존 사용자 pipeline).
##### (3) Dataset: MMLU 57 + WildJailbreak 8K (existing), BoolQ, TrueFalse (추가 재활용).
##### (4) Tools: FAISS-GPU, vLLM fork, signal-specific probe (~200 LOC).
##### (5) Baseline: WildGuard [NeurIPS 2024], LlamaGuard-3, FJD [EMNLP 2025 Findings], OmniGuard, Task-Cond Routing, separate-classifier-stack (domain-only + safety-only). Peer-reviewed ratio: WildGuard + LlamaGuard + FJD = **3/6 = 50%** 충족.
##### (6) Implementation Steps:

| Week | Task |
|------|------|
| W1 | URFB bank 구축 (기존 data 재활용) |
| W2 | FAISS IVF 2-head k-NN |
| W3 | Multi-signal fusion MMD/JS analysis |
| W4 | vLLM serving integration + latency/throughput 측정 |
| W5 | Baseline 재현 (FJD/OmniGuard/WildGuard/LlamaGuard-3) |
| W6 | 표/figure, polish |

##### (7) Preliminary Analysis:
- Baseline reproduction: 96.2% MMLU 4-cat / 94% WildJailbreak 재현.
- Bottleneck: FAISS lookup vs activation hook overhead 분해.
- Roofline: WildGuard 200ms 대비 50-100x 가속 목표 (OmniGuard 120x 근방).
- Micro-benchmark: 2 task 동시 lookup vs single task latency ≤2x.

#### 2.3.5 예상 효과

- WildGuard-7B 대비 **latency 50-100× ↓** (200ms → 2-4ms)
- Memory **28GB → <50MB** (LoRA head weight 제거)
- 2-task (domain + safety) unified accuracy **96.2% / 94%** 유지
- Production deployment path: vLLM plugin, NeMo Guardrails adapter

---

## 3. Tier-2 독립 Top 3 — 상세 (압축)

### 3.1 T1: DISCRETE-VEIL-Lite (IEEE CAL 4p 또는 DSN practical 6p)

- **Scope**: Qwen3 + PAIR 공격 200 prompts 단일 (AutoDAN drop, Mixtral drop, GCG drop).
- **Mechanism**: Mech M1 (DRO-Attack) only, Mech M2 drop.
- **Baseline 2 종**: FJD + OmniGuard.
- **8주 완결**: W1 reproduction, W2 attack 생성, W3 DRO-Attack, W4 fingerprint 재추출, W5 k-NN 재평가, W6 baseline, W7 표, W8 draft.
- **용도**: Tier-1 S&P 투고 전 precedence claim + preliminary result. "first-to-report MoE embedding-PGD robustness" 권리 확보.

### 3.2 T2: TALLY-Spinoff (DATE 2027 4p WIP)

- **Scope**: LEAP interpretability only. Expert-j-at-layer-i labeling via SAFEx-style (safety-critical / domain-critical / general) class assignment.
- **Mechanism**: LEAP MoE-specific axis pruning + SAFEx-style label attachment. FAISS/QIVF drop (LOOM' 과 중복 회피).
- **Baseline**: PCA, IPCA, SAFEx direct.
- **4주 완결 possible**.
- **용도**: LOOM' Section 7 과 별개 venue. 사용자 lab publication 개수 증가.

### 3.3 T3: BEACON-GUARD-Lite Tier-2 DATE 6p (fallback)

- **Scope**: Tier-1 ATC reject 시 DATE 6p 로 재제출. Multi-signal Fusion 축만 유지.
- **기간**: 이미 ATC 원고 있으면 2주 압축.

---

## 4. 미선정 아이디어 로그 (상세)

| ID | 이름 | 판정 | 미선정 사유 (구체) | 재방문 조건 |
|----|------|------|---------------------|-------------|
| A1 | DISCRETE-VEIL 원안 | **Refined → DISCRETE-VEIL'** | scope 4 models × 3 attacks 은 10주+; Mech 1.2 entropy tripwire 가 ASE 2025 NIER (routing entropy OOD) 와 50-60% 겹침 | GateBreaker 대응 defense 가 본 연구 scope 에 포함되면 Mech 3 추가로 재방문 |
| A2 | BEACON-GUARD 원안 (Tier-1 USENIX Security) | **DOWNGRADED → Tier-2 ATC/DATE** | Task-Cond Routing + MultiTaskGuard + vLLM Semantic Router Iris + ASE 2025 NIER concurrent 압박. Tier-1 novelty 미달. OOD 축은 DISCRETE-VEIL' defense-in-depth 로 이식 | Multi-signal interpretability proof + vLLM upstream PR 가 완료되면 Tier-1 재투고 |
| A3 | THRESHOLD 원안 | **DROP (LOOM' M2 에 흡수)** | EMBER 와 early-exit axis 중복 (token vs layer). 단독 유지 시 artificial split | Crescendo / Skeleton-Key attack 이 production 주류 되면 LOOM' 의 Streaming Extension Section 으로 확장 |
| A4 | LOOM 원안 | **MERGED → LOOM'** | vLLM Semantic Router Iris scoop; EMBER+TALLY+THRESHOLD 와 vLLM fork 공유 → artificial split | (별도 재방문 없음; LOOM' 로 확정) |
| A5 | EMBER 원안 | **MERGED → LOOM' M2** | LOOM 과 vLLM fork/FusedMoE hook 공유 | LOOM' reject 되면 standalone ATC 재제출 |
| A6 | TALLY 원안 | **DOWNGRADED → LOOM' M4 + DATE spinoff** | 단독 NeurIPS Systems 는 FAISS IVF-PQ + axis pruning 의 incremental novelty 부족 | (현 spinoff 유지) |
| B1 | Cross-model fingerprint alignment (Procrustes / CCA) | **Deferred** | 현재 실험 데이터 (각 모델 독립) 에서 직접 도출 불가; alignment 추가 실험 scope 가 "+α 최소" 조건 위배. 미래 vector 로 summary appendix 에 기록 | 사용자가 6개월+ 연구 방향 선정 후 별도 세션 |
| B2 | Hardware side-channel MoE guard | **Out-of-scope** | 2026-04-21 이전 세션 (FARD-C/ZMSP/PhantomRoute) 가 covered. 중복 회피 | 없음 |
| B3 | VLM/VLA MoE fingerprinting | **Out-of-scope** | 2026-04-22 이전 세션 covered. 본 세션은 text LLM | VLM/VLA 확장 별도 세션 |

---

## 5. References (핵심 arXiv 링크)

### Training-free LLM guard
- FJD: [arXiv:2509.14558](https://arxiv.org/abs/2509.14558) [EMNLP 2025 Findings]
- OmniGuard: [arXiv:2505.23856](https://arxiv.org/abs/2505.23856) [preprint, multi-lang/modal]
- HSF: [arXiv:2409.03788](https://arxiv.org/abs/2409.03788)
- HiddenDetect: [arXiv:2502.14744](https://arxiv.org/abs/2502.14744) [VLM]
- Jailbreaking Leaves a Trace: [arXiv:2602.11495](https://arxiv.org/abs/2602.11495)
- Do Internal Layers Reveal Patterns: [arXiv:2510.06594](https://arxiv.org/abs/2510.06594)
- MultiTaskGuard/UniGuard: [arXiv:2504.19333](https://arxiv.org/abs/2504.19333)
- WildGuard: [arXiv:2406.18495](https://arxiv.org/abs/2406.18495) [NeurIPS 2024]

### MoE safety + routing
- Task-Cond. Routing Sig.: [arXiv:2603.11114](https://arxiv.org/abs/2603.11114) [preprint, OLMoE single]
- What Gets Activated (Domain/Driver): [arXiv:2601.10159](https://arxiv.org/abs/2601.10159)
- L³ (Lobotomy, attack): [arXiv:2602.08741](https://arxiv.org/abs/2602.08741)
- SAFEx: [arXiv:2506.17368](https://arxiv.org/abs/2506.17368) [NeurIPS 2025 Poster]
- RASA: [arXiv:2602.04448](https://arxiv.org/abs/2602.04448)
- SteerMoE: [arXiv:2509.09660](https://arxiv.org/abs/2509.09660)
- RouteMark: [arXiv:2508.01784](https://arxiv.org/abs/2508.01784)
- MoE Lens: [arXiv:2603.05806](https://arxiv.org/abs/2603.05806)
- GateBreaker: [arXiv:2512.21008](https://arxiv.org/abs/2512.21008)
- Expert Selections Reveal Text: [arXiv:2602.04105](https://arxiv.org/abs/2602.04105)

### Adaptive adversarial
- Obfuscated Activations: [arXiv:2412.09565](https://arxiv.org/abs/2412.09565)
- V-MoE Adversarial Robustness: [OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp)
- PGD on LLM generation: [arXiv:2402.09154](https://arxiv.org/abs/2402.09154)

### MoE serving
- MoE-Infinity: [arXiv:2401.14361](https://arxiv.org/abs/2401.14361) [ATC 2024]
- ProMoE: [arXiv:2410.22134](https://arxiv.org/abs/2410.22134)
- DuoServe-MoE: [arXiv:2509.07379](https://arxiv.org/abs/2509.07379)
- PreScope: [arXiv:2509.23638](https://arxiv.org/abs/2509.23638)
- MoE-Beyond: [arXiv:2508.17137](https://arxiv.org/abs/2508.17137)
- Pre-Attention Expert Prediction: [arXiv:2511.10676](https://arxiv.org/abs/2511.10676)
- BuddyMoE: [arXiv:2511.10054](https://arxiv.org/abs/2511.10054)
- Semantic Parallelism: [arXiv:2503.04398](https://arxiv.org/abs/2503.04398)
- Gimbal: [arXiv:2602.21626](https://arxiv.org/abs/2602.21626) [MLSys 2026]
- METRO: [arXiv:2512.09277](https://arxiv.org/abs/2512.09277)
- In-depth Caching/Prefetching: [arXiv:2511.05814](https://arxiv.org/abs/2511.05814)
- OD-MoE: [arXiv:2512.03927](https://arxiv.org/abs/2512.03927)

### Benchmarks + evaluation
- WildJailbreak: [arXiv:2406.18510](https://arxiv.org/abs/2406.18510)
- JailbreakBench: [arXiv:2404.01318](https://arxiv.org/abs/2404.01318)
- HarmBench: [arXiv:2402.04249](https://arxiv.org/abs/2402.04249)
- MMLU: [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)

### vLLM Semantic Router (production)
- v0.1 Iris (2026-01 blog): vLLM project Semantic Router plugin (search vLLM docs)

### 이전 aica-research-bot 세션
- 2026-04-21 mode1 MoE fingerprinting (side-channel 전제, FARD-C/ZMSP/PhantomRoute)
- 2026-04-21 mode2 ACE-MoE VLM/VLA extension
- 2026-04-22 ~ 2026-04-24 (각 link 는 `__research_wiki/index.md` 참조)
