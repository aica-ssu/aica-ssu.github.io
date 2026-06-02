# 미선정 / Drop Ideas — VLM Edge Multi-turn/Multi-focus Vision-KV Cache-Reuse

> 본 세션 (2026-06-02 Mode 2) 의 Top-M 6 (Tier-1 {R1, R2, R3} + Tier-2 {R4, R5, R6}) 에 선정되지 않은 idea. 각 항목: GAP / 시도 overview / 미선정 사유 (scoop name + 유사도 또는 reviewer finding) / 재방문 조건 + R67 ★/▼ 1줄. **net drop 0** — drop-as-standalone 은 모두 Top-M idea 에 흡수됨.

---

## 1. R7 COMPUTE-PREFETCH — Speculative Compute-Idle Dequant (Major Revision, mean 5.90)

- **GAP**: multi-focus 대화에서 다음 turn focus 가 R1 cold tier 에 있으면 cold→hot promote latency 가 TTFT 를 지배. focus trajectory 의 spatial autocorrelation 으로 다음 focus 예측 가능하다면 speculative prefetch 로 latency hiding 가능.
- **시도 overview**: 경량 next-focus predictor (query history + spatial autocorrelation, training-free Markov/CLIP saliency) → cold-tier vision KV 의 **dequant compute 를 decode 의 compute-idle 슬롯에 speculative 실행** (CF-1 전면 재설계: `cudaMemPrefetchAsync` Orin 미지원 → `cudaStreamCreateWithPriority` 저우선 stream; memory bandwidth piggyback 불가 → compute piggyback, Modality Inflation GPU underutilization 근거). hit 해도 R1 cache 비파괴라 정합성 안전.
- **미선정 사유**:
  - **standalone novelty 부족** — prefetch 는 교과서 system 기법 (novelty Mech 5/10 전체 최저). focus-prediction head 가 유일한 신규인데 **InfiniGen (OSDI'24, speculative KV prefetch)** 이 직접 인접 (text KV → vision focus 차별화는 입증 부담).
  - **R1 강한 종속** — latency-hiding enhancement layer, R1 cold tier 없이 성립 안 함 → 독립 paperable unit 아님 (diff F07, impact F03).
  - **전제 evidence 0건** — "focus trajectory 예측 가능" 가정의 직접 evidence 가 corpus 31편 + step0a 어디에도 없음. R4 benchmark 가 spatial autocorr 를 먼저 측정해야 성립 → **R4 dependency chain** (circular: R7 gate 가 R4 산출물에 의존).
  - CF-1 재설계로 AI-impl 4.0→5.5, arch 5→6.5 (dead→working path) 회복했으나 mean 5.90 = Major Revision boundary, Tier-2 독립 3 (R4/R5/R6) 대비 전 축 열세.
- **재방문 조건**: R4 benchmark 가 **focus spatial autocorrelation > uniform** 입증 + Orin decode **compute-idle 존재** pilot 통과 시, R1 의 latency-hiding component 로 W14-16 부활. 그 전엔 R1 reactive promote 로 충분.
- **R67 ★/▼**: ★ AI-impl-D6 7 (CF-1 dead→working path) / ▼ **Novelty-Mech 5 (전체 최저)**.

---

## 2. A SubCover — Submodular Coverage Retention with (1−1/e) Multi-Focus Recall Guarantee (DROP-as-standalone → R4 흡수)

- **GAP**: SparseVILA/CSP/StreamMem 의 token 보존이 전부 heuristic (보장 없음). 같은 image/video 에 turn 마다 다른 focus 질문 = maximum coverage 문제 → 보존 subset S 가 future-focus 집합 F 를 최대 cover.
- **시도 overview**: coverage objective `g(S)=Σ_f w_f·𝟙[∃t∈S: attn(f→t)>τ]` (monotone submodular) 를 edge KV budget(cardinality |S|≤k) 하 greedy(CELF lazy-eval) 선택 → multi-focus recall 에 **(1−1/e)≈0.632 근사 보장** (Nemhauser-Wolsey-Fisher 1978, tight).
- **미선정 사유 (scoop + 유사도)**: **MMTok ([arXiv:2508.18264](https://arxiv.org/abs/2508.18264), maximum coverage for VLM vision token, single-prompt within-instance)** + **Adaptive Greedy Frame ([arXiv:2603.20180](https://arxiv.org/abs/2603.20180), explicit (1−1/e) under frame budget, single-query)** 가 coverage + (1−1/e) for VLM token 을 **~70% 선점** (novelty F07 critical scoop). (1−1/e) standalone 주장 유지 불가.
- **재방문 조건 / 흡수**: R4 FOCUS-COVERAGE 에 **distribution-marginal coverage** 로만 흡수 (single-prompt 아닌 multi-turn future-focus distribution marginal). (1−1/e) standalone 주장 폐기. coverage 함수 submodularity (diminishing returns) toy 검증은 R4 gate 로 잔존.
- **R67 ★/▼**: ★ 수학적 보장 (coverage 정식화) / ▼ **Novelty 6 (MMTok/Adaptive Greedy ~70% scoop)**.

---

## 3. G FocusBench — Multi-Focus Recall Benchmark + Coverage Metric (DROP-as-standalone → R4 흡수)

- **GAP**: Video-MME/MLVU/EgoSchema 전부 single-turn MCQ. SparseVILA/StreamMem 도 multi-focus recall 정량 미공개. 사용자 insight 를 측정할 metric/benchmark 자체가 부재.
- **시도 overview**: 같은 image/video 에 N개 focus trajectory T=(f_1,...,f_n) 평가셋 + **Focus-Coverage Recall (FCR)** + Multi-Focus Accuracy Gap Δ + Coverage-Energy Pareto metric. grounding(object/region) 으로 focus 후보 + vqa_oracle GT.
- **미선정 사유 (reviewer finding)**: standalone benchmark D2=4 (triviality) — 정책 없는 측정도구 단독은 novelty 약함 (novelty reviewer). "Are We Using the Right Benchmark"([arXiv:2510.07143](https://arxiv.org/abs/2510.07143))는 compression bench noise 만 critique 하나 multi-turn eval 미제안 → metric 신규성은 인정되나 standalone unit 부족.
- **재방문 조건 / 흡수**: R4 FOCUS-COVERAGE 에 흡수 — FCR/Coverage-Energy Pareto metric 정식 + B(distribution-marginal retention) + A(coverage marginal) 와 묶어 "측정도구 + 방어" 두 기둥 paperable unit. GT 외적 타당성 (human drift 상관) 은 R4 gate.
- **R67 ★/▼**: ★ Impact-Breadth (device/model-agnostic community 자산) / ▼ **D2=4 (standalone triviality)**.

---

## 4. VISION-BLASST — Cache-Resident Vision-Aware Softmax-Threshold Block Skipping (DROP-as-standalone → R6 흡수)

- **GAP**: BLASST 의 online-softmax threshold block-skip 은 KV 비파괴로 query 별 V-load 만 skip 하나 vision token redundancy 를 미활용 (vision=text 동일 취급).
- **시도 overview**: BLASST 를 vision-aware 로 확장 — vision KV block 의 spatial/semantic redundancy 를 softmax-threshold 와 결합해 cache 보존(multi-focus 안전)하되 memory-bound decode 의 V-load 를 vision-block 단위로 더 공격적으로 skip (λ_vision < λ_text).
- **미선정 사유**: legacy-system 의 **F6 (Idea F6) 의 부분집합** — F6 가 modality-aware λ + Ampere kernel + solution-first 정당화 + Quest 차별화까지 포함하므로 VISION-BLASST(ai-opt)는 F6 에 완전 포함. 중복 제거 차원의 merge.
- **재방문 조건 / 흡수**: R6 VKV-SKIP 으로 통합 (VISION-BLASST + F6 → R6). Quest/MInference scoop 차별화 + vision vs text redundancy 분포 분리(KL/Wasserstein) + Ampere SM87 kernel 이 R6 의 gate.
- **R67 ★/▼**: ★ Diff-Clarity (cache 비파괴=multi-focus 안전) / ▼ AI-Kernel (Ampere kernel, R6 로 계승되어 4.5 전체 최저).

---

## 5. PREFETCH standalone — Speculative Next-Focus Prediction (DROP-as-standalone → R7 component)

- **GAP**: multi-focus 대화에서 다음 focus 가 cold tier 에 있으면 promote latency 가 TTFT 지배. saliency/query-history 의 sequential 패턴은 예측 가능.
- **시도 overview**: speculative decoding 의 draft-verify 를 focus-prediction 으로 전용 — 경량 predictor 가 다음 focus region 예측 → cold-tier vision KV 를 decode-idle 중 prefetch/dequant (token speculation 이 아닌 KV-block speculation).
- **미선정 사유 (HW 사실 + 종속)**: **Orin NX `cudaMemPrefetchAsync` / `cudaMemAdvise(PreferredLocation)` 미지원** (CF-1, `concurrentManagedAccess=0`, CUDA for Tegra appnote) → managed-memory prefetch path 사실 무효. + R1 cold tier 강한 종속 (standalone novelty 부족).
- **재방문 조건 / 흡수**: R7 COMPUTE-PREFETCH 로 재설계 (PREFETCH-FOCUS + F4 → R7), compute-idle dequant piggyback 으로 전환. 단 R7 자체가 미선정 (위 §1) → R1 component 로만 부활 가능.
- **R67 ★/▼**: ★ Latency (focus-shift TTFT hiding) / ▼ **Novelty-Mech (prefetch 교과서 + Orin API 미지원)**.

---

## 미선정 종합

| idea | 유형 | 흡수처 | 핵심 사유 | 재방문 게이트 |
|---|---|---|---|---|
| **R7 COMPUTE-PREFETCH** | 미선정 (Major Revision) | R1 component | standalone novelty 5/10 + focus autocorr evidence 0 | R4 autocorr>uniform + compute-idle pilot |
| **A SubCover** | drop-as-standalone | R4 (distribution-marginal) | MMTok/Adaptive Greedy coverage+(1−1/e) ~70% scoop | (1−1/e) standalone 폐기, marginal 만 |
| **G FocusBench** | drop-as-standalone | R4 (metric+retention 묶음) | standalone benchmark D2=4 triviality | R4 GT 외적 타당성 |
| **VISION-BLASST** | drop-as-standalone | R6 | F6 부분집합 (중복) | R6 Quest 차별화 + Ampere kernel |
| **PREFETCH standalone** | drop-as-standalone | R7 (→R1 component) | Orin prefetch API 미지원 + R1 종속 | R7 부활 게이트 |

**net drop 0** — 모든 자산이 Top-M 6 또는 그 component 로 흡수 (19 Phase 1 idea → 7 refined → Top-M 6 + R7 component). Phase 1' merge: 12 absorbed, 0 net standalone drop.
