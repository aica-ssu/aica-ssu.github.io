# 미선정 아이디어 모음 — 2026-04-24 MoE Fingerprint Security+Serving

> [← Session Overview](/research-wiki/2026-04/moe-fingerprint-security-serving/README.md)

원안 6 아이디어 → Tier-1 Top 3 + Tier-2 독립 Top 3 = 6 selected. 단, 원안 6 중 5 가 변경/흡수/강등 (-7 mechanism). 본 파일은 각 변경/흡수/drop 의 사유 + 재방문 조건.

---

## A1: DISCRETE-VEIL 원안 (4 models × 3 attacks 큰 scope)

- **초안 score**: Phase 2 평균 7.83 (Nov 6.5 / Diff 8.0 / Imp 9.0 / Feas 7.0)
- **최종 판정**: Refined → DISCRETE-VEIL' Tier-1 S&P (scope 축소: 2 models × 2 attacks). 8주 → 10주 feasibility 향상.
- **연구 GAP**: 원안과 동일 (text MoE routing 에 embedding-PGD 첫 정량화)
- **제안 overview**: 원안 = 4 models × 3 attacks (PAIR + GCG + AutoDAN) × 3 threat models (white + gray + black) + Mech M1 (DRO-Attack) + Mech M2 (Routing-Entropy Tripwire).
- **변경 사유 (Phase 1' Improve-First)**:
  - Scope 축소: 4 models → 2 (Qwen3 + Mixtral), 3 attacks → 2 (AutoDAN drop, complexity), 3 threat models → 2 (white + gray)
  - Mech M2 entropy tripwire 가 ASE 2025 NIER "routing entropy OOD" 와 50-60% 겹침 → "DRO-attack 특화 entropy-sharpening KS-test" 로 재정의 (distribution test 로 차별화)
  - Mechanism count Δ = 0 (improve only, no add)
- **재방문 조건**: GateBreaker ([arXiv:2512.21008](https://arxiv.org/abs/2512.21008)) 대응 defense 가 본 연구 scope 에 포함되면 Mech 3 추가로 4-models × 3-attacks 재확장 가능.

---

## A2: BEACON-GUARD 원안 (Tier-1 USENIX Security)

- **초안 score**: Phase 2 평균 6.33 (Nov 4.5 / Diff 6.0 / Imp 8.5)
- **최종 판정**: DOWNGRADED → Tier-2 ATC/DATE (BEACON-GUARD-Lite). novelty axis 재pivot.
- **연구 GAP (원안)**: training-free + no-finetune + multi-task (domain + safety + OOD) unified guard
- **제안 overview**: URFB `(L, 4E)` FAISS + 3-head k-NN (domain + safety + OOD)
- **미선정 사유 상세 (3-5 문장)**:
  - **Concurrent 압박**: Task-Cond Routing ([arXiv:2603.11114](https://arxiv.org/abs/2603.11114)) preprint 가 routing signature 92.5% 4-way 보고 (60-70% 겹침). MultiTaskGuard/UniGuard ([arXiv:2504.19333](https://arxiv.org/abs/2504.19333)) 가 multi-task LoRA-shared 14x faster 발표. **vLLM Semantic Router v0.1 Iris (2026-01)** 이 jailbreak+domain+PII+fact-check classifier production merge.
  - Mech 2.2 OOD-distance gate 가 ASE 2025 NIER "routing entropy OOD" 와 50-60% 겹침
  - **Tier-1 USENIX Security novelty 축 미달**: "조합 novelty" (OmniGuard + Task-Cond Routing + ASE NIER OOD) 라는 평가
  - DOWNGRADE 결정: Tier-1 ATC (systems/deployment cost 축) 또는 DATE 6p 로 venue 전환
  - OOD Mech 2.2 는 DISCRETE-VEIL' 의 defense-in-depth section 으로 이식
- **재방문 조건 (Tier-1 USENIX Security 재도전)**:
  - Multi-signal interpretability proof (formal MMD bound) 추가
  - vLLM upstream PR (URFB) merge 완료
  - Adversarial robustness section 추가 (DISCRETE-VEIL 의 entropy KS-test 통합)

---

## A3: THRESHOLD 원안 (NDSS 단독)

- **초안 score**: Phase 2 평균 6.00 (Nov 5.5 / Diff 5.0 / Imp 7.5)
- **최종 판정**: DROP, LOOM' M2 의 token-axis 로 흡수
- **연구 GAP**: streaming k*-token prefix convergence + drift-aware re-check (Crescendo / Skeleton-Key 방어)
- **제안 overview**: Mech 3.1 (k=16 token prefix detection) + Mech 3.2 (decode 32-step drift re-check)
- **미선정 사유 상세**:
  - EMBER 의 layer-axis early-exit 와 axis 중복 (= same underlying mechanism, 다른 quantization 축)
  - 단독 NDSS 유지 시 reviewer 의 "EMBER 와 sibling 이중 제출" 의심
  - LOOM' 으로 통합 → token×layer 2D Pareto frontier 라는 일관된 framing
- **재방문 조건**: Crescendo / Skeleton-Key attack 이 production 주류가 되면 LOOM' Streaming Extension Section 으로 독립 확장

---

## A4: LOOM 원안 (3 mechanism only)

- **초안 score**: Phase 2 평균 6.67 (Nov 5.0 / Diff 7.0 / Imp 8.0)
- **최종 판정**: MERGED → LOOM' (4 mechanism, EMBER + THRESHOLD + TALLY 흡수)
- **연구 GAP (원안)**: shared observability substrate (3 consumer fan-out)
- **제안 overview (원안)**: 3 mechanism (EPRT + SRMC + ACP)
- **미선정 사유**:
  - vLLM Semantic Router Iris v0.1 (2026-01 production merge) 와 framing 50-55% 겹침
  - "Shared read 공짜" 주장이 engineering 으로 평가
  - Information-theoretic Fisher info Pareto proof 가 없으면 systems-theory contribution 부족
- **변경 (LOOM' refinement)**:
  - EMBER + THRESHOLD + TALLY 단독 candidates 를 4 mechanism 으로 merge (artificial split 방지)
  - Information-theoretic shared-substrate Pareto proof 추가 (systems-theory)
  - Mechanism count Δ = +1 (critical merge, justified)
- **재방문 조건 (별도 변경 없음)**: LOOM' 으로 확정.

---

## A5: EMBER 원안 (ATC/OSDI 단독)

- **초안 score**: Phase 2 평균 6.67 (Nov 5.5 / Diff 7.0 / Imp 7.5)
- **최종 판정**: MERGED → LOOM' M2 (layer-axis early-exit)
- **연구 GAP**: mid-prefill jailbreak rejection (Qwen1.5-MoE L02 evidence)
- **제안 overview**: Mech 1 (ELPT early-layer partial fingerprint) + Mech 2 (ALB adaptive layer budget) + Mech 3 (BPEE batch-packed early exit)
- **미선정 사유**:
  - LOOM 과 vLLM fork / FusedMoE hook 공유 → artificial split 위험
  - EMBER 의 L_k=2 가설은 Qwen3 의 L22/48 optimal 과 상충 → adaptive layer budget 없으면 model-generality 부족
  - LOOM' M2 의 token×layer 2D Pareto 로 흡수하여 일관된 framing
- **재방문 조건**: LOOM' rejection 시 standalone ATC resubmit (M2 분리)

---

## A6: TALLY 원안 (단독 NeurIPS Sys)

- **초안 score**: Phase 2 평균 3.83 (Nov 3.0 / Diff 3.0 / Imp 5.5)
- **최종 판정**: DOWNGRADED → LOOM' M4 흡수 + DATE 4p WIP spinoff (TALLY-Spinoff)
- **연구 GAP**: fingerprint index 경량화
- **제안 overview**: Mech 1 (LEAP layer-expert variance pruning) + Mech 2 (QIVF FAISS IVF-PQ) + Mech 3 (FSE streaming eviction)
- **미선정 사유 (단독 NeurIPS Systems)**:
  - FAISS IVF-PQ 는 standard, reservoir sampling 1985 Vitter — 단독 novelty 부족
  - "LEAP MoE-specific" 주장이 PCA 대비 F1 retention 차이 ≤1%p 라 weak
  - Self-critique 가 이미 "LOOM/EMBER sub-component" 인정
- **분할 처리**:
  - LOOM' M4 로 흡수 (Tier-1 LOOM' paper 의 sub mechanism)
  - DATE 4p WIP spinoff 으로 LEAP + SAFEx interpretability axis 단독 publication ([tier2/02-tally-spinoff.md](/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/02-tally-spinoff.md))
- **재방문 조건 (단독 Tier-1)**: 별도 없음 (현 spinoff 유지).

---

## B1: Cross-Model Fingerprint Alignment (Procrustes / CCA) — Out-of-Scope

- **상태**: Deferred
- **사유**: 현재 실험 데이터 (각 모델 독립 fingerprint) 에서 직접 도출 불가; alignment 추가 실험 scope 가 사용자 "+α 최소" 조건 위배.
- **재방문**: 사용자가 6개월+ 연구 방향 선정 후 별도 세션

---

## B2: Hardware Side-Channel MoE Guard — Out-of-Scope

- **상태**: 이전 세션 covered
- **사유**: 2026-04-21 mode1 MoE fingerprinting 세션 (FARD-C / ZMSP / PhantomRoute) 이 다룸. 본 세션은 model-owner in-worker fingerprint, 전제·축 독립.
- **재방문**: 없음

---

## B3: VLM/VLA MoE Fingerprinting — Out-of-Scope

- **상태**: 이전 세션 covered
- **사유**: 2026-04-22 이전 세션 covered. 본 실험은 text LLM 4 모델 한정.
- **재방문**: VLM/VLA 확장 별도 세션
