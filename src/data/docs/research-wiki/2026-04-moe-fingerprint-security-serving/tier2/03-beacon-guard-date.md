# Multi-Signal Fingerprint Fusion Geometric Analysis for Two-Task Mixture-of-Experts Guard (BEACON-GUARD-Lite DATE)

> [← Session Overview](/summary/2026-04-24-mode1-moe-fingerprint-security-serving/README.md) · **Tier-2 독립 #3** (Tier-1 ATC fallback)

> ## 📖 약어 풀이 (R35)
>
> - **MoE / Routing fingerprint / k-NN / FJD / OmniGuard / Task-Cond Routing / MMD / JS divergence** — Tier-1 [BEACON-GUARD idea 파일](/summary/2026-04-24-mode1-moe-fingerprint-security-serving/tier1/03-beacon-guard.md) 약어집 참조.
> - **DATE** (Design, Automation and Test in Europe) — Tier-2 short paper venue 6 page. Hardware/system 분야 본 spinoff 의 primary target.
> - **Multi-signal fusion** — 단일 signal classifier 대신 여러 signal (discrete topk_index + continuous softmax_scores + integer activation_count) 의 geometric property 통합 분석.
> - **Cosine similarity** — 두 vector 의 각도, `cos(θ) = (a·b) / (|a||b|)`. ([Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity))
> - **Ensemble (signal-weighted)** — 여러 signal 의 prediction 을 weight 곱한 sum. weight 는 grid search 로 task 별 최적값 결정.

**🎯 Target Venue**: DATE 2027 6p (primary)
**📊 Score** (Tier-2 rubric): Novelty 5.5 / Diff 6.5 / Impact 7.5 / Feasibility 9.0 = 평균 **6.90**
**✅ 판정**: Accept Tier-2 (Tier-1 ATC reject 시 fallback)

---

## 1. 개요

BEACON-GUARD Tier-1 ATC ([../tier1/03-beacon-guard.md](/summary/2026-04-24-mode1-moe-fingerprint-security-serving/tier1/03-beacon-guard.md)) 의 **Mech 2 (Multi-Signal Fingerprint Fusion Geometric Analysis) 단독 축소**. ATC 12p 에서 LOOM' 와 paper-pair 진행이 어려운 경우 DATE 6p deployment cost 차트 중심 fallback.

**Metaphor 부속 (R30)**: "Beacon-Lite" — Tier-1 의 multi-task 축 제외, fusion analysis 단독.

---

## 2. 기존 연구의 한계 / GAP

| 기존 | 한계 (DATE 6p fallback 대비) |
|------|------------------------------|
| OmniGuard ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) | hidden state single signal |
| FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558)) | first-token logit single |
| Task-Cond Routing ([arXiv:2603.11114](https://arxiv.org/abs/2603.11114)) | routing single signature |
| MultiTaskGuard ([arXiv:2504.19333](https://arxiv.org/abs/2504.19333)) | LoRA training required |

**GAP**: 3-signal MoE geometric analysis (discrete + continuous + integer) + LoRA-free 는 DATE scope 에서 first-to-report.

---

## 3. 제안 기법 (1 mechanism — BEACON-GUARD M2 만)

### M1: Multi-Signal Fingerprint Fusion Geometric Analysis (Tier-1 M2 의 단독 분리)

**① Scheme**:

`analysis/fusion_geometry.py` (~150 LOC). 3 signal (topk_index discrete + softmax_scores continuous + activation_count integer) 의 per-class centroid MMD + JS divergence + cosine matrix. Signal-weighted ensemble 의 optimal weight grid search.

> ⚠️ source proposed: `analysis/fusion_geometry.py`
> ✅ closest existing: vllm-project/vllm@`main` `vllm/model_executor/layers/fused_moe/layer.py` (3 signal 모두 이미 계산)

**② 문제 + evidence**: 단일 signal classifier (OmniGuard hidden state, FJD logit, Task-Cond routing) 대비 multi-signal geometric property 미탐구.

**③ Step-by-step**:
1. 3 signal per-class centroid 계산 (사용자 5,900 runs 재활용).
2. MMD + JS + cosine matrix (per task: WildJailbreak / MMLU).
3. Signal-weighted ensemble grid search.
4. Single-signal vs multi-signal F1 비교 ablation.
5. Signal-specific interpretability plot (어느 signal 이 어느 task 에 dominant).

**④ 차별화**: 단일 signal only 인 기존 대비 3-signal fusion. PCA fusion 보다 interpretable axis 보존.

---

## 4. 실험 플랜 (R27-β + R31 + R32)

### (1) HW: RTX 5090 32GB × 1.
### (2) Model: 4 모델 (사용자 기존).
### (3) Dataset: 사용자 5,900 runs + WildJailbreak 8K + MMLU 14K.
### (4) Tools: scipy, scikit-learn, FAISS.
### (5) Baseline: FJD + OmniGuard + Task-Cond Routing (single-signal trio).

### (6) Implementation Steps (R31)

| Step | 의존성 | Component / File (R32) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | 3 signal extraction (재활용). **`vllm/model_executor/layers/fused_moe/layer.py` ✅** | torch + vLLM 0.19+ | per-class centroid table |
| Step 2 | Step 1 | MMD + JS + cosine matrix | scipy | geometric matrix figure |
| Step 3 | Step 2 | Signal-weighted ensemble grid search | scikit-learn | optimal weight per task |
| Step 4 | Step 3 | Single-signal vs multi-signal F1 비교 | k-NN classifier | ablation table |
| Step 5 | Step 4 | Baseline 재현 (FJD/OmniGuard/Task-Cond) | each | comparison |
| Step 6 | Step 5 | 6p DATE draft + polish | manual | submission-ready |

**참고 소요**: 약 4-5 weeks (DATE 6p, ATC fallback).

### (7) Preliminary Analysis

| 측정 지표 | 도구 | 조건 | 기대 | 목표 |
|---|---|---|---|---|
| MMLU 4-cat (single signal vs fusion) | k-NN | Qwen3 | single 96.2% | fusion ≥96.5% (+0.3pp 이상) |
| WildJailbreak 2-way (fusion) | k-NN | Qwen3 | single 94% | fusion ≥94.5% |
| Latency (3 signal vs 1) | wall-clock | per query | 1ms (single) | ≤3ms (multi) |
| MMD matrix correlation | scipy | per-class | — | discrete > continuous in safety task (가설 검증) |

**Preliminary Study**: (i) single-signal F1 reproduction → (ii) MMD/JS matrix 작성 → (iii) ensemble weight grid → (iv) multi vs single ablation.

---

## 5. 예상 효과

| 지표 | Baseline | 목표 |
|---|---|---|
| Multi-signal fusion F1 | single 96%/94% | +0.3-0.5pp |
| Latency overhead | single 1ms | ≤3ms |
| Interpretability plot | 0 | per-signal-per-task heatmap |

---

## 6. ATC Tier-1 으로 scale-up 가이드

본 DATE fallback 이 Tier-1 ATC 로 가려면:
- Mech M1 (URFB unified bank) 추가
- 2-task → 3-task (OOD 추가)
- vLLM serving integration + production latency
- WildGuard / LlamaGuard baseline 추가

→ Tier-1 [03-beacon-guard.md](/summary/2026-04-24-mode1-moe-fingerprint-security-serving/tier1/03-beacon-guard.md) 참조.

---

## 7. References

- OmniGuard: [arXiv:2505.23856](https://arxiv.org/abs/2505.23856)
- FJD: [arXiv:2509.14558](https://arxiv.org/abs/2509.14558)
- Task-Cond Routing: [arXiv:2603.11114](https://arxiv.org/abs/2603.11114)
- MultiTaskGuard: [arXiv:2504.19333](https://arxiv.org/abs/2504.19333)
