# MoE-Specific Layer-Expert Variance Pruning with SAFEx-Style Interpretability (TALLY-Spinoff)

> [← Session Overview](/research-wiki/2026-04/moe-fingerprint-security-serving/README.md) · **Tier-2 독립 #2**

> ## 📖 약어 풀이 (R35)
>
> - **MoE / Routing fingerprint / FAISS IVF-PQ / k-NN** — Tier-1 [LOOM idea 파일](/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/02-loom.md) 의 약어집 참조.
> - **LEAP** (Layer-Expert Axis Pruning) — MoE fingerprint 의 layer-expert axis 별 variance + label MI 기준 top-D dim 만 유지하는 본 spinoff 의 핵심 mechanism. PCA 와 다르게 MoE-specific block structure 보존.
> - **PCA** (Principal Component Analysis) — 표준 dim reduction. linear projection 으로 분산 최대 axis 추출. ([Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis))
> - **IPCA** (Incremental PCA) — 메모리 제약 환경의 streaming PCA 변형. ([scikit-learn docs](https://scikit-learn.org/stable/modules/decomposition.html#incremental-pca))
> - **PQ** (Product Quantization) — 고차원 벡터를 sub-vector 로 나누고 각각 codebook quantize. FAISS IVF-PQ 의 P. ([Jegou et al. 2011](https://hal.inria.fr/inria-00514462v2/document))
> - **SAFEx** — NeurIPS 2025 Poster. Stability-based Expert Selection 으로 MoE 의 safety-critical expert 식별. 본 spinoff 는 SAFEx 의 stability score 를 LEAP axis 의 interpretable label 로 attachment. ([arXiv:2506.17368](https://arxiv.org/abs/2506.17368))
> - **MI** (Mutual Information) — random variable 간 정보 의존성. `I(X;Y) = H(X) + H(Y) - H(X,Y)`.
> - **Interpretable axis** — dim reduction 후 각 axis 가 도메인 의미를 가지는가. PCA 는 noise; SAFEx label 은 "L22 expert 47 = safety-critical" 같은 의미 보존.

**🎯 Target Venue**: DATE 2027 4p WIP (primary)
**📊 Score** (Tier-2 rubric): Novelty 6.0 / Diff 6.5 / Impact 7.0 / Feasibility 8.0 = 평균 **6.80**
**✅ 판정**: Accept Tier-2 (LOOM' M4 와 중복 회피하여 단독 publication)

---

## 1. 개요

LOOM' M4 (LEAP+QIVF) 의 단독 분리. MoE fingerprint 의 layer-expert axis variance 를 분석하여 top-D 차원 만 유지 (LEAP) + 각 axis 에 SAFEx-style **interpretable label** (safety-critical / domain-critical / general) 을 attachment. PCA 가 놓치는 MoE-specific block structure 보존 + production-ready compressed index. LOOM' Section 7 와 별개 venue 로 publication priority 확보.

**Metaphor 부속 (R30)**: "Tally" — counting + interpretable axis labels (SAFEx-style).

---

## 2. 기존 연구의 한계 / GAP

| 기존 | 한계 (TALLY-Spinoff 대비) |
|------|---------------------------|
| FAISS IVF-PQ (general) | MoE-unspecific axis selection |
| PCA / IPCA | linear projection, layer-expert block structure 손실 |
| **SAFEx** ([arXiv:2506.17368](https://arxiv.org/abs/2506.17368)) [NeurIPS 2025 Poster] ✓ | safety expert identification only, compression 미연관 |
| Task-Cond Routing ([arXiv:2603.11114](https://arxiv.org/abs/2603.11114)) | logreg single-signature, dim reduction 없음 |
| MoE Lens ([arXiv:2603.05806](https://arxiv.org/abs/2603.05806)) [ICLR 2025 Workshop] ✓ | top-1 expert characterization, indexing 없음 |

**GAP**: **MoE-specific layer-expert variance pruning + SAFEx-style interpretable label + production index** 의 통합은 공개 없음. PCA 대비 interpretability 보존 + FAISS 대비 MoE-aware axis selection.

---

## 3. 제안 기법 (1 mechanism — LOOM' M4 분리 + SAFEx label 추가)

### M1: LEAP with SAFEx-Style Interpretable Axis Labeling

**① Scheme**:

Offline preprocessing module `analysis/leap_safex.py` (~250 LOC). 
1. Train pool (253K) 의 per-dim variance + label mutual information 계산.
2. Top-D=256 dim selection (LEAP).
3. 각 selected dim 에 SAFEx-style label attachment: `(layer_i, expert_j) ∈ {safety-critical, domain-critical, general}` (SAFEx 의 stability score 사용).
4. FAISS IVF-PQ index 빌드 시 SAFEx label 을 metadata 로 보존.

> ⚠️ source proposed: `analysis/leap_safex.py` — 신규.
> ✅ closest existing: facebookresearch/faiss `IndexIVFPQ` + SAFEx repo (NeurIPS 2025 published).

**② 문제 + evidence**: 6144-dim × 253K × fp32 = ~6GB. 단순 PCA 256-dim = 8MB but interpretability 손실. LEAP+SAFEx 는 256-dim with interpretable labels.

**③ Step-by-step**:
1. Per-dim variance + MI 계산 (사용자 5,900 runs 재활용).
2. Top-D=256 selection.
3. SAFEx stability score 로 각 dim 의 label 부여.
4. FAISS IVF-PQ (nlist=256, PQ=32×8bit) 빌드 + label metadata.
5. Classifier F1 retention check (degrade ≤1%) + interpretability plot ("safety-critical" axis 가 WildJailbreak 분리에 dominant 인가 검증).

**④ 차별화**: PCA 는 interpretable axis 손실. SAFEx 는 expert identification 만. 본 연구 = 둘의 통합.

---

## 4. 실험 플랜 (R27-β + R31 + R32)

### (1) HW: RTX Pro 6000 96GB or RTX 5090 32GB.
### (2) Model: 4 모델 (사용자 기존 pipeline).
### (3) Dataset: 사용자 기존 5,900 runs + MMLU 14K + WildJailbreak 8K.
### (4) Tools: FAISS-GPU 1.10, scikit-learn (PCA baseline), SAFEx repo.
### (5) Baseline: PCA, IPCA, SAFEx direct, brute-force k-NN.

### (6) Implementation Steps (R31)

| Step | 의존성 | Component / File (R32) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | LEAP variance + MI 계산 (사용자 5,900 runs 재활용) | scikit-learn, numpy | per-dim score table |
| Step 2 | Step 1 | Top-D=256 selection + SAFEx label | SAFEx repo (NeurIPS 2025) | label-attached axis set |
| Step 3 | Step 2 | FAISS IVF-PQ + metadata | FAISS-GPU `IndexIVFPQ` | 8MB index + ≤3ms query |
| Step 4 | Step 3 | F1 retention check (k-NN 재학습) | scikit-learn k-NN | F1 degrade ≤1%p |
| Step 5 | Step 4 | Interpretability plot (label vs task discriminability) | matplotlib | DATE 4p figure 1-2 |
| Step 6 | Step 5 | Baseline 재현 (PCA, IPCA, SAFEx direct, brute) | each | comparison table |
| Step 7 | Step 6 | 4p draft + polish | manual | submission-ready |

**참고 소요**: 약 3-4 weeks (DATE 4p WIP).

### (7) Preliminary Analysis

| 측정 지표 | 도구 | 측정 조건 | 기대 | 목표 |
|---|---|---|---|---|
| Index memory | FAISS index file size | 253K vectors | brute 6GB / PCA 8MB | LEAP+QIVF ~10MB |
| Query latency | wall-clock | k=15 lookup | brute 40ms / PCA 5ms | ≤3ms |
| F1 retention | k-NN classifier | WildJailbreak 8K | 94% (full) | ≥93% |
| Label-task correlation | scipy correlation | safety-critical axis ↔ WildJailbreak | — | r > 0.7 (interpretability) |

**Preliminary Study**: (i) full k-NN 재현 → (ii) PCA 256-dim degrade 측정 → (iii) LEAP 256-dim degrade → (iv) SAFEx label correlation.

---

## 5. 예상 효과

| 지표 | Baseline | 목표 |
|---|---|---|
| Index memory | 6GB (brute) | 10MB |
| Query latency | 40ms | ≤3ms |
| F1 retention | 94% | ≥93% (degrade ≤1%p) |
| Interpretable label coverage | 0% (PCA) | 60%+ axis 에 label |

---

## 6. LOOM' 과의 관계

본 spinoff 의 LEAP + SAFEx 부분을 LOOM' M4 가 흡수 (single mechanism 인용). LOOM' Tier-1 paper 에서는 systems-theory shared-substrate Pareto 가 main, M4 는 sub. 본 spinoff 는 interpretability + compressed index 자체를 main contribution 으로 분리.

---

## 7. References

- SAFEx [NeurIPS 2025 Poster]: [arXiv:2506.17368](https://arxiv.org/abs/2506.17368)
- MoE Lens [ICLR 2025 Workshop]: [arXiv:2603.05806](https://arxiv.org/abs/2603.05806)
- Task-Cond Routing: [arXiv:2603.11114](https://arxiv.org/abs/2603.11114)
- FAISS: facebookresearch/faiss
