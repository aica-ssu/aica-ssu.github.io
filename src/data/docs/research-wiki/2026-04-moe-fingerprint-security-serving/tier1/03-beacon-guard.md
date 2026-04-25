# Training-Free Multi-Task Guard via Mixture-of-Experts Routing Fingerprint (BEACON-GUARD)

> [← Session Overview](/summary/2026-04-24-mode1-moe-fingerprint-security-serving/README.md)

> ## 📖 약어 / 핵심 용어 풀이 (R35)
>
> - **MoE** (Mixture of Experts) — 모델 일부만 sparse activate 하는 transformer 변형.
> - **URFB** (Unified Routing-Feature Bank) — 본 idea Mechanism M1. 4 signal (`router_logits`, `softmax_scores`, `topk_weights`, `activation_count`) 모두 concat → `(L, 4E)` 통합 feature bank.
> - **Training-free** — 모델 weight / classifier head fine-tune 없음. 본 idea 는 k-NN 만 사용 → 학습 0회.
> - **k-NN** (k-Nearest Neighbors) — non-parametric classifier. train pool 의 k 개 가장 가까운 점의 다수결.
> - **FAISS IVF** — Meta vector similarity search. IVF (Inverted File) cluster 기반 approximate k-NN. ([공식 docs](https://github.com/facebookresearch/faiss/wiki))
> - **WildGuard** — Allen Institute 의 jailbreak guard model. Mistral-7B fine-tune. +200ms +28GB. 본 idea 의 직접 비교 대상. ([arXiv:2406.18495](https://arxiv.org/abs/2406.18495), NeurIPS 2024)
> - **LlamaGuard-3** — Meta 의 guard. 8B Llama fine-tune. +100ms +16GB.
> - **OmniGuard** — 주 LLM 의 internal representation 으로 multi-lingual / multi-modal jailbreak detection, 120x faster. ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856))
> - **FJD** (Free Jailbreak Detection) — EMNLP 2025 Findings. dense LLM first-token logit 기반 training-free detection. ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558))
> - **MultiTaskGuard / UniGuard** — 14x faster multi-task guard. 단 LoRA training 요구. ([arXiv:2504.19333](https://arxiv.org/abs/2504.19333))
> - **vLLM Semantic Router Iris** — 2026-01 production merge. LoRA-shared multi-classifier 가 본 idea 와 가장 가까운 industry baseline. 본 idea 차별점 = LoRA 학습 0회.
> - **Task-Conditioned Routing** — arXiv preprint, OLMoE 단일 모델에서 routing signature → task classification 92.5%. domain 측면에서 60-70% concurrent. ([arXiv:2603.11114](https://arxiv.org/abs/2603.11114))
> - **MMD** (Maximum Mean Discrepancy) — 두 분포 간 거리, RKHS embedding mean 의 차이. 본 idea Mech M2 의 multi-signal fusion analysis 에 사용. ([Gretton et al. 2012](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html))
> - **JS divergence** (Jensen-Shannon) — 두 분포 간 symmetric 거리, KL divergence 의 symmetric 변형. ([Wikipedia](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence))
> - **Multi-task** — 한 모델 / system 이 여러 task (domain classification + safety detection + OOD detection) 를 동시에 처리.
> - **OOD** (Out-Of-Distribution) — 학습 데이터 분포 외의 입력. 본 idea 의 OOD detection 은 k-NN distance 가 95-percentile 초과 시 fallback.

**🎯 Target Venue**: USENIX ATC 2027 (12p) (primary) / DATE 2027 (6p, fallback paper-pair with LOOM)
**📊 Score** (Tier-2 rubric): Novelty 5.5 / Diff 6.5 / Impact 8.0 / Feasibility 9.0 = 평균 **7.25**
**✅ 판정**: Conditional Accept Tier-2 (Paper pair with LOOM)

---

## 1. 개요

`(L, 4E)` 통합 Unified Routing-Feature Bank (URFB) + FAISS IVF 로 **training-free** multi-task (domain + safety) guard. Multi-signal (discrete topk_index + continuous softmax_scores + integer activation_count) fingerprint fusion 의 geometric property 를 mechanism-interpretability 축으로 정립. **추가 forward 0 + LoRA training 0** 의 deployment cost 차트 중심 pivot.

원안 Tier-1 USENIX Security 는 novelty 약 (Task-Cond Routing + MultiTaskGuard + vLLM Semantic Router Iris + ASE 2025 NIER concurrent 압박) → **Tier-2 ATC systems / DATE 6p 강등**.

**Metaphor 부속 (R30)**: "Beacon" = 등대. 하나의 fingerprint 가 domain + safety 두 방향을 동시에 비춤.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 한계 (BEACON-GUARD 대비) |
|-----------|-------------------------------|
| WildGuard ([arXiv:2406.18495](https://arxiv.org/abs/2406.18495)) [NeurIPS 2024] ✓ | safety only, +200ms +28GB |
| LlamaGuard-3 [Meta] | safety only, +100ms +16GB |
| OmniGuard ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) | safety only, 120x faster but multi-task 아님 |
| FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558)) [EMNLP 2025 Findings] ✓ | safety only, first-token logit |
| Task-Cond Routing ([arXiv:2603.11114](https://arxiv.org/abs/2603.11114)) | domain only, OLMoE 단일, preprint |
| MultiTaskGuard/UniGuard ([arXiv:2504.19333](https://arxiv.org/abs/2504.19333)) | multi-task BUT **LoRA training required** |
| vLLM Semantic Router Iris v0.1 (2026-01) | LoRA classifier heads (training required) |

**GAP**: **training-free + no-finetune + no-LoRA + multi-task (domain + safety) MoE guard** 는 vLLM Semantic Router Iris + MultiTaskGuard 가 LoRA training 요구하는 점에서 차별화. Deployment cost 축 main claim.

---

## 3. 제안 기법 (Core Mechanisms, 2 mechanisms — improve+pivot ΔM=0)

### M1: Unified Routing-Feature Bank (URFB)

**① Scheme — Source Verified (R32)**:

vLLM v1 model runner forward 말미에 hook 추가 (LOOM EPRT 와 공유). 4 signal (router_logits, softmax_scores, topk_weights, activation_count) concat → `(L, 4E)` bank. FAISS IVF IndexIVFFlat(nlist=64). Domain + safety 2 head k-NN 조회 (병렬 lock-free).

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/worker/gpu_model_runner.py`
> ✅ source verified: vllm-project/vllm@`main` `vllm/model_executor/layers/fused_moe/layer.py`
> ⚠️ source proposed: `serving/urfb_index.py` — 신규 module (FAISS wrapper).

**② 문제 + evidence**: 기존 stack: 3 task × 200ms = 600ms + 수 GB 메모리. 본: 1 forward + kNN <5ms. 기존 5,900 runs 데이터 100% 재활용.

**③ Step-by-step**:
1. 기존 5,900 runs 에서 4 signal 통합 (L, 4E) bank 구성 (재활용).
2. FAISS IndexIVFFlat(nlist=64) 빌드.
3. 2-head (domain-57 / 4-cat + safety-2 / 4-way) k-NN classifier 학습.
4. vLLM hook 통합 + 병렬 lookup.
5. Latency / throughput 측정.

**④ 차별화**: OmniGuard/FJD 는 safety only. Task-Cond Routing 은 domain only. WildGuard/LlamaGuard 는 별도 forward 필요. MultiTaskGuard/Iris 는 LoRA training 필요. 본 연구는 **완전 no-train k-NN + 추가 forward 0**.

### M2: Multi-Signal Fingerprint Fusion Geometric Analysis

**① Scheme**:

`analysis/fusion_geometry.py` (~150 LOC). 3 signal 각각 per-class centroid 에 대해 MMD + JS divergence + cosine similarity 계산. Signal-specific discriminability matrix → optimal weighted ensemble.

> ⚠️ source proposed: `analysis/fusion_geometry.py` — 신규 analysis 모듈

**② 문제**: 기존 single signal only (OmniGuard hidden state, FJD logit). Multi-signal fusion 의 geometric property 미탐구. "어떤 signal 이 어떤 task 에 dominant" 가 mechanism-interpretability 축.

**③ Step-by-step**:
1. 3 signal per-task per-class centroid 계산 (재활용).
2. MMD + JS divergence + cosine matrix 작성.
3. Signal-weighted sum ensemble 의 optimal weight grid search.
4. Signal-specific interpretability plot (어느 signal 이 어느 task 에 dominant 인가).
5. Ablation: signal subset vs multi-signal full.

**④ 차별화**: OmniGuard 는 hidden state single. Task-Cond Routing 은 single signal 92.5%. 본 연구 = **3 signal MoE geometric analysis + LoRA-free**.

**Tier 구성**: physical 1-tier + software 1-tier (k-NN single classifier).

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32)

### (1) HW: RTX Pro 6000 96GB × 1 (ATC 충분), 보조 RTX 5090 32GB.
### (2) Model: 4 모델 (기존 사용자 pipeline).
### (3) Dataset: MMLU 57 + WildJailbreak 8K (existing) + BoolQ + TrueFalse (재활용).
### (4) Tools: FAISS-GPU + vLLM fork + signal-specific probe (~200 LOC).
### (5) Baseline: WildGuard [NeurIPS 2024], LlamaGuard-3, FJD [EMNLP 2025 Findings], OmniGuard, Task-Cond Routing, separate-classifier-stack (domain-only + safety-only).

Peer-reviewed: WildGuard + LlamaGuard + FJD = **3/6 = 50%**.

### (6) Implementation Steps (R31)

| Step | 의존성 | Component / File (R32) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | URFB bank 구축. **`vllm/v1/worker/gpu_model_runner.py` ✅ + `vllm/model_executor/layers/fused_moe/layer.py` ✅** | torch + vLLM 0.19+ | (L, 4E) tensor + 기존 5900 runs 재활용 |
| Step 2 | Step 1 | FAISS IVF 2-head k-NN | FAISS-GPU 1.10, `IndexIVFFlat` | 4 모델 × 2 task k-NN 학습 + 기존 96.2% / 94% 재현 |
| Step 3 | Step 1 | Multi-signal fusion MMD/JS analysis | scipy, scikit-learn | 3 signal interpretability matrix + plot |
| Step 4 | Step 2 | vLLM serving integration + latency/throughput | vLLM `benchmark_serving.py` | latency ≤5ms / query, throughput overhead <3% |
| Step 5 | Step 4 | Baseline 재현 (FJD/OmniGuard/WildGuard/LlamaGuard-3/Task-Cond Routing/separate stack) | each repo | baseline table 완성 |
| Step 6 | Step 3-5 | 표 / figure / draft (12p ATC) | matplotlib, manual writing | submission-ready |

**참고 소요**: 약 6-8 weeks (Tier-2 ATC). DATE 6p 로는 약 4-5 weeks.

### (7) Preliminary Analysis

| 측정 지표 | 도구 | 측정 조건 | 기대 | 목표 |
|---|---|---|---|---|
| MMLU 4-cat accuracy | k-NN + faiss | Qwen3, k=1 | 96.2% | 96.2% 재현 |
| WildJailbreak 2-way | k-NN + faiss | Qwen3, k=15 | 94.0% | 94.0% 재현 |
| FAISS IVF query latency | wall-clock | 8K pool | brute 40ms | ≤5ms |
| Activation hook overhead | Nsight Systems | per prompt | — | <0.5ms |
| WildGuard latency | WildGuard-7B forward | 동일 prompt | +200ms | reference |
| 본 latency vs WildGuard | wall-clock | 동일 prompt | — | **50-100× 빠름 (2-4ms)** |
| 2-task 동시 lookup vs single | FAISS parallel | 동일 query | — | ≤2× single latency |

**Preliminary Study 4-단계**: (i) MMLU 96.2% / WildJailbreak 94% 재현 → (ii) FAISS lookup vs activation hook overhead 분해 → (iii) WildGuard 200ms 대비 50-100x roofline → (iv) 2-task 동시 lookup latency micro-benchmark.

---

## 5. 예상 효과

| 지표 | Baseline | 목표 |
|---|---|---|
| Latency | WildGuard +200ms | **+2-4ms (50-100× ↓)** |
| Memory | WildGuard +28GB | **+<50MB FAISS index** |
| MMLU 4-cat accuracy | 96.2% | 96.2% 유지 |
| WildJailbreak 2-way | 94.0% | 94.0% 유지 |
| Deployment path | 신규 모델 학습 | **vLLM plugin / NeMo Guardrails adapter** (no train) |

---

## 6. (Tier-2 → Tier-1 재시도 가이드)

본 idea 가 Tier-1 USENIX Security 로 재도전 가능하려면:
- **Multi-signal interpretability proof** (formal MMD bound) 추가
- vLLM upstream PR (URFB) merge 완료
- Adversarial robustness section 추가 (DISCRETE-VEIL 의 entropy KS-test 통합)

→ 위 3 조건 충족 시 USENIX Security 재도전 가능. 현 Phase 에서는 ATC/DATE Tier-2 로 진행.

---

## 7. References

- WildGuard [NeurIPS 2024]: [arXiv:2406.18495](https://arxiv.org/abs/2406.18495)
- LlamaGuard-3 [Meta]
- OmniGuard: [arXiv:2505.23856](https://arxiv.org/abs/2505.23856)
- FJD [EMNLP 2025 Findings]: [arXiv:2509.14558](https://arxiv.org/abs/2509.14558)
- Task-Cond Routing: [arXiv:2603.11114](https://arxiv.org/abs/2603.11114)
- MultiTaskGuard/UniGuard: [arXiv:2504.19333](https://arxiv.org/abs/2504.19333)
- vLLM Semantic Router Iris v0.1 (2026-01)
- ASE 2025 NIER "Unseen data detection via routing entropy"
- HSF: [arXiv:2409.03788](https://arxiv.org/abs/2409.03788)
- HiddenDetect: [arXiv:2502.14744](https://arxiv.org/abs/2502.14744)
