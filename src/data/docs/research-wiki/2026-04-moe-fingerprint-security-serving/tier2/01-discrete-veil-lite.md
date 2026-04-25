# Embedding-Space PGD on Mixture-of-Experts Routing Classifier — Single-Model Precedence Study (DISCRETE-VEIL-Lite)

> [← Session Overview](/research-wiki/2026-04/moe-fingerprint-security-serving/README.md) · **Tier-2 독립 #1**

> ## 📖 약어 풀이 (R35, 핵심만)
>
> - **MoE / Routing fingerprint / Jailbreak / DRO-Attack / PGD / Gumbel-softmax / WildJailbreak / vLLM** — 모두 Tier-1 [DISCRETE-VEIL idea 파일](/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/01-discrete-veil.md) 의 약어집 참조.
> - **Precedence claim** — 동일 attack 공간에서 "first-to-report" 우선 publication 으로 inventor 주장. Tier-1 large paper 보다 먼저 short paper 로 publish 하여 priority 확보 전략.
> - **IEEE CAL** (IEEE Computer Architecture Letters) — short paper venue, 4 page, computer architecture letters journal. Tier-2 의 대표적 publication 통로.

**🎯 Target Venue**: IEEE CAL 4p (primary) / DSN 2027 practical 6p (alternative)
**📊 Score** (Tier-2 rubric): Novelty 6.5 / Diff 7.0 / Impact 7.5 / Feasibility 7.5 = 평균 **7.10**
**✅ 판정**: Accept Tier-2 (precedence claim 우선)

---

## 1. 개요

DISCRETE-VEIL Tier-1 의 single-mechanism 축소 study. **Qwen3 + PAIR 단일 공격 200 prompts** 로 "first-to-report MoE embedding-PGD on text routing classifier" precedence claim. Tier-1 S&P submission 전 IEEE CAL 4p publication priority 확보 또는 Tier-1 reject 시 fallback.

**Metaphor 부속 (R30)**: "Veil-Lite" — Tier-1 의 축소 버전.

---

## 2. 기존 연구의 한계 / GAP

DISCRETE-VEIL Tier-1 [01-discrete-veil.md](/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/01-discrete-veil.md) 의 §2 와 동일하지만, scope 축소로 V-MoE PGD baseline + GateBreaker 비교는 1-2 줄 brief reference 만:

| 기존 | 본 Lite 와 차별 |
|------|-----------------|
| Obfuscated Activations ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)) | dense only |
| V-MoE Adversarial ([OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp)) | image, text 미적용 |
| FJD ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558)) | adversarial 평가 없음 |

**GAP**: text MoE routing classifier 에 embedding-PGD 가 **first-to-report** in narrow scope.

---

## 3. 제안 기법 (1 mechanism — DISCRETE-VEIL Tier-1 의 M1 만)

### M1: DRO-Attack (Tier-1 M1 와 동일, scope 축소)

**① Scheme — Source Verified (R32)**:

`attacks/dro_attack.py` (Tier-1 와 공유). vLLM hook 위치 동일.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/worker/gpu_model_runner.py`
> ✅ source verified: vllm-project/vllm@`main` `vllm/model_executor/models/qwen3_moe.py`

**② 문제 + evidence**: Qwen3 WildJailbreak recall 94% (clean) → DRO-attack 후 변화 정량 (narrow scope, single model).

**③ Step-by-step**:
1. PAIR 200 prompts 생성 (Qwen3 only).
2. Embedding PGD (L∞ ≤ 8/255, 20 iter) + Gumbel-softmax surrogate (τ=0.5, hard=True).
3. Joint loss: `L = CE + λ·MSE(topk_surrogate, benign)`.
4. 공격 후 fingerprint 재추출 + k-NN 재평가.

**④ 차별화**: V-MoE PGD (image) 와 modality 구분, GateBreaker (weight ablation) 와 attack surface 구분 — 본 Lite 가 text MoE routing 에 PGD 적용 first-to-report.

---

## 4. 실험 플랜 (R27-β + R31 + R32)

### (1) HW: RTX Pro 6000 96GB × 1.
### (2) Model: Qwen3-30B-A3B BF16 only.
### (3) Dataset: WildJailbreak 8K + PAIR 200 attacked prompts.
### (4) Tools: vLLM fork + nanoGCG / harmbench PAIR + PyTorch 2.5.
### (5) Baseline: FJD + OmniGuard + hidden-state probe (Obfuscated Activations 재현). Peer-reviewed = 1/3 (FJD), single model 임을 감안하여 Tier-2 rubric 충족.

### (6) Implementation Steps (R31)

| Step | 의존성 | Component / File (R32) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | vLLM fork 검증 + 기존 fingerprint 재현. **`vllm/v1/worker/gpu_model_runner.py` ✅** | vLLM 0.19+ | Qwen3 94% ±1% 재현 |
| Step 2 | Step 1 | PAIR 200 prompts 생성 | `harmbench/attacks/pair.py` | clean attack corpus |
| Step 3 | Step 2 | DRO-Attack PGD + Gumbel-softmax (Tier-1 모듈 재사용) | `gumbel_softmax(hard=True)` + custom PGD | 200 perturbed fingerprints |
| Step 4 | Step 3 | k-NN 재평가 + recall delta 측정 | FAISS-GPU | recall 분포 (clean vs attacked) |
| Step 5 | Step 4 | 3 baseline 재현 (FJD / OmniGuard / hidden probe) | each repo | baseline table |
| Step 6 | Step 5 | 표 + figure 작성 (4p IEEE CAL) | matplotlib | submission-ready |

**참고 소요**: 약 4-6 weeks (Tier-2 short paper).

### (7) Preliminary Analysis

| 측정 지표 | 도구 | 측정 조건 | 기대 | 목표 |
|---|---|---|---|---|
| Qwen3 clean recall | k-NN | 94% reproduction | 94% ±1% | 재현 |
| DRO-Attack 후 recall | k-NN | 200 PGD prompts | — | **≥55% (precedence claim) / 30-55% (partial)** |
| Hidden probe drop | linear probe | 동일 200 prompts | 100% → ? | 0-20% 붕괴 |
| 격차 | — | discrete vs continuous | — | >30%p (성공 기준) |

**Preliminary Study**: (i) baseline 재현 → (ii) DRO PGD 수렴 확인 → (iii) recall drop curve → (iv) hidden probe 대조군.

---

## 5. 예상 효과

| 지표 | 목표 |
|---|---|
| First-to-report MoE embedding-PGD (text) | **precedence 확보** |
| Qwen3 recall 유지 | ≥55% (Tier-1 prep) or 30-55% (Lite-only) |
| Submission timeline | 4-6 weeks (Tier-2) |
| Tier-1 S&P 진행 가능성 | scope 확장 시 |

---

## 6. Tier-1 으로 scale-up 가이드

본 Lite 가 Tier-1 S&P 로 scale-up 가능하려면:
- 4 모델 (Mixtral 추가)
- 2 공격 (GCG 추가)
- Mech M2 entropy KS-test
- V-MoE PGD text adaptation baseline
- 4-attack-Venn diagram

→ Tier-1 [01-discrete-veil.md](/research-wiki/2026-04/moe-fingerprint-security-serving/tier1/01-discrete-veil.md) 참조.

---

## 7. References

DISCRETE-VEIL Tier-1 §7 와 동일 subset.
- Obfuscated Activations: [arXiv:2412.09565](https://arxiv.org/abs/2412.09565)
- V-MoE Adversarial: [OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp)
- FJD [EMNLP 2025 Findings]: [arXiv:2509.14558](https://arxiv.org/abs/2509.14558)
- OmniGuard: [arXiv:2505.23856](https://arxiv.org/abs/2505.23856)
- WildJailbreak: [arXiv:2406.18510](https://arxiv.org/abs/2406.18510)
