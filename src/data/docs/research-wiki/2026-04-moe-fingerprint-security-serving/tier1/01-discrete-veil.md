# Adversarial Robustness of Discrete Routing in Mixture-of-Experts Jailbreak Detection (DISCRETE-VEIL)

> [← Session Overview](/research-wiki/2026-04/moe-fingerprint-security-serving/README.md)

> ## 📖 약어 / 핵심 용어 풀이 (R35)
>
> - **MoE** (Mixture of Experts) — 모델 weight 의 일부만 sparse activate 하는 transformer 변형. ([Switch Transformer](https://arxiv.org/abs/2101.03961))
> - **Routing fingerprint** — MoE forward 1회로 얻는 `(L, E)` (layer × expert) 행렬, 어떤 expert 가 얼마나 활성됐는지의 prompt-level signature.
> - **Jailbreak** — LLM 의 safety alignment 를 우회하여 정책 위배 출력을 유도하는 입력. WildJailbreak / HarmBench 등이 표준 평가셋.
> - **DRO-Attack** (Discrete Routing Obfuscation Attack) — 본 연구 신규 공격. MoE router 의 top-k 선택을 "jailbreak generation 성공" + "benign-look routing pattern" 두 목표로 동시에 perturb. 핵심 trick: discrete argmax 의 미분 불가능을 Gumbel-softmax surrogate 로 우회. (본 idea Mech M1)
> - **PGD** (Projected Gradient Descent) — Adversarial example 생성의 표준 iterative attack. 입력 x 에 perturbation δ 를 ∇L 방향으로 step α 만큼 더하고 ε-ball 안에 projection 을 N iter 반복. ([Madry et al. 2018 ICLR arXiv:1706.06083](https://arxiv.org/abs/1706.06083))
> - **ε-ball** — adversarial perturbation 의 크기 제약. L∞ ε=8/255 = 픽셀당 최대 8/255 변화 허용 (이미지 표준), L2 ε=0.3 = norm 제약.
> - **Gumbel-softmax** — `argmax` 같은 discrete sampling operation 의 미분 가능 surrogate. Temperature τ → 0 시 hard argmax 에 수렴. backprop 가능. ([Jang et al. 2017 ICLR arXiv:1611.01144](https://arxiv.org/abs/1611.01144))
> - **Hard mode (`hard=True`)** — Gumbel-softmax 가 forward 시 one-hot 출력 (실제 argmax 와 동일), backward 시 미분 가능 surrogate gradient 를 사용하는 straight-through estimator 모드.
> - **KS-test** (Kolmogorov-Smirnov test) — 두 분포가 같은지 비교하는 non-parametric 통계 검정. KS-statistic = sup|F1(x) - F2(x)|. 본 연구는 clean fingerprint vs adversarial-attacked fingerprint 의 entropy 분포 KS-statistic 으로 anomaly detection. ([scipy.stats.ks_2samp 문서](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html))
> - **Shannon entropy** — 분포의 불확실성 측정. `H = -Σ p log p`. 본 연구: per-layer softmax_scores 의 entropy 가 PGD 공격 후 감소 (sharpen) 한다는 가설.
> - **Obfuscated Activations** — Adversarial defense 분야 ICLR 2025 논문. Dense LLM 의 hidden-state probe (linear / MLP / SAE / latent OOD) 5 종 전부 adaptive PGD 에 recall 100% → 0% 로 무력화. 본 idea 의 직접 baseline. ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565))
> - **V-MoE** — Vision MoE. ImageNet 분류용 V-MoE 의 adversarial robustness 가 OpenReview Fd05J4Bu5Sp 에 보고. **Image PGD** 임을 본 연구의 text embedding-PGD 와 modality 구분 의무. ([OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp))
> - **GateBreaker** — 2025-12 arXiv. MoE safety expert weight-level 의 3% neuron 만 disable 해서 ASR (Attack Success Rate) 7.4 → 64.9% 증가. 본 연구의 가장 가까운 적대적 reference. ([arXiv:2512.21008](https://arxiv.org/abs/2512.21008))
> - **PAIR / GCG / AutoDAN** — 대표 jailbreak attack 알고리즘. PAIR ([arXiv:2310.08419](https://arxiv.org/abs/2310.08419)) 는 LLM-on-LLM iterative refinement, GCG ([arXiv:2307.15043](https://arxiv.org/abs/2307.15043)) 는 gradient-based suffix optimization, AutoDAN ([arXiv:2310.04451](https://arxiv.org/abs/2310.04451)) 는 hierarchical genetic algorithm.
> - **WildJailbreak** — Allen Institute 의 jailbreak 평가 benchmark. 4 class (vanilla/adversarial × benign/harmful), balanced 8K eval. ([arXiv:2406.18510](https://arxiv.org/abs/2406.18510))
> - **vLLM** — UC Berkeley high-throughput LLM serving stack. 본 연구는 vLLM 0.19+ fork 위 forward hook 으로 fingerprint 추출. ([공식 docs](https://docs.vllm.ai/))
> - **Venn diagram (4-attack space)** — 본 idea 의 Section 3 표준 figure. V-MoE PGD (image) / GateBreaker (weight ablation) / Expert Selections (privacy leakage) / DISCRETE-VEIL (text embedding-PGD on routing) 4 attack 의 공격 공간 경계를 시각화 의무.

**🎯 Target Venue**: IEEE S&P 2027 (13p) (primary) / DSN 2027 practical (6p, fallback)
**📊 Score**: Novelty 7.0 / Diff 8.0 / Impact 9.0 / Feasibility 8.0 = 평균 **8.00**
**✅ 판정**: Accept strong (Phase 2' scope 축소 후)

---

## 1. 개요 (Overview)

기존 dense LLM hidden-state probe 는 Obfuscated Activations ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)) 의 adaptive PGD 공격에 의해 5 종 latent-space defense (linear/MLP probe + SAE + representation-probing + latent OOD) 모두 **recall 100% → 0%** 로 붕괴함이 보고되었다. 본 연구는 MoE router 의 **top-k argmax (discrete combinatorial signal)** 가 continuous hidden state 와 질적으로 다른 attack surface 를 갖는지 정량 검증한다. 가설: Gumbel-softmax surrogate 를 통한 PGD 가 jailbreak 의도를 유지하면서 routing pattern 을 benign pattern 에 맞추려면 **non-differentiable combinatorial objective** 를 풀어야 하므로 attack budget 이 동일할 때 dense 대비 robustness 가 30%p+ 더 보존된다.

**Metaphor 부속 (R30)**: "Veil" = 얇은 가림막. Discrete routing 이라는 얇은 layer 가 combinatorial 구조로 latent-space attack 을 차단하는지 검증하는 metaphor.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 공격/방어 축 | Signal | 한계 (본 연구 대비) |
|-----------|------------|--------|---------------------|
| **Obfuscated Activations** ([arXiv:2412.09565](https://arxiv.org/abs/2412.09565)) | embedding-PGD attack | dense hidden state | MoE routing 미다룸; discrete signal 공격 안 함 |
| **V-MoE Adversarial Robustness** ([OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp)) | image PGD | V-MoE router | **이미지 공격 공간**; text LLM embedding-PGD 와 modality 구분 |
| **GateBreaker** ([arXiv:2512.21008](https://arxiv.org/abs/2512.21008)) | weight-level neuron ablation | MoE safety expert | 3% neuron disable → ASR 7.4 → 64.9%; **embedding 공격 아님** (white-box weight access) |
| **Expert Selections Reveal** ([arXiv:2602.04105](https://arxiv.org/abs/2602.04105)) | text reconstruction attack | clean routing indices | **privacy 공격** 방향 (token 91.2% 복원); detection 공격 아님 |
| **L³ "Lobotomy"** ([arXiv:2602.08741](https://arxiv.org/abs/2602.08741)) | refusal-expert silencing | routing pattern (attack) | attack 방향, detection 공격 아님 |
| **FJD** ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558), EMNLP 2025 Findings) | — (no adversarial eval) | first-token logit | dense LLM 가정, adaptive attack 평가 부재 |
| **OmniGuard** ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) | limited (non-adaptive) | dense internal repr | adaptive attack 평가 미비 |

**GAP**: **MoE routing classifier 에 대한 embedding-space PGD attack 과 그 robustness 정량화** 는 현재 공개 논문 없음.

---

## 3. 제안 기법 (Core Mechanisms, 2 mechanisms — improve-only ΔM=0)

### M1: Discrete Routing Obfuscation Attack (DRO-Attack)

**① 추가되는 Scheme — Source Verified (R32)**:

새 attack module `attacks/dro_attack.py` (~200 LOC, PyTorch 2.5+) 를 vLLM fork 에 추가하고, MoE forward 시 token embedding 에 PGD perturbation 을 주입하여 (a) jailbreak generation 성공 (b) routing pattern benign-look 두 목표를 동시에 추구하는 joint loss 를 minimize. 구체 hook 위치: vLLM `vllm/v1/worker/gpu_model_runner.py` 의 model forward 호출 직전 input embedding 에 perturbation `δ` add. Mechanism 은 기존 model code 미수정 — input embedding 에만 작용.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/worker/gpu_model_runner.py` (확인일: 2026-04-25, gh api success)
> ✅ source verified: vllm-project/vllm@`main` `vllm/model_executor/models/qwen3_moe.py`
> ✅ source verified: vllm-project/vllm@`main` `vllm/model_executor/models/mixtral.py`
> ⚠️ source proposed (not in upstream): `attacks/dro_attack.py` — 본 연구 신규 module. 기존 vLLM `vllm/v1/worker/` 와 별도 directory `attacks/` 로 추가하여 production code 비침투.

**② 해결하는 문제 + Workload evidence**:

사용자 실측 WildJailbreak 2-way **94.0%** (Qwen3 k-NN k=15 topk_weights+L1, balanced 8K eval) 은 clean setting 수치. Obfuscated Activations 는 dense hidden probe 를 recall 100→0 으로 무력화했다. **MoE discrete routing 이 이 공격 공간에서 얼마만큼 방어력을 가지는지** 는 미측정. S&P reviewer 는 training-free detection 의 adaptive robustness 를 반드시 요구.

**③ Step-by-step (3-5 steps)**:

1. PAIR (harmbench) + GCG (nanoGCG) 로 clean jailbreak prompts **각 200 = 400 prompts** 생성 (Qwen3 + Mixtral 2 모델).
2. 각 prompt token embedding `e_t` 에 perturbation `δ_t` (L∞ ≤ 8/255, L2 ≤ 0.3) 할당, PGD 20 iteration.
3. Joint loss: `L = CE(jailbreak_generation_target) + λ · MSE(topk_index_continuous_surrogate(e+δ), benign_target_pattern)`. `topk_index_continuous_surrogate` 은 Gumbel-softmax relaxation (temperature τ=0.5, hard=True) 으로 discrete argmax 의 미분 가능 surrogate.
4. 공격 후 perturbed input 으로 fingerprint 재추출 (기존 vLLM fork pipeline 그대로).
5. k-NN classifier 로 detection recall 재측정.

**④ 기존 접근 실패 + 본 mechanism 차별화**:

(a) hidden-state linear probe 는 continuous h 에 gradient 직접 적용 → recall 0%. (b) MoE routing 은 argmax 라 differentiable surrogate 없으면 gradient 안 먹음 → Gumbel-softmax surrogate 필수. (c) 본 연구는 처음으로 "surrogate 를 통한 PGD 가 실제 discrete argmax 의 shift 를 얼마나 유도하는가" 를 WildJailbreak 에서 측정. V-MoE PGD (image) 와 modality 구분, GateBreaker (weight ablation) 와 attack surface 구분 — 본 idea 파일 §5 의 Venn diagram 참조.

### M2: Routing-Entropy Sharpening KS-Test Tripwire

**① 추가되는 Scheme — Source Verified (R32)**:

새 detection module `detection/entropy_tripwire.py` (~80 LOC) 를 vLLM fork 또는 별도 inference wrapper 에 추가. 기존 fingerprint 의 `softmax_scores` `(L, E)` 행렬에서 per-layer Shannon entropy `H_l = -Σ_e p_l(e) log p_l(e)` 계산 → `(L,)` vector 를 추가 feature 로 concat 하여 k-NN classifier 에 공급. Forward hook 은 LOOM 의 EPRT (M1 of LOOM) 와 공유 가능.

> ⚠️ source proposed: `detection/entropy_tripwire.py` — 본 연구 신규 module.
> ✅ closest existing: vllm-project/vllm@`main` `vllm/model_executor/layers/fused_moe/layer.py` (FusedMoE forward 에서 softmax_scores 가 이미 계산됨)

**② 해결하는 문제 + evidence**:

Mechanism M1 이 PGD 로 routing 을 조작하면 **softmax distribution 이 sharpen** (entropy 감소). 자연 jailbreak 의 entropy 분포와 adaptive-attacked jailbreak 의 entropy 분포 간 KS-test 로 구분. ASE 2025 NIER 의 "routing entropy OOD" 와 구분점: 단순 entropy threshold 가 아니라 **distribution 간 KS-statistic** 으로 PGD-specific anomaly forensics.

**③ Step-by-step**:

1. Clean WildJailbreak 8K 와 DRO-Attack post WildJailbreak 800 두 set 의 per-prompt layer-entropy vector 수집 (M1 결과 재활용).
2. 두 set 의 entropy 분포 간 layer-wise KS statistic + p-value 계산.
3. KS p < 0.01 인 layer 집합 `L_KS` 를 tripwire 로 선정 (예상: 큰 모델 중~후반 layer).
4. Feature = concat(topk_weights, entropy_vec restricted to L_KS) → k-NN 재학습 (기존 train/test split 동일).
5. Clean vs DRO-attacked detection recall + AUC + entropy distribution overlap 측정.

**④ 기존 실패 + 차별화**:

HSF / HiddenDetect 는 hidden state 단일 통계 (Mean, std). RASA 는 supervised finetune 필요. ASE 2025 NIER 의 entropy 는 OOD 축 (natural 다른 분포) 이지만 본 Tripwire 는 **adversarial-specific entropy sharpen 을 KS distribution test 로 forensics** → adaptive-attack 검증 축.

### Mechanism 간 상호작용

M1 (공격) + M2 (방어) 의 짝. M1 의 perturbed fingerprint 가 M2 의 KS-test 통계를 형성. M1 단독으로 recall 30-50% drop 후 M2 가 10%p+ 복구하면 defense-in-depth 성립. Ablation 시 M1 only / M1+M2 / M2 only 3 cell.

**Tier 구성**: physical 1-tier (single workstation 2 GPU) + software 1-tier (single classifier, single fingerprint type). Tiering rule R1/R1b 모두 ≤3-4 안전.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### (1) Hardware

- **Primary**: RTX Pro 6000 96GB × 1 (Qwen3-30B-A3B BF16 ≈ 60GB, 여유 36GB 로 PGD adversarial 계산)
- **Secondary**: RTX 5090 32GB × 1 (Mixtral-8x7B-Instruct AWQ-4bit ≈ 24GB)
- **Storage**: NVMe 4TB (8K + 800 prompts × 48-layer × 128-expert × bf16 ≈ 4GB raw trace per model)
- **Network/multi-node**: single-workstation scope

### (2) Model

- **Qwen3-30B-A3B** (HuggingFace `Qwen/Qwen3-30B-A3B`) BF16, fingerprint dim 48×128=6,144
- **Mixtral-8x7B-Instruct-v0.1** (`mistralai/Mixtral-8x7B-Instruct-v0.1`) AWQ 4-bit, fingerprint dim 32×8=256
- Precision: router stays BF16 even under AWQ
- Inference: vLLM fork (사용자 기존 0.19.0 + custom capture)

### (3) Dataset · Workload

- WildJailbreak 8K balanced eval (사용자 v2 sampling 재활용)
- JailbreakBench 100 behaviors
- HarmBench 200 val (OOD probe)
- **Adaptive attack corpus**: PAIR 200 + GCG 200 = 400 prompts × 2 모델 = 800 prompts
- Metrics: detection recall (clean), recall-under-attack, AUC, balanced_acc, per-class confusion

### (4) Simulator · Tools

- vLLM fork (existing + DRO hook 100 LOC)
- PAIR: `harmbench/attacks/pair.py`, GCG: `nanoGCG` 0.2
- Obfuscated Activations baseline 재현: original `obfuscated-activations` repo
- PyTorch 2.5 + `torch.nn.functional.gumbel_softmax(hard=True)` + custom PGD loop (~150 LOC)
- Profiling: Nsight Compute counters (`lts__t_sectors`, `sm__warps_active`)
- Library: FAISS-GPU 1.10 (k-NN), Nsight Systems

### (5) Ablation · Baseline

**Baselines (8 종)**:

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **FJD** | [arXiv:2509.14558](https://arxiv.org/abs/2509.14558), **EMNLP 2025 Findings** ✓ | dense first-token logit |
| (b) | **OmniGuard** | [arXiv:2505.23856](https://arxiv.org/abs/2505.23856) (preprint) | internal repr classifier 120x |
| (c) | **WildGuard-7B** | [arXiv:2406.18495](https://arxiv.org/abs/2406.18495), **NeurIPS 2024** ✓ | 별도 forward |
| (d) | **HSF** | [arXiv:2409.03788](https://arxiv.org/abs/2409.03788) | hidden state filter |
| (e) | **LlamaGuard-3** | Meta | 별도 forward |
| (f) | **hidden-state linear probe** | Obfuscated Activations 재현 | 본 연구의 직접 대조군 |
| (g) | **Task-Cond Routing** | [arXiv:2603.11114](https://arxiv.org/abs/2603.11114) (preprint) | domain axis only |
| (h) | **V-MoE PGD text adaptation** | [OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp) | image PGD → text 이식 |

Peer-reviewed ratio: 4/8 = **50%** (R2 ≥25% 충족).

**Ablation matrix**: (Clean vs DRO-Attack) × (M1 only vs M1+M2) × (Qwen3 vs Mixtral) × (PAIR vs GCG) = 2^4 = 16 cells.

**Parameter sweep**: PGD ε ∈ {4/255, 8/255, 16/255}, Gumbel τ ∈ {0.1, 0.5, 1.0}, KS threshold p ∈ {0.001, 0.01, 0.05}.

**Fallback mode**: DRO-Attack 이 Qwen3 에서 recall 을 20% 미만으로 떨어뜨리면 (=hidden probe 수준 붕괴), negative result 를 **"MoE routing 도 adaptive adversary 에 취약"** 로 positioning 하여 ACSAC/DSN tier 로 전환. Mech M2 가 어느 정도 복구하면 defense-in-depth 논문으로 재구성.

### (6) Implementation Steps (Step-Level, R31)

| Step | 의존성 | Component / File (R32 verified) | 사용 API/Library | 완료 판정 (Success Criterion) |
|------|--------|---------|---------|---------|
| Step 1 | — | vLLM fork 검증 + 기존 fingerprint 재현. **vllm-project/vllm@`main` `vllm/v1/worker/gpu_model_runner.py` ✅** | vLLM 0.19.0, FAISS-GPU 1.10 | Qwen3 WildJailbreak 2-way 94% ±1% 재현 |
| Step 2 | Step 1 | 4-attack-공간 Venn diagram 작성 (V-MoE PGD / GateBreaker / Expert Selections / 본 연구) | matplotlib + tikzplotlib | Section 3 figure 1 완성 |
| Step 3 | Step 1 | PAIR + GCG 공격 400 prompts 생성 (Qwen3 + Mixtral) | `harmbench/attacks/pair.py`, `nanoGCG` 0.2 | clean attack 후 corpus dump |
| Step 4 | Step 3 | DRO-Attack (M1) PGD + Gumbel-softmax 구현 | `torch.nn.functional.gumbel_softmax(hard=True)`, custom PGD ~150 LOC | unit test 8/8 (joint loss converges) |
| Step 5 | Step 4 | 2 모델 × 2 공격 × clean/DRO fingerprint 수집 (~1,600 runs) | 기존 vLLM pipeline | perturbed fingerprint dataset 저장 |
| Step 6 | Step 5 | M2 entropy KS-test + k-NN 재학습 | scipy `ks_2samp`, scikit-learn | KS p < 0.01 인 layer 집합 확정 |
| Step 7 | Step 4-5 | 8 baseline 구현/재현 | (a)-(h) 각 source repo | baseline table 완성 |
| Step 8 | Step 6, 7 | 표 1-4 + figure 1-5 작성 | matplotlib, pandas | manuscript draft 70% |
| Step 9 | Step 8 | adaptive-attack discussion section + mechanistic interpretability (SAFEx-style expert attribution) | manual writing | 13p draft 완성 |
| Step 10 | Step 9 | polish, artifact evaluation 준비, OpenReview 게시 | git + README | submission-ready |

**참고 시간 (단일-workstation 기준, hard deadline 아님)**: 약 8-10 weeks 분포. 환경/사전 지식에 따라 다름.

### (7) Preliminary Analysis Metrics (R27-β)

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 / 검증 기준 |
|---|---|---|---|---|
| Clean WildJailbreak 2-way recall | k-NN classifier output | Qwen3, balanced 8K eval, k=15 | **94.0% ± 1%** (재현) | 94.0% 재현 |
| DRO-Attack 후 Qwen3 recall | k-NN re-eval after PGD | 800 perturbed prompts | — (미측정) | **≥ 55% 유지 (Tier-1 성공 조건)**, 30-55% Tier-2, <30% negative result |
| Hidden-state linear probe drop | Obfuscated Activations 재현 | 동일 800 prompts | 100% → ? | 0-20% 붕괴 (예상, dense 한계) |
| Entropy KS-statistic | scipy `ks_2samp(clean, attacked)` | per-layer | clean baseline ≈ uniform | KS p < 0.01 인 layer 5+ 개 (Mech M2 성공) |
| M2 단독 recall 복구 | k-NN with entropy feature | post-DRO | M1 only 대비 | +10%p 이상 |
| PGD iteration → recall drop curve | matplotlib | iter ∈ {1, 5, 10, 20} | — | asymptote 수렴 확인 |

**Preliminary Study 4-단계**:
- **(i) Baseline reproduction**: 사용자 기존 Qwen3 WildJailbreak 2-way 94% ± 1% 재현. Mixtral 90.1% ± 1% 재현.
- **(ii) Bottleneck attribution**: DRO-Attack 후 recall drop 분해. 목표: Qwen3 ≥55% / Mixtral ≥45% 유지 시 discrete-robust 가설 성립.
- **(iii) Roofline**: PGD iter 별 recall drop curve. 20 iter 이후 converge 여부. ε=8 → 16/255 증가 시 추가 drop 의 asymptote.
- **(iv) Micro-benchmark**: Mech M2 entropy KS-test 단독 (Mech M1 제외) 복구 quantify.

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 조건 / 가설 검증 |
|---|---|---|---|
| Qwen3 WildJailbreak recall (clean) | 94.0% | 94.0% (재현) | Step 1 |
| Qwen3 recall (DRO-Attack 후) | — | **≥ 55% 유지 (가설 성립)** | Step 5 |
| Hidden-state linear probe (대조군) | 100% | **0-20% 붕괴 (예상)** | Obfuscated Activations 재현 |
| 격차 (>30%p) | — | discrete-robust 가설 성립 | 본 논문 핵심 contribution |
| M2 entropy tripwire 복구 | — | +10%p | defense-in-depth 검증 |
| Latency overhead | WildGuard +200ms | +<5ms (k-NN lookup) | 기존 OmniGuard 120x 와 유사 |

**과학적 contribution**: MoE discrete routing 과 dense hidden state 의 adaptive-attack robustness 를 처음으로 공식 구분. Venn diagram 표준 포맷으로 V-MoE PGD / GateBreaker / Expert Selections / DISCRETE-VEIL 4 attack 공간 경계 확정.

**실용적**: 만약 recall ≥55% 유지 시 → MoE serving 에 fingerprint-based guard 를 "adaptive-adversary certified" 로 deploy 가능. NeMo Guardrails / vLLM / NVIDIA AI Enterprise 인용 근거.

**Scope 제한**: MoE 가 전체 LLM 배포의 30-40% 라는 한계. 단 reviewer 는 scope 자체가 differentiation 역할 — "dense probe 는 무너지는데 MoE discrete 는 버티는가" 라는 질적 novelty question.

---

## 6. (Tier-1 → Tier-2 변환 가이드)

본 idea 를 **Tier-2 DSN practical 6p / IEEE CAL 4p** 로 분리 publication 시:

- **Single mechanism**: M1 only (DRO-Attack), M2 entropy KS-test 는 Tier-1 main paper 로 보존.
- **Scope 축소**: Qwen3 단일 모델 + PAIR 단일 공격 = 200 prompts.
- **Baseline 2-3 편**: FJD + OmniGuard + hidden-state probe.
- **참고 소요**: 약 4-6 weeks.
- **Tier-1 과의 관계**: precedence claim ("first-to-report MoE embedding-PGD on text routing classifier"). Tier-1 S&P submission 전 IEEE CAL 4p 로 publication priority 확보 가능.
- 상세는 **Tier-2 [01-discrete-veil-lite.md](/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/01-discrete-veil-lite.md)** 참조.

---

## 7. Reference 목록 (이 idea 핵심)

- **Obfuscated Activations Bypass Latent-Space Defenses**: [arXiv:2412.09565](https://arxiv.org/abs/2412.09565)
- **V-MoE Adversarial Robustness**: [OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp)
- **GateBreaker**: [arXiv:2512.21008](https://arxiv.org/abs/2512.21008)
- **Expert Selections Reveal Text**: [arXiv:2602.04105](https://arxiv.org/abs/2602.04105)
- **L³ "Lobotomy"**: [arXiv:2602.08741](https://arxiv.org/abs/2602.08741)
- **FJD** [EMNLP 2025 Findings]: [arXiv:2509.14558](https://arxiv.org/abs/2509.14558)
- **OmniGuard**: [arXiv:2505.23856](https://arxiv.org/abs/2505.23856)
- **HSF**: [arXiv:2409.03788](https://arxiv.org/abs/2409.03788)
- **WildGuard** [NeurIPS 2024]: [arXiv:2406.18495](https://arxiv.org/abs/2406.18495)
- **Task-Cond Routing**: [arXiv:2603.11114](https://arxiv.org/abs/2603.11114)
- **WildJailbreak benchmark**: [arXiv:2406.18510](https://arxiv.org/abs/2406.18510)
- **JailbreakBench**: [arXiv:2404.01318](https://arxiv.org/abs/2404.01318)
- **HarmBench**: [arXiv:2402.04249](https://arxiv.org/abs/2402.04249)
