# Q1 DRIFT (Tier-1 #2, avg 7.4)

**DRIFT: Tower-Asymmetric Post-Training Quantization via Co-Init Sensitivity Divergence for Dual-Tower Omnimodal Models** · NeurIPS / ICML / MLSys 2027

---

## 1. 개요

- **metaphor↔mechanism**: DRIFT(표류) = 같은 Qwen3-VL-8B weight 로 출발한 두 tower 가 서로 다른 objective(AR next-token CE vs rectified-flow velocity MSE)로 학습되며 quant 민감도가 비대칭으로 표류(발산)하는 현상. 같은 배에서 출발한 두 척이 다른 해류로 흘러가듯, init 은 같지만 sensitivity 가 갈린다.
- **한 줄**: co-init 두 tower 의 quant-sensitivity 발산 ρ_ℓ 을 교란변수(init·구조 고정) 통제 하 측정하고, 그 발산을 근거로 단일 UMA 예산 하 per-tower bit allocation.
- **conditional-gain 정직성**: 이득은 **ρ_ℓ 발산이 material 할 때만** 존재 — 발산 작으면 "균일이 충분"이 정답(negative-but-publishable).

## 2. 기존 연구 한계 · GAP (workload evidence 수치 포함)

- **두 정설이 한 모델에 공존**: LLM(AR) activation outlier 민감(AWQ [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), SmoothQuant [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)) vs DiT(generator) W4A4 내성(SVDQuant [arXiv:2411.05007](https://arxiv.org/abs/2411.05007)) — **같은 모델 두 tower 에 공존하는 첫 사례**.
- **measurement evidence**:
  - 두 tower 동일 Qwen3-VL-8B co-init (tech report line 933)이나 서로 다른 objective 로 sensitivity 발산.
  - 16B FP16=32GB; Orin NX 는 INT4(8GB) 필수, AGX Orin INT8/INT4 현실적.
  - decode 가 prefill 대비 **192-569× latency 지배** ([arXiv:2511.01866](https://arxiv.org/abs/2511.01866)) → reasoner 정밀도가 핵심.
- **공백**: 발산을 *교란변수 통제 하* 측정한 연구 부재 + 단일 UMA 예산 per-tower bit allocation 부재. 기존 mixed-precision(Mix-QViT [arXiv:2501.06357](https://arxiv.org/abs/2501.06357), KL-Lens [arXiv:2604.13440](https://arxiv.org/abs/2604.13440))은 전부 tower-agnostic / 단일 objective.

## 3. 제안 기법 (mechanism ≤3)

### M1. ρ_ℓ divergence 측정 study (Q4 SIEVE 흡수 — motivation + 제1 contribution)

- **① 추가 scheme**: layer ℓ, tower τ 의 `S_ℓ^τ=Tr(H_ℓ^τ)·‖ΔW_ℓ^τ‖_F²` (HAWQ-style, Hutchinson HVP, reasoner=AR CE / generator=flow-MSE task loss).
- **② 해결 문제**: tower 비대칭 sensitivity 가 init/구조 교란 없이 측정된 적 없음.
- **③ 동작 원리**: (1) 두 tower 별 Hessian trace × weight-drift 산출. (2) **divergence ratio ρ_ℓ=S_ℓ^R/S_ℓ^G** (null: co-init 이라 ρ_ℓ≈1). (3) outlier kurtosis κ_ℓ^τ 측정. (4) weight-drift δ_ℓ=‖W_ℓ^R−W_ℓ^G‖_F/‖W_init‖. (5) **(δ_ℓ,ρ_ℓ) 상관 = "표류한 만큼 민감도 발산하는가" 인과 가설** 검증.
- **④ 기존 실패 이유 + 차별화**: 기존은 tower-agnostic single-objective → co-init 통제 불가. DRIFT 는 동일 init 두 objective 라 교란변수 통제 첫 사례.

### M2. ILP/greedy bit allocation under UMA budget

- **① 추가 scheme**: `min Σ S_ℓ^τ(b) s.t. Σ bits ≤ B`, B∈{16GB-equiv, 24GB, 32GB}.
- **② 해결 문제**: 단일 UMA 예산 하 어느 tower/layer 에 bit 를 배분할지 미해결.
- **③ 동작 원리**: (1) S_ℓ^τ(b) sensitivity 곡선 산출. (2) reasoner=AWQ/GPTQ 경로, generator=SVDQuant(low-rank rank 16-64) 경로. (3) cross-attn proj 별도 sensitivity bucket. (4) ILP/greedy 로 예산 하 sensitivity 최소 bit 할당.
- **④ 기존 실패 이유 + 차별화**: 균일 정밀도는 sensitivity 무시 → 비효율. DRIFT 는 tower별 분리 경로.

### M3. Deploy validation

- **① 추가 scheme**: VLM MMMU / gen GenEval / policy RoboLab-120 subset 성공률.
- **② 해결 문제**: bit-alloc 이득이 실배포에서 유효한지 미검증.
- **③ 동작 원리**: (1) 할당 bit 로 양자화. (2) AGX Orin 배포 (footprint/tok-s/J). (3) 3 task 품질 측정. (4) FP16 대비 Δ + 균일-W4 대비 conditional-gain.
- **④ 기존 실패 이유 + 차별화**: 기존은 단일 task — DRIFT 는 VLM/gen/policy 3 mode 횡단.

## 4. 평가 · 실험 플랜 (R20-β 7요소)

- **HW**: calib+eval RTX Pro 6000; 배포 metric AGX Orin (footprint/tok-s/J, ncu 불요).
- **모델**: Cosmos3-Nano + Policy-DROID; fallback BAGEL(일반성).
- **워크로드**: GenEval, MMMU, RoboLab-120 subset.
- **도구**: HAWQ Hutchinson HVP, SVDQuant 이식 fork (BF16-only flag).
- **ablation + baseline**: 2 method × {8,4} × per-tower (전수 금지, 16wk gate). baseline — (a) SVDQuant generator-only W4A4([arXiv:2411.05007](https://arxiv.org/abs/2411.05007)), (b) uniform AWQ whole-model([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)), (c) uniform GPTQ whole-model, (d) BitVLA ternary-all([arXiv:2506.07530](https://arxiv.org/abs/2506.07530)) — 전부 tower-agnostic.
- **주차별 구현**:

  | 주차 | 작업 |
  |---|---|
  | W1-4 | ρ_ℓ/κ_ℓ/δ_ℓ 측정 (Hessian trace, weight-drift) + (δ_ℓ,ρ_ℓ) 상관 |
  | W5-8 | ILP/greedy bit-alloc + tower별 quant 경로 (AWQ/SVDQuant) |
  | W9-12 | AGX Orin 배포 + 3-task 품질 + baseline 비교 |
  | W13-14 | cross-model(BAGEL) 일반성 + conditional-gain 판정 |

- **preliminary metrics**: ρ_ℓ 분포 plot(novel artifact), (δ_ℓ,ρ_ℓ) 상관계수.

## 5. 예상 효과 (보수치, scope 명시)

- 동일 예산서 균일-W4 대비 **품질 0.5-1.5%p 회복 또는 footprint 10-18% 절감** (**ρ_ℓ 발산 클 때만**).
- 발산 작으면 negative 기여 ("균일이 충분" — 측정 자체가 Q4 SIEVE spinoff 로 발행가치).
- scope: PTQ 한정 (QAT/학습 제외), Cosmos3-Nano + BAGEL.

## Tier-2 scope-축소 variant (R15-β.3 = Q4 SIEVE)

- **measurement-only co-init divergence letter** — IEEE CAL / DAC-short.
- **mechanism 축소 benefit**: M2(bit-alloc 해법) 제외, M1 의 순수 측정만 — ρ_ℓ/κ_ℓ/δ_ℓ + (δ_ℓ,ρ_ℓ) 상관 + cross-model(BAGEL) 일반성으로 "법칙" 격상. risk 최저(~6주).
- **venue**: IEEE CAL / DAC short.
- **정량 표**:

  | 지표 | Baseline | variant | 개선 |
  |---|---|---|---|
  | ρ_ℓ 측정 | (없음) | tower별 sensitivity 분포 | novel artifact |
  | (δ_ℓ,ρ_ℓ) 상관 | (없음) | 인과 가설 검증 | 법칙 후보 |

- **승격 가능성**: ρ_ℓ divergence ≥1.5× material + bit-alloc 이득 측정가능(회복 ≥0.5%p) 확인 시 full Q1(Tier-1)로 승격; Q1 full 제출 시 motivation 섹션으로, 분리 필요 시 CAL 단독.
