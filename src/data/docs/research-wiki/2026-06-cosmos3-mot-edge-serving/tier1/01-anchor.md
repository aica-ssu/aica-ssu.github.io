# Q2 ANCHOR (Tier-1 #1, avg 7.6)

**ANCHOR: One-Shot Quantization of Static Conditioning KV with a Step-Dependent Flow-Matching Denoising Error Bound for Dual-Tower Omnimodal Models** · MLSys 주 / ICML 차선 2027

---

## 1. 개요

- **metaphor↔mechanism**: ANCHOR(닻) = denoising 전 기간 고정되어 있는 K_AR/V_AR. 배가 닻에 묶여 움직이지 않듯, conditioning KV 는 N step × CFG 2 pass 내내 read-only 로 고정 → 닻을 "한 번만 정밀하게 내려두면(1-shot quant)" 모든 step 이 그 위에서 흔들림 없이 재사용한다.
- **한 줄**: Cosmos3 의 정적·1회·read-only K_AR/V_AR 을 1-shot 으로 저비트 양자화하되, denoising 출력 오차를 flow-step 수 N 에 명시 의존하는 닫힌 error bound 로 보증해 bit 를 layer별 자동 선택.
- **G4 소유**: static K_AR/V_AR 의 **numeric(quant+bound)** 전권. A2 가 quant 를 위임(layout-only), L3 이 placement 를 위임 → Q2 가 G4 양자화 전부 소유 (no double-claiming).

## 2. 기존 연구 한계 · GAP (workload evidence 수치 포함)

- **decode-time KV quant 과 정반대 가정**: KIVI([arXiv:2402.02750](https://arxiv.org/abs/2402.02750))/KVQuant([arXiv:2401.18079](https://arxiv.org/abs/2401.18079))는 online·매 토큰 cheap + growing KV 가정 → 비싼 1-shot 양자화를 amortize 할 수 없고 step 개념이 없어 N-의존 분석 불가.
- **Cosmos3 K_AR/V_AR 은 정적**: Eq.8(tech report line 784) `O_DM=Attn_full(Q_DM,[K_AR;K_DM],[V_AR;V_DM])` 에서 K_AR/V_AR 은 N step × CFG 2 pass 전체 read-only (reasoner caching §5.3.2). T2V N=50, **policy N=4**.
- **workload evidence**:
  - modality inflation: 3,715 token conditioning → prefill **95.78J / 278.26ms** ([arXiv:2512.22695](https://arxiv.org/abs/2512.22695)); decode/denoise memory-bound ([arXiv:2512.01644](https://arxiv.org/abs/2512.01644)).
  - T2V-720p Nano 1-GPU H100 ~286-297s (Fig.16) → edge 에서 KV BW 가 step 마다 반복 read.
  - Cosmos3-Nano **BF16-only** (HF card) → quant path = reference Diffusers + llm-compressor/QuaRot fork (비공식 정밀도 명시).
- **공백**: static-KV 특화 1-shot quant amortize + flow-step N-의존 closed-form denoising error bound 가 부재 (QuantKeys/33-Method study 둘 다 bound 없음, WebFetch 확인).

## 3. 제안 기법 (mechanism ≤3)

### M1. Hadamard-rotated 1-shot K_AR/V_AR quantization

- **① 추가 scheme**: 모든 layer cached K_AR/V_AR 을 per-channel asymmetric INT4/INT3 + Hadamard rotation(QuaRot([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) 식) + MSE-optimal clipping.
- **② 해결 문제**: G4 — K_AR 이 정적이라 비싼 rotation matmul 을 1회만 치르고 amortize (online 엔 불가능).
- **③ 동작 원리**: (1) reasoner forward 종료 → K_AR/V_AR freeze. (2) K 에는 RoPE 적용 후 Hadamard rotation(MRoPE 순서규칙 준수), V 는 RoPE 무관 자유 rotation. (3) per-channel MSE-optimal clipping range 산출. (4) bit∈{8,6,4,3,2} 전 sweep 으로 양자화. (5) bound-tightness plot 산출 (필수 artifact).
- **④ 기존 실패 이유 + 차별화**: KIVI/KVQuant 는 online decode-KV 라 매 토큰 cheap quant 만 가능. ANCHOR 은 static prefix 라 1-shot amortize.

### M2. Flow-matching denoising error bound (핵심 수학 객체)

- **① 추가 scheme**: KV quant error 를 denoising 출력 오차로 전파하는 N-의존 닫힌 bound.
- **② 해결 문제**: bit 선택의 정량 근거 부재 — 어느 bit 가 품질 안전한지 측정 없이 알 수 없음.
- **③ 동작 원리**: (1) value-perturbation `‖Ô_DM−O_DM‖≤‖A‖_∞·‖ΔV_AR‖` (A row-stochastic, ‖A‖_∞≤1). (2) key-perturbation softmax-Lipschitz `‖ΔA‖≤L_softmax·‖ΔK_AR‖`, `L_softmax≤‖Q_DM‖/√d`. (3) velocity net Lipschitz `L_v^{(n)}` power iteration 측정. (4) 최종 `‖x̂_0−x_0‖≤(Σ_{n=1}^{N}Δ_n·L_v^{(n)})·(L_softmax·‖ΔK_AR‖+‖ΔV_AR‖)`. (5) N 작을수록(policy N=4) bound tight = policy 가 저비트에 관대 → 검증가능 예측.
- **④ 기존 실패 이유 + 차별화**: online KV quant 엔 step 개념 자체가 없어 N-의존 bound 불가. ANCHOR 의 N-bound 는 static prefix + flow-matching 조합에서만 정의.

### M3. Deploy & bound-validation

- **① 추가 scheme**: K_AR cache hook(구현된 reasoner caching) → L_v 측정 → bound vs 실측 비교 → bit 자동 선택.
- **② 해결 문제**: bound 가 실용적인지(tight 한지) 실증 없이 신뢰 불가.
- **③ 동작 원리**: (1) reasoner caching hook 으로 K_AR 확보. (2) 64 trajectory jacobian spectral norm 으로 L_v 측정. (3) bit별 bound vs 실측 출력 error tightness plot. (4) bound-tolerance τ 하 layer별 최소 bit 선택.
- **④ 기존 실패 이유 + 차별화**: 경쟁작은 실증 sweep 만 (bound 없음) → bit 선택이 휴리스틱. ANCHOR 은 bound 가 선택 기준.

## 4. 평가 · 실험 플랜 (R20-β 7요소)

- **HW**: 양자화+bound calib = RTX Pro 6000(ncu BW 측정 가능); policy footprint/latency = AGX Orin (BW 는 ncu 미지원 → latency proxy).
- **모델**: Cosmos3-Nano(T2V/I2V), Cosmos3-Nano-Policy-DROID(N=4 bound-favorable); fallback BAGEL-7B-MoT(14B/7B-active, vLLM upstream 미지원 → method 일반성).
- **워크로드**: VBench/GenEval(생성), RoboLab-120 subset(20-30 task ablation, full 최종) action-MSE+성공률, MMMU(VLM sanity).
- **도구**: llm-compressor/QuaRot fork on Diffusers reference path, power iteration jacobian.
- **ablation + baseline**: bit{8,6,4,3,2} × Hadamard on/off × N={4,50}. baseline — (a) FP16 KV 상한, (b) KIVI INT4([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)), (c) KVQuant per-channel([arXiv:2401.18079](https://arxiv.org/abs/2401.18079)), (d) naive RTN INT4, (e) QuantKeys([arXiv:2605.26266](https://arxiv.org/abs/2605.26266)).
- **주차별 구현**:

  | 주차 | 작업 |
  |---|---|
  | W1-3 | reasoner caching hook + 1-shot Hadamard quant 구현 (bit sweep) |
  | W4-7 | L_v power-iteration 측정 + denoising error bound 유도·구현 |
  | W8-10 | bit별 bound vs 실측 tightness plot (N=4 vs N=50 falsify gate) |
  | W11-13 | RoboLab/VBench full + bit 자동선택 + baseline 비교 |

- **preliminary metrics**: INT4 bound-tightness ratio @policy N=4, conditioning-KV 압축비.

## 5. 예상 효과 (보수치, scope 명시)

- conditioning-KV **2-4× 압축(INT4/INT3) 품질손실 <1%** (정적 K_AR 한정, decode-KV 아님).
- policy N=4 → **INT3, bound-favorable 시 INT2** (action-MSE <2%, bound 허용시만).
- step KV-read latency bit 비례 절감 (edge LPDDR5 BW-bound 구간).
- 핵심 산출물 = **bit별 bound-tightness plot** (경쟁 KV-quant 전부 부재).

## Tier-2 scope-축소 variant (R15-β.3)

- **policy-mode 4-step static-KV INT4 + measured action-MSE budget** — 단일 메커니즘(M1+M3 의 policy slice) DATE/IEEE-CAL letter.
- **mechanism 축소 benefit**: M2 의 full N-bound 유도를 생략하고 policy N=4 에 한정, bound 의 N-의존 예측을 RoboLab 에서 action-MSE budget τ 하 최소 bit 로 실증 (full Q2 의 가장 검증 쉬운 슬라이스).
- **venue**: DATE / IEEE-CAL short.
- **정량 표**:

  | 지표 | Baseline | variant | 개선 |
  |---|---|---|---|
  | policy K_AR bit | INT8 | INT4/INT3 | 2× footprint |
  | action-MSE | FP16 상한 | budget τ 하 | <2% (τ 명시) |

- **승격 가능성**: policy N=4 bound-tightness ≤3× 확인 + T2V N=50 으로 확장 시 full Q2(Tier-1)로 승격.
