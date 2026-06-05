# SPENDTHRIFT — Spectral Verbosity Budget for State-Quantized Hybrid Decoding

> **Hook**: 양자화된 SSM state의 오차가 ∏A로 동방향 누적돼 출력 길이(verbosity)를 낭비한다 — state error를 throughput 예산의 "지출"로 회계해 high-ρ 층만 차등 승급하고, bias-canceling stochastic rounding으로 verbosity를 −25~35% 억제한다.

- **Score (Phase 2', 5축 = novelty/diff/impact/ai-impl/arch-impl)**: 7 / 7 / 8 / 7 / 8 = **avg 7.4** (Δ0 vs Phase 2 — M3 제거 simplify로 novelty 손실 없음)
- **Venue target**: Track A — **ICML 2027** (이론 트랙, 대안 NeurIPS 2026).
- **Tier 위치**: Tier-1 Top-3 (3위) — SSM-recurrent bias-drift closed-form 이론 + Q-Overthink 반대 결론 정면 + BLOAT 실증 pair. decode-dominated라 impact 8 견고.
- **Paper pair**: **SPENDTHRIFT(이론) × BLOAT(실증) 수직 pair 채택** — 동일 vLLM Mamba2 state-cache hook 공유(구현 1회·논문 2편). BLOAT가 먼저 det vs SR 인과 확립 → SPENDTHRIFT 이론이 그 위에.
- **Cross-share ownership**: **state-quant→verbosity 인과 = SPENDTHRIFT 단일 소유**(BLOAT=실증 plane, SILT=D_verb, GOVERNOR=verbosity-guard 위임 모두 SPENDTHRIFT 정리 의존). SR Triton kernel = SPENDTHRIFT M2·BLOAT M1 공유.

---

## 1. Research Questions

- **RQ-master**: SSM-recurrent hybrid decoding에서 state 양자화 오차가 `h_q,t − h_t = Σ(∏A)e_i`로 동방향 누적(bias drift)돼 verbosity를 유발한다는 가설을, layer별 spectral radius ρ_ℓ에 따라 high-ρ 층만 차등 승급하는 **spectral-budgeted allocator** + **bias-canceling stochastic rounding(SR)**으로 검증해, 출력 token을 **−25~35%** 억제하면서 accuracy를 **±0.5%p** 유지할 수 있는가?
- **RQ-1**: deterministic rounding 대비 SR(`E[e_i]=0`)이 SSM-recurrent verbosity를 제거하면 "bias-drift가 verbosity 인과"임이 식별되는가 (Q-Overthink의 transformer 반대 결론과 SSM-recurrent는 다른가)?
- **RQ-2**: ρ_ℓ 기반 차등 bit 할당이 균일 state cast(naive +40% verbosity, Super 보고) 대비 실효 tok/s를 **+18~30%** 끌어올리는가 (decode-dominated 91~98.6%라 ΔL이 거의 그대로 throughput, high-ρ 층 다수 조건)?

## 2. 기준 코드베이스 (Baseline Source)

- **vLLM (V1 core)**: `https://github.com/vllm-project/vllm`
  - 검증 commit **`7c37096`** (`7c37096620fa4e161c1d8c1db5c43c8514545d84`, HEAD `[Core][Refactor] thread scheduler_block_size into KVCacheManager`, 2026-06-02 fetch). ai-impl 리뷰가 `~/code_baseline/vllm-project--vllm`에 local clone navigation으로 검증.
- **Model**: HuggingFace **Nemotron-3-Nano-30B-A3B** (BF16 / FP8 / NVFP4). selective-FP8 island는 ModelOpt checkpoint의 per-module quant config로 처리(`quantization/modelopt.py`, 서빙-path 커스텀 코드 불요). `registry.py:180 "NemotronHForCausalLM"` 직접 지원.
- **컨테이너**: vLLM NGC container `nvcr.io/nvidia/vllm:26.01-py3` (Jetson pip 미지원 → NGC + commit pin).
- **Target HW**: 1차 **AGX Orin 64GB**(edge decode-dominated, EdgeReasoning 재현 — RTX 5090은 BW 과대로 edge 미재현). BLOAT pair는 Orin Nano 8GB(Nemotron-3-Nano-4B proxy) 보조.

## 3. 배경 / GAP

- Nemotron-3-Nano = 31.6B total / 3.2B active, **Mamba-2 23층 / MoE 23층 / Attention 6층**. expert 10MB FP8, population 30GB, 모델 32GB FP8(W4 22GB). AGX Orin 64GB FP8 full-resident.
- **per-token expert read 1.84GB/token** → 출력 token 1개 = **1.84GB LPDDR 트랜잭션** → verbosity 억제 = 에너지/열 직결. ~9 tok/s 상한(EdgeReasoning [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) 4.7~9.3 TPS 정합).
- **SSM state 양자화 → verbosity +40%** (Super 보고, Eq3 `h_q,t = h_t + Σ(∏A)e_i`). **단 출처가 Super(이질 모델)** — Nano-급 hybrid 재측정이 BLOAT 기여.
- decode-dominated **91.0~98.6%** ([arXiv:2605.26297](https://arxiv.org/abs/2605.26297)) → ΔL(출력 길이 증가)이 거의 그대로 실효 throughput 손실. turn 12~62 누적 ∏A 전파(자기상관).
- **state 절감은 작다**: state quant로 state pool 절감(48→24MB/agent)은 작아 verbosity 손실이 거의 항상 이김 → cross-over는 agent 매우 많을 때만(정직한 결론, BLOAT가 측정).
- **이론 backbone**: 정리1(bias drift) deterministic rounding `E[h_q,t−h_t]=Σ(∏A)E[e_i]≠0`, ‖drift‖~Σρ^{t−i}·b vs SR `E[e_i]=0`→`E[h_q,t−h_t]=0`, Var=`σ²/(1−ρ²)` bounded(ρ<1 selective decay 보장). 정리2(verbosity bound, 조건부) `E[ΔL] ≤ κ·‖drift‖·(fork-token 민감 position 밀도)` — **ρ가 큰 layer에서만 drift→verbosity 성립**(scope 제한, Q-Overthink 반대 결론 방어).
- **회계식 backbone**: 허용 throughput 손실 x% ⇒ 허용 ‖drift‖ ⇒ per-layer ρ_ℓ에 따라 차등 bit. drift gain `g_ℓ = 1/(1−ρ_ℓ)`이므로 ρ_ℓ→1 층이 verbosity 지출의 1차 동인 → 예산 β를 high-ρ 층에 우선 배분(KKT closed-form). 저-ρ 층은 INT8로 내려도 drift 누적이 빨리 감쇠 → verbosity 영향 미미.
- **bias vs variance 분해**: deterministic INT8은 bias(`E[e_i]≠0`) 누적 → drift; SR은 bias 제거(`E[e_i]=0`)하나 variance `σ²/(1−ρ²)` 도입(ρ<1 selective decay로 bounded). 따라서 SR이 verbosity를 줄이면 "verbosity 동인은 bias이지 noise가 아님"이 식별됨 — 이것이 Q-Overthink 반대 결론(transformer는 spurious)과 SSM-recurrent를 가르는 검증 축.
- **기존 연구 한계**: Q-Overthink [arXiv:2606.00206](https://arxiv.org/abs/2606.00206)는 "Quantized Reasoning Models Think They Need to Think Longer, but They Do **Not**" — quant-verbosity가 spurious(**transformer**). SPENDTHRIFT는 "SSM-recurrent는 다르다(bias가 ∏A로 동방향 누적)"를 검증 가능 가설로 정면 배치. Quamba [arXiv:2410.13229](https://arxiv.org/abs/2410.13229)/Q-S5 [arXiv:2406.09477](https://arxiv.org/abs/2406.09477)는 accuracy만. Q-Hurt [arXiv:2504.04823](https://arxiv.org/abs/2504.04823)는 attention/weight·empirical. CQD [arXiv:2601.00938](https://arxiv.org/abs/2601.00938)는 latent state R-D(verbosity 무관). SSM-state-quant→verbosity의 closed-form bias-drift 모델은 2nd sweep에서도 무경쟁.

## 4. Mechanisms

### M1 — Spectral-budgeted mixed-precision allocator

#### 동작 원리
- **(scheme)** layer별 A spectral radius ρ_ℓ를 calibration 1K로 추정 → drift gain `g_ℓ=1/(1−ρ_ℓ)`. budget β 제약 `Σ g_ℓ·σ²(bit_ℓ) ≤ β` 하 bit 최소화(KKT closed-form). 구현: vLLM model loader per-module quant config(selective-FP8 dict 확장) + `MambaSpec.dtypes`. high-ρ 층만 승급(AutoQuantize knapsack 불요).
- **(문제)** 균일 state cast가 +40% verbosity(Super). 분석적으로 어느 층이 verbosity 동인인지 알면 high-ρ 층만 bit 승급해 메모리·verbosity trade를 최적화.
- **(step-by-step)** (1) ρ_ℓ calibration(1K sample), (2) closed-form bit 할당(KKT), (3) per-layer dtype 주입(`MambaSpec.dtypes`).
- **(차별화)** Quamba/Q-S5는 accuracy만. Q-Overthink/Q-Hurt는 attention/weight·empirical. SPENDTHRIFT는 **per-layer ρ → drift gain closed-form bit 할당**이 신규(verbosity를 예산으로 회계).

#### 기대 효과 (Gain)
- **primary axis [Performance]**: 출력 token(verbosity) −25~35% (vs naive state, reasoning-on·high-ρ 층 다수) → 실효 tok/s +18~30% (decode-dominated).
- **secondary axis [Energy]**: LPDDR 트랜잭션/decode −25~35% (verbosity = 트랜잭션 직결), accuracy ±0.5%p.

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/model_executor/layers/quantization/modelopt.py` | per-module quant config (selective-FP8) | per-module dtype honor(이미 동작) | per-layer Mamba state dtype을 ρ_ℓ closed-form으로 주입 | extend |
| `vllm/v1/core/kv_cache_interface.py` | `MambaSpec.dtypes` (L566) | state cache dtype 보유 | per-layer 차등 dtype(FP8/INT8/INT16) 주입 | extend |
| (offline calibration, 신규) | `rho_calibrator` + KKT bit allocator | 부재 | ρ_ℓ 추정 + budget β 제약 bit 최소화 | 신규 (offline) |
- **검증 출처**: per-module quant config는 ai-impl SPENDTHRIFT §M1 [✅ 경로](selective FP8가 이미 이렇게 동작) @ `7c37096`. `MambaSpec.dtypes`(kv_cache_interface.py:566) ai-impl §0 [✅]. allocator 자체는 알고리즘, 주입 지점 실재 — ρ_ℓ calibration은 offline이라 서빙-path 변경 최소.

#### 검증 시나리오 (Test Plan)
- **Unit test (분)** — KKT allocator 정합:
  - 목적: KKT bit allocator가 budget β 제약을 만족하며 high-ρ 층에 더 많은 bit 할당하는지.
  - Input: 합성 ρ_ℓ 분포 + budget β.
  - Expected: `Σg_ℓσ²(bit_ℓ)≤β`, ρ↑층일수록 bit↑ (단조).
  - metric: 제약 위반 횟수=0, 할당 단조성.
  - 실행시간: ~5분. 실패 시 액션: KKT 유도·g_ℓ 정의 재검토.
- **Mechanism-isolated test (시간)** — verbosity 억제:
  - 목적: 차등 bit이 균일 cast 대비 verbosity를 억제하는지.
  - Input: Nemotron-3-Nano, AIME25/LiveCodeBench, {균일 INT8 / spectral-budgeted} @ AGX Orin 64GB.
  - Expected: 출력 token −25~35%, accuracy ±0.5%p.
  - metric: 출력 token 수, accuracy, 실효 tok/s.
  - 실행시간: ~10시간(calibration+eval). 실패 시 액션: ρ 추정 calibration shift → 조건부(high-ρ만) scope 제한.

### M2 — Bias-canceling stochastic rounding kernel (Philox)

#### 동작 원리
- **(scheme)** SSM state write-back 시 SR(정리1: `E[e_i]=0` → drift 제거). 구현: `mamba_mixer2.py` state update 직후 Triton(`layers/mamba/ops`) SR 캐스트(Philox RNG). **BLOAT M1과 공유.**
- **(문제)** deterministic bias 누적이 verbosity 동인(정리1). SR로 unbiased 캐스트하면 drift 제거되나 RNG 오버헤드가 이득을 잠식할 위험.
- **(step-by-step)** (1) state update, (2) SR 캐스트(Philox), (3) **per-step Philox 오버헤드 선측정 gate**(verbosity 이득 < RNG 오버헤드면 net 손해 → gate로 차단).
- **(차별화)** 기존 SR 연구는 verbosity 환산 부재 — SPENDTHRIFT는 SR로 bias-drift를 verbosity 인과 식별에 연결(BLOAT가 실증 plane).

#### 기대 효과 (Gain)
- **primary axis [Performance]**: SR로 bias-drift 제거 → verbosity 억제(M1과 결합).
- **secondary axis [측정/인과]**: deterministic→SR에서 verbosity 소멸 시 bias-drift 인과 확립(Q-Overthink 반대 결론 방어).

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/model_executor/layers/mamba/mamba_mixer2.py` | state update write-back | FP/deterministic cast | state update 직후 SR 캐스트 hook | extend |
| `vllm/model_executor/layers/mamba/ops/` | (신규 Triton kernel) | 부재 | Philox bias-canceling SR cast kernel | 신규 kernel |
- **검증 출처**: `mamba_mixer2.py` state update + Triton `layers/mamba/ops`는 ai-impl SPENDTHRIFT §M2 [⚠️ 신규 kernel] @ `7c37096`. BLOAT M1과 hook 공유(ai-impl BLOAT §M1: `MambaSpec.dtypes` + mixer state cast). 대체 path = PyTorch native `torch.compile` custom op 또는 기존 SR util.

#### 검증 시나리오 (Test Plan)
- **Unit test (분)** — SR unbiasedness:
  - 목적: SR 캐스트가 unbiased(`E[e_i]=0`)인지.
  - Input: 고정 state 텐서 반복 SR 캐스트(N회).
  - Expected: 평균 round error → 0 (N↑), deterministic은 0이 아닌 bias 잔존.
  - metric: 평균 quant error(SR) vs deterministic bias.
  - 실행시간: ~5분. 실패 시 액션: Philox seed/RNG 스트림 검토.
- **Mechanism-isolated test (시간)** — 인과 식별 + 오버헤드 gate:
  - 목적: SR per-step 오버헤드 gate 측정 + verbosity 소멸 여부(bias-drift 인과).
  - Input: {deterministic INT8 / SR-Philox INT8} @ AGX Orin 64GB, AIME25.
  - Expected: SR에서 verbosity 소멸(인과 확립) **AND** RNG 오버헤드 < verbosity 이득.
  - metric: 출력 token Δ, per-step kernel µs, net tok/s.
  - 실행시간: ~6시간. 실패 시 액션: SR 오버헤드가 이득 잠식 → gate로 SR 비활성, M1 단독 보고.

### (제거) M3 — logit guard

- Q-Overthink [arXiv:2606.00206](https://arxiv.org/abs/2606.00206)의 overthinking-marker logit penalty가 **선점** → simplify로 제거(mechanism 2개). baseline 인용으로만 사용. (verbosity-guard 역할은 GOVERNOR/BLOAT에 위임 — cross-share.)

## 5. 실험 플랜 (7-요소)

1. **Hardware**: 1차 **AGX Orin 64GB**(연구실 보유/확보 필요 — edge decode-dominated, EdgeReasoning 재현. RTX 5090은 BW 과대로 edge 미재현 → 제외). BLOAT pair 보조 **Orin Nano 8GB**(확보 필요 — Nemotron-3-Nano-4B proxy, 동일 hybrid, state-quant verbosity 재현). RTX 5090 32GB(직접 사용 가능, lab 보조).
2. **Model**: primary **Nemotron-3-Nano-30B-A3B FP8**(직접 사용). state INT8/INT16 변형 sweep. proxy 보조 **Nemotron-3-Nano-4B**(BLOAT pair, Orin Nano 8GB) / Falcon-H1-1.3B(`falcon_h1.py`) / Bamba-9B(`bamba.py`).
3. **Dataset / Workload**: reasoning + agentic — **AIME25** / **LiveCodeBench** / **τ²-Bench** / **BFCL v4**(budget 고정 조건 — GOVERNOR verbosity 마스킹 회피). 출력 token·accuracy·tok/s 동시 로깅, agent 1→8 sweep.
4. **Tools**: vLLM fork(`7c37096` pin) + per-module quant config + Philox SR Triton kernel + **tegrastats/jtop**(LPDDR 트랜잭션·junction temp) + **nsys**(kernel timeline, SR 오버헤드). ncu Orin 미지원. gem5 미사용.
5. **Ablation + Protocol**: factorial = {FP16 / INT8-deterministic / INT8-SR-Philox / INT16-block128} × {균일 cast vs spectral-budgeted} × {reasoning-on/off} × {agent 1→8}. **핵심 ablation: deterministic→SR에서 verbosity 소멸 = bias-drift 인과 식별**(BLOAT plane). Baselines: **Q-Overthink [arXiv:2606.00206] (preprint, transformer 반대 결론)** 정면 배치, **Quamba [arXiv:2410.13229] (preprint)** accuracy-only, **Q-S5 [arXiv:2406.09477] (preprint)**, **Q-Hurt [arXiv:2504.04823] (preprint)**, **CQD [arXiv:2601.00938] (preprint)**.
6. **Implementation Steps** (Step별 dependency + 완료 판정 수치):
   - Step 1: ρ_ℓ calibration 모듈 + 1K calibration set으로 23 Mamba층 spectral radius 추정 (완료=ρ_ℓ 추정 안정 변동 <5%).
   - Step 2 (←1): KKT closed-form bit allocator + `MambaSpec.dtypes`/per-module config 주입 (완료=budget β 제약 만족 bit map 주입 smoke).
   - Step 3 (병렬, BLOAT 공유): Philox SR Triton kernel(`layers/mamba/ops`) + per-step 오버헤드 선측정 gate (완료=SR unbiased Unit test pass, 오버헤드 µs 측정).
   - Step 4 (←2,3): det vs SR 인과 식별 측정(BLOAT plane 먼저) (완료=SR에서 SSM-recurrent verbosity 소멸 여부 결론).
   - Step 5 (←4): spectral-budgeted full eval — verbosity·accuracy·tok/s (완료=출력 token −25~35%, accuracy ±0.5%p, tok/s +18~30%).
7. **Preliminary Analysis Metrics** (4단계):
   - **baseline repro**: 도구 tegrastats/nsys / 조건 Nemotron-3-Nano FP16 state @ AGX Orin 64GB, AIME25 / 기대 baseline ~9 tok/s, FP16 verbosity 기준선 / 목표 Δ=±10% repro.
   - **bottleneck attribution**: 도구 출력 token 로깅 / 조건 naive INT8 state cast / 기대 baseline +40% verbosity(Super 보고) Nano 재측정 / 목표 Δ=verbosity 증가가 decode-dominated 91%+ 통해 실효 tok/s 직격 정량.
   - **upper bound**: 도구 offline KKT allocator / 조건 ρ_ℓ 분포 / 기대 baseline 균일 cast +40% / 목표 Δ=high-ρ 층만 승급 시 verbosity 이론 −25~35% 상한.
   - **micro-pilot**: 도구 vLLM fork + SR kernel / 조건 det vs SR, 균일 vs spectral / 기대 baseline naive INT8 verbosity +40% / 목표 Δ=SR로 인과 식별 + spectral −25~35% verbosity, +18~30% tok/s.

## 6. 관련 연구 · 차별점

| 연구 | venue tag | 핵심 접근 | SPENDTHRIFT 차별 axis (self-contained) |
|---|---|---|---|
| Q-Overthink [arXiv:2606.00206](https://arxiv.org/abs/2606.00206) | preprint 2026 | quant-verbosity가 spurious("they do NOT need longer"), **transformer** empirical | SPENDTHRIFT는 **SSM-recurrent bias가 ∏A로 동방향 누적**(closed-form)을 검증 가능 가설로 정면 배치 — transformer 반대 결론과 다른 메커니즘 |
| Q-Hurt [arXiv:2504.04823](https://arxiv.org/abs/2504.04823) | preprint | path-forking empirical(attention/weight) | SPENDTHRIFT는 **bias-drift closed-form 모델**(empirical 아닌 정리1/2) |
| Quamba [arXiv:2410.13229](https://arxiv.org/abs/2410.13229) | preprint | SSM state quant accuracy만 | SPENDTHRIFT는 **verbosity→throughput 회계**(accuracy 외 실효 tok/s) |
| Q-S5 [arXiv:2406.09477](https://arxiv.org/abs/2406.09477) | preprint | recurrent weight 민감도 | SPENDTHRIFT는 **bias-vs-variance 분해**(deterministic drift vs SR variance) |
| CQD [arXiv:2601.00938](https://arxiv.org/abs/2601.00938) | preprint 2026 | latent state rate-distortion(verbosity 무관) | SPENDTHRIFT는 **state error → verbosity 인과**(R-D가 아닌 출력 길이 지출) |
| BLOAT (본 세션 pair) | Track B / IISWC 2027 | det vs SR 인과 식별 측정 plane | SPENDTHRIFT 정리1 SR bias 제거를 **실증** → 이론(SPENDTHRIFT)이 그 위에 (수직 pair) |

## 7. 리스크 & Decision-Tree 분기

- **Pass** (verbosity −25~35% **AND** accuracy ±0.5%p **AND** SR에서 verbosity 소멸=인과 식별): ICML 2027 본 투고. SSM-recurrent bias-drift 이론 확립, Q-Overthink 반대 결론 방어 성공.
- **Below** (verbosity −10~25% **OR** accuracy ±0.5~1%p): 조건부 주장(high-ρ 층만)으로 scope 제한, κ 실측 보정. NeurIPS efficiency로 전환, BLOAT 실증 우선 publish.
- **Critical-fail** (SR에서도 verbosity 잔존=bias-drift 아님 **OR** Q-Overthink처럼 SSM도 spurious): 이론 정리2 폐기 → BLOAT 측정 letter(IISWC/CAL)가 "SSM도 transformer와 동일(spurious)"을 first-to-report(여전히 valid 발견).
- **Outperform** (verbosity −35% 이상 **AND** SR 오버헤드 무시 가능 **AND** accuracy 향상): spectral budget을 GOVERNOR thermal budget·MOORING state-resident와 통합, full-stack verbosity 제어로 flagship.
- **임계값 backbone**: naive state quant +40% verbosity(Super), decode-dominated 91~98.6%(ΔL≈Δtok/s), Var bound `σ²/(1−ρ²)`(ρ<1), per-step Philox 오버헤드 gate(이득 > RNG cost).

## 8. Tier-2 scope 축소 variant

SPENDTHRIFT는 이론 main이므로 Tier-2 spinoff 역할은 **pair 논문 BLOAT**(`Deterministic-vs-Stochastic Rounding and the Verbosity Cost of SSM-State Quantization`, Track B / IISWC 2027)가 수행한다 — **pair 참조**. BLOAT는 `MambaSpec.dtypes` 4 변형(FP16 / INT8-deterministic / INT8-SR-Philox / INT16-block128)을 swap-in해 출력 token·useful tok/s·accuracy·LPDDR 메모리를 agent 1→8 sweep으로 로깅하고, **deterministic→SR에서 verbosity 소멸 여부로 bias-drift 인과를 식별**하는 first-to-report 측정 plane이다. 동일 vLLM Mamba2 state-cache hook(SR Triton kernel)을 SPENDTHRIFT M2와 공유(구현 1회·논문 2편). 선점 방어상 BLOAT를 빠른 letter로 먼저 publish하고 SPENDTHRIFT ICML 이론을 분리하는 전략이 valid. SILT(D_verb)·GOVERNOR(verbosity-guard)도 SPENDTHRIFT 정리에 의존하나 별도 idea(SILT는 Tier-2 독립).

## 9. 약어 / 용어 풀이

- **SSM (State Space Model)**: recurrent 상태 텐서로 O(n) 추론하는 시퀀스 모델. Nemotron-3-Nano Mamba-2 23층, state 양자화 오차가 누적되는 대상.
- **Mamba-2**: selective SSM 2세대. state transition matrix A의 spectral radius ρ가 클수록 양자화 오차가 ∏A로 오래 전파(bias drift).
- **MoE (Mixture of Experts)**: 토큰마다 일부 expert FFN 활성화. per-token read 1.84GB → 출력 token 1개 = 1.84GB LPDDR 트랜잭션 → verbosity가 에너지 직결.
- **GQA (Grouped-Query Attention)**: query head가 KV head 공유. Nano attention 6층 KV-head=2(맥락).
- **NoPE (No Positional Encoding)**: 명시적 위치 인코딩 없는 attention 변형(일부 hybrid 채택, 맥락).
- **LPDDR5**: edge SoC 저전력 DRAM. AGX Orin 204.8 GB/s. verbosity = LPDDR 트랜잭션 = 에너지/열.
- **EMC (External Memory Controller)**: Jetson LPDDR 접근 중재. tegrastats EMC util이 대역폭 포화 지표.
- **JS-divergence (Jensen-Shannon divergence)**: 분포 간 대칭 거리. 본 idea 직접 사용 아님(TIDEMARK/HEARTH 영역) — 세션 공통 용어로 기재.
- **bias drift**: deterministic rounding 시 양자화 오차 `e_i`의 기대값이 0이 아니라 `Σ(∏A)e_i`로 동방향 누적되는 현상. SPENDTHRIFT 정리1의 verbosity 동인.
- **stochastic rounding (SR)**: 확률적 반올림으로 `E[e_i]=0`(unbiased) 보장 → bias drift 제거. Philox RNG kernel로 구현, BLOAT가 인과 식별 plane.
- **spectral radius (ρ)**: state transition matrix A의 최대 고유값 절댓값. ρ가 클수록 drift gain `g=1/(1−ρ)` 커짐 → high-ρ 층만 bit 승급.
- **verbosity (ΔL)**: 양자화로 인한 출력 token 수 증가. decode-dominated 91~98.6%라 ΔL이 거의 그대로 실효 throughput 손실.
- **union-growth**: 동시 agent expert working-set 합집합 증가(TIDEMARK 영역) — 세션 공통 용어로 기재(본 idea 직접 사용 아님).
