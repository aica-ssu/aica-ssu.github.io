# BLOAT — Deterministic-vs-Stochastic Rounding and the Verbosity Cost of SSM-State Quantization

> **Hook**: SSM state를 INT8로 양자화하면 메모리는 줄지만 출력 token이 늘어난다(Super 보고 +40% verbosity). 그런데 그 메모리 절감(48→24MB/agent)은 너무 작아서, verbosity로 인한 throughput 손실이 거의 항상 이긴다. BLOAT는 이 useful-throughput cross-over를 측정하고, 더 중요하게 — **deterministic rounding을 stochastic rounding(Philox)으로 바꿨을 때 verbosity가 사라지는지**를 대조해 "bias-drift가 원인"이라는 인과를 식별한다. 이것이 SPENDTHRIFT 이론(정리1 SR bias 제거)의 실증 plane이다.
>
> **Score (Phase 2')**: novelty 5 / diff 7 / impact 6 / ai-impl 7 / arch-impl 8 = **avg 6.6** (Δ0). Tier-2(pair).
> **Venue**: IISWC 2027 (대안 ISPASS / IEEE CAL). Track B 독립 측정 letter.
> **Pair**: SPENDTHRIFT × BLOAT 수직 paper-pair (§7 참조).

---

## 1. Research Questions

- **RQ1 (정량)**: Nemotron-Nano급 hybrid에서 SSM state INT8 양자화 시 **출력 token 증가율(verbosity)**은 얼마이며(Super의 +40%를 검증/반박), state 절감(48→24MB/agent)을 고려한 **useful tok/s 순효과의 cross-over agent 수**는 몇인가? (예상: 절감이 작아 "거의 항상 verbosity 우세").
- **RQ2 (인과)**: deterministic rounding → stochastic rounding(Philox)으로 바꿨을 때 **verbosity가 사라지는가**? 사라지면 "bias-drift가 verbosity 원인"(SPENDTHRIFT 정리1)이 식별되고, 사라지지 않으면 일반 quant-overthinking(Q-Overthink, transformer)으로 귀결 — 이 분리가 사활.

---

## 2. 기준 코드베이스 (Baseline Source)

- **서빙 스택**: vLLM v1, commit **`7c37096620`** pin. NGC container `nvcr.io/nvidia/vllm:26.01-py3`(Jetson aarch64 pip 미지원 대응). state cast hook = `mamba_mixer2.py` state update 직후 + `MambaSpec.dtypes`(state dtype 4 변형 swap-in).
- **모델**: Nemotron-3-Nano-30B-A3B (HF, 풀모델 FP8, AGX Orin) + **[Nemotron-3-Nano-4B](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b)**(동일 hybrid 구조 proxy, Orin Nano 8GB). Quamba 비교군 동일 SKU(Orin Nano 8GB)로 valid 비교.
- **측정 도구**: tegrastats(tok/s·LPDDR 메모리) + vLLM 로깅(출력 token·useful tok/s·accuracy). ncu 불요(전부 user-space). SR kernel = Triton(`layers/mamba/ops`) Philox 캐스트.
- **benchmark**: **BFCL v4 + AIME25**(budget 고정 — GOVERNOR verbosity 마스킹 회피).
- **HW 측정 가능성 검증** (arch-impl review [✅] 인용):
  | 항목 | 판정 | 근거 |
  |---|---|---|
  | useful tok/s · 출력 token · accuracy · LPDDR 메모리 측정 | [✅] | tegrastats + vLLM 로깅, ncu 불요 |
  | 4-변형 state cast swap-in (FP16/INT8-det/INT8-SR/INT16-block128) | [✅] | `MambaSpec.dtypes` application-level |
  | Orin Nano 8GB 풀모델 미적재 | [⚠️] | 4B proxy 대표성 리스크(arch 인지) — 동일 Mamba-2 구조 small variant로 완화 |
  | Quamba 비교군 SKU 일치 | [✅] | Orin Nano 8GB 동일 → 비교 valid |
  - arch feasibility 8/10 — 명확한 단일 mechanism, 실디바이스, simulator 불요.

---

## 3. 배경 / GAP

- SSM state 양자화 → verbosity **+40%** ([Super 보고, Eq3 `h_q,t = h_t + Σ(∏A)e_i`]) — 단 **Super는 이질 모델**, Nano-급 hybrid 재측정이 BLOAT 고유 기여.
- decode-dominated **91.0–98.6%** ([arXiv:2605.26297](https://arxiv.org/abs/2605.26297)) → 출력 token 수 = 거의 그대로 실효 throughput 손실 → verbosity가 throughput에 직격.
- per-token expert read **1.84GB/token** → 출력 token 1개 = 1.84GB LPDDR 트랜잭션 → verbosity = 메모리/에너지 직결.
- **state 절감이 작다**: 48→24MB/agent → state pool 절감이 미미해 verbosity 손실이 거의 항상 이김 → cross-over는 agent가 매우 많을 때만(정직한 결론, 그 자체가 valid first-to-report).
- **인과 식별 부재**: Quamba/QMamba는 SSM state quant의 **accuracy/latency만** 보고 — verbosity·useful-throughput·인과(deterministic vs SR) 측정이 전무.
- **Q-Overthink 반대 결론**: [arXiv:2606.00206](https://arxiv.org/abs/2606.00206)는 "quantized reasoning models think they need to think longer, but they do **NOT**"(transformer, spurious) — **+40%가 SSM-recurrent 고유인지 일반 quant-overthinking인지** 분리가 BLOAT의 핵심 질문.
- stochastic rounding(SR)의 bias 제거 효과를 **verbosity로 환산한 연구가 없음** → SR을 인과 식별 도구로 쓰는 것이 신규.

---

## 4. Mechanism (단일)

### 4.1 동작 원리 — M1: Bias-controlled rounding 대조 + useful-throughput cross-over 측정
- `MambaSpec.dtypes` + `mamba_mixer2.py` state cast에 **4 변형**(FP16 / INT8-deterministic / INT8-SR-Philox / INT16-block128) swap-in.
- 동일 task·budget 고정에서 출력 token·useful tok/s·accuracy·LPDDR 메모리 로깅. agent 1→8 sweep.

### 4.2 기대 효과
- 출력 token 증가율(Super +40% 검증/반박) + useful tok/s 순효과 cross-over agent 수 보고.
- **deterministic→SR에서 verbosity 소멸 여부 = bias-drift 인과 식별**(SPENDTHRIFT 정리1 실증).

### 4.3 구현 변경점
| 구분 | Phase-1 원본 | Phase-1' 정제 |
|---|---|---|
| 핵심 | 4-변형 throughput cross-over | **deterministic vs SR 인과 식별** 추가 |
| +40% | 검증/반박 | Super(이질 모델) 명시 + **Nano-4B 재측정이 기여** |
| SPENDTHRIFT | 별개 | **측정→이론 검증 plane pair 명시** |
| proxy | 소형 hybrid | Nemotron-3-Nano-4B 명시 |
| R47 path | — | vLLM v1 fork, `MambaSpec.dtypes` 4 변형 + SR Triton kernel + 로깅, gem5 미사용, 실디바이스 |

### 4.4 검증 시나리오
1. 동일 task(BFCL v4 + AIME25, budget 고정)에서 4 변형 실행.
2. per-config 출력 token·useful tok/s·accuracy·LPDDR 메모리 측정.
3. agent 1→8 sweep → useful tok/s 순효과 cross-over agent 수 산출.
4. **deterministic → SR에서 verbosity 소멸 여부** 확인 → bias-drift 인과 식별 / 일반 overthinking 분리.

---

## 5. 실험 플랜 (7-요소, 단일 scope)

1. **목표 지표**: 출력 token 증가율(+X%, Super +40% 검증/반박) / useful tok/s cross-over agent 수 / **인과 식별(SR에서 verbosity 소멸 여부)** — 전부 측정치.
2. **device**: **AGX Orin 64GB**(Nano 풀모델 FP8) + **Orin Nano 8GB**(Nemotron-3-Nano-4B proxy, state-quant verbosity 재현). Quamba 비교군 동일 Orin Nano 8GB SKU.
3. **모델/precision**: 풀모델 FP8 + state 4 변형(FP16 / INT8-deterministic / **INT8-SR-Philox** / INT16-block128).
4. **대조 핵심**: **deterministic vs stochastic rounding (Philox)** — bias 통제로 인과 식별. SPENDTHRIFT M2 SR kernel과 hook 공유.
5. **sweep 축**: state precision 4 변형 × agent 1→8.
6. **측정**: 출력 token(vLLM 로깅) · useful tok/s · accuracy(BFCL v4/AIME25) · LPDDR 메모리(tegrastats). budget 고정 필수.
7. **분석**: per-config 비교 → cross-over agent 수 + det→SR verbosity delta로 bias-drift 인과 확립.

| 축 | baseline | 목표(측정) | 조건 |
|---|---|---|---|
| [측정] 출력 token 증가율 | FP16 state | +X% (Super +40% 검증/반박) | INT8-det, decode |
| [측정] useful tok/s 순효과 | 메모리 절감 가정 | cross-over agent 수 | multi-agent |
| [측정] 인과 식별 | — | SR에서 verbosity 소멸 여부 | det vs SR 대조 |

> 보수적: state 절감 작아(48→24MB) cross-over가 "거의 항상 verbosity 우세"로 나올 수 있음 — 그 자체가 valid first-to-report.

---

## 6. 관련 연구 · 차별점

- **Quamba** [arXiv:2410.13229](https://arxiv.org/abs/2410.13229) (arXiv) — SSM state quant accuracy/latency만. 차별: useful tok/s cross-over + **det vs SR 인과 식별** first-to-report.
- **QMamba** [arXiv:2501.13624](https://arxiv.org/abs/2501.13624) (arXiv) — vision PTQ. 차별: serving verbosity·throughput.
- **Q-Overthink** [arXiv:2606.00206](https://arxiv.org/abs/2606.00206) (arXiv 2026) — transformer, verbosity가 spurious("do NOT"). 차별: **transformer 대조군** — +40%가 SSM-recurrent 고유(bias-drift)인지 일반 overthinking인지 분리.

---

## 7. 왜 Tier-2 only인가

- **Top-tier scale-up 불가 이유**: 측정 harness라 **mechanism novelty가 본질적으로 낮음**(novelty 5, mechanism 3). 새로운 알고리즘이 아니라 4-변형 swap + 인과 식별 측정이므로 single-insight scope. state 절감이 작아(48→24MB) impact도 제한(impact 6) — cross-over가 "거의 항상 verbosity 우세"로 나올 가능성. ICML/MLSys급 이론 contribution은 SPENDTHRIFT가 담당하고, BLOAT는 그 실증 plane이라 독립 Track-A scale-up은 부적합.
- **Tier-1 승격 조건**: det vs SR 인과 식별이 **단순 검증을 넘어 새로운 SSM-recurrent verbosity 이론으로 일반화**되거나(그 경우 SPENDTHRIFT에 흡수), 다수 hybrid 모델 family로 cross-over 법칙을 일반화하면.
- **SPENDTHRIFT paper-pair 관계 명시**: **BLOAT(인과 식별 plane) → SPENDTHRIFT(이론) 수직 pair**. 같은 vLLM Mamba2 state-cache hook 공유(구현 1회·논문 2편). BLOAT가 먼저 cross-over·인과를 확립(정리1 SR bias 제거 실증) → SPENDTHRIFT 이론이 그 위에. +40% 출처가 Super(이질)임을 명시, **Nano-급 hybrid 재측정이 BLOAT 기여**. 선점 방어상 **BLOAT 빠른 letter(IISWC/CAL) + SPENDTHRIFT ICML 분리** 전략 valid.

---

## 8. 약어 / 용어 풀이

- **state**: Mamba-2 recurrent SSM 상태 텐서(heads 64 × head-dim 64 × d_state 128 × 23 Mamba층 ≈ 48MB/agent FP16, context 무관 상수).
- **verbosity**: state 양자화로 인한 출력 token 수 증가(decode-dominated라 throughput 손실 직결).
- **deterministic rounding**: 양자화 시 반올림 — `E[h_q,t−h_t] = Σ(∏A)E[e_i] ≠ 0`로 bias가 ∏A 방향으로 누적.
- **stochastic rounding (SR)**: 확률적 반올림 — `E[e_i]=0` → bias 제거, Var는 `σ²/(1−ρ²)` bounded. Philox RNG로 구현.
- **Philox**: counter-based RNG(GPU 병렬 친화) — SR 캐스트의 난수원.
- **useful tok/s**: verbosity를 보정한 실효 throughput(메모리 절감 vs 출력 token 증가의 순효과).
- **cross-over agent 수**: state-quant 메모리 절감 이득이 verbosity 손실을 넘어서는 동시 agent 수.
- **bias-drift**: deterministic rounding 오차가 SSM recurrence(∏A)를 따라 동방향 누적되는 현상(SPENDTHRIFT 정리1).
- **INT16-block128**: 128-element block 단위 INT16 양자화 변형.
