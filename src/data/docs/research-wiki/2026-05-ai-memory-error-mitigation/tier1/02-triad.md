# TRIAD — Protect–Migrate–Recompute Trichotomy Controller

## 1. Research Questions
- **RQ-2.1**: tensor class(weight / critical-KV / transient-activation)별로 "가장 cheap-sufficient 한 mitigation" 을 runtime 에 선택(weight→exponent-ECC, critical-KV→migrate, activation→recompute)하면, uniform ECC 대비 protection cost(area/energy)를 얼마나 회수하면서 동일 robustness 를 유지하는가?
- **RQ-2.2**: 세 mitigation arm 의 cost·coverage 가 tensor class 와 어떻게 정렬되며, runtime selector 가 static per-class assignment(SHIELD류) 대비 추가 이득이 있는가?

## 2. 배경 및 동기
- **문제 정의**: uniform ECC 는 모든 bit 을 동일 보호 → ECC-based FT([arXiv:2508.12347](https://arxiv.org/abs/2508.12347))에서 BER 1e-1 대응에 ~47.5% area. 하지만 AI workload 의 fault tolerance 는 tensor class 별로 극심하게 다름.
- **기존 접근의 한계**:
  - SHIELD([arXiv:2604.07396](https://arxiv.org/abs/2604.07396)) — tensor-class BER-asymmetry 를 확립했으나 mitigation 은 **refresh-relaxation 1종**.
  - REACH([arXiv:2512.18152](https://arxiv.org/abs/2512.18152)) / Domain-Specific ECC([arXiv:2507.02654](https://arxiv.org/abs/2507.02654)) — ECC **strength** 만 importance-aware 로 조절(다른 mechanism 부재).
  - Ranger([arXiv:2003.13874](https://arxiv.org/abs/2003.13874))(range-clip), Checkmate([arXiv:1910.02653](https://arxiv.org/abs/1910.02653))(rematerialization) — 각 arm 이 isolated.
- **Research Gap (GAP-5)**: 세 mitigation arm(protect/migrate/recompute)을 **runtime 에 cost-min 으로 선택하는 통합 selector** 부재. scoop 검증 ADJACENT — premise(class asymmetry)는 SHIELD 가 점유, 차별점은 **3-arm runtime selector** 임을 명확히 해야 함.

## 3. 핵심 가설
mitigation 을 "가장 cheap-sufficient 한 arm" 으로 tensor class 별 동적 선택하면(weight→exponent-only ECC, critical-KV→migration, activation→recompute), uniform ECC 의 protection cost 를 2.5-4× 회수하면서 동일 robustness 를 유지한다 — 왜냐하면 각 class 의 (값 분포·access·recompute-ability)가 가장 싼 충분 보호를 결정하기 때문.

## 4. Falsifiable predictions
1. exponent/MSB-only ECC 가 weight 의 catastrophic flip(BFA exponent overflow [arXiv:1903.12269](https://arxiv.org/abs/1903.12269))을 uniform ECC 와 동등하게 막아야 한다.
2. activation recompute-on-error 의 비용이 activation ECC 보호 비용보다 낮아야 한다(activation 은 short-lived·recompute-able).
3. 통합 selector 가 SHIELD-style static per-class assignment 대비 >5% cost 추가 회수해야 한다. 아니면 static 으로 충분.

## 5. 접근법 — Mechanism

### M1 — Region Classifier
- **① Scheme**: vLLM/PyTorch allocator 측 `mitigation/region_tag.py` — 각 메모리 영역(weight tensor / KV block / activation buffer)에 tensor-class tag + read/write/recompute 프로파일 부착.
- **② 문제**: class 를 모르면 mitigation 선택 불가.
- **③ 동작**: (1) model load 시 weight region tag. (2) KV block 은 KEEPER M1 criticality 와 연동(critical/cold). (3) activation buffer 는 transient + recompute-able tag. 파라미터: class threshold.
- **④ 차별화**: SHIELD 도 class 구분하나 단일 mitigation. TRIAD 는 class→arm 매핑의 입력.

### M2 — Cheapest-Sufficient Selector
- **① Scheme**: 정책 controller `mitigation/selector.py` — class 별 mitigation arm 선택. arm: {exponent-ECC(HW), migration(KEEPER 재사용), recompute(Checkmate-style)}.
- **② 문제**: uniform ECC 는 recompute-able activation 까지 보호 → 낭비.
- **③ 동작**: (1) weight → exponent/MSB-only ECC (catastrophic flip 차단, mantissa 미보호). (2) critical-KV → KEEPER migration. (3) activation → 무보호 + 오류 검출 시 layer block recompute. (4) cost model 이 robustness 제약 하 arm 선택. 파라미터: per-class robustness target.
- **④ 차별화**: REACH/Domain-ECC 는 strength dial 만; TRIAD 는 **다른 종류 mitigation 을 선택**.

### M3 — Cost Model
- **① Scheme**: arm 별 (area/energy/latency) cost 와 coverage 를 테이블화, min-cost s.t. robustness.
- **② 문제**: arm 선택의 정량 근거 필요.
- **③ 동작**: exponent-ECC area(Verilog 합성 gate), migration bandwidth(KEEPER 측정), recompute FLOPs(layer cost)를 cost 로 두고 LP/greedy. 
- **④ 차별화**: cost-min selector 가 핵심 — SHIELD 엔 부재.

### R50-γ ECC/Overhead 검증
| 항목 | Baseline | 본 mechanism | 초과? | Mitigation |
|------|---------|-------------|-------|-----------|
| weight ECC overhead | uniform SECDED 12.5% | exponent-only: BF16 8 exp bit 중 1 parity = 12.5% of exp ≈ **6.25% of word** | ✓ 이내 | asymmetric (R50-γ.3e) — 단 REACH 점유 axis 라 migration/recompute arm 으로 차별 |
| activation | ECC 12.5% | 0 (recompute) | ✓ | recompute-on-error |
| critical-KV | ECC 12.5% | 0 (migration, KEEPER) | ✓ | migration |

## 6. 예상 효과
| 지표 | Baseline | 목표 | 조건 | 근거 |
|------|---------|------|------|------|
| **[Memory eff.]** protection area/overhead | uniform ECC ~48% (BER 1e-1, ECC-FT) | ~12-18% | 양자화 모델 | arm 별 cheapest |
| **[Energy]** mitigation energy/token | uniform scrub+ECC | -20~-35% | decode | activation 무보호 |
| **[Robustness]** critical-region UE/accuracy | uniform 과 동등 | 동등 유지 | iso-robustness | exponent+migration coverage |

**적용 범위**: 양자화 LLM inference(weight+KV+activation 혼재). **미적용**: 전부 critical 한 소형 모델. **불확실**: recompute latency 가 decode SLO 압박할 수 있음.

## 7. 실험 설계
1. **Hardware**: RTX 5090(#3) / Pro 6000(#5). exponent-ECC 는 Yosys/Sky130 합성.
2. **Model**: LLaMA-3-8B(W4/FP16), Qwen2.5-14B. HF checkpoint. vLLM.
3. **Dataset**: MMLU/HumanEval(accuracy), WikiText(perplexity), LongBench.
4. **Tools**: GoldenTransformer(weight/activation/KV inject), lm-eval, Yosys+Sky130(exponent-ECC area/power), Checkmate-style recompute hook. Ramulator2(보조 error-rate).
5. **Ablation/Protocol**: (a) uniform-ECC vs SHIELD-static vs TRIAD-selector 3-way; (b) per-arm on/off(2^3 factorial); (c) BER sweep. baseline: uniform ECC, SHIELD-mimic, REACH-mimic(strength-only). runtime ~2일/config.
6. **Implementation Steps**:
   - W1-2: STAGE 0 injector + region tag.
   - W3: exponent-only ECC Verilog + 합성(area/power).
   - W4: recompute-on-error hook(activation).
   - W5: migration arm = KEEPER 연동.
   - W6: cost-model selector.
   - W7-8: 통합 + ablation. 완료: TRIAD cost < uniform, robustness 동등.
7. **Preliminary Metrics**: (a) class별 BER-accuracy 곡선; (b) exponent-only ECC area vs full ECC(합성); (c) recompute latency(layer 단위 측정); (d) selector cost-min 결과.

## 8. Code Implementation Verification
- **Clone**: vllm, goldentransformer, (exponent-ECC Verilog 자작). 
- **Patch**: `mitigation/region_tag.py`(Add), `mitigation/selector.py`(Add), activation recompute hook in `model_executor`(Wrap), exponent-ECC = standalone Verilog 합성.
- **Smoke**: inject 3-class faults → selector 가 arm 선택 → robustness 동등 + cost 측정.

## 9. 관련 연구 및 차별점
| Paper | venue/링크 | Date | 관계 | 차별점 |
|-------|-----------|------|------|------|
| SHIELD | [2604.07396](https://arxiv.org/abs/2604.07396) | 2026 | closest (premise 점유) | class asymmetry 를 refresh-relaxation 1종에만. TRIAD 는 3-arm runtime selector |
| REACH / Domain-ECC | [2512.18152](https://arxiv.org/abs/2512.18152) / [2507.02654](https://arxiv.org/abs/2507.02654) | 2025 | strong baseline | ECC strength dial 만. TRIAD 는 migration+recompute 까지 |
| Checkmate | [1910.02653](https://arxiv.org/abs/1910.02653) / MLSys'20 | 2020 | recompute-arm 기원 | memory rematerialization. TRIAD 는 error-recovery 용으로 전용 |
| Ranger | [2003.13874](https://arxiv.org/abs/2003.13874) / DSN'21 | 2020 | adjacent(activation 보호) | range-clip. TRIAD recompute arm 의 대안 |

## 10. 리스크 / 한계
| 리스크 | 영향 | 완화 |
|--------|------|-----|
| SHIELD 가 premise 점유 — "static 으로 충분" critique | 높음 — novelty ADJACENT | selector 의 dynamic 이득(>5% cost 회수)을 ablation 으로 입증. 아니면 recompute-arm 단독 CADENCE(T2)로(decision tree ②) |
| recompute latency 가 decode SLO 압박 | 중간 — p99 위반 | recompute 는 prefill/non-critical activation 한정, critical path 회피 |
| exponent-only ECC 가 mantissa multi-bit 못 막음 | 중간 — 일부 SDC | mantissa 는 error-tolerant(SHIELD 근거), 허용 |

## 11. Boundary Probing 5-axis
| Boundary | 작동 | 깨질 조건 | 진단 |
|----------|------|---------|------|
| Distributional | 양자화 LLM | 전부 critical 모델 | class 분리 가정 |
| Scale | 1B-70B | activation>weight 인 training | inference 가정 |
| Adversarial | random + exponent-targeted | mantissa-targeted backdoor | mantissa 무보호 risk |
| Compositional | 3-arm 공존 | arm 간 자원 경쟁 | selector 조정 |
| Temporal | class 고정 | dynamic class 변화 | re-tag |

## 12. Scoring
| Reviewer | score | 최고 | 최저 |
|----------|-------|------|------|
| novelty | 7.8 | 3-arm runtime selector 통합 | premise(asymmetry)는 SHIELD 점유(ADJACENT) |
| differentiation | 7.9 | migration+recompute 포함 직교 | REACH 와 표면 importance-aware 겹침 |
| impact | 8.1 | uniform ECC 48% headroom 회수 | recompute SLO 영향 |
| ai-implementation | 8.2 | GoldenTransformer 3-arm mimic 명확 | recompute hook 위치 |
| arch-system-implementation | 7.8 | exponent-ECC 합성 가능 | selector cost model 검증 |
| **mean** | **7.96** | — | — |

- **★ 최고**: ai-implementation.3-arm-mimic = 8.4 — GoldenTransformer 가 weight/activation/KV 3종 inject 지원해 selector 검증 직접 가능.
- **▼ 최저**: novelty.premise = 7.4 — class asymmetry 자체는 SHIELD 가 확립. selector 의 dynamic 이득 입증이 novelty 의 사활.
- **전문가 합의**: ai-optimization ✅ / system-robustness ✅ / legacy-system ◯ (SHIELD 차별화 ablation 필수 조건부).
