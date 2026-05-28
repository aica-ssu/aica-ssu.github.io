# ROWPRESS-AI — Inference-Induced Read-Disturbance Characterization & Mitigation

## 1. Research Questions
- **RQ-3.1**: long-context decode 의 heavy-hitter KV 반복 read 가 모델 자신의 access pattern 으로 neighbor row 에 read-disturbance(RowPress-류)를 유발하는 **benign self-aggressor** 인가?
- **RQ-3.2**: 그렇다면 hot-KV block rotation 또는 victim-row guard 로 disturbance-유발 bit-flip 을 얼마나 낮추며, throughput cost 는 얼마인가?

## 2. 배경 및 동기
- **문제 정의**: RowHammer/RowPress 는 통상 **외부 공격자** 의 의도적 hammering 으로 다뤄짐. 그러나 LLM long-context decode 는 동일 heavy-hitter KV row 를 수십만 회 반복 read 한다(H2O [arXiv:2306.14048](https://arxiv.org/abs/2306.14048): 소수 token 이 attention mass 지배). 이 benign 반복 read 자체가 neighbor row 에 read-disturbance 를 유발하는 self-aggressor 일 수 있다.
- **기존 접근의 한계**:
  - GPUHammer([arXiv:2507.08166](https://arxiv.org/abs/2507.08166)), DeepHammer([arXiv:2003.13746](https://arxiv.org/abs/2003.13746)) — 외부 공격자 hammering 가정.
  - HBM2 read-disturbance([arXiv:2310.14665](https://arxiv.org/abs/2310.14665)) — synthetic access pattern 만.
  - RowPress(ISCA'23 계열) — row 장시간 open + 반복 access 시 disturbance. LLM access 가 이 조건에 부합하는지 미검증.
- **Research Gap (GAP-3)**: **LLM 자신의 benign access pattern 을 self-induced read-disturbance aggressor 로 특징화** 한 사례 없음(scoop 검증 CLEAR).

## 3. 핵심 가설
long-context decode 의 heavy-hitter KV 반복 read 는 그 row 의 activation/open 횟수를 RowPress threshold 근처까지 누적시켜, neighbor row 에 benign self-induced read-disturbance 를 유발한다 — 따라서 hot-KV rotation 으로 단일 row 누적을 분산하면 disturbance flip 을 낮출 수 있다.

## 4. Falsifiable predictions
1. Ramulator2 access trace 에서 heavy-hitter KV row 의 refresh-window 당 activation 이 일반 row 대비 **수십~수백 배** 높아야 한다. 아니면 self-aggressor 가설 기각.
2. literature RowPress BER 모델 하, heavy-hitter row 의 neighbor disturbance 확률이 무시 못할 수준(>baseline)이어야 한다.
3. rotation 적용 시 단일 row 최대 activation 이 균등화돼 disturbance 추정치가 낮아져야 한다. 아니면 완화 무용.

## 5. 접근법 — Mechanism

### M1 — Access-Trace → Activation Count
- **① Scheme**: vLLM 에서 KV access trace 추출 hook + Ramulator2 PRAC plugin 으로 per-row activation/open count 산출.
- **② 문제**: self-disturbance 정량화하려면 실제 row 별 access 분포 필요.
- **③ 동작**: (1) decode 중 KV block→physical row 매핑 trace 기록. (2) Ramulator2 에 trace 주입, PRAC counter 로 per-row activation. (3) refresh-window 단위 누적. 파라미터: ctx length, batch.
- **④ 차별화**: 기존은 synthetic; 본 idea 는 실제 LLM trace.

### M2 — Self-Disturbance Quantification
- **① Scheme**: per-row activation 을 literature RowPress/read-disturbance BER 모델(HBM2 [arXiv:2310.14665](https://arxiv.org/abs/2310.14665), DiscoRD [arXiv:2603.12435](https://arxiv.org/abs/2603.12435) RDT spatial variation)에 대입해 neighbor flip 확률 추정.
- **② 문제**: 실측 disturbance 는 testbed 필요 → 불가. literature 모델로 추정.
- **③ 동작**: (1) DiscoRD 의 RDT 분포 + activation count → flip 확률. (2) heavy-hitter vs cold row 비교. (3) GoldenTransformer 로 해당 flip 을 KV 에 주입해 accuracy 영향 측정.
- **④ 차별화**: benign workload 의 self-disturbance map — 최초.

### M3 — Mitigation: Hot-KV Rotation / Guard
- **① Scheme**: `core/block_manager` 에 hot-KV block periodic remap(rotation) — 동일 row 누적 분산. 또는 victim-row guard(인접 row buffer).
- **② 문제**: 단일 hot row 누적이 disturbance 원인.
- **③ 동작**: (1) M1 hot row 식별. (2) N step 마다 hot-KV block 을 다른 physical row 로 rotate. (3) 단일 row 최대 activation 균등화. 파라미터: rotation period.
- **④ 차별화**: 공격 방어(RowHammer mitigation)가 아닌 self-disturbance 완화.

### R50-γ
- 새 parity 없음, block remap·trace 만. metadata 무시 가능. baseline 이내.

## 6. 예상 효과
| 지표 | Baseline | 목표 | 조건 | 근거 |
|------|---------|------|------|------|
| **[Robustness]** heavy-hitter row 최대 activation/window | 수십만 | rotation 후 균등 | 32K-128K ctx | M3 분산 |
| **[Robustness]** disturbance-유발 flip rate (literature 모델) | rotation 없음 | 저감 | DiscoRD RDT 모델 | 조건부 |
| **[Performance]** rotation throughput cost | — | <2% | event-driven | rare |

**적용 범위**: 매우 긴 context + heavy-hitter 집중 workload. **미적용**: short-context, uniform attention. **불확실(중대)**: 실측 disturbance BER 불가 → **literature 모델 가정에 의존**. 효과는 모델 의존적.

## 7. 실험 설계
1. **Hardware**: RTX 5090(#3, trace 생성) + Ramulator2(CPU). 실HW disturbance 측정 **불가** 명시.
2. **Model**: LLaMA-3-8B, Qwen2.5-14B long-ctx.
3. **Dataset**: RULER 128K, LongBench, 합성 long-ctx.
4. **Tools**: vLLM(trace), Ramulator2(PRAC activation count), DiscoRD/HBM2-RD BER 모델(literature), GoldenTransformer(flip inject), lm-eval.
5. **Ablation/Protocol**: (a) heavy-hitter vs cold row activation 분포; (b) ctx length sweep; (c) rotation period sweep; (d) BER 모델 sensitivity(여러 RDT 가정). baseline: rotation 없음, RowHammer-mitigation(BreakHammer-mimic). runtime ~2일/config.
6. **Implementation Steps**: W1-2 trace hook + Ramulator2 활성. W3 activation 분포 분석. W4 BER 모델 결합. W5 rotation mitigation. W6-7 sensitivity + 분석. 완료: heavy-hitter row activation 우위 정량 + rotation 균등화.
7. **Preliminary Metrics**: (a) per-row activation 히스토그램; (b) heavy-hitter/cold ratio; (c) BER 모델 하 flip 확률; (d) rotation 후 분포.

## 8. Code Implementation Verification
- **Clone**: vllm, ramulator2(PRAC), goldentransformer.
- **Patch**: vLLM KV trace hook(Add), Ramulator2 activation export(기존 PRAC), rotation in `block_manager`(Modify).
- **Smoke**: long-ctx decode → trace → Ramulator2 activation → heavy-hitter row 우위 확인.

## 9. 관련 연구 및 차별점
| Paper | venue/링크 | Date | 관계 | 차별점 |
|-------|-----------|------|------|------|
| GPUHammer | [2507.08166](https://arxiv.org/abs/2507.08166) / USENIX Sec'25 | 2025 | closest(GPU disturbance) | 외부 공격자. ROWPRESS-AI 는 benign self-aggressor |
| HBM2 read-disturbance | [2310.14665](https://arxiv.org/abs/2310.14665) | 2023 | baseline(disturbance char) | synthetic pattern. 본 idea 는 실 LLM trace |
| DiscoRD | [2603.12435](https://arxiv.org/abs/2603.12435) | 2026 | RDT 모델 출처 | RDT spatial 측정 방법. 본 idea 가 BER 모델로 사용 |
| BreakHammer | [2404.13477](https://arxiv.org/abs/2404.13477) / MICRO'24 | 2024 | mitigation baseline | suspect thread throttle(공격). 본 idea 는 benign rotation |

## 10. 리스크 / 한계
| 리스크 | 영향 | 완화 |
|--------|------|-----|
| **실측 disturbance 불가 — 가설이 literature BER 모델에만 의존** | 높음 — 효과 입증이 가정적 | (a) 여러 RDT 가정에 대한 sensitivity 로 robust 결론. (b) characterization(activation 우위)은 측정 가능하므로 그 부분만으로도 letter 성립. legacy-system-expert reservation 반영 — "부재 입증"보다 "조건부 gain" 으로 포지셔닝 |
| heavy-hitter row 가 실제로는 자주 close/refresh 돼 누적 안 됨(RowPress 조건 미충족) | 높음 — 가설 기각 | M1 에서 row open duration 도 측정. 미충족 시 characterization-only letter(T2)로 reposition(decision tree ②) |
| rotation 이 cache locality 깨 throughput 저하 | 중간 | event-driven + 낮은 빈도, locality-aware target row |

## 11. Boundary Probing 5-axis
| Boundary | 작동 | 깨질 조건 | 진단 |
|----------|------|---------|------|
| Distributional | heavy-hitter 집중 attention | uniform attention | 가설 핵심 |
| Scale | 매우 긴 ctx(128K+) | short ctx | 누적 부족 |
| Adversarial | benign | 의도적 hammer | 본 idea scope 밖 |
| Compositional | KV migration(KEEPER)과 공존 | rotation×migration 충돌 | 조정 필요 |
| Temporal | hot-set 지속 | hot-set 급변 | rotation 무효 |

## 12. Scoring
| Reviewer | score | 최고 | 최저 |
|----------|-------|------|------|
| novelty | 8.8 | benign self-aggressor framing(CLEAR, 최초) | RowPress 자체는 알려진 현상 |
| differentiation | 8.5 | 모든 선행은 공격자 가정 | 완화책(rotation)은 기존 remap 류 |
| impact | 7.6 | 새 reliability 위협 부류 제기 | 효과가 모델/가정 의존 |
| ai-implementation | 7.4 | trace+inject 경로 명확 | rotation 구현 |
| arch-system-implementation | 7.0 | Ramulator2 activation 측정 가능 | 실 disturbance BER 측정 불가 |
| **mean** | **7.86** | — | — |

- **★ 최고**: novelty.self-aggressor = 9.0 — benign inference access 를 disturbance aggressor 로 본 framing 은 선행 0편, 새 위협 부류.
- **▼ 최저**: arch-system.disturbance-validation = 6.6 — 실측 불가, literature BER 모델 의존이 가장 큰 약점. characterization 부분(activation 우위)은 측정 가능.
- **전문가 합의**: system-robustness ✅ (novelty) / ai-optimization ◯ / legacy-system △ (BER-모델 의존을 conditional-gain 으로 명시하는 조건부, characterization-only fallback 권장).
