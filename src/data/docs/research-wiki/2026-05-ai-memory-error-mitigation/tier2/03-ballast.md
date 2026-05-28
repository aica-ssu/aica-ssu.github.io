# BALLAST — Reliability-Cost Term for KV Tiering Placement

## 1. Research Questions
- **RQ-T3**: InfiniGen/LMCache 의 KV tiering placement objective 에 per-frame reliability-cost term 을 추가하면, 성능 저하 없이 critical KV 가 healthy frame 으로 자연 배치되는가?
- **RQ-T3.2**: reliability term 이 multi-objective(latency/throughput/reliability) Pareto 에서 의미 있는 trade-off 곡선을 형성하는가, 아니면 단순 "+1 term" 인가?

## 2. 배경 및 동기
- **문제**: KEEPER 는 event-driven 사후 migration. 더 단순한 접근은 **애초에** placement scheduler 가 reliability 를 objective 에 포함하는 것.
- **한계**: 모든 KV tiering(FlexGen LP [arXiv:2303.06865](https://arxiv.org/abs/2303.06865), InfiniGen [arXiv:2406.19707](https://arxiv.org/abs/2406.19707), LMCache [arXiv:2510.09665](https://arxiv.org/abs/2510.09665), [arXiv:2508.13231](https://arxiv.org/abs/2508.13231), Kareto [arXiv:2603.08739](https://arxiv.org/abs/2603.08739))은 latency/throughput/capacity objective 만.
- **Gap (GAP-4)**: reliability-cost term 부재(scoop 검증 CLEAR, 단 incremental).

## 3. 핵심 가설
기존 KV tiering placement objective 에 per-frame reliability-cost term 을 추가하면, scheduler 가 critical KV 를 healthy frame 으로, cold KV 를 weak frame 으로 자연 배치하여 성능 저하 거의 없이(<1%) silent-corruption 노출을 줄인다 — placement 인프라가 이미 존재하므로 term 추가만으로 충분.

## 4. Falsifiable predictions
1. reliability term 추가가 latency/throughput 를 <1% 만 변화시켜야 한다(거의 free).
2. critical-KV-on-weak-frame 비율이 baseline 대비 유의하게 감소해야 한다.
3. reliability vs performance Pareto 가 비자명한 곡선(단순 "+1 term" 이상)이어야 Tier-2 단독 가치.

## 5. 접근법 — Mechanism (단일 mechanism)
- **① Scheme**: LMCache connector / InfiniGen placement policy 에 per-frame `reliability_cost[frame]` 추가. objective = α·latency + β·capacity + **γ·reliability_cost**.
- **② 문제**: 성능-only placement 가 critical KV 를 weak frame 에 둘 수 있음.
- **③ 동작**: (1) frame-health map(KEEPER M2 재사용 또는 Ramulator2 PRAC)에서 frame 별 reliability_cost 산출. (2) KV block criticality × frame reliability_cost 를 placement objective 에 반영. (3) scheduler 가 critical→healthy, cold→weak 배치. 파라미터: γ weight.
- **④ 차별화**: 기존 objective 에 term 추가(co-design). KEEPER 의 event-driven migration 보다 단순·preventive.

### R50-γ
- 새 parity 없음, placement 정책만. baseline 무관.

## 6. 예상 효과
| 지표 | Baseline | 목표 | 조건 |
|------|---------|------|------|
| **[Robustness]** critical-KV-on-weak-frame 비율 | 무대응(높음) | 대폭 감소 | γ tuned |
| **[Performance]** placement 변경 latency overhead | — | <1% | term 만 추가 |
| **[Robustness]** silent-corruption 노출 accuracy 영향 | 기준 | 저감 | weak-frame 회피 |

**적용 범위**: KV tiering 사용 long-context serving. **미적용**: tiering 없는 in-HBM-only. **불확실**: "+1 term" 으로 보일 incremental risk.

## 7. 실험 설계
1. **Hardware**: RTX 5090(#3)/Pro 6000(#5) + Ramulator2.
2. **Model**: LLaMA-3-8B/70B(W4), Qwen2.5-14B.
3. **Dataset**: LongBench, RULER, ShareGPT, KVCache-in-wild trace.
4. **Tools**: LMCache/vLLM fork, Ramulator2(frame-health), GoldenTransformer(inject), lm-eval.
5. **Ablation**: (a) γ sweep(reliability weight); (b) reliability term on/off; (c) Pareto(latency vs reliability). baseline: InfiniGen/LMCache vanilla, KEEPER(event-driven). runtime ~2일/config.
6. **Implementation Steps**: W1 LMCache connector + baseline. W2 frame-health map 통합. W3 reliability term 추가. W4 Pareto 분석. 완료: <1% perf overhead + critical-KV-on-weak 감소.
7. **Preliminary Metrics**: (a) baseline placement 의 critical-on-weak 비율; (b) γ-Pareto; (c) perf overhead.

## 8. Code Implementation Verification
- **Clone**: LMCache(github), vllm.
- **Patch**: connector placement objective 에 reliability term(Modify), frame-health(reuse KEEPER M2).
- **Smoke**: weak-frame map 주입 → placement 가 critical 회피 확인.

## 9. 관련 연구 및 차별점
| Paper | venue/링크 | Date | 관계 | 차별점 |
|-------|-----------|------|------|------|
| InfiniGen | [2406.19707](https://arxiv.org/abs/2406.19707) / OSDI'24 | 2024 | base(KV tiering) | perf-only. BALLAST 는 reliability term |
| LMCache | [2510.09665](https://arxiv.org/abs/2510.09665) | 2025 | base(multi-tier connector) | reliability-tier 부재. BALLAST 가 추가 |
| Kareto | [2603.08739](https://arxiv.org/abs/2603.08739) | 2026 | adjacent(multi-objective tiering) | latency/throughput/cost objective. reliability 부재 |
| KEEPER | (본 세션 Tier-1) | 2026 | sibling | event-driven migration vs preventive placement |

## 10. 리스크 / 한계
| 리스크 | 영향 | 완화 |
|--------|------|-----|
| "+1 term" incremental critique | 높음 — novelty 약 | multi-objective Pareto 의 비자명성 입증. 약하면 KEEPER 에 흡수(decision tree ②) |
| frame-health 신호 가용성(KEEPER 와 동일 telemetry 한계) | 중간 | Ramulator2 mimic + design-only 명시 |

## 11. Scoring
| Reviewer | score | 최고 | 최저 |
|----------|-------|------|------|
| novelty | 7.2 | reliability term in KV tiering(CLEAR) | incremental "+1 term" |
| differentiation | 7.0 | InfiniGen/LMCache perf-only 대비 | KEEPER 와 목적 인접 |
| impact | 7.6 | 기존 stack 에 즉시 적용 가능 | preventive 만으론 부족할 수 |
| ai-implementation | 8.4 | LMCache connector fork 단순(~300 LoC) | 효과 크기 |
| arch-system-implementation | 7.6 | Ramulator2 frame-health 재사용 | telemetry bridge |
| **mean** | **7.56** | — | — |

- **★ 최고**: ai-implementation = 8.5 — LMCache connector 추상화로 engine 변경 없이 reliability-tier 추가, 구현 risk 최저.
- **▼ 최저**: novelty = 7.2 — "+1 term" 으로 보이지 않으려면 Pareto 비자명성 + KEEPER 와의 명확한 분리(preventive vs reactive) 필요.
- **전문가 합의**: legacy-system ✅ / ai-optimization ◯ (incremental risk 로 Pareto 분석 조건부).
