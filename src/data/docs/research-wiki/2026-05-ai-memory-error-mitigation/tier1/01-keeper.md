# KEEPER — ECC-Telemetry-Driven Critical-KV Cache Migration

## 1. Research Questions
- **RQ-1.1**: HBM3 errors-per-row counter / DDR5 PRAC counter / ECS error log 를 free한 "frame-health map" 으로 재해석하면, heavy-hitter KV block 을 weak frame 으로부터 event-driven migrate 하여 silent KV corruption 의 accuracy 영향을 page-offlining 대비 capacity 손실 없이 방어할 수 있는가?
- **RQ-1.2**: migration trigger 의 criticality term(attention mass)이 reliability-blind placement 대비 동일 migration budget 에서 더 큰 accuracy 보존을 주는가?

## 2. 배경 및 동기
- **문제 정의**: GPUHammer([arXiv:2507.08166](https://arxiv.org/abs/2507.08166))는 GDDR6 단일 bit-flip 이 DNN accuracy 를 80%까지 떨어뜨림을 실증. 한편 H100 HBM3 의 per-GB MTBE 는 A100 대비 24%↓이고 row-remap/page-offline budget 이 이미 포화([arXiv:2503.11901](https://arxiv.org/abs/2503.11901)). LLM serving 에서 메모리의 지배적·성장 소비자는 **KV cache** 인데, 기존 reliability 방어는 거의 전부 **static weight** 만 대상으로 함.
- **기존 접근의 한계**:
  - DNN-Defender([arXiv:2305.08034](https://arxiv.org/abs/2305.08034)) / DRAM-Locker([arXiv:2312.09027](https://arxiv.org/abs/2312.09027)) — critical **weight** row swap/lock 만. KV 동적 특성 미대응.
  - 성능-only KV placement (InfiniGen [arXiv:2406.19707](https://arxiv.org/abs/2406.19707), LMCache [arXiv:2510.09665](https://arxiv.org/abs/2510.09665), [arXiv:2508.13231](https://arxiv.org/abs/2508.13231)) — reliability 신호 부재.
  - page-offlining (ICCD'21, RL HPDC'24 [arXiv:2407.16377](https://arxiv.org/abs/2407.16377)) — capacity 손실 + criticality-blind.
- **Research Gap**: (a) 동적 KV 의 error-aware 배치/migration 미탐구 (GAP-1), (b) PRAC/ECS counter 를 reliability health-map 으로 재활용한 사례 없음 (GAP-2, scoop 검증 CLEAR — 모든 PRAC 연구는 보안용).

## 3. 핵심 가설
KV block 의 criticality(attention mass)와 frame 의 health(error telemetry)를 결합하면, **소수 heavy-hitter KV 만 healthy frame 으로 옮기는 event-driven migration** 으로 silent-corruption 유발 accuracy drop 을 capacity 손실 없이 방어할 수 있다 — 왜냐하면 (i) accuracy 는 소수 heavy-hitter KV 에 집중되고(H2O), (ii) error 는 소수 weak frame 에 집중되며(field study), (iii) 둘의 교집합(critical-KV on weak-frame)만 처리하면 충분하기 때문이다.

## 4. Falsifiable predictions
1. reliability-blind random migration 과 비교해 동일 migration count 에서 accuracy 보존이 **유의미하게(>10%) 우월**해야 한다. 아니면 criticality term 무용.
2. heavy-hitter KV 비율이 충분히 작아(<20%) 전체 KV 의 일부만 옮겨도 accuracy 대부분 보존돼야 한다.
3. migration bandwidth overhead 가 decode throughput 의 **<2%** 여야 한다 (event-driven rare). 초과 시 always-on 보호 대비 이점 상실.

## 5. 접근법 — Mechanism (R20-α 4요소)

### M1 — Criticality Scorer
- **① 추가 Scheme**: vLLM `vllm/attention/backends/` 에 per-block attention-mass accumulator 추가. PagedAttention block 마다 누적 attention weight(softmax score 합)를 running sum 으로 유지하는 `block_criticality[block_id]` 테이블.
- **② 해결 문제**: 어떤 KV block 이 accuracy-critical 인지 모르면 모든 KV 를 동일 보호 → 비용 과다. H2O([arXiv:2306.14048](https://arxiv.org/abs/2306.14048))는 소수 heavy-hitter 가 attention mass 대부분 차지함을 보임.
- **③ 동작 원리 (step-by-step)**: (1) attention kernel 이 score 계산 시 block 별 mass 를 atomic add. (2) 주기적(매 N decode step)으로 top-k% block 을 heavy-hitter 로 flag. (3) flag 를 migration controller 에 전달. 파라미터: k∈[5,20]%, N∈[16,64].
- **④ 기존 해법 실패 이유 + 차별화**: H2O 는 eviction(버림)용, 본 idea 는 **보존 우선순위**용으로 재사용. attention-mass 는 이미 kernel 이 계산하므로 추가 비용 <3% FLOPs.

### M2 — Frame-Health Map
- **① 추가 Scheme**: 새 모듈 `reliability/frame_health.py` — Ramulator2 PRAC/ECS 출력(per-row activation count, CE count, ECS error log)을 읽어 frame 별 health score 유지하는 weak-frame 우선순위 큐.
- **② 해결 문제**: weak frame 을 알아야 회피 가능. 기존엔 이 신호를 보안(RowHammer)에만 사용.
- **③ 동작 원리**: (1) Ramulator2 PRAC plugin(`example_config_prac.yaml`)에서 per-row activation·CE count export. (2) CE rate·activation 누적이 임계 초과 frame 을 weak 로 분류. (3) HBM3 ECS "errors-per-row counter" / RCC 모드 결과를 동일 map 에 통합. 파라미터: CE threshold, EWMA window.
- **④ 차별화**: MOAT/CnC-PRAC([arXiv:2506.11970](https://arxiv.org/abs/2506.11970))/Citadel([arXiv:2409.15463](https://arxiv.org/abs/2409.15463))는 counter 를 refresh·보안 isolation 에 사용 — **reliability placement 신호로 재활용은 본 idea 가 최초** (scoop 검증 CLEAR).

### M3 — Migration Controller
- **① 추가 Scheme**: vLLM `core/block_manager` 에 reliability-aware block (re)allocation hook. critical-KV block 이 weak frame 에 있으면 healthy frame 으로 copy + 매핑 갱신; cold KV 는 weak frame 으로 park.
- **② 해결 문제**: critical-KV on weak-frame 교집합이 silent corruption → accuracy drop. offlining 은 frame 을 retire 해 capacity 손실(KV 거대 → 치명적).
- **③ 동작 원리**: (1) M1 critical flag × M2 weak flag 교집합 탐지. (2) 교집합 block 을 healthy free frame 으로 `cudaMemcpy`(또는 mimic). (3) page table/block table 갱신. (4) cold KV 는 weak frame 으로 우선 배치(capacity 활용). event-driven(교집합 발생 시만) → rare.
- **④ 차별화**: migration(capacity 보존) ≠ offlining(capacity 손실). DNN-Defender 의 in-DRAM swap 은 weight·HW 전용, KEEPER 는 framework-level KV·동적.

### R50-γ ECC/Overhead 검증
- KEEPER 는 **새 parity/ECC bit 을 추가하지 않음** — 기존 telemetry 재활용 + block 이동만. metadata: block_criticality(int32/block) + frame_health(int16/frame) = KV block 당 수 byte → PagedAttention block(예: 16 token × n_head × head_dim × 2byte) 대비 <0.1%. **baseline 12.5% 이내, mitigation 불필요.**

## 6. 예상 효과 (보수적)
| 지표 | Baseline | 목표 | 적용 조건 | 근거 |
|------|---------|------|----------|------|
| **[Robustness]** BER 1e-5 long-context accuracy drop | -8~-15% (reliability-blind) | -1~-3% | LLaMA-3-8B, 32K ctx | heavy-hitter 보호 시 대부분 회복 |
| **[Memory eff.]** mitigation 당 capacity 손실 | offlining: row retire | 0 | migration | capacity 보존 |
| **[Performance]** migration bandwidth overhead | — | <1-2% | event-driven | rare trigger |
| **[Robustness]** vs reliability-blind random migration | 기준 | +>10% accuracy 보존 | iso-budget | RQ-1.2 |

**적용 범위**: long-context decode(KV 지배적), BER 1e-6~1e-3 가정 구간. **미적용**: prefill-only/short-context(KV 작음), weight-dominant 모델. **불확실**: 실제 HBM3 errors-per-row counter 의 host-노출 여부(현재 미노출 → simulation mimic 으로 검증, 실HW 적용은 future).

## 7. 실험 설계 (R20-β 7요소)
1. **Hardware**: RTX 5090 32GB single host (#3) / RTX Pro 6000 96GB(#5, 70B). CPU 측 Ramulator2.
2. **Model**: primary LLaMA-3-8B(FP16/W4), secondary Qwen2.5-14B, robustness Mistral-7B. checkpoint: HF. inference base: vLLM 0.6+.
3. **Dataset/Workload**: LongBench, RULER(32K-128K), ShareGPT. real trace + KVCache-in-wild([arXiv:2506.02634](https://arxiv.org/abs/2506.02634)) reuse 분포 mimic. metric: accuracy/EM, perplexity, migration count.
4. **Simulator/Tools**: Ramulator2(PRAC config, errors-per-row) — error-rate/weak-frame 표 추출(보조 eval, **R55: contribution 아님**). GoldenTransformer([arXiv:2509.10790](https://arxiv.org/abs/2509.10790)) — KV bit-flip injection. lm-evaluation-harness — degradation/recovery 측정.
5. **Ablation/Protocol**: (a) criticality-aware vs reliability-blind vs random migration 3-way; (b) k%(5/10/20), BER sweep(1e-6~1e-3); (c) baseline: no-mitigation, page-offlining, DNN-Defender-mimic(weight), always-on KV ECC. expected runtime ~2-3일/config, ~8 config. fallback: 8B 불가 시 1-3B PoC.
6. **Implementation Steps (week-level)**:
   - W1-2: STAGE 0 — GoldenTransformer KV/weight injector + vLLM KV block 접근 hook + lm-eval baseline.
   - W3: Ramulator2 PRAC/ECS error-rate·weak-frame 표 추출 파이프라인.
   - W4-5: M1 criticality scorer (`attention/backends` accumulator) + 검증.
   - W6: M2 frame-health map (`reliability/frame_health.py`).
   - W7-8: M3 migration controller (`core/block_manager` hook) + 통합.
   - W9-10: ablation sweep + 분석.
   - 완료 판정: criticality-aware 가 reliability-blind 대비 accuracy 보존 우위 통계적 유의.
7. **Preliminary Analysis Metrics**: (a) heavy-hitter KV 비율 측정(torch hook, 기대 5-20%); (b) BER vs accuracy degradation 곡선(injection 후 lm-eval); (c) weak-frame 분포(Ramulator2 PRAC); (d) migration count vs accuracy 보존 trade-off. Preliminary 4단계: baseline reproduction → heavy-hitter attribution → injection-accuracy 곡선 → migration pilot.

## 8. Code Implementation Verification
- **Clone Spec**: `git clone https://github.com/vllm-project/vllm` (PagedAttention block_manager); `git clone https://github.com/CMU-SAFARI/ramulator2`(PRAC); `git clone https://github.com/FuzzyNum/goldentransformer`.
- **Patch outline**:
  | M# | File | Type | 검증 |
  |----|------|------|------|
  | M1 | `vllm/attention/backends/flash_attn.py` (block mass accumulator) | Wrap | ⚠️ partial — kernel hook 비용 측정 필요 |
  | M2 | `reliability/frame_health.py` (new) | Add | ✅ Ramulator2 PRAC config 존재 확인 |
  | M3 | `vllm/core/block_manager.py` (reliability alloc hook) | Modify | ⚠️ block table 매핑 ripple 검증 필요 |
- **Smoke test**: vLLM serve LLaMA-3-8B → KV inject BER 1e-4 → lm-eval accuracy → KEEPER on/off 비교. log `[KEEPER] migrated_blocks=X` 패턴.

## 9. 관련 연구 및 차별점
| Paper | venue/링크 | Date | 관계 | 차별점 |
|-------|-----------|------|------|------|
| DNN-Defender | [2305.08034](https://arxiv.org/abs/2305.08034) / DAC'24 | 2024 | closest competitor | weight row 만 in-DRAM swap. KEEPER 는 동적 KV + telemetry trigger + heavy-hitter criticality 로 직교 |
| Dynamic KV placement (heterog. mem) | [2508.13231](https://arxiv.org/abs/2508.13231) | 2025 | adjacent (같은 KV placement, 다른 objective) | bandwidth/capacity LP 만. KEEPER 는 reliability 신호 추가 |
| MOAT / CnC-PRAC | [2506.11970](https://arxiv.org/abs/2506.11970) | 2024-25 | orthogonal (PRAC 출처) | PRAC 를 RowHammer 방어에. KEEPER 는 reliability health-map 으로 재활용 |
| RL adaptive mitigation | [2407.16377](https://arxiv.org/abs/2407.16377) / HPDC'24 | 2024 | strong baseline | page-offlining trigger, criticality-blind, capacity 손실 |
| GPUHammer | [2507.08166](https://arxiv.org/abs/2507.08166) / USENIX Sec'25 | 2025 | motivation/threat | 단일 flip→80% drop. KEEPER 의 방어 대상 |

## 10. 리스크 / 한계
| 리스크 | 영향 | 완화 |
|--------|------|-----|
| 실제 HBM3 errors-per-row / PRAC counter 가 userspace 에 미노출 — telemetry→framework 연결이 실HW 에서 불가능할 수 있음 | 높음 — 실배포 불가 시 simulation-only 로 한정 | (a) Ramulator2 로 telemetry 가용성 모델링하여 "신호가 있다면" 효과 입증. (b) NVIDIA NVML ECC error counter(현재 aggregate 수준) 로 coarse-grained fallback. (c) design-only 명시 후 vendor 협력 future work |
| heavy-hitter 분포가 workload 따라 평탄 — criticality term marginal | 중간 — reliability-blind 와 차이 작아짐 | k% sweep + workload별 측정. marginal 시 BALLAST(T2)로 강등(decision tree ②) |
| migration 이 decode critical path 에 jitter | 중간 — p99 latency 위반 | event-driven + async copy, per-window batch migration. always-on 아님 |

## 11. Boundary Probing 5-axis
| Boundary | 작동 영역 | 깨질 조건 | 진단 |
|----------|---------|---------|------|
| Distributional | LLaMA/Qwen/Mistral long-ctx | attention 평탄 모델(uniform) | heavy-hitter 가정 1차 risk |
| Scale | 1B-70B | KV << weight 인 소형/MoE | KV 지배 가정 |
| Adversarial | GPUHammer 류 random flip | 정밀 targeted flip(critical bit 직격) | TRIAD weight-arm 필요 |
| Compositional | quant(W4/INT8)+KV migration | quant scale 손상 시 | scale region 별도 보호 |
| Temporal | 긴 decode 중 hot-set drift | hot-set 급변 | scorer window 적응 |

## 12. Scoring (Phase 2' 최종)
| Reviewer | score | 최고 sub-axis | 최저 sub-axis |
|----------|-------|--------------|--------------|
| novelty | 8.5 | telemetry-as-health-map (CLEAR scoop) | KV migration verb 자체는 placement 류와 표면 유사 |
| differentiation | 8.4 | DNN-Defender(weight) 대비 KV 동적 직교 | InfiniGen 류와 placement 표면 겹침 |
| impact | 8.3 | KV=지배적 성장 소비자 + GPUHammer urgent | 실HW telemetry 노출 불확실 |
| ai-implementation | 8.0 | vLLM/GoldenTransformer 경로 명확 | block table ripple 검증 필요 |
| arch-system-implementation | 7.6 | Ramulator2 PRAC 기성 | telemetry→framework bridge 는 mimic |
| **5-reviewer mean** | **8.16** | — | — |

- **★ 최고**: novelty.telemetry-as-health-map = 8.7 — PRAC/ECS counter 를 reliability placement 신호로 재활용한 선행 0편(scoop 검증 CLEAR). 가장 강한 차별 축.
- **▼ 최저**: arch-system.telemetry-bridge = 7.3 — 실 GPU 가 per-row counter 를 host 에 노출하지 않아 실배포는 simulation mimic 에 의존. (사용자 요구가 simulation-feasible 이므로 본 세션 scope 내에서는 수용 가능.)
- **전문가 합의**: system-robustness ✅ / ai-optimization ✅ / legacy-system ◯ (telemetry bridge 를 design-only 로 명시하는 조건부 Accept).
