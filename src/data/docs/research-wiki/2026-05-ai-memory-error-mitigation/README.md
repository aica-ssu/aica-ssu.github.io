# AI-Workload-Aware Memory Error Mitigation & Page Migration on HBM/GDDR/LPDDR — Session Overview

> CXL 메모리 pool 의 page migration error-mitigation 아이디어를, **testbed 가 필요 없는 on-package 메모리(HBM/GDDR/LPDDR)** 로 옮기고, **최신 AI workload 의 메모리 사용·error-sensitivity skew** 를 1급 신호로 삼아 error mitigation + page migration 을 재설계한다. 모든 idea 는 **simulator (Ramulator2) + AI serving framework error-injection mimic (GoldenTransformer/vLLM)** 의 hybrid 로 single GPU 에서 검증 가능.

---

## 1. Research Questions

### 1.1 전체 주제 RQ
- **RQ-Master**: 최신 LLM serving 의 메모리 access·error-sensitivity skew (weights vs KV-heavy-hitter vs transient activation) 를 신호로 사용하면, **on-package 메모리(HBM/GDDR/LPDDR) 의 error mitigation + page migration 을 capacity 손실·overhead 를 baseline ECC(12.5%) 이내로 유지하면서 silent-corruption 유발 accuracy drop 을 얼마나 줄일 수 있는가?** (CXL testbed 없이 Ramulator2 + framework mimic 으로 검증)

### 1.2 각 idea 별 RQ
#### Tier-1 #1 — KEEPER (ECC-telemetry-driven Critical-KV Migration)
- **RQ-1.1**: HBM3 errors-per-row counter / DDR5 PRAC counter / ECS error log 를 free한 "frame-health map" 으로 재해석하면, heavy-hitter KV block 을 weak frame 으로부터 event-driven migrate 하여 silent KV corruption 의 accuracy 영향을 page-offlining 대비 capacity 손실 없이 방어할 수 있는가?
- **RQ-1.2**: migration trigger 의 criticality term (attention mass) 이 reliability-blind placement 대비 동일 migration budget 에서 더 큰 accuracy 보존을 주는가?

#### Tier-1 #2 — TRIAD (Protect-Migrate-Recompute Trichotomy)
- **RQ-2.1**: tensor class(weight / critical-KV / transient-activation)별로 "가장 cheap-sufficient 한 mitigation" 을 runtime 에 선택(weight→exponent-ECC, critical-KV→migrate, activation→recompute)하면, uniform ECC 대비 protection cost(area/energy)를 얼마나 회수하면서 동일 robustness 를 유지하는가?

#### Tier-1 #3 — ROWPRESS-AI (Inference-Induced Read-Disturbance)
- **RQ-3.1**: long-context decode 의 heavy-hitter KV 반복 read 가 모델 자신의 access pattern 으로 neighbor row 에 read-disturbance(RowPress-류) 를 유발하는 self-aggressor 인가? 그렇다면 hot-KV rotation/guard 로 disturbance-유발 bit-flip 을 얼마나 낮추는가?

#### Tier-2 #1 — TALLY (Block-Syndrome KV Checksum)
- **RQ-T1**: PagedAttention block(16 token) 단위 shared-syndrome checksum 으로 per-element ECC 없이 silent KV corruption 을 검출(→migrate/recompute trigger)하면서 overhead 를 SECDED 12.5% 이내로 유지할 수 있는가?

#### Tier-2 #2 — TEMPER (Error-Robust KV Quantization)
- **RQ-T2**: KIVI/KVQuant 를 random-bit-error-robust 로 calibrate 하면 KV cache 가 보호 없이 weak memory 의 높은 BER 에서 정확도를 유지하여 protection/migration 수요 자체를 줄일 수 있는가?

#### Tier-2 #3 — BALLAST (Reliability-Cost Tiering Term)
- **RQ-T3**: InfiniGen/LMCache 의 KV tiering placement objective 에 per-frame reliability-cost term 을 추가하면, 성능 저하 없이 critical KV 가 healthy frame 으로 자연 배치되는가?

---

## 2. Essential Reading (5편)

1. **GPUHammer: Rowhammer Attacks on GPU Memories are Practical** — Lin, Qu, Saileshwar, [arXiv:2507.08166](https://arxiv.org/abs/2507.08166), **USENIX Security 2025**
   - **읽어야 하는 이유**: GDDR6 단일 bit-flip 이 DNN accuracy 를 최대 80% 떨어뜨림을 실증 — 본 세션 전체의 threat/motivation. on-die ECC 가 단일 flip 을 mask 한다는 관찰이 "selective stronger protection" 논리의 출발점.
   - **분담**: 전 idea 공통 motivation.
2. **Story of Two GPUs (H100/A100 resilience field study)** — Cui, Patke, Nguyen, [arXiv:2503.11901](https://arxiv.org/abs/2503.11901), 2025
   - **읽어야 하는 이유**: H100 HBM3 의 per-GB MTBE 가 A100 대비 24%↓, row-remap/page-offline budget 이 이미 포화 — software/page-level migration 이 두 번째 방어선으로 필요하다는 근거.
   - **분담**: KEEPER / TRIAGE / BALLAST.
3. **DNN-Defender: Victim-Focused In-DRAM Defense** — Zhou, Ahmed, Rakin, [arXiv:2305.08034](https://arxiv.org/abs/2305.08034), **DAC 2024**
   - **읽어야 하는 이유**: critical **weight** row 를 in-DRAM swap 으로 보호하는 closest competitor. 본 세션이 **KV cache(동적)** 로 타겟을 옮긴 차별점을 직접 비교.
   - **분담**: KEEPER / TRIAD.
4. **Domain-Specific ECC for AI Inference + REACH** — Xie et al., [arXiv:2507.02654](https://arxiv.org/abs/2507.02654) / [arXiv:2512.18152](https://arxiv.org/abs/2512.18152), 2025
   - **읽어야 하는 이유**: "AI memory 의 reliability 는 tunable, importance-aware" 라는 핵심 prior art — ECC **strength** 축을 점유. 본 세션은 migration/recompute/placement 축으로 직교화해야 함을 보여주는 scoop-경계.
   - **분담**: TRIAD / TALLY (차별화 필수).
5. **Understanding the Security Benefits and Overheads of PRAC** — Olgun et al., [arXiv:2406.19094](https://arxiv.org/abs/2406.19094), **DRAMSec 2024** (Ramulator2 PRAC open-source)
   - **읽어야 하는 이유**: PRAC per-row counter 의 동작/overhead 와 Ramulator2 구현 진입점. KEEPER 의 "telemetry-as-health-map" 와 ROWPRESS-AI 의 activation-count 분석이 이 구현 위에서 동작.
   - **분담**: KEEPER / ROWPRESS-AI.

> 그 외 30+ 분석 논문은 [../../papers.md](../../papers.md) 참조. idea 별 추가 baseline 은 각 tier 파일 §관련연구.

---

## 3. 연구 개요 + 기존 GAP outline

### 3.1 연구 개요
- **본 세션 도메인**: on-package AI 가속기 메모리(HBM3/3e, GDDR6/7, LPDDR5x) 의 reliability — **CXL 제외** (testbed 회피, R20-γ single-system).
- **타겟 workload**: LLM long-context inference (KV cache 지배적), 양자화 모델 (INT8/W4/FP8), edge/datacenter serving.
- **핵심 측정 metric**: silent-corruption 유발 accuracy/perplexity drop, mitigation overhead (capacity / bandwidth / area / energy), migration count.
- **본 세션 hypothesis**: AI workload 의 메모리 criticality 는 극심하게 skew (weight 의 소수 exponent bit, KV 의 소수 heavy-hitter, recompute-able activation) → mitigation 을 **uniform 이 아니라 criticality-aware** 로 하면 capacity·overhead 를 baseline 이내로 유지하며 robustness 확보.
- **기여 범위**: 새로운 mitigation **mechanism + policy** (remap HW 자체가 아닌 trigger·criticality·placement). 모두 Ramulator2 + framework mimic 으로 single GPU 검증.

### 3.2 기존 연구의 한계 (GAP outline)
- **GAP-1 (KV 부재)**: 모든 weight-protection 연구(DNN-Defender [arXiv:2305.08034](https://arxiv.org/abs/2305.08034), DRAM-Locker [arXiv:2312.09027](https://arxiv.org/abs/2312.09027), REACH)는 **static weight** 만 보호. 동적·지배적·성장하는 **KV cache** 의 error-aware 배치/migration 은 미탐구.
- **GAP-2 (telemetry 미활용)**: 모든 PRAC/ECS 연구(MOAT, CnC-PRAC [arXiv:2506.11970](https://arxiv.org/abs/2506.11970), Citadel [arXiv:2409.15463](https://arxiv.org/abs/2409.15463))는 counter 를 RowHammer 방어·보안에만 사용. **reliability health-map 으로 재활용해 placement 를 구동**한 사례 없음.
- **GAP-3 (self-aggressor 미관찰)**: 모든 read-disturbance×DNN 연구(GPUHammer, DeepHammer)는 **외부 공격자** 가정. LLM 자신의 benign decode access 가 self-induced disturbance aggressor 라는 framing 부재. HBM2 read-disturbance [arXiv:2310.14665](https://arxiv.org/abs/2310.14665) 도 synthetic pattern 만.
- **GAP-4 (perf-only tiering)**: 모든 KV tiering(InfiniGen [arXiv:2406.19707](https://arxiv.org/abs/2406.19707), LMCache [arXiv:2510.09665](https://arxiv.org/abs/2510.09665), [arXiv:2508.13231](https://arxiv.org/abs/2508.13231))은 bandwidth/latency/capacity objective 만 — **reliability term 부재**.
- **GAP-5 (단일 mitigation)**: SHIELD [arXiv:2604.07396](https://arxiv.org/abs/2604.07396) 가 tensor-class BER-asymmetry 를 확립했으나 mitigation 은 refresh-relaxation 1종. **protect/migrate/recompute 를 runtime 에 cost-min 선택하는 selector** 부재.
- **GAP-6 (detection cost)**: ECC-based FT [arXiv:2508.12347](https://arxiv.org/abs/2508.12347) 는 uniform ECC 로 ~48% area. KV block-level **detection-only** shared-syndrome (overhead<<ECC) 로 migration/recompute 를 trigger 하는 경량 검출 부재.

### 3.3 GAP → idea 대응
| GAP | Tier-1 대응 | Tier-2 대응 |
|-----|-------------|-------------|
| GAP-1 KV 부재 | KEEPER (critical-KV migration) | TEMPER (error-robust KV), BALLAST (KV tiering) |
| GAP-2 telemetry 미활용 | KEEPER (telemetry→health-map) | — |
| GAP-3 self-aggressor | ROWPRESS-AI | — |
| GAP-4 perf-only tiering | — | BALLAST (reliability term) |
| GAP-5 단일 mitigation | TRIAD (trichotomy selector) | — |
| GAP-6 detection cost | (KEEPER 의 corruption trigger) | TALLY (block-syndrome checksum) |

---

## 4. Implementation-Priority Decision Tree

```
                         ┌──────────────────────────────────────────────┐
                         │ STAGE 0 (공유 인프라, 2주)                    │
                         │ vLLM PagedAttention + GoldenTransformer 기반   │
                         │ KV/weight bit-flip injector + lm-eval harness  │
                         │ + Ramulator2 PRAC/ECS 로 error-rate 표 추출    │
                         └───────────────┬──────────────────────────────┘
                                         │ (TALLY 검출기는 KEEPER 의 trigger 로 재사용 — dependency edge)
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
 ┌─────────────┐                 ┌──────────────┐                 ┌──────────────┐
 │ KEEPER (T1) │                 │ TRIAD (T1)   │                 │ ROWPRESS (T1)│
 │ critical-KV │                 │ trichotomy   │                 │ self-disturb │
 │ migration   │                 │ selector     │                 │ characterize │
 └──────┬──────┘                 └──────┬───────┘                 └──────┬───────┘
        │ MVP: accuracy 보존 vs           │ MVP: protection cost            │ MVP: Ramulator2 에서
        │ reliability-blind baseline       │ 회수 vs uniform ECC             │ heavy-hitter row 의
        │                                  │                                 │ activation-count 우위
   ┌────┴─────┐                      ┌─────┴─────┐                     ┌─────┴─────┐
   ▼          ▼                      ▼           ▼                     ▼           ▼
 Pass      Below                   Pass        Below                disturb     disturb
(±10%)    (10-30%↓)               (cost-       (회수<10%)            우위 명확   우위 미미/
   │      → criticality term       회수>20%)    → TRIAD 를 KEEPER 의   │         모델 의존
   │        제거하고 BALLAST(T2)    │ → MICRO/   weight-arm 으로 흡수    │         → characterization
   │        로 강등                 │   ISCA     → Tier-2(CADENCE)강등   │         letter(T2)로
   ▼                                ▼                                  ▼         reposition
 Outperform(>20%):               Outperform:                       Outperform:
 telemetry-driven security        recompute-arm 이 activation         최초 benign self-
 (GPUHammer 방어) 추가             SDC 까지 cover → 독립 paper          aggressor 정량화
 → ISCA/MICRO + 보안 venue         pair (TRIAD + TALLY)                → ISCA main track
```

**결과 분기별 액션 (양식 C — idea × branch)**

| Idea | Pass (±10%) | Below (10-30%↓) | Critical fail (>30%↓) | Outperform (>20%↑) |
|------|-------------|-----------------|----------------------|---------------------|
| KEEPER | Tier-1 진행, ASPLOS/MICRO | criticality term 제거 → BALLAST(T2) 로 강등 | telemetry-framework 연결 불가 입증 시 drop | GPUHammer 방어 추가 → ISCA + 보안 venue |
| TRIAD | Tier-1 진행, MICRO/ISCA | recompute-arm 만 남겨 CADENCE(T2)로 | SHIELD 대비 차별화 실패 시 drop | activation SDC 까지 cover → TALLY 와 paper pair |
| ROWPRESS-AI | characterization+mitigation, ISCA | mitigation gain 미미 → characterization-only letter(T2) | disturbance 효과 부재 입증 → drop, KEEPER 에 흡수 | 최초 self-aggressor 정량화 → ISCA main |
| TALLY (T2) | DAC/DATE, RTL 합성 첨부 | overhead>SECDED → shared-syndrome 강화 | 검출률<90% → drop | KEEPER trigger 채택 → Tier-1 승격 spinoff |
| TEMPER (T2) | DAC/DATE | robust calibration 효과<예상 → ablation 축소 | accuracy 회복 실패 → drop | weak-memory 직접 운용 가능 → MLSys |
| BALLAST (T2) | EuroSys-short/DAC | +1 term incremental → KEEPER 에 흡수 | placement 효과 없음 → drop | multi-objective Pareto → EuroSys |

**Inter-idea dependency (shared infrastructure)**
- **STAGE 0 injector + harness** → 6개 idea 전부 공유 (가장 먼저 구축).
- **TALLY 의 block-syndrome 검출기** → KEEPER 의 silent-corruption trigger 로 재사용 (TALLY 먼저 prototype 하면 KEEPER setup 비용 ↓). **Paper pair 후보**.
- **Ramulator2 PRAC/ECS error-rate 표** → KEEPER · ROWPRESS-AI · BALLAST 공유.
- **권고 순서 (1-2 학생, 12-16주)**: STAGE 0(2주) → TALLY 검출기(3주) → KEEPER(4주) → TRIAD(3주) → ROWPRESS-AI 또는 TEMPER/BALLAST(나머지).

**전문가 상호 검증 (R14.4.3)**: system-robustness-expert 와 ai-optimization-expert 가 ②③ branch 에 합의; legacy-system-expert 는 ROWPRESS-AI 의 "disturbance 효과 부재 시 drop" branch 에 reservation 1줄 — "Ramulator2 의 disturbance 모델이 parameterized 이므로 '부재 입증' 보다 'literature BER 모델 하 conditional gain' 으로 재포지셔닝 권장".

---

## 5. Tier-1 Top 3

| Rank | Title | Score | 링크 |
|------|-------|-------|------|
| 🥇 | **ECC-Telemetry-Driven Critical-KV Cache Migration (KEEPER)** | 8.16 | [tier1/01-keeper.md](tier1/01-keeper.md) |
| 🥈 | **Protect–Migrate–Recompute Trichotomy Controller (TRIAD)** | 7.96 | [tier1/02-triad.md](tier1/02-triad.md) |
| 🥉 | **Inference-Induced Read-Disturbance Characterization & Mitigation (ROWPRESS-AI)** | 7.86 | [tier1/03-rowpress-ai.md](tier1/03-rowpress-ai.md) |

### 🥇 KEEPER — Contribution
- **(a) Mechanism 정성적 benefit**:
  - M1 Criticality scorer — attention-mass(H2O 재사용)로 KV block 을 heavy-hitter/cold 로 ranking, <3% FLOPs 로 accuracy-critical KV 식별.
  - M2 Frame-health map — HBM3 errors-per-row / DDR5 PRAC counter / ECS log 를 **free reliability 신호로 재해석** (별도 측정 비용 0), weak-frame 목록 유지.
  - M3 Migration controller — PagedAttention block 단위로 critical-KV 를 weak frame 으로부터 event-driven copy, cold-KV 는 weak frame 으로 park (offlining 과 달리 capacity 보존).
- **(b) Closest competitor**: DNN-Defender ([arXiv:2305.08034](https://arxiv.org/abs/2305.08034), DAC'24) — critical **weight** row swap. KEEPER 는 **동적 KV** + **telemetry-driven trigger** + **heavy-hitter criticality** 로 직교. 성능-only KV placement ([arXiv:2508.13231](https://arxiv.org/abs/2508.13231)) 는 reliability term 부재.
- **(c) 예상 gain**:
  | 지표 | Baseline | 본 idea | 개선 |
  |------|---------|---------|------|
  | [Robustness] BER 1e-5 하 long-context accuracy drop | -8~-15%(reliability-blind) | -1~-3% | 5-8× |
  | [Memory eff.] mitigation 당 capacity 손실 | page-offlining: row 단위 retire | 0 (migration) | capacity 보존 |
  | [Performance] migration bandwidth overhead | — | <1-2% (event-driven, rare) | acceptable |
- **(d) Tier 강등 risk**: criticality term 이 reliability-blind migration 대비 marginal(<10%) 이면 BALLAST(T2)로 강등.
- **(e) Outperform 가능성**: telemetry-driven 이 GPUHammer류 공격 row 도 회피 → reliability+security dual-use 로 ISCA+보안 venue.
- **(f) Implementation envelope**: vLLM PagedAttention fork ~600 LoC + GoldenTransformer injector + Ramulator2 PRAC error-rate 표. RTX 5090 single host fit.

### 🥈 TRIAD — Contribution
- **(a) Mechanism 정성적 benefit**:
  - M1 Region classifier — 메모리 영역을 tensor class(weight / critical-KV / transient-activation)로 tagging.
  - M2 Cheapest-sufficient selector — class 별 최저비용 충분 mitigation 선택: weight→exponent/MSB-only ECC, critical-KV→KEEPER migration, activation→recompute-on-error (Checkmate-류 rematerialization [arXiv:1910.02653](https://arxiv.org/abs/1910.02653)).
  - M3 Cost model — 각 arm 의 area/energy/latency 를 cost 로 두고 robustness 제약 하 min.
- **(b) Closest competitor**: SHIELD ([arXiv:2604.07396](https://arxiv.org/abs/2604.07396)) — tensor-class asymmetry 를 refresh-relaxation **1종** mitigation 으로만. TRIAD 는 **3-arm runtime selector**. REACH/Domain-ECC 는 ECC **strength** 만 조절.
- **(c) 예상 gain**:
  | 지표 | Baseline | 본 idea | 개선 |
  |------|---------|---------|------|
  | [Memory eff.] protection area/overhead | uniform ECC ~48%(ECC-based FT) | ~12-18% | 2.5-4× 회수 |
  | [Energy] mitigation energy/token | uniform scrub+ECC | -20~-35% | |
  | [Robustness] critical-region UE 방어 | 동일 | 동일 유지 | iso-robustness |
- **(d) 강등 risk**: 회수<10% 이면 recompute-arm 만 남긴 CADENCE(T2).
- **(e) Outperform**: activation recompute-arm 이 training SDC([arXiv:2604.00726](https://arxiv.org/abs/2604.00726))까지 cover → 독립 paper.
- **(f) Envelope**: 정책 controller (Python) + GoldenTransformer 3-arm mimic + exponent-ECC Verilog(<10K gate) 합성.

### 🥉 ROWPRESS-AI — Contribution
- **(a) Mechanism 정성적 benefit**:
  - M1 Access-trace → Ramulator2 per-row activation/open-count 추출 (decode 중 heavy-hitter KV row 반복 read).
  - M2 Self-disturbance 정량화 — heavy-hitter row 의 누적 activation 이 RowPress/read-disturbance threshold 에 근접하는지 literature BER 모델로 평가.
  - M3 완화 — hot-KV block periodic rotation 또는 victim-row guard insertion.
- **(b) Closest competitor**: GPUHammer ([arXiv:2507.08166](https://arxiv.org/abs/2507.08166)) / HBM2 read-disturbance ([arXiv:2310.14665](https://arxiv.org/abs/2310.14665)) — 모두 **외부 공격자/synthetic**. ROWPRESS-AI 는 **benign inference self-aggressor** framing 이 최초.
- **(c) 예상 gain**:
  | 지표 | Baseline | 본 idea | 개선 |
  |------|---------|---------|------|
  | [Robustness] disturbance-유발 flip rate (모델 하) | rotation 없음 | rotation 적용 | literature BER 모델 하 조건부 |
  | [Robustness] heavy-hitter row 최대 activation | 수십만/refresh window | rotation 후 균등화 | quantified |
- **(d) 강등 risk**: disturbance 효과가 모델 의존적/미미 → characterization-only letter(T2).
- **(e) Outperform**: 최초 self-aggressor 정량화면 ISCA main track characterization.
- **(f) Envelope**: Ramulator2 activation-count plugin + vLLM access trace. **단 실측 disturbance BER 불가 → literature 모델 가정 명시 의무**.

---

## 6. Tier-2 독립 Top 3

| Rank | Title | Score | 링크 |
|------|-------|-------|------|
| T1 | **Block-Syndrome KV Checksum for Silent-Corruption Detection (TALLY)** | 7.78 | [tier2/01-tally.md](tier2/01-tally.md) |
| T2 | **Error-Robust KV Cache Quantization Co-Design (TEMPER)** | 7.64 | [tier2/02-temper.md](tier2/02-temper.md) |
| T3 | **Reliability-Cost Term for KV Tiering Placement (BALLAST)** | 7.56 | [tier2/03-ballast.md](tier2/03-ballast.md) |

### TALLY — Contribution
- **(a)** PagedAttention block(16 token) 단위 **shared-syndrome checksum** — per-element ECC 없이 silent KV corruption 검출, migration/recompute trigger.
- **(b)** vs ECC-based FT ([arXiv:2508.12347](https://arxiv.org/abs/2508.12347)) — uniform per-element ECC(~48% area). TALLY 는 block 공유 syndrome **detection-only** → overhead 1자리 수 작음. vs SDC-in-training ([arXiv:2604.00726](https://arxiv.org/abs/2604.00726)) — training param/step 대상, KV inference 아님.
- **(c)** | 지표 | Baseline | 본 idea | 개선 | : [Memory eff.] overhead | SECDED 12.5% | <2-4% (block-shared) | 3-6× | ; [Robustness] 검출률 | — | >95% single-flip | — |.
- **(d)** 검출률<90% 시 syndrome bit 증설(overhead↑) trade-off.
- **(e)** KEEPER 가 trigger 로 채택 → Tier-1 spinoff.
- **(f)** Verilog checksum unit <10K gate, Yosys/Sky130 합성 + GoldenTransformer KV inject.

### TEMPER — Contribution
- **(a)** KIVI/KVQuant 를 **random-bit-error-robust** 로 calibrate (RANDBET-style) → KV 가 보호 없이 높은 BER 견딤.
- **(b)** vs RANDBET ([arXiv:2006.13977](https://arxiv.org/abs/2006.13977)) — **weight/SRAM voltage** 대상. TEMPER 는 **KV cache + HBM/GDDR BER**. vs RESQ ([arXiv:2603.15413](https://arxiv.org/abs/2603.15413)) — quant+harden 이나 KV 특화 아님.
- **(c)** [Robustness] BER 1e-4 하 perplexity 상승 | robust calibration 없음: +large | 있음: +<0.3 | ; [Memory eff.] 보호 overhead 0 (algorithm-only).
- **(d)** robust calibration 의 clean accuracy 손실>1% 시 ablation 축소.
- **(e)** weak-memory 직접 운용 가능하면 MLSys.
- **(f)** algorithm-only — vLLM KIVI fork + lm-eval, GPU 직접 측정 (R55 safe).

### BALLAST — Contribution
- **(a)** InfiniGen/LMCache placement objective 에 **per-frame reliability-cost term** 추가 → critical KV 자연히 healthy frame.
- **(b)** vs InfiniGen ([arXiv:2406.19707](https://arxiv.org/abs/2406.19707))/2508.13231 — perf-only. BALLAST 는 reliability term 추가. (incremental — "+1 term" critique 대비 multi-objective Pareto 분석 필요.)
- **(c)** [Robustness] critical-KV-on-weak-frame 비율 | 무대응 | -대폭 | ; [Performance] placement 변경 latency overhead <1%.
- **(d)** +1 term 으로만 보이면 KEEPER 에 흡수.
- **(e)** multi-objective Pareto 면 EuroSys.
- **(f)** LMCache connector fork ~300 LoC.

---

## 7. 미선정 아이디어 요약
- **AEGIS (SR5)** — random-fault + adversarial critical-bit 통합 placement. RESQ([arXiv:2603.15413](https://arxiv.org/abs/2603.15413)) 와 unified 목표 겹침(ADJACENT). KEEPER 의 Outperform branch(보안 dual-use)로 흡수.
- **RELAY (LS2)** — GPU in-kernel error-triggered remap. kernel-level remap 구현·검증 risk 과다, simulation-only. Tier-2 future work.
- **MASON (LS3)** — fault-geometry-aware KV layout. RAMPART([arXiv:2310.16354](https://arxiv.org/abs/2310.16354)) device-confinement 와 인접, KEEPER 의 placement 에 부분 흡수.
- **TRIAGE (SR3)** — criticality migrate-vs-offline RL trigger. KEEPER 의 migration controller 와 trigger 중복 → 흡수. RL 정책은 KEEPER 확장으로.
- 상세: [unselected.md](unselected.md).

## 8. 참고 / 관련 자료
- 상세 Phase 로그 (Phase 1→2→1'→2'→1'') 및 Phase 1 idea pool staging 은 ideation harness 의 내부 세션 기록에 보관 (homepage 미게시).
- 관련 이전 세션: 없음 (메모리 reliability 주제는 본 wiki 최초).

## 9. 약어 / 핵심 용어 풀이

### 9.1 도메인 약어
- **PRAC** — Per-Row Activation Counting (DDR5 RowHammer 방어용 per-row counter; 본 세션은 reliability health 신호로 재해석).
- **ECS** — Error Check & Scrub (HBM3/DDR5 의 주기적 in-field error 검사·정정).
- **RFM** — Refresh Management (JEDEC RowHammer 완화 refresh).
- **ODECC / S-ECC** — On-Die ECC / System ECC.
- **heavy-hitter** — attention mass 대부분을 차지하는 소수 KV token/block (H2O).
- **KV cache** — transformer decode 의 key/value 저장, write-once/read-many, HBM/GDDR 지배적 소비자.
- **RowPress** — row 를 장시간 open 한 채 반복 access 시 발생하는 read-disturbance 변종.
- **SDC** — Silent Data Corruption (검출되지 않은 데이터 손상).
- **MTBE** — Mean Time Between Errors.

### 9.2 Polysemous term
- **migration** (본 세션) — 물리 메모리 row/frame 간 데이터 이동(reliability 회피). LLM "request migration" 과 무관.
- **page** (본 세션) — PagedAttention KV block 또는 DRAM row/page-frame. OS virtual page 와 문맥 구분.

### 9.3 Idea Metaphor
- **KEEPER** — weak frame 으로부터 critical KV 를 "지킨다(keep away)".
- **TRIAD** — protect/migrate/recompute 3-arm.
- **ROWPRESS-AI** — RowPress 를 AI workload self-aggressor 로.
- **TALLY** — block 단위로 무결성을 "집계(tally)".
- **TEMPER** — error 에 강하도록 KV 를 "단련(temper)".
- **BALLAST** — placement 의 reliability "평형추".
