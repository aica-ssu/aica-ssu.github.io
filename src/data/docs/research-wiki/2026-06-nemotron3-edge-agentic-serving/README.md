# Edge Multi-Agent Hybrid-SSM/MoE 서빙 최적화 — Nemotron 3 Nano/Super 분석 기반 Ideation (2026-06-04)

> NVIDIA Nemotron 3 Nano (30B-A3B) / Super (120B-A12B) tech report를 분석해, **hybrid Mamba-MoE 모델을 edge 디바이스에서 여러 agent로 동시 서빙할 때의 병목 이동 (KV cache → expert read·SSM state)** 을 출발점으로 Tier-1 3편 + Tier-2 3편의 연구 아이디어를 도출한 세션. Phase 1 15개 → 리뷰 5인 + 상호리뷰 → 정제 12개 → Top-M 6 + paper pair 1쌍.

- **사용자 쿼리**: 두 모델 아키텍처 분석 + 2026 agentic AI 최적화 논문 탐색 + "Super 대비 Nano 특이점 × agent 수 증가 × edge 리소스 특성" ideation.
- **중점 축**: agent-count scaling / LPDDR·thermal 자원 천장 / SSM state·expert 객체의 1급화. **의도적 제외 축**: training-side 최적화 (inference-only scope), multi-node 분산 (single-system 규율 R20-γ), LatentMoE 재설계 (모델 수정은 scope 밖 — serving-side만), DVFS 단독·L2 pinning 단독 (이전 세션 소진).
- **전문가**: ai-optimization (primary) + legacy-system + algorithm | **리뷰어**: novelty / differentiation+similarity / impact / ai-implementation (vLLM clone 검증) / arch-system-implementation + 전문가 상호리뷰 패널 + Phase 2' 통합 재리뷰.

---

## 1. Research Questions

**Master RQ**:
- **RQ-1 (agent-count)**: 단일 edge 디바이스에서 hybrid SSM-MoE 모델을 공유하는 agent 수 N이 증가할 때, throughput cliff가 발생하는 임계 N*를 **배치 전에 closed-form으로 예측**할 수 있는가? (목표: 예측 오차 ±1 agent)
- **RQ-2 (자원 객체)**: KV cache가 사실상 상수인 모델 클래스에서, agent 컨텍스트 전환의 1급 객체를 SSM state (48MB/agent)로 바꾸면 동시 agent 수와 전환 latency가 얼마나 개선되는가? (목표: SLO 유지 동시 agent +30~50%)
- **RQ-3 (압축의 숨은 비용)**: SSM state 양자화가 유발하는 출력 길이 증가 (verbosity +40%, Super 실측)는 어떤 조건에서 메모리 절감 이득을 역전시키며, 그 인과는 bias drift인가 noise인가?

**Idea별 RQ** (1줄씩):
- TIDEMARK: expert working-set union의 성장 곡선으로 N*를 ±1 정확도로 예측 가능한가?
- MOORING: tool-call 윈도우 (실행시간 2-29%) 안에 state swap을 완전히 숨길 수 있는가?
- SPENDTHRIFT: state 양자화 오차 예산을 throughput 예산으로 환산하는 닫힌 식이 성립하는가?
- KILN: thermal 정상상태에서 "sustainable agent count" 곡선은 cold-start 대비 얼마나 낮은가 (가설 −40~60%)?
- QUARRY: expert read의 DRAM row-buffer hit율을 layout으로 +20%p 올릴 잠재 이득이 있는가 (trace-driven)?
- BLOAT: verbosity 증폭은 stochastic rounding으로 bias를 제거하면 사라지는가 (인과 식별)?

## 2. Essential Reading (5편 한정)

| Paper | 왜 읽어야 하는가 | 어느 idea의 baseline |
|---|---|---|
| [arXiv:2605.26297](https://arxiv.org/abs/2605.26297) Agentic AI Workload Characteristics | agent 실행의 91-98.6%가 decode, prefix reuse 84.6-99.5%, turn 12-62 — 본 세션 전체 workload evidence의 원천 | TIDEMARK / MOORING / KILN |
| [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) EdgeReasoning [IISWC 2025, NVIDIA] | Jetson decode가 DRAM util 20-60%에서 memory-bound라는 정량 baseline — expert read 1.84GB/token 상한 계산과 정합 | TIDEMARK / QUARRY / KILN |
| [arXiv:2510.18586](https://arxiv.org/abs/2510.18586) Tokencake | MOORING의 closest competitor (concurrent 50-60%) — agent-aware tool-stall KV offload. **state-vs-KV 차별화를 직접 확인해야 할 1순위** | MOORING |
| [arXiv:2502.05370](https://arxiv.org/abs/2502.05370) FineMoE [EuroSys 2026] | expert cache의 input-conditioned 인접 경계 — HEARTH 미선정 사유이자 TIDEMARK/QUARRY가 밟지 말아야 할 영역 | TIDEMARK / (HEARTH 부활 시) |
| [arXiv:2512.20848](https://arxiv.org/abs/2512.20848) Nemotron 3 Nano | 대상 모델 그 자체 — §4.2 selective FP8과 Figure 2 레이어 패턴이 모든 idea의 전제 | 전체 |

## 3. 연구 개요 + GAP Outline

### 3.1 개요 — Nano vs Super 아키텍처 특이점

| 축 | Nano 30B-A3B | Super 120B-A12B | 서빙 함의 |
|---|---|---|---|
| 레이어 | 52 = Mamba-2 23 / MoE 23 / **Attention 6** (KV-head 2) | 88, [mixer+LatentMoE] 쌍, attention 소수 anchor | KV cache가 사실상 상수 → per-agent 메모리 기울기 평탄 |
| MoE | **standard** 128 expert top-6 + shared 2 (full-d routing) | **LatentMoE** 512 expert top-22, d/ℓ=4 압축 | Nano는 expert read가 LPDDR 직격 (1.84GB/token) — Super만 구조적 완화 |
| Spec decoding | 없음 (모델에 MTP head 부재) | MTP 2층 native | Nano 외부 draft 필요하나 Mamba Drafters가 선점 |
| Pretrain 정밀도 | BF16 (FP8는 PTQ, selective: attention 6층+선행 Mamba+Conv1D=BF16) | NVFP4 25T 전구간 (4.75-bit AutoQuantize) | Nano는 균일 양자화 불가 — precision island가 thermal/메모리 비대칭 유발 |
| State quant | (해당 없음 — BF16) | **SSM state FP16 naive cast 시 verbosity +40%** (Eq3) | state 압축의 진짜 비용은 출력 길이 폭발 |
| Router | RL 동안 freeze (expert bias만 업데이트) | 동일 계열 | expert 분포가 정적 → 예측/배치 모델링 가능 |

- 정량 backbone (재검산): expert 9.98M param (10MB FP8) → population ~30GB, 모델 FP8 32GB. **AGX Orin 64GB full-resident / RTX 5090 32GB 경계 / Orin NX 16GB는 W4 (실측 22GB)도 불가**. SSM state 48MB/agent. shared expert read = per-token 25%.
- 치명 발견 4종: llama.cpp hybrid 로딩 깨짐 (#20570) → vLLM 단일 path / ncu가 Orin iGPU 미지원 → 실기 DRAM 카운터 측정 불가 / RTX (1.79TB/s)는 edge BW-bound proxy 불가 / Thor·DGX Spark 273GB/s (용량≠대역폭).

### 3.2 GAP

- **G1**: multi-agent KV 관리 연구 (KVFlow [arXiv:2507.07400](https://arxiv.org/abs/2507.07400), Continuum [arXiv:2511.02230](https://arxiv.org/abs/2511.02230), TokenDance [arXiv:2604.03143](https://arxiv.org/abs/2604.03143))는 전부 attention-KV 가정 → hybrid의 state/expert 객체에 부적용.
- **G2**: agent-count scaling 연구는 LPDDR BW·thermal 정상상태를 독립 변수로 안 넣음 (Sustained-Load [arXiv:2603.23640](https://arxiv.org/abs/2603.23640)의 "2-iteration 반토막"이 N-agent에서 어떻게 가속되는지 미측정).
- **G3**: expert cache 정책 (FlashMoE [arXiv:2601.17063](https://arxiv.org/abs/2601.17063))이 다중 agent round-robin의 recency 오염을 자인 — cliff의 사전 예측 부재.
- **G4**: SSM state 압축의 정확도 외 비용 (출력 길이)이 측정된 적 없음 (Quamba [arXiv:2410.13229](https://arxiv.org/abs/2410.13229)는 정확도만).
- **G5**: 단일 edge 디바이스 multi-agent 서빙을 정조준한 시스템 연구가 사실상 1편 ([arXiv:2603.04428](https://arxiv.org/abs/2603.04428)) — datacenter 가정 (throughput)을 edge 비용 모델 (prefill latency·RAM·재로딩)로 재정립 공백.

### 3.3 GAP→Idea Mapping

| GAP | Idea | 메커니즘 1줄 |
|---|---|---|
| G3+G2 | TIDEMARK (T1-1) | Markov union-growth로 N* closed-form 예측 + admission |
| G1+G5 | MOORING (T1-2) | SSM state 1급 swap 객체화 + tool-window 은닉 |
| G4 | SPENDTHRIFT (T1-3) × BLOAT (T2-3) | bias-drift→verbosity 이론 + 인과 측정 pair |
| G2 | KILN (T2-1) | thermal 정상상태 sustainable agent-count 곡선 |
| G3+G1 | QUARRY (T2-2) | expert read의 row-buffer locality trace-driven 정량화 |

## 4. Implementation-Priority Decision Tree

```
[Step 1] KILN — 측정 인프라 구축 (외부 전력계+tegrastats, 2-4주, 최저 위험)
  │  지표: 정상상태 vs cold throughput gap, sustainable agent-count 곡선
  ├─ Pass(곡선 확보) ──────────────► Track-A 전체의 공통 baseline 확보 + DATE/ISLPED letter 제출
  ├─ Below(gap < 15%로 미미) ─────► thermal 축 약화 → KILN은 IISWC poster로 축소, GOVERNOR 영구 drop
  └─ Outperform(island 기여 분리 성공) ► GOVERNOR 부활 검토 (predictive thermal actuator)
        ▼
[Step 2] BLOAT — verbosity 인과 식별 (det vs SR 대조, 3-5주)
  │  지표: verbosity Δ% (det-INT8 vs SR-INT8 vs BF16), useful-tok/s cross-over
  ├─ Pass(bias 인과 확인: SR로 verbosity 소멸) ► SPENDTHRIFT 이론 go + IISWC/CAL letter
  ├─ Below(증폭 < 10%: Nano-급에서 미약) ────► SPENDTHRIFT를 "bound가 tight한 조건 규명"으로 재포지셔닝
  └─ Critical(verbosity가 SR로도 잔존) ──────► SPENDTHRIFT 정리 전제 붕괴 → drop, BLOAT 단독 (반례 보고 letter — Q-Overthink 계열과 직접 대화)
        ▼
[Step 3] MOORING MVP — state swap 비용 실측 gate (4-6주)
  │  지표: 48MB state swap latency (목표 < 10ms), tool-window 은닉률 (목표 > 90%)
  ├─ Pass ────────► full 구현 (M1-M3) → MLSys/EuroSys
  ├─ Below(swap이 KV reload 대비 < 5× 우위) ► Tier-2 측정 letter로 강등 (전환 비대칭 비용)
  └─ Outperform(동시 agent 2× 이상) ────────► OSDI/SOSP 상향 검토
        ▼
[Step 4] TIDEMARK — N* 검증 (MOORING 인프라 재사용, 4-6주)
  │  지표: N* 예측 오차 (목표 ±1 agent), admission 후 tok/s 개선 (+25~40%)
  ├─ Pass ───► MLSys 제출 (MOORING과 통합 평가 시나리오 포함)
  ├─ Below(오차 ±2-3) ► Markov 차수 상향 또는 empirical 보정 항 추가 후 재시도 1회
  └─ Critical(union growth가 모델과 질적으로 다름) ► 측정 데이터로 CAL letter 전환
        ▼
[Step 5] SPENDTHRIFT — 이론+실증 (BLOAT gate 통과 전제, 6-8주)
  │  지표: bound tightness (실측 대비 ≤ 2×), allocator 이득 (출력 −25~35%)
  └─ 분기: Pass → ICML/NeurIPS / Below → MLSys 시스템 논문으로 전환 / SR 오버헤드 > 5% → allocator만 발표
        ▼
[Step 6] QUARRY — trace-driven (학기 후반 병행, Ramulator2/DRAMSim3)
  └─ row-buffer hit 잠재 이득 < 10% 시 negative result 포함 letter (측정 자체가 기여)
```

**Dependency edges**: KILN→{GOVERNOR 부활, 모든 Track-A의 thermal baseline} / BLOAT→SPENDTHRIFT (go/no-go gate) / MOORING↔TIDEMARK (state pool 인프라 + admission 공식 상호 인용) / QUARRY↔CORAL (C[i,j] graph 공유, CORAL 부활 조건).

**전문가 상호 검증 trace** (R14.4.3):

| 분기 | ai-optimization | legacy-system | algorithm |
|---|---|---|---|
| Step 1 KILN 우선 | 동의 — 인프라 hub | 동의 — 최저 위험 | 동의 (reservation: 곡선 함수형 가정 필요) |
| Step 2 BLOAT gate | 동의 | 동의 (reservation: 4B proxy 대표성) | 동의 — 인과 식별이 이론 전제 |
| Step 3 MOORING < 10ms 임계 | 동의 | reservation: UMA 경합 시 305MB/s 실효까지 악화 가능 → 임계 20ms 완화안 | 동의 |
| Step 4 TIDEMARK ±1 | 동의 | 동의 | reservation: diversity 분포 변동 시 ±2 현실적 |

## 5. Tier-1 Top 3

| # | Idea | Avg | Venue | 파일 |
|---|---|---|---|---|
| 1 | TIDEMARK — Critical Agent-Count via Working-Set Union Growth | 7.6 | MLSys 2027 | [tier1/01-tidemark.md](tier1/01-tidemark.md) |
| 2 | MOORING — LPDDR-Warm Tiered SSM-State Pool + Tool-Window Swap | 7.8 | MLSys/EuroSys 2027 | [tier1/02-mooring.md](tier1/02-mooring.md) |
| 3 | SPENDTHRIFT — Spectral Verbosity Budget for State-Quantized Decoding | 7.4 | ICML/NeurIPS 2027 | [tier1/03-spendthrift.md](tier1/03-spendthrift.md) |

### TIDEMARK Contribution
- **(a) M1 정성 benefit**: 2-state Markov union-growth 모델 — expert 분포의 token 자기상관을 반영해 i.i.d. coupon-collector의 과소예측을 보정, **사후 완화가 아닌 배치 전 admission 공식** 제공.
- **(a) M2 정성 benefit**: diversity-aware admission sidecar — vLLM 외부 프로세스로 router freeze가 보장하는 분포 정상성을 활용, 코어 수정 없이 배포 가능.
- **(b) Closest competitor**: Concur [arXiv:2601.22705](https://arxiv.org/abs/2601.22705) — 사후 완화·attention-KV 가정·예측 부재. TIDEMARK는 hybrid의 expert union 객체 + closed-form 예측이 unique.
- **(c) 정량 gain**:

| 지표 | Baseline | 본 idea | 개선 |
|---|---|---|---|
| N* 예측 오차 | 없음 (사후 발견) | ±1 agent | 신규 능력 |
| cliff 이후 tok/s | 붕괴 방치 | +25~40% (admission) | 보수 추정 |
- **(d) Tier 강등 risk**: Markov 보정으로도 오차 ±2-3이면 측정 letter (CAL) 전환 (§4 Step 4 분기).
- **(e) Outperform 가능성**: N* 공식이 hybrid 전 클래스 (Qwen3-Next/Granite 4.0/Falcon-H1)에서 일반화 입증 시 capacity-planning 표준 → 인용 허브.
- **(f) 구현 envelope**: vLLM `RoutedExpertsCapturer` 프로파일러 + 외부 sidecar ~1.5K LoC, AGX Orin 64GB 단일.

### MOORING Contribution
- **(a) M1**: LPDDR-warm tiered state pool — agent state (48MB)를 GPU-resident/LPDDR-warm 2-tier로 계류, host round-trip 제거 (host-sync-free swap은 KV reload ~500ms 대비 수십× 저렴).
- **(a) M2**: tool-window swap scheduler — tool 실행 (E2E의 2-29%) 동안 state를 선제 demote/promote, 신호는 orchestrator↔vLLM 경계에서 추출 (스케줄러 침습 없음).
- **(a) M3**: state-vs-KV 비대칭 admission — TIDEMARK N* 공식을 state 축으로 확장 (cross-share).
- **(b) Closest**: Tokencake [arXiv:2510.18586](https://arxiv.org/abs/2510.18586) (concurrent 50-60%) — agent-aware지만 attention-KV offload. 관리 객체가 다르면 (state는 prefix-restore 불가·소형·상수) 최적 정책이 다름을 정면 비교.
- **(c) 정량 gain**:

| 지표 | Baseline (KV-중심 evict/reload) | 본 idea | 개선 |
|---|---|---|---|
| agent 전환 latency | ~500ms (KV reload 류) | < 10ms (48MB swap) | 수십× |
| SLO 유지 동시 agent | state-resident 한계 | +30~50% | 보수 추정 |
- **(d) 강등 risk**: swap 우위가 5× 미만 실측 시 전환-비용 측정 letter로 (§4 Step 3).
- **(e) Outperform**: 동시 agent 2× 입증 시 OSDI/SOSP 상향.
- **(f) envelope**: vLLM `MambaSpec`/`MambaManager` 확장 ~2-3K LoC, AGX Orin 64GB 또는 Thor.

### SPENDTHRIFT Contribution
- **(a) M1**: bias-drift→verbosity closed-form 정리 — state 양자화 오차의 bias 성분이 ∏A 누적으로 decode entropy를 올려 출력 길이를 늘린다는 인과를 수식화 ("오차 예산 = throughput 예산").
- **(a) M2**: ρ-기반 spectral mixed-precision allocator — 층별 recurrence 감쇠율로 state 비트를 배분 (SR 커널 공유, BLOAT pair).
- **(b) Closest**: Q-Overthink [arXiv:2606.00206](https://arxiv.org/abs/2606.00206) — weight/attention quant의 overthinking (SSM recurrence 미취급, 반대 결론) → "SSM은 다르다"가 검증 가능 가설.
- **(c) 정량 gain**:

| 지표 | Baseline (naive INT8 state) | 본 idea | 개선 |
|---|---|---|---|
| 출력 토큰 수 | +40% (Super 실측 류) | −25~35% (할당 최적화) | 보수 |
| 실효 tok/s | 역전 위험 | +18~30% | 동일 accuracy |
- **(d) 강등 risk**: BLOAT gate에서 SR로도 verbosity 잔존 시 정리 전제 붕괴 → drop (§4 Step 2 Critical).
- **(e) Outperform**: bound tight (≤2×) 입증 시 hybrid 양자화 설계의 표준 수식 → ICML spotlight 급.
- **(f) envelope**: vLLM per-module quant + `MambaSpec.dtype` 주입 + Philox SR 커널 ~1K LoC, RTX 5090/AGX Orin.

## 6. Tier-2 독립 Top 3

| # | Idea | Avg | Venue | 파일 |
|---|---|---|---|---|
| 1 | KILN — Thermal-Bounded Sustainable Agent-Count | 6.8 | DATE/ISLPED/IISWC | [tier2/01-kiln.md](tier2/01-kiln.md) |
| 2 | QUARRY — Row-Buffer Locality of Expert Read | 7.1 | DAC/DATE/IEEE CAL | [tier2/02-quarry.md](tier2/02-quarry.md) |
| 3 | BLOAT — Verbosity Cost of SSM-State Quantization | 6.6 | IISWC/ISPASS/CAL | [tier2/03-bloat.md](tier2/03-bloat.md) |

### KILN Contribution
- **(a) 정성 benefit**: cold-start peak가 아닌 **thermal 정상상태의 sustainable agent-count 곡선**을 최초 측정 — agent 수 × precision island × expert-resident 비율 × nvpmodel 4축 sweep, 회귀로 island vs expert-DMA 열 기여 분리.
- **(b) Baseline**: Sustained-Load [arXiv:2603.23640](https://arxiv.org/abs/2603.23640) (단일 추론, multi-agent/island 축 부재) + EdgeReasoning [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) (thermal 미통제).
- **(c) 정량**: 정상상태 vs cold gap 가설 −40~60% / sustainable N 곡선 자체가 신규 데이터.
- **(d) 왜 Tier-2 only**: mechanism 제안 없는 측정 letter — 단 Track-A 3편의 baseline 인프라 hub (승격 경로: GOVERNOR 결합 시 제어 논문).
- **(f) envelope**: vLLM 무수정 + tegrastats/jtop/INA3221 + 외부 전력계, AGX Orin 64GB + Orin Nano 8GB (4B proxy).

### QUARRY Contribution
- **(a)**: vLLM `fused_moe` access trace → Ramulator2/DRAMSim3 standalone으로 expert read의 row-buffer hit 잠재 이득 정량화 (frozen router의 정적 co-activation 활용 layout 가정 비교).
- **(b)**: RoMe [HPCA 2026] (CPU 일반 워크로드 row-buffer; MoE/LPDDR gap) + MoE CXL-NDP [arXiv:2512.04476](https://arxiv.org/abs/2512.04476).
- **(c) 정량**: row-buffer hit +20%p 가설 / 유효 BW +15~25% 잠재 (시뮬레이션 한정 명시).
- **(d) 왜 Tier-2 only**: 실기 검증 경로 부재 (ncu Orin 미지원 + 물리주소 user-space 제어 불가) — 강등이 ai-impl 4→7/arch 3→8 구제 (Δavg +1.4). 승격 경로: 벤더 협업으로 실기 layout 제어 확보 시.
- **(f) envelope**: trace 추출 (AGX Orin) + Ramulator2/DRAMSim3 (워크스테이션), batch-1 decode 한정.

### BLOAT Contribution
- **(a)**: deterministic vs stochastic rounding (Philox) 대조로 **verbosity 증폭의 인과 식별** (bias drift vs noise) + useful-tok/s cross-over 지점 측정 — "메모리를 줄였는데 느려지는" 역전 조건의 최초 보고.
- **(b)**: Quamba [arXiv:2410.13229](https://arxiv.org/abs/2410.13229) (정확도만) + Super 보고서 (+40%, 이질 모델·재현 불가).
- **(c) 정량**: verbosity Δ% 곡선 (비트 폭별) + cross-over 비트 폭 = 신규 데이터.
- **(d) 왜 Tier-2 only**: 측정 letter — 단 SPENDTHRIFT의 go/no-go gate (paper pair, vLLM hook 공유로 구현 1회·논문 2편).
- **(f) envelope**: vLLM `MambaSpec.dtype` + SR 커널, AGX Orin (full) + Orin Nano 8GB (4B proxy).

## 7. 미선정 (8건)

상세: [unselected.md](unselected.md)

| ID | Score | 미선정 사유 1줄 | 재방문 조건 |
|---|---|---|---|
| HEARTH | 7.3 | FineMoE [arXiv:2502.05370](https://arxiv.org/abs/2502.05370) 인접 35-40% + Tier-1 diversity | agent별 expert 분포 분기 실측 유의 시 최우선 부활 |
| TOLLGATE | 7.1 | BW 직접 노브 부재 (근사 제어로 약화) | TIDEMARK 실증 후 2계층 stack 논문 |
| CORAL | 7.0 | placement 단독 novelty 부족 (이론은 우아) | QUARRY trace가 locality 구조 입증 시 |
| GOVERNOR (구 TEMPER) | 6.7 | predictive 이득 미입증 + thermal 여유 시 소멸 | KILN 곡선 확보 후 |
| CINDER | 6.4 | 32GB-class 한정으로 동기 축소 | storage-pressure SKU 시나리오 구체화 시 |
| SILT | 6.2 | CQD [arXiv:2601.00938](https://arxiv.org/abs/2601.00938) 60-70% scoop 경계 | D_verb 좁힌 재포지셔닝 한정 |
| PILOT | drop | Mamba Drafters [arXiv:2506.01206](https://arxiv.org/abs/2506.01206) 45% + vLLM `nemotron_h_mtp` 실존 | 신규 white space 발견 시만 |
| ANCHOR-S | 흡수 | HEARTH M3 부분집합 + 이름 충돌 | (HEARTH 부활 시 ablation으로) |

## 8. 참고

- **Phase 상세 로그**: `sessions/2026-06-04-mode2-nemotron3-edge-agentic-serving.md` (로컬 wiki, 미publish)
- **Staging** (재현/감사용): `__research_wiki/staging/2026-06-04-nemotron3-agentic-*.md` — step0a (nano/super/workload) + step0 search 2 + phase1 ×3 + phase2 리뷰 ×6 + phase1' ×2 + phase2' ×1
- **누적 파일 갱신**: index / ideas / papers / trends (T15-T17) / concepts (C40-C43) / README
- **Deferred 검증 항목** (구현 착수 전 의무): shared expert 수 HF config 확정 (보고서=2, vLLM recipe 표기 충돌) / RoMe arxiv ID 혼선 (2512.01541 vs 2512.01644) / KAIROS ID 미검증 / 2026 preprint re-WebFetch

## 9. 약어 / 핵심 용어 풀이

> **CTRL+F (Mac: Cmd+F)** 로 검색하세요.

- **SSM (State Space Model)**: 시퀀스를 고정 크기 recurrent state로 처리하는 모델 계열 — KV cache 없이 길이 무관 메모리.
- **Mamba-2**: selective SSM의 대표 구현 (state dim 128). Nemotron 3 Nano의 52층 중 23층.
- **MoE (Mixture-of-Experts)**: 토큰별로 일부 expert FFN만 활성화하는 sparse 구조. Nano: 128 expert 중 top-6 + shared 2 활성.
- **Shared expert**: 모든 토큰에 무조건 활성화되는 expert (DeepSeekMoE 스타일) — per-token expert read의 25%.
- **LatentMoE**: token을 latent 차원 (d→ℓ)으로 압축 후 expert 연산 — expert read·all-to-all을 d/ℓ배 절감 (Super 전용, Nano 부재).
- **MTP (Multi-Token Prediction)**: 추가 head로 다음 k 토큰을 draft하는 native speculative decoding (Super 전용).
- **GQA (Grouped-Query Attention)**: Q-head 여러 개가 KV-head를 공유 — Nano는 32:2 (KV cache 16× 절감).
- **NoPE**: positional embedding 부재 — Mamba가 위치 정보를 암묵 인코딩.
- **Selective FP8 / precision island**: attention 6층 + 선행 Mamba 6층 + 전체 Conv1D만 BF16, 나머지 FP8 — 민감도 국소화.
- **NVFP4**: NVIDIA 4-bit FP 포맷 (E2M1 + 16-elem block scale E4M3) — Super pretrain 정밀도.
- **SSM state cache**: Mamba 층의 recurrent state 메모리 (Nano ~48MB/agent, 상수). polysemy 주의 — attention KV cache와 구분.
- **Verbosity amplification**: state 양자화 bias가 출력 토큰 수를 늘리는 현상 (Super 실측 +40%) — 본 세션 SPENDTHRIFT/BLOAT의 대상.
- **Working-set union |∪S_a|**: N개 agent가 일정 기간 활성화한 expert 집합의 합집합 — TIDEMARK cliff 예측의 핵심량.
- **Union growth / coupon-collector**: 표본을 뽑을수록 새 항목 발견율이 줄어드는 확률 모델 — agent 추가 시 expert union 성장 모델링.
- **2-state Markov 보정**: token 간 expert 선택의 자기상관을 반영한 union-growth 보정 (i.i.d. 가정 완화).
- **Admission control**: 새 요청/agent의 수락 여부를 자원 모델로 결정하는 제어 — TIDEMARK 단일 소유 (polysemy: 네트워크 admission과 구분).
- **LPDDR5/5x**: 모바일/edge DRAM 규격 — AGX Orin 204.8GB/s, Thor·DGX Spark 273GB/s. HBM (H100 3.35TB/s) 대비 1/12~1/16.
- **UMA (Unified Memory Architecture)**: CPU/GPU가 물리 메모리 공유 (Jetson/DGX Spark) — tool 실행 (CPU)과 decode (GPU)가 같은 BW 경쟁.
- **EMC (External Memory Controller)**: Jetson의 DRAM 컨트롤러 — tegrastats로 utilization 측정.
- **tegrastats / jtop / INA3221**: Jetson user-space 전력·열·메모리 텔레메트리 도구/센서.
- **nvpmodel**: Jetson 전력 모드 설정 도구 (user-space, R45.1 적합).
- **MPS (Multi-Process Service)**: NVIDIA GPU 프로세스 공유 서비스 — active-thread-percentage로 SM 점유 근사 제어.
- **Green Context**: CUDA SM partition API — Thor 완전 지원, Orin 제한적.
- **MambaSpec / MambaManager**: vLLM v1의 Mamba state 관리 클래스 (clone 검증 완료) — MOORING/BLOAT 구현 지점.
- **RoutedExpertsCapturer**: vLLM의 expert 라우팅 관측 hook (실존 검증) — TIDEMARK/QUARRY 프로파일러.
- **cache_salt**: vLLM prefix cache 키 분리 인자 — agent-conditioned 키의 실존 진입점.
- **fused_moe**: vLLM의 MoE GEMM 커널 모듈 — QUARRY trace 추출 지점.
- **Ramulator2 / DRAMSim3**: cycle-accurate DRAM 시뮬레이터 (활성 유지보수 확인) — QUARRY trace-driven 경로.
- **Stochastic Rounding (SR) / Philox**: 확률적 반올림 (bias 제거) / counter-based RNG — Super의 state quant 해법이자 BLOAT 대조군.
- **JS-divergence**: 분포 간 거리 (대칭 KL) — HEARTH의 agent 분포 분기 측정량.
- **Decision tree Pass/Below/Critical/Outperform**: 실험 결과 4분기 (기대 ±10% / 10-30% 미달 / >30% 미달 / >20% 초과) — §4.
