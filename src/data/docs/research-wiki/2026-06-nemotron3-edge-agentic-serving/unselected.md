# 미선정 idea (전수 8건) — 2026-06-04-mode2-nemotron3-edge-agentic-serving

> Phase 2' integrated re-review 결과 Tier-1(MOORING/TIDEMARK/SPENDTHRIFT/HEARTH/TOLLGATE/CORAL/GOVERNOR/CINDER) 및 Tier-2 독립(KILN/QUARRY/BLOAT/SILT) 선정에서 제외되었거나, drop/흡수/조건부로 정리된 8건.
> 형식: R10-α.3 bullet. 5축 = nov/diff/impact/ai-impl/arch-impl.

---

### HEARTH — Agent-Conditioned Expert Working-Set Caching
- **GAP**: 글로벌 LRU는 agent N 증가 시 round-robin recency 오염으로 expert cache hit이 붕괴 — agent-program-level 안정성(router RL-freeze)을 활용한 분기 cache 부재.
- **시도한 overview**: agent-id(`cache_salt`)로 per-(agent,layer) warm working-set(EWMA top-K)을 분할하되, JS-divergence/I(agent;expert) gate로 분기가 임계 미만이면 global-LRU fallback(M1). working-set admission gate(M2) + shared-pin(ANCHOR-S 흡수, shared=per-token 25%, M3).
- **Phase 2' 최종 score**: **7.3** (nov 6.5 / diff 7 / impact 8 / ai-impl 7 / arch-impl 8).
- **미선정 사유**: FineMoE [arXiv:2502.05370](https://arxiv.org/abs/2502.05370)(EuroSys 2026, input-semantic cache) 인접 **35–40%** — agent-program-level 차별은 견고하나 agent-key가 세션 내 가장 generic building block이라 Top-3 경계에서 SPENDTHRIFT에 밀림. **Tier-1 diversity 구성**상 차순위(Tier-1 명목 유지, Top-3 미포함).
- **재방문 조건**: FineMoE 대비 agent-program-level 안정성을 ablation으로 실증(JS-divergence gate 단독 기여)하면 Top-3 재진입.
- **흡수처/cross-share**: agent-key(`cache_salt`→`generate_block_hash_extra_keys`) 1급 소유 → CINDER M3·CORAL이 차용. ANCHOR-S 흡수(shared-pin M3). co-activation capturer를 CORAL/QUARRY와 공유.

### TOLLGATE — Phase-Aware Approximate Bandwidth Arbitration
- **GAP**: multi-agent edge에서 한 agent의 CPU-bound tool/retrieval burst(RAG 81–89% 점유)가 다른 agent GPU decode와 동일 LPDDR 채널 경합 → memory-bound decode TPS가 선형 이상 붕괴. 직접 BW QoS 노브 부재.
- **시도한 overview**: orchestrator↔vLLM 경계 phase 신호(running/tool-suspended)로 근사 actuator(cgroup `cpu.weight`/`cpuset` + MPS SM% + Green Context[Thor 한정]) 동적 배분(M1) + expert-DMA tool-window piggyback(M2) + fair-queueing/network-calculus SLO bound 정식화(M3).
- **Phase 2' 최종 score**: **7.1** (nov 7 / diff 7 / impact 8 / ai-impl 6.5 / arch-impl 7).
- **미선정 사유**: ai/arch +0.7 회복은 강하나 user-space LPDDR BW QoS 노브 부재(F04)로 **근사 제어 effectiveness가 직접-BW-QoS 대비 유의 강점인지 실측 선행 필요**. peer-review UMA baseline ≥50% 보강 의무 미실행.
- **재방문 조건**: 근사 제어(cgroup/MPS) effectiveness가 직접-BW-QoS 대비 유의 강점임을 AGX Orin 실측으로 보이고 peer-review UMA scheduling baseline 1편 보강 시.
- **흡수처/cross-share**: tool-signal(orchestrator↔vLLM 경계) = MOORING과 공유. SLO 위반 시 admission defer → TIDEMARK N*. expert-DMA M2 → arbitration은 **TIDEMARK stack(admission→arbitration 2계층) 후보**. `RoutedExpertsCapturer` p_a를 TIDEMARK와 공유.

### CORAL — Frozen-Router Co-activation Graph for 2-Tier Placement
- **GAP**: frozen router 하 expert pair 공활성 `C[i,j]`가 정적·heavy-tailed인데, random placement는 tier 간 cut(=migration block time 90%↑)이 큼 — 보장 있는 placement 부재.
- **시도한 overview**: `RoutedExpertsCapturer`로 `C[i,j]` offline 수집 → normalized-cut/Louvain spectral partition → 2-tier(LPDDR-resident/NVMe-cold) 배치 맵(M1) + 닫힌형 modularity Cheeger-type bound(Q>Q₀ → cut (1−λ₂/d) 감소) + H(e_next|e_cur) prefetch 상한(M2) + partition-block prefetch lookahead(M3, Tier-2).
- **Phase 2' 최종 score**: **7.0** (nov 7 / diff 7 / impact 7 / ai-impl 7 / arch-impl 7).
- **미선정 사유**: agent-key/bank 제거로 graph-theory 차별축은 깨끗해졌으나, migration이 **부분-offload/32GB-class에서만 발생**(AGX Orin 64GB full-resident에선 미발생)이라 impact 상한. net 점수 불변(Δ0) — Top-3 경쟁에서 밀림.
- **재방문 조건**: 32GB-class/부분-offload 시나리오에서 modularity bound가 random placement 대비 migration 실측 우위를 보이고 DanceMoE/MoE-Beyond 보강 시.
- **흡수처/cross-share**: **co-activation `C[i,j]` 그래프 → QUARRY 공유**(QUARRY는 row-buffer reorder trace에 동일 graph 사용, 프로파일러 1회 구현). agent-key는 HEARTH cross-share로 제거.

### GOVERNOR — Predictive Thermal-Aware Reasoning-Budget Control (舊 TEMPER)
- **GAP**: edge thermal 시정수(수십초~분) ≫ 제어주기(decode step 수백ms)라 reactive 제어는 늦음 — reasoning token 수가 thermal 정상상태·동시 agent 수의 1차 결정 요인인데 선제 제어 부재.
- **시도한 overview**: 외부 sidecar에서 predictive thermal 모델(부하→junction temp 회귀, KILN 데이터 학습)로 throttle 도달 전 per-agent `max_reasoning_tokens`를 chat-template budget으로 동적 주입(M1) + thermal-derived total을 agent 난이도로 가중 분배(M2). verbosity-guard는 BLOAT에 위임.
- **Phase 2' 최종 score**: **6.7** (nov 6.5 / diff 7.5 / impact 6.5 / ai-impl 6 / arch-impl 8). 본문 §4 6.6과 ±0.1 반올림 차.
- **미선정 사유**: predictive 모델이 reactive 대비 유의 이득을 KILN 데이터로 보여야 하는 **입증 부담** + 능동 냉각 시 이득 소멸·accuracy-floor 비단조로 impact 6.5 + **KILN 측정 선행 의존 리스크**. Tier-1 명목 유지하나 추천 우선순위 하위(조건부).
- **재방문 조건**: KILN 데이터로 predictive thermal 모델이 reactive 대비 유의 이득(시정수 mismatch 해소)을 실증하면 강한 MLSys 후보, 실패 시 ISLPED measurement letter로 강등.
- **흡수처/cross-share**: thermal 데이터 = **KILN 소유 입력**(predictive 모델 학습). state-quant→verbosity 인과는 SPENDTHRIFT/BLOAT 위임(舊 M3 verbosity-guard 제거). throttLL'eM(freq)+Token-Budget(thermal-blind) 결합 미점유.

### CINDER — Energy-Objective Expert Tiering
- **GAP**: 32GB-class edge에서 expert population 30GB가 KV/state와 경합해 부분 offload 강제 시, storage tier는 latency를 prefetch로 가려도 **per-token energy ~12× 폭증**(DRAM 대비 Flash 50×) — latency-only tier 결정의 energy 맹점.
- **시도한 overview**: tier 결정변수에 per-access energy 포함한 J/token 목적함수(weighted facility-location)로 {LPDDR-hot, NVMe-cold} 배치(M1) + prefetch-under-energy-budget(M2) + NVMe-access 회계(INA3221 + 외부 전력계, M3).
- **Phase 2' 최종 score**: **6.4** (nov 6 / diff 7 / impact 7 / ai-impl 6 / arch-impl 6). surviving Tier-1 중 최하위.
- **미선정 사유**: corrected math로 동기가 **32GB-class(RTX 5090) 부분-offload 강제**로 좁아짐 — AGX Orin 64GB full-resident에선 energy 압박 미발생(impact 7.5→7 하향). mechanism이 energy 1축(facility-location 표준 framing)으로 thin. 외부 전력계 setup 부담(ai/arch 6).
- **재방문 조건**: 32GB-class 부분-offload binding workload(KV/state 큰 multi-agent) 정의 + ASPLOS/EuroSys storage peer-review baseline 보강 시 Tier-1 강화.
- **흡수처/cross-share**: agent-key = HEARTH M1 차용. diversity-sizing = TIDEMARK union-growth 인용. CINDER 고유축 = **energy facility-location tier placement 단독**. SSD-harmful [arXiv:2508.06978](https://arxiv.org/abs/2508.06978)(IEEE) backbone.

### CINDER 관련 주: 위 항목은 Tier-1(조건부)로 명목 잔류하나 Top-M 추천 미포함이라 미선정 목록에 병기.

### SILT — Verbosity-Coupled Distortion of Multi-Agent SSM State Checkpoints
- **GAP**: agent state를 rate R로 압축할 때 진짜 한계는 verbosity-aware distortion `D_verb(R)`(출력 token + accuracy)인데, 기존 R-D 이론은 verbosity와 무연결 + serving 미적용.
- **시도한 overview**: agent 전환 시 SSM state PCA top-r 절단(또는 per-component bit) R sweep → effective-rank(특이값 spectrum) 측정 + empirical `D_verb(R)` 곡선 + verbosity 폭발 임계 R* 보고 + state-resident agent 수 확장 정량(M1, `MambaManager`/`SimpleCPUOffloadScheduler` offload state codec).
- **Phase 2' 최종 score**: **6.2** (nov 6 / diff 6 / impact 6 / ai-impl 6 / arch-impl 8). diff 5→6 회복(integrity 해소).
- **미선정 사유**: CQD [arXiv:2601.00938](https://arxiv.org/abs/2601.00938)(latent state R-D + spectral hard-thresholding) + Math Formalism SSM [arXiv:2410.03158](https://arxiv.org/abs/2410.03158)(selective SSM R-D bound)가 핵심 bound(reverse water-filling)를 **60–70% 경계**로 선점 — R-D bound를 main에서 내리고 effective-rank+D_verb+multi-agent로 좁혀 생존하나 이론 코어 약화 + state 48MB 작아 동기 약함.
- **재방문 조건**: MOORING state-resident 확장과 결합해 D_verb(R)·effective-rank가 독립 letter로 충분히 분리됨을 보이고 CQD/2410.03158 차별을 정량화하면.
- **흡수처/cross-share**: state-quant→verbosity 인과 = SPENDTHRIFT 정리 의존(D_verb). multi-agent state-resident 동기 = MOORING 결합. CQD/2410.03158 인용 의무.

### PILOT — Constant-State SSM Draft Sidecar for Spec Decode
- **GAP**: spec-decode draft의 KV cache가 context에 비례해 multi-agent footprint를 키우는데, SSM draft는 context-무관 constant-memory라 agent scaling에 유리할 잠재.
- **시도한 overview**: 소형 Mamba-2 draft로 constant-state draft pool(M1) + draft-guided expert prefetch(router dry-run, M2) + adaptive draft depth(M3). hybrid Mamba target + SSM draft verify.
- **Phase 2' 최종 score**: **drop**.
- **미선정 사유**: Mamba Drafters [arXiv:2506.01206](https://arxiv.org/abs/2506.01206)(EMNLP 2025 Findings)가 SSM draft constant-memory를 single-stream으로 **정확히 동일 논거 45% 선점** + vLLM **`nemotron_h_mtp` 인프라 실존**(`config/speculative.py:42`, `models/nemotron_h_mtp.py`)이 "MTP 부재→외부 draft 유일" 전제 반박 + hybrid spec-decode production 미성숙(Qwen3-Next MTP 76% regression #35387, first-attempt >500 LoC).
- **재방문 조건**: `nemotron_h_mtp` MTP vs 외부 SSM draft의 multi-agent footprint scaling 비교가 독립 측정 노트(Tier-2 후보)로 가치를 보이면.
- **흡수처/cross-share**: M2(draft-guided expert prefetch) → **HEARTH prefetch / MOORING(舊 RELAY merge) state-pool ablation 1줄로 흡수**. M1(constant-state draft pool)·M3(adaptive depth) drop.

### ANCHOR-S — Shared-Expert Residency Pinning
- **GAP**: shared expert(2/layer)가 routed와 같은 풀에서 경쟁하면 evict→reload 반복 — 매 token 발생하는 shared read 트래픽 손실.
- **시도한 overview**: expert를 shared(2, 상시 pin)/routed(128, streaming) 태깅해 shared를 `cpu_offload_params` 제외로 LPDDR-resident 영구 상주(단일 mechanism).
- **Phase 2' 최종 score**: **HEARTH M3로 흡수**(독립 arch score 8이었으나 흡수).
- **미선정 사유**: 원래 ANCHOR-S의 "shared = 전체 1.5%" 평가가 corrected math로 **틀림**(shared = routed 6+shared 2 중 2/8 = **per-token 25%**, 0.46GB/token)으로 정정되며 pin 이득 상한이 구조적으로 커졌으나, 독립 idea로는 mechanism이 trivial-pin 1개라 thin + **이름 충돌**(ANCHOR 금지명) → HEARTH M3(ablation/디폴트 정책)로 흡수.
- **재방문 조건**: shared-pin on/off ablation이 HEARTH 내에서 독립 contribution(25% 트래픽 상수화)으로 분리 입증되면 별도 letter 가능.
- **흡수처/cross-share**: **HEARTH M3 흡수** — shared/routed 구조적 분리(BuddyMoE redundancy의 정확도 trade와 달리 무손실). "pinned tier"는 L2(4MB) 아닌 LPDDR-resident region.

---

## 종합 — 미선정 5축 score 요약

| ID | nov | diff | impact | ai | arch | avg | 처리 |
|---|---|---|---|---|---|---|---|
| HEARTH | 6.5 | 7 | 8 | 7 | 8 | **7.3** | Tier-1 차순위(Top-3 미포함) |
| TOLLGATE | 7 | 7 | 8 | 6.5 | 7 | **7.1** | Tier-1, 근사제어 실측 선행 |
| CORAL | 7 | 7 | 7 | 7 | 7 | **7.0** | Tier-1, 32GB-class 의존 |
| GOVERNOR | 6.5 | 7.5 | 6.5 | 6 | 8 | **6.7** | Tier-1(조건부), KILN 의존 |
| CINDER | 6 | 7 | 7 | 6 | 6 | **6.4** | Tier-1(조건부), 최하위 |
| SILT | 6 | 6 | 6 | 6 | 8 | **6.2** | Tier-2, CQD 선점 |
| PILOT | — | — | — | — | — | **drop** | Mamba Drafters 선점 + nemotron_h_mtp 반박 |
| ANCHOR-S | — | — | — | — | (8) | **흡수** | HEARTH M3 흡수 + 이름 충돌 |

> 참고: HEARTH/TOLLGATE/CORAL/GOVERNOR/CINDER는 Tier-1 명목 자격은 유지하나 Top-M 추천(TIDEMARK·MOORING·SPENDTHRIFT)에서 제외되어 본 미선정 목록에 병기. PILOT만 완전 drop, ANCHOR-S는 흡수.
