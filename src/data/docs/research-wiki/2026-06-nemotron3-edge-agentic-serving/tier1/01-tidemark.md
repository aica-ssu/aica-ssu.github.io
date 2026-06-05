# TIDEMARK — Predicting the Critical Agent-Count via Working-Set Union Growth in Frozen-Router Hybrid Serving

> **Hook**: frozen-router hybrid MoE multi-agent에서 expert working-set union이 LPDDR-resident budget B에 닿는 만조 임계 **N\***를 closed-form으로 ±1 agent 정확도로 사전 예측해, throughput cliff 직전에 admission을 차단한다.

- **Score (Phase 2', 5축 = novelty/diff/impact/ai-impl/arch-impl)**: 7 / 7.5 / 8.5 / 7 / 8 = **avg 7.6** (Δ0 vs Phase 2 — 이미 최고점·경량 정제로 유지가 정답)
- **Venue target**: Track A — **MLSys 2027** (이론+측정). 2차 NeurIPS efficiency track.
- **Tier 위치**: Tier-1 Top-3 (1위) — 5축 최고점·scoop 위험 최저(clear <30%)·cross-share admission 단일 hub.
- **Cross-share ownership**: **admission(N\*) = TIDEMARK 단일 소유** (MOORING이 state footprint, TOLLGATE이 BW 신호, HEARTH가 working-set 회계를 입력으로 제공). `RoutedExpertsCapturer`(p_a/q_a 프로파일러) = TIDEMARK·TOLLGATE-M2·CORAL·HEARTH 공유.

---

## 1. Research Questions

- **RQ-master**: frozen-router hybrid MoE multi-agent serving에서, 동시 agent 수 N에 대한 expert working-set union을 **2-state Markov union-growth closed-form**으로 예측해, union이 LPDDR-resident budget B(≈20GB)에 닿는 임계 **N\***를 **±1 agent** 정확도로 cliff 직전에 사전 차단할 수 있는가?
- **RQ-1**: 2-state Markov(같은 expert 유지 확률 q) 보정 union-growth가 i.i.d. coupon-collector baseline 대비 N\* 예측 오차를 **±1 agent** 이내로 줄이는가 (i.i.d.는 자기상관 무시로 union 과대추정 → N\* 과소예측)?
- **RQ-2**: N≤N\* admission 차단이 무제어 admission 대비 cliff 후 sustained throughput을 **+25~40%** 보존하는가 (AGX Orin 64GB, agent 1→16 sweep, thermal steady-state)?

## 2. 기준 코드베이스 (Baseline Source)

- **vLLM (V1 core)**: `https://github.com/vllm-project/vllm`
  - 검증 commit **`7c37096`** (`7c37096620fa4e161c1d8c1db5c43c8514545d84`, HEAD `[Core][Refactor] thread scheduler_block_size into KVCacheManager`, 2026-06-02 fetch). ai-impl 리뷰가 `~/code_baseline/vllm-project--vllm`에 local clone navigation으로 검증(단순 fetch 아님).
- **Model**: HuggingFace **Nemotron-3-Nano-30B-A3B** (BF16 / FP8 / NVFP4 공개). vLLM 공식 recipe 존재(`docs.vllm.ai/projects/recipes/.../Nemotron-3-Nano-30B-A3B.html`, vLLM ≥0.12.0 권장·0.11.2 동작). `registry.py:180 "NemotronHForCausalLM": ("nemotron_h", ...)` — Nano 3 류 직접 지원.
- **컨테이너**: vLLM NGC container `nvcr.io/nvidia/vllm:26.01-py3` (Jetson은 pip 미지원 → NGC container + commit pin).
- **Target HW**: 1차 **AGX Orin 64GB**(204.8 GB/s LPDDR5, FP8 full-resident), 보조 Thor 128GB / RTX Pro 6000 96GB(sweep 상한 확장).

## 3. 배경 / GAP

- Nemotron-3-Nano = 31.6B total / 3.2B active, **Mamba-2 23층 / MoE 23층 / Attention 6층(KV-head 2) = 52층**. expert ≈9.98M param(2-matrix FC1/FC2, squared-ReLU) → **10MB FP8**, routed 128/층 + shared 2/층 → expert population **≈30GB FP8**, 모델 전체 **32GB FP8**(W4-GGUF 실측 22GB). → AGX Orin 64GB는 **FP8 full-resident(offload 불필요)**.
- **per-token expert read 1.84GB/token**(batch-1, FP8: (6 routed+2 shared)×10MB×23층) → 204.8 GB/s ÷ 1.84GB = **~9 tok/s 상한**, EdgeReasoning [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) 실측 4.7~9.3 TPS와 정합.
- **핵심 cliff 동기**: per-token read는 batch 내 expert union으로 amortize되나 **agent↑ → union↑ → amortization 붕괴**. union이 LPDDR-resident budget B를 넘으면 expert가 read-thrash → throughput superlinear drop (cliff). batch-1 full-resident에서도 1.84GB read 자체가 BW 천장.
- B 정의: AGX Orin 64GB는 **expert full-resident** → B ≈ 64 − 32(weight) − KV/overhead ≈ **~20GB working budget**. (cross-review §0b의 "weight offload·offload-bound" 가정은 64GB에서 거짓 → B는 weight-resident 상수를 뺀 잔여로 well-defined.)
- "diverse 2 ≈ homogeneous 16" ([arXiv:2602.03794](https://arxiv.org/abs/2602.03794)) → union 크기가 agent diversity의 함수 → uniform admission은 diverse agent에서 union 폭증.
- Agent workload: decode-dominated **91.0~98.6%**, prefix/KV reuse **84.6~99.5%**, turn 평균 12~62(최대 786), tool 실행 2~29% ([arXiv:2605.26297](https://arxiv.org/abs/2605.26297)) → decode token 호출은 강한 자기상관(연속 token이 같은 expert, frozen-router라 더 심함).
- **기존 연구 한계**: Concur [arXiv:2601.22705](https://arxiv.org/abs/2601.22705)는 proactive admission이나 **collapse 위치 예측 부재 + attention-KV 대상**. Roofline Scaling Laws [arXiv:2602.10377](https://arxiv.org/abs/2602.10377)는 단일 모델 roofline(multi-agent union-growth 미적용). Diversity scaling [arXiv:2602.03794](https://arxiv.org/abs/2602.03794)는 품질만 다루고 diversity→union sizing 미연결. closed-form N\* 예측 + cliff 위치 예측은 2nd similarity sweep(2025-12~2026-06)에서도 **직접 경쟁 부재(가장 깨끗)** 재확인.
- **union-growth math anchor**: M개 expert/층, agent a의 frozen 분포 p_a, KT decode step. i.i.d. coupon-collector는 `E|∪S_a| = M − Σ_e ∏_a (1−p_a(e))^{KT}` — 매 step 독립 추출 가정. 2-state Markov 보정은 `(1−p_a(e))^{KT}` 항을 "agent a가 KT step 동안 expert e를 한 번도 안 거치는 확률"로 교체(같은 expert 유지 q_a, 전환 1−q_a) → 자기상관이 클수록(q_a↑) 실효 추출 횟수↓ → union↓. i.i.d.는 union 과대추정 → N\* 과소예측(보수적 차단이나 throughput 손실).
- **device 별 cliff 양상**: AGX Orin 64GB(204.8 GB/s)는 B≈20GB working budget에서 용량 cliff + BW-bound superlinear drop 동시 재현. RTX 5090 32GB는 B≈경계라 용량 cliff는 재현하나 1.79 TB/s로 **cliff 후 BW-bound drop 미재현**(arch F01) → 메커니즘 검증은 LPDDR SKU에 한정. Thor/Spark 128GB(273 GB/s)는 cliff가 더 큰 N에서 발생(sweep 상한 확장).
- **frozen-router 안정성 전제**: Nano insight 5 — post-training RL-freeze로 router 가중치 고정 → agent별 expert activation 분포가 시간적으로 안정. 부분 freeze/expert-bias 업데이트 시 p_a/q_a 재프로파일 필요(post-training 레시피 의존, 리스크 §7).

## 4. Mechanisms

### M1 — Markov-corrected union-growth N\* estimator

#### 동작 원리
- **(scheme)** frozen p_a 프로파일을 `RoutedExpertsCapturer.capture(layer_id, topk_ids)`로 agent별 expert 분포 수집 → union 기대값을 **i.i.d.가 아닌 2-state Markov**로 추정: `E|∪S_a| = M − Σ_e ∏_a P(expert e 미호출 | Markov q_a, KT step)`. N\* = max{N : E|∪S_a(N)| × s_expert(9.98MB FP8) ≤ B}. i.i.d. coupon-collector 버전은 lower bound(union 과대추정)로 병기.
- **(문제)** i.i.d. 가정은 decode token의 자기상관(연속 token이 같은 expert) 무시 → union 과대추정 → N\* 과소예측(보수적이나 부정확). frozen-router라 자기상관이 더 심함.
- **(step-by-step)** (1) offline으로 agent별 p_a·전이확률 q_a 측정(capturer 출력), (2) 2-state Markov union closed-form으로 N\* 산출, (3) 실측 cliff 위치 vs 예측 N\* calibration.
- **(차별화)** Concur는 proactive admission이나 collapse 위치 예측 부재·attention-KV — 본안은 hybrid 상수-state + frozen-MoE union closed-form 예측 + Markov 자기상관 보정이 신규.

#### 기대 효과 (Gain)
- **primary axis [측정]**: N\* 예측 오차 = **±1 agent** (Markov 보정 후, frozen 분포) vs Concur 사후-완화(예측 부재).
- **secondary axis [Performance]**: cliff 후 throughput 보존 +25~40% (M2 차단과 결합).

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | `RoutedExpertsCapturer.capture(layer_id, topk_ids)` (L110) | routed expert topk_ids 관측만 | agent별(=`cache_salt`) p_a/q_a 분할 집계 출력 consume | extend (기존 hook consume) |
| (offline 분석 모듈, 신규) | `n_star_estimator` (Markov closed-form) | 부재 | 2-state Markov union-growth N\* 산출 + i.i.d. lower bound 병기 | 신규 (offline, vLLM 외부) |
- **검증 출처**: `RoutedExpertsCapturer`(routed_experts_capturer.py:58/110)는 ai-impl 리뷰 §0 "MoE 라우팅 capture hook [✅]"로 clone-verified @ `7c37096`. N\* 추정은 offline closed-form이라 vLLM 코드 변경 불요(ai-impl TIDEMARK §M1 [✅]).

#### 검증 시나리오 (Test Plan)
- **Unit test (분)** — closed-form 수렴:
  - 목적: Markov union closed-form이 q→0.5에서 i.i.d. lower bound로 수렴하는지.
  - Input: 합성 p_a·q_a 분포.
  - Expected: q=0.5에서 i.i.d.와 일치, q↑일수록 union↓.
  - metric: union 기대값 상대오차.
  - 실행시간: ~5분. 실패 시 액션: closed-form 유도 재검토.
- **Mechanism-isolated test (시간)** — cliff calibration:
  - 목적: 실측 cliff 위치 vs 예측 N\* calibration.
  - Input: Nemotron-3-Nano FP8, agent 1→16 sweep on AGX Orin 64GB(thermal steady-state).
  - Expected: 예측 N\*가 실측 cliff와 ±1 agent.
  - metric: N\* 예측 오차(agent), union byte vs 측정 resident set.
  - 실행시간: ~6시간(sweep+calibration). 실패 시 액션: q_a 재프로파일·Markov 차수 상향(3-state).

### M2 — Diversity-aware admission (외부 sidecar)

#### 동작 원리
- **(scheme)** agent 분포 유사도(**JS-divergence**)로 union 증가율 추정 → **vLLM 외부 sidecar**가 admission threshold 산출, vLLM `request_queue.py` 정책 또는 `max_num_running_reqs`(scheduler.py:103)에 N≤N\* 주입. **MOORING state footprint + TOLLGATE BW 신호를 입력으로 통합 admission**(cross-share, admission 로직은 TIDEMARK 단일 소유).
- **(문제)** 균일 admission은 diverse agent에서 union 폭증(과소 admission 필요)·homogeneous에서 과대 admission(여유 낭비).
- **(step-by-step)** (1) 신규 agent 도착 시 기존 active set과 분포 유사도(JS-divergence) 계산, (2) union 증분 추정, (3) N\* 초과 예측 시 admission defer.
- **(차별화)** agent scaling 연구(diversity)는 정확도만 다룸 — expert working-set union sizing에 연결한 admission은 신규. admission을 vLLM 내부가 아닌 외부 sidecar로 둬 fork 유지비↓(ai-impl 정정).

#### 기대 효과 (Gain)
- **primary axis [Performance]**: cliff 후 throughput 보존 +25~40% (N≤N\* 차단 시).
- **secondary axis [Memory eff.]**: diverse 혼재 workload에서 over-admission 방지로 thrash 회피.

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/v1/core/sched/scheduler.py` | waiting loop (L554), `max_num_running_reqs` (L103) | KV block 기반 admission(`get_num_blocks_to_allocate`) | N≤N\* threshold 주입 hook (sidecar 산출값) | extend |
| `vllm/v1/core/sched/request_queue.py` | `create_request_queue` 정책 | FIFO/priority | diversity-aware defer 정책 확장 | extend |
| (admission sidecar, 신규) | JS-divergence + N\* threshold 산출 | 부재 | MOORING state·TOLLGATE BW·HEARTH working-set 입력 통합 | 신규 (외부 sidecar) |
- **검증 출처**: `scheduler.py` schedule(L335)/waiting loop(L554)/`max_num_running_reqs`(L103), `request_queue.py create_request_queue`는 ai-impl 리뷰 TIDEMARK §M1/M2 [✅ 경로]/[⚠️ patch 부재]로 검증 @ `7c37096`. admission이 현재 KV block 기준이라 expert working-set 기준은 신규 결정변수.

#### 검증 시나리오 (Test Plan)
- **Unit test (분)** — 유사도→union 단조:
  - 목적: JS-divergence 기반 union 증분 추정이 실제 union 측정과 단조 관계인지.
  - Input: 2~4 agent 합성 분포(homogeneous→diverse).
  - Expected: diverse일수록 union 증분↑.
  - metric: 추정 union vs 실측 Spearman ρ.
  - 실행시간: ~10분. 실패 시 액션: 유사도 metric 교체(I(agent;expert)).
- **Mechanism-isolated test (시간)** — admission 보존 효과:
  - 목적: N≤N\* admission이 cliff 후 throughput을 보존하는지.
  - Input: agent 1→16 sweep, {무제어 admission / TIDEMARK admission} 2구성.
  - Expected: TIDEMARK가 cliff 후 sustained tok/s +25~40%.
  - metric: aggregate tok/s, admission defer 횟수, SLO 위반율.
  - 실행시간: ~8시간. 실패 시 액션: threshold margin 조정(N\*−1 보수 차단).

### M3 (Tier-2) — State-swap 우선 eviction

#### 동작 원리
- **(scheme)** 메모리 압력 시 expert union이 아닌 **state(MOORING M1) 우선 swap** — admission이 N\*에 닿기 전 inactive agent state를 demote해 budget 확보. **MOORING과 cross-share(eviction 객체=state, MOORING 소유)** — TIDEMARK은 admission 트리거만 제공.
- **(문제)** cliff 직전 미세 budget 확보로 N\*를 한 칸 연장.
- **(step-by-step)** (1) N→N\* 접근 시 inactive agent state를 MOORING host-cold demote, (2) expert budget 확보, (3) N\* 한 칸 연장.
- **(차별화)** MOORING과 공유 — TIDEMARK은 N\* admission 트리거만 소유(재발명 제거).

#### 기대 효과 (Gain)
- **primary axis [Memory eff.]**: state demote로 N\* +1 agent 연장(state-bound 영역).
- **secondary axis [Performance]**: cliff 직전 graceful degradation.

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager` (L854) | request-level state block | (MOORING M1 `agent_state_pool` 위임) demote 트리거 | reference (MOORING 소유) |
| `vllm/v1/kv_offload/simple_kv_offload/manager.py` | `SimpleCPUOffloadScheduler` (L67, MambaSpec-aware L215) | 부분 mamba offload | state host-cold demote 호출 | reference (MOORING 소유) |
- **검증 출처**: `MambaManager`/`SimpleCPUOffloadScheduler`는 ai-impl §0 [✅]/[✅ partial] @ `7c37096`. M3는 MOORING M1과 cross-share이므로 구현 상세는 02-mooring.md 참조(여기서는 admission 트리거만).

#### 검증 시나리오 (Test Plan)
- **Unit test (분)** — demote 회계:
  - 목적: state demote 후 expert budget 증가량이 demote된 state byte와 일치하는지.
  - Input: mock state pool.
  - Expected: budget Δ = demote byte.
  - metric: budget 회계 정합.
  - 실행시간: ~5분. 실패 시 액션: demote 회계 버그 추적.
- **Mechanism-isolated test (시간)** — N\* 연장:
  - 목적: state-swap eviction이 N\*를 연장하는지.
  - Input: state-bound 시나리오(긴 prefix·다수 inactive agent).
  - Expected: N\* +1~2.
  - metric: 달성 동시 agent 수.
  - 실행시간: ~4시간. 실패 시 액션: MOORING M1 단독 검증으로 분리(02-mooring 참조).

## 5. 실험 플랜 (7-요소)

1. **Hardware**: 1차 **AGX Orin 64GB**(연구실 보유/확보 필요 — B≈20GB working budget, cliff 재현 device). 보조 **Thor 128GB / RTX Pro 6000 96GB**(확보 필요 — sweep 상한 확장, cliff 늦게). RTX 5090 32GB(B=32GB budget 변형). **RTX 5090은 1.79 TB/s라 cliff 후 BW-bound superlinear drop 미재현(용량 cliff만, 메커니즘은 LPDDR)** — 정직 명시.
2. **Model**: primary **Nemotron-3-Nano-30B-A3B FP8**(직접 사용, proxy 불요 — ai-impl 권고). proxy 보조 **Falcon-H1-1.3B**(`falcon_h1.py`) / **Qwen3-Next-80B-A3B**(`qwen3_next.py`, gdn_attn) / Bamba-9B(`bamba.py`) — 동일 hybrid Mamba+MoE 구조.
3. **Dataset / Workload**: agentic trace — **SWE-bench(Pro)** / **BFCL v4** / **τ-bench(tau-bench)** + synthetic multi-agent(diversity 제어: homogeneous→diverse role mix로 union 증가율 sweep). decode-dominated·prefix-reuse 특성 보존.
4. **Tools**: vLLM fork(`7c37096` pin) + `RoutedExpertsCapturer` offline p_a/q_a 수집 + **tegrastats/jtop**(EMC util·junction temp) + **nsys**(timeline, ncu는 Orin iGPU 미지원). gem5 미사용.
5. **Ablation + Protocol**: factorial = {i.i.d. coupon-collector vs 2-state Markov} × {무제어 admission vs TIDEMARK N\* admission} × {homogeneous vs diverse agent mix} × {N 1→16}. Baselines: **Concur [arXiv:2601.22705] (preprint)** proactive admission, **vLLM 기본 continuous batching(framework)**, **Roofline Scaling Laws [arXiv:2602.10377] (preprint)** 단일모델 예측. **TOLLGATE stack 통합 평가 1 시나리오**: agent N을 1→16 sweep — (1) TIDEMARK이 N\*에서 admission 차단(용량/공간 축), (2) admit된 agent들이 tool-burst/decode 혼재 시 TOLLGATE이 LPDDR BW arbitration(대역폭/시간 축) → aggregate TPS·SLO 위반율을 {no-control / TIDEMARK-only / TOLLGATE-only / stack} 4구성으로 비교. admission=TIDEMARK 단일 소유, arbitration=TOLLGATE(독립 논문 유지, venue 다름).
6. **Implementation Steps** (Step별 dependency + 완료 판정 수치):
   - Step 1: `RoutedExpertsCapturer` consume 모듈로 agent별 p_a/q_a offline 수집 (완료=16 agent×23층 분포 dump, q_a 추정 수렴 R²≥0.9).
   - Step 2 (←1): Markov union closed-form `n_star_estimator` 구현 + i.i.d. lower bound (완료=합성 분포에서 union 추정 상대오차 ≤5%).
   - Step 3 (←2): 외부 admission sidecar + `scheduler.py`/`request_queue.py` threshold 주입 (완료=N≤N\* 차단 동작 smoke).
   - Step 4 (←3): agent 1→16 cliff sweep + N\* calibration (완료=예측 N\* vs 실측 cliff ±1 agent).
   - Step 5 (←4, Tier-2): MOORING M1 연계 state-swap eviction N\* 연장 측정 (완료=N\* +1 입증 또는 분리 보고).
7. **Preliminary Analysis Metrics** (4단계):
   - **baseline repro**: 도구 nsys/tegrastats / 측정조건 Nemotron-3-Nano FP8 batch-1 decode @ AGX Orin 64GB / 기대 baseline ~9 tok/s(EdgeReasoning 4.7~9.3 정합) / 목표 Δ=±10% repro.
   - **bottleneck attribution**: 도구 `RoutedExpertsCapturer`+nsys / 조건 agent N sweep / 기대 baseline expert read가 critical path 90%+ / 목표 Δ=union이 B 넘는 지점에서 tok/s superlinear drop 관측.
   - **upper bound**: 도구 offline Markov closed-form / 조건 frozen 분포 / 기대 baseline i.i.d. N\* 과소예측 / 목표 Δ=Markov가 i.i.d. 대비 N\* +1~3 agent 정확.
   - **micro-pilot**: 도구 vLLM fork + sidecar / 조건 N 1→16, 2 admission 구성 / 기대 baseline 무제어 cliff 후 −40~60% drop / 목표 Δ=TIDEMARK +25~40% throughput 보존, N\* ±1.

## 6. 관련 연구 · 차별점

| 연구 | venue tag | 핵심 접근 | TIDEMARK 차별 axis (self-contained) |
|---|---|---|---|
| Concur [arXiv:2601.22705](https://arxiv.org/abs/2601.22705) | preprint 2026 | attention-KV proactive admission으로 collapse 완화 | TIDEMARK은 hybrid frozen-MoE expert **union closed-form으로 collapse 위치(N\*) 사전 예측** — Concur는 사후 완화만, 위치 예측 부재 |
| Roofline Scaling Laws [arXiv:2602.10377](https://arxiv.org/abs/2602.10377) | preprint 2026 | 단일 모델 roofline 성능 예측 | TIDEMARK은 **multi-agent expert working-set union-growth**가 메모리 budget에 닿는 임계를 예측(단일모델 roofline 아님) |
| Diversity scaling [arXiv:2602.03794](https://arxiv.org/abs/2602.03794) | preprint 2026 | agent diversity와 출력 품질 관계("diverse 2≈homogeneous 16") | TIDEMARK은 diversity를 **expert union sizing(메모리)**에 연결 — diversity scaling은 품질만 |
| Science of Scaling Agent Systems [arXiv:2512.08296](https://arxiv.org/abs/2512.08296) | preprint 2025 | agent scaling 일반 법칙 | TIDEMARK은 **메모리-bound N\* 예측**(LPDDR budget 천장)으로 특화 |
| AMV-L [arXiv:2603.04443](https://arxiv.org/abs/2603.04443) | preprint 2026 (⚠️ snippet-only, R13.1 paper-final 전 ID 검증 필요) | lifecycle-tier agent memory | TIDEMARK은 expert working-set union 예측(agent memory lifecycle 아님) — 인접 가능성, 차별 axis 검증 의무 |
| vLLM Hybrid KV+State V1 ([PyTorch blog](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)) | framework | Mamba state/KV first-class allocator | substrate(baseline·구현 path) — agent-grained N\* admission 정책 공백 |

## 7. 리스크 & Decision-Tree 분기

- **Pass** (N\* 예측 오차 ≤±1 agent **AND** cliff 후 throughput 보존 ≥+25%): MLSys 2027 본 투고. Markov 보정의 정확성 입증 → cross-share admission hub 확정.
- **Below** (N\* 오차 ±2~3 agent **OR** 보존 +10~25%): 3-state Markov / input-dependent 분포 shift online recalibration 추가. scope를 "frozen-router + LPDDR budget 고정 + thermal steady-state"로 명시 제한, NeurIPS efficiency track으로 전환.
- **Critical-fail** (i.i.d.와 Markov 차이 무의미 **OR** cliff가 예측 불가능하게 input-dependent): closed-form 예측 contribution 폐기 → Tier-2 "N\* cliff 실측 곡선" measurement letter(IEEE CAL/DATE)로 강등.
- **Outperform** (N\* 오차 ±0 agent **AND** 보존 ≥+40%): admission sidecar를 MOORING/TOLLGATE 입력 통합 full-stack admission으로 확장, 시스템 전체 hub로 flagship 격상.
- **임계값 backbone**: cliff = tok/s superlinear drop 지점(union > B≈20GB), N\* = max{N: union×10MB ≤ B}, SLO TPOT ≤ 100ms(MLPerf Interactive 30ms 보조).

## 8. Tier-2 scope 축소 variant

TIDEMARK의 Tier-2 축소는 **"frozen-router hybrid MoE의 N\* cliff 실측 곡선"** measurement letter(IEEE CAL / DATE 2027)다 — agent 1→16 sweep에서 expert union 크기와 tok/s cliff를 단일 HW(AGX Orin 64GB)에서 측정해 Markov-예측 N\*와 calibration하는 first-to-report. 이론 closed-form 입증이 흔들려도(Below/Critical-fail 분기) 실측 N\* 곡선 자체가 actionable contribution이며, M3(state-swap eviction)을 단독으로 떼어 검증 가능. 측정 letter라 narrow scope는 비감점.

## 9. 약어 / 용어 풀이

- **SSM (State Space Model)**: 입력을 recurrent 상태 텐서로 압축해 O(n) 추론하는 시퀀스 모델. Nemotron-3-Nano의 Mamba-2 23층이 이에 해당, KV cache 없이 상수 크기 state 유지.
- **Mamba-2**: selective SSM의 2세대(multi-head, d_state=128). conv1d state + SSM state를 함께 보유, context 길이와 무관하게 ~48MB/agent.
- **MoE (Mixture of Experts)**: 토큰마다 router가 일부 expert FFN만 활성화하는 sparse 구조. Nano는 routed 128 + shared 2/층, top-6 routed 선택.
- **GQA (Grouped-Query Attention)**: 여러 query head가 KV head를 공유해 KV cache를 줄이는 attention. Nano attention 6층의 KV-head=2(극소 KV).
- **NoPE (No Positional Encoding)**: 명시적 위치 인코딩 없이 동작하는 attention 변형(일부 hybrid 모델 채택). 본 idea에선 attention 6층 특성 맥락.
- **LPDDR5**: 모바일/edge SoC용 저전력 DRAM. AGX Orin은 204.8 GB/s, CPU·GPU가 공유(UMA).
- **EMC (External Memory Controller)**: Jetson SoC의 LPDDR 접근을 중재하는 컨트롤러. tegrastats의 EMC util이 메모리 대역폭 포화 지표.
- **JS-divergence (Jensen-Shannon divergence)**: 두 확률분포 간 대칭 거리(0~1 bounded). agent별 expert 분포 유사도 측정 → union 증가율 추정에 사용.
- **union-growth**: 동시 N개 agent의 expert working-set 합집합(∪S_a) 크기가 N에 따라 커지는 현상. coupon-collector(i.i.d.) 또는 2-state Markov(자기상관)로 closed-form 추정, B 천장에 닿는 N\*가 cliff.
- **N\* (critical agent-count)**: union×expert_size가 LPDDR-resident budget B에 닿는 임계 동시 agent 수. 초과 시 expert read-thrash로 throughput superlinear drop.
- **frozen-router**: post-training 후 MoE router 가중치를 고정(RL-freeze)해 expert activation 분포가 시간적으로 안정 → union closed-form 예측의 전제.
