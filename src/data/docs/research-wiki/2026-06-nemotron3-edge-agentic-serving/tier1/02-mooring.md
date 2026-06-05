# MOORING — Tiered SSM-State Pooling for Host-Sync-Free Agent Context Switching on Unified-Memory Edge SoC

> **Hook**: agent의 Mamba-2 SSM state(48MB 상수)를 host로 내보내지 않고 LPDDR-warm 계류지에 묶어두었다가 tool-window idle에 device-내 swap하면, 동시 agent 수를 KV 한계가 아닌 state-resident 한계까지 확장하면서 tool 복귀 prefill을 제거한다.

- **Score (Phase 2', 5축 = novelty/diff/impact/ai-impl/arch-impl)**: 8 / 7.5 / 8.5 / 7 / 8 = **avg 7.8** (Δ+0.6 vs Phase 2 — 병합으로 mechanism 완성 + ai/arch-impl 각 +1.0)
- **Venue target**: Track A — **MLSys 2027** (대안 EuroSys 2027 / OSDI 2027 — 메모리-시스템 프레이밍이 systems venue에 강함).
- **Tier 위치**: Tier-1 Top-3 (2위, avg 최고) — SSM-state-swap white space(2nd sweep 후에도 무경쟁). 단 핵심 가설(state restore ≪ KV) 실측 의존이 TIDEMARK보다 risk↑.
- **병합 출처**: NIMBLE(ai-opt A2) + RELAY(legacy IDEA-3) = MOORING (≥85% mechanism 중복 → 병합으로 자해 회피).
- **Cross-share ownership**: **state pool = MOORING 단일 소유**. admission = TIDEMARK 소유(MOORING은 state-cost provider). tool-signal(orchestrator↔vLLM 경계) = MOORING·TOLLGATE 공유.

---

## 1. Research Questions

- **RQ-master**: hybrid SSM-MoE multi-agent 전환에서, 23 Mamba층 SSM state(48MB 상수)를 (1) host round-trip 없이 LPDDR-warm 계류지에 유지하고 (2) tool-wait idle window에 device-내 swap하면, 동시 agent 수를 **KV 한계가 아닌 state-resident 한계(64GB에서 수십 agent)**까지 확장하면서 tool 복귀 prefill을 제거할 수 있는가?
- **RQ-1**: SSM state device-내 copy(48MB 상수) 기반 agent 전환이 KV-centric baseline(Tokencake/Continuum류) 대비 전환 latency를 **−60~80%** 줄이는가 (host round-trip 제거 + prefill 재계산 회피)?
- **RQ-2**: tool-window idle swap이 tool 복귀 TTFT를 prefill 재실행 대비 **−50% 이상** 줄이고, SLO 유지 동시 agent 수를 KV-bound admission 대비 **1.5~2.5×** 늘리는가 (state-resident 한계까지, KV 작아 자명할 위험 → ablation 분리)?

## 2. 기준 코드베이스 (Baseline Source)

- **vLLM (V1 core)**: `https://github.com/vllm-project/vllm`
  - 검증 commit **`7c37096`** (`7c37096620fa4e161c1d8c1db5c43c8514545d84`, HEAD `[Core][Refactor] thread scheduler_block_size into KVCacheManager`, 2026-06-02 fetch). ai-impl 리뷰가 `~/code_baseline/vllm-project--vllm`에 local clone navigation으로 검증.
- **Model**: HuggingFace **Nemotron-3-Nano-30B-A3B** (BF16 / FP8 / NVFP4). `registry.py:180 "NemotronHForCausalLM"` 직접 지원, vLLM 공식 recipe(≥0.12.0 권장·0.11.2 동작).
- **컨테이너**: vLLM NGC container `nvcr.io/nvidia/vllm:26.01-py3` (Jetson pip 미지원 → NGC + commit pin).
- **Target HW**: 1차 **AGX Orin 64GB**(FP8 full-resident + state pool). 보조 Thor 128GB / DGX Spark 128GB(수십 agent headroom). Orin NX 16GB·Nano 8GB는 모델 미적재 → 부적합.

## 3. 배경 / GAP

- Nemotron-3-Nano = 31.6B total / 3.2B active, **Mamba-2 23층 / MoE 23층 / Attention 6층(KV-head 2)**. expert 9.98M param(10MB FP8), population **30GB FP8**, 모델 전체 **32GB FP8**(W4 실측 22GB). → AGX Orin 64GB는 **FP8 full-resident(offload 불필요)**, cross-review §0b의 "expert-offload-bound" 전제는 64GB에서 거짓 → 동기를 per-token expert read·state 트래픽으로 재정렬.
- **SSM state ≈ 48MB/agent**(23 Mamba층, heads64×dim64×d_state128 ≈ layer당 ~1MB FP16, context 무관 상수). **KV**(6층×KV-head 2): ~수십 KB/token → 32K context에서도 agent당 ~200MB 미만. → 전환 비용 지배항이 작은 KV가 아닌 **상수 state(48MB)**.
- **state-resident 한계가 넓다**: 48MB × 16 agent = **770MB** → AGX Orin 64GB에서 FP8 weight 32GB 상주 후에도 LPDDR-warm에 충분(host-cold tier는 수십 agent에서만 필요).
- **동기 정직화** (cross-review §0c): state 48MB는 KB-scale KV 대비 **크다**. "수십× 싸다"는 absolute size가 아니라 **(a) host round-trip 제거 + (b) prefill 재계산 O(prefix) 회피**에서 온다.
- Agent workload: agent turn 12~62(최대 786), inter-turn tool-wait **~100s 집중** ([arXiv:2605.26297](https://arxiv.org/abs/2605.26297), [arXiv:2505.09999](https://arxiv.org/abs/2505.09999)) → 48MB swap은 100s window에 충분히 hidden. decode-dominated **91.0~98.6%** ([arXiv:2605.26297]) → tool 복귀 시 prefill 재실행은 이 비율을 깨는 직격 손실. tool 실행 2~29% runtime ([arXiv:2605.26297]).
- per-token expert read **1.84GB/token** → ~9 tok/s 상한, EdgeReasoning [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) 4.7~9.3 TPS 정합.
- **비대칭 비용 anchor**: state restore = 48MB device-내 copy(상수) ≈ 48MB / 204.8 GB/s ≈ **234µs**(이론). KV recompute = O(prefix len) prefill 재실행(수십 ms~). hybrid는 KV가 작아 비대칭이 자명할 수 있으나, MOORING의 진짜 이득은 (a) host round-trip(PCIe/CPU 왕복 수 ms) 제거 (b) decode-dominated 91~98.6%를 깨는 prefill 회피 — 둘을 ablation으로 분리해 정직하게 보고.
- **state-resident 한계 scaling**: 48MB × N agent가 LPDDR-warm 예산을 채우는 지점이 동시성 천장(KV 한계가 아님). 64GB에서 weight 32GB + overhead 6~10GB 제외 후 ~22GB → state-only로 수백 agent 이론상 가능(실측은 expert union·KV와 경합 → TIDEMARK N\*가 실효 천장 결정).
- **기존 연구 한계**: vLLM V1 hybrid allocator(`MambaManager`/`MambaSpec`/`block_pool`)는 state를 1급으로 alloc하나 **agent-grained tiering·tool-aware swap 정책 부재**([PyTorch blog](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)) — substrate 존재, 정책 공백. Tokencake/Continuum/Pancake류는 모두 **attention-KV tiering**으로 hybrid SSM state 미관리. RELAY 원안의 SGLang dual-pool은 검증 substrate(vLLM)와 불일치 → vLLM 통일(ai-impl RELAY §M1).

## 4. Mechanisms

### M1 — Tiered LPDDR-warm SSM-state pool (host round-trip 제거)

#### 동작 원리
- **(scheme)** vLLM V1 `MambaManager` 위에 agent-grained 신규 레이어 `agent_state_pool` — 각 agent의 23 Mamba층 **conv1d state + SSM state를 단일 contiguous slot으로 번들**(`MambaSpec.shapes`가 이미 conv+ssm을 tuple로 보유 → 번들 alloc은 부분 기존, agent 단위 1급 객체화가 신규). 2-tier: **LPDDR-warm(default) / host-cold(수십 agent 초과 시만)**. swap은 `mamba_utils.batch_memcpy`(device-내 copy). API: `checkpoint(agent_id)`/`restore(agent_id)`.
- **(문제)** 현재 vLLM은 state alloc mechanism은 제공하나 agent 전환 시 state evict/recompute 정책 부재 → tool 복귀 시 prefill 재실행(decode-dominated 91~98.6% 붕괴).
- **(step-by-step)** (1) agent가 tool-call 진입(M2 신호) → conv+ssm 번들 state를 LPDDR-warm slot에 유지(demote 아님, host 미개입), (2) 다른 agent decode 진행, (3) tool 복귀 → device-내 copy로 restore, (4) agent 수가 LPDDR-warm 예산 초과 시에만 invocation-distance LRU로 host-cold demote.
- **(차별화)** Tokencake/Continuum은 attention-KV tiering — hybrid SSM state는 미관리. NIMBLE/RELAY 원안의 "host-cold 우선"을 **LPDDR-warm 우선**으로 정정(770MB는 LPDDR에 충분 → host round-trip 자체 제거).

#### 기대 효과 (Gain)
- **primary axis [Performance]**: agent 전환 latency −60~80% (host round-trip 제거가 주 동인, prefill 재계산 회피분).
- **secondary axis [Memory eff.]**: 동시 agent 수 1.5~2.5× (state-resident 한계까지).

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/v1/core/single_type_kv_cache_manager.py` | `MambaManager` (L854) | request-level Mamba state block 관리 | 위에 `agent_state_pool` 신규 레이어(agent-grained, 2-tier) | 신규 레이어 |
| `vllm/v1/core/kv_cache_interface.py` | `MambaSpec` (L563), `.shapes`/`.dtypes` (L566) | conv+ssm shape를 tuple 보유 | conv+ssm 번들을 agent 단위 1급 slot으로 alloc | extend |
| `vllm/v1/worker/mamba_utils.py` | `batch_memcpy` (L195), `collect_mamba_copy_meta` (L572) | device-내 state copy | checkpoint/restore device-내 swap에 활용 | reuse |
| `vllm/v1/kv_offload/simple_kv_offload/manager.py` | `SimpleCPUOffloadScheduler` (L67, MambaSpec-aware L215) | 부분 mamba block offload | multi-request 일반화 → host-cold demote | extend |
- **검증 출처**: 전 경로 ai-impl 리뷰 §0 [✅]/[✅ partial] @ `7c37096` (`MambaManager`/`MambaSpec`/`batch_memcpy`/`SimpleCPUOffloadScheduler`). NIMBLE 원안의 `mamba state manager`(미존재 클래스명)는 ai-impl F04로 정정 → 실 클래스 `MambaManager`. `MambaSpec.shapes` conv+ssm 번들은 RELAY M3가 신규로 가정했으나 부분 기존(ai-impl RELAY §M3 [✅ 경로]) → agent 단위 1급화만 신규.

#### 검증 시나리오 (Test Plan)
- **Unit test (분)**: 목적=checkpoint/restore round-trip이 state를 bit-exact 보존 / Input=1 agent decode 중 checkpoint→restore / Expected=restore 후 출력 토큰이 비-swap과 동일 / metric=state tensor L∞ diff=0 / 실행시간 ~5분 / 실패 시=conv/ssm 번들 contiguity 검토.
- **Mechanism-isolated test (시간)**: 목적=device-내 state restore latency vs KV recompute(O(prefix)) 비대칭 측정(핵심 가설) / Input=Nemotron-3-Nano FP8, agent 2~16, prefix 길이 sweep @ AGX Orin 64GB / Expected=state restore 48MB 상수 latency ≪ KV prefill / metric=nsys timeline상 restore µs vs prefill ms / 실행시간 ~6시간 / 실패 시=KV가 hybrid에서 작아 비대칭 자명 → ablation으로 host-round-trip 제거분 분리.

### M2 — Tool-window state swap scheduler (Continuum TTL과 비용 모델 차별)

#### 동작 원리
- **(scheme)** tool-call 신호를 **orchestrator↔vLLM 경계**(running/tool-suspended request 상태)에서 추출 — vLLM scheduler가 출력 토큰을 inspect하지 않음(layering 위반 회피). OpenAI tool-call streaming 단계(`entrypoints/openai/`)에서 감지 → engine hint → `agent_state_pool`에 swap 윈도우 등록(`idle_window_queue`).
- **(문제)** tool I/O 동안 가속기 유휴(tool 실행 2~29% runtime, inter-turn ~100s) → 그동안 다른 agent state restore+decode로 채움. Continuum은 KV를 TTL pin(시간 기반) — 본안은 state 객체를 swap(객체+비용 기반).
- **(step-by-step)** (1) request가 tool-suspended로 전이 → 해당 agent idle 등록, (2) idle window 동안 곧 복귀할/대기 agent state restore + decode, (3) tool 완료 콜백 → 재admission.
- **(차별화)** Continuum TTL(KV *유지*, eviction 비용=재prefill O(prefix)) vs MOORING swap(state *이동*, 비용=48MB device-copy 상수) — 동일 객체 아니므로 직접 경쟁 아님. Tokencake는 function-call stall 중 KV proactive offload(동일 시점이나 KV-only).

#### 기대 효과 (Gain)
- **primary axis [Performance]**: tool 복귀 TTFT −50% 이상 (idle window hiding 적중 시).
- **secondary axis [Memory eff.]**: idle window 활용으로 동시 agent throughput 유지.

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/entrypoints/openai/` (serving_chat 등) | tool-call streaming 핸들러 | tool-call 토큰 streaming만 | running/tool-suspended 라벨 추출 → engine hint | extend (light hook) |
| `vllm/v1/core/sched/scheduler.py` | schedule (L335), preempt (L464-L500) | request 상태 전이 | tool-suspended 시 `idle_window_queue` 등록 | extend |
| (M1 `agent_state_pool` 내) | `idle_window_queue` | 부재 | swap 윈도우 등록·복귀 콜백 | 신규 |
- **검증 출처**: `scheduler.py` schedule(L335)/preempt(L464-L500)는 ai-impl §0 [✅] @ `7c37096`. NIMBLE 원안의 "scheduler가 reasoning 출력 tool-tag 파싱"은 ai-impl NIMBLE §M2에서 **layering 위반 소지**로 flag → tool 신호를 `entrypoints/openai/` tool-call streaming 단계에서 감지 후 engine hint로 정정(ai-impl 대체 path).

#### 검증 시나리오 (Test Plan)
- **Unit test (분)**: 목적=tool-suspended 전이가 idle_window_queue에 정확히 등록/해제 / Input=mock tool-call 시퀀스 / Expected=suspend→register, complete→deregister / metric=queue 상태 일관성 / 실행시간 ~5분 / 실패 시=상태 전이 race 점검.
- **Mechanism-isolated test (시간)**: 목적=tool-window swap이 복귀 TTFT를 hiding하는지 / Input=BFCL/τ-bench tool-interleaved trace, agent 4~8 / Expected=tool 복귀 TTFT prefill 재실행 대비 −50%+ / metric=복귀 TTFT, idle window hit율, aggregate tok/s / 실행시간 ~8시간 / 실패 시=tool-wait window가 짧은 trace에선 이득↓ → tool-interleaved multi-turn으로 scope 명시.

### M3 — State-vs-KV 비대칭 admission/배치 (TIDEMARK N\* cross-share)

#### 동작 원리
- **(scheme)** `agent_state_pool`에 비대칭 비용 함수 — (state restore = 48MB 상수) vs (KV recompute = O(prefix len)). scheduler admission(scheduler.py:554)·preempt(L464-L500) 우선순위에 반영. **admission 자체는 TIDEMARK N\* 공식으로 단일화**; MOORING은 state footprint를 TIDEMARK에 입력으로 제공.
- **(문제)** 균일 eviction은 비싼 KV를 state처럼 버려 prefill 폭발. state pool + weight-resident 합이 LPDDR 천장 접근 시 어느 객체를 demote할지 결정 부재.
- **(step-by-step)** (1) 메모리 압력 시 후보 산출, (2) state-only agent(KV 작음) 우선 LPDDR-warm 유지·필요 시 host-cold demote, (3) 긴 prefix KV agent는 보존(재prefill 회피), (4) state+expert 통합 admission은 TIDEMARK 호출.
- **(차별화)** 기존 KV eviction(Autellix/Justitia류)은 state 비용을 모름 — hybrid 비대칭 명시가 신규. **cross-share 명시: admission 로직은 TIDEMARK 소유, MOORING은 state-cost provider.**

#### 기대 효과 (Gain)
- **primary axis [Memory eff.]**: 압력 시 올바른 demote 선택으로 prefill 폭발 회피.
- **secondary axis [Performance]**: 긴 prefix KV 보존 → 재prefill 회피.

#### 구현 변경점 (Code Changes)
| file path | Class · Function | as-is | to-be | 변경 type |
|---|---|---|---|---|
| `vllm/v1/core/sched/scheduler.py` | waiting loop (L554), preempt (L464-L500) | 균일 eviction/admission | state-vs-KV 비대칭 cost 함수 반영 | extend |
| (TIDEMARK admission sidecar) | N\* estimator | (TIDEMARK 소유) | MOORING state footprint를 입력 제공 | reference (TIDEMARK 소유) |
- **검증 출처**: `scheduler.py` preempt(L464-L500)/waiting loop(L554) ai-impl §0/NIMBLE §M3 [⚠️ cost 함수 신규] @ `7c37096`. admission 단일화는 cross-review §3.4(RELAY co-budgeting ↔ TIDEMARK 중복 → 단일화) 반영 — 본 mechanism은 state-cost provider 역할만, N\* 로직은 01-tidemark.md M2 참조.

#### 검증 시나리오 (Test Plan)
- **Unit test (분)**: 목적=비대칭 cost 함수가 state(상수) < 긴 prefix KV(O(prefix))를 올바르게 순위 / Input=합성 후보 set(state-only vs 긴 prefix) / Expected=state-only 우선 demote / metric=demote 선택 순위 정확도 / 실행시간 ~5분 / 실패 시=cost 함수 가중치 보정.
- **Mechanism-isolated test (시간)**: 목적=비대칭 admission이 균일 eviction 대비 prefill 폭발 회피 / Input=메모리 압력 multi-agent(긴 prefix 혼재) / Expected=재prefill 횟수 감소 / metric=재prefill 횟수, aggregate TTFT / 실행시간 ~4시간 / 실패 시=TIDEMARK N\* 단일화 경계 재정의(01-tidemark 연계).

## 5. 실험 플랜 (7-요소)

1. **Hardware**: 1차 **AGX Orin 64GB**(연구실 보유/확보 필요 — FP8 32GB full-resident + state pool 770MB + KV<200MB×agent + overhead ~6~10GB → 64GB 여유, agent 8~16 state-resident 실험). 보조 **Thor 128GB / DGX Spark 128GB**(확보 필요 — agent 수십 host-cold 불필요 headroom, 단 273 GB/s swap BW 천장). Orin NX 16GB·Nano 8GB는 모델 미적재 → 제외.
2. **Model**: primary **Nemotron-3-Nano-30B-A3B FP8**(직접 사용). proxy 보조 **Qwen3-Next-80B-A3B**(`qwen3_next.py`, gdn_attn — 80GB+ 필요 Thor) / **Falcon-H1-1.3B**(`falcon_h1.py`) / Bamba-9B(`bamba.py`) — 동일 hybrid Mamba state 구조(ai-impl 권고).
3. **Dataset / Workload**: agentic trace — **SWE-bench(Pro)**(긴 turn·context 성장) / **BFCL v4** / **τ-bench**(tool-interleaved multi-turn) + synthetic multi-agent(tool-wait window 길이·agent 수 sweep). tool 실행 2~29% 비율 보존.
4. **Tools**: vLLM fork(`7c37096` pin) + **nsys**(swap latency timeline 측정 — ncu는 Orin iGPU 미지원) + **tegrastats/jtop**(LPDDR util·junction temp). gem5 미사용.
5. **Ablation + Protocol**: factorial = {KV-centric baseline vs MOORING state-swap} × {LPDDR-warm only vs 2-tier} × {tool-window swap on/off} × {agent 1→16}. **핵심 ablation: host-round-trip 제거분 vs prefill 회피분 분리**(KV가 hybrid에서 작아 비대칭 자명할 위험). Baselines: **Tokencake [arXiv:2510.18586] (preprint)** KV proactive offload, **Continuum [arXiv:2511.02230] (preprint)** KV TTL pin, **vLLM Hybrid V1 ([PyTorch blog])** substrate, **SGLang(published system)** + EuroSys/ASPLOS state-management baseline 1편(peer-review 보강 의무).
6. **Implementation Steps** (Step별 dependency + 완료 판정 수치):
   - Step 1: HF config로 shared expert 수(=2 추정)·layer pattern 1회 확정 (완료=`n_shared_experts`/layer pattern 확정).
   - Step 2 (←1): `MambaManager` 위 `agent_state_pool` 신규 레이어 + checkpoint/restore device-내 copy (완료=Unit test bit-exact pass, request=agent 1:1 가정 smoke).
   - Step 3 (←2): tool-signal `entrypoints/openai/` 감지 → engine hint → `idle_window_queue` (완료=tool-suspend 전이 정확 등록).
   - Step 4 (←2,3): `SimpleCPUOffloadScheduler` multi-request 일반화 → host-cold 2-tier demote (완료=16 agent 초과 시 host-cold demote 동작).
   - Step 5 (←3,4): tool-window swap latency·복귀 TTFT 측정 (완료=state restore latency vs KV prefill 비대칭 nsys 측정, TTFT −50%).
   - Step 6 (←5): TIDEMARK admission 연계 비대칭 cost (완료=state footprint를 N\* 입력 제공, 통합 평가).
7. **Preliminary Analysis Metrics** (4단계):
   - **baseline repro**: 도구 nsys/tegrastats / 조건 Nemotron-3-Nano FP8 @ AGX Orin 64GB single-agent decode / 기대 baseline ~9 tok/s(EdgeReasoning 정합) / 목표 Δ=±10% repro.
   - **bottleneck attribution**: 도구 nsys timeline / 조건 KV-centric baseline tool-interleave / 기대 baseline tool 복귀 시 prefill 재실행이 TTFT 지배 / 목표 Δ=prefill 손실이 decode-dominated 91%+ 비율 깸 정량.
   - **upper bound**: 도구 nsys device-copy 측정 / 조건 48MB state copy @ 204.8 GB/s / 기대 baseline ~234µs 이론 swap / 목표 Δ=실측 swap이 tool-wait 100s window에 fully hidden.
   - **micro-pilot**: 도구 vLLM fork + state pool / 조건 agent 2~16 tool-interleaved / 기대 baseline KV-centric 전환 latency / 목표 Δ=전환 −60~80%, 동시 agent 1.5~2.5×, TTFT −50%.

## 6. 관련 연구 · 차별점

| 연구 | venue tag | 핵심 접근 | MOORING 차별 axis (self-contained) |
|---|---|---|---|
| Tokencake [arXiv:2510.18586](https://arxiv.org/abs/2510.18586) | preprint 2025 | KV-cache-centric multi-agent serving, function-call stall 중 KV proactive offload/upload(concurrent 50~60%) | MOORING은 **SSM state-object swap**(48MB 상수, host-sync-free, hybrid 한정 자산) — Tokencake는 attention-KV offload만 |
| Continuum [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) | preprint 2025 | tool window 동안 KV를 TTL pin(시간 기반 유지) | MOORING은 **state 객체를 swap 이동**(비용=48MB device-copy 상수) — Continuum은 KV 유지(eviction 비용=재prefill O(prefix)), 동일 객체 아님 |
| Pancake [arXiv:2602.21477](https://arxiv.org/abs/2602.21477) | preprint 2026 | multi-agent serving용 hierarchical memory(KV/index 중심) | MOORING은 **SSM recurrent state 1급화**(KV/index 아닌 Mamba state) |
| Compiler-First SSD [arXiv:2603.09555](https://arxiv.org/abs/2603.09555) | preprint 2026 | host-sync 없는 on-device state cache(단일 stream) | MOORING은 **multi-agent eviction/admission 정책**(단일 stream 기술 차용, 정책이 신규) |
| Agent Memory (Persistent Q4 KV) [arXiv:2603.04428](https://arxiv.org/abs/2603.04428) | preprint 2026 | agent별 KV disk persist/reload | MOORING은 **state(48MB device-copy)** — KV reload는 비쌈, state는 device-내 상수 |
| Software-Defined Agentic Serving [arXiv:2601.03197](https://arxiv.org/abs/2601.03197) | preprint 2026 | SDN-inspired control-plane orchestration | MOORING은 **data-plane state swap**(control-plane vs data-plane 추상화 레벨 상이) — 인용 권고 인접작 |
| vLLM Hybrid KV+State V1 ([PyTorch blog](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)) | framework | Mamba pool(request-level)/KV pool(token-level) allocator | substrate(baseline·구현 path) — agent-grained tiering·tool-aware swap 정책 공백 |

## 7. 리스크 & Decision-Tree 분기

- **Pass** (전환 latency −60% 이상 **AND** 동시 agent ≥1.5× **AND** state restore ≪ KV prefill 비대칭 실측 입증): MLSys 2027 본 투고. host round-trip 제거 + prefill 회피 동기 확정.
- **Below** (전환 −30~60% **OR** 동시 agent 1.2~1.5×): tool-interleaved multi-turn으로 scope 명시 제한, always-active agent 제외. ablation으로 host-round-trip vs prefill 회피 기여 분리 후 EuroSys/OSDI로 전환.
- **Critical-fail** (KV가 hybrid에서 작아 비대칭이 자명 **OR** device-copy가 tool-wait window보다 김): 핵심 가설("state restore ≪ KV") 폐기 → Tier-2 "state-vs-KV 전환 비대칭 비용 측정" letter(ISPASS/IEEE CAL)로 강등.
- **Outperform** (전환 −80% **AND** 동시 agent ≥2.5× **AND** TTFT −50% 이상): TIDEMARK admission·TOLLGATE BW와 full-stack 통합, OSDI flagship 격상.
- **임계값 backbone**: state restore = 48MB / 204.8 GB/s ≈ 234µs(이론), tool-wait ~100s 집중, KV agent당 <200MB(32K context), SLO TPOT ≤ 100ms.

## 8. Tier-2 scope 축소 variant

MOORING의 Tier-2 축소는 **"hybrid 모델 agent 전환의 state-vs-KV 전환 비대칭 비용 측정"** letter(ISPASS / IEEE CAL)다 — deterministic state-swap latency(device-copy, 48MB 상수)와 KV recompute(O(prefix)) 곡선을 agent 수·prefix 길이 sweep으로 first-to-report. 핵심 가설(state restore ≪ KV)이 실측에서 자명하거나 무너져도(Critical-fail 분기), 두 비용의 정량 곡선 자체가 hybrid serving의 actionable first-to-report이며 정책(M2/M3) 없이도 측정만으로 성립. NIMBLE/RELAY 공통 Tier-2 spinoff.

## 9. 약어 / 용어 풀이

- **SSM (State Space Model)**: 입력을 recurrent 상태 텐서로 압축해 O(n) 추론하는 시퀀스 모델. Nemotron-3-Nano의 Mamba-2 23층 — KV cache 없이 상수 크기 state.
- **Mamba-2**: selective SSM 2세대(multi-head, d_state=128). conv1d state + SSM state를 함께 보유, context 무관 ~48MB/agent. `MambaSpec.shapes`가 conv+ssm을 tuple로 1급 관리.
- **state (SSM state)**: Mamba-2 recurrent 상태 텐서(23층 ≈ 48MB/agent 상수). MOORING의 swap 대상 객체.
- **KV cache**: attention 6층의 key-value 캐시(KV-head 2, 극소). hybrid에서 context당 ~수십 KB/token → agent당 <200MB.
- **MoE (Mixture of Experts)**: 토큰마다 router가 일부 expert FFN만 활성화. Nano routed 128 + shared 2/층, per-token read 1.84GB.
- **GQA (Grouped-Query Attention)**: query head가 KV head 공유해 KV 축소. Nano attention 6층 KV-head=2.
- **LPDDR5**: edge SoC 저전력 DRAM. AGX Orin 204.8 GB/s, CPU·GPU 공유(UMA). state pool이 상주하는 "LPDDR-warm 계류지".
- **EMC (External Memory Controller)**: Jetson LPDDR 접근 중재 컨트롤러. tegrastats EMC util이 대역폭 포화 지표.
- **host round-trip**: device(GPU) state를 host(CPU) DRAM으로 내렸다 올리는 왕복. MOORING의 핵심 제거 대상(LPDDR-warm 유지로 회피).
- **orchestrator**: LangChain류 상위 agent 런타임(vLLM 밖). tool-call 신호를 orchestrator↔vLLM 경계(running/tool-suspended)에서 추출.
- **TTL pin (Time-To-Live pin)**: Continuum이 KV를 일정 시간 유지(고정)하는 시간-기반 정책. MOORING의 객체-이동 swap과 비용 모델 상이.
