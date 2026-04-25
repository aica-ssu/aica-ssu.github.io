# Token × Layer 2D Early-Exit Pareto for Multi-Consumer MoE Fingerprint Serving (LOOM)

> [← Session Overview](/research-wiki/2026-04/moe-fingerprint-security-serving/README.md)

> ## 📖 약어 / 핵심 용어 풀이 (R35)
>
> - **MoE** (Mixture of Experts) — 모델 일부만 sparse activate 하는 transformer 변형. ([Switch Transformer](https://arxiv.org/abs/2101.03961))
> - **Routing fingerprint** — MoE forward 1회로 얻는 `(L, E)` (layer × expert) 행렬, 어떤 expert 가 얼마나 활성됐는지의 prompt-level signature.
> - **Prefill / Decode** — LLM 추론 두 단계. Prefill = 입력 prompt 전체를 batch 1번에 forward, Decode = autoregressive 한 token 씩 생성.
> - **TTFT** (Time To First Token) — 사용자가 입력을 보낸 후 첫 응답 token 이 나오기까지 시간. Serving SLO 의 표준 지표.
> - **TPOT** (Time Per Output Token) — Decode 단계의 token 당 평균 latency. Generation throughput 직결.
> - **SLO** (Service Level Objective) — 서비스가 충족해야 할 정량 목표 (예: TTFT ≤ 500ms, TPOT ≤ 50ms). Serving paper 에서 throughput / latency trade-off 의 기준.
> - **EPRT** (End-of-Prefill Pooled Router Tap) — 본 idea Mechanism M1. vLLM model runner 에 추가하는 forward hook 이름. prefill 종료 시점에 `(L, E)` fingerprint 를 publish.
> - **Pareto frontier** — 다목적 최적화에서 한 목표를 더 좋게 하려면 다른 목표가 나빠지는 경계 곡선. 본 idea 는 (token prefix length k*) × (layer depth L_k) 2D 공간의 (F1 / FLOPs / latency) Pareto frontier 탐색. ([Wikipedia](https://en.wikipedia.org/wiki/Pareto_front))
> - **k-NN** (k-Nearest Neighbors) — 새 입력을 train pool 의 k 개 가장 가까운 점의 다수결 / 거리 평균으로 분류. 본 idea 의 detection consumer.
> - **FAISS IVF-PQ** — Meta vector similarity search. IVF (Inverted File) cluster + PQ (Product Quantization). 6144-dim × 253K vectors → 8MB index + ms-단위 query. ([공식 docs](https://github.com/facebookresearch/faiss/wiki/Faster-search))
> - **LEAP** (Layer-Expert Axis Pruning) — 본 idea Mechanism M4. fingerprint 의 per-dim variance + label MI (mutual information) 기준 top-D 차원만 유지하여 6144-dim → 256-dim 압축.
> - **MI** (Mutual Information) — 두 random variable 의 dependence 측정. `I(X;Y) = H(X) + H(Y) - H(X,Y)`. ([Wikipedia](https://en.wikipedia.org/wiki/Mutual_information))
> - **Fisher information** — 관측치가 확률 모델 parameter 에 대해 가지는 정보량. 본 idea 의 systems-theory contribution: fingerprint F 의 Fisher info 를 N task 에 분배하는 Pareto bound. ([Wikipedia](https://en.wikipedia.org/wiki/Fisher_information))
> - **MoE-Infinity** — USENIX ATC 2024. expert activation tracing + offload-aware prefetching. 본 idea 의 핵심 baseline. ([arXiv:2401.14361](https://arxiv.org/abs/2401.14361))
> - **vLLM Semantic Router Iris** — 2026-01 production merge. LoRA-shared classifier heads (jailbreak / domain / PII / fact-check). 본 idea Mechanism M3 와 framing 50-55% 겹침 → systems-theory proof 로 차별화.
> - **Continuous batching** — vLLM 의 핵심 throughput 기법. 동일 batch 안에서 prefill 과 decode 를 mixing 하여 GPU utilization 극대화.
> - **Roofline model** — Compute-bound vs memory-bound 분류 + 이론적 상한. arithmetic intensity (FLOP/byte) vs hardware peak. ([Williams et al. 2009](https://dl.acm.org/doi/10.1145/1498765.1498785))
> - **Nsight Compute counter** — NVIDIA GPU profiling. `lts__t_sectors_*` (L2 cache hit), `dram__throughput.*` (DRAM BW), `sm__warps_active.*` (occupancy) 등.

**🎯 Target Venue**: MLSys 2027 (18p) (primary) / ASPLOS 2027 (18p, secondary)
**📊 Score**: Novelty 6.0 / Diff 7.5 / Impact 8.5 / Feasibility 7.5 = 평균 **7.38**
**✅ 판정**: Accept (Major Revision 후 EMBER+THRESHOLD+TALLY merge)

---

## 1. 개요 (Overview)

MoE forward 1회에서 얻는 `(L, E)` router fingerprint 를 **shared observability substrate** 로 정의하고, 세 consumer (jailbreak detection + domain-aware expert residency + next-layer prefetch hint) 가 공유 read 하는 serving stack. 4 mechanism 통합 (M1 EPRT end-of-prefill pooled tap + M2 token×layer 2D early-exit Pareto + M3 multi-consumer fan-out + safety-aware admission + M4 LEAP+QIVF compressed index) 와 **information-theoretic Fisher information Pareto proof** 로 vLLM Semantic Router Iris (production engineering) 대비 systems-theory contribution.

**Metaphor 부속 (R30)**: "Loom" = 베틀. 실 하나(routing fingerprint)가 베틀에서 여러 가닥(detection + residency + prefetch consumer)으로 엮인다.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 다루는 축 | 한계 (LOOM 대비) |
|-----------|-----------|-------------------|
| **MoE-Infinity** ([arXiv:2401.14361](https://arxiv.org/abs/2401.14361)) [USENIX ATC 2024] ✓ | expert residency + prefetch | detection consumer 미포함, batch-level trace |
| **PreScope** ([arXiv:2509.23638](https://arxiv.org/abs/2509.23638)) | cross-layer prefetch | detection 미포함, best-effort async |
| **DuoServe-MoE** ([arXiv:2509.07379](https://arxiv.org/abs/2509.07379)) | affinity routing + cache | safety-aware admission 없음 |
| **Gimbal** ([arXiv:2602.21626](https://arxiv.org/abs/2602.21626)) [MLSys 2026] ✓ | prefix+KV+stickiness | expert footprint 미사용 |
| **vLLM Semantic Router Iris v0.1** (2026-01 production merge) | LoRA-shared classifier heads | serving cache/prefetch consumer 없음, LoRA 학습 필요 |
| **FJD** ([arXiv:2509.14558](https://arxiv.org/abs/2509.14558)) [EMNLP 2025 Findings] ✓ | dense first-token logit | MoE-specific 아님, 1D token axis only |
| **HSF** ([arXiv:2409.03788](https://arxiv.org/abs/2409.03788)) | hidden state final layer | 1D layer axis only |
| **OmniGuard** ([arXiv:2505.23856](https://arxiv.org/abs/2505.23856)) | dense internal repr safety | safety 단일, MoE routing 미사용 |

**GAP**: **MoE router fingerprint 를 shared observability substrate 로 정의 + Fisher information 분배 Pareto proof + token×layer 2D early-exit framework** 는 공개 없음. vLLM Semantic Router Iris 는 framing 유사하지만 serving cache/prefetch consumer 가 없고 LoRA 학습 필요 + 1D classifier head only.

---

## 3. 제안 기법 (Core Mechanisms, 4 mechanisms — critical merge ΔM=+1)

### M1: End-of-Prefill Pooled Router Tap (EPRT)

**① 추가되는 Scheme — Source Verified (R32)**:

vLLM v1 의 model runner 말미에 hook 등록. 각 MoE layer forward 직후 `router_logits` + `topk_weights` + `activation_count` 를 prefill token 축에 대해 mean pool → `(L, E)` tensor 로 aggregate. SchedulerOutput 에 `fingerprint_tensor` 필드 추가.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/worker/gpu_model_runner.py`
> ✅ source verified: vllm-project/vllm@`main` `vllm/model_executor/layers/fused_moe/layer.py` (FusedMoE 내부 router_logits 계산)
> ⚠️ source proposed: SchedulerOutput 의 `fingerprint_tensor` 필드 — 신규 추가 필요 (vLLM PR 후보)

**② 문제 + Workload evidence**:

vLLM PR #6734 ("router logit telemetry") 가 per-token overhead 이유로 merge 되지 않음. Per-prompt pooled (~4KB per prompt, L=27 E=160 기준) 은 이 overhead 우려를 우회. MoE-Infinity 는 batch-level trace 로 per-prompt 정렬 어려움.

**③ Step-by-step**:
1. `vllm/model_executor/layers/fused_moe/layer.py` 의 `FusedMoE.forward` 내 이미 계산된 `topk_weights` 를 per-prompt buffer 에 add.
2. Prefill 종료 시 buffer flush → `(L, E)` tensor.
3. `vllm/v1/worker/gpu_model_runner.py` 의 model 실행 후 hook 에서 SchedulerOutput.fingerprint_tensor 로 attach.
4. Non-blocking queue 로 publish.

**④ 차별화**: MoE-Infinity 는 scheduler 레벨만, model runner 와 decouple. PR #6734 per-token overhead 우회.

### M2: Token × Layer 2D Early-Exit Pareto (EMBER + THRESHOLD merged)

**① Scheme — Source Verified (R32)**:

vLLM v1 model runner 에 layer-wise hook 추가. 각 layer 마다 partial fingerprint margin 계산. token prefix length **k\* ∈ {1, 4, 16, 64, full}** × layer depth **L_k ∈ {2, 5, 10, 22, full}** 2D grid 에서 (F1 / FLOPs / latency) Pareto frontier 탐색. Adaptive layer budget runtime: margin > m_high 시 즉시 decision, < m_low 면 다음 layer.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/worker/gpu_model_runner.py`
> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/engine/core.py` (per-request state)

**② 문제 + evidence**: 사용자 실험: Qwen3 L22/48, DeepSeek L22/27, Qwen1.5 L02/24 모델별 optimal layer 다름. FJD 는 token k=1 only, HSF 는 final layer only — 2D joint 미개척.

**③ Step-by-step**:
1. 사용자 기존 trace 에서 token position × layer depth 2D grid 별 margin 측정 (재활용).
2. 각 (k*, L_k) 쌍의 F1 / FLOPs / latency 계산.
3. Pareto frontier 도출 + plot.
4. Runtime adaptive layer budget 구현 (margin threshold m_high / m_low).
5. Mid-prefill abort path (`raise EarlyReject` → scheduler skip decode).

**④ 차별화**: FJD (k=1) / HSF (layer=final) / HiddenDetect (layer, dense VLM) 모두 1D. 2D joint 는 공개 없음.

### M3: Multi-Consumer Fan-Out + Safety-Aware Admission

**① Scheme — Source Verified (R32)**:

`vllm/v1/engine/core.py` 에 `FingerprintBus` 추가. 3 consumer:
- (A) `JailbreakDetector` (k-NN on 253K pool)
- (B) `ExpertResidencyPolicy` (cluster → replica pin)
- (C) `PrefetchHintProducer` (next-layer top-8 expert prior)

Admission controller 가 (A) 결과로 reject / sandbox / full_decode 결정.

> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/engine/core.py`
> ✅ source verified: vllm-project/vllm@`main` `vllm/v1/core/sched/scheduler.py` (admission control 후크 위치)

**② 문제**: vLLM Semantic Router Iris 는 classifier heads 만 공유, serving-level consumer 미포함. 세 consumer 가 각자 forward 돌리면 3x cost.

**③ Step-by-step**:
1. Fingerprint ring buffer (size = max concurrent prompts).
2. 3 consumer lock-free read.
3. RequestMetadata union (cluster_id + jailbreak_score + prefetch_hint).
4. Scheduler admission: jailbreak_score > τ → reject / sandbox / full_decode.
5. Cluster_id → replica affinity routing.

**④ 차별화**: Semantic Parallelism within-replica only. Gimbal 은 prefix+KV+stickiness. vLLM Semantic Router Iris 는 classifier head shared but serving consumer 없음. 본 연구 = detection + serving 3-consumer joint + safety-aware admission first.

### M4: Compressed Index (LEAP + QIVF, TALLY merged)

**① Scheme**:

Offline: train pool variance + label mutual info → top-D=256 dim 만 유지 (LEAP). Online: FAISS IVF-PQ 256-dim × 32 subvector × 8-bit = 32 byte/vector. 253K pool → ~8MB.

> ⚠️ source proposed: `index/leap_qivf.py` — 신규 module.
> ✅ closest existing: facebookresearch/faiss `IndexIVFPQ` — standard usage.

**② 문제**: 기존 k-NN = 6144-dim × 253K × fp32 = ~6GB. QIVF → ~8MB, query 40ms → ≤3ms.

**③ Step-by-step**:
1. Per-dim variance + MI 계산.
2. Top-D selection.
3. FAISS IVF-PQ (nlist=256, PQ=32×8bit) 빌드.
4. Classifier F1 retention check (degrade ≤1%p 검증).
5. Full-precision re-ranker fallback (top-M re-score).

**④ 차별화**: FAISS IVF-PQ 는 general (MoE-unspecific). LEAP 의 MoE-specific layer-expert block structure → PCA 가 놓치는 interpretable axis (e.g., "L22 expert 47 = safety-critical" SAFEx-style label) 보존.

### Information-Theoretic Shared-Substrate Pareto (systems-theory contribution)

Fisher information 분배 정리: fingerprint F 의 N task 분배 Pareto 상한 = `I_max(task_i) ≤ trace(F^{-1}) · eigenvalue(task_correlation_matrix)`. 단일 read 로 3 consumer 가 동시 최적일 조건은 task correlation eigenvalue 분포에 의존. 수식 + proof 는 paper appendix. 이 proof 가 vLLM Semantic Router Iris 대비 contribution 의 핵심.

**Tier 구성**: physical 2-replica mini-cluster + software 3-consumer (≤4 안전).

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32)

### (1) Hardware
- Primary: RTX Pro 6000 96GB + RTX 5090 32GB (2-replica mini-cluster)
- Multi-replica extrapolation: trace replay + simulator (Orca / vLLM-v1 simulator)

### (2) Model
- Primary: Qwen3-30B-A3B BF16, DeepSeek-V2-Lite BF16
- Secondary: Mixtral-8x7B AWQ-4bit
- Robustness: Qwen1.5-MoE (edge case)

### (3) Dataset · Workload
Mixed: WildJailbreak 8K + MMLU 14K + WildChat 10K + LMSys-Chat 5K, Poisson arrival λ=2-10 req/s, jailbreak injection rate 1%/5%/10%. Metrics: detection F1, expert miss rate, p50/p99 decode latency, throughput, SLO attainment (TTFT ≤500ms, TPOT ≤50ms).

### (4) Tools
- vLLM fork + M1-M3 integration
- FAISS-GPU 1.10
- Nsight Compute (`lts__*`, `dram__*`), Nsight Systems
- vLLM `benchmark_serving.py`

### (5) Ablation · Baseline
**Baselines (11)**: vLLM v1 vanilla, vLLM+OmniGuard, vLLM+MoE-Infinity, vLLM+PreScope, vLLM+DuoServe-MoE, vLLM+Gimbal, vLLM+Semantic Router Iris, vLLM+FJD, vLLM+WildGuard, ProMoE, BuddyMoE.

Peer-reviewed: MoE-Infinity (ATC 2024) + Gimbal (MLSys 2026) + vLLM (SOSP 2023) + FJD (EMNLP 2025 Findings) = **4/11 = 36%** (R2 ≥25% 충족).

**Ablation**: (M1) / (M1+M2) / (M1+M3) / (M1+M2+M3+M4). K ∈ {5, 15, 50}, k-NN K ∈ {1, 15}, token k ∈ {1, 4, 16, 64, full}, layer L_k ∈ {2, 5, 10, 22, full}.

**Fallback**: M4 LEAP+QIVF F1 degrade 2%+ 시 full-precision re-ranker.

### (6) Implementation Steps (R31)

| Step | 의존성 | Component / File (R32) | 사용 API/Library | 완료 판정 |
|------|--------|---------|---------|---------|
| Step 1 | — | vLLM fork + EPRT (M1) hook. **`vllm/v1/worker/gpu_model_runner.py` ✅ + `vllm/model_executor/layers/fused_moe/layer.py` ✅** | torch + vLLM 0.19+ | (L, E) tensor in SchedulerOutput, unit test 8/8 |
| Step 2 | Step 1 | Token×Layer 2D Pareto (M2) — 기존 trace 재분석 + k×L_k grid | matplotlib, pandas | Pareto frontier 1st draft + figure |
| Step 3 | Step 2 | Multi-consumer fan-out (M3) + admission. **`vllm/v1/engine/core.py` ✅ + `vllm/v1/core/sched/scheduler.py` ✅** | asyncio, ring buffer | 3-consumer prototype + jailbreak admission unit test |
| Step 4 | Step 1 | LEAP + QIVF index (M4) | FAISS-GPU 1.10, `IndexIVFPQ` | 8MB index + ≤3ms query, F1 degrade ≤1%p |
| Step 5 | Step 1-4 | Mixed workload benchmark | vLLM `benchmark_serving.py` | throughput/latency 표 1 |
| Step 6 | Step 5 | Baseline 재현 (11 baselines) | each repo + vLLM integration | baseline table 완성 |
| Step 7 | Step 5 | Information-theoretic Pareto proof + figure | sympy, matplotlib | systems-theory section 완성 |
| Step 8 | Step 6 | Jailbreak injection 1%/5%/10% | WildJailbreak inject + vLLM bench | safety-aware admission 평가 표 |
| Step 9 | Step 6 | Mixtral / Qwen1.5 generalization | 추가 model run | cross-model validation 표 |
| Step 10 | Step 7-9 | Draft (18p) + polish | manual writing | submission-ready |
| Step 11 | Step 10 | Artifact evaluation 준비 | docker + reproducible scripts | artifact badge 신청 |

**참고 시간**: 약 12-14 weeks. hard deadline 아님.

### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 + counter | 측정 조건 | 기대 baseline | 개선 후 목표 |
|---|---|---|---|---|
| MoE-Infinity miss rate 재현 | Nsight Compute, batch trace | 512-token decode | 15-28% | 재현 ±3pp |
| EPRT hook overhead | Nsight Systems timeline | 100 prompt batch | <0.5ms | 유지 |
| FAISS IVF-PQ query latency | wall-clock | 253K pool, k=15 | 40ms (brute) | ≤3ms |
| Detection F1 | k-NN classifier output | WildJailbreak 8K | 94% | ≥93% (M4 압축 후) |
| Decode p50 latency | vLLM bench output | mixed workload | baseline | -15% |
| Expert cache miss rate | MoE-Infinity equivalent | jailbreak inject 5% | 15-28% | -20% |
| 2-replica throughput | Poisson λ=10 req/s | SLO TTFT ≤500ms | baseline | +25% |

**Preliminary Study 4-단계**: (i) baseline reproduction → (ii) bottleneck attribution (EPRT vs FAISS vs admission) → (iii) roofline (2-replica 이론적 throughput 상한) → (iv) micro-benchmark (mech 별 단독 gain additivity).

---

## 5. 예상 효과

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| Detection F1 | 94% | 93-94% | M4 LEAP+QIVF 압축 후 |
| Expert cache miss rate | 15-28% | -20-25% | M3 cluster-aware admission |
| Decode p50 latency | baseline | -15% | M2 early-exit |
| Memory | WildGuard +28GB | +<10MB (QIVF) | M4 LEAP+QIVF |
| 2-replica throughput | baseline | +25% | M3 admission + replica affinity |
| vLLM upstream PR path | — | M1 EPRT 후보 | 산업 impact |

---

## 6. (Tier-1 → Tier-2 변환 가이드)

LOOM 의 4 mechanism 중 단일 mechanism 만 분리 publication 시:
- **M2 단독 (token×layer 2D early-exit)** → **DAC 6p / IEEE CAL 4p**: detection 단독 축소, FLOPs / latency 측정 중심.
- **M4 단독 (LEAP+QIVF interpretability)** → **TALLY-Spinoff DATE 2027 4p WIP**, 별도 [tier2/02-tally-spinoff.md](/research-wiki/2026-04/moe-fingerprint-security-serving/tier2/02-tally-spinoff.md) 참조.

---

## 7. References

- MoE-Infinity [ATC 2024]: [arXiv:2401.14361](https://arxiv.org/abs/2401.14361)
- ProMoE: [arXiv:2410.22134](https://arxiv.org/abs/2410.22134)
- DuoServe-MoE: [arXiv:2509.07379](https://arxiv.org/abs/2509.07379)
- PreScope: [arXiv:2509.23638](https://arxiv.org/abs/2509.23638)
- BuddyMoE: [arXiv:2511.10054](https://arxiv.org/abs/2511.10054)
- Semantic Parallelism: [arXiv:2503.04398](https://arxiv.org/abs/2503.04398)
- Gimbal [MLSys 2026]: [arXiv:2602.21626](https://arxiv.org/abs/2602.21626)
- METRO: [arXiv:2512.09277](https://arxiv.org/abs/2512.09277)
- vLLM Semantic Router Iris v0.1 (2026-01 production blog)
- FJD [EMNLP 2025 Findings]: [arXiv:2509.14558](https://arxiv.org/abs/2509.14558)
- OmniGuard: [arXiv:2505.23856](https://arxiv.org/abs/2505.23856)
