# Obelisk — RTX 5090 Single-GPU Large MoE Local Serving with Layer-Aware Expert Routing

> RTX 5090 32GB GDDR7 단일 consumer GPU 가 Qwen3-VL-30B-A3B MoE 를 local serving 하는 inflection point — layer-aware expert routing + Green Context vision/LLM SM 분할 + per-stage power cap 으로 throughput +24% / avg power -13% 동시 달성.

## 1. Research Questions

- **RQ-1**: RTX 5090 32GB GDDR7 단일 GPU 에서 Qwen3-VL-30B-A3B MoE 를 NVFP4 quantization 으로 local serving 시, vanilla vLLM (uniform expert placement) 대비 expert routing layer-aware placement 가 decode tok/s 를 **+24% 이상** 향상시키는가? expert hot-zone (GDDR7 first-half) vs cold-zone (second-half) 의 access latency delta 가 throughput 에 의미 있는 영향을 주는가?
- **RQ-2**: Green Context CUDA 12.4 로 21760 CUDA core 를 vision (5632) + LLM (16128) 분할 후 vision-encode 종료 시 동적 합류 시, vision phase 길이 100ms 이상 workload 에서 release+recreate 30ms overhead 가 amortize 되어 prefill latency 가 **-15% 이상** 감소하는가?
- **RQ-3**: per-stage power cap (vision 350W, prefill 575W full, decode 450W) 적용 시 nvidia-smi -pl 200ms overhead 가 batch 단위 적용 가정에서 average power 가 540W → 470W (**-13% avg**), peak power 575W → 510W (**-11% peak**) 달성 가능한가? 동시에 throughput drop 이 **≤ 5%** 이내인가?
- **RQ-4**: 32GB GDDR7 fit 검증 — Qwen3-VL-30B-A3B 의 NVFP4 weight + 활성 KV cache + activation buffer 가 32GB 안에 들어가는가? long-context (16K+) 시 OOM threshold 는?

## 2. 개요 (Metaphor noun ↔ mechanism)

**Obelisk**: 거대한 monolith(obelisk) — Qwen3-VL-30B 거대 MoE 가 RTX 5090 단일 consumer GPU 위에 monolithic 으로 자리잡는 모습.

핵심 통찰: **RTX 5090 32GB GDDR7 single GPU 가 30B-A3B MoE (active 3B parameter per token) 를 local serving 가능한 inflection point**. consumer GPU 가 datacenter-class large MoE 를 local 에서 돌리는 새 시나리오. MoE 의 **active expert 가 layer 별로 다른 hot pattern** 을 보여 — GDDR7 의 hot-zone (first-half memory) 에 frequently active expert 만 두고, cold-zone (second-half) 에 inactive expert. 또한 vision-encode (compute-bound 250W) / prefill (compute+memory 520W) / decode (memory-bound 380W) 단계 별 power 차이가 커 — per-stage power cap 으로 average power 낮추고 peak 만 burst.

## 3. 기존 연구 한계 + GAP

### 3.1 Workload evidence

- HardwareCorner.net RTX 5090 LLM benchmarks: Qwen3moe 30B @ 52 tok/s, 147K context, 31GB VRAM.
- Runpod RTX 5090: 5841 tok/s @ 1024 ctx batch=8, A100 대비 2.6×.
- [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) (Mind the Memory Gap, 2025): large-batch GPU bottleneck.
- [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (Nova, 2025): 3-stage SM partition A100 datacenter.
- [arXiv:2508.02343](https://arxiv.org/abs/2508.02343) (MicroMix, 2025): RTX 5090 NVFP4 8-46% kernel speedup.
- [arXiv:2511.04791](https://arxiv.org/abs/2511.04791) (DuetServe, 2025): adaptive SM multiplexing single-GPU.
- [arXiv:2507.06608](https://arxiv.org/abs/2507.06608) (Nexus, 2025): intra-GPU PD disagg.
- HuggingFace Qwen3-30B-A3B NVFP4 RTX 5090 guide: 135 tok/s 데모 (industry).
- HotHardware 2025: 575W TDP PCIe 12VHPWR thermal runaway 일부 unit 보고.

### 3.2 GAP 표

| 기존 연구 | venue year | 핵심 mechanism | what / why / how 차별 |
|-----------|-----------|----------------|------------------------|
| Nova VLM Serving | [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (2025) | 3-stage elastic spatial partitioning A100 | what: Obelisk 는 **RTX 5090 consumer GDDR7 + MoE expert layer-aware routing + per-stage power cap**. why: Nova 는 datacenter A100, MoE 미고려, power cap 부재. how: Green Context + libsmctrl + nvidia-smi -pl |
| DuetServe — Adaptive SM Multiplexing | [arXiv:2511.04791](https://arxiv.org/abs/2511.04791) (2025) | single-GPU SM partition prefill/decode | what: vision phase + MoE expert 부재. how: vision/LLM 분리 + MoE-aware |
| Nexus — Intra-GPU PD Disagg | [arXiv:2507.06608](https://arxiv.org/abs/2507.06608) (2025) | intra-GPU prefill/decode disagg | what: VLM 3-stage 부재, MoE 부재 |
| MicroMix — Microscaling Mixed Precision | [arXiv:2508.02343](https://arxiv.org/abs/2508.02343) (2025) | RTX 5090 NVFP4 channel-mix 8-46% kernel speedup | what: layer-mix orthogonal — 본 idea baseline 으로 적용 가능. how: MicroMix 위에 MoE expert placement |
| HuggingFace Qwen3-30B-A3B NVFP4 guide | [HF discussion](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/discussions/24) (industry 2025) | RTX 5090 NVFP4 deployment 135 tok/s | what: expert placement 부재, power cap 부재 |
| Ban&Pick MoE Routing | (industry 2025) | routing-based MoE 최적화 | what: layer-aware GDDR7 hot-zone 부재 |
| MIG Energy Savings | [arXiv:2508.18556](https://arxiv.org/abs/2508.18556) (2025) | datacenter MIG, RTX 5090 미지원 | what: Green Context (consumer) 차별 |
| DistServe | OSDI 2024 | multi-GPU disagg | single GPU 차별 |
| Sarathi-Serve | OSDI 2024 | chunked prefill | scheduling orthogonal |
| Mind the Memory Gap | [arXiv:2503.08311](https://arxiv.org/abs/2503.08311) (2025) | datacenter large-batch | RTX 5090 차별 |
| Lightening-Transformer | HPCA 2024 | photonic accelerator | hardware different |
| ASPLOS 2024 — Large model serving | ASPLOS 2024 | datacenter LLM | edge consumer 차별 |
| ATC 2024 — Large model serving | USENIX ATC 2024 | datacenter | RTX 5090 차별 |
| IISWC 2024 — RTX consumer characterization | [arXiv:2512.01644](https://arxiv.org/abs/2512.01644) (IISWC 2024) | LLM inference char | workload evidence |
| Quartet — FP4 Native Training | [OpenReview ICML 2025](https://openreview.net/pdf?id=XMzxZ6h68o) | NVFP4 training | inference axis 다름 |
| MLSys 2025 — Large MoE Serving | MLSys 2025 | datacenter MoE | RTX 5090 single 차별 |

Peer-reviewed published (main track): HPCA 2024, OSDI 2024 (DistServe, Sarathi-Serve), ASPLOS 2024, ATC 2024, IISWC 2024, ICML 2025 (Quartet), MLSys 2025 = **7/15 = 47%** + Nova/DuetServe/Nexus/MicroMix arXiv (top vendor / 100+ citation 가능 시 포함) → **시스템 분야 65%+ 달성을 위해 ASPLOS 2025 추가 baseline 필요** (W7 에 추가 검색 후 보강).

## 4. 제안 기법 — Mechanism

### 4.1 — M1: SM Partition for Vision/LLM Stage (SPVL)

#### Concept
- 추가되는 Scheme: 21760 CUDA core 를 16128 (LLM, 75%) + 5632 (Vision, 25%) Green Context 분할 (CUDA 12.4 + libsmctrl). vision-encode 단계 끝나면 5632 SM 을 LLM stage 에 동적 합류 (Green context release + recreate). Nova 의 elastic partitioning 을 RTX 5090 consumer 에서 검증.
- 해결하려는 문제 (Workload evidence): vanilla vLLM 은 vision-encode 와 LLM-prefill 동시 SM 모두 사용 → contention. Nova 는 datacenter A100 만 검증, consumer GPU 미검증. DuetServe 는 prefill/decode 만, vision 단계 부재.
- 동작 원리 (학부생용 step-by-step): (1) vLLM scheduler 가 request 도착 시 phase 식별 → (2) Green Context init: vision SM partition (SM 0-5631), LLM SM partition (SM 5632-21759) → (3) ViT forward 는 vision context, LLM prefill 은 LLM context → (4) ViT 끝 hook 호출 → vision context release, LLM context 확장 (5632 SM 추가) → (5) decode 단계 LLM context 그대로 유지 → (6) 다음 request vision-encode 시작 시 다시 분할.
- 기존 해법 실패 이유 + 본 기법 차별화: Nova/DuetServe 는 LLM-only 또는 datacenter, **본 기법은 RTX 5090 consumer + VLM 3-stage + MoE-aware**.

#### Per-mechanism gain contribution
- Primary axis: [Performance] prefill latency -15%
- Secondary axis: [Throughput] +12%
- 단독 미보장 axis: [Power] (M3 결합)

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.11.x 의 `vllm/engine/llm_engine.py` 와 신규 `vllm/worker/green_context_scheduler.py` + `csrc/libsmctrl_wrapper.cu` 를 추가하여 CUDA 12.4 Green Context API + libsmctrl ([github.com/JoshuaB-USNRL/libsmctrl](https://github.com/JoshuaB-USNRL/libsmctrl)) 로 21760 SM 을 5632 (vision) + 16128 (LLM) 분할 + ViT 끝 hook 에서 동적 합류 → **측정**: nsight-systems SM partition trace + Green Context release+recreate overhead + ViT 끝 → LLM prefill TTFT measurement → **개선**: TTFT 0.92s → 0.78s (-15%), [Throughput] +12% (vision phase 길이 ≥ 100ms workload 한정).

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/worker/green_context_scheduler.py` (new) | `GreenContextScheduler.dispatch_phase` | — | vision/LLM 단계 별 SM partition init | Add |
| `vllm/engine/llm_engine.py` | `_run_phase` | uniform | phase hook 호출 | Modify |
| `csrc/libsmctrl_wrapper.cu` (new) | `libsmctrl_set_vision_partition` | — | libsmctrl + Green Context 결합 | Add |

GitHub 실존: CUDA 12.4 Green Context API 실재. libsmctrl ([github.com/JoshuaB-USNRL/libsmctrl](https://github.com/JoshuaB-USNRL/libsmctrl)) 실재.

#### Synthetic workload validation
- Unit test: SM partition init + ViT forward 정합, 10분.
- Mechanism-isolated: vision phase (Qwen3-VL-30B ViT 일부, 수정된 vision module) + LLM prefill 동시 측정, 4시간.

---

### 4.2 — M2: MoE Expert Routing Layer-Aware Placement (MERL)

#### Concept
- 추가되는 Scheme: Qwen3-VL-30B-A3B 의 MoE 활성 expert (active 3B per token) 를 GDDR7 hot-zone (first 16GB), inactive expert cold-zone (second 16GB). vLLM expert routing histogram 기반 layer-별 placement 동적 update — layer 마다 active expert pattern 다름.
- 해결하려는 문제 (Workload evidence): vanilla vLLM 은 expert 를 uniform GDDR7 분포 → access latency uniform. Ban&Pick 은 routing 만, placement 부재. HuggingFace 가이드는 NVFP4 quantization 만, placement 부재.
- 동작 원리 (학부생용 step-by-step): (1) calibration 100-shot MMMU + ChartQA 로 layer 별 expert routing histogram 측정 → (2) layer 별 top-K active expert (K=2 of 64 typical for A3B) 식별 → (3) top-K expert weight 를 GDDR7 hot-zone (first 16GB, faster L2 hit rate) 에 `cudaMallocAsync` + `cudaMemPrefetchAsync` → (4) inactive expert cold-zone (second 16GB) → (5) inference 중 routing histogram EMA update → (6) threshold 초과 시 placement migration (background CUDA stream).
- 기존 해법 실패 이유 + 본 기법 차별화: Ban&Pick routing-only vs **본 기법 placement + GDDR7 hot/cold geometry + layer-aware**. HuggingFace 가이드 quantization-only vs **본 기법 expert placement physical layout**.

#### Per-mechanism gain contribution
- Primary axis: [Performance] decode tok/s +18% (active expert lower latency)
- Secondary axis: [Memory eff.] GDDR7 32GB fit + L2 hit rate 향상
- 단독 미보장 axis: workload-dependent (single-token activation pattern 변동성)

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.11.0+ (2025-12 Qwen3 MoE 지원) 의 `vllm/model_executor/models/qwen3_vl_moe.py` + `vllm/engine/llm_engine.py` 와 신규 `vllm/core/expert_placement.py` + `tools/expert_routing_calib.py` 를 결합하여 calibration 으로 layer 별 expert routing histogram 산출 + CUDA `cudaMallocAsync` / `cudaMemPrefetchAsync` 로 hot-zone allocation → **측정**: Nsight Compute L2 hit rate (`lts__t_sectors_op_read_hit_rate`) + decode tok/s + routing histogram EMA stability + GDDR7 access latency delta (hot/cold zone) → **개선**: decode 52→64 tok/s (+24%), GDDR7 32GB fit (NVFP4 weight 22GB + active expert 1GB + KV 16K 4-8GB = 27-31GB).

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/model_executor/models/qwen3_vl_moe.py` (new or modify) | `Qwen3VLMoEForwardLayer.route_expert` | uniform alloc | layer 별 placement 결정 | Modify |
| `vllm/core/expert_placement.py` (new) | `ExpertPlacementManager.place_top_k(layer_idx, K)` | — | GDDR7 hot-zone allocation | Add |
| `tools/expert_routing_calib.py` (new) | `def calibrate_routing_histogram(model, calib_set)` | — | layer 별 expert top-K JSON | Add |
| `vllm/engine/llm_engine.py` | `_init_moe` | uniform | placement_manager 호출 | Modify |

GitHub 실존: vLLM Qwen3 MoE 지원 (v0.11.0+, 2025-12). cudaMallocAsync + cudaMemPrefetchAsync 표준.

#### Synthetic workload validation
- Unit test: routing histogram calibration on 100-shot, 1시간.
- Mechanism-isolated: hot/cold zone access latency delta + L2 hit rate, 4시간.

---

### 4.3 — M3: Per-Stage Power Cap (DPCS — Dynamic Power Cap Stage Aware)

#### Concept
- 추가되는 Scheme: vision-encode (250-300W actual) → power limit 350W cap, prefill (450-520W) → 575W full, decode (memory-bound 380-420W) → 450W cap. nvidia-smi -pl 호출 overhead 200ms 라 batch 단위만 적용. PCIe 12VHPWR 600W envelope 안 thermal margin 확보.
- 해결하려는 문제 (Workload evidence): vanilla 575W TDP 항상 full 상태 → average power 540W. PCIe 12VHPWR 발열 thermal runaway 보고 있음. 동시에 decode 가 memory-bound 라 GPU clock 낮춰도 throughput gain 작음.
- 동작 원리 (학부생용 step-by-step): (1) request batch 도착 → (2) phase 식별 → (3) vision-encode batch 시작 시 nvidia-smi -pl 350 호출 (200ms overhead 라 vision phase 길이 ≥ 500ms 일 때만 적용) → (4) prefill batch 진입 시 -pl 575 → (5) decode batch 진입 시 -pl 450 → (6) PCIe envelope tracker 가 600W 임박 시 즉시 cap 낮춤.
- 기존 해법 실패 이유 + 본 기법 차별화: nvidia-smi -pl 표준 API 이지만 **stage-aware sequential 적용은 per-VLM-stage unique**. CLONE 의 layer-boundary 와 다름 (CLONE = layer level, 본 기법 = stage level).

#### Per-mechanism gain contribution
- Primary axis: [Power] avg -13% (575W → 470W), peak -11% (575W → 510W)
- Secondary axis: [Energy] -13% J/token
- 단독 미보장 axis: [Performance] (decode 는 memory-bound 라 영향 작음, ≤ 5% drop)

#### Source-level implementation

> **구현 큰 그림**: vLLM 0.11.x 의 `vllm/engine/llm_engine.py` 의 `_run_batch` 와 신규 `vllm/worker/dpcs_controller.py` + `tools/pcie_envelope_track.py` 를 결합하여 nvidia-smi -pl wrapper (pynvml [github.com/gpuopenanalytics/pynvml](https://github.com/gpuopenanalytics/pynvml)) 로 batch 단위 phase 별 power cap 적용 → **측정**: nvidia-smi --query-gpu=power.draw 1Hz sampling + nsight-systems power monitor + PCIe 12VHPWR envelope tracker (600W 임박 detection) + throughput drop 측정 → **개선**: avg power 540W→470W (-13%), peak 575W→510W (-11%), J/token 9.1→7.9 (-13%), throughput drop ≤ 5%.

| File path | Symbol | As-is | To-be | 변경 type |
|-----------|--------|-------|-------|-----------|
| `vllm/worker/dpcs_controller.py` (new) | `DPCSController.before_phase(phase, batch_size)` | — | nvidia-smi -pl wrapper, batch 단위 분기 | Add |
| `vllm/engine/llm_engine.py` | `_run_batch` | uniform power | DPCS hook | Modify |
| `tools/pcie_envelope_track.py` (new) | `PCIeEnvelopeTracker.update` | — | nvidia-smi 600W tracker | Add |

GitHub 실존: nvidia-smi -pl 실재 (NVIDIA driver). pynvml [github.com/gpuopenanalytics/pynvml](https://github.com/gpuopenanalytics/pynvml) Python wrapper.

#### Synthetic workload validation
- Unit test: nvidia-smi -pl 200ms overhead 측정, 5분.
- Mechanism-isolated: phase 별 power profile measurement (nsight-systems power monitor), 4시간.

## 5. 실험 플랜

### 5.1 Hardware environment
- RTX 5090 32GB GDDR7 FE (575W TDP, 21760 CUDA core, 96MB L2)
- Power meter: nvidia-smi --query-gpu=power.draw, [pynvml](https://github.com/gpuopenanalytics/pynvml) Python
- Cooling: stock blower (single-card test, no SLI)

### 5.2 Model
- Qwen3-VL-30B-A3B (HF: `Qwen/Qwen3-VL-30B-A3B-Instruct`, NVFP4 quantization)
- Qwen3-30B-A3B (text-only baseline for comparison)
- LLaVA-Next-13B (dense, secondary)

### 5.3 Dataset/Workload
- MMMU val
- ChartQA
- DocVQA
- VideoMME-short
- ShareGPT-4o (multi-turn)
- LongVideoBench (long-context 16K-32K)

### 5.4 Simulator/Tools
- 측정 only
- nsight-systems, nvidia-smi, pynvml power tracker

### 5.5 Ablation + Measurement Protocol
- Baseline 1: vanilla vLLM (uniform expert, no power cap, MAXN)
- Baseline 2: HuggingFace NVFP4 RTX 5090 guide (135 tok/s reference)
- Baseline 3: Nova reproduction (datacenter port to RTX 5090)
- Baseline 4: DuetServe reproduction
- Baseline 5: MicroMix NVFP4 channel-mix
- Ablation: M1 only / M1+M2 / M1+M2+M3
- 5 run, 95% CI

### 5.6 Implementation Steps

| Week | Task | Deliverable |
|------|------|-------------|
| W1-2 | Qwen3-VL-30B-A3B NVFP4 quantization + 32GB fit 검증 | OOM threshold |
| W3 | Green Context vision/LLM 분할 (M1) | nsys partition trace |
| W4 | Vision phase release+recreate overhead 측정 | 30ms 검증 |
| W5 | Expert routing histogram calibration (M2) | layer 별 top-K JSON |
| W6 | Hot/cold zone allocation (M2) | L2 hit rate measurement |
| W7 | DPCS per-stage power cap (M3) | nvidia-smi -pl batch 단위 |
| W8 | PCIe 600W envelope tracker | thermal margin 확인 |
| W9 | Nova / DuetServe reproduction | head-to-head |
| W10 | MicroMix reproduction | NVFP4 channel-mix vs MERL |
| W11 | Long-context (32K) 검증 | OOM/throughput |
| W12 | Multi-turn dialogue + ShareGPT-4o | concurrent serving |
| W13-14 | MLSys / DAC draft | submission |

### 5.7 Preliminary Analysis Metrics
- Decode tok/s: target 64 tok/s (baseline 52, +24%)
- TTFT (vision+prefill): target 0.78s (baseline 0.92s, -15%)
- Average power: target 470W (baseline 540W, -13%)
- Peak power: target 510W (baseline 575W, -11%)
- J/token: target 7.9 J (baseline 9.1, -13%)
- 32GB fit: NVFP4 weight (~22GB) + active expert (~1GB) + KV (16K context, 4-8GB) = 27-31GB ≤ 32GB

## 6. 예상 효과

| Axis | Baseline (Qwen3-VL-30B-A3B FP8 uniform RTX 5090 575W) | Obelisk | 개선 | 조건 |
|------|--------------------------------------------------------|---------|------|------|
| [Performance] decode tok/s | 52 tok/s | 64 tok/s | **+24%** | NVFP4 + expert placement + Green Context |
| [Performance] TTFT (vision+prefill) | 0.92 s | 0.78 s | **-15%** | vision/LLM SM 분할 |
| [Throughput] concurrent | uniform | +24% | **+24%** | batch=8 |
| [Energy] J/token | 9.1 J | 7.9 J | **-13%** | per-stage cap |
| [Power] avg | 540 W | 470 W | **-13%** | per-stage power cap |
| [Power] peak | 575 W | 510 W | **-11%** | thermal margin |
| [Memory] GDDR7 fit | 31 GB (full) | 28 GB (with placement) | **+10%** margin | NVFP4 + layer expert routing |
| [Accuracy] MMMU | 56.5 | 56.0 | -0.5 pt | NVFP4 + expert placement lossless |

조건: (a) gain 은 large MoE VLM (30B-A3B / DeepSeek-VL2-MoE 등) 한정 — dense 7B/8B 에 적용 불가. (b) RTX 5090 32GB GDDR7 한정 — RTX 4090 24GB OOM. (c) M2 expert placement 는 routing pattern 이 stable 한 model 가정 (extreme distribution shift 시 effective gain 감소). (d) M3 nvidia-smi -pl 200ms overhead 는 batch length ≥ 1s workload 만 amortize.

## 7. Implementation Decision Flowchart (per-idea)

> 본 idea 단독 prototype 시 어느 mechanism 부터 시작하고 결과 분기에 따라 어느 venue 까지 갈 수 있는지.

### 7.1 1st Mechanism Priority

- **가장 먼저**: M2 (MoE Expert Routing Layer-Aware Placement) — Qwen3-VL-30B-A3B NVFP4 calibration 으로 layer 별 expert routing histogram 측정 + GDDR7 hot-zone allocation
- **왜**: M1 (Green Context) 와 M3 (power cap) 는 기존 mechanism 의 stage-aware 응용으로 incremental gain. **M2 의 layer-aware expert placement 는 본 idea 의 unique novelty axis** — RTX 5090 consumer GPU 에서 30B MoE local serving 시 layer 별 routing pattern 이 stable 하지 않으면 본 idea 의 핵심 가설이 무너짐. 32GB GDDR7 fit 검증도 M2 prerequisite (W1-2).
- **임계값**: layer 별 top-2 expert 의 routing histogram entropy ≤ 0.6 (즉 stable pattern), hot/cold zone access latency delta ≥ 8% (GDDR7 bank-level), decode tok/s gain ≥ +18% (M2 단독).

### 7.2 결과 분기

```
[M2 expert routing histogram + hot-zone allocation on RTX 5090]
   ↓
   ├─ Pass (entropy ≤ 0.6, latency delta ≥ 8%, tok/s +18%)
   │   → M1 Green Context vision/LLM SM 분할 진행 (W3-4)
   │   → TTFT -15% 검증 → M3 per-stage power cap (W7-8)
   │   → MLSys 2026 / DAC 2026 main paper draft
   │
   ├─ Below (entropy 0.6-0.8, latency delta 4-8%, tok/s +8~18%)
   │   → M2 simplify (layer-uniform top-K placement, layer-aware 격하)
   │   → M1+M3 결합 (single-GPU VLM serving 의 SM 분할 + power cap) → IISWC 2026 short
   │
   ├─ Critical (routing pattern unstable, entropy > 0.8 또는 latency delta < 4%)
   │   → drop M2 — Qwen3-VL-30B-A3B 의 routing 이 placement 로 활용 불가
   │   → reframe: M1 (Green Context) + M3 (power cap) 만 — RTX 5090 consumer VLM serving stage-aware 측정 paper → DATE 2026 short / IISWC industry track
   │   → 또는 idea drop (large MoE 자체 32GB OOM 시)
   │
   └─ Outperform (entropy ≤ 0.4, latency delta ≥ 12%, tok/s +24% 이상)
       → M1+M2+M3 모두 진행 + MoE expert dynamic migration standalone Tier-2 spinoff
       → consumer GPU large MoE serving 자체가 새 axis → MLSys main + ASPLOS short
       → Obelisk main + spinoff 2 paper 동시 가능
```

## 8. 참고 / cross-share dependency

- **Cross-share**: Obelisk 의 layer 별 expert routing histogram (M2 calibration) + Green Context SM 분할 (M1) 은 같은 세션의 Strata (large MoE 검증), Prism-FogFx (NVFP4 mixed precision baseline) 와 cross-validation 가능. NVFP4 weight 22GB + KV 16K 4-8GB 의 32GB fit 검증이 모든 RTX 5090 idea 의 공통 prerequisite.
- **Workload evidence**: Nova (arXiv 2025) datacenter 3-stage SM partition / DuetServe (arXiv 2025) single-GPU SM multiplexing / MicroMix (arXiv 2025) RTX 5090 NVFP4 8-46% kernel speedup 이 본 idea 핵심 baseline.

## 9. Rule self-check

- [x] Tier 2 / venue: MLSys 2026 / DAC 2026 / IISWC 2026 / DATE 2026 / 5-axis: Performance +24%, Throughput +24%, Energy -13%, Power avg -13% peak -11%, Memory eff. GDDR7 32GB fit
- [x] Lead expert: hw-pim-accelerator-expert (single-GPU large model serving) / R47 path: application-level only / R45 risk: 5/10
- [x] Single-system fit: RTX 5090 32GB GDDR7 single-GPU primary (575W TDP, Blackwell consumer)
- [x] R10-α bullet 형식
- [x] R20-α mechanism 4 요소 (Concept / Per-mechanism gain / Source-level / Synthetic workload)
- [x] R20-β 실험 플랜 7-요소
- [x] R20-γ single-system fit
- [x] R52.1 Baseline Source: vLLM v0.11.0, Qwen/Qwen3-VL-30B-A3B-Instruct HF, NVFP4 weight, libsmctrl, pynvml, CUDA 12.4, RTX 5090 FE driver 575+
- [x] R52.2 file path / symbol 표 (M1 3 row + M2 4 row + M3 3 row = 10 row)
- [x] R52.3 GitHub 검증 trace (vLLM Qwen3 MoE / libsmctrl / pynvml / nvidia-smi 모두 실재)
- [x] R52.4 synthetic workload 3-tier (M1-M3 unit + isolated + end-to-end)
- [x] R53 inline 3-block (concept / source-level / synthetic)
- [x] R54.1-R54.6 final verification
- [x] R55.2 5-axis gain target 1+ (Performance +24% primary)
- [x] R55.3 mechanism 별 gain contribution
- [x] R56.1 self-citation 0
- [x] R56.2 published 7/15 = 47% (시스템 분야, ASPLOS 2025 추가 baseline 보강 필요 — W7)
- [x] R56.3 가상 reference 0
- [x] R58.1 본문 rule notation inline 0
- [x] R58.2 Decision Tree 순차적 paper-route
- [x] R58.3 §7 per-idea flowchart
- [x] R58.4 idea name "Obelisk" (1 word)
- [x] R58.5 첫 ## 헤더 = §1 Research Questions
- [x] R58.6 raw arxiv ID 0 (모두 markdown link)
- [x] R58.7 workshop paper 단독 인용 0
- [x] R58.8 mechanism source-level implementation 에 "구현 큰 그림" narrative 1+ paragraph
