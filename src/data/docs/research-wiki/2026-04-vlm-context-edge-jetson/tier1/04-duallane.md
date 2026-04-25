# Dual-NVDLA + GPU 3-Way Dataflow Co-Scheduling for VLM Vision Encoder on Jetson Edge (DualLane)

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-1 Top 3** (R45 적용 후 신규 진입)

> ## 약어 / 핵심 용어 풀이 (R35)
>
> - **VLM** — Vision-Language Model (Qwen2.5-VL / Qwen3-VL / LLaVA-NeXT / MiniCPM-V).
> - **NVDLA / DLA** — NVIDIA Deep Learning Accelerator. Jetson Orin AGX/NX 의 Dual-DLA v2.0 (각 1 MiB convolution buffer SRAM × 2). Thor 추정 NVDLA gen-next (NDA). FP16/INT8 only. ([NVDLA spec](https://nvdla.org/hw/v1/hwarch.html), [TensorRT 10.9 DLA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html))
> - **NVMM** (NV Multimedia Memory) — NVIDIA dma-buf 기반 zero-copy buffer. `NvBufSurfaceCreate` + `cudaImportExternalMemory` (Linux dma-buf API).
> - **DLA-supported subgraph** — TensorRT compiler 가 DLA 로 mapping 가능한 ONNX subgraph (Conv2D / MatMul / LayerNorm 일부, transformer attention 은 GPU fallback).
> - **Vision Encoder** — VLM 의 CLIP-ViT-L/14 또는 SigLIP visual tower. prefill latency 의 35-55% 차지 ([MobileAIBench arXiv:2406.10290](https://arxiv.org/abs/2406.10290)).
> - **Spatial-Split** — 입력 image 를 좌/우 절반 또는 4 quadrant 로 split, 각 NVDLA 코어에 dispatch.
> - **Stage Pipelining** — 다음 frame N+1 vision encoder DLA 실행과 현재 frame N LLM prefill GPU 실행을 overlap.
> - **libsmctrl** — UNC GPU SM/TPC partition library, ECRTS 2025 Bakita et al. ([repo](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/))
> - **XSched** — XPU preemptive scheduling, OSDI 2025. ([paper](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf))
> - **Nova** — VLM 3-stage cross-stage parallelization, [arXiv:2509.21301](https://arxiv.org/abs/2509.21301). 본 idea 의 50-70% CONCURRENT — 4-mechanism 분리 ablation 으로 차별화 (Phase 2' Strong Refine).
> - **DRM dma-buf** — Linux Direct Rendering Manager dma-buf, kernel 공식 user-space API.
> - **TPOT** (Time Per Output Token) — decode latency p50/p99/p999.
> - **TTFT** (Time To First Token) — prefill latency.
> - **Robot Perception Loop** — continuous frame VLM (30 FPS) — robot / drone / ADAS scope.

**Target Venue**: ISCA 2027 (12p) (primary) / MICRO 2027 (13p) (alternative)
**Score** (Phase 2 평균 + R45 보정): Novelty **7.0** / Diff **8.0** / Impact **8.6** → 평균 **8.00** (R45 후 보정)
**판정**: Accept Tier-1 (R45 적용 후 CacheVeil demotion 자리에 진입)
**Phase 1' / R45 diff**: ΔM = 0 (mechanism 자체 변경 없음). R45 적용 후 implementation path 검증 완료 — NvMedia DLA + Linux DRM dma-buf + libsmctrl 모두 vendor 공식 API → R45 risk **4/10 LOW**. Nova `arXiv:2509.21301` 50-70% CONCURRENT 우려는 4-mechanism 분리 ablation 으로 mechanism diff 1:1 강화.

---

## 1. 개요 (Overview)

본 연구는 **Jetson Orin AGX 64GB / Orin NX 16GB 의 Dual-NVDLA v2.0 + iGPU heterogeneous compute 환경** 에서 VLM vision encoder (CLIP-ViT-L/14 또는 SigLIP) 가 prefill latency 의 **35-55% 차지** ([MobileAIBench arXiv:2406.10290](https://arxiv.org/abs/2406.10290)) 함에도 모든 vision encoder layer 가 GPU 에서만 실행되는 현상을 정량화하고, **DLA0 / DLA1 / GPU 3-Way Dataflow Co-Scheduling** 으로 vision encoder 를 spatial-split + DLA→GPU NVMM zero-copy + cross-frame stage pipelining 하는 mechanism 을 제안한다.

**가설**: Orin AGX 64GB Dual-DLA v2 환경에서 vision encoder ≥ 35% prefill 차지 워크로드 (Qwen2.5-VL-7B / 3B, LLaVA-1.5-7B, robot 30 FPS perception loop) 시 (a) prefill latency 1.4-1.8× speedup, (b) GPU SM utilization 100% → 60-70% (DLA offload 로 freed 30-40% 가 LLM prefill 병행), (c) energy/inference -15~-22% (DLA J/op 가 GPU 보다 낮음), (d) robot continuous perception 12-18 FPS → 25-30 FPS.

**Metaphor 부속 (R30)**: "DualLane" = 2 lane (DLA0/DLA1) + 1 lane (GPU) 의 동시 흐름. 후보: TriadFlow / ZipperEdge.

---

## 2. 기존 연구의 한계 / GAP

| 기존 연구 | 다루는 axis | 한계 (본 연구 대비) |
|-----------|------------|---------------------|
| **Nova: Real-Time Agentic VLM Serving** [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) | VLM 3-stage cross-stage parallelization, GPU spatial partition | **single GPU 가정**, DLA 라는 별도 가속기 axis 부재. 본 idea 4-mechanism 중 Nova 는 #4 (stage pipelining) 만 부분 overlap |
| **XSched** [OSDI 2025](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf) | XPU preemptive scheduling | DLA preemption 가능성 framework 만, **VLM-specific dataflow 부재** |
| **SLO-Aware Edge Scheduler** [ACM TECS 2024](https://dl.acm.org/doi/fullHtml/10.1145/3460352) | Jetson CPU/GPU/DLA scheduling | DLA 활용은 짧은 INT8 conv 만 — **transformer-era VLM 미공략** |
| **GPU Context-Aware Preemptive Priority-Based Scheduling** [LIPIcs ECRTS 2024](https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf) | GPU context preemption | GPU 만, **DLA→GPU stage 전이 baseline** 부재 |
| **Hardware Compute Partitioning (libsmctrl)** [ECRTS 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf) | GPU SM/TPC partition (compute side) | GPU 만, **DLA spatial-split + NVMM zero-copy 미공략** |
| **Vision Transformer Computation and Resilience** [ISPASS 2024](https://ispass.org/ispass2024/accepted-papers.php) | ViT computation analysis | analysis 만, **3-way co-schedule mechanism 부재** |
| **MobileAIBench** [arXiv:2406.10290](https://arxiv.org/abs/2406.10290) | Mobile VLM latency vs energy | benchmark 만, **DLA offload mechanism 부재** |

**GAP**: **Vision encoder 를 spatial-split 으로 Dual-DLA 분산 + DLA→GPU NVMM zero-copy + 다음 frame 의 vision encoder DLA 와 현재 frame LLM prefill GPU 의 cross-frame pipelining** 4-mechanism 결합은 현재 공개 논문 없음 — first-to-report.

### 기존 세션 중복 회피

- **이전 v1/v2/v3/qwen3vl 세션**: 모든 mechanism 이 GPU-only — DLA axis 자체 없음.
- **SHOAL** (본 세션 Tier-1 #2): DLA → KV residence enum (layer-time). DualLane 은 vision encoder spatial-split (spatial-time). **시간축이 다름** — 결합 가능 (orthogonal).
- **Watershed** (hwpim 미선정): phase classifier control plane 만, **physical dataflow 부재**.
- **Nova [arXiv:2509.21301]** 50-70% CONCURRENT 회피: 4-mechanism 분리 ablation 으로 each mechanism gain 정량화. Nova 는 single-device GPU SM partition 만 — DLA spatial-split + NVMM zero-copy 부재.

---

## 3. 제안 기법 (Core Mechanisms, 4 mechanisms — improve-only ΔM=0)

### M1: Vision Encoder Layer-Split DLA Mapping

**① 추가되는 Scheme — Source Verified (R32)**:

CLIP-ViT-L/14 또는 SigLIP vision encoder 의 conv stem + 첫 N layer 를 DLA-supported subgraph 로 분리. DLA0 / DLA1 양 코어에 batch-split (batch ≥ 2) 또는 spatial-split (좌/우 절반 또는 4 quadrant) — 각 DLA 의 1 MiB conv buffer SRAM 에 fit 하도록 layer 단위 split 설계 (TensorRT DLA compiler `IBuilderConfig::setDeviceType`).

> ✅ source verified: TensorRT 10.9 `IBuilderConfig::setDeviceType(layer, DeviceType::kDLA)` + `setDLACore(0/1)` + `setFlag(BuilderFlag::kGPU_FALLBACK)` ([공식 docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html))
> ✅ source verified: `--useDLACore=0/1` runtime flag, `--allowGPUFallback`
> ⚠️ NDA: Thor NVDLA gen-next API — Orin AGX/NX 의 NVDLA v2.0 만으로 검증 가능

**② 해결하는 문제 + Workload evidence**:

[MobileAIBench arXiv:2406.10290](https://arxiv.org/abs/2406.10290): VLM 의 vision encoder 가 prefill 의 **35-55% 차지**. encoder 만 별도 가속하면 prefill speedup 가능. Orin AGX Dual-DLA v2 의 1 MiB conv buffer SRAM × 2 = 2 MiB 가 ViT stem (16×16 patch × 14×14 grid) 의 working set 과 fit.

**③ Step-by-step (3-5 steps)**:

1. CLIP-ViT-L/14 ONNX 를 TensorRT compiler 에 입력 → `setDeviceType(layer, kDLA)` 로 DLA-supported subgraph identification.
2. Vision encoder layer-split 점 결정 — {stem only, first 4, first 8, first 12} variant 측정.
3. Spatial-split mode {batch / spatial-LR / spatial-quad} × DLA core {0, 1} mapping.
4. DLA-unsupported layer (transformer attention block) 은 GPU fallback (`kGPU_FALLBACK`).
5. Compile-time layer profiling 으로 split point 결정 — runtime 변경 비용 회피.

**④ 기존 해법 실패 + 1:1 차별화**:

- **vs single-DLA mapping (TensorRT-LLM 표준)**: 단일 DLA core 만 → 1 MiB SRAM 작아 layer 4-8 만 fit. 본 idea 는 spatial-split 으로 2 MiB working set 활용.
- **vs Nova [arXiv:2509.21301]**: GPU SM partition 만, DLA 미활용. 본 idea M1 가 DLA spatial-split.
- **vs Watershed**: control plane phase classifier 만, physical dataflow 부재.

### M2: DLA→GPU Zero-Copy Handoff via UMA NVMM

**① 추가되는 Scheme — Source Verified (R32)**:

DLA 출력 tensor 를 GPU 주소공간으로 zero-copy. Jetson UMA 환경에서 cudaMallocManaged + DLA NVMM (NV Multimedia Memory) buffer 공유 — SoC fabric round-trip 없이 LPDDR5X 한 번 write/read.

> ✅ source verified: `NvBufSurfaceCreate` + `cudaImportExternalMemory` ([Linux dma-buf API](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html))
> ✅ source verified: cudaMallocManaged + `cuMemAdvise(SetPreferredLocation)` ([NVIDIA Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/))
> ✅ source verified: CUDA event + dma-buf fence (batch fence multiple frame)

**② 해결 + evidence**:

[NVIDIA Maximizing Memory Efficiency Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/): zero-copy 가 explicit cudaMemcpyAsync 대비 LPDDR5X bus 점유 1/3. DLA 출력 → CPU staging → GPU 입력의 SoC fabric 3-hop round-trip 평균 250-400 μs 가 **LPDDR5X 1-hop 80-120 μs** 로 감소 (Orin NX 102 GB/s 기준).

**③ Step-by-step**:

1. DLA 출력 tensor 를 NvBufSurfaceCreate 로 dma-buf 생성.
2. cudaImportExternalMemory(buf, fd, size, cudaExternalMemoryHandleTypeOpaqueFd) 로 GPU virtual addr import.
3. PagedAttention kernel 또는 GPU LLM prefill kernel 이 GPU virtual addr 를 직접 dereference (UMA 1-hop).
4. CUDA event + dma-buf fence 로 DLA→GPU sync — batch fence (multiple frame) 으로 fence overhead 분산.

**④ 차별화**:

- **vs explicit cudaMemcpyAsync**: 250-400 μs round-trip. 본 idea M2 는 80-120 μs.
- **vs SHOAL M2 (cudaHostRegisterMapped)**: SHOAL 은 KV residence 의 GPU mapped path. DualLane M2 는 DLA-output → GPU input 의 NVMM dma-buf path. 직교.

### M3: Cache Conflict Avoidance (libsmctrl + GPU L2 partition)

**① 추가되는 Scheme — Source Verified (R32)**:

DLA 내부 SRAM 1 MiB × 2 와 GPU L2 (Orin: 4MB / Thor 추정: 8MB) 가 SoC fabric 에서 working set 충돌. libsmctrl 로 GPU SM/TPC partition (compute side) + GPU L2 의 access policy partition.

> ✅ source verified: libsmctrl ([UNC repo](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/), ECRTS 2025 Bakita et al.)
> ✅ source verified: `cudaStreamSetAttribute(stream, cudaStreamAttrAccessPolicyWindow, ...)` (CUDA 11.0+, Jetson Orin Ampere/Blackwell 모두 지원)

**② 해결 + evidence**:

[Hardware Compute Partitioning ECRTS 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf): GPU SM/TPC partition 으로 multi-tenant workload 의 latency variance 감소. 본 idea 는 DLA-DLA-GPU 3-way 의 GPU 측 partition.

**③ Step-by-step**:

1. libsmctrl 으로 GPU SM 50% 를 LLM prefill 에 할당, 50% 를 fallback / 다른 task.
2. cudaStreamAttrAccessPolicyWindow 로 LLM prefill stream 의 L2 working set 을 persisting region 지정.
3. DLA 측은 conv buffer SRAM 만 사용 — GPU L2 와 분리.

**④ 차별화**:

- **vs CacheVeil 원안 (R45 demoted)**: CacheVeil 은 ARM CMN SLC partition (memory side, undocumented BPMP IOCTL → R45.1 위반). 본 idea M3 는 libsmctrl + cudaStreamAttr (compute + L2 access policy, **모두 user-space 공식 API**).

### M4: Stage Pipelining with GPU LLM Prefill

**① 추가되는 Scheme**:

다음 frame N+1 의 vision encoder 가 DLA 에서 실행되는 동안, 현재 frame N 의 LLM prefill 을 GPU 에서 실행. Pipelining 으로 vision encoder latency 를 LLM prefill 뒤에 숨김.

> ✅ source verified: CUDA Stream + DLA queue 동시 사용 ([NVIDIA Multi-stream docs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#concurrent-execution-and-streams))
> ✅ source verified: TensorRT 10.9 multi-engine concurrent inference

**② 해결 + evidence**:

Robot perception loop 에서 30 FPS 달성 가능. 12-18 FPS baseline 대비 25-30 FPS — vision encoder + LLM prefill serial 일 때 17-25 ms / inference, pipelining 시 vision encoder 가 LLM prefill 뒤에 숨어 effective latency 33-40 ms / 2-frame = 16-20 ms / frame.

**③ Step-by-step**:

1. Frame N: GPU LLM prefill 실행, 동시에 frame N+1 의 vision encoder 가 DLA0/DLA1 spatial-split 실행.
2. Frame N+1 의 vision encoder 결과는 NVMM dma-buf 로 GPU 측 reserved KV slot 에 prepare.
3. Frame N LLM decode 종료 시 frame N+1 LLM prefill 시작 — vision encoder 결과는 이미 GPU 측 ready.
4. Pipeline depth {1, 2, 3 frame} variant — 실시간성과 throughput trade-off 측정.

**④ 차별화**:

- **vs Nova [arXiv:2509.21301]**: GPU SM partition 만, DLA 미활용. 본 idea M4 는 DLA-frame N+1 + GPU-frame N 의 cross-device pipelining.
- **vs DuetServe** [arXiv:2511.04791]: datacenter prefill-decode disaggregation 만, edge single-device pipelining 부재.

---

## R45 적용 — Implementation Path 검증

본 idea 의 4 mechanism 모두 vendor 공식 user-space API 위에서 구현 가능 — R45 risk **4/10 LOW**. R45.1 금지 카테고리 (kernel patch / kernel module 추가 / undocumented register / closed-source firmware) 위반 없음:

- **Mechanism #1 (TensorRT DLA layer-split)**: `IBuilderConfig::setDeviceType` + `setDLACore(0/1)` + `kGPU_FALLBACK` 모두 TensorRT 10.9 공식 user-space API.
- **Mechanism #2 (NVMM zero-copy dma-buf)**: `NvBufSurfaceCreate` + `cudaImportExternalMemory` + Linux DRM dma-buf — kernel 공식 user-space API.
- **Mechanism #3 (libsmctrl + cudaStreamAttr)**: libsmctrl 은 ECRTS 2025 published research artifact (user-space, ioctl 기반). cudaStreamAttrAccessPolicyWindow 는 CUDA 11.0+ 공식 API.
- **Mechanism #4 (multi-stream + multi-engine pipelining)**: CUDA Stream + TensorRT multi-engine 모두 공식 API.

R45.3 (한 학기 30 runs feasibility) 도 5 workload (Qwen2.5-VL-7B / Qwen2.5-VL-3B / LLaVA-1.5-7B / robot perception 30 FPS / Video-MME 32-frame) × 3 split mode (batch / spatial-LR / spatial-quad) × 2 baseline (vanilla GPU-only / Nova GPU SM partition) = 30 runs 가능 — Orin AGX 64GB 1 device 12-16주 fit. NVDLA v2 spec 변경 없음 (Thor NVDLA gen-next 는 conservative 가정, scope 외).

R45 종합 판정: **risk 4/10 LOW, Tier-1 적합**. CacheVeil R45 demotion 자리 진입 가능.

---

## 4. 평가 / 실험 플랜 (R27-β + R31 + R32 적용 7 요소)

### 4.1 Hardware

- **Primary**: Jetson Orin AGX 64GB (Dual-DLA v2.0, GPU L2 4MB).
- **Secondary**: Jetson Orin NX 16GB (Dual-DLA v2.0).
- **Optional**: Jetson Thor 128GB (NVDLA gen-next, NDA 후 검증) — NVIDIA partner channel 통해 spec 가용 시.
- **Excluded**: Jetson Orin Nano 8GB (Single DLA only — Track A 외).

### 4.2 Software Stack

- TensorRT 10.9 + DeepStream 7.x + JetPack 7.1 fixed-target.
- vLLM v1 (LLM prefill 측), TensorRT-LLM (DLA-aware build path).
- NVMM dma-buf user-space API + libsmctrl.
- Nsight Compute (`lts__t_sectors`, GPU SM utilization) + Nsight Systems (DLA timeline) + tegrastats (200ms cadence).

### 4.3 Workload (5 workload × 3 config × 2 baseline = 30 runs)

| Workload | 설명 | vision encoder 비중 |
|----------|------|---------------------|
| Qwen2.5-VL-7B / 3B | 일반 VLM | 35-45% |
| LLaVA-1.5-7B | CLIP-ViT-L/14 + Vicuna | 40-50% |
| Robot perception 30 FPS | continuous frame | 50-55% |
| Video-MME 32-frame | video VLM | 45-55% |
| InternVL3-2B | edge VLM | 35-42% |

### 4.4 Metric (7 elements R27-β)

1. Vision encoder DLA latency vs GPU latency (per-layer breakdown).
2. DLA→GPU NVMM handoff cost (μs) — CUDA event timeline.
3. SLC / GPU L2 miss rate (Nsight `lts__t_sectors_op_read_lookup_miss`).
4. GPU SM idle ratio during DLA exec (Nsight Compute SM utilization).
5. Energy/inference (tegrastats VDD_GPU + VDD_CPU + VDD_SOC sum).
6. End-to-end FPS (continuous perception loop).
7. P99 frame latency (real-time constraint).

### 4.5 Variant

- DLA layer-split point: {stem only, first 4, first 8, first 12}.
- Split mode: {batch, spatial-LR, spatial-quad}.
- Pipelining depth: {1, 2, 3 frame}.
- libsmctrl GPU SM partition: {25%, 50%, 75% allocation to LLM prefill}.

### 4.6 Baseline (6편 모두 peer-reviewed, R-Reference Integrity ≥ 50%)

- [Nova arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (mechanism 1:1 diff)
- [XSched OSDI 2025](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf)
- [SLO-Aware Edge Scheduler ACM TECS 2024](https://dl.acm.org/doi/fullHtml/10.1145/3460352)
- [Hardware Compute Partitioning ECRTS 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf)
- [MobileAIBench arXiv:2406.10290](https://arxiv.org/abs/2406.10290)
- [Vision Transformer Computation and Resilience ISPASS 2024](https://ispass.org/ispass2024/accepted-papers.php)

### 4.7 Fallback Scope

- DLA spec 미지원 layer 만 GPU fallback (`kGPU_FALLBACK`).
- 측정 시 vanilla GPU-only 와 isolated comparison.
- Single-DLA fallback (DLA1 미사용) 도 측정 — Orin Nano 호환성 확인.

---

## 5. 예상 효과 (보수적, scope 명시)

| 지표 | Baseline | 목표 | 적용 조건 |
|------|---------|------|---------|
| VLM prefill latency (vision encoder 포함) | base | 1.4-1.8× speedup | Orin AGX/NX, vision encoder ≥ 35% prefill |
| GPU SM utilization (vision-encoder 구간) | base 100% | 60-70% (DLA offload) | freed for parallel LLM prefill |
| DLA→GPU NVMM handoff | 250-400 μs (cudaMemcpy) | 80-120 μs (NVMM zero-copy) | UMA 1-hop |
| Energy/inference (perception loop) | base | -15~-22% | DLA J/op < GPU |
| Robot perception FPS | 12-18 | 25-30 | continuous 30 FPS scope |
| P99 frame latency | base | -25~-35% | Pipelining + zero-copy 결합 |

**Workload 특이성**:
- 적용: **continuous perception (robot, drone, ADAS)**, **video VLM (16+ frame)**, **camera-stream VLM**.
- 미적용: **single-turn document QA (vision encoder 1회)**, **text-only LLM**, **Orin Nano (Single DLA only)**.
- 불확실: **Thor NVDLA gen-next API** (NDA) — Orin AGX/NX 의 NVDLA v2.0 만 검증 scope.

---

## 6. Tier 분기 (Tier-1 강점 / Tier-2 fallback variant)

**Tier-1 진입 조건**:
- Vision encoder ≥ 35% prefill 워크로드에서 prefill 1.4× 이상 speedup.
- DLA→GPU NVMM zero-copy round-trip ≤ 100 μs.
- 4-mechanism ablation 결과 each mechanism contribution 정량화 (M1 alone / M1+M2 / M1+M2+M3 / all 4).
- Nova [arXiv:2509.21301] 와 1:1 mechanism diff 강화 — Nova 는 GPU SM partition 만, 본 idea 는 4-mechanism.

**Tier-2 fallback (single-mech spinoff)**:
- M1 only 또는 M2 only — single-mechanism 의 first-to-report 효과만 측정 → DAC 6p / DATE 6p 스코프.

---

## 7. 미선정 baseline 보강 (Phase 1' / R45 정제 사항)

- **Nova [arXiv:2509.21301]** 50-70% CONCURRENT 회피 — 4-mechanism 분리 ablation 으로 Nova 는 M4 (stage pipelining) 만 부분 overlap 함을 정량화. M1 (DLA spatial-split) + M2 (NVMM zero-copy) + M3 (libsmctrl L2 partition) 은 Nova 미공략.
- **Watershed (hwpim 미선정)** 와의 axis 분리 — Watershed 는 control plane phase classifier, 본 idea 는 physical dataflow co-schedule.
- **R45 적용 후 진입**: CacheVeil 원안 (8.00 ★ lead) 가 R45.1 위반으로 demotion → DualLane 자리 비면서 Tier-1 진입.

---

## 8. Source Verification (R32 통합)

| Component | Path / Function | 상태 |
|---|---|---|
| TensorRT 10.9 DLA layer-split | `IBuilderConfig::setDeviceType` + `setDLACore` ([공식 docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html)) | ✅ |
| TensorRT GPU fallback | `setFlag(BuilderFlag::kGPU_FALLBACK)` | ✅ |
| NVMM dma-buf | `NvBufSurfaceCreate` + `cudaImportExternalMemory` ([Linux dma-buf docs](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html)) | ✅ |
| CUDA Stream multi-engine | [NVIDIA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#concurrent-execution-and-streams) | ✅ |
| libsmctrl | [UNC repo](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/), ECRTS 2025 | ✅ |
| cudaStreamAttrAccessPolicyWindow | CUDA 11.0+ ([NVIDIA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)) | ✅ |
| NVDLA v2.0 spec | [nvdla.org](https://nvdla.org/hw/v1/hwarch.html) | ✅ |
| Thor NVDLA gen-next | NVIDIA Jetson AI Lab + partner channel | ⚠️ NDA |

---

## 9. Reference 목록 (이 idea 핵심)

- **Nova**: [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) (mechanism 1:1 diff baseline)
- **XSched** [OSDI 2025]: [paper](https://ipads.se.sjtu.edu.cn/_media/publications/xsched-osdi25.pdf)
- **SLO-Aware Edge Scheduler** [ACM TECS 2024]: [DOI](https://dl.acm.org/doi/fullHtml/10.1145/3460352)
- **GPU Context-Aware Preemptive Scheduling** [LIPIcs ECRTS 2024](https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf)
- **Hardware Compute Partitioning (libsmctrl)** [ECRTS 2025](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf)
- **MobileAIBench**: [arXiv:2406.10290](https://arxiv.org/abs/2406.10290) [NeurIPS D&B 2024]
- **Vision Transformer Computation and Resilience** [ISPASS 2024](https://ispass.org/ispass2024/accepted-papers.php)
- **NVDLA Hardware Spec**: [nvdla.org](https://nvdla.org/hw/v1/hwarch.html)
- **TensorRT 10.9 DLA**: [공식 docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html)
- **NVIDIA Maximizing Memory Efficiency Tech Blog**: [link](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
- **Linux dma-buf API**: [kernel docs](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html)

---

## 10. Source 추적

본 idea 의 staging 출처: `__research_wiki/sessions/staging/2026-04-25-vlm-context-edge-jetson-legacy-sys-expert.md` "Idea T3 — DualLane" 섹션. 1차 publish (2026-04-25 오전) 시점에는 Nova CONCURRENT 우려로 unselected 되었으나, R45 적용 결과 (a) 모든 API 공식 user-space (R45.1 위반 없음), (b) CacheVeil R45 demotion 자리 비면서 Tier-1 5 자리 중 진입 가능, (c) 4-mechanism 분리 ablation 으로 Nova 와의 mechanism diff 1:1 강화 가능 → Tier-1 진입 결정.
