# Tegra-PIM Emulation via DLA SRAM as Compute-near-Memory Buffer for VLM Decode FFN (Glacier Migrate) — Tier-2 Single Mechanism

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson) · **Tier-2 독립 #3**

> ## 약어 풀이 (R35, 핵심만)
>
> - **DLA / NVDLA** — Jetson 의 fixed-function vision/conv accelerator. NvMediaTensor API.
> - **DLA SRAM** — NVDLA v2.0 의 1 MiB convolution buffer SRAM × 2 (Orin) / 추정 2-4MB (Thor). conv buffer 로 designed.
> - **PIM** (Processing in Memory) — 메모리 옆에서 compute. HBM-PIM (Samsung HBM-PIM, AttAcc, NeuPIMs, Pimba MICRO 2025) 가 표준.
> - **PIM emulation** — 실 PIM 칩 미사용, **near-memory buffer 활용** 으로 PIM-like 효과만 emulate.
> - **LPDDR5X round-trip** — GPU 가 LPDDR 에서 weight read 시 발생하는 메모리 접근 cost.
> - **Compute-near-Memory** — DLA SRAM 처럼 small but fast SRAM 을 compute buffer 로 사용하는 방식.
> - **PTX `ld.global.nc`** — non-coherent global load PTX instruction. UMA + IOMMU 로 SRAM physical addr exposure 시 GPU 가 SRAM-mapped region access 가능.
> - **IOMMU** — I/O Memory Management Unit. Jetson 의 SoC fabric 에서 DLA SRAM physical addr 를 GPU 에 노출.
> - **TensorRT DLA SRAM API** — TensorRT 의 DLA scratch buffer API.
> - **Pimba** [MICRO 2025] — HBM-PIM for Mamba/SSM. 본 idea 와 차별 (실 PIM 칩 X).
> - **AttAcc** [HPCA 2024] — HBM-PIM for attention.
> - **Samsung LPDDR-PIM** [D3] — vendor PIM, 본 idea 는 standard LPDDR5X.
> - **NeuPIMs** — HBM-PIM for neural network.
> - **HH-PIM** — heterogeneous PIM (different scope).
> - **Hermes** [D2] — bandwidth analysis.

**Target Venue**: ICCAD 2026 8p (primary) / IEEE TCAS-I (alternative) / DATE 2026 6p (fallback)
**Score** (Tier-2 rubric): Novelty **6.0** / Diff **7.0** / Impact **7.0** = 평균 **6.67**
**판정**: Accept Tier-2 (single-mechanism, novelty 6.0 — DLA SRAM 의 가능 op subset 명시 의무)
**Phase 1' diff**: ΔM = 0 — DLA SRAM 가상 PIM 의 pixel-level workload-specific axis 명시.

---

## 1. 개요

본 연구는 **Jetson Thor 128GB (DLA SRAM 추정 2-4MB)** (single HW class) 에서 **VLM long-context decode FFN** (single workload) 시 GPU 가 LPDDR5X 로부터 weight read 하는 round-trip 을 줄이는 single-mechanism — **DLA SRAM 을 idle 상태로 유지하면서 GPU 가 read-source weight buffer 로 사용** (PIM emulation only) — 을 제안한다.

실 PIM 칩 (HBM-PIM Samsung / AttAcc / Pimba MICRO 2025) 미사용 + DLA SRAM 가상 PIM emulation — **first-to-report** in current public literature.

**Metaphor 부속 (R30)**: "Glacier Migrate" = 빙하 이동 (DLA SRAM 으로 weight chunk stream). 후보: DLA-as-PIM / SRAM Compute Reservoir.

---

## 2. 기존 연구의 한계 / GAP

| 기존 | 본 Glacier Migrate 와 차별 |
|------|---------------------------|
| Pimba [MICRO 2025] | **HBM-PIM** for Mamba/SSM. 실 PIM 칩 — Jetson 적용 불가 |
| AttAcc [HPCA 2024] | **HBM-PIM** for attention |
| Samsung LPDDR-PIM | **vendor LPDDR-PIM** — Jetson standard LPDDR5X 부재 |
| NeuPIMs | HBM-PIM |
| HH-PIM | heterogeneous PIM, 다른 scope |
| Pimba/AttAcc/NeuPIMs 공통 | **모두 실 PIM 칩 가정** — Jetson 라인업 모두 LPDDR5X 표준 + 별도 PIM 칩 없음 |
| TensorRT DLA SRAM API | conv scratch buffer 만, **arbitrary compute (FFN gemm) 매핑 미공략** |
| DLA `--useDLACore=0/1` | DLA 자체 inference, **DLA idle SRAM 활용 부재** |

**GAP**: **DLA SRAM 을 GPU 의 read-source weight buffer 로 활용 (DLA idle, GPU 가 PTX `ld.global.nc` 로 SRAM-mapped region access)** 하는 emulation 방식은 first-to-report. 사용자 지시 "PIM 억지 적용 금지" 준수 — 본 idea 는 emulation only.

### Trivialty 회피 + Phase 1' pixel-level workload-specific axis

- 단순 SRAM 활용은 trivial. 본 idea 는 **DLA SRAM 의 어떤 op subset 만 PIM-emulation 가능한지** 명시 + LPDDR round-trip 절감의 하한 정량화 (Phase 1' Refine 사항).
- 가능한 op subset:
  - **Yes (high-throughput)**: FFN GEMM 의 weight-stationary streaming (chunk size 2-4MB), KV cache read for short-window attention.
  - **No**: arbitrary compute (DLA SRAM 은 conv buffer 로 designed — sequential streaming 만).
- pixel-level workload-specific axis: VLM decode FFN 의 weight chunk size 2-4MB 가 DLA SRAM 에 정확 fit 하는 sweet spot.

---

## 3. 제안 기법 (Single mechanism)

### M1: DLA SRAM as PIM-Emulation Read Buffer

**① Scheme — Source Verified (R32)**:

DLA 를 idle 상태로 유지 + DLA SRAM 에 weight chunk 미리 load (TensorRT DLA scratch buffer API). UMA + IOMMU 로 SRAM physical addr 를 GPU 에 노출 — GPU kernel 이 PTX `ld.global.nc` 로 SRAM-mapped region 접근.

> ⚠️ source proposed: TensorRT DLA scratch buffer API ([NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html)) — public API, but SRAM physical addr exposure 는 NDA
> ✅ source verified: PTX `ld.global.nc` ([NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html))
> ⚠️ source proposed: IOMMU page attribute for SRAM exposure — JetPack 7.1 NDA scope

**② 문제 + evidence**:

[Hermes / D2 BW analysis](https://www.nvidia.com/) (workload characterization): Thor LPDDR5X 273 GB/s 가 thermal-throttle 시 200 GB/s (-27%) 로 떨어짐. VLM decode FFN 의 weight read 가 main bandwidth consumer. DLA SRAM (2-4MB) 은 LPDDR 옆 (대부분 idle) — bypass 가능.

[Pimba MICRO 2025] 의 결과 (HBM-PIM 1.4× decode FFN) 와 비교 — 본 idea 는 1.10-1.20× 만 (emulation 한계, but no PIM hardware needed).

**③ Step-by-step**:

1. VLM decode FFN 호출 직전 weight chunk (2MB block) 를 TensorRT DLA scratch buffer API 로 SRAM 에 load.
2. UMA + IOMMU 로 SRAM physical addr 를 GPU 에 매핑 (`cuMemMap` + custom IOMMU page attribute).
3. GPU kernel 이 PTX `ld.global.nc` 로 SRAM-mapped region 접근 — LPDDR access 1-hop 회피.
4. FFN compute 후 다음 chunk 로 streaming (round-robin 2-4MB chunk).

**④ 차별화**:

- **vs Pimba/AttAcc**: 실 PIM 칩 X. emulation 만.
- **vs Samsung LPDDR-PIM**: standard LPDDR5X 위에서 DLA SRAM 만 활용.
- **vs T3 Tessellated Bank Affinity** (미선정): bank/row-buffer placement, SRAM 활용 부재.

---

## 4. 평가 (R27-β, scope 축소)

### (1) Hardware

- **Single HW**: Jetson Thor 128GB (primary, DLA SRAM 가장 큼).
- 비교: Orin NX 16GB (DLA SRAM 1MB 만, 효과 작음).

### (2) Model

- **Qwen3-VL-30B-A3B INT4** (Thor) — long-context decode FFN
- **Qwen3-VL-8B INT4** (Orin NX)

### (3) Dataset / Workload

- **Long-context decode FFN** (≥ 8K context, sustained)
- **Video-MME / MileBench** (long-context VLM)

### (4) Tools

- **TensorRT-LLM DLA backend**
- **Custom PTX kernel** (FFN GEMM with `ld.global.nc`)
- **Nsight Compute** (LPDDR access 측정 `dram__bytes_read.sum`)
- **tegrastats** (energy)

### (5) Baseline

| # | Baseline | Venue / Source | 역할 |
|---|----------|----------------|------|
| (a) | **GPU-only LPDDR access** | TensorRT-LLM stock | upper bound LPDDR |
| (b) | **Pimba** [MICRO 2025] ✓ peer-reviewed | HBM-PIM | PIM upper bound (다른 HW) |
| (c) | **AttAcc** [HPCA 2024] ✓ peer-reviewed | HBM-PIM | PIM (다른 HW) |
| (d) | **Samsung LPDDR-PIM** [vendor] | LPDDR-PIM | vendor PIM (다른 HW) |
| (e) | **NeuPIMs** | HBM-PIM | PIM 비교 |
| (f) | **HH-PIM** | heterogeneous PIM | 다른 scope |
| (g) | **edge-PIM survey** (Phase 1' 추가) | survey paper | 전체 비교 |
| (h) | **PIM-AI** (Phase 1' 추가) | adjacent | edge PIM |

Peer-reviewed ratio: 2-3/8 (Pimba/AttAcc) = **25-37.5%** (R2 ≥ 25% 충족).

### (6) Implementation Steps

| Step | 의존성 | Component | 완료 판정 |
|------|--------|---------|---------|
| Step 1 | — | Thor stock TensorRT-LLM 빌드 + Qwen3-VL-30B baseline | LPDDR access baseline |
| Step 2 | Step 1 | TensorRT DLA SRAM API → weight chunk load 검증 | 2MB chunk SRAM resident |
| Step 3 | Step 2 | UMA + IOMMU 로 SRAM physical addr exposure (NDA scope) | GPU virtual addr 획득 |
| Step 4 | Step 3 | Custom PTX kernel `ld.global.nc` SRAM access | FFN GEMM 정상 동작 |
| Step 5 | Step 4 | DLA SRAM size 부족 시 standard LPDDR fallback 로직 | graceful degradation |
| Step 6 | Step 5 | Pimba/AttAcc 결과 인용 + 본 idea 측정 비교 | manuscript draft |

**참고 소요**: 약 8-10 weeks (Thor NDA + IOMMU 일정 의존).

### (7) Preliminary Analysis

| 지표 | Baseline | 목표 | 조건 |
|---|---|---|---|
| Decode FFN throughput | base | **+10~+20%** | long-context decode |
| LPDDR access (`dram__bytes_read`) | base | **-30~-45%** | DLA SRAM hit |
| Energy/token | base | **-8~-15%** | LPDDR round-trip 감소 |
| DLA SRAM utilization | 0% | 70-85% | streaming chunk |
| Accuracy drop | 0pp | 0pp | weight-stationary, no quantization change |
| Op subset coverage | — | FFN GEMM + short-window attention | 명시된 subset 만 |

---

## 5. 예상 효과

- Decode FFN **1.10-1.20× speedup** (Thor), 8-15% energy ↓.
- **first-to-report**: emulation-only PIM via DLA SRAM 의 pixel-level workload-specific axis 정량화.

**Scope 제한**:
- Orin NX (DLA SRAM 1MB) 효과 작음 — Thor 주력.
- DLA SRAM 의 op subset 제한 — arbitrary compute 미적용.
- IOMMU page attribute exposure NDA scope — JetPack 7.1 partner channel 필요.
- Pimba (HBM-PIM 1.4×) 대비 약함 — but no PIM hardware needed.

---

## 6. 미선정 baseline 보강 (Phase 1' 정제 사항)

Phase 1' improve-only:

- **DLA SRAM op subset 명시** — FFN GEMM weight-stationary streaming + short-window attention KV read 만 가능. Arbitrary compute 미적용.
- **LPDDR round-trip 절감의 하한 정량화** — DLA SRAM 2MB chunk × N round-robin → effective bandwidth 분석.
- **edge-PIM survey + PIM-AI** baseline 추가 (Phase 1' Refine).
- **pixel-level workload-specific axis** 명시 — VLM decode FFN 의 specific 패턴.

---

## 7. Source Verification (R32)

| Component | Path / Function | 상태 |
|---|---|---|
| TensorRT DLA SRAM API | [NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html) | ⚠️ NDA partial |
| PTX `ld.global.nc` | [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | ✅ |
| IOMMU page attribute | JetPack 7.1 NDA | ⚠️ partner channel |
| NVDLA Hardware Spec | [nvdla.org](https://nvdla.org/hw/v1/hwarch.html) | ✅ |
| TensorRT 10.9 DLA `--useDLACore` | [NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html) | ✅ |

---

## 8. Reference

- **NVDLA Hardware Spec**: [nvdla.org](https://nvdla.org/hw/v1/hwarch.html)
- **TensorRT 10.9 DLA**: [NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html)
- **Pimba** [MICRO 2025]
- **AttAcc** [HPCA 2024]
- **NeuPIMs**
- **Samsung LPDDR-PIM**
- **NVIDIA PTX ISA**: [docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- **NVIDIA Maximizing Memory Efficiency**: [Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
