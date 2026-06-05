# QUARRY — Trace-Driven Row-Buffer Locality of Standard-MoE Expert Reads on LPDDR

> **Hook**: standard-MoE expert GEMM이 LPDDR row-buffer locality를 얼마나 살리는지(또는 못 살리는지)는 아무도 측정한 적이 없다. 실기 측정은 막혀 있다 — Orin iGPU(ga10b)는 ncu 미지원이라 `dram__sectors`를 못 읽고, user-space는 CUDA UM virtual addressing 때문에 물리 row/bank 매핑을 제어할 수 없다. QUARRY는 vLLM `fused_moe` access trace를 추출해 Ramulator2/DRAMSim3 standalone에 흘려, 현재 layout의 row-buffer hit rate와 co-activation reorder의 잠재 이득 상한을 **trace-driven으로** first-to-report한다.
>
> **Score (Phase 2')**: novelty 7 / diff 7.5 / impact 7 / ai-impl 7 / arch-impl 8 = **avg 7.1** (Δ+1.4 vs Phase 2 5.7). Tier-2 Top — **강등이 점수를 살린 케이스**.
> **Venue**: DAC 2027 (대안 DATE / IEEE CAL). Track B trace-driven 측정 letter. (Phase 1' staging은 ISPASS/IISWC 표기 — 본 요약은 venue 지시값 DAC/DATE/IEEE CAL 채택.)
> **강등 이력**: Track A(MICRO/HPCA, 실기 layout 제어) → **Track B(trace-driven, R47.4)**. 실기 측정 불가가 강등 근거이자 재설계 동력.

---

## 1. Research Questions

- **RQ1 (정량)**: vLLM `fused_moe` grouped-GEMM이 batch-1 decode에서 발행하는 expert weight 주소 stream을 Ramulator2(LPDDR5 config)에 흘렸을 때, 현재 layout의 **row-buffer hit rate**는 얼마이며 — 이미 row-streaming에 가까운가(go/no-go)?
- **RQ2 (정량)**: CORAL의 co-activation 그래프 `C[i,j]`로 expert read 순서를 reorder하면 row-buffer hit rate가 얼마나 오르고, 그에 따른 **잠재 유효 BW 이득 상한(+X%)**은 얼마인가? (batch-1 decode 한정 — continuous batching은 active expert 합집합 팽창으로 정적 layout 무력).

---

## 2. 기준 코드베이스 (Baseline Source)

- **서빙 스택**: vLLM v1, commit **`7c37096620`** pin. NGC container `nvcr.io/nvidia/vllm:26.01-py3`(Jetson aarch64 pip 미지원 대응). access trace 추출은 vLLM `fused_moe` grouped-GEMM 경로.
- **모델**: Nemotron-3-Nano-30B-A3B (HF), **FP8**. (llama.cpp 경로 제거 — [Issue #20570](https://github.com/ggml-org/llama.cpp/issues/20570) GGUF hybrid 로딩 blocker로 vLLM-FP8 only.)
- **trace 추출**: **Nsight Systems (nsys)** — Orin iGPU 지원 ✅(timeline/trace). batch-1 decode expert read 주소 stream 캡처.
- **DRAM simulator** (**GitHub 활성 확인 명시**):
  - **Ramulator2** [CMU-SAFARI/ramulator2](https://github.com/CMU-SAFARI/ramulator2) — **active(MIT 라이선스)**, trace-frontend + OoO core frontend, LLM MM/MVM trace 사례 보유. LPDDR5 모델로 expert read trace의 row-buffer hit 검증 가능.
  - **DRAMSim3** [umd-memsys/DRAMsim3](https://github.com/umd-memsys/DRAMsim3) — **active**, LPDDR3/4 + thermal-capable, gem5/ZSim 연동. BW 경합 trace 보조.
  - 둘 다 active → R45.9(dead-tool reject) 해당 없음. **R47.4 trace-driven**: gem5+vLLM 동시 실행 아님(R47.1 준수) — nsys로 trace 추출만 한 뒤 simulator standalone.
- **HW 측정 불가 검증** (arch-impl review 인용 — 이것이 Track A 강등 근거):
  | 항목 | 판정 | 근거 |
  |---|---|---|
  | user-space row/bank 물리 매핑 제어 | [❌] | CUDA UM virtual addressing, `cudaMemAdvise`는 locality hint만 (R45.1 위반) |
  | Orin iGPU `dram__sectors` (row-buffer) ncu 측정 | [❌] | ga10b ncu 미지원 (`Profiling not supported on device 0`) |
  | RTX dGPU 대체 측정 | [❌] | GDDR7 → LPDDR row geometry 무관(device 모순) |
  | nsys access trace 추출 (AGX Orin) | [✅] | nsys Orin iGPU 지원 — trace 추출만 가능 |
  | Ramulator2/DRAMSim3 standalone row-buffer 측정 | [✅] | active, trace-frontend LPDDR5 config |
  - arch feasibility 3→8 (Δ+5.0): 실기 측정 불가를 trace-driven으로 전환해 측정 가능성 확보.

---

## 3. 배경 / GAP

- per-token expert read **1.84GB/token**(8 expert × 23 MoE층 × 10MB) → AGX Orin 204 GB/s ÷ 1.84GB = **~9 tok/s 상한**. expert-read가 LPDDR BW 천장이라 row-buffer locality가 유효 BW를 좌우할 잠재력.
- **user-space 물리 row/bank 제어 불가**: CUDA unified memory는 virtual address space — physical page는 MMU/커널 관할 ([CUDA UM guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)). row-aligned packing·bank interleaving은 R45.1(no-kernel-patch) 위반.
- **Orin ncu 미지원**: ga10b는 `dram__sectors` 등 row-buffer metric 측정 불가 ([forum](https://forums.developer.nvidia.com/t/nsight-compute-error/234599)). RTX dGPU는 GDDR7로 LPDDR row geometry 무관 → 실기 측정 device 모순.
- vLLM `fused_moe` grouped-GEMM이 **이미 row-streaming일 가능성** (cross-review 지적) → 먼저 trace를 확인해야 idea 유효성 판정 가능(go/no-go).
- frozen router(RL-freeze) → co-activation `C[i,j]`가 정적 → reorder 가능성의 전제(Nano insight 5).
- 기존 DRAM-layout 연구는 HBM 재설계(RoMe)나 general data 대상 — **commodity-LPDDR × standard-MoE expert** trace-driven 측정은 white space.

---

## 4. Mechanism (단일)

### 4.1 동작 원리 — M1: fused_moe access trace 추출 + Ramulator2/DRAMSim3 row-buffer 측정
- vLLM `fused_moe` grouped-GEMM이 발행하는 expert weight 주소 stream을 nsys로 캡처(batch-1 decode) → Ramulator2(LPDDR5) / DRAMSim3(thermal-capable) standalone에 trace-frontend로 입력 → row-buffer hit rate 측정.
- CORAL의 co-activation `C[i,j]`로 reorder한 trace를 A/B로 흘려 잠재 이득을 simulator에서 정량화.

### 4.2 기대 효과
- 현재 fused_moe layout의 row-buffer hit rate를 first-to-report.
- co-activation reorder의 **잠재 유효 BW 이득 상한(+X%)** 보고.
- go/no-go: 현재 layout이 이미 row-streaming이면 idea 무력 — 그 자체가 정직한 contribution.

### 4.3 구현 변경점
| 구분 | Phase-1 원본(IDEA-2) | Phase-1' 정제 |
|---|---|---|
| Track | A (MICRO/HPCA) | **B (trace-driven 측정 letter) 강등** |
| M1 layout 제어 | user-space row-align(불가) | **제거**, nsys trace 추출만 |
| M2 bank interleave | user-space(불가) | **제거** |
| M3 SLC pin | L2(4MB < 10MB expert 불가) | 제거(HEARTH M3 흡수) |
| 측정 | 실기 ncu(불가) | **Ramulator2/DRAMSim3 R47.4** |
| CORAL | 별개 | `C[i,j]` graph 공유(프로파일러 cross-share) |

### 4.4 검증 시나리오
1. nsys로 batch-1 decode expert read 주소 trace 추출(AGX Orin, vLLM-FP8).
2. Ramulator2 LPDDR5 config로 baseline row-buffer hit rate 측정.
3. CORAL `C[i,j]` co-activation reorder trace A/B 측정.
4. 잠재 유효 BW 이득 상한 보고 + 현재 layout이 이미 row-streaming인지 go/no-go 판정.

---

## 5. 실험 플랜 (7-요소, 단일 scope)

1. **목표 지표**: row-buffer hit rate(현재 layout) / reorder 후 hit rate 상한 / 잠재 유효 BW 이득(+X%) — **측정치(개선치 아님)**.
2. **실기 device**: **AGX Orin 64GB**(vLLM-FP8로 **access trace 추출만**). RTX 부적합(GDDR7 row geometry 무관).
3. **분석 환경**: **Ramulator2(LPDDR5 config) + DRAMSim3** standalone(R47.4 trace-driven, gem5+vLLM 동시 아님).
4. **trace 축**: batch-1 decode 한정(continuous batching은 active expert 합집합 팽창으로 정적 layout 무력 → 명시 제외).
5. **A/B 조건**: 현재 fused_moe layout vs CORAL `C[i,j]` co-activation reorder.
6. **측정**: nsys(주소 stream) → Ramulator2 row-buffer hit rate + DRAMSim3 thermal/BW 보조.
7. **분석**: baseline vs reorder hit rate delta → 잠재 BW 이득 상한, go/no-go 판정.

| 축 | baseline | 목표(측정) | 조건 |
|---|---|---|---|
| [측정] row-buffer hit rate | 현재 fused_moe layout | 정량화 + reorder 상한 | batch-1 decode |
| [측정] 잠재 유효 BW 이득 | simulator baseline | +X% 상한 보고 | LPDDR5 config |

---

## 6. 관련 연구 · 차별점

- **RoMe** [arXiv:2512.01541](https://arxiv.org/abs/2512.01541) (HPCA 2026) — HBM 재설계, MoE co-activation 미적용. 차별: **commodity-LPDDR trace-driven** + MoE co-activation reorder. (arXiv ID 2512.01541 vs .01644 혼선 — paper-final 전 R13.1 검증 의무.)
- **Precision-Aware Bank Separated** ([ACM ISMS](https://dl.acm.org/doi/full/10.1145/3767110.3767112), peer-reviewed) — general data bank 분리, MoE expert 무관. 차별: standard-MoE expert read 특정.
- **TurboMind layout** [arXiv:2508.15601](https://arxiv.org/abs/2508.15601) (arXiv) — dense GEMM layout. 차별: MoE expert grouped-GEMM × row-buffer.

---

## 7. 왜 Tier-2 only인가

- **Top-tier scale-up 불가 이유**: (a) **실기 row-buffer 측정 자체가 불가** — Orin ncu 미지원 + user-space 물리 매핑 제어 불가(R45.1)로 Track A(MICRO/HPCA가 기대하는 실HW microarch 측정·재설계)가 원천 차단. (b) trace-driven simulation은 "잠재 이득 상한"만 보고 가능 — 실제 hardware 개선 데모가 불가능하므로 measurement letter scope. (c) batch-1 decode 한정으로 일반화 제한. 이 셋이 Track A 강등의 직접 근거였고, 강등이 오히려 ai-impl 4→7·arch-impl 3→8로 점수를 살림.
- **Tier-1 승격 조건**: LPDDR row/bank 매핑을 제어 가능한 plat(예: 커널 협조 가능 SoC)이 확보되거나, simulator 결과를 실HW 개선으로 연결하는 경로(예: alloc-granularity 수준 user-space 정렬이 실측 BW 이득)가 검증되면 Track A 재승격.
- **CORAL 공유**: `C[i,j]` co-activation 프로파일러를 CORAL과 공유(구현 1회) — QUARRY는 row-buffer trace, CORAL은 tier placement에 같은 graph 사용.

---

## 8. 약어 / 용어 풀이

- **fused_moe**: vLLM의 MoE expert grouped-GEMM 커널 경로(expert weight 주소 stream 발행처).
- **row-buffer hit**: DRAM row를 activate한 상태에서 같은 row의 후속 access가 재activate 없이 처리되는 경우(BW 효율 직결).
- **bank interleaving**: 여러 DRAM bank에 분산 배치해 access 병렬화 — user-space 물리 제어 불가.
- **trace-frontend**: 메모리 access 주소 stream을 입력으로 받는 simulator 실행 모드(실 core 시뮬 없이).
- **R47.4**: trace-driven simulation 경로(gem5+vLLM 동시 실행 금지하는 R47.1 준수).
- **R45.1**: no-kernel-patch 제약(user-space만 허용).
- **ga10b**: AGX Orin/Orin Nano iGPU 아키텍처(ncu 미지원).
- **co-activation `C[i,j]`**: frozen router 하 expert pair i,j가 같은 token에서 동시 활성화되는 빈도 행렬.
- **batch-1 decode**: 단일 요청 decode — 정적 layout이 유효한 유일 scope(continuous batching은 expert union 팽창으로 무력).
