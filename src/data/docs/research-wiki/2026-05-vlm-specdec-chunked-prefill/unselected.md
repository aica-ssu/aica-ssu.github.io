# 미선정 / Embedded 아이디어 — 2026-05-26 Mode-1 VLM × SpecDec × Chunked Prefill

본 세션에서 도출된 20 raw idea (Phase 1 + Phase 1' refined → 18 정식 평가) 중 Top-M 6 (Tier-1 3 + Tier-2 3) 에 선정되지 못한 12 idea 의 상세 사유 + R67 sub-axis Scoring + ★/▼. Phase 2' integrated 5-reviewer re-review 의 score 와 cross-redundancy 분석에 근거.

처리 분류:
- **Embedded (1)**: B14' → B11' (algorithm-expert 명시)
- **Merged (1)**: C3L' → A3' (legacy-system-expert 명시)
- **Phase 3 elective Tier-1 후보 (5)**: C6L' > C1L' > B12' > A3' > C2L'
- **Tier-2 reposition or partner (3)**: A1', A4', A5'
- **Conditional / gate-dependent (2)**: B13' (pre-experiment fail risk), B8' (VAST-γ merge candidate)
- **Modality-merge candidate (1)**: B10' (with C2L')

---

## A1' Mosaic Maestro v2 (Region-Atomic Chunk × Spec Acceptance Window)

- **GAP**: Sarathi-Serve stall-free `max(...)` budget 초과 chunk 형성 시 visual region 이 split 되어 spec acceptance window 가 desync
- **시도한 overview**: Region-atomic chunk = `min(token_budget, round_up_to_region(...))` + spec acceptance window 와 chunk boundary 동기화 + Sarathi-Serve OSDI 2024 baseline 위 application-level
- **미선정 사유**: Top-M chunk boundary axis 묶음 (A1' / B10' / C1L') 에서 C4L' (mean 8.40, reference diversity 10/10) 가 우선; A1' application-level + C1L' HW-level 의 layered stack partner 로 Phase 3 활용
- **재방문 조건**: C4L' 와 layered stack PR 분리 시 Phase 3 elective Tier-1 (rank 6+); SARATHI-Serve plugin 공유 (C1L'/C4L'/C5L' 와)
- **R67 Scoring sub-axis 요약** (Phase 2' mean **7.85, Accept**):
  - ★ 최고 sub-axis: **arch-sys-impl R47 (9/10)** — application-level 명확 + `min(token_budget, ...)` 정정으로 Sarathi invariant 준수
  - ▼ 최저 sub-axis: **differentiation Positioning (7/10)** — CoVSpec chunk-boundary 차별화 axis 명시했으나 chunk-only narrow scope

---

## A3' Power Grid Conductor v2 (3-Sub-Phase DVFS + Acceptance Feedback, C3L' merged)

- **GAP**: 기존 DVFS (VoltanaLLM/GreenLLM/DualScale) 가 vision encode stage 미포함 — VLM 의 3-stage (encode/prefill/decode) DVFS + HBM clock + acceptance feedback 의 3-axis 결합 부재
- **시도한 overview**: 3-sub-phase NVML clock control + HBM memory clock + acceptance×power feedback (Idea 16 merge); Day 1 calibration 의무 (NVML supported clocks 실측)
- **미선정 사유**: mean 8.05 **Strong Accept** 이나 DVFS axis 단일 candidate — venue 분포 (NeurIPS/MLSys/ASPLOS/MICRO) 에서 DVFS 는 ISLPED/DAC Tier-2 venue-fit; Tier-2 슬롯에서 C5L' (novelty 9/10) 우선
- **재방문 조건**: Phase 3 elective Tier-1 우선순위 4 위; NVML `nvmlDeviceGetSupportedGraphicsClocks` RTX Pro 6000 실측 (≥1주) 통과 시 진입
- **R67 Scoring sub-axis 요약** (Phase 2' mean **8.05, Strong Accept**):
  - ★ 최고 sub-axis: **arch-sys HW-fit (9/10) + R47 (9/10)** — Tier-2 simulator (Ramulator 2.0) 분리로 R47 clean
  - ▼ 최저 sub-axis: **differentiation Positioning (7/10)** — VoltanaLLM frequency controller mis-reading 정정 후에도 standalone DVFS magnitude 가 venue 분포에 미스매치

---

## A4' Pipeline Conveyor v2 (M3 Spec Prelude as Primary, VLMOpt Baseline)

- **GAP**: VLMOpt baseline 외에 M3 spec prelude 의 cross/self-attention 분기 단독 contribution + BaseMultiModalProcessor L1113 + INT8 calibration
- **시도한 overview**: M1/M2 demote to baseline-instance, M3 spec prelude promote primary + Day 1 oneDNN AMX 측정 의무
- **미선정 사유**: CPU vision offload axis 묶음 (A4' / B10' / C2L') 에서 C2L' (dual-path + acceptance-aware + DVFS) 우선; VLMOpt scoop 후 incremental gain 만 17%
- **재방문 조건**: MLSys 2026 industry track 6p Tier-2 reposition (M3 spec prelude single contribution); Qwen2-VL-7B placeholder re-compute < 15% 통과 시
- **R67 Scoring sub-axis 요약** (Phase 2' mean **7.65, Accept**):
  - ★ 최고 sub-axis: **differentiation Positioning (8/10)** — VLMOpt baseline 명시 채택 + cross/self-attention 분기 명시
  - ▼ 최저 sub-axis: **impact Magnitude (8/10)** — VLMOpt baseline 대비 incremental gain 만 -17% 보수화

---

## A5' Heartbeat Synchronizer v2 (Triton Backend + LVSpec 3-Alternative Signal)

- **GAP**: FlashAttention softmax_lse internal-only — 3-alternative signal source (Triton / LVSpec-logit / sub-module) 가 lazy KV materialization 의 heartbeat signal 로 활용 가능
- **시도한 overview**: 3-alternative signal source + async prefetch + double-buffer + Triton backend production-ready
- **미선정 사유**: KV materialization axis 묶음에서 C6L' (UnifiedKV-NVMe, +165-230% capacity magnitude) 우선; A5' lazy KV 는 C6L' (HBM 내 lazy) 의 부분집합 (HBM/DDR5/NVMe tier)
- **재방문 조건**: Phase 3 cross-share component (Triton backend signal source) 로 활용 가능; Triton overhead > 5% 시 DROP risk
- **R67 Scoring sub-axis 요약** (Phase 2' mean **7.65, Accept**):
  - ★ 최고 sub-axis: **arch-sys R47 (8/10) + HW-fit (8/10) + D6 sys (8/10)** — Async prefetch + double-buffer 의무, Triton backend production-ready
  - ▼ 최저 sub-axis: **impact Magnitude (7/10)** — TPOT -17% 보수화 (async prefetch overhead 반영)

---

## B8' VAST-Spec v2 (LVSpec Continuous Diff + v1 path 정정)

- **GAP**: LVSpec strict/relaxed binary 가 continuous closed-form 으로 확장 가능 — visual saliency H_v 기반 sequential adaptive draft tree (CoVSpec parallel branching 과 orthogonal)
- **시도한 overview**: `vllm/v1/spec_decode/eagle.py` L432 verified + runtime `draft_token_ids_list` (L196/L208/L212/L270) + CPU saliency async +0.5-1ms/step
- **미선정 사유**: **B8' + B12' = "VAST-γ" strong merge 권고** 적용 시 단일 candidate; B12' (mean 8.40) 가 closed-form algebra + chunked prefill coupling 추가 강점
- **재방문 조건**: VAST-γ merged candidate Phase 1'' 평가에서 Tier-1 venue 추가 진입 가능 (B12' 와 묶음)
- **R67 Scoring sub-axis 요약** (Phase 2' mean **7.95, Accept-strong**):
  - ★ 최고 sub-axis: **differentiation Coverage (9/10) + Contribution (9/10)** — CoVSpec parallel branching vs 본 idea sequential adaptive 명시
  - ▼ 최저 sub-axis: **novelty Mechanism (7/10)** — LVSpec continuous extension 의 axis 가 narrow

---

## B10' Modality-Boundary Chunked Prefill (HiViS-orthogonal)

- **GAP**: HiViS drafter-side vs target prefill-side 의 orthogonality + Nova/ElasticMM/HydraInfer 가 chunk boundary 미명시 — modality boundary 기반 chunk split 부재
- **시도한 overview**: `vllm/v1/core/sched/scheduler.py` L158 + L202/L205/L401/L404 verified + modality-boundary chunked prefill + CPU SigLIP encoder
- **미선정 사유**: A1' application-level + C1L' HW-level 의 layered stack 우선; B10' CPU SigLIP encoder 는 C2L' (CDVV-Lite) dual-path 와 70% merge
- **재방문 조건**: C2L' 과 merge candidate (algorithm-expert 명시 권고) — Phase 3 cross-share dependency 의 핵심 component
- **R67 Scoring sub-axis 요약** (Phase 2' mean **8.05, Accept**):
  - ★ 최고 sub-axis: **differentiation Coverage (9/10) + impact Adoption (9/10)** — Nova / ElasticMM / HydraInfer baseline 추가
  - ▼ 최저 sub-axis: **novelty Mechanism (8/10) + arch-sys Sim-trace (8/10)** — A4' 와 70% overlap, merge candidate

---

## B12' Entropy-Gated γ* v2 (Closed-form Algebra Fix + Saliency Integration)

- **GAP**: Entropy-gated γ* closed-form 의 P_d<<P_v approximation 정정 + chunked prefill axis 추가 (F36) + SpecExec / Lookahead Decoding prior 명시
- **시도한 overview**: `vllm/v1/spec_decode/eagle.py` L44 `num_speculative_tokens` runtime reread + AHASD/EAGLE evidence + 600 LoC + DVFS coordination
- **미선정 사유**: mean 8.40 **Strong Accept** 이나 **B8' + B12' = "VAST-γ" merge 권고** → 단일 candidate 시 Tier-1 4번째 슬롯 가능; Top-M 6 에서는 training/quantization/system 3-axis 우선
- **재방문 조건**: Phase 3 elective Tier-1 우선순위 3 위 (VAST-γ merged 형태) — saliency H_v + token entropy H_t joint signal
- **R67 Scoring sub-axis 요약** (Phase 2' mean **8.40, Strong Accept**):
  - ★ 최고 sub-axis: **arch-sys 4 axis 모두 (9/10)** — 600 LoC + AHASD/EAGLE evidence + 모든 sub-axis 9 (19 idea 중 최고 tie)
  - ▼ 최저 sub-axis: **impact Magnitude/Applicability (8/10)** — Energy -15-25% 가 venue magnitude 측면 sufficient 하나 Tier-1 3개 axis 우선

---

## B13' Cross-Chunk MRoPE Spec v2 (Pre-Experiment Gate 의존)

- **GAP**: MRoPE boundary token drop 시 cross-chunk speculation 의 PPL drift 가 1.5% 이내 가능한지 — multi-image dialogue 워크로드 한정
- **시도한 overview**: `MRotaryEmbedding` L985 verified + Lookahead Decoding baseline + 3-day pre-experiment gate (PPL > 1.5% fail → DROP)
- **미선정 사유**: Pre-experiment gate fail risk 가 mean 7.30 Conditional Accept 의 핵심 위험; multi-image dialogue magnitude narrow workload (5-turn dialogue only)
- **재방문 조건**: Phase 1'' gate 통과 시에만 진입; 부분 통과 (1% < PPL < 1.5%) 시 "cross-chunk speculation only" reposition (MRoPE boundary token drop)
- **R67 Scoring sub-axis 요약** (Phase 2' mean **7.30, Conditional Accept**):
  - ★ 최고 sub-axis: **novelty Mechanism (8/10) + Hypothesis (8/10)** — MRoPE × spec decoding 의 새 axis
  - ▼ 최저 sub-axis: **impact Combinatorial (6/10)** — multi-image cumulative -20-25% 가 narrow workload 한정

---

## B14' Cross-Modal Hessian Calibration → **B11' embedded**

- **GAP**: Modality-specific Hessian profiling 의 calibration framework — Sub-2-bit 양자화의 사전 단계
- **시도한 overview**: ~400 LoC training-free calibration phase + 128 sample × 5 task Hessian 측정
- **미선정 사유**: algorithm-expert 명시 "Idea 14' ↔ Idea 11' embedded merge" 권고 — B11' 의 Mechanism (1) Per-Modality Hessian Profiling 의 사전 단계로 자연 흡수
- **재방문 조건**: B11' Phase 3 main paper 의 sub-contribution 으로 자동 포함 (별도 paper 불필요)
- **R67 Scoring sub-axis 요약** (Phase 2' mean **n/a — embedded**):
  - ★ 최고 sub-axis: **n/a (embedded)** — B11' 의 sub-mechanism 으로만 scoring
  - ▼ 최저 sub-axis: **n/a (embedded)** — independent contribution 부재

---

## C1L' VTAP-Sched v2 (HBM3e ECS-aligned + L2 Persistent + TBC Cluster-8)

- **GAP**: Chunk boundary 의 HW-level (HBM3e 256B page-table align + L2 persistent + TBC) layered stack — application-level (A1') 와 결합 가능
- **시도한 overview**: `cudaAccessPolicyWindow` Tier-1 (CUDA 11.0+ public API) + Green Context Tier-2 + HBM3e ECS Ramulator 2.0 분리 + TBC Blackwell SM120 portable cluster size 8 확인
- **미선정 사유**: mean 8.05 **Strong Accept** 이나 C4L' (mean 8.40, reference diversity 10/10) 이 같은 HW-level axis 에서 우선; A1' 와 layered stack partner 로 Phase 3 cross-share dependency 핵심 component
- **재방문 조건**: Phase 3 elective Tier-1 우선순위 2 위 (C6L' 다음); A1' + C1L' layered stack 의 cross-share PR 묶음
- **R67 Scoring sub-axis 요약** (Phase 2' mean **8.05, Strong Accept**):
  - ★ 최고 sub-axis: **novelty Mechanism (9/10) + arch-sys R47/HW-fit/D6 (9/9/9)** — TBC + L2 persistent + 256B page-table align 의 핵심 novelty
  - ▼ 최저 sub-axis: **impact Applicability (7/10) + Adoption (7/10)** — HW-level optimization 의 specific hardware fit (RTX Pro 6000 + HBM3e)

---

## C2L' CDVV-Lite v2 (AMX/VNNI Dual-Path + CXL Tier-2)

- **GAP**: CPU vision offload 의 Intel AMX (Path A) + AMD Zen 5 AVX-512 VNNI BF16 (Path B) dual-path + acceptance-aware dGPU partition
- **시도한 overview**: VLMOpt baseline + acceptance-aware dGPU partition + per-stage DVFS + CXL Tier-2 simulator (gem5 + CXL.mem emulation)
- **미선정 사유**: mean 7.75 가 Tier-1 venue (B9'/B11'/C4L' mean 8.40+) 보다 낮음; Tier-2 슬롯 C5L' (novelty 9/10) 우선; OSDI/ASPLOS 진입 시 acceptance-aware dGPU partition single-axis novelty 약함
- **재방문 조건**: Phase 3 elective Tier-1 우선순위 5 위; CPU vision offload 의 broader applicability proof 시; Threadripper PRO 9965WX + oneDNN AVX-512 BF16 SigLIP-SO400M 실측 통과 시
- **R67 Scoring sub-axis 요약** (Phase 2' mean **7.75, Accept**):
  - ★ 최고 sub-axis: **novelty Combinatorial (9/10) + differentiation Coverage (9/10)** — AMD/Intel dual-path + acceptance-aware partition + DVFS combination
  - ▼ 최저 sub-axis: **arch-sys R47 (7/10)** — CXL Tier-2 분리 후에도 dual-path 의 R47 path complexity 잔존

---

## C6L' UnifiedKV-Hier-NVMe v2 (HBM3e Hot / DDR5 Warm / NVMe Gen5 Cold + Spec Attn Prefetch)

- **GAP**: Long-context video VLM 의 KV tier 계층화 + spec attention IoU 기반 prefetch — Mooncake KVCache-centric 의 single-machine 변종
- **시도한 overview**: io_uring async + liburing + `cache_engine_hier.py` 신규 + RTX Pro 6000 + DDR5 256GB + NVMe Gen5 4TB single workstation fit + Spec attention IoU 70-85% 가정
- **미선정 사유**: mean 8.15 **Strong Accept** 이나 long-context video VLM narrow workload 가 Phase 3 단일 main paper risk; Top-M 6 는 broader applicability (B9' universal / B11' memory / C4L' system) 우선
- **재방문 조건**: Phase 3 elective Tier-1 우선순위 1 위 (즉시 진입 가능); EAGLE-3 attention prob 의 target verify hit ≥ 60% 통과 시
- **R67 Scoring sub-axis 요약** (Phase 2' mean **8.15, Strong Accept**):
  - ★ 최고 sub-axis: **impact Magnitude (9/10) + Adoption (9/10) + differentiation Coverage (9/10)** — Context capacity +165-230% 가 magnitude top + Cost eff +90-130%
  - ▼ 최저 sub-axis: **novelty Hypothesis (8/10)** — Spec attention IoU 측정 method 명시 했으나 가설 layer 잔존

---

## 종합 미선정 12 + Embedded 1 = 13 idea 요약

| # | Idea | Score | Grade | 처리 분류 | Phase 3 Status |
|---|---|---|---|---|---|
| 1 | A1' Mosaic Maestro | 7.85 | Accept | layered partner | Phase 3 partner |
| 2 | A3' Power Grid (= C3L') | 8.05 | Strong Accept | DVFS axis solo | elective rank 4 |
| 3 | A4' Pipeline Conveyor | 7.65 | Accept | CPU vision overlap | Tier-2 reposition |
| 4 | A5' Heartbeat Synchronizer | 7.65 | Accept | KV materialization | Phase 3 component |
| 5 | B8' VAST-Spec | 7.95 | Accept-strong | VAST-γ merge | elective rank 3 |
| 6 | B10' Modality-Boundary | 8.05 | Accept | C2L' merge | merge candidate |
| 7 | B12' Entropy-Gated γ* | 8.40 | Strong Accept | VAST-γ merge | elective rank 3 |
| 8 | B13' Cross-Chunk MRoPE | 7.30 | Conditional | gate-dependent | DROP risk |
| 9 | B14' Hessian Calib | n/a | embedded | → B11' | Phase 3 sub-step |
| 10 | C1L' VTAP-Sched | 8.05 | Strong Accept | HW chunk axis | elective rank 2 |
| 11 | C2L' CDVV-Lite | 7.75 | Accept | CPU vision main | elective rank 5 |
| 12 | C6L' UnifiedKV-NVMe | 8.15 | Strong Accept | KV tier solo | **elective rank 1** |

**총 처리: 17 정식 idea 중 6 Top-M selected + 13 미선정/embedded (위 12 + B14' embedded)**

*Last updated: 2026-05-26 — Phase 2' integrated re-review 의 17 정식 idea 의 Top-M 미선정 12 + embedded 1 = 13 entry.*
