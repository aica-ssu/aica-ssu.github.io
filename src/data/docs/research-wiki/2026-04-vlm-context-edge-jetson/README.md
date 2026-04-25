# VLM Context-aware Serving on Jetson Edge (Session Overview)

> **Date**: 2026-04-25 · **Mode**: 1 (sentence + folder input) · **Lab**: SSU AICA · **Student**: 이방산
>
> **Bundle 형식**: 2026-04-25 R28-R37 hierarchical 적용. 학생 / AI 가 본 README + 자기 소관 idea 파일 1-2 개 만 읽어도 (a) 전체 ideation flow, (b) 시작점 결정, (c) Tier 분기 판단 자율 수행 가능.
>
> **세션 staging**: `__research_wiki/sessions/staging/2026-04-25-vlm-context-edge-jetson-*.md` (3 expert + 3 reviewer = 6 파일)

---

## 0. Executive Summary

본 세션은 **Jetson edge device 3종** (Thor 128GB / Orin NX 16GB / Orin Nano 8GB) 에서 VLM context-aware serving 의 성능·에너지 동시 최적화 axis 를 도출. 이전 v1/v2/v3 세션이 **datacenter HBM 가정** 으로 한 axis 를 의도적으로 회피하고, edge-only 4 가지 제약 (LPDDR5X UMA 102-273 GB/s · DLA-iGPU heterogeneous · thermal/power 7-130 W · NVFP4/INT4 native HW Thor) 위 first-to-report 가능한 axis 만 다룸.

**3 expert × 7 idea = 21 candidate** 중 Phase 2' (novelty / diff / impact 3-axis review + scoop-similarity critique) 로 **Tier-1 Top 3 + Tier-2 Top 3 = 6 개** 최종 선정. 미선정 15 편 의 사유는 [unselected.md](/research-wiki/2026-04/vlm-context-edge-jetson/unselected).

### 0.1 Tier-1 Top 3 (Top-tier venue)

| Rank | Title | Score (N/D/I 평균) | 핵심 mechanism (1줄) | 링크 |
|------|-------|-------------------|---------------------|------|
| 🥇 ★lead | **CacheVeil** — Register-Level SLC Way-Partition for VLM Visual-Token KV Pinning + CPU Workload Isolation. Target: **MICRO 2027 / HPCA 2027** (legacy-sys) | N 8.5 / D 7.0 / I 8.5 = **8.00** | ARM CMN `por_hnf_pwpr` SLC way-partition 으로 visual KV 를 4-way 에 pin + camera ISP/CPU workload 를 나머지 4-way 격리 → L2 conflict miss 28-45% ↓ | [tier1/01-cacheveil.md](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil.md) |
| 🥈 | **SHOAL** — DLA tile-stream KV residence orchestrator. Target: **MLSys 2027 / ASPLOS 2027** (ai-opt) | N 7.5 / D 8.5 / I 8.2 = **8.07** | vLLM PagedAttention block table 에 `residence ∈ {GPU_HBM, GPU_UMA, CPU_pinned}` enum 추가, DLA → KV 전송 destination 을 layer-단위 dynamic 결정 + thermal closed-loop | [tier1/02-shoal.md](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/02-shoal.md) |
| 🥉 | **VESPER** — UMA zero-copy KV ledger with CPU-side patch pruner. Target: **OSDI 2027 / SOSP 2027** (ai-opt) | N 8.0 / D 7.5 / I 8.8 = **8.10** | cudaMallocManaged dual-view KV buffer + CPU NEON SIMD 가 attention score 미만 KV block in-place prune (idle CPU 41-67% 활용) | [tier1/03-vesper.md](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/03-vesper.md) |

### 0.2 Tier-2 독립 Top 3 (4-6p venue)

| Rank | Title | Score | 핵심 (1줄) | 링크 |
|------|-------|-------|------------|------|
| T1 | **TUFA** — Orin Nano-only INT4 vision token early-exit. Target: **IEEE CAL 4p** (ai-opt) | N 6.5 / D 6.5 / I 6.4 = **6.47** | CLIP-ViT-L/14 24-layer 중 spatial entropy 로 12-24 layer 동적 선택 + 7W mode thermal first-to-report | [tier2/01-tufa.md](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/01-tufa.md) |
| T2 | **ShelfSwap** — Thermal-Envelope driven KV zone migration in UMA. Target: **ISLPED 6p / DATE 6p** (legacy-sys) | N 5.5 / D 6.0 / I 6.7 = **6.07** | Skin temp > 78°C 시 visual KV cold layer 를 CPU-affine UMA zone 으로 `cuMemAdvise` migration → GPU 발열 5-9°C ↓ | [tier2/02-shelfswap.md](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap.md) |
| T3 | **Glacier Migrate** — Tegra-PIM emulation via DLA SRAM. Target: **ICCAD 8p / IEEE TCAS-I** (hwpim) | N 6.0 / D 7.0 / I 7.0 = **6.67** | Thor DLA SRAM (~2-4MB) 을 LPDDR5X round-trip bypass buffer 로 활용 (실 PIM 칩 없이 PIM-emulation 효과) | [tier2/03-glacier-migrate.md](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/03-glacier-migrate.md) |

### 0.3 핵심 차별화 (vs 이전 v1/v2/v3 / qwen3vl-deepstack 세션)

이전 세션 (HRTS / ContextMIG / PhaseGraph-VLA / RESIDUA / PRISMATIC / DELTANET TIDE / EXPERT MURMUR / TEMPO BEACON / Triptych / Cartographer / Mosaic / Echo Chamber / Estuary / Interleaved Harmonic / NVFP4 Vault / Dual-Engine Echo) 모두 **datacenter HBM 가정** + GPU-only assumption. 본 세션은 (i) UMA LPDDR5X (zero-copy, page-fault), (ii) DLA + iGPU heterogeneous, (iii) edge SLC 4-16MB cache partition, (iv) thermal envelope 7-130W 위 **edge-only 첫 mechanism** 도출. Metaphor 명사 (CacheVeil / SHOAL / VESPER / TUFA / ShelfSwap / Glacier Migrate) 모두 신규 — 이전 idea 와 충돌 없음.

---

## 1. 학생 연구 실행 흐름 (Post-Ideation Decision Tree)

> Summary 를 읽은 학생이 **어떤 순서로 실험·구현을 진행하고**, 측정 결과에 따라 **Tier-1 vs Tier-2 venue 를 결정**하는 가이드. ideation 과정이 아니라 **실험 plan 결정 트리** 다.

<svg viewBox="0 0 880 1280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="학생 연구 실행 흐름 결정 트리" style="width:100%;max-width:880px;height:auto;display:block;margin:1rem auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1E2761"/>
    </marker>
    <marker id="arrCoral" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#F96167"/>
    </marker>
  </defs>
  <text x="440" y="28" text-anchor="middle" font-size="20" font-weight="700" fill="#1E2761">학생 연구 실행 흐름 — Tier-1/2 분기 결정 트리</text>
  <text x="440" y="52" text-anchor="middle" font-size="12" fill="#666" font-style="italic">각 ◇ 분기는 자기 측정값으로 직접 판단. 화살표 옆 숫자 = 권장 주차 (총 12주 가정).</text>
  <!-- Step 1 -->
  <rect x="140" y="80" width="600" height="105" rx="10" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
  <text x="440" y="106" text-anchor="middle" font-size="15" font-weight="700" fill="#1E2761">Step 1 — 가용 Jetson HW + Tier-1 idea 매칭 (Wk 0)</text>
  <text x="440" y="128" text-anchor="middle" font-size="12" fill="#1a1a1a">Jetson Thor 128GB → ★ CacheVeil (SLC way-partition, MICRO/HPCA)</text>
  <text x="440" y="146" text-anchor="middle" font-size="12" fill="#1a1a1a">Jetson Orin NX/AGX → SHOAL (DLA tile-stream, MLSys/ASPLOS)</text>
  <text x="440" y="164" text-anchor="middle" font-size="12" fill="#1a1a1a">Jetson Orin Nano 8GB → VESPER (UMA pruner, OSDI/SOSP) 또는 Tier-2</text>
  <line x1="440" y1="185" x2="440" y2="215" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
  <!-- Step 2 -->
  <rect x="140" y="215" width="600" height="80" rx="10" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
  <text x="440" y="240" text-anchor="middle" font-size="15" font-weight="700" fill="#1E2761">Step 2 — Baseline 재현 (Wk 1-2)</text>
  <text x="440" y="262" text-anchor="middle" font-size="12" fill="#1a1a1a">Qwen3-VL-4B / InternVL3-2B prefill+decode latency 측정 (3-run avg)</text>
  <text x="440" y="280" text-anchor="middle" font-size="12" fill="#1a1a1a">Tegrastats + Nsight Compute counter (lts__t_sectors / dram__bytes_read)</text>
  <line x1="440" y1="295" x2="440" y2="325" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
  <!-- Decision 1 -->
  <polygon points="440,325 720,378 440,430 160,378" fill="#FFFFFF" stroke="#F96167" stroke-width="2.5"/>
  <text x="440" y="370" text-anchor="middle" font-size="14" font-weight="700" fill="#1E2761">◇ Baseline ±5% 일치?</text>
  <text x="440" y="390" text-anchor="middle" font-size="11" fill="#666">기준: paper / vendor 보고 latency</text>
  <text x="450" y="450" font-size="13" font-weight="700" fill="#1E2761">Yes</text>
  <line x1="440" y1="430" x2="440" y2="465" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
  <text x="730" y="372" font-size="12" font-weight="700" fill="#F96167">No → 환경 보정</text>
  <line x1="720" y1="378" x2="850" y2="378" stroke="#F96167" stroke-width="2" marker-end="url(#arrCoral)"/>
  <rect x="780" y="395" width="90" height="58" rx="6" fill="#F9E795" stroke="#F96167" stroke-width="1.5"/>
  <text x="825" y="416" text-anchor="middle" font-size="10" font-weight="600" fill="#1E2761">JetPack /</text>
  <text x="825" y="430" text-anchor="middle" font-size="10" font-weight="600" fill="#1E2761">CUDA / PyTorch</text>
  <text x="825" y="445" text-anchor="middle" font-size="10" font-weight="600" fill="#1E2761">버전 정렬</text>
  <!-- Step 3 -->
  <rect x="140" y="465" width="600" height="85" rx="10" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
  <text x="440" y="490" text-anchor="middle" font-size="15" font-weight="700" fill="#1E2761">Step 3 — Mechanism #1 구현 (Wk 3-6)</text>
  <text x="440" y="512" text-anchor="middle" font-size="12" fill="#1a1a1a">Source-verified path (R32 ✅) 사용 — vLLM/SGLang/cuDNN/NVDLA primitives</text>
  <text x="440" y="530" text-anchor="middle" font-size="12" fill="#1a1a1a">Single-mech delta 측정 → 첫 paper-worthy 신호 확보</text>
  <line x1="440" y1="550" x2="440" y2="580" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
  <!-- Decision 2 -->
  <polygon points="440,580 720,633 440,685 160,633" fill="#FFFFFF" stroke="#F96167" stroke-width="2.5"/>
  <text x="440" y="620" text-anchor="middle" font-size="14" font-weight="700" fill="#1E2761">◇ Mech #1 예상 Δ 도달?</text>
  <text x="440" y="640" text-anchor="middle" font-size="11" fill="#666">예: latency -10% / energy -8% 일관 측정</text>
  <text x="440" y="658" text-anchor="middle" font-size="11" fill="#666">(idea 별 "예상 효과" 표 기준)</text>
  <text x="450" y="705" font-size="13" font-weight="700" fill="#1E2761">Yes</text>
  <line x1="440" y1="685" x2="440" y2="720" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
  <text x="730" y="628" font-size="12" font-weight="700" fill="#F96167">No → Tier-2 단축</text>
  <line x1="720" y1="633" x2="850" y2="633" stroke="#F96167" stroke-width="2" marker-end="url(#arrCoral)"/>
  <rect x="780" y="650" width="90" height="58" rx="6" fill="#F9E795" stroke="#F96167" stroke-width="1.5"/>
  <text x="825" y="671" text-anchor="middle" font-size="10" font-weight="600" fill="#1E2761">single-mech</text>
  <text x="825" y="685" text-anchor="middle" font-size="10" font-weight="600" fill="#1E2761">Tier-2 spinoff</text>
  <text x="825" y="700" text-anchor="middle" font-size="10" font-weight="600" fill="#1E2761">으로 마무리</text>
  <!-- Step 4 -->
  <rect x="140" y="720" width="600" height="90" rx="10" fill="#CADCFC" stroke="#1E2761" stroke-width="2"/>
  <text x="440" y="745" text-anchor="middle" font-size="15" font-weight="700" fill="#1E2761">Step 4 — Mech #2 + #3 통합 + ablation (Wk 7-12)</text>
  <text x="440" y="767" text-anchor="middle" font-size="12" fill="#1a1a1a">2^N factorial / per-mech contribution / 3+ peer-reviewed baseline</text>
  <text x="440" y="785" text-anchor="middle" font-size="12" fill="#1a1a1a">Combined gain 측정 — Tier-1 cut (≥15% latency / ≥20% energy) 검증</text>
  <text x="440" y="803" text-anchor="middle" font-size="11" fill="#666" font-style="italic">기존 baseline: VL-Cache · VLCache · Nova · throttLL'eM · Four Over Six · FastVLM · SparseDVFS</text>
  <line x1="440" y1="810" x2="440" y2="840" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
  <!-- Decision 3 -->
  <polygon points="440,840 720,893 440,945 160,893" fill="#FFFFFF" stroke="#F96167" stroke-width="2.5"/>
  <text x="440" y="880" text-anchor="middle" font-size="14" font-weight="700" fill="#1E2761">◇ Tier-1 threshold 도달 + 3+ baseline?</text>
  <text x="440" y="898" text-anchor="middle" font-size="11" fill="#666">latency ≥15% / energy ≥20% / OR HW novelty 명확</text>
  <text x="440" y="916" text-anchor="middle" font-size="11" fill="#666">+ peer-reviewed baseline 비율 ≥50% (R-Reference Integrity)</text>
  <line x1="320" y1="935" x2="220" y2="975" stroke="#1E2761" stroke-width="2" marker-end="url(#arr)"/>
  <text x="265" y="965" font-size="13" font-weight="700" fill="#1E2761">Yes</text>
  <line x1="560" y1="935" x2="660" y2="975" stroke="#F96167" stroke-width="2" marker-end="url(#arrCoral)"/>
  <text x="600" y="965" font-size="13" font-weight="700" fill="#F96167">No</text>
  <!-- Tier-1 outcome -->
  <rect x="80" y="975" width="300" height="120" rx="12" fill="#E5F0E8" stroke="#2C5F2D" stroke-width="2.5"/>
  <text x="230" y="1002" text-anchor="middle" font-size="15" font-weight="700" fill="#2C5F2D">★ Tier-1 Submit</text>
  <text x="230" y="1024" text-anchor="middle" font-size="12" fill="#1a1a1a">CacheVeil → MICRO/HPCA 2027 (12-15p)</text>
  <text x="230" y="1042" text-anchor="middle" font-size="12" fill="#1a1a1a">SHOAL → MLSys/ASPLOS 2027 (12-18p)</text>
  <text x="230" y="1060" text-anchor="middle" font-size="12" fill="#1a1a1a">VESPER → OSDI/SOSP 2027 (12-15p)</text>
  <text x="230" y="1080" text-anchor="middle" font-size="11" fill="#666" font-style="italic">3-mech 모두 + adaptive attacker discussion</text>
  <!-- Tier-2 outcome -->
  <rect x="500" y="975" width="300" height="120" rx="12" fill="#FFF7DA" stroke="#B8860B" stroke-width="2.5"/>
  <text x="650" y="1002" text-anchor="middle" font-size="15" font-weight="700" fill="#8B5A00">Tier-2 Submit (Spinoff)</text>
  <text x="650" y="1024" text-anchor="middle" font-size="12" fill="#1a1a1a">CacheVeil-Lite → DAC 6p (single way-partition)</text>
  <text x="650" y="1042" text-anchor="middle" font-size="12" fill="#1a1a1a">또는 독립 Tier-2 (TUFA / ShelfSwap / Glacier)</text>
  <text x="650" y="1060" text-anchor="middle" font-size="12" fill="#1a1a1a">single-mech + single-HW + first-to-report scope</text>
  <text x="650" y="1080" text-anchor="middle" font-size="11" fill="#666" font-style="italic">IEEE CAL 4p / DATE 6p / ISLPED 6p / ICCAD 8p</text>
  <!-- Caption -->
  <text x="440" y="1140" text-anchor="middle" font-size="12" font-weight="600" fill="#1E2761">실패 시 회복 경로 (모든 분기)</text>
  <text x="440" y="1160" text-anchor="middle" font-size="11" fill="#1a1a1a">Step 2 ❌ → Wk 1-2 환경 보정 후 재시도 / Step 3 ❌ → Tier-2 spinoff 으로 마무리</text>
  <text x="440" y="1178" text-anchor="middle" font-size="11" fill="#1a1a1a">Step 4 ❌ → Tier-1 cut 미달 시 Tier-2 spinoff (single-mech, single-HW) 제출</text>
  <text x="440" y="1196" text-anchor="middle" font-size="11" fill="#1a1a1a">미선정 15편의 자세한 사유 + 재방문 조건 → "미선정 로그" 페이지 참조</text>
  <text x="440" y="1228" text-anchor="middle" font-size="11" fill="#666" font-style="italic">Idea 선택은 [Tier-1 표 (§2)] / [Tier-2 표 (§3)] 의 score + venue + Mechanism 행을 본인 자원·관심에 맞춰 결정.</text>
</svg>

**결정 가이드 요약**:
- **Step 1 매칭** — 본인이 사용할 수 있는 Jetson HW 기준으로 idea 1 개 선택. Thor 가용 → CacheVeil; Orin NX → SHOAL; Orin Nano 만 → VESPER 또는 Tier-2.
- **Step 2 baseline** — paper/vendor 가 보고한 latency 와 ±5% 일치 안 하면 mechanism 측정 무의미. JetPack/CUDA/PyTorch 버전 + thermal mode (NVPModel) 정렬 우선.
- **Step 3 mech #1** — single mechanism 만으로도 본인 idea 의 "예상 효과" 표 첫 행 (예: SHOAL 의 "p50 latency -10%") 에 도달해야 Tier-1 진행 정당화.
- **Step 4 통합** — 3 개 mechanism 결합 후 Tier-1 cut (latency ≥15% / energy ≥20%) 도달 시 top-tier 제출. 미달 시 single-mech Tier-2 spinoff 으로 빠른 publication.
- **Score 7.5 cut + Concurrent scoop check** — Phase 2' similarity critique 통해 Nova `arXiv:2509.21301` (50-70% concurrent) 등 강한 scoop 압박 시 mechanism repositioning 또는 미선정.

### Tier-1 Path 가이드 (가장 유망 ideation 부터)

**Path Lead — CacheVeil ★ (legacy-sys 출처, score 8.00, MICRO/HPCA 후보)**:
- **시작점**: Orin AGX 64GB (4MB SLC, 8-way 가정) baseline TPOT/Decode 측정 → ARM CMN `por_hnf_pwpr` partition policy IOCTL 노출 확인.
- **Tier-1 진입 조건**: 4-4 partition 시 visual KV pinning 후 (a) L2 conflict miss -28% 이상, (b) decode TPOT p99 -15% 이상.
- **추가 의무 (Phase 1' 정제 사항)**:
  - Baseline 보강: VL-Cache OpenReview verified ([HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub)), Mind the Memory Gap [arXiv:2503.08311](https://arxiv.org/abs/2503.08311), libsmctrl ECRTS 2025
  - JetPack 7.1 fixed-target + sysfs fallback 명시 (BPMP IOCTL 비공식 우려 해소)
  - Mirror Lake 와의 axis 분리 (cache layer vs IOMMU layer) — 본 idea 는 cache layer 만

**Path B — SHOAL (ai-opt, score 8.07, MLSys/ASPLOS)**:
- **시작점**: Jetson Thor stock vLLM v1 빌드 + Qwen3-VL-4B baseline TTFT/decode → DRM enum 추가 micro-benchmark.
- **Tier-1 진입 조건**: image patch ≥ 256, batch ≥ 2 일 때 prefill TTFT 18-24% ↓ + accuracy drop ≤ 0.5pp.
- **Phase 1' 정제**: baseline FastVLM (CVPR 2025) + V2Drop 추가, DLA 부재 환경 fallback (`GPU_HBM ↔ GPU_UMA` 만) 강화.

**Path C — VESPER (ai-opt, score 8.10, OSDI/SOSP)**:
- **시작점**: Orin NX 16GB cudaMallocManaged + memAdvise path 검증 → NEON kernel skeleton.
- **Tier-1 진입 조건**: long-context (≥ 8K) decode tok/s 22-28% ↑ + algorithmic eviction (H2O baseline) 와 system gain 분리 ablation.
- **Phase 1' 정제**: VLCache [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) + Dissecting CPU-GPU UPM [arXiv:2508.12743](https://arxiv.org/abs/2508.12743) baseline 추가.

### Tier-2 Path 가이드

- **TUFA (T2)**: Orin Nano 7W single mechanism. **TUFA 의 trivialty 회피 명시** — vision early-exit 자체는 CVPR 2024-25 다수 (DynamicViT/A-ViT/V2Drop) 이지만 Orin Nano 7W thermal envelope 측정은 first-to-report.
- **ShelfSwap (T2)**: novelty 5.5 — SparseDVFS / Four Over Six 등 adjacent 명시. 단일 mechanism + ISLPED 6p scope 만.
- **Glacier Migrate (T2)**: pixel-level workload-specific axis 명시 — DLA SRAM 의 어떤 op subset 만 PIM-emulation 가능한지 정량화. ICCAD 8p 분량으로 격상.

### Drop / Pivot 가이드

- **Phase 1' improve-first**: 모든 6 idea 의 mechanism diff = 0 (변경 없음, 보강만). Phase 1'' 최종 선정 confirmed.
- **DualLane 미선정 사유**: Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) 50-70% CONCURRENT, repositioning 부담 큼. VESPER 가 더 distinctive UMA axis.
- **PageWeave 미선정 사유**: VESPER 와 axis 부분 중복, novelty 비슷하나 reviewer-impact 격차.
- **Watershed/KILN/STELE/ThermalLoom/Mirror Lake/TGQ/Tessellated Bank Affinity**: CONCURRENT scoop 강 또는 score < 7.5.

---

## 2. 연구 진행 Meta

### 2.1 Input

- **사용자 쿼리 원문** (한 글자도 변경 없이 인용):
  > "VLM context-aware serving on Jetson edge (Thor 128GB / Orin NX 16GB / Orin Nano 8GB). 이전 v1/v2/v3 + qwen3vl-deepstack 세션이 데이터센터 HBM 가정으로 한 axis 만 다뤘으니, 본 세션은 edge-only 4 axis (LPDDR5X UMA / DLA-iGPU heterogeneous / thermal envelope / NVFP4 native) 위에서 first-to-report 가능한 mechanism 만 도출."
- **Mode**: 1 (sentence + folder input)
- **실행 일시**: 2026-04-25
- **관련 이전 세션**:
  - [2026-04-23 edge-vlm-energy](/summary/2026-04-23-energy-efficient-edge-vlm.md) (전제: Jetson edge but energy-only)
  - [2026-04-23 prism-vlm-kv-extension](/summary/2026-04-23-prism-vlm-kv-extension.md) (KV management extension)
  - [2026-04-24 qwen3vl-deepstack-edge](/summary/2026-04-24-qwen3vl-deepstack-edge.md) (DeepStack 6-tier scheduler 가정)

### 2.2 접근 키워드 (4-8개)

- **도메인 (A)**: VLM context-aware serving, Jetson Thor / Orin NX / Orin Nano edge inference, vLLM v1 + SGLang + llama.cpp + TensorRT-LLM
- **HW 제약 (B)**: LPDDR5X UMA 102-273 GB/s, DLA + iGPU heterogeneous compute, SLC/L3 cache 4-16MB, thermal envelope 7-130W (DVFS), NVFP4 (Thor only)
- **제안 mechanism 축**: KV residence enum (SHOAL), UMA dual-view + NEON pruner (VESPER), SLC way-partition (CacheVeil), vision early-exit (TUFA), thermal-zone migration (ShelfSwap), DLA SRAM PIM-emulation (Glacier Migrate)
- **타겟 지표**: TPOT p99, decode tok/s, energy/token, thermal throttle entry, L2/SLC miss rate

### 2.3 중점적으로 고려한 축

1. **Edge-only first-to-report** — 모든 idea 가 edge-specific HW 제약 위에서만 mechanism 정당화. Datacenter 가정 절대 금지.
2. **Multi-platform source verification (R32)** — vLLM v1 / SGLang / llama.cpp / TensorRT-LLM 4 platform path 명시 + ✅/⚠️/🔧 표.
3. **Triple-expert orthogonality** — ai-opt (serving 축) / legacy-sys (cache+UMA+thermal) / hwpim (compute-unit + memory-near).
4. **Dual Track (Tier-1 + Tier-2)** — Top-tier 13-18p + Tier-2 4-8p paper-pair 가능성 동시 평가.
5. **Workload-driven (R23)** — Step 0-α W1-W7 (MobileAIBench / EdgeMoE / PerfVec / LLMCarbon / Jetson NVIDIA blog / VLMBench / PowerInfer-2) 모두 인용.

### 2.4 의도적으로 제외한 축 + 이유

| 제외 축 | 이유 |
|---------|------|
| **Datacenter HBM-only assumption** | 이전 v1/v2/v3 모두 다룸. 본 세션 핵심 차별. |
| **VLM training / RLHF / Alignment** | 본 세션 inference-only. |
| **실 PIM 칩 (HBM-PIM, AttAcc, Pimba MICRO 2025)** | Jetson 라인업 모두 LPDDR5X 표준 + 별도 PIM 칩 없음. 사용자 지시 "PIM 억지 적용 금지". 단, S3 Glacier Migrate 만 DLA SRAM 가상 PIM 으로 emulation. |
| **Cross-device / multi-Jetson cluster** | Jetson 은 single-device edge 가 표준 deployment. ModServe NSDI 2024 가정 불일치. |
| **VLA (Vision-Language-Action) robot control** | 2026-04-23 PhaseGraph-VLA 별도 세션 다룸. |
| **Side-channel security on routing** | 2026-04-21 mode1 MoE fingerprinting 등 별도 세션. |

### 2.5 외부 탐색 범위

- **Phase 0/1**: 3 expert × ~22 paper = 약 66 reference (peer-reviewed ratio 50%+ 모두 충족)
- **Reference Integrity R1**: 모든 arxiv ID + venue 확인. OpenReview verified: VL-Cache [HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub)
- **Phase 2 similarity critique**: 21 idea 모두 scoop-check + 6 CONCURRENT idea 차별화 강화 권고

### 2.6 평가 기준

- **Tier-1 rubric**: Novelty 7.5+ / Diff 7.5+ / Impact 8.0+ / Mechanism 2-4 OK / Triple platform path
- **Tier-2 rubric**: Novelty first-to-report in narrow scope / Diff 2-3 baseline clear delta / Mechanism 1 권장 / Single HW
- **Mechanism budget**: 아이디어당 ≤4 (CacheVeil 4 / SHOAL 4 / VESPER 4 = 균형)
- **Tier budget**: physical ≤3 + software ≤3-4

### 2.7 사용된 전문가 에이전트

- `ai-optimization-expert` (서빙 최적화 / KV management / specdec)
- `legacy-system-expert` (UMA + page-fault / cache partition / thermal envelope)
- `hw-pim-accelerator-expert` (DLA + GPU heterogeneous / LPDDR5X bank / NVFP4 thermal)
- 리뷰어: `novelty-reviewer`, `differentiation-reviewer`, `impact-reviewer`

---

## 3. Tier-1 Top 3 표 (간략 link 표)

| # | Idea | Venue | Score (N / D / I) | 핵심 mechanism (1줄) | 출처 expert |
|---|------|-------|-------------------|---------------------|-------------|
| 1 ★ | [CacheVeil](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil.md) | MICRO 2027 / HPCA 2027 | 8.5 / 7.0 / 8.5 = 8.00 | ARM CMN SLC 4-way pin (visual KV) + 4-way isolation (CPU/ISP) | legacy-sys |
| 2 | [SHOAL](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/02-shoal.md) | MLSys 2027 / ASPLOS 2027 | 7.5 / 8.5 / 8.2 = 8.07 | DLA → KV residence enum (GPU_HBM/UMA/CPU_pinned) layer-단위 dynamic | ai-opt |
| 3 | [VESPER](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/03-vesper.md) | OSDI 2027 / SOSP 2027 | 8.0 / 7.5 / 8.8 = 8.10 | UMA cudaMallocManaged dual-view + NEON SIMD CPU pruner | ai-opt |

## 4. Tier-2 독립 Top 3 표

| # | Idea | Venue | Score (N / D / I) | 핵심 mechanism (1줄) | 출처 expert |
|---|------|-------|-------------------|---------------------|-------------|
| T1 | [TUFA](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/01-tufa.md) | IEEE CAL 4p | 6.5 / 6.5 / 6.4 = 6.47 | CLIP-ViT 24-layer 중 spatial entropy 로 12-24 layer 동적 exit (Orin Nano 7W only) | ai-opt |
| T2 | [ShelfSwap](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap.md) | ISLPED 6p / DATE 6p | 5.5 / 6.0 / 6.7 = 6.07 | Skin temp > 78°C → visual KV cold layer CPU-affine zone migration | legacy-sys |
| T3 | [Glacier Migrate](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/03-glacier-migrate.md) | ICCAD 8p / IEEE TCAS-I | 6.0 / 7.0 / 7.0 = 6.67 | Thor DLA SRAM 2-4MB 을 LPDDR5X round-trip bypass (PIM emulation) | hwpim |

## 5. 미선정 15편 요약 표

상세는 [unselected.md](/research-wiki/2026-04/vlm-context-edge-jetson/unselected):

| Idea | 1줄 요약 | 미선정 사유 (간략) |
|------|---------|-------------------|
| KILN | NVFP4-aware sample-level fragility gating | CONCURRENT 55% Four Over Six, score 5.5 |
| STELE | Thermal-budget-aware speculative decoding | CONCURRENT 60% SLED+Ghidorah, ThermalLoom overlap |
| CINDER | Sustained-throttle DVFS lookahead | Linux schedutil + nvpmodel adjacent, score 5.0 |
| RIVET | Orin DLA-only INT8 image embedding LRU cache | SGLang RadixAttention overlap, score 4.0 |
| ThermalLoom | Thermal-throttle BW degradation + NVPModel hide | ShelfSwap + STELE 와 thermal axis 중복 |
| PageWeave | Visual-token lifetime classifier + cgroup | VESPER 와 UMA axis 중복 |
| DualLane | Dual-NVDLA + GPU 3-way + SLC partition | Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) 50-70% CONCURRENT |
| FrostHint | ISP zero-copy + L2 pre-warming hint | CacheVeil baseline 으로 흡수 권고, score 3.5 |
| TileGate | L2-capacity-aware tile size | CacheVeil baseline 으로 흡수 권고, score 3.5 |
| Watershed | Phase-level coarse DLA+GPU dispatcher | DuetServe + Nova CONCURRENT 58% |
| TGQ (Thermal Gradient Quantizer) | NVFP4 thermal recalibration (Thor) | CONCURRENT 55% Four Over Six + NVFP4 QAD |
| Tessellated Bank Affinity | LPDDR5X bank-modality placement | NVIDIA bank mapping API 부재, score 7.0 borderline |
| Mirror Lake | UMA modality isolation IOMMU | CacheVeil 와 cache axis overlap |
| Pinwheel | Orin Nano DLA-only short VQA | TUFA 와 Orin Nano scope 충돌, score 4.0 |
| Diode Ladder | NVFP4 bias compensation circuit | 회로 시뮬 필요 + Thor only, score 3.0 |

---

## 6. R35 약어 / 핵심 용어 풀이

본 세션 전체에서 등장하는 핵심 약어 + 알고리즘 + 도메인 용어. 각 idea 파일도 자기에게 필요한 것을 다시 풀이 (학생 self-sufficient).

- **VLM** (Vision-Language Model) — 이미지+텍스트 동시 처리 모델 (예: Qwen3-VL-4B, InternVL3-2B, LLaVA-NeXT-7B, MiniCPM-V-2.6).
- **VLA** (Vision-Language-Action) — 로봇 제어용 VLM 변형. 본 세션 직접 다루지 않음.
- **Jetson Thor** — NVIDIA 차세대 edge SoC. **128GB LPDDR5X 273 GB/s**, Blackwell-gen GPU + Thor DLA + 14-core Neoverse V3AE, MAXN_SUPER 130W. NVFP4 native, L3 16MB. ([RidgeRun spec](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Thor:_Powering_the_Future_of_Physical_AI))
- **Jetson Orin NX 16GB** — 102 GB/s LPDDR5X UMA, 8-core Cortex-A78AE, Ampere-gen GPU, NVDLA v2.0 ×2, SLC 4MB. 25W mode. 본 세션 secondary HW.
- **Jetson Orin Nano 8GB** — 68 GB/s LPDDR5, 6-core Cortex-A78AE, Ampere GPU, NVDLA v2.0 ×1, SLC 매우 작거나 없음. 7W mode (TUFA scope).
- **DLA / NVDLA** (NVIDIA Deep Learning Accelerator) — Jetson 의 fixed-function vision/conv accelerator. NvMediaTensor API, FP16/INT8 only, GPU fallback 가능 (`--useDLACore=0/1`). ([NVDLA spec](https://nvdla.org/hw/v1/hwarch.html), [TensorRT 10.9 docs](https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/inference-library/work-with-dla.html))
- **LPDDR5X** — Mobile/edge용 DDR memory standard. Jetson 의 single physical pool 을 GPU/CPU/DLA 가 공유 (= UMA).
- **UMA** (Unified Memory Architecture) — Jetson 처럼 GPU+CPU+DLA 가 동일 LPDDR5X physical pool 공유. CUDA Managed Memory + memAdvise 로 access 가능.
- **SLC** (System Level Cache) — Jetson Orin AGX 4MB / Thor 추정 16MB. CPU L3 와 GPU L2 사이에 위치, ARM CMN `por_hnf_pwpr` register 로 way-partition 가능.
- **NVFP4** — NVIDIA Blackwell-gen FP4 numeric format (E2M1 + E4M3 sub-block scale). Thor only native, Orin 미지원. 1.6× speedup over INT8 but accuracy drop 0.3-1.8pp 가능.
- **DVFS** (Dynamic Voltage and Frequency Scaling) — runtime clock 조정. Jetson 의 Linux thermal governor 또는 사용자 NVPModel.
- **NVPModel** — Jetson power/thermal mode preset (예: MAXN_SUPER 130W / 60W / 30W / 15W / 7W). 전이 비용 평균 40-110ms ([EdgeReasoning IISWC 2025](https://iiswc.org/iiswc2025/accepted-papers.html)).
- **cudaMallocManaged / cuMemAdvise** — CUDA Unified Memory API. Jetson UMA 환경에서 zero-copy + page-fault 기반 migration 지원. ([NVIDIA Tech Blog](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/))
- **WCRT** (Worst-Case Response Time) — real-time scheduling 의 latency 상한 분석. ECRTS / RTAS / RTSS 의 표준 metric.
- **IISWC** (IEEE International Symposium on Workload Characterization) — workload characterization 의 top venue. EdgeReasoning / Mind the Memory Gap 등 본 세션 motivation.
- **ISPASS** (IEEE International Symposium on Performance Analysis of Systems and Software) — performance analysis venue. Vision Transformer Computation 2024 / CPU-GPU Coupled 2025 인용.
- **PagedAttention** — vLLM 의 KV cache management 기법. KV block size = 16 token, block table = logical-to-physical address mapping. 본 세션 다수 idea 가 block table extension. ([vLLM docs](https://docs.vllm.ai/))
- **RadixAttention** — SGLang 의 prefix tree KV management. token prefix sharing. 본 세션 RIVET (미선정) baseline.
- **NEON / SVE2** — ARM Cortex-A 의 SIMD intrinsic. 16 KV block 단위 in-place pruning 가능 (VESPER mechanism).
- **NvMediaTensor** — NVIDIA DLA 의 non-public scheduling/buffer API. SHOAL/DualLane/Watershed 가 ⚠️ 의존.
- **BPMP** (Boot and Power Management Processor) — Tegra (Jetson) 의 별도 ARM Cortex-R 코어. Cache partition / DVFS / sysfs IOCTL 노출 (CacheVeil mechanism).
- **`por_hnf_pwpr`** — ARM CoreLink CMN System Level Cache 의 way-allocation policy register. partition-wise allocation policy. ([ARM KA005251](https://developer.arm.com/documentation/ka005251/latest/))
- **DC CIVAC** (Data Cache Clean and Invalidate by Virtual Address to Coherency point) — ARMv8-A cache flush instruction. CacheVeil 의 phase transition 시 사용.
- **EAGLE-3 / Medusa** — speculative decoding 기법. EAGLE-3 = draft-target decoupled hidden-state, Medusa = multi-head parallel draft. STELE (미선정) 의 baseline.
- **VL-Cache** — VLM visual KV layer-adaptive sparsity-aware compression. ICLR 2025 OpenReview verified [HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub). CacheVeil / VESPER 직접 baseline.
- **VLCache** (대문자 V — 별도 paper) — 98% vision token reuse, [arXiv:2512.12977](https://arxiv.org/abs/2512.12977). VESPER 직접 baseline.
- **FastVLM** — vision encoder cost reduction, CVPR 2025. SHOAL 직접 baseline.
- **V2Drop** — vision token dropping, [arXiv:2509.01552](https://arxiv.org/abs/2509.01552). SHOAL/TUFA baseline.
- **TokenFLEX** — vision token granularity dynamic, [arXiv:2504.03154](https://arxiv.org/abs/2504.03154). TUFA adjacent.
- **Nova** — VLM 3-stage cross-stage parallelization, [arXiv:2509.21301](https://arxiv.org/abs/2509.21301). DualLane (미선정) 50-70% CONCURRENT scoop.
- **Dissecting CPU-GPU UPM** — UMA + page-fault overhead 정량, [arXiv:2508.12743](https://arxiv.org/abs/2508.12743). VESPER baseline.
- **ThrottLL'eM** — predictive throttling ASPLOS-style, [arXiv:2408.05235](https://arxiv.org/abs/2408.05235). STELE/ThermalLoom baseline.
- **CLONE** — LLM layer-boundary DVFS adaptation, USENIX ATC 2025. ThermalLoom baseline.
- **DuetServe** — disaggregated prefill-decode harmonization, [arXiv:2511.04791](https://arxiv.org/abs/2511.04791). Watershed CONCURRENT 58%.
- **XSched** — XPU preemptive scheduling, OSDI 2025. DualLane baseline.
- **Hardware Compute Partitioning (libsmctrl)** — GPU SM/TPC partition, ECRTS 2025 [Bakita et al.](https://drops.dagstuhl.de/storage/00lipics/lipics-vol335-ecrts2025/LIPIcs.ECRTS.2025.21/LIPIcs.ECRTS.2025.21.pdf). CacheVeil 의 직접 reference (compute side, CacheVeil 은 memory side).
- **EdgeReasoning** — Jetson reasoning LLM, IISWC 2025 [accepted](https://iiswc.org/iiswc2025/accepted-papers.html). DVFS 전이 비용 40-110ms motivation.
- **Mind the Memory Gap** — edge LLM memory-bound, [arXiv:2503.08311](https://arxiv.org/abs/2503.08311). CacheVeil/Tessellated motivation.
- **MMMU / RefCOCO+ / VQAv2 / DocVQA / MileBench / MVBench / Video-MME** — VLM 평가 benchmark. 본 세션 idea 별 dataset 선택 근거.
- **Tier-1 vs Tier-2 (venue)** — Tier-1 = MICRO / HPCA / ISCA / ASPLOS / OSDI / SOSP / MLSys 13-18p. Tier-2 = DAC / DATE / ISLPED / ICCAD / IEEE CAL / IEEE ESL / TCAS-I 4-8p.
- **Scoop / Concurrent / Adjacent** — Phase 2 similarity-critique 분류. 70%+ mechanism 일치 = Scoop (DROP), 50-70% = Concurrent (차별화 강화 의무), 30-50% = Adjacent (baseline 만 포함).

---

## 7. 다음 단계 (학생 / AI 추천 read order)

본 bundle 을 학생 또는 AI agent 가 활용할 때 권장 4-step:

1. **README** (현재 파일) — 전체 ideation flow + Tier 분기 + 약어 glossary 자율 학습.
2. **Flow Chart 의 분기 노드** — 자기 환경 (Thor 가용? DLA 사용?) 에 맞는 path 선택.
3. **Tier-1 lead idea 파일** ([CacheVeil](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/01-cacheveil.md)) → Mechanism 4 elements + Implementation Steps 정독.
4. **Preliminary Analysis Metrics** 항목 → 자기 자원 (workstation/lab) 으로 Step 1 baseline 재현 → Tier-1 진입 조건 만족 여부 판단 → Tier-2 fallback 또는 Tier-1 paper 작성.

추가 탐색 자료:
- 상세 Phase 로그: `__research_wiki/sessions/staging/2026-04-25-vlm-context-edge-jetson-*.md` (3 expert + 3 reviewer)
- 미선정 사유 상세: [unselected.md](/research-wiki/2026-04/vlm-context-edge-jetson/unselected)
- 관련 이전 세션:
  - [2026-04-23 edge-vlm-energy](/summary/2026-04-23-energy-efficient-edge-vlm.md)
  - [2026-04-24 qwen3vl-deepstack-edge](/summary/2026-04-24-qwen3vl-deepstack-edge.md)
- wiki entry: `__research_wiki/ideas.md`, `papers.md`, `concepts.md`, `index.md`, `README.md`
