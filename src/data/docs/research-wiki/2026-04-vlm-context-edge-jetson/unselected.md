# 미선정 아이디어 15편 — Phase 1'' Drop / Hold 사유

> [← Session Overview](/research-wiki/2026-04/vlm-context-edge-jetson)

본 세션 21 idea (3 expert × 7) 중 Phase 2 (novelty / diff / impact 3-axis review + scoop-similarity critique) 후 Phase 1'' 최종 선정 사항. **2026-04-25 R45 적용 후 갱신**:
- **Tier-1 Top 3 채택**: [VESPER](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/03-vesper.md) (★ lead, R45 후 promotion) / [SHOAL](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/02-shoal.md) / [DualLane](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/04-duallane.md) (R45 후 신규 진입)
- **Tier-2 독립 Top 3 채택**: [TUFA](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/01-tufa.md) / [ShelfSwap](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/02-shelfswap.md) / [CacheVeil-Sim](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/03-cacheveil-sim.md) (CacheVeil 원안의 simulator-path spinoff)
- **R45 demotion 2 idea**: CacheVeil 원안 (Tier-1 → Tier-2 simulator spinoff) / Glacier Migrate (Tier-2 → unselected)
- **미선정 15편 + R45 demotion 2편 = 17편**: 본 파일.

학생 / AI 가 "이 axis 가 왜 빠졌는가" 빠르게 판단할 수 있도록 idea / venue 원안 / 미선정 사유 / 재방문 조건 / 누가 도출했는지 명시.

---

## 미선정 표 (15편)

| # | Idea | 출처 expert | venue 원안 | Phase 2 score (N/D/I) | 미선정 사유 | 재방문 조건 |
|---|------|-------------|-----------|----------------------|-------------|-------------|
| 1 | **KILN** (NVFP4-aware sample-level fragility gating) | ai-opt | HPCA / MICRO | 5.5 / 7.0 / 8.1 | **CONCURRENT 55% Four Over Six** + NVFP4 QAD scoop. fragility classifier 의 generalization 이 unproven (MMMU 학습 → MathVista transfer 의 fragile 분류 정확도). | NVFP4 QAD 와 차별화 mechanism 추가 + 3-platform fragility classifier transferability 검증 시 재고려 |
| 2 | **STELE** (thermal-budget-aware speculative decoding) | ai-opt | HPCA / MLSys | 6.0 / 5.0 / 7.7 | **CONCURRENT 60% SLED + Ghidorah** scoop. ThermalLoom 와 thermal axis 중복. tegrastats 200ms cadence 가 throttle 진입 (1-2s) 대비 충분한가 미검증. | β=acceptance × thermal_headroom unique metric 정량 + 100ms cadence sampling overhead ≤ 1% 검증 시 Tier-2 격하 가능 |
| 3 | **CINDER** (sustained-throttle DVFS lookahead) | ai-opt | DAC / IEEE CAL | 5.0 / 4.5 / 6.6 | Linux schedutil + nvpmodel adjacent. STELE 와 axis 부분 중복. mechanism 신규성 약함. | VLM-aware lookahead prediction 가 Linux ondemand 대비 +20%+ 개선 시 Tier-2 가능 |
| 4 | **RIVET** (Orin DLA-only INT8 image embedding LRU cache) | ai-opt | DATE / ISLPED | 4.0 / 6.0 / 6.8 | SGLang RadixAttention 의 token-prefix tree adjacent. cross-request hash collision rate 의 robustness 미검증. | Production dashcam-style cross-request 분포에서 hash hit rate ≥ 50% 검증 시 재고려 |
| 5 | **ThermalLoom** (Thermal-throttle BW degradation + NVPModel hide) | legacy-sys | HPCA / MICRO | 7.5 / 7.0 / 7.6 | **CONCURRENT 50% CLONE + TAPAS** scoop. ShelfSwap + STELE 와 thermal axis 3-way 중복. CLONE [USENIX ATC 2025] 의 layer-aware DVFS 와 차별화 부족. | CLONE 과 mechanism 1:1 diff 강화 + sustained workload BW kernel re-tile 가 unique axis 임을 정량화 시 Tier-1 재진입 |
| 6 | **PageWeave** (cudaMallocManaged page-fault soft-IRQ hide via lifetime classifier) | legacy-sys | ASPLOS / OSDI | 8.0 / 7.5 / 8.3 | VESPER 와 UMA axis 중복. Visual-token lifetime classifier 의 OOD transferability 미검증. **Phase 2' 권고는 VESPER 와 통합** (KV pruner + page-fault hide + lifetime classifier 하나의 OSDI paper) — 본 bundle 은 분리 유지하되 미선정. | VESPER 와 통합 paper framing 시 OSDI 2027 super-paper 가능 (mechanism 6+ → 4 로 압축 필요) |
| 7 | ~~**DualLane**~~ (Dual-NVDLA + GPU 3-way + SLC partition) | legacy-sys | ISCA / MICRO | 7.0 / 8.0 / 8.6 | **R45 적용 후 Tier-1 재진입 (4/10 LOW)** — CacheVeil R45 demotion 자리 비면서 진입. NvMedia DLA + DRM dma-buf + libsmctrl 모두 공식 API. 4-mechanism 분리 ablation 으로 Nova [arXiv:2509.21301] CONCURRENT 차별화. → [tier1/04-duallane](/research-wiki/2026-04/vlm-context-edge-jetson/tier1/04-duallane.md) | — (Tier-1 진입) |
| 8 | **CacheVeil 원안** (R45 demotion) | legacy-sys | (1차 publish: Tier-1 #1 ★ lead → R45 적용 후 demotion) | 8.5 / 7.0 / 8.5 = **8.00** | **R45.1 위반** — ARM CMN `por_hnf_pwpr` partition register 가 vendor 공식 user-space API 부재 (undocumented BPMP IOCTL `tegra_bpmp_transfer(MRQ_CMN_SLC_PARTITION)` 의존). Simulator-path Tier-2 spinoff 으로 변환 → [tier2/03-cacheveil-sim](/research-wiki/2026-04/vlm-context-edge-jetson/tier2/03-cacheveil-sim.md) | Thor JetPack 7.x 에서 BPMP cache partition IOCTL 공식화 시 또는 NVIDIA partner channel 통한 BPMP firmware extension 공개 시 Tier-1 재진입 가능 |
| 9 | **FrostHint** (ISP zero-copy + L2 pre-warming hint) | legacy-sys | DATE / ISLPED | 3.5 / 5.5 / 7.0 | **DROP 또는 Repositioning** — ThermalLoom 또는 CacheVeil 의 baseline experiment 에 흡수 권고. cudaStreamAttribute persisting region 자체는 known API. | CacheVeil baseline 표 의 row 로 흡수 (M1 visual KV pinning 의 보조 효과 측정) |
| 10 | **TileGate** (L2-capacity-aware visual token tile size) | legacy-sys | IEEE CAL | 3.5 / 5.0 / 6.6 | **DROP** — CacheVeil 의 baseline table 에 흡수 권고. tile size sweep 자체는 mechanism 약함. | CacheVeil baseline 의 ablation row 로 흡수 |
| 11 | **Watershed** (Heterogeneous Phase Dispatcher) | hwpim | MICRO / HPCA | 6.5 / 7.0 / 8.0 | **CONCURRENT 58% DuetServe + Nova** scoop. SHOAL + DualLane 과 DLA+GPU axis 공유 — phase classifier 가 추가 mechanism 이지만 differentiation 약함. | DuetServe (datacenter) + Nova 와 1:1 mechanism diff 강화 + phase classifier 가 unique control plane 임을 정량화 시 Tier-1 재진입 |
| 12 | **Thermal Gradient Quantizer (TGQ)** (NVFP4 thermal recalibration on Thor) | hwpim | MICRO | 5.0 / 6.5 / 7.4 | **CONCURRENT 55% Four Over Six + NVFP4 QAD** scoop. KILN 과 NVFP4 axis 중복. Thor only — scope 좁음. | NVFP4 QAD 의 calibration cost 와 차별화 mechanism (thermal sensor 결합) 정량화 시 Tier-2 가능 |
| 13 | **Tessellated Bank Affinity** (LPDDR5X bank-modality placement) | hwpim | ICCAD / DAC | 7.0 / 7.5 / 7.2 | NVIDIA bank mapping API 부재 (driver-level 권한 필요). 효과 정량화 어려움. score 7.0 borderline — Phase 1' 권고는 Mirror Lake + Tessellated 통합. | NVIDIA bank mapping API 노출 또는 sysfs 우회 검증 시 Tier-1 재진입 가능 |
| 14 | **Mirror Lake** (UMA modality isolation IOMMU) | hwpim | ISCA / HPCA | 8.0 / 6.5 / 7.5 | **CacheVeil 과 cache axis overlap** — CacheVeil = SLC way-partition (cache layer), Mirror Lake = IOMMU page attribute (kernel layer). Phase 2' 권고는 nested 결합 (CacheVeil 의 backing 으로 Mirror Lake) 또는 sister paper 분리. 본 bundle 은 CacheVeil 단독 진입 + Mirror Lake 미선정. | CacheVeil 과 stacked benefit 측정 — superposition 시 +10%+ 추가 효과 검증 시 sister paper 가능 |
| 15 | **Pinwheel** (Orin Nano DLA-only short VQA) | hwpim | DAC | 4.0 / 5.5 / 7.0 | **TUFA 와 Orin Nano scope 충돌** — TUFA 는 vision tower early-exit (layer 축), Pinwheel 은 DLA-tile-pinwheel scheduler (compute 축). TUFA 가 더 distinctive. | DLA-only short VQA 의 first-to-report 정량화 시 Tier-2 가능 (TUFA 와 별도 mechanism) |
| 16 | **Diode Ladder** (NVFP4 Bias Compensation circuit) | hwpim | IEEE TCAS-I / ESSDERC | 3.0 / 4.0 / 6.2 | 회로 시뮬 필요 + Thor only. mechanism 회로-수준 single layer — VLM serving 본질에서 거리. | 회로 시뮬 결과 + Thor NVFP4 drift 0.6pp+ 방어 검증 시 Tier-2 가능 |
| 17 | **Glacier Migrate** (R45 demotion) | hwpim | (1차 publish: Tier-2 #3 → R45 적용 후 unselected) | 6.0 / 7.0 / 7.0 = **6.67** | **R45.1 위반** — DLA SRAM physical addr exposure 의무 (PTX `ld.global.nc` + IOMMU mapping) 가 undocumented kernel-level path. **R45.3 위반** — simulator path 의 gem5 PIM-extension 12-20주 소요로 단일 학기 fit 불가. | JetPack 의 DLA SRAM 공식 API 노출 시 또는 AttAcc/LLMServingSim (ASPLOS'24+, R45.9 active) 의 PIM extension 공개 시 (NVDLA-sim 은 R45.9 deprecated 로 금지) Tier-2 재진입 가능 |

> **참고**: 위 표의 "#" 컬럼은 본 unselected.md 의 행 번호가 아닌 21 idea 전체 enumeration 에서의 원안 번호. **R45 적용 (2026-04-25 추가) 후 변동**: (a) #7 DualLane 은 Tier-1 진입 (R45 risk 4/10), (b) #8 CacheVeil 원안 은 R45.1 위반으로 Tier-2 simulator-path spinoff (CacheVeil-Sim) 으로 demotion, (c) #17 Glacier Migrate 는 R45.1 위반 + R45.3 비현실로 unselected 이동.

---

## 통합 권고 6 pair (Phase 2' Cross-expert duplication)

미선정 idea 들의 통합 가능성 — 향후 super-paper / sister paper 로 재방문 시 활용:

| 통합 대상 | mechanism 결합 | 권장 framing |
|----------|---------------|-------------|
| **STELE ↔ ThermalLoom ↔ ShelfSwap** | thermal trigger + DVFS / kernel re-tile / zone migration | thermal axis 3-way 통합 또는 명확 분리 (sustained BW / burst specdec / zone migration) |
| **CacheVeil ↔ Mirror Lake** | SLC partition / UMA cache coherence partition | nested 결합 가능 (cache layer + IOMMU layer) — 같은 paper 통합 또는 sister paper |
| **VESPER ↔ PageWeave** | UMA + KV management | 통합하여 OSDI 2027 unified UMA-VLM paper (KV pruner + page-fault hide + lifetime classifier) — mechanism 6 → 4 압축 필요 |
| **SHOAL ↔ DualLane ↔ Watershed** | DLA + GPU axis | SHOAL = data plane KV / DualLane = dataflow / Watershed = control plane phase classifier — 단일 framework 의 3 angle |
| **KILN ↔ TGQ** | NVFP4 fragility / thermal | NVFP4 single paper 에 sample-level + thermal recalibration 통합 |
| **Tessellated ↔ Mirror Lake** | LPDDR5X bank / UMA modality | bank-level + IOMMU-level placement nested |

---

## Phase 2' / Phase 1'' R45 최종 선정 표 요약 (2026-04-25 R45 적용 후 갱신)

| 채택 | Phase 2 평균 | R45 risk | 선정 사유 |
|------|-------------|----------|----------|
| ★ VESPER (Tier-1 lead) | 8.10 | 3/10 LOW | 최고 impact 8.8, UMA dual-view + NEON SIMD CPU pruner. cudaMallocManaged + NEON 모두 공식 user-space API → R45 OK. CacheVeil R45 demotion 후 ★ lead 승격 |
| SHOAL (Tier-1) | 8.07 | 4/10 LOW | 최고 diff 8.5, DLA → KV residence enum layer-단위 dynamic. NvMediaTensor 공식 API + 선택적 AttAcc/LLMServingSim simulator path (R45.9 active; NVDLA-sim deprecated 금지) |
| DualLane (Tier-1) | 8.00 (R45 보정) | 4/10 LOW | NvMedia DLA + DRM dma-buf + libsmctrl 모두 공식 API. 4-mechanism 분리 ablation 으로 Nova CONCURRENT 차별화. CacheVeil 자리 비면서 신규 진입 |
| TUFA (Tier-2) | 6.47 | 3/10 LOW | Orin Nano-specific 7W constraint first-to-report. trivialty 회피 명시. 모두 user-space API |
| ShelfSwap (Tier-2) | 6.07 | 4/10 LOW | thermal-driven UMA zone migration first-to-report. cuMemAdvise 공식 API |
| CacheVeil-Sim (Tier-2) | 6.5 (CacheVeil 원안 8.00 → R45 -1.5) | 7/10 MED-HIGH | gem5 + ChampSim simulator-path. CacheVeil 원안 R45.1 위반의 Tier-2 spinoff. R45.2 exception + R45.3 30-run feasibility 확보 |

### R45 demotion 2 idea (선정 → demotion)

| Demoted | 1차 publish 위치 | 변경 후 위치 | R45 사유 |
|---------|----------------|------------|---------|
| **CacheVeil 원안** | Tier-1 ★ lead (8.00) | Tier-2 spinoff (CacheVeil-Sim) | R45.1 위반 (ARM CMN `por_hnf_pwpr` undocumented BPMP IOCTL 의존) |
| **Glacier Migrate** | Tier-2 #3 (6.67) | unselected | R45.1 위반 (DLA SRAM physical addr exposure) + R45.3 위반 (gem5 PIM-extension 12-20주) |

---

## 재방문 timing

본 세션의 미선정 idea 중 우선 재방문 후보 (다음 mode1 세션 또는 Tier-1 paper 의 sister paper 로):

1. **VESPER + PageWeave 통합** → OSDI 2027 unified UMA-VLM super-paper (Tier-1 격상)
2. **CacheVeil + Mirror Lake 통합** → ISCA 2027 sister paper (cache layer + IOMMU layer nested)
3. **DualLane refresh** → Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) 와 mechanism 1:1 diff 강화 후 MICRO 2027 재진입 가능
4. **ThermalLoom refresh** → CLONE [USENIX ATC 2025] 와 차별화 강화 후 HPCA 2027 재진입 가능
5. **Tessellated Bank Affinity refresh** → NVIDIA bank mapping API 노출 시 ICCAD 2027 재진입

---

## 출처 staging 파일 (idea 별 도출 expert 추적)

| Idea | 출처 staging 파일 | 본 세션 expert |
|------|-------------------|---------------|
| KILN, STELE, CINDER, RIVET | `2026-04-25-vlm-context-edge-jetson-aiopt-expert.md` | ai-opt (T3, T4, S2, S3) |
| ThermalLoom, PageWeave, DualLane, FrostHint, TileGate | `2026-04-25-vlm-context-edge-jetson-legacy-sys-expert.md` | legacy-sys (T1, T2, T3, B1, B2) |
| Watershed, TGQ, Tessellated, Mirror Lake, Pinwheel, Diode Ladder | `2026-04-25-vlm-context-edge-jetson-hwpim-expert.md` | hwpim (T1, T2, T3, T4, S1, S2) |

검토 reviewer staging 파일:
- novelty: `2026-04-25-vlm-context-edge-jetson-phase2-novelty.md`
- diff: `2026-04-25-vlm-context-edge-jetson-phase2-diff.md`
- impact: `2026-04-25-vlm-context-edge-jetson-phase2-impact.md`
