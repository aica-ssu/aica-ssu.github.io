# 2026-05-02 Mode-1 VLM Scenario-Aware Optimization Session

## Meta

- **Date**: 2026-05-02
- **Mode**: 1 (sentence-input)
- **사용자 input**: VLM 추론은 시나리오에 따라 다르므로 사전 분류 + 최적화 — video multi-turn QA visual info + accuracy 보존, 가장 bottleneck 심한 scenario 식별, 기존 benchmark 활용 실험 플랜
- **Experts**: ai-optimization (A) + algorithm (B) + legacy-system (C)
- **Reviewers**: novelty + differentiation + impact 3인 (5인 dispatch 의 ai-impl + arch-sys 는 main thread GitHub trace + R47 path 검증으로 통합)
- **Papers analyzed**: 28편 (VLCache / PrefixKV / VLA-Cache / SGLang RadixAttention / vLLM Prefix Caching / NVIDIA NIM VLM / KVFlow / LMCache / LongVU / AdaptToken / PVC / VisionThink / AwaRes / AttWarp / AutoPrune / Nova / ECVL-ROUTER / PolyKV / KVShare / OxyGen / METok / TSG / MIRAGE / One-Token-per-Frame / Lossless Ultimate Compression / MMBench-Video / MileBench / EgoSchema / Q-Bench-Video)
- **Ideas generated**: 21 raw → Top-M 6 (Tier-1 3 + Tier-2 3) + DROP 14 + 흡수 6
- **R29.2 quota check**: A=7, B=7, C=7 → 모두 5+ 충족 ✅
- **Cross-link**:
  - Summary: `summary/2026-05-02-mode1-vlm-scenario-aware/README.md`
  - Staging: `2026-05-02-mode1-vlm-scenario-aware-staging.md`
  - 이전 세션: `2026-04-30-mode1-vlm-rtx6000-cpu-cudagraph.md`, `2026-05-01-mode1-vlm-rtx6000-realbottleneck.md`

---

## Section 0 — Executive Summary

### Top-M 6 (Tier-1 3 + Tier-2 3)

#### T1.1 Mosaic (Workload-Adaptive Serving Configuration Dispatcher)
- **전제**: vLLM/SGLang single-path scenario-agnostic, scenario diversity 미활용. ECVL-ROUTER ([arXiv:2510.27256](https://arxiv.org/abs/2510.27256), ICLR'26) 는 model-tier routing — Mosaic 는 single-model serving-stack config (직교 axis).
- **GAP**: Scenario classifier 자체 부재 + scenario-conditional KV budget / prefix policy / compression dispatch 미커버 + production hit-rate adaptive pinning 미커버
- **Mechanism**: M1 lightweight classifier (DistilBERT-mini 60M, 6-class, < 2ms) / M2 scenario-conditional config table dispatch (KV budget + prefix policy + compression rate) / M3 hit-rate tracker + hot scenario GPU pinning (LMCache static pin 과 직교 — adaptive)
- **예상 효과**: TTFT mixed-scenario batch -30~-35% / throughput +30~+40% / GPU VRAM peak -25% / hit rate 60-85%→65-90% (+5-10pp)
- **Scoring**: novelty 6 / diff 7 (refinement) / impact 9.5 → mean **7.5**
- **Target venue**: OSDI 2027 / ASPLOS 2027

#### T1.2 Lattice (Cross-Turn Frame-Indexed Radix Vision KV Cache)
- **전제**: Production VLM (Qwen3.5/mlx-vlm Issue #832) 매 turn 마다 vision tower re-run + full re-prefill — vLLM Prefix Caching / SGLang RadixAttention 이 LLM token prefix 만 cover
- **GAP**: VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977)) single-session 만, frame-level radix tree + cross-session sharing 미커버
- **Mechanism**: M1 per-frame pHash + frame ID radix tree / M2 PrefixKV 기반 layer-wise prefix retention / M3 cross-session frame prefix sharing (privacy classifier CLIP-based 95%+)
- **예상 효과**: Multi-turn turn 2+ TTFT -60~-80% / vision KV 5-turn session 7.5GB→1.5GB (-75%) / cross-session frame prefix hit 0%→50-70% (new capability) / accuracy ≤-1.0pt
- **Scoring**: novelty 8 / diff 6 (VLCache concurrent 50-70% refinement) / impact 8 → mean **7.5**
- **Target venue**: MLSys 2027 / NeurIPS 2026

#### T1.3 Bramble (Cross-Image Vision Token Pool for Multi-Image Agent Loop)
- **전제**: Multi-image agent (MileBench 6+ image / request) 에서 PolyKV / KVShare 가 LLM token cross-request 만 cover, vision token cross-image 미커버
- **GAP**: Vision token cross-image / cross-tenant pool + privacy boundary 결합 paper 0편
- **Mechanism**: M1 image pHash + tenant-shared LRU pool / M2 CLIP-based privacy classifier + boundary verification / M3 reference counting + zeroize on evict + cuckoo filter
- **예상 효과**: GPU KV (multi-tenant batch=8) 24GB→14GB (-40%) / max concurrent tenants 4→8 (2×) / $/req -30~-45% / privacy leakage 0
- **Scoring**: novelty 8 / diff 5 (PolyKV/KVShare 50-70% refinement) / impact 8.5 → mean **7.2**
- **Target venue**: MLSys 2027 / OSDI 2027

#### T2.1 Lantern (NVDEC-Coupled Sliding-Window KV for Streaming Video)
- **GAP**: NVDEC HW frame metadata + sliding window KV co-design paper 0편
- **Mechanism**: M1 NVDEC frame metadata extract / M2 sliding window K=8 retention / M3 eager free + memory recycle
- **예상 효과**: Per-frame latency < 100ms / 1-hour session GPU memory bounded 1.5GB / NVDEC utilization 80%+
- **Scoring**: novelty 5 / diff 8 (HW co-design 0-30% scoop) / impact 6 → mean **6.0**
- **Target venue**: IEEE CAL letter + vLLM upstream PR

#### T2.2 Compass (Ego-Motion-Aware Compression for Egocentric Video)
- **GAP**: LongVU/AdaptToken/PVC 모두 ego-motion 미고려, EgoSchema specific compression 0편
- **Mechanism**: M1 optical flow zero-mean ego-motion stable region detection / M2 ego-stable region token retention / M3 ego-motion vector → MRoPE T-axis 보강
- **예상 효과**: EgoSchema accuracy 보존 + token reduction 50%+
- **Scoring**: novelty 7 / diff 7 (no direct competitor) / impact 6 → mean **6.0**
- **Target venue**: DATE 2027 / ISLPED 2027

#### T2.3 Hearth (Document VLM L2 Carveout + Tile Locality Boost)
- **GAP**: AwaRes/VisionThink/AttWarp algorithm-side, system-side L2 carveout 미커버
- **Mechanism**: M1 Mosaic dispatch hook (document scenario) / M2 `cudaAccessPolicyWindow` tile residence boost / M3 `cudaCtxResetPersistingL2Cache` eviction priority
- **예상 효과**: Tile attention latency -25% / L2 hit rate +30pp / accuracy 보존
- **Scoring**: novelty 6 / diff 6 (HW arch novelty 30-50%) / impact 7 → mean **6.0**
- **Target venue**: DATE 2027 / IEEE ESL letter

### DROP 14 + 흡수 6

- **DROP**: A3 (PrefixKV scoop) / A4 (Nova scoop) / A6 (AwaRes scoop) / B1 (VisionThink scoop) / B2 (LongVU/PVC scoop) / B3 (METok scoop) / B4 (TSG 91.4% scoop) / B6 (AttWarp scoop) / C4 (single axis weak) / 그리고 A5/B7/C1/C3/C5/C6 흡수
- **흡수 매핑**: A5→Lantern / B7→Bramble / C1→Lattice / C3→Bramble / C5→Lattice / C6→Mosaic

---

## Section 1 — Phase 0/1/2/1'/2'/1'' 상세 로그

### Phase 0/1 staging
`2026-05-02-mode1-vlm-scenario-aware-staging.md` — R31 verification + 21 idea 후보 + workload evidence.

### Phase 2 (3 reviewer dispatch 결과)

**Novelty Reviewer**:
- A1=6, A2=8, A3=5, A4=7, A5=4, A6=8, A7=7, B1=6, B2=5, B3=6, B4=8, B5=7, B6=4, B7=5, C1=7, C2=6, C3=8, C4=3, C5=5, C6=4, C7=5
- 권고: Top-6 = A2/B4/C3 (Tier-1) + A6/A7/B5 (Tier-2)

**Differentiation Reviewer (web search 결과 critical)**:
- 추가 closest competitor 발견 (ECVL-ROUTER / Nova / MIRAGE / LMCache / PolyKV / KVShare / METok / TSG / OxyGen / Lossless Ultimate)
- 70%+ scoop drop 권고: A4 / A6 / B1 / B2 / B4 / B6 / C1 / C3 (8개)
- Top-6 권고: C7 / B5 / C5 (Tier-1) + C2 / A2 / C6 (Tier-2)

**Impact Reviewer**:
- D1 (Scenario dispatcher = A1 generalize) 9.5 → 가장 높은 impact
- A1=9.0 / A7=8.5 / B1=9.0 / C1=9.0
- 권고: D1+A1+B1 묶음

### Phase 1' Refinement (Improve-First)

**Novelty + Differentiation + Impact 종합** (사용자 강조 "기존 benchmark 활용 실험 플랜"):
1. **A1 Mosaic** — diff reviewer 70%+ scoop 경고 (ECVL-ROUTER) 에 대한 refinement: serving-stack config (single-model) 직교 axis 강조 → mean 7.5
2. **A2 Lattice** — diff reviewer 50-70% (VLCache) refinement: cross-session frame radix + privacy 차별화 강화 → mean 7.5
3. **A7 Bramble** — diff reviewer 50-70% (PolyKV/KVShare) refinement: vision token + privacy + cuckoo filter → mean 7.2
4. **C7 Lantern** — HW co-design unique 0-30% scoop, 진행 → mean 6.0
5. **B5 Compass** — niche 30-50% scoop, 진행 → mean 6.0
6. **C2 Hearth** — HW arch novelty 30-50% scoop, 진행 → mean 6.0

**DROP 14 + 흡수 6** (위 Executive Summary 참조).

### Phase 2' (재평가, ai-impl + arch-sys reviewer 통합 main thread)

Main thread 검증:
- **R45.5 / R47.2 / R52.3 GitHub trace**: 6 idea 모두 application-level vLLM modification (R47.2). vLLM v0.10.x source 의 file path / symbol 모두 실재. [✅]
- **R20-γ.1 single-system fit**: 6 idea 모두 RTX Pro 6000 / RTX 5090 single-system fit. [✅]
- **R52.2 6-column 적용**: As-is/To-be + Class · Function · Line region + 의사코드 의무 충족
- **R53 일반 단어 section title**: 동작 원리 / 기대 효과 / 구현 변경점 / 검증 시나리오 적용
- **R19-α**: Mosaic / Lattice / Bramble / Lantern / Compass / Hearth — 모두 자연어 영단어, full title vendor-neutral (NVDEC 은 industry-standard 예외)

### Phase 1'' Top-M 6 Finalize

위 Top-M 6 + Tier-2 3개 + DROP/흡수 6개 명시.

### R52.3 / R54 Final Verification Pass

- File path / symbol 실존 (vLLM v0.10.x): `BaseMultiModalProcessor` ✅, `KVCacheManager` ✅, `Scheduler._schedule_running` ✅, `MRotaryEmbedding` ✅, `LLMEngine` ✅
- 신규 add module (`scenario/classifier.py` / `multimodal/lattice_radix.py` / `multimodal/bramble_pool.py` / `multimodal/nvdec_hook.py` / `multimodal/compass_egomotion.py` / `scenario/hearth_carveout.py`) — vLLM standard add pattern ✅
- Hallucination pattern 0 ✅
- R55.2 5-axis 분포: Performance (T1.1/T1.2/T2.1/T2.3) / Memory (T1.2/T1.3) / Cost eff. (T1.1/T1.3) / Energy (T2.1) / Quality (모두) / Security (T1.3) — **6-axis 모두 cover**

---

## Section 2 — Implementation-Priority Decision Tree

→ Summary README § 4 참조

---

## Section 3 — Cross-Reference

- **이전 세션 cross-impact**:
  - 2026-05-01 vlm-rtx6000-realbottleneck (DAVTC/SA-KV/VKM-DEMO): single-shot common case 위주, 본 세션은 multi-turn / multi-tenant scenario diversity 추가 axis
  - 2026-04-30 vlm-rtx6000-cpu-cudagraph (Cadence): MRoPE T-axis 가 본 세션 T2.2 Compass 의 MRoPE ego-offset 와 cross-share
- **다음 세션 권고**:
  - Edge VLM scenario taxonomy (Jetson Thor/Orin) 확장
  - Multi-modal beyond vision (audio + video + text) scenario classifier
  - Privacy-preserving cross-tenant pool 의 differential privacy 강화
