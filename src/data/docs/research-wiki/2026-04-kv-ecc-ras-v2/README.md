# KV cache ECC + Memory RAS v2 (Session #1, 2026-04-26)

> **Date**: 2026-04-26 · **Mode**: 1 · **Lab**: SSU AICA · **Author**: system-robustness-expert
>
> **Bundle 형식**: R28-R50 hierarchical bundle. v1 (2026-04-25) 의 3 사고 (R48 누적 갱신 누락 / R49 deprecated simulator 잔존 / R50 modern memory standard 미반영) 회피.
>
> **Phase staging**: 01-step0 · 02-phase1 · 03-phase2 · 04-phase1' · 05-phase2' · 06-phase1''

---

## 1. 세션 Meta

- **사용자 쿼리 핵심**: v1 (2026-04-25) 의 6 idea publish 후 발견된 **3 사고 (R48 누적 갱신 누락 / R49 ChaosMem 잔존 / R50 LPDDR5x/HBM3/CXL 3.x 신규 RAS feature 미반영) 의 회피**. KV cache ECC + 메모리 RAS axis 의 **modern memory standard-aware + outlier-aware + multi-tenant CXL-aware** ideation 재실행.
- **키워드 (4-8)**: `KV Cache` `ECC` `RAS` `LPDDR5x` `HBM3` `CXL 3.x` `Outlier-Aware` `Multi-Tenant Isolation`
- **중점 고려 축**: (a) modern memory standard 신규 feature 의 핵심 mechanism integration (R50.2), (b) workload-driven evidence (IISWC/ISPASS/MLPerf 9 source, R23), (c) v1 6 idea 와 axis 직교 강제, (d) R49 cross-check (ChaosMem 0건 잔존 검증), (e) R47 path strict (gem5+vLLM 동시사용 0).
- **의도 제외 축**: real DRAM rowhammer rig / FPGA / kernel patch / TensorRT-LLM cross-system / multi-node distributed serving (단 LLMServingSim cluster sim 만 허용).

---

## 2. R47 적용 결과 — 6 final idea path 분류

| # | Idea | Tier | R47 primary path | R47 secondary | R47.1 위반 |
|---|------|------|---------------------|---------------|-----------:|
| 1 | **P1 PrefixGuard** | T1 | R47.2 vLLM `LMCacheConnector` + Linux 6.16 EDAC sysfs polling | R47.3 LLMServingSim cluster | ❌ |
| 2 | **P3 Quarantine** | T1 | R47.2 vLLM `kv_transfer/` agent_id × DPA table | R47.3 LLMServingSim multi-agent | ❌ |
| 3 | **P4 PATroller** | T1 | R47.2 vLLM `block_manager_v2.py` PAT polling thread | R47.3 NeuroSim V1.4 + LLMServingSim | ❌ |
| 4 | **P8 ECS-Trace** | T2 | R47.2 vLLM `block_manager_v2.py` ECS query thread | R47.3 LLMServingSim + NeuroSim V1.4 | ❌ |
| 5 | **V3 Quarantine-Mini** | T2 | R47.2 vLLM RFC #19329 token-range recompute | (none) | ❌ |
| 6 | **V1 PrefixGuard-Lite** | T2 | R47.2 vLLM `LMCacheConnector` + EDAC sysfs write | (none) | ❌ |

**R47.1 위반 0건 / 6, ChaosMem 잔존 0건 / 6, R47.4 trace+gem5 primary 0건 / 6** — clean.

---

## 3. R50.2 Modern Memory Standard 활용 매트릭스

| Idea | LPDDR5x JESD209-5C | HBM3 JESD238B | CXL 3.x | Mechanism integration depth |
|------|---------------------|----------------|----------|------------------------------|
| P1 PrefixGuard | — | (보조 ECS) | **Patrol Scrub Control + ECS mailbox + DPA poison** | 핵심 (3 feature) |
| P3 Quarantine | — | — | **DPA tracking + poison + ECS mailbox + Memory Event Record** | 핵심 (4 feature) |
| P4 PATroller | (보조 ARM) | **PAT + ECS + RCC** | — | 핵심 (3 feature) |
| P8 ECS-Trace | — | **ECS + IEEE 1500 TAP + RCC** | — | 핵심 (3 feature) |
| V3 Quarantine-Mini | — | — | **ECS mailbox + DPA poison + Memory Event Record** | 핵심 (3 feature) |
| V1 PrefixGuard-Lite | — | — | **Patrol Scrub Control + Linux 6.16 EDAC** | 핵심 (2 feature) |

**6/6 idea 가 modern memory standard 신규 RAS feature 핵심 mechanism integration** — R50.2 충족 (단순 motivation 언급 0건).

---

## 4. R44 SVG Post-Ideation Decision Tree

> 학생/AI 가 본 세션 6 idea 중 어떤 것부터 착수할지, 그리고 baseline 재현 → Mech 단독 → 통합 → Tier 분기 결정을 자율 수행하는 흐름.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 1380" width="920" height="1380" font-family="Inter, system-ui, sans-serif" font-size="13">
<rect x="0" y="0" width="920" height="1380" fill="#fafafa"/>
<rect x="240" y="14" width="440" height="74" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>
<text x="460" y="38" text-anchor="middle" font-weight="bold">Step 1 — HW + idea 매칭 (R47 path)</text>
<text x="460" y="58" text-anchor="middle">CXL-attached DRAM access (P1/P3/V3/V1) · HBM3 GPU only (P4/P8) · single H100 (V3/V1) · cluster 8-32 GPU (P1/P3/P4)</text>
<text x="460" y="78" text-anchor="middle">All R47.2 vLLM source mod primary; R47.4 gem5 trace = 0/6</text>
<rect x="40" y="120" width="270" height="60" rx="6" fill="#fef3c7" stroke="#d97706"/>
<text x="175" y="142" text-anchor="middle" font-weight="bold">CXL pool (P1/P3/V1/V3)</text>
<text x="175" y="162" text-anchor="middle">vLLM 0.7+ + LMCache + sysfs EDAC</text>
<rect x="325" y="120" width="270" height="60" rx="6" fill="#fef3c7" stroke="#d97706"/>
<text x="460" y="142" text-anchor="middle" font-weight="bold">HBM3 GPU (P4/P8)</text>
<text x="460" y="162" text-anchor="middle">vLLM + NeuroSim V1.4 + LLMServingSim</text>
<rect x="610" y="120" width="270" height="60" rx="6" fill="#fef3c7" stroke="#d97706"/>
<text x="745" y="142" text-anchor="middle" font-weight="bold">Single H100 (V3/V1)</text>
<text x="745" y="162" text-anchor="middle">vLLM only, single workload (8주)</text>
<line x1="460" y1="88" x2="175" y2="120" stroke="#94a3b8" stroke-width="1"/>
<line x1="460" y1="88" x2="460" y2="120" stroke="#94a3b8" stroke-width="1"/>
<line x1="460" y1="88" x2="745" y2="120" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="220" width="360" height="60" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="460" y="244" text-anchor="middle" font-weight="bold">Step 2 — Baseline 재현 (Wk 2-3)</text>
<text x="460" y="264" text-anchor="middle">vanilla vLLM + Linux EDAC scrub default + ±5% match</text>
<line x1="175" y1="180" x2="460" y2="220" stroke="#94a3b8" stroke-width="1"/>
<line x1="460" y1="180" x2="460" y2="220" stroke="#94a3b8" stroke-width="1"/>
<line x1="745" y1="180" x2="460" y2="220" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="310" width="360" height="60" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="460" y="334" text-anchor="middle" font-weight="bold">Step 3 — Mech #1 단독 (Wk 4-7)</text>
<text x="460" y="354" text-anchor="middle">solo eval: PAT polling / scrub interval / ECS query thread</text>
<line x1="460" y1="280" x2="460" y2="310" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="400" width="360" height="60" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="460" y="424" text-anchor="middle" font-weight="bold">Step 4 — Mech #2/#3 통합 (Wk 8-12)</text>
<text x="460" y="444" text-anchor="middle">5 workload × 3 config × 2 baseline = 30 runs</text>
<line x1="460" y1="370" x2="460" y2="400" stroke="#94a3b8" stroke-width="1"/>
<rect x="240" y="490" width="440" height="92" rx="8" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>
<text x="460" y="518" text-anchor="middle" font-weight="bold">Decision — Tier-1 cut (Wk 12)</text>
<text x="460" y="538" text-anchor="middle">silent corruption rate ≥ 90% reduction</text>
<text x="460" y="556" text-anchor="middle">OR cross-agent cross-isolation 100%</text>
<text x="460" y="574" text-anchor="middle">OR fault rate ≥ 95% reduction (HBM3 PAT)</text>
<line x1="460" y1="460" x2="460" y2="490" stroke="#94a3b8" stroke-width="1"/>
<rect x="60" y="620" width="340" height="84" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>
<text x="230" y="648" text-anchor="middle" font-weight="bold">Tier-1 (조건 충족)</text>
<text x="230" y="670" text-anchor="middle">P1 OSDI 2027 (13p) / P3 USENIX Security 2027 (13p)</text>
<text x="230" y="688" text-anchor="middle">P4 HPCA 2027 (12p)</text>
<rect x="520" y="620" width="340" height="84" rx="8" fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>
<text x="690" y="648" text-anchor="middle" font-weight="bold">Tier-2 (cut 미달)</text>
<text x="690" y="670" text-anchor="middle">P8 ITC 2027 6p / V3 DAC 2027 6p</text>
<text x="690" y="688" text-anchor="middle">V1 DATE 2027 6p / IEEE TCAD short</text>
<line x1="340" y1="582" x2="230" y2="620" stroke="#94a3b8" stroke-width="1"/>
<line x1="580" y1="582" x2="690" y2="620" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="740" width="360" height="62" rx="8" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="460" y="764" text-anchor="middle" font-weight="bold">Step 5 — R50.2 modern memory feature 인용</text>
<text x="460" y="784" text-anchor="middle">LPDDR5x JESD209-5C / HBM3 JESD238B / CXL 3.x cite</text>
<line x1="230" y1="704" x2="460" y2="740" stroke="#94a3b8" stroke-width="1"/>
<line x1="690" y1="704" x2="460" y2="740" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="830" width="360" height="60" rx="8" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="460" y="854" text-anchor="middle" font-weight="bold">Step 6 — 7 element experiment plan (R27-β)</text>
<text x="460" y="874" text-anchor="middle">HW / Model / Dataset / Sim / Ablation / Steps / Metric</text>
<line x1="460" y1="802" x2="460" y2="830" stroke="#94a3b8" stroke-width="1"/>
<rect x="280" y="918" width="360" height="60" rx="8" fill="#fce7f3" stroke="#db2777" stroke-width="2"/>
<text x="460" y="942" text-anchor="middle" font-weight="bold">Step 7 — Tier-2 paper-pair 분기 (R21)</text>
<text x="460" y="962" text-anchor="middle">P3 ↔ V3 paper pair 1 쌍 (Option C). V4 보조 wiki only</text>
<line x1="460" y1="890" x2="460" y2="918" stroke="#94a3b8" stroke-width="1"/>
<rect x="40" y="1010" width="270" height="100" rx="8" fill="#fed7aa" stroke="#ea580c"/>
<text x="175" y="1034" text-anchor="middle" font-weight="bold">Workload (5)</text>
<text x="175" y="1056" text-anchor="middle">Llama-3.1-8B (FP16/INT4)</text>
<text x="175" y="1074" text-anchor="middle">Llama-3.1-70B-Instruct</text>
<text x="175" y="1092" text-anchor="middle">Qwen3-30B-MoE / Qwen3-VL</text>
<rect x="325" y="1010" width="270" height="100" rx="8" fill="#fed7aa" stroke="#ea580c"/>
<text x="460" y="1034" text-anchor="middle" font-weight="bold">Config (3)</text>
<text x="460" y="1056" text-anchor="middle">No-protection / SEC-DED / Proposed</text>
<text x="460" y="1074" text-anchor="middle">CE rate 1/min, 1/hr, 1/day</text>
<text x="460" y="1092" text-anchor="middle">+ Kelle/Beluga/TraCT compare</text>
<rect x="610" y="1010" width="270" height="100" rx="8" fill="#fed7aa" stroke="#ea580c"/>
<text x="745" y="1034" text-anchor="middle" font-weight="bold">Baseline (2-3)</text>
<text x="745" y="1056" text-anchor="middle">vanilla vLLM + EDAC default</text>
<text x="745" y="1074" text-anchor="middle">v1 BlockShard / OAEP-KV</text>
<text x="745" y="1092" text-anchor="middle">+ R47.2/R47.3 simulator-only</text>
<line x1="460" y1="978" x2="175" y2="1010" stroke="#94a3b8" stroke-width="1"/>
<line x1="460" y1="978" x2="460" y2="1010" stroke="#94a3b8" stroke-width="1"/>
<line x1="460" y1="978" x2="745" y2="1010" stroke="#94a3b8" stroke-width="1"/>
<rect x="180" y="1150" width="560" height="120" rx="8" fill="#cffafe" stroke="#0891b2" stroke-width="2"/>
<text x="460" y="1176" text-anchor="middle" font-weight="bold">Final Submission</text>
<text x="460" y="1198" text-anchor="middle">Tier-1: OSDI 2027 (Apr) / USENIX Security 2027 (Feb) / HPCA 2027 (Aug)</text>
<text x="460" y="1220" text-anchor="middle">Tier-2: ITC 2027 / DAC 2027 / DATE 2027 6p / IEEE TCAD short</text>
<text x="460" y="1242" text-anchor="middle">Manuscript 70% by Wk 13 → polish + artifact eval Wk 14-16</text>
<line x1="460" y1="1110" x2="460" y2="1150" stroke="#94a3b8" stroke-width="1"/>
<rect x="40" y="1300" width="840" height="60" rx="8" fill="#fff7ed" stroke="#92400e" stroke-width="1"/>
<text x="460" y="1322" text-anchor="middle" font-weight="bold">Threshold legend</text>
<text x="460" y="1342" text-anchor="middle">Step 2: ±5% baseline match · Step 3: solo Mech #1 ≥ 50% target · Step 4: integrated ≥ 90% / cross-isolation 100% / fault ≥ 95%</text>
</svg>

---

## 5. Tier-1 Top 3 카드

### 🥇 [P1 PrefixGuard](tier1/01-prefixguard.md) — Score 8.9 (OSDI 2027 13p Accept)

**Hypothesis**: CXL-attached prefix block 의 hour-scale lifetime 에 맞춰 Patrol Scrub Control 의 hour-단위 interval 을 차등 (long-lived 1hr / short-lived disabled) 하면 silent corruption 90% 감소 + scrub overhead <30%.

### 🥈 [P3 Quarantine](tier1/02-quarantine.md) — Score 8.5 (USENIX Security 2027 13p Conditional Accept)

**Hypothesis**: CXL 3.x DPA tracking + poison + ECS mailbox + Memory Event Record 4 feature 를 vLLM agent isolation 과 결합하여 multi-agent KV 의 cross-agent silent corruption 을 100% 차단 + throughput drop <5%.

### 🥉 [P4 PATroller](tier1/03-patroller.md) — Score 8.1 (HPCA 2027 12p Conditional Accept)

**Hypothesis**: HBM3 PAT counter top-k row (1s polling, top-32) 를 software-level KV migration trigger 로 활용 (8K activations/sec threshold) → Rowhammer-induced silent corruption 95% 감소 + migration overhead <3%.

---

## 6. Tier-2 독립 Top 3 카드

### T1 [P8 ECS-Trace](tier2/01-ecs-trace.md) — Score 7.9 (ITC 2027 6p Accept)

**Hypothesis**: HBM3 ECS mailbox (IEEE 1500 TAP, 10s query interval) history 를 KV block lifetime 의 reliability trace 로 활용 → 누적 CE 가 collapse 직전인 block 을 prefetch eviction → long-context (>128k) silent corruption 90% 차단.

### T2 [V3 Quarantine-Mini](tier2/02-quarantine-mini.md) — Score 7.5 (DAC 2027 6p Accept, paper pair V3 ↔ P3)

**Hypothesis**: CXL ECS mailbox poison event detect 후 vLLM RFC #19329 affected token-range recompute path 의 latency 가 token range × layer linear → 8K token + 32 layer 에서 detect-to-recompute < 200ms (TPOT 40ms × 5 budget 내).

### T3 [V1 PrefixGuard-Lite](tier2/03-prefixguard-lite.md) — Score 7.25 (DATE 2027 6p Accept)

**Hypothesis**: Linux 6.16 EDAC scrub_subsystem 의 hour-단위 interval knob 을 vLLM prefix lifetime histogram 에 fit → optimal scrub interval = p75 lifetime 일 때 silent corruption rate 70% 감소 + scrub overhead <5%.

---

## 7. 미선정 로그

[unselected.md](unselected.md) — P2 SinkShield (KVSink concurrent), P5 Watermark (LM-Fix concurrent, Tier-2 보조), P6 VideoVeil (Sali-Cache adjacent), P7 EdgeARM (Tier-2 보조), V4 PATroller-Solo (R21 paper pair 한계로 미선정).

---

## 8. R49 cross-check 종합 (6 final)

| Idea | 개요 simulator | M1/M2/M3 R47 path | 실험 (4) Simulator/Tools | 일관 | ChaosMem? |
|------|----------------|-------------------|---------------------------|------|-----------|
| P1 | vLLM + LLMServingSim | R47.2 / R47.2 / R47.3 | vLLM 0.7+ + LLMServingSim 1.x | ✅ | ❌ |
| P3 | vLLM + LLMServingSim | R47.2 / R47.2 / R47.3 | vLLM 0.7+ + LLMServingSim 1.x | ✅ | ❌ |
| P4 | vLLM + NeuroSim V1.4 + LLMServingSim | R47.2 / R47.2 / R47.3 | vLLM 0.7+ + NeuroSim V1.4 + LLMServingSim | ✅ | ❌ |
| P8 | vLLM + LLMServingSim + NeuroSim V1.4 | R47.2 / R47.2 / R47.3 | vLLM 0.7+ + LLMServingSim + NeuroSim V1.4 | ✅ | ❌ |
| V3 | vLLM | R47.2 only | vLLM 0.7+ | ✅ | ❌ |
| V1 | vLLM | R47.2 only | vLLM 0.7+ | ✅ | ❌ |

**6/6 R49 일관, 6/6 ChaosMem 미사용, 6/6 R47.1 (gem5+vLLM 동시 X) 충족**.
