# DVFS-PILOT: Power-Mode-Aware Prune-vs-Retrieve-vs-Recompute Vision Recovery Policy with Ski-Rental Competitive Bound for Commodity Edge VLM

**Tier-1 · Grade: Accept (mean 7.75 / Total mean 7.33) · TRACE-C producer · 최저 scoop risk (3/10)**

> **Metaphor noun:** *DVFS-PILOT* — "pilot" 가 turn 별 3-way recover modality 를 power-mode 에 맞춰 조종.
> **Merge 출처:** ENERGY-PILOT(ai-opt) + algo-F EnergyPolicy + F3(legacy). F3 의 3-way recover modality + nvpmodel rail-level energy model + algo-F 의 ski-rental 2-competitive bound + ENERGY-PILOT 의 vLLM scheduler hook.

📖 약어 / 핵심 용어:
- **3-way recover modality**: focus 가 캐시된 vision 정보를 요구할 때 (a) prune 후 vision-encoder **재encode** (compute-bound), (b) cache **retrieve** (memory-bound), (c) 부분 attention **recompute** (소량) 중 택일.
- **DVFS / nvpmodel**: Dynamic Voltage Frequency Scaling. Jetson 은 `nvpmodel -m {0..}` 으로 power-mode (10/15/25/40W) 와 GPU/memory clock 을 조정. user-space CLI.
- **Ski-rental**: 미래 재방문 미지 하의 rent-vs-buy 고전 online problem. retain(누적 cost) vs recompute(1회 cost) 전환 임계 = 2-competitive (offline-optimal 대비 ≤2×).
- **rail-level energy**: tegrastats / INA3221 sysfs 의 VDD_GPU_SOC / VDD_CPU_CV rail 전력 적분 (token 당 J).

---

## 1. Research Questions

- **Master RQ**: vision-encode(compute-bound) vs decode/retrieve(memory-bound) 의 bottleneck 비대칭과 nvpmodel power-mode 를 결합한 rail-level energy cost model 로, turn 별 3-way recover modality 를 online 결정하면 고정 정책 대비 session energy 를 절감하면서 worst-case 를 보장할 수 있는가?
- **RQ-1**: 3-way recover modality 의 energy-optimal 선택이 nvpmodel mode (10/15/25W) 에 따라 **역전(cross-over)** 하는가, 그리고 그 cross-over 가 정량적으로 존재하는가?
- **RQ-2**: ski-rental 기반 DVFS-aware online 정책이 always-prune / always-retrieve 고정 정책 대비 session energy 를 **15-30%** 절감하고 **2-competitive** (offline-optimal 대비 ≤2×) 를 보장하는가?
- **RQ-3**: tegrastats 100ms 해상도의 variance 가 정책 간 energy 차이보다 작게 (INA3221 직접 polling + N≥30 turn averaging) 통제되어 EDP −12~20% 가 통계적으로 유의한가?

## 2. Two-Sentence Pitch

Jetson 0.7-1.1 token/J 환경에서 multi-focus turn 마다 vision 정보 recover 의 3-way 선택 — (a) prune 후 vision-encoder 재encode(compute-bound), (b) cache retrieve(memory-bound), (c) 부분 attention recompute — 의 energy-optimal modality 는 nvpmodel power-mode 에 따라 역전할 수 있다. 본 idea 는 vision-encode(compute) vs decode/retrieve(memory) bottleneck 비대칭과 nvpmodel 을 결합한 rail-level energy cost model 로 turn 별 modality 를 online 결정하되, 미래 focus 재방문 미지 하에서 **ski-rental 2-competitive** 보장으로 worst-case 를 막는다.

## 3. 가설 + Falsification

**가설:** vision-encode 는 compute-bound, retrieve 는 memory-bound 라는 비대칭(Modality Inflation) 때문에 3-way recover energy 의 optimal 선택이 nvpmodel mode (10/15/25W) 에 따라 갈리며, ski-rental 기반 DVFS-aware online 정책은 always-prune/always-retrieve 고정 정책 대비 session energy 를 15-30% 절감하고 2-competitive 를 보장한다.

**Falsification (4):**
1. 3-way modality energy 순위가 모든 nvpmodel mode 에서 동일 (cross-over 없음) → DVFS-aware 가치 소멸 → **measurement-only 로 reposition** (refuted-hypothesis insight, drop 아님).
2. ski-rental switch (retain-cost = recompute-cost) 의 online overhead 가 절감분 상쇄 → 기각.
3. tegrastats 100ms 해상도의 variance 가 정책 간 energy 차이보다 큼 → INA3221 직접 sysfs polling + N≥30 turn averaging 으로 완화, 실패 시 측정 재설계.
4. DVFS mode-switch latency 가 turn-level 보다 느려 동적 전환 비실용 → turn-window 단위 mode 고정 + within-window modality 선택.

## 4. Workload Evidence (Step 0-α 정량 인용)

- **Bottleneck 이분법 (verified, Modality Inflation [arXiv:2512.22695](https://arxiv.org/abs/2512.22695)):** vision encoding 은 **compute-bound** (FP16 tensor core saturate, bandwidth 무시 가능), language decoding 은 **memory-bandwidth-bound** (weight+KV HBM streaming). energy overhead 모델별 **17-94%**. video 시 CPU 100% / GPU ~20%. → 3-way modality energy 비대칭의 직접 근거.
- **DVFS 민감도 (verified, PAISE'25 [arXiv:2506.09554](https://arxiv.org/abs/2506.09554)):** memory freq 하향(Power Mode H) 시 latency **+370%**, energy **+72%** (power −52%) → **memory bandwidth 가 dominant**. Power Mode B 는 MAXN 대비 power −51%. → memory freq 가 energy 1차 변수, DVFS calibration anchor.
- **Edge energy 기준점:** Jetson token/J **0.7-1.1** (CHIME 측정 baseline). edge energy 실측 방법론은 Sustainability-Aware (Jetson Orin NX) batch size 4 sweet spot.

## 5. 기준 코드베이스 (R52.1)

- **vLLM** `main` commit `7c37096`. `Scheduler.schedule`(L335) 는 **batch-level** (per-request policy hook 아님, CF) → Orin bs=1 edge serving 에선 server loop 직접 삽입(단순). `EncoderCacheManager.can_allocate`(L119) 로 retrieve 가능 판단.
- **SGLang** `main` commit `e958f45` (secondary 대조). **llama.cpp** `main` commit `dbe9c0c` (Jetson container).
- **HF model:** `Qwen/Qwen2.5-VL-7B-Instruct`.
- **deps:** tegrastats / INA3221 sysfs (user-space, kernel-patch 아님 — R45.1), nvpmodel/jetson_clocks (user-space CLI), flash-attention skill (decode kernel cost profile).
- **Jetson Orin NX 16GB 정정 spec (CF-4):** CUDA core **1024**, LPDDR5 **102.4 GB/s**, nvpmodel **10/15/25W+MAXN (JetPack 5.x); 40W = JetPack 6.2 Super Mode 한정**, Ampere **sm_87**, `concurrentManagedAccess=0`. tegrastats/INA3221 rail-level (VDD_GPU_SOC/VDD_CPU_CV) energy 측정 ✅ feasible ([appnote](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)). secondary RTX 5090 (고정 power 대조).

## 6. 동작 원리 (R53 inline)

### M1 — Rail-level 3-way Recover Energy Cost Model

**① 동작원리 4요소**
- **(무엇을)** re-encode(vision-tower forward, compute/tensor core) vs retrieve(LPDDR5 read, memory) vs recompute(attention only, 소량) 의 energy 를 nvpmodel mode × focus-shift × KV-residency 함수로 모델링.
- **(왜 작동)** power-mode 민감도가 modality 마다 다름 (PAISE'25: memory freq 가 energy 1차) → re-encode(compute) 와 retrieve(memory) 의 optimal 이 mode 에 따라 cross-over.
- **(구현변경점, CF)** tegrastats/INA3221 sysfs polling (VDD_GPU_SOC) per-token J. **recompute(attention) vs reencode(vision tower) 별도 modality 명시 측정** (arch-sys F25, 혼동 금지).
- **(검증 시나리오)** cross-over pilot: 3 modality × 3 mode (10/15/25W) energy sweep, N≥30 turn averaging.

**② 기대효과**: cross-over regime 식별 → mode 별 optimal modality table.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Power telemetry | rail-level J/token | Jetson SW | tegrastats / sysfs INA3221 (VDD_GPU_SOC) | external probe | 중 | OS ✓ |
| Retrieve 가능 판단 | encoder cache hit/miss | vLLM V1 | [`vllm/v1/core/encoder_cache_manager.py#L119-L160`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/encoder_cache_manager.py#L119-L160) `can_allocate` | read | 중 | clone ✓ |
| Decode kernel cost | memory-bound retrieve cost | flash-attention | [`~/skills/AI-Research-SKILLs/10-optimization/flash-attention`](https://github.com) (local skill) | profile | 중 | skill ✓ |

**④ 검증 trace (R52.3)**: `can_allocate`(L119) clone ✓. tegrastats/INA3221 = user-space. **recompute/reencode modality 분리 측정.**

### M2 — Ski-Rental DVFS-Aware Online Selector (2-competitive)

**① 동작원리 4요소**
- **(무엇을)** 미래 focus 재방문 미지 하에서 retain(누적 retrieve cost) 이 recompute 1회 cost 도달 시 전환하는 ski-rental selector. turn-window 단위 nvpmodel 고정 + within-window modality 선택.
- **(왜 작동)** ski-rental 의 2-competitive 보장이 worst-case (focus 재방문 패턴 적대적) 를 막음 — offline-optimal 대비 ≤2×.
- **(구현변경점, CF)** vLLM `Scheduler.schedule` 는 batch-level → **Orin bs=1 edge server loop 에 selector 직접 삽입** (단순). nvpmodel CLI (`sudo nvpmodel -m`, user-space).
- **(검증 시나리오)** always-prune / always-retrieve / ski-rental 3-way session energy 비교 (재방문 skew sweep).

**② 기대효과**: session energy −15~30% (재방문 skew 시) + 2-competitive worst-case.
**③ 구현변경점 (R52.2 7-col 표)**

| 컴포넌트 | 역할 | 프레임워크 | 파일 (GitHub, R68 main+line-anchored) | 수정 유형 | 난이도 | 검증 |
|---|---|---|---|---|---|---|
| Policy selector | 3-way modality 결정 (ski-rental) | vLLM V1 / edge loop | [`vllm/v1/core/sched/scheduler.py#L335-L400`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py#L335-L400) `schedule` (batch) → bs=1 server loop | metadata layer | 상 | clone ✓ |
| DVFS 제어 | nvpmodel mode | Jetson SW | `nvpmodel`/`jetson_clocks` (user-space) | CLI hook | 하 | OS ✓ |

**④ 검증 trace (R52.3)**: `scheduler.py schedule`(L335, batch-level hook 정정) clone ✓. nvpmodel = user-space (R45.1 kernel-patch 아님). **CF-4: 40W=JetPack6.2 명시.**

## 7. End-to-End Evaluation

- **Multi-turn / multi-focus:** R4 FOCUS-COVERAGE (focus-trajectory + Coverage-Energy Pareto). MMDU / ConvBench / MileBench / MultiVerse.
- **Long-video:** Video-MME / MLVU (focus 재방문 시나리오 합성).
- **합성:** focus 재방문 skew(uniform~heavy-tail) 제어 — ski-rental switch 임계 검증.
- **Baseline:** always-prune (SparseVLM 재encode 잦음), always-retrieve (SparseVILA full cache), V-Rex(HW), Sustainability-Aware(text routing), EdgeReasoning(EDP Pareto). offline-optimal (oracle 재방문) 으로 competitive ratio 측정.

## 8. 실험 7요소 (12-16주)

1. **Hardware**: Jetson Orin NX 16GB (1024 core, nvpmodel 10/15/25W). secondary RTX 5090 (고정 power 대조).
2. **Model**: Qwen2.5-VL-7B (vision tower + LLM backbone re-encode/retrieve/recompute 분리 계측).
3. **Framework**: bs=1 edge server loop (vLLM V1 / llama.cpp container) + ski-rental selector.
4. **Energy 측정 (TRACE-C producer)**: INA3221 sysfs polling (VDD_GPU_SOC/VDD_CPU_CV) per-token J 적분 protocol, N≥30 turn averaging. tegrastats 100ms 한계 자체를 falsification 으로 명시.
5. **Pilot (gate)**: cross-over pilot (3 modality × 3 mode 소규모). 존재 시 Tier-1(MLSys 승격), 부재 시 Tier-2 measurement (DATE/ISLPED/IISWC).
6. **Steps**: (a) INA3221 sysfs probe + per-token J protocol → (b) 3 modality 분리 계측 (reencode/retrieve/recompute) → (c) cross-over pilot → (d) ski-rental selector (bs=1 loop) → (e) fixed-policy 비교.
7. **Metrics**: session energy (J), EDP, e2e latency, competitive ratio (vs offline-optimal), cross-over 존재 여부.

## 9. 예상효과 5-axis 표 (Energy 강조)

| Axis | 예상 개선 | 조건/scope |
|---|---|---|
| **Energy/Power** ★★ | session energy −15~30% (재방문 skew) + 2-competitive worst-case | cross-over regime 존재 시 |
| Latency | EDP −12~20%, e2e −10~20% (freq↓ 회피) | thermal-bound 아닌 구간 |
| Performance | latency-energy Pareto 개선 | — |
| Cost eff. | SW-only, HW 추가 없이 V-Rex 류 효과 일부 | commodity Jetson |
| Memory eff. | retrieve 선택 시 cache 보존 (간접) | — |

## 10. 관련연구 + 차별화

- **Modality Inflation** [arXiv:2512.22695](https://arxiv.org/abs/2512.22695) — energy 17-94%, "compute-heavy vision encoders vs visual token sequences during prefill". → **차별화: stage-DVFS 확인하나 vision-KV recover 정책 미결합, A100(edge 아님).**
- **V-Rex** [arXiv:2512.12284](https://arxiv.org/abs/2512.12284) (HPCA'26) — edge KV retrieval 3.1-18.5× energy, dedicated HW. → **차별화: commodity Jetson SW-only.**
- **Sustainability-Aware** [arXiv:2512.04088](https://arxiv.org/abs/2512.04088) — Jetson Orin NX text-LLM routing energy. → **차별화: vision-KV recover modality energy. baseline 격상.**
- **EdgeReasoning** [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) (IISWC) — Jetson accuracy-latency-energy Pareto. → 누락 baseline 추가 (EDP 비교).
- **PAISE'25** [arXiv:2506.09554](https://arxiv.org/abs/2506.09554) — memory freq↓ → latency +370%/energy +72%. → DVFS calibration anchor.
- ski-rental 2-competitive (classical online algorithm) — worst-case 보장, 경쟁자 전무.

## 11. Implementation Consistency

- **R47 path**: Application-level. edge bs=1 server loop 에 ski-rental selector + OS tegrastats probe. Jetson llama.cpp/vLLM container.
- **CF 일관성**: scheduler hook=batch-level → bs=1 server loop (CF). recompute/reencode modality 분리 (arch-sys F25). nvpmodel 40W=JetPack 6.2 (CF-4). tegrastats/INA3221/nvpmodel = user-space.
- **TRACE-C producer**: per-token J 적분 protocol 을 R1/R4/R5 가 energy axis 측정에 재사용.

## 12. Reproducibility Checklist (5 필드)

1. **Clone Spec**: vLLM `7c37096`. `scheduler.py schedule`(L335, batch-level), `encoder_cache_manager.py can_allocate`(L119) verified, hallucinated 0건.
2. **Environment**: JetPack 6.2, CUDA 12.x, tegrastats/INA3221 sysfs, nvpmodel/jetson_clocks, Python 3.10+.
3. **Build**: bs=1 edge server loop 패치 + INA3221 polling daemon. nvpmodel mode 전환 smoke.
4. **Patch List**: M1(INA3221 probe + can_allocate read + modality 분리 계측) / M2(ski-rental selector + scheduler/server loop + nvpmodel CLI).
5. **Smoke Test**: 3 modality × 3 mode energy sweep 동작, ski-rental switch 임계 트리거, N≥30 averaging variance < 정책 차이.

## 13. Scoring 및 이유 (R67, phase2prime Section D)

| Reviewer | sub1 | sub2 | sub3 | sub4 | rev-mean |
|---|---|---|---|---|---|
| Novelty (Mech/Comb/Hypo/D2) | 7 | 8 | 8 | 8 | 7.75 |
| Differentiation (RW/Clarity/Pos/Scope) | 8 | 7 | 8 | 7 | 7.5 |
| Impact (Mag/Breadth/Adopt/D1) | 6.5 | 6 ▼ | 7 | 6.5 | 6.5 |
| AI-impl (Src/Kernel/Integ/D6) | 6.5 | 7.5 | 6.5 | 7 | 6.875 |
| Arch-sys (R47/fit/HW/D6) | 8 | 8 | 8 | 8 | 8.0 ★ |

- **★ 전체 최고: Arch-fit/D6 = 8.5 (개별)** — rail-level INA3221 측정이 세션 최고 feasibility, tegrastats 100ms 한계 자체를 falsification 으로 전환.
- **▼ 전체 최저: Impact-Breadth = 6** — nvpmodel-bound (commodity Jetson 한정).
- **이유**: 세션 최강 novelty (ski-rental 2-competitive + commodity SW-only energy), **최저 scoop 3/10** (V-Rex=HW, Sustainability=text 직교). cross-over pilot 통과 시 MLSys, 부재 시 measurement (IISWC/DATE) 로 생존. **Total mean 7.33 / Grade Accept.**

## 14. R14.4 Decision Tree

```
preliminary gate: cross-over pilot (3 modality × 3 mode energy sweep)
├─ cross-over 존재 (modality 순위가 mode 에 따라 역전)
│   ├─ ski-rental 절감 ≥15% + 2-competitive 유지 → DVFS-aware policy [MLSys Tier-1]
│   └─ online overhead 상쇄 (F2) → turn-window mode 고정 + within-window 선택만
├─ cross-over 부재 (F1) → DVFS-aware 가치 소멸 → "edge VLM 3-way recover energy 첫 정량 characterization" measurement paper [IISWC/DATE/ISLPED]
└─ tegrastats variance > 정책 차이 (F3) → INA3221 직접 polling 강화, 실패 시 측정 재설계
  └─ DVFS switch latency > turn (F4) → turn-window 단위 mode 고정
```

## 15. Inter-idea Dependency

- **R2 = TRACE-C producer**: rail-level energy probe (tegrastats/INA3221 sysfs) + per-token J 적분 protocol (N≥30 turn averaging). **R1/R4/R5 가 energy axis 측정에 재사용.**
- **R2 ← R4 (BENCH-D)**: multi-focus 재방문 시나리오 토대.
- **독립 paper unit**: R2 = energy MLSys/IISWC (Paper Pair 아님, 독립).

## 16. Stakeholder (7-row)

| Stakeholder | 관심사 | R2 제공 가치 |
|---|---|---|
| Edge 운영자 | battery/thermal budget | session energy −15~30% SW-only |
| MLSys / 시스템 연구자 | online algorithm 보장 | ski-rental 2-competitive |
| Sustainability 평가자 | carbon/J per token | rail-level 정량 + cross-over characterization |
| HW vendor (Jetson) | nvpmodel 활용 효율 | mode 별 optimal modality table |
| 후속 연구자 (R1/R4/R5) | energy 측정 protocol | TRACE-C producer (재사용) |
| 측정/벤치 커뮤니티 | edge VLM energy 부재 | 3-way recover 첫 정량 (measurement fallback) |
| 프레임워크 개발자 | bs=1 edge serving | server loop selector 패치 |

## 17. Boundary (5-axis)

| Axis | In-scope | Out-of-scope |
|---|---|---|
| Modality | vision recover 3-way (encode/retrieve/recompute) | text-only routing (Sustainability) |
| Device | commodity Jetson (nvpmodel DVFS) | dedicated HW accelerator (V-Rex) |
| Mechanism | energy cost model + ski-rental online | offline schedule (재방문 oracle 만 비교군) |
| Measurement | SW-only rail-level (INA3221) | kernel-level power patch (R45.1 밖) |
| Batch | bs=1 edge serving | datacenter batch scheduling |

## 18. Self-Check

- [x] RQ 3개 의문문 + 정량 (cross-over 존재 / 15-30%·2-competitive / EDP −12~20%)
- [x] CF-1 (energy 측정은 tegrastats/INA3221 user-space, prefetch API 무관)
- [x] CF-2 (can_allocate=encoder embedding hit/miss read, KV 아님 — modality 분리 측정)
- [x] CF-3 (retrieve 는 LPDDR5 read, NVMe round-trip 비교는 R1/R5)
- [x] CF-4 (1024 core, 40W=JetPack6.2, INA3221 feasible)
- [x] R68 GitHub link `blob/main/{path}#L{A}-L{B}` (fixed path)
- [x] R8 arxiv clickable markdown
- [x] R67 ★(Arch-fit/D6 8.5) / ▼(Impact-Breadth 6)
- [x] vendor-neutral title (device명 title 없음)
- [x] 18 의무 섹션 전부
