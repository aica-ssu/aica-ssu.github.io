# KILN — Thermal-Bounded Sustained Throughput Characterization for Multi-Agent Hybrid Edge Serving

> **Hook**: edge multi-agent hybrid 서빙의 실효 throughput 천장은 peak가 아니라 **thermal 정상상태**다. sustained load 2 iteration 내 throughput 반토막([arXiv:2603.23640](https://arxiv.org/abs/2603.23640))인데, agent 수 × precision island 비율 × expert-resident 비율을 동시에 sweep한 정상상태 곡선은 아무도 측정한 적이 없다. KILN은 단일 AGX Orin에서 "thermal-bounded sustainable agent count"를 first-to-report로 규명한다.
>
> **Score (Phase 2')**: novelty 5 / diff 6.5 / impact 5.5 / ai-impl 8 / arch-impl 9 = **avg 6.8** (Δ0). Tier-2 Top.
> **Venue**: DATE 2027 (대안 ISLPED / IEEE CAL). Track B 측정 letter — 가장 안전(arch 9 / impl 8).
> **舊 이름**: FORGE → KILN (기존 wiki FORGE acronym 충돌 + 가마=일정 온도 지속 가열 은유).

---

## 1. Research Questions

- **RQ1 (정량)**: agent 수(1→8+)·precision island 비율(BF16 attention+선행 Mamba+Conv1D vs FP8 routed)·expert-resident 비율(offload 강도 0→부분)·nvpmodel(15W/30W/50W/MAXN) 조합별 **thermal 정상상태(≥수 분 가열 후) tok/s·junction temp·DVFS clock**은 cold-start 대비 얼마나 떨어지며, "sustainable agent count vs power-mode" 곡선은 어떤 형태인가? (예상: 정상상태 vs cold gap −40~60%).
- **RQ2 (정량)**: 정상상태 thermal 부하의 1차 동인은 **BF16 precision island(BW 2×)인가 expert-DMA(per-token 1.84GB read)인가**? expert-offload 트래픽을 공변량으로 추가한 회귀로 island 효과를 expert-DMA 교란에서 분리 가능한가?

---

## 2. 기준 코드베이스 (Baseline Source)

- **서빙 스택**: vLLM v1, commit **`7c37096620`** pin (R30.6.4 reproducibility). Jetson aarch64는 pip 미지원 → NGC container `nvcr.io/nvidia/vllm:26.01-py3`(Orin/Thor 지원) 사용.
- **모델**: Nemotron-3-Nano-30B-A3B (HF, total 31.6B / active 3.2B), **FP8** selective island. proxy = [Nemotron-3-Nano-4B](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b)(동일 hybrid 구조, Orin Nano 8GB용).
- **측정 도구**: **tegrastats**(CPU/GPU/EMC clock + thermal zone 온도 + 전력), **jtop/jetson_stats**(INA3221 rail Python API), **INA3221 sysfs**(`/sys/bus/i2c/drivers/ina3221/1-0040/...` → VDD_GPU_SOC / VDD_CPU_CV / VIN_SYS_5V0 분리), **nvpmodel / jetson_clocks**(power-mode·clock·팬 lock, kernel patch 불요), **외부 전력계**(Orin Nano carrier는 rail 통합 → per-component 분리에 필수). ncu 불요(Orin iGPU ga10b 미지원), simulator 불요.
- **HW 측정 가능성 검증** (arch-impl review [✅] 인용):
  | 항목 | 판정 | 근거 |
  |---|---|---|
  | tok/s·junction temp·GPU/EMC clock·power sweep | [✅] | tegrastats user-space, nvpmodel power-mode grid 확인 |
  | BF16 island 전력 분리 | [✅] | INA3221 VDD_GPU_SOC rail + island 통과 구간 회귀, nsys timeline 정렬 |
  | Orin Nano 풀모델 미적재 | [✅] | 소형 4B proxy로 thermal-tight 측정(풀모델은 64GB) |
  | 외부 power meter + ambient 통제 | [⚠️] | 측정 letter 특성, 실험실 enclosure 통제 권고 |
  - arch feasibility score 9/10 — 본 세션 HW feasibility 가장 깨끗(simulator·kernel patch·ncu 전부 불요).

---

## 3. 배경 / GAP

- sustained load에서 thermal이 peak compute를 압도, **2 iteration 내 throughput 반토막** ([arXiv:2603.23640](https://arxiv.org/abs/2603.23640)) — 단 단일 모델·단일 요청 측정에 그침.
- edge decode가 prefill 대비 **192–569× 점유**·memory-bandwidth-bound, decode-dominated 91.0–98.6% ([arXiv:2511.01866](https://arxiv.org/abs/2511.01866), IISWC 2025) — latency/energy는 보고하나 thermal 정상상태·multi-agent 차원 미측정.
- per-token expert read **1.84GB/token**(6 routed + 2 shared × 9.98MB FP8 × 23 MoE층) → AGX Orin 204 GB/s ÷ 1.84GB = **~9 tok/s 상한**. expert-DMA가 thermal 1차 동인 후보(precision island보다 클 수 있음 — cross-review 권고로 공변량 분리 필요).
- BF16 island(attention 6층 + 선행 Mamba 6층 + Conv1D)는 FP8 대비 BW **2×** → island 통과 구간이 thermal spike 유발 가능.
- Jetson junction **80°C 부근 downclock** → cold-start TPS는 sustained workload를 과대평가.
- GPU 0.5–1.7 vs CPU 27–77 MJ/Mtok ([arXiv:2604.24785](https://arxiv.org/abs/2604.24785)) — energy/thermal이 edge 1급 제약이라 정상상태 측정이 운영상 핵심.
- 동일 LPDDR5x 273 GB/s를 공유하는 DGX Spark·Thor 128GB(고용량)도 capacity≠bandwidth — agent scaling headroom이 thermal·BW 양쪽에 묶임.
- 기존 edge 측정은 대부분 단발·cold-start로 **2-iteration 반토막을 미보정** → "지속 가능한 동시 agent 수"라는 운영 질문에 답이 없음.

---

## 4. Mechanism (단일)

### 4.1 동작 원리 — M1: Sustained multi-agent thermal characterization harness + 공변량 회귀
- vLLM-FP8 on AGX Orin에서 **agent 수 × precision island 비율 × expert-resident 비율(offload 강도) × phase** 4차원 grid를 정상상태(cold→warmup→≥수 분 가열)에서 sweep.
- 정상상태 window에서 tok/s·junction temp·GPU/EMC clock(tegrastats)·전력(INA3221 VDD_GPU_SOC) 동시 기록. 전부 공식 user-space(nvpmodel/jetson_clocks/sysfs), kernel patch 없음.

### 4.2 기대 효과
- "thermal-bounded sustainable agent count vs power-mode" 곡선을 first-to-report로 산출.
- **island vs expert-DMA thermal 기여 분리** — expert-offload 트래픽을 회귀 공변량으로 추가해 island 효과를 expert-DMA 교란에서 떼어냄(cross-review 권고).
- 정상상태 vs cold gap을 −40~60% 수준으로 정량화 → 운영자에게 실효 천장 제공.

### 4.3 구현 변경점
| 구분 | Phase-1 원본(FORGE) | Phase-1' 정제(KILN) |
|---|---|---|
| 이름 | FORGE(wiki 충돌) | **KILN** |
| sweep 축 | agent × island × phase | **+ expert-resident 비율(offload 강도)** |
| 회귀 | island thermal만 | **expert-offload 트래픽 공변량 추가** |
| 분리 | island 기여만 | **island vs expert-DMA 분리** |
| R47 path | — | application-level only, vLLM-FP8 + tegrastats/nvpmodel/jetson_clocks + 외부 power meter, gem5 미사용 |

### 4.4 검증 시나리오
1. cold→warmup→정상상태(≥수 분) 가열 프로토콜로 2-iteration 반토막 영역 진입.
2. 정상상태 window에서 tok/s·temp·clock·power 수집.
3. agent × island × offload grid sweep (각 셀 정상상태까지 hold).
4. 회귀에 expert-offload 트래픽 공변량 추가 → island 계수와 expert-DMA 계수 분리.
5. sustainable agent count vs power-mode 곡선 도출 + cold gap 정량화.

---

## 5. 실험 플랜 (7-요소, 단일 scope)

1. **목표 지표**: sustainable agent count 곡선(신규 데이터) / 정상상태 vs cold gap(−40~60%) / island vs expert-DMA thermal 기여(공변량 회귀 분리).
2. **device**: **AGX Orin 64GB**(주 측정, FP8, INA3221 rail 분리) + **Orin Nano 8GB**(thermal-tight, Nemotron-3-Nano-4B proxy) + **외부 전력계**(Nano rail 통합 대응) + ambient 통제 enclosure.
3. **모델/precision**: 풀모델 FP8(AGX Orin) / 4B proxy FP8(Orin Nano). selective-FP8 island 보존.
4. **sweep 축**: **agent 수(1→8+) × precision island 비율 × expert-resident 비율(offload 강도) × nvpmodel(15W/30W/50W/MAXN)**.
5. **측정**: tegrastats(tok/s·junction temp·GPU/EMC clock·power) + INA3221 VDD_GPU_SOC rail + nsys timeline(island 구간 정렬) + 외부 전력계(Nano per-component).
6. **baseline 비교**: cold-start TPS vs 정상상태 TPS / 단일 모델([arXiv:2603.23640]) vs multi-agent+island+expert-DMA.
7. **분석**: 정상상태 window 회귀(expert-offload 공변량) → island/expert-DMA 계수 분리 + sustainable count vs power-mode 곡선.

| 축 | baseline | 목표(측정) | 조건 |
|---|---|---|---|
| [측정] sustainable agent count 곡선 | 미측정 | 정량 규명(신규 데이터) | thermal 정상상태 |
| [측정] 정상상태 vs cold gap | 미보고 | −40~60% 정량화 | sustained load |
| [측정] island vs expert-DMA thermal 기여 | 미측정 | 공변량 회귀 분리 | offload 강도 sweep |

---

## 6. 관련 연구 · 차별점

- **LLM Inference at Edge Sustained Load** [arXiv:2603.23640](https://arxiv.org/abs/2603.23640) (arXiv) — 단일 모델·단일 요청 sustained thermal. 차별: **multi-agent + hybrid island + expert-DMA 분리** first-to-report.
- **EdgeReasoning** [arXiv:2511.01866](https://arxiv.org/abs/2511.01866) (IISWC 2025) — latency/energy 특성화. 차별: thermal **정상상태** sustainable agent count.
- **Characterizing Mobile SoC** ([ACM 10.1145/3731569.3764808](https://dl.acm.org/doi/10.1145/3731569.3764808), peer-reviewed) — UMA BW 특성화. 차별: BW만이 아닌 thermal steady-state × agent scaling.

---

## 7. 왜 Tier-2 only인가

- **Top-tier scale-up 불가 이유**: 측정 letter 본질상 **mechanism novelty가 낮음**(novelty 5 / framing 6) — 새로운 알고리즘·제어 기법이 아니라 단일 HW(1–2종)에서의 특성화이므로 일반화가 제한된다. impact 5.5도 측정 letter 천장. MLSys/ASPLOS급 scale-up은 mechanism contribution이 필요하나 KILN은 데이터 contribution이 전부.
- **Tier-1 승격 조건**: sustainable agent count 곡선 + island/expert-DMA 분리가 **다수 HW SKU(Thor/DGX Spark 추가)로 일반화**되고 그 데이터가 새로운 예측 모델(예: 정상상태 throughput 회귀)로 발전하면 Track-A 후보.
- **인프라 hub 가치**: KILN의 진짜 가치는 점수가 아니라 **GOVERNOR predictive thermal 모델의 학습 데이터 제공** + TIDEMARK/HEARTH 등 모든 Track-A idea의 thermal steady-state baseline이라는 점. 측정 인프라 hub로서 세션 전체의 전제.

---

## 8. 약어 / 용어 풀이

- **expert**: MoE FFN 서브네트워크(2-matrix FC1/FC2, ≈9.98M param, FP8 ≈10MB).
- **precision island**: BF16로 보존되는 층 묶음(attention 6층 + 선행 Mamba 6층 + Conv1D) — FP8 대비 BW 2×.
- **expert-resident 비율**: expert population(30GB FP8) 중 LPDDR-resident 비율(offload 강도의 역).
- **junction temp**: SoC die 온도(~80°C 부근 downclock 트리거).
- **nvpmodel**: Jetson power-mode CLI(AGX Orin 15W/30W/50W/MAXN).
- **EMC**: memory controller clock — 메모리 BW 활용 proxy.
- **INA3221**: AGX Orin on-board power monitor(VDD_GPU_SOC / VDD_CPU_CV rail 분리).
- **sustainable agent count**: thermal 정상상태에서 SLO를 유지하며 동시 서빙 가능한 agent 수.
- **decode-dominated**: agentic workload에서 decode가 LLM time의 91.0–98.6% 점유.
