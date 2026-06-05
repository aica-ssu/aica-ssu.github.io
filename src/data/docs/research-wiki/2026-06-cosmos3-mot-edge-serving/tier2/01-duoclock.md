# S2 DUOCLOCK (Tier-2 #1 독립, avg 7.4)

**Phase-Aware Decoupled Memory(EMC)/GPU DVFS for the Intra-Request AR-decode↔DM-denoise Asymmetry of Dual-Tower MoT on Edge Unified Memory** · DATE / ISLPED / IISWC 2027

---

## 1. 개요

- **metaphor↔mechanism**: DUOCLOCK = EMC clock + GPU clock 두 시계를 phase 에 맞춰 독립 제어. AR phase 는 memory 시계를, DM phase 는 compute 시계를 각각 최대로 — 두 시계가 phase 마다 역할을 주고받는다.
- **한 줄**: 한 inference 내부 AR(memory-bound)↔DM(compute-bound) 비대칭을 Jetson EMC/GPU clock-domain 물리 분리 DVFS 로 각 phase 에 맞춰 독립 제어 + J/chunk 재정의.
- **single-insight scope**: L2(TWOCLOCKS) reposition + Tier-1→Tier-2 강등 (DVFS governor 성숙 + concurrent 다수 → measurement-결합 letter 가 정직).

## 2. 기존 연구 한계 · GAP (workload evidence 수치 포함, 3 reposition 축)

- **(a) intra-request dual-regime**: 한 inference 내부에서 AR↔DM 교대 — 기존(DualScale [arXiv:2602.18755](https://arxiv.org/abs/2602.18755), GreenLLM [arXiv:2508.16449](https://arxiv.org/abs/2508.16449), SparseDVFS [arXiv:2603.21908](https://arxiv.org/abs/2603.21908))은 across-request / 동일 AR 모델 prefill-decode phase 스케줄. 한 요청 내 AR↔DM tower 전환 freq 비대칭 분리는 미개척.
- **(b) UMA EMC+GPU decoupled control**: 기존은 discrete GPU(GPU clock only) 또는 cloud — 데이터센터 DVFS 에 EMC 분리 개념 자체가 없음. (단 SparseDVFS 가 EMC축 선점 → 약화.)
- **(c) J/chunk metric**: tegrastats 33-50ms < step → J/step 직접 측정 불가 → **J/chunk(policy 15Hz)·J/inference-phase 재정의** (자체가 edge MoT energy 측정 방법론 기여).
- **workload evidence**:
  - AR decode = memory-bound (decode 가 prefill 대비 **192-569× latency 지배** AGX Orin, [arXiv:2511.01866](https://arxiv.org/abs/2511.01866)); DM denoise = compute-bound([arXiv:2312.14385](https://arxiv.org/abs/2312.14385)).
  - GPU freq 2842→180MHz: **에너지 42%↓, latency 1-6%↑** (decode freq 둔감, [arXiv:2501.08219](https://arxiv.org/abs/2501.08219)); DynamoLLM **에너지 52%↓** ([arXiv:2408.00741](https://arxiv.org/abs/2408.00741)).
  - policy forward ~263ms/forward(2.1s/8) → J/chunk 직접 측정 가능.

## 3. 제안 기법 (mechanism ≤2)

### M1. Phase-transition-triggered decoupled EMC/GPU DVFS governor (intra-request)

- **① 추가 scheme**: runtime phase boundary(AR-prefill done→DM-denoise start→re-prefill)를 hook 으로 노출, EMC freq governor 와 GPU freq governor 독립 set. `/sys/class/devfreq/{gpu,emc}/.../{governor,userspace/set_freq}` + nvpmodel pinning.
- **② 해결 문제**: 단일 nvpmodel 모드가 두 상반 phase 동시 최적 불가(G1).
- **③ 동작 원리**: (1) phase boundary callback 등록(S1 TowerPhaseFSM hook 재사용 — inter-idea 연계). (2) per-phase {GPU_freq, EMC_freq} LUT 를 J/chunk 특성화로 사전 산출. (3) AR phase: EMC max + GPU↓(memory-bound, compute freq 둔감); DM phase: GPU max + EMC↓(compute-bound). (4) **전환 granularity = chunk(2.1s) 단위**(step 아님) → settle cost 흡수, 15Hz 주기 내 잦은 전환 회피.
- **④ 기존 실패 이유 + 차별화**: DualScale=disaggregated(across-request), GreenLLM=GPU-freq only(EMC 無), SparseDVFS=operator-sparsity triplet(discrete GPU), DynamoLLM=cluster. DUOCLOCK = edge Jetson EMC↔GPU clock-domain 물리 분리 + diffusion denoise phase + 한 요청 내 AR↔DM 전환.

### M2. J/chunk + J/inference-phase 특성화 LUT

- **① 추가 scheme**: Cosmos3-Nano phase별 에너지-주파수 곡선을 Orin/Thor INA3221 rail 적분 실측 → governor LUT. (S3/LEDGERMARK 와 공유 산출물 — inter-idea producer.)
- **② 해결 문제**: edge MoT J/step 특성화 공백 + tegrastats 33-50ms < step → J/step 직접 불가.
- **③ 동작 원리**: GPU/EMC freq grid sweep × {VLM-decode, T2I-denoise, policy 15Hz} × INA3221(VDD_GPU_SOC) 적분, policy forward(263ms) 직접 / gen step(<50ms) N-step 적분.
- **④ 기존 실패 이유 + 차별화**: EdgeReasoning(AR-only)·Modality-Inflation(A100-only) 미커버. diffusion denoise 의 EMC-둔감성 실측이 신규.

## 4. 평가 · 실험 플랜 (R20-β 7요소)

- **HW**: AGX Orin 64GB(EMC/GPU freq 자유도 큼, 1차) / Orin NX 16GB / Thor(차세대 freq governor 대조) / RTX Pro 6000(phase 측정 baseline, DVFS 비대상).
- **모델**: Cosmos3-Nano; fallback BAGEL-7B-MoT(VLM+T2I) + Cosmos3-Nano-Policy-DROID(policy).
- **워크로드**: VLM-decode-heavy / T2I-denoise-heavy / policy(15Hz 교대) — 단일 workload 중심(policy 우선).
- **도구**: `/sys/class/devfreq` userspace governor, INA3221 sysfs 적분 harness, tegrastats.
- **ablation + baseline**: diffusion EMC-둔감성 ablation. baseline — (b1) 고정 nvpmodel MAXN, (b2) GPU-only DVFS([arXiv:2501.08219](https://arxiv.org/abs/2501.08219)) 적응, (b3) DynamoLLM-style([arXiv:2408.00741](https://arxiv.org/abs/2408.00741)) GPU-freq, (b4) GreenLLM phase-DVFS([arXiv:2508.16449](https://arxiv.org/abs/2508.16449)) 적응(EMC 미분리 대조).
- **주차별 구현**:

  | 주차 | 작업 |
  |---|---|
  | W1-2 | `/sys/class/devfreq` EMC/GPU userspace governor 제어 검증 + INA3221 적분 harness |
  | W3-5 | phase boundary hook(S1 FSM 재사용) + per-phase freq LUT sweep |
  | W6-8 | freq settle latency 실측 + chunk-granularity 전환 정책 + SLO gate |
  | W9-10 | baseline 비교 + diffusion EMC-둔감성 ablation |

- **preliminary metrics**: SparseDVFS triplet 대비 modality-regime-aware 순증분 J/chunk, freq settle latency.

## 5. 예상 효과 (보수치, scope 명시)

- J/chunk **−25~38%** (AR phase GPU-freq↓ + DM phase EMC-freq↓, **phase별 독립 가정 명시 — 단순 합산 아님**).
- SLO 위반(policy 2.1s 주기) 추가 **<3%p**.
- latency overhead **<6%** (freq settle chunk-단위 흡수).
- scope: edge Jetson 한정, SparseDVFS 순증분 ≥5%p 일 때만 standalone letter (미달 시 S3 흡수).
