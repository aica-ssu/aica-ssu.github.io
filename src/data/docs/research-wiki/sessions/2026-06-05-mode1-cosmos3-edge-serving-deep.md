# Session — Cosmos3 MoT (16B dual-tower) 단일 edge GPU Omnimodal 서빙 DEEP re-spec (R72 첫 적용)

- **Date**: 2026-06-05
- **Mode**: 1 개선 모드 (직전 2026-06-04 cosmos3 세션의 Top-6 계승 + R72 Document-Depth Discipline 첫 적용) — 신규 ideation 없음, **구현 디테일 deep re-spec** 전용
- **사용자 쿼리 (원문 요지, 2026-06-05 피드백)**: "직전 세션(2026-06-04 cosmos3 MoT edge serving) 산출물의 **구현 디테일이 부족**하다 — tier 문서가 ~80줄에 그치고 R52.2(코드 anchor)/R70(완료 판정)/pseudo-code/preliminary study 가 누락. source-code 를 직접 검증하며 문서를 깊게 다시 써라."
- **직전 세션**: [sessions/2026-06-04-mode2-cosmos3-mot-edge-serving.md](2026-06-04-mode2-cosmos3-mot-edge-serving.md) · [summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md](../summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md) (R72 적용으로 **superseded**)
- **참여 agent (총 18)**: setup 2 · Phase 1-deep per-document 전담 6 · depth-gate fix 4 · Phase 2 검증 1 · Phase 1' fix 3 · Phase 3 wiki writer 2
- **세션 결과 (1줄)**: Top-6 **불변** (Tier-1 Q2 ANCHOR 7.6 / Q1 DRIFT 7.4 / S1 TIDELOOM 7.4 + Tier-2 S2 DUOCLOCK 7.4 / L3 KEELKV 6.6 / S4 SIDEPOOL 6.2, pair Q2×L3), 6 tier 문서를 ~80줄 → **363-404줄** source-code-anchored 로 재작성, anchor 검증율 47/48(98%), 정정 5건, 신규 코드-수준 발견 10건, 신규 must-cite 2편.

> Summary 번들: [summary/2026-06-05-mode1-cosmos3-edge-serving-deep/README.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/README.md)

---

## Section 0 — Executive Summary (R10 의무)

### Top-M 6 (R10-α.2) — 직전 세션 계승, 이번 재수행에서 깊어진 점 명시

#### ① Q2 — ANCHOR (Tier-1, MLSys/ICML 2027, avg 7.6 ★ 최강) · [01-anchor.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier1/01-anchor.md) (388줄)
- **전제**: generator tower 가 reasoner 의 K_AR/V_AR 을 N denoising step × CFG 2 pass 전 기간 read-only static prefix 로 cross-attend (`O_DM=Attn_full(Q_DM,[K_AR;K_DM],[V_AR;V_DM])`, Eq.8).
- **GAP**: decode-time KV quant(online·growing) 과 정반대인 정적·1회·read-only prefix → 비싼 1-shot Hadamard quant amortize + flow-step N 의존 closed-form bound. 경쟁작(QuantKeys / 33-method) 전부 bound 부재.
- **Mechanism**: M1 Hadamard-rotated 1-shot per-channel K_AR/V_AR INT4/INT3 quant / M2 `‖x̂_0−x_0‖ ≤ (Σ_n Δ_n·L_v^{(n)})·(L_softmax‖ΔK_AR‖+‖ΔV_AR‖)` / M3 bit-tightness plot 로 layer별 bit 자동 선택.
- **예상 효과**: conditioning-KV 2-4× 압축(품질손실 <1%), policy N=4 → INT3(bound-favorable 시 INT2).
- **Scoring**: nov 8 / diff 8 / impact 8 / ai-impl 7 / arch-impl 7 → **7.6**.
- **이번에 깊어진 점**: ❶ **post-RoPE K 캐시 확정** — cosmos3 `cached_kv` 가 RoPE 적용 *후* 저장됨을 소스 확인(L548) → Hadamard 를 RoPE 뒤에 걸어야 위치정보 보존이 코드로 입증. ❷ **K_AR 재산정** — 16×16 patch, policy ~1,050 tok, per-layer BF16 ~4.3MB / INT4 ~1.08MB (GQA 8-head, 36 gen layer); 단 serving 코드상 K_AR 은 **text-only**, 관측은 GEN tower 주입 → 두 시나리오 병기.

#### ② Q1 — DRIFT (Tier-1, NeurIPS/ICML/MLSys 2027, avg 7.4) · [02-drift.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier1/02-drift.md) (404줄)
- **전제**: 두 tower 가 동일 Qwen3-VL-8B co-init(L933) 이나 AR-CE vs flow-MSE 두 objective 로 quant 민감도 비대칭 발산.
- **GAP**: 교란변수 통제 하 발산 측정 부재 + UMA 예산 per-tower bit allocation 부재.
- **Mechanism**: M1 ρ_ℓ=S_ℓ^R/S_ℓ^G (Hutchinson HVP) divergence study (Q4 흡수) / M2 ILP/greedy bit-alloc (reasoner=AWQ/GPTQ, generator=SVDQuant) / M3 MMMU/GenEval/RoboLab subset.
- **예상 효과**: 균일-W4 대비 품질 0.5-1.5%p 회복 또는 footprint 10-18% 절감 (ρ_ℓ material 시만, conditional-gain 정직).
- **Scoring**: nov 7 / diff 8 / impact 8 / ai-impl 7 / arch-impl 7 → **7.4**.
- **이번에 깊어진 점**: ❶ **quant 메뉴 소스 확인** — vllm-omni 가 int8/mxfp8/mxfp4/mxfp4_dualscale/inc/auto-round/gguf 제공 → DRIFT 의 per-tower 경로(AWQ↔SVDQuant)가 실 메뉴에 mapping 가능. ❷ **qwen3_omni thinker 의 tower-prefix quant routing 선례**(L1150-1151) — tower 별 quant 분기가 upstream 에 이미 존재하는 hook point.

#### ③ S1 — TIDELOOM (Tier-1, MLSys/ASPLOS 2027, avg 7.4) · [03-tideloom.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier1/03-tideloom.md) (404줄)
- **전제**: 한 요청이 AR(memory-bound)→DM(compute-bound)을 단일 edge GPU 직렬 traverse 하나 vLLM(AR)+vLLM-Omni(DM) 분리 엔진. phase별 활성은 항상 8B.
- **GAP**: 분리 엔진은 phase 전환을 한 process·한 footprint 로 못 다룸 / 16B 전량 상주 낭비 / weight residency·런타임 통합 부재.
- **Mechanism**: M1 TowerPhaseFSM (plain-PyTorch 단일-process) + K_AR zero-copy / M2 phase-결정적 active-8B residency (double-buffered `cudaMemcpyAsync`, Orin prefetch 미지원 대응) / M3 chunk-pipelined AR∥DM overlap (MPS).
- **예상 효과**: peak resident −35~45%, RTX Pro 6000 ×2→×1 단일화, e2e −10~18%, chunk −15~25%.
- **Scoring**: nov 7 / diff 8 / impact 8 / ai-impl 7 / arch-impl 7 → **7.4**.
- **이번에 깊어진 점**: ❶ **dual hook 확정** — `_forward_local`(L648) + `_forward_sp`(L675) 두 경로에 FSM hook 삽입점이 코드로 특정됨. ❷ **diffusion 스택의 stream/staging greenfield 입증** — 현 코드에 stream-overlap·double-buffer 구현 부재 확인 → S1 의 systems 기여가 빈 영역임이 소스로 확정.

#### ④ S2 — DUOCLOCK (Tier-2, DATE/ISLPED/IISWC 2027, avg 7.4) · [01-duoclock.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier2/01-duoclock.md) (363줄)
- **전제**: 한 inference 내부 AR↔DM 교대 — 단일 nvpmodel 고정 freq 가 두 상반 phase 에 동시 최적 불가.
- **GAP**: 기존 phase-DVFS 전부 across-request / EMC 분리 개념 부재 / tegrastats 33-50ms<step → J/step 직접 불가.
- **Mechanism**: M1 decoupled EMC/GPU DVFS governor (AR: EMC max+GPU↓ / DM: GPU max+EMC↓) / M2 J/chunk·J/phase LUT (INA3221 적분).
- **예상 효과**: J/chunk −25~38%, SLO 추가 <3%p, overhead <6%.
- **Scoring**: nov 7 / diff 8 / impact 7 / ai-impl 8 / arch-impl 7 → **7.4**.
- **이번에 깊어진 점**: **Orin EMC governor 현실 정정** — EMC freq 직접 제어는 BPMP debugfs experimental 경로 → 미지원 시 **nvpmodel fallback** 으로 governor 설계를 이중화 (sysfs `/sys/class/devfreq/{gpu,emc}` 가용성 step0 실측 gate).

#### ⑤ L3 — KEELKV (Tier-2, DAC/DATE/IEEE-CAL 2027, avg 6.6) · [02-keelkv.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier2/02-keelkv.md) (380줄)
- **전제**: static cross-tower K_AR/V_AR(read-only)을 denoise loop 동안 per-layer L2 set-aside pin.
- **GAP**: vLLM/SGLang/llama.cpp/TRT 전부 L2 set-aside 미사용 / video-K_DM L2 pollution(2.15×) 회피.
- **Mechanism**: M1 `cudaAccessPolicyWindow` per-layer K_AR persisting + video-K_DM streaming demote (Orin sm_87 3MB 확정) / M2 SMEM staging fallback.
- **예상 효과**: per-step DRAM read −10~20%, latency −5~12%.
- **Scoring**: nov 6 / diff 7 / impact 6 / ai-impl 8 / arch-impl 7 → **6.6**.
- **이번에 깊어진 점**: **K_AR L2-fit 수치 출처 확정** — per-layer policy K_AR INT4 ~1.08MB(Q2 산출물)가 Orin 3MB L2 에 4× 여유 fit 임을 GQA 8-head/36-layer/1,050-tok 으로 재유도; pair {Q2×L3} 전제(Q2 INT4 → L3 L2-fit) 가 수치로 강화.

#### ⑥ S4 — SIDEPOOL (Tier-2, DAC/DATE/ISPASS 2027, avg 6.2) · [03-sidepool.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier2/03-sidepool.md) (372줄)
- **전제**: frozen VAE decode ∥ next-chunk denoise GPU stream-priority overlap + accelerator-complex placement LUT.
- **GAP**: 단일 edge GPU stream-priority overlap + iGPU/DLA-2D/PVA/CPU placement LUT 미존재.
- **Mechanism**: M1 low-priority CUDA stream VAE ∥ high-priority denoise + `cudaEvent` 동기 / M2 placement LUT (DLA 는 검증된 2D-subgraph 만).
- **예상 효과**: e2e chunk −15~30% (generation-mode only), J/chunk −10~20%.
- **Scoring**: nov 6 / diff 6 / impact 6 / ai-impl 7 / arch-impl 7 → **6.2**.
- **이번에 깊어진 점**: **DLA 비호환 op 목록 + NVDLA 버전 정정** — CausalConv3d / RMS_norm / AttentionBlock 이 DLA 미지원 op 임을 명시, NVDLA **v2.0**(구 문서 오기) 로 정정 → placement LUT 의 "DLA = 2D-subgraph only" 정직화가 op-level 로 뒷받침됨.

### Paper Pair
- **{Q2 ANCHOR (T1) × L3 KEELKV (T2)}** — Q2 의 INT4 K_AR(~1.08MB/layer) 산출물이 L3 의 per-layer L2 set-aside pin 을 Orin 3MB 에 4× 여유 fit 시키는 producer-consumer. 이번 deep 재수행에서 두 문서의 K_AR footprint 수치가 동일 산정식(16×16 patch / 1,050 tok / GQA 8-head / 36 gen layer)으로 정합 확인.

### 미선정 9건 (직전 세션 계승, 본 세션 재이데이션 없음)
- drop 3 (A5 Herald / A6 Switchback / Q3 PRISM) + 흡수 6 (A2 Keystone / Q5 RELAY / A3 Cascade / A4+L4 / Q4 SIEVE). 상세는 직전 세션 [unselected.md](../summary/2026-06-04-mode2-cosmos3-mot-edge-serving/unselected.md) (본 세션 deep 번들에도 동일 계승).

---

## Section 1 — R72 Document-Depth Discipline 신설 (Rule 41)

### 배경
- 직전 2026-06-04 cosmos3 세션 산출물의 tier 문서가 **~80줄**에 그쳐 (a) R52.2 source-code anchor, (b) R70 완료 판정, (c) pseudo-code, (d) preliminary study 가 누락 → 구현 가능성 검증 불가. 사용자 피드백이 직접 트리거.

### R72 내용 요약
- **R72.1 길이상한 금지**: tier 문서에 줄 수 상한을 두지 않음 (depth-first). 직전 세션의 ~80줄 압축이 anti-pattern.
- **R72.2 per-document 전담 agent**: 문서 1편당 전담 agent 1 — 한 agent 가 여러 문서를 얕게 쓰는 것을 금지.
- **R72.3 문서 agent 의 source-code 검증 직접 수행**: 각 전담 agent 가 인용 코드(repo clone)를 **직접 열어** anchor(파일#L줄)를 확정. 2차 검증으로 미루지 않음.
- **R72.4 depth-gate**: 작성 후 게이트가 (anchor 존재 / github 링크 / pseudo-code / 완료 판정 / preliminary) 항목을 PASS/FAIL 판정 — FAIL 시 fix agent 재작성.
- **R72.5 staging 완화**: deep 문서 작성 중 staging 제약(분할 저장 등)을 완화 적용.
- **파일**: `skill.md` + `references/document-depth-discipline.md` 신설.

---

## Section 2 — Setup 로그 (setup agent 2)

### Repo shallow clone (4종, source-of-truth)
| Repo | SHA | 용도 |
|---|---|---|
| nvidia/cosmos | `7f5797f` | Cosmos3 모델 본체 (cached_kv / dual hook / CFG call) |
| cosmos-framework | `003d66d` | 학습/serving 프레임 (policy server, guidance) |
| vllm-omni (vllm-project) | `95d56cf` | diffusion serving 스택 (quant 메뉴 / stream-staging greenfield) |
| vllm | `063ce98` | upstream (qwen3_omni thinker tower-prefix quant routing 선례) |

- **repo-map 핵심**: cosmos3 cached_kv = `Cosmos3VFMTransformer.cached_kv` (per-layer (K,V) list), CFG 2-call 경로, dual forward hook(`_forward_local`/`_forward_sp`), vllm-omni quant 메뉴(int8/mxfp8/mxfp4/mxfp4_dualscale/inc/auto-round/gguf), qwen3_omni thinker tower-prefix quant routing.
- **step0-refresh 결과**: **판정 변동 0** (Top-6·tier 불변). 신규 must-cite **2건**:
  - [arXiv:2602.02110](https://arxiv.org/abs/2602.02110) — World Model Quantization → **Q2(ANCHOR)** 보강 인용.
  - [arXiv:2511.11418](https://arxiv.org/abs/2511.11418) — Optimal-Transport Quant for Flow Matching → **Q2** 보강 인용.

---

## Section 3 — Phase 1-deep 로그 (per-document 전담 6 agent, R72.2)

| 문서 | 전담 agent | 최종 줄 수 | anchor 수 |
|---|---|---|---|
| tier1/01-anchor (Q2) | deep-spec-1 | 388 | 8 |
| tier1/02-drift (Q1) | deep-spec-2 | 404 | 8 |
| tier1/03-tideloom (S1) | deep-spec-3 | 404 | 9 |
| tier2/01-duoclock (S2) | deep-spec-4 | 363 | 7 |
| tier2/02-keelkv (L3) | deep-spec-5 | 380 | 8 |
| tier2/03-sidepool (S4) | deep-spec-6 | 372 | 8 |
| **합계** | 6 | **2,311** | **48** |

- 각 agent 가 R72.3 에 따라 repo clone 을 직접 열고 anchor(파일#L줄) 를 본문에 삽입.

---

## Section 4 — depth-gate (R72.4) — 규칙 효과 입증 사례

### 1차 gate: 2 PASS / 4 FAIL
- **FAIL 원인 (4 문서 공통)**: 본문에 **github 링크 0건** — anchor 를 `파일#L줄` 텍스트로만 적고 clickable repo URL 미부착 (R72.4 항목 누락).
- 이 결함은 직전 세션이 가졌던 "anchor 부재" 문제와 동질 — **R72.4 depth-gate 가 실제로 결함을 잡아낸 첫 사례** (규칙 도입 효과 입증).

### fix: 4 fix agent → 2차 gate 전원 PASS
- 4 fix agent 가 각 FAIL 문서에 github blob URL(SHA-pinned) 부착 → 2차 gate **전원 PASS**.

---

## Section 5 — Phase 2 검증 리뷰 (검증 agent 1) + Phase 1' fix (3 agent)

### 검증 결과: anchor 47/48 정확 (98%)
- 48 anchor 중 47 이 실제 소스 라인과 일치. 1건은 fix-now 대상으로 회수.

### fix-now 5건 (상세)
1. **K_AR 16×16 재산정** — patch 16×16, policy ~1,050 tok, per-layer BF16 ~4.3MB / INT4 ~1.08MB (구 산정식이 patch/tok 수 부정확) → Q2/L3 두 문서 동기화.
2. **policy CFG ON 정정** — 구 문서가 일부 맥락에서 CFG OFF 로 기술 → policy(DROID) 는 guidance scale 3.0 + CFG-parallelism = **CFG ON / cached K_AR 2벌(cond/uncond)** 로 정정 (tech report §4.2.5 L2007 + `action_policy_robolab_server.md` L22/L53). guidance_scale=1.0 은 forward/inverse-dynamics·padding pass 용.
3. **ANCHOR bound 36-layer 누적 항 추가** — 구 bound 식이 단일-layer 기준 optimistic-invalid → **36 gen layer 누적** 항을 명시(layer-recursion 반영).
4. **KEELKV 출처** — L2-fit 수치(3MB / per-layer fit)의 산정 출처를 GQA 8-head/36-layer/1,050-tok 으로 명기.
5. **코드상 K_AR text-only 발견** — serving 코드 경로상 cached K_AR 이 **text-only** 임을 발견 (관측은 GEN tower 주입) → Q2/ANCHOR 에 "serving K_AR text-only" 와 "관측 GEN-tower 주입" **두 시나리오 병기** 로 정직화.

### Phase 1' fix (3 agent)
- 3 fix agent 가 5건을 해당 문서(anchor / drift / keelkv 등)에 반영 → **최종 gate 전원 PASS** (363-404줄 유지).

---

## Section 6 — 신규 코드-수준 발견 10건

| # | 발견 | anchor | 영향 문서 |
|---|---|---|---|
| 1 | post-RoPE K 캐시 (Hadamard 를 RoPE 뒤에) | cosmos `cached_kv` L548 | Q2 ANCHOR |
| 2 | dual forward hook (FSM 삽입점) | `_forward_local` L648 + `_forward_sp` L675 | S1 TIDELOOM |
| 3 | qwen3_omni thinker tower-prefix quant routing 선례 | vllm L1150-1151 | Q1 DRIFT |
| 4 | quant 메뉴 (int8/mxfp8/mxfp4/mxfp4_dualscale/inc/auto-round/gguf) | vllm-omni quant config | Q1 DRIFT |
| 5 | diffusion 스택 stream/staging **greenfield** 입증 (구현 부재) | vllm-omni diffusion path | S1 TIDELOOM |
| 6 | CFG 2-call + policy CFG ON 정정 | cosmos L1497/L1511 | Q2 ANCHOR |
| 7 | K_AR 재산정 (16×16 patch, ~1,050 tok, BF16 4.3MB / INT4 1.08MB per-layer) | GQA 8-head / 36 gen layer | Q2 ANCHOR / L3 KEELKV |
| 8 | serving K_AR **text-only** (관측은 GEN tower 주입, 두 시나리오 병기) | cosmos serving path | Q2 ANCHOR |
| 9 | ANCHOR bound **36-layer 누적 항** (구 식 optimistic-invalid) | bound 유도 | Q2 ANCHOR |
| 10 | Orin EMC = BPMP debugfs experimental → nvpmodel fallback / DLA 비호환 op(CausalConv3d·RMS_norm·AttentionBlock) + NVDLA v2.0 정정 | Jetson sysfs / trtexec | S2 DUOCLOCK / S4 SIDEPOOL |

---

## Section 7 — 산출물 링크

- **신규 summary 번들**: [summary/2026-06-05-mode1-cosmos3-edge-serving-deep/README.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/README.md)
  - Tier-1: [01-anchor.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier1/01-anchor.md) (388) · [02-drift.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier1/02-drift.md) (404) · [03-tideloom.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier1/03-tideloom.md) (404)
  - Tier-2: [01-duoclock.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier2/01-duoclock.md) (363) · [02-keelkv.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier2/02-keelkv.md) (380) · [03-sidepool.md](../summary/2026-06-05-mode1-cosmos3-edge-serving-deep/tier2/03-sidepool.md) (372)
- **직전 세션(superseded)**: [summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md](../summary/2026-06-04-mode2-cosmos3-mot-edge-serving/README.md)
- **신규 must-cite**: [arXiv:2602.02110](https://arxiv.org/abs/2602.02110) (World Model Quantization) · [arXiv:2511.11418](https://arxiv.org/abs/2511.11418) (OT Quant for Flow Matching)
