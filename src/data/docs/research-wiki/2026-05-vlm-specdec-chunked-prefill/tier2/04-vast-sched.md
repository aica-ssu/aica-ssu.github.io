# Heartbeat Pulse : Visual-Activity-Tied Speculative Token Scheduler with Time-Since-Region-Completion Heartbeat (R19-α)

> Tier-2 candidate #1 (Independent) — venue target: DAC / DATE / ICCAD / CAL (~300 LOC, 2-mechanism, single workload, training-free)

## 1. Research Questions (R28.2.5)

- **RQ1** — Cross-finding distillation 의 visual-activity heartbeat (= `time_since_last_completed_visual_region` 의 EWMA) 와 EAGLE-2 의 acceptance rate 사이에 Pearson |ρ| ≥ 0.4 의 monotonic 관계가 존재하는가?
- **RQ2** — `num_speculative_tokens` 의 heartbeat-piecewise-linear (PWL) function (heartbeat > 50 ms → 5 / 20–50 ms → 3 / < 20 ms → 2) 이 fixed `num_speculative_tokens = 4` baseline 대비 wasted draft compute ≥ −30% + acceptance rate ≥ +5pp 달성하는가?
- **RQ3** — 본 idea 의 ≤300 LOC + 3 file modify 의 implementation footprint 가 vLLM b6553be1 base 의 다른 Tier-1 idea (B12', A1') 와 propose() signature 공유 시 single PR 묶음 가능한가?

## 2. Two-Sentence Pitch (R32.4 A9)

VLM speculative decoding currently sets a fixed `num_speculative_tokens` regardless of visual context state because vLLM's spec config is initialization-time static, leading to wasted draft compute when the prompt is mid-visual-region with low acceptance probability. We tie `num_speculative_tokens` to a **visual-activity heartbeat metric — the EWMA of time since last completed visual region** — using a piecewise-linear policy, which works because (a) SpecVLM's empirics show acceptance naturally drops with incomplete visual grounding, and (b) the heartbeat metric is O(1)-computable from existing `mm_positions` without any new attention export.

## 3. 가설 + Falsification (R33.3)

### 3.1 가설
- **H1**: Visual heartbeat ↔ acceptance rate Pearson correlation |ρ| ≥ 0.4 on MMMU sample (n=200, 95% CI), monotonically increasing.
- **H2**: PWL `f(heartbeat) → num_speculative_tokens` 이 fixed-4 baseline 대비 wasted draft compute −30%, acceptance +5pp, energy −10%, lossless.

### 3.2 Falsification
- **F1**: |ρ| < 0.3 → heartbeat 가 acceptance 의 predictor 아님 → idea drop.
- **F2**: PWL function control overhead > 1% step latency → 무효.
- **F3**: Energy gain < 5% on MMMU 평균 → fail (gain margin 부족).
- **F4**: 모든 workload 에서 fixed `num_speculative_tokens=4` 이 optimal (heartbeat-aware 가 wasted compute 줄였지만 acceptance 도 같이 떨어짐) → narrow scope reposition.
- **F5**: `propose()` signature 변경이 B12' Entropy-γ idea 의 signature 와 conflict → integration sprint 추가.

## 4. Workload Evidence (R17.1)

- **SpecVLM [arXiv:2509.11815](https://arxiv.org/abs/2509.11815)** — Visual context completion ↔ acceptance rate; incomplete region 시 acceptance ↓ 직접 evidence.
- **VoltanaLLM [arXiv:2509.04827](https://arxiv.org/abs/2509.04827)** — Feedback-driven frequency control (SGLang, 36.3% energy saving) — feedback signal pattern 의 precedent.
- **SparseVILA [arXiv:2510.17777](https://arxiv.org/abs/2510.17777)** — Visual token sparsity time-varying — heartbeat metric 의 temporal lever evidence.
- **AHASD [arXiv:2604.25326](https://arxiv.org/abs/2604.25326) (DAC 2026)** — Entropy-history pre-verification 의 NPU+PIM-specific 사례, Tier-2 venue (DAC) 의 positioning reference.

## 5. 기준 코드베이스 (Baseline Source) (R52.1)

- **Framework**: [vLLM commit b6553be1](https://github.com/vllm-project/vllm/tree/b6553be1) (2026-05-fetch).
- **Model**: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) primary.
- **Dependencies**: `transformers==4.57.x`, `torch==2.6.0`, `cuda==12.4`.
- **Hardware target**: RTX Pro 6000 96GB primary, RTX 5090 32GB secondary (Tier-2 single-config workstation).
- **Clone Spec (R30.6.1)**:
  - Repo URL: `https://github.com/vllm-project/vllm.git`
  - Commit hash: `b6553be1`
  - Tag: untagged
  - Fetched: 2026-05-26

## 6. 동작 원리 (Mechanism)

### M1 Visual-Activity Heartbeat Metric

#### 6.M1.1 동작 원리 (R20-α 4 요소)
- **① 추가 Scheme**: `vllm/v1/spec_decode/metrics.py:L17 SpecDecodingStats` 에 신규 field `time_since_last_visual_region: float` (ms unit) + EWMA buffer.
- **② 해결 문제**: SpecVLM 의 incomplete visual context → acceptance drop evidence — 현재 vLLM 의 spec config 는 visual state 미인지.
- **③ 동작 원리 step-by-step**:
  - Step 1: 신규 field `last_visual_region_complete_ts: float` (timestamp) 도입 (per request).
  - Step 2: Scheduler 의 chunk emit 시 만약 chunk 가 visual region 의 마지막 token 포함하면 timestamp 업데이트.
  - Step 3: 매 spec decode step 마다 `heartbeat = current_ts - last_visual_region_complete_ts` 계산.
  - Step 4: EWMA `h_ewma = α · h + (1-α) · h_ewma_prev`, α=0.2.
  - Step 5: `SpecDecodingStats.get_heartbeat() -> float` API 노출.
- **④ 차별화**:
  - vs VoltanaLLM: P/D feedback frequency, visual-specific heartbeat 부재.
  - vs SpecVLM: acceptance rate feedback only, time-since-region 미사용.
  - vs AHASD: entropy-history (token-level), visual-region temporal 미고려.

#### 6.M1.2 기대 효과
- **Primary**: Reliability — heartbeat 가 acceptance 의 predictor (|ρ| ≥ 0.4 invariant).
- **Secondary**: Implementation — O(1) cost per step, no new attention export.
- **단독 미보장**: Magnitude — Tier-2 scope (300 LOC).

#### 6.M1.3 구현 변경점 (R52.2, R68)

| File path | Class · Function · Line region | As-is | To-be | 변경 type | GitHub Link |
| --- | --- | --- | --- | --- | --- |
| `vllm/v1/spec_decode/metrics.py` | `class SpecDecodingStats` L17, `draft_acceptance_rate` L77, `mean_acceptance_length` L81 | aggregate only | `last_visual_region_complete_ts: float`, `heartbeat_ewma: float`, `get_heartbeat() -> float` | Modify | [vllm/v1/spec_decode/metrics.py#L17](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py#L17) |
| `vllm/v1/core/sched/output.py` | `class NewRequestData` L22, `mm_positions: list[PlaceholderRange]` L28 | read-only existing | scheduler 가 mm_positions 의 region end 추적 시 read | Read-only | [vllm/v1/core/sched/output.py#L28](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/output.py#L28) |
| `vllm/v1/core/sched/scheduler.py` | `class Scheduler.schedule` L158–L410 | chunk emit | chunk end 시점 region complete check + `SpecDecodingStats.update_heartbeat()` 호출 | Modify | [vllm/v1/core/sched/scheduler.py#L158-L410](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py#L158-L410) |

**R52.3 verification trace**:
- `vllm/v1/spec_decode/metrics.py:L17 SpecDecodingStats` 실재 ([github.com — metrics.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py), commit `b6553be1`). [✅]
- `vllm/v1/core/sched/output.py:L28 mm_positions` 실재 ([github.com — output.py#L28](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/output.py#L28), commit `b6553be1`). [✅]
- `vllm/v1/core/sched/scheduler.py:L158 schedule()` 실재 (commit `b6553be1`, [github.com — scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)). [✅]

#### 6.M1.4 검증 시나리오
- **Unit test** (5 min): 목적 — EWMA 계산 정합 / Input — 합성 heartbeat 시계열 (10/20/30 ms) / Expected — EWMA α=0.2 분석값과 일치 (rtol 1e-6) / 검증 metric — assert / 실행 시간 — 5 min / 실패 시 액션 — α coefficient 정정.
- **Mechanism-isolated test** (2h): 목적 — heartbeat ↔ acceptance correlation / Input — Qwen2-VL-7B + MMMU 200 sample / Expected — Pearson |ρ| ≥ 0.4 / 검증 metric — scipy.stats.pearsonr / 실행 시간 — 2h / 실패 시 액션 — F1 violation 시 idea drop.

### M2 Heartbeat-PWL Dynamic `num_speculative_tokens` Clip

#### 6.M2.1 동작 원리
- **① 추가 Scheme**: `vllm/v1/spec_decode/eagle.py:L78 propose()` signature 에 `num_speculative_tokens: int` runtime argument 추가 (B12' 와 공유 — single PR 묶음).
- **② 해결 문제**: vLLM 의 `num_speculative_tokens` 가 init-time static (`EagleProposer.__init__` L28). Fixed 4 인 환경에서 incomplete visual region (low acceptance) 시 wasted compute.
- **③ 동작 원리 step-by-step**:
  - Step 1: `Scheduler.schedule()` 의 chunk emit 직후 `h_ewma = SpecDecodingStats.get_heartbeat()` poll.
  - Step 2: PWL function:
    - `heartbeat > 50 ms` → `num_spec = 5` (region complete 후 충분 시간, high acceptance regime)
    - `20 ms ≤ heartbeat ≤ 50 ms` → `num_spec = 3`
    - `heartbeat < 20 ms` → `num_spec = 2` (region mid, low acceptance regime)
  - Step 3: `propose(num_speculative_tokens=num_spec, ...)` 호출 — runtime reread.
  - Step 4: `EagleProposer.propose()` 의 internal loop `for _ in range(self.num_speculative_tokens - 1)` (L208) 가 runtime arg 사용 (B12' 와 동일 path).
- **④ 차별화**:
  - vs static config: visual-aware adaptation 부재.
  - vs B12' Entropy-Gated γ: token entropy + closed-form, 본 idea 는 visual heartbeat (orthogonal signal) — combine 가능.
  - vs A6' Tortoise & Hare: CPU draft worker placement, 본 idea 는 spec count adaptation.

#### 6.M2.2 기대 효과
- **Primary**: Performance — wasted draft compute −35% (vs fixed 4).
- **Secondary**: Performance — acceptance rate +6pp; Energy — J/token −11%.
- **단독 미보장**: Lossless (modified rejection sampling 보존).

#### 6.M2.3 구현 변경점 (R52.2)

| File path | Class · Function · Line region | As-is | To-be | 변경 type | GitHub Link |
| --- | --- | --- | --- | --- | --- |
| `vllm/v1/spec_decode/eagle.py` | `class EagleProposer.propose` L78, `num_speculative_tokens` L44 (init capture) | init-time static | propose() signature `num_speculative_tokens: int` 추가, runtime reread (B12' 와 공유 — single PR) | Modify (signature) | [vllm/v1/spec_decode/eagle.py#L78-L130](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py#L78-L130) |
| `vllm/v1/core/sched/scheduler.py` | `class Scheduler.schedule` L158, `num_spec_tokens` L140–L145 | init-time | PWL function call `_pwl_num_spec(h_ewma) -> int` | Modify | [vllm/v1/core/sched/scheduler.py#L140-L145](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py#L140-L145) |
| `vllm/v1/spec_decode/heartbeat_pwl.py` (NEW) | new module | n/a | `pwl_num_spec(heartbeat_ms: float) -> int` API | Add | [vllm/v1/spec_decode/](https://github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode) |

**R52.3 verification trace**:
- `vllm/v1/spec_decode/eagle.py:L78 propose()` 실재 (commit `b6553be1`, [github.com — eagle.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py)). [✅]
- `vllm/v1/core/sched/scheduler.py:L140-L145 num_spec_tokens` 실재 (commit `b6553be1`, [github.com — scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)). [✅]

#### 6.M2.4 검증 시나리오
- **Unit test** (3 min): 목적 — PWL function 의 boundary / Input — heartbeat ∈ {10, 19, 20, 50, 51, 100} / Expected — {2, 2, 3, 3, 5, 5} (boundary inclusive) / 검증 metric — assert / 실행 시간 — 3 min / 실패 시 액션 — boundary inclusive/exclusive 정정.
- **Mechanism-isolated test** (3h): 목적 — Wasted draft compute reduction / Input — Qwen2-VL-7B + MMMU 200 sample / Expected — wasted draft tokens (rejected) −35% / 검증 metric — `SpecDecodingStats.num_draft_tokens - num_accepted_tokens` aggregate / 실행 시간 — 3h / 실패 시 액션 — PWL boundary tune (e.g., 30/60).

## 7. 전체 평가 시나리오 (E2E) (R52.4-C)

- **Synthetic Tier-A** (1h): 50 sample 합성 heartbeat → PWL boundary 정합.
- **Tier-B** (2h): MMMU dev 100 sample 의 fixed-4 vs heartbeat-PWL 비교.
- **Tier-C real benchmark** (12h): MMMU dev 200 / MMBench 200 / DocVQA 200 / TextVQA 200.
- **실험 환경**: RTX Pro 6000 96GB single workstation.
- **모델**: Qwen2-VL-7B primary (Tier-2 single-config; secondary 평가 생략 — DAC 6p scope).
- **Metric**: acceptance rate, wasted draft compute (rejected count), J/token (NVML), MMMU score (lossless verify).
- **실행 시간**: 총 ~15h.
- **실패 시 액션**: F1 ρ < 0.3 시 idea drop; F3 energy < 5% 시 scope narrow (high-heartbeat regime only).

## 8. 실험 설계 7-요소 (R27-β)

1. **Hardware**: RTX Pro 6000 96GB (Tier-2 single-config workstation).
2. **Model**: Qwen2-VL-7B-Instruct (primary). LLaVA-1.5-7B (robustness, optional).
3. **Dataset**: MMMU dev (primary), MMBench / DocVQA / TextVQA.
4. **Tools**: vLLM b6553be1, NVML energy counter, scipy.stats.
5. **Ablation**: (a) fixed-4 baseline / (b) heartbeat metric only (no PWL) / (c) full (M1+M2).
6. **Implementation Schedule** (8-week, Tier-2 short scope):

| Week | 작업 |
| --- | --- |
| 1 | Heartbeat metric instrument (SpecDecodingStats field 추가) |
| 2 | Pilot correlation 측정 (MMMU 200) |
| 3 | PWL function + propose() signature 변경 |
| 4 | B12' signature merge (single PR) |
| 5 | MMMU/MMBench evaluation |
| 6 | DocVQA/TextVQA |
| 7 | Ablation (3 config) + energy 측정 |
| 8 | Paper writing (DAC/DATE 6p) |

7. **Preliminary Metrics**: NVML `nvmlDeviceGetTotalEnergyConsumption`, `SpecDecodingStats.num_draft_tokens`, MMMU score (95% CI).

## 9. 예상 효과 표 (R55.2 5-axis)

| Axis | 지표 | Baseline (fixed num_spec=4) | 본 idea | 개선 | 조건 / 근거 |
| --- | --- | --- | --- | --- | --- |
| Performance | Wasted draft compute (rejected count) | 100% | 65% | **−35%** | EWMA-clip num_spec |
| Performance | Acceptance rate | 38% | 44% | +6pp | visual-grounded regime only |
| Energy | J/token | 0.62 | 0.55 | **−11%** | fewer draft compute |
| Accuracy | MMMU dev score | 50.2 | 50.2 ± 0.2 | **lossless** | modified rejection 보존 |
| Memory | Peak VRAM | 42 GB | 42 GB | 0% | no new buffer |

## 10. 관련 연구 + 차별화

- Closest competitor: **AHASD [arXiv:2604.25326](https://arxiv.org/abs/2604.25326) (DAC 2026)** — Entropy-history pre-verification (NPU+PIM specific).
- 차별화 axis: 본 idea 의 **visual-region temporal heartbeat** 가 AHASD 의 entropy history 와 orthogonal — VLM-specific lever. AHASD NPU+PIM, 본 idea standard CUDA GPU.
- Baseline list (Tier-2, 4 편):
  1. [SpecVLM arXiv:2509.11815](https://arxiv.org/abs/2509.11815)
  2. [VoltanaLLM arXiv:2509.04827](https://arxiv.org/abs/2509.04827)
  3. [SparseVILA arXiv:2510.17777](https://arxiv.org/abs/2510.17777)
  4. [AHASD arXiv:2604.25326](https://arxiv.org/abs/2604.25326) DAC 2026

## 11. Implementation Consistency (R52.5)

- R47.2 application-level only — vLLM source modify (~300 LoC), no driver/kernel/firmware.
- Simulator path 잔재 0.

## 12. Reproducibility Checklist (R30.6.4)

- **Clone Spec**: vLLM `b6553be1` only.
- **Environment**: Ubuntu 22.04, CUDA 12.4, Python 3.11, PyTorch 2.6.0.
- **Build Sequence**: `git clone https://github.com/vllm-project/vllm.git && cd vllm && git checkout b6553be1 && pip install -e .` → patch apply.
- **Patch List**: `vllm/v1/spec_decode/metrics.py` (heartbeat field), `vllm/v1/spec_decode/heartbeat_pwl.py` (NEW), `vllm/v1/spec_decode/eagle.py` (propose signature, B12'/A6'/A1' 와 공유), `vllm/v1/core/sched/scheduler.py` (PWL call).
- **Smoke Test**: `vllm serve Qwen/Qwen2-VL-7B-Instruct --speculative-config '{"method":"eagle","num_speculative_tokens":4,"heartbeat_pwl":true}'` → MMMU 10 sample, wasted draft count 측정.

## 13. Scoring 및 이유 (R67) — 5 reviewer × 4 sub-axis

| Reviewer | Sub-axis 1 (Mech/Source) | Sub-axis 2 (Comb/Kernel) | Sub-axis 3 (Hyp/Framework) | Sub-axis 4 (D2/D6) | 평균 |
| --- | --- | --- | --- | --- | --- |
| novelty | 6 | 7 | 7 | 8 | 7.00 |
| differentiation | 7 | 7 | 7 | 8 | 7.25 |
| impact | 6 | 7 | 7 | 7 | 6.75 |
| ai-impl | 8 | 8 | 8 | 8 | 8.00 |
| arch-sys | 8 | 7 | 8 | 7 | 7.50 |

### ★ 전체 최고 sub-axis: **ai-impl Source-level (8/10)** + Kernel/Framework/D6 (8/10)
~300 LOC, 3 file modify, B12' Entropy-Gated γ 와 `propose()` signature 공유 — single PR 묶음 가능. ~1 week implementation budget. Tier-2 implementation efficiency 우수.

### ▼ 전체 최저 sub-axis: **novelty Mechanism (6/10)** + impact Magnitude (6/10)
Visual heartbeat 가 새 lever 이나 mechanism 자체는 EWMA + PWL function 으로 narrow. Tier-2 DAC/DATE 6p 적합. Magnitude (energy −11%, wasted compute −35%) 가 Tier-1 venue 진입 불가 수준.

## 14. R14.4 Implementation-Priority Decision Tree

- **Preliminary study (Week 1–2)**: Heartbeat metric instrument + pilot correlation.
  - 측정: Pearson |ρ| (heartbeat vs acceptance) on MMMU 200.
  - Pass (|ρ| ≥ 0.4): 다음 stage 진입.
  - Below (|ρ| ∈ [0.3, 0.4]): scope narrow (high-heartbeat regime only).
  - Critical (|ρ| < 0.3): F1 violation — idea drop.

- **Minimum viable prototype (Week 3–4)**: PWL function + propose() signature.
  - 측정: wasted draft −30%, acceptance +5pp.
  - **① Outperform** (full target hit): Week 5+ full eval.
  - **② Pass**: PWL boundary tune (30/60 ms).
  - **③ Below** (wasted draft −20%): boundary 추가 fine-tune.
  - **④ Critical**: F2 (overhead > 1%) — fixed num_spec=4 fallback.

- **Full evaluation (Week 5–8)**: 4 benchmark × seed 5.
  - ① Outperform: DAC submission.
  - ② Pass: DATE / ICCAD submission.
  - ③ Below: CAL letter.
  - ④ Critical: drop.

## 15. Inter-idea Dependency

- **Shared infrastructure**:
  - vLLM commit `b6553be1`.
  - `vllm/v1/spec_decode/eagle.py:L78 propose()` signature 변경 path (B12', A1', A3' 와 공유) — **single PR 묶음 권고**.
  - `vllm/v1/spec_decode/metrics.py:L17 SpecDecodingStats` (B9' KL-Bounded distillation 의 EWMA buffer 와 sibling field).
- **Free-combine partner**:
  - B12' Entropy-Gated γ: heartbeat + token entropy → joint signal for `num_speculative_tokens` (Phase 1' 'VAST-γ' merge candidate).
  - B9' KL-Bounded Distillation: training-only orthogonal — KL-distilled draft 가 dynamic num_spec 환경의 robustness 강화.
- **Provenance**: Phase 2 reviewer cross-finding (Mosaic M3 DVFS drop + Origami sequential adaptive + Idea 3 acceptance feedback) 의 정합 distillation 의 independent Tier-2.

## 16. Stakeholder Rotation 7-row (R32.7 A7)

| Stakeholder | Concern | 답변 |
| --- | --- | --- |
| End user | TPOT 영향? | 평균 −5–10% (acceptance gain, num_spec adapt) |
| Developer | 통합 난이도? | 3 file × <300 LoC, B12'/A1' 와 single PR 묶음 |
| Theorist | Heartbeat 의 mathematical foundation? | SpecVLM 의 acceptance ↔ visual context evidence 만, formal proof 미주장 (Tier-2 적정) |
| Adversary | Heartbeat manipulation? | Heartbeat 는 mm_positions 기반 — adversarial prompt 의 region structure 만으로 manipulate 가능, 단 acceptance 하락이 자연 penalty |
| Ethicist | Lossless 보장? | modified rejection sampling 보존 |
| Regulator | Energy accounting? | NVML counter logged |
| Operator | Deploy cost? | ~1 week 단일 PR + B12' 와 묶음 |

## 17. Boundary Probing 5-axis (R32.6 A5)

| Axis | 경계 시나리오 | 본 idea 응답 |
| --- | --- | --- |
| Distributional | Text-only prompt (no visual) | heartbeat → ∞ → num_spec=5 (high-acceptance regime), fixed-4 와 동등 또는 우월 |
| Scale | 11B/13B target | PWL boundary 동일, scaling-out 자연 (signature share) |
| Adversarial | Region boundary manipulation | acceptance 하락이 자연 penalty — idea robust |
| Compositional | Multi-image (5+) batch | per-image heartbeat 추적, last region timestamp update |
| Temporal | Video stream | frame 별 region 정의 따라 heartbeat (future work — frame-level region) |

## 18. Self-Check (R52 + R53 + R54 + R28.2 + R68)

- [x] R52.1 Baseline Source 5필드 ✅
- [x] R52.2 7-column 표 ≥ 3 row per mechanism ✅ (M1: 3 / M2: 3)
- [x] R52.3 verification trace [✅] mark + commit hash `b6553be1` ✅
- [x] R52.4 synthetic 3-tier (A-B-C) ✅
- [x] R52.5 Implementation vs Simulator 잔재 0 ✅
- [x] R53 4-section inline ✅
- [x] R54.1–6 Final verification ✅
- [x] R68 GitHub link main branch + line anchor — 5/6 line-anchored (83% ≥ 50%) ✅
- [x] R28.2.5 첫 ## heading = "Research Questions" ✅
- [x] R28.2.6 raw arxiv ID 0 ✅
- [x] R10-α bullet 의무 ✅
- [x] R19-α vendor-neutral title ("Visual-Activity-Tied", "Time-Since-Region-Completion") ✅
