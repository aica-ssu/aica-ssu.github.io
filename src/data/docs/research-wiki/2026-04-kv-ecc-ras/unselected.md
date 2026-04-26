# 미선정 아이디어 모음 — 2026-04-25 KV Cache Memory ECC + RAS (Session #2)

> [← Session Overview](/research-wiki/2026-04/kv-ecc-ras/README.md)

3 expert × 7 idea = 21 → 최종 6 (Tier-1 3 + Tier-2 3) → 미선정 15. 본 파일은 각 미선정 idea 의 사유 + 재방문 조건. 6 self-scoop pair 는 모두 본 파일의 merge note 와 일치.

---

## A1: Sentinel (system-robustness T1)

- **Score**: Phase 2 평균 7.20 (Nov 7.2 / Diff 6.5 / Imp 7.5 / Feas 7.5)
- **최종 판정**: MERGED → Tier-1 [OAEP-KV](/research-wiki/2026-04/kv-ecc-ras/tier1/01-oaep-kv.md) 의 Mech M2 (DRAM-row-stripe alignment + sub-page retirement)
- **연구 GAP**: outlier 채널을 DRAM row 의 contiguous stripe 에 align → sub-page retire 시 outlier-stripe 만 보호
- **미선정 사유 상세**:
  - **C2 self-scoop pair (Sentinel ↔ OAEP-KV)**: 두 idea 의 outlier-channel ECC 축이 거의 동일. Sentinel 만의 unique 부분 = DRAM-row-stripe alignment + sub-page retire — OAEP-KV merged 측의 Mech M2 로 흡수.
  - 단독 Tier-1 제출 시 OAEP-KV 와 동일 venue (DSN/HPCA/MICRO) 에서 self-collision.
  - DRAM-mapping vendor-specific 한계 (channel interleave / rank striping) 는 OAEP-KV 의 ablation section 에서 명시.
- **재방문 조건**: DRAM-row stripe alignment 가 vendor 무관 universal 임이 입증되면 Sentinel 만의 single-mech sub-paper 가능. Tier-2 IEEE TCAD short 형태로 분리.

---

## A2: Lattice (system-robustness T3)

- **Score**: Phase 2 평균 8.50 (Nov 8.5 / Diff 8.0 / Imp 8.5 / Feas 8.0) — **단독 score 가장 높음**
- **최종 판정**: MERGED → Tier-1 [BlockShard](/research-wiki/2026-04/kv-ecc-ras/tier1/02-blockshard.md) 의 Mech M2 (vLLM RFC #19329-aware recompute reschedule)
- **연구 GAP**: vLLM 16-token block 을 ECC + retirement granularity unit 으로 재정의
- **미선정 사유 상세**:
  - **C3 self-scoop pair (Lattice ↔ BlockShard)**: 동일 핵심 insight (vLLM block ≪ OS page). Lattice = vLLM RFC #19329 graceful error 패턴 재사용 (R45-strict). BlockShard = Linux soft-offline ABI 확장 (R45-borderline kernel patch).
  - **Merge 결정**: Lattice 의 R45-clean path (vLLM-only) 를 메인으로, BlockShard 의 ABI 제안은 position section 으로 보강.
  - 단독 paper 분리 시 **두 paper 가 사실상 같은 venue (ASPLOS/OSDI/DSN) 에서 self-collision** — reviewer 가 self-plagiarism 으로 평가 가능.
- **재방문 조건**: SGLang RadixAttention 도 동일 block-level fault isolation 추가 시 cross-system generality paper 로 분리 가능. ASPLOS 2028.

---

## A3: Hourglass (system-robustness T2)

- **Score**: Phase 2 평균 7.80 (Nov 7.8 / Diff 7.0 / Imp 7.5)
- **최종 판정**: DROP — chunked-prefill 보편화로 boundary 좁음
- **연구 GAP**: Phase-asymmetric ECC (prefill DEC / decode SEC-DED + fallback)
- **미선정 사유 상세**:
  - **Boundary issue**: Sarathi-Serve / DistServe / Dynamo 등 chunked-prefill (prefill-decode 분리 부재) 가 dominant 2025-2026 scheduler. Hourglass 의 prefill-decode split 가정이 적용 모델 수가 줄어듦.
  - Mechanism #2 (post-CE migration) 의 capacity overhead 정량 부족.
  - BlockShard 의 Mech M1 (block-level fault classifier) 가 사실상 동일 효과 (background migration via vLLM block table) 를 R45-clean path 로 제공.
- **재방문 조건**: prefill-decode disaggregation (LLM-d, llumnix) 이 다시 dominant 가 되면 Hourglass single-mech IEEE TCAD short 로 부활 가능.

---

## A4: Chameleon (system-robustness T4)

- **Score**: Phase 2 평균 7.50 (Nov 7.5 / Diff 7.0 / Imp 7.0)
- **최종 판정**: DROP — ScaleShield (algorithm T3) 와 self-scoop, GenBFA-only scope 한정적
- **연구 GAP**: Metadata-vs-mantissa unequal ECC + scale replication (anti-GenBFA defense)
- **미선정 사유 상세**:
  - **C1 self-scoop pair (Chameleon ↔ ScaleShield)**: 두 idea 의 metadata-mantissa 차별 보호 mechanism 동일. Chameleon = security framing (DSN/USENIX Security), ScaleShield = RAS framing (MLSys/ASPLOS).
  - 두 idea 모두 단독 Tier-1 venue 분리 시 self-collision. 본 세션은 둘 다 미선정 → OAEP-KV 의 scale-FP16 SEC-DED + RS(15,13) 부분으로 흡수.
  - GenBFA 측 attack scope 만 가지면 단독 security paper novelty 부족 (DeepNcode / NAPER 와 직접 경쟁).
- **재방문 조건**: GenBFA 가 production 위협이 되면 Chameleon single-mech (scale replication) 로 USENIX Security 6p 부활.

---

## A5: ScaleShield (algorithm T3)

- **Score**: Phase 2 평균 7.00 (Nov 7.0 / Diff 7.0 / Imp 8.5)
- **최종 판정**: DROP — Chameleon 과 self-scoop. Scale-byte protection 만 OAEP-KV 의 sub-mechanism 으로 흡수
- **연구 GAP**: Hierarchical ECC anchored at quantization-scale tensors + scale-parity-driven page retirement
- **미선정 사유 상세**:
  - **C1 self-scoop pair**: Chameleon 과 동일.
  - **R45 borderline**: page migration via cudaMemcpyAsync between HBM regions 가 vLLM 가 노출하지 않는 page-granular HBM allocation control 가정. R45 strict 평가 위배.
  - "scale FP16 SEC-DED + RS(15,13)" 의 핵심 idea 만 OAEP-KV 의 scale protection step 으로 흡수.
- **재방문 조건**: HBM ECC vendor (Samsung HBM3e PIM SKU) 가 region-level ECC mode software 노출 시 ScaleShield M2 (page retirement RAS) single-mech MLSys 6p 부활 가능.

---

## A6: RotECC (algorithm T4)

- **Score**: Phase 2 평균 6.50 (Nov 6.5 / Diff 8.0 / Imp 7.0) — Diff 가장 높음
- **최종 판정**: DROP — pipeline-conditional (QuaRot 한정), OAEP-KV 의 limitation note 로 흡수
- **연구 GAP**: Rotation-bounded ECC (Hadamard rotation diagonalize outlier mass) + low-rank residual corrector
- **미선정 사유 상세**:
  - **Pipeline-conditional**: QuaRot/SpinQuant/Quirk 만 — 현재 deployment 의 20-30%, 늘어나지만 niche.
  - OAEP-KV 와 mutual-exclusive (rotated 시 outlier 평탄화 → OAEP-KV gain 사라짐). 두 idea 가 같은 paper 의 두 branch 로 unified 가능 — 그 framing (outlier-spectrum-aware ECC) 은 Phase 3 후속 paper 후보.
  - 단독 ICLR 제출 시 theoretical contribution 강하나 empirical scope 좁음 (LLaMa-2 한정).
- **재방문 조건**: QuaRot/SpinQuant 가 production mainstream 되면 단독 ICLR/HPCA paper 로 부활. OAEP-KV 와 통합한 "outlier-spectrum-aware ECC" framing 은 Phase 3 추가 세션에서 정식.

---

## A7: VLM-MAP 원안 (algorithm T5)

- **Score**: Phase 2 평균 4.50 (Nov 4.5 / Diff 5.0 / Imp 5.5)
- **최종 판정**: MERGED → Tier-2 [VLM-MAP](/research-wiki/2026-04/kv-ecc-ras/tier2/01-vlm-map.md) (legacy-system T4 + B3 와 통합 single-mech)
- **연구 GAP**: Modality-conditioned Hamming code (vision SEC-DED, text parity)
- **미선정 사유 상세**:
  - **C5 self-scoop triple-duplication (VLM-MAP ↔ ModalSplit ↔ ECCLite)**: 셋 모두 modality-aware ECC. VLM-MAP 단독 novelty 가 SGLang vs vLLM 차이만 — research 측면 무의미.
  - Tier-2 측에서 ModalSplit + ECCLite 와 통합한 single-mech VLM-MAP DATE 6p 형태로 흡수.
- **재방문 조건**: 별도 변경 없음 (Tier-2 측에서 통합 publication).

---

## A8: DeepStackECC (algorithm T6)

- **Score**: Phase 2 평균 5.00 (Nov 5.0 / Diff 6.0 / Imp 5.0)
- **최종 판정**: HOLD/DROP — architecture-fragile (Qwen3-VL 한정)
- **연구 GAP**: Multi-tier residual sum self-detection (Qwen3-VL DeepStack architecture redundancy reuse)
- **미선정 사유 상세**:
  - **Architecturally fragile**: Qwen3-VL DeepStack family only — 다른 VLM (LLaVA, InternVL) 은 DeepStack 없음.
  - **Detection-only** (no correction); recompute fallback 미명시.
  - FP16 round-off tolerance (3σ × num-additions) 가 single bit-flip 검출에 너무 loose 가능 — 정밀도 reviewer 우려.
- **재방문 조건**: Qwen3-VL DeepStack 가 VLM mainstream 이 되면 IEEE CAL 4p 로 부활 가능. "architectural-redundancy-as-detection" framework 으로 일반화 (DeepStack + MoE shared experts + residual streams) 시 Tier-1 후속 가능.

---

## A9: KV-Wear-Sketch (algorithm T7)

- **Score**: Phase 2 평균 5.50 (Nov 5.5 / Diff 6.0 / Imp 5.5)
- **최종 판정**: REPOSITIONING — Tier-2 EntropyECC 또는 Tier-1 LayerTier 의 calibration feature 로 흡수
- **연구 GAP**: Quantization clipping rate as online BER monitor
- **미선정 사유 상세**:
  - **Content-induced clipping confounds wear signal** — false positives during long-context outlier-heavy prompts.
  - Independent value modest; works best as add-on signal feeding ErrorOracle's predictor.
  - 단독 Tier-2 제출 시 thin novelty.
- **재방문 조건**: ErrorOracle (legacy-system T3) 가 부활하면 KV-Wear-Sketch 가 그 predictor 의 52-feature 중 하나로 흡수. 또는 LayerTier 의 reliability-zone migration trigger 의 추가 signal.

---

## A10: ErrorOracle (legacy-system T3)

- **Score**: Phase 2 평균 8.00 (Nov 8.0 / Diff 8.0 / Imp 8.0) — **score 두 번째로 높음**
- **최종 판정**: DROP — 16주 budget 초과 (52-feature LightGBM 학습 시간 + ML systems engineering)
- **연구 GAP**: KV-cost-aware ML predictor + KV pre-emption interface
- **미선정 사유 상세**:
  - **16주 budget 초과**: 52-feature LightGBM + synthetic CE log generator + 1M event 학습 + integration + ablation 합산 시 16주 빠듯.
  - **Real log validation 우려**: MareNostrum-class real log access uncertain.
  - 단독 Tier-1 score 매우 강함 — Phase 3 후속 세션에서 6개월+ scope 으로 부활 권장.
- **재방문 조건**: (a) 학생이 ML systems engineering 경험 + (b) 16주 → 24주 budget 확장 + (c) MareNostrum 또는 Alibaba real log access 확보 — 셋 충족 시 HPCA/DSN 단독 Tier-1 제출.

---

## A11: ModalSplit (legacy-system T4)

- **Score**: Phase 2 평균 6.50 (Nov 6.5 / Diff 7.0 / Imp 6.5)
- **최종 판정**: MERGED → Tier-2 [VLM-MAP](/research-wiki/2026-04/kv-ecc-ras/tier2/01-vlm-map.md)
- **연구 GAP**: VLM modality × ECC zone (chip-kill) selection
- **미선정 사유 상세**:
  - **C5 self-scoop triple-duplication**: VLM-MAP, B3 ECCLite 와 핵심 mechanism 동일.
  - **R45 borderline**: BIOS-level chip-kill region selectability 가정 — 일부 SKU 만 지원.
  - Tier-2 측에서 ModalSplit + VLM-MAP + ECCLite 통합 single-mech 으로 흡수.
- **재방문 조건**: 별도 변경 없음 (Tier-2 통합 paper).

---

## A12: B1 FrostFloor (legacy-system)

- **Score**: Phase 2 평균 4.00-5.00 (Nov 4.0 / Diff 5.5 / Imp 4.5)
- **최종 판정**: ACCEPT Tier-2 → [FrostFloor](/research-wiki/2026-04/kv-ecc-ras/tier2/02-frostfloor.md)
- **연구 GAP**: Sub-page bit-error map for edge LLM (Jetson Orin LPDDR5)
- **선정 사유**:
  - **Edge LLM ECC characterization first-to-report**: Jetson Orin LPDDR5 의 mcelog 미지원 → userspace bitmap allocator 가 유일한 software RAS option.
  - DATE 6p scope tight (single mechanism + INT4 only + Jetson only).
- **재방문 조건**: 별도 없음 — 이미 Tier-2 채택.

---

## A13: B2 MigGate (legacy-system)

- **Score**: Phase 2 평균 4.50 (Nov 4.5 / Diff 5.0 / Imp 5.0)
- **최종 판정**: MERGED → Tier-1 [LayerTier](/research-wiki/2026-04/kv-ecc-ras/tier1/03-layertier.md) 의 Mech M2 (calibration-stability lemma)
- **연구 GAP**: 100-prompt calibration set 만으로 layer hot/cold ranking 95%+ Spearman ρ stable
- **미선정 사유 상세**:
  - **Descriptive only**: characterization paper. ρ>0.95 stability 는 useful finding 이나 single-observation paper 는 generative novelty 부족.
  - **C7 self-scoop pair (LayerTier ↔ MigGate)**: MigGate 가 LayerTier 의 calibration-once 정당성의 lemma 제공 → LayerTier 의 Mech M2 로 흡수.
- **재방문 조건**: 별도 없음 (LayerTier 의 lemma 로 통합 publication).

---

## A14: B3 ECCLite (legacy-system)

- **Score**: Phase 2 평균 3.50 (Nov 3.5 / Diff 3.5 / Imp 4.5) — **score 가장 낮음**
- **최종 판정**: MERGED → Tier-2 [VLM-MAP](/research-wiki/2026-04/kv-ecc-ras/tier2/01-vlm-map.md)
- **연구 GAP**: VLM vision-token outlier ratio R²>0.85 correlation
- **미선정 사유 상세**:
  - **C5 triple-duplication**: ModalSplit + VLM-MAP 과 동일 axis. Single observation + lightweight mechanism — 독립 novelty 부족.
  - VLM-MAP 의 step 5 (option) 으로 outlier ratio 임계 mechanism 흡수.
- **재방문 조건**: 별도 없음.

---

## A15: S1 Driftwood (system-robustness)

- **Score**: Phase 2 평균 6.00 (Nov 6.0 / Diff 6.5 / Imp 5.5)
- **최종 판정**: MERGED → Tier-2 [EntropyECC](/research-wiki/2026-04/kv-ecc-ras/tier2/03-entropy-ecc.md) 의 Driftwood section (long-context recency bias)
- **연구 GAP**: Position-dependent bit-criticality drift in long-context Llama-3-8B-W4
- **미선정 사유 상세**:
  - **Pure characterization**: mitigation hook 없음 → Tier-2 only.
  - "recency-weighted ECC" follow-up 이 implied 됐을 뿐 prototype 없음.
  - EntropyECC 의 Mech M1 에 position p × entropy 가중치 형태로 흡수 → mechanism 으로 통합.
- **재방문 조건**: 별도 없음 (EntropyECC 통합 publication).

---

## A16: S2 Honeycomb-Lite (system-robustness)

- **Score**: Phase 2 평균 5.50 (Nov 5.5 / Diff 5.0 / Imp 5.5)
- **최종 판정**: REPOSITIONING → Tier-1 OAEP-KV 의 per-head ablation 으로 흡수
- **연구 GAP**: Per-head ECC strength assignment in MQA/GQA Llama-3-8B (8 KV heads)
- **미선정 사유 상세**:
  - 사실상 Sentinel/OAEP-KV 의 outlier-stripe granularity sub-case (heads = stripes).
  - Marginal independent novelty.
- **재방문 조건**: 별도 없음 (OAEP-KV ablation 으로 통합).

---

## A17: S3 Hourglass-Mini (system-robustness)

- **Score**: Phase 2 평균 5.80 (Nov 5.8 / Diff 6.0 / Imp 5.5)
- **최종 판정**: DROP (Hourglass DROP 와 함께) — VLM-MAP 의 RowPress note 로 축소
- **연구 GAP**: VLM prefill RowPress amplification (image-token long contiguous runs) + single-knob row-open cap
- **미선정 사유 상세**:
  - **Hourglass DROP 와 동반**: chunked-prefill 보편화 issue.
  - Trivially derived from Hourglass + RowPress facts; novelty mostly in empirical measurement.
  - VLM-MAP 의 motivation note 로 흡수.
- **재방문 조건**: VLM serving 이 prefill-decode 분리 (DistServe-style) 되면 Hourglass-Mini 단독 IEEE TCAD short 부활.

---

## R45 / R46 종합 status (15 미선정 idea)

| Idea | R45 status | R46 status | merge / drop reason |
|------|-----------|-----------|---------------------|
| Sentinel | clean | verified | C2 self-scoop (OAEP-KV merge) |
| Lattice | clean (vLLM-only) | verified | C3 self-scoop (BlockShard merge) |
| Hourglass | clean | verified | DROP — chunked-prefill boundary |
| Chameleon | clean | verified | C1 self-scoop (ScaleShield merge); GenBFA scope 좁음 |
| ScaleShield | borderline (HBM region) | verified | C1 self-scoop; HBM region R45 위배 |
| RotECC | clean | verified | DROP — pipeline-conditional |
| VLM-MAP 원안 | clean | verified | C5 triple-duplication (VLM-MAP merge) |
| DeepStackECC | clean | verified | HOLD — Qwen3-VL 한정 |
| KV-Wear-Sketch | clean | verified | repositioning — feature add-on |
| ErrorOracle | clean | verified | DROP — 16주 budget 초과 |
| ModalSplit | borderline (BIOS) | verified | C5 (VLM-MAP merge) |
| B2 MigGate | clean | verified | C7 self-scoop (LayerTier lemma merge) |
| B3 ECCLite | clean | verified | C5 (VLM-MAP merge) |
| S1 Driftwood | clean | verified | C8 (EntropyECC merge) |
| S2 Honeycomb-Lite | clean | verified | repositioning — OAEP-KV ablation |
| S3 Hourglass-Mini | clean | verified | DROP w/ Hourglass |

R46 verified 0 missing. R45 strict 평가 — 4 idea borderline (모두 미선정 또는 merge 시 R45-clean path 로 우회).

**R47 적용 결과 (2026-04-25 신규)**: 15 미선정 idea 의 simulator path 도 R47 규칙 (gem5+vLLM 동시 사용 금지 + R47.2 application-level 우선) 에 준해 재평가됨. 미선정 idea 모두 (a) merged target idea 의 R47 path 에 흡수되거나 (b) 단독 부활 시 R47.2 vLLM source 수정 path 가 default. ScaleShield/ModalSplit 등 R45 borderline idea 도 R47.2 적용 시 application-level path 로 R45 clean 승격 가능. 전체 미선정 표는 변경 X — sim path 재평가만 추가 명시.

---

## Kelle MICRO 2025 baseline 보강 종합 (R46 의무)

`R46 verified: title="Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing", venue=MICRO 2025 (Seoul, Oct 18-22), arXiv:2510.16040`

| Idea (status) | Kelle 와의 1:1 차별화 |
|---|---------------------|
| OAEP-KV (Tier-1, selected) | refresh vs ECC + token-importance vs outlier-mask R² ablation |
| BlockShard (Tier-1, selected) | refresh vs retirement axis + 동일 BER head-to-head plot |
| LayerTier (Tier-1, selected) | layer access × reliability vs token importance × bit-position |
| VLM-MAP (Tier-2, selected) | modality flag vs token importance ablation |
| EntropyECC (Tier-2, selected) | entropy vs token importance R²<0.5 (orthogonal) 의무 |
| Sentinel (merged in OAEP-KV) | DRAM-row spatial axis vs Kelle bit-position axis (orthogonal) |
| Chameleon (DROP) | metadata-mantissa hierarchy vs Kelle bit-position MSB/LSB |
| ScaleShield (DROP) | scale-mantissa hierarchy vs Kelle 2DRP refresh |
| ModalSplit (merged in VLM-MAP) | modality 가 importance 의 specialization 인지 별도 signal 인지 |

---

## C: Out-of-Scope 카테고리

- **Distributed/multi-node KV serving (LMCache 2025, BanaServe)** — 본 세션 single-node sim. 별도 세션.
- **Real DRAM rowhammer rig / FPGA prototype** — R45 strict.
- **Linux kernel mainline upstream merge** — R45 borderline; BlockShard 가 ABI position section 만 포함.
- **Adversarial bit-flip attack ML pipeline** — 본 세션 reliability axis. GenBFA 는 baseline 인용만.
- **Cross-system generality (TensorRT-LLM, DeepSpeed-MII)** — 본 세션 vLLM/SGLang 한정.
