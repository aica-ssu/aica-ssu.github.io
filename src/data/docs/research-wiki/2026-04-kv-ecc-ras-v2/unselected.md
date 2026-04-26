# 미선정 로그 — 2026-04-26 KV cache ECC + Memory RAS v2

> [← Session Overview](README.md)

본 세션에서 도출되었으나 Top-M 6 final 외로 분류된 idea 의 사유 및 재방문 조건. Phase 1 의 8 candidate 중 2 + Phase 1' 의 9 변형 중 3 = 5 미선정.

---

## P2 SinkShield (Tier-1 Major Revision → Replace 권고)

**Title**: SinkShield — Attention-Sink-Aware Refresh Management Frequency Boost for KV Cache Retention in HBM3/LPDDR5x

**Hypothesis**: StreamingLLM (E1, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)) 의 attention sink 4 initial token 이 4M streaming stability 단독 결정 → HBM3/LPDDR5x RFM (Refresh Management) command 를 sink token block 에만 high-frequency 발행. 단 1 sink token corruption 도 PPL collapse 차단 + RFM overhead <2%.

**미선정 사유**:

1. **KVSink/SinkQ concurrent (55-65%)**: KVSink ([arXiv:2508.04257](https://arxiv.org/abs/2508.04257), COLM 2025) + SinkQ ([OpenReview bJ33TvbJW0](https://openreview.net/forum?id=bJ33TvbJW0)) 가 모두 attention sink 보존 motivation. KVSink = activation outlier 로 sink dynamic 예측 (quantization layer), SinkQ = 2-bit KV quant + dynamic sink tracking — **mechanism axis 직교 (RFM hardware vs quantization software)** 이지만 **motivation 동일**.
2. **v1 EntropyECC 와 차별성 약화**: v1 의 entropy 기반 ECC 가 sink token 의 entropy 와 매핑 가능성. axis 분리가 weak.
3. **HBM3 RFM granularity 문제**: legacy-system-expert Phase 2 — HBM3 JESD238B RFM 은 row-level + bank-level 까지만, 단일 KV block (16-token) 단위 RFM 발행은 datasheet 와 mismatch. sink_flag → row mapping 추가 명시 필요.
4. **Score 7.3** (Phase 2) — Tier-1 cut 미달, Tier-2 venue switch 권고했으나 P5 Watermark 와 axis 겹침으로 Tier-2 도 제외.

**재방문 조건**:
- KVSink/SinkQ 와 stack-able layer (RFM hardware × quantization software) 로 reposition + 산업적 stack 가치 입증.
- v1 EntropyECC 와의 mechanism level 명확한 차별화 paragraph.
- HBM3 RFM row/bank-level mapping 의 sink_flag → row aggregation 정확 문서화.

---

## P5 Watermark (Tier-2 보조, Top-M 외)

**Title**: Watermark — KV-Cache-Specific Lightweight CRC Watermark for Stack-able Bit-Flip Detection in Quantized LLM Inference

**Hypothesis**: SilentStriker ([arXiv:2509.17371](https://arxiv.org/abs/2509.17371)) 의 50-bit flip → GSM8K 65.7% → 7.6% naturalness 유지 stealthy attack 의 **KV cache 영역 한정** 16-bit CRC + per-block secret watermark. detection rate >99% + overhead <1%.

**미선정 사유**:

1. **LM-Fix/RoR/BitFlipScope concurrent (60-70%)**: LM-Fix ([arXiv:2511.02866](https://arxiv.org/abs/2511.02866)) 가 weight 영역 detection 94%/100% 1.9-7.7% overhead, Rotated Robustness ([arXiv:2603.16382](https://arxiv.org/abs/2603.16382)) Householder rotation prevention (50-bit BFA → 17,000+ bit), BitFlipScope ([arXiv:2512.22174](https://arxiv.org/abs/2512.22174)) differential analysis fault localization — 3 concurrent paper.
2. Phase 1' 에서 **KV-cache-specific reposition + ChampSim 제거 + R50.2 mechanism 격상** 으로 6.0 → 6.5 회복했으나 여전히 Tier-2 독립 Top 3 외.
3. **R21 paper pair 1 쌍 한계**: V3+V4 우선, V1 (P1 paper pair) 와 함께 Option C 선택 시 P5 외.
4. **Score 6.5** (Phase 1') — Tier-2 보조 후보 수준.

**재방문 조건**:
- LM-Fix/RoR 와 stack-able layer (KV cache vs weight axis) 의 정량적 stack 가치 입증.
- attention kernel fuse 로 0.2-0.4ms overhead 실측 (Triton/CUTLASS pattern).
- Tier-2 venue 가 추가 신설 시 (DAC 2027 6p 추가 slot 등) 독립 contribution 으로 재진입 가능.

---

## P6 VideoVeil (Tier-2 Major Revision → Replace 권고)

**Title**: VideoVeil — Frame-Importance-Aware ECC Strength for Video VLM Long-Context KV Cache

**Hypothesis**: 비디오 VLM 의 frame-level KV importance 8-15× variance (saliency, optical flow) → important frame 만 strong ECC (DEC-3) + 나머지 SECDED. video QA accuracy drop <2%.

**미선정 사유**:

1. **Sali-Cache adjacent (40-50%)**: Sali-Cache ([arXiv:2602.14236](https://arxiv.org/abs/2602.14236)) 가 saliency + optical flow 기반 video VLM KV 압축 — frame-importance source 동일. ECC strength 차등은 axis 차별화이나 **measurement source overlap**.
2. **v1 VLM-MAP 와 axis 분리 약화**: v1 의 vision-vs-text axis 와 frame-axis 가 일부 겹침.
3. **HBM3 RFM frame-level granularity 문제**: legacy-system-expert Phase 2 — frame KV block 이 multi-row 분산 시 RFM 명령 다수 발행 overhead.
4. **Score 6.7** (Phase 2) — Tier-2 venue 적합하지만 Top 3 외.

**재방문 조건**:
- Video-specific RAS 의 다른 axis (예: vision token outlier × HBM3 RFM, video stream의 temporal outlier × ECS) 로 reposition.
- Sali-Cache 의 frame-importance source 를 reuse 하면서 **compression × reliability stack** 의 정량적 가치.
- v1 VLM-MAP 와 mechanism level 명확한 차별화.

---

## P7 EdgeARM (Tier-2 보조, Top-M 외)

**Title**: EdgeARM — LPDDR5x Adaptive Refresh Management Counter as Hot-Block Identifier for Edge LLM KV Reliability with llama.cpp/MLC LLM Cross-Runtime Validation

**Hypothesis**: LPDDR5x JESD209-5C ARM counter 가 mobile/edge SoC (Jetson Orin) hardware-level → vLLM-edge polling 으로 hot KV block 식별 → reliability-tagged sub-page 에 placement. silent KV corruption 90% 감소 + ARM polling overhead <0.5%.

**미선정 사유**:

1. **Kelle adjacent (35-45%)**: Kelle ([arXiv:2510.16040](https://arxiv.org/abs/2510.16040), MICRO 2025) 의 2DRP (importance-based refresh) 가 custom eDRAM 위에 — P7 은 LPDDR5x 표준 적용 측면에서 deployment 차별화 명확하지만 mechanism 의 idea 자체는 같은 family.
2. **R21 paper pair 1 쌍 한계**: V3 (P3 paper pair) + V1 (P1 sister) 의 Option C 선택 시 P7 외.
3. **Vendor-specific TAP register layout**: legacy-system-expert Phase 2 — Qualcomm/MediaTek/Samsung Exynos 별 register map 다름. sysfs/mock emulation 의 fidelity 추가 검증 필요.
4. **Score 7.1** (Phase 1') — Tier-2 독립 Top 3 외 5위.

**재방문 조건**:
- Qualcomm SNPE SDK / Samsung Exynos / MediaTek Dimensity 3 vendor register map 모두 검증 후 commodity SoC deployment generality 입증.
- llama.cpp / MLC LLM cross-runtime ablation 결과 산업 deployment 가치 정량화.
- Kelle 와의 deployment differential (custom vs commodity) 의 산업 백서 형식 reposition.

---

## V4 PATroller-Solo (Tier-2 보조, R21 paper pair 한계)

**Title**: PATroller-Solo — HBM3 IEEE 1500 TAP PAT Counter Polling Overhead Profile for vLLM BlockManager

**Hypothesis**: HBM3 IEEE 1500 TAP 으로 PAT counter top-32 query 의 polling latency 가 1s interval 에서 hot-path 영향 <0.05% (TPOT 40ms 의 0.02ms budget 내).

**미선정 사유**:

1. **R21 paper pair 1 쌍 한계**: V3 ↔ P3 + V4 ↔ P4 동시 포함 시 2 쌍 위반. Option C (P8 + V3 + V1) 선택 — V4 미선정.
2. system-robustness-expert Phase 2' 권고: V3 가 vLLM RFC #5067 mainstream supplement 로 production impact 강함, V4 보다 narrative fit. → V4 = Option D (P8 + V4 + V1) 의 alternative.
3. **Score 7.5** — V3 (7.5) 와 동률, narrative 만 차이.

**재방문 조건**:
- 추가 Tier-2 venue slot 확보 시 (ITC 2027 + DATE 2027 + DAC 2027 기존, IEEE CAL 추가 시) 독립 contribution 으로 재진입.
- Option D (P8 + V4 + V1) 로 재선택 시 V3 와 swap 가능.

---

## 미선정 정리 표

| Idea | Type | 미선정 사유 | Score | 재방문 조건 |
|------|------|--------------|------:|--------------|
| P2 SinkShield | Tier-1 후보 | KVSink/SinkQ concurrent 55-65%, v1 EntropyECC overlap | 7.3 | KVSink 와 stack-able layer reposition |
| P5 Watermark | Tier-2 보조 | LM-Fix/RoR concurrent 60-70%, Top 3 외 | 6.5 | KV-cache vs weight axis stack 가치 입증 |
| P6 VideoVeil | Tier-2 후보 | Sali-Cache adjacent 40-50%, v1 VLM-MAP overlap | 6.7 | Video-specific RAS 다른 axis reposition |
| P7 EdgeARM | Tier-2 보조 | Kelle adjacent 35-45%, R21 paper pair 한계 | 7.1 | 3 vendor register map 검증 + commodity deployment 백서 |
| V4 PATroller-Solo | Tier-2 variant | R21 paper pair 1 쌍 한계 (V3 우선) | 7.5 | 추가 Tier-2 slot 확보 시 V3 와 swap |
