# TEMPER — Error-Robust KV Cache Quantization Co-Design

## 1. Research Questions
- **RQ-T2**: KIVI/KVQuant 를 random-bit-error-robust 로 calibrate 하면 KV cache 가 보호 없이 weak memory 의 높은 BER 에서 정확도를 유지하여 protection/migration 수요 자체를 줄일 수 있는가?
- **RQ-T2.2**: clean accuracy 손실을 최소화하면서 (RANDBET-style) random-bit-error training/calibration 을 KV 의 K/V 비대칭 구조에 맞게 적용하는 최적 방식은?

## 2. 배경 및 동기
- **문제**: KEEPER/TALLY 는 사후 대응(migration/검출). 근본적으로 KV 가 error 에 강하면 보호 수요 자체가 감소. RANDBET([arXiv:2006.13977](https://arxiv.org/abs/2006.13977))은 weight 를 random-bit-error-robust 로 training 해 낮은 SRAM voltage(높은 BER) 운용 가능을 보임.
- **한계**: RANDBET 은 **weight/SRAM**, RESQ([arXiv:2603.15413](https://arxiv.org/abs/2603.15413))는 quant+harden 이나 KV 특화 아님. KIVI([arXiv:2402.02750](https://arxiv.org/abs/2402.02750))/KVQuant([arXiv:2401.18079](https://arxiv.org/abs/2401.18079))는 압축만, error-robustness 미고려.
- **Gap**: KV cache 를 HBM/GDDR BER 에 강하도록 co-design (algorithm-only, scoop 검증 CLEAR-adjacent — RANDBET 인접하나 KV·HBM 대상은 미점유).

## 3. 핵심 가설
KV quantization 을 random-bit-error-aware 로 calibrate(clipping + per-channel K/per-token V 구조 보존 + error-augmented calibration)하면, KV 가 보호 없이 BER 1e-4 수준 weak memory 에서 perplexity 상승을 <0.3 으로 억제 가능 — KV 정보가 소수 outlier 에 집중되어 그 부분만 robust 하게 표현하면 되기 때문.

## 4. Falsifiable predictions
1. error-robust calibration 이 vanilla KIVI/KVQuant 대비 동일 BER 에서 perplexity 상승을 유의하게 낮춰야 한다.
2. clean accuracy 손실 <1% 여야 한다(아니면 trade-off 불리).
3. K(per-channel)와 V(per-token)에 비대칭 robustness 적용이 균일 적용보다 우월.

## 5. 접근법 — Mechanism (단일 mechanism)
- **① Scheme**: vLLM KIVI/KVQuant fork 에 error-aware calibration pass 추가 — calibration 중 KV 에 random bit-error 주입(GoldenTransformer)하고 그 하에서 quant param(scale/clip/outlier set)을 robust 하게 fit.
- **② 문제**: weak memory(높은 BER)에 KV 를 두면 corruption. 사전에 robust 하면 보호 불필요.
- **③ 동작**: (1) calibration set 으로 KV 분포 측정. (2) random bit-error augmentation 하에 clipping threshold·outlier isolation(KVQuant dense-sparse) 최적화. (3) K=per-channel, V=per-token 비대칭으로 robustness 배분. (4) error-augmented 로 quant param 확정. 파라미터: target BER, clip ratio, outlier %.
- **④ 차별화**: RANDBET(weight) → KV; KVQuant(압축만) → +error-robustness.

### R50-γ
- algorithm-only, parity/ECC 추가 없음 — quant 표현 size 변화 없음(KV 2-3bit 유지). baseline 무관, **R55 safe**(model inference 직접 측정).

## 6. 예상 효과
| 지표 | Baseline | 목표 | 조건 |
|------|---------|------|------|
| **[Robustness]** BER 1e-4 perplexity 상승 | vanilla KIVI: +large | TEMPER: +<0.3 | LLaMA-3-8B |
| **[Memory eff.]** 보호 overhead | ECC 12.5% | 0 (algorithm) | calibration only |
| **[Robustness]** clean accuracy 손실 | — | <1% | calibration trade-off |

**적용 범위**: 양자화 KV(2-3bit). **미적용**: FP16 KV(quant 없음). **불확실**: 매우 높은 BER(1e-3+)에서 한계.

## 7. 실험 설계
1. **Hardware**: RTX 5090(#3) / Pro 6000(#5). algorithm-only, GPU 직접 측정.
2. **Model**: LLaMA-3-8B, Qwen2.5-7B/14B, Mistral-7B.
3. **Dataset**: WikiText/C4(perplexity), LongBench, MMLU. calibration: small subset.
4. **Tools**: vLLM KIVI/KVQuant fork, GoldenTransformer(KV BER inject), lm-eval. Ramulator2(보조 — 어떤 BER 가 현실적인지).
5. **Ablation**: (a) vanilla vs error-robust calibration; (b) K/V 비대칭 vs 균일; (c) BER sweep(1e-6~1e-3); (d) target-BER calibration 일치/불일치. baseline: KIVI, KVQuant, RANDBET-mimic(weight). runtime ~1-2일/config.
6. **Implementation Steps**: W1 KIVI/KVQuant fork + baseline perplexity. W2 GoldenTransformer KV BER inject pipeline. W3 error-augmented calibration. W4 K/V 비대칭. W5 ablation. 완료: 동일 BER 에서 perplexity 상승 < vanilla, clean<1% 손실.
7. **Preliminary Metrics**: (a) BER-perplexity 곡선(vanilla); (b) outlier % 측정; (c) calibration 후 robustness; (d) clean accuracy delta.

## 8. Code Implementation Verification
- **Clone**: vllm(KIVI/KVQuant path), goldentransformer.
- **Patch**: KV quant calibration 에 error-aug pass(Modify `quantization/kv_cache`), GoldenTransformer KV inject(reuse).
- **Smoke**: calibrate → KV inject BER 1e-4 → perplexity vs vanilla.

## 9. 관련 연구 및 차별점
| Paper | venue/링크 | Date | 관계 | 차별점 |
|-------|-----------|------|------|------|
| RANDBET | [2006.13977](https://arxiv.org/abs/2006.13977) / MLSys'21 | 2020 | closest(error-robust train) | weight/SRAM voltage. TEMPER 는 KV/HBM BER |
| KVQuant | [2401.18079](https://arxiv.org/abs/2401.18079) / NeurIPS'24 | 2024 | base(KV quant) | 압축만, error 미고려. TEMPER 는 +robustness |
| KIVI | [2402.02750](https://arxiv.org/abs/2402.02750) / ICML'24 | 2024 | base(K/V 비대칭) | TEMPER 가 비대칭 robustness 로 확장 |
| RESQ | [2603.15413](https://arxiv.org/abs/2603.15413) | 2026 | adjacent(reliability+security quant) | weight 중심. TEMPER 는 KV cache 특화 |

## 10. 리스크 / 한계
| 리스크 | 영향 | 완화 |
|--------|------|-----|
| error-robust calibration 의 clean accuracy 손실>1% | 중간 — trade-off 불리 | clip ratio 미세조정, robustness budget 제한. 손실 큼 시 ablation 축소(decision tree ②) |
| 매우 높은 BER 에서 outlier 도 손상 | 중간 | TALLY 검출 결합(outlier만 검출) |

## 11. Scoring
| Reviewer | score | 최고 | 최저 |
|----------|-------|------|------|
| novelty | 7.6 | KV cache 대상 error-robust quant(CLEAR-adjacent) | RANDBET 의 방법론 재사용 |
| differentiation | 7.4 | KVQuant +robustness 직교 | RESQ/RANDBET 와 컨셉 인접 |
| impact | 7.6 | 보호 수요 자체 제거(근본적) | 매우 높은 BER 한계 |
| ai-implementation | 8.2 | algorithm-only, GPU 직접 측정(R55 safe) | calibration cost |
| arch-system-implementation | 7.4 | Ramulator2 로 현실 BER 근거 | HW 측 기여 적음 |
| **mean** | **7.64** | — | — |

- **★ 최고**: ai-implementation = 8.4 — algorithm-only 라 simulator/HW 불필요, vLLM+lm-eval 만으로 완결(가장 낮은 구현 risk, 1-student 적합).
- **▼ 최저**: differentiation = 7.4 — RANDBET 방법론 재사용이라 "KV 적용 delta" critique 대비 K/V 비대칭 robustness 가 차별 핵심.
- **전문가 합의**: ai-optimization ✅ / algorithm ✅. KEEPER 와 상보(robust KV 는 migration 수요↓).
