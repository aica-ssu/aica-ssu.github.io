# TALLY — Block-Syndrome KV Checksum for Silent-Corruption Detection

## 1. Research Questions
- **RQ-T1**: PagedAttention block(16 token) 단위 shared-syndrome checksum 으로 per-element ECC 없이 silent KV corruption 을 검출(→migrate/recompute trigger)하면서 overhead 를 SECDED 12.5% 이내로 유지할 수 있는가?
- **RQ-T1.2**: 검출-only(정정 없음) 가 inference KV 에 충분한가 — 검출 후 migration/recompute 로 복구하면 per-element ECC 의 정정 비용을 회피할 수 있는가?

## 2. 배경 및 동기
- **문제**: uniform per-element ECC 는 비쌈(ECC-based FT [arXiv:2508.12347](https://arxiv.org/abs/2508.12347): ~47.5% area @ BER 1e-1). KV 는 대부분 error-tolerant(KVQuant [arXiv:2401.18079](https://arxiv.org/abs/2401.18079): bulk sparse KV 견딤)이고 silent corruption 만 막으면 됨.
- **한계**: SDC-in-training([arXiv:2604.00726](https://arxiv.org/abs/2604.00726))은 training param/step 대상. LMCache 는 storage-layer 오류 보고에 의존(자체 checksum 없음).
- **Gap (GAP-6)**: inference KV block-level **detection-only** shared-syndrome 으로 migration/recompute 를 trigger 하는 경량 검출 부재(scoop 검증 CLEAR).

## 3. 핵심 가설
PagedAttention block 단위 shared-syndrome checksum 은 per-element ECC 대비 1자리 수 작은 overhead 로 single-bit silent KV corruption 을 >95% 검출하고, 검출 시 migration/recompute 로 복구하면 정정 코드 없이도 충분하다 — KV 는 정정보다 검출+복구가 cost-effective 하기 때문.

## 4. Falsifiable predictions
1. block-shared syndrome overhead 가 SECDED 12.5% 대비 1자리 수 작아야(<2-4%) 한다.
2. single-bit flip 검출률 >95%. 미달 시 syndrome bit 증설 필요 → overhead 이점 상실.
3. 검출→recompute/migration 복구가 per-element ECC 정정과 동등 accuracy 회복.

## 5. 접근법 — Mechanism

### M1 — Block Shared-Syndrome Checksum (단일 mechanism, Tier-2 규칙)
- **① Scheme**: PagedAttention block(16 token × n_head × head_dim) 당 shared syndrome word(예: 32-64 bit XOR-folded + light CRC) 를 block metadata 에 부착. HW 측은 Verilog checksum unit(<10K gate).
- **② 문제**: silent KV corruption 을 cheap 하게 검출해야 migration/recompute 를 trigger 가능.
- **③ 동작 (step-by-step)**: (1) KV block write 시 syndrome 계산·저장. (2) block read(attention) 시 syndrome 재계산·비교. (3) 불일치 → corruption flag → KEEPER migration 또는 layer recompute trigger. (4) R50-γ.3a shared-metadata: block 당 1 syndrome 으로 per-element overhead 분산. 파라미터: syndrome bit width, block size.
- **④ 차별화**: per-element ECC(정정) ≠ block-shared(검출-only). Frugal-ECC 류 compression-parity 와 달리 KV block 의 token granularity 활용.

### R50-γ ECC/Overhead 검증
| 항목 | Baseline | 본 mechanism | 초과? | Mitigation |
|------|---------|-------------|-------|-----------|
| parity/data | SECDED 12.5% | block 64-bit syndrome / (16 token block ≈ 수십 KB) = **<0.5%** | ✓ 이내 | (a) shared-metadata block 단위 분산 |
| 검출 vs 정정 | 정정(비쌈) | 검출-only + migration/recompute 복구 | — | recompute/migration 으로 정정 대체 |

## 6. 예상 효과
| 지표 | Baseline | 목표 | 조건 |
|------|---------|------|------|
| **[Memory eff.]** overhead | SECDED 12.5% | <2-4% | block-shared |
| **[Robustness]** single-flip 검출률 | — | >95% | 32-64bit syndrome |
| **[Robustness]** 검출→복구 후 accuracy | per-elem ECC 동등 | 동등 | migration/recompute |

**적용 범위**: KV cache(token-block 구조). **미적용**: weight(static, exponent-ECC 적합). **불확실**: multi-bit 동시 flip 검출률.

## 7. 실험 설계
1. **Hardware**: RTX 5090(#3) + Yosys/Sky130(checksum unit 합성). 
2. **Model**: LLaMA-3-8B, Qwen2.5-7B.
3. **Dataset**: LongBench, WikiText(perplexity), MMLU.
4. **Tools**: GoldenTransformer(KV inject), lm-eval, Yosys+Sky130(area/power/timing), vLLM(block metadata).
5. **Ablation**: (a) syndrome width(16/32/64) vs 검출률; (b) block size; (c) single vs multi-bit; baseline: SECDED per-element, no-protection. RTL 합성 결과(area/power/Fmax) 첨부. runtime ~1-2일/config.
6. **Implementation Steps**: W1 GoldenTransformer KV inject + baseline. W2 syndrome 계산/검증 SW. W3 vLLM block metadata 통합. W4 Verilog checksum unit + 합성. W5 검출률/overhead ablation. 완료: 검출률>95% + RTL area/power.
7. **Preliminary Metrics**: (a) 검출률 vs syndrome width; (b) RTL gate count(합성); (c) overhead %; (d) 복구 후 accuracy.

## 8. Code Implementation Verification
- **Clone**: vllm, goldentransformer; Verilog 자작 + Yosys.
- **Patch**: block syndrome in `block_manager`(Add metadata), check in attention read(Wrap), Verilog checksum module(standalone 합성).
- **Smoke**: KV inject single-bit → syndrome mismatch detected → flag.

## 9. 관련 연구 및 차별점
| Paper | venue/링크 | Date | 관계 | 차별점 |
|-------|-----------|------|------|------|
| ECC-based FT for DNN | [2508.12347](https://arxiv.org/abs/2508.12347) | 2025 | closest(ECC overhead) | uniform per-element(~48% area). TALLY 는 block-shared 검출-only |
| SDC in LLM training | [2604.00726](https://arxiv.org/abs/2604.00726) / CCGrid'26 | 2026 | adjacent(검출+recompute) | training param/step. TALLY 는 inference KV block |
| Frugal-ECC | MICRO'15 | 2015 | foundational(shared parity) | compression-parity. TALLY 는 KV token-block |

## 10. 리스크 / 한계
| 리스크 | 영향 | 완화 |
|--------|------|-----|
| multi-bit 동시 flip 미검출(XOR-fold 약점) | 중간 — 일부 SDC 누락 | light CRC 추가, syndrome width sweep. 검출률<90% 시 강화(decision tree ②) |
| 검출 후 복구 비용(recompute/migration)이 큼 | 중간 | KEEPER migration 재사용(rare event) |

## 11. Scoring
| Reviewer | score | 최고 | 최저 |
|----------|-------|------|------|
| novelty | 8.0 | inference KV block-shared 검출-only(CLEAR) | checksum 자체는 고전 기법 |
| differentiation | 7.8 | per-element ECC 대비 overhead 직교 | SDC-training 과 검출+recompute 컨셉 인접 |
| impact | 7.5 | 경량 검출로 migration enable | 검출-only 단독 가치 제한 |
| ai-implementation | 7.6 | GoldenTransformer KV inject 명확 | block metadata ripple |
| arch-system-implementation | 8.0 | Verilog 합성으로 area/power 입증 | multi-bit coverage |
| **mean** | **7.78** | — | — |

- **★ 최고**: arch-system.RTL-synth = 8.2 — checksum unit Yosys/Sky130 합성으로 area/power/timing 정량 입증 가능(Tier-2 DAC/DATE 적합).
- **▼ 최저**: impact.standalone = 7.2 — 검출-only 는 KEEPER trigger 와 결합 시 가치 극대화(paper pair).
- **전문가 합의**: algorithm ✅ / system-robustness ✅. **Tier-1 KEEPER 와 paper pair** (KEEPER 의 corruption trigger 로 채택 시 Tier-1 spinoff).
