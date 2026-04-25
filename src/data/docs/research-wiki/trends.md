# Research Trends

`aica-research-bot` Mode 2 / Mode 3 실행에서 도출된 학회별 또는 분야별 트렌드를 시간 역순으로 기록.

---

### VLM + PIM / GPU-PIM Heterogeneous Serving — 2026-04 시점 미니 트렌드
- **Date analyzed**: 2026-04-22
- **Session**: [링크](/research-wiki/2026-04/vlm-pim-extension)
- **Experts**: hw-pim-accelerator-expert (primary), ai-optimization-expert, legacy-system-expert

**신호 1 — VLM-specific HW accelerator 가 2025-Q4 ~ 2026-Q1 폭발적 증가**: Focus ([arXiv:2512.14661](https://arxiv.org/abs/2512.14661), HPCA 2026 Best Paper Candidate, Duke), V-Rex ([arXiv:2512.12284](https://arxiv.org/abs/2512.12284), HPCA 2026, KAIST), ORCHES (MICRO 2025, vision reasoning 3.10×), Pimba ([arXiv:2507.10178](https://arxiv.org/abs/2507.10178), MICRO 2025 post-transformer) 등 4편이 6개월 내 연속 등장. 공통 테마: **VLM 특이성 (visual token 비대칭, multi-image long-context, video streaming) 을 HW 에 직접 반영**. AttAcc (ASPLOS 2024) / NeuPIMs / IANUS / TransPIM 의 LLM 가정은 이미 outdated.

**신호 2 — KV hierarchy 는 레드오션화**: PAM ([arXiv:2602.11521](https://arxiv.org/abs/2602.11521), 2026.02, HBM-PIM + DRAM-PIM + SSD-PIM 3-tier + PAMattention), PIMphony/LoL-PIM ([arXiv:2412.20166](https://arxiv.org/abs/2412.20166), HPCA 2026), LongSight (MICRO 2025, NPU HBM dense + CXL sparse), Scalable PNM ([arXiv:2511.00321](https://arxiv.org/abs/2511.00321), 2025.11 CXL multi-PNM 1M-Token), LeoAM ([arXiv:2506.20187](https://arxiv.org/abs/2506.20187), adaptive GPU-CPU-Disk), Strata ([arXiv:2508.18572](https://arxiv.org/abs/2508.18572), hierarchical context caching) — 3-tier + hot/cold migration 이 이미 6편 이상. **단독 3-tier 제안은 통과 불가**.

**신호 3 — VLM serving scheduling 도 레드오션화**: ModServe ([arXiv:2502.00937](https://arxiv.org/abs/2502.00937), 2025.02), RPS-Serve ([arXiv:2603.26498](https://arxiv.org/abs/2603.26498), 2026.03, MMMU tail 해결), ElasticMM ([arXiv:2507.10069](https://arxiv.org/abs/2507.10069), 2025.07), Dual-Pool Token-Budget ([arXiv:2604.08075](https://arxiv.org/abs/2604.08075), 2026.04), PolyServe ([arXiv:2507.17769](https://arxiv.org/abs/2507.17769)), SLOs-Serve ([arXiv:2504.08784](https://arxiv.org/abs/2504.08784)) — 1년 사이 6편 이상. Modality-aware admission / disaggregation / pool partition 모두 선점.

**신호 4 — Image-hash KV reuse 가 SGLang 에 공식 구현**: VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977), 2025.12) 가 Qwen3-VL 대상 2% 계산 + 98% 재사용 + 1.2-16× TTFT 달성. vLLM prefix caching 도 image hash 공식 지원 (PR #11187). **Cross-request image KV sharing** 은 더 이상 novel axis 아님. MPIC ([arXiv:2502.01960](https://arxiv.org/abs/2502.01960)), KVShare ([arXiv:2503.16525](https://arxiv.org/abs/2503.16525)), CacheBlend ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444)), DroidSpeak ([arXiv:2411.02820](https://arxiv.org/abs/2411.02820)) 추가.

**신호 5 — VLM KV quantization 은 per-sample / per-layer 분리 시작**: VL-Cache ([arXiv:2410.23317](https://arxiv.org/abs/2410.23317), ICLR 2025 layer-adaptive budget), AKVQ-VL ([arXiv:2501.15021](https://arxiv.org/abs/2501.15021) attention-aware saliency), MBQ ([arXiv:2412.19509](https://arxiv.org/abs/2412.19509), CVPR 2025 modality-balanced), SparseVLM ([arXiv:2410.04417](https://arxiv.org/abs/2410.04417), ICML 2025), SparseVILA ([arXiv:2510.17777](https://arxiv.org/abs/2510.17777), ICCV 2025). 모두 **layer 또는 token granularity**. **Per-sample fragility gating 은 아직 empty axis**.

**신호 5-보강 (2026-04-22 post-literature survey) — VLM quantization × attention distribution dynamics 는 측정 공백**: 본 연구의 W8A8 +66.85pp visual attention collapse (5.73% → 72.58%) 관찰을 계기로 MBQ/Q-VLM/VLMQ/MQuant/AKVQ-VL/VL-Cache/LLM.int8/SmoothQuant/AWQ/QuaRot/SpinQuant/QUIK 전체 정독 결과: **누구도 layer-wise visual attention ratio 를 quantization 전/후로 직접 측정하지 않음**. MBQ 는 gradient sensitivity 로 vision 10× 둔감 보고, VLMQ 는 Hessian 편향, MQuant 는 per-tensor scale mismatch, LLM.int8 는 outlier → softmax mass 방향 영향 — 모두 **간접 signal**. Visual Attention Sink ([arXiv:2503.03321](https://arxiv.org/abs/2503.03321)) 는 FP16 내재적 sink 발견이지만 quantization 영향 미분석. 실배포 W8A8 모델카드 (RedHatAI 등) 도 task accuracy 만 보고. **+66pp extreme collapse 정량화는 공백 + Qwen-VL family specific 측정도 공백**. F2 는 이 공백을 측정 axis + runtime detection axis 로 채움.

**VLM quantization 의 세 가지 mechanism 통합 (F2 post-survey 제안)**: (1) **Vision-side activation outlier 증폭** (MBQ + LLM.int8 합성) — per-tensor W8A8 이 vision-channel outlier 를 기준으로 scale 설정 → softmax logit 분포 왜곡. (2) **Attention sink 증폭** (Visual Attention Sink + Attention Sinks and Compression Valleys [arXiv:2510.06477](https://arxiv.org/abs/2510.06477)) — massive activation quantization 으로 sink-KV dot-product magnitude spiky 증가. (3) **Rotation-invariance 미적용** (MQuant + SpinQuant [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) + QuaRot) — rotation-free recipe 에서 vision-channel outlier 보존. 이 세 mechanism 이 VLM 배포 setting 에서 catastrophic regime 로 수렴하는 경로를 본 연구의 W8A8 collapse 로 최초 정량화. "Seeing but Not Believing" ([arXiv:2510.17771](https://arxiv.org/abs/2510.17771)) 의 attention-accuracy decoupling 으로 MBQ "W8A8 lossless (task acc)" 주장과 +66pp collapse 양립 설명.

**신호 6 — Post-transformer PIM 이 독립 영역으로 분화**: Pimba ([arXiv:2507.10178](https://arxiv.org/abs/2507.10178), Mamba/SSM/linear attn), P3-LLM ([arXiv:2511.06838](https://arxiv.org/abs/2511.06838), NPU-PIM mixed precision with dynamic smoothing), HPIM ([arXiv:2509.12993](https://arxiv.org/abs/2509.12993), SRAM-PIM + HBM-PIM 2-tier), AQPIM (HPCA 2026 in-memory activation quant). Qwen3.5 hybrid linear 같은 post-transformer VLM 은 Pimba 계열이 strong baseline.

**신호 7 — FP16 overflow 가 VLM 특화 numerical safety 이슈로 부상**: 본 연구의 Qwen3-VL L27 self-attn Q·Kᵀ > 65504 overflow 발견 (bf16 migration 강제). 이는 LM head 직전 sharpened representation 의 FP16 5-bit exponent 한계 문제 — **VLM 의 긴 sequence (L≥1000) 에서 정규화**. FlashAttention-3 ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608)) FP32 softmax rescale 은 표준 해법이지만, VLM PIM macro 에서 HW-level 감지 + fallback 은 미탐색.

**HW venue 전략적 교훈**:
1. VLM HW 는 SW+HW co-design 필수 — single-layer contribution 은 distinct competitor 5편 이상에 밀림.
2. **DeepStack topology + quantization robustness 는 2026 시점 unique axes**. Architecture-level novelty 가 단독으로는 약해도 결합 시 defensible.
3. 3-tier hierarchy / VLM serving scheduling 은 DROP 또는 사용 시 반드시 VLM-specific 추가 axis 결합.
4. **Cross-request image KV sharing 은 scooped** — coherence protocol / coarse-grained consistency 축으로 전환 필요.
5. Post-transformer (Mamba, hybrid linear) 확장은 Pimba/P3-LLM baseline 강력 — Qwen3-VL 같은 standard transformer scope 로 좁히는 것이 안전.

**Top-tier venue alignment 2026-2027**:
- **HPCA**: VLM accelerator + quantization + coherence (Focus/V-Rex/AQPIM)
- **MICRO**: GPU-PIM heterogeneous scheduling + post-transformer PIM (ORCHES/Pimba)
- **ASPLOS / MLSys**: end-to-end VLM serving (EPD disaggregation, chunked prefill extensions)
- **ISCA**: large-scale memory hierarchy + cross-tier consistency (PAM 계열 follow-up)
- **NeurIPS / ICLR**: VLM algorithm (per-sample fragility, modality-balanced, DeepStack-aware)

---

### BNN/TNN + CIM + Non-Image 도메인 — 2026-04 시점 미니 트렌드
- **Date analyzed**: 2026-04-22
- **Session**: 링크
- **Experts**: algorithm-expert, hw-pim-accelerator-expert
- **Papers analyzed**: 25+ (ISLPED/ISSCC/Interspeech/arxiv 2022-2026)
- **Trigger**: 공영호 교수의 PRISM (ISLPED'26 투고) 확장 쿼리. scratch training + small model + CIM target 제약.

#### Hot Topics (2024-2026 반복 등장)
| Topic | 대표 논문 | 한 줄 설명 |
|-------|---------|----------|
| **Low-rank scaling factor decomposition** | [LoRDS](https://arxiv.org/abs/2601.22716), PRISM (ISLPED'26 투고) | PRISM 과 LLM PTQ 쪽 (LoRDS) 이 2026 년 거의 동시에 "SF 를 low-rank 로 분해" 원리를 독립 발견 — domain-specific 확장이 open |
| **Streaming/temporal CIM** | [PSCNN](https://arxiv.org/abs/2205.01569), [CIMR-V](https://arxiv.org/abs/2503.22072), [Nonlinear Analog CIM LSTM KWS](https://arxiv.org/abs/2512.06362) | SRAM-CIM 의 streaming audio 가 production 에 근접 — 885 TOPS/W 달성. Crossbar 구조에서 temporal reuse / warm-LUT 는 미개척 |
| **Event-driven SNN on CIM (2025-11 집중)** | [CADC](https://arxiv.org/abs/2511.22166), [SOT-MRAM Event-Driven Spiking CIM](https://arxiv.org/abs/2511.03203), [ASTER](https://arxiv.org/abs/2511.06770), [SpikeFit](https://arxiv.org/abs/2604.14487) | 2025-11 에 SNN-CIM 논문 집중 발표 (243.6 TOPS/W) — **이 트랙은 scoop 매우 위험, 본 세션에서 미선정** |
| **Test-Time Adaptation for quantization** | [TTAQ](https://arxiv.org/abs/2412.09899), [Tent](https://arxiv.org/abs/2006.10726), [RTF-Q](https://arxiv.org/abs/2408.05752) | FP/INT8 양자화 모델이 distribution shift 하에서 성능 저하 — on-device 보정. BNN 쪽은 empty |
| **Biosignal BNN (매우 희박)** | [Arrhythmia-BNN](https://arxiv.org/abs/2304.01568), [BNN-HAR RISC-V](https://arxiv.org/abs/2205.12781), [MINIMALIST](https://arxiv.org/abs/2505.08599) | 2022-2023 의 소수 ancestor. **2024-2026 BNN+CIM+biosignal+subject-shift 4-way intersection 은 empty** (blue ocean) |
| **Wearable KWS BNN** | [BiFSMNv2](https://arxiv.org/abs/2211.06987), [SparkNet](https://arxiv.org/abs/2406.06634) | CPU/MCU 타겟은 성숙, CIM 결합은 "PSCNN 외 없음" — KWS+BNN+CIM+streaming 4-way 도 sparse |

#### Emerging Trends
| Trend | 관련 논문/신호 | 해석 |
|-------|---------------|------|
| Rank decomposition 이 양자화 도메인 전반으로 확산 | LoRDS(2026-01) + PRISM(2026-04 투고) | LoRA 의 training efficiency 이득이 inference quantization 로 확장 중. BNN/domain-특화 확장은 **지금이 golden window** |
| SNN-CIM 2025-11 에 집중 발표 | CADC+SOT-MRAM+ASTER+SpikeFit+ETBW-DVS 5편이 2개월 내 집중 | 2025-11 은 SNN-CIM 의 "hype moment". 후발 주자는 매우 빠른 출시 혹은 composition 전략 필요 |
| Wearable always-on 이 BNN 의 killer use case | ISSCC 2025/2026 의 ultra-low-power biomedical SoC 다수 | <100 µW 24h 대역이 연구→산업 전환점. Medical + HAR 융합이 clinical narrative 강점 |
| RRAM/SRAM collab CIM 실물 chip 가 ISSCC 에 | ISSCC 2025/2026 | precision mix 가 multi-modal workload 용 기술 성숙 중. per-modality partition 은 미답 |
| Streaming / online learning CIM | CIMR-V 등 | batched CIM 에서 streaming 으로 전환 중. Cold-start / warm-LUT 이슈가 새 연구 공간 |

#### Open Problems (이번 세션 도출)
| Open Problem | 관련 idea |
|-------------|----------|
| **Time-axis rank decomposition**: rank decomposition 을 spatial 외 time dimension 으로 확장하는 조건 | F1 TempoPRISM-CoDesign |
| **Subject-shift 를 scalar shift 로 근사 가능한 조건** (rank-1 Taylor 근사의 tightness) | F2 PhysioPRISM |
| **RRAM endurance 보존한 per-subject adaptation**: shadow SRAM 혹은 hybrid 메모리 설계 | F2 VitalXbar |
| **Heterogeneous-precision 다중 region CIM 의 joint training 안정화**: stagewise 외 다른 방법 | F2 VitalXbar (scope 축소) |
| **BNN + CIM + biosignal + subject-shift 4-way intersection** 최초 탐사 | F2 |
| **SNN-CIM 후발 주자의 합리적 repositioning** (CADC 와 composition / N_active-binned SF workshop) | SNN track (미선정) |

#### 유의사항 (Caveats)
- PRISM 의 rank decomposition 과 LoRDS 의 동시 발견 → **rank decomposition for quantization 은 2026 년 hot area** 로 전환 중. 후속 연구가 계속 쏟아질 것.
- SNN-CIM 영역은 이미 포화 — 단순 "RRAM 위에서 같은 아이디어 재실행" 은 scoop 위험. Composition / theoretical lemma 필요.
- Biosignal BNN 은 blue ocean 이나 subject-shift 평가의 variance 가 커 reviewer 의 statistical significance 요구가 높음 — paired t-test / cluster-balanced LOSO 필수.

---

### MoE Expert Fingerprinting (Security + Systems Dual Axis) — 2026-04 시점 미니 트렌드
- **Date analyzed**: 2026-04-21
- **Session**: 링크
- **Experts**: system-robustness-expert, ai-optimization-expert
- **Papers analyzed**: 30+ (Phase 1 expert-memory 기반 + Phase 2 리뷰어 arxiv 재검색)
- **Trigger**: User 공영호 교수가 "fingerprinting accuracy 85-90%, expert 많은 Qwen3에서 더 잘 됨" premise 공유 후 보안+시스템 연구 주제 도출 요청.

#### Hot Topics (2024-2026에 반복 등장)
| Topic | 대표 논문 | 한 줄 설명 |
|-------|---------|----------|
| **MoE expert prediction** | MoE-Beyond([2508.17137](https://arxiv.org/abs/2508.17137), 97.5%), Pre-Attention([2511.10676](https://arxiv.org/abs/2511.10676), 93-94%), OD-MoE([2512.03927](https://arxiv.org/abs/2512.03927), 99.94%), PROBE([2602.00509](https://arxiv.org/abs/2602.00509)) | 2025년 하반기부터 predictor accuracy가 90%+ 고지를 본격 돌파 — premise 85-90%가 오히려 conservative |
| **MoE side-channel attack (NEW)** | MoEcho([2508.15036](https://arxiv.org/abs/2508.15036), CCS'25), Expert Selections([2602.04105](https://arxiv.org/abs/2602.04105)), Stealing User Prompts([2410.22884](https://arxiv.org/abs/2410.22884)) | 2025-2026에 MoE 고유의 routing leakage가 security 영역에서 확립. 향후 2-3년 주요 threat class |
| **MoE serving infrastructure** | vLLM/SGLang EP deployment, Kimi K2.6, GLM-5.1 (754B MoE), DeepSeek-V3 EPLB | 2026년 현재 production MoE 서빙은 fine-grained (128+ experts) 중심 |
| **MoE prefetch family** | HOBBIT, ProMoE, PreScope, BuddyMoE, DuoServe-MoE, SP-MoE, MoE-SpeQ, MoE-SpAc | 포화 상태 — novelty는 formal bound / schedulability 방향으로 이동 |
| **MoE affinity dispatcher** | Semantic Parallelism, Gimbal, DualMap, XShare, METRO | 2025-2026 신흥 영역 — demand-side routing이 supply-side(EPLB) 보완 축으로 부상 |
| **MoE privacy defense** | CryptoMoE, SecMoE (crypto 100×), NoEsis (training DP), CacheSolidarity (KV cache) | Routing-layer에 대한 practical-overhead 방어는 blue ocean |

#### Emerging Trends
| Trend | 관련 논문/신호 | 해석 |
|-------|---------------|------|
| Predictor accuracy crosses threshold | OD-MoE 99.94% | 60-70% 정확도로 설계된 기존 prefetcher들의 "miss 경로 과부하" 설계 전제가 무너지는 중 |
| Side-channel-as-universal-threat | MoEcho + Whisper Leak + NetEcho | LLM에 대한 side-channel은 hardware(MoEcho), network(Whisper), cache timing(InputSnatch) 다축으로 성숙 |
| WCRT for GPU-based inference | RTAS 2025-2026 | GPU serving에서 formal real-time 분석이 점차 수용됨. 단 probabilistic-RT (p99.9) framing 필수 |
| Cross-request expert affinity | Semantic Parallelism, Gimbal | batch 단일 → replica pool 레벨로 dispatcher가 "올라가는" 추세 |
| MoE architecture growth trend | Mixtral-8 → Qwen3-Next-128 → DeepSeek-V3-256 | Expert granularity가 privacy liability (attack) + defense enabler (decoy budget) 양면성 |

#### Open Problems (이번 세션 도출)
| Open Problem | 관련 idea |
|-------------|----------|
| **Fingerprint-driven dispatcher의 signature-entropy scaling law** 형식화 (LSH cohort purity vs N_experts) | I4 FARD-C |
| **MoE expert scheduling의 WCRT theorem** (bounded-miss EDF 변형) | I3 ZMSP |
| **Routing-layer formal MI bound** as function of (k, ε, N_experts) | I2 PhantomRoute |
| **Privacy-vs-expert-count 측정** (MoEcho가 측정 안 한 scaling trend) | I2 motivation (I1 흡수) |
| **VLA phase-conditional fingerprint의 schedulability** | I5 (deferred) |
| Cross-idea predictor 공유 시 artificial-split rejection 회피 전략 | 전체 세션 output strategy |

#### 유의사항 (Caveats)
- MoEcho의 존재를 Phase 1 expert-memory가 놓침 → Phase 2 arxiv 재검색의 중요성 재확인. Web-based related work search는 **모든 세션에서 필수**.
- 85-90% premise는 OD-MoE(99.94%) / MoE-Beyond(97.5%) 대비 conservative. 상한 accuracy로 설계 재평가 시 idea 가치 상승 가능.
- MoE architecture는 few-expert(Mixtral-8) → many-expert(128+)로 명백한 수렴 중. 모든 아이디어는 many-expert regime에 최적화.

---

### VLM/VLA Software-Side Serving Optimization — 2026-04 시점 미니 트렌드 분석
- **Date analyzed**: 2026-04-21
- **Session**: 링크
- **Experts**: ai-optimization-expert
- **Papers analyzed**: 2 (in-depth) + 7 (referenced)
- **Trigger**: ACE-MoE의 future direction (VLM/VLA 확장)을 user가 명시적으로 요청

#### Hot Topics (반복 등장)
| Topic | 대표 논문 | 한 줄 설명 |
|-------|---------|----------|
| MoE expert offloading + caching | ACE-MoE [iccad-2026], HybriMoE [DAC'25], Edge-MoE [ICCAD'23] | GPU memory 한계 극복 위한 expert를 CPU/PIM/Storage로 offload + 활성화 시 transfer |
| Visual KV cache compression / eviction | VL-Cache [arXiv'24], PyramidKV [NeurIPS'24], SnapKV [NeurIPS'24], H2O [NeurIPS'23] | VLM의 visual KV가 sequence 86%+ 차지하나 attention 11% 만 받음 — software-side compression 활발 |
| Importance-driven scheduling | ACE caching, Cache-Conditional MoE [arXiv'24], cumulative score 류 | "hit rate가 아닌 accuracy preservation" 패러다임 전환 |

#### Emerging Trends
| Trend | 대표 논문 | 왜 주목 |
|-------|---------|--------|
| VLM-MoE inference (사용자 워크로드) | MoE-LLaVA [arXiv'24], Uni-MoE [arXiv'24], DeepSeek-VL2 | 모델 크기 증가 + 추론 cost 압박으로 VLM에서도 MoE 채택 가속 |
| VLA serving (real-time robotics) | OpenVLA [RSS'24], Pi-0 [arXiv'24], RT-2-X | LLM-style serving optimization을 hard real-time constraint에 적용해야 |
| Hierarchical multimodal injection | Qwen3-VL DeepStack, BLIP-3, Mantis | ViT 중간 출력을 LM 여러 layer에 inject하는 trend → layer-uniform 최적화의 가정 깨짐 |
| Cross-request sharing (multi-turn / multi-user) | vLLM prefix caching, SGLang RadixAttention, PromptCache | KV-level 공유에서 expert/visual 단위로 확장될 흐름 |
| Joint multi-axis resource budget | (emerging, no canonical paper yet) | 단일 axis 최적화의 한계, multi-resource (token×expert×bandwidth) 최적화 필요성 인지 시작 |

#### Open Problems (저자들이 언급한 미해결)
| Problem | 언급 출처 | 현재 접근법의 한계 |
|---------|---------|-----------------|
| VLM의 prefill TTFT가 LLM 대비 6-22× 길어짐 | VLM exploration (Choi 2026) | software-side로는 visual token 줄이기 외 직접 attack 부재 |
| ACE-MoE의 multimodal 확장 | ACE-MoE 결론 | future work로 명시, 아직 답 없음 |
| Layer-wise visual attention asymmetry 활용한 software-side 최적화 | VL-Cache, ACE-MoE 둘 다 layer dimension 다름 | 두 dim(token×layer, expert×layer)을 모두 다룬 통합 framework 부재 |
| VLA의 hard real-time inference + 정확도 보존 | OpenVLA 후속 | latency budget formulation의 control-theoretic 통합 부재 |
| Cross-request visual sharing의 expert-level extension | vLLM prefix caching 한계 | KV에 한정, expert activation 정보는 미공유 |

#### Cross-domain Opportunities
| Domain A | Domain B | 연결 시나리오 |
|----------|----------|-------------|
| MoE expert offloading | VLM serving | ACE 원칙을 modality-aware로 확장 (I1) |
| KV cache compression | MoE caching | Token+expert joint budget (I3) |
| LLM/VLM serving | Robotics control | Hard real-time SLO 도입 (I5) |
| Production workload analysis | Cache management | Image-level cross-request sharing (I6) |
| Model architecture analysis | Optimization | DeepStack/HIL boundary handling (I1 흡수) |

---

_이 트렌드 분석은 ACE-MoE 본 논문과 VLM exploration 1편의 깊은 분석에서 파생됨. Mode 3 (전 분야 3년 스캔)을 통해 보다 종합적인 트렌드 분석을 수행하면 본 entry가 보강될 것._

---

### 2026-04-21 보강 — 2024-2026 arxiv 관련 연구 검색 결과 기반 업데이트

arxiv API WebFetch 검색(9개 쿼리, 100+ 논문 검토)으로 아래 추가 트렌드 발견:

#### 가속 트렌드 (2025-2026 sub-themes)
- **VLA-specific inference optimization 폭증** (2025-09 이후): ActionFlow(2.55× FPS), HyperVLA(120× speedup via hypernetwork), CogVLA(NeurIPS'25), A1(72% latency reduction via budget-aware truncation), SemanticVLA(2.7×) 등. **VLA serving optimization이 2025년 하반기부터 집중 연구 분야**로 부상.
- **MoE edge inference의 budget/precision co-design 증가**: DyMoE, Dynamic Expert Quantization, Context-Aware MoE on CXL-NDP, SliceMoE 등이 모두 **importance + precision + scheduling의 joint approach** 채택. 연구 설계 자체의 **multi-axis 방향으로의 shift**.
- **Speculative + Prefetch + Quantize 3-way co-design 유행**: MoE-SpeQ, SpecMoE, MoE-SpAc 등.
- **Modality-aware vs modality-agnostic 논쟁 가시화**: AlignMamba-2 (modality-specific + shared experts) vs ERNIE 5.0 (modality-agnostic routing). I1과 같은 direction이 단일 옳은 답이 아님을 인지해야.
- **Null expert / big-little expert 방식 등장**: Improving MoE Compute Efficiency (null experts), MoBiLE (big+little), BuddyMoE (redundancy fallback) — **expert 자체의 heterogeneity 탐구**.

#### Emerging Trends (updated)
| Trend | 대표 논문 | 왜 주목 |
|-------|---------|--------|
| Budget-constrained optimization formulation | Dynamic Expert Quantization (2025-11), A1 (2026-04), DyMoE (2026-03) | "budget-aware"가 paper title 차원에서 급증 — optimization의 formalism이 정착 중 |
| VLA + MoE (architectural fusion) | HEX (2026-04), HY-Embodied-0.5 (2026-04), Qwen3.5-Omni (2026-04) | VLA/embodied foundation에 MoE 채택이 2026년부터 본격화 |
| Speculative-driven MoE inference | MoE-SpeQ (2025-11), SpecMoE (2026-04), MoE-SpAc (2026-02) | speculation으로 prefetch 예측 정확도 ↑ |
| Visual token pruning의 메타 학습 | VisPCO (2026-04, Pareto-frontier), PixelPrune (2026-04, predictive coding) | task-specific에서 generic framework로 이동 |
| KV cache의 multi-agent sharing | TokenDance (2026-04), MemServe | multi-agent serving 패턴 정착 |

#### Open Problems (updated)
| Problem | 최신 언급 |
|---------|---------|
| Modality-aware vs modality-agnostic routing의 empirical 비교 부재 | AlignMamba-2 vs ERNIE 5.0 — 직접 비교 실험 없음 |
| VLA의 action chunk boundary가 ACE / caching의 ideal granularity인가? | Adaptive Action Chunking (2026-04) 제기한 문제, ACE-VLA의 direct question |
| Joint token-expert budget optimizer의 closed-form 존재 여부 | 여전히 미해결, I3 papers에서 정면 접근 시 contribution |
| Visual encoder + LM decoder + expert skip의 end-to-end orchestration | SemanticVLA/CogVLA/ACE-VLA 세 방향이 병렬 전개 중 |

_본 보강은 이 세션의 Related Work 강화를 위해 WebFetch(arxiv API)로 수집. 향후 Mode 2/3 실행 시 매번 관련 분야 최신 6-12개월 논문을 재검색할 것을 권장 (`references/related-work-search.md` 워크플로우 참조)._
