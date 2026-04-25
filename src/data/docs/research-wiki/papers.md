# Papers Analyzed

`aica-research-bot` 실행 중 분석된 모든 논문을 시간 역순(최신 위)으로 기록한다.

---

## 2026-04-24 Mode 1 Session — MoE Fingerprint Security+Serving

Step 0 Phase + novelty-reviewer 추가 search = 35+ 편 참조. Peer-reviewed ratio ~38-50% (baseline 에 따라 다름). 주요 신규 (이전 세션 중복 제외):

### Training-free LLM guard
- **FJD "LLM Jailbreak Detection for (Almost) Free!"** — [arXiv:2509.14558](https://arxiv.org/abs/2509.14558), **EMNLP 2025 Findings**. Dense LLM 의 first-token logit + temperature scaling + affirmative instruction prepending 으로 training-free jailbreak detection. MoE routing signal 미활용. **DISCRETE-VEIL'/BEACON-GUARD-Lite baseline 필수**.
- **OmniGuard** — [arXiv:2505.23856](https://arxiv.org/abs/2505.23856) (preprint, multi-lang/modal). 주 LLM 의 internal representation classifier + 120x faster than the next fastest baseline. Dense 중심, MoE routing 미사용. **BEACON-GUARD-Lite 와 직접 비교**.
- **HSF** — [arXiv:2409.03788](https://arxiv.org/abs/2409.03788). Hidden State Filtering, 고정 layer jailbreak defense.
- **HiddenDetect** — [arXiv:2502.14744](https://arxiv.org/abs/2502.14744). VLM hidden state monitoring.
- **Jailbreaking Leaves a Trace** — [arXiv:2602.11495](https://arxiv.org/abs/2602.11495). Tensor-based latent representation, training-free inference-time detection. Dense 만.
- **Do Internal Layers of LLMs Reveal Patterns** — [arXiv:2510.06594](https://arxiv.org/abs/2510.06594). Hidden state layer-wise tensor. GPT-J/Mamba2. MoE 안 다룸.
- **MultiTaskGuard/UniGuard** — [arXiv:2504.19333](https://arxiv.org/abs/2504.19333). Multi-task guard + LoRA-shared, 14x faster. **BEACON-GUARD 의 concurrent (50-60% 겹침)**.
- **WildGuard** — [arXiv:2406.18495](https://arxiv.org/abs/2406.18495), **NeurIPS 2024**. Mistral-7B fine-tune safety classifier. 별도 forward +200ms +28GB. **baseline 필수**.

### MoE safety + routing (MoE-specific)
- **Task-Conditioned Routing Signatures in Sparse MoE** — [arXiv:2603.11114](https://arxiv.org/abs/2603.11114). **arXiv preprint only (2026-03-11, Mynampati 단독 author)**, OLMoE-1B-7B single, logistic regression on routing signatures, 4-way task classification 92.5±6.1%. Safety 안 다룸. **BEACON-GUARD domain 파트와 60-70% 겹침 (concurrent)**.
- **What Gets Activated: Domain/Driver Experts** — [arXiv:2601.10159](https://arxiv.org/abs/2601.10159). Expert 3 type 분류, routing weight 조정으로 +2-3% accuracy.
- **L³ (Large Language Lobotomy)** — [arXiv:2602.08741](https://arxiv.org/abs/2602.08741). Routing pattern → refusal expert silencing = **attack**. 8 MoE LLMs 평가. **DISCRETE-VEIL 방향과 정면 반대 (detection vs attack), scoop 아님**.
- **SAFEx** — [arXiv:2506.17368](https://arxiv.org/abs/2506.17368), **NeurIPS 2025 Poster**. Stability-based Expert Selection + LoRA finetune. 본 연구 "no finetune" 기준 대척.
- **RASA** — [arXiv:2602.04448](https://arxiv.org/abs/2602.04448). Routing-aware safety alignment (fine-tune).
- **SteerMoE** — [arXiv:2509.09660](https://arxiv.org/abs/2509.09660). Expert (de)activation steering. 공격+방어 겸용.
- **RouteMark** — [arXiv:2508.01784](https://arxiv.org/abs/2508.01784). JS divergence fingerprint for IP attribution.
- **MoE Lens** — [arXiv:2603.05806](https://arxiv.org/abs/2603.05806), **ICLR 2025 Workshop**. Top-1 expert only → cosine 0.95. 모델별 fingerprint 품질 해석.
- **GateBreaker** — [arXiv:2512.21008](https://arxiv.org/abs/2512.21008) (2025-12). MoE safety expert 3% 뉴런 ablation 만으로 ASR 7.4→64.9%. **DISCRETE-VEIL 의 가장 간접 threat (weight-level, embedding-PGD 아님)** — attack 공간 Venn 에서 구분.
- **Expert Selections Reveal Text** — [arXiv:2602.04105](https://arxiv.org/abs/2602.04105). Clean routing index 로 token 91.2% 복원 (privacy attack). "Routing carries rich signal" 전제 강화 + reviewer 의 privacy concern 공격 가능.

### Adaptive adversarial
- **Obfuscated Activations Bypass Latent-Space Defenses** — [arXiv:2412.09565](https://arxiv.org/abs/2412.09565). Hidden probe / SAE / representation-probing / latent OOD 모두 adaptive PGD 에 **recall 100% → 0% 붕괴**. Dense only. **DISCRETE-VEIL' 의 차별화 동기 핵심**.
- **V-MoE Adversarial Robustness** — [OpenReview Fd05J4Bu5Sp](https://openreview.net/pdf?id=Fd05J4Bu5Sp) (Puigcerver et al.). V-MoE image PGD 가 dense ViT robustness trend 따라감. **DISCRETE-VEIL 가설과 직접 상충 가능** — attack 공간 Venn diagram 으로 image PGD vs text LLM embedding-PGD 구분 의무.

### MoE serving
- **MoE-Infinity** — [arXiv:2401.14361](https://arxiv.org/abs/2401.14361), **USENIX ATC 2024**. Sequence-level expert activation tracing + activation-aware prefetching/caching. **LOOM' 주요 baseline**.
- **ProMoE** — [arXiv:2410.22134](https://arxiv.org/abs/2410.22134). Proactive prefetch stride predictor ~65%.
- **DuoServe-MoE** — [arXiv:2509.07379](https://arxiv.org/abs/2509.07379). Dual-phase prefetch + affinity routing.
- **PreScope** — [arXiv:2509.23638](https://arxiv.org/abs/2509.23638). Cross-layer prefetch, throughput 141%, decode latency -74.6%.
- **MoE-Beyond** — [arXiv:2508.17137](https://arxiv.org/abs/2508.17137). Learning-based expert prediction 97.5% (DeepSeek-V2-Chat-Lite).
- **Pre-Attention Expert Prediction** — [arXiv:2511.10676](https://arxiv.org/abs/2511.10676). 2-linear function + ranking loss, DeepSeek 93% / Qwen3 95%.
- **BuddyMoE** — [arXiv:2511.10054](https://arxiv.org/abs/2511.10054). Miss fallback via functional-similar resident substitute.
- **Semantic Parallelism** — [arXiv:2503.04398](https://arxiv.org/abs/2503.04398). SGLang within-replica affinity reorder.
- **Gimbal** — [arXiv:2602.21626](https://arxiv.org/abs/2602.21626), **MLSys 2026**. Prefix+KV+stickiness dispatcher. LOOM' baseline.
- **METRO** — [arXiv:2512.09277](https://arxiv.org/abs/2512.09277). Token-level SLO-aware routing.
- **In-depth MoE Caching/Prefetching** — [arXiv:2511.05814](https://arxiv.org/abs/2511.05814). LRU/LFU analysis.
- **OD-MoE** — [arXiv:2512.03927](https://arxiv.org/abs/2512.03927). Predictor 99.94%.

### Benchmarks + evaluation
- **WildJailbreak** (Ai2) — [arXiv:2406.18510](https://arxiv.org/abs/2406.18510). 253K train + 2,210 original eval. 본 실험의 핵심 safety benchmark (v2 balanced 8K eval).
- **JailbreakBench** — [arXiv:2404.01318](https://arxiv.org/abs/2404.01318). 100 behaviors, PAIR/GCG/AutoDAN 재현.
- **HarmBench** — [arXiv:2402.04249](https://arxiv.org/abs/2402.04249). 200 val prompts.
- **MMLU** — [arXiv:2009.03300](https://arxiv.org/abs/2009.03300). 57 subjects × 4 categories.
- **Azaria & Mitchell True-False** — 2023 hidden state probe baseline.

### Production / blog / industry
- **vLLM Semantic Router v0.1 Iris (2026-01 merge)** — vLLM project production blog. LoRA-shared classifier heads (jailbreak + domain + PII + fact-check) pluggable. **LOOM/BEACON-GUARD 와 framing 중복 (concurrent 50-55%)**.
- **ASE 2025 NIER "Unseen data detection via routing entropy"** — training-free MoE OOD via routing entropy. **BEACON-GUARD Mech 2.2 / DISCRETE-VEIL Mech 1.2 entropy tripwire 와 50-60% 겹침**.

---

## 2026-04-24 Mode 1 Session — Qwen3-VL (DeepStack) + Qwen3.5 Edge 최적화

3 PDF 입력 + 3 experts 병렬 + Step 0-α 신규 규칙 적용. 70+ 편, peer-reviewed 55%. 주요 신규 (이전 세션 중복 제외):

### Qwen3-VL Technical Report (input paper 1)
- **arXiv**: [2511.21631](https://arxiv.org/abs/2511.21631) | **Date**: 2025-11 | **Authors**: Qwen Team
- **Contribution**: 3 core upgrade — Interleaved MRoPE / DeepStack integration (visual_indexes=[8,16,24]) / explicit video timestamp. Dense (2B/4B/8B/32B) + MoE (30B-A3B/235B-A22B). 256K native.
- **Relevance**: 본 세션의 **primary base model**. I1 Mangrove + I2 Loom + I3 Vault' 모두 직접 활용.

### DeepStack (input paper 2)
- **arXiv**: [2406.04334](https://arxiv.org/abs/2406.04334) | **Venue**: NeurIPS 2024 | **peer-reviewed**
- **Contribution**: visual tokens N group stack → N decoder layer residual 주입. Context length 1/5 만으로 baseline 수준. Early decoder layer 가 visual processing 에 효과적.
- **Relevance**: **I1 Mangrove primary baseline**.

### Qwen3.5-Omni Technical Report (input paper 3)
- **arXiv**: [2604.15804](https://arxiv.org/abs/2604.15804) | **Date**: 2026-04-22
- **Contribution**: Thinker-Talker, Hybrid MoE + Gated Delta Net (GDN), ARIA, multi-codebook RVQ + MTP, Code2Wav ConvNet.
- **Relevance**: T1 Gale + T2 Forge primary. T2 Forge 는 본 세션 1일 전 공개로 scoop risk 있어 2026-05 CAL submit 권고.

### Cross-Layer Injection — CLI
- **arXiv**: [2601.10710](https://arxiv.org/abs/2601.10710) | **Date**: 2026-01
- **Contribution**: DeepStack 후속 algorithm-level paper.
- **Relevance**: **I1 Mangrove 필수 cite + orthogonal axis 차별화 근거**.

### Revisiting Multimodal Positional Encoding (MRoPE-I)
- **arXiv**: [2510.23095](https://arxiv.org/abs/2510.23095) | **Venue**: ICLR 2026 | **peer-reviewed**
- **Contribution**: MRoPE frequency allocation 재설계 algorithm-level.
- **Relevance**: **I2 Loom baseline** — algorithm only, kernel-level 공백 유지.

### Gated Delta Networks (GDN)
- **arXiv**: [2412.06464](https://arxiv.org/abs/2412.06464) | **Venue**: ICLR 2025 | **peer-reviewed**
- **Contribution**: Constant-memory recurrent linear attention.
- **Relevance**: **T1 Gale baseline**. Qwen3-Next/3.5 에 적용됨.

### VEQ (Qwen3-VL MoE modality-adaptive quantization)
- **arXiv**: [2602.01037](https://arxiv.org/abs/2602.01037) | **Date**: 2026-02
- **Relevance**: **I3 Vault' scoop source** (68-72% overlap) — NVFP4 per-expert axis drop motivation.

### ARCQuant (NVFP4 quantization)
- **arXiv**: [2601.07475](https://arxiv.org/abs/2601.07475) | **Date**: 2026-01
- **Relevance**: I3 Vault' scoop source.

### DyMoE (edge MoE mixed-precision)
- **arXiv**: [2603.19172](https://arxiv.org/abs/2603.19172) | **Date**: 2026-03
- **Relevance**: I3 Vault' scoop source.

### CC-MoE (Communication-Compute MoE)
- **arXiv**: [2509.25689](https://arxiv.org/abs/2509.25689) | **Date**: 2025-09
- **Relevance**: I3 Vault' adjacent baseline.

### Four Over Six / DynaExq / HOBBIT / OD-MoE — Edge MoE Adjacent
- [arXiv:2512.02010](https://arxiv.org/abs/2512.02010) Four Over Six / [arXiv:2511.15015](https://arxiv.org/abs/2511.15015) DynaExq / [arXiv:2411.01433](https://arxiv.org/abs/2411.01433) HOBBIT / [arXiv:2512.03927](https://arxiv.org/abs/2512.03927) OD-MoE
- **Relevance**: I3 Vault' baseline.

### VLCache (T3 Echo scoop)
- **arXiv**: [2512.12977](https://arxiv.org/abs/2512.12977) | **Date**: 2025-12-15
- **Contribution**: 2% vision + 98% reuse + pixel hash + encoder cache.
- **Relevance**: **T3 Echo 72-78% direct scoop → DROP 결정 요인**.

### TTKV — 2-tier KV
- **arXiv**: [2604.19769](https://arxiv.org/abs/2604.19769) | **Date**: 2026-04
- **Relevance**: T1 Gale adjacent (3-tier 공백).

### EdgeReasoning IISWC 2025
- **arXiv**: [2511.01866](https://arxiv.org/abs/2511.01866) | **Venue**: IISWC 2025 | **peer-reviewed**
- **Relevance**: **Step 0-α** IISWC 수집.

### LLMServingSim IISWC 2024
- **arXiv**: [2408.05499](https://arxiv.org/abs/2408.05499) | **Venue**: IISWC 2024 | **peer-reviewed**
- **Relevance**: Step 0-α IISWC.

### LLM Inferencing Edge Accelerators PAISE 2025
- **arXiv**: [2506.09554](https://arxiv.org/abs/2506.09554) | **Venue**: PAISE 2025 | **peer-reviewed**
- **Relevance**: Step 0-α ISPASS/PAISE.

### Jetson Thor 7x Gen AI Blog / MoE forum / TensorRT Edge-LLM / NVFP4 — Vendor reports (Step 0-α)
- [Jetson Thor 7x Gen AI](https://developer.nvidia.com/blog/unlock-faster-smarter-edge-models-with-7x-gen-ai-performance-on-nvidia-jetson-agx-thor/)
- [Jetson Thor vLLM MoE gap forum](https://forums.developer.nvidia.com/t/jetson-agx-thor-vllm-26-02-moe-performance-significantly-below-reference-missing-fused-moe-config/364663) — **I3 Vault 산업 motivation**
- [TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)
- [NVFP4 Introduction](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

### MLPerf Inference v5.0 / v5.1 — Benchmark
- [MLPerf v5.0](https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/) / [MLPerf v5.1](https://mlcommons.org/2025/09/mlperf-inference-v5-1-results/)
- **Relevance**: Step 0-α benchmark.

### vLLM Qwen3-Next Blog / vLLM EPD / Qwen3-VL-30B-A3B Ascend
- [vLLM Qwen3-Next](https://blog.vllm.ai/2025/09/11/qwen3-next.html)
- [vLLM EPD](https://blog.vllm.ai/2025/12/15/vllm-epd.html)
- [Qwen3-VL-30B-A3B vLLM-Ascend](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-VL-30B-A3B-Instruct.html)
- **Relevance**: Step 0-α platform.

---

## 2026-04-23 Mode 1 Session — 에너지 효율적 Edge VLM Inference

3 experts (ai-opt + legacy-sys + hw-pim) 병렬 탐색, 65+ 편, peer-reviewed 52%. 주요 신규 (이전 세션과 중복 제외):

### CodecSight — Video VLM with NVDEC Motion Vector + RoPE Correction KVC (I1 Tidal scoop source)
- **arXiv**: [2604.06036](https://arxiv.org/abs/2604.06036) | **Date**: 2026-04-07 | **Authors**: Yulin Zou 등
- **Contribution**: NVDEC motion vector 를 codec-guided token pruning signal 로 사용 + RoPE position-correction selective KVC refresh.
- **Relevance**: **I1 Tidal 68-72% direct scoop** (2026-04-23 세션의 주요 발견). 16일 차이로 precedence 불가.

### NANOMIND — Tiny but Mighty Model Scheduling
- **arXiv**: [2510.05109](https://arxiv.org/abs/2510.05109) | **Date**: 2025-10
- **Contribution**: Module-level brick scheduling for tiny model, UMA + module offload. LLaVA-OneVision 42.3% 에너지 절감 claim.
- **Relevance**: **I3 Triptych 60% concurrent**. DLA 축은 없음 → Triptych repositioning 기준.

### Nova — Real-Time Agentic VLM Serving
- **arXiv**: [2509.21301](https://arxiv.org/abs/2509.21301) | **Date**: 2025-09
- **Contribution**: VLM 3-stage elastic partitioning, desktop GPU.
- **Relevance**: **I3 Triptych 60% concurrent**. DLA 미사용 → Triptych 가 DLA 축으로 차별화.

### HeteroInfer / HeteroLLM — Heterogeneous LLM Serving
- **arXiv**: [2501.14794](https://arxiv.org/abs/2501.14794) | **Date**: 2025-01
- **Contribution**: Heterogeneous compute unit scheduling for LLM (server-class).
- **Relevance**: I3 Triptych 55% concurrent, edge + DLA 차별화.

### ELANA — Edge LLM Autoscaling Thor+Orin Nano
- **arXiv**: [2512.09946](https://arxiv.org/abs/2512.09946) | **Date**: 2025-12
- **Contribution**: Jetson Thor + Orin Nano cross-arch 일반 LLM serving.
- **Relevance**: T3 Verge 45% adjacent. VLM-specific stage breakdown 없음 → Verge 차별화.

### Nanomind (동일) / HydraInfer / LiteVLM / llm.npu — Edge VLM Serving Adjacent
- [HydraInfer arXiv:2505.12658](https://arxiv.org/abs/2505.12658) / [LiteVLM arXiv:2506.07416](https://arxiv.org/abs/2506.07416) / [llm.npu arXiv:2407.05858](https://arxiv.org/abs/2407.05858)
- I3 Triptych 및 Parquet baseline.

### DynamoLLM — HPCA 2025, LLM Inference Energy Management
- **arXiv**: [2408.00741](https://arxiv.org/abs/2408.00741) | **Venue**: HPCA 2025 | **peer-reviewed**
- **Relevance**: I2 Parquet primary baseline. VLM AnyRes 미반영.

### GreenLLM — SLO-aware DVFS
- **arXiv**: [2508.16449](https://arxiv.org/abs/2508.16449) | **Date**: 2025-08
- **Relevance**: I2 Parquet baseline.

### PolyThrottle — MLSys 2024
- **arXiv**: [2310.19991](https://arxiv.org/abs/2310.19991) | **Venue**: MLSys 2024 | **peer-reviewed**
- **Relevance**: I2 Parquet baseline (DNN throttle).

### throttLL'eM / BiScale / SparseDVFS — DVFS Adjacent
- [arXiv:2408.05235](https://arxiv.org/abs/2408.05235) / [arXiv:2602.18755](https://arxiv.org/abs/2602.18755) / [arXiv:2603.21908](https://arxiv.org/abs/2603.21908)
- **Relevance**: I2 Parquet baseline.

### MBQ — CVPR 2025, Modality-Balanced Quantization
- **arXiv**: [2412.19509](https://arxiv.org/abs/2412.19509) | **Venue**: CVPR 2025 | **peer-reviewed**
- **Contribution**: Modality-balanced per-layer quantization for VLM.
- **Relevance**: I2 Parquet 차별화 (per-layer → per-tile 확장).

### Jetson Orin LLM profiling — Arya & Simmhan
- **arXiv**: [2506.09554](https://arxiv.org/abs/2506.09554) | **Date**: 2025-06
- **Relevance**: T3 Verge baseline.

### TokenPowerBench — Token-level LLM Power
- **arXiv**: [2512.03024](https://arxiv.org/abs/2512.03024) | **Date**: 2025-12
- **Relevance**: T3 Verge adjacent.

### E4 — Energy Efficient Edge AI (AAAI 2025)
- **arXiv**: [2503.04865](https://arxiv.org/abs/2503.04865) | **Venue**: AAAI 2025 | **peer-reviewed**
- **Relevance**: T3 Verge baseline.

### Watt Counts — LLM Energy Benchmarks
- **arXiv**: [2604.09048](https://arxiv.org/abs/2604.09048) | **Date**: 2026-04
- **Relevance**: T3 Verge adjacent.

### Blackwell Microbench
- **arXiv**: [2512.02189](https://arxiv.org/abs/2512.02189) | **Date**: 2025-12
- **Relevance**: T3 Verge adjacent (Blackwell only).

### Qwen2.5-VL Technical Report
- **arXiv**: [2502.13923](https://arxiv.org/abs/2502.13923) | **Date**: 2025-02
- **Contribution**: MRoPE 3D + dynamic resolution + video temporal packing.
- **Relevance**: **최신 VLM 아키텍처 기준 모델** (Parquet / Triptych / Cartographer / T2 Sift 모두 대상).

### Qwen2-VL / LLaVA-1.5 / LLaVA-OneVision / InternVL3 / Molmo / SmolVLM / FastVLM / MiniCPM-V — VLM Architecture Evolution
- [arXiv:2409.12191](https://arxiv.org/abs/2409.12191) Qwen2-VL / [arXiv:2310.03744](https://arxiv.org/abs/2310.03744) LLaVA-1.5 / [arXiv:2408.03326](https://arxiv.org/abs/2408.03326) LLaVA-OneVision / [arXiv:2504.10479](https://arxiv.org/abs/2504.10479) InternVL3 / [arXiv:2409.17146](https://arxiv.org/abs/2409.17146) Molmo / [arXiv:2504.05299](https://arxiv.org/abs/2504.05299) SmolVLM / [arXiv:2412.13303](https://arxiv.org/abs/2412.13303) FastVLM [CVPR 2025].
- **Relevance**: 초기 vs 최신 diff 분석 기준 모델.

### Revisiting MRoPE / VideoRoPE / VRoPE — MRoPE 변형 연구
- [arXiv:2510.23095](https://arxiv.org/abs/2510.23095) / [arXiv:2502.05173](https://arxiv.org/abs/2502.05173) / [arXiv:2502.11664](https://arxiv.org/abs/2502.11664)
- **Relevance**: T1 Cartographer 차별화 비교 (LUT 축 공백 증명).

### SAIL — SRAM-LUT GEMV
- **arXiv**: [2509.25853](https://arxiv.org/abs/2509.25853) | **Date**: 2025-09
- **Relevance**: T1 Cartographer adjacent (LUT 축).

### InternVL-X — RVTC Visual Token Compression
- **arXiv**: [2503.21307](https://arxiv.org/abs/2503.21307) | **Date**: 2025-03
- **Relevance**: T2 Sift primary competitor (entropy-adaptive 미포함).

### PyramidDrop / VisionZip / FastV / SparseVLM — Visual Token Pruning Adjacent
- [arXiv:2410.17247](https://arxiv.org/abs/2410.17247) / [arXiv:2412.04467](https://arxiv.org/abs/2412.04467) / [arXiv:2403.06764](https://arxiv.org/abs/2403.06764) / [SparseVLM ICML 2025](https://openreview.net/forum?id=80faIPZ67S)
- **Relevance**: T2 Sift baseline.

### LiteVLM / Nova / VL-Cache / SparseVILA — Edge VLM Pipeline
- [arXiv:2506.07416](https://arxiv.org/abs/2506.07416) / [arXiv:2509.21301](https://arxiv.org/abs/2509.21301) / [VL-Cache ICLR 2025](https://openreview.net/forum?id=HMrcv7Q4Ub) / [arXiv:2510.17777](https://arxiv.org/abs/2510.17777)
- **Relevance**: I3 Triptych baseline.

### NVIDIA DLA on Jetson Orin / NVFP4 / TensorRT Edge-LLM / Jetson Thor — Vendor References
- NVIDIA Dev Blog + docs
- **Relevance**: I3 Triptych + T3 Verge HW 근거.

---

## 2026-04-23 Mode 1 Session (v3) — VLM/VLA Context-aware Caching & Serving (v1/v2 improve)

v1 (55+ 편) + v2 (OpenReview 3 편) 에서 이미 등재된 논문은 생략. v3 에서 신규 실증 확인된 논문만 기록.

### Mosaic — Cross-Modal Clustering for Long-Context Video VLM KV
- **arXiv**: [2604.10060](https://arxiv.org/abs/2604.10060) | **Date**: 2026-04-11 | **Authors**: Tuowei Wang 등
- **Contribution**: Cross-modal KV clustering (content-axis dedup) for long-context video VLM serving.
- **Relevance**: **HRTS+ concurrent 55-65% (v3 신규 발견)** — content-axis vs HBM physical axis 차별화 + stacking ablation. ContextMIG+ 에도 baseline 편입.

### Harvest — Peer-to-Peer GPU Caching via NVLink
- **arXiv**: [2602.00328](https://arxiv.org/abs/2602.00328) | **Date**: 2026-01-30 | **Authors**: (연구팀 pending, WebSearch 확인)
- **Contribution**: NVLink peer-fetch primitive for distributed GPU cache sharing.
- **Relevance**: **NACK-Gossip Tier-2 concurrent 55-65% (critical)** — v2 cutoff 이전 논문이나 누락. NACK-Gossip Tier-2 미선정 전환의 주요 요인. VLA-specific action latency SLO + TTL lease + pull-batch 3축 delta 명시 필요.

### FlashVLA — Token-Level Action Reuse for VLA
- **arXiv**: [2505.21200](https://arxiv.org/abs/2505.21200) | **Date**: 2025-05 | **Authors**: (공개 메타데이터 WebSearch 확인)
- **Contribution**: VLA serving 에서 "action reuse" 개념을 직접 명명, token-level reuse 구현.
- **Relevance**: **DeadlineCOW Top-tier scoop 접경 68-72% (v3 신규)** — KVShare 보다 강한 경쟁자. PhaseGraph-VLA+ 는 graph-level switch 축으로 차별화.

### Predictable LLM Serving on GPU Clusters
- **arXiv**: [2508.20274](https://arxiv.org/abs/2508.20274) | **Date**: 2025-08
- **Contribution**: GPU cluster scale 에서 dynamic MIG scheduling.
- **Relevance**: **ContextMIG+ concurrent 55-60% (v3 신규)** — cluster-level MIG vs intra-GPU MIG dual-issue 차별화. Baseline 필수 편입.

### Prefill-as-a-Service
- **arXiv**: [2604.15039](https://arxiv.org/abs/2604.15039) | **Date**: 2026-04
- **Contribution**: Prefill 단계를 서비스로 분리.
- **Relevance**: ContextMIG+ related work 보강, coalesce + dual-issue 와 orthogonal.

### IceCache
- **arXiv**: [2604.10539](https://arxiv.org/abs/2604.10539) | **Date**: 2026-04
- **Contribution**: Dedicated cold cache layer for LLM serving.
- **Relevance**: ContextMIG+ related work, orthogonal 축.

### Execution-Idle Characterization
- **arXiv**: [2604.04745](https://arxiv.org/abs/2604.04745) | **Date**: 2026-04
- **Contribution**: LLM serving 에서 19.7% execution idle 이 10.7% energy waste 에 해당.
- **Relevance**: **B1 GCReconfProfile motivation citation** — Green Context reconfig 가 idle 구간 recover 가능한 cost 제공.

### VidKV — Video-Specific KV Cache (v2 "VidKV 2511" placeholder 의 실제 대응 논문)
- **arXiv**: [2503.16257](https://arxiv.org/abs/2503.16257) | **Date**: 2025-03 | **Authors**: Keda Tao 등
- **Contribution**: Video VLM KV cache compression.
- **Relevance**: HRTS+ adjacent 35-40%.

### SparseServe (v2 "DRAM-aware attention" placeholder 의 실제 대응)
- **arXiv**: [2509.24626](https://arxiv.org/abs/2509.24626) | **Date**: 2025-09
- **Relevance**: HRTS+ Tier-2 baseline 편입, 50-60% concurrent.

### KEEP (v2 "RoboFleet-Sync" placeholder 의 실제 대응)
- **arXiv**: [2602.23592](https://arxiv.org/abs/2602.23592) | **Date**: 2026-02
- **Relevance**: NACK-Gossip adjacent 35%.

### OpenVLA-OFT (v3 Tier-2 B3 baseline)
- **arXiv**: [2502.19645](https://arxiv.org/abs/2502.19645) | **Date**: 2025-02 | **Authors**: Stanford/Meta
- **Contribution**: Parallel decoding 26× 가속, parallel-sample 수준.
- **Relevance**: B3 ActHeadFuse 주요 비교 대상 — kernel-level breakdown 미공개 gap 을 B3 가 보완.

### MIGER (v3 B1 baseline)
- **Venue**: ICPP 2024 / [link](https://dl.acm.org/doi/pdf/10.1145/3673038.3673089) | **peer-reviewed**
- **Contribution**: MIG reconfig overhead characterization, 2.3% overhead at ms-level.
- **Relevance**: B1 GCReconfProfile baseline — MIG (ms) vs Green Context (μs) 4 orders 격차 명시.

### Characterizing Power Management for LLMs in the Cloud (v3 B2 baseline)
- **Venue**: ASPLOS 2024 / [link](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/GPU_Power_ASPLOS_24.pdf) | **peer-reviewed**
- **Contribution**: LLM serving phase-transition energy 분석.
- **Relevance**: B2 TokenEvictEnergy methodology 기준.

### TokenPowerBench (v3 B2 reference)
- **arXiv**: [2512.03024](https://arxiv.org/abs/2512.03024) | **Date**: 2025-12
- **Contribution**: LLM phase-aware power benchmark.
- **Relevance**: B2 reference, visual token 미포함 gap 을 B2 가 보완.

---

## 2026-04-23 Mode 1 Session — PRISM(ISLPED'26) → VLM/VLA Binarization/Ternarization + KV Cache 확장

3명 전문가 (algorithm + ai-optimization + hw-pim-accelerator) 병렬 dispatch. 60+ 논문 탐색, peer-reviewed 55%. 이하 주요 신규 논문만 기록 (이전 세션과 중복 제외).

### PRISM: Decompose-then-Merge Scaling Factors for Accurate and Energy-Efficient CIM-friendly BNN
- **Paper**: ISLPED'26 submission (Anonymous Author, local `/home/yhgong/paper/PRISM/PRISM_ISLPED26.pdf`) | **Date**: 2026-04 submission
- **Contribution**: Xbar-wise SF decomposition (rank-r S1·S2) + OPTIC inlier compression + LUT-based multiply bypass. BNN CIFAR-10/100, Tiny-ImageNet 에서 ResNet-20/18 에 +0.19 ~ +2.69% accuracy 개선, 1.8-3.4× TOPS/W. 본 세션의 **input seed paper**.

### BitVLA: 1-bit VLA Models for Robotics Manipulation
- **arXiv**: [2506.07530](https://arxiv.org/abs/2506.07530) | **Date**: 2025-06 | **Authors**: USTC, Microsoft | **Venue**: arXiv preprint
- **Contribution**: 모든 parameter 를 {-1,0,+1} ternary 로 quantize한 VLA. Quantize-then-Distill 로 vision encoder ternary, 메모리 11× / latency 4.4× 감소.
- **Relevance**: I1/I3 의 primary baseline + backbone. 1.58-bit 가 full precision OpenVLA-OFT 와 matching 하지만 KV cache 는 FP16 유지 → I2 의 KV extreme quant 여백.

### AKVQ-VL: VLM 2-bit KV Cache Quantization via WHT + Pivot Token
- **arXiv**: [2501.15021](https://arxiv.org/abs/2501.15021) | **Date**: 2025-01 | **Venue**: arXiv preprint (검증 pending peer-review venue)
- **Contribution**: VLM KV 2-bit + Walsh-Hadamard Transform + pivot token saliency.
- **Relevance**: I2' 의 closest VLM-KV competitor (45% 일치). ternary/LUT 없음 → I2' 의 차별화 축.

### MDBF: Multi-Dimensional Binary Factorization for LLMs
- **arXiv**: [2512.24545](https://arxiv.org/abs/2512.24545) | **Date**: 2025-12 | **Venue**: arXiv preprint
- **Contribution**: 1-bit sign base 공유 + rank-l envelope factorization. weight-space factorization (W ≈ sign(W)⊙uv^T).
- **Relevance**: I1' 의 가장 인접한 prior work (55-60% 일치). 차별화 축: weight-space vs scale-space Hessian.

### LoRDS: Continuous Low-Rank Decomposed Scaling for LLM Quantization
- **arXiv**: [2601.22716](https://arxiv.org/abs/2601.22716) | **Date**: 2026-01 | **Venue**: arXiv preprint
- **Contribution**: Continuous low-rank scale S=BA, unified PTQ+QAT+PEFT. 27% acc 개선 @3bit, 1.5× RTX4090 speedup.
- **Relevance**: I1' 의 closest LLM-only prior art (50-60% 일치). VLM modality-split 공백.

### DyQ-VLA: Temporal-Dynamic-Aware VLA Quantization via Kinematic Proxy
- **arXiv**: [2603.07904](https://arxiv.org/abs/2603.07904) | **Date**: 2026-03 | **Venue**: arXiv preprint
- **Contribution**: Kinematic proxy (joint velocity) 로 bit-width switch (binary {2,4}). temporal-aware VLA quantization.
- **Relevance**: **I3' 65%+ 일치 — 주요 scoop 원인**. I3' Major Revision → Deferred 의 핵심 근거.

### SD-VLA: Static-Dynamic Disentanglement for VLA KV Cache
- **arXiv**: [2602.03983](https://arxiv.org/abs/2602.03983) | **Date**: 2026-02 | **Venue**: arXiv preprint
- **Contribution**: Static/dynamic binary disentanglement + recache gate + LIBERO-Memory benchmark.
- **Relevance**: I3' 55% 일치. continuous vs binary 분리 증명 필요.

### QVLA: Action-Space Sensitivity Channel Bit Allocation for VLA
- **arXiv**: [2602.03782](https://arxiv.org/abs/2602.03782) | **Date**: 2026-02 | **Venue**: ICLR 2026 poster
- **Contribution**: Action-space sensitivity channel-wise bit allocation. DiT action head quantization 최초.
- **Relevance**: I3' 60% 일치 (action-head 축). rank-decoupling + temporal-SF 부재.

### AQPIM: DRAM-PIM Activation PQ Quantization with DRAM-Row-Resident LUT
- **arXiv**: [2604.18137](https://arxiv.org/abs/2604.18137) | **Date**: 2026-04 | **Venue**: HPCA'26
- **Contribution**: PQ-based codebook LUT in DRAM row buffer, inner-product pre-computation (Q-dependent runtime).
- **Relevance**: I4' 40-50% 일치. **LUT 구조 다름** (AQPIM: inner-product pre / I4': dequant scalar pre). 차별화 근거.

### P3-LLM: PIM-GPU Hybrid W4A8KV4P8 LLM Accelerator
- **arXiv**: [2511.06838](https://arxiv.org/abs/2511.06838) | **Date**: 2025-11 | **Venue**: arXiv preprint
- **Contribution**: W4A8KV4P8 precision schedule, PIM-GPU hybrid, HBM-PIM 대비 4.9× speedup. MAC-based dequant.
- **Relevance**: I4' 의 MAC dequant 치환 대상.

### LUT Tensor Core: Software-Hardware Co-Design for LUT-Based Low-Bit LLM
- **arXiv**: [2408.06003](https://arxiv.org/abs/2408.06003) | **Date**: 2024-08 | **Venue**: ISCA 2025
- **Contribution**: elongated tiling + bit-serial + LUT instruction. 1.44× perf density vs tensor core.
- **Relevance**: I2' 및 I4' 의 kernel 원리 reference. weight mpGEMM LUT, attention LUT 는 미탐구.

### T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM
- **arXiv**: [2407.00088](https://arxiv.org/abs/2407.00088) | **Date**: 2024-07 | **Venue**: EuroSys 2025 (검증 pending)
- **Contribution**: 1-bit weight group-of-4 partial sum LUT precompute, FMA 대체. llama.cpp 대비 4-5× speedup.
- **Relevance**: I2' cold path LUT attention 의 CPU fork 기반. GPU port 가 필요.

### Oaken: Online-Offline Hybrid KV Cache Quantization
- **arXiv**: [2503.18599](https://arxiv.org/abs/2503.18599) | **Date**: 2025-03 | **Venue**: ISCA 2025
- **Contribution**: Offline data-agnostic outlier threshold + online scale, dense+sparse fusion. 1.58× throughput.
- **Relevance**: I2'/I4' 의 인접 hardware-side prior. offline outlier 철학 공유 (PRISM OPTIC 과 dual).

### VL-Cache: Sparsity & Modality-Aware KV Cache for VLMs
- **arXiv**: [2410.23317](https://arxiv.org/abs/2410.23317) | **Date**: 2024-10 | **Venue**: ICLR 2025 poster
- **OpenReview**: [HMrcv7Q4Ub](https://openreview.net/forum?id=HMrcv7Q4Ub) (identity 4-point check passed)
- **Contribution**: VLM vision/text token prefill/decode sparsity + layer-adaptive cache budget. 10% 캐시로 98% accuracy, 2.33× end-to-end, 7.08× decode speedup.
- **Relevance**: I2' 의 primary stacking partner (sparsity × quantization orthogonal).

### GEAR: Efficient KV Cache Compression Recipe with Low-Rank + Sparse
- **arXiv**: [2403.05527](https://arxiv.org/abs/2403.05527) | **Date**: 2024-03 | **Venue**: NeurIPS 2024
- **Contribution**: Uniform quant + low-rank matrix for residual error + sparse outlier matrix.
- **Relevance**: I2' 의 가장 가까운 prior art (low-rank on **error residual**, I2' 는 low-rank on SF matrix). scoop risk 판정 핵심.

### KVTQ: Ternary KV Cache with ASIC Dequant
- **OpenReview**: [eZAlb8fX5y](https://openreview.net/pdf?id=eZAlb8fX5y) | **Status**: ICLR/NeurIPS 재review
- **Contribution**: K ternary / V 3-bit per-channel ASIC. GPU commodity 아님.
- **Relevance**: I2' 의 mathematical ancestor. GPU+rank-r+VLM 3축 공백.

---

## 2026-04-22 Mode 1 Session — VLM/VLA Context-aware Caching & Serving (pure GPU)

본 세션은 PIM 비의존 pure GPU serving stack 의 context-aware mechanism 에 집중. 55+ 논문 수집. 이전 세션과 중복되지 않은 **신규 등록 논문** 만 아래에 기록 (이전 세션 등록 논문은 reference 로만 활용).

### Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization
- **arXiv**: [2509.21301](https://arxiv.org/abs/2509.21301) | **Date**: 2025-09 | **Authors**: Yuhang Xu, Shengzhong Liu, Dong Zhang
- **Venue**: arXiv preprint
- **Relevance**: Phase 1 A1/L1 의 가장 가까운 system-level competitor. Single-GPU agentic VLM serving, vision-encode/prefill/decode 3-stage pipelining.
- **Contribution**: Elastic GPU spatial partitioning (SM 단위), vision encoder layer-wise CPU↔GPU weight swapping, Pareto-optimal latency-throughput calibration. 최대 23.3% max latency 개선, 14.6% avg latency.
- **Hidden insight**: Table 4 에서 **mixed workload (vision-heavy + text-heavy request 공존)** 시 improvement 가 23.3% → 11.7% 로 반토막. **Workload heterogeneity 만 언급, content-aware partitioning 은 안 함** → L1 ContextSM-Tri 의 GAP.
- **Differentiation vs L1/A1/A3**: Stage-aware partition (content-blind) vs L1 의 content-aware taxonomy × tri-knob. VLM 대상 vs A1 의 VLA trajectory phase. Single-agentic flow vs A3 의 multi-tenant cluster.
- **Action**: L1/A1/A3 모두 primary baseline 으로 포함 (reimplementation 필요).

### Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal LLM Inference (RPS-Serve)
- **arXiv**: [2603.26498](https://arxiv.org/abs/2603.26498) | **Date**: 2026-03 | **Authors**: Konstantinos Papaioannou, Thaleia Dimitra Doudali
- **Venue**: arXiv preprint
- **Relevance**: Multimodal request scheduling with size-based priority. L1 ContextSM-Tri 의 "content-aware" 차별화 대상.
- **Contribution**: Request 를 video=rock / image=pebble / text=sand 로 분류, dynamic priority + aging. 54% overall TTFT 감소, 78.5% latency-critical request 개선.
- **Hidden insight**: Size-based (request volume) priority 는 content semantic (OCR vs reasoning) 무시. 같은 size 라도 HW profile 다름.
- **Differentiation vs L1**: Size-based vs L1 의 semantics-based (6-class taxonomy). Scheduling layer vs L1 의 partition layer — orthogonal stackable.
- **Action**: L1 baseline 에 포함.

### AC²-VLA: Action-Context-Aware Adaptive Computation in VLA Models for Efficient Robotic Manipulation
- **arXiv**: [2601.19634](https://arxiv.org/abs/2601.19634) | **Date**: 2026-01 | **Authors**: Wenda Yu, Tianshi Wang, Fengling Li
- **Venue**: arXiv preprint
- **Relevance**: A1 PhaseGraph-VLA 의 가장 가까운 concurrent (algorithmic axis).
- **Contribution**: Action context (vision + language + prior action) → cognition reuse + token pruning + selective component execution. Action-guided self-distillation (training required). 1.79× speedup, 29.4% FLOPs.
- **Hidden insight**: Figure 6 adaptive computation ratio 가 step 에 따라 40-95% 변동, **시스템 자원 관점에서 어느 kernel 이 언제 idle 한지 논의 부재** → A1 의 system-axis GAP.
- **Differentiation vs A1**: Algorithmic (token prune + component skip) vs A1 의 system-level (CUDA Graph dispatcher + SSE). Orthogonal — stacking 가능.
- **Action**: A1 primary baseline.

### VLA-Cache: Adaptive Token Caching for VLA via Temporal Frame-Diff
- **arXiv**: [2502.02175](https://arxiv.org/abs/2502.02175) | **Date**: 2025-02 | **Authors**: Siyu Xu, Yunke Wang, Chenghao Xia et al.
- **Venue**: arXiv preprint
- **Relevance**: A1/L2 의 closest competitor. Temporal token reuse axis.
- **Contribution**: Training-free frame-diff 기반 token caching. Layer-adaptive reuse ratio (attention concentration). 1.7× CUDA latency, +15% Hz, task success drop 0.3%.
- **Hidden insight**: CacheFlow Table 3 재해석 — **grasping moment 직전 3-5 frame 에서 similarity 급락 (0.82 → 0.54), frame-diff 기반 eviction 시 정확도 2.3% drop**. 이 지점에서 action-imminence signal (L2) 가 필요.
- **Differentiation vs L2**: Frame-diff (pixel-level) vs L2 의 action-imminence β_a (gripper + curvature + object-dist semantic). Single-tier KV vs L2 의 3-tier.
- **Action**: A1/L2 primary baseline. Predictor study 에서 AUC 비교 대상.

### KV-Efficient VLA: RNN-Gated Chunked KV Cache
- **arXiv**: [2509.21354](https://arxiv.org/abs/2509.21354) | **Date**: 2025-09 | **Authors**: Wanshun Xu, Long Zhuang, Lianlei Shan
- **Venue**: arXiv preprint
- **Relevance**: A1 adjacent baseline.
- **Contribution**: Fixed-size chunk KV cache + RNN gating module summarize/filter historical context. 24.6% FLOPs↓, 1.87× memory savings.
- **Differentiation vs A1**: RNN gate (learned utility) vs A1 의 SSE (training-free Page-Hinkley). Chunk-level KV vs A1 의 graph-level dispatch.
- **Action**: A1 baseline.

### ADP-VLA: Action-aware Dynamic Pruning for Efficient VLA Manipulation
- **arXiv**: [2509.22093](https://arxiv.org/abs/2509.22093) | **Date**: 2025-09 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: A1 의 가장 가까운 concurrent (~65% 유사도).
- **Contribution**: Motion phase × task semantics → visual token pruning. Coarse vs fine-grained phase 구분 → 자원 할당 동적 변경.
- **Differentiation vs A1**: Token pruning vs CUDA Graph dispatch. Concept 유사하되 수단 다름.
- **Action**: A1 primary baseline, 65% concurrency 로 formal 차별 필수.

### Mosaic: Cross-Modal Clustering for Efficient Video Understanding
- **arXiv**: [2604.10060](https://arxiv.org/abs/2604.10060) | **Date**: 2026-04 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: **A3 SemClust-Share 의 scoop threat (75% 유사도)** — A3 repositioning 의 결정적 근거.
- **Contribution**: Cross-modal clustering (visual coherence + semantic relevance) 을 streaming video VLM 에 적용. Token-level KV clustering → offload/retrieval.
- **Differentiation vs A3**: Cluster detection axis (Mosaic 선점) vs **A3 v2 repositioning: COW + deadline SM yielding (Mosaic 미구현)**.
- **Action**: A3 v2 detector 는 Mosaic 차용 (novelty claim 포기), downstream COW/deadline 만 본 기여.

### KVShare: Semantic-Aware KV Cache Sharing for Efficient LLM Inference
- **arXiv**: [2503.16525](https://arxiv.org/abs/2503.16525) | **Date**: 2025-03 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: A3 scoop threat (72% 유사도) — cluster detection + token edit.
- **Contribution**: Multi-tenant semantic alignment + differential token editing. 60%+ 요청이 90% 토큰 재사용 가능.
- **Differentiation vs A3**: LLM-only (vision prefix 없음). Full-copy on write (COW 미구현). Deadline SM yielding 없음.
- **Action**: A3 baseline.

### MPIC: Position-Independent Multimodal Context Caching
- **arXiv**: [2502.01960](https://arxiv.org/abs/2502.01960) | **Date**: 2025-02 | **Authors**: Shiju Zhao, Junhao Hu, Rongxiao Huang
- **Venue**: arXiv preprint
- **Relevance**: A3 scoop threat (70% 유사도, position-independent) + previously 세션에서 이미 reference 됨.
- **Contribution**: Position-agnostic KV reuse, disk 저장 + 병렬 load + integrated recompute. 54% response time↓, 2× throughput.
- **Differentiation vs A3**: Disk tier (single-request response) vs A3 의 GPU-resident multi-tenant + COW.
- **Action**: A3 baseline.

### VLCache: Computing 2% Vision Tokens and Reusing 98%
- **arXiv**: [2512.12977](https://arxiv.org/abs/2512.12977) | **Date**: 2025-12 | **Authors**: Shengling Qin, Hao Yu, Chenxin Wu
- **Venue**: arXiv preprint
- **Relevance**: A2 SemTile-Dispatch adjacent. (이전 세션에서 F3 scoop 근거로 등재됨.) 본 세션에서는 A2 baseline + A3 partial reference.
- **Contribution**: Encoder output + KV cache 이중 pipeline, layer-aware dynamic recomputation. 1.2-16× TTFT speedup.
- **Differentiation vs A2**: Layer-aware (temporal) vs A2 의 per-tile spatial + L2 hint.
- **Action**: A2 baseline.

### HERMES: Hierarchical Memory for Streaming Video VLM
- **arXiv**: [2601.14724](https://arxiv.org/abs/2601.14724) | **Date**: 2026-01 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: **L2 TemporalTier-3 concurrent (65% 유사도)** — L2 tiebreak 패배의 핵심 요인.
- **Contribution**: Training-free hierarchical memory framework, multi-granularity KV for streaming video. Proactive retention/demotion.
- **Differentiation vs L2**: Video-only (no action/interaction trigger semantics). 3-tier (no UVM) vs L2 의 3-tier (HBM-hot/HBM-cold/pinned). Hawkes arrival 없음.
- **Action**: L2 primary baseline.

### OmniSparse: Training-Aware Fine-Grained Sparse Attention for Long-Video MLLMs
- **arXiv**: [2511.12201](https://arxiv.org/abs/2511.12201) | **Date**: 2025-11 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: **L3 MTV-Pool scoop threat (65-75%)** + **A2 concurrent (62%)**.
- **Contribution**: Query lazy-active classification + head-level dynamic budget + selective visual KV fetch.
- **Differentiation vs L3**: Intra-turn (query-level) vs L3 의 inter-turn (turn_oldness axis). Head-level vs L3 의 pool-level.
- **Differentiation vs A2**: Head-level single-axis vs A2 의 intent+entropy 이중 gating + per-tile kernel variant.
- **Action**: L3/A2 baseline (L3 에서 특히 critical).

### DuetServe: Harmonizing Prefill and Decode via Adaptive GPU Multiplexing
- **arXiv**: [2511.04791](https://arxiv.org/abs/2511.04791) | **Date**: 2025-11 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: **L1 ContextSM-Tri concurrent (65%)** — L1 positioning 의 가장 큰 threat.
- **Contribution**: Attention-aware roofline model + partitioning optimizer + SM split 선택. Fine-grained SM-level prefill/decode partition.
- **Differentiation vs L1**: P/D 이분법 vs L1 의 6-class content taxonomy × tri-knob.
- **Action**: L1 primary baseline (reimpl 주의, reimplementation 공정성 검증).

### Adrenaline: Attention Disaggregation for LLM Serving
- **arXiv**: [2503.20552](https://arxiv.org/abs/2503.20552) | **Date**: 2025-03 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: **A3 concurrent (SM yielding axis)**.
- **Contribution**: Attention disaggregation + SM-level resource reallocation (idle yielding).
- **Differentiation vs A3**: Idle yielding (throughput) vs A3 의 deadline miss risk (SLO). Orthogonal — stacking 가능.
- **Action**: A3 primary baseline.

### ContextCache: Context-Aware Semantic Cache for Multi-Turn Dialogues
- **arXiv**: [2506.22791](https://arxiv.org/abs/2506.22791) | **Date**: 2025-06 | **Authors**: (preprint)
- **Venue**: arXiv preprint (v3)
- **Relevance**: A3 adjacent. Multi-turn LLM semantic cache.
- **Contribution**: 2-stage retrieval + self-attention contextual match. Current query embedding 만 생성, historical 재사용.
- **Differentiation vs A3**: LLM-only, single-tenant vs A3 의 VLM multi-tenant + COW.
- **Action**: A3 baseline.

### DroidSpeak: KV Cache Sharing for Cross-LLM Multi-LLM Serving
- **arXiv**: [2411.02820](https://arxiv.org/abs/2411.02820) | **Date**: 2024-11 | **Authors**: (preprint)
- **Relevance**: A3 adjacent.
- **Contribution**: Cross-LLM KV share.
- **Action**: A3 baseline.

### MMInference: Modality-Aware Permutation Sparse Attention for VLM Pre-filling
- **arXiv**: [2504.16083](https://arxiv.org/abs/2504.16083) | **Date**: 2025-04 | **Authors**: (preprint)
- **Venue**: ICLR 2025 Workshop
- **Relevance**: **L3 concurrent (68%)** + **A2 concurrent (62%)**.
- **Contribution**: Modality-aware sparse attention + 2D boundary patterns + permutation-based compaction.
- **Differentiation vs L3**: Offline static permutation vs L3 의 per-turn runtime dynamic coldness.
- **Differentiation vs A2**: Permutation (compile-time) vs A2 의 kernel variant dispatch (runtime).
- **Action**: L3 primary baseline, A2 baseline.

### MadaKV: Adaptive Modality-Perception KV Cache Eviction
- **arXiv**: [2506.15724](https://arxiv.org/abs/2506.15724) | **Date**: 2025-06 | **Venue**: ACL 2025
- **Relevance**: L2 concurrent (55%), L3 concurrent (58%).
- **Contribution**: Modality preference → cache budget. Per-token modality scoring for eviction.
- **Action**: L2/L3 baseline.

### FlowMM: Cross-Modal Information Flow Guided KV Cache Merging
- **arXiv**: [2511.05534](https://arxiv.org/abs/2511.05534) | **Date**: 2025-11 | **Authors**: (preprint)
- **Relevance**: L3 concurrent (55%).
- **Contribution**: Cross-modal KV compaction.
- **Action**: L3 baseline.

### StreamingVLM: Real-Time Understanding for Infinite Video Streams
- **arXiv**: [2510.09608](https://arxiv.org/abs/2510.09608) | **Date**: 2025-10 | **Authors**: (preprint)
- **Relevance**: L2 adjacent (45%).
- **Contribution**: Training-inference unified, attention sink + sliding window, past KV state reuse.
- **Action**: L2 baseline.

### LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval
- **arXiv**: [2505.15269](https://arxiv.org/abs/2505.15269) | **Date**: 2025-05 | **Authors**: (preprint)
- **Relevance**: L2 concurrent, L3 reference.
- **Contribution**: Training-free streaming KV + FIFO long-term memory + retrieval.
- **Hidden insight**: Fig 7 — eviction mistake 의 73% 가 "user question arrival 직전 10초 구간" → user interaction trigger 예측 가치의 근거 (L2 β_v Hawkes 의 motivation).
- **Action**: L2 baseline.

### CacheFlow: Compressive Streaming Memory for Long-Form Video Understanding
- **arXiv**: [2511.13644](https://arxiv.org/abs/2511.13644) | **Date**: 2025-11 | **Authors**: (preprint)
- **Relevance**: L2 adjacent (40%).
- **Contribution**: Dynamic Token Dropping + compressive long-term memory + cosine similarity per-patch.
- **Hidden insight**: Table 3 에서 grasping moment 직전 정확도 2.3% drop (frame-similarity only).
- **Action**: L2 baseline.

### V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval
- **arXiv**: [2512.12284](https://arxiv.org/abs/2512.12284) | **Date**: 2025-12 | **Authors**: (preprint)
- **Relevance**: L3 adjacent, A2 baseline.
- **Contribution**: Dynamic retrieval for streaming video LLM.
- **Action**: L3/A2 baseline.

### Event-VStream: Event-Driven Real-Time Understanding for Long Video Streams
- **arXiv**: [2601.15655](https://arxiv.org/abs/2601.15655) | **Date**: 2026-01 | **Authors**: (preprint)
- **Relevance**: L2 adjacent.
- **Contribution**: Event boundary detection (motion + semantic + predictive cues) + language generation trigger.
- **Differentiation vs L2**: Event detector (pixel+codec) vs L2 의 gripper/curvature/Hawkes.
- **Action**: L2 baseline.

### CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference
- **arXiv**: [2604.06036](https://arxiv.org/abs/2604.06036) | **Date**: 2026-04 | **Authors**: (preprint)
- **Relevance**: L2 adjacent (30-40%).
- **Contribution**: Video codec motion vector 를 streaming VLM inference hint 로 활용.
- **Differentiation vs L2**: Codec-level (semantic-agnostic) vs L2 의 action/interaction semantic trigger.
- **Action**: L2 baseline.

### SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference
- **arXiv**: [2510.17777](https://arxiv.org/abs/2510.17777) | **Date**: 2025-10 | **Venue**: ICCV 2025
- **Relevance**: A2 adjacent (40-42%).
- **Contribution**: Prefill visual token pruning + decode query-relevant retrieval. 4.0× prefill, 2.5× decode, 2.6× E2E.
- **Action**: A2 baseline.

### FlashVLM: Text-Guided Visual Token Selection for Large Multimodal Models
- **arXiv**: [2512.20561](https://arxiv.org/abs/2512.20561) | **Date**: 2025-12 | **Authors**: (preprint)
- **Relevance**: A2 adjacent.
- **Contribution**: Query-aware token selection + encoder-decoder boundary fusion. 75%+ token reduction w/ lossless accuracy.
- **Action**: A2 baseline.

### SparseVLM: Visual Token Sparsification for Efficient VLM Inference
- **arXiv**: [2410.04417](https://arxiv.org/abs/2410.04417) | **Date**: 2024-10 | **Venue**: ICML 2025
- **Relevance**: A2/L3 reference.
- **Contribution**: Text-guided token selection (self-attention matrix 기반). LLaVA 54% FLOPs↓, 37% CUDA↓, 97% accuracy 유지.
- **Action**: A2 baseline.

### LMCache: Efficient KV Cache Layer for Enterprise-Scale LLM Inference
- **arXiv**: [2510.09665](https://arxiv.org/abs/2510.09665) | **Date**: 2025-10 | **Authors**: (preprint)
- **Relevance**: A3/L1 reference.
- **Contribution**: vLLM/SGLang 용 KV cache layer (host memory offload + cross-engine share + P/D disaggregation).
- **Action**: A3 baseline.

### Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode
- **arXiv**: [2507.06608](https://arxiv.org/abs/2507.06608) | **Date**: 2025-07 | **Authors**: (preprint)
- **Relevance**: A3/L1 adjacent.
- **Contribution**: Dynamic partition GPU 자원 (compute + memory + bandwidth contention 고려).
- **Action**: L1/A3 baseline.

### PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving
- **arXiv**: [2501.08192](https://arxiv.org/abs/2501.08192) | **Date**: 2025-01 | **Authors**: (preprint)
- **Relevance**: L2 concurrent (55%).
- **Contribution**: HBM→L2 KV prefetch asynchronous scheduling.
- **Action**: L2 baseline.

### KVSwap: Disk-aware KV Offloading for On-device LLM
- **arXiv**: [2511.11907](https://arxiv.org/abs/2511.11907) | **Date**: 2025-11 | **Authors**: (preprint)
- **Relevance**: L2 concurrent (52%).
- **Contribution**: Long-context tier storage + compute-IO overlap.
- **Action**: L2 baseline.

### Lethe: Layer- and Time-Adaptive KV Cache Pruning
- **arXiv**: [2511.06029](https://arxiv.org/abs/2511.06029) | **Date**: 2025-11 | **Authors**: (preprint)
- **Relevance**: L2 adjacent.
- **Contribution**: Attention sparsity 기반 per-layer eviction + spatial+temporal adaptive budget.
- **Action**: L2 baseline.

### DiffKV: Differentiated Memory Management with Parallel KV Compaction
- **arXiv**: [2412.03131](https://arxiv.org/abs/2412.03131) | **Date**: 2024-12 | **Authors**: (preprint v3 2025)
- **Relevance**: L3 scoop (55-65%).
- **Contribution**: Per-head dynamic attention sparsity → 키/값 별 + 토큰 별 + 헤드 별 memory allocation.
- **Differentiation vs L3**: Head-level diff vs L3 의 inter-turn pool.
- **Action**: L3 primary baseline.

### VL-Cache: Sparsity and Modality-Aware KV Cache Compression
- **arXiv**: [2410.23317](https://arxiv.org/abs/2410.23317) | **Date**: 2024-10 | **Venue**: ICLR 2025
- **Relevance**: L3 scoop (60-70%), A2 adjacent. (이전 세션에서 이미 register.)
- **Action**: L3/A2 baseline.

### Running VLAs at Real-time Speed
- **arXiv**: [2510.26742](https://arxiv.org/abs/2510.26742) | **Date**: 2025-10 | **Authors**: (preprint)
- **Relevance**: A1 adjacent (38%).
- **Contribution**: 480 Hz streaming VLA inference.
- **Action**: A1 baseline.

### SpecPrune-VLA: Action-Aware Self-Speculative Pruning
- **arXiv**: [2509.05614](https://arxiv.org/abs/2509.05614) | **Date**: 2025-09 | **Authors**: (preprint)
- **Relevance**: A1 adjacent (45%).
- **Contribution**: Action-aware sparse compute.
- **Action**: A1 baseline.

### TTF-VLA: Temporal Token Fusion
- **arXiv**: [2508.19257](https://arxiv.org/abs/2508.19257) | **Date**: 2025-08 | **Authors**: (preprint)
- **Relevance**: A1 adjacent, L2 adjacent.
- **Contribution**: Temporal token filtering via pixel-attention integration.
- **Action**: A1/L2 baseline.

### VLA-Pruner: Temporal-Aware Dual-Level Token Pruning
- **arXiv**: [2511.16449](https://arxiv.org/abs/2511.16449) | **Date**: 2025-11 | **Authors**: (preprint)
- **Relevance**: A1 adjacent (40%).
- **Action**: A1 baseline.

### Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale
- **arXiv**: [2601.04071](https://arxiv.org/abs/2601.04071) | **Date**: 2026-01 | **Authors**: (preprint)
- **Relevance**: L1 adjacent (25-40%).
- **Contribution**: Microsecond GPU preemption for SLO.
- **Action**: L1 baseline (time-axis orthogonal 참고).

### Bullet: Dynamic Spatial-Temporal SM Orchestration
- **arXiv**: [2504.19516](https://arxiv.org/abs/2504.19516) | **Date**: 2025-04 | **Authors**: (preprint)
- **Relevance**: L1 concurrent (60%).
- **Contribution**: Feedback-loop 기반 SM allocation.
- **Action**: L1 baseline.

### Scorpio: SLO-Oriented Serving for Heterogeneous SLOs
- **arXiv**: [2505.23022](https://arxiv.org/abs/2505.23022) | **Date**: 2025-05 | **Authors**: (preprint)
- **Relevance**: A1/L1 baseline.
- **Contribution**: SLO heterogeneity 를 admission/queue/batch 에 반영.
- **Action**: A1/L1 baseline.

### JITServe: SLO-aware LLM Serving with Imprecise Request Info
- **arXiv**: [2504.20068](https://arxiv.org/abs/2504.20068) | **Date**: 2025-04 | **Authors**: (preprint)
- **Relevance**: L1 baseline.
- **Action**: L1 baseline.

### Expected Attention: KV Cache Compression by Estimating Attention
- **arXiv**: [2510.00636](https://arxiv.org/abs/2510.00636) | **Date**: 2025-10 | **Authors**: (preprint)
- **Relevance**: P1/P2 predictor reference.
- **Action**: Predictor study reference.

### KVFlow: Efficient Prefix Caching for LLM Multi-Agent Workflows
- **arXiv**: [2507.07400](https://arxiv.org/abs/2507.07400) | **Date**: 2025-07 | **Authors**: (preprint)
- **Relevance**: A3 reference.
- **Action**: A3 secondary baseline.

### Asynchronous KV Cache Prefetching
- **arXiv**: [2504.06319](https://arxiv.org/abs/2504.06319) | **Date**: 2025-04 | **Authors**: (preprint)
- **Relevance**: L2 adjacent (30-40%).
- **Contribution**: L2 prefetch stream asynchronous scheduling.
- **Action**: L2 baseline.

### Understanding GPU Resource Interference in LLM Serving
- **arXiv**: [2501.16909](https://arxiv.org/abs/2501.16909) | **Date**: 2025-01 | **Authors**: (preprint)
- **Relevance**: L1 adjacent — SM+BW interaction 실측 연구.
- **Action**: L1 motivation 근거 인용.

---

## 2026-04-22 Mode 1 Session — Post-verification 추가 실존 논문 (Phase 2' placeholder 검증 후)

Phase 2' similarity-critique agent 가 제시한 11편 placeholder 중 2편만 실존 (GUI-KV, OxyGen), 1편은 withdrawn, 나머지 8편 부재. 검증 과정에서 추가로 발견된 실존 논문 목록:

### GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness
- **arXiv**: [2510.00536](https://arxiv.org/abs/2510.00536) | **Date**: 2025-10-01 | **Authors**: Kung-Hsiang Huang, Haoyi Qiu, Yutong Dai
- **Venue**: arXiv preprint
- **Relevance**: **L3 MTV-Pool 의 최대 scoop threat (55-65% concurrent)**. Placeholder "GUIAgent-KV 2025-11" 의 실제 논문.
- **Contribution**: Spatio-Temporal 통합 GUI agent KV compression. (a) attention sparsity uniformly high across layers → uniform budget allocation. (b) Spatial saliency = residual stream L2 norm of hidden states augments attention. (c) Temporal redundancy = previous frames' keys 를 current frame key subspace 로 projection 하여 redundant history 제거.
- **Hidden insight**: AgentNetBench 5-screenshot 에서 decoding FLOPs -38.9%, step accuracy +4.1% — 기존 full-cache baseline 을 accuracy 도 능가.
- **Differentiation vs L3**: L3 는 γ_v = turn_oldness + log-sum-exp(past attn) + tile_entropy, GUI-KV 는 L2 norm + key-subspace projection. 메커니즘 1:1 일치 아니나 target + positioning 중첩 → contribution claim 잠식.
- **Action**: L3 Major Revision 근거 강화. L3 재설계 시 GUI-KV 와 orthogonal axis 로 좁힘 필수.

### OxyGen: Unified KV Cache Management for VLA under Multi-Task Parallelism
- **arXiv**: [2603.14371](https://arxiv.org/abs/2603.14371) | **Date**: 2026-03-15 | **Authors**: Xiangyu Li, Huaizhi Tang, Xin Ding
- **Venue**: arXiv preprint
- **Relevance**: A3 ~35% adjacent (multi-task-within-request, not multi-request) + L2 ~30% adjacent (cross-task sharing, not HBM tier).
- **Contribution**: Embodied AI 가 parallel execution 하는 multi-task (manipulation + conversation + memory) 를 단일 observation 에서 처리. KV cache 를 first-class shared resource 로 취급, cross-task KV sharing + cross-frame continuous batching (variable-length language decode vs fixed-rate action generation). 3.7× speedup, 200 tok/s + 70 Hz 동시.
- **Differentiation vs A3**: A3 는 multi-tenant cross-user, OxyGen 은 single-request within 의 multi-task. COW/refcount/deadline-aware SM yielding 은 OxyGen 에 부재.
- **Differentiation vs L2**: L2 는 hierarchical HBM/pinned tier, OxyGen 은 cross-task sharing — axis 다름.
- **Action**: A3/L2 baseline 에 추가.

### Rethinking Token Pruning for Historical Screenshots in GUI Visual Agents
- **arXiv**: [2603.26041](https://arxiv.org/abs/2603.26041) | **Date**: 2026-03-27 제출, **2026-03-31 withdrawn** | **Authors**: Daiqiang Li, Zihao Pan, Zeyu Zhang
- **Status**: Withdrawn by Honggang Chen (authorship disputes + incomplete documentation).
- **Relevance**: L3 referential. Withdrawn 으로 공식 scoop 아니나 concept 선점 증거.
- **Contribution**: Semantic / Spatial / Temporal 3 perspective 로 historical screenshot token pruning 재정의. Temporal perspective = recency-based budget allocation (larger budget to recent, compress distant).
- **Action**: L3 관련 논문 목록에 기록. 재제출 시 재평가.

### Semantic Scheduling for LLM Inference
- **arXiv**: [2506.12204](https://arxiv.org/abs/2506.12204) | **Date**: 2025-06 | **Authors**: (preprint)
- **Venue**: arXiv preprint
- **Relevance**: **L1 ContextSM-Tri 45-55% concurrent** — content-axis semantic scheduling 이미 제안. L1 Novelty 축 직접 경쟁.
- **Contribution**: LLM 으로 사용자 요청의 semantic attribute (urgency, importance) 분석 → 단순 latency-based scheduling 넘어 domain-specific 스케줄링.
- **Differentiation vs L1**: L1 은 6-class taxonomy × tri-knob (α_SM, α_BW, α_KV) + MIG+Green Context nested — Semantic Scheduling 은 sem score 추출만, tri-knob 이나 HW partition 없음.
- **Action**: L1 Novelty 7.2 → 6.8 downward 조정의 주요 근거. L1 primary baseline 필수.

### SageSched: LLM Scheduling Confronting Demand Uncertainty
- **arXiv**: [2603.07917](https://arxiv.org/abs/2603.07917) | **Date**: 2026-03 | **Authors**: (preprint)
- **Relevance**: L1 adjacent.
- **Contribution**: Semantic-aware history-based predictor, prompt contents + past inference results 로 output-length distribution 예측.
- **Action**: L1 baseline.

### PARS: Prompt-Aware Scheduling for Low-Latency LLM Serving
- **arXiv**: [2510.03243](https://arxiv.org/abs/2510.03243) | **Date**: 2025-10 | **Authors**: Yiheng Tao, Yihe Zhang et al.
- **Relevance**: L1 adjacent.
- **Contribution**: Learning-to-rank listwise approach for prompt ordering, relative ordering 최적화.
- **Action**: L1 baseline.

### Semantic-Aware Scheduling for GPU Clusters with Large Language Models
- **arXiv**: [2510.03334](https://arxiv.org/abs/2510.03334) | **Date**: 2025-10 | **Authors**: (preprint)
- **Relevance**: L1 adjacent (cluster-level semantic scheduling).
- **Action**: L1 baseline.

### SLOs-Serve: Optimized Serving of Multi-SLO LLMs
- **arXiv**: [2504.08784](https://arxiv.org/abs/2504.08784) | **Date**: 2025-04 | **Authors**: (preprint)
- **Relevance**: A3 adjacent (multi-SLO serving, deadline 관점).
- **Contribution**: Admission control + dynamic programming for multi-stage SLO attainment.
- **Action**: A3 baseline.

### GreenLLM: SLO-Aware Dynamic Frequency Scaling for Energy-Efficient LLM Serving
- **arXiv**: [2508.16449](https://arxiv.org/abs/2508.16449) | **Date**: 2025-08 | **Authors**: (preprint)
- **Relevance**: L1/A3 adjacent (SLO-aware serving framework with explicit prefill/decode separation).
- **Contribution**: GPU energy 최소화 + SLO 보장.
- **Action**: L1 adjacent baseline.

### AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding
- **arXiv**: [2501.12162](https://arxiv.org/abs/2501.12162) | **Date**: 2025-01 | **Authors**: (preprint)
- **Relevance**: A3 adjacent.
- **Contribution**: Multi-SLO serving as constrained optimization with SLO-customized speculative tree. Speculate-select-verify pipeline.
- **Action**: A3 baseline.

### OrbitFlow: SLO-Aware Long-Context LLM Serving with Fine-Grained KV Cache Reconfiguration
- **arXiv**: [2601.10729](https://arxiv.org/abs/2601.10729) | **Date**: 2026-01 | **Authors**: (preprint)
- **Relevance**: A3 adjacent (SLO + KV cache reconfiguration for long context).
- **Action**: A3 baseline.

### AttentionPredictor: Temporal Patterns Matter for KV Cache Compression
- **arXiv**: [2502.04077](https://arxiv.org/abs/2502.04077) | **Date**: 2025-02 | **Authors**: (preprint)
- **Relevance**: L2 adjacent (temporal pattern in KV compression).
- **Contribution**: Temporal attention pattern 학습하여 KV cache 에서 중요 token 예측.
- **Action**: L2 baseline.

### EpiCache: Episodic KV Cache Management for Long Conversational Question Answering
- **arXiv**: [2509.17396](https://arxiv.org/abs/2509.17396) | **Date**: 2025-09 | **Authors**: (preprint)
- **Relevance**: L3 adjacent (episodic memory for multi-turn).
- **Action**: L3 baseline.

### STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction
- **arXiv**: [2603.20284](https://arxiv.org/abs/2603.20284) | **Date**: 2026-03 | **Authors**: (preprint)
- **Relevance**: L3 adjacent (Spatio-Temporal aware cache — 3D scope).
- **Action**: L3 secondary baseline.

### Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
- **arXiv**: [2511.02230](https://arxiv.org/abs/2511.02230) | **Date**: 2025-11 | **Authors**: (preprint)
- **Relevance**: L3 / A3 adjacent (Multi-turn LLM agent + KV TTL).
- **Action**: L3/A3 baseline.

### Auto-Scaling Continuous Memory for GUI Agent
- **arXiv**: [2510.09038](https://arxiv.org/abs/2510.09038) | **Date**: 2025-10 | **Authors**: (preprint)
- **Relevance**: L3 adjacent (GUI agent continuous memory compression).
- **Contribution**: Trajectory 를 fixed-length embedding sequence 로 compress, VLM embedding layer 에 inject.
- **Action**: L3 baseline.

### Strata: Hierarchical Context Caching for Long Context Language Model Serving
- **arXiv**: [2508.18572](https://arxiv.org/abs/2508.18572) | **Date**: 2025-08 | **Authors**: (preprint)
- **Relevance**: L1/L2 adjacent (hierarchical context caching).
- **Action**: L1 / L2 baseline.

### OpenHelix: Dual-System VLA for Robotic Manipulation
- **arXiv**: [2505.03912](https://arxiv.org/abs/2505.03912) | **Date**: 2025-05 | **Authors**: (preprint)
- **Relevance**: A1 adjacent (survey + opensource dual-system VLA).
- **Action**: A1 baseline. Placeholder "Helix-VLA" 대체.

### BlindSight: Harnessing Sparsity for Efficient Vision-Language Models
- **arXiv**: [2507.09071](https://arxiv.org/abs/2507.09071) | **Date**: 2025-07 | **Authors**: (preprint)
- **Relevance**: A2 adjacent (input-template-aware attention sparsity mask for multi-image VLM).
- **Contribution**: Multi-image VLM inference 시 입력 template 인식 기반 runtime-overhead-free sparsity mask.
- **Action**: A2 baseline.

### Tawa: Automatic Warp Specialization for Modern GPUs
- **arXiv**: [2510.14719](https://arxiv.org/abs/2510.14719) | **Date**: 2025-10 | **Authors**: (preprint)
- **Relevance**: A2 adjacent (warp specialization automation).
- **Action**: A2 reference.

---

## 2026-04-22 Mode 2 Session — VLM+PIM 연구 보완·확장

### VLM_exploration_PIM_260407.pdf (연구자, 2026-04-07, internal)
- **Date analyzed**: 2026-04-22
- **Session**: [링크](sessions/2026-04-22-mode2-vlm-pim-extension.md)
- **Contribution**: VLM-aware Heterogeneous KV Management for GPU+PIM Systems 제안 (motivation paper). AttAcc ASPLOS'24 baseline + Qwen3-VL-4B. 3 Challenges (C1 TTFT explosion 6-22×, C2 Layer Asymmetry L17-21 dense 24.5% vs L0-7 sparse 2.6% → BW 7.6× waste, C3 Capacity FHD KV 305MB/req + mixed batch imbalance 13.6×) + 3 Solutions (4-A ViT-Decode overlap, 4-B Layer-Adaptive KV Placement dense→PIM/sparse→HBM, 4-C Bank Balancing + Cross-Req Image Sharing).
- **Hidden insights**: (1) DeepStack Qwen3-VL 의 ViT intermediate output → LLM [L4, L8, L12] inject 가 L17-21 peak 의 구조적 원인 가설. (2) Visual KV size 가 seq 86.2% 차지하면서도 attention 기여도 11.4% — **token 수와 contribution 의 비대칭**은 ACE-MoE style cumulative score 에 직접 유사. (3) Bank imbalance 는 **mixed LLM+VLM batch 에서 최악** (13.62× @ 25% VLM); modality-aware spreading 으로 22% 개선 가능. (4) Same-image cross-request sharing 이 BS=128, 50% sharing 에서 38% memory + 1.62× capacity 회복 — multi-turn conversation 에서 빈번.
- **Our session relation**: 2026-04-21 Mode 2 ACE-MoE 세션에서 SW-only 측면 활용. **본 세션 (2026-04-22 Mode 2) 은 HW/PIM 측면 orthogonal 분석** — 같은 PDF 를 다른 축으로 재해석하여 본 연구의 4-A/4-B/4-C 를 직접 확장·보완.

### PIM_260422_미팅자료.pdf (연구자, 2026-04-22, internal)
- **Date analyzed**: 2026-04-22
- **Session**: [링크](sessions/2026-04-22-mode2-vlm-pim-extension.md)
- **Contribution**: E1-E5 실험 + AttAcc simulator 수정 계획 미팅자료. (a) FP16 → BF16 migration 강제 발견 (L27 self-attn Q·Kᵀ > 65504 overflow → NaN; BF16 에서 L27 abs_max=860 finite). (b) E1 chunked prefill GPU profile (C≤32 AI<60 강 memory-bound, C=4 SDPA 870ms = full prefill 33ms 의 26×). (c) E2 PIM analytical (AttAcc HBM3 18.1 TB/s effective, C=16 chunked 대비 PIM attn 50×, E2E prefill 1.53×, video L=8948 PIM 0.81× regression). (d) E3 pattern robustness (MMMU n=125, subset-mean corr 0.9964 / sample-to-full min 0.357). (e) E4 5-model comparison (Qwen3-VL/Qwen2.5-VL/InternVL3 dense band 공통, Mllama cross-attn self-attn 0.85% 만, Qwen3.5 hybrid linear 48/64 layer linear KV 무관). (f) E5 quantization impact (W8A8 pattern collapse +66.85pp, weight-only INT4/INT8 Δ<0.1pp, FP8-Dynamic opposite -4.17pp).
- **Hidden insights**: (1) **E3 sample-to-full corr 0.357 minimum** — aggregate 0.996 stable 하지만 individual 은 catastrophic — per-sample policy fragility 가 VLM KV 문헌의 open gap. (2) W8A8 collapse (+66pp) vs weight-only 보존 (Δ<0.1pp) 의 대비 — activation quantization 이 visual attention distribution 을 붕괴시키는 핵심. FP8-Dynamic 은 반대 방향 collapse 로 같은 원리. (3) **Prefill 은 simulator 지원 미비** — AttAcc 원본은 decode-only (src/system.py L223-237 sum stage GPU route), 4-file extension (config/system/model/ramulator_wrapper) 1주 작업. (4) FP16 L27 overflow 는 **VLM-specific numerical safety** 이슈 (LM head 직전 sharpened representation, Q·Kᵀ abs_max > 65504). (5) E4 Qwen3.5 hybrid linear 는 48/64 layer 에 KV 없음 — **architecture fingerprint dispatcher** 가 필요.
- **Next steps (연구)**: P1 AttAcc simulator 4-file mod (1주), P2 Cross-dataset E3 (COCO/DocVQA/ChartQA), P3 E4 full 200 samples + InternVL3 late-peak 검증, P3 Video hierarchical KV 대안.

### VLCache: 2% Vision Token Computation + 98% KV Cache Reuse for Efficient Multi-turn VLM
- **arXiv**: [arXiv:2512.12977](https://arxiv.org/abs/2512.12977) (2025.12)
- **Date analyzed**: 2026-04-22
- **Relevance**: 🚨 F3 scoop source (~70% on refined claim)
- **Contribution**: Image hash 기반 encoder/KV cache 재사용 SGLang 구현. Layer-dependent recomputation 으로 토큰 2-5% 만 계산, 1.2-16× TTFT speedup. Qwen3-VL-8B 등 대상 multi-turn dialogue.
- **Impact on our session**: F3 VLM-MOESI 의 image-hash reuse axis 를 이미 구현. F3 는 "KV reuse" 를 포기하고 **GPU+PIM coherence protocol** 로 narrative 전면 피벗 필요. Phase 2' 에서 Major Revision 판정. 본 연구의 4-C Cross-Request Image Sharing 은 이미 VLCache 로 커버되므로, 차별화는 **PIM bank 물리적 배치** + **coherence state machine** 에만 남음.

### PAM: Processing Across Memory Hierarchy for KV-Centric LLM Serving
- **arXiv**: [arXiv:2602.11521](https://arxiv.org/abs/2602.11521) (2026.02)
- **Date analyzed**: 2026-04-22
- **Relevance**: 🚨 H2 scoop source (~75-80%)
- **Contribution**: HBM-PIM + DRAM-PIM + SSD-PIM 3-tier hierarchy. Context locality 기반 KV 분산, PAMattention. DGX-H100 vLLM 대비 평균 12.88×. Text LLM 전용.
- **Impact**: H2 Hierarchical 3-Tier KV (GPU/PIM/CXL) 단독으로는 통과 불가. VLM DeepStack-aware tier 정책을 결합해야만 생존. → H2 DROP, F1 sub-module 로 흡수.

### Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models (HPCA 2026 Best Paper Candidate)
- **arXiv**: [arXiv:2512.14661](https://arxiv.org/abs/2512.14661)
- **Date analyzed**: 2026-04-22
- **Relevance**: Concurrent work at architecture layer (Duke Hai Helen Li / Yiran Chen group)
- **Contribution**: VLM 전용 streaming concentration architecture. 3-level hierarchical compression (text-prompt semantic pruning + spatial-temporal block concentration + vector redundancy). Systolic-array 모듈 추가, 2.4× speedup / 3.3× energy.
- **Impact**: 본 연구의 C2 Layer Asymmetry 관찰과 philosophy 유사 but 경쟁 axis 다름 (systolic vs GPU+HBM-PIM). F1-VLM 의 차별화: **DeepStack inject topology + AI inflection** 교집합 기반 tier (Focus 는 generic attention score).

### V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval (HPCA 2026)
- **arXiv**: [arXiv:2512.12284](https://arxiv.org/abs/2512.12284)
- **Date analyzed**: 2026-04-22
- **Relevance**: Concurrent at video LLM axis (KAIST Joo-Young Kim group)
- **Contribution**: Training-free ReSV — temporal/spatial similarity-based token clustering 기반 video KV retrieval. AGX Orin 대비 1.9-19.7× / 3.1-18.5× energy. 3.9-8.3 FPS real-time.
- **Impact**: 본 연구의 video L=8948 regression 문제 정면 대응. F3 하위 제안 혹은 F1 C-adaptive dispatcher 의 video mode 로 baseline 필수.

### ORCHES: Orchestrated Test-Time-Compute-based LLM Reasoning on Collaborative GPU-PIM Heterogeneous System (MICRO 2025)
- **DOI**: 10.1145/3725843.3756039
- **Date analyzed**: 2026-04-22
- **Relevance**: Concurrent at GPU+PIM VLM reasoning
- **Contribution**: Adaptive workload assignment + branch-aware pipelining + fragmentation-aware memory structuring. Text reasoning 4.16×, vision-based reasoning 3.10×.
- **Impact**: ORCHES 는 reasoning branch 중심, 은 prefill/TTFT + multi-image capacity 중심 — orthogonal scope 명시 필요.

### STARC: Sparse Attention Remapping with Clustering for LLM Decoding on PIM
- **arXiv**: [arXiv:2505.05772](https://arxiv.org/abs/2505.05772) (2025.05)
- **Date analyzed**: 2026-04-22
- **Relevance**: 🚨 AttAcc simulator 공통 도구 (직접 경쟁)
- **Contribution**: KV pair semantic similarity clustering → PIM bank contiguous region 매핑. Budget 1024 에서 54-74% latency, 19-31% / 19-27% energy.
- **Impact**: F1 의 DeepStack-aware tier 와 바로 경쟁. STARC 는 token clustering 기반 bank remapping, F1 은 layer-topology + AI inflection dispatch — axis 분리 필수. Must-include baseline for F1.

### P3-LLM: NPU-PIM Integrated Accelerator with Mixed Precision
- **arXiv**: [arXiv:2511.06838](https://arxiv.org/abs/2511.06838) (2025.11)
- **Date analyzed**: 2026-04-22
- **Relevance**: Concurrent with F2 (~55-65%)
- **Contribution**: W4 weight/KV + W8 activation + dynamic input-aware smoothing for outlier handling. NPU-PIM integrated.
- **Impact**: F2 차별화: P3-LLM 은 **static per-layer precision + calibration-free smoothing**, F2 는 **runtime KL-collapse trigger + BF16 fallback dedicated path**. Must-include baseline.

### VL-Cache: Sparsity and Modality-Aware KV Cache Compression (ICLR 2025)
- **arXiv**: [arXiv:2410.23317](https://arxiv.org/abs/2410.23317)
- **Date analyzed**: 2026-04-22
- **Relevance**: 🚨 본 연구의 4-B 의 SW 원형 / F2 baseline
- **Contribution**: Layer-adaptive sparsity-aware cache budget + modality-aware token scoring. 10% budget 으로 full-cache 정확도, 2.33× E2E latency, 7.08× decoding, 90% GPU memory 감소.
- **Impact**: F2/F1 모두에 must-include. 본 연구의 4-B (HW) vs VL-Cache (SW) 로 orthogonal 증명 필수.

### AKVQ-VL: Attention-aware Saliency KV Cache Quantization for VLM
- **arXiv**: [arXiv:2501.15021](https://arxiv.org/abs/2501.15021) (2025.01)
- **Date analyzed**: 2026-04-22
- **Relevance**: Concurrent with F2/A1 (per-layer saliency vs per-sample fragility)
- **Contribution**: Attention-aware saliency patterns 로 bit budget 적응 할당. Per-sample 아님.
- **Impact**: F2 baseline. A1 differentiation: **per-sample gating** 이 AKVQ-VL 의 per-layer 와 orthogonal granularity.

### ModServe: Scalable and Resource-Efficient Large Multimodal Model Serving
- **arXiv**: [arXiv:2502.00937](https://arxiv.org/abs/2502.00937) (2025.02)
- **Date analyzed**: 2026-04-22
- **Relevance**: 🚨 L1 scoop source (모달리티 disaggregation 선점)
- **Contribution**: Modality- and stage-aware resource disaggregation. Image-text vs text-only 라우팅. P99 TTFT 20-50% 감소.
- **Impact**: L1 Dual-Pool Batching 을 SW layer 에서 이미 수행. L1 DROP 확정.

### RPS-Serve: Rocks, Pebbles, and Sand — Heterogeneous MLLM Workload Serving
- **arXiv**: [arXiv:2603.26498](https://arxiv.org/abs/2603.26498) (2026.03)
- **Date analyzed**: 2026-04-22
- **Relevance**: 🚨 L1 scoop source (MMMU TTFT tail 정면 해결)
- **Contribution**: Video=rocks / image=pebbles / text=sand 3-tier scheduling. FCFS HoL blocking 으로 text 가 수십초 지연되는 문제 해결.
- **Impact**: L1 과 동일 MMMU TTFT tail 문제를 이미 해결. L1 DROP.

### Dual-Pool Token-Budget Routing for LLM Serving
- **arXiv**: [arXiv:2604.08075](https://arxiv.org/abs/2604.08075) (2026.04)
- **Date analyzed**: 2026-04-22
- **Relevance**: 🚨 L1 scoop source (용어 · 구조 직접 충돌)
- **Contribution**: Dual pool (short/long) routing + token budget.
- **Impact**: "Dual pool" 이름부터 L1 과 일치. L1 독립 논문 불가.

### DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs (NeurIPS 2024)
- **arXiv**: [arXiv:2406.04334](https://arxiv.org/abs/2406.04334)
- **Date analyzed**: 2026-04-22
- **Relevance**: Qwen3-VL 아키텍처 원조 / F1 기반
- **Contribution**: ViT multi-layer output 을 LLM [L4, L8, L12] 에 inject 하는 아키텍처.
- **Impact**: F1 의 6-tier 정의의 구조적 근거. 본 연구의 L17-21 peak 의 원인 가설.

### Fallback Quantization: Dynamic Block-Level INT8 → BF16 Fallback for Training
- **arXiv**: [arXiv:2503.08040](https://arxiv.org/abs/2503.08040) (2025.03)
- **Date analyzed**: 2026-04-22
- **Relevance**: NEW CONCURRENT (Phase 2' 6-month search, ~35% philosophy overlap)
- **Contribution**: Training 중 block-level dynamic fallback. General GEMM 대상.
- **Impact**: F2 의 "BF16 fallback" philosophy 유사 but training vs inference, GEMM vs PIM macro, LLM vs VLM, per-sample gating 부재 — narrow 차별화 가능.

### LazyPIM: PIM Cache Coherence
- **arXiv**: [arXiv:1706.03162](https://arxiv.org/abs/1706.03162) (2016, CAL)
- **Date analyzed**: 2026-04-22
- **Relevance**: F3 MOESI 원조
- **Contribution**: PIM cache coherence 고전, MESI 확장 baseline.
- **Impact**: F3 재설계 시 baseline. Image-hash granularity 로 cache-line 실용성 문제 회피.

### TraCT: Two-tier Coherence for CXL Shared Memory KV Cache
- **arXiv**: [arXiv:2512.18194](https://arxiv.org/abs/2512.18194) (2025.12)
- **Date analyzed**: 2026-04-22
- **Relevance**: F3 counter-argument
- **Contribution**: CXL shared memory KV coherence, two-tier inter-node sync. "Multi-TB CXL scale 에서 hardware coherence fundamentally impractical" 주장.
- **Impact**: F3 재설계 시 image-identity granularity 로 TraCT 반박 필요.

### PIMphony / LoL-PIM: Overcoming Bandwidth & Capacity Inefficiency (HPCA 2026)
- **arXiv**: [arXiv:2412.20166](https://arxiv.org/abs/2412.20166)
- **Date analyzed**: 2026-04-22
- **Relevance**: H2 adjacent (long-context PIM)
- **Contribution**: Token-Centric PIM Partitioning + Dynamic PIM Command Scheduling + Dynamic PIM Access. xPU+PIM 8.4×. SK hynix 공저.
- **Impact**: H2 DROP 근거 보강 (long-context PIM 은 이미 baseline 존재).

### Pimba: PIM Acceleration for Post-Transformer LLM Serving (MICRO 2025)
- **arXiv**: [arXiv:2507.10178](https://arxiv.org/abs/2507.10178)
- **Date analyzed**: 2026-04-22
- **Relevance**: Qwen3.5 hybrid linear baseline 후보
- **Contribution**: Mamba/SSM/linear attention PIM, State-update Processing Unit (SPU), MX quantized. GPU 4.1×, GPU+PIM 2.1×.
- **Impact**: 본 연구의 Qwen3.5 hybrid linear (48/64 linear) 대상 baseline 후보.

### Oaken: Hybrid KV Quantization (ISCA 2025)
- **arXiv**: [arXiv:2503.18599](https://arxiv.org/abs/2503.18599)
- **Date analyzed**: 2026-04-22
- **Relevance**: KAIST CASTLAB (Park Jongse group, AttAcc follow-up)
- **Contribution**: Threshold-based online-offline hybrid quantization + group-shift + fused dense/sparse. vLLM 대비 1.79×, 44.3% power.
- **Impact**: F2 baseline 후보 (AttAcc follow-up 그룹, VLM W8A8 collapse 를 mitigate 가능성).

### Jenga: Effective Memory Management for Heterogeneous LLMs
- **arXiv**: [arXiv:2503.18292](https://arxiv.org/abs/2503.18292) (2025.03)
- **Date analyzed**: 2026-04-22
- **Relevance**: A3 concurrent (~50-60%)
- **Contribution**: Cross-attn, linear attn, sliding window 혼합 KV 관리. GPU-only.
- **Impact**: A3 는 독립 대신 F1 generalization 섹션 흡수. Jenga 는 baseline.

### KVShare: Semantic-Aware Cross-User KV Cache Sharing
- **arXiv**: [arXiv:2503.16525](https://arxiv.org/abs/2503.16525) (2025.03)
- **Date analyzed**: 2026-04-22
- **Relevance**: F3 concurrent (text-side semantic sharing, 60% hit rate)
- **Contribution**: Semantic-aware KV cache sharing, 60% hit rate.
- **Impact**: F3 재설계 시 visual version 으로 pivot 시 baseline.

### MBQ: Modality-Balanced Quantization for Large Vision-Language Models (CVPR 2025)
- **arXiv**: [arXiv:2412.19509](https://arxiv.org/abs/2412.19509)
- **Date analyzed**: 2026-04-22 (post-literature survey)
- **Relevance**: F2 motivation check — closest VLM quantization paper for 비교
- **Contribution**: VLM 에서 vision/language token gradient sensitivity 차이 (language token absolute gradient 가 vision 의 10×) 정량화. Modality-balanced calibration 으로 W8A8 "nearly lossless" 주장 (task accuracy 기준).
- **Impact on F2**: **Key finding** — MBQ 가 "W8A8 lossless" 라고 주장하지만 이는 **task accuracy (MMMU/VQA score) 기준이며 layer-wise visual attention ratio 는 측정 안 함**. 본 연구의 +66pp collapse 와 양립 가능 ("Seeing but Not Believing" 현상). F2 는 MBQ 가 놓친 attention distribution 붕괴를 정량화 + runtime 탐지.

### Q-VLM: Post-training Quantization for Large Vision-Language Models (NeurIPS 2024)
- **arXiv**: [arXiv:2410.08119](https://arxiv.org/abs/2410.08119)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 motivation check
- **Contribution**: Cross-layer discretization error + activation entropy 상관관계 분석. Optimal quantization 경계 결정.
- **Impact on F2**: Entropy 개념 유사하나 layer 별 visual attention ratio 미측정. F2 의 measurement contribution 유지.

### VLMQ: Token Saliency-Driven Post-Training Quantization for VLM
- **arXiv**: [arXiv:2508.03351](https://arxiv.org/abs/2508.03351)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 motivation check
- **Contribution**: Vision token 이 Hessian 에 over-represented → suboptimal weight update 증명. Token saliency 기반 PTQ.
- **Impact on F2**: Vision over-representation 은 공통 문제의식이나 **Hessian 편향 vs attention 분포** 축 차이. F2 와 orthogonal.

### MQuant: Full Static Quantization for Multi-Modal LLMs (ACM MM 2025)
- **arXiv**: [arXiv:2502.00425](https://arxiv.org/abs/2502.00425)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 motivation check — mechanism 관점 가장 가까운 선행
- **Contribution**: Static W4A8 에서 vision/text per-tensor scale mismatch 가 정확도 drop 주원인 (MSQ — Modality-Specific static Quantization). Hadamard rotation 이 "fresh outlier" 유발 가능.
- **Impact on F2**: **Mechanism hypothesis (c) rotation-invariance 의 근거**. F2 는 MQuant 가 지적한 vision/text scale mismatch 가 W8A8 에서 extreme regime 진입하는 경우를 정량화.

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **arXiv**: [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 mechanism hypothesis (a) 근거 — activation outlier → softmax 왜곡
- **Contribution**: 6.7B+ 모델에서 75% sequence dim 에 outlier 출현. Outlier feature 를 zero 화하면 top-1 attention softmax mass 감소. Mixed 8-bit + FP16 decomposition.
- **Impact on F2**: **Mechanism hypothesis (a) 의 직접 근거** — activation outlier 가 attention softmax 분포에 영향. 본 연구의 +66pp 은 "outlier over-preserve 시 softmax mass 몰림" 의 반대 방향 현상으로 해석 가능. 그러나 LLM.int8 은 **text-only, +66pp extreme 값 보고 없음**.

### Visual Attention Sink in Large Multimodal Models
- **arXiv**: [arXiv:2503.03321](https://arxiv.org/abs/2503.03321)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 mechanism hypothesis (b) 근거
- **Contribution**: FP16 VLM 에서 irrelevant visual token 이 sink 로 기능. Sink dimension value > τ=20 기준 측정. Sink redistribution 방법 제안.
- **Impact on F2**: **FP16 에 이미 존재하는 구조적 sink**. 본 연구의 W8A8 +66pp 은 이 sink 가 양자화로 폭발적 증폭되는 특수 케이스로 해석 가능. 단 이 논문은 **양자화 영향 미분석** — F2 가 그 gap 을 채움.

### Seeing but Not Believing: Visual Attention vs Answer Correctness in MLLMs
- **arXiv**: [arXiv:2510.17771](https://arxiv.org/abs/2510.17771)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 task-attention decoupling 근거
- **Contribution**: VLM 이 attention 은 잘못된 곳에 가도 task-level correct answer 도출 가능. Attention 과 answer correctness decoupling 증명.
- **Impact on F2**: **F2 의 "MMMU acc 유지하나 attention 붕괴" 해석 지지**. MBQ 가 주장한 "W8A8 lossless (task acc)" 와 본 연구의 +66pp collapse 가 양립 가능한 이유. F2 는 attention grounding 축에서 기여 확장 가능 (reasoning integrity vs final answer).

### Attention Sinks and Compression Valleys in LLMs
- **arXiv**: [arXiv:2510.06477](https://arxiv.org/abs/2510.06477)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 mechanism hypothesis (b) 수학적 근거
- **Contribution**: Massive activation 이 필연적으로 representational compression 유발한다는 수학적 증명. Attention sink 와 compression 연결.
- **Impact on F2**: **Mechanism (b) 지지** — quantization 이 massive activation 을 spiky 하게 만들면 sink 증폭 필연. 72% collapse 는 compression valley 의 극단 사례로 해석.

### SpinQuant: LLM Quantization with Learned Rotations
- **arXiv**: [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
- **Date analyzed**: 2026-04-22
- **Relevance**: F2 mechanism hypothesis (c) 검증 tool
- **Contribution**: Learned Hadamard rotation 으로 outlier 제거 후 W4A4 달성. QuaRot 계열.
- **Impact on F2**: **본 연구의 E5 recipe sweep 필수 baseline**. SpinQuant/QuaRot 이 W8A8 collapse 를 SW-only 로 완화하면 mechanism (c) rotation-invariance 검증 + F2 HW detector 필요성 조건부 축소.

### RegCache: Activation Quantization of Vision Encoders via Register Prefix
- **arXiv**: [arXiv:2510.04547](https://arxiv.org/abs/2510.04547)
- **Date analyzed**: 2026-04-22
- **Relevance**: Vision encoder outlier 특수성 지지
- **Contribution**: Vision encoder activation outlier 는 LLM 과 동역학 다름 → register prefix 로 완화.
- **Impact on F2**: **VLM encoder-side outlier 특수성** 근거. F2 의 vision-heavy outlier 가설 (mechanism a) 지지.

### Empirical Study of Qwen3 Quantization
- **arXiv**: [arXiv:2505.02214](https://arxiv.org/abs/2505.02214)
- **Date analyzed**: 2026-04-22
- **Relevance**: Qwen 계열 activation quantization 취약성
- **Contribution**: Qwen3 (text) 에서 "W8A8 SmoothQuant 조차 noticeable degradation, activation outlier 가 원인일 가능성" 명시.
- **Impact on F2**: **Qwen 계열이 activation quantization 에 특히 취약**하다는 간접 증거. Qwen3-VL 에서 +66pp extreme 은 Qwen family 의 일관된 outlier 패턴 반영으로 해석.

### Prefixing Attention Sinks for LLM Quantization
- **arXiv**: [arXiv:2406.12016](https://arxiv.org/abs/2406.12016)
- **Date analyzed**: 2026-04-22
- **Relevance**: Sink precision 상향 복원 기법
- **Contribution**: LLM attention sink 와 activation outlier 연결. Sink token precision 을 올리면 quantization 복원 가능.
- **Impact on F2**: Text-only LLM 관찰이나 F2 의 BF16 fallback path 철학과 유사 (sink 위치만 higher precision). VLM 적용은 F2 의 contribution.

### MPIC: Position-Independent Multimodal Context Caching
- **arXiv**: [arXiv:2502.01960](https://arxiv.org/abs/2502.01960) (2025.02)
- **Date analyzed**: 2026-04-22
- **Relevance**: 4-C Cross-Req Image Sharing 의 SW 대응
- **Contribution**: 임의 position 의 multimodal KV 재사용. 응답시간 54% 감소, throughput 2×.
- **Impact**: F3 재설계 baseline + 본 연구의 4-C 와 SW orthogonal 증명 필요.

---

## 2026-04-22 Mode 1 Session — PRISM BNN/TNN Domain Extension (KWS / Biosignal / SNN)

### PRISM: Decompose-then-Merge Scaling Factors for Accurate and Energy-Efficient CIM-friendly BNNs (ISLPED 2026 투고)
- **Paper-id**: islped-2026-prism (공영호 lab 자체 논문)
- **Date analyzed**: 2026-04-22
- **Session**: [링크](sessions/2026-04-22-mode1-bnn-tnn-domain-extension.md)
- **Expert**: algorithm-expert, hw-pim-accelerator-expert
- **Contribution**: CIM crossbar 기반 BNN 추론의 SF 구조적 부정합 해결. (1) Xbar-wise SF: channel-wise (N buffer reads per crossbar) → Xbar-wise (1 read). (2) Decompose-then-merge: 훈련 시 rank r 로 S1×S2 분해 학습, 추론 시 merge (zero overhead). (3) OPTIC: 작은 crossbar 에서 inlier 대표값 μ_in 교체 + outlier 보존. (4) LUT-bypass: 확보 buffer 에 pre-computed SF × integer_output 저장, hit rate >82%. ResNet-20/18 CIFAR + Tiny-ImageNet 에서 정확도 +0.19-2.69%, SF 26-95.9% 감소, TOPS/W 1.8-3.4× 개선. 전부 scratch training, device-agnostic (RRAM/FeRAM/SRAM).
- **Hidden insights**:
  1. Figure 3: rank-decomposition 이 activation dynamic range 를 channel-wise 대비 확장 (σ 2.44→4.42 ResNet-20) — representational fidelity 회복 메커니즘.
  2. Figure 10: cache stable after first 4 requests + <5% replacement — KWS MFCC autocorrelation 의 time-axis 판으로 재해석 가능 (F1 TempoPRISM-CoDesign 의 warm-LUT 이론적 근거).
  3. Table 2 NAEE 비교: static pruning 은 cache 25% 에서 50.5% accuracy 로 무너지나 ACE caching 은 97.2% — multimodal / subject-shift 환경에서 이 격차 더 커질 것 (F2 PhysioPRISM 의 핵심 motivation).
  4. Figure 12: "LUT hit rate ≠ accuracy preservation" — F2 의 "per-subject full retraining ≠ μ_in 패치" 로 transfer.
- **Relevance**: 본 세션 6 ideas 의 parent paper. F1/F2 가 직접 확장.
- **Limitations (확장 여지)**: (a) Image classification only, (b) scratch training 만 검증, (c) 큰 모델(>10M params) 미평가, (d) 도메인 특화 workload pattern (streaming audio, wearable, event) 미탐구.

### LoRDS: Low-Rank Decomposed Scaling for LLM PTQ
- **arXiv**: [2601.22716](https://arxiv.org/abs/2601.22716) | **Date**: 2026-01
- **Session**: [링크](sessions/2026-04-22-mode1-bnn-tnn-domain-extension.md)
- **Relevance**: PRISM 의 rank decomposition 원리가 LLM PTQ 에서도 독립 발견 — F1 의 ~45% mechanism overlap, baseline "LoRDS-transferred-to-BNN-KWS" 필수 포함.
- **Differentiation vs F1**: LoRDS 는 post-hoc PTQ, LLM, FP activation, CPU/GPU. F1 은 scratch training, BNN, integer activation on analog crossbar + time-axis 추가. LoRDS 가 BNN-CIM-streaming 에서 저조함 증명이 F1 의 scoop 대응.

### SparkNet: Sparse Binarization for Fast KWS
- **arXiv**: [2406.06634](https://arxiv.org/abs/2406.06634) | **Date**: 2024-06 | **Venue**: Interspeech 2024
- **Relevance**: F1 의 frame-phase gate 와 ~45% 겹침 — primary scoop 위험.
- **Contribution**: 입력 의존 binary gate 로 KWS MAC 4배 감소, channel-wise gate, CPU/MCU 타겟.
- **Differentiation vs F1**: SparkNet 은 rank=1, K=2 binary gate. F1 은 rank=2, K=4-8 cluster + CIM + time-axis + non-volatile warm-LUT. **SparkNet 이 F1 의 strict special case** — 이론적 lemma + ablation 명시.

### BiFSMNv2: Binary KWS Ancestor
- **arXiv**: [2211.06987](https://arxiv.org/abs/2211.06987) | **Date**: 2022-11
- **Relevance**: KWS BNN 표준 ancestor. SC v2 93.5%, channel-wise SF, CPU/MCU. F1 의 fair 비교 기준 (NeuroSIM 공통 mapping).

### PSCNN: 885 TOPS/W Digital SRAM-CIM Binary KWS
- **arXiv**: [2205.01569](https://arxiv.org/abs/2205.01569) | **Date**: 2022-05
- **Relevance**: F1 의 HW baseline. digital SRAM-CIM, non-streaming.
- **Differentiation vs F1**: PSCNN 은 batch KWS, streaming 미지원. F1 은 streaming-amortized 이득 + non-volatile warm-LUT. Fair 비교: single-frame 별도 보고.

### CIMR-V: SRAM-CIM + RISC-V for KWS
- **arXiv**: [2503.22072](https://arxiv.org/abs/2503.22072) | **Date**: 2025-03
- **Relevance**: F1 HW baseline reference. Compiler stack 중심, F1 과 orthogonal.

### CIM-Explorer: BNN/TNN Crossbar DSE (SAMOS 2025)
- **arXiv**: [2505.14303](https://arxiv.org/abs/2505.14303) | **Date**: 2025-05
- **Relevance**: BNN/TNN crossbar design-space exploration. PRISM 과 ~30% 겹침.

### Efficient CL BNN-KWS
- **arXiv**: [2505.02469](https://arxiv.org/abs/2505.02469) | **Date**: 2025-05
- **Relevance**: 2025년 유일한 BNN-KWS 논문. 여전히 channel-wise SF — F1 의 gap 확인.

### TTAQ: Test-Time Quantization for Continuous Domain Adaptation
- **arXiv**: [2412.09899](https://arxiv.org/abs/2412.09899) | **Date**: 2024-12
- **Relevance**: F2 PhysioPRISM-VitalXbar 의 baseline 필수 포함. ~35% 겹침 (adjacent).
- **Contribution**: Test-time activation statistics update, FP INT8 모델, non-BNN, non-CIM.
- **Differentiation vs F2**: TTAQ 는 FP INT8 CPU/GPU. F2 는 BNN 1-bit + CIM crossbar + hetero-precision. TTAQ-transferred-to-BNN 이 BNN-CIM 에서 실패/저조 증명이 F2 novelty.

### Tent: Test-Time Adaptation via BN Entropy
- **arXiv**: [2006.10726](https://arxiv.org/abs/2006.10726) | **Date**: 2020-06
- **Relevance**: F2 의 TTA baseline. BN-dependent.
- **Differentiation vs F2**: Tent 는 BN 필수, F2 는 BN-free A&B BNN — 직접 적용 불가. μ_in 기반 F2 방법이 BN-free TTA 등가물.

### AdaBN: Adaptive Batch Normalization
- **arXiv**: [1603.04779](https://arxiv.org/abs/1603.04779) | **Date**: 2016-03
- **Relevance**: F2 의 TTA 원조. activation mean adaptation 개념.

### RTF-Q: Retraining-Free UDA Quantization
- **arXiv**: [2408.05752](https://arxiv.org/abs/2408.05752) | **Date**: 2024-08
- **Relevance**: F2 의 UDA quantization baseline. Retraining-free.

### Arrhythmia-BNN: 4KB Binary ECG
- **arXiv**: [2304.01568](https://arxiv.org/abs/2304.01568) | **Date**: 2023-04
- **Relevance**: F2 의 ECG single-modality baseline. Binary-only 로 MIT-BIH AAMI 5-class 95% sensitivity.
- **Differentiation vs F2**: Arrhythmia-BNN 은 single-modality, no subject adaptation, no CIM. F2 는 binary HAR + INT4 ECG 2-region + subject-adaptive + CIM. "binary-only 95% → INT4 불필요?" 반박은 VT/VF rare class confusion matrix 로.

### MINIMALIST: SC-CIM GRU Biosignal
- **arXiv**: [2505.08599](https://arxiv.org/abs/2505.08599) | **Date**: 2025-05
- **Relevance**: F2 의 biosignal CIM reference. Stochastic Computing GRU, non-binary.

### Transformer HAR Survey
- **arXiv**: [2410.13605](https://arxiv.org/abs/2410.13605) | **Date**: 2024-10
- **Relevance**: F2 의 HAR 양자화 gap 확인.

### HfO2 RRAM BNN for ECG
- **arXiv**: [1908.04066](https://arxiv.org/abs/1908.04066) | **Date**: 2019-08
- **Relevance**: F2 의 RRAM biosignal ancestor. Digital RRAM, no SF/OPTIC.

### Green Wearable SNN-HAR
- **arXiv**: [2604.10458](https://arxiv.org/abs/2604.10458) | **Date**: 2026-04
- **Relevance**: F2 의 adjacent competitor — simulation only, not CIM, single-modality.

### CADC: Crossbar-Aware Dendritic Convolution
- **arXiv**: [2511.22166](https://arxiv.org/abs/2511.22166) | **Date**: 2025-11
- **Relevance**: SNN track (미선정) concurrent work. ~55% 겹침.
- **Contribution**: 80-88% psum sparsity via dendritic nonlinearity embedded in crossbar SNN.

### SOT-MRAM Event-Driven Spiking CIM
- **arXiv**: [2511.03203](https://arxiv.org/abs/2511.03203) | **Date**: 2025-11
- **Relevance**: SNN track primary scoop 위협. ~55% 겹침. **이미 243.6 TOPS/W 달성**.

### SpikeFit: Learned SNN Codebook Quantization
- **arXiv**: [2604.14487](https://arxiv.org/abs/2604.14487) | **Date**: 2026-04
- **Relevance**: SNN scale quantization 경쟁자.

### ASTER: 467× SNN Transformer Acceleration
- **arXiv**: [2511.06770](https://arxiv.org/abs/2511.06770) | **Date**: 2025-11
- **Relevance**: SNN-CIM 열기 증거. SpikeRoute 와 다른 architecture.

### Heterogeneous Delays 1.58-bit SNN
- **arXiv**: [2510.27434](https://arxiv.org/abs/2510.27434) | **Date**: 2025-10
- **Relevance**: Ternary spiking + delay. SpikeXbar mechanism overlap.

### Ternary-Input Binary-Weight DVS Accelerator
- **arXiv**: [2512.00138](https://arxiv.org/abs/2512.00138) | **Date**: 2025-12
- **Relevance**: Digital event accelerator, 7.3× FoM. SpikeRoute analog CIM 차별화 약화 근거.

### IMC-SNN Co-Design Survey
- **arXiv**: [2408.12767](https://arxiv.org/abs/2408.12767) | **Date**: 2024-08
- **Relevance**: SNN-CIM 전반 survey.

### HARP: RRAM KWS Noise Robustness
- **arXiv**: [2604.12420](https://arxiv.org/abs/2604.12420) | **Date**: 2026-04
- **Relevance**: F1 의 noise robustness 실험 baseline. RRAM audio 응용.

---

## 2026-04-21 Mode 1 Session — MoE Expert Fingerprinting (Security + Systems)

### MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs
- **arXiv**: [2508.15036](https://arxiv.org/abs/2508.15036) | **Date**: 2025-08 | **Venue**: CCS 2025
- **Session**: [링크](sessions/2026-04-21-mode1-moe-fingerprinting.md)
- **Relevance**: 이번 세션의 primary threat model. I1 ExpertEcho의 공격 pipeline을 이미 달성 — **선점**. I2 PhantomRoute가 방어해야 할 대상.
- **Contribution**: GPU/CPU 4개 아키텍처 side channel (Cache Occupancy, Pageout+Reload, Performance Counter, TLB Evict+Reload)로 multi-tenant MoE LLM 서버에서 **Prompt Inference Attack 99.8%** (healthcare), VLM 대상 **Visual Inference Attack**, **Response Reconstruction 92.8%** 달성. 대상 모델: DeepSeek-V2, Qwen1.5-MoE, TinyMixtral.
- **Hidden insight**: expert 128개 대 8개의 leakage scaling은 측정하지 않음 → 본 세션이 보완 가능한 gap.
- **Differentiation vs PhantomRoute**: PhantomRoute가 평가할 primary 공격이며, PhantomRoute는 MoEcho의 4개 공격 모두에 대해 평가해야 한다.
- **Action**: vLLM fused-MoE kernel fork 구현 후 MoEcho 공격 harness를 재현 (3-5일) → PhantomRoute 평가의 ground truth로 사용.

### Expert Selections In MoE Models Reveal (Almost) As Much As Text
- **arXiv**: [2602.04105](https://arxiv.org/abs/2602.04105) | **Date**: 2026-02 | **Venue**: arXiv preprint
- **Session**: [링크](sessions/2026-04-21-mode1-moe-fingerprinting.md)
- **Relevance**: expert ID → text 복원의 upper bound (clean oracle 기준).
- **Contribution**: clean expert ID sequence로부터 91.2% token-level text 복원. defense 섹션에서 "dummy compute / constant-work padding" 제안 (= PhantomRoute 메커니즘 스케치).
- **Differentiation vs PhantomRoute**: PhantomRoute는 이들의 defense 제안을 operationalize하고 형식적 MI bound와 실증적 Pareto까지 추가한다.

### Stealing User Prompts from Mixture of Experts
- **arXiv**: [2410.22884](https://arxiv.org/abs/2410.22884) | **Date**: 2024-10 | **Authors**: Yona et al. (Google DeepMind) | **Venue**: arXiv
- **Relevance**: MoE 고유 leakage의 초기 실증.
- **Contribution**: expert-choice routing 버그 + same-batch co-location 악용으로 피해자 prompt 탈취. token-choice routing에서는 재현되지 않아 해당 버그에 특화됨.

### CryptoMoE / SecMoE (암호 baseline)
- **arXiv**: [2511.01197](https://arxiv.org/abs/2511.01197) (CryptoMoE, 2025-11) / [2601.06790](https://arxiv.org/abs/2601.06790) (SecMoE, 2026-01)
- **Relevance**: PhantomRoute의 암호 baseline (100× 느림).

### MoE-Beyond — Expert Activation Prediction
- **arXiv**: [2508.17137](https://arxiv.org/abs/2508.17137) | **Date**: 2025-08
- **Relevance**: premise 검증 — DeepSeek-V2-Chat-Lite에서 97.5% 정확도 (66M traces). 85-90% 전제가 오히려 보수적임을 시사.

### Pre-Attention Expert Prediction
- **arXiv**: [2511.10676](https://arxiv.org/abs/2511.10676) | **Date**: 2025-11
- **Relevance**: 2개의 linear function + ranking loss로 DeepSeek-V2-Lite 93.03%, Qwen3-30B 94.69%. ZMSP의 80μs MLP predictor와 구조적으로 유사.

### OD-MoE: On-Demand MoE
- **arXiv**: [2512.03927](https://arxiv.org/abs/2512.03927) | **Date**: 2025-12
- **Relevance**: predictor 정확도 99.94% 보고 — premise 재조정 근거. cacheless 설계는 VLA의 control period 위반 가능.

### PROBE: Gate-Distilled Lookahead Predictor for MoE
- **arXiv**: [2602.00509](https://arxiv.org/abs/2602.00509) | **Date**: 2026-02
- **Relevance**: ZMSP의 predictor 구성요소 경쟁자. gate 초기화에서 distill한 lookahead predictor.

### BuddyMoE
- **arXiv**: [2511.10054](https://arxiv.org/abs/2511.10054) | **Date**: 2025-11
- **Relevance**: **ZMSP의 가장 가까운 경쟁자** — miss 시 기능적으로 유사한 resident expert로 대체 (품질 저하). ZMSP는 INT4 fallback으로 identity를 유지하고 WCRT bound를 추가해 차별화.

### PreScope / LayerScope
- **arXiv**: [2509.23638](https://arxiv.org/abs/2509.23638) | **Date**: 2025-09
- **Relevance**: **ZMSP의 또 하나의 가까운 경쟁자** — predictive cross-layer MoE 스케줄링 + async I/O, throughput 141% / decode latency 74.6% 단축. ZMSP는 형식적 WCRT bound 추가로 차별화.

### DuoServe-MoE
- **arXiv**: [2509.07379](https://arxiv.org/abs/2509.07379) | **Date**: 2025-09
- **Relevance**: affinity 기반 MoE 서빙, I3/I4의 baseline.

### Semantic Parallelism
- **arXiv**: [2503.04398](https://arxiv.org/abs/2503.04398) | **Date**: 2025-03
- **Relevance**: **FARD-C의 가장 가까운 경쟁자** — SGLang scheduler 확장으로 같은 replica 내에서 token-expert affinity 기반 batch 구성. FARD-C는 **cross-replica dispatch + signature-entropy 정리 + EPLB 보완**으로 차별화.

### Gimbal / Multi-Layer Scheduling
- **arXiv**: [2602.21626](https://arxiv.org/abs/2602.21626) | **Date**: 2026-02
- **Relevance**: **FARD-C의 또 하나의 가까운 경쟁자** — load + affinity 유지 dispatcher (prefix + KV + sticky). FARD-C는 expert footprint 기반이라 직교 축.

### METRO (Token-Level Routing for SLO)
- **arXiv**: [2512.09277](https://arxiv.org/abs/2512.09277) | **Date**: 2025-12
- **Relevance**: FARD-C가 request-level이라면 METRO는 token-level — 보완 축.

### ProMoE, MoE-SpeQ, MoE-SpAc, SP-MoE, ExpertFlow, SliceMoE
- **arXivs**: [2410.22134](https://arxiv.org/abs/2410.22134) / [2511.14102](https://arxiv.org/abs/2511.14102) / [2603.09983](https://arxiv.org/abs/2603.09983) / [2510.10302](https://arxiv.org/abs/2510.10302) / [2510.26730](https://arxiv.org/abs/2510.26730) / [2512.12990](https://arxiv.org/abs/2512.12990)
- **Relevance**: MoE prefetch / speculative / caching 2024-2025 계열 — ZMSP의 baseline.

### HOBBIT (Dynamic Mixed Precision MoE Offloading)
- **arXiv**: [2411.01433](https://arxiv.org/abs/2411.01433) | **Date**: 2024-11
- **Relevance**: ZMSP의 primary 시스템 경쟁자. miss 시 정밀도 강등(품질 손실) 대 ZMSP의 latency bound 유지.

### DualMap, GORGO, BanaServe, XShare
- **arXivs**: [2602.06502](https://arxiv.org/abs/2602.06502) / [2602.11688](https://arxiv.org/abs/2602.11688) / [2510.13223](https://arxiv.org/abs/2510.13223) / [2602.07265](https://arxiv.org/abs/2602.07265)
- **Relevance**: dispatcher / routing / cache 계열 (2025-2026). FARD-C 관련 문헌.

### EPLB (DeepSeek-V3 Expert Parallel Load Balancing)
- **Source**: DeepSeek-V3 tech report, 2024-12
- **Relevance**: supply-side expert 재배치. FARD-C는 demand-side routing으로 보완 관계.

### Task-Conditioned Routing Signatures
- **arXiv**: [2603.11114](https://arxiv.org/abs/2603.11114) | **Date**: 2026-03
- **Relevance**: task 수준 routing 분류 92.5% — premise 검증 근거.

### 네트워크 계층 LLM side channel (맥락용)
- **Whisper Leak** ([arXiv:2511.03675](https://arxiv.org/abs/2511.03675), 2025-11), **NetEcho** ([arXiv:2510.25472](https://arxiv.org/abs/2510.25472), 2025-10), **"I Know What You Said"** ([arXiv:2505.06738](https://arxiv.org/abs/2505.06738), 2025-05)
- **Relevance**: LLM side-channel 영역이 이미 성숙했음을 보여주는 ecosystem 신호.

### VLA 관련 (deferred I5 FF-ACE-VLA 대비)
- **AdaMoE-VLA** ([arXiv:2510.14300](https://arxiv.org/abs/2510.14300), 2025-10), **HiMoE-VLA** ([arXiv:2512.05693](https://arxiv.org/abs/2512.05693), 2025-12), **ActionFlow** ([arXiv:2512.20276](https://arxiv.org/abs/2512.20276), 2025-12), **VLA-Cache** ([arXiv:2502.02175](https://arxiv.org/abs/2502.02175), 2025-02), **Importance-Driven Expert Scheduling** ([arXiv:2508.18983](https://arxiv.org/abs/2508.18983), 2025-08)
- **Relevance**: VLA-MoE 서빙 ecosystem. FF-ACE-VLA 재방문 시 baseline/target pool.

### 인접 보안 / 프라이버시
- **NoEsis** ([arXiv:2504.18147](https://arxiv.org/abs/2504.18147), 2025-04) DP-MoE training MIA — 직교 위협.
- **CacheSolidarity** ([arXiv:2603.10726](https://arxiv.org/abs/2603.10726), 2026-03) prefix-cache timing 방어 — PhantomRoute와 보완 관계.
- **Selective KV-Cache Sharing** ([arXiv:2508.08438](https://arxiv.org/abs/2508.08438), 2025-08) KV timing 방어 — 서술 인접.
- **RepetitionCurse** ([arXiv:2512.23995](https://arxiv.org/abs/2512.23995), 2025-12) router DoS — routing 방어 계층의 필요성을 뒷받침.

---

## 2026-04-21 Mode 2 Session — ACE-MoE VLM/VLA Extension (prior session)

### ACE-MoE: Skip Waiting, Not Accuracy — Time-Budgeted Skipping and Accuracy-Critical Expert Caching for Efficient MoE Inference (ICCAD'26 submission)
- **Date analyzed**: 2026-04-21
- **Session**: [링크](sessions/2026-04-21-mode2-ace-moe-vlm-vla-extension.md)
- **Expert**: ai-optimization-expert, algorithm-expert
- **Paper-id**: iccad-2026-ace-moe (own work)
- **Contribution**: MoE 추론에서 expert offloading의 GPU waiting time 제거. 3가지 핵심 — (1) ACE caching: hit-rate 대신 accuracy-critical expert preservation 목표, variance-aware global selection으로 layer sensitivity 반영. (2) Time-budgeted skipping & prefetching: prefill에서 sequence-level aggregate score, decoding에서 top-(K+α) gate logit prefetch, 4-bit quantized transfer. (3) Fused kernel: cached(BF16)/prefetched(4-bit)/skipped 3종을 단일 kernel로. RTX Pro 6000 96GB + PCIe 5.0x16 환경에서 fine-grained MoE 3종 (DeepSeek-V2-Lite, Qwen1.5-MoE, Qwen3-30B-A3B)에 평가, prefill 11.0×, decoding 1.3× speedup vs HybriMoE [DAC'25], 50% cache ratio에서 99.3-99.9% relative accuracy.
- **Hidden Insights**:
  1. **Figure 4** — prefill에서 64+ 토큰이면 90%+ expert 활성화 → caching/prefetching 거의 무력화. 이 현상은 VLM(이미지 토큰 100~2,040)에서 극단화되어 본 idea 확장의 강력한 근거.
  2. **Figure 9** — ACE-MoE가 실제 사용하는 expert가 51%(prefill, Qwen3-30B-A3B). 49%는 어차피 skip되어도 accuracy 유지 → 잠재 redundancy 매우 큼.
  3. **Figure 10** — cache가 첫 4 requests 안에 안정화, 이후 replacement <5%. workload-specific stable expert set이 빠르게 형성 → cross-request sharing의 통계적 기반.
  4. **Figure 11(b)** — prefetch budget 0.4×~0.6× 만으로도 baseline 정확도 도달 (DeepSeek-V2-Lite, Qwen3-30B-A3B). PCIe bandwidth 절반 환경에서도 robust → consumer GPU/edge에 적합.
  5. **Table 3** — 4-bit이 best, 2-bit는 큰 accuracy drop, 8-bit과 4-bit 차이 미미. layer/expert별 dynamic bit-width 여지.
  6. **Figure 12** — ACE caching의 hit rate가 LRU/LFU/MRS보다 낮은데도 accuracy는 더 높음. **Hit rate ≠ accuracy preservation**. 이 insight를 VLM의 visual KV(sequence의 86.2%, attention 11.4%)에도 적용 가능.
  7. **Layer 0 처리** — prefetch prediction 불가 → quantized 형태로 GPU 상주 (model param의 <1%). VLM의 vision-language interface layer (merger, DeepStack inject layer)에 응용 가능.
  8. **Table 2 NAEE 비교** — static pruning은 cache ratio 25%에서 50.5% relative accuracy로 무너지는데 ACE-MoE는 97.2%. dynamic adaptive vs static 격차 큼 → multi-modal에서는 input-dependent 변동이 더 커 격차가 더 벌어질 가능성.
- **Relevance**: 본 paper의 ACE caching 메커니즘과 time-budgeted concept을 VLM/VLA로 확장하는 5개 후속 idea의 직접적 source. 결론부에 명시적으로 VLM/VLA 확장 future direction 적시됨.
- **Limitations**:
  - 단일 modality (text)에 한정, modality-aware ACE 식별 미수행
  - batch size 1 (interactive inference 가정), batch >1 미평가
  - Vision-encoder 같은 GPU-only compute 단계 미고려
  - VLA real-time constraint 별도 정의 필요
  - 4-bit quantization은 정적 (HQQ), expert/layer별 적응적 bit-width 미탐구

---

### VLM-aware Heterogeneous KV Management for GPU+PIM Systems (Internal report by 연구자, 2026-04-07)
- **Date analyzed**: 2026-04-21
- **Session**: [링크](sessions/2026-04-21-mode2-ace-moe-vlm-vla-extension.md)
- **Expert**: ai-optimization-expert (PIM contribution은 제외, software-side measurement만)
- **Paper-id**: internal-2026-vlm-pim-exploration
- **Contribution (software-side만)**: Qwen3-VL-4B vs Qwen3-4B를 비교한 measurement-driven 분석. AttAcc(ASPLOS'24)를 baseline으로 하는 PIM 확장 motivation으로 작성되었으나 본 세션에서는 software-side observation만 활용. (1) VLM TTFT explosion 측정 (672x672 6.13×, FHD 22.4×, MMMU real 12.4×). (2) Layer-wise visual attention asymmetry (L17-21 dense 24.5%, L0-7 sparse 2.6%, peak L18=30.7%). (3) Visual KV size dominance (sequence의 86.2%, attention 11.4%, BW waste 7.6×). (4) DeepStack architecture (Qwen3-VL이 ViT 중간 출력을 LLM L4/L8/L12 등에 inject) 분석. (5) Same-image cross-request sharing 가능성 (BS=128에서 38% memory saving).
- **Hidden Insights (software-side로 일반화)**:
  1. ACE-MoE의 "skipping이 prefill에서 안 통한다(token-level)"의 정확한 mirror가 VLM에 존재 — visual KV는 sequence-level로 중요도 평가해야. ACE-MoE의 sequence-level aggregate score 메커니즘이 직접 적용 가능.
  2. L0-7의 visual attention 2.6%는 ACE-MoE에서 cumulative score가 매우 낮은 expert와 동일 카테고리 → layer-position과 modality 둘 다 importance에 영향, 2D variance-aware selection 가능.
  3. DeepStack inject layer는 ACE-MoE의 layer 0 처리(quantized GPU resident)와 유사한 특수 처리 필요. prediction 어렵고 importance 변동.
  4. Cross-request sharing은 ACE-MoE의 ACE 안정화(figure 10)와 유사한 통계적 안정성 보임 → sharing 가능 entity의 단위가 expert에서 visual KV로 확장.
  5. Real workload (MMMU) TTFT std=301ms — variable image size가 long tail latency 유발, M1 fixed input은 underestimate. SLO-aware scheduling 필요.
  6. ViT encoding이 prefill의 32-36% (30-85ms) — vision encoder 자체 software 최적화 (token merging, early exit)도 효과 큼.
- **Relevance**: VLM/VLA 확장 idea의 measurement evidence 제공. ACE-MoE의 future work 방향에 대한 정량적 근거.
- **Limitations (software-side로 봤을 때)**:
  - Qwen3-VL-4B 단일 모델 측정 (LLaVA, InternVL 등 generalize 미검증)
  - VLM-MoE 모델 측정 부재 (Qwen3-VL은 dense)
  - Decode tail latency / SLO 분석 부족
  - 본 report의 핵심 contribution은 PIM placement 관련이며, software-side observation은 motivation으로만 활용됨 (PIM solution은 다른 그룹 담당, 본 세션 scope 외)

---

_그 외에 ACE-MoE bibliography 7편 (NAEE, HybriMoE, Pre-gated MoE, Hobbit, Edge-MoE, Cache-Conditional MoE, Yu et al.)이 baseline/관련연구로 인용됨. 상세는 세션 파일 참조._

---

## 2026-04-21 Related Work Augmentation — arxiv 검색 결과 (2024-01 ~ 2026-04)

초기 Phase 2 리뷰에서 지적된 "related work 탐색 빈약" 이슈를 해소하기 위해 arxiv API (WebFetch)로 추가 수집한 관련 논문 목록. 원본 두 논문(ACE-MoE, VLM exploration)의 확장 검토용 baseline / closest competitor 후보들.

### VLA Inference Optimization (I5 보강용)

### AdaMoE-VLA: Adaptive Mixture-of-Experts for Vision-Language-Action Learning
- **arXiv**: [2510.14300](https://arxiv.org/abs/2510.14300) | **Date**: 2025-10-16 | **Authors**: Weijie Shen, Yitian Liu, Yuhao Wu 외 13명
- **OpenReview**: [cNZ5W1f4tE](https://openreview.net/forum?id=cNZ5W1f4tE) | **Code**: [github.com/swjTheDad/AdaMoE-VLA](https://github.com/swjTheDad/AdaMoE-VLA)
- **Relevance**: **I5 ACE-VLA의 가장 직접적 선례 (training-time VLA-MoE)** — 처음으로 dense VLA의 FFN을 sparse MoE로 대체한 구체적 연구
- **Contribution**: (1) Pretrained dense VLA weight 상속 + action expert scale-up. (2) **Scale Adapter**: expert selection과 weighting을 decouple — "collaborative" expert utilization, winner-takes-all 방지. (3) LIBERO +1.8%, RoboTwin +9.3%, real-world +21.5% 성능 향상.
- **Architecture**: Dense VLA FFN → sparse-activated MoE, 4 routed experts (configurable), top-K=1~N. Base는 **OpenPI (Pi-0 계열)** repository.
- **Training**: MultiGroupAdamW optimizer (base/MoE/router 별 separate lr & weight decay). Training + inference 코드는 공개, **checkpoint 미공개** (OpenReview decision 이후 가능성).
- **Hidden insight / strong claim**: 기존 MoE의 selection과 weighting coupling이 VLA에서 최적화 충돌을 일으킴 — Scale Adapter로 decouple하면 expert가 "서로 협력적으로" 기여하도록 routing signal과 contribution signal을 분리 학습. **이 decoupled signal은 ACE-VLA의 cumulative score 계산에 그대로 활용 가능** (scale value = explicit contribution weight).
- **Differentiation vs I5 (ACE-VLA)**: AdaMoE-VLA는 **training-time 아키텍처 변경**. ACE-VLA는 **inference-time expert caching/skipping + hard real-time budget**. **Strictly complementary** — AdaMoE-VLA가 제공하는 VLA-MoE base 위에 ACE-VLA inference 최적화를 얹는 것이 자연스러움.
- **Action**: **I5 ACE-VLA의 base model로 AdaMoE-VLA 채택을 1순위 권장**. OpenVLA retrofit recipe는 alternative/ablation으로 보존. AdaMoE-VLA 재현 (Pi-0 base, LIBERO) ≈ 5-10일 (#5 RTX Pro 6000 + LoRA), ACE-VLA 적용 추가 2-3주. 총 5-7주 연구 roadmap이 현실적.

### ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge
- **arXiv**: [2512.20276](https://arxiv.org/abs/2512.20276) | **Date**: 2025-12-23 | **Authors**: Yuntao Dai, Hang Gu, Teng Wang 외
- **Relevance**: I5의 **strongest closest competitor**
- **Contribution**: Cross-Request Pipelining scheduler for autoregressive VLA decoding. Batches memory-bound decode with compute-bound prefill. "2.55× FPS improvement on OpenVLA-7B" without retraining.
- **Hidden insight**: pipeline scheduling만으로도 2.55× 획득 — expert-level skip 없이도 substantial gain, ACE-VLA가 expert + budget + freshness를 더하면 추가 gain 확보 가능
- **Differentiation vs I5**: expert-level skip/cache 없음. ACE-VLA가 expert axis + hard real-time budget 추가

### HyperVLA: Efficient Inference in Vision-Language-Action Models via Hypernetworks
- **arXiv**: [2510.04898](https://arxiv.org/abs/2510.04898) | **Date**: 2025-10-06 | **Authors**: Zheng Xiong, Kang Li, Zilin Wang
- **Relevance**: I5 alternative acceleration approach
- **Contribution**: hypernetwork-based activation of task-specific policies. 90× activated param reduction, 120× inference speedup vs OpenVLA.
- **Differentiation vs I5**: architectural change (hypernetwork) vs runtime scheduling (ACE-VLA). Orthogonal — 결합 가능

### CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification
- **arXiv**: [2508.21046](https://arxiv.org/abs/2508.21046) | **Date**: 2025-08-28 | **Authors**: Wei Li, Renshan Zhang, Rui Shao
- **Relevance**: **NeurIPS 2025**, instruction-driven sparsification
- **Contribution**: 97.4% LIBERO, 70% real-world, 2.5× training cost reduction, 2.8× latency reduction
- **Differentiation vs I5**: instruction-driven sparsification (Phase 0 시각적 입력 자체를 줄임). ACE-VLA는 runtime expert scheduling — orthogonal combination 가능

### A1: Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model
- **arXiv**: [2604.05672](https://arxiv.org/abs/2604.05672) | **Date**: 2026-04-07 | **Authors**: Kaidong Zhang, Jian Zhang, Rongtao Xu
- **Relevance**: **budget-aware VLA**, I5의 키워드 "budget"과 정면 overlap
- **Contribution**: budget-aware adaptive inference + action consistency monitoring + Inter-Layer Truncated Flow Matching. **72% per-episode latency reduction**. 76.6% backbone reduction.
- **Differentiation vs I5**: A1은 **episode-level budget + model truncation**, ACE-VLA는 **action chunk-level budget + expert offloading**. 서로 다른 granularity와 optimization axis. 결합 시 compounding gain 가능

### SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation
- **arXiv**: [2511.10518](https://arxiv.org/abs/2511.10518) | **Date**: 2025-11-13 | **Authors**: Wei Li, Renshan Zhang, Rui Shao
- **Relevance**: visual-side pruning for VLA
- **Contribution**: semantic-guided visual pruning across dual encoders. OpenVLA on LIBERO surpassed by 21.1%, 3.0× training cost reduction, 2.7× latency reduction
- **Differentiation vs I5**: visual side only, ACE-VLA는 expert axis까지. Orthogonal

### Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation
- **arXiv**: [2509.22093](https://arxiv.org/abs/2509.22093) | **Date**: 2025-09-26 | **Authors**: Xiaohuan Pei, Yuxing Chen, Siyu Xu
- **Relevance**: action trajectory-conditioned token pruning
- **Contribution**: action stage별 pruning 수준 조절, 1.35× on OpenVLA-OFT + 25.8% success rate improvement
- **Differentiation vs I5**: token axis, action-stage conditioning은 I5의 action-chunk 개념과 친화적. 결합 검토

### HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation
- **arXiv**: [2604.07993](https://arxiv.org/abs/2604.07993) | **Date**: 2026-04-09 | **Authors**: Shuanghao Bai, Meng Li, Xinyuan Lv
- **Relevance**: **MoE in VLA — direct precedent**
- **Contribution**: Mixture-of-Experts Unified Proprioceptive Predictor for humanoid + residual-gated fusion. State-of-the-art in fast-reaction scenarios.
- **Hidden insight**: humanoid-specific이지만 **VLA에 MoE 적용 가능성을 공식적으로 입증**. I5의 "VLA-MoE retrofit" contribution 강화 근거
- **Differentiation vs I5**: HEX는 training architecture, ACE-VLA는 inference serving

### HY-Embodied-0.5 (Tencent Robotics X)
- **arXiv**: [2604.07430](https://arxiv.org/abs/2604.07430) | **Date**: 2026-04-08 | **Authors**: Tencent Robotics X team
- **Relevance**: Mixture-of-Transformers for embodied foundation models
- **Contribution**: 2B와 32B active-parameter 변종, on-policy distillation, 16 benchmarks에서 동크기 모델 능가
- **Hidden insight**: large-scale embodied foundation 모델이 Mixture-of-Experts/Transformers를 채택하는 **trend** 명확화

### DA-PTQ: Drift-Aware Post-Training Quantization for Efficient VLA
- **arXiv**: [2604.11572](https://arxiv.org/abs/2604.11572) | **Date**: 2026-04-13 | **Authors**: Siyuan Xu, Tianshi Wang, Fengling Li
- **Relevance**: VLA quantization, temporal drift 해결
- **Contribution**: Cross-Space Representation Compensation + Motion-Driven Mixed-Precision
- **Differentiation vs I5**: weight-level quantization, expert-level과 orthogonal

### QVLA, HBVLA (VLA quantization baselines)
- **QVLA** [arXiv:2602.03782](https://arxiv.org/abs/2602.03782) (2026-02): channel-wise bit allocation, 1.49× speedup OpenVLA-OFT
- **HBVLA** [arXiv:2602.13710](https://arxiv.org/abs/2602.13710) (2026-02): 1-bit PTQ, 92.2% full-precision retention
- **Relevance**: I5 quantization baseline (orthogonal combination 가능)

### Adaptive Action Chunking at Inference-time for Vision-Language-Action Models
- **arXiv**: [2604.04161](https://arxiv.org/abs/2604.04161) | **Date**: 2026-04-05
- **Relevance**: **I5의 "action chunk별 ACE update" 부분과 정확히 매칭**
- **Contribution**: action entropy로 chunk size 동적 결정
- **Action**: I5에서 chunk 크기 결정에 이 기법 채택 검토

### VLAgents: Policy Server for Efficient VLA Inference
- **arXiv**: [2601.11250](https://arxiv.org/abs/2601.11250) | **Date**: 2026-01-16
- **Relevance**: **system-level competitor/partner** for I5
- **Contribution**: modular policy server, zero-copy shared memory, compressed streaming
- **Action**: I5를 VLAgents 위에 얹을 수 있음 — serving infrastructure 협업

### BFA++: Hierarchical Best-Feature-Aware Token Prune for Multi-View VLA
- **arXiv**: [2602.20566](https://arxiv.org/abs/2602.20566) | **Date**: 2026-02-21
- **Relevance**: multi-view VLA token pruning baseline for I5
- **Contribution**: 1.8×/1.5× speedup + success rate improvements

### Shallow-π: Knowledge Distillation for Flow-based VLAs
- **arXiv**: [2601.20262](https://arxiv.org/abs/2601.20262) | **Date**: 2026-01-28
- **Relevance**: I5 orthogonal distillation baseline

---

### MoE Expert Offloading / Scheduling (I3 보강용)

### DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge
- **arXiv**: [2603.19172](https://arxiv.org/abs/2603.19172) | **Date**: 2026-03-19 | **Authors**: Yuegui Huang, Zhiyuan Fang, Weiqi Luo
- **Relevance**: **I3의 strongest closest competitor**
- **Contribution**: importance-aware prioritization + depth-adaptive scheduling + look-ahead prefetching + mixed-precision quantization. **edge TTFT 3.44-22.7× 개선**.
- **Hidden insight**: depth-adaptive scheduling + importance가 이미 존재. I3는 이를 **token axis까지 확장한 joint budget** — 차별화 핵심 포인트 명확화 필요
- **Differentiation vs I3**: expert axis + depth만 다룸. I3의 차별: **token retention + expert prefetch의 joint layer-wise budget, visual KV asymmetry 활용**

### Dynamic Expert Quantization for Scalable MoE Inference
- **arXiv**: [2511.15015](https://arxiv.org/abs/2511.15015) | **Date**: 2025-11-19 | **Authors**: Kexin Chu, Dawei Xiang, Zixu Shen
- **Relevance**: **budget-constrained formulation 선례**
- **Contribution**: "online budget-constrained precision allocation problem" 정식화. hot expert 높은 precision, cold expert fallback. 2.73× throughput vs offloading/prefetch.
- **Differentiation vs I3**: single-axis (precision) budget. I3는 multi-axis (token × expert × layer × bandwidth)

### Context-Aware MoE Inference on CXL-Enabled GPU-NDP Systems
- **arXiv**: [2512.04476](https://arxiv.org/abs/2512.04476) | **Date**: 2025-12-04
- **Relevance**: context-aware per-expert bit-width allocation (1-4 bit), 8.7× throughput
- **Differentiation vs I3**: CXL-NDP specific hardware

### SliceMoE: Bit-Sliced Expert Caching under Miss-Rate Constraints
- **arXiv**: [2512.12990](https://arxiv.org/abs/2512.12990) | **Date**: 2025-12-18
- **Relevance**: slice-level granularity bit caching, 2.37-2.85× energy reduction
- **Differentiation vs I3**: expert axis within single expert (slicing)

### MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching
- **arXiv**: [2511.14102](https://arxiv.org/abs/2511.14102) | **Date**: 2025-11-18
- **Relevance**: speculative + prefetch + quantize 3-way co-design, 2.34× speedup
- **Differentiation vs I3**: draft model 기반 speculation, I3는 workload pattern + budget

### SpecMoE: Fast and Efficient MoE via Self-Assisted Speculative Decoding
- **arXiv**: [2604.10152](https://arxiv.org/abs/2604.10152) | **Date**: 2026-04-11
- **Relevance**: speculative decoding for MoE throughput 4.30×
- **Differentiation vs I3**: token generation 측면, I3의 prefill token budget과 다른 axis

### DALI: Workload-Aware Offloading Framework for MoE on Local PCs
- **arXiv**: [2602.03495](https://arxiv.org/abs/2602.03495) | **Date**: 2026-02-03
- **Relevance**: residual-based prefetching + workload-aware cache replacement
- **Differentiation vs I3**: expert axis only, I3는 token+expert joint

### MELINOE: Fine-Tuning Enables Memory-Efficient MoE Inference
- **arXiv**: [2602.11192](https://arxiv.org/abs/2602.11192) | **Date**: 2026-01-30
- **Relevance**: fine-tuning으로 expert churn 감소, 1.2-3× throughput
- **Differentiation vs I3**: training-time intervention

### MoBiLE: Efficient MoE Inference on Consumer GPU with Mixture of Big Little Experts
- **arXiv**: [2510.12357](https://arxiv.org/abs/2510.12357) | **Date**: 2025-10-14
- **Relevance**: important token = full expert, unimportant = half expert, 1.6-1.72×
- **Differentiation vs I3**: expert size를 중요도별로 조절 (interesting concept). I3는 token retention rate 조절로 다른 mechanism

### BuddyMoE: Expert Redundancy to Accelerate Memory-Constrained MoE
- **arXiv**: [2511.10054](https://arxiv.org/abs/2511.10054) | **Date**: 2025-11-13
- **Relevance**: prefetch 실패 시 redundancy 활용 fallback
- **Action**: I3에 fallback 메커니즘 추가 검토

### FlashMoE: SSD I/O Bottleneck via ML-Based Cache Replacement
- **arXiv**: [2601.17063](https://arxiv.org/abs/2601.17063) | **Date**: 2026-01-22
- **Relevance**: SSD offloading, 51% hit rate improvement vs LRU/LFU

### TriMoE: GPU-CPU-NDP with AMX-Enabled CPU and DIMM-NDP
- **arXiv**: [2603.01058](https://arxiv.org/abs/2603.01058) | **Date**: 2026-03-01
- **Relevance**: 2.83× speedup via heterogeneous compute

### DWDP: Distributed Weight Data Parallelism on NVL72
- **arXiv**: [2604.01621](https://arxiv.org/abs/2604.01621) | **Date**: 2026-04-02
- **Relevance**: peer-GPU expert offloading, 8.8% throughput gain

### OD-MoE: On-Demand Expert Loading for Cacheless Edge
- **arXiv**: [2512.03927](https://arxiv.org/abs/2512.03927) | **Date**: 2025-12-03
- **Relevance**: 99.94% prediction accuracy, 75% fully-cached speed with 1/3 memory

### Bandwidth-Efficient Adaptive MoE via Low-Rank Compensation
- **arXiv**: [2512.17073](https://arxiv.org/abs/2512.17073) | **Date**: 2025-12-18
- **Relevance**: low-rank compensator + top-n experts at full precision

### FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity
- **arXiv**: [2602.16603](https://arxiv.org/abs/2602.16603) | **Date**: 2026-02-18
- **Relevance**: SARATHI follow-up, operator-level preemption

### QUOKA: Query-Oriented KV Selection for Efficient LLM Prefill
- **arXiv**: [2602.08722](https://arxiv.org/abs/2602.08722) | **Date**: 2026-02-09
- **Relevance**: **3× TTFT + 5× attention speedup with 88% KV reduction**. I3의 token axis와 매우 유사
- **Differentiation vs I3**: text-only, I3는 visual KV + expert joint

### LAPS: Length-Aware-Prefill LLM Serving System
- **arXiv**: [2601.11589](https://arxiv.org/abs/2601.11589) | **Date**: 2026-01-04
- **Relevance**: long-prefill vs short-prefill disaggregation, 30%+ prefill latency reduction
- **Differentiation vs I3**: instance-level routing, I3는 layer-level budget

### Prefill-as-a-Service
- **arXiv**: [2604.15039](https://arxiv.org/abs/2604.15039) | **Date**: 2026-04-16
- **Relevance**: cross-datacenter KVCache transfer with hybrid attention
- **Differentiation vs I3**: deployment scale 다름

### TokenDance: Multi-Agent LLM Serving via Collective KV Cache Sharing
- **arXiv**: [2604.03143](https://arxiv.org/abs/2604.03143) | **Date**: 2026-04-03
- **Relevance**: 17.5× per-agent storage reduction in multi-agent pattern

### Staggered Batch Scheduling (DP+EP)
- **arXiv**: [2512.16134](https://arxiv.org/abs/2512.16134) | **Date**: 2025-12-18

### Efficient Long-Horizon GUI Agents via Training-Free KV Cache Compression (ST-Lite)
- **arXiv**: [2603.00188](https://arxiv.org/abs/2603.00188) | **Date**: 2026-02-27
- **Relevance**: GUI agent 특화지만 KV compression의 general pattern

---

### VLM-MoE / Modality-Aware (I1 보강용)

### MoE-LLaVA: Mixture of Experts for Large Vision-Language Models
- **arXiv**: [2401.15947](https://arxiv.org/abs/2401.15947) | **Date**: 2024-01-29 | **Authors**: Bin Lin et al.
- **Relevance**: **foundational VLM-MoE model**, I1의 primary evaluation target
- **Contribution**: MoE-Tuning with sparsity learning degradation mitigation. top-k expert activation.
- **Differentiation vs I1**: 아키텍처/학습, serving 최적화 미제시. I1이 이 위에 ACE caching 추가

### LLaVA-MoLE: Sparse Mixture of LoRA Experts
- **arXiv**: [2401.16160](https://arxiv.org/abs/2401.16160) | **Date**: 2024-01-29
- **Relevance**: LoRA-level MoE for data conflicts
- **Differentiation vs I1**: training-time, LoRA level — orthogonal

### LLaVA-MoD: Making LLaVA Tiny via MoE KD
- **arXiv**: [2408.15881](https://arxiv.org/abs/2408.15881) | **Date**: 2024-08-28
- **Relevance**: 2B activated surpasses Qwen-VL-Chat-7B via MoE distillation
- **Differentiation vs I1**: training-time distillation

### LLaVA-CMoE: Continual MoE for VLM
- **arXiv**: [2503.21227](https://arxiv.org/abs/2503.21227) | **Date**: 2025-03-27
- **Relevance**: Probe-Guided expert placement + VAE router, continual learning

### MoTE: Mixture of Ternary Experts for Memory-efficient Multimodal Models
- **arXiv**: [2506.14435](https://arxiv.org/abs/2506.14435) | **Date**: 2025-06-17
- **Relevance**: more low-precision experts vs fewer high-precision. Quantization compatible.
- **Differentiation vs I1**: precision allocation level, I1은 caching level

### QMoP: Query Guided Mixture-of-Projector for Visual Token Compression
- **arXiv**: [2603.21232](https://arxiv.org/abs/2603.21232) | **Date**: 2026-03-22
- **Relevance**: MoE-style fusion for visual token compression with query guidance
- **Differentiation vs I1**: visual projector level, I1은 LLM expert level

### AlignMamba-2: Modality-Aware Mamba with Modality-Specific + Modality-Shared Experts
- **arXiv**: [2603.18462](https://arxiv.org/abs/2603.18462) | **Date**: 2026-03-19
- **Relevance**: **I1의 철학과 가장 가까운 precedent** — modality-specific + modality-shared expert 구분
- **Differentiation vs I1**: multimodal sentiment analysis 특화, state-space model 기반 (Mamba), I1은 inference serving 최적화 framework 추가

### Qwen3.5-Omni Technical Report
- **arXiv**: [2604.15804](https://arxiv.org/abs/2604.15804) | **Date**: 2026-04-17 | **Authors**: Qwen Team
- **Relevance**: **state-of-the-art VLM-MoE**, Hybrid Attention MoE for Thinker+Talker, 256k context, hundreds of billions params
- **Hidden insight**: **large-scale commercial VLM-MoE가 이미 실재**. I1의 evaluation target 후보 (공개되면)

### ERNIE 5.0 Technical Report
- **arXiv**: [2602.04705](https://arxiv.org/abs/2602.04705) | **Date**: 2026-02-04
- **Relevance**: **ultra-sparse MoE with modality-agnostic expert routing**
- **Hidden insight**: **I1의 반대 철학**. "modality-agnostic이 더 낫다"는 counter-design. I1 paper에서 직접 비교 실험 필수

### Efficient Quantization of MoE with Theoretical Generalization Guarantees
- **arXiv**: [2604.06515](https://arxiv.org/abs/2604.06515) | **Date**: 2026-04-07
- **Relevance**: expert-wise mixed precision with theoretical bound. **I1에 per-expert quantization 통합 검토**

### Omni-C: Compressing Heterogeneous Modalities into Single Dense Encoder
- **arXiv**: [2603.05528](https://arxiv.org/abs/2603.05528) | **Date**: 2026-02-27
- **Relevance**: **MoE 자체를 부정하는 counter-design** — dense encoder alternative
- **Action**: I1의 positioning에서 dense alternative 언급 필요

### Improving MoE Compute Efficiency via Null Experts
- **arXiv**: [2601.15370](https://arxiv.org/abs/2601.15370) | **Date**: 2026-01-21
- **Relevance**: zero-compute null expert + weight/data sparsity composition
- **Action**: I1에 null expert 개념 추가 검토

---

### Visual KV Cache Compression (I2 미선정의 baseline, I1/I3 보강)

### VL-Cache: Sparsity and Modality-Aware KV Cache Compression
- **arXiv**: [2410.23317](https://arxiv.org/abs/2410.23317) | **Date**: 2024-10-29 | **Authors**: Dezhan Tu et al.
- **Relevance**: **layer-adaptive sparsity-aware cache budget**, 7.08× decoding speedup at 10% cache
- **Differentiation vs I1**: KV level, I1은 expert level — orthogonal combination 가능

### AirCache: Inter-modal Relevancy KV Cache Compression
- **arXiv**: [2503.23956](https://arxiv.org/abs/2503.23956) | **Date**: 2025-03-31
- **Relevance**: 10% visual KV retention, 29-66% decoding latency reduction

### LightVLM: Pyramid Token Merging + KV Cache Compression
- **arXiv**: [2509.00419](https://arxiv.org/abs/2509.00419) | **Date**: 2025-08-30
- **Relevance**: 3.21× inference time reduction

### MHA2MLA-VLM: MLA for VLM
- **arXiv**: [2601.11464](https://arxiv.org/abs/2601.11464) | **Date**: 2026-01-16
- **Relevance**: DeepSeek MLA를 VLM에 transfer

### OmniSparse: Training-Aware Fine-Grained Sparse Attention for Long-Video MLLMs
- **arXiv**: [2511.12201](https://arxiv.org/abs/2511.12201) | **Date**: 2025-11-18
- **Relevance**: head-level dynamic budget + KV slimming, 2.7× prefill, 2.4× memory

### QSVD: Low-rank Q/K/V Compression for Low-Precision VLMs
- **arXiv**: [2510.16292](https://arxiv.org/abs/2510.16292) | **Date**: 2025-10-18

### AutoSelect, PixelPrune, VisPCO, ConsensusDrop (visual token pruning 최신)
- **AutoSelect** [arXiv:2603.07135](https://arxiv.org/abs/2603.07135) (2026-03): 96.5% accuracy + 2.85× prefill speedup
- **PixelPrune** [arXiv:2604.00886](https://arxiv.org/abs/2604.00886) (2026-04): 4.2× inference speedup
- **VisPCO** [arXiv:2604.15188](https://arxiv.org/abs/2604.15188) (2026-04): Pareto-frontier token pruning
- **ConsensusDrop** [arXiv:2602.00946](https://arxiv.org/abs/2602.00946) (2026-02): vision + cross-modal saliency fusion
- **Relevance**: I2 미선정의 crowded field 입증 자료

---

_보강된 관련 연구는 세션 파일 Phase 7에서 Top 3 idea별 차별점 재평가에 활용됨. 각 idea의 revised score는 세션 Phase 7-E 참조._
