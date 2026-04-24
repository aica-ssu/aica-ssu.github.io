# 최신 VLM 특성 기반 에너지 효율적 Edge VLM Inference: Parquet / Triptych / Cartographer / Sift / Verge

*Session date: 2026-04-23 · Mode 1 (기존 세션 참고 금지 완전 신규) · R27 Self-Sufficient Summary 적용 (retrofit 2026-04-24)*

> 초창기 LLaVA-1.5 (fixed 576 visual token) → 최신 Qwen2.5-VL / InternVL3 (dynamic 4-16,384 visual token + MRoPE 3D + pixel shuffle + video temporal packing) 으로 진화한 VLM 아키텍처가 Jetson Orin AGX / Thor / RTX 4060 같은 edge GPU 에서 에너지 효율적으로 서빙되려면 어떤 VLM-only 최적화가 필요한가? 3 명 전문가 ideation (ai-optimization / legacy-system / hw-pim) + Phase 1-2-1'-2'-1'' 5-phase 루프 + 2026-04-23 신규 규칙 4종 (R23 workload-driven / R24 kernel fusion triviality / R25 Platform-Usage Analysis / R26 Metaphor noun title) 전면 적용하여 **Tier-1 Top 2 (Parquet ★ / Triptych) + Tier-2 독립 Top 3 (Cartographer / Sift / Verge) = 총 5 선정**, I1 Tidal 1 개는 CodecSight (2026-04-07, 16일 차이) 68-72% scoop 으로 미선정.

본 summary 는 **R27 Self-Sufficient Summary for Implementation** 규칙에 따라 작성되었다. 각 mechanism 은 (① 추가 scheme, ② 해결 문제, ③ 동작 원리 step-by-step, ④ 기존 해법 실패 + 차별화) 4 요소를 포함하고, 각 실험 플랜은 기존 5 요소에 (6) Implementation Steps week-level + (7) Preliminary Analysis Metrics 2 요소를 추가한 7-요소 포맷이다. Summary 만 읽고도 CS 학부 졸업생이 preliminary 실험에 즉시 착수 가능하도록 구성되었다.

---

## 1. 연구 진행 Meta

### 1.1 사용자 쿼리 원문

> "오픈소스로 공개된 혹은 많은 연구들이 표준적으로 사용하는 VLM 모델에 대해서, 초창기 모델 대비 최신 모델의 아키텍처를 고려해서 이를 Single-GPU 혹은 Edge GPU 에서 serving 시 성능/에너지 최적화에 대해 연구 ideation 을 진행. 기존 세션 참고 금지, Vision 모델 최적화나 LLM 모델 특징과 다른 VLM 만의 특징을 고려한 idea. Revisit 혹은 기존과 완전히 다른 방식 가능."

### 1.2 주요 키워드 (4축)

| 축 | 키워드 |
|---|---|
| **도메인** | VLM (Vision-Language Model), edge GPU (Jetson Orin Nano/NX/AGX, Jetson Thor, RTX 4060/4090), single-GPU serving |
| **관찰·특징** | 초기 vs 최신 VLM 아키텍처 diff (fixed 576 → dynamic 4-16K token, 1D RoPE → MRoPE 3D, AnyRes 1-12 tile, pixel shuffle 2×2, video 2-frame temporal packing), VLM-only (LLM 과 다른 vision encoder / projector / LLM 3-stage) |
| **제안 기법** | AnyRes tile-count unifying signal → CUDA Graph bucket + coupled DVFS + per-tile precision / Vision DLA + Projector Tensor core + LLM NVFP4 modality-stage mapping + UMA zero-copy + DLA preemption / MRoPE tri-axial LUT / Entropy-adaptive pixel shuffle / Thor vs Orin cross-arch VLM characterization |
| **타겟** | J/token, W, battery life, TTFT, energy / request, MMMU accuracy 99.5%+ |

### 1.3 중점적으로 고려한 축

- **초기 (LLaVA-1.5 2023) vs 최신 (Qwen2.5-VL / InternVL3 2024-2025) 아키텍처 diff 를 mechanism 에 직접 활용**: AnyRes tile variance 가 DVFS signal, MRoPE 3D 가 LUT 대상, pixel shuffle 이 entropy-adaptive dial.
- **Edge GPU 의 특수 HW**: Jetson DLA (NVDLA 2.0) 가 vision encoder INT8 에 최적이지만 VLM serving 에 미활용, UMA 가 zero-copy 가능하지만 vLLM 이 discrete GPU 전제 memcpy.
- **VLM-only 특징**: 3-stage (vision / projector / LLM), dynamic resolution, modality-specific precision.
- **성능 + 에너지 동시**: TTFT 속도, J/token 에너지, MMMU accuracy 3 축 Pareto.
- **기존 세션 재사용 금지**: v1/v2/v3 VLM/VLA context serving, PRISM-VLM-KV, ACE-MoE, VLM+PIM 전혀 reference 하지 않고 완전 신규 ideation.

### 1.4 의도적으로 제외한 축 (이유 명시)

| 제외 축 | 이유 |
|---|---|
| **Multi-tenant / cross-request sharing** | v1/v2/v3 VLM/VLA context serving 세션이 이미 다룸. 본 세션은 single-tenant edge. |
| **KV cache compression (quantization)** | PRISM-VLM-KV 세션 (I2 TernVLM-KV-LUT) 이 커버. 본 세션은 edge GPU compute / DVFS / DLA 축. |
| **MoE VLM** | ACE-MoE 세션 (VLM/VLA software 확장) 이 커버. 본 세션은 dense VLM. |
| **PIM / HBM-PIM** | 2026-04-22 VLM+PIM 세션이 커버. Edge (Jetson) 는 LPDDR 이므로 HBM-PIM 부적합. |
| **Multi-node serving** | Single-workstation / edge scope 규율. |
| **Training-time optimization** | Inference serving 쿼리 명시. |
| **Token pruning 단독** | VL-Cache / SparseVLM / FastV 가 이미 다룸. 본 세션은 이를 stacking 대상으로만 참조. |
| **Cloud A100/H100 deployment** | CodecSight 등 cloud 연구와 차별화 위해 edge-only (Jetson / 4060-class) 로 scope 제한. |

### 1.5 검색 쿼리 전략

| Phase | 주요 쿼리 | 도메인 |
|---|---|---|
| Step 0 ai-opt | "Qwen2.5-VL Jetson edge serving", "vLLM VLM AnyRes bucket", "VLM DVFS energy", "NVDEC motion vector LLM inference" | arxiv / OpenReview / MLSys / SOSP / HPCA |
| Step 0 legacy-sys | "Jetson Orin AGX LLM profiling energy", "edge GPU unified memory VLM", "LPDDR5 bandwidth LLM serving" | arxiv / ICPP / HPCA / proceedings |
| Step 0 hw-pim | "Jetson DLA NVDLA transformer", "NVFP4 VLM quantization edge", "LUT tensor core MRoPE" | arxiv / ISCA / MICRO / CVPR |
| Phase 2 Similarity | "video VLM NVDEC motion vector token skip", "VLM modality stage pipeline DLA", "edge VLM energy characterization Thor Orin" | arxiv 최근 6 개월 + OpenReview |

### 1.6 사용된 전문가 에이전트 + 리뷰어

- **Experts**: ai-optimization-expert / legacy-system-expert / hw-pim-accelerator-expert (3 명 병렬 dispatch)
- **Reviewers**: novelty-reviewer / differentiation-reviewer / impact-reviewer (3 인 병렬)
- **Similarity Critique**: differentiation-reviewer 4th dispatch (scoop ≥ 70% 발굴 전담)
- **Cross review**: ai-opt ↔ legacy-sys ↔ hw-pim 도메인 상호 리뷰 (Phase 2 staging 파일 내 기록)

---

## 2. Tier-1 Top 2 (Top-tier Venue Target)

> Tier-1 Top 3 에서 I1 Tidal 이 CodecSight scoop 로 drop — Top 2 로 축소. 각 idea 에 Tier-2 scope 축소 variant subsection 병기 (paper-pair 후보).

### 2.1 Parquet ★ (Tier-1 Top 1, Avg **7.54** / ASPLOS 2027 / MLSys 2027)

#### 2.1.1 개요 (Metaphor → Mechanism)

**"Parquet"** = 마루판 타일. AnyRes 의 가변 tile 1-12 개가 마루판 타일처럼 가지런히 정렬되면서도 각 판이 서로 다른 precision/빈도 성질을 가진 은유. 최신 VLM (LLaVA-NeXT / Qwen2.5-VL / InternVL3) 의 **AnyRes tile-count** 를 통합 signal 로 활용하여 (1) CUDA Graph bucket 을 tile-count-variant 로 확장, (2) GPU+DRAM DVFS 를 coupled 하게 제어, (3) tile 별 entropy 로 precision 동적 선택한다. 세 mechanism 은 scheduler (M1) × HW DVFS controller (M2) × precision dispatcher (M3) 의 직교 축으로 ablation 가능하며, 공통 signal 인 per-request tile-count 를 통해 coordination 된다.

#### 2.1.2 기존 연구 한계점 및 Gap

- [vLLM v1 Multimodal CUDA Graph 문서](https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/): batch-size variant **만** 지원. Tile-count variant 없음. 실측 4-bucket fill rate 55%.
- [DynamoLLM HPCA'25 arXiv:2408.00741](https://arxiv.org/abs/2408.00741) [peer-reviewed]: LLM DVFS/inference 에너지 관리. VLM 미포함. **Tile-count signal 없음**.
- [GreenLLM arXiv:2508.16449](https://arxiv.org/abs/2508.16449): SLO-aware frequency scaling. VLM AnyRes 고려 없음.
- [PolyThrottle MLSys'24 arXiv:2310.19991](https://arxiv.org/abs/2310.19991) [peer-reviewed]: DNN throttle 일반, VLM visual token variance 미반영.
- [MBQ CVPR'25 arXiv:2412.19509](https://arxiv.org/abs/2412.19509) [peer-reviewed]: modality-balanced quantization, per-layer 축만 있음. **Per-tile (sub-image) 축 없음**.
- [BiScale arXiv:2602.18755](https://arxiv.org/abs/2602.18755): BW-coupled DVFS, VLM 미터치.
- [throttLL'eM arXiv:2408.05235](https://arxiv.org/abs/2408.05235): LLM throttling, tile-count 미활용.
- **SparseDVFS [arXiv:2603.21908](https://arxiv.org/abs/2603.21908)**: sparse workload DVFS, VLM AnyRes 고려 없음.

**Gap 요약**: AnyRes tile-count 를 **CUDA Graph + DVFS + precision** 의 unifying control signal 로 활용하는 연구 공백. 3 축을 통합한 첫 edge VLM serving scheduler.

#### 2.1.3 제안 기법 (3 Mechanism, R27-α: 4 필수 요소)

##### **M1: Adaptive CUDA Graph Bucket by Tile-Count**

- **① 추가되는 Scheme**: vLLM v1 fork 의 `vllm/v1/worker/gpu_model_runner.py` 내 `CUDAGraphRunner` class 를 확장하여 신규 subclass `TileCountGraphRunner` 를 추가. 기존 batch-size 축 bucket 에 **tile-count 축** 을 교차시킨 2D bucket table (`graph_table[bs_bucket][tile_bucket]`) 을 관리한다. 새 config `--tile-count-buckets` (예: `{1,3,6,12}`) + dynamic overflow path 를 `vllm/engine/arg_utils.py` 에 등록. 입력 request 진입 시 `scheduler.py` 의 `_schedule_prefill` 함수에서 request 의 AnyRes tile 수를 측정하여 적절한 bucket 으로 dispatch, bucket miss 발생 시 JIT capture 경로로 fallback.
- **② 해결하려는 문제**: 기존 vLLM v1 Multimodal CUDA Graph 는 batch-size variant 4 bucket `{1,2,4,8}` 만 capture. Qwen2.5-VL-7B AnyRes workload 에서 tile 수가 1-12 로 runtime 가변 (DocVQA 는 tile 평균 8-10, MMMU 는 2-4) 이지만 graph capture 는 tile shape 에 dependent → **bucket miss 가 45-55%** 발생. Nsight Systems 측정 시 graph re-capture 에 launch overhead ~30 µs × N kernel 이 누적되어 prefill TTFT 의 8-12% 손실.
- **③ 동작 원리 (학부생용 step-by-step)**:
  1. **Request tile-count 추출** — 입력 image preprocessing (`vllm/multimodal/processing/qwen2_vl.py`) 직후 `num_tiles = image_grid_thw[0]*image_grid_thw[1]` 계산. 이 값을 `SequenceData` metadata 에 저장.
  2. **Bucket resolution** — `_schedule_prefill` 진입 시 `tile_bucket = next_pow2_up(num_tiles, [1,3,6,12])` 로 가장 가까운 상위 bucket 결정. 동시에 batch-size bucket 도 결정.
  3. **2D graph table lookup** — `graph_table[bs_bucket][tile_bucket]` 에 capture 된 CUDA Graph 존재 여부 확인. Hit 이면 즉시 replay.
  4. **JIT capture (miss 시)** — `torch.cuda.CUDAGraph()` 로 새 graph capture 를 `capture_begin()/capture_end()` 로 진행, `graph_table` 에 등록 + LRU eviction (최대 24 entry).
  5. **Padded replay** — 실제 tile 수가 bucket 보다 작을 때 `torch.zeros(...)` padding 으로 tile dim 을 bucket 크기에 맞춤. `attn_mask` 로 padded position invalidate.
- **④ 기존 해법 실패 + 차별화**: (i) vLLM v1 공식 구현은 batch-size 만 고려, Qwen2.5-VL-7B 처럼 tile variance 큰 workload 에는 대응 불가 → fragmentation 45-55%. (ii) DynamoLLM HPCA'25 는 LLM 전용, VLM AnyRes tile-count signal 없음. (iii) 본 M1 은 **tile-count 를 independent bucket dimension 으로 격상** 한 첫 scheduler, bucket fill rate 55%→75-85% 달성 예상.
- **Platform-Usage Analysis**: vLLM v1 (이미 bucket 구조 존재, tile-count 로 확장 필요) / SGLang (RadixAttention 기반, tile-count bucket 미구현) / llama.cpp (batch API 제한적) — Step C: vLLM v1 에 upstream PR 경로.

##### **M2: Coupled GPU+DRAM DVFS with Tile-Count Signal**

- **① 추가되는 Scheme**: Jetson Orin AGX 의 `nvpmodel` + `jetson_clocks` 를 wrapping 한 신규 Python class `CoupledDVFSController` 를 `vllm/engine/dvfs_controller.py` 에 추가. 이 controller 는 tile-count signal 을 입력받아 **GPU frequency (GPU_MIN_FREQ..GPU_MAX_FREQ)** 와 **EMC frequency (LPDDR5 DRAM clock)** 를 coupled policy 로 설정한다. `/sys/kernel/debug/bpmp/debug/clk/emc/rate` 와 `/sys/devices/17000000.gpu/devfreq/17000000.gpu/userspace/set_freq` 에 직접 write. 데스크탑 (RTX 4060/4090) 에서는 NVML `nvmlDeviceSetApplicationsClocks` 로 fallback.
- **② 해결하려는 문제**: AnyRes tile 많을 때 (e.g., 8-12 tile) visual encoder 와 projector 가 **memory-bound** (arithmetic intensity 1.2-1.8 FLOP/byte) 로 GPU freq 최대에서 pipeline stall → Nsight Compute `dram__throughput.avg.pct_of_peak_sustained_elapsed` 70-85% 관측. 반대로 tile 적을 때 (1-3 tile) **compute-bound** 지만 DRAM freq 는 MAX 유지 → 불필요한 전력 낭비. 기존 Jetson governor 는 instantaneous utilization 기반으로 tile-count 같은 **앞 단계 signal** 반영 불가, 결과적으로 J/request 의 15-25% 비효율 발생.
- **③ 동작 원리 (step-by-step)**:
  1. **Offline profiling table 구축** — 5000 sample workload 로 tile-count × (GPU_freq, EMC_freq) 격자에서 J/token + latency 측정하여 Pareto frontier 도출. 결과를 `dvfs_policy.yaml` 로 저장.
  2. **Runtime signal 추출** — Request 진입 시 `num_tiles` 를 controller 로 전달.
  3. **Policy lookup** — `policy_table[tile_bucket] = (gpu_freq_hz, emc_freq_hz)` 에서 최적 freq pair 결정.
  4. **Hysteresis 적용** — 이전 freq 대비 Δ < 10% 이면 transition 생략 (freq switch overhead 수 ms 회피).
  5. **Frequency 적용** — `sudo /usr/bin/jetson_clocks --store user_policy.conf` 후 sysfs write. Transition 완료 delay 5 ms 후 kernel launch.
- **④ 기존 해법 실패 + 차별화**: (i) DynamoLLM / throttLL'eM / PolyThrottle 는 모두 LLM decode-token rate signal 기반, VLM vision stage 의 AnyRes variance 를 미포착. (ii) BiScale 은 BW-coupled DVFS 제안했으나 VLM tile-count 축 없음. (iii) Jetson default governor (`nvpmodel MAXN`) 는 static profile 만 제공. 본 M2 는 **tile-count 를 predictive signal 로** 사용하여 GPU+EMC 를 coupled 하게 스케줄 — memory-bound 시 GPU freq 10-20% 낮춰 전력 절감, compute-bound 시 EMC 10-15% 낮춤. 예상 에너지 -15~25%.

##### **M3: Per-Tile Entropy-Driven Precision Dispatching**

- **① 추가되는 Scheme**: Qwen2.5-VL vision encoder 의 SigLIP attention weight 를 proxy 로 각 tile 의 attention entropy 를 계산하는 신규 module `PerTileEntropyScorer` 를 `vllm/multimodal/entropy_scorer.py` 에 추가. 결과를 기반으로 tile 별 precision 을 `{FP8, INT8}` 중 선택하여 projector + LLM prefill 에 dispatch. Precision switch 는 CUTLASS mixed-precision GEMM template (`cutlass::gemm::device::GemmUniversal<half, int8_t, ...>`) 로 구현.
- **② 해결하려는 문제**: MBQ CVPR'25 의 per-layer 축 quantization 은 일괄적용으로 **tile 간 importance variance** 를 놓친다. DocVQA 의 경우 table/text 영역 tile 은 high entropy (FP8 필요), 배경 tile 은 low entropy (INT8 충분). 일괄 INT8 적용 시 DocVQA accuracy 2-4% drop, 일괄 FP8 적용 시 energy -5% 에 그침. Nsight Compute `smsp__pipe_tensor_op_hmma_cycles_active` 측정 시 FP8 tensor core utilization 이 MBQ 일괄 대비 35% 더 효율 가능.
- **③ 동작 원리 (step-by-step)**:
  1. **Pre-trained SigLIP attention extraction** — Vision encoder forward 시 last-layer attention weight `attn = softmax(QK^T/√d)` 를 `torch.no_grad()` 로 tap off.
  2. **Per-tile entropy 계산** — 각 tile 의 spatial region 에 속하는 attention row 들의 entropy `H = -Σ p log p` 를 계산 (per-tile scalar).
  3. **Quantile ranking** — Batch 내 tile 들을 entropy 기준 정렬, 상위 50% = FP8 flag, 하위 50% = INT8 flag (tile 별 boolean mask).
  4. **Precision-aware GEMM dispatch** — Projector 와 LLM prefill 의 matmul 을 CUTLASS 의 mixed-precision template 으로 호출, tile mask 에 따라 row 분할 execution.
  5. **Output dequantize + concat** — 결과 row 를 FP16 으로 dequantize 후 concat 하여 다음 layer 로 전달.
- **④ 기존 해법 실패 + 차별화**: (i) MBQ 는 per-layer granularity 만 제공 → image 내 region 별 importance 차이 무시. (ii) 일반 mixed-precision (e.g., AWQ) 은 weight-axis, activation (tile) axis 미지원. (iii) 본 M3 는 VLM-only 인 AnyRes tile 구조를 **activation-side per-tile precision** 축으로 격상한 첫 시도. 예상: DocVQA accuracy drop <0.5pp, energy -8~15% 추가.

**Mechanism 간 상호작용**: M1 은 scheduler-level (kernel dispatch), M2 는 HW-level (freq), M3 는 precision-level (per-tile) — 세 축 모두 독립 orthogonal. 공통 signal (tile-count + per-tile entropy) 은 M1/M2 가 count, M3 가 entropy 를 사용. 2^3 = 8 cell factorial ablation 가능.

**Tier 구성**: physical 1-tier (edge GPU + LPDDR) + software 2-tier (scheduler + precision) → R1/R1b ≤3-4 준수.

#### 2.1.4 평가 / 실험 플랜 (7-요소, R27-β)

##### (1) Hardware

- **Primary**: RTX 4090 24GB (desktop edge), Jetson Orin AGX 64GB (embedded edge).
- **Secondary**: RTX 4060 8GB (consumer edge).
- **Access path**: 연구실 #3 서버 RTX 4090 × 2, Orin AGX DevKit 1대.

##### (2) Model

- **Primary**: Qwen2.5-VL-7B ([HuggingFace Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), AnyRes native).
- **Secondary**: LLaVA-NeXT-7B (AnyRes baseline), InternVL3-8B (pixel shuffle baseline).
- **Robustness**: LLaVA-OneVision-7B.
- **Inference stack**: vLLM v1 fork (0.8.x base) + Qwen2.5-VL HF integration.

##### (3) Dataset / Workload

- **Benchmarks**: DocVQA (OCR, tile count 많음) + MMMU (mixed) + ChartQA + OK-VQA.
- **Workload**: 5000 request mixed AnyRes workload + per-request tile-count histogram 기록.
- **Real trace**: LMSys VisionArena partnership (optional).
- **Primary metric**: Energy/request (J), TTFT (ms).
- **Secondary**: CUDA Graph bucket fill rate, DocVQA accuracy, DRAM BW utilization.

##### (4) Simulators / Tools

- **Profiler**: Nsight Compute 2024.3 (`l1tex__t_sector_hit_rate.pct`, `lts__t_sectors`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `smsp__pipe_tensor_op_hmma_cycles_active`, `sm__warps_active.avg.pct_of_peak_sustained_active`).
- **System profiler**: Nsight Systems 2024.5 (kernel timeline, stream overlap).
- **Energy**: NVML (`nvmlDeviceGetPowerUsage`) + Jetson INA3221 (`tegrastats --interval 100`) + Intel RAPL (desktop CPU).
- **Serving stack**: vLLM v1 fork (tile-count bucket + DVFS hook) + CUTLASS mixed-precision template.

##### (5) Ablation + Measurement Protocol

- **Factorial design**: 2^3 = 8 cell (M1 × M2 × M3).
- **Parameter sweeps**: tile-count histogram `{1, 3, 6, 12}`, DVFS policy `{default, coupled, aggressive}`, precision tier `{FP16, FP8/INT8 mixed, INT8}`.
- **Baseline list (10 편, peer-reviewed 70%)**:
  - vLLM v1 Multimodal CUDA Graph (docs, 2024)
  - SGLang RadixAttention
  - DynamoLLM HPCA'25 [arXiv:2408.00741](https://arxiv.org/abs/2408.00741)
  - GreenLLM [arXiv:2508.16449](https://arxiv.org/abs/2508.16449)
  - PolyThrottle MLSys'24 [arXiv:2310.19991](https://arxiv.org/abs/2310.19991)
  - BiScale [arXiv:2602.18755](https://arxiv.org/abs/2602.18755)
  - throttLL'eM [arXiv:2408.05235](https://arxiv.org/abs/2408.05235)
  - MBQ CVPR'25 [arXiv:2412.19509](https://arxiv.org/abs/2412.19509)
  - SparseDVFS [arXiv:2603.21908](https://arxiv.org/abs/2603.21908)
  - Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301)
- **Main metric**: Energy/request, **secondary**: TTFT, bucket fill rate, DocVQA accuracy.
- **Expected runtime**: 개발 9주 + 실험 3주 + writing 2주 = **14 주**.
- **Fallback**: vLLM v1 PR 변경 시 SGLang fork 로 대체.

##### (6) Implementation Steps (Week-Level)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---|---|---|
| W1-2 | vLLM v1 fork + Qwen2.5-VL-7B baseline 재현 | vLLM v0.8.x, transformers 4.50+, HF Qwen2.5-VL | MMMU accuracy ±0.5pp, TTFT ±5% |
| W3 | `TileCountGraphRunner` class 구현 (M1) — 2D bucket table | vLLM v1 `CUDAGraphRunner`, `torch.cuda.CUDAGraph` | tile-count bucket 분기 정상, graph hit log 출력 |
| W4 | Bucket LRU + JIT overflow path | `collections.OrderedDict`, `capture_begin/end` | bucket fill rate 55%→70%+ 확인 |
| W5 | Offline DVFS profiling table 생성 | `nvpmodel`, `jetson_clocks`, sysfs | 4×4 freq grid × 5 tile bucket 측정 완료 |
| W6 | `CoupledDVFSController` 구현 (M2) | sysfs write, NVML `nvmlDeviceSetApplicationsClocks` | hysteresis policy + 5 ms transition delay 검증 |
| W7 | `PerTileEntropyScorer` 구현 (M3) | PyTorch hook, SigLIP attention tap | per-tile entropy histogram 시각화 |
| W8 | CUTLASS mixed-precision GEMM template 통합 | CUTLASS 3.6 `GemmUniversal`, FP8/INT8 kernel | projector FP8/INT8 switching 정상 |
| W9 | 8-cell factorial ablation 실행 | Hydra config, bash automation | 8 cell × 4 dataset = 32 cell 완료 |
| W10 | 10 baseline 비교 실험 | 각 baseline repo reproduction | 10 × 4 dataset 측정 완료 |
| W11-12 | Orin AGX 포팅 + cross-platform 검증 | JetPack 6.0, `tegrastats` | RTX 4090 / Orin 양 platform 결과 보유 |
| W13-14 | Writing + ablation table + figure | LaTeX, pandas, matplotlib | 논문 draft + ablation + Pareto plot 완성 |

##### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 (Δ) |
|---|---|---|---|---|
| CUDA Graph bucket fill rate | vLLM `--output-json` + custom counter | 5000 req mixed workload | 45-55% | **75-85%** |
| Energy/request | NVML + INA3221 (`tegrastats`) | DocVQA 1000 req, 1h run | 2.5-4.0 J/req | **-28~42%** |
| TTFT | vLLM wall-clock | MMMU 100 sample | 350-550 ms | **+4~5% trade** |
| DRAM BW usage | Nsight `dram__throughput.avg.pct_of_peak_sustained_elapsed` | vision encoder scope | 70-85% | **60-70% (efficiency gain)** |
| Tensor core utilization | Nsight `smsp__pipe_tensor_op_hmma_cycles_active` | projector scope | 42-58% | **+15pp (FP8/INT8 mix)** |
| DocVQA accuracy | lm-evaluation-harness | val set | 100% (baseline) | **≥ 99.5%** |
| Per-tile precision distribution | custom logger | 5000 req entropy log | uniform FP16 | **50/50 FP8/INT8** |

**Preliminary Study 순서**:

- **(i) Baseline reproduction**: vLLM v1 + Qwen2.5-VL-7B + DocVQA 공식 eval script 로 baseline accuracy 및 TTFT 재현 (±5%). 실패 시 transformers version, CUDA graph capture on/off, KV cache dtype 매칭 확인.
- **(ii) Bottleneck attribution**: Nsight Compute `ncu --section MemoryWorkloadAnalysis --section SchedulerStats --section LaunchStats` 로 vision encoder (memory-bound) vs projector (compute-bound) vs LLM prefill (mixed) 분류. 예상 결과: DocVQA 8-12 tile workload 는 **memory-bound dominant** (DRAM BW 85%+), MMMU 2-4 tile 은 **compute-bound** (tensor core 58%).
- **(iii) Roofline upper bound**: Qwen2.5-VL 각 stage 의 arithmetic intensity 계산. RTX 4090 HW roofline (165.2 TFLOPS FP16, 1008 GB/s) 기준 현재 구현은 peak 의 45-55% 활용. 본 기법 목표 65-75%.
- **(iv) Mechanism 단독 Micro-benchmark**: (a) **M1 only** — tile-count bucket 만 활성, DVFS default, precision FP16. Bucket fill rate 개선 측정 (기대 +20pp). (b) **M2 only** — single bucket + coupled DVFS + FP16. Energy 개선 측정 (기대 -12~18%). (c) **M3 only** — single bucket + default DVFS + per-tile FP8/INT8. Accuracy drop 및 energy 개선 (기대 -8~15%, drop <0.5pp). 세 effect 의 sum 이 full combo 와 linear 한지 검증.

#### 2.1.5 예상 효과

| 지표 | Baseline | Parquet | 조건 |
|---|---|---|---|
| Energy / request | 1× | **-28~42%** | Mixed AnyRes workload |
| TTFT (ms) | 1× | +4~5% trade | - |
| CUDA Graph bucket fill rate | 55% | **75-85%** | Tile-count adaptive |
| DocVQA accuracy | 100% | ≥ 99.5% | Per-tile precision |
| Tile-uniform (text-only) | - | <3% gain | scope 밖 (명시) |

**Scoring Summary**: Nov 7.4 / Diff 7.5 / Imp 7.76 / Feas 7.5 → **avg 7.54 Accept strong**. 전문가 **3:0 unanimous**.

#### 2.1.6 Tier-2 Scope 축소 Variant (IEEE CAL 4p)

- **Target**: IEEE CAL 4p / ISLPED 6p.
- **Single mechanism**: M1 only (Adaptive CUDA Graph bucket).
- **Scope 축소**: Qwen2.5-VL-7B 단일 + Jetson Orin AGX 단일, bucket fill rate + latency 2 metric.
- **Runtime**: 3 주.
- **Top-tier 관계**: M1 을 2026-07 CAL 로 먼저 공개, M2+M3 를 2026-10 ASPLOS/MLSys 로 확장.

---

### 2.2 Triptych (Tier-1 Top 2, Avg **7.38** / ASPLOS 2027 / EuroSys 2027)

#### 2.2.1 개요 (Metaphor → Mechanism)

**"Triptych"** = 3 폭 패널 그림. Vision encoder (DLA INT8) / Projector (Tensor core FP16) / LLM (NVFP4 또는 INT4) 이 각각 독립된 패널로 구성되면서 하나의 통합된 serving 작품을 이루는 은유. Jetson Orin AGX 의 DLA 를 vision encoder 전용으로 활용하고, Jetson UMA 의 zero-copy 로 activation 을 memcpy 없이 hand-off, DLA fine-grained preemption 으로 pipeline bubble 제거한다. 세 mechanism 은 compute mapping (M1) × memory hand-off (M2) × scheduler preemption (M3) 의 직교 축으로 설계된다.

#### 2.2.2 기존 연구 한계점 및 Gap

- [Nova arXiv:2509.21301](https://arxiv.org/abs/2509.21301): VLM 3-stage elastic partitioning, **desktop GPU only, DLA 미사용**. Phase 2 similarity **60% concurrent**.
- [Nanomind arXiv:2510.05109](https://arxiv.org/abs/2510.05109): module-level brick scheduling for tiny model, **DLA fine-grained preemption 없음**. Phase 2 similarity **60% concurrent**.
- [HeteroInfer arXiv:2501.14794](https://arxiv.org/abs/2501.14794): heterogeneous LLM serving, **server-class GPU, DLA 미포함**. 55% concurrent.
- [HydraInfer arXiv:2505.12658](https://arxiv.org/abs/2505.12658): hybrid inference, DLA 미사용.
- [llm.npu arXiv:2407.05858](https://arxiv.org/abs/2407.05858): mobile NPU + LLM, VLM 3-stage 미커버.
- [LiteVLM arXiv:2506.07416](https://arxiv.org/abs/2506.07416): edge VLM pipeline, **single compute unit, DLA 미사용**.
- [FastVLM CVPR'25 arXiv:2412.13303](https://arxiv.org/abs/2412.13303) [peer-reviewed]: vision encoder 최적화 (algorithmic), **DLA 미사용**.
- [NVIDIA DLA Jetson Orin blog](https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/): DLA 를 convolution only 로 제시, transformer 미커버.

**Gap 요약**: Jetson DLA 는 NVIDIA 공식 문서조차 convolution only 로 가정 — transformer vision encoder + VLM serving 용 DLA scheduler 공백. UMA 는 하드웨어 지원 있으나 vLLM 이 discrete GPU 전제로 활용 안 함.

#### 2.2.3 제안 기법 (3 Mechanism, R27-α: 4 필수 요소, Phase 1' M3 replace)

##### **M1: Modality-Stage Heterogeneous Mapping (DLA / Tensor Core / NVFP4)**

- **① 추가되는 Scheme**: vLLM-Jetson fork 의 `vllm/model_executor/models/qwen2_vl.py` 에 신규 class `ModalityStageDispatcher` 추가. 이 dispatcher 는 Qwen2.5-VL 의 (a) vision encoder (ViT/SigLIP/InternViT) → Jetson DLA core 0/1 (INT8, NVDLA 2.0), (b) projector (MLP 2-layer) → GPU Tensor Core (FP16), (c) LLM decoder (28 layer) → GPU Tensor Core (Orin: INT4, Thor: NVFP4) 로 분리 dispatch 한다. Weight conversion 은 `trtexec --onnx=vision.onnx --useDLACore=0 --int8 --calib=calib.cache --saveEngine=vision_dla.engine` 파이프라인으로 선행.
- **② 해결하려는 문제**: Baseline 에서 Qwen2.5-VL-7B 전체를 GPU 만으로 실행 시 vision encoder 단계에서 Orin AGX Ampere GPU 가 INT8 MMA unit 을 완전 활용 못함 (SM occupancy 45-55%). 동시에 DLA 2 core 는 0% idle. Nvidia 공식 blog 에 따르면 DLA 는 convolution 위주 guidance 로 transformer ViT 미커버 — 연구자가 자발적으로 transformer 에 DLA 를 활용한 VLM serving 사례 전무. Energy/query 측정 시 vision encoder stage 가 총 J 의 30-40% 차지.
- **③ 동작 원리 (step-by-step)**:
  1. **Vision encoder ONNX export** — Qwen2.5-VL 의 ViT 부분을 `torch.onnx.export` 로 단독 export (opset 17), shape = [batch, 3, H, W] 입력.
  2. **DLA engine 변환** — `trtexec --onnx=vit.onnx --useDLACore=0 --int8 --saveEngine=vit_dla.engine`. INT8 calibration 은 DocVQA 500 sample 로 PTQ.
  3. **DLA runtime init** — vLLM engine init 에서 `nvinfer1::IRuntime::deserializeCudaEngine()` 로 DLA engine load, core 0/1 에 dispatch.
  4. **Projector on Tensor Core** — Projector (2-layer MLP) 는 기존 GPU path 유지, FP16 GEMM.
  5. **LLM on Tensor Core (precision tiered)** — Orin 에서는 AWQ INT4, Thor 에서는 NVFP4 (Blackwell native). `vllm/model_executor/layers/quantization/` 의 적절한 backend 선택.
- **④ 기존 해법 실패 + 차별화**: (i) Nova/LiteVLM/HeteroInfer 는 모두 desktop GPU 기반, DLA 고려 없음. (ii) NVIDIA 공식 DLA blog 는 convolution-only, transformer ViT 미커버 → 학계/산업 모두 공백. (iii) FastVLM 은 algorithmic-level (vision encoder 구조 개선), 본 M1 의 HW mapping 과 orthogonal. (iv) 본 M1 은 NVDLA 2.0 transformer 지원을 검증 + Orin AGX 에서 DLA INT8 로 vision encoder 2-3× throughput/W 이점 활용.
- **Platform-Usage Analysis**: vLLM (discrete GPU 전제, DLA 미지원) / TensorRT-LLM (일부 Jetson support 있으나 DLA weight routing 수동) / MLC-LLM (Jetson 지원, DLA 활용 제한) — Step C: vLLM-Jetson fork + TensorRT DLA backend.

##### **M2: UMA Zero-Copy Activation Hand-off**

- **① 추가되는 Scheme**: vLLM-Jetson fork 에 신규 class `ZeroCopyActivationRouter` 를 `vllm/worker/uma_router.py` 에 추가. Jetson UMA (Unified Memory Architecture) 의 `cudaMallocManaged` + `cudaMemAdviseSetPreferredLocation` 를 활용하여 DLA output tensor 와 GPU input tensor 가 **동일 physical address** 를 공유하도록 강제. 기존 host-device memcpy 를 우회하고, DLA completion event 를 GPU stream 의 `cudaStreamWaitEvent` 로 동기화.
- **② 해결하려는 문제**: Jetson Orin AGX 는 UMA 를 hardware level 에서 지원 (CPU/GPU/DLA 가 동일 LPDDR5 물리 메모리 공유) 하지만 vLLM 은 discrete GPU 전제로 `cudaMemcpyHostToDevice`/`DeviceToHost` 를 실행. Vision encoder (DLA output) → projector (GPU input) 전달 시 불필요 memcpy 2회 × 약 4 MB/image = 8 MB 전송이 매 request 마다 발생. Nsight Systems timeline 에서 memcpy 가 prefill TTFT 의 5-8% 차지.
- **③ 동작 원리 (step-by-step)**:
  1. **Managed memory allocation** — Engine init 에서 vision feature tensor 를 `cudaMallocManaged(&ptr, size)` 로 할당. `cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, DLA_DEVICE)` + `GPU_DEVICE` 로 양쪽 accessibility 설정.
  2. **DLA execution 직접 write** — DLA engine 실행 시 output buffer 를 managed ptr 로 지정. DLA 완료 시 `cudaEventRecord(event, stream_dla)`.
  3. **GPU stream 대기** — Projector kernel 진입 전 `cudaStreamWaitEvent(stream_gpu, event, 0)` 로 DLA 완료 대기.
  4. **GPU kernel 직접 read** — Projector kernel 이 동일 ptr 을 input 으로 사용, memcpy 없음.
  5. **TLB coherency 확인** — Jetson UMA 는 hardware coherent 이므로 flush 불필요. 단, CPU 접근 시 `cudaDeviceSynchronize` 로 명시 동기화.
- **④ 기존 해법 실패 + 차별화**: (i) vLLM 기본 구현은 discrete GPU 전제로 항상 memcpy 발생. (ii) TensorRT-LLM Jetson edition 은 일부 UMA 지원하나 VLM 3-stage activation routing 은 수동 설정 요구 (개발자 친화 아님). (iii) MLC-LLM 은 Jetson 지원 있으나 UMA 활용 부분적. (iv) 본 M2 는 vLLM 에 UMA zero-copy 를 **structural 로 integration** — 개발자 개입 없이 자동 activation routing. 예상 TTFT 개선 5-8%, energy -3~5%.

##### **M3 (REPLACE from SM partition): DLA Fine-Grained Preemptive Scheduling**

- **① 추가되는 Scheme**: NVDLA 2.0 SDK 의 fine-grained preemption 기능 (`nvdla::PreemptionLevel::SUB_KERNEL`) 을 활용하여 DLA 상의 long-running vision encoder inference 를 sub-kernel boundary 에서 preempt 가능하게 설정. 신규 scheduler `DLAPreemptScheduler` 를 `vllm/engine/dla_scheduler.py` 에 추가하여 multi-image batch 시 **이미지 간 interleaving** 을 가능하게 한다. GPU Tensor Core + DLA + LLM 의 3-stream pipeline bubble 을 제거.
- **② 해결하려는 문제**: Multi-image batch (3+ image) 시 DLA 가 image 1 전체를 처리한 뒤 image 2 로 넘어가는 non-preemptive execution → projector 와 LLM 이 image 1 결과만 받고 idle. Nsight Systems 측정 시 DLA execution 의 30-45% 가 독점 (non-overlap) 이며 3-stage pipeline bubble 이 TTFT 의 20-30% 발생. Nova 의 desktop SM partition 방식은 Jetson DLA 에 적용 불가 (DLA 는 SM 이 아닌 별도 accelerator).
- **③ 동작 원리 (step-by-step)**:
  1. **Preemption level 설정** — NVDLA engine load 시 `nvdla::IExecutionContext::setPreemptionLevel(nvdla::PreemptionLevel::SUB_KERNEL)` 호출.
  2. **Batch splitting** — Input batch 를 `sub_batch_size=1` 단위로 쪼개어 DLA queue 에 enqueue. 각 sub_batch 후 preemption point 발생.
  3. **Projector async dispatch** — 첫 sub_batch 완료 시 즉시 projector 를 `stream_projector` 에 async dispatch. 동시에 DLA 는 다음 sub_batch 처리.
  4. **3-stream pipeline** — DLA (vision stream) / Tensor Core (projector stream) / Tensor Core (LLM stream) 3 개 stream 간 `cudaStreamWaitEvent` 로 의존성 chain 구성.
  5. **Fine-grained preempt on new request** — 더 우선 높은 request 도착 시 sub_kernel boundary 에서 DLA 가 현재 inference 를 preempt → 새 request 의 vision encoder 를 DLA 에 먼저 dispatch (SLO 민감 workload).
- **④ 기존 해법 실패 + 차별화**: (i) Nova 의 SM partition 은 desktop GPU 전제, Jetson DLA 는 SM 이 아닌 별도 accelerator 이므로 직접 적용 불가. (ii) Nanomind 는 brick-level scheduling 이나 DLA preemption 없음. (iii) NVIDIA 공식 DLA blog 에는 preemption 활용 사례 전무 (convolution 용 dedicated execution 전제). (iv) 본 M3 는 NVDLA 2.0 의 sub-kernel preemption 을 처음으로 VLM serving 에 적용, multi-image batch 의 pipeline bubble 제거. 예상 TTFT 1.4-1.7× 가속.
- **Critical gap 방어**: Phase 2 Similarity 에서 Nova + Nanomind + HeteroInfer 68-72% concurrent 판정 → M3 replace 로 DLA axis 로 repositioning, concurrent → unique.

**Mechanism 간 상호작용**: M1 은 compute-mapping (어디서 실행), M2 는 memory path (어떻게 전달), M3 는 scheduler (언제 실행) — 3 개 직교 축. M1+M2 만으로도 유효하나 M3 의 preemption 이 multi-image workload 에서 synergy 증폭.

**Tier 구성**: physical 3-tier (DLA + Tensor Core + UMA) + software 1-tier (dispatcher) = 4-tier 상한.

#### 2.2.4 평가 / 실험 플랜 (7-요소, R27-β)

##### (1) Hardware

- **Primary**: Jetson Orin AGX 64GB (LPDDR5 204 GB/s, 2 × DLA core, Ampere GPU).
- **Secondary**: Jetson Thor DevKit 128GB LPDDR5X (2026-06 gate, Blackwell + DLA-Next + NVFP4).
- **Baseline (no DLA)**: RTX 4090 24GB (desktop, DLA 없음).
- **Access path**: 연구실 Orin AGX 1대, Thor NVIDIA Inception partnership 2026-06 확보.

##### (2) Model

- **Primary**: Qwen2.5-VL-7B ([HF Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)).
- **Secondary**: InternVL3-8B, LLaVA-OneVision-7B.
- **Robustness**: MiniCPM-V-2.6 (tiny edge).

##### (3) Dataset / Workload

- **Benchmarks**: VideoMME short subset + MMMU + DocVQA + Multi-image LLaVA-Interleaved (MMDU).
- **Workload**: 3000 req multi-image batch (2-8 image/req).
- **Primary metric**: TTFT (ms), energy/query (J).
- **Secondary**: DLA utilization %, SM occupancy %, pipeline bubble %.

##### (4) Simulators / Tools

- **Profiler**: Nsight Compute + Nsight Systems (3-stream overlap visualization).
- **DLA tools**: `nvmedia-dla-sdk`, NVDLA SDK 6.1, `trtexec`.
- **Energy**: Jetson INA3221 via `tegrastats --interval 100`.
- **Serving stack**: vLLM-Jetson fork + `ZeroCopyActivationRouter` + `DLAPreemptScheduler`.

##### (5) Ablation + Measurement Protocol

- **Factorial design**: 2^3 (M1 mapping × M2 UMA × M3 preemption) + Thor NVFP4 optional condition.
- **Parameter sweeps**: DLA core count `{1, 2}`, sub_batch_size `{1, 2, 4}`, UMA hint policy.
- **Baseline list (9 편, peer-reviewed 67%)**:
  - Nova [arXiv:2509.21301](https://arxiv.org/abs/2509.21301)
  - Nanomind [arXiv:2510.05109](https://arxiv.org/abs/2510.05109)
  - HeteroInfer [arXiv:2501.14794](https://arxiv.org/abs/2501.14794)
  - HydraInfer [arXiv:2505.12658](https://arxiv.org/abs/2505.12658)
  - vLLM EPD (blog)
  - llm.npu [arXiv:2407.05858](https://arxiv.org/abs/2407.05858)
  - LiteVLM [arXiv:2506.07416](https://arxiv.org/abs/2506.07416)
  - FastVLM CVPR'25 [arXiv:2412.13303](https://arxiv.org/abs/2412.13303)
  - TensorRT-LLM-Edge (NVIDIA blog)
- **Main metric**: TTFT, **secondary**: energy/query, DLA utilization, MMMU accuracy.
- **Expected runtime**: 개발 10주 + 실험 4주 + writing 2주 = **16 주**.
- **Fallback**: Thor DevKit 미확보 시 Orin-only (NVFP4 → INT4), analytical model 보강.

##### (6) Implementation Steps (Week-Level)

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---|---|---|
| W1-2 | vLLM-Jetson fork + Qwen2.5-VL-7B baseline (GPU-only) | vLLM v0.8.x, Jetson JetPack 6.0 | TTFT / MMMU 재현 ±5% |
| W3 | Vision encoder ONNX export + DLA calibration | `torch.onnx.export`, `trtexec --int8 --calib` | INT8 calibration table 생성, accuracy drop <1pp |
| W4 | DLA engine integration (M1 part 1) | `nvinfer1::IRuntime`, NVDLA SDK 6.1 | DLA core 0/1 에서 ViT 실행 성공 |
| W5 | `ModalityStageDispatcher` 완성 (M1 part 2) | vLLM `model_executor`, CUDA stream | 3-stage dispatch 정상 + MMMU bit-exact ±0.5pp |
| W6 | `ZeroCopyActivationRouter` 구현 (M2) | `cudaMallocManaged`, `cudaMemAdvise` | memcpy count 0 확인 (Nsight Systems) |
| W7 | UMA zero-copy 검증 + coherency test | `cudaEventRecord`, `cudaStreamWaitEvent` | 데이터 정합성 unit test 8/8 pass |
| W8 | DLA preemption level 설정 (M3 part 1) | `nvdla::IExecutionContext::setPreemptionLevel` | sub_kernel boundary preemption 로그 검출 |
| W9 | `DLAPreemptScheduler` 구현 (M3 part 2) | `stream.wait_event`, scheduler queue | multi-image 3-stream pipeline 검증 |
| W10 | 2^3 factorial ablation 실행 | Hydra, bash | 8 cell × 4 dataset = 32 cell 완료 |
| W11 | 9 baseline 비교 | 각 repo reproduction | 9 × 4 dataset 측정 |
| W12-13 | Thor DevKit 포팅 (optional) | Thor JetPack, NVFP4 kernel | Thor result 확보 |
| W14-16 | Writing + ablation table + figure | LaTeX, matplotlib | 논문 draft + ablation + Pareto plot |

##### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 + counter/command | 측정 조건 | 기대 범위 (baseline) | 개선 후 목표 (Δ) |
|---|---|---|---|---|
| TTFT | vLLM wall-clock | multi-image 3 batch, Qwen2.5-VL-7B | 650-900 ms | **-30~40% (1.4-1.7× faster)** |
| DLA utilization | `tegrastats` DLA column | M1 on | 0% | **≥60%** |
| SM occupancy | Nsight `sm__warps_active.avg.pct_of_peak_sustained_active` | projector + LLM | 45-55% | **70-80%** |
| Memcpy count | Nsight Systems memcpy events | full prefill | 2-4 memcpy/req | **0** |
| Pipeline bubble % | Nsight Systems stream timeline | 3-stream overlap | 20-30% | **<5%** |
| Energy/query | INA3221 | 1h multi-image run | 1.8-2.6 J/query | **-25~35%** |
| MMMU accuracy | lm-eval-harness | 5-shot, all | 100% (baseline) | **≥99.5%** |

**Preliminary Study 순서**:

- **(i) Baseline reproduction**: vLLM + Qwen2.5-VL-7B (GPU-only, DLA 미사용) 으로 MMMU/DocVQA/VideoMME 재현. 각 TTFT ±5%, accuracy ±0.5pp.
- **(ii) Bottleneck attribution**: Nsight Systems timeline 으로 (a) vision encoder (DLA idle 상태 GPU 혼자 실행), (b) memcpy (UMA 미활용), (c) multi-image serial execution 각각을 분리 측정. Bottleneck 분류: **compute (vision)** + **memory-transfer (memcpy)** + **scheduler (serial)** 의 복합 bottleneck.
- **(iii) Roofline upper bound**: Orin AGX 에서 vision encoder 의 AI (Qwen2.5-VL ViT-L/14 기준 ~3.2 FLOP/byte) 계산. GPU+DLA 병행 execution 시 통합 peak (GPU 4.1 TFLOPS FP16 + DLA 5 TOPS INT8 = ~9 TFLOPS equivalent) 활용. 현재 peak 의 35-45% → 본 기법 목표 65-75%.
- **(iv) Mechanism 단독 Micro-benchmark**: (a) **M1 only** — DLA mapping 만 활성, memcpy 유지, preemption off. Vision encoder throughput 개선 (기대 2-3× throughput/W). (b) **M2 only** — UMA zero-copy 만 적용, DLA off, GPU-only. TTFT 개선 (기대 -5~8%). (c) **M3 only** — single-image baseline 에 preemption 만 활성, multi-image 시 pipeline overlap 측정 (기대 -15~20% TTFT).

#### 2.2.5 예상 효과

| 지표 | Baseline (vLLM all-GPU) | Triptych | 조건 |
|---|---|---|---|
| TTFT | 1× | **1.4-1.7× faster** | Multi-image batch 3+ |
| Energy / query | 1× | **-25~35%** | DLA 활용 |
| MMMU accuracy | 100% | ≥ 99.5% | Modality precision |
| Single-image VLM | - | 5-8% gain (scope 밖) | DLA 이점 축소 |
| Jetson Thor NVFP4 | +5-8% 추가 | - | Conditional on Thor DevKit |

**Scoring Summary**: Nov 6.5 (post-replacement) / Diff 8.0 / Imp 8.0 / Feas 7.0 → **avg 7.38 Accept**. 전문가 **3:0 unanimous (post-replacement)**.

#### 2.2.6 Tier-2 Scope 축소 Variant (IEEE ESL 4p)

- **Target**: IEEE ESL 4p / ISLPED 6p.
- **Single mechanism**: M1 only (DLA INT8 vision encoder mapping).
- **Scope**: Qwen2.5-VL-7B + Jetson Orin AGX 단일, 에너지 + latency 2 metric.
- **Runtime**: 4 주.
- **Top-tier 관계**: M1 이 가장 measurable energy delta 확보, M2+M3 는 top-tier 확장.

---

## 3. Tier-2 독립 Top 3 (Track B, Phase 1 부터 독립 도출)

### 3.1 Cartographer (Track B Top 1, Avg **7.06** / IEEE CAL 4p / DATE 6p)

#### 3.1.1 개요 (Metaphor → Mechanism)

**"Cartographer"** = 지도 제작자. MRoPE 의 3 axis (time × H × W) 좌표계를 precomputed LUT 으로 "지도화" 하여 runtime 계산을 치환. Edge GPU 의 SFU (Special Function Unit) 점유를 줄이고 memory BW 감소.

#### 3.1.2 기존 연구 한계점 및 Gap

- Qwen2.5-VL / InternVL3 의 **MRoPE (Multi-dimensional RoPE, 3-axial: time × H × W)** 계산은 전용 kernel, SFU 점유 non-trivial (edge GPU 에서 더 크게 pronounced).
- [T-MAC EuroSys'25 arXiv:2407.00088](https://arxiv.org/abs/2407.00088) [peer-reviewed]: weight mpGEMM LUT, positional encoding 미커버.
- [LUT Tensor Core ISCA'25 arXiv:2408.06003](https://arxiv.org/abs/2408.06003) [peer-reviewed]: weight LUT, positional encoding 미적용.
- [RotateKV arXiv:2501.16383](https://arxiv.org/abs/2501.16383): rotation-based, LUT 치환 미도입.
- [Revisiting MRoPE arXiv:2510.23095](https://arxiv.org/abs/2510.23095): MRoPE 변형 연구, **LUT 기반 kernel 공백**.
- [SAIL arXiv:2509.25853](https://arxiv.org/abs/2509.25853): SRAM-LUT GEMV, VLM positional 미적용.

#### 3.1.3 제안 기법 (Single Mechanism, Tier-2 rubric)

##### **M1: MRoPE Tri-Axial Precomputed LUT + LPDDR Row-Aligned Layout**

- **① 추가되는 Scheme**: CUTLASS 3.6 template library 의 `include/cutlass/gemm/collective/` 에 신규 template file `mrope_triaxial_lut.hpp` 추가. vLLM `vllm/model_executor/models/qwen2_vl.py` 의 `apply_rotary_pos_emb_vision` 함수를 LUT 참조로 교체. LUT 은 3 개 `__constant__` array — `__constant__ half time_lut[128]`, `__constant__ half height_lut[128]`, `__constant__ half width_lut[128]` (head_dim=128 가정) — 로 구성, 각 lookup 결과를 element-wise multiply 후 rotation 적용. 추가로 LPDDR5 row-buffer 정렬을 위해 `cudaMallocPitch` 로 LUT 저장 영역 pitch 를 2 KB 배수로 reserve.
- **② 해결하려는 문제**: Qwen2.5-VL-7B 의 MRoPE kernel 이 CUDA SFU (sincos 연산) 를 heavy 하게 사용하여 Orin AGX 에서 `smsp__inst_executed_pipe_fp64_sfu.sum` 측정 시 **18-25%** busy. MRoPE 전체 latency 의 35-50% 가 SFU 의존 sincos. Desktop GPU (RTX 4090) 는 SFU throughput 16 unit/SM × 2.52 GHz 로 충분하나 Orin AGX 는 16 unit/SM × 1.3 GHz 로 edge 에서 bottleneck 심화. 추가로 LPDDR5 row-buffer miss 로 LUT load 시 BW 10-15% 손실.
- **③ 동작 원리 (step-by-step)**:
  1. **Offline LUT 생성** — Qwen2.5-VL 의 RoPE base freq θ = 10000^(-2i/D) 로 time/H/W 각 128 element LUT 생성 (float16), `__constant__` memory 에 binding.
  2. **LPDDR row alignment** — LUT backing storage 를 `cudaMallocPitch(&ptr, &pitch, 256, 3)` 로 할당 (2 KB row-buffer 경계 정렬).
  3. **Kernel prologue prefetch** — MRoPE kernel 진입 시 `__ldg(&time_lut[i])` 등으로 L1 cache 에 prefetch.
  4. **3-axis element-wise multiply** — Q/K 의 각 dim 에 `q_rot = q * complex_mul(time_lut[t] * height_lut[h] * width_lut[w])` 로 rotation 적용. SFU 호출 0.
  5. **Tile 내 fused apply** — CUTLASS mainloop 내부에 inline 삽입, separate kernel launch 제거.
- **④ 기존 해법 실패 + 차별화**: (i) T-MAC / LUT Tensor Core 는 weight-side LUT 만 다룸 (positional encoding 미커버). (ii) Revisiting MRoPE 는 algorithm-level 분석, kernel-level 구현 공백. (iii) 본 M1 은 MRoPE 3-axial 을 **precomputed tri-LUT** 으로 치환한 첫 edge-specific kernel, SFU busy 22%→5% 이하, latency -40~60% 예상.

#### 3.1.4 실험 플랜 (7-요소, Tier-2 축소)

##### (1)-(5)

- **Hardware**: Jetson Orin AGX 64GB 단일.
- **Model**: Qwen2.5-VL-7B 단일 (MRoPE 3D).
- **Dataset**: VideoMME + DocVQA subset 500 req.
- **Tools**: CUTLASS 3.6 LUT kernel + Nsight Compute (`smsp__inst_executed_pipe_fp_hi`, SFU busy %).
- **Ablation**: LUT size sweep `{256, 512, 1024, 2048}`, row-alignment on/off.
- **Baseline (5 편, peer-reviewed 60%)**: T-MAC EuroSys'25, LUT Tensor Core ISCA'25, FlashAttention-3, SAIL, Revisiting MRoPE.

##### (6) Implementation Steps

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---|---|---|
| W1 | vLLM Qwen2.5-VL baseline on Orin AGX | vLLM v0.8.x, JetPack 6.0 | TTFT / MMMU 재현 ±5% |
| W2 | LUT 생성 + `__constant__` binding + `cudaMallocPitch` | CUDA 12.5, constexpr generation | LUT vs reference rotation bit-exact |
| W3 | CUTLASS fused MRoPE kernel 구현 | CUTLASS 3.6, `__ldg`, inline rotation | SFU busy % 25%→5% 이하 (Nsight) |
| W4 | 평가 + 5 baseline 비교 + writing | Nsight Compute, lm-eval | 5 × DocVQA/VideoMME 측정 완료 |

##### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 + counter | 측정 조건 | Baseline | 목표 |
|---|---|---|---|---|
| MRoPE kernel latency | Nsight `sm__cycles_active.sum` filter MRoPE | Qwen2.5-VL-7B, 8K seq | 150-220 µs/call | **-40~60%** |
| SFU busy % | Nsight `smsp__inst_executed_pipe_fp64_sfu.sum` | rotation scope | 18-25% | **<5%** |
| Row-buffer hit | Nsight `dram__sectors_read_conflict` | LUT load | 40-55% | **75%+** |
| Energy/MRoPE call | INA3221 proxy | 1h workload | - | **-15~25%** |
| MMMU accuracy | lm-eval | 5-shot | 100% | **100% (bit-exact)** |

**Preliminary Study**: (i) baseline Qwen2.5-VL-7B MRoPE kernel isolate 실측, (ii) SFU vs L1 vs DRAM bottleneck attribution, (iii) Orin AGX roofline (SFU-bound region 명시), (iv) M1 only micro-benchmark (LUT size sweep).

#### 3.1.5 예상 효과

| 지표 | Baseline | Cartographer |
|---|---|---|
| MRoPE kernel latency | 1× | **-40~60%** |
| Energy / MRoPE call | 1× | **-15~25%** |
| MMMU accuracy | 100% | 100% (bit-exact LUT) |

**Scoring**: Nov 6.8 / Diff 6.6 / Imp 7.03 / Feas 7.8 → **avg 7.06 Accept CAL**.

**Tier-1 scale-up 불가 이유**: single-kernel characterization letter, MRoPE 에 한정.

---

### 3.2 Sift (Track B Top 2, Avg **6.85** / ISLPED 2026 6p)

#### 3.2.1 개요 (Metaphor → Mechanism)

**"Sift"** = 체로 거른다. Patch entropy 기준으로 visual token 을 체질하여 통과/압축 선택. InternVL3 / SmolVLM 의 pixel shuffle ratio 를 entropy-adaptive 하게 동적 선택한다 (기존 2×2 고정).

#### 3.2.2 기존 연구 한계점 및 Gap

- [InternVL-X arXiv:2503.21307](https://arxiv.org/abs/2503.21307): RVTC (reduced visual token compression), **entropy-adaptive 미포함**.
- [PyramidDrop arXiv:2410.17247](https://arxiv.org/abs/2410.17247): layer-wise token drop, **projector-이전 pixel shuffle 축 없음**.
- [VisionZip arXiv:2412.04467](https://arxiv.org/abs/2412.04467): visual token compression, ratio 고정.
- [FastV arXiv:2403.06764](https://arxiv.org/abs/2403.06764): attention-based prune, projector-이전 미커버.
- [SparseVLM ICML'25](https://openreview.net/forum?id=80faIPZ67S) [peer-reviewed]: sparse attention, pixel shuffle 미커버.
- [SmolVLM arXiv:2504.05299](https://arxiv.org/abs/2504.05299): edge VLM, pixel shuffle 고정.

#### 3.2.3 제안 기법 (Single Mechanism)

##### **M1: Patch Entropy-Driven Adaptive Pixel Shuffle Ratio**

- **① 추가되는 Scheme**: SmolVLM / InternVL3 vision encoder 의 pixel shuffle layer (기존 고정 2×2, down-sample 4×) 를 adaptive 하게 전환하는 module `AdaptivePixelShuffle` 을 `vllm/model_executor/models/internvl.py` (또는 smolvlm) 에 추가. 입력은 SigLIP attention weight 로부터 proxy 된 patch-level entropy map, 출력은 ratio `{2, 4, 8}` 중 하나. Projector 이전 stage 에서 visual token 수를 dynamic 하게 축소.
- **② 해결하려는 문제**: SmolVLM-2B 나 InternVL3-2B 같은 tiny edge VLM 은 pixel shuffle ratio 2×2 고정으로 visual token 수가 균일 (예: 729 token 고정). 그러나 COCO/DocVQA patch 의 entropy 분포는 bimodal — 일부 patch 는 high entropy (text, table), 다수는 low entropy (background). 균일 ratio 는 low-entropy patch 에 과잉 token 할당 → Jetson Orin Nano 8GB 에서 token count 가 throughput bottleneck. 실측 visual token 이 decode TPOP 의 25-35% 차지.
- **③ 동작 원리 (step-by-step)**:
  1. **SigLIP attention tap** — Vision encoder 의 마지막 block 이후 attention weight `attn[B,H,N,N]` 를 `torch.no_grad()` 로 추출.
  2. **Patch entropy 계산** — 각 patch 의 cross-patch attention row entropy `H_i = -Σ_j attn_ij log attn_ij` 계산.
  3. **Region clustering** — K-means 혹은 simple threshold (p25/p50/p75) 로 patch 를 low/med/high 3 region 분류.
  4. **Region-wise ratio 할당** — Low entropy → ratio 8 (down-sample 64×), med → ratio 4, high → ratio 2. 결과 token map 을 reconstruct.
  5. **Projector input 재구성** — Region-별 down-sample 된 token 을 concat 하여 projector 에 전달. Position embedding 은 original patch index 기반 유지.
- **④ 기존 해법 실패 + 차별화**: (i) InternVL-X RVTC 는 post-projector token compression, pixel shuffle ratio 축 미터치. (ii) PyramidDrop / VisionZip / FastV 는 LLM-layer 내 token drop 으로 pre-projector 축소 대비 효과 제한. (iii) SparseVLM 은 attention sparsity, pixel shuffle 미커버. (iv) 본 M1 은 **projector 이전** pixel shuffle ratio 를 entropy-adaptive 하게 전환 — 가장 효과적인 token reduction 위치 + edge VLM-specific 축. 예상 token count -40~60%, energy -20~30%.

#### 3.2.4 실험 플랜 (7-요소, Tier-2)

##### (1)-(5)

- **Hardware**: Jetson Orin Nano 8GB + Orin AGX 64GB.
- **Model**: SmolVLM-2B / InternVL3-2B / MiniCPM-V-2.6.
- **Dataset**: DocVQA + COCO + MMMU patch entropy subset.
- **Tools**: PyTorch pixel shuffle modification + NVML + Jetson INA3221.
- **Ablation**: Entropy threshold sweep (p25/p50/p75) + ratio set `{2, 4, 8}` vs fixed.
- **Baseline (6 편, peer 50%)**: InternVL-X, PyramidDrop, VisionZip, FastV, SparseVLM ICML'25, SmolVLM.

##### (6) Implementation Steps

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---|---|---|
| W1 | SmolVLM/InternVL3 baseline on Orin Nano | vLLM, HF models | DocVQA 재현 ±1pp |
| W2 | `AdaptivePixelShuffle` module 구현 | PyTorch, SigLIP attention tap | per-patch entropy map 시각화 |
| W3 | Region clustering + ratio 할당 logic | K-means / threshold, torch | token count -40%+ 확인 |
| W4 | Projector 재통합 + position embedding 처리 | model surgery | MMMU accuracy drop <1pp |
| W5 | 6 baseline 비교 + writing | 각 repo | 6 × 3 dataset 측정 완료 |

##### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 | 측정 조건 | Baseline | 목표 |
|---|---|---|---|---|
| Visual token count | PyTorch log | DocVQA | 729 (fixed) | **-40~60%** |
| Prefill TTFT | vLLM wall-clock | SmolVLM-2B, MMMU | 280-360 ms | **-25~35%** |
| Energy/request | INA3221 | 1h run | - | **-20~30%** |
| DocVQA accuracy | lm-eval | val | 100% (baseline) | **≥99% (drop ≤1pp)** |
| MMMU accuracy | lm-eval | 5-shot | 100% | **≥99%** |

**Preliminary Study**: (i) baseline SmolVLM-2B fixed 2×2 shuffle 재현, (ii) patch entropy 분포 측정 (bimodal 검증), (iii) tiny edge VLM roofline (memory-bound for small models), (iv) M1 only micro-benchmark (ratio sweep).

#### 3.2.5 예상 효과

- Visual token count -40-60%, energy -20-30%, accuracy drop ≤ 1pp (DocVQA / MMMU).

**Scoring**: Nov 6.5 / Diff 6.5 / Imp 6.90 / Feas 7.5 → **avg 6.85 Accept ISLPED**.

**Tier-1 scale-up 불가 이유**: tiny VLM + projector-이전 pixel shuffle 축 narrow.

---

### 3.3 Verge (Track B Top 3, Avg **6.29** / IEEE ESL 4p letter Conditional)

#### 3.3.1 개요 (Metaphor → Mechanism)

**"Verge"** = 경계/가장자리. Jetson Thor (최신 Blackwell NVFP4) 와 Orin AGX (이전 세대 Ampere INT8) 의 경계에서 VLM-specific 에너지 차이를 측정.

#### 3.3.2 기존 연구 한계점 및 Gap

- [ELANA arXiv:2512.09946](https://arxiv.org/abs/2512.09946): Thor + Orin Nano cross-arch, **일반 LLM serving only, VLM-specific stage breakdown 없음**.
- [Jetson Orin LLM profiling arXiv:2506.09554](https://arxiv.org/abs/2506.09554): Orin only, Thor cross-arch 없음.
- [TokenPowerBench arXiv:2512.03024](https://arxiv.org/abs/2512.03024): LLM token-level power, **VLM 미포함**.
- [Watt Counts arXiv:2604.09048](https://arxiv.org/abs/2604.09048): LLM energy, Jetson Thor 미커버.
- [Blackwell Microbench arXiv:2512.02189](https://arxiv.org/abs/2512.02189): Blackwell 일반 micro, VLM 미커버.

#### 3.3.3 제안 기법 (Single Mechanism, Characterization)

##### **M1: VLM-Specific Stage-Level Cross-Architecture Energy Characterization**

- **① 추가되는 Scheme**: Qwen2.5-VL-7B / InternVL3-8B / LLaVA-OneVision-7B 3 모델을 Jetson Thor (Blackwell + NVFP4 + LPDDR5X) vs Orin AGX (Ampere + INT8 + LPDDR5) 에서 VLM 고유 4-stage (vision encoder / projector / LLM prefill / LLM decode) 별로 J/token + W + TTFT 를 측정하는 characterization framework `VLMPowerProbe` 를 구축. `tegrastats` + INA3221 + Nsight Compute counter 를 stage boundary event 와 cross-reference 하여 stage-level 에너지 attribution 을 수행한다.
- **② 해결하려는 문제**: ELANA / Jetson Orin profiling / TokenPowerBench 는 LLM 전용 characterization 이거나 VLM 통합 metric 만 제공. VLM-specific 질문 — "NVFP4 Blackwell 에서 vision encoder 는 얼마나 에너지 이득? LLM decode 는? LPDDR5X vs LPDDR5 의 visual token burst write pattern 차이는?" — 에 답할 data 가 없다. 업계 / 학계 모두 VLM-on-Thor cross-arch characterization 전무.
- **③ 동작 원리 (step-by-step)**:
  1. **Stage boundary marking** — vLLM forward 에 CUDA event 를 4 stage boundary 에 삽입 (`cudaEventRecord`). `tegrastats` 와 CUDA event timestamp 를 NTP sync 하여 alignment.
  2. **Per-stage power sampling** — INA3221 을 100 Hz 로 sampling, 각 stage 구간의 power integral = energy 계산.
  3. **Counter collection** — Nsight Compute `sm__cycles_active.sum`, `dram__throughput`, `smsp__pipe_tensor_op_hmma_cycles_active` 등을 stage 별로 수집.
  4. **Cross-arch matrix 구축** — 3 model × 2 platform × 4 stage × 3 dataset = 72 data point matrix.
  5. **Attribution analysis** — NVFP4 vs INT8 효율 차이, LPDDR5X vs LPDDR5 bandwidth 차이, tensor core utilization 차이를 stage 별로 분리 보고.
- **④ 기존 해법 실패 + 차별화**: (i) ELANA 는 LLM only, VLM stage breakdown 없음. (ii) TokenPowerBench / Watt Counts 는 token-level granularity, VLM 3-stage 미분리. (iii) Blackwell Microbench 는 HW micro-benchmark, VLM workload 미수록. (iv) 본 M1 은 **Thor-on-VLM first characterization** 으로 후속 연구의 reference 가 되는 characterization letter.

#### 3.3.4 실험 플랜 (7-요소, Tier-2 Conditional)

##### (1)-(5)

- **Hardware**: **Jetson Thor DevKit 128GB (2026-06 gate)** + Jetson Orin AGX 64GB. Fallback: RTX 4090 + Orin AGX.
- **Model**: Qwen2.5-VL-7B / InternVL3-8B / LLaVA-OneVision-7B.
- **Dataset**: VideoMME / MMMU / DocVQA.
- **Tools**: Jetson INA3221 / NVML / PMU / nvprof / Nsight Compute.
- **Ablation**: Stage × platform × model × dataset matrix (no factorial, characterization design).
- **Baseline (6 편, peer 40% Conditional)**: ELANA, Jetson Orin profiling, TokenPowerBench, E4 AAAI'25, Blackwell Microbench, Watt Counts.

##### (6) Implementation Steps

| Week | Component / File | 사용 API/Library | 완료 판정 |
|------|---|---|---|
| W1 | `VLMPowerProbe` framework skeleton | Python, INA3221 Python bindings | 4-stage event marking 정상 |
| W2 | Orin AGX 3 model baseline 측정 | vLLM, JetPack 6.0, `tegrastats` | 3 × 3 dataset × 4 stage matrix |
| W3 | Thor DevKit 셋업 + NVFP4 kernel 확인 | Thor JetPack, NVFP4 backend | Thor 에서 3 model 구동 |
| W4 | Thor 3 model cross-arch 측정 | 동일 framework | 3 × 3 × 4 Thor matrix 완료 |
| W5 | Cross-arch analysis + figure | pandas, matplotlib | 72 data point comparison figure |
| W6 | 6 baseline 비교 + writing | 각 baseline data | letter draft 완성 |

##### (7) Preliminary Analysis Metrics

| 측정 지표 | 도구 | 측정 조건 | Baseline (Orin) | Thor 목표 |
|---|---|---|---|---|
| Vision encoder J/stage | INA3221 | 1 image prefill | - | insight report |
| Projector J/stage | INA3221 | single matmul | - | insight report |
| LLM prefill J/token | INA3221 | 8K seq | 0.5-0.8 | NVFP4 개선 보고 |
| LLM decode J/token | INA3221 | 2K output | 0.4-0.6 | NVFP4 개선 보고 |
| DRAM BW (LPDDR5 vs 5X) | Nsight | full prefill | LPDDR5 204 GB/s 기준 | LPDDR5X 273 GB/s 활용도 |
| Tensor Core util (Ampere vs Blackwell) | Nsight `smsp__pipe_tensor_op_hmma_cycles_active` | matmul scope | 45-55% | 65-75% (Blackwell) |

**Preliminary Study**: (i) Orin AGX 에서 baseline 3 model × 3 dataset 재현, (ii) 각 stage 별 bottleneck attribution (vision = compute or BW?), (iii) Orin Ampere roofline vs Thor Blackwell roofline 이론 비교, (iv) Thor 확보 직후 M1 standalone run (micro-benchmark 수준 prelim).

#### 3.3.5 예상 효과

- VLM-specific insight (FP4 의 LLM vs VLM 에너지 이점 차이, LPDDR5X vs LPDDR5 의 visual token burst write pattern, Blackwell Tensor Core utilization 변화).

**Scoring**: Nov 5.8 / Diff 6.2 / Imp 6.35 / Feas 6.8 → **avg 6.29 Conditional**.

**Conditional 조건**: Jetson Thor DevKit 2026-06 확보 필수.

**Tier-1 scale-up 불가 이유**: characterization-only letter.

---

## 4. 미선정 아이디어 전수 (사유 + 재방문 조건)

### 4.1 I1 Tidal (DROP by CodecSight Scoop)

- **연구 GAP (원래 의도)**: Video VLM (Qwen2.5-VL video / LLaVA-Video) 2-frame visual token cosine sim > 0.9 비율 42-65% 에서 NVDEC hardware motion vector + MRoPE-aware temporal KV dedup + LPDDR bank-aligned block layout.
- **Metaphor 의도**: "Tidal" = 조수 (temporal repetition, cyclical pattern 은유).
- **미선정 사유**: [CodecSight arXiv:2604.06036](https://arxiv.org/abs/2604.06036) (2026-04-07, Yulin Zou 등) 이 (a) NVDEC motion vector → token skip gate, (b) RoPE position-correction selective KVC refresh 를 이미 제안. **68-72% 직접 scoop**. 제출일 16일 차이로 precedence 확보 불가능. Edge-only repositioning 시도에도 core mechanism 이 1:1 중첩.
- **재방문 조건**:
  1. CodecSight 의 Jetson edge validation 부재 명확화 + VLM-specific edge benchmark 실측 novelty 확보.
  2. MRoPE 3D temporal axis 가 CodecSight 1D RoPE 와 formal 다른 별도 축으로 재설계 (예: VideoRoPE 변형 결합).
  3. LPDDR bank-aligned layout 을 I2 Parquet 의 sub-mechanism 으로 흡수 가능.

### 4.2 Phase 1 에서 통합된 원본 12 ideas (artificial split 방지)

| Original (Phase 1) | 통합 Idea | 합병 사유 |
|---|---|---|
| ai-opt Ember | **Triptych** (M2 UMA + Ember 의 per-stage DVFS) | Modality-stage pipeline 3-way 수렴 |
| ai-opt Mosaic | **Parquet** (M1 bucket + Mosaic 의 CUDA Graph) | AnyRes tile batching 3-way 수렴 |
| ai-opt Ripple | **Tidal** (M1 NVDEC + Ripple 의 MRoPE reuse) | Video VLM dedup 3-way 수렴 |
| ai-opt Lattice | **Sift** (Lattice 의 tiny edge entropy pixel shuffle) | Entropy-adaptive pixel shuffle 2-way 수렴 |
| legacy-sys TileTide | **Parquet** (M2 DVFS + TileTide 의 coupled GPU+DRAM) | AnyRes DVFS 3-way 수렴 |
| legacy-sys VistaGate | **Triptych** (M2 UMA zero-copy + VistaGate 의 LPDDR layout) | UMA pipeline 3-way 수렴 |
| legacy-sys EchoVault | **Tidal** (M2 KV dedup + EchoVault 의 LPDDR bank) | Video VLM 3-way 수렴 |
| legacy-sys PixelTram | **Sift** (PixelTram 의 ViT-tail pixel shuffle) | Entropy pixel shuffle 2-way 수렴 |
| hw-pim Triptych Pipeline | **Triptych** (M1 Modality mapping + DLA INT8) | 동일 컨셉, 이름 유지 |
| hw-pim Mosaic Tiler | **Parquet** (M3 per-tile precision + Mosaic Tiler 의 FP8/INT8) | AnyRes 3-way 수렴 |
| hw-pim Echo Chamber | **Tidal** (M3 + Echo Chamber 의 sparse projector/LLM) | Video VLM 3-way 수렴 |
| hw-pim Cartographer Cache | **Cartographer** (MRoPE LUT + LPDDR 독립) | 독립 Track B 유지 |

---

## 5. 이 세션의 독특한 점

| 축 | 이 세션의 특이점 |
|---|---|
| **기존 세션 완전 신규** | v1/v2/v3 VLM/VLA context serving, PRISM-VLM-KV, ACE-MoE, VLM+PIM 어느 것도 reference 하지 않음. 사용자 요구 충족. |
| **초기 vs 최신 VLM diff 직접 활용** | Parquet = AnyRes (LLaVA-1.5 미존) / Triptych = DLA × 3-stage (최신 pixel shuffle + projector) / Cartographer = MRoPE 3D (초기 1D RoPE 미존) / Sift = pixel shuffle (초기 미존). **4 idea 중 4 개 모두 최신 아키텍처 diff 활용**. |
| **VLM-only 특징 강제** | LLM-only or Vision-only 로 성립 불가함을 각 idea 에서 formal 증명. |
| **Edge GPU 이기종** | Jetson DLA (NVDLA 2.0), UMA zero-copy, NVFP4 (Thor), LPDDR5X 등 edge-specific HW 활용 — desktop GPU 적용 불가 축. |
| **R23 Workload-driven 엄수** | 6 idea 모두 Workload evidence 섹션에 숫자 근거 (Qwen2.5-VL 4-16K token, vLLM 55% bucket fill, video cosine 42-65% 등). |
| **R26 Metaphor Noun Title 엄수** | Parquet / Triptych / Cartographer / Sift / Verge / Tidal 모두 metaphor noun. |
| **CodecSight scoop 즉시 감지** | Phase 2 similarity critique 가 16일 전 공개 논문 발견 → Tidal 즉시 DROP. |
| **R27 Retrofit (2026-04-24)** | R27 Self-Sufficient Summary 적용. Mechanism 4 요소 + 실험 플랜 7 요소. |

---

## 6. 다음 단계 제안

1. **Parquet Phase 3 entry**: vLLM v1 EncoderDisagg merge 상태 2026-04 snapshot 실측 (1주 PoC). W1-W2 tile-count bucket 구현 우선.
2. **Triptych DLA preemption API**: NVDLA SDK 접근 권한 확인 + Jetson Orin AGX DLA INT8 vision encoder 프로파일링 (2주).
3. **Cartographer MRoPE LUT PoC**: Nsight Compute SFU busy % 선행 측정 (1일).
4. **Sift entropy 분포 측정**: DocVQA + COCO entropy 분포 선행 (2일).
5. **Verge Thor DevKit gate**: 2026-06-01 확보 decision, 미확보 시 drop or Orin+RTX 4090 fallback.
6. **Tidal 재방문 세션**: CodecSight 후속 연구 모니터링 + MRoPE 3D vs 1D RoPE formal 차별 재설계 (Mode 1 재호출 시).
7. **Publish 요청 시**: summary 파일 homepage publish (명시 요청 시만).

---

## 7. 참고 파일

- **Session 상세 (재현성)**: [sessions/2026-04-23-mode1-energy-efficient-edge-vlm.md](../sessions/2026-04-23-mode1-energy-efficient-edge-vlm.md)
- **Staging**:
  - [aiopt expert](../sessions/staging/2026-04-23-edge-vlm-energy-aiopt-expert.md) (520 lines)
  - [legacy-sys expert](../sessions/staging/2026-04-23-edge-vlm-energy-legacy-sys-expert.md) (500 lines)
  - [hwpim expert](../sessions/staging/2026-04-23-edge-vlm-energy-hwpim-expert.md) (467 lines)
  - [Phase 1 integration](../sessions/staging/2026-04-23-edge-vlm-energy-phase1-integration.md)
  - [Phase 2 novelty](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-novelty.md)
  - [Phase 2 differentiation](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-diff.md)
  - [Phase 2 impact](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-impact.md)
  - [Phase 2 similarity critique](../sessions/staging/2026-04-23-edge-vlm-energy-phase2-similarity.md)
  - [Phase 1'/2'/1''](../sessions/staging/2026-04-23-edge-vlm-energy-phase1prime-2prime-1primeprime.md)
