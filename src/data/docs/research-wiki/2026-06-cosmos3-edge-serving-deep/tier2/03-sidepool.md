# S4 SIDEPOOL (Tier-2 #3 독립, 직전 avg 6.2 → placement-LUT 중심 재작성)

**SIDEPOOL: Placement and Overlap of Frozen Multimodal Tokenizers across the Edge SoC Accelerator Complex for Omnimodal Generation**
· DATE / ISPASS / DAC 2027 (placement & overlap characterization study)

> 본 문서는 직전(빈약) 버전(`2026-06-04-mode2/.../tier2/03-sidepool.md`)을 **Tier-1 동급 detail(R28-α)** 로 전면 재작성한 것이다. 직전 세션 rescope 결정(R-S5) + Phase-2' scoop 알람 + Step-0 refresh 를 모두 계승한다. **핵심 기여 = placement LUT (이종 가속기 복합체 위 modality 별 tokenizer 배치의 체계적 특성화 + 정책)**, overlap 은 placement 의 한 활용처(PipeDiT/DeDiVAE 가 overlap 발상을 선점했으므로 보조로 강등).

---

## 0. 약어 Glossary

| 약어 | 풀이 | 본 문서 맥락 |
|---|---|---|
| **VAE** | Variational Auto-Encoder | latent↔pixel 변환 tokenizer. Cosmos3 는 Wan2.2 video VAE(4×16×16) + audio VAE(AVAE). **frozen** (학습 후 고정). |
| **tokenizer** | (multimodal) tokenizer | 본 문서에서 "frozen tokenizer" = {video VAE encode/decode, audio VAE encode/decode, (조건부) ViT vision encoder}. denoise transformer 의 **밖**에 있는 modality↔latent 변환기. |
| **DLA** | Deep Learning Accelerator (NVDLA) | Jetson SoC 의 고정기능 conv/pool/activation 가속기. AGX Orin = **2× NVDLA v2.0**. **2D-only**, FP16/INT8. |
| **cuDLA** | CUDA DLA runtime API | CUDA stream 에서 DLA submission 을 제출하는 런타임(`cudaStream` ↔ DLA task 통합). hybrid mode 로 GPU↔DLA 비동기 협업. |
| **PVA** | Programmable Vision Accelerator | Jetson 의 VLIW 비전 가속기(AGX Orin = PVA v2.0). filter/conv/SfM 등 고정 비전 op. audio/소형 op 후보. |
| **TensorRT DLA delegate** | `trtexec --useDLACore` / TRT builder flag | ONNX/TRT engine 을 DLA core 에 매핑. 미지원 layer 는 `--allowGPUFallback` 로 GPU 로 자동 회귀. |
| **GroupNorm** | Group Normalization | VAE 의 정규화 layer 후보. (Cosmos3 Wan2.2 VAE 는 실제로 `RMS_norm` 사용 — 아래 §3 정정.) **DLA 미지원**. |
| **3D conv** | 3D convolution (`nn.Conv3d`) | video VAE 의 시공간 conv. Wan2.2 VAE 의 `CausalConv3d` 가 대표. **DLA 미지원**(2D-only). |
| **CausalConv3d** | causal 3D conv (temporal padding 단방향) | Wan2.2 VAE 의 지배 연산. `nn.Conv3d` 상속 + temporal cache. DLA 비호환 핵심. |
| **chunked encoding** | 시간축 청크 단위 VAE 인코딩 | 256p 68f / 480p 24f / 720p 12f 단위 청크 (temporal cache 유지하며 VAE 통과). 메모리 cap 목적. |
| **placement LUT** | placement Look-Up Table | {tokenizer}×{가속기}×{해상도}×{dtype} → (latency, energy, accuracy, GPU-fallback%) 측정 grid + 최적 배치 룩업표. **본 idea 의 핵심 산출물.** |
| **DVFS** | Dynamic Voltage & Frequency Scaling | (인접 idea S2 DUOCLOCK 영역. 본 문서는 placement 가 자원-축, DVFS 는 freq-축으로 직교.) |
| **UMA** | Unified Memory Architecture | Jetson 의 iGPU/CPU/DLA 가 LPDDR5 물리 메모리 공유. placement 가 copy-free 일 수 있으나 EMC BW 경합. |
| **ISPASS** | IEEE Int'l Symp. on Performance Analysis of Systems and Software | 측정/특성화 letter 적합 venue (DATE/DAC 와 함께 본 idea 타깃). |
| **EMC** | External Memory Controller (clock) | Jetson LPDDR5 메모리 컨트롤러. UMA 공유 BW 의 병목. placement 간 간섭의 측정축. |
| **SDPA** | Scaled Dot-Product Attention | VAE 중간 `AttentionBlock` 의 attention(F.scaled_dot_product_attention). DLA 미지원(GPU 유지). |

---

## 1. 개요

### 1.1 메타포 ↔ 메커니즘
- **SIDEPOOL** = SIDE(본류=denoise transformer 옆의 측면) + POOL(SoC 의 이종 가속기 복합체 = iGPU + DLA + PVA + CPU 가 이루는 보조 자원 풀).
- **그림**: denoise 가 흐르는 본류(main stream) 옆에 측면 풀(side pool)을 판다. frozen VAE/audio 인코더를 이 측면 풀(DLA/PVA/CPU/유휴 GPU slack)로 빼돌리면 (a) 본류의 iGPU 점유를 해방하고 (b) 연속 generation 시 본류(다음 요청 denoise)와 측면 풀(현 요청 VAE)이 겹친다.
- **그러나 핵심 질문은 "겹침"이 아니라 "어디에 둘 것인가"이다.** 측면 풀에는 여러 가속기가 있고, modality(video/audio/vision)×해상도×dtype 조합마다 최적 자원이 다르다. SIDEPOOL 은 그 **배치 지도(placement LUT)** 를 체계적으로 측정·구축하고, 그 위에 비동기 dispatch 정책을 얹는다.

### 1.2 Research Question (RQ)
> **frozen tokenizer 들은 단일 edge SoC 의 어느 가속기 자원에 두는 것이 latency/energy 최적인가 — modality × 해상도 × dtype × 자원 조합의 체계적 placement 지도를 어떻게 구축하고, 그 지도를 서빙 런타임의 비동기 dispatch 정책으로 어떻게 활용하는가?**

부속 질문:
- (RQ-a) Wan2.2 video VAE 의 `CausalConv3d` 지배 연산은 DLA 2D-only 제약상 GPU fallback 이 불가피한데, **chunked 2D-decomposable subgraph 로 변환해 DLA 후보를 만들 수 있는가**, 그 때 fallback 비율은 얼마인가?
- (RQ-b) audio VAE(저부하, ~25 tok/s 급)·ViT encoder 같은 소형/2D 친화 tokenizer 는 DLA/PVA/CPU 중 어디가 유리한가?
- (RQ-c) placement 가 결정되면, `_decode_latents` 동기 실행을 별도 stream/cuDLA submission 으로 비동기화해 다음 요청 denoise 와 겹칠 때 실효 이득은?

### 1.3 PipeDiT / DeDiVAE 와의 관계 (정직한 차별화)
- [PipeDiT (arXiv:2511.12056)](https://arxiv.org/abs/2511.12056) 와 [DeDiVAE (arXiv:2512.07350)](https://arxiv.org/abs/2512.07350) 는 **"VAE decode ∥ denoise pipeline overlap"** 핵심 발상을 이미 점유한다 (Phase-2' scoop 알람, worst-overlap concurrent 45-55%). **단, 둘 다 multi-GPU group 분리** (VAE 를 별도 GPU rank 에 떼어 그룹 간 pipeline) 다.
- **따라서 본 idea 는 overlap 을 핵심 기여로 주장하지 않는다.** 본 idea 의 핵심은 **단일 SoC 내 이종 가속기 복합체(iGPU/DLA/PVA/CPU)에 modality 별 tokenizer 를 배치하는 체계적 특성화 + placement LUT** 이고, overlap(M2)은 그 LUT 가 "denoise 와 다른 자원" 을 지정했을 때 자연히 따라오는 **한 활용처(corollary)** 다.
- multi-GPU 전제(PipeDiT/DeDiVAE)는 edge SoC 의 단일-die·UMA·이종-고정기능-가속기 현실에 적용 불가 — **placement 라는 문제 자체가 multi-GPU 에는 없다**(거기선 "어느 GPU" 가 동질적). 이것이 본 idea 의 빈 영역(Step-0 refresh: 단일-device placement 신규 경쟁작 0건, ACM TACO 2025 JDIMO 도 diffusion VAE 특화 아님).

### 1.4 직전 버전 대비 변경 (R28-α 재작성)
- 직전(6-2): M1 GPU stream-overlap 을 사실상 동격으로 둠 → **본 재작성: M1 = placement LUT(standalone 핵심), M2 = LUT-driven async dispatch(overlap, 보조)** 로 위계 재정립.
- frozen tokenizer 범위를 video VAE 뿐 아니라 **audio VAE·ViT encoder 까지** 명시적으로 grid 에 포함 (제목의 "multimodal tokenizers" 정합).
- DLA 버전 정정: AGX Orin = **NVDLA v2.0** (직전 문서/spec 의 "3.1" 은 Thor-class. web 확인). DLA-incompat op 목록을 실제 소스(`wan2pt2_vae_4x16x16.py`)에서 재확인하여 GroupNorm→`RMS_norm` 정정.

---

## 2. Mechanism Summary Table (R70)

| | **M1 — Tokenizer Placement Characterization (LUT)** | **M2 — LUT-Driven Async Dispatch (overlap, 보조)** |
|---|---|---|
| **한 줄** | frozen tokenizer × 가속기 × 해상도 × dtype 의 latency/energy/accuracy/fallback% grid 측정 → placement LUT | LUT 가 지정한 비-denoise 자원으로 `_decode_latents`/encode 를 별도 stream/cuDLA submission 비동기 dispatch, 다음 요청 denoise 와 overlap |
| **무엇을 바꾸나** | (측정 + 정책 산출물) — 코드 변경은 ONNX export + TRT/cuDLA 변환 파이프라인 | [`pipeline_cosmos3.py#L1003 self.vae.decode`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1003) 동기 호출 → async dispatch wrapper |
| **앵커** | [`wan2pt2_vae_4x16x16.py`](https://github.com/nvidia/cosmos-framework/blob/main/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py) (op 목록), [`pipeline_cosmos3.py#L987 _decode_latents`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L987) / [`#L1131 _encode_video_tensor`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1131) | [`pipeline_cosmos3.py#L1867 video=self._decode_latents(latents)`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1867) (diffuse 종료 후 동기 실행) |
| **차별 핵심** | 이종-가속기 placement 지도 = 미존재 (PipeDiT/DeDiVAE 는 multi-GPU 동질) | 단일-SoC 이종 가속기 + LUT 정책 (PipeDiT: GPU 내 SM 분할 / 우리: DLA·CPU·GPU-stream 혼합) |
| **이득 성격** | 인용 가능 측정 letter (특성화 자체로 ISPASS 성립) | e2e latency↓ 또는 GPU 점유 해방 → throughput↑ (연속 generation 한정) |
| **risk** | DLA fallback 과다(3D conv) → DLA 후보 축소 | 연속 generation 아니면 overlap 기회 없음 / RTX(DLA 무) 는 GPU-only |
| **falsification gate** | DLA-supported subgraph 비율 <20% → DLA drop, placement = {GPU,CPU,PVA} 한정 | overlap% 가 PipeDiT-식 GPU-내 overlap 대비 우위 없으면 M2 drop, M1 단독 letter |

- **의존성**: M2 ⟸ M1 (placement LUT 가 dispatch 대상 자원을 결정). M1 단독으로도 letter 성립(특성화). M2 는 M1 의 한 활용처.

---

## 3. GAP + Workload Evidence + Baseline Source (R52.1)

### 3.1 GAP — G8 (VAE/tokenizer 병목)
- frozen tokenizer(VAE/audio encoder)가 denoise transformer 와 **같은 iGPU 에서 직렬 실행** → SM·EMC BW 경합. denoise 종료 후 `_decode_latents` 가 **동기로 1회** 실행되는 구조 (`pipeline_cosmos3.py#L1867`).
- Edge/Nano 급에서 **transformer 가 작아 VAE 비용이 amortize 되지 않음** → VAE 가 비-amortized 지배 비용 (tech report §5.2.6 — 학습 측 실측이나 추론에도 시사). cloud 급(큰 transformer + 큰 batch)에서는 VAE 가 무시 가능하나 edge single-stream 에서는 지배적.
- frozen tokenizer 는 가중치 갱신이 없어 **별도 가속기로 떼어내기에 이상적**(컴파일·고정-engine 가능)인데, **이종 가속기 placement 가 미활용**.

### 3.2 Workload Evidence (수치 + clickable)
- **해상도 정의** (tech report / inference_benchmarks.md L39 확인): 256p = 320×192, 480p = 832×480, 720p = 1280×720. chunked encoding: 256p 68f / 480p 24f / 720p 12f.
- **t2v 720p latency** (vLLM-Omni, inference_benchmarks.md §37): RTX PRO 6000 **369.67s**(720p/1), H100 NVL 311.13s. 256p/1 Diffusers ~9-11s. → 고해상도일수록 denoise 가 길어 VAE 비중 상대 감소, 저해상도(256p)일수록 VAE 상대 비중↑ (edge 표적과 정합).
- **audio VAE 저부하**: audio token ~25 tok/s 급 (tech report §2.4 modality 비대칭: video 수만 token ≫ audio 25/s ≫ action ~32). → audio tokenizer 는 GPU 에 둘 필요 없는 소형 후보(CPU/PVA).
- **Orin SM 유휴**: "SM issue-slot 25-40%" ([arXiv:2508.08430](https://arxiv.org/abs/2508.08430)) → denoise 가 iGPU 점유해도 issue-slot 여유 有, VAE 분리 시 iGPU 를 denoise 에 집중 가능.
- **Jetson SoC 자원** (web 확인): AGX Orin 64GB = 2048-core Ampere iGPU + **2× NVDLA v2.0** + **PVA v2.0** + 12-core Cortex-A78AE CPU, 256-bit LPDDR5 204.8GB/s. cuDLA/TensorRT DLA delegate 경로 가용. DLA dtype: **FP16/INT8** ([Working with DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)).
- **DLA 비호환 op (소스 재확인, R72.3)**: `cosmos_framework/.../tokenizers/wan2pt2_vae_4x16x16.py` 실측 op 목록 — `CausalConv3d`(L74, `nn.Conv3d` 상속, DLA **3D conv 미지원**), `RMS_norm`(L102, DLA layer 목록 부재 — GroupNorm 과 동일 운명), `Upsample`(L117, nearest-exact `F.interpolate` — DLA 제한적), `AttentionBlock`(L256, `F.scaled_dot_product_attention` — DLA 미지원). **2D-friendly**: `nn.Conv2d`(L142,151,267 — DLA 후보), `SiLU`(L237). → **video VAE 의 지배 연산(CausalConv3d)이 DLA 미지원** = "frozen VAE→DLA" 전제 fragile (R-S5 계승).
- **scope (정직)**: policy mode 는 **video-latent decode skip** (tech report §4.2.5 — policy 는 action token 직접 출력, video 안 그림) → **S4 는 generation 모드(T2I/T2V/I2V) 한정**. policy on-robot 임팩트 제한 명시.

### 3.3 Baseline Source (R52.1) — 4종 + 실행 명령
- **(a) 전부-GPU 동기** (vllm-omni 기본): 현 코드 그대로. denoise 후 `_decode_latents` 동기. 측정 명령:
  ```bash
  # AGX Orin, generation-mode, 전부-iGPU 동기 baseline
  python -m vllm_omni.entrypoints.diffusion_cli \
    --model nvidia/Cosmos3-Nano --task t2v --resolution 256p --num-frames 68 \
    --num-inference-steps 30 --guidance 6.0 --profile-stage-durations \
    --device cuda:0   # _decode_latents 동기 (변경 없음)
  ```
- **(b) vae_patch_parallel (multi-GPU — 개념 대비)**: `diffusion/distributed/vae_patch_parallel.py#L71 _distributed_tiled_decode`. **rank별 tile 분할 + rank0 gather** = multi-GPU 전용. 단일 SoC 에는 직접 적용 불가 → **개념 대비**(같은 VAE 부하 분산이나 "여러 동질 GPU" 전제). 단일-GPU 강제 축소로 overhead 측정:
  ```bash
  python -m vllm_omni.entrypoints.diffusion_cli --model nvidia/Cosmos3-Nano \
    --task t2v --resolution 480p --vae-patch-parallel-size 1   # 단일 rank 강제 (tile path overhead)
  ```
- **(c) PipeDiT-식 GPU-내 overlap 재현**: VAE decode 를 별도 **CUDA stream(GPU 내)** 으로 분리, 다음 요청 denoise 와 SM 공유 overlap (DLA/CPU 미사용 = PipeDiT 의 단일-GPU 등가). M2 의 정면 비교군 (우리는 이종 가속기 추가):
  ```bash
  python bench/sidepool_overlap.py --mode gpu-stream-only \
    --decode-stream-priority -1 --denoise-stream-priority 0 --requests 100
  ```
- **(d) CPU-offload VAE**: `_decode_latents` 의 `self.vae.decode` 를 ARM CPU 로 (diffusers `enable_model_cpu_offload` 등가 또는 ONNX-CPU EP). 저해상도/audio 에서 유효성 측정:
  ```bash
  python bench/sidepool_placement.py --tokenizer video_vae --device cpu \
    --resolution 256p --frames 68 --dtype fp16 --measure latency,energy
  ```

---

## 4. 제안 기법

### M1 — Tokenizer Placement Characterization (LUT) — **standalone 핵심**

- **① 추가 scheme**: frozen tokenizer 3종 × 가속기 ×해상도×dtype 의 grid 를 자동 측정해 **placement LUT** 를 산출한다. 측정 차원:
  - tokenizer ∈ {video VAE encode, video VAE decode, audio VAE encode/decode, ViT vision encoder}
  - 가속기 ∈ {iGPU, DLA(변환 가능 subgraph), PVA, CPU(ARM)}
  - 해상도 ∈ {256p, 480p, 720p}(720p 는 시간상 subset)
  - dtype ∈ {INT8, FP16} (DLA 제약; iGPU 는 BF16 도)
  - 측정값 = (latency, energy[J], output accuracy[PSNR/SSIM vs BF16-GPU ref], **GPU-fallback%**)

- **② 해결 문제**: 단일 placement(전부 iGPU)가 모든 modality/해상도/dtype 에 최적이 아님. 특히 (RQ-a) DLA 변환 가능성·fallback 비율이 사전 미지(未知) → 측정 없이는 정책 불가. 이종-가속기 placement 지도가 미존재(PipeDiT/DeDiVAE 는 multi-GPU 동질이라 "어느 가속기" 문제 자체가 없음).

- **③ 동작 원리 (≥5 step)**:
  1. **ONNX export**: 각 frozen tokenizer 를 `torch.onnx.export` 로 추출. video VAE 는 chunk 단위(256p 68f 등) static-shape 로 export (DLA 는 dynamic shape 미지원 → static 필수).
  2. **DLA 변환 + fallback 측정**: `trtexec --onnx=video_vae_dec_256p.onnx --useDLACore=0 --fp16 --allowGPUFallback --verbose 2>&1 | grep -c "running on DLA\|GPU fallback"` 로 **DLA 에 올라간 layer 수 vs GPU fallback layer 수** 집계 → fallback% 산출. 이것이 W1-2 falsification gate.
  3. **2D-decomposable 변환 시도 (정직)**: `CausalConv3d(C,C,(3,3,3))` 를 (a) temporal 축 분리 → frame 별 `Conv2d(3,3)` + 별도 temporal `Conv1d/CausalConv3d(3,1,1)`(temporal-only 는 작음, GPU 유지) 로 **chunked 2D 변환**, 또는 (b) temporal kernel=1 인 layer 만 2D 로 강등. 변환 가능 subgraph(2D conv 부분)만 DLA, 잔여(temporal 3D + attention + RMS_norm)는 GPU fallback. **변환 불가능하면 정직하게 fallback% 그대로 보고** (DLA 무용 결론도 letter 가치).
  4. **cuDLA hybrid submission 측정**: 변환 성공 subgraph 를 `cudaStream` 에 cuDLA task 로 제출(`cudlaSubmitTask`), GPU↔DLA 데이터 이동(UMA 이므로 zero-copy 시도, 안 되면 `cudaMemcpyAsync`) 포함 end-to-end latency 측정.
  5. **energy 측정**: 각 placement 에서 tegrastats(VDD_GPU_SOC / VDD_CPU_CV rail) 적분. video VAE chunk(>50ms) 직접 측정, 소형 audio(<33ms sampling) 는 N-chunk 적분/평균 (CF-B 계승: INA3221 33-50ms < step).
  6. **accuracy 측정**: DLA INT8 / FP16 출력 vs BF16-GPU reference 의 PSNR/SSIM(video), spectral L2(audio). 정확도 손실이 placement 정책의 제약조건.
  7. **LUT 구축**: 위 grid 를 `(tokenizer, resolution, dtype) → argmin_latency / argmin_energy / accuracy-constrained placement` 룩업표로 직렬화 (JSON/YAML), 서빙 런타임이 load.

- **③' Pseudo-code (M1 측정 matrix 자동화, ≥10줄)**:
  ```python
  # sidepool_characterize.py — placement LUT grid 측정 자동화
  TOKENIZERS = ["video_vae_dec", "video_vae_enc", "audio_vae", "vit_encoder"]
  ACCEL      = ["igpu", "dla0", "pva", "cpu"]
  RES        = ["256p", "480p", "720p"]    # 720p subset
  DTYPE      = ["int8", "fp16"]
  lut = {}
  for tok in TOKENIZERS:
      onnx_path = export_onnx(tok, static_shape=chunk_shape(tok))      # step 1
      for acc in ACCEL:
          for res in RES:
              for dt in DTYPE:
                  if acc.startswith("dla"):
                      eng, fb = build_trt_dla(onnx_path, res, dt,      # step 2-3
                                              core=int(acc[-1]), gpu_fallback=True)
                      fb_ratio = fb.dla_fallback_layers / fb.total_layers
                  else:
                      eng, fb_ratio = build_backend(onnx_path, acc, res, dt), 0.0
                  lat  = bench_latency(eng, acc, n=50)                  # step 4
                  enrg = integrate_tegrastats(eng, acc, rail="VDD_GPU_SOC")  # step 5
                  psnr = accuracy_vs_ref(eng, ref=bf16_gpu_ref(tok,res))     # step 6
                  lut[(tok,acc,res,dt)] = dict(lat=lat, J=enrg,
                                               psnr=psnr, dla_fallback=fb_ratio)
  dump_lut(lut, "placement_lut.json")     # step 7: argmin policy 후처리
  ```

- **④ 기존 실패 이유 + 차별화**:
  - **vLLM-Omni VAE-Patch-Parallel** (`vae_patch_parallel.py`) = **multi-GPU** tile 분산 — 단일 SoC 이종 가속기 placement 아님.
  - **PipeDiT [arXiv:2511.12056] / DeDiVAE [arXiv:2512.07350]** = multi-GPU group 으로 VAE↔denoise 분리 — "어느 가속기" 문제 부재(GPU 동질).
  - **LiteVLA-Edge [arXiv:2603.03380]** = AR backbone only(no VAE placement).
  - **EC-Diff [arXiv:2507.11980]** = edge-cloud 분산 토폴로지 (SoC 내 placement 아님).
  - **차별 = 단일 edge SoC 의 이종 가속기 복합체(iGPU/DLA-2Dsubgraph/PVA/CPU)에 대한 modality×해상도×dtype placement LUT** = 직접 prior art 부재 (Step-0 refresh: 신규 0건). DLA-only 주장(직전 L5)을 accelerator-complex 측정으로 정직화.

> ✅ source verified: vllm-omni@95d56cf [`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L987-1004`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L987-L1004) (`_decode_latents`, `self.vae.decode` 동기), [`#L1131-1153`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1131-L1153) (`_encode_video_tensor`)
> ✅ source verified: vllm-omni@95d56cf [`vllm_omni/diffusion/distributed/vae_patch_parallel.py#L71`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/distributed/vae_patch_parallel.py#L71) (`_distributed_tiled_decode`, multi-GPU tile)
> ✅ source verified: cosmos-framework@003d66d [`cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py#L74`](https://github.com/nvidia/cosmos-framework/blob/main/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py#L74) (`CausalConv3d(nn.Conv3d)`), [`#L102`](https://github.com/nvidia/cosmos-framework/blob/main/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py#L102) (`RMS_norm`), [`#L117`](https://github.com/nvidia/cosmos-framework/blob/main/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py#L117) (`Upsample`), [`#L256-283`](https://github.com/nvidia/cosmos-framework/blob/main/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py#L256-L283) (`AttentionBlock`, SDPA)
> ✅ source verified: cosmos-framework@003d66d [`cosmos_framework/tools/flops/wan_vae.py#L55`](https://github.com/nvidia/cosmos-framework/blob/main/cosmos_framework/tools/flops/wan_vae.py#L55) (`_causalconv3d_flops`) — VAE FLOPs roofline 근거
> (proposed) placement LUT grid 측정 자동화, 2D-decomposable 변환 파이프라인, cuDLA hybrid submission — 신규.

#### M1 R52.2 표 (코드 사실 vs 제안)

| 항목 | 코드에 있는 것 (verified) | 제안 (proposed) |
|---|---|---|
| VAE decode | `self.vae.decode(latents)` 동기, iGPU, BF16 ([`pipeline#L1003`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1003)) | ONNX export → 이종 가속기 placement grid |
| VAE op | `CausalConv3d`/`RMS_norm`/`Upsample`/SDPA ([소스 확인](https://github.com/nvidia/cosmos-framework/blob/main/cosmos_framework/model/vfm/tokenizers/wan2pt2_vae_4x16x16.py#L74)) | 2D-decomposable subgraph 추출 + DLA 매핑 |
| 분산 | multi-GPU tile ([`vae_patch_parallel`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/distributed/vae_patch_parallel.py#L71)) | 단일-SoC 이종 가속기 placement LUT |
| dtype | BF16 only | INT8/FP16(DLA 제약) + accuracy grid |
| 측정 | `stage_durations` (phase별 시간, [`diffusion_pipeline_profiler#L107`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py#L107)) | + fallback%, energy, PSNR grid |

---

### M2 — LUT-Driven Async Dispatch (overlap, **보조**)

- **① 추가 scheme**: M1 의 placement LUT 가 video VAE decode 를 "denoise 와 다른 자원"(예: DLA-subgraph + GPU-fallback, 또는 별도 GPU stream)으로 지정하면, `_decode_latents`(현 요청 i)를 **별도 CUDA stream(낮은 priority) 또는 cuDLA submission** 으로 비동기 dispatch 하고, **다음 요청 i+1 의 denoise**(높은 priority)와 겹친다. 연속 generation 시나리오 전용.
- **② 해결 문제**: G8 — VAE 가 denoise SM/BW 경합. denoise 종료 후 동기 VAE(`pipeline#L1867`)가 critical path 에 직렬 추가 → 연속 요청에서 VAE 가 다음 denoise 와 겹칠 수 있는데 미활용. iGPU issue-slot 25-40% 유휴 보전.
- **③ 동작 원리 (≥5 step)**:
  1. **stream 구성**: `denoise_stream = torch.cuda.Stream(priority=high)`, `vae_stream = torch.cuda.Stream(priority=low)` (`cudaStreamCreateWithPriority` — Orin 지원 확인). LUT 가 DLA 지정이면 `cudaStreamToDla` 매핑 + cuDLA task.
  2. **double-buffer**: 요청 i 의 latents 를 `latents_buf[i%2]` 에 보관, denoise(i+1)는 `denoise_stream`, decode(i)는 `vae_stream`/cuDLA 로 동시 발행.
  3. **이벤트 동기**: `ev_i = torch.cuda.Event(); ev_i.record(vae_stream)` 후, 출력 수집 시점에 `ev_i.synchronize()` — denoise(i+1) 진행과 무관하게 decode(i) 완료 보장.
  4. **cuDLA submission (DLA placement 시)**: `cudlaSubmitTask(dla_dev, &task, stream)` 로 DLA-subgraph 제출, GPU fallback 잔여는 `vae_stream` 에서 실행 (hybrid). UMA zero-copy 시도, 실패 시 `cudaMemcpyAsync`.
  5. **간섭 측정**: denoise(i+1) ∥ decode(i) 의 EMC BW 경합(tegrastats EMC%)·SM 경합(wall-clock 증가율) 측정 → 간섭이 overlap 이득을 잠식하면 LUT 가 CPU/DLA(SM 비경합 자원)로 재지정.
  6. **fallback 경로**: 단일 요청(연속 아님)이면 overlap 기회 없음 → M1 의 best-single-placement 로 동기 실행(graceful).
- **③' Pseudo-code (M2 async dispatcher, ≥10줄)**:
  ```python
  # sidepool_dispatch.py — LUT-driven async VAE dispatch (연속 generation)
  class SidepoolDispatcher:
      def __init__(self, lut, vae):
          self.lut = lut
          self.denoise_s = torch.cuda.Stream(priority=-1)   # high
          self.vae_s     = torch.cuda.Stream(priority=0)    # low
          self.pending   = []   # (event, out_buf) 큐
      def run(self, requests):
          for i, req in enumerate(requests):
              with torch.cuda.stream(self.denoise_s):
                  latents = denoise(req)                      # 본류 (high prio)
              place = self.lut.lookup(req.modality, req.res, req.dtype)  # M1 정책
              if place.accel == "dla":
                  ev = submit_cudla_decode(latents, place, self.vae_s)   # 측면 풀
              else:                                                       # gpu-stream/cpu
                  with torch.cuda.stream(self.vae_s):
                      out = vae_decode(latents); ev = torch.cuda.Event(); ev.record(self.vae_s)
              self.pending.append((ev, req.id))
              self._drain_completed()        # 완료된 decode 수확 (denoise 와 겹친 채)
          self._drain_all()
  ```
- **④ 기존 실패 이유 + 차별화**:
  - **PipeDiT [arXiv:2511.12056]**: GPU-내 SM 분할 pipeline (VAE 와 denoise 가 같은 GPU 의 SM 을 시분할/공간분할). **우리: 이종 가속기(DLA/CPU)로 VAE 를 물리적으로 다른 die-block 에 보내 SM 경합 자체를 제거** + LUT 가 자원 선택. **DeDiVAE [arXiv:2512.07350]**: VAE 를 별도 GPU group decoupling — 단일 SoC 불가.
  - **핵심 차별**: PipeDiT/DeDiVAE 의 overlap 은 "같은/다른 GPU" 차원, 우리는 "단일 SoC 의 이종 고정기능 가속기 + LUT 정책" 차원. **M2 단독 promote 불가** (overlap 발상은 선점됨) — **M1 placement LUT 가 standalone 핵심, M2 는 LUT 의 활용처**.

> ✅ source verified: vllm-omni@95d56cf [`pipeline_cosmos3.py#L1867`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1867) (`video = self._decode_latents(latents)` — diffuse 종료 후 동기), [`#L1282 diffuse`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1282), [`#L1332 reset_cache`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1332)
> ✅ source verified: vllm-omni@95d56cf [`pipeline_cosmos3.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py) 전역 grep — `cuda.Stream`/`cudaEvent`/`priority` **0건** = stream overlap 완전 greenfield (트랩 §S4 계승)
> ✅ source verified: vllm-omni@95d56cf [`diffusion/profiler/diffusion_pipeline_profiler.py#L80-111`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py#L80-L111) (`stage_durations` — denoise vs vae phase 관찰점)
> (proposed) LUT-driven stream/cuDLA async dispatch, double-buffer, 간섭 측정 — 신규.

#### M2 R52.2 표

| 항목 | 코드에 있는 것 (verified) | 제안 (proposed) |
|---|---|---|
| decode 실행 | diffuse 종료 후 동기 ([`#L1867`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1867)) | LUT-지정 자원에 async stream/cuDLA dispatch |
| stream | 명시적 stream/event 0건 (greenfield) | `cudaStreamCreateWithPriority` + `cudaEvent` double-buffer |
| 연속 요청 | 요청 간 overlap 없음 | decode(i) ∥ denoise(i+1) |
| 간섭 | 미측정 | EMC%/SM wall-clock 경합 측정 → LUT 재지정 |

#### 의존성 그래프 (M1 → M2)
```
[ONNX export] ──> [trtexec --useDLACore + fallback% 측정]  (M1.2-3)
       │                         │
       v                         v
[2D-decomp 변환]          [cuDLA hybrid latency]  (M1.4)
       │                         │
       └──────> [placement LUT (latency/J/PSNR/fallback)] (M1.7)  ★standalone letter
                          │
                          v
                [LUT-driven async dispatch] (M2)  — 연속 generation 한정, 보조
                          │
                          v
                [간섭 측정 → LUT 재지정 루프] (M2.5)
```

---

## 5. 평가 · 실험 플랜 (R20-β 7요소)

### 5.1 HW
- **AGX Orin 64GB** (primary): 2048-core Ampere iGPU + **2× NVDLA v2.0** + **PVA v2.0** + 12-core Cortex-A78AE, 204.8GB/s LPDDR5. DLA 2기 → 2개 tokenizer 동시 DLA 가능성도 측정.
- **Orin NX 16GB** (보조): DLA 有(소형), capacity-wall — video VAE buffer 와 denoise 경합 측정.
- **Thor** (forward-looking, conditional): 차세대 DLA(NVDLA 3.x 급) — DLA 지원 op 확대 가능성 측정 (가용 시).
- **RTX Pro 6000 Blackwell 96GB** (**DLA 없음 → GPU-only 대조군**): placement 의 GPU-stream-only baseline + accuracy reference(BF16). ncu 가용(sm_120)이나 본 idea 는 ncu 불요(Orin 측정 완결).

### 5.2 모델
- **Cosmos3-Nano** generation-mode: T2I + T2V-256p/480p (720p 는 시간상 subset). VAE = Wan2.2 video VAE(4×16×16, frozen) + audio VAE(AVAE, frozen) + ViT vision encoder(I2V conditioning).
- fallback: VAE 단독 microbench (모델 적재 없이 tokenizer 만).

### 5.3 Workload (generation-mode only)
- **연속 50-100 generation** 시퀀스 (M2 overlap 측정 — 단일 요청은 overlap 기회 없음 명시).
- I2V (encode-heavy: conditioning encode `_encode_video_tensor`) / T2V-짧은clip (decode-heavy: `_decode_latents`) / T2I + Audio (audio VAE placement).
- **policy mode 제외** (video-latent decode skip, §3.2 — VAE 이득 無).

### 5.4 Tools
- `trtexec --useDLACore=N --fp16/--int8 --allowGPUFallback` (DLA 변환 + fallback% gate)
- cuDLA API (`cudlaSubmitTask`, hybrid GPU↔DLA)
- `torch.onnx.export` (static-shape chunk export)
- `torch.cuda.Stream(priority)` / `cudaEvent` (M2 async)
- Nsight Systems (timeline/range marker — 자원별 overlap 시각화)
- tegrastats (EMC%/전력 적분 — energy, ncu 불요)

### 5.5 Ablation + Metrics
- **placement ablation**: {GPU, DLA(2Dsubgraph), PVA, CPU} × {encode, decode} × {256p, 480p, 720p} × {INT8, FP16}.
- **metrics**: (1) e2e chunk latency, (2) **GPU-점유 해방률**(VAE 제거 후 denoise stream 점유 회복%), (3) **J/image (J/chunk)**, (4) **DLA fallback 비율**(W1-2 gate), (5) output PSNR/SSIM(accuracy 제약), (6) overlap%(M2, vs PipeDiT-식 baseline c).

### 5.6 주차별 표 (~10주, 파일경로 + 완료판정)

| 주차 | 작업 | 산출 파일 | 완료 판정 |
|---|---|---|---|
| **W1-2** | baseline (a)/(c)/(d) 측정 + VAE 단독 프로파일 (stage_durations 로 VAE 가 e2e 의 몇 %인지) | `bench/baseline_{a,c,d}.json`, `prof/vae_share.csv` | 4 baseline e2e 수치 확보 + VAE share% 산출 |
| **W3-4** | ONNX export + DLA 변환(`trtexec --useDLACore`) + **fallback% 측정** (M1.2-3) + **INT8 DLA calibration set 구축**(대표 latent 수백 샘플 PTQ → INT8 scale, FP16 은 calib 불요) | `onnx/*.onnx`, `trt/dla_fallback.json`, `calib/vae_latent_calib.npz` | **video VAE DLA fallback% gate** 측정 완료 + INT8 engine 빌드용 calib set 확보 + INT8 PSNR accuracy gate |
| **W5-6** | 2D-decomposable 변환 시도 + cuDLA hybrid latency + audio/ViT placement → **LUT 구축** (M1.7) | `placement_lut.json`, `bench/cudla_hybrid.csv` | LUT grid {4 tok × 4 accel × 3 res × 2 dtype} 채움 |
| **W7-8** | LUT-driven async dispatch (M2) 구현 + double-buffer + overlap% 측정 | `sidepool_dispatch.py`, `bench/overlap.csv` | 연속 50-100 gen overlap% vs baseline(c) 비교 |
| **W9-10** | e2e (placement LUT + async) + 간섭 측정 + ablation + scope 검증 | `results/e2e_ablation.csv`, paper draft | 전 metric 표 완성 + decision-tree 분기 판정 |

### 5.7 Preliminary 4단계 (R20-β)
1. **reproduction**: vllm-omni Cosmos3-Nano T2V-256p generation 1회 재현 (`diffusion_cli` 정상 출력) — 환경 sanity.
2. **attribution** (VAE 가 e2e 의 몇 %): `stage_durations` 활성화 후
   ```bash
   python -m vllm_omni.entrypoints.diffusion_cli --model nvidia/Cosmos3-Nano \
     --task t2v --resolution 256p --num-frames 68 --profile-stage-durations
   # 출력: denoise_ms vs vae_decode_ms → VAE share = vae_ms / total_ms
   ```
   VAE share **<8% 면 drop** (decision-tree). 10-25% 면 진행.
3. **roofline** (VAE FLOPs·BW): `cosmos_framework/tools/flops/wan_vae.py compute_wan_vae_encoder_flops` 로 256p 68f FLOPs 계산 + LPDDR5 204.8GB/s 대비 arithmetic intensity → VAE 가 compute-bound 인지 BW-bound 인지 판정 (placement 자원 선택 근거).
4. **micro-benchmark** (DLA vs GPU 단일 conv block A/B): 단일 `Conv2d(256,256,3)` block 을 iGPU vs DLA(`trtexec --useDLACore=0`) 로 latency/J A/B — DLA 이득 상한 추정 (3D 변환 전 sanity).

---

## 6. 예상 효과 · Risk · Tier · Scoring · Decision-tree

### 6.1 예상 효과 (보수, scope 명시)
- **VAE 비중 10-25% 구간**에서 e2e latency **5-15%↓** (placement + async overlap, VAE critical path 비중만큼 — DLA 무관하게 GPU-stream/CPU 분리로도 성립) **또는** GPU 점유 해방으로 **throughput 동등 향상** (denoise stream 회복).
- J/chunk **5-15%↓** (DLA/CPU 2D-subgraph 오프로드 성공 시 — **부분**, fallback% 의존).
- **특성화-only 도 letter 성립 (명시)**: DLA 전환 전부 실패해도 **"단일 edge SoC 에서 frozen multimodal tokenizer 의 이종-가속기 placement 특성화 + fallback% 정량 + roofline"** = ISPASS measurement letter 로 독립 성립 (negative result 도 인용 가치). 이것이 본 idea 의 risk floor.

### 6.2 Risk + 완화
- **R1 (DLA fallback 과다)**: video VAE CausalConv3d 3D conv DLA 미지원 → fallback% 높음. → **2D-decomposable 변환 시도**(M1.3) + 실패 시 **CPU/GPU-stream 조합으로 placement**(DLA 없이도 M1 LUT·M2 overlap 성립). fallback% step0 gate.
- **R2 (3D-conv 변환 불가)**: temporal 3D 가 분리 안 되면 → **encode 만 DLA**(encode 경로의 2D 비중이 다를 수 있음 — 측정으로 판정), decode 는 GPU-stream/CPU.
- **R3 (overlap 발상 선점)**: PipeDiT/DeDiVAE → **M1 placement LUT 가 standalone 핵심, M2 overlap 은 보조** (위계 명확). M2 단독 promote 금지.
- **R4 (RTX DLA 없음)**: GPU-only control 정직 비교 (placement 의 GPU-stream baseline).
- **R5 (policy VAE-skip)**: generation-mode 한정 scope 명시, robotics on-robot 임팩트 제한 정직화.
- **R6 (DLA INT8 calibration 비용·품질)**: DLA **INT8 engine 빌드는 calibration dataset 필요** — VAE decode 의 대표 latent 분포(수백 샘플) 로 PTQ calibration 을 해야 INT8 scale 산출 가능(FP16 DLA 는 calib 불요). → calibration set 구축·실행 비용(시간·대표성)이 placement grid 의 INT8 cell 마다 발생, **calib 분포가 빈약하면 INT8 PSNR/SSIM 저하**(accuracy 제약 위반)로 INT8 placement 가 무용. → **calibration set(대표 latent) 구축을 W3-4 항목에 명시**하고, INT8 cell 은 calib 품질을 accuracy gate 와 병행 판정. (DLA fallback% 와 별개로 INT8 정확도 gate 추가.)

### 6.3 R52.4 Tier-A/B/C
- **Tier-A (필수, 무조건 산출)**: placement LUT grid {tokenizer × 가속기 × 해상도 × dtype → latency/J/PSNR/fallback%} + VAE share% attribution + DLA fallback% 측정. = 특성화 letter 의 골격.
- **Tier-B (조건부, gate 통과 시)**: LUT-driven async dispatch(M2) e2e 이득 + cuDLA hybrid 실측. (연속 generation + DLA subgraph ≥20% 조건.)
- **Tier-C (stretch)**: Thor 차세대 DLA op-확대 비교 + 2 DLA 동시(2 tokenizer 병렬) + ViT encoder placement 까지 확장.

### 6.4 Tier-1 승격 조건
- (1) placement LUT 가 **단일 자원 대비 e2e 25%+ 이득**을 일관되게 보이고, (2) DLA 2D-decomposable 변환이 video VAE 에서 **fallback% <30%** 로 성공해 DLA 가 실효 자원이 되며, (3) 이종-가속기 placement 가 **여러 모델(Cosmos3 외 1개)로 일반화**되면 → "edge SoC tokenizer placement runtime" 으로 Tier-1(시스템 런타임) 승격 가능. 현 상태(DLA fragile)에서는 Tier-2 measurement/placement study 가 정직.

### 6.5 Scoring (직전 6.2 → 재작성 반영)
| 축 | 점수 | 근거 |
|---|---|---|
| novelty | **6** ▼ | placement study, 직접 prior art 부재 강점이나 overlap 발상 PipeDiT/DeDiVAE 선점 (worst-overlap concurrent 45-55%) |
| differentiation | 6 | vLLM-Omni VAE-Patch-Parallel(multi-GPU) + PipeDiT 대비 단일-SoC 이종 가속기 LUT |
| impact | 6 | generation-mode 한정 (policy VAE-skip) |
| ai-impl | 6 | DLA 변환 fragile 하나 GPU-stream/CPU placement 견고 |
| arch-impl | **7** ★ | ncu 불요 Orin 측정 완결 + stream/cuDLA dispatch 견고 + 특성화-only floor |
| **avg** | **6.2** | 직전 동일 (★arch 7 / ▼novelty 6 — task 지정 일치) |

### 6.6 Decision-tree 분기
```
W1-2: VAE share% 측정 (attribution)
 ├─ VAE share < 8%  ──────────────> DROP (VAE 가 비병목, idea 무의미)
 └─ VAE share 8-25%+ ─> 진행
      W3-4: DLA fallback% 측정 (trtexec --useDLACore)
       ├─ DLA-supported subgraph < 20% (fallback >50%) ─> 특성화-only:
       │     placement LUT = {GPU-stream, CPU, PVA} 한정, DLA = negative-char 보고
       │     → ISPASS measurement letter (Tier-A 만)
       └─ DLA subgraph ≥ 20% ─> 진행
            W7-8: overlap% vs PipeDiT-식 baseline(c)
             ├─ M2 overlap% ≤ baseline(c) (이종가속기 우위 없음) ─> M2 drop,
             │     M1 placement LUT 단독 letter (Tier-A + cuDLA latency)
             └─ M2 overlap% > baseline(c) ─> 전체 (M1 LUT + M2 async) Tier-A+B
                  → Tier-1 승격 검토 (§6.4 3조건)
```

---

## 7. Inter-idea / 종합

- **S3 LEDGERMARK(특성화 producer)와 보완**: LEDGERMARK 가 phase-transition/co-residency/J-chunk 의 일반 ledger 라면, SIDEPOOL 은 **frozen tokenizer 자원-축 placement** 의 특화 ledger — modality working-set(video 수만 ≫ audio 25/s) 측정을 공유 가능.
- **S1 TIDELOOM 의 `DECODE_OUT` phase 와 연계**: TIDELOOM 의 FSM `DECODE_OUT`(VAE) 단계가 SIDEPOOL 의 placement LUT 를 호출해 자원 선택 → runtime 통합 시 자연 흡수점.
- **핵심 한 줄**: PipeDiT/DeDiVAE 가 "VAE∥denoise overlap"(multi-GPU)을 선점했으므로, SIDEPOOL 은 **단일 edge SoC 이종 가속기 복합체 위 frozen multimodal tokenizer 의 placement 특성화 + LUT 정책**을 핵심 기여로, overlap 은 그 활용처로 정직하게 재정위한 placement study (ISPASS/DATE Tier-2).
