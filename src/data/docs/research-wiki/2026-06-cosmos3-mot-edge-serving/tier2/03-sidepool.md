# S4 SIDEPOOL (Tier-2 #3 독립, avg 6.2, placement-LUT 중심 rescope)

**Phase-Overlapped VAE Encoder/Decoder Placement and Overlap Study across the Edge SoC Accelerator Complex (iGPU/DLA/PVA/CPU) for Dual-Tower MoT Generation** · DATE / DAC / ISPASS 2027

---

## 1. 개요

- **metaphor↔mechanism**: SIDEPOOL = SIDE(VAE 를 denoise 본류 옆 side-accelerator stream 으로 분리) + POOL(iGPU/DLA/PVA/CPU accelerator complex pool). denoise 본류 옆 별도 pool 에서 VAE 를 흘려 본류와 겹친다.
- **한 줄**: VAE encode/decode 를 denoise 와 같은 iGPU 직렬 실행하던 것을 GPU stream-priority overlap + accelerator-complex placement LUT 로 분리.
- **rescope (R-S5)**: Wan2.2 video VAE 의 지배 연산(3D causal conv/GroupNorm)이 DLA 미지원 → "frozen VAE→DLA" 전제 fragile. **placement LUT(M2)를 standalone 핵심**으로, M1 GPU stream-overlap 은 단일-device 차별로 (PipeDiT/DeDiVAE overlap 선점 반영).

## 2. 기존 연구 한계 · GAP (workload evidence 수치 포함)

- **GAP G8(VAE 병목)**: VAE encode/decode 가 denoise 와 같은 iGPU 직렬 실행 → SM·BW 경합. Edge/Nano 급에서 **VAE 가 비-amortized 지배 비용**(transformer 작아 amortize 안 됨, §3.4/5.2.6).
- **workload evidence**:
  - VAE = Wan2.2 video VAE(4× temporal, 32×32 spatial, frozen) + audio VAE(48kHz, frozen). chunked encoding(256p 68f/480p 24f/720p 12f). critical path: I2V conditioning encode + (policy 외) latent decode.
  - AGX Orin/NX **2× NVDLA 3.1**(conv/deconv/pool/activation, INT8/FP16) + **PVA**. Orin "SM issue-slot 25-40%" ([arXiv:2508.08430](https://arxiv.org/abs/2508.08430)) → VAE 분리 시 GPU 를 denoise 집중.
  - Wan2.2 video VAE 3D causal conv·GroupNorm = **DLA 미지원→GPU fallback 빈발** (NVIDIA Working-with-DLA 문서).
- **공백**: PipeDiT([arXiv:2511.12056](https://arxiv.org/abs/2511.12056))/DeDiVAE([arXiv:2512.07350](https://arxiv.org/abs/2512.07350))가 "VAE∥denoise pipeline overlap" 핵심 발상 보유(단 multi-GPU group 분리) → 단일-device placement LUT 가 빈 영역.

## 3. 제안 기법 (mechanism ≤2)

### M1. GPU stream-priority overlap of VAE ∥ next-chunk denoise (DLA 무관)

- **① 추가 scheme**: VAE decode(현 chunk)를 별도 CUDA stream(낮은 priority) + `cudaEvent` 동기로 denoise transformer(다음 chunk, 높은 priority)와 overlap.
- **② 해결 문제**: G8 — VAE 가 denoise SM/BW 경합, GPU 활용 갭(25-40% issue-slot) 보전.
- **③ 동작 원리**: (1) `cudaStreamCreateWithPriority`(Orin 지원)로 denoise=high, VAE=low. (2) double-buffer 로 VAE(chunk i) ∥ denoise(chunk i+1). (3) `cudaEvent` 로 chunk 경계 동기. (4) overlap% 측정(VAE critical path 비중).
- **④ 기존 실패 이유 + 차별화**: vLLM-Omni VAE-Patch-Parallel = multi-GPU; LiteVLA-Edge([arXiv:2603.03380](https://arxiv.org/abs/2603.03380)) = AR backbone only(no VAE). PipeDiT([arXiv:2511.12056](https://arxiv.org/abs/2511.12056))/DeDiVAE([arXiv:2512.07350](https://arxiv.org/abs/2512.07350))는 multi-GPU group decoupling → 단일 edge GPU stream-overlap 미개척 (단 M1 단독 promote 불가, M2 가 standalone 핵심).

### M2. Accelerator-complex placement LUT (iGPU/DLA-2Dsubgraph/PVA/CPU) — standalone 핵심

- **① 추가 scheme**: VAE 종류(video/audio)·해상도(256/480/720p)·chunk frame 수별 best placement LUT. DLA 는 검증된 2D-subgraph 만 선택 오프로드(3D conv/GroupNorm 은 GPU 유지), audio VAE(작음)=CPU/PVA.
- **② 해결 문제**: 단일 placement 가 모든 모달리티/해상도에 최적 아님 + DLA 부분 오프로드 ROI 정량.
- **③ 동작 원리**: (1) VAE ONNX export → `trtexec --useDLACore=0 --fp16 --allowGPUFallback` 로 **DLA-supported subgraph 비율 step0 실측**(falsification gate). (2) iGPU vs DLA(2D-subgraph) vs PVA vs CPU(ARM) 3-4-way placement × 3-resolution × {video,audio} latency/J grid. (3) DLA-friendly subgraph 비율이 이득 결정 → 미지원 多면 GPU stream-overlap(M1)만으로 graceful.
- **④ 기존 실패 이유 + 차별화**: 모달리티-aware accelerator placement(G6+G8) 미존재. DLA-only 주장(L5) → accelerator-complex 측정으로 정직화.

## 4. 평가 · 실험 플랜 (R20-β 7요소)

- **HW**: AGX Orin 64GB(2× DLA + PVA) / Orin NX 16GB(DLA 有) / Thor(차세대 DLA) / RTX Pro 6000(DLA 없음 → iGPU-only control, overlap baseline).
- **모델**: Cosmos3-Nano(Wan2.2 VAE + audio VAE); fallback VAE 단독 microbench.
- **워크로드 (generation-mode only)**: I2V(encode-heavy) / T2V-짧은clip(decode-heavy) / T2I+Audio. (policy 는 video-latent decode skip → VAE 이득 無, 명시 제외.)
- **도구**: `trtexec --useDLACore`, `cudaStreamCreateWithPriority`, tegrastats(ncu 불요 → Orin 측정 완결).
- **ablation + baseline**: M1 overlap on/off, DLA 2D-subgraph on/off. baseline — (b1) all-on-iGPU naive serial, (b2) vLLM-Omni VAE-Patch-Parallel single-GPU 축소, (b3) CPU-VAE(ARM) fallback.
- **주차별 구현**:

  | 주차 | 작업 |
  |---|---|
  | W1-2 | **DLA-supported subgraph 비율 step0 실측**(falsification gate) + VAE microbench |
  | W3-5 | GPU stream-priority overlap(M1) 구현·overlap% 측정 |
  | W6-8 | placement LUT(iGPU/DLA-2D/PVA/CPU) grid + modality-aware |
  | W9-10 | baseline 비교 + generation-mode scope 검증 |

- **preliminary metrics**: DLA-supported subgraph 비율(W1-2 gate), VAE critical path 비중.

## 5. 예상 효과 (보수치, scope 명시)

- e2e chunk latency **−15~30%** (M1 GPU stream-overlap, VAE critical path 비중만큼 — **DLA 무관 성립**).
- J/chunk **−10~20%** (DLA 2D-subgraph 오프로드 성공 시, **부분**).
- denoise GPU occupancy 회복.
- scope: **generation-mode 한정**(T2I/T2V/I2V), policy on-robot 임팩트 제한 정직화. DLA subgraph <20% 면 M1 만, M1 PipeDiT 대비 무우위면 M2 placement LUT 만 standalone.
