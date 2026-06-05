# L3 KEELKV (Tier-2 #2, deep) — L2-Resident Static Conditioning KV for Repeated-Denoise Inference on Edge GPUs

**KEELKV: L2-Resident Static Conditioning KV for Repeated-Denoise Inference on Edge GPUs**
· 목표 Venue: **DAC / DATE 2027** (차선: IEEE CAL letter)
· 작성일 2026-06-05 · Mode-1 deep-spec (R28-α: Tier-2 도 Tier-1 동일 detail)
· G4 소유 축 = **on-chip L2 placement** (numeric quant=Q2, runtime layout=A2 위임)
· Pair: **Q2 ANCHOR** (producer-consumer — INT4 K_AR(~1.08MB/layer) 이 3MB L2 set-aside 에 여유로 상주)

> **metaphor**: 용골(keel)은 배를 처음 건조할 때 밑바닥에 한 번 깔리면 항해 내내 떼어내지 않는 구조재다. denoise loop 내내 **read-only** 인 conditioning KV (`K_AR`/`V_AR`) 를 L2 set-aside 에 한 번 "깔아두고" 항해(=N step × CFG 2 pass) 내내 고정한다. 매 step 흘러가는 video-K_DM 은 streaming 으로 흘려보내 용골을 덮지 않게 한다.

---

## 0. 약어 Glossary

| 약어 | 정의 |
|---|---|
| **L2 set-aside** | GPU L2 cache 의 일부를 persisting access 전용으로 떼어내는 영역. `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, n)` 로 크기 지정. 이 영역에 매핑된 주소는 evict 우선순위가 낮아져 반복 read 시 상주. |
| **cudaAccessPolicyWindow** | CUDA stream/graph 에 부착하는 구조체. `{base_ptr, num_bytes, hitRatio, hitProp, missProp}` 로 특정 global-memory 윈도우의 L2 캐싱 정책을 지정. ([CUDA C Programming Guide §L2 Access Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-cache-set-aside-for-persisting-accesses)) |
| **hitRatio** | window 내 access 중 persisting 정책을 적용할 비율(0~1). `hitRatio≈1.0` = window 전량을 persisting 취급(window ≤ set-aside 크기일 때만 권장; 초과 시 thrash). |
| **hitProp / missProp** | window 안(hit)·밖(miss) access 의 캐싱 속성. `cudaAccessPropertyPersisting` (상주) / `cudaAccessPropertyStreaming` (1회 쓰고 즉시 evict 후보) / `Normal`. |
| **persisting / streaming access** | persisting = 반복 read 데이터(우리의 K_AR/V_AR). streaming = 1회성 흘러가는 데이터(video-K_DM, noisy latent). |
| **GQA** (Grouped-Query Attention) | query head 수 > KV head 수. KV head 수가 적어 K/V 메모리가 작다 — KEELKV 의 fit 에 유리. Cosmos3 cross-attn 은 `num_kv_heads`(=8) 로 GQA. |
| **KV-head** | key/value projection 의 head. K_AR footprint = `tokens × num_kv_heads × head_dim × 2(K,V) × precision`. |
| **ncu** (Nsight Compute) | per-kernel HW counter 프로파일러. `l2_tex_hit_rate`, `lts__t_sectors_*` 등. **Orin(ga10b/sm_87) 미지원** — RTX/Thor 에서만. |
| **K_AR / V_AR** | reasoner(understanding tower) 가 1회 forward 로 만든 cross-tower conditioning KV. denoise 전 기간 read-only·step-invariant. 코드상 `Cosmos3VFMTransformer.cached_kv`. |
| **K_DM / V_DM** | generator(denoise) self-attention 의 KV. 매 step noisy latent 에서 재계산 = streaming. |
| **EMC** | Jetson 의 External Memory Controller (LPDDR5). `tegrastats` 가 EMC %·freq 노출. Orin BW proxy. |
| **denoise loop** | sampler 의 `for t in timesteps` 루프 (`pipeline_cosmos3.diffuse`). policy mode N=4 step × CFG 2 pass. |

---

## 1. 개요

### 1.1 한 줄 RQ (single-insight)

> **read-only 로 반복-read 되는 conditioning KV 를 edge GPU 의 L2 set-aside 에 못 박으면, denoise-loop 의 DRAM 왕복 BW 를 얼마나 줄일 수 있는가?**

Cosmos3 dual-tower MoT 의 generator 는 매 denoise step·매 layer 에서 reasoner 가 만든 `K_AR/V_AR` 을 cross-attention 입력으로 **전량 다시 읽는다**(`cat([k_und, k])`, L648). 이 텐서는 denoise 전 기간 단 한 번 쓰이고(L1474-1479) 그 뒤로는 **절대 갱신되지 않는** 순수 read-only 데이터다. 그럼에도 현 구현은 매 step 이를 DRAM(LPDDR5) 에서 재독출하고, 동시에 흘러가는 video-K_DM(수만 token) 이 L2 를 점유해 K_AR 을 축출(eviction)한다. KEELKV 는 **단 하나의 통찰** — "step-invariant read-only 데이터는 명시적으로 on-chip(L2) 에 못 박아야 한다" — 을 edge GPU 의 `cudaAccessPolicyWindow` 메커니즘으로 실현한다.

### 1.2 왜 "edge GPU" 인가 (체급 차이)

datacenter GPU(H100 50MB, B200 큰 L2)는 L2 가 풍부하고, 그쪽 KV 는 decode-time growing-KV 라 거대해 L2-pin 의 동기 자체가 약하다. **edge GPU(AGX Orin L2 4MB)** 에서는 정반대다: (i) L2 가 작아 video-K_DM streaming 이 K_AR 을 쉽게 축출하고, (ii) policy-mode 정적 K_AR 은 작아서(아래 §1.3) **per-layer 단위로 set-aside 에 통째로 들어간다**. 즉 "작은 L2 × 작은 정적 KV × 반복 read" 라는 edge 특유의 3박자가 KEELKV 를 정확히 정당화한다.

### 1.3 K_AR 크기 산정 (본 문서 핵심 grounding)

GQA: `num_kv_heads = 8`, `head_dim = 128`, `num_layers = 36`. per-layer K_AR+V_AR = `tokens × 8 × 128 × 2(K,V) × bytes`.

**policy mode** (conditioning token 산정):
- 텍스트 프롬프트: 예시 입력 `action_policy_av.json` 의 prompt = "You are an autonomous vehicle planning system." ≈ 10 tok + system overhead → **~30 tok** (보수적 상한 ~300 tok 도 병기).
- 관측 ViT 토큰 (**3-view, 540×640 canvas**): Qwen3-VL ViT patch **16×16 + 2×2 spatial merge → effective patch 32×32**. per-view = ⌈540/32⌉×⌈640/32⌉ = 17×20 = **340 tok**, 3-view = **~1,020 tok**. (image_size=480 config 기준이면 per-view ⌈480/32⌉²=15×15=225, 3-view ~675 tok.)
- → policy conditioning ≈ **~1,050 tok** (텍스트 ~30 + 3-view ViT ~1,020). 본 문서 채택값 **~1,050 tok**.

> 📌 **출처**: "3-view / 540×640 canvas" 는 예시 입력 JSON 이 아니라 **tech report §4.2.5 (Cosmos3-Nano-Policy-DROID)** 가 근거다. `action_policy_av.json` 은 `view_point: ego_view` **단일-뷰**·`image_size=480` 의 예시 입력일 뿐이고, 실제 policy 배포 구성(3-view·540×640·32 action @15Hz·~2.1s)은 tech report §4.2.5 에서 명시된다 (step0a 분석 L37·L71-72). ViT patch 16×16+2×2 merge 도 동일 tech report §2.1·step0a L37 근거.

| precision | per-layer K_AR+V_AR (~1,050 tok) | 36-layer 합산 | 3MB set-aside fit? |
|---|---|---|---|
| BF16 (2B) | 1,050×8×128×2×2 = **~4.3 MB** | ~155 MB | per-layer ✗ (3MB 초과) |
| INT8 (1B) | **~2.15 MB** | ~77 MB | **per-layer 여유 fit ✓** |
| **INT4 (0.5B)** | **~1.08 MB** | ~39 MB | **per-layer 2개 동시 상주 ✓** |

> ⚠ 주의: 직전(빈약) 버전 및 phase1' 는 "prompt ~300 tok" 만 가정해 per-layer BF16 1.2MB / INT4 0.3MB 으로 산정했다. **그 산정은 텍스트만 센 것**이고, tech report §4.2.5 의 policy 배포는 **3-view 관측 ViT 토큰이 지배적**(~1,020 tok ≫ 텍스트 30 tok)이다. 따라서 본 문서는 두 산정을 모두 명시한다:
> - **텍스트-only 하한** (prompt 300 tok): BF16 1.2MB / INT4 0.3MB/layer.
> - **관측 포함 현실치** (~1,050 tok, ViT 16×16+2×2): **BF16 ~4.3MB / INT8 ~2.15MB / INT4 ~1.08MB/layer.**
> → BF16 현실치(~4.3MB)는 per-layer 도 3MB set-aside 를 **초과**한다. 반면 **INT8(~2.15MB)·INT4(~1.08MB) 는 둘 다 per-layer fit** 이므로, **Q2 INT4 또는 INT8 가 per-layer fit 의 전제**가 된다(BF16 단독으론 부분 pin 필요). 이것이 Q2↔L3 pair 가 단순 "보너스"가 아니라 **현실 policy workload 에서 L3 의 per-layer 전량 pin 을 가능케 하는 조건**임을 의미한다.
>
> 🔎 **CFG 2벌 (policy guidance=3.0 → CFG ON)**: policy 는 CFG scale 3 (tech report §4.2.5·[`action_policy_robolab_server.md`])이라 cached K_AR 이 **cond/uncond 2벌** 존재한다([`pipeline_cosmos3.diffuse` L1486-1530](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1486-L1530): cond_cache L1496/L1507·uncond_cache L1510/L1521 독립 populate). **단, 코드상 K_AR 은 관측 토큰이 아니라 text-id 토큰만으로 만들어진다** — `cached_kv = self.language_model(text_ids, …)`(L1474)는 `embed_tokens(text_ids)` lookup(L917)뿐이고 ViT/관측 embedding merge 가 없다. cond 는 positive prompt(`cond_ids`), uncond 는 negative prompt(`uncond_ids`)를 각각 tokenize([L1086-1091](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1086-L1091)) → **관측 토큰은 uncond 는 물론 cond K_AR 에도 들어가지 않고**, observation 은 `action_latents`/conditioning latent 로 GEN tower(L1440-1457)에 별도 주입된다. 따라서 두 시나리오를 정직히 병기한다:
> - **(A) tech-report 관측-지배 시나리오** (위 표): K_AR ≈ 1,050 tok 가정 — cond/uncond 모두 동일 크기로 보면 set-aside 압력 **2벌**(INT4 2×1.08=2.16MB, INT8 2×2.15=4.3MB, BF16 2×4.3=8.6MB).
> - **(B) 코드-grounded text-only 시나리오** (vllm-omni 현 구현): K_AR = text-id KV 만 → cond(positive)·uncond(negative, 보통 더 짧음)로 **2벌 중 uncond 는 훨씬 작다**(negative prompt ≪ positive). 이 경우 per-layer 가 텍스트-only 하한(BF16 1.2MB 이하)에 가까워 INT4 2벌도 ~0.6MB.
> 시나리오 A 가 **set-aside fit 상 더 보수적(worst-case)** 이므로 fit 판정은 A 기준으로 한다(아래 §4 M1-③).

**T2I/T2V mode**: conditioning K_AR = 텍스트 프롬프트(짧음, ~512 tok) 면 BF16 per-layer 2.0MB → fit. 단 i2v/긴 prompt 면 K_AR 이 커져 **부분 pin**(앞쪽 token segment 만) 필요.

### 1.4 Q2 와의 pair 구도 (producer-consumer)

```
Q2 ANCHOR (numeric)         L3 KEELKV (placement)
─────────────────────       ──────────────────────
K_AR/V_AR 을 INT4 로  ──────▶ 작아진 K_AR 을 L2 set-aside 에
1-shot 양자화 (Hadamard)      per-layer pin (1.08MB ≤ 3MB)
(producer: footprint↓)        (consumer: BW↓, fit 가능)
```
Q2 는 K_AR 을 **4× 작게** 만들고(BF16 ~4.3MB → INT4 ~1.08MB), L3 는 그 작아진 텐서를 set-aside 에 못 박는다. 둘은 추상화 계층이 달라 비충돌이며(numeric vs HW placement). **전제(강) vs standalone 경계 명시**: policy 현실치 BF16(~4.3MB/layer)는 3MB set-aside 를 초과하므로 **per-layer 전량 pin 은 Q2 INT4(~1.08MB)·INT8(~2.15MB) 를 전제로 한다**(CFG 2벌이면 INT4 2.16MB ≤ 3MB·INT8 4.3MB 초과 → INT4 권장). 다만 Q2 없이도 L3 는 BF16 의 **앞쪽 token segment 부분 pin** 또는 **INT8 1벌 fit** 으로 **standalone 성립**한다(pair 는 fit 여유·이득을 키우는 producer).

---

## 2. Mechanism Summary Table (R70)

| # | Mechanism | 한 줄 | 입력→출력 | source / proposed | Verified-link | 핵심 metric |
|---|---|---|---|---|---|---|
| **M1** | Persistent-Window Pinning | `cached_kv` 를 연속 buffer 로 재배치 후 denoise loop 동안 `accessPolicyWindow{Persisting}` 으로 L2 set-aside 에 pin; video-K_DM 은 `Streaming` demote | `cached_kv` tensor ptr → L2-resident K_AR | **(proposed)** — greenfield, 코드 0건 (pin 대상 anchor 검증됨) | pin 대상 [`cached_kv` L1474-1479](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474-L1479) · 소비 [cat L648-649](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L648-L649) · 순차 [L1499-1507](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1499-L1507) | `l2_tex_hit_rate`(K_AR) ↑, per-step DRAM read ↓ |
| **M2** | Verification & Fallback | RTX Pro 6000 ncu 로 hit-rate 인과검증 → Orin step-latency/EMC proxy; set-aside 미지원·미효과 시 SMEM 수동 캐시 fallback | ncu counters → 판정 → (pin \| SMEM) | **(proposed)** + ncu(source-tool) | — (proposed, 측정 프로토콜·CUDA ext greenfield) | hit-rate Δ, latency Δ, fallback 발동 여부 |

> single-insight 준수: M1 = 통찰 실현, M2 = 통찰 검증+안전망. 새 알고리즘 추가 없음.

---

## 3. GAP + Workload Evidence + Baseline Source

### 3.1 GAP: 반복-read 정적 KV 가 매 step DRAM 왕복 + L2 오염

denoise loop 의 cross-attention 은 매 step·매 layer 에서 `k_all = torch.cat([k_und, k], dim=1)` (L648) 로 **K_AR 을 전량 다시 읽는다**. K_AR 은 step-invariant read-only 인데도 현재:
1. **DRAM 재독출**: per-step·per-layer 마다 K_AR(INT4 ~1.08MB) 을 LPDDR5 에서 L2/SM 으로 다시 fetch.
2. **L2 오염(pollution)**: 같은 attention 에서 video-K_DM(수만 token, 수십 MB) 이 streaming 으로 L2 를 휩쓸어 K_AR 을 축출 → 다음 step 에 K_AR cache-miss 보장. co-location pollution **2.15×** ([arXiv:2501.16909](https://arxiv.org/abs/2501.16909)).

### 3.2 Workload evidence (DRAM 왕복량 계산식)

denoise 동안 K_AR 재독출 총량:
```
read_bytes(K_AR) = N_step × CFG × N_layer × (per-layer K_AR+V_AR bytes)
                   (CFG=2 = cond/uncond 2벌, policy guidance=3.0 → CFG ON)
policy INT4:  4 × 2 × 36 × 1.08 MB  ≈ 311 MB
policy BF16:  4 × 2 × 36 × 4.3 MB   ≈ 1,243 MB
```
AGX Orin LPDDR5 BW ≈ **204.8 GB/s**. 위 read 가 전부 DRAM 왕복이면:
```
policy INT4:  311 MB / 204.8 GB/s   ≈ 1.5 ms (전 denoise)
policy BF16:  1,243 MB / 204.8 GB/s ≈ 6.1 ms
```
L2 hit 으로 이 왕복을 제거하면 그만큼 EMC traffic·step latency 절감 — **절약 상한은 곧 위 read_bytes** 이며, 실제 이득은 `read_bytes(K_AR) / read_bytes(전체)` 비율에 비례(정직한 상한, §6).

> ✅ source verified: `vllm-omni@95d56cf transformer_cosmos3.py#L1474-1479` ([link](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474-L1479)) — `cached_kv_full = self.language_model(...)`; `self.cached_kv = [(k[:,:max_real_len], v[:,:max_real_len]) ...]` (pin 대상 tensor 할당 site).
> ✅ source verified: `vllm-omni@95d56cf transformer_cosmos3.py#L648-649` ([link](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L648-L649)) — `k_all = torch.cat([k_und, k], dim=1)` / `v_all = torch.cat([v_und, v], dim=1)` (매 step·매 layer 반복 read 소비 site).
> ✅ source verified: `vllm-omni@95d56cf transformer_cosmos3.py#L1499-1507` ([link](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1499-L1507)) — `for layer, (k_und, v_und) in zip(self.gen_layers, self.cached_kv): hidden_gen = layer(..., k_und=k_und, v_und=v_und, ...)` (layer 순차 실행 = rolling-window 가능 근거).
> ✅ source verified: `action_policy_av.json` ([link](https://github.com/nvidia/cosmos-framework/blob/main/inputs/omni/action_policy_av.json)) — policy prompt="You are an autonomous vehicle planning system.", `image_size=480`, **`view_point: ego_view` (단일-뷰)**. 이 JSON 은 **예시 입력**일 뿐이며, token 산정에 쓴 **3-view·540×640 canvas 구성은 tech report §4.2.5 (Cosmos3-Nano-Policy-DROID)** 근거다 (step0a 분석 L71-72). ViT 16×16 patch+2×2 merge 도 tech report §2.1·step0a L37.
> ✅ source verified: K_AR 은 **text-id 토큰만**으로 생성 — `cached_kv = self.language_model(text_ids, …)` ([transformer_cosmos3.py#L1474](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474)) 가 `embed_tokens(text_ids)` lookup ([L917](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L917))뿐 (관측 ViT merge 없음). CFG cond/uncond 는 positive/negative prompt tokenize ([pipeline L1086-1091](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1086-L1091)). → §1.3 시나리오 A(tech-report 관측-지배)·B(코드 text-only) 병기 근거.

### 3.3 HW 사실 (본인 재확인)

- **AGX Orin (sm_87, ga10b)**: L2 = **4 MB**, set-aside `l2_max_perst_spc` = **W1-2 에서 실측 확정**(직전 리뷰 추정 ~3MB — 추정치이며 fit 판정 기준은 실측으로 대체). 4MB 중 가용 set-aside 가 3MB 미만이면 INT4 CFG 2벌(2.16MB) 마진이 소진되므로 **실측 우선**. ncu **미지원** → `dram__bytes_read`/`l2_tex_hit_rate` 커널 카운터 사용 불가. BW 는 Nsight Systems + tegrastats(EMC %) + wall-clock proxy.
- **RTX PRO 6000 Blackwell (GB202)**: L2 = **128 MB** (web 확인, [NVIDIA RTX PRO Blackwell GPU Architecture v1.0](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)). ncu 지원 → hit-rate 인과검증을 **여기서** 수행하고, Orin 엔 latency proxy 만 적용. (RTX L2 가 128MB 라 set-aside 만으로도 K_AR 상주가 쉬워 hit-rate 인과를 깨끗이 측정하기 좋으나, **Orin 의 작은 L2 가 진짜 타깃 — RTX 는 인과 증명용 대리 플랫폼**.)
- **PyTorch 노출 경로**: `torch.cuda.Stream` 은 `accessPolicyWindow` attribute 를 **노출하지 않는다**(미노출 확인). 따라서 구현 경로 두 가지:
  - (a) **custom CUDA extension (pybind)**: `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)` 직접 호출 + `cudaDeviceSetLimit` — 권장(정밀 제어).
  - (b) **cupy**: `cupy.cuda.runtime` 로 `cudaStreamSetAttribute` 호출(프로토타입 빠름, dtype 캐스팅 주의).

### 3.4 Greenfield 확인

> ✅ source verified: `vllm-omni@95d56cf` + `vllm@063ce98` + `cosmos-framework@003d66d` 전체에서 `accessPolicyWindow|cudaAccessPolicy|PersistingL2|cudaLimitPersisting|hitRatio|cudaCtxResetPersisting` grep → **유효 0건** (유일한 2건은 vLLM `rust/.../deepseek_v32/fixtures/` 의 wildlife PDF 테스트 텍스트 "lands set aside for wildlife" — 코드 무관). → **완전 greenfield**: L2 persistence 메커니즘이 서빙 스택에 전무.

### 3.5 Baseline Source (R52.1)

| # | Baseline | 무엇 | 한계 / 차별 |
|---|---|---|---|
| (a) | **기본 (no policy window)** | 현 vLLM-Omni `diffuse` — 매 step K_AR DRAM 재독출, L2 정책 없음 | K_AR pollution 무대응 (= GAP) |
| (b) | **read-only 힌트만** (`__ldg`/`const __restrict__`) | kernel 에 read-only 캐시 힌트만 부여, set-aside 없음 | 힌트는 L1/texture-path 우선일 뿐 **L2 상주 보장 못 함** — pollution 시 여전히 축출 |
| (c) | **SMEM 수동 캐싱** (= M2 fallback) | cross-attn kernel 에서 K_AR(k_und) tile 을 SMEM(Orin 192KB/SM) 에 step 시작 1회 stage | SMEM 작아 K_AR 전량 불가 → tile 반복 reload; set-aside 불가 HW 용 fallback |
| (d) | **Q2 INT4 만 (pin 없음)** | K_AR 을 INT4 로만 줄이고 L2-pin 안 함 | footprint↓ 로 DRAM read 절대량은 줄지만 **반복 왕복 자체는 잔존** — L2 hit↑ 없음 (L3 의 순증분이 이 delta) |

**플랫폼 사용 분석 (T2 규칙)**: vLLM / SGLang / llama.cpp / TensorRT(-LLM) 코드베이스에서 `cudaAccessPolicyWindow`/`accessPolicyWindow` 사용 **0건**(공개 코드 grep + 본 repo grep 일치). **이유 추정**: (i) datacenter L2 가 풍부(H100 50MB)해 명시 pin 의 한계이득이 작고, (ii) 그쪽 KV 는 paged decode-time growing-KV 라 거대·동적이어서 정적 윈도우로 못 박을 대상 자체가 없다 — **edge 의 작은·정적 cross-tower KV 와는 다른 체급**. KEELKV 는 이 빈틈(작은 L2 × 정적 KV)을 정확히 겨냥.

> 인접 선행(인용·차별): [Async KV Prefetch→L2 (arXiv:2504.06319)](https://arxiv.org/abs/2504.06319) — KV 를 L2 로 **prefetch**(상주 pin 아님)·decode-time growing-KV·discrete GPU 대상. KEELKV = denoise 전기간 read-only static cross-tower K_AR 을 **per-layer set-aside pin + video-K_DM streaming demote**(Orin sm_87). step0-refresh(2026-06-05) worst-overlap ~45% adjacent, 판정 유지. (인접 참고: [arXiv:2508.13231](https://arxiv.org/abs/2508.13231) dynamic KV placement(DRAM↔HBM tier, L2 아님), [arXiv:2603.28405](https://arxiv.org/abs/2603.28405) EdgeDiT(mobile NPU, Orin L2 무관) — 둘 다 다른 계층.)

---

## 4. 제안 기법

### M1. Persistent-Window Pinning

**① 추가 scheme** — `Cosmos3VFMTransformer.cached_kv`(현재 36개 `(k,v)` 튜플 리스트, 각각 독립 할당) 를 **연속(contiguous) buffer 로 재배치**한 뒤, denoise loop 진입 시 그 buffer 의 base_ptr 에 `accessPolicyWindow{base_ptr, num_bytes, hitRatio≈1.0, hitProp=Persisting, missProp=Streaming}` 를 stream attribute 로 설정. 같은 attention 에서 흐르는 video-K_DM read 는 window 밖이라 `missProp=Streaming` 으로 자동 demote → K_AR 축출 방지. loop 종료 시 reset.

**② 해결 문제** — (3.1) video-K_DM streaming 이 K_AR 을 L2 에서 축출 → 매 step K_AR cache-miss → DRAM 왕복.

**③ 동작 원리 (≥5 step)**:
1. **set-aside 확보**: `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside_bytes)` — Orin 은 `l2_max_perst_spc` 실측값(≤ ~3MB) 상한. 1회(엔진 init).
2. **연속 재배치**: `cached_kv` 를 `pinned_buf[layer][k|v]` 형태의 **단일 연속 텐서**로 copy(reasoner forward 직후 L1479 hook). per-layer stride 고정 → window base/size 계산이 layer index 의 affine 함수.
3. **window 설정**: denoise loop 진입 직전, 현재 set-aside 에 들어갈 layer 군집의 `base_ptr`/`num_bytes` 로 `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)`. **CFG 2벌 반영**: policy 는 guidance=3.0 → CFG ON 이라 cond/uncond K_AR 이 **2벌** 존재한다(§1.3, pipeline L1496-1521). 관측-지배 시나리오 A(INT4 ~1.08MB/layer)에서 3MB set-aside 는 **1-layer 의 cond+uncond 2벌(2×1.08=2.16MB ≤ 3MB)** 을 담는다 — 즉 "2-layer 동시 상주"가 아니라 **"1-layer × cond/uncond 2copy"** 가 정확한 해석이다. INT8(2벌 4.3MB)·BF16(2벌 8.6MB)은 2벌 동시 상주 불가 → INT4 권장(또는 cond 만 pin·uncond 는 streaming). (CFG OFF 가정이면 1벌이라 INT4 2-layer·INT8 1-layer 동시 상주 가능하나, tech report 는 policy CFG=3 → ON 이므로 2벌 기준으로 산정.)
4. **rolling window** (layer-순차 실행 활용, L1499-1507): gen_layer 는 0→35 순차 실행이므로, set-aside 가 전 layer 를 담지 못해도 "현재 layer 의 cond+uncond 2벌"(+여유 시 다음 layer) 만 윈도우에 두고 layer 진행에 따라 base_ptr 을 rolling 으로 갱신(prefetch-like). 정적 schedule 이라 분기 없음.
5. **streaming demote**: video-K_DM 텐서는 window 밖 → `missProp=Streaming`. 추가로 K_DM 전용 stream 에 `Normal/Streaming` 명시(오염 차단 강화).
6. **reset**: loop 종료 시 `cudaCtxResetPersistingL2Cache()` + `cudaStreamSetAttribute` 로 window 해제(다음 request/VAE phase 가 L2 전체 복원).

**④ 기존 실패 이유 + 차별화** — (b) read-only 힌트는 L1/texture path 만 건드려 **L2 상주를 보장 못 함**; pollution 시 축출 동일. prefetch(2504.06319)는 상주가 아니라 미리 끌어옴(다음 step 또 evict). KEELKV 는 **window 가 살아있는 동안 강제 상주** + streaming demote 로 oxidation 차단. **INT4(Q2) 면 ~1.08MB/layer(CFG 2벌=2.16MB)** 라 set-aside 에 layer(cond/uncond 2copy) 단위로 들어가고, 못 채우면 rolling window 로 일반화.

```python
# (proposed) M1: custom CUDA ext (pybind) + python hook  — denoise loop 경계
# ── C++ side (l2pin.cu, pybind) ──────────────────────────────────────────
#   void set_persisting(uintptr_t stream, uintptr_t base, size_t nbytes, float hit) {
#     cudaStreamAttrValue v{};
#     v.accessPolicyWindow.base_ptr  = (void*)base;
#     v.accessPolicyWindow.num_bytes = nbytes;          // ≤ set-aside size
#     v.accessPolicyWindow.hitRatio  = hit;             // ≈1.0
#     v.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
#     v.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
#     cudaStreamSetAttribute((cudaStream_t)stream,
#                            cudaStreamAttributeAccessPolicyWindow, &v); }
#   void reset(uintptr_t stream){ cudaCtxResetPersistingL2Cache();
#     cudaStreamAttrValue z{}; cudaStreamSetAttribute((cudaStream_t)stream,
#                            cudaStreamAttributeAccessPolicyWindow, &z); }
# ── Python side (patch Cosmos3OmniDiffusersPipeline.diffuse) ──────────────
import l2pin, torch
def pin_kv(self):                       # call once after reasoner UND forward
    flat = [torch.cat([k.reshape(-1), v.reshape(-1)]) for k, v in self.transformer.cached_kv]
    self.kv_buf = torch.cat(flat).contiguous()        # 연속 buffer
    self.layer_stride = self.kv_buf.numel() // len(flat)
def diffuse(self, *a, **kw):
    l2pin.set_limit(self.set_aside_bytes)             # cudaDeviceSetLimit (1회)
    s = torch.cuda.current_stream().cuda_stream
    base = self.kv_buf.data_ptr()
    win = min(self.set_aside_bytes, self.kv_buf.element_size()*self.layer_stride*2)  # CFG cond+uncond 2copy of 1 layer (INT4 2.16MB ≤ 3MB)
    l2pin.set_persisting(s, base, win, 1.0)           # ← K_AR pin
    out = self._orig_diffuse(*a, **kw)                # for t in timesteps: ...
    l2pin.reset(s)                                    # ← 해제
    return out
```

> ✅ source verified: pin 대상 = `cached_kv` ([L1474-1479](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474-L1479)), 소비 = `_forward_local` cat ([L648-649](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L648-L649)), 순차 layer = [L1499-1507](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1499-L1507).
> (proposed): 위 pybind ext·hook·rolling-window schedule 전부 greenfield 신규.

**R52.2 표 — M1 baseline 대비**

| 항목 | (a) no-window | (b) `__ldg` 힌트 | (c) SMEM(=M2) | (d) Q2 INT4 only | **M1 (proposed)** |
|---|---|---|---|---|---|
| K_AR L2 상주 보장 | ✗ | ✗(L1 only) | △(tile) | ✗ | **✓ (set-aside)** |
| pollution 차단 | ✗ | ✗ | ✓(SMEM 격리) | ✗ | **✓ (streaming demote)** |
| per-step DRAM 왕복 제거 | ✗ | ✗ | △ | 절대량만↓ | **✓ (hit↑)** |
| HW 요구 | — | — | SMEM | — | set-aside 지원 |

### M2. Verification & Fallback

**① 추가 scheme** — RTX Pro 6000 에서 ncu 로 `l2_tex_hit_rate`·`lts__t_sectors_srcunit_tex_op_read` 를 pin on/off A/B 측정해 **K_AR hit 상승의 인과**를 증명한 뒤, Orin 엔 ncu 가 없으므로 step latency + tegrastats EMC counter 를 proxy 로 적용. set-aside 미지원(실측 `l2_max_perst_spc`=0)·효과 미미(<3%) 시 **SMEM 수동 캐시 fallback**(cross-attn kernel 의 `k_und` 부분만 SMEM tile staging) 으로 자동 전환.

**② 해결 문제** — (i) Orin 의 ncu 부재로 직접 BW counter 못 봄 → RTX 인과 + Orin proxy 의 2-tier honest scope. (ii) set-aside 미지원/협소 HW 일반화.

**③ 동작 원리 (≥5 step)**:
1. **RTX 인과 측정**: `ncu --set full --metrics l2_tex_hit_rate,lts__t_sectors_srcunit_tex_op_read.sum --kernel-name regex:.*attn.* -o pin_on ...` 와 pin off 두 회 → kernel 별 hit-rate delta 추출.
2. **판정**: `Δhit_rate(K_AR kernel) ≥ τ_hit (예 +15%p)` & `Δlts_read_sectors ≤ 0` 면 인과 성립.
3. **Orin proxy**: 동일 pin on/off 를 Orin 에서 step-latency(Nsight Systems wall-clock) + `tegrastats` EMC%·EMC-freq 평균으로 측정 → `Δlatency`, `ΔEMC_traffic_proxy`.
4. **fallback gate**: `l2_max_perst_spc` 실측이 0(=set-aside 미지원) 또는 Orin `Δlatency < 3%` 면 → SMEM staging path 활성.
5. **SMEM path**: cross-attn kernel 에서 step 시작 시 `k_und`(K_AR) tile 을 SMEM 으로 1회 load(static 이라 step 시작 1회로 충분 — FlashAttention 의 매-step reload 와 다름), tile 내 attention 은 SMEM read.

```bash
# (proposed) M2: ncu 명령 + 판정 스크립트
# --- RTX Pro 6000: pin on/off A/B ---
ncu --target-processes all --set full \
    --metrics l2_tex_hit_rate,lts__t_sectors_srcunit_tex_op_read.sum \
    --kernel-name-base demangled --kernel-name "regex:.*Cosmos3CrossAttn.*" \
    -o keelkv_pin_on  python serve_policy.py --keelkv-pin 1 --steps 4
ncu (... 동일 ...)  -o keelkv_pin_off python serve_policy.py --keelkv-pin 0 --steps 4
# --- 판정 (ncu --import → csv) ---
python - <<'PY'
import pandas as pd
on  = pd.read_csv("keelkv_pin_on.csv");  off = pd.read_csv("keelkv_pin_off.csv")
h_on  = on [on.Kernel.str.contains("CrossAttn")]["l2_tex_hit_rate"].mean()
h_off = off[off.Kernel.str.contains("CrossAttn")]["l2_tex_hit_rate"].mean()
r_on  = on [on.Kernel.str.contains("CrossAttn")]["lts__t_sectors_srcunit_tex_op_read.sum"].sum()
r_off = off[off.Kernel.str.contains("CrossAttn")]["lts__t_sectors_srcunit_tex_op_read.sum"].sum()
print(f"Δhit={h_on-h_off:+.1f}%p  read_sectors {r_off}->{r_on} ({(r_on-r_off)/r_off:+.1%})")
print("CAUSAL-PASS" if (h_on-h_off)>=15 and r_on<=r_off else "FALLBACK->SMEM")
PY
```

> (proposed): ncu A/B 프로토콜·fallback gate·SMEM staging 전부 신규. ncu 자체는 source-tool(검증된 NVIDIA 도구).

**R52.2 표 — M2 측정 매트릭스**

| 플랫폼 | ncu | 측정 대상 | 역할 |
|---|---|---|---|
| RTX Pro 6000 (128MB L2) | ✓ | `l2_tex_hit_rate`, read sectors | **인과 증명** (hit↑→read↓) |
| AGX Orin (4MB L2) | ✗ | step latency, EMC%/freq (tegrastats) | **proxy 적용** (honest scope) |

### 의존성 그래프 (Q2 의존 = optional)

```
        ┌─────────────┐  producer(전량 pin 전제)  ┌──────────────┐
        │ Q2 ANCHOR   │ ───────────────────────▶ │   M1 pin     │
        │ INT4 K_AR   │ K_AR ~4.3MB→~1.08MB(fit↑) │ (Persisting) │
        └─────────────┘                           └──────┬───────┘
                                                         │ verify
   standalone: BF16 앞쪽 segment 부분 pin 또는           ▼
   INT8 1벌 fit 으로도 M1 성립 (Q2 없이)             ┌──────────────┐
                                                  │   M2 verify  │
                                                  │  +SMEM fallbk│
                                                  └──────────────┘
```
- **전제(강) vs standalone(optional) 경계** (§1.4 와 동일 문구): policy 현실치 BF16(~4.3MB/layer)는 set-aside(3MB) 초과 → **per-layer 전량 pin 은 Q2 INT4(~1.08MB) 를 사실상 전제로 한다**(CFG 2벌이면 INT4 2.16MB ≤ 3MB; INT8 2벌 4.3MB·BF16 2벌 8.6MB 는 전량 2벌 불가). 다만 **standalone 성립**도 동시에 참이다 — BF16 **앞쪽 token segment 부분 pin**, 또는 **INT8 1벌 fit**(cond 만 pin·uncond streaming), 또는 CFG OFF 모드면 INT8 per-layer 여유 fit. 요컨대 **per-layer 전량 pin = INT4 전제(강), 부분 pin·INT8·1벌 = standalone(약)** 으로 조건 분기된다.

---

## 5. 평가·실험 플랜 (7요소)

### 5.1 HW
- **primary**: AGX Orin 64GB (sm_87, L2 4MB, set-aside `l2_max_perst_spc` 실측) — 진짜 타깃.
- **검증**: RTX Pro 6000 (GB202, L2 128MB, ncu 지원) — hit-rate 인과 증명용 대리.

### 5.2 모델 / workload
- **1순위 Policy-DROID** (`model_mode=policy`): K_AR 가 작아(~1,050 tok, INT4 ~1.08MB/layer; CFG 2벌 = 2.16MB ≤ 3MB) per-layer fit — 가장 깨끗한 케이스. workload = policy 연속 chunk(action_chunk_size=60).
- **2순위 T2I/T2V**: K_AR 가 클 수 있어(i2v 긴 conditioning) **부분 pin**. workload = T2I 배치.

### 5.3 tools
custom CUDA ext(pybind, `cudaStreamSetAttribute`/`cudaDeviceSetLimit`) · RTX ncu(`l2_tex_hit_rate`, `lts__t_sectors_srcunit_tex_op_read`) · Orin Nsight Systems + tegrastats(EMC%/freq) · cupy(프로토타입).

### 5.4 ablation
`pin ∈ {off, K only, K+V, +INT4} × window-크기 sweep {1-layer, 2-layer, rolling}`.
- metrics: **step latency**(Orin), **DRAM read bytes**(RTX ncu), **J/chunk**(Orin tegrastats power×time).

### 5.5 주차별 표 (~9주)

| 주차 | 작업 | 파일/산출물 | 완료 판정 |
|---|---|---|---|
| **W1-2** | baseline(a-d) 재현 + K_AR 크기 실측 산정 | `bench/policy_baseline.py`, `notes/kar_size.md` | policy/T2I per-layer K_AR bytes 실측 확정, baseline latency 표 |
| **W3-4** | pybind ext(`l2pin.cu`) + 연속 재배치 hook + diffuse window 삽입 | `ext/l2pin/`, `pipeline_cosmos3.diffuse` patch | pin on/off 토글 동작, 수치 무변(품질 Δ=0) |
| **W5-6** | RTX Pro 6000 ncu 인과 측정 (hit-rate A/B) | `results/rtx_ncu_*.csv`, 판정 스크립트 | `Δl2_tex_hit_rate ≥ +15%p` & read sectors↓ → CAUSAL-PASS |
| **W7-8** | Orin 적용 + latency/EMC proxy + INT4(Q2) pair + ablation sweep | `results/orin_proxy.csv`, sweep 표 | Orin Δlatency·ΔEMC 측정, window-크기 최적점 |
| **W9** | rolling-window·부분 pin 일반화 + SMEM fallback(M2) + 정리 | `ext/smem_stage.cu`, 최종 plot | fallback gate 검증, T2V scope-gate 문서화 |

### 5.6 preliminary 4단계

1. **reproduction**: 현 vLLM-Omni policy diffuse 재현, per-step latency·step 수(N=4×CFG2) 확인. 도구: Nsight Systems. 기대: baseline latency 분포.
2. **attribution**: `cached_kv` read 가 cross-attn kernel BW 에서 차지하는 비중을 RTX ncu(`lts__t_sectors_srcunit_tex_op_read` 의 K_AR kernel 분율) 로 측정. 도구: ncu. 기대: K_AR read 비중 x% (이득 상한의 근거).
3. **roofline**: 절약 가능 bytes 상한 = §3.2 의 `read_bytes(K_AR)`(INT4 ~311MB / BF16 ~1,243MB, CFG 2벌 포함) 계산 + LPDDR5 204.8GB/s 대비 latency 상한. 도구: 계산. 기대: 절약 상한 ms.
4. **micro-benchmark**: 합성 tensor(~1.08MB, CFG 2벌이면 2copy) 반복-read 커널에 window on/off A/B (K_DM streaming 동시 부하 주입). 도구: ncu(RTX) + 합성 kernel. 명령: `ncu --metrics l2_tex_hit_rate ./synth_repeat_read --pin {0,1}`. 기대값: pin on 시 hit-rate ~100%, read sectors ≈ 1회분.

---

## 6. 예상 효과 / Risk / Tier / Scoring / Decision-tree

### 6.1 예상 효과 (보수 + 정직한 상한)

- **policy-mode step latency 5-12%↓** + **EMC traffic 감소** — 단 **K_AR 비중 조건부**. 효과는 `read_bytes(K_AR)/read_bytes(전체 cross-attn read)` 비율에 **선형 비례**.
- 정직한 상한: video-K_DM(수만 token) 이 cross-attn read 를 지배하면 K_AR 비중이 낮아 이득 상한도 낮다. **상한 = K_AR read 가 전체의 x% 일 때 step latency 절감 ≤ x%**. policy(K_DM 작음)에서 비중 높음 → 이득 큼; T2V(K_DM 큼)에서 비중 낮음 → 이득 작지만 pollution 회피 가치는 반대로 큼(축출 빈도↑).

### 6.2 Risk + 완화

| Risk | 완화 |
|---|---|
| `l2_max_perst_spc` 실측 0 (set-aside 미지원) | → **SMEM fallback (M2-c)** 자동 전환 |
| 효과 < 3% (K_AR 비중 작음) | → **측정-letter(IEEE CAL) 전환** — "edge 정적 cross-tower KV 의 L2 pin 한계 특성화" negative-but-publishable |
| BF16 K_AR ~4.3MB/layer > 3MB set-aside (CFG 2벌이면 8.6MB) | → Q2 INT4(~1.08MB, 2벌 2.16MB)/INT8(~2.15MB, 1벌) fit, 또는 BF16 앞쪽 segment 부분 pin |
| `l2_max_perst_spc` 실측이 3MB 미만 → INT4 2벌(2.16MB) 마진 소진 | → **W1-2 실측 우선**(추정치 의존 제거); 마진 부족 시 cond 만 pin·uncond streaming 으로 1벌(1.08MB) 운용 |
| ncu Orin 미지원 | → RTX 인과 + Orin latency/EMC proxy (honest scope 명시) |
| window reset 누락 → 다음 phase L2 오염 | → `diffuse` finally 블록 + VAE phase 진입 시 reset 강제 |

### 6.3 R52.4 Tier-A/B/C

- **Tier-A (확실)**: greenfield 확인(코드 0건), `cached_kv`/cat anchor, K_AR INT4 ~1.08MB/layer(CFG 2벌 2.16MB) ≤ 3MB per-layer fit(관측-지배 시나리오 A), CUDA set-aside API 존재.
- **Tier-B (개연)**: RTX hit-rate 상승의 인과(set-aside 동작 사실에 근거), policy K_AR 비중이 높아 이득 material 할 것.
- **Tier-C (추측)**: Orin latency 5-12%↓ 수치(proxy 측정 전), T2V 부분 pin 의 실효, rolling-window 일반화 이득.

### 6.4 Tier-1 승격 조건

(i) **rolling-window 일반화**가 임의 K_AR 크기(전 36-layer)로 확장되어 set-aside 크기에 무관한 schedule 로 정립 + (ii) **멀티 워크로드**(policy + T2I + T2V) 에서 일관된 이득 + (iii) Orin 실측 ≥10% latency↓ 가 동시 충족되면 DAC full → MLSys/ASPLOS 급 systems contribution 으로 승격.

### 6.5 Scoring

base **6.6** (phase2' avg). 조정: **★ ai-impl 8** (greenfield·~9wk·anchor 견고·표준 CUDA API), **▼ novelty 6** (L2-pin 발상은 2504.06319 인접; static cross-tower 한정이 차별). 종합 유지 6.6.

### 6.6 Decision-tree 분기

```
W5-6 RTX ncu A/B
 ├─ CAUSAL-PASS (Δhit≥+15%p, read↓)
 │    └─ W7-8 Orin proxy
 │         ├─ Δlatency ≥ 5%  → DAC/DATE full (정상 경로)
 │         │     └─ (rolling+멀티WL+≥10%) → Tier-1 승격 시도
 │         └─ Δlatency < 3%  → 측정-letter(CAL) "한계 특성화"
 └─ CAUSAL-FAIL / set-aside 미지원
      └─ SMEM fallback(M2-c) 측정 → 이득 시 fallback-중심 letter
                                  → 무이득 시 negative-characterization 보고
```
```
K_AR fit 분기 (관측-지배 시나리오 A + CFG 2벌 반영, ~1,050 tok)
 ├─ INT4 ~1.08MB → 2벌 2.16MB ≤ 3MB → 1-layer cond+uncond pin  ← Q2 pair 권장
 ├─ INT8 ~2.15MB → 1벌 fit·여유; 2벌 4.3MB > 3MB → cond-only pin(standalone)
 └─ BF16 ~4.3MB  → 1벌도 초과 → 앞쪽 token segment 부분 pin 또는 Q2 의존
```

---

### Appendix — anchor 재확인 요약 (R72.3)

| anchor | 파일#라인 | 확인 |
|---|---|---|
| pin 대상 tensor 할당 | [`transformer_cosmos3.py#L1474-1479`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474-L1479) | ✅ `cached_kv` 할당 |
| 반복 read 소비 | [`transformer_cosmos3.py#L648-649`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L648-L649) | ✅ `cat([k_und,k])` |
| layer 순차 실행(rolling 근거) | [`transformer_cosmos3.py#L1499-1507`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1499-L1507) | ✅ for-zip |
| GQA head/dim | [`transformer_cosmos3.py#L567-578`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L567-L578),[`L625-629`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L625-L629) | ✅ num_kv_heads/head_dim |
| policy 예시 입력 (단일-뷰) | [`inputs/omni/action_policy_av.json`](https://github.com/nvidia/cosmos-framework/blob/main/inputs/omni/action_policy_av.json) | ✅ prompt·`image_size=480`·`ego_view`(단일) |
| 3-view·540×640·16×16+2×2 patch | tech report §4.2.5·§2.1 (step0a L37·L71-72) | ✅ token 산정 실근거 |
| K_AR = text-id KV (관측 merge 無) | [`transformer_cosmos3.py#L1474`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474)·[L917](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L917) | ✅ `embed_tokens(text_ids)` only |
| CFG cond/uncond 2벌 | [`pipeline_cosmos3.py#L1486-1530`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1486-L1530) | ✅ cond_cache/uncond_cache 독립 |
| greenfield | vllm-omni+vllm+cosmos-framework grep | ✅ 유효 0건 |
| RTX L2 128MB | NVIDIA RTX PRO Blackwell Arch v1.0 | ✅ web 확인 |
