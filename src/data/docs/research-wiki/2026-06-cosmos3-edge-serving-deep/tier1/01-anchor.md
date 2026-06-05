# Q2 ANCHOR (Tier-1 #1, 직전 세션 avg 7.6) — DEEP

**ANCHOR: One-Shot Quantization of Static Conditioning KV with a Step-Dependent Flow-Matching Denoising Error Bound for Dual-Tower Omnimodal Models**

- **Venue**: MLSys 2027 주(主) / ICML 2027 차선
- **G4 소유**: static conditioning-KV 의 **numeric(quant + bound)** 전권. A2 Keystone 이 layout/CFG-dedup, L3 KEELKV 가 L2 placement 를 분담 (no double-claiming).
- **Metaphor**: 닻(anchor) = denoising 전 기간 고정된 conditioning KV. 한 번 정밀하게 내려두면(1-shot quant) 모든 step 이 그 위에서 흔들림 없이 재사용한다.

---

> ## 📖 약어/핵심 용어 풀이
>
> - **K_AR / V_AR (conditioning KV)**: reasoner(understanding) tower 가 텍스트/멀티모달 프롬프트를 1회 forward 해 만든 Key/Value. 코드상 `Cosmos3VFMTransformer.cached_kv` (per-layer `(K,V)` list). denoising N step × CFG 2 pass 내내 read-only. **본 문서의 양자화 대상**.
> - **K_DM / V_DM**: generator(diffusion) tower 가 매 step noisy latent 로부터 새로 만드는 Key/Value (video/action/sound 토큰). step 마다 바뀜 → 양자화 대상 아님.
> - **MoT (Mixture-of-Transformers)**: modality-disjoint dual-tower. reasoner(und)·generator(gen) 가 별도 weight 를 갖되 cross-attention 으로 결합. cf. [arXiv:2411.04996](https://arxiv.org/abs/2411.04996).
> - **CFG (Classifier-Free Guidance)**: cond(real prompt)·uncond(null prompt) 두 번 forward 해 `noise = uncond + w·(cond−uncond)` 로 결합. K_AR cache 가 cond/uncond 별도 = **2벌**. **policy(DROID) 는 guidance scale 3.0 + CFG-parallelism → CFG ON → cached K_AR 2벌 (cond/uncond)** (tech report §4.2.5 L2007: "guidance scale of 3"; `action_policy_robolab_server.md#L22,L53`: guidance 3.0, 4 denoising steps). guidance_scale=1.0 은 forward/inverse-dynamics·padding pass 용이며 policy 모드가 아님 → policy 는 항상 CFG ON / 2벌.
> - **Flow matching / rectified flow**: noise→data 를 velocity field `v_θ(x_t,t)` 로 ODE 적분. Cosmos3 generator 의 학습 objective (velocity-MSE). cf. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
> - **N (flow steps / denoising steps)**: ODE 적분 step 수. T2V N=50, audiovisual N=35, **policy(action) N=4** (cookbook 확정값). bound 가 N 에 명시 의존.
> - **RTN (Round-To-Nearest)**: 가장 단순한 PTQ — scale 로 나눠 반올림. baseline.
> - **per-channel asymmetric quant**: channel(=feature dim)마다 scale·zero-point 를 따로 두는 비대칭 양자화. `q = round((x−z)/s)`. K_AR 의 channel-wise outlier 흡수에 유리.
> - **Hadamard rotation**: orthogonal Hadamard 행렬 H 로 `x→xH` 회전해 outlier energy 를 channel 전체로 spread → quant error↓. inference 시 `(xH)(H^T y)=xy` 로 정확 복원. cf. QuaRot [arXiv:2404.00456](https://arxiv.org/abs/2404.00456).
> - **MRoPE (Multi-dimensional RoPE)**: Qwen3-VL 의 3D(temporal/height/width) rotary position embedding. K 에 RoPE 가 적용된 **후** Hadamard 를 걸어야 위치정보를 깨지 않음 (cosmos3 cached K 는 post-RoPE 로 저장됨 — §4.M1 verified).
> - **Lipschitz 상수 (L)**: `‖f(x)−f(y)‖ ≤ L·‖x−y‖` 의 최소 L. 본 bound 에서 softmax-Jacobian(L_softmax)·velocity-net(L_v) 두 곳에 사용.
> - **power iteration**: 행렬 J 의 최대 특이값(spectral norm)을 `v←J^TJv/‖·‖` 반복으로 추정. velocity-net Jacobian 의 `L_v^{(n)}` 측정 도구.
> - **GQA (Grouped-Query Attention)**: KV head 수 < Q head 수. cosmos3 reasoner KV = 8 head × 128 dim, 36 gen layer, K+V (per-layer K_AR footprint 산정 근거: policy ~1,050 tok 기준 per-layer BF16 ~4.3MB / INT4 ~1.08MB).
> - **1-shot quant**: 양자화를 추론 전 1회만 수행하고 모든 step 이 재사용 → 비싼 calibration(rotation/clipping search)을 amortize. cf. online per-token quant(KIVI)와 정반대.

---

## 1. 개요

- **한 줄 정의**: Cosmos3 의 정적·1회·read-only 인 K_AR/V_AR 을 추론 전 1-shot 으로 저비트 양자화하되, denoising 출력 오차를 flow-step 수 N 에 명시 의존하는 **닫힌(closed-form) error bound** 로 보증해 layer/modality 별 bit 를 자동 선택한다.
- **metaphor ↔ mechanism 대응**:
  - 닻을 한 번 정밀하게 내림 ↔ **M1**: 추론 전 K_AR/V_AR 에 Hadamard+MSE-clipping+per-channel quant 를 1회 적용 (`cached_kv` 직후 hook).
  - 닻이 항해 내내 고정 ↔ **M2**: 그 고정성(read-only, step-invariant)이 있어야 출력 오차를 N step 으로 closed-form 전파 가능 (online KV 엔 정의 불가).
  - 닻의 강도를 바닥 깊이에 맞춰 선택 ↔ **M3**: bound 를 실측 L_v 로 채워 bit 를 layer 별 자동 선택·검증.
- **RQ**:
  - (RQ1) static cross-tower conditioning-KV 의 1-shot 저비트 양자화가 어느 bit 까지 품질을 보존하는가?
  - (RQ2) KV quant error → denoising 출력 error 를 flow-step N 에 명시 의존하는 closed-form bound 로 유도 가능한가, 그리고 그 bound 는 tight 한가?
  - (RQ3) policy N=4 가 T2V N=50 보다 저비트에 관대하다는 N-의존 예측이 RoboLab 에서 실증되는가?
- **G4 소유 구도**: Q2 = 수치(bit/bound) 단독 소유. A2 = runtime layout + CFG K_AR-block dedup (quant off 무손실). L3 = on-chip L2 set-aside placement. 세 축이 같은 자원(static K_AR)을 노리되 추상화 계층이 달라 producer-consumer 관계.
  - **pair 격상 (L3 KEELKV 와 동기화)**: policy 현실 K_AR(3-view 관측 지배, ~1,050 tok)의 per-layer footprint = **BF16 ~4.3MB > 3MB L2 set-aside** 초과. 따라서 L3 의 per-layer pin 은 **Q2 INT4(1.08MB)를 사실상 전제**로 한다 (단순 보너스 아님 — KEELKV 가 "Q2 INT4 = L2 fit 의 전제"로 격상). CFG ON(policy guidance 3.0)이므로 cond/uncond 2벌 = **2×1.08 = 2.16MB ≤ 3MB** 로 INT4 2벌 동시 상주가 fit 의 조건.
  - **전제(강) vs optional(standalone) 경계 (양쪽 동일 문구)**: per-layer 전량 pin 은 **INT4 전제**; INT8(2벌 4.3MB)은 경계 초과 → 부분 pin, BF16 은 부분 pin 으로만 **L3 standalone** 성립. Q2 의존은 "전량 pin = 강전제 / 부분 pin = optional" 의 조건 분기로 KEELKV §1.4·§4 와 일치.

---

## 2. Mechanism Summary Table (R70, 7열)

| M# | 수정 대상 `file#L` | 유형 | 왜 | Verified link | 타당성 1줄 | 상세 anchor |
|---|---|---|---|---|---|---|
| **M1** | `transformer_cosmos3.py#L1474-1479` (`cached_kv` 할당 직후) | hook + 신규 wrapper class | static K_AR/V_AR 생성 site → 여기서 1-shot quantize | [L1474-1479](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474-L1479) | cache 가 1회만 채워지므로(`if self.cached_kv is None`) quantize 도 1회 | §4.M1 |
| **M1** | `transformer_cosmos3.py#L648-649` `_forward_local` (`cat([k_und,k])`) | hook (dequant) | cached K/V 가 매 step 소비되는 site → dequant 삽입 | [L648-L649](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L648-L649) | softmax 입력 직전 = quant error 가 logit 에 작용하는 정확한 지점 | §4.M1 |
| **M1** | `transformer_cosmos3.py#L685-680` `_forward_sp` (`joint_key/value`) | hook (dequant) | SP 경로의 cached K/V 소비 site (TP/Ulysses) | [L676-L679](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L656-L679) | local·SP 두 소비 경로 모두 dequant 필요 (누락 시 SP 무손상 안 됨) | §4.M1 |
| **M2** | `transformer_cosmos3.py#L628` cross-attn `softmax_scale=1/√d` | 신규 분석 모듈 (offline) | bound 의 `L_softmax≤‖Q_DM‖/√d` 가 이 scale 에 의존 | [softmax_scale](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L628) | 코드의 scale 이 bound 상수와 1:1 → 가정 검증 가능 | §4.M2 |
| **M2** | `pipeline_cosmos3.py#L1492` diffuse step 루프 | 신규 분석 모듈 (offline) | step n 별 Δ_n·L_v^{(n)} 합산이 bound 의 N-의존 핵심 | [L1492](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1492) | 실제 step 루프 구조가 bound 의 Σ_{n=1}^N 와 동형 | §4.M2 |
| **M3** | `quantization/component_config.py#L63-73` `resolve()` | 수정 (config 플러밍) | layer/component 별 bit 를 bound 가 선택 → 여기로 주입 | [L63-L73](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/quantization/component_config.py#L63-L73) | longest-prefix match 가 이미 per-tower 라우팅 지원 → KV 별 bit 확장 자연 | §4.M3 |
| **M3** | `transformer_cosmos3.py#L1281-1283` `reset_cache()` | hook (bit-selection 진입점) | request 시작 시 cache reset → 이때 bit 결정·calib trigger | [L1281-L1283](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1281-L1283) | reset 이 모든 request 의 cache 생애 시작 → bit-policy 적용 lifecycle hook | §4.M3 |

---

## 3. 기존 연구 한계 · GAP

### 3.1 GAP (정량)

- **decode-time KV quant 과 정반대 가정**:
  - KIVI([arXiv:2402.02750](https://arxiv.org/abs/2402.02750))·KVQuant([arXiv:2401.18079](https://arxiv.org/abs/2401.18079)) = online·매 토큰 cheap quant + **growing KV** 가정.
  - Cosmos3 K_AR/V_AR = **정적·1회·read-only** (verified: `if self.cached_kv is None:` 로 1회만 생성, `reset_cache()` 까지 불변).
  - 함의 (i): 비싼 1-shot 양자화(Hadamard matmul + MSE-clipping search)를 **N step × CFG 2 pass 에 걸쳐 amortize** 가능 — online 엔 매 토큰 비용이라 불가.
  - 함의 (ii): step 개념이 online 엔 없음 → **N-의존 closed-form denoising error bound 자체가 정의 불가**. 이 static-KV 특화 bound 가 문헌 전반에 부재.
- **video-diffusion KV quant 실증은 있으나 bound 부재**:
  - QuantKeys([arXiv:2605.26266](https://arxiv.org/abs/2605.26266)) = video-diffusion KV quant + Jensen-bias 보정, 단 **autoregressive self-forcing growing-KV** 대상·순수 실증.
  - 33-Method study([arXiv:2603.27469](https://arxiv.org/abs/2603.27469)) = self-forcing video KV 33종 실증, **closed-form/denoising/flow-step bound 일절 없음** (WebFetch 확인).
- **flow-matching 양자화 인접 선행(신규 must-cite)**:
  - **[arXiv:2511.11418](https://arxiv.org/abs/2511.11418) "Low-Bit, High-Fidelity: Optimal Transport Quantization for Flow Matching"** — 2-Wasserstein 최소화 PTQ. 단 (a) **weight 양자화**(KV 아님), (b) "quant error 가 integration step 통해 비선형 전파"를 *정성* 언급할 뿐 **step-수 의존 closed-form bound 없음**, (c) conditioning-KV 무관. overlap ~30%. → ANCHOR 의 N-의존 닫힌 bound 차별축 무손상. related-work 에 인용.

### 3.2 Workload evidence (수치)

- Eq.8 (tech report line 784): `O_DM = Attn_full(Q_DM, [K_AR;K_DM], [V_AR;V_DM])` = 단일 softmax over concat. K_AR/V_AR 은 모든 step×CFG read-only. (코드 verified: `_forward_local#L648 k_all=torch.cat([k_und,k])`.)
- modality inflation: 3,715 token conditioning → prefill **95.78J / 278.26ms** ([arXiv:2512.22695](https://arxiv.org/abs/2512.22695)); decode/denoise memory-bound ([arXiv:2512.01644](https://arxiv.org/abs/2512.01644)).
- T2V-720p Nano 1-GPU H100 ~286-297s (tech report Fig.16) → edge LPDDR5 에서 KV BW 가 step 마다 반복 read.
- per-layer K_AR/V_AR footprint (policy, **~1,050 tok** = 3-view 관측 ViT 토큰 지배[16×16 patch + 2×2 merge → effective 32×32, 540×640 canvas: per-view ⌈540/32⌉×⌈640/32⌉=17×20=340 tok, 3-view 1,020 + text ≈ 1,050; tech report §4.2.5], GQA 8×128, K+V): **BF16 ~4.3MB / INT8 ~2.15MB / INT4 ~1.08MB** — bit 가 직접 BW·footprint 비례 절감. (구 "~300 tok" 은 텍스트만 센 과소산정 — policy conditioning 은 관측 ViT 토큰이 텍스트 ~10-30 tok 을 압도.) policy 는 **CFG ON(guidance 3.0) → cond/uncond 2벌** 이므로 residency·BW 모두 ×2.
- Cosmos3-Nano **BF16-only** (HF model card) → quant path = reference Diffusers / vllm-omni `ComponentQuantizationConfig` 확장 (비공식 정밀도 신규 도입 명시).

### 3.3 Baseline Source (R52.1)

| # | Baseline | venue | repo / branch | 실행 진입점 | 예상 실행 명령 |
|---|---|---|---|---|---|
| (a) | **FP16/BF16 KV 상한** | — | vllm-omni @95d56cf | `pipeline_cosmos3.diffuse` (quant off) | `vllm-omni serve nvidia/Cosmos3-Nano` (기본 BF16) |
| (b) | **KIVI INT4** | ICML'24 | `github.com/jy-yuan/KIVI` @main | per-token online quant 을 K_AR 에 강제 적용 | KIVI `quant_kv` 를 `cached_kv` 텐서에 1회 호출(online overhead 그대로) |
| (c) | **KVQuant** | NeurIPS'23 | `github.com/SqueezeAILab/KVQuant` @main | per-channel + dense-and-sparse | KVQuant calibration → K_AR per-channel INT4 |
| (d) | **naive RTN INT4** | — | 자체 구현 | `cached_kv` 직후 round-to-nearest | scale = max(|x|)/7, `round(x/scale)` |
| (e) | **QuantKeys** | (arXiv) | [arXiv:2605.26266](https://arxiv.org/abs/2605.26266) repo | Jensen-bias 보정 KV quant | self-forcing 가정 제거 후 K_AR 에 bias-corrected INT4 |
| (rel) | **OT-Quant flow matching** | (arXiv) | [arXiv:2511.11418](https://arxiv.org/abs/2511.11418) | weight-quant (related-work only) | 인용 — KV 비교 대상 아님, bound 부재 차별 |

> **차별 (1줄)**: (b)(c)는 growing-KV·online 가정이라 1-shot amortize 불가, (e)는 self-forcing growing-KV·**bound 부재**. ANCHOR 만 static prefix 특화 1-shot Hadamard amortize + N-의존 closed-form bound.

---

## 4. 제안 기법

### M1. Hadamard-rotated 1-shot K_AR/V_AR quantization

- **① 추가 scheme (실제 anchor)**:
  - `transformer_cosmos3.py#L1479` `self.cached_kv = [(k[:,:max_real_len], v[:,:max_real_len]) for k,v in cached_kv_full]` **직후**에 `QuantizedKVCache` wrapper 로 감싼다 (≈40-60 LoC 신규 클래스 + L1479 1줄 수정).
  - `QuantizedKVCache` fields: `k_q: int8/int4-packed tensor [B,S,H_kv,D]`, `v_q`, `k_scale/k_zp: [H_kv, D]` (per-channel asymmetric), `v_scale/v_zp`, `had_K: [D,D] orthogonal Hadamard`, `had_V: [D,D]`, `bits: int`. dtype: scale/zp = bf16, packed = uint8 (INT4 = 2 elem/byte).
  - dequant hook: `_forward_local#L648` `k_all = torch.cat([k_und, k], dim=1)` 의 `k_und` 가 `QuantizedKVCache.dequant_K()` 호출 결과가 되도록 (≈10 LoC). SP 경로 `_forward_sp#L676` 의 `joint_key=k_und` 도 동일 dequant 적용 (누락 시 SP 무손상 깨짐 — verified 두 경로 분기).
- **② 해결 문제 (workload 1:1)**: K_AR 이 정적이라 Hadamard matmul(D×D=128×128 per head per layer)을 **1회**만 치르고 N=4~50 step × CFG 2 pass 에 amortize → online(KIVI)이라면 매 step 재계산되어 절대 불가. policy footprint ~4.3MB(BF16)→~1.08MB(INT4) per layer (~1,050 tok 기준), CFG ON 이라 cond/uncond 2벌 모두에 동일 1-shot quant 적용.
- **③ 동작 원리 (step-by-step)**:
  1. **freeze**: `if self.cached_kv is None:` 분기(L1460)에서 reasoner `language_model(text_ids, freqs_und)`(L1474) 가 per-layer `(K,V)` list 반환. K 는 **post-norm·post-RoPE** (verified: `Cosmos3CausalAttention.forward` L550 `return ..., k, v` 가 `_apply_rotary_pos_emb` 이후) → Hadamard 는 RoPE 뒤에 적용되어야 위치정보 보존(MRoPE 호환). 자료구조: list[(K,V)], len = num gen layers.
  2. **rotation**: K←`K @ had_K`, V←`V @ had_V` (had = normalized Hadamard, `had @ had^T = I`). V 는 RoPE 무관이라 자유 rotation, K 는 post-RoPE 라 rotation 후에도 cross-attn 의 `q·k^T` 에서 `q' = q @ had_K` 로 대칭 적용해 정확 복원 (`(q·had)(k·had)^T = q·k^T`). parameter: had_K size = head_dim(128).
  3. **clipping**: per-channel(=D axis) MSE-optimal clipping range `[c_min, c_max]` 를 grid search(예: 100-point, [0.5,1.0]×max). MSE = `E[(x − dequant(quant(x;c)))²]`. range: clip ratio ∈ [0.7, 1.0].
  4. **quantize**: per-channel asymmetric `s = (c_max−c_min)/(2^b−1)`, `z = round(−c_min/s)`, `q = clamp(round(x/s)+z, 0, 2^b−1)`. bit b ∈ {8,6,4,3,2} 전 sweep. INT4 packing = 2 nibble/byte.
  5. **store & amortize**: `QuantizedKVCache` 저장. 이후 모든 step 의 `_forward_local`/`_forward_sp` 가 dequant(`x = (q−z)*s`) → unrotate(`@ had^T`)는 q 대칭 적용으로 생략 가능(K), V 는 `@ had_V^T` 한 번. bound-tightness plot 산출(필수 artifact).
- **④ 기존 실패 이유 + 차별화**:
  - KIVI/KVQuant = online decode-KV → 매 토큰 cheap quant 만 가능, Hadamard 같은 비싼 rotation 은 throughput 파괴라 금지. ANCHOR = static prefix → 1회 amortize 라 Hadamard 자유.
  - QuantKeys = self-forcing growing-KV + Jensen-bias 보정만, **rotation 기반 outlier spread 부재 + bound 부재**.

```python
# M1: 1-shot quantize hook  (transformer_cosmos3.py, after L1479)
# self.cached_kv : list[(K,V)] per gen layer, K/V are post-norm post-RoPE  (verified L550)
def quantize_cached_kv(self, bits_per_layer):
    qkv = []
    for layer_idx, (k, v) in enumerate(self.cached_kv):          # k,v: [B,S,H_kv,D]
        b = bits_per_layer[layer_idx]                            # from M3 bound-selection
        had_K, had_V = hadamard(self.head_dim), hadamard(self.head_dim)
        k_rot = k @ had_K                                        # RoPE already applied -> safe
        v_rot = v @ had_V                                        # V RoPE-free -> free rotation
        k_q, k_s, k_z = quant_per_channel(k_rot, b, clip=mse_clip(k_rot, b))
        v_q, v_s, v_z = quant_per_channel(v_rot, b, clip=mse_clip(v_rot, b))
        qkv.append(QuantizedKVCache(k_q, v_q, k_s, k_z, v_s, v_z, had_K, had_V, b))
    self.cached_kv = qkv

# dequant at consumption (transformer_cosmos3.py _forward_local L648)
def _forward_local(self, q, k, v, k_und, v_und):
    if isinstance(k_und, QuantizedKVCache):
        k_und = k_und.dequant_K(); v_und = v_und.dequant_V()    # (q-z)*s @ had^T
    k_all = torch.cat([k_und, k], dim=1)                        # verified L648
    v_all = torch.cat([v_und, v], dim=1)                        # verified L649
    return self.attn(q, k_all, v_all).reshape(q.shape[0], q.shape[1], -1)
```

> ✅ source verified: vllm-omni@95d56cf `vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474-L1479` (cached_kv 할당), `#L648-L649` (`k_all=torch.cat([k_und,k])` / `v_all`), `#L656-L679` (`_forward_sp` joint_key/value), `#L528-L550` (Causal K/V post-RoPE 반환). `QuantizedKVCache` 클래스 = (proposed, not in current upstream).

### M2. Flow-step-N 의존 denoising error bound (핵심 수학 객체)

- **① 추가 scheme (실제 anchor)**: `transformer_cosmos3.py#L628 softmax_scale=1/√d` (cross-attn) 와 `pipeline_cosmos3.py#L1492 for t in timesteps` 를 입력으로 받는 **offline 분석 모듈** `anchor/bound.py` (신규 파일, ≈120 LoC). 코드 자체엔 양자화-runtime 외 변경 없음; bound 는 M3 의 bit-selection 에 상수로 공급.
- **② 해결 문제 (workload 1:1)**: bit 선택의 정량 근거 부재. T2V N=50 vs policy N=4 가 bit 에 얼마나 다르게 관대한지 측정 없이 알 수 없음 → 휴리스틱 bit 선택은 품질 risk.
- **③ 동작 원리 (유도 전 단계)**:
  1. **value-perturbation (per layer ℓ)**: gen layer ℓ 의 cross-attn 출력 `O_DM^{(ℓ)} = A^{(ℓ)}·[V_AR^{(ℓ)};V_DM^{(ℓ)}]` (A = softmax attention matrix, row-stochastic ⇒ `‖A‖_∞ ≤ 1`). V_AR^{(ℓ)} 만 ΔV_AR^{(ℓ)} 교란되므로 `‖ΔO_DM^{(ℓ)}‖ ≤ ‖A_{:,AR}^{(ℓ)}‖_∞ · ‖ΔV_AR^{(ℓ)}‖ ≤ ‖ΔV_AR^{(ℓ)}‖`. (자료구조: K_AR/V_AR 은 **36 gen layer 전부에서 read**(L1499-1507) → 교란은 layer 마다 독립 entry.)
  2. **key-perturbation (softmax-Lipschitz, per layer)**: logit `ℓ_logit^{(ℓ)} = Q_DM^{(ℓ)}·K_AR^{(ℓ)T}·(1/√d)` (verified scale L628). softmax Jacobian spectral norm `≤ 1/2`. `‖ΔA^{(ℓ)}‖ ≤ L_softmax^{(ℓ)}·‖ΔK_AR^{(ℓ)}‖`, `L_softmax^{(ℓ)} ≤ ‖Q_DM^{(ℓ)}‖/(2√d)`. 출력 변화 `‖ΔO_DM^{(ℓ)}‖ ≤ ‖ΔA^{(ℓ)}‖·‖[V_AR^{(ℓ)};V_DM^{(ℓ)}]‖`.
  3. **layer-국소 perturbation**: layer ℓ 의 cross-attn 출력 perturbation `ε^{(ℓ)} ≤ L_softmax^{(ℓ)}·‖ΔK_AR^{(ℓ)}‖·‖V^{(ℓ)}‖ + ‖ΔV_AR^{(ℓ)}‖ =: ‖ΔV_AR^{(ℓ)}‖ + c^{(ℓ)}·‖ΔK_AR^{(ℓ)}‖` (c^{(ℓ)} = L_softmax^{(ℓ)}·‖V^{(ℓ)}‖).
  4. **layer-간 누적 (depth 전파, L=36)**: 한 step 의 generator forward 안에서 K_AR/V_AR 은 **36개 gen layer 전부에서 read** 되므로(L1499-1507) 각 layer 의 ε^{(ℓ)} 가 이후 layer 로 forward 전파된다. layer ℓ→끝 의 Jacobian spectral norm 곱을 `Π_{j>ℓ} L_blk^{(j)}` 로 두면 step n 의 cross-attn-stack 누적 출력 perturbation:
     `E_n ≤ Σ_{ℓ=1}^{L} ( Π_{j=ℓ+1}^{L} L_blk^{(j,n)} ) · ε^{(ℓ,n)}`  (per-layer Lipschitz 전파 형태). 보수 상계로는 `E_n ≤ L · max_ℓ ( (Π_{j>ℓ} L_blk^{(j,n)})·ε^{(ℓ,n)} )` (L=36-layer 합을 L·max 로 대체).
  5. **velocity-net Lipschitz (step 간 전파)**: generator velocity head `v_θ^{(n)}` 가 36-layer cross-attn-stack 출력을 입력받음. Jacobian spectral norm `L_v^{(n)} = ‖∂v_θ^{(n)}/∂O_DM‖` 를 **power iteration**(64 trajectory, 20 iter)으로 측정 — 단 K_AR 이 layer 마다 read 되므로 **layer-resolved Jacobian** `L_v^{(ℓ,n)}` 측정으로 확장(아래 bound-tightness 실험 참조). step n 기여 `Δ_n · L_v^{(n)} · E_n`.
  6. **N×L-합산 최종 bound (step×layer 이중 누적)**:
     `‖x̂_0 − x_0‖ ≤ ( Σ_{n=1}^{N} Δ_n·L_v^{(n)} ) · ( Σ_{ℓ=1}^{L} ( Π_{j>ℓ} L_blk^{(j)} )·( L_softmax^{(ℓ)}·‖ΔK_AR^{(ℓ)}‖ + ‖ΔV_AR^{(ℓ)}‖ ) )`.
     - 보수(loose) 형태: `≤ ( Σ_{n=1}^{N} Δ_n·L_v^{(n)} ) · L · max_ℓ ( L_softmax^{(ℓ)}·‖ΔK_AR^{(ℓ)}‖ + ‖ΔV_AR^{(ℓ)}‖ )` (L=36).
     - **N-의존성**: step 합 `Σ_{n=1}^N Δ_n·L_v^{(n)}` 이 N 에 단조 증가 → **N 작을수록(policy N=4) bound tight = policy 가 저비트에 관대**라는 검증가능 예측 (RQ3).
     - **L-누적의 정직한 함의 (수학적)**: 36-layer 합/Π 항을 더하면 단일 cross-attn perturbation 만 보던 구 bound 대비 bound 가 **더 loose 해진다**(최대 ~36× 보수, layer Lipschitz 곱이 1 초과 시 더). 즉 구 single-layer 식은 layer-간 누적을 누락해 **bound 를 과소평가(optimistic invalid)** 했고, 정정 후엔 bound 보수성이 증가한다 → **tightness plot 의 필요성이 더욱 커진다**(과소·과대 모두 위험하므로 실측 대조 필수).
     - ΔK_AR^{(ℓ)}/ΔV_AR^{(ℓ)} 은 M1 의 quant error → bit 의 함수 (per-channel, per-layer). bound 를 최소화하는 bit/clipping 을 layer 별 선택(M3).
- **④ 기존 실패 이유 + 차별화**: online KV quant 엔 step 개념 자체가 없어 N-의존 bound 불가. OT-Quant ([arXiv:2511.11418](https://arxiv.org/abs/2511.11418))는 "step 통해 전파"를 정성 언급할 뿐 closed-form 아님. QuantKeys/33-method 는 bound 전무. ANCHOR 의 N-bound = static prefix + flow-matching 조합에서만 정의.

```python
# M2: bound computation  (anchor/bound.py, proposed)
# inputs grounded in code: softmax_scale=1/sqrt(d) (L628), N steps (pipeline L1492),
# K_AR/V_AR read at all 36 gen layers (L1499-1507) -> layer accumulation term
def denoising_error_bound(dK_per_layer, dV_per_layer, Q_DM_norm_per_layer, V_norm_per_layer,
                          d, step_sizes, Lv_per_step, L_blk_per_layer, conservative=False):
    # per-layer local cross-attn perturbation eps^(l) = ||dV_AR^l|| + c^l*||dK_AR^l||
    L_softmax = [q / (2.0 * d**0.5) for q in Q_DM_norm_per_layer]   # <=1/2, scale 1/sqrt(d)
    eps = [L_softmax[l] * dK_per_layer[l] + dV_per_layer[l]
           for l in range(len(dK_per_layer))]                       # one per gen layer
    Lnum = len(eps)                                                 # = 36
    # layer-to-end Jacobian product Pi_{j>l} L_blk^(j)  (depth propagation, L=36)
    suffix_prod = [1.0] * (Lnum + 1)
    for l in range(Lnum - 1, -1, -1):
        suffix_prod[l] = suffix_prod[l + 1] * L_blk_per_layer[l]
    if conservative:                                                # E_n <= L * max_l(...)
        layer_term = Lnum * max(suffix_prod[l + 1] * eps[l] for l in range(Lnum))
    else:                                                           # tight: sum_l Pi_{j>l}*eps^l
        layer_term = sum(suffix_prod[l + 1] * eps[l] for l in range(Lnum))
    step_gain = sum(dt * Lv for dt, Lv in zip(step_sizes, Lv_per_step))  # sum_{n=1}^N
    return step_gain * layer_term                  # closed-form ||x_hat0 - x0||, step x layer

def measure_Lv(velocity_fn, x_t_traj, n_iter=20):  # power iteration over N steps
    Lv = []
    for n, x in enumerate(x_t_traj):              # one entry per diffuse step (L1492)
        u = torch.randn_like(x)
        for _ in range(n_iter):
            Jv = jvp(velocity_fn[n], x, u)        # J @ u  (autograd jvp)
            u = Jv / Jv.norm()
        Lv.append(jvp(velocity_fn[n], x, u).norm().item())
    return Lv                                     # L_v^{(n)} per step
    # NOTE: bound-tightness 실험에서 L_v 를 layer-resolved L_v^{(l,n)} 로 분해 측정해
    #       layer 누적 항(suffix_prod)의 보수성을 실측 대조 (per-layer jvp hook).
```

> ✅ source verified: vllm-omni@95d56cf `transformer_cosmos3.py#L628` (`softmax_scale=1.0/(self.head_dim**0.5)`), `pipeline_cosmos3.py#L1492` (`for t in self.progress_bar(timesteps)` = Σ_{n=1}^N 구조), `transformer_cosmos3.py#L1499-1507` (`for layer,(k_und,v_und) in zip(self.gen_layers,self.cached_kv)` = 36 gen layer 전부에서 K_AR/V_AR read → layer 누적 항 Σ_{ℓ=1}^{L} 근거). `anchor/bound.py` = (proposed, not in current upstream).

### M3. Deploy & bound-validation (bit 자동 선택)

- **① 추가 scheme (실제 anchor)**: `quantization/component_config.py#L63-73 resolve()` 를 확장해 prefix→bit 라우팅에 **per-layer KV bit** 를 추가 (≈15 LoC 수정). bit-selection 진입은 `transformer_cosmos3.py#L1281-1283 reset_cache()` 직후 hook (≈20 LoC) — request 시작마다 bound-tolerance τ 하 layer 별 최소 bit 결정.
- **② 해결 문제 (workload 1:1)**: bound 가 실용적(tight)인지 실증 없이 신뢰 불가. policy(action-MSE budget τ) vs T2V(VBench budget) 별로 안전 bit 가 다름 → 자동 선택 필요.
- **③ 동작 원리**:
  1. **L_v 측정**: M2 의 `measure_Lv` 로 64 trajectory jacobian spectral norm (offline, 모델·N 별 1회). layer-누적 항 검증을 위해 layer-resolved `L_v^{(ℓ,n)}`·`L_blk^{(ℓ)}` 도 측정.
  2. **bit→ΔK/ΔV 매핑**: M1 quant 을 bit∈{8,6,4,3,2} 로 실행해 layer 별 `‖ΔK_AR^{(ℓ)}‖,‖ΔV_AR^{(ℓ)}‖` 측정 (36 layer × bit grid).
  3. **bound 계산**: 각 bit-할당에 **step×layer 이중 누적** `denoising_error_bound` 적용 (layer 합 Σ_{ℓ=1}^{36} 또는 보수 L·max) → 예측 출력 error.
  4. **bit-selection loop**: tolerance τ 하 `min Σ_layer bits s.t. bound(bit-vector) ≤ τ`. layer-누적이라 한 layer 의 고bit↑가 전체 bound 의 해당 suffix-product 항을 통해 후속 layer 에 전파 → greedy 는 큰 `Π_{j>ℓ}L_blk·L_v` 민감 layer 부터 고bit.
  5. **validation**: bound vs 실측 출력 error tightness ratio plot (bit별, 핵심 artifact). layer 누적으로 bound 가 더 보수적이라 ratio 가 커질 수 있음 → tightness plot 이 더욱 필수. ratio > 5× loose → empirical-tightening fallback (decision-tree).
- **④ 기존 실패 이유 + 차별화**: 경쟁작(QuantKeys/33-method)은 실증 sweep 만 → bit 선택이 휴리스틱(품질 측정 후 사후 결정). ANCHOR = bound 가 **사전(predictive)** 선택 기준.

```python
# M3: bit-selection loop  (hook after reset_cache, transformer_cosmos3.py L1283)
# bound is step x layer accumulated (K_AR read at all 36 gen layers, L1499-1507)
def select_bits(self, Lv, L_blk, step_sizes, tau, Q_norm_pl, V_norm_pl, d):
    L = len(self.gen_layers)                               # = 36
    bits = [8] * L                                         # start high
    # greedy: try lowering each layer's bit, keep if global accumulated bound <= tau
    order = sorted(range(L), key=lambda l: Lv_layer_gain(l, Lv, L_blk), reverse=True)
    for layer_idx in order:                               # sensitive (high suffix-prod*Lv) first
        for b in (6, 4, 3, 2):                             # high -> low
            bits[layer_idx] = b
            dK = [quant_error(self.cached_kv[l], bits[l])[0] for l in range(L)]
            dV = [quant_error(self.cached_kv[l], bits[l])[1] for l in range(L)]
            bound = denoising_error_bound(dK, dV, Q_norm_pl, V_norm_pl,
                                          d, step_sizes, Lv, L_blk)  # sum_l Pi_{j>l} term
            if bound > tau:                                # this bit breaks global bound
                bits[layer_idx] = (b * 2 if b < 8 else 8)  # revert to last safe
                break
    return bits   # -> ComponentQuantizationConfig.resolve per-layer bit (L63-73)
```

> ✅ source verified: vllm-omni@95d56cf `quantization/component_config.py#L63-L73` (`resolve` longest-prefix match), `transformer_cosmos3.py#L1281-L1283` (`reset_cache`). bit-selection hook = (proposed, not in current upstream).

### As-Is / To-Be (R52.2, 통합 7-column)

| file | symbol | as-is line region | as-is 동작 | to-be 변경 | 변경 LoC | GitHub link |
|---|---|---|---|---|---|---|
| `transformer_cosmos3.py` | `Cosmos3VFMTransformer.forward` | L1474-1479 | reasoner K/V 1회 계산→`cached_kv` (bf16) | L1479 직후 `quantize_cached_kv()` 호출 + `QuantizedKVCache` 저장 | +50 (class) / ~3 (call) | [L1474-L1479](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1474-L1479) |
| `transformer_cosmos3.py` | `Cosmos3CrossAttention._forward_local` | L648-649 | `cat([k_und,k])` (bf16 k_und) | k_und/v_und 가 `QuantizedKVCache` 면 dequant 후 cat | ~8 | [L648-L649](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L648-L649) |
| `transformer_cosmos3.py` | `Cosmos3CrossAttention._forward_sp` | L656-679 | `joint_key=k_und` (bf16) | joint_key/value dequant 적용 | ~6 | [L656-L679](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L656-L679) |
| `transformer_cosmos3.py` | `reset_cache` | L1281-1283 | cache None 초기화 | bit-selection hook trigger | ~20 | [L1281-L1283](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1281-L1283) |
| `quantization/component_config.py` | `resolve` | L63-73 | prefix→config | per-layer KV bit 라우팅 확장 | ~15 | [L63-L73](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/quantization/component_config.py#L63-L73) |
| `anchor/bound.py` | `denoising_error_bound`,`measure_Lv` | — | (없음) | 신규 offline 분석 모듈 | +120 | (proposed) |

### M1-M3 의존성 그래프

```
            reasoner forward (L1474)  ──produces──►  cached_kv (bf16, post-RoPE)
                                                          │
                                  M1: quantize_cached_kv (L1479+)  ──► QuantizedKVCache
                                                          │                      ▲
   M2: measure_Lv (offline) ──► L_v^{(n)} ──┐            │ (bit per layer)       │
                                            ▼            │                       │
   M2: denoising_error_bound ──► bound(layer,bit) ──► M3: select_bits ──────────┘
                                                          │
                                  M3: resolve (L63-73) ──► per-layer bit policy
                                                          │
            diffuse step loop (L1492) ──consumes──► dequant @ _forward_local/_sp (L648/L676)
```

---

## 5. 평가 · 실험 플랜 (R20-β 7요소)

### (1) HW

- **RTX Pro 6000 96GB (Blackwell)**: calibration(Hadamard+clipping search) + L_v power-iteration + **ncu BW 측정** (`dram__bytes_read`, `l2_tex_hit_rate` 지원). bound-validation 의 정밀 reproduction 환경.
- **AGX Orin 64GB (sm_87)**: 배포 latency·footprint. **ncu 미지원(ga10b)** → Nsight Systems + tegrastats + wall-clock latency proxy 명시. KV-read latency delta 는 BW-bound 구간에서 간접 관찰.
- **Thor (sm_101) 입수 시 조건부**: Blackwell-class edge → ncu 지원 가능성. 입수 확정 시 Orin 결과를 Thor 로 cross-check (조건부, 비필수).

### (2) 모델

- **`nvidia/Cosmos3-Nano`** (BF16-only 공식) — quant 는 vllm-omni `ComponentQuantizationConfig` 확장 경로로 비공식 정밀도 신규 도입. T2V/T2I.
- **`nvidia/Cosmos3-Nano-Policy-DROID`** (N=4, **guidance scale 3.0 → CFG ON → K_AR cache 2벌** [cond/uncond], CFG-parallelism; tech report §4.2.5). bound-favorable 검증 주력 (RQ3).
- **fallback BAGEL-7B-MoT** (14B/7B-active, vLLM upstream 미지원) — method 일반성(cross-model bound) 확인용.

### (3) Workload / Dataset

- **VBench / GenEval**: T2V / T2I 생성 품질 (FP16 대비 Δ).
- **RoboLab-120**: subset 20-30 task = ablation, full 120 = final. action-MSE + 성공률.
- **MMMU**: VLM sanity (reasoner 출력이 quant 으로 깨지지 않는지 — K_AR 은 reasoner 출력이므로 sanity 필요).

### (4) Tools

- **rotation**: llm-compressor 또는 QuaRot fork (Hadamard). 예: `llmcompressor.transform.hadamard(head_dim=128)` 를 `cached_kv` 텐서에 적용.
- **power-iteration jacobian**: 자체 스크립트 `measure_Lv` (autograd jvp, 64 traj × 20 iter). 예: `python anchor/measure_lv.py --model Cosmos3-Nano-Policy --N 4 --traj 64`.
- **profiler**: vllm-omni `DiffusionPipelineProfilerMixin.stage_durations` + `OmniTorchProfilerWrapper.annotate_context_manager` (record_function). 예: `ncu --metrics dram__bytes_read.sum,l2_tex_hit_rate -k regex:attention python diffuse.py`.

### (5) Ablation + Measurement Protocol

- **grid**: bit{8,6,4,3,2} × Hadamard{on,off} × N{4,50} (= 5×2×2 = 20 cells).
- **baseline 5종 (venue tag)**: FP16(상한) / KIVI INT4(ICML'24) / KVQuant(NeurIPS'23) / RTN INT4 / QuantKeys(arXiv). + OT-Quant(related-work, 비교 제외).
- **main metric**: (i) action-MSE (policy), (ii) VBench score (T2V/T2I), (iii) **bound-tightness ratio = bound / 실측 출력 error** (bit별, 핵심 artifact).
- **예상 총 GPU-시간**: calibration+L_v(RTX) ~80h, grid×model×bench(RTX) ~260h, Orin 배포 측정 ~60h, RoboLab full ~120h ≈ **~520 GPU-h**.

### (6) Implementation Steps (주차별)

| 주차 | 변경 component / 파일 경로 | 사용 API / 도구 | 완료 판정 조건 |
|---|---|---|---|
| W1 | `transformer_cosmos3.py` `QuantizedKVCache` 골격 | PyTorch, llm-compressor | bit-exact RTN INT8 round-trip pass |
| W2 | `transformer_cosmos3.py#L1479` quantize hook + `#L648/676` dequant | torch.cat, QuaRot fork | T2I 1-step output ≈ bf16 (INT8) |
| W3 | Hadamard rotation (had_K post-RoPE, had_V) | QuaRot Hadamard | RoPE 보존 verify (q·k^T 불변) |
| W4 | per-channel MSE-optimal clipping | grid-search 스크립트 | INT4 MSE < RTN INT4 MSE |
| W5 | `anchor/bound.py` value/key-perturbation 항 | numpy/torch | softmax-Lipschitz 단위테스트 pass |
| W6 | `anchor/measure_lv.py` power-iteration (step + **layer-resolved** L_v^{(ℓ,n)}, L_blk^{(ℓ)}) | autograd jvp (per-layer hook) | L_v 수렴(<2% drift @20iter), 36-layer L_blk 측정 |
| W7 | bound 통합 (**N×L 이중 누적**: Σ_n step × Σ_ℓ layer 또는 L·max 보수) | bound.py | N=4 vs N=50 bound 분리 + single-layer vs 36-layer 누적 bound 비교 산출 |
| W8 | bit별 bound vs 실측 tightness plot (layer-누적 보수성 포함) | matplotlib, profiler | INT4@N=4 tightness ratio 측정 (누적으로 ratio↑ 예상 — plot 필수성↑) |
| W9 | `component_config.py#L63` per-layer bit 라우팅 | ComponentQuantizationConfig | per-layer bit 적용 e2e run |
| W10 | `reset_cache` bit-selection hook | M3 select_bits | τ 하 자동 bit 선택 동작 |
| W11 | RoboLab subset + VBench subset 평가 | RoboLab-120, VBench | baseline 5종 비교표 |
| W12 | RoboLab full 120 + Orin 배포 latency | Nsight Systems, tegrastats | Orin latency delta 측정 |
| W13 | bound-tightness 최종 plot + 논문 figure | matplotlib | 핵심 artifact 완성 |

### (7) Preliminary Analysis Metrics + 4단계

- **(i) baseline reproduction**: vllm-omni cosmos3 T2I/policy 실행.
  - 명령: `vllm-omni serve nvidia/Cosmos3-Nano` (T2I), `run_id_with_vllm` (policy N=4, **guidance=3.0 → CFG ON, 2벌**).
  - 기대: T2V-720p Nano 1-GPU ~286-297s(H100, Fig.16), policy chunk latency RTX 기준 측정.
- **(ii) bottleneck attribution**: `cached_kv` read BW 비중.
  - 명령: `ncu --metrics dram__bytes_read.sum,l2_tex_hit_rate -k regex:cross_attention python diffuse.py` (RTX).
  - 기대: cross-attn KV read 가 step BW 의 의미있는 비중(>15% T2V long-context).
- **(iii) roofline**: KV read bytes/step 계산식.
  - `bytes/step = num_layers × S_AR × H_kv × D × 2(K,V) × precision_bytes`. policy 36L×**1,050**×8×128×2×2(bf16) ≈ **155MB/step** → INT4 ≈ **39MB/step** (per CFG pass; policy 는 CFG ON 이라 cond/uncond 2-pass → step 당 실 traffic ×2 ≈ 310MB(bf16)/78MB(INT4)).
  - 명령: 위 계산 + ncu 실측 대조. 기대: 측정 bytes/step 이 계산식 ±15%.
- **(iv) micro-benchmark**: `cat([k_und,k])` 구간 isolation.
  - 명령: `nsys profile -t cuda --capture-range cudaProfilerApi python microbench_crossattn.py` (L648 구간만 stub).
  - 기대: INT4 dequant+cat 이 bf16 cat 대비 latency neutral 또는 BW 절감으로 net 이득.

---

## 6. 예상 효과 / Risk / 검증 / Tier-2 / Scoring / Decision-tree

### 예상 효과 (보수치 + scope)

- conditioning-KV **2-4× 압축 (INT4/INT3) 품질손실 <1%** (정적 K_AR 한정, decode-KV 아님). policy per-layer BF16 ~4.3MB→INT4 ~1.08MB (~1,050 tok, CFG ON 2벌).
- **policy N=4 → INT3, bound-favorable 시 INT2** (action-MSE <2%, bound 허용 시만). T2V N=50 은 INT4 까지 보수적. 단 bound 는 36-layer 누적이라 보수적 → 실측 tightness 가 bit 하한을 결정.
- step KV-read latency bit 비례 절감 (edge LPDDR5 BW-bound 구간; Orin latency proxy). policy roofline ~155MB/step(bf16, per CFG pass)→INT4 ~39MB/step.
- 핵심 산출물 = **bit별 bound-tightness plot** (step×layer 이중 누적 bound 의 실측 대조; 경쟁 KV-quant 전부 부재 = clear novelty artifact).

### Risk + 완화

- (R) **bound loose** (L_v 과대추정 + **36-layer 누적 항이 bound 를 ~36× 보수화**) → 실용 bit 선택 무용. (M) layer-resolved L_v^{(ℓ,n)}·L_blk^{(ℓ)} 측정으로 누적 항을 실측 채워 tighten; tightness plot 필수, empirical-tightening + 상대 ordering 가이드로도 유효; ratio >5× 시 fallback (decision-tree). 단 누락 시 bound 가 layer 누적을 빠뜨려 **과소평가(optimistic invalid)** 되므로 누적 항 유지가 soundness 전제.
- (R) **Hadamard × MRoPE 간섭** → K 가 post-RoPE 저장(verified L550)이므로 RoPE 뒤 rotation + q 대칭 적용으로 위치정보 보존. QuaRot 순서규칙 준수.
- (R) **BF16-only fork 부담** → `ComponentQuantizationConfig` 확장 경로 활용, BAGEL 로 method 일반성 먼저.
- (R) **SP/TP 경로 dequant 누락** → `_forward_local`·`_forward_sp` 두 소비처 모두 hook (verified 분기 L648/L676).

### R52.4 Synthetic 검증 (Tier-A/B/C)

- **Tier-A (unit, 1 prompt 1 step)**: bit-exactness — INT8 round-trip 이 bf16 와 `allclose(rtol=1e-2)`. Hadamard `had@had^T=I` 단위테스트. q·k^T rotation 불변 검증.
- **Tier-B (mechanism-isolated, 10-50 prompt)**: bound vs actual — 50 prompt 에서 예측 bound 가 실측 출력 error 를 상회(soundness)하고 ratio ≤5× (tightness). N=4 vs N=50 bound 분리 확인.
- **Tier-C (e2e, VBench subset)**: VBench subset(20 prompt) + RoboLab subset(20 task) 에서 INT4 품질손실 <1%, policy INT3 action-MSE <2%.

### Tier-2 variant (직전 계승 + 보강)

- **policy-mode 4-step static-KV INT4 + measured action-MSE budget** — DATE / IEEE-CAL short.
- M2 의 full N-bound 유도를 생략하고 **policy N=4 한정**, bound 의 N-의존 예측을 RoboLab action-MSE budget τ 하 최소 bit 로 실증 (full Q2 의 가장 검증 쉬운 슬라이스).
- 정량 표:

  | 지표 | Baseline | variant | 개선 |
  |---|---|---|---|
  | policy K_AR bit | INT8 | INT4/INT3 | 2× footprint |
  | action-MSE | FP16 상한 | budget τ 하 | <2% (τ 명시) |
  | bound-tightness | — | ≤3× @N=4 | 승격 gate |

- **승격**: policy N=4 tightness ≤3× 확인 + T2V N=50 확장 시 full Q2(Tier-1)로.

### Scoring 박스

| 축 | 점수 | 사유 |
|---|---|---|
| novelty | **8 ★** | flow-step-N 의존 closed-form denoising error bound = 경쟁 KV-quant(QuantKeys/33-method/OT-Quant) 전부 부재 (WebFetch+grep 확인). static cross-tower KV 특화 1-shot amortize. |
| differentiation | 8 | 모든 baseline(KIVI/KVQuant/QuantKeys) growing-KV·online·bound-free. ANCHOR static prefix + N-bound 단독. |
| impact | 8 | edge BW-bound denoise 의 conditioning-KV 2-4× 압축 + Q2→L3 paper-pair. policy 현실 K_AR(BF16 ~4.3MB/layer) > 3MB L2 set-aside → **L3 per-layer pin 이 Q2 INT4(1.08MB)를 사실상 전제** (CFG ON 2벌 = 2.16MB ≤ 3MB fit). |
| ai-impl | **7 ▼** | BF16-only fork + L_v jacobian calib + SP/TP 두 경로 dequant ~13wk. bound tightness 가 load-bearing risk. |
| arch-impl | 7 | `cached_kv` hook·`component_config.resolve` 확장이 기존 plumbing 위 자연 삽입 (anchor 확인). |
| **avg** | **7.6** | Tier-1 최강 단일 후보. |

### Decision-tree 분기

- **pass**: INT4 bound-tightness ratio ≤3× @N=4 (policy) → Tier-1 MLSys 제출 진행, T2V N=50 INT4 확장.
- **below (bound loose)**: tightness ratio 5-10× → bound 를 상대 ordering 가이드로 강등, "1-shot Hadamard static-KV quant (실증-중심, bound 없이)" 로 reposition (여전히 publishable, novelty↓).
- **N-의존성 falsify**: policy N=4 와 T2V N=50 의 bound-tightness 차이가 예측대로 안 나오면 → Tier-1 핵심 가설 붕괴, 1차 gate (W7-8).
- **outperform**: policy INT2 성공(action-MSE <2% + bound 허용) → ICML 본선 (저비트 extreme + bound 정당화).

---

> ✅ 전체 source-verified anchor (본인 grep/Read 확인): vllm-omni@95d56cf `transformer_cosmos3.py` {L1460, L1474, L1479, L648-649, L656-679, L528-550, L628, L1281-1283, L777-846, L900-924}, `pipeline_cosmos3.py` {L1282, L1332, L1492, L1486-1521}, `component_config.py` {L51-108, L63-73}, cosmos-framework@003d66d `unified_mot.py#L1224` (ReasonerKVCache trap 확인 — text-AR decode 용, serving cached_kv 아님).
