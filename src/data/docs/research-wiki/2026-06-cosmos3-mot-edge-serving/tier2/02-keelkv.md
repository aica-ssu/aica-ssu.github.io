# L3 KEELKV (Tier-2 #2 독립, avg 6.6)

**KEELKV: Per-Layer L2 Set-Aside Pinning of Static Cross-Tower Conditioning KV during the Denoise Loop** · DAC / DATE / CAL 2027

---

## 1. 개요

- **metaphor↔mechanism**: KEELKV = denoise 항해 내내 고정된 K_AR/V_AR 을 L2 "용골(keel)"에 고정. 배의 용골이 항해 중 흔들리지 않듯, conditioning KV 를 L2 set-aside 에 pin 해 매 step DRAM 으로 떠내려가지 않게 한다.
- **한 줄**: denoise 전 기간 read-only 인 static cross-tower K_AR/V_AR 을 per-layer L2 set-aside window 에 pin 하고, 매 step streaming 되는 video-K_DM(수만 token)을 demote 해 K_AR 축출(L2 pollution)을 막는다.
- **G4 소유**: static K_AR/V_AR 의 **on-chip L2 placement**. quant 은 Q2, layout 은 A2 위임.

## 2. 기존 연구 한계 · GAP (workload evidence 수치 포함)

- **GAP + size 수학**: K_AR/V_AR 은 denoise 전 기간 read-only 인데 매 step streaming 되는 video-K_DM(수만 token)이 L2 점유 → K_AR 축출 → DRAM 재독출 (L2 pollution **2.15×** [arXiv:2501.16909](https://arxiv.org/abs/2501.16909)).
- **policy mode K_AR fit 검증**: per-layer = prompt(~300 tok)×8 KV-head×128 dim×2(K,V)×precision = **BF16 1.2MB / INT8 0.6MB / INT4 0.3MB → 3MB L2 set-aside 에 per-layer fit 확정** (Q2 INT4 시 4× 여유). 전체 36L 합산(29-44MB)은 안 들어가나 cross-attn 커널은 한 layer 만 read 하므로 무관.
- **workload evidence**:
  - reasoner caching K_AR/V_AR read-only static (§5.3.2); GQA 8 KV-head; AGX Orin sm_87 **L2=4MB, set-aside 3MB(`l2_max_perst_spc`) 확정 지원**.
  - vLLM/SGLang/llama.cpp/TRT **전부 L2 set-aside 미사용** → static cross-tower KV L2-pin 은 어느 엔진도 미적용.

## 3. 제안 기법 (mechanism ≤2)

### M1. per-layer K_AR/V_AR persisting L2 window + video-K_DM streaming demote

- **① 추가 scheme**: denoise 진입 시 `cudaStreamSetAttribute accessPolicyWindow(base_ptr=K_AR, hitProp=Persisting, num_bytes=per-layer K_AR/V_AR)`; video-K_DM read 는 `Streaming` prop.
- **② 해결 문제**: video-K_DM 수만 token 이 K_AR 을 L2 에서 축출 → DRAM 재독출 (per-step BW 낭비).
- **③ 동작 원리**: (1) `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)` 로 3MB set-aside 확보. (2) denoise loop 진입 시 K_AR(policy 1.2MB BF16 / 0.3MB INT4)을 Persisting window 로 pin. (3) video-K_DM 은 Streaming prop → K_AR 축출 방지. (4) loop 종료 시 `cudaCtxResetPersistingL2Cache()`.
- **④ 기존 실패 이유 + 차별화**: Async KV Prefetch→L2([arXiv:2504.06319](https://arxiv.org/abs/2504.06319))는 decode-time growing KV 대상·prefetch(상주 pin 아님)·discrete GPU. KEELKV 는 denoise 전기간 read-only static cross-tower K_AR 을 per-layer set-aside pin + K_DM streaming demote (Orin sm_87 3MB 확정).

### M2. SMEM staging fallback

- **① 추가 scheme**: set-aside 미지원/협소 HW 에서 GQA-8 K_AR tile 을 SMEM(192KB/SM) step 시작 1회 stage (FlashAttention-style).
- **② 해결 문제**: L2 set-aside 미지원 HW 일반화.
- **③ 동작 원리**: (1) step 시작 시 GQA-8 K_AR tile 을 SMEM 로 1회 load. (2) tile 내 attention 은 SMEM read. (3) L2 지원 확정으로 M1 우선, M2 는 일반성.
- **④ 기존 실패 이유 + 차별화**: 기존 FlashAttention SMEM staging 은 매 step 재load — KEELKV 는 static K_AR 이라 step 시작 1회만.

## 4. 평가 · 실험 플랜 (R20-β 7요소)

- **HW**: L2-hit 인과검증 = RTX Pro 6000(ncu 지원); AGX Orin = latency/BW delta(Nsight Systems) 간접; Orin indirect 만 honest scope.
- **모델**: Cosmos3-Nano policy + T2I/T2V-짧은clip.
- **워크로드**: per-denoise-step latency, DRAM read bytes (policy 우선 단일 workload, T2V 보조).
- **도구**: `cudaAccessPolicyWindow`, RTX ncu `l2_tex_hit_rate`, Nsight Systems(Orin).
- **ablation + baseline**: pin on/off × {policy, T2V} × {BF16, Q2-INT4}. baseline — (a) reasoner-caching only(no L2 pin), (b) naive(K_AR DRAM 재독출), (c) RTX control L2-pin 상한.
- **주차별 구현**:

  | 주차 | 작업 |
  |---|---|
  | W1-2 | per-layer K_AR fit 수학 검증 + `cudaAccessPolicyWindow` set-aside 구현 |
  | W3-5 | video-K_DM streaming demote + RTX ncu `l2_tex_hit_rate` pin on/off 측정 |
  | W6-8 | Orin latency/BW delta 간접 측정 + Q2-INT4 pair + baseline |
  | W9 | SMEM fallback(M2) + scope 검증(T2V K_AR>3MB → policy-only) |

- **preliminary metrics**: RTX `l2_tex_hit_rate`(K_AR) pin on/off, per-step DRAM read.

## 5. 예상 효과 (보수치, scope 명시)

- K_AR L2 hit↑로 per-step DRAM read **10-20%↓** (K_AR 비중만큼).
- per-step latency **5-12%↓**.
- T2V K_DM streaming 클수록 pollution 회피 이득 큼.
- scope: per-step DRAM/latency, **T2V 한정 단일 metric**; T2V K_AR>3MB(긴 conditioning) → policy-mode-only scope 축소.
- **Q2→L3 paper-pair**: Q2 INT4 K_AR 이 3MB L2 에 4× 여유 pin → BW·SRAM 동시 절감 (단, BF16 에서도 standalone 성립 — pair 는 보너스).
