# Q1 DRIFT (Tier-1 #2, 직전 avg 7.4)

**DRIFT: Quantization-Sensitivity Divergence of Co-Initialized Towers in Mixture-of-Transformers Omnimodal Models and Budget-Constrained Asymmetric Bit Allocation**

- **Venue 후보**: NeurIPS / ICML / MLSys 2027
- **G4 ownership**: weight 양자화 sensitivity 측정·배분 (G4 static-K_AR 축과 직교 — DRIFT 는 weight, Q2/L3 는 KV)
- **Metaphor (DRIFT = 표류)**: 같은 항구(`Qwen3-VL-8B` co-init)에서 출발한 두 배 — reasoner tower(understanding)와 generator tower — 가 서로 다른 학습 objective(AR next-token cross-entropy vs rectified-flow velocity matching MSE)라는 해류를 타고 표류한다. 출항 시점엔 동일한 weight 였으나, 항해(training)가 끝났을 때 두 배 사이 벌어진 거리 = **quantization-sensitivity divergence ρ_ℓ**. 같은 init 에서 출발했기에 이 거리는 순수하게 objective 차이로 귀속된다 (controlled natural experiment).

> 본 문서는 R72.1 (디테일 생략 금지) 하에 작성. 모든 source anchor 는 본인이 `/tmp/cosmos3-repos/` (vllm-omni `95d56cf`, cosmos-framework `003d66d`, cosmos `7f5797f`, vllm `063ce98`) 에서 재확인함 (R72.3).

---

## 0. 약어 glossary

| 약어 / 기호 | 정의 |
|---|---|
| **ρ_ℓ** | layer ℓ 의 tower-간 quant-sensitivity divergence ratio = `S_ℓ^R / S_ℓ^G` (R=reasoner, G=generator). co-init null hypothesis 는 ρ_ℓ ≈ 1. |
| **S_ℓ^τ** | layer ℓ, tower τ∈{R,G} 의 quant-sensitivity score = `Tr(H_ℓ^τ) · ‖ΔW_ℓ^τ(b)‖_F²` (HAWQ-style). H=Hessian, ΔW(b)=bit-b 양자화 perturbation. |
| **Hessian trace** | `Tr(H)` = loss 의 2차 곡률 합. 큰 trace = 작은 weight 섭동에도 loss 급증 = 양자화 민감. Hutchinson estimator(stochastic HVP)로 추정. |
| **HAWQ** | Hessian-AWare Quantization [arXiv:1905.03696](https://arxiv.org/abs/1905.03696). per-layer Hessian eigenvalue/trace 로 mixed-precision bit 배정. DRIFT 의 sensitivity 정의의 이론적 조상 (tower 개념 없음). |
| **ILP** | Integer Linear Programming. bit∈{8,6,4,3} 이산 변수에 대한 정수 최적화. PuLP / OR-Tools 로 풀이. |
| **kurtosis (κ)** | activation/weight 분포의 4차 모멘트 표준화값. 큰 κ = heavy-tail = outlier 多 = per-tensor 양자화에 취약. AR tower 의 activation outlier 진단 지표. |
| **weight drift δ_ℓ** | `‖W_ℓ^R − W_ℓ^G‖_F / ‖W_init‖_F`. co-init 동일 weight 가 training 후 얼마나 갈라졌는지. ρ_ℓ 과의 상관 = "표류 거리 ↔ 민감도 발산" 인과 가설. |
| **PTQ / QAT** | Post-Training / Quantization-Aware Training. 본 idea 는 **PTQ 한정** (학습 재개 없음). |
| **mxfp4 / mxfp8** | Microscaling FP4(E2M1)/FP8 — block-shared exponent 양자화. vllm-omni 에 W4A4/W8A8 구현 존재 (NPU 경로). |
| **GPTQ / AWQ** | reasoner-tower(LLM) PTQ 정설. GPTQ [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) = 2차근사 column-wise; AWQ [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) = activation-aware salient channel scaling. |
| **SVDQuant** | generator-tower(DiT) PTQ 정설 [arXiv:2411.05007](https://arxiv.org/abs/2411.05007). low-rank branch 로 outlier 흡수 후 W4A4. |
| **Fisher proxy** | Hessian 대신 gradient² 기대값(empirical Fisher)으로 곡률 근사. HVP 불가/비싼 경우 fallback. |
| **UMA** | Unified Memory Architecture. Jetson Orin 의 CPU/GPU 공유 LPDDR5. 모델·KV·activation 이 단일 예산 B 를 공유 → tower-간 bit trade-off 의 물리적 근거. |

---

## 1. 개요

- **metaphor ↔ mechanism 대응**:
  - *같은 항구* = `Cosmos3VFMTransformer` 의 두 tower 가 모두 `Qwen3-VL-8B` 에서 co-init (tech report §2.3.1, line 933). reasoner=`self.language_model`(`Cosmos3LanguageModel`), generator=`self.gen_layers`(`Cosmos3GenDecoderLayer` × num_hidden_layers). **둘 다 동일 hidden_size·num_heads·head_dim·rope_theta 로 생성** (본인 확인: `transformer_cosmos3.py#L1058-1111` 에서 두 tower 가 같은 hyperparameter dict 로 instantiate).
  - *다른 해류* = reasoner 는 AR next-token CE, generator 는 rectified-flow velocity MSE 로 학습. ViT encoder 는 reasoner 와 공동학습(co-trained), VAE 는 frozen — 따라서 **ρ_ℓ 측정 대상은 오직 transformer decoder layer 의 두 parameter set** (encoder/VAE 제외; 이것이 controlled experiment 의 통제 경계).
  - *표류 거리* = ρ_ℓ. 발산이 크면(ρ_ℓ ≫ 1 또는 ≪ 1) tower 별로 다른 bit 가 정당화되고, ρ_ℓ ≈ 1 이면 "균일이 충분" (negative-but-publishable).
- **Research Question**: *co-init 된 두 tower 의 양자화 민감도는 (a) 얼마나, (b) 어디서(어느 layer/sub-module) 발산하는가? (c) 그 발산을 알면 동일 UMA 메모리 예산에서 무엇을(품질 회복 또는 footprint 절감) 얻는가?*
- **이중 기여**:
  1. **측정 과학 (M1)**: co-init 이라는 *controlled natural experiment* — init·구조·data 가 통제된 상태에서 발산이 순수하게 objective(AR-CE vs flow-MSE) 차이로 귀속됨. ρ_ℓ / κ_ℓ^τ / δ_ℓ 분포 + (δ_ℓ, ρ_ℓ) 상관을 "법칙" 후보로 제시. 이것이 직전 흡수된 Q4 SIEVE 의 핵심.
  2. **시스템 (M2)**: UMA 예산 B 하 per-(tower, layer) bit 배분을 ILP/greedy 로 풀고, vllm-omni 의 `ComponentQuantizationConfig` prefix 라우팅(이미 존재)으로 **실제 서빙 스택에 배선**. 측정만 하는 선행연구와 달리 deployable.
- **conditional-gain 정직성 (직전 review 반영)**: 이득은 ρ_ℓ 발산이 material 할 때만 존재. **kill condition: ρ_ℓ < 1.5× across ≥80% layers → Tier-2 측정-only letter 로 강등** (phase2' Task 5 AGREE). 두 번째 gate: 균일-W4 대비 품질 회복 < 0.5%p 이면(이득 측정불가) demote.

---

## 2. Mechanism Summary Table (R70 7열)

| # | Mechanism | 무엇을 하나 | Source anchor (verified) | 추가물(novel) | 핵심 metric | 차별화 1줄 |
|---|---|---|---|---|---|---|
| **M1** | ρ_ℓ divergence 측정 (구 Q4 흡수) | tower·layer 별 Hessian-trace sensitivity S_ℓ^τ, ρ_ℓ, κ_ℓ^τ, δ_ℓ 산출 + (δ_ℓ,ρ_ℓ) 상관 | tower 분리: `transformer_cosmos3.py#L1058-1111` (`language_model` vs `gen_layers`); [blob](https://github.com/vllm-project/vllm-omni/blob/95d56cf/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1058-L1111) | tower-쌍 차분 sensitivity, weight-drift δ_ℓ 와의 인과 상관 | ρ_ℓ 분포, corr(δ_ℓ,ρ_ℓ), κ gap | co-init controlled experiment — 발산이 순수 objective 귀속 |
| **M2** | 예산 제약 asymmetric bit 배분 | UMA B 하 per-(tower,layer) bit∈{8,6,4,3} ILP/greedy | `component_config.py#L51-73` `ComponentQuantizationConfig.resolve`; [blob](https://github.com/vllm-project/vllm-omni/blob/95d56cf/vllm_omni/quantization/component_config.py#L51-L73) | tower-asymmetric ILP 목적함수 + prefix dict 자동생성 | footprint(GB) @ iso-quality, 품질 Δ @ iso-memory | 모든 baseline tower-agnostic; prefix 라우팅에 직접 배선 |
| **M3** | 배포 검증 (iso-memory) | asymmetric vs uniform 동일 메모리 품질/latency 비교 | quant build 진입: `factory.py#L310-338` `_build_component_config`; [blob](https://github.com/vllm-project/vllm-omni/blob/95d56cf/vllm_omni/quantization/factory.py#L310-L338) | 3-mode(VLM/gen/policy) 횡단 검증 | MMMU/VBench/action-MSE, tok/s, J | iso-memory 비교 + tower 별 품질 분리 측정 |

---

## 3. GAP + workload evidence + Baseline Source (R52.1)

### 3.1 GAP — 두 정설이 한 모델 두 tower 에 공존

- **LLM(AR) tower 는 activation outlier 에 민감** → per-channel / salient-scaling 필요 (AWQ [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), SmoothQuant [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)).
- **DiT(generator) tower 는 W4A4 에 상대적으로 내성** → low-rank outlier 흡수로 4-bit 가능 (SVDQuant [arXiv:2411.05007](https://arxiv.org/abs/2411.05007)).
- 이 **두 정설이 같은 모델의 두 tower 에 동시에 적용되어야 하는 첫 사례** 가 Cosmos3 의 dual-tower MoT. 그런데 기존 mixed-precision 은 전부 단일 objective·tower-agnostic.

### 3.2 측정 공백 — co-init 통제 실험의 부재 (★ M1 ④ 의 정면 배치)

- **선점된 상위 명제**: [arXiv:2602.02110](https://arxiv.org/abs/2602.02110) *An Empirical Study of World Model Quantization* (Huawei Noah, 2026-02, overlap ~40%) 가 **DINO-WM 의 encoder vs predictor 모듈 간 quant-sensitivity 가 highly asymmetric** 임을 규명. "world-model 모듈 간 비대칭 quant 민감도"라는 상위 명제를 **이미 선점**. 보조로 [arXiv:2602.11882](https://arxiv.org/abs/2602.11882) *Where Bits Matter in World Model Planning* (DINO-WM uniform/mixed/asymmetric bit, 3-regime 8/6 ok·4 transition·3 collapse).
- **DRIFT 의 차별화 (M1 ④ 핵심)**: 위 선행은 (i) **단일 world-model 내 encoder/predictor** 로, 두 모듈은 *서로 다른 구조·다른 init* 이라 비대칭이 구조·init·objective 가 뒤섞인 confounded 관측. DRIFT 는 (ii) **co-init 된 두 tower** — *동일 init·동일 구조* 에서 출발했으므로 발산이 **순수하게 objective(AR-CE vs flow-MSE) 차이로 귀속되는 controlled natural experiment**. 더하여 (iii) 측정에 그치지 않고 **serving-stack 통합 bit 배분**(M2, `ComponentQuantizationConfig`)까지. 이 두 축(controlled attribution + deployable allocation)이 2602.02110 대비 결정적 차별.
- 일반 mixed-precision 선행(Mix-QViT [arXiv:2501.06357](https://arxiv.org/abs/2501.06357), KL-Lens forward-only sensitivity [arXiv:2604.13440](https://arxiv.org/abs/2604.13440), understanding↔generation MoE Symbiotic-MoE [arXiv:2604.07753](https://arxiv.org/abs/2604.07753))도 전부 tower-agnostic / 단일 objective.

### 3.3 시스템 공백 — 단일 UMA 예산 하 per-tower 배분 부재

- 16B(reasoner 8B + generator 8B 급) FP16 = 32GB. Orin NX(8/16GB)는 INT4 필수, AGX Orin(32/64GB)은 INT8/INT4 현실적 (arch-sys review §1.5). UMA 단일 예산 B 에서 두 tower 가 메모리를 경합 → 한쪽 bit↓ 가 다른쪽 bit↑ 여유를 만듦.
- decode 가 prefill 대비 **192-569× latency 지배** ([arXiv:2511.01866](https://arxiv.org/abs/2511.01866) EdgeReasoning) → reasoner tower 정밀도가 end-to-end 품질의 핵심 lever.

### 3.4 Baseline Source (R52.1) — repo 명시

| Baseline | venue / arXiv | repo | tower 처리 | DRIFT 와의 차이 |
|---|---|---|---|---|
| **SVDQuant** | ICLR'25 [arXiv:2411.05007](https://arxiv.org/abs/2411.05007) | `mit-han-lab/deepcompressor` | generator-tower(DiT)-only W4A4 | ρ_ℓ 의 분모(S_ℓ^G)만 다룸 — reasoner 분자 부재, divergence 개념 없음 |
| **AWQ** | MLSys'24 [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | `mit-han-lab/llm-awq` | whole-model uniform | tower 무구분, salient-channel 만 |
| **GPTQ** | ICLR'23 [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | `IST-DASLab/gptq` | whole-model uniform | 2차근사 column-wise, tower 개념 없음 |
| **HAWQ-계열** | CVPR'19 [arXiv:1905.03696](https://arxiv.org/abs/1905.03696) | `Zhen-Dong/HAWQ` | per-layer Hessian mixed-precision | DRIFT sensitivity 정의의 조상, 단 tower-쌍 divergence 와 co-init attribution 없음 |
| **World-Model Quant** | [arXiv:2602.02110](https://arxiv.org/abs/2602.02110) | (Huawei Noah) | encoder/predictor 모듈 비대칭 | 모듈-비대칭 선행이나 co-init 통제·serving 부재 (§3.2) |
| **BitVLA** | [arXiv:2506.07530](https://arxiv.org/abs/2506.07530) | `ustcwhy/BitVLA` | ternary-all (1.58bit 전체) | tower 무구분 극단 압축, sensitivity-aware 아님 |

> 신규 인접 참고(인용 선택): flow-matching weight-quant 선행 [arXiv:2511.11418](https://arxiv.org/abs/2511.11418) (OT-quant, step-N bound 부재), 다단 추론 누적 noise Quantization Trap [arXiv:2602.13595](https://arxiv.org/abs/2602.13595).

---

## 4. 제안 기법

### 의존성 그래프

```
[HF ckpt: nvidia/Cosmos3-Nano (BF16) 또는 framework state_dict]
        │  (tower prefix 분리: language_model.* / gen_layers.* 또는 *_moe_gen)
        ▼
M1: per-(tower,layer) Hessian-trace S_ℓ^τ(b)  ──┐  (sensitivity 곡선)
    + κ_ℓ^τ, δ_ℓ, corr(δ_ℓ,ρ_ℓ)               │
        │ (ρ_ℓ = S_ℓ^R/S_ℓ^G ≥1.5× material?)  │
        ▼ (gate 통과 시)                         │
M2: ILP/greedy  min Σ S_ℓ^τ(b) s.t. Σ mem(b) ≤ B ◄┘
        │  (bit map → per-prefix quant dict)
        ▼
   build_quant_config({"language_model": {...}, "gen_layers": {...}})
        │  → ComponentQuantizationConfig.resolve(prefix)  (이미 존재)
        ▼
M3: 배포 — asymmetric vs uniform iso-memory (MMMU/VBench/action-MSE, tok/s, J)
```

---

### M1. ρ_ℓ 측정 (흡수된 구 Q4 SIEVE)

**① anchor + 추가물 상세**

- **anchor (verified)**: tower 의 물리적 분리는 `transformer_cosmos3.py#L1058-1111` 에서 확인 — reasoner 는 `self.language_model = Cosmos3LanguageModel(..., prefix="language_model")` (L1058-1071), generator 는 `self.gen_layers = nn.ModuleList([Cosmos3GenDecoderLayer(..., prefix=f"gen_layers.{i}")])` (L1096-1111). **두 생성자가 동일 `hidden_size/num_attention_heads/num_key_value_heads/head_dim/rms_norm_eps` 인자를 받음** → 구조 동일성(controlled experiment 의 전제) 코드로 확정.
  > ✅ source verified: vllm-omni@95d56cf `vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1058-1111` — `language_model`(reasoner) / `gen_layers`(generator) 동일 hyperparameter co-instantiation.
- **framework 측 weight set (verified)**: 학습 체크포인트의 tower 구분은 `cosmos-framework unified_mot.py#L430-495` `PackedAttentionMoT.__init__` 에서 reasoner=`q_proj/k_proj/v_proj/o_proj`(L464-467) vs generator=`q_proj_moe_gen/k_proj_moe_gen/v_proj_moe_gen/o_proj_moe_gen`(L486-495). `MoTDecoderLayer`(L868) 도 `mlp`/`mlp_moe_gen`(L900-904), `input_layernorm`/`input_layernorm_moe_gen`(L906-907) 2벌.
  > ✅ source verified: cosmos-framework@003d66d `cosmos_framework/model/vfm/mot/unified_mot.py#L430-495, L868-909` — `_moe_gen` 접미사 = generator tower weight.
- **ckpt key 매핑 (verified)**: framework `_moe_gen` 이름 → 서빙 prefix 변환은 `pipeline_cosmos3.py#L486-588 _remap_ckpt_key`. 예: `input_layernorm_moe_gen.` → `{gen_lp}.input_layernorm.` (L570), `mlp_moe_gen.gate_proj.` → `{gen_lp}.mlp.gate_proj.` (L577), where `gen_lp = f"transformer.gen_layers.{layer_idx}"` (L550), `und_lp = f"transformer.language_model.layers.{layer_idx}"` (L549). → **ρ_ℓ 측정은 HF/framework state_dict 의 prefix 만으로 두 tower parameter set 을 분리 가능** (모델 forward 불요한 weight-drift 측정의 경우).
  > ✅ source verified: vllm-omni@95d56cf `pipeline_cosmos3.py#L486-588` — `_remap_ckpt_key` framework→serving prefix 변환.
- **추가물**: HAWQ 의 per-layer Hessian trace 에 **tower 축 추가** + **weight-drift δ_ℓ 와의 상관**(co-init 이라 가능). 기존 HAWQ 는 단일 모델 layer-importance 만.

**② 해결 문제**: tower 비대칭 sensitivity 가 init·구조 교란 없이 측정된 적 없음. 2602.02110 은 encoder/predictor(다른 구조·다른 init)라 confounded.

**③ step-by-step (≥5)**

1. **tower parameter set 분리**: state_dict 를 prefix 로 partition — reasoner = `transformer.language_model.layers.{ℓ}.*`, generator = `transformer.gen_layers.{ℓ}.*`. (weight-drift δ_ℓ 는 여기서 즉시 계산; sensitivity 는 forward 필요.)
2. **modality 별 calibration set 구성**: reasoner CE loss 는 text/VLM prompt 128-sample (MMMU/RoboLab instruction subset), generator flow-MSE 는 video/action latent 128-sample (각 modality 별로 분리 calibration — mixed 하면 두 objective gradient 가 섞여 sensitivity 오염). action mode 는 DROID trajectory.
3. **per-(tower,layer) Hessian-trace 추정**: Hutchinson estimator 로 `Tr(H_ℓ^τ) ≈ (1/M) Σ_m vᵀ (H_ℓ^τ v_m)`, v_m ~ Rademacher. HVP 는 `torch.autograd.grad(grad·v, params)` double-backward. reasoner 는 CE loss, generator 는 sampled-timestep flow-MSE loss 에 대해.
4. **sensitivity score**: `S_ℓ^τ(b) = Tr(H_ℓ^τ) · ‖W_ℓ^τ − Q_b(W_ℓ^τ)‖_F²` for b∈{8,6,4,3}. `Q_b` = RTN 또는 AWQ-식 양자화. → bit 별 sensitivity 곡선.
5. **divergence ratio**: `ρ_ℓ = S_ℓ^R(b) / S_ℓ^G(b)` (동일 b 기준). null hypothesis ρ_ℓ ≈ 1 (co-init). per-layer plot.
6. **activation outlier 진단**: tower 별 activation kurtosis `κ_ℓ^τ` + outlier-channel 비율 (|x| > 6σ channel 분포). AR tower 가 더 heavy-tail 일 것이라는 가설.
7. **인과 상관**: `δ_ℓ = ‖W_ℓ^R − W_ℓ^G‖_F / ‖W_init‖_F` 와 ρ_ℓ 의 Pearson/Spearman 상관 + bootstrap CI. "표류한 만큼 민감도 발산하는가" 검증.

**④ 차별화**: 기존(2602.02110, Mix-QViT, KL-Lens)은 tower-agnostic single-objective 또는 다른 구조 두 모듈 → co-init 통제 불가. DRIFT 는 동일 init·동일 구조 두 objective 라 **교란변수 통제 첫 사례** — 발산이 순수 objective 귀속.

**pseudo-code (M1: Hessian-trace 측정 루프 w/ torch.autograd)**

```python
import torch

def tower_hessian_trace(model, loss_fn, layer_params, calib_loader,
                        n_hutchinson=64, device="cuda"):
    """layer_params: dict {layer_idx -> list[Parameter]} for ONE tower.
    loss_fn: tower-specific (reasoner=CE, generator=flow-MSE)."""
    traces = {ℓ: 0.0 for ℓ in layer_params}
    n_batches = 0
    for batch in calib_loader:                       # 128-sample modality-specific
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch.to(device))       # AR-CE or flow-matching MSE
        for ℓ, params in layer_params.items():
            g = torch.autograd.grad(loss, params, create_graph=True,
                                    retain_graph=True)          # 1st-order
            tr = 0.0
            for _ in range(n_hutchinson):
                v = [torch.randint_like(p, 0, 2).mul_(2).sub_(1) for p in params]  # Rademacher
                gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
                Hv = torch.autograd.grad(gv, params, retain_graph=True)            # HVP
                tr += sum((Hi * vi).sum().item() for Hi, vi in zip(Hv, v))
            traces[ℓ] += tr / n_hutchinson
        n_batches += 1
    return {ℓ: t / n_batches for ℓ, t in traces.items()}        # Tr(H_ℓ^τ)

def quant_perturb_fro2(W, bits):                     # ‖W - Q_b(W)‖_F^2  (per-channel RTN)
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=-1, keepdim=True) / qmax
    Wq = (W / scale).round().clamp(-qmax - 1, qmax) * scale
    return (W - Wq).pow(2).sum().item()

# S_ℓ^τ(b) = Tr(H_ℓ^τ) * Σ_{p∈layer ℓ} ‖W_p - Q_b(W_p)‖_F^2
# ρ_ℓ(b)  = S_ℓ^R(b) / S_ℓ^G(b)
```

> 측정 letter scope (Tier-2 fallback): n_hutchinson sweep + bootstrap CI 만으로도 ρ_ℓ/κ/δ 분포 발행 가능 (M2 불요).

**R52.2 7-column 표 (M1)**

| 항목 | 기존(HAWQ/2602.02110) | DRIFT M1 | 출처(verified) | 측정 지표 | 위험 | 완화 |
|---|---|---|---|---|---|---|
| sensitivity 축 | layer / 모듈 | (tower, layer) 쌍 | transformer L1058-1111 | ρ_ℓ 분포 | Hessian 분산 | probe sweep + bootstrap |
| attribution | confounded (구조·init 혼재) | co-init 통제 | unified_mot L430-495 | corr(δ_ℓ,ρ_ℓ) | δ 측정 noise | per-channel norm |
| calibration | 단일 set | modality-분리 set | 본 idea | κ_ℓ^τ gap | mixed 오염 | tower별 loss 분리 |

---

### M2. 예산 제약 asymmetric bit 배분

**① anchor + 추가물 상세**

- **anchor (verified)**: per-prefix → quant-config 라우팅의 정확한 플러밍 지점 = `component_config.py#L51-73` `ComponentQuantizationConfig`. `resolve(prefix)` (L63-73) 가 `self._sorted_prefixes`(길이 내림차순 정렬, L61) 를 순회하며 **longest-prefix match** 로 module 별 다른 `QuantizationConfig` 반환. docstring(L4-9) 예시: `{"transformer": fp8, "vae": None}`.
  > ✅ source verified: vllm-omni@95d56cf `vllm_omni/quantization/component_config.py#L51-73` — `ComponentQuantizationConfig.resolve` longest-prefix → per-tower config 라우팅.
- **build 진입 (verified)**: `factory.py#L310-338 _build_component_config` 가 per-component spec dict → `ComponentQuantizationConfig` 생성. `build_quant_config({"language_model": {"method":"int8"}, "gen_layers": {"method":"mxfp4"}, "default": None})` 형태로 호출 가능. `prefix == "default"` 는 default_config 로 분리(L329-330).
  > ✅ source verified: vllm-omni@95d56cf `vllm_omni/quantization/factory.py#L310-338` — per-prefix dict → ComponentQuantizationConfig 빌더.
- **precedent (verified, 결정적)**: per-tower 라우팅이 *가설이 아니라 작동하는 코드*임의 증거 — `qwen3_omni/qwen3_omni_moe_thinker.py#L1150-1151` 에서 `ComponentQuantizationConfig(component_configs={"language_model": quant_config})` 로 **이미 `language_model` prefix 만 양자화**하는 실사용 존재. DRIFT 는 여기에 `gen_layers` prefix 를 추가해 tower-asymmetric 으로 확장.
  > ✅ source verified: vllm-omni@95d56cf `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py#L1150-1151` — `language_model`-only component routing 실사용 (DRIFT 의 cosmos3 generator-tower 라우팅은 greenfield).
- **사용 가능 quant method (본인 ls/Read 확정)**: `vllm_omni/quantization/factory.py#L142-153` `_OVERRIDES` = `{int8, mxfp8, mxfp4, mxfp4_dualscale, inc(=auto-round/auto_round), gguf}` + vLLM registry(GPTQ/AWQ/fp8 등 35+). 즉 **reasoner tower → AWQ/GPTQ/int8(W8A8), generator tower → mxfp4(W4A4)/mxfp8/int8** 를 prefix 별로 다르게 배선 가능. mxfp4 는 W4A4 MXFP4(E2M1, NPU 경로; `mxfp4_config.py#L3` "W4A4 MXFP4 online/offline").
- **추가물**: tower-asymmetric ILP 목적함수 + ILP 해 → per-prefix dict 자동 생성 어댑터.

**② 해결 문제**: 단일 UMA 예산 B 하 어느 tower/layer 에 bit 를 배분할지 미해결. 균일 정밀도는 ρ_ℓ 발산을 무시 → 비효율 (둔감한 tower 에 과도한 bit).

**③ step-by-step (≥5)**

1. **sensitivity 곡선 입력**: M1 의 `S_ℓ^τ(b)` for τ∈{R,G}, ℓ∈[L], b∈{8,6,4,3}.
2. **메모리 비용 함수**: `mem_ℓ^τ(b) = (#params_ℓ^τ) · b / 8` bytes. cross-attention proj(`gen_layers.{i}.cross_attention`, `transformer_cosmos3.py#L777-810`)는 별도 sensitivity bucket(generator→reasoner 정보 흐름의 병목).
3. **ILP 정식화**: 이진 결정변수 `x_{ℓ,τ,b} ∈ {0,1}` (layer ℓ tower τ 에 bit b 선택).
   - 목적: `min Σ_{ℓ,τ,b} x_{ℓ,τ,b} · S_ℓ^τ(b)`
   - 제약 ①(예산): `Σ_{ℓ,τ,b} x_{ℓ,τ,b} · mem_ℓ^τ(b) ≤ B`
   - 제약 ②(유일 선택): `Σ_b x_{ℓ,τ,b} = 1 ∀(ℓ,τ)`
   - (선택) 제약 ③(인접 bit smoothness): `|b(ℓ) − b(ℓ+1)| ≤ 2` (mixed-precision kernel 단편화 완화).
4. **풀이**: PuLP(CBC) 또는 OR-Tools CP-SAT. 변수 수 ≈ L·2·4 (Nano L≈36 → ~288 binary) — 즉시 해. greedy+Lagrangian fallback: `λ` dual 로 `S + λ·mem` 최소 bit 선택, B 만족까지 λ 이분탐색.
5. **bit map → prefix dict**: ILP 해를 `{"language_model.layers.{ℓ}": {"method": map_to_method(b)}, "gen_layers.{ℓ}": {...}, "default": None}` 으로 변환. `map_to_method`: 8→int8, 4→mxfp4, 6→mxfp8 등.
6. **build & inject**: `build_quant_config(prefix_dict)` → `od_config.quantization_config` 로 주입 (`transformer_cosmos3.py#L1056` `quant_config = getattr(od_config, "quantization_config", None)` 이 ColumnParallelLinear/RowParallelLinear 의 `quant_config=` 로 전파, L482-509, L584-618).

**④ 차별화**: SVDQuant 는 generator-tower-only (ρ 분모만), AWQ/GPTQ 는 whole-model uniform (tower 무구분). DRIFT 는 **단일 예산 하 tower-쌍 sensitivity 곡선을 동시에 풀고** prefix 라우팅에 직배선 — 어떤 baseline 도 두 objective tower 를 한 예산에서 trade-off 하지 않음.

**pseudo-code (M2: ILP 정식화 + solver 호출)**

```python
import pulp

def asymmetric_bit_ilp(S, mem, B, bits=(8, 6, 4, 3)):
    """S[(tau, l, b)] = sensitivity, mem[(tau, l, b)] = bytes, B = budget bytes."""
    prob = pulp.LpProblem("DRIFT_alloc", pulp.LpMinimize)
    x = {(t, l, b): pulp.LpVariable(f"x_{t}_{l}_{b}", cat="Binary")
         for (t, l, b) in S}
    # objective: minimize total sensitivity
    prob += pulp.lpSum(x[k] * S[k] for k in S)
    # constraint 1: budget
    prob += pulp.lpSum(x[k] * mem[k] for k in S) <= B
    # constraint 2: exactly one bit per (tower, layer)
    layers = {(t, l) for (t, l, _) in S}
    for (t, l) in layers:
        prob += pulp.lpSum(x[(t, l, b)] for b in bits if (t, l, b) in S) == 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return {(t, l): b for (t, l, b) in S if x[(t, l, b)].value() == 1}

def to_prefix_dict(alloc, method_of={8: "int8", 6: "mxfp8", 4: "mxfp4", 3: "mxfp4"}):
    pref = {"default": None}
    base = {"R": "language_model.layers", "G": "gen_layers"}
    for (t, l), b in alloc.items():
        pref[f"{base[t]}.{l}"] = {"method": method_of[b]}
    return pref
# from vllm_omni.quantization import build_quant_config
# qcfg = build_quant_config(to_prefix_dict(alloc))   # -> ComponentQuantizationConfig
```

**R52.2 7-column 표 (M2)**

| 항목 | 기존(SVDQuant/AWQ) | DRIFT M2 | 출처(verified) | 측정 지표 | 위험 | 완화 |
|---|---|---|---|---|---|---|
| 배분 단위 | tower-agnostic / DiT-only | per-(tower,layer) | component_config L51-73 | footprint @ iso-quality | ILP 해 단편화 | smoothness 제약 ③ |
| 예산 모델 | 없음(고정 bit) | UMA B 제약 ILP | factory L310-338 | bit map | mixed kernel 비용 | method 정렬 |
| 배선 | 외부 도구 | prefix dict 직배선 | thinker L1150-1151(precedent) | 라우팅 검증 | prefix remap 누락 | resolve docstring L66-69 경고 준수 |

---

### M3. 배포 검증 (iso-memory)

**① anchor + 추가물 상세**

- **anchor (verified)**: 양자화 ckpt 적재·tower 별 weight 매핑 = `pipeline_cosmos3.py#L589-637 load_weights` + `_remap_ckpt_key`(L486-588). quant build 진입 = `factory.py#L341 build_quant_config`. mxfp4_dualscale 은 이미 **per-layer BF16 fallback**(`num_bf16_fallback_layers` 기본 5, `mxfp4_config.py#L18`; offline 은 `ignored_layers` from config.json, L103-134) → mixed-precision 인프라가 부분적으로 존재 (단 tower-aware 아님).
  > ✅ source verified: vllm-omni@95d56cf `mxfp4_config.py#L16-18, L103-134` — `num_bf16_fallback_layers`/`ignored_layers` per-layer BF16 fallback (DRIFT 가 tower-aware 로 확장할 기존 mixed-precision hook).
- **추가물**: **iso-memory 비교 프로토콜** — asymmetric(ILP) 과 uniform 을 *동일 footprint* 로 맞춘 뒤 품질/latency 비교 (uniform 이 같은 B 에서 어떤 균일 bit 가 되는지 계산해 공정 비교).

**② 해결 문제**: bit-alloc 이득이 실배포(iso-memory)에서 유효한지, 3-mode(VLM/gen/policy) 횡단으로 미검증.

**③ step-by-step (≥5)**

1. **iso-memory anchor 계산**: 예산 B 에 대해 uniform bit `b_u` = max{b : L·(p_R+p_G)·b/8 ≤ B}. asymmetric 은 동일 B 의 ILP 해.
2. **양자화 적용**: 두 config 로 각각 `build_quant_config` → `load_weights`.
3. **tower 별 품질 분리 측정**: reasoner 품질 = MMMU + RoboLab instruction-following; generator 품질 = VBench + action-MSE (tower 별 metric 분리 — 한쪽 degrade 가 다른쪽에 가려지지 않도록).
4. **배포 metric**: AGX Orin 64GB 에서 footprint(GB), tok/s, J/inference (tegrastats). RTX Pro 6000 에서 calibration·sensitivity.
5. **conditional-gain 판정**: ρ_ℓ < 1.5× across ≥80% layers → asymmetric 이득 미미 예측 → Tier-2 letter 강등. 발산 클 때만 iso-memory 품질 우위 또는 footprint 절감 보고.

**④ 차별화**: 기존은 단일 task 단일 tower. DRIFT 는 VLM/gen/policy 3 mode 횡단 + iso-memory 공정 비교 + tower 별 분리 품질.

> ✅ source verified: vllm-omni@95d56cf `pipeline_cosmos3.py#L589-637` — `load_weights` tower 별 양자화 weight 적재.

**R52.2 7-column 표 (M3)**

| 항목 | 기존 | DRIFT M3 | 출처(verified) | 측정 지표 | 위험 | 완화 |
|---|---|---|---|---|---|---|
| 비교 기준 | 고정 bit | iso-memory | factory L341 | 품질 Δ @ 동일 GB | uniform 정의 모호 | b_u 명시 계산 |
| 품질 측정 | 통합 단일 | tower 분리 | thinker precedent | MMMU/VBench/MSE 분리 | mode 편향 | 3-mode 횡단 |
| 배포 | 미검증 | Orin 실측 | pipeline L589-637 | GB, tok/s, J | BF16-only fork | 비공식 quant path 명시 |

---

## 5. 평가 · 실험 플랜 (7요소)

### 5.1 HW

- **Calibration / sensitivity 측정**: RTX Pro 6000 Blackwell (Hessian HVP·128-sample calib·full-precision teacher 적재). ncu BW counter 불요(weight-quant 은 메모리-footprint 가 metric).
- **배포 검증**: AGX Orin 64GB (footprint·tok/s·J via tegrastats). Orin NX 8/16GB 는 INT4 footprint fit 데모(stretch). ncu(Nsight Compute)는 Orin(ga10b) 미지원 → Orin 은 wall-clock + tegrastats energy 만.

### 5.2 모델

- **주 모델**: **Cosmos3-Nano** (BF16-only 공식, HF model card). FP8/INT8/INT4/mxfp4 = 전부 **비공식 정밀도 신규도입** → reference path = vllm-omni `Cosmos3OmniDiffusersPipeline` + llm-compressor fork / custom PyTorch RTN·AWQ. upstream 미지원 명시.
- **policy mode**: Cosmos3-Nano-Policy-DROID (action-MSE 품질, N=4-step).
- **fallback**: **BAGEL** — 단, BAGEL 은 **vLLM 미지원 단일 MoT-experts 모델**(14B/7B-active, understanding/generation expert 가 한 backbone 의 두 expert set 이지 별도 co-init tower 아님). 따라서 BAGEL 은 *method 일반성*(sensitivity 측정 파이프라인 이식) 입증용이며, co-init-tower controlled-experiment 의 핵심 주장은 Cosmos3 에 한정 (BAGEL 의 두 expert 는 동일 backbone 공유라 co-init attribution 이 다름 — 이 한계 정직히 명시).

### 5.3 Workload (tower 별 품질 지표 분리)

| mode | reasoner 품질 | generator 품질 |
|---|---|---|
| VLM | MMMU (subset → full) | — |
| gen (T2V/T2I) | — | VBench, GenEval |
| policy | RoboLab-120 instruction-following (subset 20-30 → full) | action-MSE, 성공률 |

- reasoner=instruction-following / MMMU, generator=VBench / action-MSE 로 **분리 측정** (R72 요구). 한쪽 tower degrade 가 통합 metric 에 묻히지 않도록.

### 5.4 Tools

- **sensitivity**: torch `autograd.grad` double-backward Hutchinson HVP (M1 pseudo-code). empirical Fisher fallback.
- **quant**: llm-compressor fork (AWQ/GPTQ for reasoner), vllm-omni native `int8/mxfp4/mxfp8` config (generator), RTN baseline.
- **ILP**: PuLP(CBC) 주, OR-Tools CP-SAT 검증. greedy+Lagrangian fallback.
- **배선**: vllm-omni `build_quant_config` + `ComponentQuantizationConfig`.

### 5.5 Ablation + baseline + kill 조건

- **ablation grid**: bit-grid {8,6,4,3}×2 PTQ method (AWQ-식, RTN) × {uniform, asymmetric} × per-tower. **전수 금지** (16wk gate) — 2 method × {8,4} 핵심 + {6,3} 확장만.
- **baseline**: §3.4 (SVDQuant gen-only / AWQ uniform / GPTQ uniform / World-Model-Quant / BitVLA / HAWQ-uniform-Hessian).
- **conditional-gain kill 조건 (직전 임계값)**: **ρ_ℓ < 1.5× across ≥80% layers → Tier-2 측정 letter 강등** (phase2' Task5 AGREE). 2차 gate: 균일-W4 대비 품질 회복 < 0.5%p 이면 demote.

### 5.6 주차별 표 (12-13주)

| 주차 | 작업 | 파일경로 + API | 완료 판정 |
|---|---|---|---|
| W1-2 | reproduction: cosmos3 BF16 서빙 + tower prefix 검증 | `pipeline_cosmos3.diffuse`, `transformer_cosmos3.forward` | T2V/policy 1 sample 재현 |
| W3-4 | M1 weight-drift δ_ℓ + κ_ℓ^τ (forward-free) | state_dict prefix partition (`_remap_ckpt_key` 매핑) | δ_ℓ, κ 분포 plot |
| W5-6 | M1 Hessian-trace S_ℓ^τ(b) (Hutchinson HVP) | `torch.autograd.grad` double-backward | ρ_ℓ 분포 + corr(δ,ρ) ± CI |
| W7 | gate 판정 (ρ_ℓ ≥1.5× material?) | — | go/Tier-2 결정 |
| W8-9 | M2 ILP/greedy + prefix dict 생성 | `pulp` + `build_quant_config` | bit map → ComponentQuantizationConfig |
| W10 | M3 quant 적용 + load | `factory._build_component_config`, `pipeline.load_weights` | asymmetric ckpt 로드 성공 |
| W11-12 | M3 iso-memory 배포 (Orin) + 3-mode 품질 + baseline | tegrastats, MMMU/VBench/action-MSE | iso-memory 비교 표 |
| W13 | cross-model(BAGEL) 일반성 + conditional-gain 판정 | BAGEL state_dict | 일반성 plot, 최종 verdict |

### 5.7 Preliminary 4단계

1. **reproduction**: vllm-omni cosmos3 BF16 T2V/policy 실행 (`diffuse` 루프, UND-once cache `transformer_cosmos3.py#L1459-1479`).
2. **attribution**: layer 별 quant error 1-shot probe — 단일 layer 를 b-bit 양자화하고 나머지 BF16 유지 시 output Δ (tower 별).
3. **roofline**: 메모리 footprint vs bit 계산식 `mem = L·(p_R·b_R + p_G·b_G)/8`; B 별 uniform b_u vs asymmetric 해 비교.
4. **micro-benchmark**: 단일 layer quant-dequant 품질 probe — `Q_b(W)` 적용 후 layer output MSE (kurtosis 와의 상관 확인).

---

## 6. 예상 효과 / Risk / Tier / Scoring / Decision-tree

### 6.1 예상 효과 (보수, conditional)

- iso-memory 에서 asymmetric 이 uniform 대비 **품질 우위(0.5-1.5%p 회복)** 또는 **동품질 25-40% footprint 절감** — **단 ρ_ℓ 발산이 material(≥1.5× across material layer set)할 때만**.
- 발산 작으면 negative-but-publishable ("균일이 충분" — 측정 자체가 Tier-2 spinoff 발행가치). scope: PTQ 한정 (QAT 제외), Cosmos3-Nano + BAGEL(일반성).

### 6.2 Risk + 완화

| Risk | 완화 |
|---|---|
| ρ_ℓ ≈ 1 (발산 작음) → 이득 미미 | 측정 자체가 기여(Tier-2 letter) + activation/kurtosis 축 확장 |
| **generator HVP 에 reasoner gradient 누설** (generator 출력이 cross-attn 으로 K_AR=reasoner 출력에 의존 → flow-MSE loss 를 그대로 backprop 하면 generator Hessian 에 reasoner graph 가 섞여 co-init 통제 오염) | **reasoner tower 출력(K_AR/V_AR)을 `detach()` 한 뒤** generator flow-MSE HVP 측정 — "reasoner-detach 하 generator HVP" 프로토콜을 §M1-③3 측정에 명시. detach 가 graph 분리=계산 비용↓도 부수 효과(reasoner double-backward 불요) |
| Hessian 추정 분산 大 | Hutchinson M sweep + bootstrap CI + Fisher proxy cross-check |
| BF16-only → 비공식 quant fork 부담 | reference path(Diffusers + llm-compressor) 명시, native int8/mxfp4 우선 |
| 2602.02110 선점 (모듈 비대칭) | co-init controlled attribution + serving 통합으로 차별 (§3.2, M1④) |
| mixed-precision kernel 단편화 → latency 손해 | ILP smoothness 제약 ③ + tower 내 동일 method 정렬 |

### 6.3 R52.4 Tier-A/B/C

- **Tier-A (확실)**: ρ_ℓ/κ_ℓ^τ/δ_ℓ 측정 + corr(δ_ℓ,ρ_ℓ) — co-init controlled experiment. 발산 유무와 무관하게 발행가치.
- **Tier-B (조건부)**: ILP asymmetric 배분이 iso-memory 에서 uniform 우위 — ρ_ℓ 발산 material 시.
- **Tier-C (stretch)**: cross-model(BAGEL) 일반성 + 3-mode 횡단 일관성으로 "법칙" 격상.

### 6.4 Tier-2 variant (측정-only divergence letter)

- **measurement-only co-init divergence letter** — **IEEE CAL** / DAC-short. M2(bit-alloc) 제외, M1 순수 측정만. ρ_ℓ/κ/δ + corr + cross-model(BAGEL) 로 법칙 후보. impl ~6wk, risk 최저. Q1 full 제출 시 motivation 섹션으로, 분리 필요 시 단독.

### 6.5 Scoring 박스

```
DRIFT (Q1) — 직전 avg 7.4
  novelty       7   ▼ (sensitivity mixed-precision 기법 자체는 성숙: Mix-QViT/KL-Lens/2602.02110)
  differentiation 8 ★ (모든 baseline tower-agnostic; co-init ρ_ℓ controlled-attribution 유일)
  impact        8   (UMA 예산 enabling, 3-mode 횡단)
  ai-impl       7   (~13wk, BF16-only fork + Hessian HVP calib)
  arch-impl     7   (prefix 라우팅 이미 존재 — thinker precedent)
  ──────────────────
  avg ≈ 7.4   |  ★diff 8  ▼novelty 7
```

### 6.6 Decision-tree 분기

```
W3-4: δ_ℓ / κ_ℓ^τ 측정
  └─ tower 간 κ gap 명확? ──no──► activation 축 약화, weight-only 로 scope
W7 GATE: ρ_ℓ ≥ 1.5× across ≥80% layers?
  ├─ YES ─► M2 ILP 진행 ─► W11-12 iso-memory 비교
  │           └─ 품질 회복 ≥0.5%p OR footprint −10%? 
  │                 ├─ YES ─► Tier-1 full (NeurIPS/ICML/MLSys)
  │                 └─ NO  ─► 2차 gate fail ─► Tier-2 측정 letter (IEEE CAL)
  └─ NO ──► "균일이 충분" negative ─► Tier-2 measurement-only divergence letter
W13: BAGEL 일반성 재현? ──yes──► "법칙" 격상 (Tier-C)
                          └─no──► Cosmos3 단일-모델 scope 로 정직히 한정
```
