# S2 — DUOCLOCK (Tier-2 #1, deep idea spec)

**DUOCLOCK: Intra-Request Dual-Regime Frequency Scaling for Mixture-of-Transformers Omnimodal Inference on Unified-Memory Edge Devices**

- **Venue 후보**: DATE 2027 / ISLPED 2027 / IISWC 2027 (energy / measurement track)
- **작성일**: 2026-06-05 · **Tier**: 2 (single-insight letter, 단 Tier-1 동일 구현 깊이로 기술 — R28-α)
- **직전 버전**: `2026-06-04-mode2-.../tier2/01-duoclock.md` (avg 7.4, 빈약 — 본 문서가 full-depth 재작성)
- **메타포**: 한 요청 안에서 **두 개의 시계**(GPU clock / EMC clock)를 AR(memory-bound) ↔ DM(compute-bound) regime 에 맞춰 **따로 감는다**. AR phase 는 memory 시계를 최대로·compute 시계를 늦추고, DM phase 는 그 반대로 — 두 시계가 한 inference 내부에서 역할을 주고받는다.
- **Single-insight**: cosmos3 generator 의 **phase 경계가 in-process 로 결정적**(`Cosmos3VFMTransformer.forward` 의 UND→GEN 전환, denoise step loop) → frequency 전환을 **predictor 없이 정확히 그 경계에 건다**. SparseDVFS류가 의존하는 online sparsity/load 예측이 불필요.

---

## 0. 약어 Glossary

- **DVFS** (Dynamic Voltage & Frequency Scaling): 부하에 따라 전압·주파수를 동적 조정해 에너지를 절감하는 기법.
- **EMC** (External Memory Controller): Jetson SoC 의 LPDDR5 메모리 컨트롤러 클럭 도메인. memory bandwidth 를 결정. **GPU clock 과 물리적으로 별개 도메인**.
- **nvpmodel**: Jetson 의 power-mode 관리 CLI (`/usr/sbin/nvpmodel`). CPU online/freq cap, GPU freq cap, EMC MAX_FREQ 를 **프리셋(mode)** 단위로 묶어 설정. `/etc/nvpmodel.conf` 에 정의.
- **jetson_clocks**: 모든 클럭(CPU/GPU/EMC)을 현재 nvpmodel 모드의 **최대값에 고정**하는 스크립트 (DVFS 무력화 = MAXN-pin).
- **devfreq**: Linux 의 device frequency governor 프레임워크. GPU(`17000000.gpu`)는 devfreq 노드로 노출. governor = `userspace`/`nvhost_podgov`(default)/`performance` 등.
- **BPMP** (Boot and Power Management Processor): Tegra 의 전력관리 보조 프로세서 펌웨어. **EMC clock 은 BPMP 가 직접 제어** — 일반 devfreq userspace 노드가 아니라 debugfs(`/sys/kernel/debug/bpmp/...`) 또는 nvpmodel 경유.
- **INA3221**: Jetson devkit 의 3-channel 전류/전압 모니터 IC. rail 별(VDD_GPU_SOC, VDD_CPU_CV 등) 전력을 sysfs 로 노출.
- **J/chunk**: policy(15Hz) 한 chunk(denoise window = 4 denoise step × 2(cond/uncond, CFG ON guidance=3.0) = **8 forward**, ~2.1s 주기) 당 적분 에너지. INA3221 33-50ms 샘플링 < step 이므로 **J/step 은 직접 측정 불가** → 본 idea 의 1차 에너지 단위. (DROID policy 는 CFG ON → 매 denoise step 당 cond/uncond 2 forward.)
- **J/phase**: AR-prefill / DM-denoise-window / VAE 구간별 적분 에너지(T2I/T2V gen 측).
- **memory-bound / compute-bound**: roofline 상 성능이 메모리 BW 에 의해 제한되면 memory-bound(주파수↓해도 latency 둔감), FLOP 에 의해 제한되면 compute-bound.
- **roofline**: operational intensity(FLOP/byte) 대 attainable performance 그래프. ridge point 좌측=memory-bound, 우측=compute-bound.
- **settle time**: frequency set 명령 발행 후 클럭/전압이 목표값에 안정화될 때까지의 latency(수 ms~수십 ms). 전환 가치를 잠식하는 핵심 비용.
- **AR / DM**: AR = autoregressive understanding tower(reasoner, `language_model`/UND pass). DM = diffusion-model generation tower(generator, `gen_layers` denoise step loop).
- **MoT** (Mixture-of-Transformers): modality-disjoint dual-tower(reasoner/generator) 아키텍처. cosmos3 가 대표.
- **possible_rates / mrq_rate_locked**: BPMP EMC debugfs 의 허용 rate 목록 / rate lock flag.

---

## 1. 개요 — metaphor · single-insight RQ · 왜 Tier-2 인가

### 1.1 문제와 메타포
Cosmos3 omnimodal MoT 추론은 **한 요청 안에서** 두 regime 을 직렬로 통과한다.
- **AR phase** (reasoner UND pass): 텍스트/조건 토큰을 한 번 forward → K/V 캐시. token-by-token 성격이 강하고 KV·weight 재적재가 지배적인 **memory-bound** 구간.
- **DM phase** (generator denoise loop): 동일 K_AR 를 step 마다 재사용하며 full-sequence latent 를 반복 갱신하는 **compute-bound** 구간.

두 regime 의 frequency 요구는 **정반대**다 (§3 evidence). 그런데 Jetson 은 한 시점에 단일 nvpmodel 모드로 동작 → 두 phase 를 **동시에 최적화할 수 없다**. DUOCLOCK 은 GPU clock 과 EMC clock 두 시계를 phase 경계에서 따로 감아 각 regime 에 맞춘다.

### 1.2 Single-insight RQ
> **RQ**: cosmos3 generator 의 phase 경계가 in-process 로 100% 결정적이라는 사실을 이용하면, online sparsity/load **predictor 없이** frequency 를 정확히 그 경계에서 전환하여, J/chunk(policy)·J/phase(gen)를 latency 손실 최소로 절감할 수 있는가? 그리고 그 절감이 **freq settle time** 에 의해 잠식되지 않는 phase-길이 경계 조건은 무엇인가?

phase 경계가 결정적이므로(코드 위치가 `forward` 내부에 고정), DUOCLOCK 의 governor 는 "예측"이 아니라 "통지(notify)" 만 받는다 — 이것이 기존 DVFS governor 와의 구조적 차이다.

### 1.3 왜 Tier-2 인가 (정직한 순증분 명시)
- **SparseDVFS [arXiv:2603.21908](https://arxiv.org/abs/2603.21908)** 가 이미 **CPU/GPU/EMC freq triplet + edge(Jetson) + operator-granular(intra-inference) DVFS + latency amortization 제약**까지 도달 → DUOCLOCK 의 "EMC/GPU 분리 제어"(과거 novelty 축 (b))는 **선점됨**.
- 따라서 **남는 순증분은 정확히 세 가지뿐**이며, 본 문서는 이것만 주장한다:
  1. **intra-request modality-regime 전환(AR↔DM)**: SparseDVFS 는 operator-sparsity 기반이지 *modality regime* 전환이 아니다. DUOCLOCK 은 reasoner↔generator tower 전환을 freq plan 의 단위로 삼는다.
  2. **phase-결정성(predictor 불필요)**: SparseDVFS/DualScale/GreenLLM 은 load/sparsity 추정을 동반한다. DUOCLOCK 의 경계는 코드상 결정적이라 settle time 을 phase-길이에 맞춰 **선제(prefetch-style)** 배치할 수 있다.
  3. **MoT omnimodal workload 최초**: 동일 edge GPU 에서 AR→DM(diffusion denoise 포함)을 직렬 traverse 하는 omnimodal MoT 에 DVFS 를 적용·측정한 최초 사례.
- novelty(b) 선점 + DVFS governor 기술 성숙 → **measurement-결합 single-insight letter** 가 정직한 포지션. (Tier-1 승격 조건은 §6.4.)

---

## 2. Mechanism Summary Table (R70 7열)

| # | mechanism | 추가 scheme(무엇을 새로 넣나) | 해결 problem | 핵심 동작(1줄) | 신규/기존 | venue 기여 |
|---|---|---|---|---|---|---|
| **M1** | Phase-Anchored Frequency Plan | phase enum→{GPU freq, EMC 등급} 정책 LUT + offline roofline 캘리브레이션 + in-process 경계 통지 hook + userspace governor daemon | 단일 nvpmodel 모드가 상반 regime(AR memory-bound / DM compute-bound)을 동시 최적화 불가 | `forward` UND→GEN 경계 / denoise step 콜백에서 daemon 에 phase 신호 → daemon 이 GPU devfreq + EMC(debugfs/nvpmodel) 즉시 write | hook=verified 지점([transformer_cosmos3.py#L1459-1500](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1500), [pipeline_cosmos3.py#L1492](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1492)), daemon/LUT=신규(proposed) | edge MoT 의 phase-결정 DVFS governor 설계 |
| **M2** | J/chunk Measurement Ledger | INA3221 rail 적분 ⨉ `stage_durations` 정렬 + chunk(policy)/phase(gen) 회계 + freq settle-time kill 경계 도출 | INA3221 33-50ms > step → J/step 불가능, edge MoT 에너지 회계·전환 가치 판정 공백 | rail power 시계열을 phase boundary timestamp 로 분할·적분, settle > phase 길이면 "전환 무가치" kill | 측정 방법론 신규(proposed), `stage_durations`=verified([diffusion_pipeline_profiler.py#L80-113](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py#L80-L113)) | edge omnimodal 에너지 측정 방법론 + transition-value 경계 |

> Tier-2 single-insight 원칙에 따라 **2 mechanism 만**. M1 = 정책/제어, M2 = 측정/검증(M1 의 LUT 산출 및 kill-gate 제공). M2 는 M1 의 producer.

---

## 3. GAP + Workload Evidence + Baseline Source (R52.1)

### 3.1 GAP
- **G1 (dual-regime 동시최적 불가)**: 한 요청이 AR(memory-bound)→DM(compute-bound)을 직렬 통과하는데, Jetson 은 시점당 단일 nvpmodel 모드 → 두 regime 중 하나만 최적. 둘 다 MAXN 으로 두면 AR 구간 GPU freq 가 낭비(energy↑), 둘 다 절전 모드로 두면 DM 구간 latency↑.
- **G2 (측정 공백)**: edge omnimodal MoT 의 phase별 에너지(특히 diffusion denoise 의 EMC-둔감성)는 미측정. tegrastats/INA3221 샘플링 한계로 J/step 직접 측정도 불가 → 측정 단위 재정의 자체가 공백.

### 3.2 Workload Evidence (수치 + clickable)
- **decode 가 inference time 의 77-91% 지배** ([arXiv:2501.08219](https://arxiv.org/abs/2501.08219)) — AR/memory-bound 구간의 시간 비중이 크다 = freq plan 의 ROI 근거.
- **GPU freq 2842→180MHz: 에너지 42%↓, latency 1-6%↑** (decode 가 frequency 에 둔감, [arXiv:2501.08219](https://arxiv.org/abs/2501.08219)) — memory-bound phase 에서 GPU freq 를 깊게 내려도 latency 손실이 작다 = AR phase GPU↓의 직접 근거.
- **DynamoLLM 에너지 52%↓** (cluster GPU-freq, HPCA'25 Best Paper, [arXiv:2408.00741](https://arxiv.org/abs/2408.00741)) — phase-aware freq 의 절감 잠재력 상한 참고(단 cluster/across-request).
- **AGX Orin decode 가 prefill 대비 192-569× latency 지배** (EdgeReasoning IISWC'25, [arXiv:2511.01866](https://arxiv.org/abs/2511.01866)); AGX Orin LPDDR5 **204.8 GB/s**, 8B decode **~7.8 tok/s** — Orin 에서 AR 구간이 memory-bound 임을 실측 근거로 제공.
- **DM denoise = compute-bound** (full-sequence 반복, KV 불변, ISPASS'24 [arXiv:2312.14385](https://arxiv.org/abs/2312.14385)) — DM phase 에서 GPU max + EMC↓의 근거.
- **memory freq 하향 시 decode latency +370% / energy +72%(power −52%)** (PAISE'25 류) — memory phase 와 compute phase 의 freq 요구가 **정반대**임을 정량적으로 못박는 핵심 근거.
- **policy denoise window = 4 denoise step × 2(cond/uncond) = 8 forward/chunk @ 15Hz (~2.1s 주기)**, forward ~263ms (tech report Sec.4.2.5; DROID policy **CFG ON, guidance scale 3.0** → step 당 cond/uncond 2 forward) → **J/chunk 직접 측정 가능 영역**(263ms ≫ 50ms 샘플링). DM phase 의 forward 수(전력/J-chunk 산정의 분모)는 **8** 로 일관 적용(4 step 이 아님).
- **SparseDVFS 의 latency amortization constraint** ([arXiv:2603.21908](https://arxiv.org/abs/2603.21908)) — 전환 페널티가 이득을 상쇄하지 않도록 하는 제약. DUOCLOCK 은 이를 **settle-time kill 경계**로 정식화(M2)하되, phase-결정성으로 settle 을 선제 배치한다.

### 3.3 Baseline Source (R52.1) — 4종, 각 실행/설정 명령
모두 동일 워크로드(policy 200 chunk / T2I 100장)에서 J/chunk·latency 비교.

- **(a) MAXN 고정** — 모든 클럭을 최대 고정(DVFS 무력화, energy upper-bound).
  ```bash
  sudo /usr/sbin/nvpmodel -m 0           # MAXN mode
  sudo /usr/bin/jetson_clocks            # CPU/GPU/EMC 를 mode-max 에 pin
  sudo /usr/bin/jetson_clocks --show     # 고정 확인
  ```
- **(b) nvpmodel 프리셋 고정 3종** — `MAXN`(0), 중간 전력(예 30W=`-m 1`), 저전력(예 15W=`-m 2`)을 **요청 전 구간 단일 고정**(intra-request 전환 없음 = DUOCLOCK 의 대조).
  ```bash
  for M in 0 1 2; do sudo /usr/sbin/nvpmodel -m $M; sudo /usr/sbin/nvpmodel -q; \
    python bench_cosmos3.py --workload policy --chunks 200 --tag preset_$M; done
  ```
- **(c) SparseDVFS-style operator-granular** — 재현 가능 범위만: actmon/load 기반 GPU+EMC 동적 스케일을 **operator 입도**로 근사. 정확 재현 불가(원 구현 비공개·sparsity hook 필요)이므로 **devfreq default governor(`nvhost_podgov`) + EMC actmon DFS on** = "load-반응형 동적 DVFS" 로 근사하고 그 한계를 명시.
  ```bash
  echo nvhost_podgov | sudo tee /sys/class/devfreq/17000000.gpu/governor   # GPU load-반응
  # EMC actmon DFS(BPMP 기본) 유지: mrq_rate_locked 미설정(=동적)
  cat /sys/kernel/debug/bpmp/debug/clk/emc/rate                            # 동적 변동 확인
  ```
  > 재현 한계 명시: SparseDVFS 의 operator-sparsity 측 신호는 본 워크로드(BF16, sparsity hook 없음)에서 미적용 → "load-반응 DVFS" 상한 근사로만 비교. 순증분 gate(§6.5)에서 이 baseline 대비 차이를 핵심 판정값으로 사용.
- **(d) ondemand devfreq governor** — GPU devfreq 를 표준 load-반응 governor 로(EMC 는 BPMP 기본 DFS).
  ```bash
  echo userspace | sudo tee /sys/class/devfreq/17000000.gpu/governor     # 비교 전 reset
  echo nvhost_podgov | sudo tee /sys/class/devfreq/17000000.gpu/governor # Orin GPU 표준 동적 governor
  ```
  > 주의: Orin GPU devfreq 의 표준 동적 governor 명칭은 `nvhost_podgov`(L4T). 일반 `ondemand`/`simple_ondemand` 는 GPU 노드에 미노출일 수 있어 `available_governors` 로 확인 후 가용한 load-반응 governor 채택.

---

## 4. 제안 기법

### M1 — Phase-Anchored Frequency Plan

**① 추가 scheme**: phase enum(`AR_UND`, `DM_DENOISE`, `VAE_DECODE`)별 `{GPU freq, EMC 등급}` 정책 LUT 를 offline 캘리브레이션으로 산출하고, `forward`/`diffuse` 의 **in-process 경계**에서 userspace governor daemon 에 phase 신호를 보내 GPU devfreq + EMC(debugfs 또는 nvpmodel) 를 즉시 write. predictor 없음 — 경계가 결정적.

**② 해결 problem**: G1 — 단일 nvpmodel 모드가 상반 regime 동시 최적 불가. AR phase 에서 GPU freq↓로 42%급 절감(latency 1-6%만 손실), DM phase 에서 EMC↓로 memory power 절감(compute-bound 이므로 BW 둔감).

**③ 동작 원리 (≥5 step, sysfs 경로/명령 포함)**:
1. **(offline 캘리브레이션)** phase별 roofline 측정 → 각 phase 에서 latency 손실 <5% 를 지키는 최저 freq 선정. GPU 가용 freq 목록:
   ```bash
   cat /sys/class/devfreq/17000000.gpu/available_frequencies   # 예: 306000000 ... 1300500000
   cat /sys/kernel/debug/bpmp/debug/clk/emc/possible_rates      # EMC 허용 rate (예: ...665600000 2133000000)
   ```
2. **(governor 준비)** GPU devfreq 를 userspace 로:
   ```bash
   echo userspace | sudo tee /sys/class/devfreq/17000000.gpu/governor
   # JP5 경로 fallback: /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/governor
   ```
3. **(AR phase 진입)** `forward` UND pass 직전 daemon 이 신호 수신 → GPU 낮춤 + EMC max:
   ```bash
   echo 612000000 | sudo tee /sys/class/devfreq/17000000.gpu/userspace/set_freq   # GPU ↓ (memory-bound)
   sudo sh -c "echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked"
   sudo sh -c "echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/state"
   sudo sh -c "echo 2133000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate"      # EMC max (BW 확보)
   ```
4. **(DM phase 진입)** UND→GEN 경계(코드 결정적)에서 daemon 이 정반대로 set — GPU max + EMC↓:
   ```bash
   echo 1300500000 | sudo tee /sys/class/devfreq/17000000.gpu/userspace/set_freq # GPU max (compute-bound)
   sudo sh -c "echo 1331200000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate"      # EMC ↓ (BW 둔감)
   ```
5. **(전환 granularity = chunk/phase 단위)** policy 는 **chunk(2.1s) 단위**로만 전환(step 마다 X) → settle cost 흡수 + 15Hz 주기 내 잦은 전환 회피. gen(T2I)은 prefill→denoise→VAE phase 단위.
6. **(EMC 직접 제어 불가 시 fallback)** debugfs EMC write 가 막힌 환경(보안/커널 옵션)이면 **nvpmodel 프리셋 조합**으로 대체: AR-친화 모드 / DM-친화 모드를 `/etc/nvpmodel.conf` 에 정의(GPU/EMC MAX_FREQ 조합)하고 `nvpmodel -m` 으로 전환(settle 더 큼 → §M2 gate 로 평가).

**Pseudo-code — userspace governor daemon loop (≥12줄, phase IPC/소켓 신호 수신→freq write)**:
```python
# duoclock_governor.py  (proposed) — root 권한 daemon
import socket, json
GPU = "/sys/class/devfreq/17000000.gpu/userspace/set_freq"
EMC_RATE = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"
LUT = {  # offline 캘리브레이션 산출 (phase -> (gpu_hz, emc_hz)); <5% latency 손실 보장값
    "AR_UND":     (612_000_000, 2_133_000_000),   # GPU↓ EMC max
    "DM_DENOISE": (1_300_500_000, 1_331_200_000), # GPU max EMC↓
    "VAE_DECODE": (1_300_500_000, 2_133_000_000), # 둘 다 高 (짧고 BW+compute 모두 부하)
}
def write(path, val):
    with open(path, "w") as f: f.write(str(val))
def apply(phase):
    gpu, emc = LUT[phase]
    write(GPU, gpu); write(EMC_RATE, emc)          # 두 시계를 따로 감는다
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind("/run/duoclock.sock")                     # in-process hook 이 phase 를 datagram 으로 통지
# EMC lock 1회 사전 설정 (mrq_rate_locked=1, state=1) 은 daemon init 에서 수행
while True:
    msg, _ = sock.recvfrom(64)                      # blocking; predictor 없음 — 통지만 수신
    ev = json.loads(msg)                            # {"phase":"DM_DENOISE","req":123,"ts":...}
    apply(ev["phase"])                              # settle time 은 M2 ledger 가 별도 계측
```

**in-process hook (경계 통지) — `Cosmos3VFMTransformer.forward` 측**:
```python
# transformer_cosmos3.py forward() 내, UND→GEN 경계(현 L1480 부근)에 삽입 (proposed)
if self.cached_kv is None:
    notify_phase("AR_UND")                          # UND pass 직전 (현 L1459-1474 사이)
    cached_kv_full = self.language_model(text_ids, freqs_und)
    ...
notify_phase("DM_DENOISE")                           # GEN layers 직전 (현 L1499 직전)
for layer, (k_und, v_und) in zip(self.gen_layers, self.cached_kv, strict=True):
    ...
```

> ✅ source verified: vllm-omni@`95d56cf` [`vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-1500`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1500) — `if self.cached_kv is None:` 블록에서 `language_model(...)` 으로 UND K/V 1회 계산(L1474), 직후 `for layer ... in zip(self.gen_layers, self.cached_kv ...)` 으로 GEN denoise 진입(L1499-1507). **UND→GEN 경계가 단일 forward 내부에 결정적으로 존재** = freq-transition hook 지점.
> ✅ source verified: vllm-omni@`95d56cf` [`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1282,L1332,L1492`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1282) — `diffuse()` 진입([L1282](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1282)), `self.transformer.reset_cache()`([L1332](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1332)), CFG-sequential denoise step loop `for t in self.progress_bar(timesteps)`(L1454/[L1492](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1492)/L1533) — step 단위 콜백 지점(VAE_DECODE 전환은 loop 종료 후).
> `notify_phase()` / `duoclock_governor.py` daemon / Unix-socket IPC / phase-LUT = **(proposed)** — 현 코드에 DVFS/freq 제어 전무(grep 0건, greenfield).

**R52.2 표 (Phase1' → 본 문서 변경)**:

| 항목 | Phase1'(refined spec) | 본 문서 | 종류 | 사유 |
|---|---|---|---|---|
| M1 제어 입도 | "chunk 단위 전환" 서술 | chunk(policy)/phase(gen) + **결정적 경계 통지(predictor 없음)** 명시 | improve | single-insight=phase 결정성 강조 |
| EMC 제어 knob | `/sys/class/devfreq/emc/...` (부정확) | **BPMP debugfs `clk/emc/{mrq_rate_locked,state,rate}` + nvpmodel fallback** | replace | EMC 는 devfreq userspace 노드 아님 — web 검증 결과 정정 |
| hook 지점 | "S1 TowerPhaseFSM 재사용" | 동일 + **본 [forward L1480](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1480) / [diffuse L1492](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1492) 직접 anchor** | improve | self-verify(R72.3) |

**의존성 그래프 (M1)**:
```
[offline roofline 캘리브레이션] ──→ [phase-LUT {gpu,emc}]
        (M2 J/chunk ledger 가 측정 제공)            │
[forward UND→GEN 경계 hook]──notify──→[duoclock daemon]──write──→ GPU devfreq + EMC debugfs/nvpmodel
[diffuse step loop 콜백]──────────────────┘                              │
                                                          [settle-time kill gate (M2)] ←─ 검증/되먹임
```

---

### M2 — J/chunk Measurement Ledger

**① 추가 scheme**: INA3221 rail 전력 시계열을 적분하고, `DiffusionPipelineProfilerMixin.stage_durations` 의 phase boundary timestamp 와 정렬하여 **chunk 단위(policy)·phase 단위(gen)** 에너지를 회계. 동시에 **freq settle time** 을 단독 micro-benchmark 로 실측하여, settle 이 phase 길이를 잠식하는 **kill 경계 조건**을 도출.

**② 해결 problem**: G2 — INA3221 33-50ms > step 이라 J/step 직접 불가 → chunk/phase 단위로 재정의해야 측정 가능. 또한 전환의 가치가 settle time 에 의해 음(-)이 되는 경계가 미지 → 정량화 필요(kill 조건).

**③ 동작 원리 (≥5 step, sysfs 경로/명령 포함)**:
1. **(rail 식별)** AGX Orin devkit INA3221 rail sysfs 확인(rail 명칭은 carrier/JP 버전별 상이):
   ```bash
   grep -r . /sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/ 2>/dev/null | grep -E "in[0-9]_label|curr"
   # 예: VDD_GPU_SOC, VDD_CPU_CV — in*_input(mV) × curr*_input(mA) = power(mW)
   ```
2. **(고속 polling harness)** INA3221 를 ~1-2kHz 로 polling(샘플 ~0.5-1ms)하여 power(t) 수집 — tegrastats(20-30Hz)보다 fine. 단 phase boundary 와 동기 위해 monotonic clock 사용.
3. **(boundary 정렬)** `stage_durations` 와 hook 의 phase timestamp 를 같은 clock 으로 기록 → power(t) 를 phase 구간으로 분할.
   ```python
   E_phase = trapz(power[t0:t1], time[t0:t1])   # J = ∫ P dt over [phase_enter, phase_exit]
   ```
4. **(회계 입도)** policy: forward(263ms) 직접 측정 → **J/chunk = Σ_{8 forward} E + transition E**. gen(T2I): step<50ms → **N-step 적분**으로 J/DM-window, 별도 J/AR-prefill, J/VAE.
5. **(settle time 단독 측정)** freq write 발행 시각 ~ cur_freq 가 목표 도달 시각 차이를 micro-bench:
   ```bash
   echo <target> | sudo tee /sys/class/devfreq/17000000.gpu/userspace/set_freq; \
   while :; do cat /sys/class/devfreq/17000000.gpu/cur_freq; done   # 도달 시각 관측
   # EMC: cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 반복
   ```
6. **(kill 경계 도출)** 전환 이득 `ΔE_save` 와 settle 동안의 손실 `E_settle(+latency)` 비교. **settle_time > phase_length 이면 그 phase 의 전환은 무가치** → 해당 phase 는 freq 전환에서 제외(kill). policy chunk(2.1s) ≫ settle(수~십 ms) → 정상; 단 step(66ms)·VAE(짧음)·nvpmodel-fallback(settle 큼) 은 gate 통과 여부를 실측으로 판정.

**Pseudo-code — 에너지 적분 정렬 코드 (≥1)**:
```python
# duoclock_ledger.py (proposed) — INA3221 적분 ⨉ stage_durations 정렬
import numpy as np
def integrate_phase(power_w, t_s, t_enter, t_exit):      # trapezoid J over [t_enter,t_exit]
    m = (t_s >= t_enter) & (t_s <= t_exit)
    return float(np.trapz(power_w[m], t_s[m]))           # Joules
def jchunk(power_w, t_s, boundaries, settle_log):        # boundaries: [(phase,t0,t1),...]
    rows, e_total = [], 0.0
    for phase, t0, t1 in boundaries:
        e = integrate_phase(power_w, t_s, t0, t1); e_total += e
        rows.append((phase, t1 - t0, e))                 # (phase, dur_s, J)
    # settle-time kill gate: 전환 이득 vs settle 손실
    for phase, dur, e in rows:
        st = settle_log.get(phase, 0.0)                  # 실측 settle (s)
        if st > dur:                                     # settle > phase 길이 → 전환 무가치
            print(f"[KILL] {phase}: settle {st*1e3:.1f}ms > phase {dur*1e3:.1f}ms")
    return e_total, rows                                  # J/chunk = e_total
```

> ✅ source verified: vllm-omni@`95d56cf` [`vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py#L80-113`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py#L80-L113) — `DiffusionPipelineProfilerMixin` 의 `_PROFILER_TARGETS = ["vae.encode","vae.decode","diffuse","text_encoder.forward","tokenizer.forward"]`(L81), `stage_durations` property([L106-109](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py#L106-L109)) 가 stage(=phase)별 duration dict 반환 = J/phase 정렬의 기존 timing infra 통합 지점.
> INA3221 1-2kHz polling harness / settle-time micro-bench / kill-gate = **(proposed)** — 현 코드에 전력측정·DVFS 전무.

**R52.2 표 (M2)**:

| 항목 | Phase1' | 본 문서 | 종류 | 사유 |
|---|---|---|---|---|
| 측정 단위 | J/chunk + J/phase | 동일 + **stage_durations 정렬 코드·polling rate 구체화** | improve | 구현 깊이(R28-α) |
| settle 처리 | "settle LUT gate" 1줄 | **kill 경계 조건 정식화 + micro-bench 절차** | add | single-insight=settle 경계가 idea 생존선 |

**의존성 그래프 (M2)**:
```
[INA3221 1-2kHz polling] ─┐
[stage_durations timestamp]─┼→ [phase 분할 적분] → [J/chunk, J/phase ledger] → M1 phase-LUT(캘리브레이션)
[freq settle micro-bench] ─┘                                   └→ [settle>phase kill-gate] → §6.5 decision-tree
```

---

## 5. 평가·실험 플랜 (7요소)

### 5.1 HW
- **AGX Orin 64GB devkit (1순위)** — INA3221 rail 가용(devkit 만 board-level rail 노출), EMC/GPU freq 자유도 큼, GPU `17000000.gpu` devfreq + EMC BPMP debugfs 접근 가능. ncu 불가는 본 idea 무관(전력/wall-clock 측정만 필요).
- **Orin NX 16GB (보조)** — 102.4 GB/s, 저전력 envelope 에서 freq plan ROI 재확인. (INA3221 가용성 carrier-dependent → 확인 후 사용.)
- **Thor (conditional)** — 차세대 freq governor·power rail 구조가 다름. 가용 시 세대 비교, 불가 시 제외(본 idea 핵심 결과는 Orin 으로 완결).

### 5.2 모델
- **Cosmos3-Nano-Policy-DROID (1순위)** — phase 가 15Hz 로 **반복**되어(연속 chunk) 통계 풍부, J/chunk 직접 측정 가능(forward 263ms). **DROID policy 는 CFG ON (guidance scale 3.0** — cosmos-framework `action_policy_robolab_server.md`·tech report L2007; guidance=1.0 은 forward/inverse-dynamics·padding pass 용) → denoise phase forward 수 = **4 step × 2(cond/uncond) = 8 forward/chunk**, K_AR cache **2벌(cond/uncond)**. (guidance=1.0 가정의 "CFG off·1벌" 서술은 오기 — 정정.)
- **Cosmos3-Nano T2I (2순위)** — prefill→denoise→VAE phase 분리 명확, J/phase 측정용. (BF16-only 공식 정밀도 — quant orthogonal.)

### 5.3 Workload
- policy: **연속 200 chunk** (≈420s, chunk 당 8 forward) — chunk 간 분산으로 J/chunk 통계.
- T2I: **100장** (각 num_steps≈30) — J/AR-prefill / J/DM-window / J/VAE 분해.

### 5.4 Tools
- **tegrastats** (EMC%/전력 coarse, 20-30Hz) — 보조·sanity.
- **jtop** (jetson-stats) — 실시간 freq/power/온도 모니터·로깅.
- **custom INA3221 polling** (1-2kHz, §M2) — J 적분 주력.
- **Nsight Systems** (timeline/range marker) — phase boundary 시각화·hook 정렬 검증(ncu 불요).

### 5.5 Ablation
- **freq plan ⨉ workload 2종**: `{(a)MAXN, (b)preset×3, M1 phase-anchored}` × `{policy, T2I}`.
- **metrics**: **J/chunk**(policy), **J/phase**(T2I), **chunk p99 latency**, **품질 무손실 확인**(policy=action MSE vs MAXN baseline 동일성, T2I=동일 seed 픽셀 동일성 — DVFS 는 결과 불변이어야 함 = correctness gate), **freq settle time**(단독).
- **예상 측정 일정**: HW/harness(W1-2) → 캘리브레이션 sweep(W3-5) → settle+정책(W6-7) → ablation 본측정(W8-9, 각 condition 3-반복) → 분석(W10).

### 5.6 주차별 표 (~10주, 파일경로 + 완료판정)

| 주차 | 작업 | 산출 파일 | 완료판정 |
|---|---|---|---|
| W1-2 | GPU devfreq userspace + EMC debugfs(`mrq_rate_locked/state/rate`) write 검증; INA3221 1-2kHz polling harness | `duoclock_ledger.py`, `setup_clocks.sh` | `cur_freq`/`emc/rate` 가 set 값 추종 확인 + power(t) 로깅 |
| W3 | in-process hook: `transformer_cosmos3.py` UND→GEN 경계 + `pipeline_cosmos3.py diffuse` step 콜백 → Unix-socket notify | `patches/duoclock_hooks.py` | Nsight Systems range 가 hook 시점과 일치 |
| W4-5 | offline roofline 캘리브레이션: phase별 freq sweep → <5% latency 손실 최저 freq → phase-LUT | `calib/phase_roofline.csv`, `phase_lut.json` | phase별 (gpu,emc) 최저 freq 확정 |
| W6 | `duoclock_governor.py` daemon: notify 수신→freq write; chunk/phase granularity | `duoclock_governor.py` | daemon 이 200 chunk 동안 무누락 전환 |
| W7 | freq settle time 단독 micro-bench(GPU/EMC/nvpmodel-fallback) + kill-gate | `settle_bench.csv` | settle(ms) 분포 + phase별 gate 판정 |
| W8-9 | ablation 본측정: {MAXN,preset×3,M1}×{policy,T2I}, 각 3-반복 + 품질 무손실 gate | `results/*.parquet` | J/chunk·p99·품질 동일성 표 완성 |
| W10 | baseline(a-d) 비교·SparseDVFS-style 순증분 산정·decision-tree 판정 | `analysis.ipynb`, draft | §6.5 분기 결정 + letter draft |

### 5.7 Preliminary 4단계 (각 도구+명령+기대값)
1. **Reproduction** — MAXN 에서 policy 200 chunk 의 J/chunk·p99 기저선 확보.
   - 도구/명령: `nvpmodel -m 0 && jetson_clocks` + INA3221 polling + `bench_cosmos3.py --workload policy --chunks 200`.
   - 기대값: chunk forward ~263ms 재현, J/chunk 안정(변동계수 <10%).
2. **Attribution (phase별 power trace 분해)** — UND vs GEN vs VAE 의 power(t) 분리.
   - 도구/명령: hook notify + `duoclock_ledger.py integrate_phase`.
   - 기대값: AR(UND)에서 GPU rail 저부하·EMC 고부하, DM(GEN)에서 GPU 고부하 — regime 비대칭이 trace 에 보일 것.
3. **Roofline (phase별 op-intensity 계산)** — FLOP/byte 추정으로 memory/compute-bound 분류.
   - 도구/명령: forward FLOP(모델 config) ÷ DRAM bytes(weight+KV+latent) → ridge point 대비 위치; GPU freq sweep 시 latency 기울기로 교차검증.
   - 기대값: AR < ridge(memory-bound, freq 둔감), DM > ridge(compute-bound, freq 민감).
4. **Micro-benchmark (freq settle time 단독)** — set→도달 latency.
   - 도구/명령: `settle_bench.csv` 절차(§M2-③5).
   - 기대값: GPU devfreq settle 수 ms~수십 ms, EMC debugfs 유사, **nvpmodel 모드 전환은 더 큼**(수십~수백 ms 가능) → fallback gate 판단 근거.

---

## 6. 예상 효과 · Risk · Tier 등급 · Scoring · Decision-tree

### 6.1 예상 효과 (보수)
- **J/chunk 15-30%↓ @ chunk p99 latency +<5%** — **조건부**: 절감폭은 AR phase 의 시간 비중(decode 77-91% evidence)과 GPU freq 둔감성(2842→180MHz 시 42% energy↓/1-6% latency↑)에 비례. 단순 phase별 절감 **합산 아님**(상호작용·VAE·transition overhead 차감 후 보수치).
- **settle-time 경계 조건**: policy chunk(2.1s) ≫ settle(수~십 ms) → 전환 가치 충분. 단 step(66ms)·짧은 VAE·nvpmodel-fallback 은 settle 비중이 커 **전환 제외 가능**(M2 kill-gate). 이 경계가 idea 생존선.

### 6.2 Risk + 완화
- **R1 — EMC 직접 제어의 프로덕션 비현실성 (격상)**: BPMP debugfs(`clk/emc/{mrq_rate_locked,state,rate}`)는 NVIDIA 공식 **"debugging/experimental only"** — on-robot 프로덕션 이미지는 debugfs 미마운트/보안 차단이 **일반적**이라 debugfs EMC 직접 write 는 **프로덕션 1차 경로로 부적합(불가 가능성)**. 따라서 본 연구는 ① **측정/특성화 가치를 우선**(phase별 EMC-둔감성·J/chunk ledger·settle 경계는 debugfs 가용한 devkit 에서 측정해 그 자체로 기여) 하고, ② **프로덕션 1차 경로는 nvpmodel 프리셋 fallback**(AR-친화/DM-친화 모드를 `/etc/nvpmodel.conf` 에 정의, `nvpmodel -m` 전환 — supported/persistent)으로 둔다. debugfs 직접 write 는 **devkit 한정 특성화 경로**로 강등. nvpmodel 경로는 settle 이 더 크므로(수십~수백 ms) M2 settle-gate 로 가치 재판정. EMC 가 어느 경로로도 phase-제어 불가/무가치하면 **GPU-only DVFS 로 축소**(여전히 AR phase GPU↓ 42%급 절감 유효 — idea 일부 생존). → 이에 따라 "DM-구간 EMC↓ 추가 절감"은 debugfs 의존이므로 Tier-A 가 아니라 **Tier-B(조건부)** 결과로 둔다(§6.3 ⑤ 참조).
- **R2 — root 권한**: devfreq/debugfs/nvpmodel write 는 root 필요. → daemon 을 systemd 서비스(최소 권한, write 대상 path whitelist)로; on-robot 배포는 capability 한정.
- **R3 — phase별 절감 비합산**: 상호작용·transition overhead. → 합산 가정 명시 + M1/M2 결합 실측으로 보수 보고.
- **R4 — INA3221 rail 가용성**: production module(devkit 아님)은 board rail 미노출. → 1순위 AGX Orin **devkit** 으로 측정 완결, NX 는 carrier 확인 후.

### 6.3 R52.4 Tier-A/B/C 분류
- **Tier-A (반드시 성립해야 letter 가 성립)**: ① UND→GEN 경계에서 freq 전환이 **품질 무손실**(action MSE/픽셀 동일) ② J/chunk **≥15%↓** ③ settle < phase(policy) 로 전환이 net-positive.
- **Tier-B (성립 시 강화)**: ④ SparseDVFS-style baseline 대비 순증분 ≥5%p ⑤ EMC 직접 제어로 DM phase 추가 절감 ⑥ T2I J/phase 에서도 동일 경향.
- **Tier-C (실패해도 무방, 부수)**: ⑦ Thor 세대 비교 ⑧ nvpmodel-fallback 이 debugfs 와 동등 ⑨ Orin NX 일반화.

### 6.4 Tier-1 승격 조건 (1줄)
multi-tenant / 멀티 phase-mix(여러 요청의 AR·DM 가 시간축에서 겹치는) 환경에서 phase-결정성을 일반 governor(요청 간 phase 충돌 해소 + freq 협상)로 확장하면 → 시스템 contribution 으로 Tier-1(MLSys/HPCA) 승격 가능.

### 6.5 Scoring & Decision-tree
- **Scoring**: 직전 avg **7.4** 유지. **★ arch-impl 8** (표준 Jetson sysfs/debugfs·INA3221 devkit·hook 지점 verified → feasibility 최상) / **▼ novelty 6** (SparseDVFS EMC-triplet 선점 — 순증분은 intra-request modality-regime + phase-결정성 + MoT-최초 3축뿐). (diff 7 / impact 8 / ai-impl 8 유지.)
- **Decision-tree 분기**:
  - **EMC 제어 경로 분기 (프로덕션 현실성)**: debugfs EMC write 는 **devkit 특성화 경로**로만 사용 → 측정 결과(EMC-둔감성·DM-EMC↓ 절감폭)는 보고하되 **프로덕션 주장은 nvpmodel 프리셋 fallback 으로 한정**. nvpmodel-fallback 의 settle 이 phase 를 잠식(settle > phase) 하면 → **프로덕션 경로는 GPU-only DVFS 로 축소**(AR-GPU↓ 절감만 유지), debugfs-기반 EMC 추가절감은 Tier-B 특성화 결과로만 남김.
  - **settle_time > phase_length** (policy chunk 기준에서도, 또는 EMC/GPU 모두 settle 과대) → **drop** (전환 자체가 무가치, idea 성립 불가).
  - **J/chunk 절감 <15%** (Tier-A② 미달) 또는 **SparseDVFS-style 대비 순증분 <5%p** → standalone letter 가치 소멸 → **측정-only letter** 로 전환(= J/chunk·phase별 power·roofline·settle 의 첫 edge MoT 에너지 ledger 를 S3/LEDGERMARK producer 데이터로 흡수, DVFS 제어 주장 철회).
  - **Tier-A 전부 성립 + 순증분 ≥5%p** → DUOCLOCK standalone energy letter(DATE/ISLPED/IISWC) 제출.

---

## 부록 A — Source Anchor 재확인 요약 (R72.3)
- [`transformer_cosmos3.py#L1459-1500`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/transformer_cosmos3.py#L1459-L1500) (vllm-omni@`95d56cf`): UND→GEN 경계 결정적 존재 ✅
- [`pipeline_cosmos3.py#L1282/L1332/L1492`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py#L1282) (vllm-omni@`95d56cf`): `diffuse()`·`reset_cache()`·denoise step loop ✅
- [`diffusion_pipeline_profiler.py#L80-113`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py#L80-L113) (vllm-omni@`95d56cf`): `stage_durations`·`_PROFILER_TARGETS` ✅
- Orin GPU freq: `/sys/class/devfreq/17000000.gpu/{governor,userspace/set_freq,available_frequencies,cur_freq}` (JP6) / `17000000.ga10b`(JP5) — 직접 제어 가능 ✅
- Orin EMC freq: `/sys/kernel/debug/bpmp/debug/clk/emc/{mrq_rate_locked,state,rate,possible_rates}` (BPMP debugfs, "experimental only") + `nvpmodel.conf EMC MAX_FREQ` (supported/persistent fallback) — **devfreq userspace 노드 아님** ✅ (web 검증, [NVIDIA L4T r36.5 Power&Perf](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html))
- DVFS 제어 코드 현 repo 전무(greenfield) — daemon/hook/ledger 전부 (proposed) ✅
