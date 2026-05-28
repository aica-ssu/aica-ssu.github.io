# 미선정 / 흡수 아이디어 — 2026-05-27 AI-Workload Memory Error Mitigation

Phase 1 에서 도출된 17 idea 중 Top-M 6 에 들지 못한 아이디어. 각 항목: GAP / 시도 overview / 미선정 사유 / 재방문 조건.

### AEGIS (SR5) — Unified Random-Fault + Adversarial Critical-Bit Placement
- **GAP**: reliability(random CE)와 security(targeted flip)가 같은 high-leverage bit 을 보호하려 함.
- **시도 overview**: 단일 migration map 이 output-proj/attention weight 의 critical bit 을 weak row + hammerable row 양쪽에서 회피.
- **미선정 사유**: RESQ([arXiv:2603.15413](https://arxiv.org/abs/2603.15413))가 reliability+security unified(quant+harden) 점유 — scoop 검증 ADJACENT(30-50%). KEEPER 의 Outperform branch(telemetry 가 공격 row 도 회피하는 보안 dual-use)로 흡수.
- **재방문 조건**: KEEPER 가 GPUHammer류 방어 실증 시 보안 venue spinoff 로 분리.

### RELAY (LS2) — GPU-Driven In-Kernel Error-Triggered Remap
- **GAP**: host-trap 기반 page fault migration 의 latency.
- **시도 overview**: GPUVM([arXiv:2411.05309](https://arxiv.org/abs/2411.05309)) 처럼 attention kernel 이 poisoned line touch 시 in-kernel remap.
- **미선정 사유**: kernel-level remap 의 구현·검증 risk 과다(CUDA kernel 내 동적 remap), single-GPU simulation 으로만 부분 검증 가능. Tier-1 scale-up 어려움.
- **재방문 조건**: KEEPER 의 migration controller 가 안정화된 후 latency 최적화 확장(future work).

### MASON (LS3) — Fault-Geometry-Aware KV Layout
- **GAP**: bank/row/column fault geometry(MICRO'25 분류)에 맞춘 KV block 정렬 부재.
- **시도 overview**: KV block 경계를 fault-containment domain 에 정렬해 bank fault 가 evictable/cold KV 만 손상.
- **미선정 사유**: RAMPART([arXiv:2310.16354](https://arxiv.org/abs/2310.16354)) device-confinement 와 인접(ADJACENT), 독립 contribution 약함. KEEPER 의 placement 정책에 geometry-aware 옵션으로 부분 흡수.
- **재방문 조건**: 실제 HBM3 fault-mode 분포 데이터 확보 시 layout 최적화 단독 연구.

### TRIAGE (SR3) — Criticality-Aware Migrate-vs-Offline RL Trigger
- **GAP**: RL adaptive mitigation([arXiv:2407.16377](https://arxiv.org/abs/2407.16377))의 trigger 가 criticality-blind.
- **시도 overview**: CE 상승 row 가 critical(weight/heavy-KV) 이면 migrate, tolerant(cold KV/activation) 이면 ignore/offline 하는 RL/cost-benefit 정책.
- **미선정 사유**: KEEPER 의 migration controller trigger 와 기능 중복 — KEEPER 에 흡수(criticality term 이 곧 TRIAGE 의 핵심). 단독 RL formulation 은 KEEPER 확장으로 충분.
- **재방문 조건**: KEEPER 의 heuristic trigger 가 RL 로 유의하게 개선되면 별도 정책 paper.

### Phase 1 흡수 로그 (cluster merge)
- **AO1 (Heavy-Hitter Protection Budget)** → KEEPER M1 흡수.
- **AO2 (Selective KV Scrubbing)** → CADENCE(TRIAD Tier-2 fallback) / SR4 흡수.
- **AO4 (Recompute-as-Correction)** → TRIAD M2 recompute-arm 흡수.
- **AL1 (Info-Theoretic Criticality Map)** → KEEPER/TRIAD scorer 의 이론 backing 보조.
- **AL3 (Optimal Migration Threshold)** → SR3 TRIAGE(흡수) 의 formal 부분.
- **LS4 (Hotness×Health Fused Placement)** → KEEPER M3 흡수.
- **SR4 (Tensor-Class Differentiated ECS/Refresh, "CADENCE")** → TRIAD 의 Tier-2 강등 fallback 으로 보존 (decision tree ② branch).
