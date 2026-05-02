# 미선정 / DROP / 흡수 아이디어 (2026-05-02 vlm-scenario-aware)

## DROP (14개 — 70%+ scoop or single axis weak)

### A3 — Scenario-Conditional KV Budget (DROP)
- **GAP**: PrefixKV / AdaptToken adaptive policy 의 scenario conditioning
- **시도한 overview**: layer-wise binary search + scenario class 별 budget table
- **미선정 사유**: PrefixKV ([NeurIPS 2025](https://github.com/THU-MIG/PrefixKV)) / AdaptToken (Microsoft 2025) **50-70% scoop**, scenario conditioning 만으로는 incremental
- **재방문 조건**: T1.1 Mosaic 의 sub-component 으로 흡수 가능

### A4 — Multi-Tenant Scenario Batch Composer (DROP)
- **GAP**: Heterogeneous scenario batch composition
- **미선정 사유**: **Nova** ([arXiv:2509.21301](https://arxiv.org/abs/2509.21301)) cross-stage parallel **70%+ scoop**

### A5 — Streaming Video Sliding Window (흡수 → T2.1 Lantern)
- **시도한 overview**: NVDEC + sliding window KV
- **미선정 사유**: T2.1 Lantern 으로 흡수 + One-Token-per-Frame ([arXiv:2604.14149](https://arxiv.org/abs/2604.14149)) 50-70% scoop

### A6 — Doc VLM Multi-Turn ROI Cache (DROP)
- **GAP**: Tile pHash radix tree for document multi-turn
- **미선정 사유**: AwaRes ([arXiv:2603.16932](https://arxiv.org/abs/2603.16932)) **70%+ scoop**

### B1 — Question-Conditional Token Eviction (DROP)
- **GAP**: Multi-turn question keyword-conditional eviction
- **미선정 사유**: VisionThink ([ICLR 2026](https://arxiv.org/abs/2507.13348)) **70%+ scoop** (RL autonomous resolution + question)

### B2 — Long Video Scene Boundary Compression (DROP)
- **미선정 사유**: LongVU ([arXiv:2410.17434](https://arxiv.org/abs/2410.17434)) / PVC (CVPR 2025) **70%+ scoop**

### B3 — Action-Dense Boundary-Preserving (DROP)
- **미선정 사유**: METok ([arXiv:2506.02850](https://arxiv.org/abs/2506.02850)) **50-70% scoop**

### B4 — Event-Graph KV Partitioning (DROP)
- **GAP**: Causal/temporal event-graph cross-event reuse
- **미선정 사유**: TSG (Temporal Semantic Graphs, [arXiv:2601.06097](https://arxiv.org/abs/2601.06097)) 91.4% token reduction **70%+ scoop**

### B6 — Doc ROI Token Prune (DROP)
- **미선정 사유**: AttWarp (ICLR 2026) **70%+ scoop**

### B7 — Multi-Image Cross-Image Dedup (흡수 → T1.3 Bramble)
- **시도한 overview**: Image-level dedup
- **미선정 사유**: Lossless Ultimate Compression ([arXiv:2512.09010](https://arxiv.org/abs/2512.09010)) 50-70% + T1.3 Bramble 으로 흡수

### C1 — Multi-Turn Video Vision KV Warm Cache (흡수 → T1.2 Lattice)
- **시도한 overview**: Host DRAM tier + frame radix tree
- **미선정 사유**: VLCache ([arXiv:2512.12977](https://arxiv.org/abs/2512.12977)) + LMCache ([arXiv:2510.09665](https://arxiv.org/abs/2510.09665)) **70%+ scoop**, T1.2 Lattice 의 layer-wise extension 으로 흡수

### C3 — Cross-Tenant Vision KV Pool (흡수 → T1.3 Bramble)
- **시도한 overview**: Global pHash + privacy boundary
- **미선정 사유**: OxyGen ([arXiv:2603.14371](https://arxiv.org/abs/2603.14371)) + KVShare ([arXiv:2503.16525](https://arxiv.org/abs/2503.16525)) **70%+ scoop**, T1.3 Bramble (image-level pHash + privacy + reference counting + zeroize) 으로 흡수

### C4 — Scenario-Aware Batch Size Optimizer (DROP)
- **미선정 사유**: T1.1 Mosaic 의 sub-component, 단독 contribution narrow

### C5 — Long Video Scene Demote (흡수 → T1.2 Lattice)
- **시도한 overview**: Scene boundary 기준 deep-layer KV demote
- **미선정 사유**: T1.2 Lattice 의 layer-wise budget sub-mechanism 으로 흡수

### C6 — Hot Scenario Cache Pinning (흡수 → T1.1 Mosaic)
- **시도한 overview**: Production hit-rate adaptive pinning
- **미선정 사유**: LMCache static pin API **50-70% scoop**, T1.1 Mosaic 의 M3 sub-mechanism 으로 흡수

---

## 통합 흡수 매핑

| 흡수된 idea | 흡수 대상 | 흡수 방식 |
|------------|---------|----------|
| A5 Streaming | T2.1 Lantern | NVDEC + sliding window 통합 |
| B7 Multi-Image Dedup | T1.3 Bramble | image-level pHash + dedup 통합 |
| C1 Vision KV Warm Cache | T1.2 Lattice | layer-wise prefix retention 통합 |
| C3 Cross-Tenant Pool | T1.3 Bramble | privacy boundary + reference counting 통합 |
| C5 Scene Demote | T1.2 Lattice | layer-wise budget sub-mech |
| C6 Hot Pinning | T1.1 Mosaic | hit-rate adaptive pinning M3 |

---

## 다음 세션 재방문 권고

- **Edge VLM** (Jetson Thor / AGX Orin) scenario taxonomy 확장 (mobile / robot / on-device document)
- **Multi-modal beyond vision** (audio + video + text) scenario classifier
- **Privacy-preserving cross-tenant pool** 의 differential privacy 강화
