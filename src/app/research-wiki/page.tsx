import Link from "next/link";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Research Wiki - AICA Lab",
  robots: { index: false, follow: false },
};

type WikiEntry = {
  slug: string;
  title: string;
  description: string;
  date: string;
  tags: string[];
};

const entries: Record<string, WikiEntry[]> = {
  "2026-04": [
    {
      slug: "2026-04/qwen3vl-deepstack-edge",
      title: "Qwen3-VL (DeepStack) + Qwen3.5 최신 아키텍처의 Edge 배포 최적화 (Loom / Mangrove / Vault / Gale / Forge)",
      description:
        "Qwen3-VL (arXiv:2511.21631, DeepStack arXiv:2406.04334 integration) + Qwen3.5-Omni (arXiv:2604.15804) 3 PDF 입력. R23 Step 0-α 신규 규칙 첫 적용 (IISWC/ISPASS/benchmark report 우선 탐색, 15편). Qwen3-VL 의 3 core upgrade (Interleaved MRoPE frequency interleave / DeepStack multi-layer residual visual_indexes=[8,16,24] / explicit video timestamp) + Qwen3.5 의 Hybrid MoE + Gated Delta Net + Thinker-Talker + ARIA 를 Jetson Orin AGX / Thor 128GB LPDDR5X / RTX 4060-5090 edge 에 배포 시 이전 Qwen2.5-VL / LLaVA-1.5 대비 아키텍처 diff 직접 mechanism 화. 3 experts × 4 idea = 12 → 3-way artificial split 통합 → Tier-1 Top 3 (★ Loom 7.90 lead Interleaved MRoPE unified LUT + FA3 fused / Mangrove 7.60 DeepStack 4-stage + LPDDR bank + DLA offload / Vault' 7.30 post-Major Revision DeepStack×MoE L2 contention) + Tier-2 독립 Top 2 (Gale 6.70 GDN:Attn 3:1 + 256K 3-tier / Forge 6.50 Thinker/Talker Tensor Core + DLA + L2) = 5 selected. T3 Echo DROP (VLCache arXiv:2512.12977 72-78% scoop + 이전 세션 Tidal 재진입). Jetson Thor vLLM MoE gap forum (2026-02) 산업 motivation. 이전 edge-vlm-energy 세션과 완전 독립.",
      date: "2026-04-24",
      tags: ["Mode 1", "Qwen3-VL", "DeepStack", "Qwen3.5", "Edge GPU", "Jetson Thor", "MoE", "R23 Step 0-α"],
    },
    {
      slug: "2026-04/energy-efficient-edge-vlm",
      title: "최신 VLM 특성 기반 에너지 효율적 Edge VLM Inference (Parquet / Triptych / Cartographer / Sift / Verge)",
      description:
        "기존 세션 완전 신규 (v1/v2/v3 VLM/VLA context serving, PRISM-VLM-KV, ACE-MoE, VLM+PIM 어느 것도 참조 안 함). 2026-04-23 신규 규칙 R23-R26 (workload-driven / kernel fusion triviality / Platform-Usage Analysis 3-step / metaphor noun title) 전면 적용 첫 세션. 초창기 LLaVA-1.5 (fixed 576 token) vs 최신 Qwen2.5-VL/InternVL3 (dynamic 4-16,384 token + MRoPE 3D + pixel shuffle + video temporal packing) 아키텍처 diff 를 mechanism 에 직접 활용. 3 experts × 4 idea = 12 후보 → 3-way artificial split 통합 → Tier-1 3 + Track B 3 = 6 후보. Phase 2 similarity critique 에서 CodecSight (arXiv:2604.06036, 2026-04-07, 16일 차이) 68-72% Tidal direct scoop 발견 → Tidal DROP. 최종 Tier-1 Top 2 (Parquet 7.54 AnyRes tile-count unifying signal × coupled DVFS × per-tile precision / Triptych 7.38 DLA + UMA zero-copy + DLA preemption) + Tier-2 독립 Top 3 (Cartographer 7.06 MRoPE tri-axial LUT / Sift 6.85 entropy-adaptive pixel shuffle / Verge 6.29 Conditional Thor vs Orin cross-arch).",
      date: "2026-04-23",
      tags: ["Mode 1", "VLM", "Edge GPU", "Jetson", "Energy", "Metaphor Naming"],
    },
    {
      slug: "2026-04/vlm-vla-context-serving-v3",
      title: "VLM/VLA Context-aware Caching & Serving (v3, Dual Top-3 New-Rules Re-run)",
      description:
        "v1/v2 (4/22, 구 규칙) 에서 도출된 10 variants 를 새 규칙 (R21 Dual Top-3 = Tier-1 Top 3 + Tier-2 독립 Top 3 총 6 아이디어 / R22 summary 블로그-style 의무 / Executive Summary 맨 앞 / arxiv 링크 포맷 강화 / Track B 독립) 으로 improve. v2 placeholder 7편 전부 재검증 (Harvest arXiv:2602.00328 55-65% concurrent 발견) + 2026-04 20+편 최신 논문 탐색 (Mosaic arXiv:2604.10060 55-65% HRTS concurrent, Predictable LLM Serving arXiv:2508.20274 60% ContextMIG concurrent, FlashVLA arXiv:2505.21200 68-72% DeadlineCOW scoop 접경). Tier-1 Top 3: HRTS+ 7.85 / ContextMIG+ 7.73 / PhaseGraph-VLA+ 7.18 (v1 A1 revival). Tier-2 독립 Top 3 (Track B): B1 GCReconfProfile 7.50 (ISLPED) / B2 TokenEvictEnergy 7.35 (IEEE ESL, negative result) / B3 ActHeadFuse 7.20 (IEEE CAL, 1-kHz control). Mechanism diff 전부 add 0 (improve-first 완전 준수), peer-reviewed ratio Tier-1 60-70% / Tier-2 67-100%.",
      date: "2026-04-23",
      tags: ["Mode 1", "VLM", "VLA", "Dual Top-3", "Track B", "v3"],
    },
    {
      slug: "2026-04/vlm-vla-context-serving-v2",
      title: "VLM/VLA Context-aware Caching & Serving (v2, Updated Harness Rules 재실행)",
      description:
        "업데이트된 harness 규칙(tiering ≤3-4, tier-aware dual-track, improve-first mechanism ≤3, reference integrity + OpenReview identity 4-point check) 전체 적용. 6 ideas → 5 (Axis A merger) → 10 variants (dual-track) → Top 3 Tier-Mix: (1) HRTS Top-tier 7.90 Accept (ASPLOS/MICRO, HBM row-tile streaming for long-context video VLM), (2) ContextMIG Top-tier 7.75 Accept (ASPLOS/MLSys, CLIP-L reuse graph × MIG dual-issue × phase coalesce for multi-tenant VLM, Mosaic scoop 정면 대응 replace-all), (3) NACK-Gossip Tier-2 7.80 Conditional Accept (IEEE ESL/ISLPED, VLA 2-GPU NVLink peer-fetch profiling). OpenReview 3편 verified (VL-Cache ICLR'25, VLA-Cache NeurIPS'25, SparseVLM ICML'25).",
      date: "2026-04-22",
      tags: ["Mode 1", "VLM", "VLA", "Tier-Mix", "Dual-Track"],
    },
    {
      slug: "2026-04/vlm-vla-context-serving",
      title: "VLM/VLA Context-aware Caching & Serving (v1, 초기 ideation)",
      description:
        "PIM 비의존 pure GPU stack 기반 VLM/VLA context-aware caching/serving ideation 초기 버전. ai-optimization-expert (A1-A3) + legacy-system-expert (L1-L3) + algorithm-expert (P1-P2 predictor) 협업. 6 ideas → Top 3: L1 ContextSM-Tri 7.00 Accept (content-axis SM/BW/KV tri-partition), A3 SemCOW-Deadline 7.15 Conditional Accept (page-refcount COW + deadline SM yielding), A1 PhaseGraph-VLA 7.08 Conditional Accept (SSE Page-Hinkley + phase CUDA Graph). Phase 2' placeholder 11편 post-verification (GUI-KV/OxyGen 실존, Rethinking Token Pruning withdrawn, 8편 부재), Semantic Scheduling concurrent 발견으로 L1 Novelty 7.2→6.8 조정. v2 재실행 세션의 baseline context.",
      date: "2026-04-22",
      tags: ["Mode 1", "VLM", "VLA", "Predictor"],
    },
    {
      slug: "2026-04/vlm-pim-extension",
      title: "VLM+PIM 내부 연구 보완·확장: DeepStack-Native 6-Tier KV + Quantization-Robust Layered Defense",
      description:
        "AttAcc(ASPLOS'24) baseline + Qwen3-VL-4B 기반 VLM+PIM 연구의 보완점 탐색. 8 ideas → 3 fused → Top 2 (F2 Quant-Robust Layered Defense HPCA/MICRO 8.43, F1 DeepStack-Native 6-Tier Pipeline ASPLOS/MLSys 7.60). VLCache/PAM/ModServe scoop 분석 + W8A8 +66pp visual attention collapse 선행 보고 부재 검증.",
      date: "2026-04-22",
      tags: ["Mode 2", "VLM", "PIM", "AttAcc"],
    },
    {
      slug: "2026-04/ace-moe-vlm-vla",
      title: "ACE-MoE 확장: VLM/VLA Software 연구 주제 Top 3",
      description:
        "ACE-MoE(ICCAD'26)을 VLM/VLA로 확장하는 3개 연구 주제 — ACE-VLA(real-time), Joint Token-Expert Budget, Modality-Aware ACE. arxiv 60+편 재검색 기반 positioning 포함.",
      date: "2026-04-21",
      tags: ["Mode 2", "MoE", "VLM", "VLA"],
    },
  ],
};

export default function ResearchWikiIndexPage() {
  const months = Object.keys(entries).sort().reverse();

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1
        className="text-2xl font-bold mb-1"
        style={{ color: "var(--text-primary)" }}
      >
        Research Wiki
      </h1>
      <p className="text-sm mb-2" style={{ color: "var(--text-muted)" }}>
        연구실 내부 공유용 ideation 노트. 네비게이션 비노출(hidden link).
      </p>
      <p className="text-xs mb-10" style={{ color: "var(--text-muted)" }}>
        aica-research-bot harness로 도출된 연구 주제 및 관련연구 정리.
      </p>

      <div className="space-y-12">
        {months.map((month) => (
          <section key={month}>
            <h2
              className="text-sm font-semibold mb-3 tracking-wide uppercase"
              style={{ color: "var(--text-muted)" }}
            >
              {month}
            </h2>
            <div
              className="space-y-0 divide-y"
              style={{ borderColor: "var(--border)" }}
            >
              {entries[month].map((entry) => (
                <Link
                  key={entry.slug}
                  href={`/research-wiki/${entry.slug}`}
                  className="block group"
                >
                  <div
                    className="py-5 flex items-start gap-3"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <div className="flex flex-col gap-1 shrink-0 min-w-[82px]">
                      {entry.tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-[10px] font-medium px-2 py-0.5 rounded inline-block text-center"
                          style={{
                            backgroundColor: "var(--bg-secondary)",
                            color: "var(--text-muted)",
                            border: "1px solid var(--border)",
                          }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div>
                      <h3
                        className="text-base font-semibold group-hover:underline"
                        style={{ color: "var(--accent)" }}
                      >
                        {entry.title}
                      </h3>
                      <p
                        className="text-sm mt-1"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {entry.description}
                      </p>
                      <p
                        className="text-xs mt-2"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {entry.date}
                      </p>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
