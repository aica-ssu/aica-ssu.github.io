import fs from "node:fs";
import path from "node:path";
import Link from "next/link";
import type { Metadata } from "next";
import PrintButton from "./PrintButton";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "ROSTER 연구 보완 노트 — 읽을 논문과 다음 실험 | AICA Lab",
  description:
    "현재 multi-turn VLM 설계를 유지하며 살펴볼 논문 4편과 보완 실험 5개. MixKV·ReKV 비교, novelty 보완, 질문별 재선택 검증 안내.",
  robots: { index: false, follow: false },
};

export default function RosterReviewPage() {
  // Repository-owned HTML: reviewed research content, with no scripts or remote assets.
  const content = fs.readFileSync(
    path.join(process.cwd(), "src/data/docs/research-wiki/2026-09-roster-review.html"),
    "utf-8"
  );

  return (
    <div className={styles.page}>
      <nav className={styles.navigation} aria-label="연구 보완 노트 안내">
        <Link href="/research-wiki">← Research Wiki</Link>
        <div>
          <a href="#papers">읽을 논문</a>
          <a href="#experiments">보완 실험</a>
          <PrintButton />
        </div>
      </nav>
      <article
        id="roster-research-note"
        className={styles.note}
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </div>
  );
}
