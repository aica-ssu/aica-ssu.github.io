import Link from "next/link";
import Image from "next/image";
import { researchAreas } from "@/data/research";
import { news } from "@/data/news";
import { publications } from "@/data/publications";

export default function Home() {
  const featuredKeys = ["lee2026date", "chung2026cal", "kwon2025access", "kim2025ksc", "moon2025ksc"];
  const featuredPubs = featuredKeys
    .map((key) => publications.find((p) => p.bibtexKey === key))
    .filter(Boolean);
  const latestNews = news.slice(0, 5);

  return (
    <div>
      {/* Hero Section */}
      <section className="py-12 md:py-16" style={{ backgroundColor: "var(--bg-secondary)" }}>
        <div className="max-w-6xl mx-auto px-4 text-center">
          <div className="flex justify-center mb-4">
            <Image src="/images/logo.png" alt="AICA Lab" width={72} height={72} />
          </div>
          <h1 className="text-2xl md:text-3xl font-bold mb-4" style={{ color: "var(--text-primary)" }}>
            <span style={{ color: "var(--accent)" }}>A</span>dvanced{" "}
            <span style={{ color: "var(--accent)" }}>I</span>ntelligent{" "}
            <span style={{ color: "var(--accent)" }}>C</span>omputing{" "}
            <span style={{ color: "var(--accent)" }}>A</span>rchitecture Laboratory
          </h1>
          <p className="text-sm font-medium mt-4 mb-8" style={{ color: "var(--accent)" }}>
            AI Optimization &middot; Memory Systems &middot; Computer Architecture
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <Link
              href="/members"
              className="px-5 py-2.5 text-sm font-medium transition-colors"
              style={{ backgroundColor: "var(--accent)", color: "#ffffff" }}
            >
              Meet Our Team
            </Link>
            <Link
              href="/publications"
              className="px-5 py-2.5 text-sm font-medium border transition-colors"
              style={{ borderColor: "var(--accent)", color: "var(--accent)" }}
            >
              View Publications
            </Link>
          </div>
        </div>
      </section>

      {/* Research Highlights */}
      <section className="py-14">
        <div className="max-w-6xl mx-auto px-4">
          <h2 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
            Research Interests
          </h2>
          <p className="text-sm mb-8" style={{ color: "var(--text-secondary)" }}>연구 분야</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px" style={{ backgroundColor: "var(--border)" }}>
            {researchAreas.map((area, i) => (
              <Link href="/research" key={i}>
                <div
                  className="p-6 h-full transition-colors hover:opacity-90"
                  style={{ backgroundColor: "var(--bg-primary)" }}
                >
                  <h3 className="text-base font-semibold mb-0.5" style={{ color: "var(--text-primary)" }}>
                    {area.title}
                  </h3>
                  <p className="text-xs mb-2" style={{ color: "var(--accent)" }}>{area.titleKo}</p>
                  <p className="text-sm mb-3" style={{ color: "var(--text-secondary)" }}>{area.description}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {area.topics.slice(0, 3).map((topic, j) => (
                      <span
                        key={j}
                        className="text-[11px] px-1.5 py-0.5"
                        style={{ backgroundColor: "var(--bg-secondary)", color: "var(--text-muted)" }}
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Latest News */}
      <section className="py-14" style={{ backgroundColor: "var(--bg-secondary)" }}>
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex justify-between items-end mb-6">
            <div>
              <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
                Latest News
              </h2>
              <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>최근 소식</p>
            </div>
            <Link href="/news" className="text-sm font-medium" style={{ color: "var(--accent)" }}>
              View All &rarr;
            </Link>
          </div>
          <div className="space-y-0 divide-y" style={{ borderColor: "var(--border)" }}>
            {latestNews.map((item, i) => (
              <div key={i} className="py-3 flex flex-col sm:flex-row sm:items-start gap-2" style={{ borderColor: "var(--border)" }}>
                <span className="text-xs font-mono font-medium whitespace-nowrap" style={{ color: "var(--accent)" }}>
                  {item.date}
                </span>
                <p className="text-sm" style={{ color: "var(--text-primary)" }}>{item.contentKo}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Selected Publications */}
      <section className="py-14">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex justify-between items-end mb-6">
            <div>
              <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
                Selected Publications
              </h2>
              <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>주요 논문</p>
            </div>
            <Link href="/publications" className="text-sm font-medium" style={{ color: "var(--accent)" }}>
              View All &rarr;
            </Link>
          </div>
          <div className="space-y-0 divide-y" style={{ borderColor: "var(--border)" }}>
            {featuredPubs.map((pub, i) => (
              <div key={i} className="py-4" style={{ borderColor: "var(--border)" }}>
                {pub.award && (
                  <span className="text-[10px] font-semibold px-1.5 py-0.5 mr-2" style={{ backgroundColor: "rgba(245,158,11,0.15)", color: "#f59e0b" }}>
                    🏆 {pub.award}
                  </span>
                )}
                <h3 className="font-semibold text-sm mt-1" style={{ color: "var(--text-primary)" }}>
                  {pub.title}
                </h3>
                <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>{pub.authors}</p>
                <p className="text-xs mt-0.5 font-medium" style={{ color: "var(--accent)" }}>{pub.venue}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
