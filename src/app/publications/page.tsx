"use client";

import { useState } from "react";
import { publications, allTags, tagColors, aicaMembers, type Tag } from "@/data/publications";

const years = [...new Set(publications.map((p) => p.year))].sort((a, b) => b - a);

function renderAuthors(authorStr: string, correspondingAuthors: string[]) {
  const authors = authorStr.split(", ");
  return authors.map((name, i) => {
    const isAica = aicaMembers.some((m) => name.includes(m) || m.includes(name));
    const isCorresponding = correspondingAuthors.some((c) => name.includes(c) || c.includes(name));
    return (
      <span key={i}>
        <span style={{ textDecoration: isAica ? "underline" : "none" }}>
          {name}
        </span>
        {isCorresponding && <sup>*</sup>}
        {i < authors.length - 1 && ", "}
      </span>
    );
  });
}

export default function PublicationsPage() {
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [selectedTag, setSelectedTag] = useState<Tag | null>(null);
  const [expandedBib, setExpandedBib] = useState<string | null>(null);

  const filtered = publications.filter((p) => {
    if (selectedYear && p.year !== selectedYear) return false;
    if (selectedTag && !p.tags.includes(selectedTag)) return false;
    return true;
  });

  const grouped = years
    .map((y) => ({ year: y, pubs: filtered.filter((p) => p.year === y) }))
    .filter((g) => g.pubs.length > 0);

  const typeLabel: Record<string, string> = {
    conference: "Conference",
    journal: "Journal",
    domestic: "Domestic",
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        Publications
      </h1>
      <p className="text-sm mb-1" style={{ color: "var(--text-secondary)" }}>
        {filtered.length} / {publications.length} papers
      </p>
      <p className="text-xs mb-8" style={{ color: "var(--text-muted)" }}>
        <span style={{ textDecoration: "underline" }}>Underline</span> is/are with the AICA Lab. * indicates (co-)correspondence.
      </p>

      {/* Tag Filter */}
      <div className="mb-3">
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setSelectedTag(null)}
            className="px-2.5 py-1 text-xs font-medium border transition-colors"
            style={{
              backgroundColor: selectedTag === null ? "var(--text-primary)" : "transparent",
              color: selectedTag === null ? "var(--bg-primary)" : "var(--text-muted)",
              borderColor: selectedTag === null ? "var(--text-primary)" : "var(--border)",
            }}
          >
            All
          </button>
          {allTags.map((tag) => {
            const count = publications.filter((p) => p.tags.includes(tag)).length;
            const colors = tagColors[tag];
            const isActive = selectedTag === tag;
            return (
              <button
                key={tag}
                onClick={() => setSelectedTag(isActive ? null : tag)}
                className="px-2.5 py-1 text-xs font-medium border transition-colors"
                style={{
                  backgroundColor: isActive ? colors.text : "transparent",
                  color: isActive ? "#fff" : colors.text,
                  borderColor: isActive ? colors.text : "var(--border)",
                }}
              >
                {tag} ({count})
              </button>
            );
          })}
        </div>
      </div>

      {/* Year Filter */}
      <div className="mb-10">
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setSelectedYear(null)}
            className="px-2.5 py-1 text-xs font-medium border transition-colors"
            style={{
              backgroundColor: selectedYear === null ? "var(--text-primary)" : "transparent",
              color: selectedYear === null ? "var(--bg-primary)" : "var(--text-muted)",
              borderColor: selectedYear === null ? "var(--text-primary)" : "var(--border)",
            }}
          >
            All Years
          </button>
          {years.map((y) => (
            <button
              key={y}
              onClick={() => setSelectedYear(selectedYear === y ? null : y)}
              className="px-2.5 py-1 text-xs font-medium border transition-colors"
              style={{
                backgroundColor: selectedYear === y ? "var(--text-primary)" : "transparent",
                color: selectedYear === y ? "var(--bg-primary)" : "var(--text-muted)",
                borderColor: selectedYear === y ? "var(--text-primary)" : "var(--border)",
              }}
            >
              {y}
            </button>
          ))}
        </div>
      </div>

      {/* Publication List */}
      <div className="space-y-10">
        {grouped.map((group) => (
          <section key={group.year}>
            <h2 className="text-lg font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
              {group.year}
              <span className="text-sm font-normal ml-2" style={{ color: "var(--text-muted)" }}>
                ({group.pubs.length})
              </span>
            </h2>
            <div className="space-y-1">
              {group.pubs.map((pub) => (
                <div key={pub.bibtexKey} className="py-4 border-b" style={{ borderColor: "var(--border)" }}>
                  {/* Badges */}
                  <div className="flex flex-wrap items-center gap-1.5 mb-1.5">
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 uppercase tracking-wide" style={{ color: "var(--text-muted)", backgroundColor: "var(--bg-secondary)" }}>
                      {typeLabel[pub.type]}
                    </span>
                    {pub.award && (
                      <span className="text-[10px] font-semibold px-1.5 py-0.5" style={{ backgroundColor: "rgba(245,158,11,0.15)", color: "#f59e0b" }}>
                        🏆 {pub.award}
                      </span>
                    )}
                  </div>
                  {/* Title */}
                  <h3 className="font-semibold text-[15px] leading-snug mb-1" style={{ color: "var(--text-primary)" }}>
                    {pub.title}
                  </h3>
                  {/* Authors */}
                  <p className="text-sm mb-1" style={{ color: "var(--text-secondary)" }}>
                    {renderAuthors(pub.authors, pub.correspondingAuthors)}
                  </p>
                  {/* Venue */}
                  <p className="text-sm mb-2" style={{ color: "var(--text-muted)" }}>{pub.venue}</p>
                  {/* Tags + Actions */}
                  <div className="flex flex-wrap items-center gap-1.5">
                    {pub.tags.map((tag) => {
                      const colors = tagColors[tag];
                      return (
                        <span
                          key={tag}
                          className="text-[10px] font-medium px-1.5 py-0.5"
                          style={{ backgroundColor: colors.bg, color: colors.text }}
                        >
                          {tag}
                        </span>
                      );
                    })}
                    <span className="flex-1" />
                    <button
                      onClick={() => setExpandedBib(expandedBib === pub.bibtexKey ? null : pub.bibtexKey)}
                      className="text-[11px] font-medium px-1.5 py-0.5 hover:underline"
                      style={{ color: "var(--accent)" }}
                    >
                      BibTeX
                    </button>
                    {pub.doi && (
                      <a
                        href={`https://doi.org/${pub.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[11px] font-medium px-1.5 py-0.5 hover:underline"
                        style={{ color: "var(--accent)" }}
                      >
                        DOI
                      </a>
                    )}
                  </div>
                  {/* BibTeX expanded */}
                  {expandedBib === pub.bibtexKey && (
                    <pre
                      className="mt-2 p-3 text-xs overflow-x-auto"
                      style={{ backgroundColor: "var(--bg-secondary)", color: "var(--text-secondary)", fontFamily: "monospace" }}
                    >
{`@${pub.type === "conference" ? "inproceedings" : "article"}{${pub.bibtexKey},
  title   = {${pub.title}},
  author  = {${pub.authors.split(", ").map((n) => { const parts = n.trim().split(" "); const last = parts.pop(); return `${last}, ${parts.join(" ")}`; }).join(" and ")}},
  ${pub.type === "conference" ? "booktitle" : "journal"} = {${pub.venue.split(",")[0]}},
  year    = {${pub.year}}${pub.doi ? `,\n  doi     = {${pub.doi}}` : ""}
}`}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {filtered.length === 0 && (
        <p className="text-center py-12" style={{ color: "var(--text-muted)" }}>
          No publications found for the selected filters.
        </p>
      )}
    </div>
  );
}
