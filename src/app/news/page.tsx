"use client";

import { useState } from "react";
import { news } from "@/data/news";

const ITEMS_PER_PAGE = 10;

function getYear(date: string) {
  return date.split(".")[0];
}

export default function NewsPage() {
  const [page, setPage] = useState(0);

  const totalPages = Math.ceil(news.length / ITEMS_PER_PAGE);
  const paged = news.slice(page * ITEMS_PER_PAGE, (page + 1) * ITEMS_PER_PAGE);

  const grouped: { year: string; items: typeof news }[] = [];
  for (const item of paged) {
    const year = getYear(item.date);
    const last = grouped[grouped.length - 1];
    if (last && last.year === year) {
      last.items.push(item);
    } else {
      grouped.push({ year, items: [item] });
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        News
      </h1>
      <p className="text-sm mb-10" style={{ color: "var(--text-secondary)" }}>소식</p>

      <div className="space-y-10">
        {grouped.map((group) => (
          <section key={group.year}>
            <h2 className="text-lg font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
              {group.year}
            </h2>
            <div className="divide-y" style={{ borderColor: "var(--border)" }}>
              {group.items.map((item, i) => (
                <div key={i} className="py-4 flex flex-col sm:flex-row sm:items-start gap-3">
                  <span
                    className="text-xs font-mono font-semibold whitespace-nowrap"
                    style={{ color: "var(--accent)" }}
                  >
                    {item.date}
                  </span>
                  <div>
                    <p className="text-sm" style={{ color: "var(--text-primary)" }}>{item.contentKo}</p>
                    <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>{item.content}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2 mt-12">
          <button
            onClick={() => setPage(page - 1)}
            disabled={page === 0}
            className="px-3 py-1.5 text-sm border transition-colors disabled:opacity-30"
            style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
          >
            &larr; Prev
          </button>
          {Array.from({ length: totalPages }, (_, i) => (
            <button
              key={i}
              onClick={() => setPage(i)}
              className="px-3 py-1.5 text-sm border transition-colors"
              style={{
                backgroundColor: page === i ? "var(--text-primary)" : "transparent",
                color: page === i ? "var(--bg-primary)" : "var(--text-muted)",
                borderColor: page === i ? "var(--text-primary)" : "var(--border)",
              }}
            >
              {i + 1}
            </button>
          ))}
          <button
            onClick={() => setPage(page + 1)}
            disabled={page === totalPages - 1}
            className="px-3 py-1.5 text-sm border transition-colors disabled:opacity-30"
            style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
          >
            Next &rarr;
          </button>
        </div>
      )}
    </div>
  );
}
