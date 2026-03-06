import { researchAreas } from "@/data/research";

export const metadata = {
  title: "Research - AICA Lab",
};

export default function ResearchPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-16">
      <h1 className="text-3xl md:text-4xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
        Research
      </h1>
      <p className="text-lg mb-12" style={{ color: "var(--text-secondary)" }}>
        연구 분야
      </p>

      <div className="space-y-16">
        {researchAreas.map((area, i) => (
          <section key={i}>
            <h2 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
              {area.title}
            </h2>
            <p className="text-base font-medium mb-4" style={{ color: "var(--accent)" }}>
              {area.titleKo}
            </p>
            <p className="mb-4 leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              {area.description}
            </p>
            <p className="mb-6 leading-relaxed" style={{ color: "var(--text-secondary)" }}>
              {area.descriptionKo}
            </p>
            <div className="flex flex-wrap gap-2">
              {area.topics.map((topic, j) => (
                <span
                  key={j}
                  className="text-sm px-3 py-1 rounded-full border"
                  style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
                >
                  {topic}
                </span>
              ))}
            </div>
            {i < researchAreas.length - 1 && (
              <hr className="mt-12" style={{ borderColor: "var(--border)" }} />
            )}
          </section>
        ))}
      </div>
    </div>
  );
}
