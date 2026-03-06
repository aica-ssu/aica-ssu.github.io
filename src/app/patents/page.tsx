import { internationalPatents, domesticPatents, type Patent } from "@/data/patents";

export const metadata = {
  title: "Patents - AICA Lab",
};

function PatentList({ patents }: { patents: Patent[] }) {
  return (
    <div className="divide-y" style={{ borderColor: "var(--border)" }}>
      {patents.map((p, i) => (
        <div key={i} className="py-4">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-semibold px-1.5 py-0.5" style={{
              backgroundColor: p.status === "Registered" ? "rgba(16,185,129,0.15)" : "rgba(99,102,241,0.15)",
              color: p.status === "Registered" ? "#10b981" : "#6366f1",
            }}>{p.status}</span>
          </div>
          <h3 className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>{p.title}</h3>
          <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>{p.inventors}</p>
          <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>{p.number} &middot; {p.date}</p>
        </div>
      ))}
    </div>
  );
}

export default function PatentsPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        Patents
      </h1>
      <p className="text-sm mb-10" style={{ color: "var(--text-secondary)" }}>
        International: {internationalPatents.length} / Domestic: {domesticPatents.length}
      </p>

      <section className="mb-14">
        <h2 className="text-lg font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          International Patents
        </h2>
        <PatentList patents={internationalPatents} />
      </section>

      <section>
        <h2 className="text-lg font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Domestic Patents / 국내 특허
        </h2>
        <PatentList patents={domesticPatents} />
      </section>
    </div>
  );
}
