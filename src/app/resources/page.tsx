import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Resources - AICA Lab",
};

const resources = [
  {
    category: "Computer Architecture",
    items: [
      {
        title: "The RISC-V Instruction Set Manual, Volume I: User-Level ISA (v2.2)",
        href: "https://github.com/magicpan-risc-v/doc/blob/master/riscv-spec-v2.2.pdf",
        description: "RISC-V user-level ISA specification",
      },
    ],
  },
];

export default function ResourcesPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        Resources
      </h1>
      <p className="text-sm mb-10" style={{ color: "var(--text-muted)" }}>
        수업 및 연구에 필요한 참고 자료
      </p>

      {resources.map((section) => (
        <div key={section.category} className="mb-10">
          <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--text-primary)" }}>
            {section.category}
          </h2>
          <div className="space-y-0 divide-y" style={{ borderColor: "var(--border)" }}>
            {section.items.map((item) => (
              <a
                key={item.href}
                href={item.href}
                target="_blank"
                rel="noopener noreferrer"
                className="block group"
              >
                <div className="py-4 flex items-start gap-3" style={{ borderColor: "var(--border)" }}>
                  <span
                    className="text-[11px] font-medium px-2 py-0.5 mt-0.5 shrink-0 rounded"
                    style={{ backgroundColor: "var(--bg-secondary)", color: "var(--text-muted)", border: "1px solid var(--border)" }}
                  >
                    PDF
                  </span>
                  <div>
                    <h3
                      className="text-base font-semibold group-hover:underline"
                      style={{ color: "var(--accent)" }}
                    >
                      {item.title}
                      <span className="ml-1 text-xs" style={{ color: "var(--text-muted)" }}>↗</span>
                    </h3>
                    <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>
                      {item.description}
                    </p>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
