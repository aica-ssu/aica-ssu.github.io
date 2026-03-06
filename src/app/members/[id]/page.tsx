import { members } from "@/data/members";
import { publications } from "@/data/publications";
import Link from "next/link";
import Image from "next/image";
import { notFound } from "next/navigation";

function slugify(name: string) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function getNameVariants(member: { name: string; nameKo: string }) {
  const variants = [member.name, member.nameKo];
  const hyphenated = member.name.replace(/(\w+)\s(\w+)\s(\w+)/, "$1-$2 $3");
  if (hyphenated !== member.name) variants.push(hyphenated);
  return variants;
}

export function generateStaticParams() {
  return members
    .filter((m) => m.category !== "alumni")
    .map((m) => ({ id: slugify(m.name) }));
}

export function generateMetadata({ params }: { params: Promise<{ id: string }> }) {
  return params.then(({ id }) => {
    const member = members.find((m) => slugify(m.name) === id);
    return { title: member ? `${member.category === "pi" ? "Prof. " : ""}${member.name} - AICA Lab` : "Member - AICA Lab" };
  });
}

export default async function MemberPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const member = members.find((m) => slugify(m.name) === id);
  if (!member) notFound();

  const isPI = member.category === "pi";

  const nameVariants = getNameVariants(member);
  const memberPubs = isPI ? [] : publications.filter((p) =>
    nameVariants.some((v) => p.authors.includes(v))
  );

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <Link href="/members" className="text-sm mb-8 inline-block" style={{ color: "var(--accent)" }}>
        &larr; Back to Members
      </Link>

      {/* Header */}
      <div className="flex flex-col sm:flex-row gap-6 mb-12">
        <div className="w-20 h-20 flex-shrink-0 rounded-full overflow-hidden" style={{ backgroundColor: "var(--bg-secondary)" }}>
          {member.image ? (
            <Image src={member.image} alt={member.name} width={80} height={80} className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-2xl" style={{ color: "var(--text-muted)" }}>&#128100;</div>
          )}
        </div>
        <div>
          <h1 className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>
            {isPI && "Prof. "}{member.name} <span className="text-base font-normal" style={{ color: "var(--text-secondary)" }}>({member.nameKo})</span>
          </h1>
          <p className="text-sm font-medium mt-1" style={{ color: "var(--accent)" }}>{member.role}</p>
          {member.background && (
            <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>{member.background}</p>
          )}
          {member.email && (
            <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>{member.email}</p>
          )}
        </div>
      </div>

      {/* Education (PI) */}
      {member.education && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Education
          </h2>
          <ul className="space-y-2">
            {member.education.map((edu, i) => (
              <li key={i} className="text-sm" style={{ color: "var(--text-secondary)", whiteSpace: "pre-line" }}>{edu}</li>
            ))}
          </ul>
        </section>
      )}

      {/* Career (PI) */}
      {member.career && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Career
          </h2>
          <ul className="space-y-2">
            {member.career.map((item, i) => (
              <li key={i} className="text-sm" style={{ color: "var(--text-secondary)" }}>{item}</li>
            ))}
          </ul>
        </section>
      )}

      {/* Research Interests */}
      {member.research && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Research Interests
          </h2>
          <div className="flex flex-wrap gap-2">
            {member.research.split("\n").map((topic, i) => (
              <span
                key={i}
                className="text-sm px-2 py-0.5"
                style={{ backgroundColor: "var(--bg-secondary)", color: "var(--text-secondary)" }}
              >
                {topic}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Teaching (PI) */}
      {member.teaching && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Teaching
          </h2>
          <div className="space-y-4">
            {member.teaching.map((t, i) => (
              <div key={i}>
                <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  {t.university} <span className="font-normal" style={{ color: "var(--text-muted)" }}>({t.period})</span>
                </p>
                <ul className="mt-1 space-y-0.5">
                  {t.courses.map((c, j) => (
                    <li key={j} className="text-sm" style={{ color: "var(--text-secondary)" }}>&#x2022; {c}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Awards (PI) */}
      {member.awards && member.awards.length > 0 && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Awards
          </h2>
          <ul className="space-y-1">
            {member.awards.map((a, i) => (
              <li key={i} className="text-sm" style={{ color: "var(--text-secondary)" }}>{a}</li>
            ))}
          </ul>
        </section>
      )}

      {/* Technical Services (PI) */}
      {member.services && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Professional Services
          </h2>
          <ul className="space-y-1">
            {member.services.map((s, i) => (
              <li key={i} className="text-sm" style={{ color: "var(--text-secondary)" }}>{s}</li>
            ))}
          </ul>
        </section>
      )}

      {/* Education (non-PI) */}
      {!isPI && member.background && !member.education && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Education
          </h2>
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{member.background}</p>
        </section>
      )}

      {/* Publications (non-PI only) */}
      {!isPI && (
        <section className="mb-10">
          <h2 className="text-base font-bold mb-3 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
            Publications
          </h2>
          {memberPubs.length > 0 ? (
            <div className="space-y-1">
              {memberPubs.map((pub) => (
                <div key={pub.bibtexKey} className="py-3 border-b" style={{ borderColor: "var(--border)" }}>
                  {pub.award && (
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 mr-2" style={{ backgroundColor: "rgba(245,158,11,0.15)", color: "#f59e0b" }}>
                      {pub.award}
                    </span>
                  )}
                  <h3 className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>{pub.title}</h3>
                  <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>{pub.authors}</p>
                  <p className="text-xs mt-0.5 font-medium" style={{ color: "var(--accent)" }}>
                    {pub.venue}
                    {pub.doi && (
                      <a href={`https://doi.org/${pub.doi}`} target="_blank" rel="noopener noreferrer" className="ml-2 hover:underline">[DOI]</a>
                    )}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm py-4" style={{ color: "var(--text-muted)" }}>To be added.</p>
          )}
        </section>
      )}
    </div>
  );
}
