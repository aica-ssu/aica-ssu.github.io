import Link from "next/link";
import Image from "next/image";
import { members } from "@/data/members";

function slugify(name: string) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function LinkedInIcon({ href }: { href: string }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="inline-block align-middle ml-1.5 hover:opacity-70">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ color: "#0A66C2" }}>
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
      </svg>
    </a>
  );
}

function GoogleScholarIcon({ href }: { href: string }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="inline-block align-middle ml-1.5 hover:opacity-70">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ color: "#4285F4" }}>
        <path d="M5.242 13.769L0 9.5 12 0l12 9.5-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 1 0 0 14 7 7 0 0 0 0-14z" />
      </svg>
    </a>
  );
}

function SocialIcons({ member }: { member: { linkedin?: string; googleScholar?: string } }) {
  if (!member.linkedin && !member.googleScholar) return null;
  return (
    <>
      {member.linkedin && <LinkedInIcon href={member.linkedin} />}
      {member.googleScholar && <GoogleScholarIcon href={member.googleScholar} />}
    </>
  );
}

export const metadata = {
  title: "Members - AICA Lab",
};

export default function MembersPage() {
  const pi = members.filter((m) => m.category === "pi");
  const graduates = members.filter((m) => m.category === "graduate");
  const undergraduates = members.filter((m) => m.category === "undergraduate");
  const alumni = members.filter((m) => m.category === "alumni");

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        Members
      </h1>
      <p className="text-sm mb-10" style={{ color: "var(--text-secondary)" }}>연구실 구성원</p>

      {/* Principal Investigator */}
      <section className="mb-14">
        <h2 className="text-lg font-bold mb-5 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Principal Investigator
        </h2>
        {pi.map((member, i) => (
          <div key={i} className="flex flex-col sm:flex-row gap-5">
            <Link href={`/members/${slugify(member.name)}`} className="w-32 h-32 flex-shrink-0 overflow-hidden relative block" style={{ backgroundColor: "var(--bg-secondary)" }}>
              {member.image ? (
                <Image src={member.image} alt={member.name} fill className="object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-4xl" style={{ color: "var(--text-muted)" }}>&#128100;</div>
              )}
            </Link>
            <div>
              <h3 className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>
                <Link href={`/members/${slugify(member.name)}`} className="hover:underline" style={{ color: "var(--text-primary)" }}>
                  {member.name}
                </Link>
                <SocialIcons member={member} />
                {" "}<span className="text-base font-normal" style={{ color: "var(--text-secondary)" }}>({member.nameKo})</span>
              </h3>
              <p className="font-medium text-sm mt-1" style={{ color: "var(--accent)" }}>{member.role}</p>
              <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>{member.background}</p>
              {member.research && (
                <p className="text-sm mt-1" style={{ color: "var(--text-secondary)", whiteSpace: "pre-line" }}>
                  Research: {member.research}
                </p>
              )}
              <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
                {member.email}
              </p>
            </div>
          </div>
        ))}
      </section>

      {/* Graduate Students */}
      <section className="mb-14">
        <h2 className="text-lg font-bold mb-5 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Graduate Students
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {graduates.map((member, i) => (
            <div key={i} className="text-center">
              <Link href={`/members/${slugify(member.name)}`} className="w-20 h-20 rounded-full mx-auto mb-3 overflow-hidden relative block" style={{ backgroundColor: "var(--bg-secondary)" }}>
                {member.image ? (
                  <Image src={member.image} alt={member.name} fill className="object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-2xl" style={{ color: "var(--text-muted)" }}>&#128100;</div>
                )}
              </Link>
              <h3 className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>
                <Link href={`/members/${slugify(member.name)}`} className="hover:underline" style={{ color: "var(--text-primary)" }}>
                  {member.name}
                </Link>
                <SocialIcons member={member} />
              </h3>
              <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{member.nameKo}</p>
              <p className="text-xs mt-1" style={{ color: "var(--accent)" }}>{member.role}</p>
              <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>{member.researchShort || member.research}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Undergraduate Interns */}
      <section className="mb-14">
        <h2 className="text-lg font-bold mb-5 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Undergraduate Interns
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {undergraduates.map((member, i) => (
            <div key={i} className="text-center">
              <Link href={`/members/${slugify(member.name)}`} className="w-20 h-20 rounded-full mx-auto mb-3 overflow-hidden relative block" style={{ backgroundColor: "var(--bg-secondary)" }}>
                {member.image ? (
                  <Image src={member.image} alt={member.name} fill className="object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-2xl" style={{ color: "var(--text-muted)" }}>&#128100;</div>
                )}
              </Link>
              <h3 className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>
                <Link href={`/members/${slugify(member.name)}`} className="hover:underline" style={{ color: "var(--text-primary)" }}>
                  {member.name}
                </Link>
                <SocialIcons member={member} />
              </h3>
              <p className="text-xs" style={{ color: "var(--text-secondary)" }}>{member.nameKo}</p>
              <p className="text-xs mt-1" style={{ color: "var(--accent)" }}>{member.role}</p>
              <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>{member.researchShort || member.research}</p>
            </div>
          ))}
          {/* Recruiting Card */}
          <Link href="/recruiting" className="text-center group">
            <div className="w-20 h-20 rounded-full mx-auto mb-3 flex items-center justify-center border-2 border-dashed transition-colors group-hover:border-[var(--accent)]" style={{ borderColor: "var(--border)" }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: "var(--text-muted)" }}>
                <path d="M12 5v14M5 12h14" strokeLinecap="round" />
              </svg>
            </div>
            <h3 className="font-semibold text-sm" style={{ color: "var(--text-muted)" }}>Join Us!</h3>
            <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>학부연구생 / 학석사연계 / 대학원생 모집중</p>
          </Link>
        </div>
      </section>

      {/* Alumni */}
      <section>
        <h2 className="text-lg font-bold mb-5 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Alumni
        </h2>
        <div className="divide-y" style={{ borderColor: "var(--border)" }}>
          {alumni.map((member, i) => (
            <div key={i} className="py-2.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
              <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                {member.name} ({member.nameKo})
                <SocialIcons member={member} />
              </span>
              <span style={{ color: "var(--text-secondary)" }}>{member.role}</span>
              <span className="font-medium" style={{ color: "var(--accent)" }}>{member.current}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
