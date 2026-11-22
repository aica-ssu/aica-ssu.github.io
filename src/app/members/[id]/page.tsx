import { members } from "@/data/members";
import { buildCv, cvFileBase, slugify } from "@/lib/cv";
import MemberCV from "@/components/MemberCV";
import CvActions from "@/components/CvActions";
import Link from "next/link";
import { notFound } from "next/navigation";

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

// 개인 페이지 = CV. 논문·특허는 publications.ts / patents.ts 에서 자동으로 모인다.
export default async function MemberPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const member = members.find((m) => m.category !== "alumni" && slugify(m.name) === id);
  if (!member) notFound();
  const cv = buildCv(member);

  return (
    <div className="cv-page max-w-4xl mx-auto px-4 py-12">
      <div className="no-print mb-6 flex flex-wrap items-center justify-between gap-3">
        <Link href="/members" className="text-sm" style={{ color: "var(--accent)" }}>
          &larr; Back to Members
        </Link>
        <CvActions fileBase={cvFileBase(cv)} />
      </div>
      <article className="cv-sheet border" style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-card)" }}>
        <MemberCV cv={cv} />
      </article>
    </div>
  );
}
