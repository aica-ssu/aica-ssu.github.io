import { members } from "@/data/members";
import { buildCv, slugify } from "@/lib/cv";
import MemberCV from "@/components/MemberCV";
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

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <Link href="/members" className="text-sm mb-6 inline-block" style={{ color: "var(--accent)" }}>
        &larr; Back to Members
      </Link>
      <article className="border" style={{ borderColor: "var(--border)", backgroundColor: "var(--bg-card)" }}>
        <MemberCV cv={buildCv(member)} />
      </article>
    </div>
  );
}
