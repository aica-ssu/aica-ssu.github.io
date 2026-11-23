// 멤버 개인 페이지의 CV. 데이터는 src/lib/cv.ts 의 buildCv() 가 members/publications/patents 에서 만든다.
import Image from "next/image";
import type { Publication } from "@/data/publications";
import PaperLinks from "@/components/PaperLinks";
import { ScholarIcon, LinkedInIcon, GitHubIcon, MailIcon } from "@/components/Icons";
import { type CvData, type LinkKind, isMe, isCorresponding, pubLabel, patentRegion, patentNumber } from "@/lib/cv";

const typeColor: Record<Publication["type"], string> = {
  conference: "#2563eb",
  journal: "#7c3aed",
  domestic: "#64748b",
};

// 원형 사진. 사진 전체를 원 안에 넣고(object-contain) 남는 좌우는 같은 사진을 흐리게 깔아
// 머리·얼굴이 잘리지 않게 한다.
function Avatar({ src, alt, size }: { src: string; alt: string; size: number }) {
  return (
    <div className="relative shrink-0 rounded-full overflow-hidden" style={{ width: size, height: size, backgroundColor: "var(--bg-secondary)" }}>
      <Image src={src} alt="" aria-hidden width={size} height={size} className="absolute inset-0 w-full h-full object-fill blur-lg scale-110" />
      <Image src={src} alt={alt} width={size} height={size} className="relative w-full h-full object-contain object-top origin-top scale-[1.16]" />
    </div>
  );
}

const linkMeta: Record<LinkKind, { label: string; Icon: typeof ScholarIcon }> = {
  scholar: { label: "Google Scholar", Icon: ScholarIcon },
  linkedin: { label: "LinkedIn", Icon: LinkedInIcon },
  github: { label: "GitHub", Icon: GitHubIcon },
};

// 이름 옆 아이콘 버튼. 링크가 있는 것만 그린다.
function SocialIcons({ cv }: { cv: CvData }) {
  if (cv.links.length === 0) return null;
  return (
    <div className="flex items-center gap-1.5">
      {cv.links.map(({ kind, href }) => {
        const { label, Icon } = linkMeta[kind];
        return (
          <a
            key={kind}
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            title={label}
            aria-label={label}
            className="inline-flex items-center justify-center w-8 h-8 rounded-full border transition-colors hover:bg-[var(--bg-secondary)]"
            style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
          >
            <Icon className="w-4 h-4" />
          </a>
        );
      })}
    </div>
  );
}

// markCorr: 교신저자 * 표시 (교수 CV에만 쓴다)
function Authors({ pub, me, markCorr }: { pub: Publication; me: string[]; markCorr: boolean }) {
  const names = pub.authors.split(", ");
  return (
    <>
      {names.map((n, i) => (
        <span key={i}>
          {isMe(n, me) ? <strong>{n}</strong> : n}
          {markCorr && isCorresponding(n, pub) && <sup>*</sup>}
          {i < names.length - 1 && ", "}
        </span>
      ))}
    </>
  );
}

// 5편/5건을 넘어 일부만 보일 때 전체 목록 페이지로 가는 링크
function MoreLink({ cv, kind }: { cv: CvData; kind: "pubs" | "patents" }) {
  const more = kind === "pubs" ? cv.pubsSelected : cv.totalPatents > cv.patents.length;
  if (!more) return null;
  return (
    <a href={kind === "pubs" ? "/publications" : "/patents"} className="text-[0.75rem] font-medium normal-case tracking-normal whitespace-nowrap hover:underline" style={{ color: "var(--accent)" }}>
      {kind === "pubs" ? `All ${cv.totalPubs} papers` : `All ${cv.totalPatents} patents`} →
    </a>
  );
}

function H({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="cv-h mt-7 mb-2.5 pb-1 border-b flex items-end justify-between gap-3" style={{ borderColor: "var(--text-primary)" }}>
      <h2 className="text-[0.78rem] font-bold uppercase tracking-[0.14em]">{children}</h2>
      {right}
    </div>
  );
}

function Row({ title, detail, period }: { title: string; detail?: string; period: string }) {
  return (
    <div className="cv-row flex justify-between gap-4 mb-1.5">
      <div>
        <span className="font-semibold">{title}</span>
        {detail && <div style={{ color: "var(--text-secondary)" }}>{detail}</div>}
      </div>
      <div className="shrink-0 text-right" style={{ color: "var(--text-secondary)" }}>{period}</div>
    </div>
  );
}

// 모바일에서는 라벨을 인용문 위로 올리고, sm 이상에서 왼쪽 열로 둔다
const LABEL_COL = "sm:w-[6.5rem] shrink-0 text-[0.75rem] font-semibold sm:pt-0.5";
const ITEM_ROW = "flex flex-col sm:flex-row gap-0.5 sm:gap-3";

export default function MemberCV({ cv }: { cv: CvData }) {
  const m = cv.member;
  return (
    <div className="p-5 sm:p-12 text-[0.85rem] leading-relaxed" style={{ color: "var(--text-primary)" }}>
      <header className="flex flex-col sm:flex-row sm:items-center gap-5">
        {m.image && <Avatar src={m.image} alt={m.name} size={104} />}
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">
              {cv.isPI && "Prof. "}{m.name} <span className="text-lg font-normal" style={{ color: "var(--text-muted)" }}>{m.nameKo}</span>
            </h1>
            <SocialIcons cv={cv} />
          </div>
          <p className="mt-1" style={{ color: "var(--text-secondary)" }}>{cv.headline}</p>
          {cv.email && (
            <a href={`mailto:${cv.email}`} className="mt-1 inline-flex items-center gap-1.5 text-[0.8rem] hover:underline" style={{ color: "var(--text-secondary)" }}>
              <MailIcon className="w-3.5 h-3.5" />
              {cv.email}
            </a>
          )}
        </div>
      </header>

      {cv.interests.length > 0 && (
        <>
          <H>Research Interests</H>
          <p>{cv.interests.join(" · ")}</p>
        </>
      )}

      {cv.education.length > 0 && (
        <>
          <H>Education</H>
          {cv.education.map((e, i) => <Row key={i} {...e} />)}
        </>
      )}

      {cv.experience.length > 0 && (
        <>
          <H>{cv.isPI ? "Professional Experience" : "Research Experience"}</H>
          {cv.experience.map((e, i) => <Row key={i} {...e} />)}
        </>
      )}

      <H right={<MoreLink cv={cv} kind="pubs" />}>{cv.pubsSelected ? "Selected Publications" : "Publications"}</H>
      {cv.pubs.length === 0 && <p style={{ color: "var(--text-muted)" }}>To be added.</p>}
      <ul>
        {cv.pubs.map((p) => (
          <li key={p.bibtexKey} className={`${ITEM_ROW} mb-3 sm:mb-2`}>
            <span className={LABEL_COL} style={{ color: typeColor[p.type] }}>{pubLabel(p)}</span>
            <span className="flex-1 min-w-0">
              <Authors pub={p} me={cv.nameVariants} markCorr={cv.isPI} />, &ldquo;{p.title},&rdquo; <em style={{ color: "var(--text-secondary)" }}>{p.venue}</em>{p.underReview && " (under review)"}.
              {p.award && <strong style={{ color: "var(--award)" }}> ({p.award})</strong>}
            </span>
            <span className="shrink-0 sm:self-center mt-1 sm:mt-0"><PaperLinks pub={p} iconOnly /></span>
          </li>
        ))}
      </ul>
      {cv.isPI && cv.pubs.length > 0 && <p className="text-[0.75rem] mt-1" style={{ color: "var(--text-muted)" }}>* Corresponding author</p>}

      {cv.patents.length > 0 && (
        <>
          <H right={<MoreLink cv={cv} kind="patents" />}>Patents</H>
          <ul>
            {cv.patents.map((p, i) => (
              <li key={i} className={`${ITEM_ROW} mb-2.5 sm:mb-1.5`}>
                <span className={LABEL_COL} style={{ color: "var(--text-secondary)" }}>
                  {patentRegion(p)}{" "}
                  <span className="font-medium" style={{ color: p.status === "Registered" ? "#16a34a" : "var(--text-muted)" }}>{p.status}</span>
                </span>
                <span>
                  {p.inventors}, &ldquo;{p.title},&rdquo; {patentNumber(p)}, {p.date}.
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      {cv.awards.length > 0 && (
        <>
          <H>Honors &amp; Awards</H>
          {cv.awards.map((a, i) => <Row key={i} title={a.title} detail={a.detail} period={a.year} />)}
        </>
      )}

      {m.teaching && (
        <>
          <H>Teaching</H>
          {m.teaching.map((t, i) => <Row key={i} title={t.university} detail={t.courses.join(", ")} period={t.period} />)}
        </>
      )}

      {m.services && (
        <>
          <H>Professional Services</H>
          {m.services.map((s, i) => <p key={i}>{s}</p>)}
        </>
      )}
    </div>
  );
}
