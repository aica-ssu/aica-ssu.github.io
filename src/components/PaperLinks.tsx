import type { Publication } from "@/data/publications";

function LinkButton({ href, label, children }: { href: string; label: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      title={label}
      aria-label={label}
      className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-1 border transition-colors hover:bg-[var(--bg-secondary)]"
      style={{ color: "var(--accent)", borderColor: "var(--border)" }}
    >
      {children}
      <span className="hidden sm:inline">{label}</span>
    </a>
  );
}

// GitHub (있을 때) → Paper(DOI) 순서로 아이콘 링크를 그린다
export default function PaperLinks({ pub }: { pub: Pick<Publication, "doi" | "github"> }) {
  if (!pub.doi && !pub.github) return null;
  return (
    <div className="flex items-center gap-1.5 shrink-0">
      {pub.github && (
        <LinkButton href={pub.github} label="GitHub">
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.4-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
          </svg>
        </LinkButton>
      )}
      {pub.doi && (
        <LinkButton href={`https://doi.org/${pub.doi}`} label="Paper">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </LinkButton>
      )}
    </div>
  );
}
