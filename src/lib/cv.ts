// CV 데이터 파생 로직.
// members.ts / publications.ts / patents.ts 만 읽어서 CV 한 장에 필요한 값을 만든다.
// 논문·특허를 데이터 파일에 추가하면 CV는 별도 수정 없이 자동으로 갱신된다.
import type { Member } from "@/data/members";
import { publications, underReviewPapers, type Publication } from "@/data/publications";
import { internationalPatents, domesticPatents, type Patent } from "@/data/patents";

// 아래 개수 제한과 "최근 3년" 기준은 교수 CV에만 적용한다. 학생 CV는 전부 보여준다.
export const MAX_PUBS = 5;
export const MAX_PATENTS = 5;
export const RECENT_YEARS = 3;

export function slugify(name: string) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

export function getNameVariants(member: { name: string; nameKo: string }) {
  const variants = [member.name, member.nameKo];
  const noHyphen = member.name.replace(/-/g, " ");
  if (noHyphen !== member.name) variants.push(noHyphen);
  const hyphenated = member.name.replace(/(\w+)\s(\w+)\s(\w+)/, "$1-$2 $3");
  if (hyphenated !== member.name) variants.push(hyphenated);
  return variants;
}

// "Something, Place (2018)" → { text: "Something, Place", period: "2018" }
export function splitPeriod(line: string) {
  const m = line.match(/^(.*?)\s*\(([^()]*)\)\s*$/);
  return m ? { text: m[1], period: m[2] } : { text: line, period: "" };
}

// 학회명 괄호 속 약어(DATE, ISLPED…)를 우선 쓰고, 저널은 아래 표로 줄인다.
// 표에 없는 새 저널은 첫 쉼표 앞 이름을 그대로 쓰므로 누락되지 않는다.
const journalShort: [RegExp, string][] = [
  [/^IEEE Computer Architecture Letters/, "IEEE CAL"],
  [/^IEEE Embedded Systems Letters/, "IEEE ESL"],
  [/^IEEE Transactions on Emerging Topics in Computing/, "IEEE TETC"],
  [/^IEEE Transactions on Computers/, "IEEE TC"],
  [/^IEEE Design & Test/, "IEEE D&T"],
  [/^Journal of Systems Architecture/, "JSA"],
  [/^Integration, the VLSI Journal/, "Integration"],
  [/^Microprocessors and Microsystems/, "MICPRO"],
];

export function venueShort(pub: Publication) {
  const paren = pub.venue.match(/\(([A-Za-z][A-Za-z0-9&\- ]{1,14})\)/);
  if (paren && pub.type !== "journal") return paren[1].replace(/\d{4}$/, "");
  for (const [re, short] of journalShort) if (re.test(pub.venue)) return short;
  return pub.venue.split(",")[0];
}

// "DATE'27" 형식
export function venueLabel(pub: Publication) {
  return `${venueShort(pub)}'${String(pub.year).slice(2)}`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// "Sep. 2026" → 202609 (정렬용)
function dateKey(date: string) {
  const m = date.match(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})/);
  return m ? Number(m[2]) * 100 + MONTHS.indexOf(m[1]) + 1 : 0;
}

export function isMe(name: string, variants: string[]) {
  return variants.includes(name.trim());
}

export function isCorresponding(name: string, pub: Publication) {
  return pub.correspondingAuthors.some((c) => name.includes(c) || c.includes(name));
}

export type LinkKind = "scholar" | "linkedin" | "github";

export interface CvEntry {
  title: string;
  detail?: string;
  period: string;
}

export interface CvAward {
  title: string;
  detail: string;
  year: string;
}

export type CvPatent = Patent & { region: "Int'l" | "KR" };

export interface CvData {
  member: Member;
  isPI: boolean;
  slug: string;
  nameVariants: string[];
  headline: string;
  email?: string;
  links: { kind: LinkKind; href: string }[];
  interests: string[];
  education: CvEntry[];
  experience: CvEntry[];
  pubs: Publication[];          // 화면에 보일 논문 (심사 중 논문이 맨 앞, 교수는 최대 MAX_PUBS)
  pubsSelected: boolean;        // 교수 CV처럼 일부만 골라낸 목록인지
  totalPubs: number;            // 게재 논문 수 (심사 중 제외)
  patents: CvPatent[];          // 화면에 보일 특허 (최신순, 교수는 최대 MAX_PATENTS)
  totalPatents: number;
  awards: CvAward[];
}

function isAuthor(authors: string, variants: string[]) {
  return authors.split(", ").some((a) => variants.includes(a.trim()));
}

// 내가 1저자이거나 교신저자인 논문
function isLeadRole(pub: Publication, variants: string[]) {
  const names = pub.authors.split(", ");
  return isMe(names[0], variants) || names.some((n) => isMe(n, variants) && isCorresponding(n, pub));
}

// 교수 CV: 최근 RECENT_YEARS 년(올해 포함, 게재 예정 포함) 논문 중 5편을
// selected 표시 → 1저자/교신 → 수상 → 최신 순으로 골라 최신순으로 보여준다.
// 국제 논문을 먼저 채우고, 모자랄 때만 국내 논문을 쓴다.
function pickPubs(mine: Publication[], variants: string[]) {
  const since = new Date().getFullYear() - (RECENT_YEARS - 1);
  const score = (p: Publication) => (p.selected ? 4 : 0) + (isLeadRole(p, variants) ? 2 : 0) + (p.award ? 1 : 0);
  const rank = (list: Publication[]) => [...list].sort((a, b) => score(b) - score(a) || b.year - a.year);
  const recent = mine.filter((p) => p.year >= since);
  return [...rank(recent.filter((p) => p.type !== "domestic")), ...rank(recent.filter((p) => p.type === "domestic"))]
    .slice(0, MAX_PUBS)
    .sort((a, b) => b.year - a.year);
}

// 같은 특허가 출원·등록으로 두 번 있으면 등록 건만 남긴다.
function dedupePatents<T extends Patent>(list: T[]) {
  const registered = new Set(list.filter((p) => p.status === "Registered").map((p) => p.title));
  return list.filter((p) => p.status === "Registered" || !registered.has(p.title));
}

// 교수·박사과정은 최대 5개, 그 외는 최대 3개
function interestLimit(member: Member) {
  return member.category === "pi" || /Ph\.D\./.test(member.role) ? 5 : 3;
}

export function buildCv(member: Member): CvData {
  const isPI = member.category === "pi";
  const nameVariants = getNameVariants(member);
  const mine = publications.filter((p) => isAuthor(p.authors, nameVariants)).sort((a, b) => b.year - a.year);

  const allPatents = dedupePatents([
    ...internationalPatents
      .filter((p) => nameVariants.some((v) => p.inventors.includes(v)))
      .map((p) => ({ ...p, region: "Int'l" as const })),
    ...domesticPatents
      .filter((p) => nameVariants.some((v) => p.inventors.includes(v)))
      .map((p) => ({ ...p, region: "KR" as const })),
  ]).sort((a, b) => dateKey(b.date) - dateKey(a.date));

  const interests = (member.interests ?? (member.research ?? member.researchShort ?? "").split(/[,\n]/))
    .map((t) => t.trim())
    .filter(Boolean)
    .slice(0, interestLimit(member));

  let education: CvEntry[];
  let experience: CvEntry[] = [];
  if (isPI) {
    education = (member.education ?? []).map((e) => {
      const [first, ...rest] = e.split("\n");
      const { text, period } = splitPeriod(first);
      return { title: text, detail: rest.map((r) => r.replace(/^-\s*/, "")).join(" "), period };
    });
    experience = (member.career ?? []).map((c) => {
      const { text, period } = splitPeriod(c);
      return { title: text, period };
    });
  } else {
    const role = splitPeriod(member.role);
    const since = role.period.replace(/~$/, "");
    const period = since ? `${since}–Present` : "";
    const bg = member.background ? [{ title: splitPeriod(member.background).text, period: splitPeriod(member.background).period }] : [];
    if (member.category === "undergraduate") {
      // 학부연구생: 학력은 학부 과정, 연구실 활동은 Research Experience 로
      education = bg;
      experience = [{ title: `${role.text}, AICA Lab, Soongsil University`, detail: "Advisor: Prof. Young-Ho Gong", period }];
    } else {
      education = [
        { title: `${role.text.replace(/ Student$/, "")}, Soongsil University`, detail: "AICA Lab (Advisor: Prof. Young-Ho Gong)", period },
        ...bg,
      ];
    }
  }

  const pubAwards: CvAward[] = mine
    .filter((p) => p.award)
    .map((p) => ({ title: p.award!, detail: `${venueShort(p)} ${p.year}`, year: String(p.year) }));
  const memberAwards: CvAward[] = (member.awards ?? []).map((a) => {
    const { text, period } = splitPeriod(a);
    const [title, ...rest] = text.split(", ");
    return { title, detail: rest.join(", "), year: period };
  });
  const awards = [...pubAwards, ...memberAwards].sort((a, b) => b.year.localeCompare(a.year));

  const links = (
    [
      ["scholar", member.googleScholar],
      ["linkedin", member.linkedin],
      ["github", member.github],
    ] as [LinkKind, string | undefined][]
  )
    .filter(([, href]) => href)
    .map(([kind, href]) => ({ kind, href: href! }));

  // 학생 CV: 심사 중 논문을 맨 앞에 두고 게재 논문은 전부
  const reviewing = underReviewPapers.filter((p) => isAuthor(p.authors, nameVariants));
  const pubs = isPI ? pickPubs(mine, nameVariants) : [...reviewing, ...mine];

  return {
    member,
    isPI,
    slug: slugify(member.name),
    nameVariants,
    headline: isPI ? `${member.role}, ${member.background ?? ""}` : `${splitPeriod(member.role).text}, AICA Lab, Soongsil University`,
    email: member.email,
    links,
    interests,
    education,
    experience,
    pubs,
    pubsSelected: isPI,
    totalPubs: mine.length,
    patents: isPI ? allPatents.slice(0, MAX_PATENTS) : allPatents,
    totalPatents: allPatents.length,
    awards,
  };
}
