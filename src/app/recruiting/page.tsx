import Link from "next/link";
import { researchTracks } from "@/data/recruiting";

export const metadata = {
  title: "Join - AICA Lab",
};

export default function RecruitingPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-3xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        Join Us
      </h1>
      <p className="text-sm mb-10" style={{ color: "var(--text-secondary)" }}>대학원생 및 학부 연구생 모집</p>

      {/* Research Areas with Papers & News */}
      <section className="mb-12">
        <h2 className="text-lg font-bold mb-6 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Research Areas / 연구 분야
        </h2>
        <div className="space-y-10">
          {researchTracks.map((track, i) => (
            <div key={i}>
              <h3 className="font-semibold" style={{ color: "var(--text-primary)" }}>{track.title}</h3>
              <p className="text-xs font-medium" style={{ color: "var(--accent)" }}>{track.titleKo}</p>
              <p className="text-sm mt-2 mb-4" style={{ color: "var(--text-secondary)" }}>{track.desc}</p>

              {/* Key Publications */}
              <div className="mb-3">
                <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-muted)" }}>Key Publications</p>
                <ul className="space-y-1">
                  {track.papers.map((p, j) => (
                    <li key={j} className="text-sm" style={{ color: "var(--text-secondary)" }}>
                      &bull; {p.title} <span style={{ color: "var(--accent)" }}>({p.venue})</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Related News */}
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-muted)" }}>Related News</p>
                <ul className="space-y-1">
                  {track.news.map((n, j) => (
                    <li key={j} className="text-xs" style={{ color: "var(--text-muted)" }}>
                      &bull;{" "}
                      {n.url ? (
                        <a href={n.url} target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: "var(--text-secondary)" }}>
                          {n.title}
                        </a>
                      ) : (
                        n.title
                      )}
                      {" "}&mdash; <span style={{ color: "var(--text-muted)" }}>{n.source}, {n.date}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {i < researchTracks.length - 1 && <hr className="mt-8" style={{ borderColor: "var(--border)" }} />}
            </div>
          ))}
        </div>
      </section>

      {/* Qualifications */}
      <section className="mb-12">
        <h2 className="text-lg font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Qualifications / 지원 자격
        </h2>
        <ul className="space-y-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li className="flex items-start gap-2">
            <span style={{ color: "var(--accent)" }}>&#x2713;</span>
            컴퓨터 아키텍처 및 시스템에 대한 높은 관심 <strong>(매우 중요)</strong>
          </li>
          <li className="flex items-start gap-2">
            <span style={{ color: "var(--accent)" }}>&#x2713;</span>
            컴퓨터 구조에 대한 이해 또는 C/C++, Python 프로그래밍 능력
          </li>
          <li className="flex items-start gap-2">
            <span style={{ color: "var(--accent)" }}>&#x2713;</span>
            <strong>대학원생:</strong> 학사 또는 석사 학위 취득 예정자
          </li>
          <li className="flex items-start gap-2">
            <span style={{ color: "var(--accent)" }}>&#x2713;</span>
            <strong>학부 연구생:</strong> 재학 중, 2학년 1학기 이상
          </li>
        </ul>
      </section>

      {/* Benefits */}
      <section className="mb-12">
        <h2 className="text-lg font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Benefits / 지원 사항
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="font-semibold text-sm mb-3" style={{ color: "var(--text-primary)" }}>Graduate Students / 대학원생</h3>
            <ul className="space-y-1.5 text-sm" style={{ color: "var(--text-secondary)" }}>
              <li>&#x2022; 등록금 및 연구 인센티브 지원</li>
              <li>&#x2022; 개인 PC 및 연구 장비 제공</li>
              <li>&#x2022; 연구 공간 제공</li>
              <li>&#x2022; 학술 논문 및 특허 지도</li>
              <li>&#x2022; 진로 및 진학 상담</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-sm mb-3" style={{ color: "var(--text-primary)" }}>Undergraduate Interns / 학부 연구생</h3>
            <ul className="space-y-1.5 text-sm" style={{ color: "var(--text-secondary)" }}>
              <li>&#x2022; 개인 PC 사용 가능</li>
              <li>&#x2022; 연구 참여 장학금</li>
              <li>&#x2022; 취업 컨설팅</li>
              <li>&#x2022; 학술 논문 및 특허 지도</li>
              <li>&#x2022; 진로 및 진학 상담</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Contact */}
      <section>
        <h2 className="text-lg font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          How to Apply / 지원 방법
        </h2>
        <p className="text-sm mb-2" style={{ color: "var(--text-primary)" }}>
          공영호 교수에게 이메일로 연락해 주세요.
        </p>
        <p className="text-base font-semibold mb-4" style={{ color: "var(--accent)" }}>
          yhgong@ssu.ac.kr
        </p>
        <p className="text-sm mb-2" style={{ color: "var(--text-secondary)" }}>
          아래 자료를 함께 보내주시면 됩니다:
        </p>
        <ul className="text-sm space-y-1" style={{ color: "var(--text-secondary)" }}>
          <li>&#x2022; 간단한 자기소개 및 연구 관심 분야</li>
          <li>&#x2022; 성적 증명서</li>
        </ul>
        <div className="mt-6 pt-3 border-t text-sm" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
          <p>Lab: 창의관 306호 지능형컴퓨팅구조연구실</p>
        </div>
      </section>
    </div>
  );
}
