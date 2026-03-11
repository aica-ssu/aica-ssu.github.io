"use client";

import { useState } from "react";

const HASH = "cadb3b36d671925c7bd64e0edbf12be56ec715dae56e759e33d7445f73bdc8a2";

async function check(input: string) {
  const encoded = new TextEncoder().encode(input);
  const buf = await crypto.subtle.digest("SHA-256", encoded);
  const hex = Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, "0")).join("");
  return hex === HASH;
}

function ExtLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="hover:underline" style={{ color: "var(--accent)" }}>
      {children}
    </a>
  );
}

export default function BackupPage() {
  const [password, setPassword] = useState("");
  const [unlocked, setUnlocked] = useState(false);
  const [error, setError] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (await check(password)) {
      setUnlocked(true);
      setError(false);
    } else {
      setError(true);
    }
  };

  if (!unlocked) {
    return (
      <div className="max-w-md mx-auto px-4 py-32 text-center">
        <h1 className="text-xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
          Access Restricted
        </h1>
        <p className="text-sm mb-6" style={{ color: "var(--text-muted)" }}>
          This page requires a password.
        </p>
        <form onSubmit={handleSubmit} className="flex flex-col gap-3 items-center">
          <input
            type="password"
            value={password}
            onChange={(e) => { setPassword(e.target.value); setError(false); }}
            placeholder="Password"
            className="w-64 px-3 py-2 text-sm border rounded outline-none focus:ring-1"
            style={{
              backgroundColor: "var(--bg-primary)",
              borderColor: error ? "#ef4444" : "var(--border)",
              color: "var(--text-primary)",
            }}
            autoFocus
          />
          {error && <p className="text-xs" style={{ color: "#ef4444" }}>Incorrect password.</p>}
          <button
            type="submit"
            className="px-5 py-2 text-sm font-medium"
            style={{ backgroundColor: "var(--accent)", color: "#ffffff" }}
          >
            Enter
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-16">
      <h1 className="text-2xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
        Research / Study Materials & Links
      </h1>
      <p className="text-sm mb-10" style={{ color: "var(--text-muted)" }}>Internal use only</p>

      {/* Before starting research */}
      <section className="mb-10">
        <h2 className="text-base font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Before Starting Research on AI System/Architecture Optimization
        </h2>

        {/* Basics of Deep Learning */}
        <h3 className="text-sm font-semibold mt-4 mb-2" style={{ color: "var(--text-primary)" }}>
          Basics of Deep Learning (ANN, CNN, RNN, Transformer, etc.)
        </h3>
        <ol className="space-y-2 text-sm list-decimal list-inside" style={{ color: "var(--text-secondary)" }}>
          <li>
            모두를 위한 딥러닝 시즌2 PyTorch<br />
            <ExtLink href="https://deeplearningzerotoall.github.io/season2/lec_pytorch.html">deeplearningzerotoall.github.io</ExtLink>
          </li>
          <li>
            [딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)<br />
            <ExtLink href="https://www.youtube.com/watch?v=AA621UofTUA">YouTube</ExtLink>
          </li>
          <li>
            서울대 이준석 교수님 ML/DL 강의 (Lecture 19부터 Transformer 내용)<br />
            <ExtLink href="https://www.youtube.com/playlist?list=PL0E_1UqNACXA5u65LBjzFCAVSZ4xuBWqj">YouTube Playlist</ExtLink>
          </li>
        </ol>

        {/* GPU */}
        <h3 className="text-sm font-semibold mt-6 mb-2" style={{ color: "var(--text-primary)" }}>
          How AI Models Are Accelerated with GPU?
        </h3>
        <ul className="space-y-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li>GPU는 어떻게 작동할까? &mdash; <ExtLink href="https://www.youtube.com/watch?v=ZdITviTD3VM">YouTube</ExtLink></li>
          <li>차이점 비교를 위한 참고.. CPU는 어떻게 작동할까? &mdash; <ExtLink href="https://www.youtube.com/watch?v=Fg00LN30Ezg">YouTube</ExtLink></li>
          <li>AI가 GPU를 좋아하는 이유 &mdash; <ExtLink href="https://www.youtube.com/watch?v=AKob3yZT0I4">YouTube</ExtLink></li>
          <li>
            GPU CUDA 병렬프로그래밍 &mdash; <ExtLink href="https://www.youtube.com/watch?v=zdqZjVxIHT4&list=PLKZ28p5qq0DGLcO6QZdMSG_jsprRtG15C">YouTube Playlist</ExtLink>
            <br /><span className="text-xs" style={{ color: "var(--text-muted)" }}>5년된 강의이긴 하지만.. 기본적인 GPU 병렬프로그래밍모델에 대해서 위의 영상들보다 좀 더 코드레벨에서 설명하는 강의</span>
          </li>
        </ul>

        {/* System/Architecture Optimization */}
        <h3 className="text-sm font-semibold mt-6 mb-2" style={{ color: "var(--text-primary)" }}>
          Optimizing AI Models in the Perspective of System/Architecture/Hardware
        </h3>
        <ul className="space-y-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li>
            MIT Han Lab EfficientAI &mdash; <ExtLink href="https://efficientml.ai">efficientml.ai</ExtLink>
            <br /><span className="text-xs" style={{ color: "var(--text-muted)" }}>2023 Fall full ver: <ExtLink href="https://hanlab.mit.edu/courses/2023-fall-65940">hanlab.mit.edu</ExtLink></span>
          </li>
          <li>
            QiShao Notes &mdash; <ExtLink href="https://github.com/qishao-chalmers/qishao-notes">GitHub</ExtLink>
            <br /><span className="text-xs" style={{ color: "var(--text-muted)" }}>On HBM, Gem5, especially running llama.cpp on Gem5 &rarr; can be extended to llama rowhammer problem examination</span>
          </li>
        </ul>
      </section>

      {/* Paper Search */}
      <section className="mb-10">
        <h2 className="text-base font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Paper Search
        </h2>
        <ul className="space-y-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li><ExtLink href="https://cspapers.org">cspapers.org</ExtLink></li>
          <li>Edge AI &mdash; <ExtLink href="https://github.com/xumengwei/Edge-AI-Paper-List/">Edge-AI-Paper-List (GitHub)</ExtLink></li>
        </ul>

        <h3 className="text-sm font-semibold mt-4 mb-2" style={{ color: "var(--text-primary)" }}>System-level Optimization</h3>
        <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li><ExtLink href="https://github.com/inpluslab-wuhui/Systems-for-Foundation-Models">Systems-for-Foundation-Models (GitHub)</ExtLink></li>
          <li><ExtLink href="https://github.com/byungsoo-oh/ml-systems-papers">ml-systems-papers (GitHub)</ExtLink></li>
        </ul>

        <h3 className="text-sm font-semibold mt-4 mb-2" style={{ color: "var(--text-primary)" }}>DRAM / NVM Study</h3>
        <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li>Parallel Data Laboratory (CMU)</li>
          <li><ExtLink href="https://ihpcs.ethz.ch/research/publications-safari-research-group.html">ETH Zurich SAFARI Research Group Publications</ExtLink></li>
        </ul>
      </section>

      {/* YouTube Channels */}
      <section className="mb-10">
        <h2 className="text-base font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Recommended YouTube Channels
        </h2>

        <h3 className="text-sm font-semibold mt-4 mb-2" style={{ color: "var(--text-primary)" }}>International</h3>

        <p className="text-xs font-medium mt-3 mb-1" style={{ color: "var(--text-muted)" }}>Short AI Papers (high-level)</p>
        <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li><ExtLink href="https://www.youtube.com/@aipapersacademy">AI Papers Academy</ExtLink></li>
          <li><ExtLink href="https://www.youtube.com/@TwoMinutePapers">Two Minute Papers</ExtLink></li>
        </ul>

        <p className="text-xs font-medium mt-3 mb-1" style={{ color: "var(--text-muted)" }}>Lectures / Papers</p>
        <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li><ExtLink href="https://www.youtube.com/@OnurMutluLectures">ETH Zurich SAFARI Lab</ExtLink></li>
          <li><ExtLink href="https://www.youtube.com/@MITHANLab">MIT HAN Lab</ExtLink> (<ExtLink href="https://efficientml.ai">efficientml.ai</ExtLink>)</li>
          <li><ExtLink href="https://www.youtube.com/@acmsigarch2299">ACM SIGARCH</ExtLink></li>
        </ul>

        <p className="text-xs font-medium mt-3 mb-1" style={{ color: "var(--text-muted)" }}>Semiconductor</p>
        <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li>Packaging &mdash; <ExtLink href="https://www.youtube.com/@semicontalk3223">Semicontalk</ExtLink></li>
        </ul>

        <h3 className="text-sm font-semibold mt-5 mb-2" style={{ color: "var(--text-primary)" }}>Only for KR</h3>
        <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          <li><ExtLink href="https://www.youtube.com/@GadgetSeoul">가젯서울</ExtLink></li>
          <li><ExtLink href="https://www.youtube.com/@unrealtech">안될공학</ExtLink></li>
          <li><ExtLink href="https://www.youtube.com/@jocoding">조코딩</ExtLink> (AI 뉴스 정리)</li>
        </ul>
      </section>

      {/* News Feed */}
      <section className="mb-10">
        <h2 className="text-base font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          News Feed
        </h2>
        <ul className="text-sm" style={{ color: "var(--text-secondary)" }}>
          <li>Geek News &mdash; <ExtLink href="https://news.hada.io/">news.hada.io</ExtLink></li>
        </ul>
      </section>

      {/* Paper Submission */}
      <section className="mb-10">
        <h2 className="text-base font-bold mb-4 pb-1 border-b" style={{ color: "var(--text-primary)", borderColor: "var(--border)" }}>
          Paper Submission
        </h2>

        <h3 className="text-sm font-semibold mt-4 mb-2" style={{ color: "var(--text-primary)" }}>Elsevier</h3>
        <div className="overflow-x-auto">
          <table className="text-sm w-full" style={{ color: "var(--text-secondary)" }}>
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                <th className="text-left py-2 pr-4 font-semibold" style={{ color: "var(--text-primary)" }}>Journal</th>
                <th className="text-left py-2 pr-4 font-semibold" style={{ color: "var(--text-primary)" }}>Abbr.</th>
                <th className="text-right py-2 font-semibold" style={{ color: "var(--text-primary)" }}>IF</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Future Generation Computer Systems</td><td className="pr-4">FGCS</td><td className="text-right">7.5</td></tr>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Expert Systems with Applications</td><td className="pr-4">ESA</td><td className="text-right">8.5</td></tr>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Computers &amp; Security</td><td className="pr-4">COSE</td><td className="text-right">5.6</td></tr>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Journal of Systems Architecture</td><td className="pr-4">JSA</td><td className="text-right">4.5</td></tr>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Internet of Things</td><td className="pr-4">IoT</td><td className="text-right">5.9</td></tr>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Microprocessors and Microsystems</td><td className="pr-4">MICPRO</td><td className="text-right">2.6</td></tr>
            </tbody>
          </table>
        </div>

        <h3 className="text-sm font-semibold mt-6 mb-2" style={{ color: "var(--text-primary)" }}>Letters</h3>
        <div className="overflow-x-auto">
          <table className="text-sm w-full" style={{ color: "var(--text-secondary)" }}>
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                <th className="text-left py-2 pr-4 font-semibold" style={{ color: "var(--text-primary)" }}>Journal</th>
                <th className="text-left py-2 pr-4 font-semibold" style={{ color: "var(--text-primary)" }}>Abbr.</th>
                <th className="text-right py-2 font-semibold" style={{ color: "var(--text-primary)" }}>IF</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Embedded Systems Letters</td><td className="pr-4">ESL</td><td className="text-right">1.6</td></tr>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}><td className="py-1.5 pr-4">Computer Architecture Letters</td><td className="pr-4">CAL</td><td className="text-right">2.3</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
