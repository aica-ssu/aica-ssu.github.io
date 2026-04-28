import { ReactNode } from "react";
import type { Metadata } from "next";
import ResearchWikiSidebar from "@/components/ResearchWikiSidebar";

export const metadata: Metadata = {
  robots: {
    index: false,
    follow: false,
    nocache: true,
    googleBot: {
      index: false,
      follow: false,
      noimageindex: true,
      "max-video-preview": -1,
      "max-image-preview": "none",
      "max-snippet": -1,
    },
  },
};

export default function ResearchWikiLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex">
      <ResearchWikiSidebar />
      <main className="flex-1 min-w-0">
        {children}
      </main>
    </div>
  );
}
