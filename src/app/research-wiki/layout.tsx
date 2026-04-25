import { ReactNode } from "react";
import ResearchWikiSidebar from "@/components/ResearchWikiSidebar";

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
