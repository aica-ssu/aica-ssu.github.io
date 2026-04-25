"use client";

import Script from "next/script";
import { useEffect, useState } from "react";

declare global {
  interface Window {
    mermaid?: {
      initialize: (cfg: Record<string, unknown>) => void;
      run: (opts?: { querySelector?: string }) => Promise<void>;
    };
  }
}

/**
 * Client-side Mermaid initializer.
 *
 * Loads Mermaid v10 UMD bundle from jsDelivr CDN via `next/script`
 * (strategy="afterInteractive"), then on `onLoad` initializes mermaid
 * and runs it against any `<pre class="mermaid">` blocks emitted by our
 * markdown renderer. Re-runs once after navigation/render.
 *
 * Use in any page that renders markdown via `markdownToHtml` —
 * ```mermaid ... ``` code blocks become `<pre class="mermaid">` HTML.
 */
export default function MermaidScript() {
  const [scriptReady, setScriptReady] = useState(false);

  useEffect(() => {
    if (!scriptReady) return;
    if (typeof window === "undefined" || !window.mermaid) return;
    try {
      window.mermaid.initialize({
        startOnLoad: false,
        theme: "neutral",
        securityLevel: "loose",
        flowchart: { htmlLabels: true, useMaxWidth: true },
      });
      window.mermaid
        .run({ querySelector: "pre.mermaid, div.mermaid" })
        .catch((e) => console.warn("mermaid.run failed", e));
    } catch (e) {
      console.warn("mermaid init failed", e);
    }
  }, [scriptReady]);

  return (
    <Script
      src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
      strategy="afterInteractive"
      onLoad={() => setScriptReady(true)}
    />
  );
}
