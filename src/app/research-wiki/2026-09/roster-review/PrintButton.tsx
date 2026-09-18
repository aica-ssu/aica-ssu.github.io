"use client";

import { useEffect } from "react";

export default function PrintButton() {
  useEffect(() => {
    let previous: { element: HTMLDetailsElement; open: boolean }[] = [];

    const expand = () => {
      previous = Array.from(
        document.querySelectorAll<HTMLDetailsElement>("#roster-research-note details"),
        (element) => ({ element, open: element.open })
      );
      previous.forEach(({ element }) => { element.open = true; });
    };
    const restore = () => {
      previous.forEach(({ element, open }) => { element.open = open; });
      previous = [];
    };

    window.addEventListener("beforeprint", expand);
    window.addEventListener("afterprint", restore);
    return () => {
      window.removeEventListener("beforeprint", expand);
      window.removeEventListener("afterprint", restore);
      restore();
    };
  }, []);

  return <button type="button" onClick={() => window.print()}>PDF · 인쇄</button>;
}
