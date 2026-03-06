"use client";

import Link from "next/link";
import Image from "next/image";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/research", label: "Research" },
  { href: "/members", label: "Members" },
  { href: "/publications", label: "Publications" },
  { href: "/patents", label: "Patents" },
  { href: "/photos", label: "Photos" },
  { href: "/news", label: "News" },
  { href: "/recruiting", label: "Join" },
];

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "dark" || (!saved && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
      setIsDark(true);
      document.documentElement.classList.add("dark");
    }
  }, []);

  const toggleDark = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  return (
    <nav className="sticky top-0 z-50 border-b backdrop-blur-sm" style={{ backgroundColor: "color-mix(in srgb, var(--bg-primary) 90%, transparent)", borderColor: "var(--border)" }}>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link href="/" className="flex items-center gap-2" style={{ color: "var(--text-primary)" }}>
            <Image src="/images/logo.png" alt="AICA Lab" width={64} height={64} />
          </Link>
          <div className="hidden lg:flex items-center gap-6">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-sm font-medium transition-colors hover:opacity-80"
                style={{ color: pathname === item.href ? "var(--accent)" : "var(--text-secondary)" }}
              >
                {item.label}
              </Link>
            ))}
            <button
              onClick={toggleDark}
              className="p-2 rounded-lg border transition-colors"
              style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
              aria-label="Toggle dark mode"
            >
              {isDark ? (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
          </div>
          <div className="flex items-center gap-2 lg:hidden">
            <button
              onClick={toggleDark}
              className="p-2 rounded-lg border"
              style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
              aria-label="Toggle dark mode"
            >
              {isDark ? "☀️" : "🌙"}
            </button>
            <button
              className="p-2"
              onClick={() => setIsOpen(!isOpen)}
              aria-label="Toggle menu"
              style={{ color: "var(--text-primary)" }}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {isOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>
        {isOpen && (
          <div className="lg:hidden pb-4 border-t" style={{ borderColor: "var(--border)" }}>
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="block py-2 text-sm font-medium"
                style={{ color: pathname === item.href ? "var(--accent)" : "var(--text-secondary)" }}
                onClick={() => setIsOpen(false)}
              >
                {item.label}
              </Link>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
}
