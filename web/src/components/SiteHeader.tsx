"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Cell", n: "01" },
  { href: "/smear", label: "Smear", n: "02" },
  { href: "/model-card", label: "Model card", n: "03" },
  { href: "/method", label: "Method", n: "04" },
];

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className="masthead">
      <nav className="wrap masthead-inner" aria-label="Main">
        {/* The brand is drawn as a glass-slide label: specimen code on the left, name on the right. */}
        <Link href="/" className="slide-label" aria-label="Cell Triage, home">
          <span className="code" aria-hidden="true">CT-01</span>
          <span className="name">
            <b>Cell Triage</b>
            <span>ALL · single-cell</span>
          </span>
        </Link>
        <ul className="nav-links">
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} aria-current={path === l.href ? "page" : undefined}>
                <span className="n" aria-hidden="true">{l.n}</span>
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
