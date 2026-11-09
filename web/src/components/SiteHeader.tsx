"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrandGlyph } from "./CellMark";

const LINKS = [
  { href: "/", label: "Analyse" },
  { href: "/model-card", label: "Model card" },
  { href: "/method", label: "Method" },
];

export function SiteHeader() {
  const path = usePathname();
  return (
    <header>
      <nav className="nav" aria-label="Main">
        <Link href="/" className="brand" aria-label="Cell Triage, home">
          <BrandGlyph />
          Cell Triage
        </Link>
        <ul className="nav-links">
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} aria-current={path === l.href ? "page" : undefined}>
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
        <Link className="btn btn-outline pill-sm" href="/#analyse">
          Try a sample
        </Link>
      </nav>
    </header>
  );
}
