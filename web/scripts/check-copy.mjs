// Fails the build if the site uses any pattern from our design no-go list. Run: node scripts/check-copy.mjs
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const RULES = [
  [/—/, "em dash in copy"],
  [/lorem ipsum/i, "lorem ipsum"],
  [/lucide/i, "Lucide icons"],
  [/["'(]\s*(Inter|Geist)\b/, "Inter or Geist font"],
  [/\bInter_|\bGeist\b/, "Inter or Geist font import"],
  [/[\u{1F300}-\u{1FAFF}\u{2728}\u{2B50}]/u, "emoji or sparkle"],
  [/backdrop-filter|glass-?card|glassmorph/i, "glass card"],
  [/gradient\([^)]*(purple|violet|#8b5cf6|#a855f7|#7c3aed)/i, "purple gradient"],
  [/transition:\s*transform|:hover[^{]*\{[^}]*transform/i, "hover animation"],
  [/radial-gradient\([^)]*dot|background-image:\s*radial-gradient\(circle/i, "dot-grid background"],
];

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(tsx?|css|md)$/.test(name) && !p.includes("DESIGN_NOTES")) out.push(p);
  }
  return out;
}

let bad = 0;
for (const file of walk("src")) {
  const text = readFileSync(file, "utf8");
  text.split(/\r?\n/).forEach((line, i) => {
    for (const [re, why] of RULES) {
      if (re.test(line)) {
        console.error(`${file}:${i + 1}: ${why}: ${line.trim().slice(0, 100)}`);
        bad++;
      }
    }
  });
}
if (bad) {
  console.error(`\n${bad} banned-pattern violation(s).`);
  process.exit(1);
}
console.log("copy check passed: no banned patterns");
