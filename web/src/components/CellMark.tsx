/**
 * Lymphoblast motif in the site greens: thin cytoplasm rim, large slightly indented nucleus with fine chromatin
 * stipple and one off-centre nucleolus. Asymmetric on purpose so it reads as a cell, not a face. Decorative.
 */
const STIPPLE: [number, number, number][] = [
  [78, 70, 2.2], [96, 62, 1.6], [116, 70, 2], [132, 86, 1.5], [70, 92, 1.8], [90, 88, 1.4], [108, 96, 2.1],
  [128, 108, 1.7], [64, 116, 1.5], [84, 112, 2], [102, 120, 1.4], [120, 128, 1.9], [78, 134, 1.6], [96, 142, 2.1],
  [114, 146, 1.4], [140, 120, 1.3], [88, 156, 1.5], [62, 102, 1.2], [136, 140, 1.6], [104, 78, 1.3],
];

export function CellMark({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 200 200" aria-hidden="true" focusable="false">
      <path
        d="M101 9c48 2 86 37 88 87 2 52-36 95-90 95-50 0-90-37-90-88C9 52 50 7 101 9z"
        fill="#d4f5c0"
        stroke="#1a7a3a"
        strokeWidth="2.5"
      />
      <path
        d="M103 30c38 1 66 30 66 68 0 21-6 33-19 43-6 5-6 13-12 20-12 13-30 17-45 15-36-4-60-31-60-68 0-44 30-79 70-78z"
        fill="#1a7a3a"
      />
      {STIPPLE.map(([x, y, r], i) => (
        <circle key={i} cx={x} cy={y} r={r} fill="#12351d" opacity=".55" />
      ))}
      <ellipse cx="124" cy="78" rx="12" ry="9" fill="#b5f800" transform="rotate(-20 124 78)" />
    </svg>
  );
}

/** Small glyph next to the wordmark (mirrors xohari's brand glyph). */
export function BrandGlyph() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <circle cx="12" cy="12" r="10" fill="#d4f5c0" />
      <path d="M12.3 5c3.6.1 6 2.7 6 6.2 0 2-.6 3-1.7 4-.6.5-.6 1.3-1.2 2-1.2 1.2-2.8 1.6-4.2 1.4C7.9 18.3 5.8 15.8 5.8 12.4 5.8 8.3 8.6 4.9 12.3 5z" fill="#1a7a3a" />
      <ellipse cx="14.3" cy="9.4" rx="1.6" ry="1.2" fill="#b5f800" />
    </svg>
  );
}
