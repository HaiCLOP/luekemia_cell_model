/** Microscope reticle drawn over the field of view: crosshair, ring ticks and a 10 µm scale bar. Decorative. */
export function Reticle({ className }: { className?: string }) {
  const ticks = Array.from({ length: 36 }, (_, i) => i * 10);
  return (
    <svg className={className} viewBox="0 0 200 200" aria-hidden="true" focusable="false">
      <circle cx="100" cy="100" r="99" fill="none" stroke="rgba(255,255,255,.35)" strokeWidth=".8" />
      {ticks.map((a) => {
        const long = a % 90 === 0;
        const r1 = long ? 88 : 93;
        const rad = (a * Math.PI) / 180;
        return (
          <line key={a} x1={100 + r1 * Math.cos(rad)} y1={100 + r1 * Math.sin(rad)} x2={100 + 99 * Math.cos(rad)} y2={100 + 99 * Math.sin(rad)}
            stroke="rgba(255,255,255,.55)" strokeWidth={long ? 1 : 0.6} />
        );
      })}
      <line x1="100" y1="6" x2="100" y2="80" stroke="rgba(255,255,255,.28)" strokeWidth=".6" strokeDasharray="2 3" />
      <line x1="100" y1="120" x2="100" y2="194" stroke="rgba(255,255,255,.28)" strokeWidth=".6" strokeDasharray="2 3" />
      <line x1="6" y1="100" x2="80" y2="100" stroke="rgba(255,255,255,.28)" strokeWidth=".6" strokeDasharray="2 3" />
      <line x1="120" y1="100" x2="194" y2="100" stroke="rgba(255,255,255,.28)" strokeWidth=".6" strokeDasharray="2 3" />
      <g transform="translate(120 160)">
        <line x1="0" y1="0" x2="36" y2="0" stroke="#fff" strokeWidth="1.6" />
        <line x1="0" y1="-3" x2="0" y2="3" stroke="#fff" strokeWidth="1" />
        <line x1="36" y1="-3" x2="36" y2="3" stroke="#fff" strokeWidth="1" />
        <text x="18" y="-6" textAnchor="middle" fill="#fff" fontSize="7" fontFamily="monospace">10 µm</text>
      </g>
    </svg>
  );
}
