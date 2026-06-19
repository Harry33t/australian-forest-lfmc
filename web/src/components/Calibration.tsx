import type { Conformal } from "../types";

// 校准曲线:名义 vs 经验覆盖率(应贴对角线)
export default function Calibration({ data }: { data: Conformal }) {
  const W = 380, H = 300, m = { l: 48, r: 14, t: 14, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const sx = (v: number) => m.l + ((v - 0.4) / 0.6) * iw;
  const sy = (v: number) => m.t + (1 - (v - 0.4) / 0.6) * ih;
  const pts = data.levels.map((a) => {
    const k = String(a);
    return { a, e: data.overall[k]?.coverage ?? 0 };
  });
  const path = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.a)},${sy(p.e)}`).join(" ");

  return (
    <svg width={W} height={H} style={{ fontFamily: "inherit", maxWidth: "100%" }}>
      {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((t) => (
        <g key={t}>
          <line x1={sx(t)} y1={m.t} x2={sx(t)} y2={m.t + ih} stroke="#e7e0d1" />
          <line x1={m.l} y1={sy(t)} x2={m.l + iw} y2={sy(t)} stroke="#e7e0d1" />
          <text x={sx(t)} y={H - 22} fontSize={9} fill="#6f665a" textAnchor="middle">{t.toFixed(1)}</text>
          <text x={m.l - 8} y={sy(t) + 3} fontSize={9} fill="#6f665a" textAnchor="end">{t.toFixed(1)}</text>
        </g>
      ))}
      {/* ideal diagonal */}
      <line x1={sx(0.4)} y1={sy(0.4)} x2={sx(1)} y2={sy(1)} stroke="#b8ad9a" strokeDasharray="5 5" />
      <path d={path} fill="none" stroke="#b35c34" strokeWidth={2.5} />
      {pts.map((p) => (
        <g key={p.a}>
          <circle cx={sx(p.a)} cy={sy(p.e)} r={4.5} fill="#b35c34" />
          <text x={sx(p.a)} y={sy(p.e) - 9} fontSize={9.5} fill="#221f1b" textAnchor="middle">
            {Math.round(p.e * 100)}%
          </text>
        </g>
      ))}
      <text x={m.l + iw / 2} y={H - 4} fontSize={11} fill="#6f665a" textAnchor="middle">Nominal coverage</text>
      <text x={14} y={m.t + ih / 2} fontSize={11} fill="#6f665a" textAnchor="middle"
        transform={`rotate(-90 14 ${m.t + ih / 2})`}>Empirical coverage</text>
    </svg>
  );
}
