import type { Summary } from "../types";
import { VEG_COLOR, VEG_LABEL } from "../lib/color";

// 分植被型 LFMC 分布(p05–q1–median–q3–p95 箱须)
export default function DistChart({ summary }: { summary: Summary }) {
  const vegs = Object.keys(summary.veg_stats);
  const W = 380, H = 300, m = { l: 44, r: 14, t: 16, b: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const max = 240;
  const sy = (v: number) => m.t + (1 - v / max) * ih;
  const bw = iw / vegs.length;

  return (
    <svg width={W} height={H} style={{ fontFamily: "inherit", maxWidth: "100%" }}>
      {[0, 50, 100, 150, 200].map((t) => (
        <g key={t}>
          <line x1={m.l} y1={sy(t)} x2={m.l + iw} y2={sy(t)} stroke="rgba(255,255,255,0.06)" />
          <text x={m.l - 7} y={sy(t) + 3} fontSize={9} fill="#8b97a6" textAnchor="end">{t}</text>
        </g>
      ))}
      <line x1={m.l} y1={sy(100)} x2={m.l + iw} y2={sy(100)} stroke="rgba(255,255,255,0.18)" strokeDasharray="3 3" />
      {vegs.map((v, i) => {
        const s = summary.veg_stats[v];
        const cx = m.l + bw * (i + 0.5);
        const col = `rgb(${VEG_COLOR[v].join(",")})`;
        const boxw = Math.min(46, bw * 0.5);
        return (
          <g key={v}>
            <line x1={cx} y1={sy(s.p05)} x2={cx} y2={sy(s.p95)} stroke={col} strokeWidth={2} />
            <rect x={cx - boxw / 2} y={sy(s.q3)} width={boxw} height={sy(s.q1) - sy(s.q3)}
              fill={col} opacity={0.32} stroke={col} rx={4} />
            <line x1={cx - boxw / 2} y1={sy(s.median)} x2={cx + boxw / 2} y2={sy(s.median)}
              stroke={col} strokeWidth={2.5} />
            <text x={cx} y={H - 26} fontSize={11} fill="#e8edf2" textAnchor="middle">{VEG_LABEL[v]}</text>
            <text x={cx} y={H - 12} fontSize={9} fill="#8b97a6" textAnchor="middle">n={s.n.toLocaleString()}</text>
          </g>
        );
      })}
      <text x={12} y={m.t + ih / 2} fontSize={11} fill="#8b97a6" textAnchor="middle"
        transform={`rotate(-90 12 ${m.t + ih / 2})`}>LFMC (% dry weight)</text>
    </svg>
  );
}
