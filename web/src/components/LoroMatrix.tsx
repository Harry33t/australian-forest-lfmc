import { useState } from "react";
import type { Loro } from "../types";
import { r2Color } from "../lib/color";

const short = (s: string) => (s.length > 16 ? s.split(" ").slice(0, 2).join(" ") : s);

export default function LoroMatrix({ data }: { data: Loro }) {
  const R = data.regions;
  const [hov, setHov] = useState<{ i: number; j: number } | null>(null);
  const N = R.length;
  const cell = 42, lab = 116, pad = 8;
  const W = lab + N * cell + pad, H = lab + N * cell + pad;

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={W} height={H} style={{ fontFamily: "inherit" }}>
        {/* column labels */}
        {R.map((r, j) => (
          <text key={"c" + j} x={lab + j * cell + cell / 2} y={lab - 6}
            transform={`rotate(-40 ${lab + j * cell + cell / 2} ${lab - 6})`}
            fontSize={10} fill="#8b97a6" textAnchor="start">{short(r)}</text>
        ))}
        {R.map((ri, i) => (
          <g key={"r" + i}>
            <text x={lab - 8} y={lab + i * cell + cell / 2 + 3} fontSize={10}
              fill="#8b97a6" textAnchor="end">{short(ri)}</text>
            {R.map((rj, j) => {
              const v = data.transfer_matrix[ri]?.[rj];
              const on = hov && hov.i === i && hov.j === j;
              return (
                <g key={j}>
                  <rect x={lab + j * cell} y={lab + i * cell} width={cell - 2} height={cell - 2}
                    rx={4} fill={v == null ? "#1a212b" : r2Color(v)}
                    stroke={on ? "#fff" : "transparent"} strokeWidth={on ? 2 : 0}
                    onMouseEnter={() => setHov({ i, j })} onMouseLeave={() => setHov(null)} />
                  {v != null && (
                    <text x={lab + j * cell + (cell - 2) / 2} y={lab + i * cell + (cell - 2) / 2 + 3}
                      fontSize={9.5} fill="#f2eadf" opacity={Math.abs(v) < 0.06 ? 0.55 : 0.95}
                      textAnchor="middle" pointerEvents="none">{v.toFixed(2)}</text>
                  )}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <div className="note" style={{ marginTop: 10 }}>
        {hov ? (
          <span>
            Train on <b style={{ color: "#e8edf2" }}>{R[hov.i]}</b> → test on{" "}
            <b style={{ color: "#e8edf2" }}>{R[hov.j]}</b>:&nbsp;
            R² = <b style={{ color: "#e8edf2" }}>{(data.transfer_matrix[R[hov.i]]?.[R[hov.j]] ?? NaN).toFixed(2)}</b>
            {hov.i === hov.j ? " (within-region)" : ""}
          </span>
        ) : (
          <span>Diagonal = within a region. Off-diagonal red = a model trained in one bioregion fails in another.</span>
        )}
      </div>
    </div>
  );
}
