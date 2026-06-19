import { useEffect, useState } from "react";
import LfmcMap from "./components/LfmcMap";
import LoroMatrix from "./components/LoroMatrix";
import Calibration from "./components/Calibration";
import DistChart from "./components/DistChart";
import type { Site, Loro, Conformal, Summary } from "./types";

const base = import.meta.env.BASE_URL;
async function load<T>(f: string): Promise<T> {
  const r = await fetch(`${base}data/${f}`);
  return r.json();
}

export default function App() {
  const [sites, setSites] = useState<Site[]>([]);
  const [outline, setOutline] = useState<unknown | null>(null);
  const [loro, setLoro] = useState<Loro | null>(null);
  const [conf, setConf] = useState<Conformal | null>(null);
  const [sum, setSum] = useState<Summary | null>(null);

  useEffect(() => {
    load<Site[]>("sites.json").then(setSites);
    load("australia.json").then(setOutline);
    load<Loro>("loro.json").then(setLoro);
    load<Conformal>("conformal.json").then(setConf);
    load<Summary>("summary.json").then(setSum);
  }, []);

  return (
    <>
      {sites.length > 0 && <LfmcMap sites={sites} outline={outline} />}

      {/* the stats strip under the hero */}
      {sum && (
        <div className="section" style={{ paddingTop: 64, paddingBottom: 24 }}>
          <div className="stats" style={{ justifyContent: "space-between" }}>
            <Stat v={sum.n_measurements.toLocaleString()} l="LFMC measurements" />
            <Stat v={sum.n_sites.toString()} l="field sites" />
            <Stat v={sum.n_bioregions.toString()} l="IBRA bioregions" />
            <Stat v={`${sum.year_min}–${sum.year_max}`} l="years" />
            <Stat v="Sentinel-2" l="10 m, Digital Earth Australia" />
          </div>
        </div>
      )}

      {/* problem */}
      <section className="section" style={{ paddingTop: 30 }}>
        <div className="kicker">Background</div>
        <h2>Forest live fuel moisture is hard to estimate from space</h2>
        <p className="lead">
          Live fuel moisture content (LFMC) — water in vegetation relative to dry mass — strongly
          affects how readily a fire spreads. Continental satellite LFMC performs well over grassland
          but only moderately over forest (R² ≈ 0.43; Yebra et al., Remote Sensing 2026), because
          ground measurements are sparse and coarse pixels average over local variation. Two questions
          follow: how far does a forest LFMC model generalise to unseen sites and regions, and how
          should each estimate carry an uncertainty?
        </p>
        {sum && (
          <div className="grid2">
            <div className="card panel">
              <h3>LFMC by vegetation type</h3>
              <div className="sub">Globe-LFMC 2.0, Australia. Forest clusters near 100%; shrub and grass span a much wider range.</div>
              <DistChart summary={sum} />
            </div>
            <div className="card panel">
              <h3>Within-site vs leave-site-out</h3>
              <div className="sub">A random split mixes nearby samples of the same site into train and test. Leaving whole sites out is the stricter, more realistic test.</div>
              <GapBars sum={sum} />
            </div>
          </div>
        )}
      </section>

      {/* cross-region */}
      <section className="section" style={{ paddingTop: 30 }}>
        <div className="kicker">Generalisation across regions</div>
        <h2>A model trained in one bioregion rarely transfers to another</h2>
        <p className="lead">
          Each cell is the R² when a random forest trained on one IBRA bioregion is evaluated on
          another (meteorology features). The diagonal — tested within the training region — can reach
          0.7, but most off-diagonal cells are low or negative. This cross-region transfer is the core
          difficulty for an operational forest LFMC product.
        </p>
        {loro && (
          <div className="card panel" style={{ marginTop: 30, display: "inline-block" }}>
            <h3>Cross-region transfer matrix (R²)</h3>
            <div className="sub">Train ↓ · Test → · Random Forest on meteorology features</div>
            <LoroMatrix data={loro} />
          </div>
        )}
      </section>

      {/* uncertainty */}
      <section className="section" style={{ paddingTop: 30 }}>
        <div className="kicker">Uncertainty</div>
        <h2>Calibrated prediction intervals</h2>
        <p className="lead">
          Split conformal prediction attaches a distribution-free interval to each estimate. Empirical
          coverage closely tracks the nominal level — at a nominal 90%, about 88% of held-out
          measurements fall within the interval.
        </p>
        {conf && (
          <div className="grid2">
            <div className="card panel">
              <h3>Conformal calibration</h3>
              <div className="sub">Nominal vs empirical coverage, close to the 1:1 line.</div>
              <Calibration data={conf} />
            </div>
            <div className="card panel">
              <h3>Reading the 3D map</h3>
              <div className="sub">&nbsp;</div>
              <p className="note">
                In the map above, column height encodes per-site uncertainty: the tallest, reddest
                columns mark where the model is least confident — and where an additional field campaign
                would be most informative. Conformal turns that qualitative cue into formal intervals,
                so a downstream fire-risk model receives a confidence band rather than a single number.
              </p>
            </div>
          </div>
        )}
      </section>

      <div className="footer">
        Data: Globe-LFMC 2.0 (Yebra et al. 2024) · Sentinel-2 NBART via Digital Earth Australia · IBRA7 bioregions.
        <br />
        Guanxiong Huang · Northwest A&amp;F University · <a href="mailto:harry.huang@nwafu.edu.cn">harry.huang@nwafu.edu.cn</a>
      </div>
    </>
  );
}

function Stat({ v, l }: { v: string; l: string }) {
  return (
    <div className="stat">
      <div className="v">{v}</div>
      <div className="l">{l}</div>
    </div>
  );
}

function GapBars({ sum }: { sum: Summary }) {
  const h = sum.headline;
  const rows = [
    { label: "Random split (leaky)", v: h.forest_random_met, c: "#4fa3ff" },
    { label: "Leave-site-out (honest)", v: h.forest_logo_met, c: "#2ec27e" },
    { label: "Yebra 2026 ground-truth", v: h.yebra_forest, c: "#ef6c5a" },
  ];
  const W = 380, bw = 250, x0 = 150;
  return (
    <svg width={W} height={150} style={{ fontFamily: "inherit", maxWidth: "100%" }}>
      {rows.map((r, i) => (
        <g key={i} transform={`translate(0 ${20 + i * 42})`}>
          <text x={x0 - 10} y={16} fontSize={11.5} fill="#c7cfda" textAnchor="end">{r.label}</text>
          <rect x={x0} y={4} width={bw} height={18} rx={4} fill="rgba(255,255,255,0.06)" />
          <rect x={x0} y={4} width={bw * Math.max(0, r.v)} height={18} rx={4} fill={r.c} />
          <text x={x0 + bw * Math.max(0, r.v) + 8} y={18} fontSize={12} fill="#e8edf2">{r.v.toFixed(2)}</text>
        </g>
      ))}
      <text x={4} y={148} fontSize={10} fill="#8b97a6">Forest LFMC R² · meteorology-only RF baseline</text>
    </svg>
  );
}
