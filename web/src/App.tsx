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
const GITHUB = "https://github.com/Harry33t/australian-forest-lfmc";

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
      <header className="page paper-head">
        <h1>
          Live fuel moisture in Australian forests: estimation,
          cross-region generalisation, and calibrated uncertainty
        </h1>
        <div className="authors"><span className="name">Guanxiong Huang</span></div>
        <div className="affil">Northwest A&amp;F University · harry.huang@nwafu.edu.cn</div>
        <div className="links">
          <a href={GITHUB} target="_blank" rel="noreferrer">Code</a>
          <a href="https://doi.org/10.1038/s41597-024-03159-6" target="_blank" rel="noreferrer">Data · Globe-LFMC 2.0</a>
          <a href="https://www.dea.ga.gov.au/" target="_blank" rel="noreferrer">Sentinel-2 · Digital Earth Australia</a>
        </div>
        <p className="abstract">
          Live fuel moisture content (LFMC) — water in vegetation relative to dry mass — strongly
          affects how readily a fire spreads. Continental satellite LFMC performs well over grassland
          but only moderately over forest. Using {sum ? sum.n_measurements.toLocaleString() : "3,000+"} field
          measurements from Globe-LFMC 2.0 paired with Sentinel-2, I evaluate forest LFMC <b>honestly</b>:
          how far models transfer across sites and bioregions, and how to attach a calibrated interval to
          every estimate. The map below shows each site's estimate (colour) and the model's uncertainty
          (height); the figures that follow quantify the generalisation gap and the calibration.
        </p>
      </header>

      <figure className="page wide figure">
        <div className="frame">{sites.length > 0 && <LfmcMap sites={sites} outline={outline} />}</div>
        <figcaption>
          <b>Figure 1.</b> Field-measured LFMC sites across Australia ({sum?.n_sites ?? "—"} sites,
          {" "}{sum ? sum.year_min : ""}–{sum ? sum.year_max : ""}). Column colour encodes the LFMC
          estimate and column height the per-site model uncertainty (random-forest tree spread). Drag
          to rotate, scroll to zoom; switch the colour encoding at lower-left.
        </figcaption>
      </figure>

      <main className="page">
        <section className="sec">
          <div className="num">1 — Background</div>
          <h2>Why forest live fuel moisture is hard to estimate</h2>
          <p>
            Australia-wide satellite LFMC works for grass but is weaker over forest (R² ≈ 0.43;
            Yebra et al., Remote Sensing 2026), because ground measurements are sparse and coarse
            pixels average over local variation. Forest LFMC is also tightly distributed near 100%,
            leaving little dynamic range; shrub and grass span much wider.
          </p>
          {sum && (
            <div className="cards">
              <figure className="fig-card">
                <h3>LFMC by vegetation type</h3>
                <DistChart summary={sum} />
                <div className="cap">
                  <b>Figure 2.</b> Distribution of measured LFMC (Globe-LFMC 2.0, Australia). Box =
                  interquartile range, whiskers = 5th–95th percentile.
                </div>
              </figure>
              <figure className="fig-card">
                <h3>Within-site vs leave-site-out</h3>
                <GapBars sum={sum} />
                <div className="cap">
                  <b>Figure 3.</b> A random split mixes samples of the same site into train and test
                  and overstates skill; leaving whole sites out is the honest test.
                </div>
              </figure>
            </div>
          )}
        </section>

        <section className="sec">
          <div className="num">2 — Generalisation across regions</div>
          <h2>A model trained in one bioregion rarely transfers to another</h2>
          <p>
            Each cell is the R² when a random forest trained on one IBRA bioregion is evaluated on
            another (meteorology features). The diagonal — tested within the training region — can
            reach 0.7, but most off-diagonal cells are low or negative. This cross-region transfer is
            the central difficulty for an operational forest LFMC product.
          </p>
          {loro && (
            <figure className="matrix-wrap">
              <LoroMatrix data={loro} />
              <div className="cap" style={{ maxWidth: 760 }}>
                <b>Figure 4.</b> Cross-region transfer matrix. Rows = training bioregion, columns =
                test bioregion; colour and value give R².
              </div>
            </figure>
          )}
        </section>

        <section className="sec">
          <div className="num">3 — Uncertainty</div>
          <h2>Calibrated prediction intervals</h2>
          <p>
            Split conformal prediction attaches a distribution-free interval to each estimate.
            Empirical coverage closely tracks the nominal level — at a nominal 90%, about 88% of
            held-out measurements fall within the interval. In Figure 1, column height is this
            per-site uncertainty: the tallest, reddest columns mark where the model is least
            confident and where additional field sampling would be most informative.
          </p>
          {conf && (
            <figure className="fig-card" style={{ marginTop: 22 }}>
              <h3>Conformal calibration</h3>
              <Calibration data={conf} />
              <div className="cap">
                <b>Figure 5.</b> Nominal versus empirical coverage of split-conformal intervals,
                against the 1:1 line.
              </div>
            </figure>
          )}
        </section>
      </main>

      <footer className="footer page">
        Data: Globe-LFMC 2.0 (Yebra et al. 2024, <i>Scientific Data</i> 11:332) · Sentinel-2 NBART via
        Digital Earth Australia · IBRA7 bioregions (© Commonwealth of Australia).<br />
        Guanxiong Huang · Northwest A&amp;F University · <a href="mailto:harry.huang@nwafu.edu.cn">harry.huang@nwafu.edu.cn</a>
        {" · "}<a href={GITHUB} target="_blank" rel="noreferrer">source code</a>
      </footer>
    </>
  );
}

function GapBars({ sum }: { sum: Summary }) {
  const h = sum.headline;
  const rows = [
    { label: "Random split (leaky)", v: h.forest_random_met, c: "#caa46a" },
    { label: "Leave-site-out (honest)", v: h.forest_logo_met, c: "#b35c34" },
    { label: "Yebra 2026 ground-truth", v: h.yebra_forest, c: "#7d9153" },
  ];
  const W = 380, bw = 232, x0 = 168;
  return (
    <svg width={W} height={150} style={{ fontFamily: "inherit", maxWidth: "100%" }}>
      {rows.map((r, i) => (
        <g key={i} transform={`translate(0 ${20 + i * 42})`}>
          <text x={x0 - 10} y={16} fontSize={11.5} fill="#3c372f" textAnchor="end">{r.label}</text>
          <rect x={x0} y={4} width={bw} height={18} rx={4} fill="#ece4d6" />
          <rect x={x0} y={4} width={bw * Math.max(0, r.v)} height={18} rx={4} fill={r.c} />
          <text x={x0 + bw * Math.max(0, r.v) + 8} y={18} fontSize={12} fill="#221f1b">{r.v.toFixed(2)}</text>
        </g>
      ))}
      <text x={4} y={148} fontSize={10} fill="#6f665a">Forest LFMC R² · meteorology-only random-forest baseline</text>
    </svg>
  );
}
