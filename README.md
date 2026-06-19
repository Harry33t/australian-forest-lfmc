# Australian Forest Live Fuel Moisture — estimation, generalisation & uncertainty

**Guanxiong Huang** · Northwest A&F University · harry.huang@nwafu.edu.cn

🌐 **Interactive 3D demo:** https://harry33t.github.io/australian-forest-lfmc/

---

A research prototype on **live fuel moisture content (LFMC)** of Australian forests: an honest
evaluation of how well forest LFMC can be estimated from satellite + meteorology, how far models
generalise across sites and bioregions, and how to attach calibrated uncertainty to each estimate.
Built with Globe-LFMC 2.0 ground truth, Sentinel-2 (Digital Earth Australia), and IBRA7 bioregions.

---

## Why this is hard
LFMC — water in vegetation relative to dry mass — strongly affects fire behaviour. Continental
satellite LFMC performs well over grassland but only moderately over forest (R² ≈ 0.43; Yebra et
al., *Remote Sensing* 2026), because ground measurements are sparse and coarse pixels average over
local variation. This project characterises that difficulty directly rather than reporting a single
optimistic score.

## Key results (honest, leave-site-out)
- **Naive cross-validation overstates skill.** Forest LFMC R² is ~0.45 under a random split but only
  ~0.11 under leave-site-out — the random split leaks information between samples of the same site.
- **Single-date satellite features don't transfer across sites.** They match meteorology in-sample
  but do not improve honest leave-site-out R²; temporal modelling is needed.
- **Cross-region transfer is the core gap.** Training on one IBRA bioregion and testing on another
  is mostly low or negative R²; within-region can reach ~0.7.
- **Self-supervised temporal pretraining** was evaluated; at this data scale it is comparable to
  baselines, so label-efficient cross-region LFMC remains an open direction.
- **Calibrated uncertainty.** Split conformal intervals track their nominal coverage closely
  (e.g. 90% nominal → ~88% empirical).

## Pipeline (CPU, reproducible)
```bash
pip install -r requirements.txt
python src/download_data.py      # Globe-LFMC 2.0 (figshare)
python src/prepare_lfmc.py       # clean → parquet + EDA
python src/baseline_rf.py --protocol both --feature-set met
python src/satellite_dea.py      # Sentinel-2 features via Digital Earth Australia (no auth)
python src/bioregion.py          # spatial join to IBRA7 bioregions
python src/loro.py               # leave-one-region-out + transfer matrix
python src/conformal.py          # split-conformal prediction intervals
python src/export_web.py         # JSON for the web demo
python viz/plot_w1.py            # static figures → outputs/figures/
```

## Web demo
```bash
cd web && npm install --legacy-peer-deps && npm run dev
```
deck.gl + MapLibre 3D map (colour = LFMC estimate, height = uncertainty), interactive cross-region
matrix, and conformal calibration. See `web/README.md`.

## Data & methods
- **Globe-LFMC 2.0** (Yebra et al. 2024, *Sci Data* 11:332) — ground-truth LFMC.
- **Sentinel-2 NBART** via **Digital Earth Australia** (public, no authentication).
- **IBRA7** bioregions (DCCEEW) for region-level generalisation.
- Models: Random Forest baselines + a temporal Transformer with masked self-supervised pretraining;
  evaluation by leave-site-out and leave-one-region-out; uncertainty via split conformal prediction.

## License / data attribution
Code released for research demonstration. Datasets retain their original licences
(Globe-LFMC 2.0 CC-BY-4.0; DEA Sentinel-2; IBRA7 © Commonwealth of Australia).
