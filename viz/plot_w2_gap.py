"""⭐ W2 诚实图:单日卫星快照的"站内 vs 跨站点"鸿沟 —— 为时序 SSL 立靶。

发现(全在同一批 S2 覆盖样本上):
  · 站内随机split:met 与 met+S2 都 ≈0.45(森林)—— 卫星光谱确实携带 LFMC 信号
  · 留站点 LOGO  :两者都崩到 ~0.05–0.11 —— 单日反射率是站点专属,跨站点不迁移
→ 这个"跨站点泛化鸿沟"正是时序自监督预训练 + LORO + conformal 要解决的。

用法:
    python viz/plot_w2_gap.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style as S
import common as C

FIG_DIR = "outputs/figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--met", default=os.path.join(C.RESULTS_DIR, "baseline_rf_au_s2cov_met.json"))
    ap.add_argument("--mets2", default=os.path.join(C.RESULTS_DIR, "baseline_rf_au_s2cov_mets2.json"))
    ap.add_argument("--out", default=os.path.join(FIG_DIR, "w2_generalization_gap.png"))
    args = ap.parse_args()
    S.apply()
    met = json.load(open(args.met))["by_veg"]
    m2 = json.load(open(args.mets2))["by_veg"]

    vegs = [v for v in C.VEG_TYPES if v in met and "logo" in met[v]]
    x = np.arange(len(vegs)); w = 0.2

    def vals(res, proto):
        return [res[v].get(proto, {}).get("r2", np.nan) for v in vegs]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    bars = [
        ("met · random",  vals(met, "random"), "#bbdefb"),
        ("met+S2 · random", vals(m2, "random"), "#64b5f6"),
        ("met · LOGO",    vals(met, "logo"),   "#90a4ae"),
        ("met+S2 · LOGO", vals(m2, "logo"),    "#37474f"),
    ]
    for i, (lab, v, col) in enumerate(bars):
        off = (i - 1.5) * w
        b = ax.bar(x + off, v, w, color=col, label=lab)
        for bb, val in zip(b, v):
            if np.isfinite(val):
                ax.text(bb.get_x()+bb.get_width()/2, val + (0.008 if val>=0 else -0.03),
                        f"{val:.2f}", ha="center", va="bottom" if val>=0 else "top", fontsize=7.5)

    # Yebra 地面验证对标带
    ax.axhspan(0.42, 0.53, color=S.ACCENT, alpha=0.08)
    ax.axhline(0.42, color=S.ACCENT, lw=1, ls=":")
    ax.text(len(vegs)-0.5, 0.475, "Yebra 2026 ground-truth 0.42–0.53",
            color=S.ACCENT, fontsize=7.5, ha="right", va="center")
    ax.axhline(0, color="#999", lw=0.8)

    ax.set_xticks(x); ax.set_xticklabels([S.VEG_LABEL[v] for v in vegs])
    ax.set_ylabel("R²")
    lo = min(np.nanmin(v) for _, v, _ in bars)
    hi = max(np.nanmax(v) for _, v, _ in bars)
    ax.set_ylim(lo - 0.08, hi + 0.16)
    ax.set_title("Single-date satellite features don't cross sites — the gap SSL must close")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.text(0.0, -0.16,
            "Within-site (random): satellite matches meteorology (~0.45, Yebra range) — the spectral signal is real.\n"
            "Cross-site (leave-site-out): both collapse — single snapshots are site-specific. "
            "Temporal SSL + LORO + conformal target this gap.",
            transform=ax.transAxes, fontsize=8.3, color="#555")
    S.savefig(fig, args.out)

    print("\n森林(focus):")
    print(f"  random : met {met['forest']['random']['r2']:+.2f}  →  met+S2 {m2['forest']['random']['r2']:+.2f}")
    print(f"  LOGO   : met {met['forest']['logo']['r2']:+.2f}  →  met+S2 {m2['forest']['logo']['r2']:+.2f}")


if __name__ == "__main__":
    main()
