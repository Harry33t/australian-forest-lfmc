"""⭐ 签名图:加入卫星光谱特征后的分植被型 R² 提升(对标 Yebra)。

公平对比:两条 baseline 跑在**同一批 S2 覆盖样本**上(同样的行、同样 LOGO 协议),
唯一差别是特征集 —— 隔离出"卫星光谱(NDII 等)到底带来多少提升"。

输入:两个 baseline 结果 JSON(--before / --after),如:
    python src/baseline_rf.py --parquet outputs/lfmc_au_s2.parquet --feature-set met    --tag s2cov_met
    python src/baseline_rf.py --parquet outputs/lfmc_au_s2.parquet --feature-set met+s2 --tag s2cov_mets2

用法:
    python viz/plot_improvement.py \
        --before outputs/results/baseline_rf_au_s2cov_met.json \
        --after  outputs/results/baseline_rf_au_s2cov_mets2.json
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


def r2_by_veg(results, proto="logo"):
    return {v: d.get(proto, {}).get("r2") for v, d in results["by_veg"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=os.path.join(C.RESULTS_DIR, "baseline_rf_au_s2cov_met.json"))
    ap.add_argument("--after", default=os.path.join(C.RESULTS_DIR, "baseline_rf_au_s2cov_mets2.json"))
    ap.add_argument("--proto", default="logo", choices=["logo", "random"])
    ap.add_argument("--out", default=os.path.join(FIG_DIR, "w2_r2_improvement.png"))
    args = ap.parse_args()
    S.apply()

    for f in (args.before, args.after):
        if not os.path.exists(f):
            sys.exit(f"缺结果 {f}。先在 lfmc_au_s2.parquet 上跑两次 baseline_rf(见本文件 docstring)。")
    before = json.load(open(args.before)); after = json.load(open(args.after))
    rb, ra = r2_by_veg(before, args.proto), r2_by_veg(after, args.proto)

    vegs = [v for v in C.VEG_TYPES + ["all"] if v in rb and v in ra
            and rb[v] is not None and ra[v] is not None]
    x = np.arange(len(vegs)); w = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    b1 = ax.bar(x - w/2, [rb[v] for v in vegs], w, color="#b0bec5",
                label="Met only (W1 floor)")
    b2 = ax.bar(x + w/2, [ra[v] for v in vegs], w, color=S.VEG_COLORS["forest"],
                label="Met + Sentinel-2 (NDII/NDVI/…)")
    for v, xi in zip(vegs, x):
        lo, hi = rb[v], ra[v]
        for val, off in [(lo, -w/2), (hi, w/2)]:
            ax.text(xi + off, val + 0.008, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        # 提升箭头 + Δ 标注
        d = hi - lo
        ax.annotate("", xy=(xi + w/2, hi), xytext=(xi - w/2, lo),
                    arrowprops=dict(arrowstyle="->", color=S.ACCENT, lw=1.4, alpha=0.8))
        ax.text(xi, max(hi, lo) + 0.04, f"Δ{d:+.2f}", ha="center",
                color=S.ACCENT, fontsize=9, fontweight="bold")

    # Yebra 地面验证对标带
    ax.axhspan(0.42, 0.53, color=S.ACCENT, alpha=0.08)
    ax.axhline(0.42, color=S.ACCENT, lw=1, ls=":")
    ax.text(len(vegs) - 0.45, 0.475, "Yebra 2026\nground-truth\n0.42–0.53",
            color=S.ACCENT, fontsize=8, ha="right", va="center")

    ax.set_xticks(x); ax.set_xticklabels([S.VEG_LABEL[v] for v in vegs])
    ax.set_ylabel(f"R² ({S.PROTO_LABEL[args.proto].split('(')[0].strip()})")
    top = max(max(rb[v], ra[v]) for v in vegs)
    ax.set_ylim(0, max(0.6, top + 0.12))
    n = after["by_veg"].get("forest", {}).get(args.proto, {}).get("n", "?")
    ax.set_title("Adding satellite spectral features lifts forest LFMC R²")
    ax.legend(loc="upper left")
    ax.text(0.0, -0.15, "Same S2-covered samples, same leave-site-out protocol — "
            "only the feature set differs. NDII is the strongest LFMC predictor.",
            transform=ax.transAxes, fontsize=8.5, color="#555")
    S.savefig(fig, args.out)
    # 控制台小结
    print("\n分植被型 R² 提升(LOGO):")
    for v in vegs:
        print(f"  {S.VEG_LABEL[v]:10s}  {rb[v]:+.3f} → {ra[v]:+.3f}   Δ{ra[v]-rb[v]:+.3f}")


if __name__ == "__main__":
    main()
