"""W3 签名图:跨区域泛化(LORO 迁移矩阵)+ conformal 校准曲线 + 逐点置信带。

产出(outputs/figures/):
  w3_loro_matrix.png    ⭐ region×region 迁移矩阵热图(train i → test j 的 R²)
  w3_conformal_calib.png   名义 vs 经验覆盖率(校准曲线)+ 区间宽度
  w3_conformal_bands.png   示例:LFMC 预测 ± 90% 置信带 vs 真值(按真值排序)

用法:
    python viz/plot_w3.py
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style as S
import common as C

FIG_DIR = "outputs/figures"


def fig_loro_matrix(loro_res, out):
    regions = loro_res["regions"]
    M = loro_res["transfer_matrix"]
    short = [r.replace(" ", "\n", 1) if len(r) > 14 else r for r in regions]
    n = len(regions)
    mat = np.full((n, n), np.nan)
    for i, ri in enumerate(regions):
        for j, rj in enumerate(regions):
            v = M[ri].get(rj)
            if v is not None:
                mat[i, j] = max(v, -1)        # 截断到 -1 便于配色

    fig, ax = plt.subplots(figsize=(8.6, 7))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(short, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7.5)
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(mat[i, j]) < 0.6 else "white")
    ax.set_xlabel("Tested on region →"); ax.set_ylabel("← Trained on region")
    ax.set_title("Cross-region transfer matrix (R²) — train on one bioregion, test on another")
    fig.colorbar(im, ax=ax, shrink=0.8, label="R² (clipped at −1)")
    ax.text(0.0, -0.30, "Diagonal = within-region (train/test split). Off-diagonal mostly low/negative: "
            "naive models don't transfer across\nbioregions — the cross-region gap that motivates SSL + "
            "region-aware calibration. (RF, meteorology features.)",
            transform=ax.transAxes, fontsize=8, color="#555")
    S.savefig(fig, out)


def fig_calibration(conf, out):
    levels = conf["levels"]
    ov = conf["overall"]
    def fetch(a, key):                 # JSON 把 float key 存成字符串
        d = ov.get(str(a)) or ov.get(a)
        return d[key]
    emp = [fetch(a, "coverage") for a in levels]
    width = [fetch(a, "width") for a in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.plot([0, 1], [0, 1], ls="--", color="#999", label="ideal")
    ax1.plot(levels, emp, marker="o", color=S.VEG_COLORS["forest"], lw=2, ms=8, label="conformal")
    for a, e in zip(levels, emp):
        ax1.text(a, e + 0.02, f"{e:.0%}", ha="center", fontsize=8)
    ax1.set_xlabel("Nominal coverage"); ax1.set_ylabel("Empirical coverage")
    ax1.set_title(f"Conformal calibration (R²={conf['r2']:.2f})")
    ax1.set_xlim(0.4, 1); ax1.set_ylim(0.4, 1); ax1.legend(loc="upper left")

    ax2.plot(levels, width, marker="s", color="#1565c0", lw=2, ms=8)
    for a, w in zip(levels, width):
        ax2.text(a, w + 2, f"±{w/2:.0f}", ha="center", fontsize=8)
    ax2.set_xlabel("Nominal coverage"); ax2.set_ylabel("Interval width (% LFMC)")
    ax2.set_title("Interval width grows with confidence")
    fig.suptitle("Per-pixel uncertainty via split conformal", fontweight="bold")
    S.savefig(fig, out)


def fig_bands(conf, out):
    b = conf["bands_example"]
    true = np.array(b["true"]); pred = np.array(b["pred"])
    lo = np.array(b["lo"]); hi = np.array(b["hi"])
    order = np.argsort(pred)               # 按预测排序:预测线平滑、带平行,真值点落带内一眼看覆盖
    x = np.arange(len(true))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(x, lo[order], hi[order], color=S.VEG_COLORS["forest"], alpha=0.2, label="90% conformal band")
    ax.plot(x, pred[order], color=S.VEG_COLORS["forest"], lw=1.6, label="prediction")
    ax.scatter(x, true[order], s=12, color="#c62828", alpha=0.7, label="true LFMC", zorder=3)
    ax.set_xlabel("Test samples (sorted by predicted LFMC)"); ax.set_ylabel("LFMC (%)")
    ax.set_title("Per-sample LFMC prediction with 90% conformal band")
    ax.legend(loc="upper left")
    S.savefig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loro", default=os.path.join(C.RESULTS_DIR, "loro.json"))
    ap.add_argument("--conformal", default=os.path.join(C.RESULTS_DIR, "conformal.json"))
    ap.add_argument("--out", default=FIG_DIR)
    args = ap.parse_args()
    S.apply()
    if os.path.exists(args.loro):
        fig_loro_matrix(json.load(open(args.loro)), os.path.join(args.out, "w3_loro_matrix.png"))
    if os.path.exists(args.conformal):
        conf = json.load(open(args.conformal))
        fig_calibration(conf, os.path.join(args.out, "w3_conformal_calib.png"))
        fig_bands(conf, os.path.join(args.out, "w3_conformal_bands.png"))
    print("完成。")


if __name__ == "__main__":
    main()
