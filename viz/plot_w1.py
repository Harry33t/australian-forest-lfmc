"""W1 签名图:从 parquet + baseline 结果 JSON 出图。

产出(outputs/figures/):
  w1_site_map.png       澳洲 LFMC 站点地图(按植被型上色)——展示数据覆盖
  w1_lfmc_dist.png      分植被型 LFMC 分布(violin)——展示三类动态范围差异
  w1_gen_gap.png   ⭐  random vs leave-site-out R²(分植被型)——签名图:诚实泛化落差

用法:
    python viz/plot_w1.py
    python viz/plot_w1.py --parquet outputs/lfmc_au.parquet --results outputs/results/baseline_rf_au.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import style as S
import common as C

FIG_DIR = "outputs/figures"


def fig_site_map(df, out):
    fig, ax = plt.subplots(figsize=(7, 6.2))
    # 按站点聚合(一个站点画一点),点大小温和反映测量数(封顶,避免单站独大)
    # 绘制顺序:草地→灌丛→森林,让主打的森林点画在最上层、最显眼
    handles = {}
    for veg in [C.VEG_GRASS, C.VEG_SHRUB, C.VEG_FOREST]:
        sub = df[df["veg_type"] == veg]
        if not len(sub):
            continue
        g = sub.groupby("site").agg(lon=("lon", "mean"), lat=("lat", "mean"),
                                     n=("lfmc", "size"))
        size = np.clip(18 + g["n"] * 1.2, 18, 90)
        zo = 3 if veg == C.VEG_FOREST else 2
        h = ax.scatter(g["lon"], g["lat"], s=size, alpha=0.75, zorder=zo,
                       color=S.VEG_COLORS[veg], edgecolor="white", linewidth=0.5,
                       label=f"{S.VEG_LABEL[veg]} ({sub['site'].nunique()} sites, n={len(sub):,})")
        handles[veg] = h
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Globe-LFMC 2.0 — Australian field sites")
    # 图例固定顺序:森林→灌丛→草地
    ax.legend([handles[v] for v in C.VEG_TYPES if v in handles],
              [handles[v].get_label() for v in C.VEG_TYPES if v in handles],
              loc="lower left", fontsize=8.5)
    ax.set_aspect(1.15)
    txt = f"{df['site'].nunique()} sites · {len(df):,} measurements · {df['date'].dt.year.min()}–{df['date'].dt.year.max()}"
    ax.text(0.99, 0.01, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#666")
    S.savefig(fig, out)


def fig_lfmc_dist(df, out):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    data, labels, colors = [], [], []
    for veg in C.VEG_TYPES:
        s = df[df["veg_type"] == veg]["lfmc"].to_numpy()
        if len(s):
            data.append(s); labels.append(f"{S.VEG_LABEL[veg]}\n(n={len(s):,})")
            colors.append(S.VEG_COLORS[veg])
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.65); pc.set_edgecolor(c)
    parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
    ax.set_ylabel("LFMC (% dry weight)")
    ax.set_title("LFMC distribution by vegetation type (Australia)")
    ax.axhline(100, color="#bbb", lw=0.8, ls="--")
    S.savefig(fig, out)


def fig_gen_gap(results, out):
    """⭐ 签名图:每个植被型 random vs LOGO 的 R²,凸显站内泄漏导致的虚高。"""
    by = results["by_veg"]
    vegs = [v for v in C.VEG_TYPES + ["all"] if v in by and "logo" in by[v]]
    x = np.arange(len(vegs)); w = 0.36
    rand = [by[v].get("random", {}).get("r2", np.nan) for v in vegs]
    logo = [by[v]["logo"]["r2"] for v in vegs]

    fig, ax = plt.subplots(figsize=(7.2, 5))
    b1 = ax.bar(x - w/2, rand, w, color=S.PROTO_COLORS["random"], label=S.PROTO_LABEL["random"])
    b2 = ax.bar(x + w/2, logo, w, color=S.PROTO_COLORS["logo"], label=S.PROTO_LABEL["logo"])
    for b in list(b1) + list(b2):
        h = b.get_height()
        if not np.isnan(h):
            ax.text(b.get_x() + b.get_width()/2, h + 0.008, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)
    # Yebra 地面验证对标带
    ax.axhspan(0.42, 0.53, color=S.ACCENT, alpha=0.10)
    ax.axhline(0.42, color=S.ACCENT, lw=1, ls=":")
    ax.text(len(vegs) - 0.5, 0.475, "Yebra 2026\nground-truth\nR² 0.42–0.53",
            color=S.ACCENT, fontsize=8, ha="right", va="center")
    ax.set_xticks(x); ax.set_xticklabels([S.VEG_LABEL[v] for v in vegs])
    ax.set_ylabel("R²"); ax.set_ylim(0, max(0.65, np.nanmax(rand) + 0.1))
    ax.set_title("Honest generalization gap — W1 baseline (no satellite features yet)")
    ax.legend(loc="upper left")
    ax.text(0.0, -0.16, "Random split overstates skill via within-site leakage; "
            "leave-site-out is the honest floor. Satellite SSL (W2) targets the gap.",
            transform=ax.transAxes, fontsize=8.5, color="#555")
    S.savefig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=C.AU_PARQUET)
    ap.add_argument("--results", default=os.path.join(C.RESULTS_DIR, "baseline_rf_au_met.json"))
    ap.add_argument("--out", default=FIG_DIR)
    args = ap.parse_args()
    S.apply()

    df = pd.read_parquet(args.parquet)
    print(f"载入 {args.parquet}:{len(df):,} 行")
    fig_site_map(df, os.path.join(args.out, "w1_site_map.png"))
    fig_lfmc_dist(df, os.path.join(args.out, "w1_lfmc_dist.png"))

    if os.path.exists(args.results):
        results = json.load(open(args.results))
        fig_gen_gap(results, os.path.join(args.out, "w1_gen_gap.png"))
    else:
        print(f"  跳过泛化落差图(缺 {args.results},先跑 baseline_rf.py)")
    print("完成。")


if __name__ == "__main__":
    main()
