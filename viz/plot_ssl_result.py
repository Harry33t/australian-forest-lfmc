"""⭐ W2c 签名图:SSL 时序预训练的效果。读 outputs/results/ssl_lfmc.json。

两张:
  w2c_ssl_vs_scratch.png   分植被型 LOGO R²:SSL-init vs 从头(frac=1.0)+ Yebra 对标带
  w2c_label_efficiency.png 森林少标签省标注曲线:R² vs 标注站点比例(SSL vs scratch)

用法:
    python viz/plot_ssl_result.py
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


def get(res, mode, frac, veg):
    return res.get(f"{mode}__frac{frac}__{veg}", {}).get("r2")


def fig_vs_scratch(res, out, min_sites=18):
    # 只展示站点数足够、LOGO 可靠的植被型(草地仅 14 站点 → 剔除,脚注说明)
    def nsites(v):
        return res.get(f"ssl__frac1.0__{v}", {}).get("n_sites", 0)
    vegs = [v for v in C.VEG_TYPES if get(res, "ssl", 1.0, v) is not None and nsites(v) >= min_sites]
    dropped = [v for v in C.VEG_TYPES if get(res, "ssl", 1.0, v) is not None and nsites(v) < min_sites]
    if not vegs:
        print("  缺 frac=1.0 结果,跳过 vs_scratch"); return
    x = np.arange(len(vegs)); w = 0.36
    sc = [get(res, "scratch", 1.0, v) for v in vegs]
    ss = [get(res, "ssl", 1.0, v) for v in vegs]
    fig, ax = plt.subplots(figsize=(7.2, 5))
    b1 = ax.bar(x - w/2, sc, w, color="#b0bec5", label="From scratch")
    b2 = ax.bar(x + w/2, ss, w, color=S.VEG_COLORS["forest"], label="SSL pretrain → fine-tune")
    for b in list(b1) + list(b2):
        h = b.get_height()
        if h is not None and np.isfinite(h):
            ax.text(b.get_x()+b.get_width()/2, h + (0.008 if h>=0 else -0.03),
                    f"{h:.2f}", ha="center", va="bottom" if h>=0 else "top", fontsize=9)
    ax.axhspan(0.42, 0.53, color=S.ACCENT, alpha=0.08)
    ax.axhline(0.42, color=S.ACCENT, lw=1, ls=":")
    ax.text(len(vegs)-0.5, 0.475, "Yebra 2026\n0.42–0.53", color=S.ACCENT, fontsize=8, ha="right", va="center")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([S.VEG_LABEL[v] for v in vegs])
    ax.set_ylabel("R² (leave-site-out)")
    lo = min(min(sc), min(ss)); hi = max(max(sc), max(ss))
    ax.set_ylim(min(-0.05, lo - 0.06), max(0.62, hi + 0.14))
    ax.set_title("Temporal self-supervised pretraining vs training from scratch")
    ax.legend(loc="upper left")
    note = "SSL pretraining on unlabeled S2 time-series enables cross-site generalization that from-scratch can't reach."
    if dropped:
        note += f"\n({', '.join(S.VEG_LABEL[v] for v in dropped)} omitted — too few sites for reliable leave-site-out.)"
    ax.text(0.0, -0.15, note, transform=ax.transAxes, fontsize=8.5, color="#555")
    S.savefig(fig, out)


def fig_label_efficiency(res, out, veg="forest"):
    fracs = sorted({float(k.split("frac")[1].split("__")[0])
                    for k in res if k.endswith(f"__{veg}")})
    if len(fracs) < 2:
        print("  少标签曲线需 ≥2 个 frac,跳过"); return
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, col, mk in [("scratch", "#90a4ae", "o"), ("ssl", S.VEG_COLORS["forest"], "s")]:
        ys = [get(res, mode, f, veg) for f in fracs]
        ax.plot(fracs, ys, marker=mk, color=col, lw=2, ms=7,
                label="SSL pretrain" if mode == "ssl" else "From scratch")
        for f, y in zip(fracs, ys):
            if y is not None:
                ax.text(f, y + 0.012, f"{y:.2f}", ha="center", fontsize=8, color=col)
    ax.set_xlabel("Fraction of labeled sites used in training")
    ax.set_ylabel("R² (leave-site-out)")
    ax.set_title(f"Label efficiency — {S.VEG_LABEL.get(veg, veg)} LFMC")
    ax.legend(loc="lower right")
    ax.text(0.0, -0.15, "SSL pretraining on unlabeled S2 time-series should win most at low label fractions "
            "— the case for active learning + label efficiency.",
            transform=ax.transAxes, fontsize=8.5, color="#555")
    S.savefig(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(C.RESULTS_DIR, "ssl_lfmc.json"))
    ap.add_argument("--out", default=FIG_DIR)
    args = ap.parse_args()
    S.apply()
    if not os.path.exists(args.results):
        sys.exit(f"缺 {args.results}(先在 AutoDL 跑 finetune_lfmc.py 并取回)")
    res = json.load(open(args.results))
    fig_vs_scratch(res, os.path.join(args.out, "w2c_ssl_vs_scratch.png"))
    fig_label_efficiency(res, os.path.join(args.out, "w2c_label_efficiency.png"))
    print("完成。")


if __name__ == "__main__":
    main()
