"""统一绘图风格(demo:干净、克制、可直接进网页/PDF)。

所有图用这里的调色板与 rcParams,保证一套视觉语言。import 即生效。
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# 植被型固定配色(全 demo 一致:森林=深绿、灌丛=橙、草地=金)
VEG_COLORS = {
    "forest": "#2e7d32",
    "shrubland": "#ef6c00",
    "grassland": "#c9a227",
    "all": "#455a64",
}
VEG_LABEL = {
    "forest": "Forest",
    "shrubland": "Shrubland",
    "grassland": "Grassland",
    "all": "All",
}

# 协议配色:乐观(站内随机)用浅、诚实(留站点)用深
PROTO_COLORS = {"random": "#90caf9", "logo": "#1565c0"}
PROTO_LABEL = {"random": "Random split (site leakage)", "logo": "Leave-site-out (honest)"}

ACCENT = "#c62828"   # 强调/对标线


def apply():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def savefig(fig, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  图写出:{path}")
