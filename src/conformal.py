"""split conformal:给 LFMC 预测每点置信带 + 校准曲线(回应"不确定性没量化")。

split conformal(无分布假设,交换性下覆盖率有保证):
  train 拟合 RF → calib 集算残差分位数 → test 形成区间 [pred±q] → 测经验覆盖率 vs 名义。
产出:多个名义置信水平的经验覆盖率(校准曲线)+ 平均区间宽度;分植被型也报一份。

用法:
    python src/conformal.py --parquet outputs/lfmc_au_s2.parquet --feature-set met+s2
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
import common as C

LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]   # 名义置信水平


def rf(seed=0):
    return RandomForestRegressor(n_estimators=300, max_depth=24, min_samples_leaf=4,
                                 max_features="sqrt", n_jobs=-1, random_state=seed)


def Xy(df, feats):
    X = df[feats].copy()
    for c in feats:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X.to_numpy(float), df["lfmc"].to_numpy(float)


def split_conformal(Xtr, ytr, Xca, yca, Xte, yte, levels, seed):
    """对称 split conformal。返回每个 level 的 (经验覆盖率, 平均区间宽度) + test 预测/上下界(用最高 level)。"""
    m = rf(seed).fit(Xtr, ytr)
    res = np.abs(yca - m.predict(Xca))           # calib 残差
    pred = m.predict(Xte)
    out = {}
    for a in levels:
        q = np.quantile(res, np.clip(a * (1 + 1 / len(res)), 0, 1))  # 有限样本校正
        lo, hi = pred - q, pred + q
        cover = float(np.mean((yte >= lo) & (yte <= hi)))
        out[a] = {"coverage": cover, "width": float(2 * q)}
    # 用 0.9 level 的区间给可视化
    q90 = np.quantile(res, np.clip(0.9 * (1 + 1 / len(res)), 0, 1))
    bands = {"pred": pred.tolist(), "lo": (pred - q90).tolist(),
             "hi": (pred + q90).tolist(), "true": yte.tolist()}
    return out, bands, float(r2_score(yte, pred))


def run(df, feats, levels, seed):
    X, y = Xy(df, feats)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a, b = int(0.5 * n), int(0.75 * n)           # 50% train / 25% calib / 25% test
    tr, ca, te = idx[:a], idx[a:b], idx[b:]
    return split_conformal(X[tr], y[tr], X[ca], y[ca], X[te], y[te], levels, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/lfmc_au_s2.parquet")
    ap.add_argument("--feature-set", default="met+s2", choices=list(C.FEATURE_SETS))
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "conformal.json"))
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    feats = [f for f in C.FEATURE_SETS[args.feature_set] if f in df.columns]
    print(f"{args.parquet}:{len(df):,} 行,feature-set={args.feature_set}")

    overall, bands, r2 = run(df, feats, LEVELS, args.seed)
    print(f"\n整体(R²={r2:.3f}):名义 → 经验覆盖率 / 区间宽度")
    for a in LEVELS:
        print(f"  {a:.0%}  →  {overall[a]['coverage']:.0%}  (±宽 {overall[a]['width']:.0f}% LFMC)")

    by_veg = {}
    for v in C.VEG_TYPES:
        s = df[df["veg_type"] == v]
        if len(s) >= 120:
            ov, _, _ = run(s, feats, LEVELS, args.seed)
            by_veg[v] = ov

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"feature_set": args.feature_set, "levels": LEVELS,
               "overall": overall, "by_veg": by_veg, "r2": r2,
               "bands_example": {k: v[:120] for k, v in bands.items()}},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n写出:{args.out}")


if __name__ == "__main__":
    main()
