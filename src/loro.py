"""leave-one-region-out(LORO)+ region×region 迁移矩阵(你的签名实验)。

回应 Yebra/Williamson 的跨区域泛化关切:模型换一个 bioregion 还行不行?
两个产出:
  ① LORO:对每个 held-out bioregion,train=其余所有区域,test=该区域 → 诚实的"迁到没见过的区域"R²。
  ② 迁移矩阵:train 单区域 i → test 区域 j 的 R²(对角=区内 split)。热图一眼看出哪些区域可互迁。

用 RF(快、无需 GPU);默认 lfmc_au.parquet(18 区、气象特征)。

用法:
    python src/loro.py                                   # met 特征
    python src/loro.py --parquet outputs/lfmc_au_s2.parquet --feature-set met+s2
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
import common as C


def rf(seed=0):
    return RandomForestRegressor(n_estimators=200, max_depth=24, min_samples_leaf=4,
                                 max_features="sqrt", n_jobs=-1, random_state=seed)


def Xy(df, feats):
    X = df[feats].copy()
    for c in feats:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X.to_numpy(float), df["lfmc"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=C.AU_PARQUET)
    ap.add_argument("--feature-set", default="met", choices=list(C.FEATURE_SETS))
    ap.add_argument("--min-samples", type=int, default=40, help="region 样本下限")
    ap.add_argument("--min-sites", type=int, default=2, help="region 站点下限")
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "loro.json"))
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if "bioregion" not in df.columns:
        sys.exit("parquet 缺 bioregion 列,先跑 src/bioregion.py")
    feats = [f for f in C.FEATURE_SETS[args.feature_set] if f in df.columns]

    # 选样本/站点足够的 region
    g = df.groupby("bioregion").agg(n=("lfmc", "size"), sites=("site", "nunique"))
    regions = sorted(g[(g["n"] >= args.min_samples) & (g["sites"] >= args.min_sites)].index.tolist())
    print(f"参与 LORO 的 bioregion({len(regions)}):")
    for r in regions:
        print(f"  {r:30s} n={int(g.loc[r,'n']):5d} 站点={int(g.loc[r,'sites'])}")
    if len(regions) < 2:
        sys.exit("可用 region < 2")

    sub = {r: df[df["bioregion"] == r] for r in regions}

    # ① LORO:train=其余区域,test=held-out
    loro = {}
    for held in regions:
        tr = pd.concat([sub[r] for r in regions if r != held])
        Xtr, ytr = Xy(tr, feats); Xte, yte = Xy(sub[held], feats)
        m = rf(args.seed).fit(Xtr, ytr)
        pred = m.predict(Xte)
        loro[held] = {"r2": float(r2_score(yte, pred)),
                      "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                      "n": int(len(yte))}
        print(f"  LORO held={held:28s} R²={loro[held]['r2']:+.3f}")

    # ② 迁移矩阵:train i → test j(对角=区内 80/20)
    matrix = {}
    for ri in regions:
        Xi, yi = Xy(sub[ri], feats)
        row = {}
        for rj in regions:
            if ri == rj:
                Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.3, random_state=args.seed)
                m = rf(args.seed).fit(Xtr, ytr); pred = m.predict(Xte)
            else:
                m = rf(args.seed).fit(Xi, yi)
                Xj, yj = Xy(sub[rj], feats); pred = m.predict(Xj); yte = yj
            row[rj] = float(r2_score(yte, pred)) if len(yte) > 5 else None
        matrix[ri] = row

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {"feature_set": args.feature_set, "parquet": args.parquet,
           "regions": regions, "loro": loro, "transfer_matrix": matrix}
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n写出:{args.out}")
    mean_loro = np.mean([v["r2"] for v in loro.values()])
    print(f"LORO 平均 R²={mean_loro:+.3f}(对每个区域都是没见过的)")


if __name__ == "__main__":
    main()
