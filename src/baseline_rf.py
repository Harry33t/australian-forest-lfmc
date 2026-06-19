"""W1 baseline:分植被型的 RandomForest LFMC 回归(零卫星,先用站点+时间特征)。

目的:复现 Yebra 量级的"分植被型 R²",作为后续 SSL+主动学习+conformal 的对照底。
对标(诚实版):Yebra RS 2026 对 Globe-LFMC 2.0 地面真值验证 R²≈0.42(同质站点)~0.53。
注意:W1 特征里 *没有卫星反射率*(NDII 等是最强预测因子),所以这版 R² 是地板,
W2 接 DEA Sentinel-2 后预计明显抬升 —— 这正是"森林 R² 提升对比图"的起点。

两种评估协议:
  random   : 站点内随机 80/20(乐观,会因同站点泄漏而虚高)
  logo     : leave-one-group-out 的站点级 CV —— 留出整个站点测试(诚实,接 W3 的 LORO)

默认按植被型各训各的(forest/shrub/grass 分别一个 RF),分层报 R²/RMSE。

用法:
    python src/baseline_rf.py                         # 澳洲子集, logo 协议
    python src/baseline_rf.py --protocol random
    python src/baseline_rf.py --parquet outputs/lfmc_clean.parquet --region global
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
import common as C


def feature_matrix(df, feature_set):
    """组装特征矩阵 X(按 feature_set 预设取列);数值缺失填中位数。

    feature_set:base(站点+时间)/ met(+气象)/ s2(+卫星)/ met+s2(全量)。
    """
    feats = [f for f in C.FEATURE_SETS[feature_set] if f in df.columns]
    X = df[feats].copy()
    for col in feats:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    return X.to_numpy(dtype=float), feats


def make_rf(seed):
    # 贴近 Yebra RS 2026 量级(她约 76 树/depth 24);demo 不调参,够稳即可
    return RandomForestRegressor(
        n_estimators=200, max_depth=24, min_samples_leaf=4,
        max_features="sqrt", n_jobs=-1, random_state=seed,
    )


def metrics(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def eval_random(X, y, seed):
    """站点内随机 80/20(乐观参考)。"""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    m = make_rf(seed).fit(Xtr, ytr)
    return metrics(yte, m.predict(Xte))


def eval_logo(X, y, groups, seed, n_splits=5):
    """站点级 GroupKFold:同一站点只在 train 或 test,杜绝时空泄漏(诚实)。"""
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        return None
    gkf = GroupKFold(n_splits=k)
    yt, yp = [], []
    for tr, te in gkf.split(X, y, groups):
        m = make_rf(seed).fit(X[tr], y[tr])
        yt.append(y[te]); yp.append(m.predict(X[te]))
    return metrics(np.concatenate(yt), np.concatenate(yp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=C.AU_PARQUET)
    ap.add_argument("--region", default="au", choices=["au", "global"], help="仅用于结果命名")
    ap.add_argument("--protocol", default="logo", choices=["logo", "random", "both"])
    ap.add_argument("--feature-set", default="met", choices=list(C.FEATURE_SETS),
                    help="base / met / s2 / met+s2(s2 需用 satellite_dea.py 产出的 parquet)")
    ap.add_argument("--tag", default=None, help="结果文件名后缀(默认=feature_set)")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--min-samples", type=int, default=30, help="某植被型样本少于此则跳过")
    args = ap.parse_args()
    fs = args.feature_set
    tag = args.tag or fs.replace("+", "_")

    if not os.path.exists(args.parquet):
        sys.exit(f"找不到 {args.parquet},先跑:python src/prepare_lfmc.py")
    df = pd.read_parquet(args.parquet)
    missing_fs = [f for f in C.FEATURE_SETS[fs] if f not in df.columns]
    if missing_fs:
        print(f"⚠️ feature-set '{fs}' 缺列 {missing_fs}(可能用错 parquet),将只用现有列")
    print(f"载入 {args.parquet}:{len(df):,} 行,{df['site'].nunique():,} 站点  "
          f"feature-set={fs}\n")

    protocols = ["logo", "random"] if args.protocol == "both" else [args.protocol]
    results = {"region": args.region, "parquet": args.parquet, "feature_set": fs,
               "by_veg": {}}

    for veg in C.VEG_TYPES + ["all"]:
        sub = df if veg == "all" else df[df["veg_type"] == veg]
        if len(sub) < args.min_samples:
            print(f"[{veg}] 样本 {len(sub)} < {args.min_samples},跳过")
            continue
        X, feats = feature_matrix(sub, fs)
        y = sub["lfmc"].to_numpy(dtype=float)
        groups = sub["site"].to_numpy()
        results["by_veg"][veg] = {"n_sites": int(len(np.unique(groups))), "features": feats}

        print(f"[{veg}] n={len(sub):,}  站点={len(np.unique(groups))}  特征={feats}")
        for proto in protocols:
            m = eval_random(X, y, args.seed) if proto == "random" else \
                eval_logo(X, y, groups, args.seed)
            if m is None:
                print(f"    {proto:7s}: 站点数不足,跳过"); continue
            results["by_veg"][veg][proto] = m
            print(f"    {proto:7s}: R²={m['r2']:+.3f}  RMSE={m['rmse']:5.1f}%  "
                  f"MAE={m['mae']:4.1f}%  (n_test 合计={m['n']:,})")
        print()

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = os.path.join(C.RESULTS_DIR, f"baseline_rf_{args.region}_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果写出:{out}")

    # 对标提示
    fr = results["by_veg"].get("forest", {}).get("logo", {})
    if fr:
        print(f"\n→ 森林 LOGO R²={fr['r2']:.3f}(对标 Yebra 地面验证 0.42–0.53;"
              f"本版无卫星特征=地板,W2 接 DEA 后应抬升)")


if __name__ == "__main__":
    main()
