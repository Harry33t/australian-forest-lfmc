"""导出 React 网页 demo 需要的 JSON(web/public/data/)。

产出:
  sites.json     逐站点:lon/lat/veg/bioregion/lfmc 实测/预测/不确定性(RF 树间方差)/n
                 → 3D 地图柱:颜色=LFMC 估值,高度=不确定性(SPEC 签名"估值/不确定性并排")
  loro.json      区域迁移矩阵(复制 results)
  conformal.json 校准曲线 + 区间(复制 results)
  summary.json   头条数字 + 分植被型分布(给叙事/图表)

用法:python src/export_web.py
"""
from __future__ import annotations
import json, os, sys, shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, os.path.dirname(__file__))
import common as C

WEB_DATA = "web/public/data"


def main():
    os.makedirs(WEB_DATA, exist_ok=True)
    df = pd.read_parquet(C.AU_PARQUET)
    feats = [f for f in C.FEATURE_SETS["met"] if f in df.columns]
    X = df[feats].copy()
    for c in feats:
        X[c] = X[c].fillna(X[c].median())
    X = X.to_numpy(float); y = df["lfmc"].to_numpy(float)
    groups = df["site"].to_numpy()

    rf = RandomForestRegressor(n_estimators=300, max_depth=24, min_samples_leaf=4,
                               max_features="sqrt", n_jobs=-1, random_state=0)
    # 诚实的逐点预测:站点级 out-of-fold
    pred = cross_val_predict(rf, X, y, groups=groups,
                             cv=GroupKFold(n_splits=5), n_jobs=-1)
    # 不确定性:全量拟合后树间预测标准差(epistemic,逐点变化 → 3D 柱高有信息)
    rf.fit(X, y)
    tree_preds = np.stack([t.predict(X) for t in rf.estimators_])
    unc = tree_preds.std(0)
    df = df.assign(pred=pred, unc=unc)

    # 逐站点聚合
    agg = df.groupby("site").agg(
        lon=("lon", "mean"), lat=("lat", "mean"),
        veg=("veg_type", "first"), bioregion=("bioregion", "first"),
        lfmc=("lfmc", "mean"), pred=("pred", "mean"), unc=("unc", "mean"),
        n=("lfmc", "size")).reset_index()
    sites = [{"site": r.site, "lon": round(r.lon, 4), "lat": round(r.lat, 4),
              "veg": r.veg, "bioregion": r.bioregion,
              "lfmc": round(r.lfmc, 1), "pred": round(r.pred, 1),
              "unc": round(r.unc, 1), "n": int(r.n)} for r in agg.itertuples()]
    json.dump(sites, open(f"{WEB_DATA}/sites.json", "w"), ensure_ascii=False)
    print(f"sites.json: {len(sites)} 站点")

    # 复制结果 JSON
    for name in ("loro.json", "conformal.json"):
        src = os.path.join(C.RESULTS_DIR, name)
        if os.path.exists(src):
            shutil.copy(src, f"{WEB_DATA}/{name}"); print(f"copied {name}")

    # summary:头条 + 分布
    veg_stats = {}
    for v in C.VEG_TYPES:
        s = df[df["veg_type"] == v]["lfmc"]
        if len(s):
            veg_stats[v] = {"n": int(len(s)), "sites": int(df[df["veg_type"]==v]["site"].nunique()),
                            "median": round(float(s.median()), 1),
                            "q1": round(float(s.quantile(.25)), 1),
                            "q3": round(float(s.quantile(.75)), 1),
                            "p05": round(float(s.quantile(.05)), 1),
                            "p95": round(float(s.quantile(.95)), 1)}
    summary = {
        "n_measurements": int(len(df)), "n_sites": int(df["site"].nunique()),
        "n_bioregions": int(df["bioregion"].nunique()),
        "year_min": int(df["date"].dt.year.min()), "year_max": int(df["date"].dt.year.max()),
        "veg_stats": veg_stats,
        "headline": {
            "forest_logo_met": 0.11, "forest_random_met": 0.45,
            "yebra_forest": 0.43, "yebra_ground_low": 0.42, "yebra_ground_high": 0.53,
        },
    }
    json.dump(summary, open(f"{WEB_DATA}/summary.json", "w"), ensure_ascii=False, indent=2)
    print(f"summary.json: {summary['n_measurements']} 测量 / {summary['n_sites']} 站点 / {summary['n_bioregions']} bioregion")
    print("完成。")


if __name__ == "__main__":
    main()
