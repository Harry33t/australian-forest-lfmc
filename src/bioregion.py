"""把 LFMC 站点经纬度空间 join 到 IBRA7 bioregion(供 leave-one-region-out 用)。

IBRA7 = Interim Biogeographic Regionalisation for Australia v7,89 个 bioregion。
shapefile:DCCEEW 官方(EPSG:4283 GDA94),字段 REG_NAME_7 / REG_CODE_7。

给 parquet 增列 bioregion / bioregion_code;点落在多个/无多边形时按最近或标 None。

用法:
    python src/bioregion.py                       # 处理 au 与 au_s2 两个 parquet
    python src/bioregion.py --parquet outputs/lfmc_au_s2.parquet
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C

SHP = "data/raw/ibra7/ibra7_regions.shp"
NAME_FIELD, CODE_FIELD = "REG_NAME_7", "REG_CODE_7"


def join_one(parquet, gdf_regions):
    import geopandas as gpd
    df = pd.read_parquet(parquet)
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    ).to_crs(gdf_regions.crs)
    joined = gpd.sjoin(pts, gdf_regions[[NAME_FIELD, CODE_FIELD, "geometry"]],
                       how="left", predicate="within")
    # 落在边界外的少数点 → 最近 bioregion 兜底
    miss = joined[NAME_FIELD].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss.values], gdf_regions[[NAME_FIELD, CODE_FIELD, "geometry"]],
                                 how="left")
        joined.loc[miss.values, NAME_FIELD] = near[NAME_FIELD].values
        joined.loc[miss.values, CODE_FIELD] = near[CODE_FIELD].values
    df["bioregion"] = joined[NAME_FIELD].values
    df["bioregion_code"] = joined[CODE_FIELD].values
    df.to_parquet(parquet, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="*", default=[C.AU_PARQUET, "outputs/lfmc_au_s2.parquet"])
    ap.add_argument("--shp", default=SHP)
    args = ap.parse_args()
    import geopandas as gpd
    if not os.path.exists(args.shp):
        sys.exit(f"找不到 {args.shp}(先下 IBRA7,见 bioregion.py docstring)")
    regions = gpd.read_file(args.shp)
    print(f"IBRA7:{len(regions)} 个 bioregion,CRS={regions.crs}")

    for pq in args.parquet:
        if not os.path.exists(pq):
            print(f"  跳过(不存在){pq}"); continue
        df = join_one(pq, regions)
        n_reg = df["bioregion"].nunique()
        print(f"\n{pq}:{len(df):,} 行 → {n_reg} 个 bioregion")
        top = df.groupby("bioregion").agg(n=("lfmc", "size"), sites=("site", "nunique")).sort_values("n", ascending=False)
        for name, r in top.head(10).iterrows():
            print(f"  {str(name):32s} n={int(r['n']):5d}  站点={int(r['sites'])}")
    print("\n完成。bioregion / bioregion_code 列已写回 parquet。")


if __name__ == "__main__":
    main()
