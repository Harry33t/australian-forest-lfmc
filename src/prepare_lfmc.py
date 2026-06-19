"""清洗 Globe-LFMC 2.0,派生植被三分类 + 时间特征,导出 parquet + EDA 摘要。

为什么单独一步:原始 .xlsx ~72.5 MB、列名带空格括号、日期是整数 YYYYMMDD、读一次很慢。
这里读一次 → rename 短名 → 解析日期/派生特征 → 落 parquet(全球 + 澳洲两份),
后续 baseline/EDA 直接吃 parquet(秒读)。

产出:
  outputs/lfmc_clean.parquet   全球清洗后(含 veg_type 的有效行)
  outputs/lfmc_au.parquet      澳洲子集(country == Australia)
  outputs/results/eda_summary.txt   计数/分布摘要(W1 EDA)

用法:
    python src/prepare_lfmc.py
    python src/prepare_lfmc.py --raw data/raw --max-lfmc 400
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C


def load_raw(raw_dir):
    xlsx = os.path.join(raw_dir, C.XLSX_NAME)
    if not os.path.exists(xlsx):
        sys.exit(f"找不到 {xlsx},先跑:python src/download_data.py")
    print(f"读 {xlsx}(sheet='{C.SHEET_DATA}',~72 MB,稍等)…")
    df = pd.read_excel(xlsx, sheet_name=C.SHEET_DATA, engine="openpyxl")
    print(f"  原始 {len(df):,} 行 × {df.shape[1]} 列")
    return df


def clean(df, max_lfmc):
    # 只取关心的列(存在才取,容忍版本列名微差)
    keep = [c for c in C.RENAME if c in df.columns]
    missing = [c for c in C.RENAME if c not in df.columns]
    if missing:
        print(f"  ⚠️ 缺列(已跳过):{missing}")
    df = df[keep].rename(columns=C.RENAME).copy()

    # 日期:整数 YYYYMMDD → datetime;无法解析的丢
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # LFMC:数值化,去缺失/非正/离群(干重百分比,极少 >400% 多为录入异常)
    df["lfmc"] = pd.to_numeric(df["lfmc"], errors="coerce")
    n0 = len(df)
    df = df[df["lfmc"].notna() & (df["lfmc"] > 0) & (df["lfmc"] <= max_lfmc)]
    df = df[df["lat"].notna() & df["lon"].notna() & df["date"].notna()]
    print(f"  清洗去无效 LFMC/坐标/日期:{n0:,} → {len(df):,} 行")

    # 植被三分类
    igbp = df["igbp"] if "igbp" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["veg_type"] = [C.map_veg_type(ft, ig) for ft, ig in zip(df["func_type"], igbp)]
    n1 = len(df)
    df = df[df["veg_type"].notna()]
    print(f"  去无法归类植被型:{n1:,} → {len(df):,} 行")

    # 时间特征:年内日序正余弦(季节性,无跳变)
    doy = df["date"].dt.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 数值化站点+气象特征(缺失保留 NaN,baseline 里填补)
    for col in ["elevation", "slope"] + C.MET_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def eda_summary(df, df_au):
    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p("=" * 64)
    p("Globe-LFMC 2.0 — W1 EDA 摘要")
    p("=" * 64)
    p(f"全球有效测量:{len(df):,}  站点:{df['site'].nunique():,}  "
      f"国家:{df['country'].nunique()}")
    p(f"日期范围:{df['date'].min().date()} → {df['date'].max().date()}")
    p("\n[全球] 按植被型计数 / LFMC 中位数 / IQR:")
    for v in C.VEG_TYPES:
        s = df[df["veg_type"] == v]["lfmc"]
        if len(s):
            p(f"  {v:10s} n={len(s):>7,}  median={s.median():6.1f}%  "
              f"IQR=[{s.quantile(.25):.0f}, {s.quantile(.75):.0f}]")

    p("\n" + "-" * 64)
    p(f"[澳洲] 有效测量:{len(df_au):,}  站点:{df_au['site'].nunique():,}")
    if len(df_au):
        p(f"日期范围:{df_au['date'].min().date()} → {df_au['date'].max().date()}")
        p("按植被型:")
        for v in C.VEG_TYPES:
            s = df_au[df_au["veg_type"] == v]
            if len(s):
                p(f"  {v:10s} n={len(s):>6,}  站点={s['site'].nunique():>4}  "
                  f"median LFMC={s['lfmc'].median():6.1f}%")
        p("\n按州/区(top):")
        for st, n in df_au["state"].value_counts().head(8).items():
            p(f"  {str(st):24s} {n:>5,}")
    # 站点特征缺失率(决定 baseline 用哪些)
    p("\n站点特征缺失率(澳洲):")
    for col in C.SITE_FEATURES:
        if col in df_au.columns:
            miss = df_au[col].isna().mean() * 100
            p(f"  {col:12s} {miss:5.1f}% 缺失")
    p("=" * 64)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=C.RAW_DIR)
    ap.add_argument("--max-lfmc", type=float, default=400.0, help="LFMC 上限(%),超过视为录入异常")
    args = ap.parse_args()

    df = clean(load_raw(args.raw), args.max_lfmc)
    df_au = df[df["country"].astype(str).str.strip().str.lower() == "australia"].reset_index(drop=True)

    os.makedirs(os.path.dirname(C.PROCESSED_PARQUET), exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    df.to_parquet(C.PROCESSED_PARQUET, index=False)
    df_au.to_parquet(C.AU_PARQUET, index=False)
    print(f"\n写出:\n  {C.PROCESSED_PARQUET}  ({len(df):,} 行)")
    print(f"  {C.AU_PARQUET}  ({len(df_au):,} 行)")

    txt = eda_summary(df, df_au)
    out = os.path.join(C.RESULTS_DIR, "eda_summary.txt")
    with open(out, "w") as f:
        f.write(txt + "\n")
    print(f"\nEDA 摘要写出:{out}")
    print("下一步:python src/baseline_rf.py")


if __name__ == "__main__":
    main()
