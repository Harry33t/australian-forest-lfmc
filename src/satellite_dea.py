"""为澳洲 LFMC 站点配 Sentinel-2 时序特征(Digital Earth Australia,免认证公共桶)。

为什么 DEA:Yebra RS 2026 本人就用 DEA 的 S2 NBART;澳洲免费、公共桶无需认证,
本地连 AWS 悉尼比国内机房稳(见 README 算力分工)。

⚡ 按**站点**批处理:一个站点常有多条测量(森林 ~4.7 条/站)。对每个站点只做一次
STAC 搜索 + 构一个惰性影像立方,再把该站点所有测量各自匹配到最近无云时次 ——
读取量只跟"实际命中的时次"成正比,比逐测量重搜快一个量级。

匹配:对每条测量,在 ±window 天内挑离采样日期最近、fmask=valid 的 S2 像元,
算反射率 + LFMC 光谱指数(NDII 最强,见 common.py)。

⚠️ Sentinel-2 仅 2015-07 起:更早的测量无 S2 → 自然落空(诚实限制)。
⚠️ 网络重活:默认先 --limit-sites 验证通路,通了再放量;按 row_id 缓存可断点续跑。

用法:
    python src/satellite_dea.py --veg forest --limit-sites 5      # 先验证通路
    python src/satellite_dea.py --veg forest                      # 森林全量(夜跑)
    python src/satellite_dea.py                                   # 全澳

产出:outputs/lfmc_au_s2.parquet(原行 + s2_* 反射率 + 指数列 + s2_date/s2_dt_days)
"""
from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C

# DEA 公共数据在 AWS ap-southeast-2 公共桶,免签名读取
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
warnings.filterwarnings("ignore")

DEA_STAC = "https://explorer.dea.ga.gov.au/stac"
S2_COLLECTIONS = ["ga_s2am_ard_3", "ga_s2bm_ard_3"]   # Sentinel-2A / 2B NBART ARD

# DEA NBART 波段名 → 我们的短名(S2 波段)
BANDS = {
    "nbart_blue": "blue",        # B2
    "nbart_green": "green",      # B3
    "nbart_red": "red",          # B4
    "nbart_nir_1": "nir",        # B8
    "nbart_nir_2": "nir_narrow", # B8A
    "nbart_swir_2": "swir1",     # B11
    "nbart_swir_3": "swir2",     # B12
}
FMASK = "oa_fmask"               # 1=valid 2=cloud 3=shadow 4=snow 5=water
OUT_PARQUET = "outputs/lfmc_au_s2.parquet"
S2_START = pd.Timestamp("2015-07-01")   # Sentinel-2 数据起点


def compute_indices(r):
    """r: dict of 反射率(0–1)。返回 LFMC 光谱指数 dict。"""
    eps = 1e-6
    def nd(a, b):
        return (a - b) / (a + b + eps)
    return {
        "ndvi": nd(r["nir"], r["red"]),
        "ndii": nd(r["nir"], r["swir1"]),                 # 最强 LFMC 预测因子
        "ndwi": nd(r["nir"], r["swir2"]),
        "gvmi": ((r["nir_narrow"] + 0.1) - (r["swir2"] + 0.02)) /
                ((r["nir_narrow"] + 0.1) + (r["swir2"] + 0.02) + eps),
        "nmdi": (r["nir_narrow"] - (r["swir1"] - r["swir2"])) /
                (r["nir_narrow"] + (r["swir1"] - r["swir2"]) + eps),
        "vari": (r["green"] - r["red"]) / (r["green"] + r["red"] - r["blue"] + eps),
    }


def extract_site(catalog, odc_stac, site_df, window_days, half_box_deg):
    """处理一个站点的所有测量。返回 [{row_id, s2_*, 指数, s2_date, s2_dt_days}, ...]。

    一次 STAC 搜索 + 一个惰性立方;只对命中的时次触发像元读取。
    """
    lat = float(site_df["lat"].iloc[0]); lon = float(site_df["lon"].iloc[0])
    dates = pd.to_datetime(site_df["date"])
    # 只处理 S2 时代内的测量
    in_era = dates >= S2_START
    if not in_era.any():
        return []

    t0 = (dates[in_era].min() - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    t1 = (dates[in_era].max() + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    bbox = [lon - half_box_deg, lat - half_box_deg, lon + half_box_deg, lat + half_box_deg]

    items = list(catalog.search(collections=S2_COLLECTIONS, bbox=bbox,
                                datetime=f"{t0}/{t1}").items())
    if not items:
        return []
    ds = odc_stac.load(items, bands=list(BANDS) + [FMASK], bbox=bbox,
                       resolution=20, groupby="solar_day", chunks={})
    if ds.sizes.get("time", 0) == 0:
        return []
    ds = ds.sel(x=lon, y=lat, method="nearest")          # 站点中心像元(仍惰性)
    cube_times = pd.to_datetime(ds["time"].values)

    # 1) 汇总每条测量的候选时次索引(±window),取并集 → 只读这些时次的 fmask
    cand_by_meas = {}
    cand_idx = set()
    for rid, d in zip(site_df["row_id"], dates):
        if d < S2_START:
            continue
        dt = (cube_times - d).days
        idx = np.where(np.abs(dt) <= window_days)[0]
        if len(idx):
            cand_by_meas[int(rid)] = (d, idx)
            cand_idx.update(int(i) for i in idx)
    if not cand_idx:
        return []
    cand_idx = sorted(cand_idx)
    fmask_vals = ds[FMASK].isel(time=cand_idx).values    # 触发读取(仅候选时次)
    valid_at = {ci: (fmask_vals[j] == 1) for j, ci in enumerate(cand_idx)}

    # 2) 为每条测量挑最近的 valid 时次
    pick = {}                # row_id -> 时次索引
    need_band_idx = set()
    for rid, (d, idx) in cand_by_meas.items():
        best, best_dt = None, None
        for ci in idx:
            if not valid_at.get(int(ci), False):
                continue
            dd = abs(int((cube_times[ci] - d).days))
            if best is None or dd < best_dt:
                best, best_dt = int(ci), dd
        if best is not None:
            pick[rid] = best
            need_band_idx.add(best)
    if not pick:
        return []

    # 3) 只读命中时次的 7 个波段(一次性)
    need_band_idx = sorted(need_band_idx)
    band_arr = {short: ds[raw].isel(time=need_band_idx).values
                for raw, short in BANDS.items()}
    pos = {ci: j for j, ci in enumerate(need_band_idx)}

    out = []
    for rid, ci in pick.items():
        j = pos[ci]
        refl = {}
        bad = False
        for short in BANDS.values():
            v = float(band_arr[short][j])
            if not np.isfinite(v) or v <= 0:
                bad = True; break
            refl[short] = v / 10000.0          # DEA NBART 0–10000 → 0–1
        if bad:
            continue
        d = cand_by_meas[rid][0]
        rec = {"row_id": int(rid)}
        rec.update({f"s2_{k}": v for k, v in refl.items()})
        rec.update(compute_indices(refl))
        rec["s2_date"] = pd.Timestamp(cube_times[ci]).strftime("%Y-%m-%d")
        rec["s2_dt_days"] = int((cube_times[ci] - d).days)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=C.AU_PARQUET)
    ap.add_argument("--out", default=OUT_PARQUET)
    ap.add_argument("--veg", default=None, choices=[None] + C.VEG_TYPES)
    ap.add_argument("--limit-sites", type=int, default=None, help="只处理前 N 个站点(验证通路)")
    ap.add_argument("--window-days", type=int, default=8, help="匹配窗口 ±天(Yebra:±3–10 天影响小)")
    ap.add_argument("--half-box-deg", type=float, default=0.0015, help="站点 bbox 半边长(度,~150m)")
    ap.add_argument("--checkpoint-every", type=int, default=20, help="每 N 站点存一次 parquet(抗崩溃/可中途出图)")
    args = ap.parse_args()

    import pystac_client
    import odc.stac as odc_stac
    catalog = pystac_client.Client.open(DEA_STAC)

    # row_id 用 AU parquet 的原始整数索引(稳定):跨 veg 子集 / 续跑都不会撞号
    df = pd.read_parquet(args.parquet).reset_index().rename(columns={"index": "row_id"})
    if args.veg:
        df = df[df["veg_type"] == args.veg]

    # 断点续跑:已成功的 row_id 跳过
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        done = set(int(r) for r in prev["row_id"]) if "row_id" in prev.columns else set()
        print(f"  已缓存 {len(done)} 条,续跑")

    sites = list(df.groupby("site"))
    if args.limit_sites:
        sites = sites[:args.limit_sites]
    n_meas = sum(len(g) for _, g in sites)
    print(f"待处理 {len(sites):,} 站点 / {n_meas:,} 测量"
          f"(veg={args.veg or 'all'},window=±{args.window_days}d)")

    base = pd.read_parquet(args.parquet).reset_index().rename(columns={"index": "row_id"})

    def save(rows):
        """把已有缓存 + 本轮 rows 的特征 join 回原行,写 parquet。返回命中行数。"""
        new = pd.DataFrame(rows)
        if os.path.exists(args.out) and done:
            prev = pd.read_parquet(args.out)
            prev_feat = prev[["row_id"] + [c for c in prev.columns
                                           if c.startswith("s2_") or c in C.S2_INDICES]]
            new = pd.concat([prev_feat, new], ignore_index=True) if len(new) else prev_feat
        if not len(new):
            return 0
        merged = base.merge(new.drop_duplicates("row_id"), on="row_id", how="inner")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        merged.to_parquet(args.out, index=False)
        return len(merged)

    rows, ok = [], 0
    for si, (site, g) in enumerate(sites):
        g = g[~g["row_id"].isin(done)]
        if not len(g):
            continue
        try:
            recs = extract_site(catalog, odc_stac, g, args.window_days, args.half_box_deg)
        except Exception as e:
            recs = []
            if si < 5:
                print(f"  [站点 {site}] 异常:{type(e).__name__}: {str(e)[:90]}")
        rows.extend(recs); ok += len(recs)
        if (si + 1) % args.checkpoint_every == 0:
            total = save(rows)
            print(f"  站点 {si+1}/{len(sites)}  本轮命中 {ok} 条  已存(累计 {total} 条)")

    if not rows and not done:
        sys.exit("没有任何成功提取(检查网络到 AWS 悉尼 / STAC 可达性)。")

    total = save(rows)
    print(f"\n命中 {total:,} 条(本轮新增 {len(rows):,})")
    merged = pd.read_parquet(args.out)
    if len(merged):
        print(f"  时间匹配 |dt| 中位数 {merged['s2_dt_days'].abs().median():.0f} 天;"
              f"NDII 范围 {merged['ndii'].min():.2f}–{merged['ndii'].max():.2f}")
    print(f"写出:{args.out}")
    print("下一步:在该 parquet 上跑 met / met+s2 两版 baseline,再 viz/plot_improvement.py")


if __name__ == "__main__":
    main()
