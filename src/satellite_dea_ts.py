"""为澳洲 LFMC 站点提**时序** Sentinel-2 特征(DEA,免认证),供时序 SSL 用。

与单日版 satellite_dea.py 的区别:每条测量提取**过去 12 个月**的月度序列 (T=12, F=8),
而不是单一最近无云日。这是时序自监督预训练(masked time-series / 对比)的输入。

每个月度槽位:在该月中心 ±tol 天内挑**最近的 fmask=valid** S2 像元(够轻:每条测量 ≤T 次读取,
站内并集去重);无有效观测 → mask=0(SSL 正好用掩码建模缺测)。
特征 F=8:6 个 LFMC 指数(ndvi/ndii/ndwi/gvmi/nmdi/vari)+ nir + swir1。

⚡ 按站点 ThreadPool 并发(瓶颈是网络延迟,并发把等待重叠 → 快数倍)。
⚠️ 仅 2015-07 起有 S2;断点续跑靠 row_id 缓存。

用法:
    python src/satellite_dea_ts.py --veg forest --limit-sites 3   # 验证通路
    python src/satellite_dea_ts.py --workers 8                    # 全澳(并发)

产出:outputs/ts/au_s2_ts.npz —— X(N,T,F) float32, mask(N,T) bool, y(N,),
       row_id/site/veg_type 对齐数组 + meta(特征名, T, lookback)。
"""
from __future__ import annotations
import argparse, os, sys, json, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
warnings.filterwarnings("ignore")

DEA_STAC = "https://explorer.dea.ga.gov.au/stac"
S2_COLLECTIONS = ["ga_s2am_ard_3", "ga_s2bm_ard_3"]
BANDS = {"nbart_blue": "blue", "nbart_green": "green", "nbart_red": "red",
         "nbart_nir_1": "nir", "nbart_nir_2": "nir_narrow",
         "nbart_swir_2": "swir1", "nbart_swir_3": "swir2"}
FMASK = "oa_fmask"
OUT_NPZ = "outputs/ts/au_s2_ts.npz"
S2_START = pd.Timestamp("2015-07-01")

LOOKBACK_MONTHS = 12       # 回看窗口长度
T = 12                     # 月度时间槽(细槽抓到的不同观测最多;稀疏由 SSL 掩码建模处理)
MONTH = 30.44              # 平均天数
SLOT_SPACING = LOOKBACK_MONTHS / T   # 槽间隔(月)= 1
TOL_DAYS = 25              # 槽位中心 ±tol 找最近有效观测
FEATURES = ["ndvi", "ndii", "ndwi", "gvmi", "nmdi", "vari", "nir", "swir1"]
F = len(FEATURES)


def indices_from_refl(r):
    eps = 1e-6
    def nd(a, b): return (a - b) / (a + b + eps)
    return {
        "ndvi": nd(r["nir"], r["red"]), "ndii": nd(r["nir"], r["swir1"]),
        "ndwi": nd(r["nir"], r["swir2"]),
        "gvmi": ((r["nir_narrow"]+0.1)-(r["swir2"]+0.02))/((r["nir_narrow"]+0.1)+(r["swir2"]+0.02)+eps),
        "nmdi": (r["nir_narrow"]-(r["swir1"]-r["swir2"]))/(r["nir_narrow"]+(r["swir1"]-r["swir2"])+eps),
        "vari": (r["green"]-r["red"])/(r["green"]+r["red"]-r["blue"]+eps),
        "nir": r["nir"], "swir1": r["swir1"],
    }


def extract_site_ts(catalog, odc_stac, site_df, tol_days, half_box_deg):
    """一个站点所有测量的时序序列。返回 [{row_id, X(T,F), mask(T,)}, ...]。"""
    lat = float(site_df["lat"].iloc[0]); lon = float(site_df["lon"].iloc[0])
    dates = pd.to_datetime(site_df["date"])
    in_era = dates >= S2_START
    if not in_era.any():
        return []
    # 立方覆盖:最早测量回看 12 个月 ~ 最晚测量
    t0 = (dates[in_era].min() - pd.Timedelta(days=MONTH*LOOKBACK_MONTHS + tol_days)).strftime("%Y-%m-%d")
    t1 = (dates[in_era].max() + pd.Timedelta(days=tol_days)).strftime("%Y-%m-%d")
    bbox = [lon-half_box_deg, lat-half_box_deg, lon+half_box_deg, lat+half_box_deg]

    items = list(catalog.search(collections=S2_COLLECTIONS, bbox=bbox, datetime=f"{t0}/{t1}").items())
    if not items:
        return []
    ds = odc_stac.load(items, bands=list(BANDS)+[FMASK], bbox=bbox,
                       resolution=20, groupby="solar_day", chunks={})
    if ds.sizes.get("time", 0) == 0:
        return []
    ctimes = pd.to_datetime(ds["time"].values)   # 保留 x/y 维:窗口内对有效像元取均值(更稳、覆盖更高)

    # 1) 每条测量 × T 槽位:找候选时次索引;并集去重 → 只读这些时次
    slot_pick = {}     # row_id -> [时次索引 or -1] 长度 T
    need = set()
    for rid, d in zip(site_df["row_id"], dates):
        if d < S2_START:
            continue
        picks = []
        for j in range(T):
            center = d - pd.Timedelta(days=MONTH * SLOT_SPACING * (T - 1 - j))  # j=0 最旧, j=T-1 ~采样
            dd = np.abs((ctimes - center).days)
            cand = np.where(dd <= tol_days)[0]
            picks.append(int(cand[np.argmin(dd[cand])]) if len(cand) else -1)
        slot_pick[int(rid)] = picks
        need.update(p for p in picks if p >= 0)
    if not need:
        return []
    need = sorted(need)

    # 2) 读候选时次的整窗 fmask + 波段(一次性);窗口内对 fmask=valid 像元取均值
    fm = ds[FMASK].isel(time=need).values                       # (n,ny,nx)
    band_cube = {short: ds[raw].isel(time=need).values for raw, short in BANDS.items()}
    pos = {ci: k for k, ci in enumerate(need)}
    valid_px = fm == 1                                          # (n,ny,nx)
    n_valid = valid_px.reshape(len(need), -1).sum(1)            # 每时次有效像元数

    def slot_refl(ci):
        """窗口内有效像元的反射率均值;无有效像元返回 None。"""
        k = pos[ci]
        if n_valid[k] == 0:
            return None
        vp = valid_px[k]
        refl = {}
        for short in BANDS.values():
            band = band_cube[short][k].astype(np.float64)
            vals = band[vp]
            vals = vals[(vals > 0) & np.isfinite(vals)]
            if vals.size == 0:
                return None
            refl[short] = vals.mean() / 10000.0
        return refl

    out = []
    for rid, picks in slot_pick.items():
        X = np.zeros((T, F), dtype=np.float32)
        mask = np.zeros(T, dtype=bool)
        for j, ci in enumerate(picks):
            if ci < 0:
                continue
            refl = slot_refl(ci)
            if refl is None:
                continue
            idx = indices_from_refl(refl)
            X[j] = [idx[f] for f in FEATURES]
            mask[j] = True
        if mask.any():
            out.append({"row_id": rid, "X": X, "mask": mask})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=C.AU_PARQUET)
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--veg", default=None, choices=[None] + C.VEG_TYPES)
    ap.add_argument("--limit-sites", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8, help="并发线程数(网络延迟瓶颈,8 通常够)")
    ap.add_argument("--tol-days", type=int, default=TOL_DAYS)
    ap.add_argument("--half-box-deg", type=float, default=0.0015)
    args = ap.parse_args()

    import pystac_client
    import odc.stac as odc_stac
    catalog = pystac_client.Client.open(DEA_STAC)

    df = pd.read_parquet(args.parquet).reset_index().rename(columns={"index": "row_id"})
    if args.veg:
        df = df[df["veg_type"] == args.veg]

    done = set()
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        done = set(int(r) for r in z["row_id"])
        print(f"  已缓存 {len(done)} 条,续跑")

    sites = [g for _, g in df.groupby("site")]
    if args.limit_sites:
        sites = sites[:args.limit_sites]
    sites = [g[~g["row_id"].isin(done)] for g in sites]
    sites = [g for g in sites if len(g)]
    n_meas = sum(len(g) for g in sites)
    print(f"待处理 {len(sites):,} 站点 / {n_meas:,} 测量,{args.workers} 线程并发")

    results = []
    def work(g):
        try:
            return extract_site_ts(catalog, odc_stac, g, args.tol_days, args.half_box_deg)
        except Exception as e:
            return [("ERR", f"{type(e).__name__}: {str(e)[:80]}")]

    errs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, g): i for i, g in enumerate(sites)}
        for n, fut in enumerate(as_completed(futs), 1):
            recs = fut.result()
            for r in recs:
                if isinstance(r, tuple):
                    errs += 1
                    if errs <= 5: print(f"  异常:{r[1]}")
                else:
                    results.append(r)
            if n % 20 == 0:
                print(f"  完成站点 {n}/{len(sites)}  累计序列 {len(results)} 条")

    if not results and not done:
        sys.exit("没有任何成功提取(检查网络 / STAC 可达)。")

    # 合并缓存 + 本轮,按 row_id 去重,join 标签
    rid_new = np.array([r["row_id"] for r in results], dtype=np.int64)
    X_new = np.stack([r["X"] for r in results]) if results else np.zeros((0, T, F), np.float32)
    M_new = np.stack([r["mask"] for r in results]) if results else np.zeros((0, T), bool)
    if done and os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        rid_all = np.concatenate([z["row_id"], rid_new])
        X_all = np.concatenate([z["X"], X_new]); M_all = np.concatenate([z["mask"], M_new])
    else:
        rid_all, X_all, M_all = rid_new, X_new, M_new
    _, uniq = np.unique(rid_all, return_index=True)
    rid_all, X_all, M_all = rid_all[uniq], X_all[uniq], M_all[uniq]

    base = pd.read_parquet(args.parquet).reset_index().rename(columns={"index": "row_id"})
    bi = base.set_index("row_id")
    y = bi.loc[rid_all, "lfmc"].to_numpy(np.float32)
    veg = bi.loc[rid_all, "veg_type"].to_numpy()
    site = bi.loc[rid_all, "site"].to_numpy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {"features": FEATURES, "T": T, "lookback_months": LOOKBACK_MONTHS, "tol_days": args.tol_days}
    np.savez_compressed(args.out, X=X_all, mask=M_all, y=y, row_id=rid_all,
                        veg_type=veg, site=site, meta=json.dumps(meta))
    cov = M_all.mean()
    print(f"\n命中 {len(rid_all):,} 条序列;月度槽位平均覆盖率 {cov:.0%}")
    for v in C.VEG_TYPES:
        m = veg == v
        if m.any():
            print(f"  {v:10s} {m.sum():5d} 条  覆盖率 {M_all[m].mean():.0%}")
    print(f"写出:{args.out}  (X {X_all.shape}, mask {M_all.shape})")
    print("下一步:rsync 到 AutoDL → python src/ssl_pretrain.py")


if __name__ == "__main__":
    main()
