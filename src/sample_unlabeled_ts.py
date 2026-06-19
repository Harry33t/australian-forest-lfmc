"""采**无标注**森林时序语料(瓦片批处理版,供 SSL 大语料预训练)。

为什么瓦片批处理:单点逐个建 cube 太慢(~78s/点)。这里撒少量瓦片中心,每瓦片只建一次
cube,在瓦片内随机采多个小窗口(窗口内有效像元均值,与标注集一致)→ 把昂贵的 STAC 搜索/
建 cube 摊销 ~每瓦片 n_per_tile 倍。

SSL 卖点 = 用海量未标注数据预训练 → 少量标签微调即达同等精度(打 Nolan "外业贵")。

用法:
    python src/sample_unlabeled_ts.py --tiles 130 --per-tile 25 --workers 8   # ~3000 序列
产出:outputs/ts/au_s2_unlabeled.npz(X, mask + 占位 y/veg/site/row_id)
"""
from __future__ import annotations
import argparse, os, sys, json, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C
import satellite_dea_ts as TS

warnings.filterwarnings("ignore")
OUT_NPZ = "outputs/ts/au_s2_unlabeled.npz"
SHP = "data/raw/ibra7/ibra7_regions.shp"


def sample_tiles(n, seed):
    """在含森林 LFMC 站点的 bioregion 内随机撒 n 个瓦片中心。返回 (lon,lat) 数组。"""
    import geopandas as gpd
    from shapely.geometry import Point
    au = pd.read_parquet(C.AU_PARQUET)
    fr = sorted(au[au["veg_type"] == "forest"]["bioregion"].dropna().unique())
    regions = gpd.read_file(SHP).to_crs("EPSG:4326")
    poly = regions[regions["REG_NAME_7"].isin(fr)].union_all()
    minx, miny, maxx, maxy = poly.bounds
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
    return np.array(pts), fr


def extract_tile(catalog, odc_stac, lat, lon, ref, per_tile, half_deg, win, tol, seed):
    """一个瓦片:建一次 cube,采 per_tile 个小窗口序列。返回 [{X,mask}, ...]。"""
    t0 = (ref - pd.Timedelta(days=TS.MONTH * TS.LOOKBACK_MONTHS + tol)).strftime("%Y-%m-%d")
    t1 = (ref + pd.Timedelta(days=tol)).strftime("%Y-%m-%d")
    bbox = [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]
    items = list(catalog.search(collections=TS.S2_COLLECTIONS, bbox=bbox, datetime=f"{t0}/{t1}").items())
    if not items:
        return []
    ds = odc_stac.load(items, bands=list(TS.BANDS) + [TS.FMASK], bbox=bbox,
                       resolution=20, groupby="solar_day", chunks={})
    if ds.sizes.get("time", 0) == 0:
        return []
    ctimes = pd.to_datetime(ds["time"].values)

    # 每个月度槽的候选时次(按 |dt| 排序),并集去重 → 只读这些时次
    slot_cands, need = [], set()
    for j in range(TS.T):
        center = ref - pd.Timedelta(days=TS.MONTH * TS.SLOT_SPACING * (TS.T - 1 - j))
        dd = np.abs((ctimes - center).days)
        cand = np.where(dd <= tol)[0]
        cand = cand[np.argsort(dd[cand])]
        slot_cands.append(list(map(int, cand)))
        need.update(int(c) for c in cand)
    if not need:
        return []
    need = sorted(need)
    pos = {ci: k for k, ci in enumerate(need)}
    valid = (ds[TS.FMASK].isel(time=need).values == 1)             # (nt,ny,nx)
    bands = {s: ds[raw].isel(time=need).values.astype(np.float32) for raw, s in TS.BANDS.items()}
    nt, ny, nx = valid.shape
    if ny < 2 * win + 1 or nx < 2 * win + 1:
        return []

    # 每槽在**瓦片级**选一个时次:候选里有效像元最多者(消掉逐窗口逐候选的 Python 循环)
    slot_time = []
    for j in range(TS.T):
        best, best_cov = -1, 0
        for ci in slot_cands[j]:
            cov = valid[pos[ci]].sum()
            if cov > best_cov:
                best_cov, best = cov, pos[ci]
        slot_time.append(best)                # -1 = 该槽全瓦片无有效观测

    rng = np.random.default_rng(seed)
    cys = rng.integers(win, ny - win, per_tile)
    cxs = rng.integers(win, nx - win, per_tile)
    out = [{"X": np.zeros((TS.T, TS.F), np.float32), "m": np.zeros(TS.T, bool)}
           for _ in range(per_tile)]
    for j, k in enumerate(slot_time):
        if k < 0:
            continue
        vk = valid[k]                          # (ny,nx)
        bk = {s: bands[s][k] for s in TS.BANDS.values()}
        for w, (cy, cx) in enumerate(zip(cys, cxs)):
            ys, xs = slice(cy - win, cy + win + 1), slice(cx - win, cx + win + 1)
            wv = vk[ys, xs]
            if not wv.any():
                continue
            refl, ok = {}, True
            for s in TS.BANDS.values():
                vals = bk[s][ys, xs][wv]
                vals = vals[vals > 0]
                if vals.size == 0:
                    ok = False; break
                refl[s] = float(vals.mean()) / 10000.0
            if not ok:
                continue
            idx = TS.indices_from_refl(refl)
            out[w]["X"][j] = [idx[f] for f in TS.FEATURES]; out[w]["m"][j] = True
    return [{"X": o["X"], "mask": o["m"]} for o in out if o["m"].any()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=130)
    ap.add_argument("--per-tile", type=int, default=25)
    ap.add_argument("--half-deg", type=float, default=0.02, help="瓦片半边长(度,~2km)")
    ap.add_argument("--win", type=int, default=3, help="子窗口半径(像元),窗口均值")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pystac_client
    import odc.stac as odc_stac
    catalog = pystac_client.Client.open(TS.DEA_STAC)

    centers, fr = sample_tiles(args.tiles, args.seed)
    au = pd.read_parquet(C.AU_PARQUET)
    fdates = pd.to_datetime(au[(au["veg_type"] == "forest") & (au["date"] >= TS.S2_START)]["date"])
    rng = np.random.default_rng(args.seed + 1)
    refs = pd.to_datetime(rng.choice(fdates.values, len(centers)))
    print(f"{len(fr)} 个森林 bioregion 内撒 {len(centers)} 瓦片 × {args.per_tile} 窗口 = 目标 ~{len(centers)*args.per_tile}")

    def work(i):
        return extract_tile(catalog, odc_stac, centers[i][1], centers[i][0], refs[i],
                            args.per_tile, args.half_deg, args.win, TS.TOL_DAYS, args.seed + i)

    results, errs = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in range(len(centers))]
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                results.extend(fut.result())
            except Exception as e:
                errs += 1
                if errs <= 5:
                    print(f"  tile err: {type(e).__name__}: {str(e)[:70]}")
            if k % 20 == 0:
                print(f"  完成瓦片 {k}/{len(centers)}  累计序列 {len(results)}")

    if not results:
        sys.exit("无结果(检查网络/STAC)。")
    X = np.stack([r["X"] for r in results]); M = np.stack([r["mask"] for r in results])
    n = len(X)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {"features": TS.FEATURES, "T": TS.T, "lookback_months": TS.LOOKBACK_MONTHS, "unlabeled": True}
    np.savez_compressed(args.out, X=X, mask=M, y=np.zeros(n, np.float32),
                        row_id=np.arange(n), veg_type=np.array(["unlabeled"] * n),
                        site=np.array([f"U{i}" for i in range(n)]), meta=json.dumps(meta))
    print(f"\n无标注语料:{n} 条序列(errs={errs}),覆盖率 {M.mean():.0%}")
    print(f"写出:{args.out}")


if __name__ == "__main__":
    main()
