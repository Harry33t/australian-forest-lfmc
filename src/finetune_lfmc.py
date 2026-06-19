"""在时序序列上微调 LFMC 回归头 + leave-site-out(LOGO)评估。在 AutoDL GPU 上跑。

核心对比(demo 签名):**SSL 预训练 init vs 从头 scratch**,在 LOGO 协议下分植被型报 R²;
配合 --label-frac 可画"少标签省标注曲线"(SSL 在少标签时应明显领先)。

用法(AutoDL):
    # 先确保有 ssl_encoder.pt(见 ssl_pretrain.py)
    python src/finetune_lfmc.py --mode ssl     --pretrained outputs/models/ssl_encoder.pt --veg forest
    python src/finetune_lfmc.py --mode scratch --veg forest
    # 少标签曲线:对 frac in 0.1 0.25 0.5 1.0 跑两种 mode(结果累积进同一 JSON)
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
import ts_model as M
import common as C


def train_fold(Xtr, mtr, ytr, Xte, mte, cfg, pretrained, epochs, lr, dev, seed):
    torch.manual_seed(seed)
    enc = M.TSEncoder(cfg["F"], cfg["T"], d_model=cfg["d_model"], nlayers=cfg["nlayers"]).to(dev)
    if pretrained is not None:
        enc.load_state_dict(pretrained["encoder"])
    head = M.RegHead(cfg["d_model"]).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()

    # y 标准化(用训练折统计),稳训练
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    Xtr_t = torch.from_numpy(Xtr).to(dev); mtr_t = torch.from_numpy(mtr).to(dev)
    ytr_t = torch.from_numpy(((ytr - ym) / ys).astype(np.float32)).to(dev)
    n = len(Xtr); bs = min(128, n)

    enc.train(); head.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            h = enc(Xtr_t[idx], mtr_t[idx])
            pred = head(h, mtr_t[idx])
            loss = lossf(pred, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    enc.eval(); head.eval()
    with torch.no_grad():
        h = enc(torch.from_numpy(Xte).to(dev), torch.from_numpy(mte).to(dev))
        pred = head(h, torch.from_numpy(mte).to(dev)).cpu().numpy()
    return pred * ys + ym       # 反标准化回 LFMC


def run(d, veg, cfg, pretrained, mu, sd, args, dev):
    """对某植被型做 LOGO,返回 pooled R²/RMSE。"""
    sel = np.ones(len(d["y"]), bool) if veg == "all" else (d["veg"] == veg)
    if args.min_obs > 1:                       # 只保留观测月数足够的序列(时序方法的前提)
        sel = sel & (d["mask"].sum(1) >= args.min_obs)
    X, mask, y, site = d["X"][sel], d["mask"][sel], d["y"][sel], d["site"][sel]
    Xs = M.apply_standardize(X, mask, mu, sd)
    groups = site
    ng = len(np.unique(groups))
    if ng < 2 or len(y) < 30:
        return None
    k = min(args.folds, ng)
    gkf = GroupKFold(n_splits=k)
    rng = np.random.default_rng(args.seed)
    yt, yp = [], []
    for tr, te in gkf.split(Xs, y, groups):
        # 少标签:在训练折里按"站点"抽 frac
        if args.label_frac < 1.0:
            tr_sites = np.unique(groups[tr])
            kk = max(1, int(round(len(tr_sites) * args.label_frac)))
            keep = set(rng.choice(tr_sites, kk, replace=False))
            tr = tr[np.isin(groups[tr], list(keep))]
            if len(tr) < 10:
                continue
        pred = train_fold(Xs[tr], mask[tr], y[tr], Xs[te], mask[te],
                          cfg, pretrained, args.epochs, args.lr, dev, args.seed)
        yt.append(y[te]); yp.append(pred)
    if not yt:
        return None
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return {"n": int(len(yt)), "n_sites": int(ng),
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/ts/au_s2_ts.npz")
    ap.add_argument("--mode", default="ssl", choices=["ssl", "scratch"])
    ap.add_argument("--pretrained", default="outputs/models/ssl_encoder.pt")
    ap.add_argument("--veg", default="forest", help="forest/shrubland/grassland/all 或 'each'")
    ap.add_argument("--label-frac", type=float, default=1.0)
    ap.add_argument("--min-obs", type=int, default=1, help="只用观测月数≥此的序列(时序方法前提)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(C.RESULTS_DIR, "ssl_lfmc.json"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = M.load_npz(args.npz)
    # 标准化统计:SSL 模式用预训练保存的 mu/sd(与预训练一致);scratch 用全数据估
    if args.mode == "ssl":
        ckpt = torch.load(args.pretrained, map_location=dev, weights_only=False)
        pretrained, mu, sd, cfg = ckpt, ckpt["mu"], ckpt["sd"], ckpt["cfg"]
    else:
        pretrained = None
        mu, sd = M.standardizer(d["X"], d["mask"])
        cfg = {"F": d["X"].shape[2], "T": d["T"], "d_model": 64, "nlayers": 3}

    vegs = C.VEG_TYPES + ["all"] if args.veg == "each" else [args.veg]
    print(f"mode={args.mode}  label_frac={args.label_frac}  dev={dev}  veg={vegs}")
    out_rows = {}
    for v in vegs:
        r = run(d, v, cfg, pretrained, mu, sd, args, dev)
        if r:
            out_rows[v] = r
            print(f"  [{v:10s}] LOGO R²={r['r2']:+.3f}  RMSE={r['rmse']:.1f}%  (n={r['n']}, sites={r['n_sites']})")
        else:
            print(f"  [{v:10s}] 样本/站点不足,跳过")

    # 累积写入(key = mode__frac__veg),便于画对比/曲线
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    allres = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for v, r in out_rows.items():
        allres[f"{args.mode}__frac{args.label_frac}__{v}"] = {**r, "mode": args.mode,
                                                              "label_frac": args.label_frac, "veg": v}
    json.dump(allres, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"结果累积写出:{args.out}")


if __name__ == "__main__":
    main()
