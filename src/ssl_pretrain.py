"""时序自监督预训练(masked time-series 重建,MAE 式)。在 AutoDL GPU 上跑。

在**全部** S2 时序序列上预训练(忽略 LFMC 标签 → 这就是"自监督/标签高效"的核心):
随机遮住已观测月度槽位,让编码器重建其特征。学到的编码器供 finetune_lfmc.py 用。

用法(AutoDL):
    python src/ssl_pretrain.py --epochs 100 --out outputs/models/ssl_encoder.pt
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
import ts_model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", default=["outputs/ts/au_s2_ts.npz"],
                    help="一个或多个 npz(无标注语料+标注集),X/mask 拼接做预训练语料")
    ap.add_argument("--out", default="outputs/models/ssl_encoder.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-frac", type=float, default=0.4, help="遮蔽比例")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    parts = [M.load_npz(p) for p in args.npz]
    X = np.concatenate([p["X"] for p in parts])
    mask = np.concatenate([p["mask"] for p in parts])
    print("预训练语料来源:" + ", ".join(f"{os.path.basename(p)}({len(d['X'])})"
                                      for p, d in zip(args.npz, parts)))
    mu, sd = M.standardizer(X, mask)
    Xs = M.apply_standardize(X, mask, mu, sd)
    N, T, F = Xs.shape
    print(f"预训练语料:{N} 条序列, T={T}, F={F}, 平均覆盖 {mask.mean():.0%}")

    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(mask))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    enc = M.TSEncoder(F, T, d_model=args.d_model, nlayers=args.nlayers).to(dev)
    head = M.SSLHead(args.d_model, F).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(args.seed)

    enc.train(); head.train()
    for ep in range(1, args.epochs + 1):
        tot, nb = 0.0, 0
        for xb, mb in dl:
            xb, mb = xb.to(dev), mb.to(dev)
            ssl = M.make_ssl_mask(mb, args.mask_frac, gen).to(dev)
            if not ssl.any():
                continue
            h = enc(xb, mb, ssl_mask=ssl)
            pred = head(h)
            # 仅在被遮蔽的观测槽位上算重建 MSE
            loss = ((pred - xb)[ssl] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  recon MSE {tot/max(nb,1):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "mu": mu, "sd": sd,
                "cfg": {"F": F, "T": T, "d_model": args.d_model, "nlayers": args.nlayers}},
               args.out)
    print(f"预训练编码器写出:{args.out}")
    print("下一步:python src/finetune_lfmc.py --pretrained " + args.out)


if __name__ == "__main__":
    main()
