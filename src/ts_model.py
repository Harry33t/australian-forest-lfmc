"""时序模型 + 数据工具(F2 SSL 共享)。在 AutoDL(torch 2.5 + CUDA)上跑。

输入数据:satellite_dea_ts.py 产出的 npz —— X(N,T,F) 月度 S2 指数序列, mask(N,T) 观测掩码,
y(N,) LFMC, veg_type/site/row_id 对齐数组。

模型:小型 Transformer 编码器(masked time-series 自监督;缺测槽位用 padding mask 不参与注意力)。
  · SSL:随机遮住部分*已观测*槽位,重建其特征向量(MAE 式)。
  · 微调:对已观测槽位做 masked mean-pool → MLP 回归 LFMC。
"""
from __future__ import annotations
import json
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_npz(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return {
        "X": z["X"].astype(np.float32),      # (N,T,F)
        "mask": z["mask"].astype(bool),      # (N,T) True=观测
        "y": z["y"].astype(np.float32),      # (N,)
        "veg": z["veg_type"], "site": z["site"], "row_id": z["row_id"],
        "features": meta["features"], "T": meta["T"],
    }


def standardizer(X, mask):
    """按特征算 mean/std(只在观测槽位上)。返回 (mean,std) 形 (F,)。"""
    obs = X[mask]                            # (n_obs, F)
    mu = obs.mean(0); sd = obs.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardize(X, mask, mu, sd):
    Xs = (X - mu) / sd
    Xs[~mask] = 0.0                          # 缺测置 0(配合 padding mask)
    return Xs.astype(np.float32)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class TSEncoder(nn.Module):
    """Transformer 编码器:Linear 嵌入 + 学习位置编码 + N 层 encoder。"""
    def __init__(self, n_feat, T, d_model=64, nhead=4, nlayers=3, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, T, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))   # SSL 遮蔽用
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.d_model = d_model
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, obs_mask, ssl_mask=None):
        """x:(B,T,F) obs_mask:(B,T) True=观测。ssl_mask:(B,T) True=被 SSL 遮蔽的观测槽。
        返回 token 表示 (B,T,d)。缺测槽位作为 key_padding_mask 不被注意。"""
        h = self.inp(x) + self.pos
        if ssl_mask is not None:
            h = torch.where(ssl_mask.unsqueeze(-1), self.mask_token, h)
        key_pad = ~obs_mask                  # True 处被忽略
        # 全缺测的样本极少;给它至少一个可注意位防 NaN
        allpad = key_pad.all(dim=1)
        if allpad.any():
            key_pad = key_pad.clone(); key_pad[allpad, 0] = False
        return self.enc(h, src_key_padding_mask=key_pad)


class SSLHead(nn.Module):
    """重建头:token 表示 → 特征向量。"""
    def __init__(self, d_model, n_feat):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, n_feat))

    def forward(self, h):
        return self.proj(h)


class RegHead(nn.Module):
    """回归头:对观测槽位 masked mean-pool → MLP → LFMC。"""
    def __init__(self, d_model, hidden=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, h, obs_mask):
        w = obs_mask.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / (w.sum(1) + 1e-6)
        return self.mlp(pooled).squeeze(-1)


def make_ssl_mask(obs_mask, frac, generator=None):
    """在每个样本的观测槽位里随机选 frac 比例遮蔽(至少 1 个,若有 ≥2 观测)。"""
    B, T = obs_mask.shape
    ssl = torch.zeros_like(obs_mask)
    for i in range(B):
        idx = torch.where(obs_mask[i])[0]
        if len(idx) < 2:
            continue
        k = max(1, int(round(len(idx) * frac)))
        perm = idx[torch.randperm(len(idx), generator=generator)[:k]]
        ssl[i, perm] = True
    return ssl
