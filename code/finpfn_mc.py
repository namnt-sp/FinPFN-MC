"""FinPFN-MC va M1: du lieu theo cap ngay, tinh chinh, suy luan co trung binh phan hoach.
Bam de cuong v3. Cac lua chon cai dat ap dung NHU NHAU cho M1 va FinPFN-MC:
  - Dau vao: dac trung doi sang hang trong ngay [-0.5, 0.5]; FinPFN-MC them 2 cot hang du bao Ridge, LGBM.
  - Ngu canh = m ma cua ngay t-1 (co nhan), cau hoi = cung m ma o ngay t; nhan chuan hoa z trong nhom.
  - Tinh chinh tu TabPFN v2 (nhu Wang & Lera): AdamW schedule-free, lr 1e-5, cat gradient 1.0,
    100 buoc/epoch, dung som khi loss kiem dinh khong giam 20 epoch.
  - Suy luan bang dung lan truyen thuan nhu luc tinh chinh (khong qua tien xu ly/xao tron cot cua
    TabPFNRegressor) -> thu tu cot co dinh, khop giua huan luyen va suy luan, tat dinh.
  - Du bao diem = trung vi cua phan phoi (nhu bai goc).
Khac nhau giua M1 va FinPFN-MC chi o 3 thanh phan cua de cuong: cot xep chong, K phan hoach, co nhom thay doi.
"""
from __future__ import annotations

import math
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from dac_trung import TEN
from mo_hinh_nen import chuan_hoa_ngay

COT_NEN = ["ridge", "lgbm"]
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*GradScaler.*")


# ----------------------------------------------------------------------------- du lieu
def chuan_bi(df_dt: pd.DataFrame, nen: pd.DataFrame | None, tu=None, den=None):
    """Tra ve (kho, ngay, feats). kho[ngay] = (ids sap xep, X float32, y_tho float32)."""
    if tu is not None:
        df_dt = df_dt[(df_dt["date"] >= tu) & (df_dt["date"] <= den)]
    z = chuan_hoa_ngay(df_dt, TEN).drop(columns="y").merge(df_dt[["date", "id", "target"]], on=["date", "id"])
    feats = list(TEN)
    if nen is not None:
        n = nen.copy()
        for c in COT_NEN:
            n[c] = n.groupby("date")[c].rank(pct=True) - 0.5
        z = z.merge(n[["date", "id"] + COT_NEN], on=["date", "id"])   # chi ngay co du bao nen
        feats += COT_NEN
    kho = {}
    for d, g in z.groupby("date"):
        g = g.sort_values("id")
        kho[np.datetime64(d, "ns")] = (g["id"].to_numpy(), g[feats].to_numpy(np.float32),
                                       g["target"].to_numpy(np.float32))
    return kho, np.array(sorted(kho), dtype="datetime64[ns]"), feats


def ghep(kho, d0, d1):
    """Ma hop le: co nhan o d0 (ngu canh) va co mat o d1."""
    i0, X0, y0 = kho[d0]
    i1, X1, y1 = kho[d1]
    ok = np.flatnonzero(~np.isnan(y0))
    ids, a, b = np.intersect1d(i0[ok], i1, return_indices=True)
    a = ok[a]
    return ids, X0[a], y0[a], X1[b], y1[b]


def zs(y):
    if np.isnan(y).all():
        return y
    m, s = np.nanmean(y), np.nanstd(y, ddof=1)
    return (y - m) / s if s > 0 else np.zeros_like(y)


# ----------------------------------------------------------------------------- mo hinh
def nap_mo_hinh(ckpt, seed=0, device="cuda"):
    from tabpfn.model.loading import load_model_criterion_config
    model, crit, cfg = load_model_criterion_config(
        str(ckpt), check_bar_distribution_criterion=False, cache_trainset_representation=False,
        which="regressor", version="v2", download=False, model_seed=seed)
    model.criterion = crit
    return model.to(device), crit.to(device), cfg


def luu_mo_hinh(model, cfg, path):
    torch.save(dict(state_dict=model.state_dict(), config=cfg.__dict__), str(path))


def _tien(model, Xc, yc, Xq, device, amp):
    with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
        return model(train_x=Xc.to(device), train_y=yc.to(device).unsqueeze(-1), test_x=Xq.to(device))


# ----------------------------------------------------------------------------- tinh chinh
def lay_batch(kho, ngay, rng, b, m):
    Xc, yc, Xq, yq = [], [], [], []
    while len(Xc) < b:
        i = int(rng.integers(1, len(ngay)))
        ids, X0, y0, X1, y1 = ghep(kho, ngay[i - 1], ngay[i])
        if len(ids) < m:
            continue
        s = rng.choice(len(ids), m, replace=False)
        Xc.append(X0[s]); yc.append(zs(y0[s])); Xq.append(X1[s]); yq.append(zs(y1[s]))
    t = lambda L: torch.tensor(np.stack(L, 1))
    return t(Xc), t(yc), t(Xq), t(yq)


def tinh_chinh(model, crit, cfg, kho, ngay_hl, kho_kd, ngay_kd, *, save_path, co_nhom=(50, 50),
               lr=1e-5, batch=64, accum=1, buoc_moi_epoch=100, max_epoch=100, kien_nhan=20,
               seed=0, device="cuda", amp=True, gioi_han_gio=None, nhat_ky=print, tiep_tuc=None):
    """co_nhom=(50,50): M1. co_nhom=(30,60): FinPFN-MC (co nhom rut ngau nhien moi buoc)."""
    from schedulefree import AdamWScheduleFree
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = AdamWScheduleFree(model.parameters(), lr=lr)
    loss_fn = FullSupportBarDistribution(borders=crit.borders, ignore_nan_targets=True).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device == "cuda")
    t0, lich_su, ep0 = time.time(), [], 1
    model.eval(); opt.eval()
    if tiep_tuc is not None and Path(tiep_tuc).exists():          # chay tiep sau khi bi ngat
        st = torch.load(tiep_tuc, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        rng.bit_generator.state = st["rng"]
        best, cho, lich_su, ep0 = st["best"], st["cho"], st["lich_su"], st["epoch"] + 1
        t0 -= st["giay"]
        nhat_ky(f"Chay tiep tu epoch {ep0} (best val loss {best['loss']:.5f})")
    else:
        best = danh_gia(model, crit, kho_kd, ngay_kd, device=device, amp=amp)
        nhat_ky(f"epoch 0 | val loss {best['loss']:.5f} | val IC {best['ic']:.4f}")
        luu_mo_hinh(model, cfg, save_path)
        cho = 0
    for ep in range(ep0, max_epoch + 1):
        model.train(); opt.train()
        tl = []
        for _ in range(buoc_moi_epoch):
            m = int(rng.integers(co_nhom[0], co_nhom[1] + 1))
            for _ in range(accum):
                Xc, yc, Xq, yq = lay_batch(kho, ngay_hl, rng, batch // accum, m)
                with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                    logits = _tien(model, Xc, yc, Xq, device, amp)
                    loss = loss_fn(logits=logits.float(), y=yq.to(device)).mean() / accum
                scaler.scale(loss).backward()
                tl.append(loss.item() * accum)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        model.eval(); opt.eval()
        kd = danh_gia(model, crit, kho_kd, ngay_kd, device=device, amp=amp)
        tot = kd["loss"] < best["loss"]
        if tot:
            best, cho = kd, 0
            luu_mo_hinh(model, cfg, save_path)
        else:
            cho += 1
        gio = (time.time() - t0) / 3600
        lich_su.append(dict(epoch=ep, train_loss=float(np.mean(tl)), val_loss=kd["loss"], val_ic=kd["ic"], gio=gio))
        nhat_ky(f"epoch {ep} | train {np.mean(tl):.5f} | val loss {kd['loss']:.5f} | val IC {kd['ic']:.4f}"
                f"{' *BEST*' if tot else ''} | {gio:.2f} gio")
        if tiep_tuc is not None:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                            rng=rng.bit_generator.state, best=best, cho=cho, lich_su=lich_su, epoch=ep,
                            giay=time.time() - t0), tiep_tuc)
        if cho >= kien_nhan or (gioi_han_gio and gio * (ep + 1) / ep > gioi_han_gio):
            nhat_ky("Dung: " + ("het kien nhan" if cho >= kien_nhan else "het thoi gian"))
            break
    return pd.DataFrame(lich_su), best


# ----------------------------------------------------------------------------- suy luan
def du_bao_ngay(model, crit, kho, d0, d1, *, K=1, co_nhom=50, rng=None, device="cuda", amp=True, bs=64,
                loss_fn=None, ids_chon=None):
    """K phan hoach ngau nhien; moi phan hoach chia deu cac ma thanh round(n/co_nhom) nhom.
    Tra ve (ids, du bao trung binh, nhan tho ngay d1[, loss trung binh])."""
    ids, X0, y0, X1, y1 = ghep(kho, d0, d1)
    if ids_chon is not None:                                  # gioi han trong mot vu tru ma
        m = np.isin(ids, ids_chon)
        ids, X0, y0, X1, y1 = ids[m], X0[m], y0[m], X1[m], y1[m]
    n = len(ids)
    g = max(1, round(n / co_nhom))
    rng = rng or np.random.default_rng(0)
    nhom = []
    for _ in range(K):
        nhom += np.array_split(rng.permutation(n), g)
    theo_dai = defaultdict(list)
    for e in nhom:
        theo_dai[len(e)].append(e)
    tong, dem, losses = np.zeros(n), np.zeros(n), []
    for L, es in theo_dai.items():
        for j in range(0, len(es), bs):
            ch = es[j:j + bs]
            Xc = torch.tensor(np.stack([X0[e] for e in ch], 1))
            yc = torch.tensor(np.stack([zs(y0[e]) for e in ch], 1))
            Xq = torch.tensor(np.stack([X1[e] for e in ch], 1))
            with torch.no_grad():
                lg = _tien(model, Xc, yc, Xq, device, amp).float()
                med = crit.median(lg).cpu().numpy()
                if loss_fn is not None:
                    yq = torch.tensor(np.stack([zs(y1[e]) for e in ch], 1)).to(device)
                    losses.append(loss_fn(logits=lg, y=yq).mean().item())
            for k, e in enumerate(ch):
                tong[e] += med[:, k]; dem[e] += 1
    pred = tong / np.maximum(dem, 1)
    return (ids, pred, y1, float(np.nanmean(losses))) if loss_fn is not None else (ids, pred, y1)


def danh_gia(model, crit, kho, ngay, *, device="cuda", amp=True):
    """Kiem dinh khi tinh chinh: 1 phan hoach co dinh (seed 0), nhom 50; tra loss TB va rank IC TB."""
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    loss_fn = FullSupportBarDistribution(borders=crit.borders, ignore_nan_targets=True).to(device)
    ics, ls = [], []
    for i in range(1, len(ngay)):
        ids, p, y, l = du_bao_ngay(model, crit, kho, ngay[i - 1], ngay[i], K=1, co_nhom=50,
                                   rng=np.random.default_rng([0, i]), device=device, amp=amp, loss_fn=loss_fn)
        ok = ~np.isnan(y)
        if ok.sum() > 5:
            ics.append(stats.spearmanr(p[ok], y[ok])[0])
        ls.append(l)
    return dict(loss=float(np.nanmean(ls)), ic=float(np.nanmean(ics)))


def du_bao_giai_doan(model, crit, kho, ngay_all, tu, den, *, K=1, co_nhom=50, seed=0, device="cuda",
                     amp=True, nhat_ky=None):
    """Du bao moi ngay trong [tu, den]; ngu canh la ngay giao dich lien truoc."""
    rows, t0 = [], time.time()
    idx = [i for i, d in enumerate(ngay_all) if i > 0 and np.datetime64(tu) <= d <= np.datetime64(den)]
    for n_, i in enumerate(idx, 1):
        ids, p, y = du_bao_ngay(model, crit, kho, ngay_all[i - 1], ngay_all[i], K=K, co_nhom=co_nhom,
                                rng=np.random.default_rng([seed, i]), device=device, amp=amp)
        rows.append(pd.DataFrame({"date": ngay_all[i], "id": ids, "pred": p, "target": y}))
        if nhat_ky and n_ % 20 == 0:
            nhat_ky(f"  {n_}/{len(idx)} ngay | {(time.time() - t0) / 60:.1f} phut")
    return pd.concat(rows, ignore_index=True)
