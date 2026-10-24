"""Buoc 4d - VONG THU CSI 2021 theo de cuong v3 (chon cau hinh truoc khi khoa).
Danh gia tren vu tru 100 ma giong H1: moi ngay 2 lan x 5 vu tru roi nhau (hat giong co dinh); IC tinh trong
tung vu tru roi lay trung binh theo ngay. Do on dinh tinh tren ca mat cat.
Mo hinh: M1 (K=1) | M1+K (K=20) | FinPFN-MC voi K = 1, 20, 50 | Ridge | LGBM | tron deu (M1+Ridge+LGBM) |
         tron theo IC cuon chieu 60 ngay.
Tieu chi khoa: IC(MC) >= IC(M1) + 0,005  VA  IC(MC) >= IC(tron deu) - 0,005.
Chay tu D:\\LuanVan\\Code:  python 33_vong_thu_2021.py      (~30-60 phut)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import finpfn_mc as F

OUT = Path("../Ketqua/vong_thu_2021"); OUT.mkdir(parents=True, exist_ok=True)
CK = Path("../Ketqua/tinh_chinh")
dev = "cuda" if torch.cuda.is_available() else "cpu"
SO_LAN, SO_VT, CO_VT = 2, 5, 100
K_MC = [1, 20, 50]
f_log = open(OUT / "nhat_ky_vong_thu.txt", "w", encoding="utf-8")


def ghi(s):
    print(s); f_log.write(s + "\n"); f_log.flush()


df = pd.read_parquet("../Data/csi500_dac_trung.parquet")
nen = pd.read_parquet("../Data/nen_csi500.parquet")
kho1, ngay1, _ = F.chuan_bi(df, None, tu="2020-12-01", den="2021-12-31")
khoM, ngayM, _ = F.chuan_bi(df, nen, tu="2020-12-01", den="2021-12-31")
ngay = np.intersect1d(ngay1, ngayM)
idx = [i for i in range(1, len(ngay)) if ngay[i] >= np.datetime64("2021-01-01")]
nen_d = nen.set_index(["date", "id"])

m1, c1, _ = F.nap_mo_hinh(CK / "m1_best.ckpt", device=dev)
mc, cm, _ = F.nap_mo_hinh(CK / "mc_best.ckpt", device=dev)
for m in (m1, mc):
    m.eval()
ghi(f"Vong thu CSI 2021: {len(idx)} ngay | {SO_LAN}x{SO_VT} vu tru {CO_VT} ma/ngay | thiet bi {dev}")


def rk(x):
    return pd.Series(x).rank(pct=True).to_numpy()


hang_ic, mat_cat, t0 = [], [], time.time()
for n_, i in enumerate(idx, 1):
    d0, d1 = ngay[i - 1], ngay[i]
    ids1 = F.ghep(kho1, d0, d1)[0]
    idsM = F.ghep(khoM, d0, d1)[0]
    chung = np.intersect1d(ids1, idsM)
    rng = np.random.default_rng([2021, i])
    seed = lambda *a: np.random.default_rng([2021, i, *a])

    # --- mat cat day du (cho do on dinh va phan tich sau)
    a, p, y = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=1, rng=seed(0), device=dev, ids_chon=chung)
    b, q, _ = F.du_bao_ngay(mc, cm, khoM, d0, d1, K=20, rng=seed(1), device=dev, ids_chon=chung)
    mat_cat.append(pd.DataFrame({"date": d1, "id": a, "m1": p, "target": y}).merge(
        pd.DataFrame({"date": d1, "id": b, "mc_K20": q}), on=["date", "id"]))

    # --- vu tru 100 ma
    for lan in range(SO_LAN):
        perm = rng.permutation(chung)[: SO_VT * CO_VT]
        for v in range(SO_VT):
            u = perm[v * CO_VT:(v + 1) * CO_VT]
            if len(u) < CO_VT:
                continue
            r = {}
            ids_u, r["M1"], y_u = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=1, rng=seed(lan, v, 0), device=dev, ids_chon=u)
            _, r["M1+K"], _ = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=20, rng=seed(lan, v, 1), device=dev, ids_chon=u)
            for k in K_MC:
                idk, pk, _ = F.du_bao_ngay(mc, cm, khoM, d0, d1, K=k, rng=seed(lan, v, 2, k), device=dev, ids_chon=u)
                assert (idk == ids_u).all()
                r[f"MC_K{k}"] = pk
            nn_ = nen_d.loc[[(pd.Timestamp(d1), x) for x in ids_u]]
            r["Ridge"], r["LGBM"] = nn_["ridge"].to_numpy(), nn_["lgbm"].to_numpy()
            r["Tron deu"] = (rk(r["M1"]) + rk(r["Ridge"]) + rk(r["LGBM"])) / 3
            ok = ~np.isnan(y_u)
            for ten, pr in r.items():
                hang_ic.append((d1, lan, v, ten, stats.spearmanr(pr[ok], y_u[ok])[0]))
            # luu hang thanh phan cho tron IC cuon chieu (tinh o vong sau, chi dung qua khu)
            hang_ic.append((d1, lan, v, "_rank", (rk(r["M1"]), rk(r["Ridge"]), rk(r["LGBM"]), y_u)))
    if n_ % 10 == 0:
        ghi(f"  {n_}/{len(idx)} ngay | {(time.time() - t0) / 60:.1f} phut")

# ---- Tron theo IC cuon chieu 60 ngay (trong so = IC TB 60 ngay truoc, khong am; <20 ngay lich su -> deu)
H = pd.DataFrame(hang_ic, columns=["date", "lan", "vt", "mo_hinh", "gia_tri"])
ic = H[H["mo_hinh"] != "_rank"].copy()
ic["gia_tri"] = ic["gia_tri"].astype(float)
ngay_ic = ic.groupby(["date", "mo_hinh"])["gia_tri"].mean().unstack()
tp = ["M1", "Ridge", "LGBM"]
qua_khu = ngay_ic[tp].shift(1).rolling(60, min_periods=20).mean()
them = []
for _, row in H[H["mo_hinh"] == "_rank"].iterrows():
    rm1, rr, rl, y_u = row["gia_tri"]
    w = qua_khu.loc[row["date"]].clip(lower=0).fillna(0).to_numpy()
    w = np.ones(3) / 3 if w.sum() <= 0 else w / w.sum()
    pr = w[0] * rm1 + w[1] * rr + w[2] * rl
    ok = ~np.isnan(y_u)
    them.append((row["date"], row["lan"], row["vt"], "Tron IC 60n", stats.spearmanr(pr[ok], y_u[ok])[0]))
ic = pd.concat([ic, pd.DataFrame(them, columns=ic.columns)])
ngay_ic = ic.groupby(["date", "mo_hinh"])["gia_tri"].mean().unstack()
ngay_ic.to_csv(OUT / "ic_theo_ngay.csv")


def nw(x, lag=5):
    x = np.asarray(x) - np.mean(x); T = len(x)
    v = x @ x / T + 2 * sum((1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / T for l in range(1, lag + 1))
    return np.sqrt(v / T)


# ---- Do on dinh tren mat cat
mc_df = pd.concat(mat_cat)
mc_df.to_parquet(OUT / "du_bao_mat_cat_2021.parquet", index=False)
od = {}
for c in ["m1", "mc_K20"]:
    w = mc_df.pivot(index="date", columns="id", values=c)
    od[c] = w.corrwith(w.shift(1), axis=1, method="spearman").mean()

thu_tu = ["M1", "M1+K"] + [f"MC_K{k}" for k in K_MC] + ["Ridge", "LGBM", "Tron deu", "Tron IC 60n"]
ghi(f"\n{'Mo hinh':12} {'IC TB':>7} {'IR':>6} {'t(NW)':>6}")
for c in thu_tu:
    x = ngay_ic[c].dropna()
    ghi(f"{c:12} {x.mean():7.4f} {x.mean() / x.std():6.3f} {x.mean() / nw(x):6.2f}")
ghi(f"\nDo on dinh (mat cat): M1 {od['m1']:.3f} | FinPFN-MC K=20 {od['mc_K20']:.3f}")

ghi("\n=== TIEU CHI KHOA (de cuong) ===")
for k in K_MC:
    c = f"MC_K{k}"
    d1_ = ngay_ic[c] - ngay_ic["M1"]
    d2_ = ngay_ic[c] - ngay_ic["Tron deu"]
    ok1, ok2 = d1_.mean() >= 0.005, d2_.mean() >= -0.005
    ghi(f"{c}: - M1 = {d1_.mean():+.4f} (t {d1_.mean() / nw(d1_):.2f}) {'DAT' if ok1 else 'CHUA'} | "
        f"- Tron deu = {d2_.mean():+.4f} (t {d2_.mean() / nw(d2_):.2f}) {'DAT' if ok2 else 'CHUA'}"
        f"{'  ==> QUA' if ok1 and ok2 else ''}")
ghi(f"\nTong thoi gian: {(time.time() - t0) / 60:.0f} phut. Ket qua: {OUT}")
