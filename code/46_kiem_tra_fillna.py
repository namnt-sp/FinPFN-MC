"""Kiem tra fillna(0) trong loi suat danh muc (CHI DOC, khong ghi de ket qua H3 da khoa).
 (1) Ty le loi suat ngay bi thieu trong khoang niem yet cua tung ma, VN 2019-2024.
 (2) Danh muc binh quan deu VN-100: cach cu (fillna(0) tren ca 100 cot) va cach dung (TB tren ma co du lieu).
 (3) Kiem tra do ben H3 (bo tro): loi suat nam giu tinh tu GIA dau ky -> cuoi ky (gia gan nhat da biet),
     thay cho nhan lien tiep loi suat ngay voi o thieu = 0. Cung logic danh muc nhu script 40.
Chay tu D:\\LuanVan\\Code:  python 46_kiem_tra_fillna.py   (~3-5 phut). Ra: ../Ketqua/kiem_tra_fillna.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd

KQ = Path("../Ketqua"); VN = KQ / "kiem_tra_vn100"
PHI, THUE, TRAN = 0.0025, 0.001, 0.067
f = open(KQ / "kiem_tra_fillna.txt", "w", encoding="utf-8")


def ghi(s=""):
    print(s); f.write(s + "\n"); f.flush()


sach = pd.read_csv("../Data/vn100_sach.csv", parse_dates=["date"])
gia = sach.pivot(index="date", columns="ma", values="gia").sort_index()
ret = sach.pivot(index="date", columns="ma", values="ret").sort_index().reindex(gia.index)
TU, DEN = "2019-01-01", "2024-12-31"
g, r = gia.loc[TU:DEN], ret.loc[TU:DEN]

# ---------- (1) o thieu trong khoang niem yet
trong = pd.DataFrame(False, index=g.index, columns=g.columns)
for c in g.columns:
    v = g[c].dropna()
    if len(v):
        trong.loc[v.index[0]:v.index[-1], c] = True
trong.iloc[0] = False                                         # phien dau khong co loi suat theo dinh nghia
n_tong, n_thieu = int(trong.values.sum()), int((r.isna() & trong).values.sum())
ghi(f"(1) VN 2019-2024: {n_thieu:,} / {n_tong:,} o loi suat thieu trong khoang niem yet ({n_thieu / n_tong:.2%})")

# ---------- (2) binh quan deu
def tuan(x):
    w = (1 + x).groupby(x.index.to_period("W-FRI")).prod() - 1
    return (1 + w).prod() ** (52 / len(w)) - 1, w.mean() / w.std() * np.sqrt(52)


cu = tuan(r.fillna(0).mean(axis=1)); dung = tuan(r.mean(axis=1))
ghi(f"(2) VN-100 binh quan deu (chua phi): cach cu LS nam {cu[0]:.2%}, Sharpe {cu[1]:.3f} | "
    f"cach dung (TB tren ma co du lieu) LS nam {dung[0]:.2%}, Sharpe {dung[1]:.3f}")

# ---------- (3) danh muc A-D, loi suat nam giu tu gia
du = pd.read_parquet(VN / "du_bao_ngay.parquet")
hang = {m: du.assign(h=du.groupby("date")[m].rank(pct=True)).pivot(index="date", columns="id", values="h")
        for m in ["M1", "MC"]}
ngay = hang["MC"].index
ky = pd.Series(ngay, index=ngay).dt.to_period("W-FRI")
cuoi = pd.Series(ngay, index=ngay).groupby(ky).max().to_numpy()
g_ff = gia.ffill()                                             # gia gan nhat da biet (ngung GD: giu gia cu)


def chay(mo_hinh, dai_tre, cach):
    tb = hang[mo_hinh].groupby(ky.to_numpy()).mean(); w = pd.Series(dtype=float); out = []
    for k in range(len(cuoi) - 1):
        d, d2 = cuoi[k], cuoi[k + 1]
        sig = tb.loc[ky.loc[d]].dropna(); n = len(sig); rk = sig.rank(ascending=False)
        rt = ret.loc[d] if d in ret.index else pd.Series(dtype=float)
        tran, san = set(rt[rt >= TRAN].index), set(rt[rt <= -TRAN].index)
        tv, tr = set(rk[rk <= np.ceil(.2 * n)].index), set(rk[rk <= np.ceil(.5 * n)].index)
        cu_ = set(w.index); giu = (cu_ & (tr if dai_tre else tv)) | (cu_ & san); nam = sorted(giu | ((tv - cu_) - tran))
        mt = pd.Series(1 / len(nam), index=nam) if nam else pd.Series(dtype=float)
        tat = mt.index.union(w.index); dw = mt.reindex(tat, fill_value=0) - w.reindex(tat, fill_value=0)
        cp = dw.clip(lower=0).sum() * PHI + (-dw).clip(lower=0).sum() * (PHI + THUE)
        if cach == "ret":
            kp = ret.loc[(ret.index > d) & (ret.index <= d2), nam].fillna(0)
            gg = (1 + kp).prod() - 1 if len(kp) else pd.Series(0.0, index=nam)
        else:
            gg = (g_ff.loc[d2, nam] / g_ff.loc[d, nam] - 1).fillna(0)
        gop = float((mt * gg).sum()) if nam else 0.0
        out.append((1 - cp) * (1 + gop) - 1)
        w = mt * (1 + gg) / (1 + gop) if nam else pd.Series(dtype=float)
    return np.array(out)


def boot(d, B=10_000, L=4, seed=0):
    T = len(d); rng = np.random.default_rng(seed); st = rng.integers(T, size=(B, T))
    nw_ = rng.random((B, T)) < 1 / L; nw_[:, 0] = True
    vt = np.maximum.accumulate(np.where(nw_, np.arange(T), 0), axis=1)
    idx = (np.take_along_axis(st, vt, axis=1) + np.arange(T) - vt) % T
    return d.mean(), (np.abs((d - d.mean())[idx].mean(1)) >= abs(d.mean())).mean()


for cach, ten in [("ret", "cach goc script 40 (nhan loi suat ngay, o thieu = 0)"), ("gia", "loi suat tu gia dau-cuoi ky")]:
    ht = {"A": chay("M1", False, cach), "B": chay("MC", False, cach), "C": chay("MC", True, cach), "D": chay("M1", True, cach)}
    ghi(f"\n(3) {ten}:")
    for k, v in ht.items():
        nav = np.cumprod(1 + v)
        ghi(f"   {k}: LS rong nam {nav[-1] ** (52 / len(v)) - 1:+.2%} | Sharpe {v.mean() / v.std(ddof=1) * np.sqrt(52):.3f}")
    for a, b in [("C", "A"), ("D", "A"), ("C", "D")]:
        m, p = boot(ht[a] - ht[b]); ghi(f"   {a} - {b}: {m * 100:+.3f}%/tuan (p {p:.4f})")
ghi(f"\nLuu: {KQ / 'kiem_tra_fillna.txt'}")
