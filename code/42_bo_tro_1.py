"""Buoc 6c - PHAN TICH BO TRO (dat truoc trong de cuong; bao cao day du, khong co tieu chi dat/khong dat).
  (1) Theo che do: ngay bien dong cao (VN va CSI 500).
      Chi so dai dien = loi suat binh quan deu cua cac ma trong ro (khong co du lieu VN-Index/CSI 500 index).
      RV20 = do lech chuan 20 ngay x sqrt(252); ngay bien dong cao: RV20_t > phan vi 90 cua RV20 cac ngay truoc
      (mo rong, toi thieu 250 ngay). (Khai bao: dung chi so dai dien thay VN-Index.)
  (2) Thien lech song sot: chi cac ma VN co du lieu tu phien dau tien (3/1/2017): IC, do on dinh, H3 (C - A).
  (3) H1 tren CSI 500 giai doan 2022 - 4/2023 (DA TIEP XUC, chi mo ta), ca mat cat.
  (4) Loi suat vuot troi cua top 20% (gop, truoc phi) song song voi IC, VN va CSI 500.
Chay tu D:\\LuanVan\\Code:  python 42_bo_tro_1.py      (~15-30 phut; dung GPU cho phan (3))
"""
import json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import finpfn_mc as F
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

KQ = Path("../Ketqua"); OUT = KQ / "bo_tro"; OUT.mkdir(parents=True, exist_ok=True)
VN, CS = KQ / "kiem_tra_vn100", KQ / "kiem_tra_csi500"
dev = "cuda" if torch.cuda.is_available() else "cpu"
f_log = open(OUT / "bo_tro_1.txt", "w", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


khoa = json.loads((KQ / "cau_hinh_khoa.json").read_text(encoding="utf-8"))
for k, p in {"mo_hinh_M1": KQ / "tinh_chinh/m1_best.ckpt", "mo_hinh_FinPFN_MC": KQ / "tinh_chinh/mc_best.ckpt",
             "code_finpfn_mc": Path("finpfn_mc.py")}.items():
    if hashlib.sha256(p.read_bytes()).hexdigest() != khoa["dau_van_tay_sha256"][k]:
        raise SystemExit(f"DAU VAN TAY KHONG KHOP: {k}")


def nw(x, lag=5):
    x = np.asarray(pd.Series(x).dropna(), float); x = x - x.mean(); T = len(x)
    v = x @ x / T + 2 * sum((1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / T for l in range(1, lag + 1))
    return np.sqrt(v / T)


def mt(x):
    x = pd.Series(x).dropna(); return f"{x.mean():+.4f} (t {x.mean() / nw(x):.2f}, n {len(x)})"


def ic_cot(d, cols, tgt="target"):
    ok = d.dropna(subset=[tgt])
    return pd.DataFrame({c: ok.groupby("date")[[c, tgt]].apply(lambda g: stats.spearmanr(g[c], g[tgt])[0])
                         for c in cols})


sach = pd.read_csv("../Data/vn100_sach.csv", parse_dates=["date"])
ret_vn = sach.pivot(index="date", columns="ma", values="ret").sort_index()
du = pd.read_parquet(VN / "du_bao_ngay.parquet")
R = {c: du.groupby("date")[c].rank(pct=True) for c in ["M1", "Ridge", "LGBM"]}
du["Tron chinh"] = (R["M1"] + R["Ridge"] + R["LGBM"]) / 3
ic_vn = pd.read_csv(VN / "ic_theo_ngay.csv", index_col=0, parse_dates=True)
MO = ["M1", "MC", "Ridge", "LGBM", "Tron chinh"]


def ngay_bien_dong_cao(ret_wide):
    idx = ret_wide.mean(axis=1)
    rv = idx.rolling(20, min_periods=20).std() * np.sqrt(252)
    nguong = rv.shift(1).expanding(min_periods=250).quantile(0.90)
    return (rv > nguong).rename("cao"), rv


# ============ (1) theo che do bien dong
ghi("=== (1) NGAY BIEN DONG CAO (bo tro) ===")
cao_vn, _ = ngay_bien_dong_cao(ret_vn)
x = ic_vn.join(cao_vn, how="inner").dropna(subset=["cao"])
ghi(f"VN 2019-2024: {int(x['cao'].sum())} ngay bien dong cao / {len(x)}")
ghi(x.groupby("cao")[MO].mean().rename(index={True: "Bien dong cao", False: "Binh thuong"}).round(4).to_string())
h = x[x["cao"] == True]
ghi(f"  Trong ngay bien dong cao: MC - M1 = {mt(h['MC'] - h['M1'])} | MC - Tron chinh = {mt(h['MC'] - h['Tron chinh'])}")
dc = pd.read_parquet("../Data/csi500_dac_trung.parquet", columns=["date", "id", "rev1"])
ret_cs = (-dc.pivot(index="date", columns="id", values="rev1")).sort_index()     # rev1 = -loi suat ngay
cao_cs, _ = ngay_bien_dong_cao(ret_cs)
ic_cs = pd.read_csv(CS / "ic_theo_ngay.csv", index_col=0, parse_dates=True)
x = ic_cs.join(cao_cs, how="inner").dropna(subset=["cao"])
ghi(f"\nCSI 500 5/2023-6/2026 (vu tru 100 ma): {int(x['cao'].sum())} ngay bien dong cao / {len(x)}")
ghi(x.groupby("cao")[MO].mean().rename(index={True: "Bien dong cao", False: "Binh thuong"}).round(4).to_string())
h = x[x["cao"] == True]
ghi(f"  Trong ngay bien dong cao: MC - M1 = {mt(h['MC'] - h['M1'])} | MC - Tron chinh = {mt(h['MC'] - h['Tron chinh'])}")

# ============ (2) thien lech song sot (VN)
ghi("\n=== (2) THIEN LECH SONG SOT: chi ma co du lieu tu 3/1/2017 (bo tro) ===")
dau = ret_vn.index[0] if ret_vn.notna().any(axis=1).iloc[0] else ret_vn.index[1]
gia = sach.pivot(index="date", columns="ma", values="gia").sort_index()
S = gia.columns[gia.iloc[0].notna()]
ghi(f"So ma: {len(S)} / {gia.shape[1]}")
ds = du[du["id"].isin(S)].copy()
for c in ["M1", "Ridge", "LGBM"]:
    ds[f"r_{c}"] = ds.groupby("date")[c].rank(pct=True)
ds["Tron chinh"] = ds[["r_M1", "r_Ridge", "r_LGBM"]].mean(axis=1)
ics = ic_cot(ds, MO)
ghi("IC TB: " + ", ".join(f"{c} {ics[c].mean():.4f}" for c in MO))


def od(d, c):
    w = d.pivot(index="date", columns="id", values=c); return w.corrwith(w.shift(1), axis=1, method="spearman")


ghi(f"H2 tren tap song sot: (a) IC MC - M1 = {mt(ics['MC'] - ics['M1'])} | (b) on dinh MC - M1 = {mt(od(ds, 'MC') - od(ds, 'M1'))}")

# H3 tren tap song sot (cung logic script 40)
PHI, THUE, TRAN = 0.0025, 0.001, 0.067
r_s = ret_vn.loc["2019-01-01":"2024-12-31", list(S)]
hang = {m: ds.assign(h=ds.groupby("date")[m].rank(pct=True)).pivot(index="date", columns="id", values="h")
        for m in ["M1", "MC"]}
ngay = hang["MC"].index
ky = pd.Series(ngay, index=ngay).dt.to_period("W-FRI")
cuoi = pd.Series(ngay, index=ngay).groupby(ky).max().to_numpy()


def chay(mo_hinh, dai_tre):
    tb = hang[mo_hinh].groupby(ky.to_numpy()).mean(); w = pd.Series(dtype=float); out = []
    for k in range(len(cuoi) - 1):
        d, d2 = cuoi[k], cuoi[k + 1]
        sig = tb.loc[ky.loc[d]].dropna(); n = len(sig); rk = sig.rank(ascending=False)
        rt = r_s.loc[d] if d in r_s.index else pd.Series(dtype=float)
        tran, san = set(rt[rt >= TRAN].index), set(rt[rt <= -TRAN].index)
        tv, tr = set(rk[rk <= np.ceil(.2 * n)].index), set(rk[rk <= np.ceil(.5 * n)].index)
        cu = set(w.index); giu = (cu & (tr if dai_tre else tv)) | (cu & san); nam = sorted(giu | ((tv - cu) - tran))
        mtg = pd.Series(1 / len(nam), index=nam) if nam else pd.Series(dtype=float)
        tat = mtg.index.union(w.index); dw = mtg.reindex(tat, fill_value=0) - w.reindex(tat, fill_value=0)
        cp = dw.clip(lower=0).sum() * PHI + (-dw).clip(lower=0).sum() * (PHI + THUE)
        kp = r_s.loc[(r_s.index > d) & (r_s.index <= d2), nam].fillna(0)
        g = (1 + kp).prod() - 1 if len(kp) else pd.Series(0.0, index=nam)
        gop = float((mtg * g).sum()) if nam else 0.0
        out.append((1 - cp) * (1 + gop) - 1)
        w = mtg * (1 + g) / (1 + gop) if nam else pd.Series(dtype=float)
    return np.array(out)


def boot(d, B=10_000, L=4, seed=0):
    T = len(d); rng = np.random.default_rng(seed); st = rng.integers(T, size=(B, T))
    nw_ = rng.random((B, T)) < 1 / L; nw_[:, 0] = True
    vt = np.maximum.accumulate(np.where(nw_, np.arange(T), 0), axis=1)
    idx = (np.take_along_axis(st, vt, axis=1) + np.arange(T) - vt) % T
    return d.mean(), (np.abs((d - d.mean())[idx].mean(1)) >= abs(d.mean())).mean()


A, C, Dd = chay("M1", False), chay("MC", True), chay("M1", True)
m, p = boot(C - A); m2, p2 = boot(Dd - A); m3, p3 = boot(C - Dd)
ghi(f"H3 tren tap song sot: C - A = {m * 100:+.3f}%/tuan (p {p:.4f}) | D - A = {m2 * 100:+.3f}% (p {p2:.4f}) | "
    f"C - D = {m3 * 100:+.3f}% (p {p3:.4f})")

# ============ (3) H1 CSI 500 2022 - 4/2023 (da tiep xuc, mo ta)
ghi("\n=== (3) CSI 500 1/2022-4/2023: M1 vs MC, ca mat cat (DA TIEP XUC - chi mo ta) ===")
df = pd.read_parquet("../Data/csi500_dac_trung.parquet"); nen = pd.read_parquet("../Data/nen_csi500.parquet")
k1, n1, _ = F.chuan_bi(df, None, tu="2021-12-01", den="2023-04-28")
kM, nM, _ = F.chuan_bi(df, nen, tu="2021-12-01", den="2023-04-28")
m1, c1, _ = F.nap_mo_hinh(KQ / "tinh_chinh/m1_best.ckpt", device=dev)
mc, cm, _ = F.nap_mo_hinh(KQ / "tinh_chinh/mc_best.ckpt", device=dev)
ng = np.intersect1d(n1, nM); rows = []
for i in range(1, len(ng)):
    if ng[i] < np.datetime64("2022-01-01"):
        continue
    ch = np.intersect1d(F.ghep(k1, ng[i - 1], ng[i])[0], F.ghep(kM, ng[i - 1], ng[i])[0])
    so = int(pd.Timestamp(ng[i]).strftime("%Y%m%d"))
    a, p1, y = F.du_bao_ngay(m1, c1, k1, ng[i - 1], ng[i], K=1, rng=np.random.default_rng([so, 0]), device=dev, ids_chon=ch)
    _, p2_, _ = F.du_bao_ngay(mc, cm, kM, ng[i - 1], ng[i], K=20, rng=np.random.default_rng([so, 1]), device=dev, ids_chon=ch)
    ok = ~np.isnan(y)
    rows.append((ng[i], stats.spearmanr(p1[ok], y[ok])[0], stats.spearmanr(p2_[ok], y[ok])[0]))
e = pd.DataFrame(rows, columns=["date", "M1", "MC"]).set_index("date")
ghi(f"{len(e)} ngay | IC M1 {e['M1'].mean():.4f} | IC MC {e['MC'].mean():.4f} | MC - M1 = {mt(e['MC'] - e['M1'])}")

# ============ (4) loi suat vuot troi top 20% (gop) song song IC
ghi("\n=== (4) LOI SUAT VUOT TROI TOP 20% (gop, truoc phi, ngay ke tiep) ===")


def top20(d, cols):
    out = {}
    for c in cols:
        def f(g):
            g = g.dropna(subset=[c, "target"]); n = max(1, int(np.ceil(0.2 * len(g))))
            return g.nlargest(n, c)["target"].mean() - g["target"].mean()
        s = d.groupby("date")[["target", c]].apply(f); out[c] = s
    return pd.DataFrame(out)


for ten, dd, cols, icx in [("VN 2019-2024", du, MO, ic_vn),
                           ("CSI 500 5/2023-6/2026 (ca mat cat)",
                            pd.read_parquet(CS / "mat_cat.parquet").rename(columns={"m1": "M1", "mc": "MC"}), ["M1", "MC"], None)]:
    t2 = top20(dd, cols)
    icc = icx if icx is not None else ic_cot(dd, cols)
    ghi(f"{ten}:")
    for c in cols:
        x = t2[c].dropna()
        ghi(f"  {c:11} vuot troi {x.mean() * 100:+.3f}%/ngay (~{x.mean() * 252 * 100:+.1f}%/nam, t {x.mean() / nw(x):.2f}) | IC {icc[c].mean():.4f}")
ghi(f"\nLuu: {OUT}")
