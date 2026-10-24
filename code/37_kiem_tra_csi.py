"""Buoc 5 - TAP KIEM TRA CSI 5/2023-6/2026, chay DUNG MOT LAN voi cau hinh da khoa (H1, H1b + kiem tra do ben).
  python 37_kiem_tra_csi.py csi500      (chinh)        |   python 37_kiem_tra_csi.py csi300   (lap lai)
Thiet ke (theo cau_hinh_khoa.json): moi ngay 10 lan chia x cac vu tru 100 ma roi nhau (CSI 500: 5; CSI 300: 2).
  Trong moi vu tru: M1 (K=1), M1+K (K=20), FinPFN-MC (K=20); Ridge, LGBM; tron deu chinh (M1,R,L);
  tron phu (M1+K,R,L); tron IC 60 ngay (chi dung qua khu). MC K=50 (do ben) chi tren lan chia dau tien.
  Mat cat day du: M1 va MC K=20 (do on dinh, phan tich sau).
Kiem tra dau van tay SHA-256 truoc khi chay. Bi ngat? Chay lai dung lenh: bo qua cac ngay da xong.
"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import finpfn_mc as F

TT = sys.argv[1].lower() if len(sys.argv) > 1 else ""
if TT not in ("csi500", "csi300"):
    raise SystemExit("Cach dung: python 37_kiem_tra_csi.py csi500   (hoac csi300)")
KQ = Path("../Ketqua"); OUT = KQ / f"kiem_tra_{TT}"; OUT.mkdir(parents=True, exist_ok=True)
TU, DEN = np.datetime64("2023-05-01"), np.datetime64("2026-06-30")
SO_LAN, CO_VT, K, K_BEN = 10, 100, 20, 50
dev = "cuda" if torch.cuda.is_available() else "cpu"
f_log = open(OUT / "nhat_ky.txt", "a", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


# ---------------- 0) kiem tra cau hinh khoa
khoa = json.loads((KQ / "cau_hinh_khoa.json").read_text(encoding="utf-8"))
TEP = {"mo_hinh_M1": KQ / "tinh_chinh/m1_best.ckpt", "mo_hinh_FinPFN_MC": KQ / "tinh_chinh/mc_best.ckpt",
       f"dac_trung_{TT}": Path(f"../Data/{TT}_dac_trung.parquet"), f"nen_{TT}": Path(f"../Data/nen_{TT}.parquet"),
       "code_finpfn_mc": Path("finpfn_mc.py"), "code_dac_trung": Path("dac_trung.py"),
       "code_mo_hinh_nen": Path("mo_hinh_nen.py")}
for k, p in TEP.items():
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != khoa["dau_van_tay_sha256"][k]:
        raise SystemExit(f"DAU VAN TAY KHONG KHOP: {k} ({p}). Tep da bi thay doi sau khi khoa -> dung lai.")
ghi(f"=== {TT.upper()} | dau van tay khop cau hinh khoa ({khoa['thoi_diem_khoa']}) ===")

# ---------------- 1) du lieu, mo hinh
df = pd.read_parquet(f"../Data/{TT}_dac_trung.parquet")
nen = pd.read_parquet(f"../Data/nen_{TT}.parquet")
kho1, n1, _ = F.chuan_bi(df, None, tu="2023-03-01", den="2026-07-31")
khoM, nM, _ = F.chuan_bi(df, nen, tu="2023-03-01", den="2026-07-31")
ngay = np.intersect1d(n1, nM)
idx = [i for i in range(1, len(ngay)) if TU <= ngay[i] <= DEN]
nen_d = nen.set_index(["date", "id"])
m1, c1, _ = F.nap_mo_hinh(KQ / "tinh_chinh/m1_best.ckpt", device=dev)
mc, cm, _ = F.nap_mo_hinh(KQ / "tinh_chinh/mc_best.ckpt", device=dev)

# ---------------- 2) tiep tuc neu bi ngat
P_IC, P_MC = OUT / "ic_vu_tru.parquet", OUT / "mat_cat.parquet"
cu_ic = pd.read_parquet(P_IC) if P_IC.exists() else pd.DataFrame()
cu_mc = pd.read_parquet(P_MC) if P_MC.exists() else pd.DataFrame()
xong = set(pd.to_datetime(cu_ic["date"]).values) if len(cu_ic) else set()
ghi(f"{len(idx)} ngay kiem tra | da xong {len(xong)} | thiet bi {dev}")


def rk(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def ic_ngay_hist():
    if not len(cu_ic):
        return pd.DataFrame(columns=["M1", "Ridge", "LGBM"], index=pd.DatetimeIndex([]), dtype=float)
    return cu_ic[cu_ic["mo_hinh"].isin(["M1", "Ridge", "LGBM"])].groupby(["date", "mo_hinh"])["ic"].mean().unstack()


hist = ic_ngay_hist()
buf_ic, buf_mc, t0, dem = [], [], time.time(), 0
for i in idx:
    d0, d1 = ngay[i - 1], ngay[i]
    if d1 in xong:
        continue
    so = int(pd.Timestamp(d1).strftime("%Y%m%d"))
    seed = lambda *a: np.random.default_rng([so, *a])
    chung = np.intersect1d(F.ghep(kho1, d0, d1)[0], F.ghep(khoM, d0, d1)[0])

    # trong so tron IC 60 ngay: chi tu cac ngay truoc d1
    h = hist[hist.index < pd.Timestamp(d1)].tail(60)
    w = h[["M1", "Ridge", "LGBM"]].mean().clip(lower=0).fillna(0).to_numpy() if len(h) >= 20 else np.zeros(3)
    w = np.ones(3) / 3 if w.sum() <= 0 else w / w.sum()

    a, p1, y = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=1, rng=seed(900), device=dev, ids_chon=chung)
    b, pm, _ = F.du_bao_ngay(mc, cm, khoM, d0, d1, K=K, rng=seed(901), device=dev, ids_chon=chung)
    buf_mc.append(pd.DataFrame({"date": d1, "id": a, "m1": p1, "target": y}).merge(
        pd.DataFrame({"date": d1, "id": b, "mc": pm}), on=["date", "id"]))

    so_vt = len(chung) // CO_VT
    rows = []
    for lan in range(SO_LAN):
        perm = seed(lan).permutation(chung)
        for v in range(so_vt):
            u = perm[v * CO_VT:(v + 1) * CO_VT]
            r = {}
            ids_u, r["M1"], y_u = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=1, rng=seed(lan, v, 0), device=dev, ids_chon=u)
            _, r["M1+K"], _ = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=K, rng=seed(lan, v, 1), device=dev, ids_chon=u)
            idm, r["MC"], _ = F.du_bao_ngay(mc, cm, khoM, d0, d1, K=K, rng=seed(lan, v, 2), device=dev, ids_chon=u)
            assert (idm == ids_u).all()
            if lan == 0:
                _, r["MC_K50"], _ = F.du_bao_ngay(mc, cm, khoM, d0, d1, K=K_BEN, rng=seed(lan, v, 3), device=dev, ids_chon=u)
            nn_ = nen_d.loc[[(pd.Timestamp(d1), x) for x in ids_u]]
            r["Ridge"], r["LGBM"] = nn_["ridge"].to_numpy(), nn_["lgbm"].to_numpy()
            rm1, rr, rl = rk(r["M1"]), rk(r["Ridge"]), rk(r["LGBM"])
            r["Tron chinh"] = (rm1 + rr + rl) / 3
            r["Tron phu"] = (rk(r["M1+K"]) + rr + rl) / 3
            r["Tron IC 60n"] = w[0] * rm1 + w[1] * rr + w[2] * rl
            ok = ~np.isnan(y_u)
            for ten, pr in r.items():
                rows.append((pd.Timestamp(d1), lan, v, ten, stats.spearmanr(pr[ok], y_u[ok])[0]))
    day = pd.DataFrame(rows, columns=["date", "lan", "vt", "mo_hinh", "ic"])
    buf_ic.append(day)
    hist.loc[pd.Timestamp(d1)] = day[day["mo_hinh"].isin(["M1", "Ridge", "LGBM"])].groupby("mo_hinh")["ic"].mean()
    dem += 1
    if dem % 10 == 0 or i == idx[-1]:
        cu_ic = pd.concat([cu_ic] + buf_ic, ignore_index=True); cu_ic.to_parquet(P_IC, index=False)
        cu_mc = pd.concat([cu_mc] + buf_mc, ignore_index=True); cu_mc.to_parquet(P_MC, index=False)
        buf_ic, buf_mc = [], []
        con = (time.time() - t0) / dem * (len(idx) - len(xong) - dem) / 3600
        ghi(f"  {pd.Timestamp(d1).date()} | xong them {dem} ngay | con lai ~{con:.1f} gio")
if buf_ic:
    cu_ic = pd.concat([cu_ic] + buf_ic, ignore_index=True); cu_ic.to_parquet(P_IC, index=False)
    cu_mc = pd.concat([cu_mc] + buf_mc, ignore_index=True); cu_mc.to_parquet(P_MC, index=False)

# ---------------- 3) kiem dinh
def nw(x, lag=5):
    x = np.asarray(pd.Series(x).dropna()); x = x - x.mean(); T = len(x)
    v = x @ x / T + 2 * sum((1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / T for l in range(1, lag + 1))
    return np.sqrt(v / T)


ic = cu_ic.groupby(["date", "mo_hinh"])["ic"].mean().unstack().sort_index()
ic.to_csv(OUT / "ic_theo_ngay.csv")
so_vt_ngay = cu_ic[["date", "lan", "vt"]].drop_duplicates().groupby("date").size().mean()
ghi(f"\n{ic.shape[0]} ngay, {so_vt_ngay:.0f} vu tru/ngay (TB)")
ghi(f"{'Mo hinh':12} {'IC TB':>7} {'IR':>6} {'t(NW)':>6}")
for c in ["M1", "M1+K", "MC", "Ridge", "LGBM", "Tron chinh", "Tron phu", "Tron IC 60n"]:
    x = ic[c].dropna()
    ghi(f"{c:12} {x.mean():7.4f} {x.mean() / x.std():6.3f} {x.mean() / nw(x):6.2f}")

d = (ic["MC"] - ic["M1"]).dropna(); se = nw(d); t = d.mean() / se
p = 2 * (1 - stats.norm.cdf(abs(t)))
ghi(f"\nH1  MC - M1 = {d.mean():+.4f} | SE(NW,5) {se:.4f} | t {t:.2f} | p hai phia {p:.4g} | "
    f"{'UNG HO H1' if d.mean() > 0 and p < 0.05 else 'KHONG UNG HO H1'}")
for ten, cot in [("H1b", "Tron chinh"), ("Do ben", "Tron phu")]:
    d = (ic["MC"] - ic[cot]).dropna(); se = nw(d); lb = d.mean() - 1.645 * se
    ghi(f"{ten:4} MC - {cot} = {d.mean():+.4f} | can duoi KTC95% mot phia {lb:+.4f} | "
        f"{'DAT (> -0.005)' if lb > -0.005 else 'KHONG DAT'}")
sub = cu_ic[cu_ic["lan"] == 0].groupby(["date", "mo_hinh"])["ic"].mean().unstack()
d = (sub["MC_K50"] - sub["MC"]).dropna()
ghi(f"Do ben K=50 - K=20 (lan chia dau) = {d.mean():+.4f} (t {d.mean() / nw(d):.2f})")
nam = ic.groupby(ic.index.year)[["M1", "MC", "Tron chinh"]].mean()
ghi("Theo nam:\n" + nam.round(4).to_string())
od = {c: cu_mc.pivot(index="date", columns="id", values=c).pipe(
    lambda w_: w_.corrwith(w_.shift(1), axis=1, method="spearman").mean()) for c in ["m1", "mc"]}
ghi(f"Do on dinh (mat cat): M1 {od['m1']:.3f} | FinPFN-MC {od['mc']:.3f}")
ghi(f"Tong thoi gian lan chay nay: {(time.time() - t0) / 3600:.1f} gio")
