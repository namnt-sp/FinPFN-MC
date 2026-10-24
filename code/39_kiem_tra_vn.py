"""Buoc 6a - TAP KIEM TRA VN-100 2019-2024, chay DUNG MOT LAN voi cau hinh da khoa.
FinPFN-MC tinh chinh tren CSI, KHONG tinh chinh tren VN; thong tin VN vao qua ngu canh va mo hinh nen VN.
H2 (de cuong): MC tot hon M1 ve (a) rank IC VA (b) do on dinh xep hang ngay-ngay; ca hai > 0, p < 0,05 hai phia,
   Newey-West do tre 5. Do on dinh = Spearman giua hai ngay lien tiep tren ma co mat ca hai ngay.
H1c lap lai: (MC - Tron chinh) theo D_t = |TB 20 ngay (IC_Ridge - IC_LGBM)|, chi qua khu.
Luu du bao hang ngay -> dung cho H3 (script 40).
Chay tu D:\\LuanVan\\Code:  python 39_kiem_tra_vn.py   (~10-30 phut)
"""
import json, hashlib, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import finpfn_mc as F
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

KQ = Path("../Ketqua"); OUT = KQ / "kiem_tra_vn100"; OUT.mkdir(parents=True, exist_ok=True)
TU, DEN, K = np.datetime64("2019-01-01"), np.datetime64("2024-12-31"), 20
dev = "cuda" if torch.cuda.is_available() else "cpu"
f_log = open(OUT / "nhat_ky.txt", "a", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


khoa = json.loads((KQ / "cau_hinh_khoa.json").read_text(encoding="utf-8"))
TEP = {"mo_hinh_M1": KQ / "tinh_chinh/m1_best.ckpt", "mo_hinh_FinPFN_MC": KQ / "tinh_chinh/mc_best.ckpt",
       "dac_trung_vn100": Path("../Data/vn100_dac_trung.parquet"), "nen_vn100": Path("../Data/nen_vn100.parquet"),
       "code_finpfn_mc": Path("finpfn_mc.py"), "code_dac_trung": Path("dac_trung.py"),
       "code_mo_hinh_nen": Path("mo_hinh_nen.py")}
for k, p in TEP.items():
    if hashlib.sha256(p.read_bytes()).hexdigest() != khoa["dau_van_tay_sha256"][k]:
        raise SystemExit(f"DAU VAN TAY KHONG KHOP: {k}. Dung lai.")
ghi(f"=== VN-100 | dau van tay khop cau hinh khoa ({khoa['thoi_diem_khoa']}) ===")

df = pd.read_parquet("../Data/vn100_dac_trung.parquet")
nen = pd.read_parquet("../Data/nen_vn100.parquet")
kho1, n1, _ = F.chuan_bi(df, None, tu="2018-11-01", den="2024-12-31")
khoM, nM, _ = F.chuan_bi(df, nen, tu="2018-11-01", den="2024-12-31")
ngay = np.intersect1d(n1, nM)
idx = [i for i in range(1, len(ngay)) if TU <= ngay[i] <= DEN]
m1, c1, _ = F.nap_mo_hinh(KQ / "tinh_chinh/m1_best.ckpt", device=dev)
mc, cm, _ = F.nap_mo_hinh(KQ / "tinh_chinh/mc_best.ckpt", device=dev)
ghi(f"{len(idx)} ngay | thiet bi {dev}")

# ---------------- du bao hang ngay tren ca mat cat VN
P = OUT / "du_bao_ngay.parquet"
if P.exists():
    du = pd.read_parquet(P); ghi("Dung lai du bao da luu.")
else:
    rows, t0 = [], time.time()
    for n_, i in enumerate(idx, 1):
        d0, d1 = ngay[i - 1], ngay[i]
        chung = np.intersect1d(F.ghep(kho1, d0, d1)[0], F.ghep(khoM, d0, d1)[0])
        so = int(pd.Timestamp(d1).strftime("%Y%m%d"))
        sd = lambda *a: np.random.default_rng([so, *a])
        a, p_m1, y = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=1, rng=sd(0), device=dev, ids_chon=chung)
        _, p_m1k, _ = F.du_bao_ngay(m1, c1, kho1, d0, d1, K=K, rng=sd(1), device=dev, ids_chon=chung)
        b, p_mc, _ = F.du_bao_ngay(mc, cm, khoM, d0, d1, K=K, rng=sd(2), device=dev, ids_chon=chung)
        _, p_mc1, _ = F.du_bao_ngay(mc, cm, khoM, d0, d1, K=1, rng=sd(3), device=dev, ids_chon=chung)
        assert (a == b).all()
        rows.append(pd.DataFrame({"date": pd.Timestamp(d1), "id": a, "M1": p_m1, "M1+K": p_m1k,
                                  "MC": p_mc, "MC_K1": p_mc1, "target": y}))
        if n_ % 100 == 0:
            ghi(f"  {n_}/{len(idx)} ngay | {(time.time() - t0) / 60:.1f} phut")
    du = pd.concat(rows, ignore_index=True)
    du = du.merge(nen, on=["date", "id"], how="left").rename(columns={"ridge": "Ridge", "lgbm": "LGBM"})
    du.to_parquet(P, index=False)

# ---------------- IC, tron, do on dinh
g = du.groupby("date")
R = {c: g[c].rank(pct=True) for c in ["M1", "M1+K", "Ridge", "LGBM"]}
du["Tron chinh"] = (R["M1"] + R["Ridge"] + R["LGBM"]) / 3
du["Tron phu"] = (R["M1+K"] + R["Ridge"] + R["LGBM"]) / 3
MO = ["M1", "M1+K", "MC_K1", "MC", "Ridge", "LGBM", "Tron chinh", "Tron phu"]
ok = du.dropna(subset=["target"])
ic = pd.DataFrame({c: ok.groupby("date")[[c, "target"]].apply(lambda d: stats.spearmanr(d[c], d["target"])[0])
                   for c in MO})
# tron IC 60 ngay (trong so tu cac ngay truoc)
w = ic[["M1", "Ridge", "LGBM"]].shift(1).rolling(60, min_periods=20).mean().clip(lower=0)
w = w.div(w.sum(axis=1), axis=0).where(w.sum(axis=1) > 0, 1 / 3).fillna(1 / 3)
du = du.join(w.add_prefix("w_"), on="date")
du["Tron IC 60n"] = du["w_M1"] * R["M1"] + du["w_Ridge"] * R["Ridge"] + du["w_LGBM"] * R["LGBM"]
ok = du.dropna(subset=["target"])
ic["Tron IC 60n"] = ok.groupby("date")[["Tron IC 60n", "target"]].apply(
    lambda d: stats.spearmanr(d["Tron IC 60n"], d["target"])[0])
ic.to_csv(OUT / "ic_theo_ngay.csv")


def od_ngay(c):
    wd = du.pivot(index="date", columns="id", values=c)
    return wd.corrwith(wd.shift(1), axis=1, method="spearman")


od = pd.DataFrame({c: od_ngay(c) for c in ["M1", "MC", "Ridge", "LGBM", "Tron chinh"]})
od.to_csv(OUT / "on_dinh_theo_ngay.csv")


def nw(x, lag=5):
    x = np.asarray(pd.Series(x).dropna()); x = x - x.mean(); T = len(x)
    v = x @ x / T + 2 * sum((1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / T for l in range(1, lag + 1))
    return np.sqrt(v / T)


ghi(f"\n{ic.shape[0]} ngay, TB {du.groupby('date').size().mean():.0f} ma/ngay")
ghi(f"{'Mo hinh':12} {'IC TB':>7} {'IR':>6} {'t(NW)':>6}")
for c in MO + ["Tron IC 60n"]:
    x = ic[c].dropna(); ghi(f"{c:12} {x.mean():7.4f} {x.mean() / x.std():6.3f} {x.mean() / nw(x):6.2f}")
ghi("Do on dinh TB: " + ", ".join(f"{c} {od[c].mean():.3f}" for c in od))

# ---------------- H2
ghi("\n=== H2 (giao-hop: ca (a) va (b) phai dat) ===")
kq = []
for ten, s in [("(a) IC", ic["MC"] - ic["M1"]), ("(b) On dinh", od["MC"] - od["M1"])]:
    s = s.dropna(); se = nw(s); t = s.mean() / se; p = 2 * (1 - stats.norm.cdf(abs(t)))
    dat = s.mean() > 0 and p < 0.05; kq.append(dat)
    ghi(f"{ten:12} MC - M1 = {s.mean():+.4f} | SE {se:.4f} | t {t:.2f} | p {p:.4g} | {'DAT' if dat else 'KHONG DAT'}")
ghi(f"==> H2: {'UNG HO' if all(kq) else 'KHONG UNG HO'}")
for ten, cot in [("MC - Tron chinh", "Tron chinh"), ("MC - Tron phu", "Tron phu")]:
    s = (ic["MC"] - ic[cot]).dropna(); se = nw(s)
    ghi(f"Mo ta: {ten} = {s.mean():+.4f} | can duoi KTC95% mot phia {s.mean() - 1.645 * se:+.4f}")

# ---------------- H1c lap lai tren VN
ghi("\n=== H1c lap lai (VN) ===")
mn = nen.merge(df[["date", "id", "target"]], on=["date", "id"]).dropna(subset=["target"])
gg = mn.groupby("date")
icf = pd.DataFrame({"R": gg[["ridge", "target"]].apply(lambda d: stats.spearmanr(d["ridge"], d["target"])[0]),
                    "L": gg[["lgbm", "target"]].apply(lambda d: stats.spearmanr(d["lgbm"], d["target"])[0])})
icf = icf.reindex(np.sort(mn["date"].unique()))
D = (icf["R"] - icf["L"]).shift(1).rolling(20, min_periods=20).mean().abs().rename("D")
x = ic.join(D, how="inner").dropna(subset=["D", "MC", "Tron chinh"])
y = (x["MC"] - x["Tron chinh"]).to_numpy(); xc = x["D"].to_numpy() - x["D"].mean(); yc = y - y.mean()
b = (xc @ yc) / (xc @ xc); u = xc * (yc - b * xc); T = len(u)
S = u @ u / T + 2 * sum((1 - l / 6) * (u[l:] @ u[:-l]) / T for l in range(1, 6))
se = np.sqrt(S * T) / (xc @ xc); p = 2 * (1 - stats.norm.cdf(abs(b / se)))
ghi(f"b = {b:+.4f} | t {b / se:.2f} | p {p:.4f} | {'UNG HO H1c' if b > 0 and p < 0.05 else 'KHONG UNG HO H1c'}")

ghi("\nTheo nam (IC TB):\n" + ic.groupby(ic.index.year)[["M1", "MC", "Tron chinh"]].mean().round(4).to_string())
ghi("Theo nam (on dinh):\n" + od.groupby(od.index.year)[["M1", "MC"]].mean().round(3).to_string())
