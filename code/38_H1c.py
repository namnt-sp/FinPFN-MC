"""Buoc 5 (tiep) - H1c (gia thuyet co che, dat truoc trong de cuong):
  Loi the IC ngay (FinPFN-MC - Tron deu chinh) lon hon vao nhung ngay cac mo hinh nen BAT DONG.
  Chi so bat dong D_t = | TB 20 ngay cua (IC_Ridge - IC_LGBM) |, chi dung cac ngay s <= t-1
  (IC tinh tren ca mat cat; IC_s biet khi dong cua s+1).
  Hoi quy: (MC - Tron chinh)_t = a + b*D_t ; UNG HO neu b > 0 va p < 0,05 (Newey-West do tre 5).
Chi doc ket qua da luu cua script 37 + du lieu; khong chay lai mo hinh.
Chay tu D:\\LuanVan\\Code:  python 38_H1c.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
KQ = Path("../Ketqua")


def hoi_quy_nw(y, x, lag=5):
    xc = x - x.mean(); yc = y - y.mean()
    b = (xc @ yc) / (xc @ xc); e = yc - b * xc; u = xc * e; T = len(u)
    S = u @ u / T + 2 * sum((1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / T for l in range(1, lag + 1))
    se = np.sqrt(S * T) / (xc @ xc)
    return b, se


def ic_ca_mat_cat(tt):
    dt = pd.read_parquet(f"../Data/{tt}_dac_trung.parquet", columns=["date", "id", "target"])
    m = pd.read_parquet(f"../Data/nen_{tt}.parquet").merge(dt, on=["date", "id"]).dropna(subset=["target"])
    g = m.groupby("date")
    ic = pd.DataFrame({"R": g[["ridge", "target"]].apply(lambda d: stats.spearmanr(d["ridge"], d["target"])[0]),
                       "L": g[["lgbm", "target"]].apply(lambda d: stats.spearmanr(d["lgbm"], d["target"])[0])})
    return ic.reindex(np.sort(m["date"].unique()))


f_log = open(KQ / "ket_qua_H1c.txt", "w", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


for tt in ["csi500", "csi300"]:
    p = KQ / f"kiem_tra_{tt}/ic_theo_ngay.csv"
    if not p.exists():
        ghi(f"{tt}: chua co ket qua script 37, bo qua"); continue
    ic_kt = pd.read_csv(p, index_col=0, parse_dates=True)
    icf = ic_ca_mat_cat(tt)
    D = (icf["R"] - icf["L"]).shift(1).rolling(20, min_periods=20).mean().abs().rename("D")
    x = ic_kt.join(D, how="inner").dropna(subset=["D", "MC", "Tron chinh"])
    ghi(f"=== {tt.upper()} | {len(x)} ngay | D: TB {x['D'].mean():.4f}, trung vi {x['D'].median():.4f} ===")
    for ten, cot in [("H1c (chinh)", "Tron chinh"), ("Do ben", "Tron phu")]:
        y = (x["MC"] - x[cot]).to_numpy()
        b, se = hoi_quy_nw(y, x["D"].to_numpy())
        pv = 2 * (1 - stats.norm.cdf(abs(b / se)))
        ket = "UNG HO H1c" if (b > 0 and pv < 0.05) else "KHONG UNG HO H1c"
        ghi(f"{ten:12} MC - {cot}: b = {b:+.4f} | SE {se:.4f} | t {b / se:.2f} | p hai phia {pv:.4f} | {ket}")
    x["nhom_D"] = pd.qcut(x["D"], 3, labels=["D thap", "D giua", "D cao"])
    bang = x.assign(chenh=x["MC"] - x["Tron chinh"]).groupby("nhom_D", observed=True)["chenh"].agg(["mean", "count"])
    ghi("Chenh lech MC - Tron chinh theo tam phan vi D (mo ta):\n" + bang.round(4).to_string() + "\n")
ghi(f"Luu: {KQ / 'ket_qua_H1c.txt'}")
