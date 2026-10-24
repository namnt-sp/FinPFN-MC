"""Buoc 4d-2 - Phep kiem gate: thanh tich tuong doi gan day cua Ridge vs LGBM co du bao duoc
mo hinh nao tot hon NGAY KE TIEP khong? (quyet dinh co tinh chinh MC-v2 voi cot dynamic_stack)

Thoi diem (as-of):  IC_s = Spearman(du bao_s, y_s), y_s = loi suat s->s+1, chi biet khi dong cua s+1.
                    gate_t = TB 20 ngay cua (IC_Ridge - IC_LGBM) tren cac ngay s <= t-1.
QUY TAC DAT TRUOC (CSI 500, 2017-2020): train MC-v2 neu
   [ b > 0 voi t(NW) >= 2  HOAC  chenh lech co dieu kien > 0 voi t(NW) >= 2 ]  VA  tuong quan > 0 o >= 3/4 nam.
Tham khao (khong dung ra quyet dinh): cua so 5/10/60; CSI 2021; VN 2018; IC cua dynamic_stack.
Chay tu D:\\LuanVan\\Code:  python 34_kiem_gate.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

OUT = Path("../Ketqua/kiem_gate"); OUT.mkdir(parents=True, exist_ok=True)
EPS, CUA_SO = 0.005, 20
f_log = open(OUT / "nhat_ky_kiem_gate.txt", "w", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


def nw_se(x, lag=5):
    x = np.asarray(x, float) - np.mean(x); T = len(x)
    v = x @ x / T + 2 * sum((1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / T for l in range(1, lag + 1))
    return np.sqrt(v / T)


def hoi_quy_nw(y, x, lag=5):
    """b va t(NW) cua y = a + b x."""
    xc = x - x.mean()
    b = (xc @ (y - y.mean())) / (xc @ xc)
    e = (y - y.mean()) - b * xc
    u = xc * e; T = len(u)
    S = u @ u / T + 2 * sum((1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / T for l in range(1, lag + 1))
    se = np.sqrt(S * T) / (xc @ xc)
    return b, b / se


def ic_ngay(f_dt, f_nen):
    dt = pd.read_parquet(f_dt, columns=["date", "id", "target"])
    nen = pd.read_parquet(f_nen)
    m = nen.merge(dt, on=["date", "id"]).dropna(subset=["target"])
    g = m.groupby("date")
    out = pd.DataFrame({
        "R": g[["ridge", "target"]].apply(lambda d: stats.spearmanr(d["ridge"], d["target"])[0]),
        "L": g[["lgbm", "target"]].apply(lambda d: stats.spearmanr(d["lgbm"], d["target"])[0]),
    }).sort_index()
    return m, out


def dynamic_ic(m, ic, w):
    """IC cua dynamic_stack (alpha_t tu IC den t-1) va cua tron deu Ridge+LGBM."""
    ic20 = ic.shift(1).rolling(w, min_periods=w).mean()
    pR, pL = ic20["R"].clip(lower=0), ic20["L"].clip(lower=0)
    alpha = ((pR + EPS) / (pR + pL + 2 * EPS)).rename("alpha")
    m = m.copy()
    m["rR"] = m.groupby("date")["ridge"].rank(pct=True)
    m["rL"] = m.groupby("date")["lgbm"].rank(pct=True)
    m = m.join(alpha, on="date").dropna(subset=["alpha"])
    m["dyn"] = m["alpha"] * m["rR"] + (1 - m["alpha"]) * m["rL"]
    m["deu"] = (m["rR"] + m["rL"]) / 2
    g = m.groupby("date")
    return (g[["dyn", "target"]].apply(lambda d: stats.spearmanr(d["dyn"], d["target"])[0]),
            g[["deu", "target"]].apply(lambda d: stats.spearmanr(d["deu"], d["target"])[0]), alpha)


def kiem(ic, tu, den, w, chi_tiet=False):
    d_ic = ic["R"] - ic["L"]
    gate = d_ic.shift(1).rolling(w, min_periods=w).mean()        # chi dung den t-1
    x = pd.DataFrame({"gate": gate, "d": d_ic}).loc[tu:den].dropna()
    b, tb = hoi_quy_nw(x["d"].to_numpy(), x["gate"].to_numpy())
    r = x["gate"].corr(x["d"])
    tren, duoi = x.loc[x["gate"] > 0, "d"], x.loc[x["gate"] <= 0, "d"]
    s = np.where(x["gate"] > 0, 1, -1) * (x["d"] - x["d"].mean())   # chenh lech co dieu kien (chuoi ngay)
    t_cd = s.mean() / nw_se(s)
    nam = x.groupby(x.index.year).apply(lambda g: g["gate"].corr(g["d"]))
    if chi_tiet:
        ghi(f"  So ngay: {len(x)} | IC TB Ridge {ic.loc[tu:den, 'R'].mean():.4f}, LGBM {ic.loc[tu:den, 'L'].mean():.4f}")
        ghi(f"  Hoi quy dIC_t ~ gate_t: b = {b:+.3f}, t(NW) = {tb:.2f} | tuong quan = {r:+.3f}")
        ghi(f"  gate>0: {len(tren)} ngay, dIC TB {tren.mean():+.4f} | gate<=0: {len(duoi)} ngay, dIC TB {duoi.mean():+.4f}"
            f" | t(NW) chenh lech co dieu kien = {t_cd:.2f}")
        ghi("  Tuong quan theo nam: " + ", ".join(f"{y}: {v:+.3f}" for y, v in nam.items()))
    return b, tb, r, t_cd, nam


# ================= CSI 500 =================
m_csi, ic_csi = ic_ngay("../Data/csi500_dac_trung.parquet", "../Data/nen_csi500.parquet")
ic_csi.to_csv(OUT / "ic_ngay_ridge_lgbm_csi500.csv")
ghi("=== CSI 500, 2017-2020 (RA QUYET DINH), cua so 20 ===")
b, tb, r, t_cd, nam = kiem(ic_csi, "2017-01-01", "2020-12-31", CUA_SO, chi_tiet=True)
dat = ((b > 0 and tb >= 2) or t_cd >= 2) and (nam > 0).sum() >= 3
ghi(f"\n  ==> QUY TAC: {'DAT -> nen train MC-v2' if dat else 'KHONG DAT -> khong train MC-v2'}")

ghi("\n--- Tham khao: cac cua so khac (CSI 2017-2020) ---")
for w in [5, 10, 60]:
    b_, tb_, r_, tc_, _ = kiem(ic_csi, "2017-01-01", "2020-12-31", w)
    ghi(f"  cua so {w:2d}: b {b_:+.3f} t {tb_:5.2f} | r {r_:+.3f} | t dieu kien {tc_:5.2f}")

ghi("\n--- Tham khao: IC cua dynamic_stack (cua so 20) so voi tron deu Ridge+LGBM, CSI 2017-2020 ---")
dyn, deu, alpha = dynamic_ic(m_csi, ic_csi, CUA_SO)
dd = (dyn - deu).loc["2017-01-01":"2020-12-31"].dropna()
ghi(f"  dynamic {dyn.loc['2017':'2020'].mean():.4f} | tron deu {deu.loc['2017':'2020'].mean():.4f} | "
    f"chenh {dd.mean():+.4f} (t {dd.mean() / nw_se(dd):.2f}) | alpha TB {alpha.loc['2017':'2020'].mean():.2f}, "
    f"do lech {alpha.loc['2017':'2020'].std():.2f}")

ghi("\n--- Tham khao: CSI 2021 (vong thu) ---")
kiem(ic_csi, "2021-01-01", "2021-12-31", CUA_SO, chi_tiet=True)

# ================= VN 2018 (giai doan phat trien) =================
ghi("\n--- Tham khao: VN 2018 (giai doan phat trien; KHONG cham 2019+) ---")
m_vn, ic_vn = ic_ngay("../Data/vn100_dac_trung.parquet", "../Data/nen_vn100.parquet")
ic_vn = ic_vn.loc[:"2018-12-31"]
kiem(ic_vn, "2018-01-01", "2018-12-31", CUA_SO, chi_tiet=True)
ghi(f"\nLuu: {OUT}")
