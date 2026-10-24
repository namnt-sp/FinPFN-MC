"""Buoc 3e - Thu y tuong xep chong (chua tinh chinh): FinPFN trong so tac gia + 2 cot du bao Ridge, LGBM
trong ngu canh. CSI 500 nam 2020 (giai doan huan luyen). Can file ../Ketqua/ket_qua_26_*.
Chay tu D:\\LuanVan\\Code:  python 27_thu_xep_chong.py   (~22 phut)
"""
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tabpfn import TabPFNRegressor
import finpfn_utils
from dac_trung import TEN
from mo_hinh_nen import chuan_hoa_ngay
from finpfn_utils import CKPT, du_bao_mot_ngay
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

KQ = Path("../Ketqua")
df = pd.read_parquet("../Data/csi500_dac_trung.parquet")
df = df[(df["date"] >= "2019-10-01") & (df["date"] <= "2020-12-31")]

nen = pd.read_parquet(KQ / "ket_qua_26_nen_2020.parquet")
for c in ["ridge", "lgbm"]:                                  # hang trong ngay, thang [-0.5, 0.5]
    nen[c] = nen.groupby("date")[c].rank(pct=True) - 0.5

z = chuan_hoa_ngay(df, TEN).drop(columns="y").merge(df[["date", "id", "target"]], on=["date", "id"])
z = z.merge(nen, on=["date", "id"])                           # chi giu ngay co du bao nen (2020)
FEATS = TEN + ["ridge", "lgbm"]

ngay = np.sort(z["date"].unique())
reg = TabPFNRegressor(model_path=CKPT, device="cuda")
parts, t0 = [], time.time()
for i in range(1, len(ngay)):
    p, _ = du_bao_mot_ngay(reg, z, FEATS, ngay[i - 1], ngay[i], np.random.default_rng([7, i]), "khong_hoan_lai")
    parts.append(p)
    print(f"  FinPFN+xep chong {str(ngay[i])[:10]} | {(time.time() - t0) / 60:.1f} phut", end="\r")
print(f"\nSo nhom dung phuong an du phong: {finpfn_utils.SO_LAN_DU_PHONG}")

xc = pd.concat(parts).rename(columns={"pred": "finpfn_xc"})
cu = pd.read_csv(KQ / "ket_qua_26_P1_csi2020.csv", parse_dates=["date"])
m = cu.merge(xc, on=["date", "id"])
m["finpfn_xc"] = m.groupby("date")["finpfn_xc"].rank(pct=True)
m.to_csv(KQ / "ket_qua_27_xep_chong_csi2020.csv", index=False)


def ic(a, b):
    return m.groupby("date")[[a, b]].apply(lambda g: stats.spearmanr(g[a], g[b])[0])


def on_dinh(c):
    w = m.pivot(index="date", columns="id", values=c)
    return w.corrwith(w.shift(1), axis=1, method="spearman").mean()


print(f"\nCSI 500 nam 2020: {m['date'].nunique()} ngay (bo ngay dau), TB {m.groupby('date').size().mean():.0f} ma/ngay")
print(f"{'Mo hinh':11} {'IC TB':>7} {'IR':>6} {'On dinh':>8}")
for c in ["finpfn", "finpfn_xc", "ridge", "lgbm", "tron_nen_2", "tron_deu_3"]:
    x = ic(c, "target")
    print(f"{c:11} {x.mean():7.4f} {x.mean() / x.std():6.3f} {on_dinh(c):8.3f}")
d = ic("finpfn_xc", "target") - ic("tron_nen_2", "target")
print(f"\nfinpfn_xc - tron_nen_2: {d.mean():+.4f} (t = {d.mean() / d.std() * np.sqrt(len(d)):.2f}, chua hieu chinh tu tuong quan)")
print(f"Tuong quan hang finpfn_xc voi: Ridge {ic('finpfn_xc', 'ridge').mean():.3f} | LGBM {ic('finpfn_xc', 'lgbm').mean():.3f}"
      f" | tron_nen_2 {ic('finpfn_xc', 'tron_nen_2').mean():.3f}")
