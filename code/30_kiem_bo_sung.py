"""Buoc 3h - FinPFN ngu canh lon co bo sung thong tin cho Ridge + LGBM khong? (CSI 2020, giai doan huan luyen)
Dung lai ket qua script 28, chay vai giay.
Chay tu D:\\LuanVan\\Code:  python 30_kiem_bo_sung.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

KQ = Path("../Ketqua")
m = pd.read_csv(KQ / "ket_qua_28_ngu_canh_lon_csi2020.csv", parse_dates=["date"])
BT = ["finpfn_L1", "finpfn_L5", "tabpfn_L10"]

cot = ["tron_nen_2"]
for b in BT:                                     # tron deu 3 thanh phan: Ridge, LGBM, bien the FinPFN
    m[f"tron3_{b}"] = m[["ridge", "lgbm", b]].mean(axis=1)
    cot.append(f"tron3_{b}")


def ic(c):
    return m.groupby("date")[[c, "target"]].apply(lambda g: stats.spearmanr(g[c], g["target"])[0])


def on_dinh(c):
    w = m.pivot(index="date", columns="id", values=c)
    return w.corrwith(w.shift(1), axis=1, method="spearman").mean()


goc = ic("tron_nen_2")
print(f"{m['date'].nunique()} ngay\n{'Phep tron':20} {'IC TB':>7} {'IR':>6} {'On dinh':>8} {'- tron_nen_2':>13} {'t':>6}")
for c in cot:
    x = ic(c)
    if c == "tron_nen_2":
        print(f"{c:20} {x.mean():7.4f} {x.mean() / x.std():6.3f} {on_dinh(c):8.3f}")
        continue
    d = x - goc
    print(f"{c:20} {x.mean():7.4f} {x.mean() / x.std():6.3f} {on_dinh(c):8.3f} {d.mean():+13.4f} "
          f"{d.mean() / d.std() * np.sqrt(len(d)):6.2f}")
