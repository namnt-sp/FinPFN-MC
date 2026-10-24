"""Buoc 2.5 - Du bao FinPFN phu thuoc cach chia nhom den muc nao? (kiem chung so bo cho P3)
Cung mo hinh, cung ngay, chi doi hat giong chia nhom -> so sanh hai lan chay.
Chay tu D:\\LuanVan\\Code:  python 05_do_nhay_chia_nhom.py   (~3-4 phut)
"""
import time
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from finpfn_utils import CKPT, doc_test, du_bao_mot_ngay, ic_theo_ngay

N_NGAY = 10
test, feats, dates = doc_test()
dates = dates[: N_NGAY + 1]
reg = TabPFNRegressor(model_path=CKPT, device="cuda")

lan = {}
for seed in (1, 2):
    rng = np.random.default_rng(seed)
    parts, phu = [], []
    t = time.time()
    for d0, d1 in zip(dates[:-1], dates[1:]):
        p, n = du_bao_mot_ngay(reg, test, feats, d0, d1, rng)
        parts.append(p); phu.append(len(p) / n)
    lan[seed] = pd.concat(parts)
    print(f"Seed {seed}: {time.time() - t:.0f}s | ty le ma co du bao/ngay: {np.mean(phu):.1%}")

m = lan[1].merge(lan[2], on=["date", "id"], suffixes=("_1", "_2"))
m = m.merge(test[["date", "id", "target"]], on=["date", "id"])
print(f"Ma co du bao o ca hai lan: {len(m) / N_NGAY:.0f} ma/ngay")
print(f"Tuong quan hang giua 2 lan chay (TB theo ngay): {ic_theo_ngay(m, 'pred_1', 'pred_2').mean():.3f}")
print(f"IC lan 1: {ic_theo_ngay(m, 'pred_1', 'target').mean():.4f} | "
      f"IC lan 2: {ic_theo_ngay(m, 'pred_2', 'target').mean():.4f}")
m["pred_tb"] = (m["pred_1"] + m["pred_2"]) / 2
print(f"IC trung binh 2 lan: {ic_theo_ngay(m, 'pred_tb', 'target').mean():.4f}")
