"""Buoc 2.6 - Tai lap FinPFN tren toan bo 301 ngay test, cach chia giong file cong bo
(khong hoan lai, 500 ma, 10 nhom/ngay). Moc: IC ~0.0405, IR ~0.68 (so voi loi suat toan bo ma).
Chay tu D:\\LuanVan\\Code:  python 06_tai_lap_day_du.py   (~35-45 phut)
Bi ngat giua chung? Chay lai lenh cu: script tu bo qua cac ngay da xong.
"""
import os, time
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from finpfn_utils import CKPT, doc_test, du_bao_mot_ngay, ic_theo_ngay

OUT = "ket_qua_06_tai_lap.csv"
SEED = 42

test, feats, dates = doc_test()
xong = set(pd.read_csv(OUT, parse_dates=["date"])["date"]) if os.path.exists(OUT) else set()
reg = TabPFNRegressor(model_path=CKPT, device="cuda")

t0 = time.time()
for i, (d0, d1) in enumerate(zip(dates[:-1], dates[1:]), 1):
    if pd.Timestamp(d1) in xong:
        continue
    rng = np.random.default_rng([SEED, i])        # seed rieng cho tung ngay -> chay lai giua chung van y het
    p, n = du_bao_mot_ngay(reg, test, feats, d0, d1, rng, cach_chia="khong_hoan_lai")
    p.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(f"[{i}/{len(dates) - 1}] {pd.Timestamp(d1).date()}: {len(p)}/{n} ma | {(time.time() - t0) / 60:.1f} phut", end="\r")

# ---- Danh gia ----
ours = pd.read_csv(OUT, parse_dates=["date"]).merge(test[["date", "id", "target"]], on=["date", "id"])
pub = pd.read_csv("FinPFN/results/finpfn_perf_csi500.csv.gz", index_col=0, parse_dates=["date"])
pub = pub.groupby(["date", "id"], as_index=False).agg(pub_pred=("finpfn_median_pred", "mean"),
                                                      target=("return", "first"))

print("\n\n              So ngay  Ma/ngay   IC TB     IR")
for ten, d, col in [("Cua minh", ours, "pred"), ("Cong bo", pub, "pub_pred")]:
    ic = ic_theo_ngay(d, col, "target")
    print(f"{ten:12} {ic.size:8d} {len(d) / ic.size:8.0f}  {ic.mean():.4f}  {ic.mean() / ic.std():.3f}")

m = ours.merge(pub[["date", "id", "pub_pred"]], on=["date", "id"])
print(f"\nMa trung nhau/ngay: {len(m) / m['date'].nunique():.0f} | "
      f"tuong quan hang minh-cong bo: {ic_theo_ngay(m, 'pred', 'pub_pred').mean():.3f}")
