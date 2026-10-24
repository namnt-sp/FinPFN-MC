"""Buoc 3g - Tao san du bao ngoai mau cua Ridge va LGBM cho TOAN BO giai doan (dau vao cho FinPFN-MC).
KHONG tinh IC hay bat ky chi so nao -> khong "nhin" tap kiem tra. Lan mo chinh thuc van la Buoc 5-6.
Ra: ../Data/nen_csi500.parquet, nen_csi300.parquet, nen_vn100.parquet
Chay tu D:\\LuanVan\\Code:  python 29_tao_du_bao_nen.py   (~1,5-2 gio, CPU; chay song song voi script 28 duoc)
Bi ngat? Chay lai: thi truong da xong se duoc bo qua.
"""
import os, time
from pathlib import Path
import pandas as pd
import mo_hinh_nen
from dac_trung import TEN

mo_hinh_nen.LGBM_THAM_SO["n_jobs"] = max(1, (os.cpu_count() or 2) // 2)   # chua nua CPU cho script 28
print("LightGBM dung", mo_hinh_nen.LGBM_THAM_SO["n_jobs"], "luong")

VIEC = [  # (ten, file dac trung, du bao tu, du bao den)
    ("csi500", "../Data/csi500_dac_trung.parquet", "2017-01-01", "2026-06-30"),
    ("csi300", "../Data/csi300_dac_trung.parquet", "2017-01-01", "2026-06-30"),
    ("vn100", "../Data/vn100_dac_trung.parquet", "2018-01-01", "2024-12-31"),
]

for ten, f, tu, den in VIEC:
    out = Path(f"../Data/nen_{ten}.parquet")
    if out.exists():
        print(f"{ten}: da co, bo qua")
        continue
    t0 = time.time()
    df = pd.read_parquet(f)
    pred = mo_hinh_nen.du_bao_ngoai_mau(df, TEN, tu_ngay=tu, den_ngay=den)
    pred.to_parquet(out, index=False)
    print(f"{ten}: {len(pred):,} dong, {pred['date'].nunique()} ngay, "
          f"{pred['date'].min().date()} -> {pred['date'].max().date()} | {(time.time() - t0) / 60:.0f} phut")
print("Xong. (Khong tinh chi so nao tren tap kiem tra.)")
