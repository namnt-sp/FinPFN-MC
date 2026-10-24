"""Buoc 4a - Chay thu tinh chinh tren GTX 1060: kiem tra code chay dung, do thoi gian va bo nho de lap
ngan sach GPU cho Kaggle. KHONG phai tinh chinh that (chi vai buoc).
Du lieu: huan luyen CSI 500 1/2017-12/2019; kiem dinh dung som 2020. Vong thu 2021 KHONG duoc dung.
Chay tu D:\\LuanVan\\Code:  python 31_thu_tinh_chinh.py   (~10-20 phut)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import finpfn_mc as F

KQ = Path("../Ketqua"); KQ.mkdir(exist_ok=True)
TABPFN = "FinPFN/models/tabpfn-v2-regressor.ckpt"
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_parquet("../Data/csi500_dac_trung.parquet")
nen = pd.read_parquet("../Data/nen_csi500.parquet")
log = open(KQ / "nhat_ky_31_thu_tinh_chinh.txt", "w", encoding="utf-8")


def ghi(s):
    print(s); log.write(s + "\n"); log.flush()


for ten, dung_nen, co_nhom in [("M1", False, (50, 50)), ("FinPFN-MC", True, (30, 60))]:
    ghi(f"\n===== {ten} =====")
    kho, ngay, feats = F.chuan_bi(df, nen if dung_nen else None, tu="2016-12-01", den="2020-12-31")
    hl = ngay[(ngay >= np.datetime64("2017-01-01")) & (ngay <= np.datetime64("2019-12-31"))]
    kd = ngay[np.searchsorted(ngay, np.datetime64("2020-01-01")) - 1:]   # ngay cuoi 2019 chi lam ngu canh
    ghi(f"So cot: {len(feats)} | ngay huan luyen: {len(hl)} | ngay kiem dinh: {len(kd)}")

    model, crit, cfg = F.nap_mo_hinh(TABPFN, device=dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t = time.time()
    v0 = F.danh_gia(model, crit, kho, kd[-21:], device=dev)        # 20 ngay de do thoi gian
    t_kd = (time.time() - t) / 20
    ghi(f"Khoi tao TabPFN, 20 ngay 2020: val loss {v0['loss']:.4f} | IC {v0['ic']:.4f} | {t_kd:.2f} giay/ngay")

    for b, acc in [(64, 2), (64, 4), (64, 8)]:                       # batch 64; tang tich luy neu het bo nho
        try:
            t = time.time()
            F.tinh_chinh(model, crit, cfg, kho, hl, kho, kd[-3:], save_path=KQ / "tam.ckpt",
                         co_nhom=co_nhom, batch=b, accum=acc, buoc_moi_epoch=3, max_epoch=1,
                         device=dev, nhat_ky=lambda s: None)
            t_buoc = (time.time() - t - 2 * t_kd * 2) / 3
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
            ghi(f"batch {b}, tich luy {acc}: ~{t_buoc:.1f} giay/buoc | GPU toi da {mem:.2f} GB")
            ep_gio = (100 * t_buoc + 243 * t_kd) / 3600
            ghi(f"  -> uoc tinh 1 epoch (100 buoc + kiem dinh ca 2020): {ep_gio:.2f} gio tren GTX 1060")
            break
        except torch.cuda.OutOfMemoryError:
            ghi(f"batch {b}, tich luy {acc}: HET BO NHO GPU -> thu tich luy lon hon")
            torch.cuda.empty_cache()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

(KQ / "tam.ckpt").unlink(missing_ok=True)
ghi("\nXong buoc 4a. Gui toan bo noi dung tren (cung luu o Ketqua/nhat_ky_31_thu_tinh_chinh.txt).")
