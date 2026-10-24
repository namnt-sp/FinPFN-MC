"""Buoc 4b/4c - Tinh chinh that M1 hoac FinPFN-MC tren CSI 500. Cung quy trinh, cung ngan sach cho ca hai.
  Huan luyen: 1/2017-12/2019 | Dung som theo loss kiem dinh: 2020 | Vong thu 2021: KHONG dung.
  Toi da 100 epoch x 100 buoc, batch 64, lr 1e-5, kien nhan 20 epoch, seed 0 (nhu Wang & Lera).
Chay:  python 32_tinh_chinh.py m1 | mc | mc2 | m1v
Bi ngat (tat may, Ctrl+C)? Chay lai dung lenh cu: tu tiep tuc tu epoch da xong.
Ra (trong ../Ketqua/tinh_chinh/): <ten>_best.ckpt, <ten>_lich_su.csv, <ten>_nhat_ky.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import finpfn_mc as F

TEN = sys.argv[1].lower() if len(sys.argv) > 1 else ""
CAU_HINH = {"m1": dict(nen=False, co_nhom=(50, 50)), "mc": dict(nen=True, co_nhom=(30, 60)),
            "mc2": dict(nen=True, co_nhom=(30, 60), dong=True),   # MC-v2: + cot dynamic_stack 20 ngay
            "m1v": dict(nen=False, co_nhom=(30, 60))}             # bac thang loai tru: M1 + co nhom thay doi
if TEN not in CAU_HINH:
    raise SystemExit("Cach dung: python 32_tinh_chinh.py m1   (hoac mc, mc2, m1v)")
ch = CAU_HINH[TEN]

OUT = Path("../Ketqua/tinh_chinh"); OUT.mkdir(parents=True, exist_ok=True)
f_log = open(OUT / f"{TEN}_nhat_ky.txt", "a", encoding="utf-8")


def ghi(s):
    print(s); f_log.write(s + "\n"); f_log.flush()


df = pd.read_parquet("../Data/csi500_dac_trung.parquet")
nen = pd.read_parquet("../Data/nen_csi500.parquet") if ch["nen"] else None
them = {"dong": True} if ch.get("dong") else {}             # chi MC-v2 can; ban khoa cua finpfn_mc khong co tham so nay
kho, ngay, feats = F.chuan_bi(df, nen, tu="2016-12-01", den="2020-12-31", **them)
hl = ngay[(ngay >= np.datetime64("2017-01-01")) & (ngay <= np.datetime64("2019-12-31"))]
kd = ngay[np.searchsorted(ngay, np.datetime64("2020-01-01")) - 1:]
ghi(f"== {TEN.upper()} | cot: {feats} | co nhom {ch['co_nhom']} | {len(hl)} ngay HL, {len(kd) - 1} ngay KD ==")

model, crit, cfg = F.nap_mo_hinh("FinPFN/models/tabpfn-v2-regressor.ckpt", seed=0, device="cuda")
lich_su, best = F.tinh_chinh(
    model, crit, cfg, kho, hl, kho, kd,
    save_path=OUT / f"{TEN}_best.ckpt", co_nhom=ch["co_nhom"],
    lr=1e-5, batch=64, accum=1, buoc_moi_epoch=100, max_epoch=100, kien_nhan=20,
    seed=0, device="cuda", nhat_ky=ghi, tiep_tuc=OUT / f"{TEN}_tiep_tuc.pt")
lich_su.to_csv(OUT / f"{TEN}_lich_su.csv", index=False)
ghi(f"XONG {TEN.upper()}: best val loss {best['loss']:.5f} | val IC 2020 {best['ic']:.4f} | "
    f"{len(lich_su)} epoch | luu {TEN}_best.ckpt")
