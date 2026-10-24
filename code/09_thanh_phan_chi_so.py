"""Buoc 3.2 - Lich su thanh phan CSI 500 va CSI 300 (anh chup dau moi thang, 7/2015 -> 10/2026).
Luu: ../Data/baostock/thanh_phan_csi500.csv, thanh_phan_csi300.csv
Chay tu D:\\LuanVan\\Code:  python 09_thanh_phan_chi_so.py   (~2-4 phut)
"""
from pathlib import Path
import baostock as bs
import pandas as pd

OUT = Path("../Data/baostock"); OUT.mkdir(parents=True, exist_ok=True)
THANG = pd.date_range("2015-07-01", "2026-10-01", freq="MS").strftime("%Y-%m-%d")


def bang(rs):
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


print("Dang nhap:", bs.login().error_msg)
for ten, ham in [("csi500", bs.query_zz500_stocks), ("csi300", bs.query_hs300_stocks)]:
    parts = []
    for d in THANG:
        t = bang(ham(date=d))
        t["ngay_chup"] = d
        parts.append(t)
    tp = pd.concat(parts, ignore_index=True)
    tp.to_csv(OUT / f"thanh_phan_{ten}.csv", index=False, encoding="utf-8-sig")

    dem = tp.groupby("ngay_chup")["code"].nunique()
    print(f"\n{ten.upper()}: {len(THANG)} anh chup | so ma/anh: min {dem.min()}, max {dem.max()}")
    print(f"  Anh chup rong: {[d for d in THANG if d not in dem.index] or 'khong co'}")
    print(f"  Cac lan cap nhat (updateDate): {tp['updateDate'].nunique()} lan | "
          f"tu {tp['updateDate'].min()} den {tp['updateDate'].max()}")
    print(f"  Tong so ma tung nam trong chi so: {tp['code'].nunique()}")
bs.logout()
