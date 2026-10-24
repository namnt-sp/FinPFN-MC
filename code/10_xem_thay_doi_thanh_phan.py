"""Buoc 3.3 - Thanh phan chi so thay doi vao nhung thang nao? (chay tren may, khong goi mang)
Chay tu D:\\LuanVan\\Code:  python 10_xem_thay_doi_thanh_phan.py
"""
import pandas as pd

for ten in ["csi500", "csi300"]:
    tp = pd.read_csv(f"../Data/baostock/thanh_phan_{ten}.csv")
    ds = tp.groupby("ngay_chup")["code"].apply(set).sort_index()

    bien = pd.DataFrame({
        "vao": [len(b - a) for a, b in zip(ds[:-1], ds[1:])],
        "ra":  [len(a - b) for a, b in zip(ds[:-1], ds[1:])],
    }, index=ds.index[1:])
    bien["thang"] = pd.to_datetime(bien.index).month

    print(f"\n=== {ten.upper()} ===")
    print("Tong so ma vao, cong theo thang chup (ca giai doan):")
    print(bien.groupby("thang")["vao"].sum().to_string())
    print("\nCac lan chup co >= 10 ma vao:")
    print(bien[bien["vao"] >= 10][["vao", "ra"]].to_string())
    print(f"\nThang co thay doi le (1-9 ma): {(bien['vao'].between(1, 9)).sum()} | "
          f"thang khong doi: {(bien['vao'] == 0).sum()}")
