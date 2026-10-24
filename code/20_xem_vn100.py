"""Buoc 1.1 - Xem cau truc file VN-100 goc (Nguyen, 2025, Mendeley). Chua sua gi.
Tu tim moi file csv/xlsx trong ../Data (bo qua thu muc baostock).
Chay tu D:\\LuanVan\\Code:  python 20_xem_vn100.py
"""
from pathlib import Path
import pandas as pd

files = [f for f in Path("../Data").rglob("*")
         if f.suffix.lower() in (".csv", ".xlsx", ".xls") and "baostock" not in f.parts]
print("File tim thay:", [str(f) for f in files] or "KHONG CO")

for f in files:
    print(f"\n========== {f} ({f.stat().st_size / 1e6:.1f} MB) ==========")
    if f.suffix.lower() == ".csv":
        df = pd.read_csv(f)
    else:
        xl = pd.ExcelFile(f)
        print("Cac sheet:", xl.sheet_names)
        df = xl.parse(xl.sheet_names[0])
    print("Kich thuoc:", df.shape)
    print("10 cot dau:", list(df.columns[:10]))
    print("\n5 dong dau (toi da 8 cot):")
    print(df.iloc[:5, :8].to_string())
    print("\n3 dong cuoi (toi da 8 cot):")
    print(df.iloc[-3:, :8].to_string())
    print("\nKieu du lieu (dem):", df.dtypes.astype(str).value_counts().to_dict())
    print("Ty le o thieu toan bang:", round(df.isna().mean().mean(), 4))
