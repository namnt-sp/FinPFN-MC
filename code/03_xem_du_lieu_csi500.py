"""Buoc 2.3 - Xem cau truc file du lieu cong bo 30features_csi500.parquet (chua chay mo hinh).
Chay tu D:\\LuanVan\\Code:  python 03_xem_du_lieu_csi500.py
"""
import pandas as pd

F = "../Data/30features_csi500.parquet"
df = pd.read_parquet(F)
df["date"] = pd.to_datetime(df["date"])

print("Kich thuoc:", df.shape)
print("Cot:", list(df.columns))
print("Ngay:", df["date"].min().date(), "->", df["date"].max().date(), "|", df["date"].nunique(), "ngay")

n = df.groupby("date")["id"].nunique()
print(f"So ma/ngay: min {n.min()}, TB {n.mean():.1f}, max {n.max()}")

# Chia theo moc cua tac gia: train < 2021, valid 2021, test >= 2022
for ten, m in [("train <2021", df["date"] < "2021-01-01"),
               ("valid 2021 ", df["date"].between("2021-01-01", "2021-12-31")),
               ("test >=2022", df["date"] >= "2022-01-01")]:
    print(f"  {ten}: {df.loc[m, 'date'].nunique():5d} ngay, {m.sum():9,d} dong")

print("\nThong ke target (loi suat ngay ke tiep, chua x100):")
print(df["target"].describe().round(4).to_string())
print("\nTy le gia tri thieu cao nhat:")
print(df.isna().mean().sort_values(ascending=False).head(5).round(4).to_string())
