"""Buoc 3a - Tao dac trung chung cho VN-100 va kiem tra dieu kien tren giai doan phat trien 4/2017-12/2018.
KHONG nhin IC tu 2019 tro di (tap kiem tra VN).
Chay tu D:\\LuanVan\\Code:  python 23_dac_trung_vn100.py
"""
import pandas as pd
import warnings
from scipy import stats
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
from dac_trung import TEN, tao_dac_trung

sach = pd.read_csv("../Data/vn100_sach.csv", parse_dates=["date"])
gia = sach.pivot(index="date", columns="ma", values="gia").sort_index()
print(f"Bang gia: {gia.shape[0]} phien x {gia.shape[1]} ma")

df = tao_dac_trung(gia, bien_do=0.07)
co_r7 = sach.set_index(["date", "ma"])["co_r7"].rename_axis(["date", "id"])
df = df.join(co_r7, on=["date", "id"])
df.to_parquet("../Data/vn100_dac_trung.parquet", index=False)
print(f"Da luu: {len(df):,} dong | cot: {list(df.columns)}")

print("\nTy le thieu tung cot (toan giai doan):")
print(df[TEN + ["target"]].isna().mean().round(3).to_string())

# ---- Kiem tra dieu kien: chi 4/2017-12/2018 ----
dev = df[(df["date"] >= "2017-04-01") & (df["date"] <= "2018-12-31")].dropna(subset=["target"])
print(f"\nGiai doan phat trien: {dev['date'].nunique()} ngay, TB {dev.groupby('date').size().mean():.0f} ma/ngay")
print(f"{'Dac trung':11} {'IC TB':>7} {'IR':>6}   (de cuong P4: rev1 ~0, rev5 +0.030, mom20 +0.023, mom60 +0.027)")
for c in TEN:
    ic = dev.dropna(subset=[c]).groupby("date")[[c, "target"]].apply(
        lambda g: stats.spearmanr(g[c], g["target"])[0] if g[c].nunique() > 1 else float("nan"))
    print(f"{c:11} {ic.mean():7.4f} {ic.mean() / ic.std():6.3f}")
