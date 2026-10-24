"""Buoc 2.7 - Kiem so lieu chan doan P1, P2, P3 tren 301 ngay (CSI 2022-4/2023, DA TIEP XUC -> chi mo ta).
Khong can GPU. Chay tu D:\\LuanVan\\Code:  python 07_kiem_chan_doan_P1_P3.py
"""
import pandas as pd
from finpfn_utils import ic_theo_ngay

pub = pd.read_csv("FinPFN/results/finpfn_perf_csi500.csv.gz", index_col=0, parse_dates=["date"])
pub = pub.groupby(["date", "id"], as_index=False).agg(
    finpfn=("finpfn_median_pred", "mean"), ridge=("ridge_pred", "first"),
    lgbm=("lgbm_pred", "first"), target=("return", "first"))
ours = pd.read_csv("ket_qua_06_tai_lap.csv", parse_dates=["date"]).rename(columns={"pred": "finpfn_minh"})
df = pub.merge(ours, on=["date", "id"])             # chi giu ma co ca hai bo du bao (~406/ngay)

# Chuyen moi du bao thanh hang phan tram trong ngay -> cung thang do de tron
models = ["finpfn", "finpfn_minh", "ridge", "lgbm"]
for c in models:
    df[c] = df.groupby("date")[c].rank(pct=True)
df["tron_deu_3"] = df[["finpfn", "ridge", "lgbm"]].mean(axis=1)
df["tb_2_phan_hoach"] = df[["finpfn", "finpfn_minh"]].mean(axis=1)


def on_dinh(c):
    """P2: tuong quan Spearman giua hang ngay t va t-1 cua cung ma (ma co mat ca hai ngay)."""
    w = df.pivot(index="date", columns="id", values=c)
    return w.corrwith(w.shift(1), axis=1, method="spearman").mean()


print(f"So ngay: {df['date'].nunique()} | ma/ngay: {len(df) / df['date'].nunique():.0f}\n")
print(f"{'Mo hinh':18} {'IC TB':>7} {'IR':>6} {'On dinh ngay-ngay':>18}")
for c in models + ["tron_deu_3", "tb_2_phan_hoach"]:
    ic = ic_theo_ngay(df, c, "target")
    print(f"{c:18} {ic.mean():7.4f} {ic.mean() / ic.std():6.3f} {on_dinh(c):18.3f}")

print(f"\nTuong quan hang FinPFN-Ridge: {ic_theo_ngay(df, 'finpfn', 'ridge').mean():.3f}"
      f" | FinPFN-LGBM: {ic_theo_ngay(df, 'finpfn', 'lgbm').mean():.3f}")
