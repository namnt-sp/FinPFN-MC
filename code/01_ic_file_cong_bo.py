"""Buoc 2.1 - Kiem tra moc tai lap tren file du bao tac gia cong bo.
Khong can GPU, khong can tai du lieu. Ky vong: FinPFN IC ~ 0.042, IR ~ 0.86.
Chay tu thu muc D:\\LuanVan\\Code:  python 01_ic_file_cong_bo.py
"""
from pathlib import Path
import pandas as pd
from scipy import stats

f = Path("FinPFN/results/finpfn_perf_csi500.csv.gz")
df = pd.read_csv(f, index_col=0, parse_dates=["date"])
print("So dong:", len(df), "| So ngay:", df["date"].nunique(),
      "|", df["date"].min().date(), "->", df["date"].max().date())

# target: loi suat chuan hoa trong nhom 50 ma; return: loi suat tren ca 500 ma (nhu notebook goc)
cols = {"FinPFN": ("target", "finpfn_median_pred"),
        "TabPFN": ("target", "tabpfn_median_pred"),
        "Ridge":  ("return", "ridge_pred"),
        "LGBM":   ("return", "lgbm_pred")}

ic = pd.DataFrame({name: df.groupby("date").apply(lambda g, y=y, p=p: stats.spearmanr(g[y], g[p])[0])
                   for name, (y, p) in cols.items()})

print("\n        IC TB    IR")
for c in ic:
    print(f"{c:7} {ic[c].mean():.4f}  {ic[c].mean() / ic[c].std():.3f}")

ok = abs(ic["FinPFN"].mean() - 0.042) < 0.002
print("\nMoc tai lap:", "DAT" if ok else "KHONG KHOP - bao lai")
