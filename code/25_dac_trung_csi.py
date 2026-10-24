"""Buoc 3c - Dung du lieu CSI 500 / CSI 300 voi bo dac trung chung (chay SAU KHI tai xong gia Baostock).
Ra: ../Data/csi500_dac_trung.parquet, ../Data/csi300_dac_trung.parquet
Chay tu D:\\LuanVan\\Code:  python 25_dac_trung_csi.py   (~3-6 phut)
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from dac_trung import TEN, tao_dac_trung
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

GOC = Path("../Data/baostock")

# 0) Chan: chi chay khi da tai du
can = set(pd.read_csv(GOC / "thanh_vien_csi500.csv")["code"]) | set(pd.read_csv(GOC / "thanh_vien_csi300.csv")["code"])
co = {f.stem for f in (GOC / "gia").glob("*.csv")}
if len(can - co):
    raise SystemExit(f"CHUA TAI XONG: thieu {len(can - co)}/{len(can)} ma. Cho lenh 12_tai_gia_baostock.py chay xong roi chay lai.")

# 1) Doc gia
parts, rong = [], []
for f in sorted((GOC / "gia").glob("*.csv")):
    d = pd.read_csv(f)
    (parts if len(d) else rong).append(d if len(d) else f.stem)
gia_dai = pd.concat(parts, ignore_index=True)
gia_dai["date"] = pd.to_datetime(gia_dai["date"])
print(f"Ma co du lieu: {gia_dai['code'].nunique()} | ma rong: {len(rong)} {rong[:10]}")

# 2) Phien ngung giao dich -> khong co gia
gia_dai.loc[gia_dai["tradestatus"] == 0, "close"] = np.nan
print(f"Phien ngung giao dich: {(gia_dai['tradestatus'] == 0).sum():,}")
gia = gia_dai.pivot(index="date", columns="code", values="close").sort_index()
st = gia_dai.pivot(index="date", columns="code", values="isST").reindex_like(gia).fillna(0)

# 3) Do ma doi so hieu trung (vd sz.302xxx): loi suat trung khop gan nhu tuyet doi voi ma khac
r = gia.pct_change(fill_method=None)
for c in [c for c in gia.columns if c.startswith("sz.302")]:
    cor = r.corrwith(r[c]).drop(c)
    nghi = cor[cor > 0.999]
    print(f"Kiem tra {c}: {'TRUNG voi ' + str(nghi.index.tolist()) if len(nghi) else 'khong trung'}")

# 4) Bien do theo bang niem yet
ngay = gia.index.to_series()
bd = pd.DataFrame(0.10, index=gia.index, columns=gia.columns)
bd = bd.mask(st == 1, 0.05)
for c in gia.columns:
    if c.startswith("sh.688"):
        bd[c] = 0.20
    elif c.startswith(("sz.300", "sz.301", "sz.302")):
        bd[c] = np.where(ngay >= "2020-08-24", 0.20, bd[c])

# 5) Dac trung tren toan bo lich su, roi loc thanh vien
dt = tao_dac_trung(gia, bien_do=bd)
print(f"Dac trung toan bo: {len(dt):,} dong")
phien = gia.index


def loc_thanh_vien(ten):
    tv = pd.read_csv(GOC / f"thanh_vien_{ten}.csv", parse_dates=["tu_ngay", "den_ngay"])
    # ngay hieu luc -> phien giao dich dau tien tu ngay do tro di
    tv["tu"] = phien[phien.searchsorted(tv["tu_ngay"])]
    tv["den"] = [phien[min(phien.searchsorted(d), len(phien) - 1)] if pd.notna(d) else pd.Timestamp("2100-01-01")
                 for d in tv["den_ngay"]]
    m = dt.merge(tv[["code", "tu", "den"]], left_on="id", right_on="code")
    m = m[(m["date"] >= m["tu"]) & (m["date"] < m["den"])].drop(columns=["code", "tu", "den"])
    return m.drop_duplicates(["date", "id"])


for ten in ["csi500", "csi300"]:
    m = loc_thanh_vien(ten)
    m = m[m["date"] >= "2016-01-01"]
    m.to_parquet(f"../Data/{ten}_dac_trung.parquet", index=False)
    n = m.groupby("date").size()
    print(f"\n=== {ten.upper()}: {len(m):,} dong, {n.size} ngay | ma/ngay min {n.min()}, TB {n.mean():.0f}, max {n.max()}")
    print("Ty le thieu:", m[TEN + ['target']].isna().mean().round(3).to_dict())

    tr = m[(m["date"] < "2021-01-01")].dropna(subset=["target"])     # chi giai doan huan luyen
    print(f"IC dac trung, giai doan huan luyen 2016-2020 ({tr['date'].nunique()} ngay):")
    for c in TEN:
        ic = tr.dropna(subset=[c]).groupby("date")[[c, "target"]].apply(
            lambda g: stats.spearmanr(g[c], g["target"])[0] if g[c].nunique() > 1 else np.nan)
        print(f"   {c:11} IC {ic.mean():7.4f}  IR {ic.mean() / ic.std():6.3f}")
