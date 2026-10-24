"""Buoc 3d - Kiem nhanh P1 voi bo dac trung chung (chi tu gia dong cua), CSI 500 nam 2020 (giai doan huan luyen).
FinPFN = trong so tac gia (chua tinh chinh lai) -> phep thu gan dung truoc khi dau tu GPU.
Chay thu 5 ngay:  python 26_kiem_P1_dac_trung_chung.py thu
Chay ca nam:      python 26_kiem_P1_dac_trung_chung.py        (~30-40 phut)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tabpfn import TabPFNRegressor
from dac_trung import TEN
from mo_hinh_nen import chuan_hoa_ngay, du_bao_ngoai_mau
import finpfn_utils
from finpfn_utils import CKPT, du_bao_mot_ngay
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

KQ = Path("../Ketqua"); KQ.mkdir(exist_ok=True)
THU = len(sys.argv) > 1 and sys.argv[1] == "thu"

df = pd.read_parquet("../Data/csi500_dac_trung.parquet")
df = df[df["date"] <= "2020-12-31"]                       # chan cung: chi giai doan huan luyen

# 1) Mo hinh nen: hoc tu 2016, du bao ngoai mau tung thang 2020
den = "2020-01-10" if THU else "2020-12-31"
F_NEN = KQ / ("ket_qua_26_nen_thu.parquet" if THU else "ket_qua_26_nen_2020.parquet")
if F_NEN.exists():                                        # da chay roi -> dung lai, khoi cho
    nen = pd.read_parquet(F_NEN)
    print("Dung lai du bao mo hinh nen da luu:", F_NEN.name)
else:
    t0 = time.time()
    nen = du_bao_ngoai_mau(df, TEN, tu_ngay="2020-01-01", den_ngay=den)
    nen.to_parquet(F_NEN, index=False)
    print(f"Mo hinh nen xong: {(time.time() - t0) / 60:.1f} phut")

# 2) FinPFN: dac trung da doi sang hang trong ngay (giong mo hinh nen), 1 phan hoach/ngay
z = chuan_hoa_ngay(df, TEN).drop(columns="y").merge(df[["date", "id", "target"]], on=["date", "id"])
ngay = np.sort(z["date"].unique())
ngay = ngay[ngay <= np.datetime64(den)]
i0 = np.searchsorted(ngay, np.datetime64("2020-01-01"))
reg = TabPFNRegressor(model_path=CKPT, device="cuda")
parts, t0 = [], time.time()
for i in range(i0, len(ngay)):
    p, _ = du_bao_mot_ngay(reg, z, TEN, ngay[i - 1], ngay[i], np.random.default_rng([7, i]), "khong_hoan_lai")
    parts.append(p)
    print(f"  FinPFN {str(ngay[i])[:10]} | {(time.time() - t0) / 60:.1f} phut", end="\r")
print(f"\nSo nhom dung phuong an du phong (bo cot gan hang so): {finpfn_utils.SO_LAN_DU_PHONG}")
fin = pd.concat(parts).rename(columns={"pred": "finpfn"})

# 3) Ghep va danh gia
m = fin.merge(nen, on=["date", "id"]).merge(df[["date", "id", "target"]], on=["date", "id"]).dropna(subset=["target"])
for c in ["finpfn", "ridge", "lgbm"]:
    m[c] = m.groupby("date")[c].rank(pct=True)
m["tron_deu_3"] = m[["finpfn", "ridge", "lgbm"]].mean(axis=1)
m["tron_nen_2"] = m[["ridge", "lgbm"]].mean(axis=1)
m.to_csv(KQ / "ket_qua_26_P1_csi2020.csv", index=False)


def ic(a, b):
    return m.groupby("date")[[a, b]].apply(lambda g: stats.spearmanr(g[a], g[b])[0])


def on_dinh(c):
    w = m.pivot(index="date", columns="id", values=c)
    return w.corrwith(w.shift(1), axis=1, method="spearman").mean()


print(f"\nCSI 500 nam 2020: {m['date'].nunique()} ngay, TB {m.groupby('date').size().mean():.0f} ma/ngay")
print(f"{'Mo hinh':11} {'IC TB':>7} {'IR':>6} {'On dinh':>8}")
for c in ["finpfn", "ridge", "lgbm", "tron_nen_2", "tron_deu_3"]:
    x = ic(c, "target")
    print(f"{c:11} {x.mean():7.4f} {x.mean() / x.std():6.3f} {on_dinh(c):8.3f}")
print(f"\nTuong quan hang FinPFN-Ridge: {ic('finpfn', 'ridge').mean():.3f} | FinPFN-LGBM: {ic('finpfn', 'lgbm').mean():.3f}"
      f" | Ridge-LGBM: {ic('ridge', 'lgbm').mean():.3f}")
