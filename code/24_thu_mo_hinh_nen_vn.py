"""Buoc 3b - Chay thu mo hinh nen tren VN, CHI giai doan phat trien: hoc tu 2017, du bao ngoai mau nam 2018.
Khong du bao/nhin 2019+ (tap kiem tra VN). Muc dich: kiem tra pipeline chay dung, khong chon cau hinh.
Chay tu D:\\LuanVan\\Code:  python 24_thu_mo_hinh_nen_vn.py   (~1 phut)
"""
import warnings
import pandas as pd
from scipy import stats
from dac_trung import TEN
from mo_hinh_nen import du_bao_ngoai_mau
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

df = pd.read_parquet("../Data/vn100_dac_trung.parquet")
df = df[df["date"] <= "2018-12-31"]                     # chan cung: khong cham 2019+

pred = du_bao_ngoai_mau(df, TEN, tu_ngay="2018-01-01", den_ngay="2018-12-31")
pred = pred.merge(df[["date", "id", "target"]], on=["date", "id"]).dropna(subset=["target"])
for c in ["ridge", "lgbm"]:
    pred[c] = pred.groupby("date")[c].rank(pct=True)
pred["tron_2"] = (pred["ridge"] + pred["lgbm"]) / 2


def ic(a, b):
    return pred.groupby("date")[[a, b]].apply(lambda g: stats.spearmanr(g[a], g[b])[0])


def on_dinh(c):
    w = pred.pivot(index="date", columns="id", values=c)
    return w.corrwith(w.shift(1), axis=1, method="spearman").mean()


print(f"Du bao ngoai mau 2018: {pred['date'].nunique()} ngay, TB {pred.groupby('date').size().mean():.0f} ma/ngay")
print(f"{'Mo hinh':8} {'IC TB':>7} {'IR':>6} {'On dinh':>8}")
for c in ["ridge", "lgbm", "tron_2"]:
    x = ic(c, "target")
    print(f"{c:8} {x.mean():7.4f} {x.mean() / x.std():6.3f} {on_dinh(c):8.3f}")
print(f"\nTuong quan hang Ridge-LGBM: {ic('ridge', 'lgbm').mean():.3f}")
