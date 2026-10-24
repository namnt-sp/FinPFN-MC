"""Buoc 3f - Thu mo rong ngu canh: toan bo ma cua L ngay gan nhat (khong chia nhom), co cot xep chong.
CSI 500 nam 2020 (giai doan huan luyen). Moi bien the thu deu ghi nhat ky (bac tu do nghien cuu).
Chay thu 5 ngay:  python 28_thu_ngu_canh_lon.py thu      (xem toc do tung bien the)
Chay ca nam:      python 28_thu_ngu_canh_lon.py
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from tabpfn import TabPFNRegressor
import finpfn_utils
from dac_trung import TEN
from mo_hinh_nen import chuan_hoa_ngay
from finpfn_utils import CKPT, fit_predict_an_toan
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", message="Found unknown categories")   # TabPFN coi co_bien_do la bien phan loai, vo hai

KQ = Path("../Ketqua")
THU = len(sys.argv) > 1 and sys.argv[1] == "thu"
TABPFN_CKPT = "FinPFN/models/tabpfn-v2-regressor.ckpt"
BIEN_THE = [("finpfn_L1", CKPT, 1), ("finpfn_L5", CKPT, 5),
            ("tabpfn_L5", TABPFN_CKPT, 5), ("tabpfn_L10", TABPFN_CKPT, 10)]
FEATS = TEN + ["ridge", "lgbm"]

df = pd.read_parquet("../Data/csi500_dac_trung.parquet")
df = df[(df["date"] >= "2019-10-01") & (df["date"] <= "2020-12-31")]
nen = pd.read_parquet(KQ / "ket_qua_26_nen_2020.parquet")
for c in ["ridge", "lgbm"]:
    nen[c] = nen.groupby("date")[c].rank(pct=True) - 0.5
z = chuan_hoa_ngay(df, TEN).merge(df[["date", "id", "target"]], on=["date", "id"]).merge(nen, on=["date", "id"])
z = z.rename(columns={"y": "yz"})                             # nhan z-score trong ngay

ngay = np.sort(z["date"].unique())
L_MAX = max(b[2] for b in BIEN_THE)
chay = ngay[L_MAX:L_MAX + 5] if THU else ngay[L_MAX:]       # cung tap ngay cho moi bien the

out = z[z["date"].isin(chay)][["date", "id", "target", "ridge", "lgbm"]].copy()
for ten, ckpt, L in BIEN_THE:
    f_bt = KQ / f"ket_qua_28_{ten}{'_thu' if THU else ''}.parquet"
    if f_bt.exists():                                       # bien the da chay xong -> dung lai
        out = out.merge(pd.read_parquet(f_bt), on=["date", "id"])
        print(f"{ten}: dung lai ket qua da luu")
        continue
    reg = TabPFNRegressor(model_path=ckpt, device="cuda")
    finpfn_utils.SO_LAN_DU_PHONG = 0
    preds, t0 = [], time.time()
    for d in chay:
        i = np.searchsorted(ngay, d)
        ctx = z[z["date"].isin(ngay[i - L:i])].dropna(subset=["yz"])
        q = z[z["date"] == d]
        p = q[["date", "id"]].copy()
        p[ten] = fit_predict_an_toan(reg, ctx[FEATS].to_numpy(np.float32), ctx["yz"].to_numpy(np.float32),
                                     q[FEATS].to_numpy(np.float32))
        preds.append(p)
        print(f"  {ten} {str(d)[:10]} | {len(ctx)} dong ngu canh | {(time.time() - t0) / 60:.1f} phut", end="\r")
    print(f"\n{ten}: {(time.time() - t0) / 60:.1f} phut, du phong {finpfn_utils.SO_LAN_DU_PHONG} lan")
    pd.concat(preds).to_parquet(f_bt, index=False)
    out = out.merge(pd.concat(preds), on=["date", "id"])

for c in ["ridge", "lgbm"] + [b[0] for b in BIEN_THE]:
    out[c] = out.groupby("date")[c].rank(pct=True)
out["tron_nen_2"] = out[["ridge", "lgbm"]].mean(axis=1)
out = out.dropna(subset=["target"])
out.to_csv(KQ / ("ket_qua_28_thu.csv" if THU else "ket_qua_28_ngu_canh_lon_csi2020.csv"), index=False)


def ic(a, b):
    return out.groupby("date")[[a, b]].apply(lambda g: stats.spearmanr(g[a], g[b])[0])


def on_dinh(c):
    w = out.pivot(index="date", columns="id", values=c)
    return w.corrwith(w.shift(1), axis=1, method="spearman").mean()


print(f"\n{out['date'].nunique()} ngay, TB {out.groupby('date').size().mean():.0f} ma/ngay")
print(f"{'Mo hinh':11} {'IC TB':>7} {'IR':>6} {'On dinh':>8} {'- tron_nen_2':>13} {'t':>6} {'tq voi tron':>11}")
for c in ["ridge", "lgbm", "tron_nen_2"] + [b[0] for b in BIEN_THE]:
    x = ic(c, "target")
    if c == "tron_nen_2":
        print(f"{c:11} {x.mean():7.4f} {x.mean() / x.std():6.3f} {on_dinh(c):8.3f}")
        continue
    d = x - ic("tron_nen_2", "target")
    t = d.mean() / d.std() * np.sqrt(len(d))
    print(f"{c:11} {x.mean():7.4f} {x.mean() / x.std():6.3f} {on_dinh(c):8.3f} {d.mean():+13.4f} {t:6.2f}"
          f" {ic(c, 'tron_nen_2').mean():11.3f}")
