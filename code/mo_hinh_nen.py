"""Mo hinh nen Ridge va LightGBM: cua so mo rong, tai huan luyen dau moi thang, du bao ngoai mau.
Dung chung cho VN va CSI. Lua chon thiet ke chot truoc (xem nhat ky):
  - Dac trung -> hang trong ngay, thang [-0.5, 0.5]; thieu -> 0 (trung vi).
  - Nhan -> z-score trong ngay.
  - Mo hinh tai thang M chi hoc dong co date < ngay giao dich dau thang M (nhan cua dong t-1 = r_t, da biet).
  - Tham so co dinh, khong tinh chinh.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

LGBM_THAM_SO = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


def chuan_hoa_ngay(df, feats):
    out = df[["date", "id"]].copy()
    g = df.groupby("date")
    for c in feats:
        out[c] = (g[c].rank(pct=True) - 0.5).fillna(0.0)
    t = g["target"]
    out["y"] = (df["target"] - t.transform("mean")) / t.transform("std")
    return out


def du_bao_ngoai_mau(df, feats, tu_ngay, den_ngay, hoc_tu=None):
    """Tra ve bang date, id, ridge, lgbm cho cac ngay trong [tu_ngay, den_ngay]."""
    z = chuan_hoa_ngay(df, feats)
    if hoc_tu is not None:
        z = z[z["date"] >= hoc_tu]
    ngay = np.sort(z["date"].unique())
    ngay = ngay[(ngay >= np.datetime64(tu_ngay)) & (ngay <= np.datetime64(den_ngay))]
    thang = pd.Series(ngay).dt.to_period("M")
    ket_qua = []
    for _, cac_ngay in pd.Series(ngay).groupby(thang):
        moc = cac_ngay.iloc[0]
        hoc = z[(z["date"] < moc)].dropna(subset=["y"])
        X, y = hoc[feats].to_numpy(), hoc["y"].to_numpy()
        ridge = Ridge(alpha=1.0).fit(X, y)
        lgbm = LGBMRegressor(**LGBM_THAM_SO).fit(X, y)
        du = z[z["date"].isin(cac_ngay)]
        out = du[["date", "id"]].copy()
        out["ridge"] = ridge.predict(du[feats].to_numpy())
        out["lgbm"] = lgbm.predict(du[feats].to_numpy())
        ket_qua.append(out)
        print(f"  {moc.date() if hasattr(moc, 'date') else str(moc)[:10]}: hoc {len(hoc):,} dong", end="\r")
    print()
    return pd.concat(ket_qua, ignore_index=True)
