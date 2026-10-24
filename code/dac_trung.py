"""Bo dac trung chung, chi tu gia dong cua - dung chung cho VN-100 va CSI (chot truoc, khong chon theo IC).
Dau vao: bang rong gia (index = moi phien giao dich cua thi truong, cot = ma), o trong = khong co gia.
"""
import numpy as np
import pandas as pd

TEN = ["rev1", "rev5", "mom20", "mom60", "vol20", "dist_hi20", "skew20", "kurt20", "co_bien_do", "ty_le_r0"]


def tao_dac_trung(gia: pd.DataFrame, bien_do, dung_sai=0.003, min_obs=15):
    """bien_do: so (vd 0.07 cho HOSE) hoac DataFrame cung kich thuoc (CSI: theo bang/ST)."""
    r = gia / gia.shift(1) - 1                                  # R6: qua ngay trong -> NaN
    roll = lambda f: getattr(r.rolling(20, min_periods=min_obs), f)()

    d = {
        "rev1": -r,                                             # dao dau: dao chieu 1 ngay
        "rev5": -(gia / gia.shift(5) - 1),                      # dao dau: dao chieu 5 ngay
        "mom20": gia.shift(5) / gia.shift(20) - 1,              # momentum 20 ngay, bo 5 ngay gan nhat
        "mom60": gia.shift(5) / gia.shift(60) - 1,
        "vol20": roll("std"),
        "dist_hi20": gia / gia.rolling(20, min_periods=min_obs).max() - 1,
        "skew20": roll("skew"),
        "kurt20": roll("kurt"),
        "co_bien_do": np.sign(r).where(r.abs() >= bien_do - dung_sai, 0).where(r.notna()),
        "ty_le_r0": (r == 0).astype(float).where(r.notna()).rolling(20, min_periods=min_obs).mean(),
    }
    dai = pd.concat({k: v.stack() for k, v in d.items()}, axis=1)
    dai["target"] = r.shift(-1).stack().reindex(dai.index)     # loi suat phien ke tiep
    dai.index.names = ["date", "id"]
    co_gia = gia.stack().dropna().index                         # chi giu (ngay, ma) co gia that
    return dai.reindex(co_gia).rename_axis(["date", "id"]).reset_index()
