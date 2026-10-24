"""Ham dung chung cho cac buoc 2.x. Quy trinh giong notebook goc cua Wang & Lera."""
import numpy as np
import pandas as pd
from scipy import stats

SPD = 50                                          # so ma/ngay trong 1 nhom ngu canh
CKPT = "FinPFN/models/finpfn_30feats_csi500.ckpt"


def doc_test(path="../Data/30features_csi500.parquet", tu_ngay="2022-01-01"):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["target"] = df["target"] * 100             # percentage=True nhu tac gia
    test = df[df["date"] >= tu_ngay]
    feats = [c for c in test.columns if c not in ("date", "id", "target")]
    return test, feats, sorted(test["date"].unique())


SO_LAN_DU_PHONG = 0       # dem so nhom phai dung phuong an du phong (ghi vao nhat ky)


def fit_predict_an_toan(reg, X_ctx, y_ctx, X_q):
    """TabPFN tu bien doi Yeo-Johnson tung cot; cot gan nhu hang so trong ngu canh 50 dong
    (vd co_bien_do, ty_le_r0) doi khi lam tran so (inf). Quy tac du phong co dinh:
    bo cac cot co <= 3 gia tri khac nhau trong ngu canh cua nhom do, roi chay lai."""
    global SO_LAN_DU_PHONG
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            reg.fit(X_ctx, y_ctx)
            return reg.predict(X_q, output_type="full")["median"]
    except ValueError:
        SO_LAN_DU_PHONG += 1
        giu = [j for j in range(X_ctx.shape[1]) if len(np.unique(X_ctx[:, j])) > 3]
        with np.errstate(over="ignore", invalid="ignore"):
            reg.fit(X_ctx[:, giu], y_ctx)
            return reg.predict(X_q[:, giu], output_type="full")["median"]


def du_bao_mot_ngay(reg, test, feats, d0, d1, rng, cach_chia="co_hoan_lai"):
    """Du bao ngay d1 dung ngu canh ngay d0.
    cach_chia="co_hoan_lai":   rut ceil(n/50) nhom co hoan lai (code dang bat trong notebook).
    cach_chia="khong_hoan_lai": xao tron, lay toi da 500 ma, chia 10 nhom rieng biet
                                (code bi comment trong notebook; khop file cong bo: dung 500 ma/ngay).
    Tra ve (bang du bao trung binh theo ma, so ma hop le trong ngay)."""
    dd = test[test["date"].isin([d0, d1])]
    # ma hop le: co mat ca hai ngay VA co nhan o ngay ngu canh d0 (nhan thieu khi ma ngung GD phien sau)
    co_nhan_d0 = dd.loc[(dd["date"] == d0) & dd["target"].notna(), "id"]
    co_d1 = dd.loc[dd["date"] == d1, "id"]
    stocks = np.intersect1d(co_nhan_d0.to_numpy(), co_d1.to_numpy())
    dd = dd.set_index("id")
    if cach_chia == "co_hoan_lai":
        k = int(np.ceil(len(stocks) / SPD))
        nhom = rng.choice(stocks, size=(k, SPD), replace=True)
    else:
        s = rng.permutation(stocks)
        k = min(500, len(s)) // SPD
        nhom = np.split(s[: k * SPD], k)
    rows = []
    for stks in nhom:
        g = dd.loc[stks].reset_index().sort_values(["date", "id"])
        g["target"] = g.groupby("date")["target"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0)
        X = g[feats].to_numpy(np.float32)
        y = g["target"].to_numpy(np.float32)
        out = g[["id", "date"]].iloc[SPD:].copy()
        out["pred"] = fit_predict_an_toan(reg, X[:SPD], y[:SPD], X[SPD:])
        rows.append(out)
    pred = pd.concat(rows).groupby(["date", "id"], as_index=False)["pred"].mean()
    return pred, len(stocks)


def ic_theo_ngay(df, a, b):
    """Rank IC (Spearman) giua hai cot, tinh theo tung ngay."""
    return df.groupby("date")[[a, b]].apply(lambda g: stats.spearmanr(g[a], g[b])[0])
