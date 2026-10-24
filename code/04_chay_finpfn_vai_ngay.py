"""Buoc 2.4 - Chay lai FinPFN (trong so tac gia) tren N ngay dau tap test, so voi file du bao cong bo.
Quy trinh giong notebook goc: ngu canh = 50 ma ngay t-1 (co nhan), du bao 50 ma ngay t,
chia ~590 ma thanh ceil(n/50) nhom, rut co hoan lai, seed 42, target x100, chuan hoa trong nhom.
Chay tu D:\\LuanVan\\Code:  python 04_chay_finpfn_vai_ngay.py
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from tabpfn import TabPFNRegressor

N_NGAY = 20                      # so ngay test chay thu; tang dan sau khi on
SPD = 50                         # so ma moi ngay trong 1 nhom ngu canh
CKPT = "FinPFN/models/finpfn_30feats_csi500.ckpt"

df = pd.read_parquet("../Data/30features_csi500.parquet")
df["date"] = pd.to_datetime(df["date"])
df["target"] = df["target"] * 100                       # percentage=True nhu tac gia
test = df[df["date"] >= "2022-01-01"]
feats = [c for c in test.columns if c not in ("date", "id", "target")]
dates = sorted(test["date"].unique())[: N_NGAY + 1]

np.random.seed(42)
reg = TabPFNRegressor(model_path=CKPT, device="cuda")
rows = []
for d0, d1 in zip(dates[:-1], dates[1:]):
    t = time.time()
    dd = test[test["date"].isin([d0, d1])]
    cnt = dd.groupby("id")["date"].count()
    stocks = cnt[cnt == 2].index.to_numpy()
    dd = dd.set_index("id")
    k = int(np.ceil(len(stocks) / SPD))
    for stks in np.random.choice(stocks, size=(k, SPD), replace=True):
        g = dd.loc[stks].reset_index().sort_values(["date", "id"])
        g["target"] = g.groupby("date")["target"].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0)
        X = g[feats].to_numpy(np.float32); y = g["target"].to_numpy(np.float32)
        reg.fit(X[:SPD], y[:SPD])
        out = g[["id", "date"]].iloc[SPD:].copy()
        out["pred"] = reg.predict(X[SPD:], output_type="full")["median"]
        rows.append(out)
    print(f"{pd.Timestamp(d1).date()}: {len(stocks)} ma, {k} nhom, {time.time() - t:.1f}s")

ours = pd.concat(rows).groupby(["date", "id"], as_index=False)["pred"].mean()
ours = ours.merge(test[["date", "id", "target"]].rename(columns={"target": "ret"}), on=["date", "id"])

pub = pd.read_csv("FinPFN/results/finpfn_perf_csi500.csv.gz", index_col=0, parse_dates=["date"])
pub = pub.groupby(["date", "id"], as_index=False).agg(pub_pred=("finpfn_median_pred", "mean"))
m = ours.merge(pub, on=["date", "id"])
print(f"\nGhep voi file cong bo: {len(m)} / {len(ours)} dong")

ic = lambda a, b: m.groupby("date").apply(lambda g: stats.spearmanr(g[a], g[b])[0])
ic_ours, ic_pub, agree = ic("pred", "ret"), ic("pub_pred", "ret"), ic("pred", "pub_pred")
print(f"IC (so voi loi suat toan bo ma) - cua minh: {ic_ours.mean():.4f} | cong bo: {ic_pub.mean():.4f}")
print(f"Tuong quan hang du bao minh vs cong bo (TB theo ngay): {agree.mean():.3f}")
ours.to_csv("ket_qua_04.csv", index=False)
