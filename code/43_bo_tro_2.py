"""Buoc 6d - PHAN TICH BO TRO (tiep), dat truoc trong de cuong, bao cao day du:
  (5) Thich nghi cau truc tin hieu (P4): muc phu thuoc cua M1 va FinPFN-MC vao tung dac trung, dac biet
      rev1 so voi rev5, tren VN va CSI 500. KHAI BAO: thay SHAP bang do nhay che (occlusion) vi chi phi:
      phu thuoc_j = TB |du bao - du bao khi cot j cua dong hoi dat ve 0 (hang trung vi)|, ngu canh giu nguyen.
  (6) Mo hinh nen hoc tren CSI so voi hoc tren VN: FinPFN-MC tren VN dung cot xep chong tu Ridge/LGBM hoc tren
      CSI 500 (cua so mo rong, hoc lai dau moi thang, cung tham so), so voi ban chinh (nen hoc tren VN).
Chay tu D:\\LuanVan\\Code:  python 43_bo_tro_2.py      (~40-70 phut; LightGBM CPU + GPU)
"""
import json, hashlib, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import finpfn_mc as F
from dac_trung import TEN
from mo_hinh_nen import chuan_hoa_ngay, LGBM_THAM_SO
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

KQ = Path("../Ketqua"); OUT = KQ / "bo_tro"; OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
f_log = open(OUT / "bo_tro_2.txt", "w", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


khoa = json.loads((KQ / "cau_hinh_khoa.json").read_text(encoding="utf-8"))
for k, p in {"mo_hinh_M1": KQ / "tinh_chinh/m1_best.ckpt", "mo_hinh_FinPFN_MC": KQ / "tinh_chinh/mc_best.ckpt",
             "code_finpfn_mc": Path("finpfn_mc.py")}.items():
    if hashlib.sha256(p.read_bytes()).hexdigest() != khoa["dau_van_tay_sha256"][k]:
        raise SystemExit(f"DAU VAN TAY KHONG KHOP: {k}")
m1, c1, _ = F.nap_mo_hinh(KQ / "tinh_chinh/m1_best.ckpt", device=dev)
mc, cm, _ = F.nap_mo_hinh(KQ / "tinh_chinh/mc_best.ckpt", device=dev)


def nw(x, lag=5):
    x = np.asarray(pd.Series(x).dropna(), float); x = x - x.mean(); T = len(x)
    v = x @ x / T + 2 * sum((1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / T for l in range(1, lag + 1))
    return np.sqrt(v / T)


def trung_vi_nhom(model, crit, Xc, yc, Xq):
    with torch.no_grad():
        lg = F._tien(model, torch.tensor(Xc[:, None]), torch.tensor(yc[:, None]), torch.tensor(Xq[:, None]), dev, True)
        return crit.median(lg.float()).cpu().numpy()[:, 0]


# ============ (5) muc phu thuoc dac trung
ghi("=== (5) MUC PHU THUOC DAC TRUNG (do nhay che, thay SHAP) ===")
for ten_tt, f_dt, f_nen, tu, den, buoc in [("VN 2019-2024", "vn100_dac_trung", "nen_vn100", "2018-11-01", "2024-12-31", 10),
                                         ("CSI 500 5/2023-6/2026", "csi500_dac_trung", "nen_csi500", "2023-03-01", "2026-06-30", 5)]:
    df = pd.read_parquet(f"../Data/{f_dt}.parquet"); nen = pd.read_parquet(f"../Data/{f_nen}.parquet")
    test_tu = "2019-01-01" if ten_tt.startswith("VN") else "2023-05-01"
    for ten_m, model, crit, dung_nen in [("M1", m1, c1, False), ("FinPFN-MC", mc, cm, True)]:
        kho, ng, feats = F.chuan_bi(df, nen if dung_nen else None, tu=tu, den=den)
        idx = [i for i in range(1, len(ng)) if ng[i] >= np.datetime64(test_tu)][::buoc]
        tong = np.zeros(len(feats)); dem = 0
        for i in idx:
            ids, X0, y0, X1, _ = F.ghep(kho, ng[i - 1], ng[i])
            for e in np.array_split(np.random.default_rng([i, 7]).permutation(len(ids)), max(1, round(len(ids) / 50))):
                yc = F.zs(y0[e]); goc = trung_vi_nhom(model, crit, X0[e], yc, X1[e])
                for j in range(len(feats)):
                    Xq = X1[e].copy(); Xq[:, j] = 0.0
                    tong[j] += np.abs(goc - trung_vi_nhom(model, crit, X0[e], yc, Xq)).mean()
                dem += 1
        imp = pd.Series(tong / dem, index=feats); ti = imp / imp.sum()
        ghi(f"{ten_tt} | {ten_m} ({len(idx)} ngay mau): rev1/rev5 = {imp['rev1'] / imp['rev5']:.2f} | ty trong: "
            + ", ".join(f"{k} {v:.1%}" for k, v in ti.sort_values(ascending=False).items()))

# ============ (6) mo hinh nen hoc tren CSI ap dung cho VN
ghi("\n=== (6) MO HINH NEN HOC TREN CSI 500, AP DUNG CHO VN ===")
csi = pd.read_parquet("../Data/csi500_dac_trung.parquet"); vn = pd.read_parquet("../Data/vn100_dac_trung.parquet")
zc = chuan_hoa_ngay(csi, TEN).dropna(subset=["y"]); zv = chuan_hoa_ngay(vn, TEN)
thang = pd.period_range("2019-01", "2024-12", freq="M"); rows, t0 = [], time.time()
for th in thang:
    moc = th.start_time
    hoc = zc[zc["date"] < moc]; du = zv[(zv["date"] >= moc) & (zv["date"] <= th.end_time)]
    if not len(du):
        continue
    X, y = hoc[TEN].to_numpy(), hoc["y"].to_numpy()
    r = Ridge(alpha=1.0).fit(X, y); l = LGBMRegressor(**LGBM_THAM_SO).fit(X, y)
    o = du[["date", "id"]].copy(); o["ridge"] = r.predict(du[TEN].to_numpy()); o["lgbm"] = l.predict(du[TEN].to_numpy())
    rows.append(o); print(f"  {th} | {(time.time() - t0) / 60:.1f} phut", end="\r")
nen_csi = pd.concat(rows, ignore_index=True)
nen_csi.to_parquet(OUT / "nen_vn_tu_csi.parquet", index=False)
# can them thang 12/2018 lam ngu canh cho ngay dau 2019: dung nen VN goc cho ngay truoc 2019 (chi lam ngu canh)
nen_vn = pd.read_parquet("../Data/nen_vn100.parquet")
nen_mix = pd.concat([nen_vn[nen_vn["date"] < "2019-01-01"], nen_csi], ignore_index=True)
kho1, n1, _ = F.chuan_bi(vn, None, tu="2018-11-01", den="2024-12-31")
khoX, nX, _ = F.chuan_bi(vn, nen_mix, tu="2018-11-01", den="2024-12-31")
ng = np.intersect1d(n1, nX); out = []
for i in range(1, len(ng)):
    if ng[i] < np.datetime64("2019-01-01"):
        continue
    ch = np.intersect1d(F.ghep(kho1, ng[i - 1], ng[i])[0], F.ghep(khoX, ng[i - 1], ng[i])[0])
    so = int(pd.Timestamp(ng[i]).strftime("%Y%m%d"))
    a, p, y = F.du_bao_ngay(mc, cm, khoX, ng[i - 1], ng[i], K=20, rng=np.random.default_rng([so, 2]), device=dev, ids_chon=ch)
    out.append(pd.DataFrame({"date": pd.Timestamp(ng[i]), "id": a, "MC_nenCSI": p, "target": y}))
x = pd.concat(out).merge(nen_csi.rename(columns={"ridge": "Ridge_CSI", "lgbm": "LGBM_CSI"}), on=["date", "id"])
x = x.dropna(subset=["target"])
ic = pd.DataFrame({c: x.groupby("date")[[c, "target"]].apply(lambda g: stats.spearmanr(g[c], g["target"])[0])
                   for c in ["MC_nenCSI", "Ridge_CSI", "LGBM_CSI"]})
icv = pd.read_csv(KQ / "kiem_tra_vn100/ic_theo_ngay.csv", index_col=0, parse_dates=True)
ic = ic.join(icv[["MC", "Ridge", "LGBM", "M1"]], how="inner")
ghi("IC TB tren VN 2019-2024: " + ", ".join(f"{c} {ic[c].mean():.4f}" for c in ic.columns))
d = (ic["MC"] - ic["MC_nenCSI"]).dropna()
ghi(f"MC (nen VN) - MC (nen CSI) = {d.mean():+.4f} (t {d.mean() / nw(d):.2f})")
d = (ic["MC_nenCSI"] - ic["M1"]).dropna()
ghi(f"MC (nen CSI) - M1 = {d.mean():+.4f} (t {d.mean() / nw(d):.2f})")
ghi(f"\nLuu: {OUT}")
