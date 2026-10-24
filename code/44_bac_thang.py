"""Buoc 6e - BAC THANG LOAI TRU (bo tro, dat truoc): M1 -> + trung binh phan hoach (M1+K) ->
+ co nhom thay doi (M1v+K) -> + xep chong (FinPFN-MC day du).
M1v = tinh chinh nhu M1 nhung co nhom 30-60, KHONG xep chong (python 32_tinh_chinh.py m1v; tinh chinh SAU khi khoa,
chi dung cho phan tich nay, cung quy trinh va ngan sach).
CSI 500: dung lai vu tru cua script 37 (lan chia 0 va 1, cung hat giong) -> so sanh cap doi. VN: ca mat cat.
Chay tu D:\\LuanVan\\Code:  python 44_bac_thang.py   (~1-2 gio)
"""
import hashlib, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import finpfn_mc as F
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

KQ = Path("../Ketqua"); OUT = KQ / "bo_tro"; OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
f_log = open(OUT / "bac_thang.txt", "w", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


CK = KQ / "tinh_chinh/m1v_best.ckpt"
ghi(f"M1v: {CK} | SHA-256 {hashlib.sha256(CK.read_bytes()).hexdigest()[:16]}...")
m1v, c1v, _ = F.nap_mo_hinh(CK, device=dev)


def nw(x, lag=5):
    x = np.asarray(pd.Series(x).dropna(), float); x = x - x.mean(); T = len(x)
    v = x @ x / T + 2 * sum((1 - l / (lag + 1)) * (x[l:] @ x[:-l]) / T for l in range(1, lag + 1))
    return np.sqrt(v / T)


def bac(ic, cot):
    ghi(f"{'Buoc':34} {'IC TB':>7} {'tang them':>10} {'t(NW)':>6}")
    truoc = None
    for ten, c in cot:
        x = ic[c]
        if truoc is None:
            ghi(f"{ten:34} {x.mean():7.4f}")
        else:
            d = (x - ic[truoc]).dropna(); ghi(f"{ten:34} {x.mean():7.4f} {d.mean():+10.4f} {d.mean() / nw(d):6.2f}")
        truoc = c


LADDER = [("M1", "M1"), ("+ trung binh phan hoach (M1+K)", "M1+K"),
          ("+ co nhom thay doi (M1v+K)", "M1v+K"), ("+ xep chong (FinPFN-MC)", "MC")]

# ============ CSI 500: cung vu tru voi script 37, lan chia 0-1
ghi("=== CSI 500 5/2023-6/2026 (vu tru 100 ma, lan chia 0-1) ===")
df = pd.read_parquet("../Data/csi500_dac_trung.parquet"); nen = pd.read_parquet("../Data/nen_csi500.parquet")
kho1, n1, _ = F.chuan_bi(df, None, tu="2023-03-01", den="2026-07-31")
khoM, nM, _ = F.chuan_bi(df, nen, tu="2023-03-01", den="2026-07-31")
ngay = np.intersect1d(n1, nM)
idx = [i for i in range(1, len(ngay)) if np.datetime64("2023-05-01") <= ngay[i] <= np.datetime64("2026-06-30")]
rows, t0 = [], time.time()
for n_, i in enumerate(idx, 1):
    d0, d1 = ngay[i - 1], ngay[i]
    so = int(pd.Timestamp(d1).strftime("%Y%m%d")); seed = lambda *a: np.random.default_rng([so, *a])
    chung = np.intersect1d(F.ghep(kho1, d0, d1)[0], F.ghep(khoM, d0, d1)[0])
    for lan in range(2):
        perm = seed(lan).permutation(chung)
        for v in range(len(chung) // 100):
            u = perm[v * 100:(v + 1) * 100]
            ids, p, y = F.du_bao_ngay(m1v, c1v, kho1, d0, d1, K=20, rng=seed(lan, v, 4), device=dev, ids_chon=u)
            ok = ~np.isnan(y)
            rows.append((pd.Timestamp(d1), lan, v, "M1v+K", stats.spearmanr(p[ok], y[ok])[0]))
    if n_ % 100 == 0:
        ghi(f"  {n_}/{len(idx)} ngay | {(time.time() - t0) / 60:.1f} phut")
moi = pd.DataFrame(rows, columns=["date", "lan", "vt", "mo_hinh", "ic"])
cu = pd.read_parquet(KQ / "kiem_tra_csi500/ic_vu_tru.parquet")
cu = cu[(cu["lan"] < 2) & cu["mo_hinh"].isin(["M1", "M1+K", "MC"])]
cu["date"] = pd.to_datetime(cu["date"])
ic = pd.concat([cu, moi]).groupby(["date", "mo_hinh"])["ic"].mean().unstack().dropna()
bac(ic, LADDER)

# ============ VN: ca mat cat
ghi("\n=== VN 2019-2024 (ca mat cat) ===")
vn = pd.read_parquet("../Data/vn100_dac_trung.parquet"); nv = pd.read_parquet("../Data/nen_vn100.parquet")
k1, a1, _ = F.chuan_bi(vn, None, tu="2018-11-01", den="2024-12-31")
kM, aM, _ = F.chuan_bi(vn, nv, tu="2018-11-01", den="2024-12-31")
ng = np.intersect1d(a1, aM); r2 = []
for i in range(1, len(ng)):
    if ng[i] < np.datetime64("2019-01-01"):
        continue
    ch = np.intersect1d(F.ghep(k1, ng[i - 1], ng[i])[0], F.ghep(kM, ng[i - 1], ng[i])[0])
    so = int(pd.Timestamp(ng[i]).strftime("%Y%m%d"))
    _, p, y = F.du_bao_ngay(m1v, c1v, k1, ng[i - 1], ng[i], K=20, rng=np.random.default_rng([so, 4]), device=dev, ids_chon=ch)
    ok = ~np.isnan(y)
    r2.append((pd.Timestamp(ng[i]), stats.spearmanr(p[ok], y[ok])[0]))
icv = pd.read_csv(KQ / "kiem_tra_vn100/ic_theo_ngay.csv", index_col=0, parse_dates=True)
icv = icv.join(pd.DataFrame(r2, columns=["date", "M1v+K"]).set_index("date"), how="inner")
bac(icv, LADDER)
ghi(f"\nLuu: {OUT / 'bac_thang.txt'}")
