"""Buoc 6b - H3 (de cuong): he thong FinPFN-MC + dai tre (C) co loi suat rong tuan cao hon he thong M1
khong dai tre (A)? Ung ho neu C - A > 0 va p < 0,05 hai phia; stationary bootstrap (Politis-Romano),
khoi TB 4 tuan, 10.000 lan.
Thiet ke (de cuong): long-only, top 20% theo tin hieu TB trong tuan (hang trong ngay, TB cac ngay trong tuan);
  tai co cau cuoi tuan (dong cua); trong so deu; phi 0,25%/chieu + thue ban 0,1%;
  khong mua ma dong cua o gia tran, khong ban ma dong cua o gia san (|r| >= 7% - 0,3 diem %);
  dai tre: vao khi lot top 20%, ra khi roi top 50%.  A=M1 | B=MC | C=MC+dai tre | D=M1+dai tre.
Do nhay (de cuong): phi 0,15%; tai co cau hang thang. Bo tro (ngoai de cuong): giao dich tre 1 phien.
Loi suat thieu trong ky nam giu (ngung GD) tinh bang 0.
Chay tu D:\\LuanVan\\Code:  python 40_H3_danh_muc.py   (~5-15 phut)
"""
from pathlib import Path
import numpy as np
import pandas as pd

KQ = Path("../Ketqua"); OUT = KQ / "kiem_tra_vn100"
PHI, THUE, TRAN = 0.0025, 0.001, 0.07 - 0.003
TOP_VAO, TOP_RA, B_LAP, KHOI = 0.20, 0.50, 10_000, 4
f_log = open(OUT / "nhat_ky_H3.txt", "w", encoding="utf-8")


def ghi(s=""):
    print(s); f_log.write(s + "\n"); f_log.flush()


du = pd.read_parquet(OUT / "du_bao_ngay.parquet")
sach = pd.read_csv("../Data/vn100_sach.csv", parse_dates=["date"])
ret = sach.pivot(index="date", columns="ma", values="ret").sort_index()
ret = ret.loc["2019-01-01":"2024-12-31"]
hang = {m: du.assign(h=du.groupby("date")[m].rank(pct=True)).pivot(index="date", columns="id", values="h")
        for m in ["M1", "MC"]}
ngay = hang["MC"].index


def lich(tan_suat, tre):
    """Ngay ra tin hieu (cuoi tuan/thang) va ngay thuc hien (cung ngay, hoac phien ke tiep neu tre=1)."""
    s = pd.Series(ngay, index=ngay)
    ky = s.dt.to_period("W-FRI" if tan_suat == "tuan" else "M")
    cuoi = s.groupby(ky).max().to_numpy()
    if not tre:
        return cuoi, cuoi, ky
    ds = list(ngay)
    th = [ds[ds.index(d) + 1] if ds.index(d) + 1 < len(ds) else None for d in cuoi]
    keep = [i for i, d in enumerate(th) if d is not None]
    return cuoi[keep], np.array([th[i] for i in keep]), ky


def chay(mo_hinh, dai_tre, phi=PHI, tan_suat="tuan", tre=0):
    tin, thuc, ky = lich(tan_suat, tre)
    H = hang[mo_hinh]
    tb_ky = H.groupby(ky.to_numpy()).mean()                    # tin hieu TB trong ky
    w = pd.Series(dtype=float)                                  # trong so hien tai (sau troi gia)
    rows = []
    for k in range(len(tin) - 1):
        d_tin, d_th, d_sau = tin[k], thuc[k], thuc[k + 1]
        sig = tb_ky.loc[ky.loc[d_tin]].dropna()
        r_th = ret.loc[d_th] if d_th in ret.index else pd.Series(dtype=float)
        tran = set(r_th[r_th >= TRAN].index); san = set(r_th[r_th <= -TRAN].index)
        n = len(sig); rk = sig.rank(ascending=False)
        top_vao = set(rk[rk <= np.ceil(TOP_VAO * n)].index)
        top_ra = set(rk[rk <= np.ceil(TOP_RA * n)].index)
        cu = set(w.index)
        giu = (cu & top_ra) if dai_tre else (cu & top_vao)
        giu |= (cu & san)                                       # khong ban duoc ma dong cua o san
        moi = (top_vao - cu) - tran                             # khong mua duoc ma dong cua o tran
        nam = sorted(giu | moi)
        muc_tieu = pd.Series(1 / len(nam), index=nam) if nam else pd.Series(dtype=float)
        tat = muc_tieu.index.union(w.index)
        dw = muc_tieu.reindex(tat, fill_value=0) - w.reindex(tat, fill_value=0)
        mua, ban = dw.clip(lower=0).sum(), (-dw).clip(lower=0).sum()
        chi_phi = mua * phi + ban * (phi + THUE)
        kp = ret.loc[(ret.index > d_th) & (ret.index <= d_sau), nam].fillna(0)
        g = (1 + kp).prod() - 1 if len(kp) else pd.Series(0.0, index=nam)
        gop = float((muc_tieu * g).sum()) if nam else 0.0
        rong = (1 - chi_phi) * (1 + gop) - 1
        w = muc_tieu * (1 + g) / (1 + gop) if nam else pd.Series(dtype=float)
        rows.append((d_sau, gop, rong, chi_phi, (mua + ban) / 2, len(nam)))
    return pd.DataFrame(rows, columns=["date", "gop", "rong", "chi_phi", "vong_quay", "so_ma"]).set_index("date")


def chi_so(x, tan=52):
    nav = (1 + x["rong"]).cumprod()
    nam = len(x) / tan
    return dict(LS_rong_nam=nav.iloc[-1] ** (1 / nam) - 1, Sharpe_rong=x["rong"].mean() / x["rong"].std() * np.sqrt(tan),
                Vong_quay_nam=x["vong_quay"].sum() / nam, Chi_phi_nam=x["chi_phi"].sum() / nam,
                Sut_giam_TD=(nav / nav.cummax() - 1).min(), So_ma_TB=x["so_ma"].mean())


def bootstrap(d, B=B_LAP, L=KHOI, seed=0):
    """Stationary bootstrap (Politis-Romano), do dai khoi ~ hinh hoc TB L. p hai phia (quanh 0) va KTC 95%."""
    d = np.asarray(d, float); T = len(d); rng = np.random.default_rng(seed)
    bat_dau = rng.integers(T, size=(B, T))
    moi = rng.random((B, T)) < 1 / L; moi[:, 0] = True
    vt = np.maximum.accumulate(np.where(moi, np.arange(T), 0), axis=1)
    idx = (np.take_along_axis(bat_dau, vt, axis=1) + (np.arange(T) - vt)) % T
    tb_null = (d - d.mean())[idx].mean(axis=1)
    tb = d[idx].mean(axis=1)
    pv = (np.abs(tb_null) >= abs(d.mean())).mean()
    return d.mean(), pv, np.percentile(tb, [2.5, 97.5])


def bang_va_kiem(phi=PHI, tan_suat="tuan", tre=0, chinh=False):
    tan = 52 if tan_suat == "tuan" else 12
    ht = {"A": chay("M1", False, phi, tan_suat, tre), "B": chay("MC", False, phi, tan_suat, tre),
          "C": chay("MC", True, phi, tan_suat, tre), "D": chay("M1", True, phi, tan_suat, tre)}
    bang = pd.DataFrame({k: chi_so(v, tan) for k, v in ht.items()}).T
    ghi(bang.to_string(float_format=lambda v: f"{v:.4f}"))
    for ten, a, b in [("C - A (H3)", "C", "A"), ("B - A", "B", "A"), ("C - D", "C", "D"), ("D - A", "D", "A")]:
        d = (ht[a]["rong"] - ht[b]["rong"]).dropna()
        m, pv, ci = bootstrap(d.to_numpy(), L=KHOI if tan == 52 else 1)
        kl = ""
        if chinh and ten.startswith("C - A"):
            kl = " ==> " + ("UNG HO H3" if m > 0 and pv < 0.05 else "KHONG UNG HO H3")
        ghi(f"  {ten:11} TB {m * 100:+.3f}%/{'tuan' if tan == 52 else 'thang'} | p {pv:.4f} | "
            f"KTC95% [{ci[0] * 100:+.3f}%, {ci[1] * 100:+.3f}%]{kl}")
    return ht


ghi("=== H3 CHINH: tuan, phi 0,25% + thue 0,1% ===")
ht = bang_va_kiem(chinh=True)
pd.concat({k: v for k, v in ht.items()}, axis=1).to_csv(OUT / "H3_loi_suat_tuan.csv")
ew = ret.fillna(0).mean(axis=1)
ew_t = (1 + ew).groupby(ew.index.to_period("W-FRI")).prod() - 1
ghi(f"Mo ta: VN-100 binh quan deu (khong phi) LS nam {(1 + ew_t).prod() ** (52 / len(ew_t)) - 1:.4f}, "
    f"Sharpe {ew_t.mean() / ew_t.std() * np.sqrt(52):.3f}")
ghi("\n=== Do nhay (de cuong): phi 0,15% ===")
bang_va_kiem(phi=0.0015)
ghi("\n=== Do nhay (de cuong): tai co cau hang thang ===")
bang_va_kiem(tan_suat="thang")
ghi("\n=== Bo tro (ngoai de cuong): giao dich tre 1 phien ===")
bang_va_kiem(tre=1)
