"""Buoc 1.3 - Lam sach VN-100 theo quy tac R1-R7 (de cuong v3). Moi bien doi ghi vao nhat ky.
Can: ../Data/vn100_2017_2024.csv, ../Data/lich_su_san_vn100.csv
Ra:  ../Data/vn100_sach.csv (dang dai), ../Data/nhat_ky_lam_sach.csv
Chay tu D:\\LuanVan\\Code:  python 22_lam_sach_vn100.py
"""
import pandas as pd

GOC = "../Data/vn100_2017_2024.csv"
SAN = "../Data/lich_su_san_vn100.csv"
R3_MA, R3_TU, R3_DEN = ["VND", "BSI", "DXS", "PAN"], "2021-03-31", "2021-09-05"
R4_MA, R4_TU = "VHM", "2018-05-17"
R5_N = 5
R7_NGUONG = 2.0          # gia dieu chinh < 1 nghin dong -> buoc gia 0,01 >= 1% (sai so lam tron lon)

p = pd.read_csv(GOC, parse_dates=["time"]).set_index("time").sort_index()
ngay0 = p.index[0]
nk = []                   # nhat ky: (quy_tac, ma, so_quan_sat, ghi_chu)


def xoa(mask, quy_tac, ghi_chu=""):
    """Chuyen o co gia trong mask thanh thieu, ghi nhat ky theo tung ma."""
    global p
    mask = mask & p.notna()
    for ma, n in mask.sum().items():
        if n:
            nk.append((quy_tac, ma, int(n), ghi_chu))
    p = p.mask(mask)


# R1: gia bang 0 -> thieu
xoa(p == 0, "R1", "gia bang 0")

# R2: chi giu giai doan niem yet HOSE
san = pd.read_csv(SAN, parse_dates=["ngay_dau_hose"])
m = pd.DataFrame(False, index=p.index, columns=p.columns)
for r in san.itertuples():
    if r.ma in p.columns:
        m.loc[p.index < r.ngay_dau_hose, r.ma] = True
    else:
        print("CANH BAO: ma trong bang san khong co trong du lieu:", r.ma)
xoa(m, "R2", "truoc ngay dau tren HOSE")

# R3: cua so tam giao dich HNX nam 2021
m = pd.DataFrame(False, index=p.index, columns=p.columns)
for ma in [x for x in R3_MA if x in p.columns]:
    m.loc[R3_TU:R3_DEN, ma] = True
xoa(m, "R3", f"tam giao dich HNX {R3_TU}..{R3_DEN}")

# R4: VHM truoc 17/5/2018
m = pd.DataFrame(False, index=p.index, columns=p.columns)
if R4_MA in p.columns:
    m.loc[p.index < R4_TU, R4_MA] = True
xoa(m, "R4", f"truoc {R4_TU}")

# R5: bo 5 phien dau sau khi ma bat dau giao dich tren HOSE (ma bat dau sau phien dau tien cua du lieu)
m = pd.DataFrame(False, index=p.index, columns=p.columns)
for ma in p.columns:
    idx = p[ma].dropna().index
    if len(idx) and idx[0] > ngay0:
        m.loc[idx[:R5_N], ma] = True
xoa(m, "R5", "5 phien dau tren HOSE")

# R6: loi suat chi khi hai phien lien tiep deu co gia (khong dien gia cu)
ret = p / p.shift(1) - 1

# R7: danh dau (khong xoa)
co_r7 = p < R7_NGUONG
for ma, n in co_r7.sum().items():
    if n:
        nk.append(("R7", ma, int(n), f"danh dau: gia < {R7_NGUONG}"))

# ---- Xuat ----
dai = (p.stack().rename("gia").to_frame()
       .join(ret.stack().rename("ret"))
       .join(co_r7.stack().rename("co_r7")))
dai.index.names = ["date", "ma"]
dai = dai.dropna(subset=["gia"])
dai.reset_index().to_csv("../Data/vn100_sach.csv", index=False)
nk = pd.DataFrame(nk, columns=["quy_tac", "ma", "so_quan_sat", "ghi_chu"])
nk.to_csv("../Data/nhat_ky_lam_sach.csv", index=False)

# ---- Doi chieu voi de cuong ----
de_cuong = {"R1": 3, "R2": 14736, "R3": 338, "R4": 130, "R5": 195, "R7": 98}
print("Quy tac | Code  | De cuong")
for q, k in de_cuong.items():
    n = nk.loc[nk["quy_tac"] == q, "so_quan_sat"].sum()
    print(f"{q:7} | {n:5d} | {k:5d}  {'OK' if n == k else '<-- LECH'}")
print("\nR3 theo ma:", nk[nk.quy_tac == "R3"].set_index("ma")["so_quan_sat"].to_dict())
print("R7 theo ma:", nk[nk.quy_tac == "R7"].set_index("ma")["so_quan_sat"].to_dict())

vuot = ret.abs() > 0.073
print(f"\nKiem tra cheo |r| > 7,3% sau lam sach: {int(vuot.sum().sum())} (de cuong: 2, ANV 1/2017)")
for ma, s in vuot.items():
    for d in s[s].index:
        print(f"   {ma} {d.date()} r = {ret.loc[d, ma]:+.3f}")
n = p.notna().sum(axis=1)
print("\nSo ma co gia/ngay theo nam (TB):", n.groupby(n.index.year).mean().round(1).to_dict())
