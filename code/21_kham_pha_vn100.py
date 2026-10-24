"""Buoc 1.2 - Kham pha VN-100 truoc khi lam sach: ap R1 (gia 0 -> thieu), R6 (khong dien gia cu),
roi tim cac ma co loi suat vuot bien do HOSE (+-7%, dung sai 0,3 diem %) -> dau hieu dang o san khac
(HNX +-10%, UPCOM +-15%) hoac phien dau niem yet (+-20%).
Chay tu D:\\LuanVan\\Code:  python 21_kham_pha_vn100.py
"""
import pandas as pd

wide = pd.read_csv("../Data/vn100_2017_2024.csv", parse_dates=["time"]).set_index("time").sort_index()
print(f"{wide.shape[0]} phien, {wide.shape[1]} ma | {wide.index[0].date()} -> {wide.index[-1].date()}")

# R1: gia bang 0 -> thieu
zero = (wide == 0)
print("R1 - gia bang 0:", int(zero.sum().sum()), "quan sat |", zero.sum()[zero.sum() > 0].to_dict())
wide = wide.mask(zero)

# R6: loi suat chi tinh khi ca hai phien lien tiep deu co gia (khong dien gia cu)
ret = wide / wide.shift(1) - 1
print("So quan sat co gia:", int(wide.notna().sum().sum()), "| co loi suat:", int(ret.notna().sum().sum()))

# Ngay dau co du lieu cua tung ma
dau = wide.apply(lambda s: s.first_valid_index())
print(f"\nMa co du lieu tu phien dau: {(dau == wide.index[0]).sum()} | bat dau muon hon: {(dau > wide.index[0]).sum()}")

# Vuot bien do HOSE
vuot = ret.abs() > 0.073
n_vuot = vuot.sum().sort_values(ascending=False)
print(f"\nTong quan sat |r| > 7,3%: {int(n_vuot.sum())} | so ma lien quan: {(n_vuot > 0).sum()}")

bang = pd.DataFrame({
    "n_vuot": n_vuot,
    "vuot_dau": vuot.apply(lambda s: s[s].index.min()),
    "vuot_cuoi": vuot.apply(lambda s: s[s].index.max()),
    "max_abs_r": ret.abs().max().round(3),
    "ngay_dau_dl": dau,
})
bang = bang[bang["n_vuot"] > 0]
for c in ["vuot_dau", "vuot_cuoi", "ngay_dau_dl"]:
    bang[c] = bang[c].dt.date
print("\nCac ma co loi suat vuot +-7,3% (sap theo so lan):")
print(bang.to_string())
bang.to_csv("ket_qua_21_vuot_bien_do.csv")
