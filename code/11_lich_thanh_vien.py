"""Buoc 3.4 - Dung lich thanh vien chi so voi ngay hieu luc chuan.
Van de: anh chup Baostock tre 1-4 thang o mot so ky (2021-08, 2022-02, 2023-04).
Quy tac: dot dieu chinh dinh ky (>=5 ma) gan vao ngay hieu luc theo quy dinh CSI:
  phien ke tiep sau thu Sau thu hai cua thang 6 / thang 12 (ky gan nhat truoc ngay chup).
  Thay doi le (<5 ma, vd huy niem yet) giu theo ngay chup (sai so toi da ~1 thang, khai bao).
Luu: ../Data/baostock/thanh_vien_csi500.csv, thanh_vien_csi300.csv  (code, tu_ngay, den_ngay)
Chay tu D:\\LuanVan\\Code:  python 11_lich_thanh_vien.py
"""
import pandas as pd

NGUONG = 5        # CSI 300 co ky chi doi 5-9 ma (vd 7/2023, 7/2025)


def ngay_hieu_luc_dinh_ky(nam, thang):
    """Ngay thuong ke tiep sau thu Sau thu hai cua thang (lich thuong, chua tru nghi le)."""
    fridays = pd.date_range(f"{nam}-{thang:02d}-01", periods=31, freq="D")
    fridays = fridays[(fridays.month == thang) & (fridays.weekday == 4)]
    return fridays[1] + pd.offsets.BDay(1)


KY = sorted(ngay_hieu_luc_dinh_ky(y, m) for y in range(2015, 2027) for m in (6, 12))

for ten in ["csi500", "csi300"]:
    tp = pd.read_csv(f"../Data/baostock/thanh_phan_{ten}.csv")
    ds = tp.groupby("ngay_chup")["code"].apply(set).sort_index()
    ds.index = pd.to_datetime(ds.index)

    su_kien, bang_gan = [], []
    for (d_a, a), (d_b, b) in zip(ds.iloc[:-1].items(), ds.iloc[1:].items()):
        vao, ra = b - a, a - b
        if not vao and not ra:
            continue
        if len(vao) >= NGUONG:
            hl = max(k for k in KY if k <= d_b)
            bang_gan.append((d_b.date(), hl.date(), (d_b - hl).days, len(vao)))
        else:
            hl = d_b
        su_kien += [(hl, c, 1) for c in vao] + [(hl, c, -1) for c in ra]

    # Chuyen su kien -> khoang thanh vien [tu_ngay, den_ngay)
    sk = pd.DataFrame(su_kien, columns=["ngay", "code", "loai"]).sort_values("ngay")
    mo = {c: ds.index[0] for c in ds.iloc[0]}
    khoang = []
    for ngay, code, loai in sk.itertuples(index=False):
        if loai == 1:
            mo.setdefault(code, ngay)
        elif code in mo:
            khoang.append((code, mo.pop(code), ngay))
    khoang += [(c, t, pd.NaT) for c, t in mo.items()]
    kq = pd.DataFrame(khoang, columns=["code", "tu_ngay", "den_ngay"])
    kq.to_csv(f"../Data/baostock/thanh_vien_{ten}.csv", index=False)

    print(f"\n=== {ten.upper()} ===")
    print(pd.DataFrame(bang_gan, columns=["ngay_chup", "ngay_hieu_luc", "tre_ngay", "so_ma"]).to_string(index=False))
    for d in ["2019-03-01", "2023-02-01", "2026-06-01"]:
        d = pd.Timestamp(d)
        n = ((kq["tu_ngay"] <= d) & (kq["den_ngay"].isna() | (kq["den_ngay"] > d))).sum()
        print(f"So thanh vien ngay {d.date()}: {n}")
    print(f"So khoang: {len(kq)} | so ma: {kq['code'].nunique()}")
