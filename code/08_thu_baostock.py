"""Buoc 3.1 - Thu ket noi Baostock: lay danh sach CSI 500 tai 1 thoi diem + gia 1 ma.
Chi kiem tra ket noi va dinh dang du lieu, chua tai hang loat.
Chay tu D:\\LuanVan\\Code:  python 08_thu_baostock.py
"""
import baostock as bs
import pandas as pd


def bang(rs):
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


lg = bs.login()
print("Dang nhap:", lg.error_code, lg.error_msg)

# 1) Thanh phan CSI 500 tai hai thoi diem (de xem co thay doi theo thoi gian)
for ngay in ["2021-01-04", "2026-06-01"]:
    tp = bang(bs.query_zz500_stocks(date=ngay))
    print(f"CSI 500 ngay {ngay}: {len(tp)} ma | vi du: {tp['code'].head(3).tolist() if len(tp) else '-'}")

# 2) Gia ngay cua 1 ma (dieu chinh hau quyen: adjustflag='1')
k = bang(bs.query_history_k_data_plus(
    tp["code"].iloc[0] if len(tp) else "sh.600008",
    "date,code,close,pctChg,tradestatus,isST",
    start_date="2026-05-01", end_date="2026-10-16", frequency="d", adjustflag="1"))
print(f"\nGia {k['code'].iloc[0] if len(k) else '?'}: {len(k)} phien, ngay cuoi co du lieu: "
      f"{k['date'].iloc[-1] if len(k) else '-'}")
print(k.tail(3).to_string(index=False))

bs.logout()
