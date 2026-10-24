"""Buoc 3.5 - Tai gia ngay tu Baostock cho moi ma tung thuoc CSI 500 / CSI 300.
Moi ma 1 file: ../Data/baostock/gia/<code>.csv. Chay lai se bo qua ma da tai xong.
Thu truoc 5 ma:   python 12_tai_gia_baostock.py
Tai toan bo:      python 12_tai_gia_baostock.py tat_ca
"""
import sys, time
from pathlib import Path
import baostock as bs
import pandas as pd

BAT_DAU, KET_THUC = "2015-07-01", "2026-07-31"
COT = "date,code,close,preclose,pctChg,tradestatus,isST"
GOC = Path("../Data/baostock")
OUT = GOC / "gia"; OUT.mkdir(parents=True, exist_ok=True)

codes = sorted(set(pd.read_csv(GOC / "thanh_vien_csi500.csv")["code"])
               | set(pd.read_csv(GOC / "thanh_vien_csi300.csv")["code"]))
print(f"Tong so ma can tai: {len(codes)}")
if len(sys.argv) < 2 or sys.argv[1] != "tat_ca":
    codes = codes[:3] + codes[-2:]           # thu 5 ma: 3 ma sh dau danh sach + 2 ma sz cuoi
    print("CHE DO THU 5 MA:", codes)


def tai_mot_ma(code):
    rs = bs.query_history_k_data_plus(code, COT, start_date=BAT_DAU, end_date=KET_THUC,
                                      frequency="d", adjustflag="1")
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


bs.login()
t0, loi = time.time(), []
for i, code in enumerate(codes, 1):
    f = OUT / f"{code}.csv"
    if f.exists():
        continue
    for lan in range(3):
        try:
            tai_mot_ma(code).to_csv(f, index=False)
            break
        except Exception as e:
            bs.logout(); time.sleep(5); bs.login()
    else:
        loi.append(code)
    con = (time.time() - t0) / i * (len(codes) - i) / 60
    print(f"[{i}/{len(codes)}] {code} | con lai ~{con:.0f} phut", end="\r")
bs.logout()

print(f"\n\nXong. Ma loi: {loi or 'khong co'}")
for code in codes[:5]:
    d = pd.read_csv(OUT / f"{code}.csv")
    print(f"{code}: {len(d)} phien | {d['date'].min()} -> {d['date'].max()} | "
          f"ngung GD: {(d['tradestatus'] == 0).sum()} | ST: {(d['isST'] == 1).sum()}")
