"""Buoc 4e - KHOA CAU HINH truoc khi cham tap kiem tra. Ghi file JSON kem dau van tay SHA-256 cua
mo hinh, du lieu va ma nguon. Moi buoc sau (5, 6) se kiem tra lai dau van tay nay truoc khi chay.
Chay tu D:\\LuanVan\\Code:  python 36_khoa_cau_hinh.py
"""
import hashlib, json, datetime
from pathlib import Path

KQ = Path("../Ketqua")


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


TEP = {
    "mo_hinh_M1": KQ / "tinh_chinh/m1_best.ckpt",
    "mo_hinh_FinPFN_MC": KQ / "tinh_chinh/mc_best.ckpt",
    "dac_trung_csi500": Path("../Data/csi500_dac_trung.parquet"),
    "dac_trung_csi300": Path("../Data/csi300_dac_trung.parquet"),
    "dac_trung_vn100": Path("../Data/vn100_dac_trung.parquet"),
    "nen_csi500": Path("../Data/nen_csi500.parquet"),
    "nen_csi300": Path("../Data/nen_csi300.parquet"),
    "nen_vn100": Path("../Data/nen_vn100.parquet"),
    "code_finpfn_mc": Path("finpfn_mc.py"),
    "code_dac_trung": Path("dac_trung.py"),
    "code_mo_hinh_nen": Path("mo_hinh_nen.py"),
}

cau_hinh = {
    "thoi_diem_khoa": datetime.datetime.now().isoformat(timespec="seconds"),
    "cau_hinh_chon": {
        "mo_hinh": "FinPFN-MC", "K_phan_hoach": 20, "co_nhom_suy_luan": 50,
        "co_nhom_tinh_chinh": [30, 60], "cot_xep_chong": ["ridge", "lgbm"], "dynamic_stack": False,
        "du_bao_diem": "trung vi", "suy_luan": "lan truyen thuan truc tiep, thu tu cot co dinh",
    },
    "doi_chung": {
        "M1": "FinPFN goc tu tinh chinh, K=1, nhom 50",
        "M1+K": "M1, K=20",
        "Tron deu (chinh, dat truoc)": "hang trung binh cua M1 (K=1), Ridge, LGBM - doi chung chinh cua H1b",
        "Tron phu (kiem tra do ben)": "hang trung binh cua M1+K (K=20), Ridge, LGBM - chi bao cao, KHONG doi tieu chi",
        "Tron IC 60n": "trong so = IC TB 60 ngay truoc (khong am), <20 ngay -> deu",
        "Ridge, LGBM": "cua so mo rong, hoc lai dau moi thang",
    },
    "vong_thu_2021": {
        "IC": {"M1": 0.0590, "M1+K": 0.0656, "MC_K1": 0.0686, "MC_K20": 0.0729, "MC_K50": 0.0736,
               "Ridge": 0.0609, "LGBM": 0.0582, "Tron deu": 0.0673, "Tron IC 60n": 0.0672},
        "on_dinh": {"M1": 0.265, "MC_K20": 0.327},
        "tieu_chi": "MC_K20: -M1 = +0.0138 (t 6.29); -Tron deu = +0.0055 (t 1.62) -> QUA",
        "so_bien_the_da_thu": 3, "bien_the": ["MC K=1", "MC K=20", "MC K=50"],
        "MC_v2": "khong tinh chinh (MC K=20 da qua tieu chi khoa, theo quy tac dat truoc)",
        "chon_K": "K=20 phien ban chinh (K=50 chi hon +0.0007 < nguong 0.002, chi phi suy luan x2.5); K=50 la kiem tra do ben",
    },
    "kiem_dinh": {
        "H1": "CSI 5/2023-6/2026, vu tru 100 ma, 10 phan hoach x 5 vu tru/ngay, hat giong co dinh; "
              "chenh lech IC MC - M1 theo ngay; Newey-West do tre 5; p < 0.05 hai phia",
        "H1b": "nhu H1; can duoi KTC 95% mot phia cua (MC - Tron deu chinh) > -0.005",
        "kiem_tra_do_ben": [
            "MC - Tron phu (M1+K, Ridge, LGBM): cung cach tinh nhu H1b; bao cao minh bach, khong thay tieu chi chinh",
            "MC_K50 thay MC_K20: bao cao IC va chenh lech so voi K20",
        ],
        "lap_lai": "CSI 300, cung cau hinh, cung giai doan",
    },
    "dau_van_tay_sha256": {k: (sha(p) if p.exists() else "KHONG TIM THAY") for k, p in TEP.items()},
}

thieu = [k for k, v in cau_hinh["dau_van_tay_sha256"].items() if v == "KHONG TIM THAY"]
if thieu:
    raise SystemExit(f"Thieu tep: {thieu}. Kiem tra duong dan roi chay lai.")
f = KQ / "cau_hinh_khoa.json"
if f.exists():
    raise SystemExit(f"{f} da ton tai - cau hinh da khoa. Khong ghi de.")
f.write_text(json.dumps(cau_hinh, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"DA KHOA: {f}")
for k, v in cau_hinh["dau_van_tay_sha256"].items():
    print(f"  {k:20} {v[:16]}...")
