"""Buoc 2.2 - Thu nap trong so FinPFN cua tac gia va du bao tren du lieu gia (ngau nhien).
Muc dich: chi kiem tra tabpfn + checkpoint + GPU chay duoc. Con so du bao KHONG co y nghia.
Chay tu D:\\LuanVan\\Code:  python 02_thu_nap_finpfn.py
"""
import time
import numpy as np
import torch
from tabpfn import TabPFNRegressor

CKPT = "FinPFN/models/finpfn_30feats_csi500.ckpt"
rng = np.random.default_rng(0)

# Mo phong 1 nhom ngu canh nhu bai goc: 50 ma ngay t-1 (co nhan) -> du bao 50 ma ngay t, 30 dac trung
X_train = rng.standard_normal((50, 30)).astype(np.float32)
y_train = rng.standard_normal(50).astype(np.float32)
X_test = rng.standard_normal((50, 30)).astype(np.float32)

for device in ["cuda", "cpu"]:
    t = time.time()
    reg = TabPFNRegressor(model_path=CKPT, device=device).fit(X_train, y_train)
    out = reg.predict(X_test, output_type="full")
    print(f"{device:4}: {time.time() - t:5.2f}s | median[:3] = {np.round(out['median'][:3], 4)}")

print("GPU RAM da dung toi da:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
print("Buoc 2.2: DAT")
