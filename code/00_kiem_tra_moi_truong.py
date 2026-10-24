"""Buoc 2.0 - Kiem tra moi truong. Chay: python 00_kiem_tra_moi_truong.py"""
import sys, platform, importlib, shutil

print("Python    :", sys.version.split()[0], "| OS:", platform.platform())
ok_py = (3, 9) <= sys.version_info[:2] <= (3, 12)
print("  -> Python 3.9-3.12:", "OK" if ok_py else "CAN DOI (khuyen nghi 3.10)")

print("git       :", "co" if shutil.which("git") else "KHONG CO (tai ZIP tu GitHub cung duoc)")

for pkg in ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "torch", "tabpfn"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:10}: {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"{pkg:10}: CHUA CAI")

try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU       : {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("GPU       : khong thay CUDA -> chi chay suy luan CPU; tinh chinh dung Colab/Kaggle")
except ImportError:
    pass
