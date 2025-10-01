import numpy, torch, scipy, numba, matplotlib

print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("Numba:", numba.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
