import torch

# More detailed information about the GPU and PyTorch version
print("*" * 50)
print("GPU information:")
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("Current CUDA device:", torch.cuda.current_device())
print("Number of CUDA devices:", torch.cuda.device_count())
print("PyTorch version:", torch.__version__)