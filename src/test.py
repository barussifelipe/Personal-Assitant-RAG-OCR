import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Test a simple CUDA operation
    x = torch.tensor([1.0, 2.0, 3.0])
    x = x.cuda()
    print(f"Tensor on GPU: {x}")
    print(f"Tensor device: {x.device}")
else:
    print("❌ CUDA not available")