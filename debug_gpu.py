# Save this as debug_gpu.py and run it
import torch
import yaml
import os

print("=== CUDA Environment ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")

print("\n=== Parameters File ===")
try:
    with open("mlcube/workspace/parameters.yaml") as f:
        params = yaml.safe_load(f)
    print(f"gpu_ids in params: {params.get('gpu_ids', 'NOT FOUND')}")
    for key, value in params.items():
        print(f"{key}: {value}")
except Exception as e:
    print(f"Error reading parameters: {e}")

print("\n=== Environment Variables ===")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

print("\n=== Test GPU Tensor ===")
if torch.cuda.is_available():
    try:
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.mm(x, y)
        print(f"GPU computation successful: {z.device}")
    except Exception as e:
        print(f"GPU computation failed: {e}")
else:
    print("CUDA not available for tensor test")