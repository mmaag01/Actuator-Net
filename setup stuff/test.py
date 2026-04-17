import torch
import time

print(f"Using device: {torch.cuda.get_device_name(0)}")

# GPU test
device = torch.device('cuda')
x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)

start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.4f}s")

# CPU test
x_cpu = x.cpu()
y_cpu = y.cpu()
start = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f}s")
print(f"GPU is {cpu_time/gpu_time:.1f}x faster")