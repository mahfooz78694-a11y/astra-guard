import torch
import torch.nn as nn
import time
import os
import psutil
from astra_guard import ZVILGuard

class DummyLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        return self.fc(x)

class DummyTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(768, 768)

    def forward(self, x):
        return self.attn(x)

class DummyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def measure_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)


def benchmark_layer(model, layer_name, input_tensor, num_warmup=10, num_iters=100):
    device = input_tensor.device
    model = model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_tensor)

    # Baseline Latency
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    mem_before = measure_memory_mb()

    with torch.no_grad():
        for _ in range(num_iters):
            model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_time = (time.time() - start_time) / num_iters * 1000  # ms
    mem_after_baseline = measure_memory_mb()

    # Create dummy dataloader for calibration
    class DummyLoader:
        def __init__(self, x):
            self.x = x
        def __iter__(self):
            yield (self.x,)

    # Attach Guard
    guard = ZVILGuard(model, target_layer=layer_name, rank_k=16)
    guard.calibrate(DummyLoader(input_tensor), num_batches=1)
    guard.attach()

    # Warmup protected
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_tensor)

    # Protected Latency
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iters):
            model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    protected_time = (time.time() - start_time) / num_iters * 1000  # ms
    mem_after_protected = measure_memory_mb()

    guard.detach()

    net_latency = protected_time - baseline_time
    mem_diff = mem_after_protected - mem_after_baseline

    print(f"--- Benchmark for {layer_name} ---")
    print(f"Tensor Shape: {list(input_tensor.shape)}")
    print(f"Baseline Latency:  {baseline_time:.4f} ms")
    print(f"Protected Latency: {protected_time:.4f} ms")
    print(f"Net Overhead:      {net_latency:.4f} ms")
    print(f"Memory Diff:       {mem_diff:.4f} MB\n")
    return baseline_time, protected_time, net_latency, mem_diff

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}...\n")

    # 2D Linear
    model_2d = DummyLinearModel()
    input_2d = torch.randn(32, 512, device=device)
    benchmark_layer(model_2d, "fc", input_2d)

    # 3D Transformer Sequence
    model_3d = DummyTransformerModel()
    input_3d = torch.randn(32, 128, 768, device=device)
    benchmark_layer(model_3d, "attn", input_3d)

    # 4D Conv2D Feature Map
    model_4d = DummyConvModel()
    input_4d = torch.randn(32, 64, 56, 56, device=device)
    benchmark_layer(model_4d, "conv", input_4d)
