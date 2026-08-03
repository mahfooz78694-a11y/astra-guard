import torch
import time
from astra_guard import ZVILGuard
import torch.nn as nn

class Dummy(nn.Module):
    def forward(self, x): return x

model = Dummy()
guard = ZVILGuard(model, '', rank_k=4)
guard.target_module = model # mock
guard.engine.calibrate_subspace(torch.randn(4, 3, 8, 8))
guard.attach()

t0 = time.time()
for _ in range(10):
    x = torch.randn(1, 3, 256, 256)
    model(x)
print("10 iters 256x256 time:", time.time() - t0)
