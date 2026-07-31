import pytest
import torch
import torch.nn as nn
from astra_guard.hooks import ZVILGuard

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def test_hook_attach_detach():
    model = DummyModel()
    guard = ZVILGuard(model, target_layer='fc1', rank_k=4)

    assert not guard.is_attached
    assert guard.hook_handle is None

    # Needs calibration to attach fully realistically or just attach testing
    guard.attach()
    assert guard.is_attached
    assert guard.hook_handle is not None

    guard.detach()
    assert not guard.is_attached

def test_hook_interception():
    model = DummyModel()
    guard = ZVILGuard(model, target_layer='fc1', rank_k=4)

    # dummy calibration
    dataloader = [(torch.randn(2, 10),) for _ in range(3)]
    guard.calibrate(dataloader, num_batches=2)

    guard.attach()
    out = model(torch.randn(2, 10))
    assert out.shape == (2, 5)
    guard.detach()
