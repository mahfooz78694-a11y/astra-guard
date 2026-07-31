import torch
import torch.nn as nn
from astra_guard.core import VORTEXSVDEngine
from astra_guard.hooks import ZVILGuard

def test_extreme_dynamic_batches():
    engine = VORTEXSVDEngine(rank_k=2)

    # Needs calibration first
    base_acts = torch.randn(10, 4)
    assert engine.calibrate_subspace(base_acts)

    batch_sizes = [1, 128, 256, 512]

    for b in batch_sizes:
        # 2D tensor
        t2d = torch.randn(b, 4)
        out2d = engine.deflect_activations(t2d)
        assert out2d.shape == t2d.shape

        # 3D tensor
        t3d = torch.randn(b, 5, 4)
        out3d = engine.deflect_activations(t3d)
        assert out3d.shape == t3d.shape

        # 4D tensor
        t4d = torch.randn(b, 4, 8, 8)
        out4d = engine.deflect_activations(t4d)
        assert out4d.shape == t4d.shape

def test_singular_value_degeneracy():
    engine = VORTEXSVDEngine(rank_k=2)

    # Zero tensors
    zero_t = torch.zeros(10, 4)
    assert engine.calibrate_subspace(zero_t)
    out_zero = engine.deflect_activations(zero_t)
    assert not torch.isnan(out_zero).any()

    # Inf/NaN tensors
    inf_t = torch.tensor([[float('inf'), float('-inf'), float('nan'), 0.0] for _ in range(10)])
    assert engine.calibrate_subspace(inf_t)
    out_inf = engine.deflect_activations(inf_t)
    assert not torch.isnan(out_inf).any()

    # Identity matrix
    id_t = torch.eye(4).repeat(10, 1)
    assert engine.calibrate_subspace(id_t)
    out_id = engine.deflect_activations(id_t)
    assert not torch.isnan(out_id).any()

def test_autograd_immunity():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    guard = ZVILGuard(model, target_layer='linear', rank_k=2)

    # Calibrate
    class DummyDataLoader:
        def __iter__(self):
            yield torch.randn(10, 4)

    guard.calibrate(DummyDataLoader(), num_batches=1)
    guard.attach()

    # Needs a parameter that requires grad in the end
    # Because ZVILGuard modifies output, which uses torch.no_grad() internally

    # Wait, deflect_activations uses `with torch.no_grad():`
    # This detaches the tensor from the computational graph, which is exactly the point of the firewall.
    # So `out = model(x)` will have `requires_grad=False`.
    # Let's add a linear layer *after* the guard, so we can run a backward pass through *that*.

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(4, 4)
            self.linear2 = nn.Linear(4, 4)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    model2 = TestModel()
    guard2 = ZVILGuard(model2, target_layer='linear1', rank_k=2)
    guard2.calibrate(DummyDataLoader(), num_batches=1)
    guard2.attach()

    x = torch.randn(10, 4, requires_grad=True)
    out = model2(x)
    loss = out.sum()
    loss.backward()

    assert guard2.engine.P_parallel.requires_grad == False
    assert guard2.engine.P_parallel.grad is None
