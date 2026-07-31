import pytest
import torch
from astra_guard.core import VORTEXSVDEngine

def test_gradient_freezing():
    engine = VORTEXSVDEngine(rank_k=4)
    acts = torch.randn(10, 16)
    engine.calibrate_subspace(acts)

    assert engine.P_parallel.requires_grad == False

    test_input = torch.randn(5, 16, requires_grad=True)
    out = engine.deflect_activations(test_input)

    # We deflect activations inside torch.no_grad(), so out shouldn't have grad_fn
    assert out.requires_grad == False
    assert out.grad_fn is None
