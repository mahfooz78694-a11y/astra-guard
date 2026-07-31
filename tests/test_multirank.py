import pytest
import torch
import torch.nn as nn
from astra_guard.core import VORTEXSVDEngine

def test_multirank_2d():
    engine = VORTEXSVDEngine(rank_k=2)
    acts = torch.randn(10, 4)
    assert engine.calibrate_subspace(acts)

    test_input = torch.randn(5, 4)
    out = engine.deflect_activations(test_input)
    assert out.shape == (5, 4)

def test_multirank_3d():
    engine = VORTEXSVDEngine(rank_k=4)
    acts = torch.randn(10, 16, 8)
    assert engine.calibrate_subspace(acts)

    test_input = torch.randn(2, 16, 8)
    out = engine.deflect_activations(test_input)
    assert out.shape == (2, 16, 8)

def test_multirank_4d():
    engine = VORTEXSVDEngine(rank_k=8)
    acts = torch.randn(10, 16, 32, 32)
    assert engine.calibrate_subspace(acts)

    test_input = torch.randn(2, 16, 32, 32)
    out = engine.deflect_activations(test_input)
    assert out.shape == (2, 16, 32, 32)

def test_unroll_restore_non_contiguous():
    engine = VORTEXSVDEngine(rank_k=2)
    # create non-contiguous tensor
    acts = torch.randn(10, 32, 16, 16).transpose(1, 2) # shape (10, 16, 32, 16) but not contiguous
    assert not acts.is_contiguous()

    assert engine.calibrate_subspace(acts)

    test_input = torch.randn(2, 32, 16, 16).transpose(1, 2)
    out = engine.deflect_activations(test_input)
    assert out.shape == (2, 16, 32, 16)
