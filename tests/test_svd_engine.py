import pytest
import torch
from astra_guard.core import VORTEXSVDEngine

def test_svd_engine_calibration():
    engine = VORTEXSVDEngine(rank_k=8, enable_basis_hopping=False, enable_watermark=False)
    acts = torch.randn(100, 32)
    assert engine.calibrate_subspace(acts)
    assert engine.P_parallel is not None
    assert engine.V_k is not None
    assert engine.P_parallel.shape == (32, 32)
    assert engine.P_parallel.dtype == torch.float64

def test_svd_zero_variance_fallback():
    engine = VORTEXSVDEngine(rank_k=4)
    # acts with zero variance
    acts = torch.ones(10, 16)
    assert engine.calibrate_subspace(acts)
    assert engine.P_parallel is not None

def test_qr_fallback_mocked(monkeypatch):
    def mock_svd(*args, **kwargs):
        raise RuntimeError("SVD did not converge")

    monkeypatch.setattr(torch.linalg, "svd", mock_svd)
    engine = VORTEXSVDEngine(rank_k=4)
    acts = torch.randn(20, 10)
    # Should trigger QR fallback
    assert engine.calibrate_subspace(acts)
    assert engine.V_k is not None
    assert engine.P_parallel is not None

def test_precision_casting():
    engine = VORTEXSVDEngine(rank_k=4, enable_basis_hopping=False, enable_watermark=False)
    acts = torch.randn(20, 10, dtype=torch.float32)
    engine.calibrate_subspace(acts)

    test_input = torch.randn(5, 10, dtype=torch.float32)
    out = engine.deflect_activations(test_input)
    assert out.dtype == torch.float32

    test_input_16 = torch.randn(5, 10, dtype=torch.float16)
    out_16 = engine.deflect_activations(test_input_16)
    assert out_16.dtype == torch.float16
