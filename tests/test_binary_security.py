import subprocess
import glob
import os
import torch
import pytest
from astra_guard.core import VORTEXSVDEngine, AstraSecurityException

def test_binary_stripping():
    # Check in multiple possible locations: build directory or inplace directory
    so_files = glob.glob('build/lib.*/**/*.so', recursive=True) + glob.glob('astra_guard/*.so')
    if not so_files:
        pytest.fail("No shared objects found! The binary audit must run against a compiled .so file.")

    so_path = so_files[0]

    # nm check
    try:
        nm_out = subprocess.check_output(['nm', '-D', so_path], stderr=subprocess.STDOUT).decode('utf-8')
    except subprocess.CalledProcessError as e:
        nm_out = e.output.decode('utf-8')

    if "no symbols" not in nm_out.lower():
        lines = nm_out.strip().split('\n')
        for line in lines:
            if line:
                sym = line.split()[-1]
                # PyInit_core should be allowed, but internal SVD stuff shouldn't
                if sym != 'PyInit_core' and 'VORTEXSVDEngine' in sym:
                     pytest.fail(f"Leaked symbol in dynamic table: {sym}")

    # strings check
    try:
         strings_out = subprocess.check_output(['strings', so_path]).decode('utf-8')
    except subprocess.CalledProcessError as e:
         strings_out = e.output.decode('utf-8')

    assert '0.15' not in strings_out
    assert '1e-6' not in strings_out
    assert '1e-7' not in strings_out
    assert 'nullspace_projection' not in strings_out


def test_stack_trace_leakage_mitigation():
    engine = VORTEXSVDEngine(rank_k=2)
    # Calibrate on dummy data first to initialize matrices
    dummy_data = torch.randn(10, 10)
    engine.calibrate_subspace(dummy_data)

    # Passing malformed input (string instead of tensor)
    with pytest.raises(AstraSecurityException, match="Processing Failed"):
        engine.deflect_activations("this is not a tensor")

    # Passing mismatched shape tensor to trigger internal failure
    with pytest.raises(AstraSecurityException, match="Processing Failed"):
        bad_tensor = torch.randn(1, 2)
        # This causes an inner matmul dimension mismatch inside the C++ wrapper
        engine.deflect_activations(bad_tensor)

def test_security_policy_and_license():
    # Verify SECURITY.md and LICENSE exist and are accessible for compliance
    assert os.path.exists('SECURITY.md'), "SECURITY.md is missing from the repository"
    assert os.path.exists('LICENSE'), "LICENSE is missing from the repository"

    with open('SECURITY.md', 'r') as f:
        content = f.read()
        assert len(content.strip()) > 0, "SECURITY.md is empty"
