import torch
import pytest
import concurrent.futures
import psutil
import os
import gc
from unittest.mock import patch
from astra_guard.core import VORTEXSVDEngine

def test_dos_protection():
    engine = VORTEXSVDEngine(rank_k=2)
    # Ultra-high resolution input 8K x 8K
    x_huge = torch.randn(8192, 8192)

    # Calibrate with huge tensor, shouldn't lock up or OOM
    engine.calibrate_subspace(x_huge)

    # Deflect huge tensor
    deflected = engine.deflect_activations(x_huge)

    # Result should be cropped/downsampled by logic we'll add
    # Specifically it crops to max 2048 x 2048 during _unroll_to_2d conceptually,
    # but the returned tensor matches orig_shape based on how _restore_shape works.
    # The actual processing inside deflect_activations will have cropped the flat tensor.
    # We just want to ensure it succeeds.
    assert deflected.shape == x_huge.shape

def test_concurrency_memory_leak():
    engine = VORTEXSVDEngine(rank_k=4)
    engine.calibrate_subspace(torch.randn(128, 128))

    def run_inference():
        x = torch.randn(128, 128)
        engine.deflect_activations(x)
        return True

    # 1000 inferences across 16 threads
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for _ in range(1000):
            futures.append(executor.submit(run_inference))

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    # Wait for all to finish
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    mem_after = process.memory_info().rss

    assert len(results) == 1000
    assert all(results)

    # Assert memory remains flat (allow small overhead margin like 5MB)
    assert (mem_after - mem_before) < 5 * 1024 * 1024, "Memory leak detected during concurrency stress test"
