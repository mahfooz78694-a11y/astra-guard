# Enterprise MLOps Debugging & Remediation Manual

This manual provides immediate operational solutions for integrating `astra-guard` into complex enterprise MLOps environments.

---

## Operational Failure Modes & Immediate Remediation

### 1. `KeyError: Layer Name Not Found in Module Tree`
* **Symptom:** PyTorch raises a `KeyError` when attempting to register forward hooks on `target_layer`.
* **Root Cause:** Model architecture renames modules or uses sequential indices.
* **Remediation:** Run the Auto-Layer Discovery scanner:
```python
from astra_guard.auto import AutoSubspaceTuner
tuner = AutoSubspaceTuner()
valid_layer = tuner.discover_optimal_layer(model)
guard = ZVILGuard(model, target_layer=valid_layer)
```

### 2. High VRAM Pressure or CUDA Memory Allocation Stalls
* **Symptom:** Micro-latency spikes occur during batch processing under heavy server load.
* **Root Cause:** Available GPU VRAM dropped below safety threshold (e.g., 1000MB remaining).
* **Remediation:** System automatically engages Circuit-Breaker safe pass-through mode to prevent process crashing.
```python
guard = ZVILGuard(model, target_layer="layer4", preallocate_buffers=True, vram_threshold_mb=1000)
```

---
## Cryptographic & DOI Verification
* Registered Research DOI: `10.5281/zenodo.21532310`
* Lead Research Architects: MD Mahfooz & Alsaad Alam