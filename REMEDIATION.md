# Enterprise MLOps Debugging & Remediation Manual

### 1. KeyError Layer Name Not Found
Run Auto-Layer Discovery scanner:
```python
from astra_guard.auto import AutoSubspaceTuner
tuner = AutoSubspaceTuner()
valid_layer = tuner.discover_optimal_layer(model)
guard = ZVILGuard(model, target_layer=valid_layer)
```

### 2. High VRAM Pressure
System automatically engages Circuit-Breaker safe pass-through mode.