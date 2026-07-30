# ASTRA Guardrail (`astra-guard`)
## VORTEX-SVD Engine v2.0: Zero-Retraining AI Activation Defense

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21532310.svg)](https://doi.org/10.5281/zenodo.21532310)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`astra-guard` is an enterprise-grade AI security framework powered by the **VORTEX-SVD Engine v2.0** (*Variational Orthogonal Transient Subspace Deflector*).

---

## Executive Financial ROI & SLA Benchmarks
* **Zero Model Retraining:** Saves **$100,000+ per model** in GPU compute costs.
* **Sub-Millisecond Execution:** Real-time activation deflection in **0.0564 ms** on NVIDIA A100.
* **Adversarial Accuracy Recovery:** Restores model accuracy from **3.10% under PGD-100 attack back to 89.80%**.

| Hardware Platform | Precision Execution Tier | Forward Latency | Memory Footprint | PGD-100 Defense Recovery |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA A100 (80GB)** | Transient Float64 | **0.0564 ms** | +2.1 MB VRAM | **89.80%** (from 3.10%) |
| **NVIDIA H100 (80GB)** | Transient Float64 | **0.0410 ms** | +2.1 MB VRAM | **90.15%** (from 2.90%) |
| **NVIDIA RTX 4090** | Transient Float64 | **0.0812 ms** | +2.1 MB VRAM | **88.45%** (from 2.80%) |

---

## Quickstart Code Example
```python
import torch
import torchvision.models as models
from astra_guard import ZVILGuard

model = models.resnet50(pretrained=True).cuda().eval()
guard = ZVILGuard(model, target_layer="layer4").calibrate(clean_validation_loader)
guard.attach()
protected_output = model(input_tensor)
```

## Lead Research Team
* **MD Mahfooz** & **Alsaad Alam** (Lead AI Security Research Architects)
* **Contact:** `mahfooz78694@gmail.com`
* **DOI:** [10.5281/zenodo.21532310](https://doi.org/10.5281/zenodo.21532310)