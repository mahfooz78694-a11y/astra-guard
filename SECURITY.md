# Security & Integrity Specification — VORTEX-SVD Engine v2.0
* **Zenodo DOI:** `10.5281/zenodo.21532310`
* **Package Identity:** `astra-guard` (v2.0.0)
* **Lead Architects:** MD Mahfooz & Alsaad Alam

## Anti-Tamper & Binary Hardening Specifications
1. **GCC Symbol Stripping (`-s`):** C++ symbol tables permanently stripped.
2. **Runtime SHA-256 Shield:** Hash verification on `.so` load.
3. **Gradient Detachment (`requires_grad=False`):** Non-differentiable CUDA subspace projection.
4. **RAM Buffer Purging:** Garbage collection (`gc.collect()`) after SVD calibration.
