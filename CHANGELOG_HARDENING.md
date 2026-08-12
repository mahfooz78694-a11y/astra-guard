# Security Hardening & Refactoring Changelog

## 1. Mathematical Terminology & README Refactoring
- Corrected subspace terminology referencing $V_k$ as spanning the **Dominant Principal Subspace** $\mathcal{S}_k$.
- Defined adversarial perturbations effectively bounded to concentrate predominantly in the **Residual Subspace** $\mathcal{S}_k^\perp$.
- Replaced absolute bounds with empirically verified empirical deflection limits ($\ell_\infty, \ell_2$).
- Added Clean Signal Truncation Error Bound formula ($\tilde{X}_{\text{deflected}} = \tilde{X}_{\text{clean}} - \epsilon_{\text{truncation}} + P_k \delta$).
- Described Autograd Detachment correctly as **Autograd Graph Severance** and updated the stochastic BPDA rotation logic notation.

## 2. Core C++ Build Enhancements & Obfuscation
- Updated `setup.py` extension compilation flags to incorporate extreme hardening: `-flto` (Link-Time Optimization) and `-fstack-protector-strong`.
- Validated that symbol visibility was successfully controlled via `-fvisibility=hidden` and unstripped logic was resolved.

## 3. Real-Time Memory Safety & Leak Mitigation
- Introduced POSIX-based `mlock()` memory pinning block implementation into the `VORTEXSVDEngine` class configuration.
- Explicitly sanitized transient data matrices (`x_f64.zero_()`, `P_f64.zero_()`, `deflected_f64.zero_()`) during the matrix multiplication lifecycle to eliminate any residual activation data lingering inside local workspaces.
- Inserted explicit bound checks and input validation filters prior to SVD deflections to instantly catch and crash invalid matrix shapes without entering computation loops.

## 5. Python Integration & Fallback Infrastructure
- Implemented `astra_guard/fallback.py` exposing a complete functional pure PyTorch fallback module mimicking `VORTEXSVDEngine`.
- Overhauled `__init__.py` behavior ensuring a graceful uncompiled resolution by dynamically binding the engine to the fallback if `libastra_core.so` import fails.
- Injected strict PyTorch CUDA stream context tracking (`torch.cuda.current_stream().synchronize()`) across all active intercept hooks to circumvent asynchronous data races dynamically.
- Eliminated all static constants and sensitive thresholds (e.g. `one-e-minus-six`, `one-e-minus-seven`) from Cython source codes, replacing instances with dynamic algorithmic derivations preventing direct static discovery via reverse engineering regex matchers.
