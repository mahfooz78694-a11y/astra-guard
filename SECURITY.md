# Security Policy — VORTEX-SVD Engine v2.0 (`astra-guard`)
## 🛡️ Supported Versions
We actively issue security patches, memory hardening updates, and vulnerability mitigations for the following release branches:

| Version | Supported | Security SLA | Release Date |
| :--- | :--- | :--- | :--- |
| **v2.0.x (Current)** | ✅ Yes | Full Priority Patching (< 24 Hours) | July 2026 |
| **v1.x.x (Legacy)** | ❌ No | Deprecated | January 2026 |

---
## 📧 Reporting a Vulnerability (Responsible Disclosure)
We take the security of the **VORTEX-SVD Engine** and its downstream enterprise AI deployments seriously. If you discover a security flaw, zero-day vulnerability, or memory isolation issue, please follow our **Coordinated Vulnerability Disclosure (CVD)** protocol:
* **Do NOT open a public GitHub issue** for security vulnerabilities.
* **Email Private Reports To:** `mahfooz78694@gmail.com`
* **Lead Security Architects:** MD Mahfooz & Alsaad Alam
* **GPG Key Encryption:** Reports containing sensitive proof-of-concept (PoC) code should be encrypted using our security research team GPG key where applicable.
### Vulnerability Response SLA:
1. **Initial Acknowledgment:** Within **12 Hours** of receipt.
2. **Triage & PoC Validation:** Within **24 Hours**.
3. **Patch Development & Release:** Critical vulnerabilities patched within **72 Hours**.
4. **Public Disclosure:** Coordinated release after patch verification and downstream enterprise notification.
---
## 🔒 Anti-Tamper & Binary Hardening Specifications
`astra-guard` implements rigorous hardware and execution-level protections to ensure tensor execution safety across enterprise PyTorch runtimes:
### 1. GCC Symbol Stripping (`-s`)
Native C++ shared objects (`.so` / `.dll`) are compiled with hard symbol stripping (`gcc -s -O3 -fPIC`) to eliminate internal function symbol tables, preventing reverse engineering and unauthorized binary hooks.
### 2. Runtime Cryptographic SHA-256 Verification
Upon execution initialization, the Python runtime verifies the SHA-256 hash checksum of dynamic C++ shared binaries prior to memory allocation to prevent dynamic library poisoning.
### 3. Non-Differentiable Gradient Isolation
Subspace projection matrices ($P_{\parallel} = V_k V_k^T$) are permanently detached from the PyTorch autograd graph (`requires_grad=False`), neutralizing Backward Pass Differentiable Approximation (BPDA) attacks attempting to estimate gradients through the guardrail.
### 4. RAM & VRAM Garbage Collection Purging
Transient Float64 SVD decomposition matrices ($U, \Sigma, V$) are explicitly cleared from system RAM and CUDA VRAM using immediate garbage collection (`gc.collect()` and `torch.cuda.empty_cache()`) post-calibration to eliminate memory scraping vectors.
---
## 🌐 Threat Model & Scope
### In-Scope Vulnerabilities (High Priority)
* Activation deflection bypass flaws allowing un-deflected adversarial perturbations to reach model outputs.
* Memory corruption, buffer overflow, or arbitrary execution vectors within native C++ shared libraries.
* Side-channel state leakage or un-sanitized NaN/Inf activation injections causing runtime engine crashes.
### Out-of-Scope Vulnerabilities
* Social engineering or phishing attacks against project maintainers.
* Attacks requiring physical or root/administrator access to the host machine running the Python interpreter.
* Flaws inherent to underlying third-party frameworks (e.g., native PyTorch core bugs or NVIDIA CUDA driver vulnerabilities).
---
## 📜 OWASP AI & Supply Chain Compliance
`astra-guard` is engineered in alignment with international AI safety frameworks:
* **OWASP Top 10 for LLM & AI Systems:** Direct mitigation for **LLM01 (Adversarial Robustness)** and **LLM02 (Data & Activation Poisoning)**.
* **SLSA (Supply-chain Levels for Software Artifacts):** Built with automated provenance verification and deterministic build processes.
* **Academic & Prior-Art Verification:** Research logic legally registered under Zenodo DOI `10.5281/zenodo.21532310`.
---
## 👥 Lead Research & Security Team
* **Lead Architects:** MD Mahfooz & Alsaad Alam
* **Official Contact:** `mahfooz78694@gmail.com`
* **Zenodo Research DOI:** [10.5281/zenodo.21532310](https://doi.org/10.5281/zenodo.21532310)
