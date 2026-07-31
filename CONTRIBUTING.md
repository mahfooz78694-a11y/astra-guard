# Contributing to ASTRA Guardrail

First off, thank you for considering contributing to `astra-guard`! We welcome contributions from academic researchers, industry security engineers, and open-source enthusiasts.

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Standard Git Workflow

1. **Fork the repository** on GitHub.
2. **Clone your fork locally**:
   `git clone https://github.com/your-username/astra-guard.git`
   `cd astra-guard`
3. **Create a dedicated feature branch**:
   `git checkout -b feature/your-feature-name`
4. **Make your changes** and commit them with descriptive, standard commit messages.
5. **Push your branch** to your fork:
   `git push origin feature/your-feature-name`
6. **Open a Pull Request (PR)** against the `main` branch of the upstream repository. Ensure you fully document the PR, explaining the purpose of the change, any related issues, and providing testing details.

## PR Guidelines

- **Keep PRs focused**: Each PR should address a single feature, bug fix, or documentation update.
- **Documentation**: If you change the API or add new features, update `README.md` and inline docstrings.
- **No API Changes**: Do not modify the core API signatures of `ZVILGuard` or `AutoSubspaceTuner` without prior discussion in an issue.
- **Maintain SLA**: Ensure that your changes do not compromise the sub-millisecond (<0.1ms) SLA for the ZVILGuard interceptor net latency overhead. You can verify this using the `scripts/reproduce_benchmarks.py` script.
- **Do not break Autograd**: Never remove or modify the IEEE 754 Float64 SVD casting or autograd detachment logic (`requires_grad = False`). This is critical for preventing adaptive white-box attacks.

## Unit Testing Requirements

All code changes must be tested.

1. We use `pytest` for testing. Run the full test suite locally before submitting a PR:
   `PYTHONPATH=. pytest tests/ -v`
2. **Coverage**: New features must include comprehensive test cases in the `tests/` directory.
3. **Python Versions**: We explicitly support and test against Python versions 3.9, 3.10, 3.11, and 3.12. Please ensure your code is compatible.
4. **Type Checking**: We use `mypy` for static analysis and type safety checks. Run `mypy astra_guard/` to ensure there are no type errors.

Thank you for helping make AI deployments more secure!
