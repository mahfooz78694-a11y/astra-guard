try:
    from astra_guard.core import VORTEXSVDEngine
except ImportError as e:
    raise ImportError("astra_guard_core is not compiled. Please install from a wheel or run 'python setup.py build_ext --inplace'.") from e

from astra_guard.hooks import ZVILGuard
from astra_guard.auto import AutoSubspaceTuner
from astra_guard.export import export_protected_onnx
from astra_guard.telemetry import start_telemetry_server

__all__ = [
    "VORTEXSVDEngine",
    "ZVILGuard",
    "AutoSubspaceTuner",
    "export_protected_onnx",
    "start_telemetry_server"
]

__version__ = "2.0.0"
__author__ = "MD Mahfooz & Alsaad Alam"
__doi__ = "10.5281/zenodo.21532310"
