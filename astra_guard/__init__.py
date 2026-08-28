try:
    from astra_guard.core import VORTEXSVDEngine
except ImportError:
    import logging

    logger = logging.getLogger("astra_guard")
    logger.warning("astra_guard.core not compiled. Falling back to pure Python implementation.")
    from astra_guard.fallback import VORTEXSVDEngine

from astra_guard.hooks import ZVILGuard
from astra_guard.auto import AutoSubspaceTuner
from astra_guard.export import export_protected_onnx
from astra_guard.telemetry import start_telemetry_server

__all__ = [
    "VORTEXSVDEngine",
    "ZVILGuard",
    "AutoSubspaceTuner",
    "export_protected_onnx",
    "start_telemetry_server",
]

__version__ = "2.0.0"
__author__ = "MD Mahfooz & Alsaad Alam"
__doi__ = "10.5281/zenodo.21532310"
