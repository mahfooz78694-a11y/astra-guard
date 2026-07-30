# -*- coding: utf-8 -*-
"""
ASTRA Guardrail — Public PyTorch Hook Interceptor & Circuit-Breaker Module
Copyright 2026 MD Mahfooz & Alsaad Alam
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Any
from astra_guard.core import VORTEXSVDEngine

logger = logging.getLogger('astra_guard')

class ZVILGuard:
    """
    2-Line Integration Guardrail Wrapper for PyTorch Models.
    Attaches non-invasive forward hooks to protect activations in real-time.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        rank_k: int = 64,
        enable_basis_hopping: bool = True,
        enable_watermark: bool = True,
        preallocate_buffers: bool = False,
        vram_threshold_mb: int = 1000
    ):
        self.model = model
        self.target_layer_name = target_layer
        self.vram_threshold_mb = vram_threshold_mb
        self.engine = VORTEXSVDEngine(
            rank_k=rank_k,
            enable_basis_hopping=enable_basis_hopping,
            enable_watermark=enable_watermark,
            preallocate_buffers=preallocate_buffers
        )
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.is_attached = False
        self.circuit_breaker_tripped = False
        self.target_module = self._find_target_layer()

    def _find_target_layer(self) -> nn.Module:
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        raise KeyError(f'[ASTRA-ERROR] Layer {self.target_layer_name} not found in model hierarchy.')

    def calibrate(self, dataloader: Any, num_batches: int = 5) -> 'ZVILGuard':
        logger.info(f'[ASTRA] Calibrating VORTEX-SVD Subspace on layer: {self.target_layer_name}...')
        self.model.eval()
        captured_activations = []
        def temp_collector(module, input_t, output_t):
            captured_activations.append(output_t.detach().cpu())
        handle = self.target_module.register_forward_hook(temp_collector)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches: break
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                if torch.cuda.is_available(): inputs = inputs.cuda()
                self.model(inputs)
        handle.remove()
        if not captured_activations:
            raise RuntimeError('[ASTRA-ERROR] No activation samples captured during calibration.')
        combined_acts = torch.cat(captured_activations, dim=0)
        success = self.engine.calibrate_subspace(combined_acts)
        if not success:
            raise RuntimeError('[ASTRA-ERROR] Subspace calibration failed.')
        logger.info('[ASTRA] Calibration completed successfully.')
        return self

    def _check_circuit_breaker(self) -> bool:
        """VRAM Circuit Breaker monitoring via torch.cuda.mem_get_info."""
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mb = free_mem / (1024 * 1024)
            if free_mb < self.vram_threshold_mb:
                if not self.circuit_breaker_tripped:
                    logger.warning(f'[ASTRA CIRCUIT-BREAKER] Low VRAM ({free_mb:.1f}MB). Switched to Safe Pass-Through Mode.')
                    self.circuit_breaker_tripped = True
                return True
        self.circuit_breaker_tripped = False
        return False

    def _forward_hook(self, module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor) -> torch.Tensor:
        if self._check_circuit_breaker():
            return output_tensor
        return self.engine.deflect_activations(output_tensor)

    def attach(self) -> None:
        if not self.is_attached and self.target_module is not None:
            self.hook_handle = self.target_module.register_forward_hook(self._forward_hook)
            self.is_attached = True
            logger.info(f'[ASTRA] Guardrail successfully attached to layer: {self.target_layer_name}')

    def detach(self) -> None:
        if self.is_attached and self.hook_handle is not None:
            self.hook_handle.remove()
            self.is_attached = False
            logger.info('[ASTRA] Guardrail successfully detached.')