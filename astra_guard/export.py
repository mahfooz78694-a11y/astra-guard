# -*- coding: utf-8 -*-
"""
ASTRA Guardrail — Custom ONNX & TensorRT Export Utility
Copyright 2026 MD Mahfooz & Alsaad Alam
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger('astra_guard')

def export_protected_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    export_path: str,
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> bool:
    """Exports PyTorch model containing VORTEX-SVD projection matrix into ONNX graph."""
    try:
        model.eval()
        if dynamic_axes is None:
            dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        logger.info(f'[ASTRA EXPORT] Exporting guarded ONNX model to {export_path}...')
        torch.onnx.export(
            model,
            (dummy_input,),
            export_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        logger.info('[ASTRA EXPORT] ONNX Export Completed Successfully.')
        return True
    except Exception as e:
        logger.error(f'[ASTRA EXPORT FAILED] {str(e)}')
        return False