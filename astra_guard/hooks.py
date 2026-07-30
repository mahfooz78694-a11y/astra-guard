# -*- coding: utf-8 -*-
import logging, torch, torch.nn as nn
from typing import Optional, Any
from astra_guard.core import VORTEXSVDEngine
logger = logging.getLogger('astra_guard')

class ZVILGuard:
    def __init__(self, model: nn.Module, target_layer: str, rank_k: int = 64, enable_basis_hopping: bool = True, enable_watermark: bool = True, preallocate_buffers: bool = False, vram_threshold_mb: int = 1000):
        self.model = model
        self.target_layer_name = target_layer
        self.vram_threshold_mb = vram_threshold_mb
        self.engine = VORTEXSVDEngine(rank_k=rank_k, enable_basis_hopping=enable_basis_hopping, enable_watermark=enable_watermark, preallocate_buffers=preallocate_buffers)
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.is_attached = False
        self.target_module = dict(model.named_modules())[target_layer]

    def calibrate(self, dataloader: Any, num_batches: int = 5) -> 'ZVILGuard':
        self.model.eval()
        acts = []
        def collector(m, i, o): acts.append(o.detach().cpu())
        h = self.target_module.register_forward_hook(collector)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches: break
                inp = batch[0] if isinstance(batch, (list, tuple)) else batch
                if torch.cuda.is_available(): inp = inp.cuda()
                self.model(inp)
        h.remove()
        self.engine.calibrate_subspace(torch.cat(acts, dim=0))
        return self

    def _forward_hook(self, module, inp, out): return self.engine.deflect_activations(out)
    def attach():
        if not self.is_attached:
            self.hook_handle = self.target_module.register_forward_hook(self._forward_hook)
            self.is_attached = True
    def detach(self):
        if self.is_attached and self.hook_handle:
            self.hook_handle.remove()
            self.is_attached = False