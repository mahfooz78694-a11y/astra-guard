# -*- coding: utf-8 -*-
import os, sys, gc, logging, torch
from typing import Optional
logger = logging.getLogger('astra_guard')
logging.basicConfig(level=logging.INFO)

class AstraSecurityException(Exception):
    pass

class VORTEXSVDEngine:
    def __init__(self, rank_k: int = 64, enable_basis_hopping: bool = True, enable_watermark: bool = True, preallocate_buffers: bool = False):
        self.rank_k = rank_k
        self.enable_basis_hopping = enable_basis_hopping
        self.enable_watermark = enable_watermark
        self.P_parallel: Optional[torch.Tensor] = None
        self.V_k: Optional[torch.Tensor] = None
        self.watermark_vector: Optional[torch.Tensor] = None

    def calibrate_subspace(self, baseline_activations: torch.Tensor) -> bool:
        with torch.no_grad():
            try:
                x_clean = torch.nan_to_num(baseline_activations, nan=0.0, posinf=1e4, neginf=-1e4)
                x_flat = self._unroll_to_2d(x_clean)

                # Downsample/crop to prevent DoS via $O(N^3)$ lockups
                if x_flat.shape[0] > 2048:
                    x_flat = x_flat[:2048, :]
                if x_flat.shape[1] > 2048:
                    x_flat = x_flat[:, :2048]

                # Zero-variance check and mitigation
                var = torch.var(x_flat, dim=0, unbiased=False)
                zero_var_mask = var < (torch.rand(1).item() * 1e-5 + 1e-7)
                if zero_var_mask.any():
                    noise = torch.randn_like(x_flat) * (torch.rand(1).item() * 1e-5 + 1e-7)
                    noise = noise * zero_var_mask.float().unsqueeze(0)
                    x_flat = x_flat + noise

                x_f64 = x_flat.to(dtype=torch.float64)
                try:
                    U, S, Vh = torch.linalg.svd(x_f64, full_matrices=False)
                    k = min(self.rank_k, Vh.shape[0])
                    self.V_k = Vh[:k, :].T
                except RuntimeError:
                    logger.warning('[VORTEX-SVD] SVD Non-Convergence. Triggering QR Fallback.')
                    Q, _ = torch.linalg.qr(x_flat.T.to(dtype=torch.float64))
                    k = min(self.rank_k, Q.shape[1])
                    self.V_k = Q[:, :k]
                self.P_parallel = torch.matmul(self.V_k, self.V_k.T)
                self.P_parallel.requires_grad = False
                num_ch = self.P_parallel.shape[0]
                rw = torch.randn(num_ch, dtype=torch.float64)
                self.watermark_vector = (rw / torch.norm(rw)) * (torch.rand(1).item() * 1e-6 + 1e-8)
                self.watermark_vector.requires_grad = False
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info(f'[VORTEX-SVD] Subspace Locked Successfully. Rank K={k}')
                return True
            except Exception as e:
                raise AstraSecurityException('Processing Failed') from None

    def deflect_activations(self, x: torch.Tensor) -> torch.Tensor:
        if self.P_parallel is None: return x
        try:
            orig_shape, orig_dtype, target_dev = x.shape, x.dtype, x.device
        except Exception:
            raise AstraSecurityException('Processing Failed') from None
        with torch.no_grad():
            try:
                x_san = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                x_flat = self._unroll_to_2d(x_san)

                # We need to make sure we don't multiply x_f64 with P_f64 if P_f64 is smaller than x_f64.
                # Actually for deflect, since we must return orig_shape,
                # if we crop x_flat for the matmul, we must pad it back or we can't restore shape.
                # Since P_parallel is size k x k or channels x channels, it is calibrated to the cropped channels.
                # If during deflect the channels exceed what P_parallel was calibrated on, we slice P_parallel or crop x_flat.
                # To be safe and just prevent DoS lockup on the math:
                if x_flat.shape[1] > self.P_parallel.shape[0]:
                    x_flat = x_flat[:, :self.P_parallel.shape[0]]

                x_f64 = x_flat.to(dtype=torch.float64, device=target_dev)
                P_f64 = self.P_parallel.to(dtype=torch.float64, device=target_dev)
                if self.enable_basis_hopping and self.V_k is not None:
                    V_f64 = self.V_k.to(dtype=torch.float64, device=target_dev)
                    rnd = torch.randn(V_f64.shape[1], V_f64.shape[1], dtype=torch.float64, device=target_dev)
                    Q, _ = torch.linalg.qr(rnd)
                    V_h = torch.matmul(V_f64, Q)
                    P_f64 = torch.matmul(V_h, V_h.T)
                deflected_f64 = torch.matmul(x_f64, P_f64)
                if self.enable_watermark and self.watermark_vector is not None:
                    deflected_f64 = deflected_f64 + self.watermark_vector[:deflected_f64.shape[1]].to(device=target_dev)
                deflected = deflected_f64.to(dtype=orig_dtype)

                # If we cropped features earlier, we pad them back with 0s to restore shape
                if deflected.shape[1] < self._unroll_to_2d(x).shape[1]:
                    pad_size = self._unroll_to_2d(x).shape[1] - deflected.shape[1]
                    deflected = torch.cat([deflected, torch.zeros(deflected.shape[0], pad_size, dtype=orig_dtype, device=target_dev)], dim=1)

                return self._restore_shape(deflected, orig_shape)
            except Exception:
                raise AstraSecurityException('Processing Failed') from None

    def _unroll_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if x.dim() == 2: return x
        elif x.dim() == 4: return x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
        elif x.dim() == 3: return x.reshape(-1, x.shape[2])
        else: raise ValueError('Invalid Rank')

    def _restore_shape(self, x_flat: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        x_flat = x_flat.contiguous()
        if len(orig_shape) == 2: return x_flat
        elif len(orig_shape) == 4: return x_flat.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1]).permute(0, 3, 1, 2).contiguous()
        elif len(orig_shape) == 3: return x_flat.reshape(orig_shape)
        else: raise ValueError('Invalid Dim')