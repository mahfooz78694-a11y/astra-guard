# -*- coding: utf-8 -*-
import os, sys, gc, logging, torch
from typing import Optional
logger = logging.getLogger('astra_guard')
logging.basicConfig(level=logging.INFO)



cdef extern from "native_crypto.hpp":
    void secure_zero_memory(void *ptr, size_t size) nogil
    int secure_mlock(void *ptr, size_t size) nogil
    int secure_munlock(void *ptr, size_t size) nogil

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

        # Static Arena Allocator Buffers
        self.arena_x_f64: Optional[torch.Tensor] = None
        self.arena_P_f64: Optional[torch.Tensor] = None
        self.arena_deflected_f64: Optional[torch.Tensor] = None
        self.arena_V_f64: Optional[torch.Tensor] = None
        self.arena_rnd: Optional[torch.Tensor] = None



    def __del__(self):
        if hasattr(self, 'arena_P_f64') and self.arena_P_f64 is not None and self.arena_P_f64.device.type == 'cpu':
            secure_munlock(<void*><Py_ssize_t>self.arena_P_f64.data_ptr(), self.arena_P_f64.numel() * self.arena_P_f64.element_size())

    def calibrate_subspace(self, baseline_activations: torch.Tensor) -> bool:
        with torch.no_grad():
            try:
                x_clean = torch.nan_to_num(baseline_activations, nan=0.0, posinf=1e4, neginf=-1e4)
                x_flat = self._unroll_to_2d(x_clean)

                x_f64 = x_flat.to(dtype=torch.float64)
                # Deterministic Ridge Regularization / Epsilon Damping for ill-conditioned matrices
                # Replaces arbitrary stochastic noise injection for zero-variance mitigation
                epsilon = 10.0 ** -7
                cov_matrix = torch.matmul(x_f64.T, x_f64) / max(1, x_f64.shape[0])
                cov_matrix = cov_matrix + torch.eye(cov_matrix.shape[0], dtype=torch.float64, device=cov_matrix.device) * epsilon

                try:
                    # Compute SVD on the regularized covariance matrix (V is both left and right singular vectors for symmetric matrix)
                    _, S, Vh = torch.linalg.svd(cov_matrix, full_matrices=False)
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
                # Dynamic randomized bound for watermark to avoid hardcoded sensitive constant (IP protection)
                watermark_scale = (torch.rand(1).item() * (0.0001 / 100.0) + (0.0001 / 10000.0))
                self.watermark_vector = (rw / torch.norm(rw)) * watermark_scale
                self.watermark_vector.requires_grad = False
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info(f'[VORTEX-SVD] Subspace Locked Successfully. Rank K={k}')
                return True
            except Exception as e:
                logger.error(f'SVD Error: {e}')
                raise AstraSecurityException('Processing Failed')

    def deflect_activations(self, x: torch.Tensor) -> torch.Tensor:
        if self.P_parallel is None: return x
        if x.shape[0] == 0 or (x.dim() > 1 and x.shape[1] == 0):
            return x
        orig_shape, orig_dtype, target_dev = x.shape, x.dtype, x.device
        with torch.no_grad():
            try:
                x_san = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                x_flat = self._unroll_to_2d(x_san)

                # Static Arena Allocation Strategy
                if self.arena_x_f64 is None or self.arena_x_f64.shape != x_flat.shape or self.arena_x_f64.device != target_dev:
                    self.arena_x_f64 = torch.empty(x_flat.shape, dtype=torch.float64, device=target_dev)

                self.arena_x_f64.copy_(x_flat)
                x_f64 = self.arena_x_f64

                if self.arena_P_f64 is None or self.arena_P_f64.device != target_dev:
                    self.arena_P_f64 = torch.empty(self.P_parallel.shape, dtype=torch.float64, device=target_dev)
                    # Only mlock once upon creation if requested
                    if target_dev.type == 'cpu':
                        res = secure_mlock(<void*><Py_ssize_t>self.arena_P_f64.data_ptr(), self.arena_P_f64.numel() * self.arena_P_f64.element_size())
                        if res != 0:
                            logger.warning("mlock failed during arena allocation.")
                self.arena_P_f64.copy_(self.P_parallel)
                P_f64 = self.arena_P_f64

                if self.enable_basis_hopping and self.V_k is not None:
                    if self.arena_V_f64 is None or self.arena_V_f64.device != target_dev:
                        self.arena_V_f64 = torch.empty(self.V_k.shape, dtype=torch.float64, device=target_dev)
                    self.arena_V_f64.copy_(self.V_k)
                    V_f64 = self.arena_V_f64

                    if self.arena_rnd is None or self.arena_rnd.shape[0] != V_f64.shape[1] or self.arena_rnd.device != target_dev:
                        self.arena_rnd = torch.empty((V_f64.shape[1], V_f64.shape[1]), dtype=torch.float64, device=target_dev)

                    self.arena_rnd.normal_()
                    Q, _ = torch.linalg.qr(self.arena_rnd)
                    V_h = torch.matmul(V_f64, Q)
                    P_f64.copy_(torch.matmul(V_h, V_h.T)) # Overwrites P_f64 arena inline

                if self.arena_deflected_f64 is None or self.arena_deflected_f64.shape != (x_f64.shape[0], P_f64.shape[1]) or self.arena_deflected_f64.device != target_dev:
                    self.arena_deflected_f64 = torch.empty((x_f64.shape[0], P_f64.shape[1]), dtype=torch.float64, device=target_dev)

                deflected_f64 = self.arena_deflected_f64
                torch.matmul(x_f64, P_f64, out=deflected_f64)

                if self.enable_watermark and self.watermark_vector is not None:
                    deflected_f64.add_(self.watermark_vector.to(device=target_dev))

                # To prevent data wiping when orig_dtype is float64, we MUST enforce copy if we are about to zero deflected_f64.
                deflected = deflected_f64.to(dtype=orig_dtype, device=target_dev, copy=True)

                # Secure sanitize temporary buffers natively
                if target_dev.type == 'cpu':
                    secure_zero_memory(<void*><Py_ssize_t>x_f64.data_ptr(), x_f64.numel() * x_f64.element_size())
                    secure_zero_memory(<void*><Py_ssize_t>P_f64.data_ptr(), P_f64.numel() * P_f64.element_size())
                    secure_zero_memory(<void*><Py_ssize_t>deflected_f64.data_ptr(), deflected_f64.numel() * deflected_f64.element_size())
                else:
                    x_f64.zero_()
                    P_f64.zero_()
                    deflected_f64.zero_()

                return self._restore_shape(deflected, orig_shape).detach().requires_grad_(False)
            except Exception:
                return x

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