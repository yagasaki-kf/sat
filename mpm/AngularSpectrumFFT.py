"""
AngularSpectrumFFT.py

Python implementation of angular-spectrum propagation used by
PhaseRetrieval.py. This mirrors AngularSpectrumFFT.jl so the full workflow
can run without Julia.
"""
from __future__ import annotations

import numpy as np


def _fftshift_freq(n: int, dx: float) -> np.ndarray:
    """Return fftshift-ordered spatial-frequency axis [1/m]."""
    start = -(n // 2)
    stop = (n + 1) // 2
    return np.arange(start, stop, dtype=np.float64) / (n * dx)


def propagate_angular_spectrum(
    u: np.ndarray,
    dz: float,
    wavelength: float,
    dx: float,
    tilt_x: float,
    tilt_y: float,
) -> np.ndarray:
    """
    Angular-spectrum propagation with linear tilt compensation.
    """
    n, m = u.shape
    k = 2.0 * np.pi / wavelength

    fx = _fftshift_freq(n, dx)
    fy = _fftshift_freq(m, dx)
    FX = fx.reshape(n, 1)
    FY = fy.reshape(1, m)

    fz_sq_raw = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    propagating = fz_sq_raw >= 0.0
    FZ = np.sqrt(np.maximum(fz_sq_raw, 0.0))

    # Hard-cut evanescent spectrum: non-propagating components are set to zero.
    H = np.exp(1j * k * dz * FZ) * propagating

    delta_x = tilt_x * dz
    delta_y = tilt_y * dz
    shift = np.exp(-2.0j * np.pi * (FX * delta_x + FY * delta_y))

    U = np.fft.fftshift(np.fft.fft2(u))
    return np.fft.ifft2(np.fft.ifftshift(U * H * shift))
